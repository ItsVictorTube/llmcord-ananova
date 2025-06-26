# -----------------------------------------------------------------------------
# llmcord.py: A Discord bot that connects to a Large Language Model (LLM)
# -----------------------------------------------------------------------------

# --- IMPORTS ---
import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
import httpx
from openai import AsyncOpenAI
import yaml

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# --- CONSTANTS ---
# Model tags that indicate vision capabilities
VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude", "gemini", "2.0", "2.5", "gemma", "llama", "pixtral", "mistral", "vision", "vl")

# Embed colors for different response states
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

# Visual indicators and timing for streaming responses
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1  # Min seconds between response edits to avoid rate limits

# Cache and message limits
MAX_MESSAGE_NODES = 500  # Max messages to keep in the cache to conserve memory

# --- CONFIGURATION ---
def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    """Loads the configuration from a YAML file."""
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)

config = get_config()

# --- GLOBAL STATE ---
# The currently active LLM model (e.g., "openai/gpt-4o")
curr_model = next(iter(config["models"]))

# A cache for processed message data to avoid re-fetching
msg_nodes = {}

# Timestamp of the last streaming edit to manage rate-limiting
last_task_time = 0

# --- BOT AND CLIENT INITIALIZATION ---
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content

activity = discord.CustomActivity(name=(config.get("status_message", "github.com/jakobdylanc/llmcord"))[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

# Asynchronous HTTP client for fetching attachment content
httpx_client = httpx.AsyncClient()


# --- DATA STRUCTURES ---
@dataclass
class MsgNode:
    """
    Represents a processed message in the conversation cache.
    This node stores the content, role, and metadata of a message to avoid
    re-processing it on subsequent replies.
    """
    # Core content
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    role: Literal["user", "assistant"] = "assistant"

    # Metadata and warnings
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    # Conversation structure
    parent_msg: Optional[discord.Message] = None

    # Concurrency control
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


# --- HELPER FUNCTIONS ---
async def _is_message_authorized(message: discord.Message, config: dict) -> bool:
    """Checks if the message author and channel are authorized based on the config."""
    is_dm = message.channel.type == discord.ChannelType.private
    permissions = config["permissions"]
    author = message.author

    # Check user permissions
    user_is_admin = author.id in permissions["users"]["admin_ids"]
    (allowed_user_ids, blocked_user_ids) = (permissions["users"]["allowed_ids"], permissions["users"]["blocked_ids"])
    (allowed_role_ids, blocked_role_ids) = (permissions["roles"]["allowed_ids"], permissions["roles"]["blocked_ids"])
    role_ids = {role.id for role in getattr(author, "roles", [])}

    allow_all_users = not allowed_user_ids and (is_dm or not allowed_role_ids)
    is_good_user = user_is_admin or allow_all_users or author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    # Check channel permissions
    (allowed_channel_ids, blocked_channel_ids) = (permissions["channels"]["allowed_ids"], permissions["channels"]["blocked_ids"])
    channel_ids = {message.channel.id, getattr(message.channel, "parent_id", None), getattr(message.channel, "category_id", None)}

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or (config.get("allow_dms") and is_dm) or (allow_all_channels or any(id in allowed_channel_ids for id in channel_ids))
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    return not (is_bad_user or is_bad_channel)


async def _build_message_chain(start_msg: discord.Message, config: dict, bot_user: discord.ClientUser) -> tuple[list[dict], set[str]]:
    """
    Builds the conversation history by traversing replies from a starting message.
    Returns a list of messages formatted for the API and a set of user warnings.
    """
    messages = []
    user_warnings = set()
    curr_msg = start_msg

    max_messages = config.get("max_messages", 10)
    max_text_per_message = config.get("max_text", 4096)
    accept_images = any(x in curr_model.lower() for x in VISION_MODEL_TAGS)
    max_images_per_message = config.get("max_images", 1) if accept_images else 0
    creator_id = config.get("creator_id")

    while curr_msg is not None and len(messages) < max_messages:
        # Use cached node if available, otherwise create a new one
        node = msg_nodes.setdefault(curr_msg.id, MsgNode())
        async with node.lock:
            # --- Populate the node if it's new ---
            if node.text is None:
                # 1. Fetch text content from message, embeds, and text attachments
                cleaned_content = curr_msg.content.removeprefix(bot_user.mention).lstrip()
                good_attachments = [att for att in curr_msg.attachments if att.content_type and (att.content_type.startswith("text") or att.content_type.startswith("image"))]
                attachment_responses = await asyncio.gather(*(httpx_client.get(att.url) for att in good_attachments))

                base_text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + [f"{embed.title}\n{embed.description}" for embed in curr_msg.embeds if embed.title or embed.description]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                # 2. Assign role and format user message with display name
                node.role = "assistant" if curr_msg.author == bot_user else "user"
                if node.role == "user":
                    user_name = config.get("creator_name", "Victor") if creator_id and curr_msg.author.id == creator_id else curr_msg.author.display_name
                    node.text = f"{user_name}: {base_text}"
                else:
                    node.text = base_text

                # 3. Fetch image attachments
                node.images = [
                    {"type": "image_url", "image_url": {"url": f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"}}
                    for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("image")
                ]

                # 4. Find the parent message in the conversation chain
                try:
                    if curr_msg.reference:
                        node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(curr_msg.reference.message_id)
                    # Implicitly link to previous message if it's a natural continuation
                    elif (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0]):
                         if prev_msg_in_channel.author in (bot_user, curr_msg.author):
                             node.parent_msg = prev_msg_in_channel

                except (discord.NotFound, discord.HTTPException):
                    node.fetch_parent_failed = True
                    logging.exception("Failed to fetch parent message.")

            # --- Format node content for the API and add warnings ---
            api_content = node.text[:max_text_per_message]
            if node.images[:max_images_per_message]:
                api_content = ([{"type": "text", "text": api_content}] if api_content else []) + node.images[:max_images_per_message]

            if api_content:
                messages.append({"role": node.role, "content": api_content})

            # Add warnings if content was truncated or had issues
            if len(node.text) > max_text_per_message:
                user_warnings.add(f"⚠️ Max {max_text_per_message:,} characters per message")
            if len(node.images) > max_images_per_message:
                user_warnings.add(f"⚠️ Max {max_images_per_message} image(s) per message" if max_images_per_message > 0 else "⚠️ This model can't see images")
            if node.fetch_parent_failed or (node.parent_msg and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Conversation history limited to the last {len(messages)} message(s)")

            curr_msg = node.parent_msg

    return messages, user_warnings


# --- DISCORD COMMANDS ---
@discord_bot.tree.command(name="model", description="View or switch the current LLM")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    """Handles the /model command to switch the active LLM."""
    global curr_model

    if model == curr_model:
        await interaction.response.send_message(f"Current model is already: `{curr_model}`", ephemeral=True)
        return

    # Only admins can change the model
    if interaction.user.id in config["permissions"]["users"]["admin_ids"]:
        curr_model = model
        output = f"Model switched to: `{model}`"
        logging.info(f"{output} (by user {interaction.user.id})")
    else:
        output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=True)


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, current_input: str) -> list[Choice[str]]:
    """Provides autocomplete suggestions for the /model command."""
    global config
    # Hot-reload the config to show the latest model list
    if current_input == "":
        config = await asyncio.to_thread(get_config)

    # Filter models based on user input
    filtered_models = [m for m in config["models"] if current_input.lower() in m.lower()]

    # Format choices, showing the current model with a special icon
    choices = [
        Choice(name=f"○ {m}", value=m) for m in filtered_models if m != curr_model
    ][:24] # Limit to 24 choices to stay under Discord's limit
    if curr_model in filtered_models:
        choices.append(Choice(name=f"◉ {curr_model} (current)", value=curr_model))

    return choices


# --- START OF CHANGE ---
@discord_bot.tree.command(name="purge", description="Admin: Deletes the bot's recent messages in this channel.")
@discord.app_commands.describe(limit="The number of bot messages to delete (default: 1, max: 100).")
@discord.app_commands.allowed_contexts(dms=True, guilds=True, private_channels=True)
# --- END OF CHANGE ---
async def purge_command(interaction: discord.Interaction, limit: int = 1) -> None:
    """Handles the /purge command to delete the bot's own messages."""
    # 1. Check for admin permissions
    if interaction.user.id not in config["permissions"]["users"]["admin_ids"]:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    # 2. Defer response as fetching and deleting can be slow
    await interaction.response.defer(ephemeral=True, thinking=True)

    # 3. Find the bot's messages to delete
    # Clamp the limit to a reasonable range
    limit = max(1, min(100, limit))
    messages_to_delete = []
    # We search a bit more than the limit in case of interspersed user messages
    async for message in interaction.channel.history(limit=limit * 5):
        if len(messages_to_delete) >= limit:
            break
        if message.author.id == discord_bot.user.id:
            messages_to_delete.append(message)

    if not messages_to_delete:
        await interaction.followup.send("No recent messages from me found to delete.")
        return

    # 4. Delete the messages
    try:
        # DMs and Group DMs do not support bulk deletion, so we delete one by one
        if isinstance(interaction.channel, (discord.DMChannel, discord.GroupChannel)):
            for msg in messages_to_delete:
                await msg.delete()
        else:
            # Bulk delete is more efficient in server text channels
            await interaction.channel.delete_messages(messages_to_delete)

        deleted_count = len(messages_to_delete)
        plural = "s" if deleted_count > 1 else ""
        await interaction.followup.send(f"Successfully deleted my last {deleted_count} message{plural}.")

    except discord.Forbidden:
        logging.warning(f"Missing permissions to delete messages in channel {interaction.channel.id}")
        await interaction.followup.send("I lack the 'Manage Messages' permission to delete messages in this channel.")
    except discord.HTTPException as e:
        # This often happens for messages older than 14 days with bulk delete
        logging.error(f"Failed to delete messages: {e}")
        await interaction.followup.send("An error occurred. I might not be able to delete messages older than 14 days.")


# --- BOT EVENTS ---
@discord_bot.event
async def on_ready() -> None:
    """Event handler for when the bot successfully connects to Discord."""
    if client_id := config.get("client_id"):
        # Corrected invite URL with the 'applications.commands' scope
        invite_url = f"https://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot%20applications.commands"
        logging.info(f"\n\nBOT INVITE URL (re-invite if slash commands don't appear):\n{invite_url}\n")
    
    # Check for a test guild ID in the config
    if guild_id := config.get("test_guild_id"):
        guild_obj = discord.Object(id=int(guild_id))
        # Copy global commands to the test guild
        discord_bot.tree.copy_global_to(guild=guild_obj)
        # Sync to the test guild (instant update)
        await discord_bot.tree.sync(guild=guild_obj)
        logging.info(f"Synced commands to test guild: {guild_id} (DMs will not have commands)")
    else:
        # Sync globally if no test guild is specified (can take up to an hour)
        await discord_bot.tree.sync()
        logging.info("Synced commands globally (will work in DMs, updates can take up to an hour).")
    
    logging.info(f"Bot '{discord_bot.user}' is ready and online.")


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    """Main event handler for processing incoming messages."""
    global last_task_time

    # 1. Initial Checks: Ignore bots and messages not mentioning the bot (in servers)
    if new_msg.author.bot or (new_msg.channel.type != discord.ChannelType.private and discord_bot.user not in new_msg.mentions):
        return

    # 2. Permissions: Check if the user and channel are authorized
    config = await asyncio.to_thread(get_config)
    if not await _is_message_authorized(new_msg, config):
        logging.warning(f"Unauthorized message from user {new_msg.author.id} in channel {new_msg.channel.id}")
        return

    # 3. Build Conversation Chain: Traverse replies to create context for the LLM
    messages, user_warnings = await _build_message_chain(new_msg, config, discord_bot.user)
    logging.info(f"Message received (user ID: {new_msg.author.id}, conversation length: {len(messages)}):\n{new_msg.content}")

    # 4. Prepare System Prompt: Add context about date, time, and user naming
    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()
        prompt = system_prompt.replace("{date}", now.strftime("%B %d, %Y")).replace("{time}", now.strftime("%I:%M %p %Z"))
        prompt += "\nEach user's message is prefixed with their display name."
        # This new line explicitly tells the bot how to format its own messages.
        prompt += "\nYou must not prefix your own responses with your name (e.g., 'Ana:')."
        if config.get("creator_id"):
            creator_name = config.get("creator_name", "Creator")
            prompt += f" The user named '{creator_name}' is your creator."
        messages.append({"role": "system", "content": prompt.strip()})

    # 5. Generate and Stream Response
    provider, model = curr_model.split("/", 1)
    provider_config = config["providers"][provider]
    openai_client = AsyncOpenAI(base_url=provider_config["base_url"], api_key=provider_config.get("api_key", "sk-no-key"))
    
    response_msgs = []
    response_contents = [""]
    edit_task = None
    finish_reason = None
    
    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    max_msg_len = 2000 if config.get("use_plain_responses") else (4096 - len(STREAMING_INDICATOR))

    try:
        async with new_msg.channel.typing():
            stream = await openai_client.chat.completions.create(
                model=model, messages=messages[::-1], stream=True, extra_body=config["models"].get(curr_model)
            )
            async for chunk in stream:
                delta_content = chunk.choices[0].delta.content or ""
                response_contents[-1] += delta_content

                if finish_reason := chunk.choices[0].finish_reason:
                    break
                
                # Split into a new message if the current one is too long
                if len(response_contents[-1]) > max_msg_len:
                    response_contents.append("")

                # For embed responses, edit the message periodically
                if not config.get("use_plain_responses"):
                    time_since_last_edit = datetime.now().timestamp() - last_task_time
                    if (edit_task is None or edit_task.done()) and time_since_last_edit >= EDIT_DELAY_SECONDS:
                        if edit_task: await edit_task # Ensure previous task is complete
                        
                        embed.description = response_contents[-1] + STREAMING_INDICATOR
                        embed.color = EMBED_COLOR_INCOMPLETE

                        if len(response_msgs) < len(response_contents): # This is a new message
                            reply_to = new_msg if not response_msgs else response_msgs[-1]
                            response_msg = await reply_to.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)
                        else: # Edit the existing message
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                        
                        last_task_time = datetime.now().timestamp()
            
        # Final update to the response message(s) after stream completion
        is_good_finish = finish_reason and finish_reason.lower() in ("stop", "end_turn")
        embed.color = EMBED_COLOR_COMPLETE if is_good_finish else EMBED_COLOR_INCOMPLETE
        
        if config.get("use_plain_responses"):
            for i, content in enumerate(response_contents):
                reply_to = new_msg if i == 0 else response_msgs[i-1]
                response_msgs.append(await reply_to.reply(content=content, suppress_embeds=True))
        elif response_msgs: # Final edit for embed response
            embed.description = response_contents[-1]
            await response_msgs[-1].edit(embed=embed)

    except Exception:
        logging.exception("Error while generating LLM response.")
        if response_msgs:
             await response_msgs[-1].edit(content="Sorry, an error occurred while generating the response.", embed=None)

    # 6. Cache final response content and clean up
    for i, response_msg in enumerate(response_msgs):
        node = msg_nodes.setdefault(response_msg.id, MsgNode(parent_msg=new_msg))
        async with node.lock:
             node.text = response_contents[i]

    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[:num_nodes - MAX_MESSAGE_NODES]:
            if node_to_delete := msg_nodes.get(msg_id):
                async with node_to_delete.lock:
                    msg_nodes.pop(msg_id, None)


# --- MAIN EXECUTION ---
async def main() -> None:
    """Starts the Discord bot."""
    try:
        await discord_bot.start(config["bot_token"])
    except discord.LoginFailure:
        logging.critical("Failed to log in. Please check your 'bot_token' in the config file.")
    except Exception:
        logging.critical("An unexpected error occurred during bot startup.", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot shutting down.")