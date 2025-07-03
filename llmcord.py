# -----------------------------------------------------------------------------
# llmcord.py: (Original, monolithic version)
# NOTE: The code has been split into modules and cogs for maintainability.
# This file is preserved as a reference.
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
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

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

def validate_config(config: dict) -> None:
    """Validate configuration structure and required fields."""
    required_fields = {
        "bot_token": str,
        "providers": dict,
        "models": dict,
        "permissions": dict
    }
    for field, expected_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
        if not isinstance(config[field], expected_type):
            raise TypeError(f"Config field {field} must be {expected_type.__name__}")
    if not config["models"]:
        raise ValueError("At least one model must be configured")

config = get_config()
validate_config(config)


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
    user_id: Optional[int] = None # Added for user-specific prompts if needed

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

    # If allowed_user_ids is empty, allow all users. If allowed_role_ids is empty, allow all roles.
    # In DMs, only user IDs matter. In guilds, both user and role IDs matter.
    allow_all_users_or_roles = not allowed_user_ids and (is_dm or not allowed_role_ids)
    is_good_user = user_is_admin or allow_all_users_or_roles or author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    # Check channel permissions
    (allowed_channel_ids, blocked_channel_ids) = (permissions["channels"]["allowed_ids"], permissions["channels"]["blocked_ids"])
    # Include the channel itself, its parent (if a thread), and category (if a channel)
    channel_ids = {message.channel.id, getattr(message.channel, "parent_id", None), getattr(message.channel, "category_id", None)}
    channel_ids.discard(None) # Remove None if no parent/category

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or (config.get("allow_dms") and is_dm) or (allow_all_channels or any(id in allowed_channel_ids for id in channel_ids))
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    return not (is_bad_user or is_bad_channel)


async def _build_message_chain(start_msg: discord.Message, config: dict, bot_user: discord.ClientUser) -> tuple[list[dict], set[str]]:
    """
    Builds the conversation history by traversing replies from a starting message.
    Returns a list of messages formatted for the API and a set of user warnings.
    NOTE: This function is currently only called by the `on_message` handler's older
    logic, and `on_message` now has its own, more detailed message chain building.
    This helper is kept for completeness as it existed in the original code.
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


@discord_bot.tree.command(name="purge", description="Admin: Deletes the bot's recent messages in this channel.")
@discord.app_commands.describe(limit="The number of bot messages to delete (default: 1, max: 100).")
@discord.app_commands.allowed_contexts(dms=True, guilds=True, private_channels=True)
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
    """Main event handler for processing incoming messages and generating LLM responses."""
    global last_task_time, msg_nodes

    # 1. Initial Checks: Ignore bots and messages not mentioning the bot (in servers)
    if new_msg.author.bot or (new_msg.channel.type != discord.ChannelType.private and discord_bot.user not in new_msg.mentions):
        return

    # 2. Permissions: Check if the user and channel are authorized
    # Hot-reload config on message to pick up changes without bot restart
    current_config = await asyncio.to_thread(get_config)
    if not await _is_message_authorized(new_msg, current_config):
        return

    # 3. Prepare LLM client and parameters
    provider_slash_model = curr_model
    provider, model = provider_slash_model.split("/", 1)
    
    provider_config = current_config["providers"][provider]
    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required") # Some models/providers don't need a key
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = current_config.get("max_text", 100000)
    max_images = current_config.get("max_images", 5) if accept_images else 0
    max_messages_history = current_config.get("max_messages", 25)
    creator_id = current_config.get("creator_id")
    creator_name = current_config.get("creator_name", "Creator")

    # 4. Build message chain for the LLM API
    # This logic is more detailed than _build_message_chain to handle specific Discord message types
    # and provide granular warnings.
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg is not None and len(messages) < max_messages_history:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text is None: # Only process if not already cached
                # Remove bot mention from content if present
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                # Filter attachments for text and image types
                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
                
                # Fetch attachment content concurrently
                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                # Combine text from content, embeds, and text attachments
                base_text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                # Set role and user_id for the message node
                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                curr_node.text = base_text

                # Convert image attachments to base64 for API
                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                # Track if any unsupported attachments were present
                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                # Determine the parent message for conversation threading
                try:
                    parent_message_to_fetch = None
                    if curr_msg.reference:
                        # Direct reply: use the referenced message
                        parent_message_to_fetch = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(curr_msg.reference.message_id)
                    elif curr_msg.channel.type == discord.ChannelType.public_thread and curr_msg.reference is None:
                        # If it's the first message in a public thread, its parent is the thread's starter message
                        parent_message_to_fetch = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(curr_msg.channel.id)
                    elif (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0]):
                        # Implicit continuation: if the previous message was by the bot or current user
                        if prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply) and \
                           prev_msg_in_channel.author in (discord_bot.user, curr_msg.author):
                            parent_message_to_fetch = prev_msg_in_channel
                    
                    curr_node.parent_msg = parent_message_to_fetch

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            # Format node content for the API request
            api_content = curr_node.text[:max_text]
            if curr_node.images[:max_images]:
                # If images are accepted and present, content is a list of text/image objects
                api_content = ([dict(type="text", text=api_content)] if api_content else []) + curr_node.images[:max_images]
            
            # Add message to the conversation history if it has content
            if api_content:
                message_for_api = dict(content=api_content, role=curr_node.role)
                if accept_usernames and curr_node.user_id is not None:
                    # Providers like OpenAI can use 'name' for per-user prompts
                    message_for_api["name"] = str(curr_node.user_id)
                messages.append(message_for_api)

            # Add user warnings based on content truncation or issues
            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments (only text/images processed)")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(messages) == max_messages_history):
                user_warnings.add(f"⚠️ Conversation history limited to the last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    # Add system prompt if configured
    if system_prompt := current_config["system_prompt"]:
        now = datetime.now().astimezone()
        prompt = system_prompt.replace("{date}", now.strftime("%B %d, %Y")).replace("{time}", now.strftime("%I:%M %p %Z"))
        prompt += "\nEach user's message is prefixed with their display name."
        # This new line explicitly tells the bot how to format its own messages.
        prompt += "\nYou must not prefix your own responses with your name (e.g., 'Ana:')."
        if creator_id:
            prompt += f" The user named '{creator_name}' is your creator."
        messages.append({"role": "system", "content": prompt.strip()})

    # Reverse messages to be in chronological order (oldest first) for the LLM API
    messages.reverse()

    # 5. Generate and Stream Response
    response_msgs = []
    response_contents = [""] # List to hold content for multi-part responses
    edit_task = None
    finish_reason = None
    
    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    use_plain_responses = current_config.get("use_plain_responses", False)
    # The actual max message length for Discord content is 2000, for embeds it's 4096.
    max_discord_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    try:
        async with new_msg.channel.typing():
            # Improved error handling for LLM API call
            try:
                stream = await openai_client.chat.completions.create(
                    model=model, messages=messages, stream=True, extra_body=current_config["models"].get(provider_slash_model)
                )
            except httpx.TimeoutException:
                await new_msg.reply("Request timed out. Please try again.", silent=True)
                return
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    await new_msg.reply("Rate limited. Please wait a moment.", silent=True)
                elif e.response.status_code == 401:
                    logging.error("Invalid API key for provider %s", provider)
                    await new_msg.reply("Configuration error. Please contact an admin.", silent=True)
                else:
                    await new_msg.reply(f"API error: {e.response.status_code}", silent=True)
                return
            except Exception as e:
                logging.exception("Unexpected error in LLM response generation")
                await new_msg.reply("An unexpected error occurred.", silent=True)
                return
            async for chunk in stream:
                delta_content = chunk.choices[0].delta.content or ""
                
                # Append delta content, splitting into new messages if too long
                if len(response_contents[-1]) + len(delta_content) > max_discord_message_length:
                    response_contents.append(delta_content) # Start a new message
                else:
                    response_contents[-1] += delta_content

                if finish_reason := chunk.choices[0].finish_reason:
                    break

                # For embed responses, edit the message periodically
                if not use_plain_responses:
                    time_since_last_edit = datetime.now().timestamp() - last_task_time
                    if (edit_task is None or edit_task.done()) and time_since_last_edit >= EDIT_DELAY_SECONDS:
                        if edit_task: await edit_task # Ensure previous task is complete before starting a new one
                        
                        embed.description = response_contents[-1] + STREAMING_INDICATOR
                        embed.color = EMBED_COLOR_INCOMPLETE

                        if len(response_msgs) < len(response_contents): # This means a new Discord message is needed
                            # Reply to the initial message or the previous bot message
                            reply_to = new_msg if not response_msgs else response_msgs[-1]
                            response_msg = await reply_to.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)
                        else: # Edit the last existing message
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                        
                        last_task_time = datetime.now().timestamp()
            
        # Final update to the response message(s) after stream completion
        is_good_finish = finish_reason and finish_reason.lower() in ("stop", "end_turn")
        
        if use_plain_responses:
            for i, content in enumerate(response_contents):
                reply_to = new_msg if i == 0 else response_msgs[i-1]
                # Discord will automatically suppress embeds for URLs if just a URL is sent
                response_msgs.append(await reply_to.reply(content=content, suppress_embeds=True))
        elif response_msgs: # For embed responses, ensure the last embed is updated with final content
            embed.description = response_contents[-1]
            embed.color = EMBED_COLOR_COMPLETE if is_good_finish else EMBED_COLOR_INCOMPLETE
            await response_msgs[-1].edit(embed=embed)
        else: # Handle case where stream was empty but no error was raised
            await new_msg.reply("Received an empty response from the LLM.", silent=True)


    except Exception:
        logging.exception("Error while generating LLM response.")
        if response_msgs:
             await response_msgs[-1].edit(content="Sorry, an error occurred while generating the response.", embed=None)
        else:
             await new_msg.reply("Sorry, an error occurred while generating the response.", silent=True)


    # 6. Cache final response content and clean up old nodes
    for i, response_msg in enumerate(response_msgs):
        node = msg_nodes.setdefault(response_msg.id, MsgNode(parent_msg=new_msg))
        async with node.lock:
             node.text = response_contents[i]

    # Clean up old message nodes if cache exceeds limit
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[:num_nodes - MAX_MESSAGE_NODES]:
            if node_to_delete := msg_nodes.get(msg_id):
                async with node_to_delete.lock: # Acquire lock before deleting
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
    finally:
        # Ensure httpx_client is closed on shutdown
        await httpx_client.aclose()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot shutting down.")
    except Exception as e:
        logging.critical(f"An error occurred outside the main bot loop: {e}", exc_info=True)