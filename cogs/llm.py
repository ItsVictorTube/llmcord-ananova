import logging
import discord
from discord.ext import commands
from utils.config import get_config
from utils.http_client import httpx_client
from openai import AsyncOpenAI
from base64 import b64encode
from datetime import datetime
import asyncio

# Model tags that indicate vision capabilities
VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude", "gemini", "2.0", "2.5", "gemma", "llama", "pixtral", "mistral", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_NODES = 500

class MsgNode:
    def __init__(self):
        self.text = None
        self.images = []
        self.role = "assistant"
        self.user_id = None
        self.has_bad_attachments = False
        self.fetch_parent_failed = False
        self.parent_msg = None
        self.lock = asyncio.Lock()

class LLM(commands.Cog):
    """Cog for handling LLM message events."""
    def __init__(self, bot):
        self.bot = bot
        self.msg_nodes = {}
        self.last_task_time = 0
        self.curr_model = None

    @commands.Cog.listener()
    async def on_ready(self):
        config = get_config()
        self.curr_model = next(iter(config["models"]))
        logging.info("LLM Cog ready. Current model: %s", self.curr_model)

    async def generate_voice_response(self, user_message: str, user_name: str, channel) -> str:
        """Generate a response for voice input."""
        try:
            current_config = await asyncio.to_thread(get_config)
            provider_slash_model = self.curr_model
            provider, model = provider_slash_model.split("/", 1)
            provider_config = current_config["providers"][provider]
            base_url = provider_config["base_url"]
            api_key = provider_config.get("api_key", "sk-no-key-required")
            openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            
            # Create messages for the API
            messages = []
            
            # Add system prompt
            if system_prompt := current_config["system_prompt"]:
                now = datetime.now().astimezone()
                prompt = system_prompt.replace("{date}", now.strftime("%B %d, %Y")).replace("{time}", now.strftime("%I:%M %p %Z"))
                prompt += "\nEach user's message is prefixed with their display name."
                prompt += "\nYou must not prefix your own responses with your name (e.g., 'Ana:')."
                creator_id = current_config.get("creator_id")
                creator_name = current_config.get("creator_name", "Creator")
                if creator_id:
                    prompt += f" The user named '{creator_name}' is your creator."
                prompt += "\nThis is a voice conversation, so keep responses concise and natural for speech."
                messages.append({"role": "system", "content": prompt.strip()})
            
            # Add user message
            messages.append({"role": "user", "content": f"{user_name}: {user_message}"})
            
            # Generate response
            response = await openai_client.chat.completions.create(
                model=model, 
                messages=messages, 
                stream=False,
                extra_body=current_config["models"].get(provider_slash_model)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Error generating voice response: {e}")
            return "I'm sorry, I encountered an error while processing your voice message."

    @commands.Cog.listener()
    async def on_message(self, new_msg: discord.Message):
        # Main event handler for processing incoming messages and generating LLM responses.
        # Ignore slash commands and bot messages
        if new_msg.author.bot or new_msg.content.startswith('/'):
            return
        if new_msg.channel.type != discord.ChannelType.private and self.bot.user not in new_msg.mentions:
            return
        current_config = await asyncio.to_thread(get_config)
        if not await self._is_message_authorized(new_msg, current_config):
            return
        provider_slash_model = self.curr_model
        provider, model = provider_slash_model.split("/", 1)
        provider_config = current_config["providers"][provider]
        base_url = provider_config["base_url"]
        api_key = provider_config.get("api_key", "sk-no-key-required")
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
        accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)
        max_text = current_config.get("max_text", 100000)
        max_images = current_config.get("max_images", 5) if accept_images else 0
        max_messages_history = current_config.get("max_messages", 25)
        creator_id = current_config.get("creator_id")
        creator_name = current_config.get("creator_name", "Creator")
        messages = []
        user_warnings = set()
        curr_msg = new_msg
        while curr_msg is not None and len(messages) < max_messages_history:
            curr_node = self.msg_nodes.setdefault(curr_msg.id, MsgNode())
            async with curr_node.lock:
                if curr_node.text is None:
                    cleaned_content = curr_msg.content.removeprefix(self.bot.user.mention).lstrip()
                    good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
                    attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])
                    base_text = "\n".join(
                        ([cleaned_content] if cleaned_content else [])
                        + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                        + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                    )
                    curr_node.role = "assistant" if curr_msg.author == self.bot.user else "user"
                    curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                    curr_node.text = base_text
                    curr_node.images = [
                        dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}") )
                        for att, resp in zip(good_attachments, attachment_responses)
                        if att.content_type.startswith("image")
                    ]
                    curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)
                    try:
                        parent_message_to_fetch = None
                        if curr_msg.reference:
                            parent_message_to_fetch = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(curr_msg.reference.message_id)
                        elif curr_msg.channel.type == discord.ChannelType.public_thread and curr_msg.reference is None:
                            parent_message_to_fetch = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(curr_msg.channel.id)
                        elif (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0]):
                            if prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply) and \
                               prev_msg_in_channel.author in (self.bot.user, curr_msg.author):
                                parent_message_to_fetch = prev_msg_in_channel
                        curr_node.parent_msg = parent_message_to_fetch
                    except (discord.NotFound, discord.HTTPException):
                        logging.exception("Error fetching next message in the chain")
                        curr_node.fetch_parent_failed = True
                api_content = curr_node.text[:max_text]
                if curr_node.images[:max_images]:
                    api_content = ([dict(type="text", text=api_content)] if api_content else []) + curr_node.images[:max_images]
                if api_content:
                    message_for_api = dict(content=api_content, role=curr_node.role)
                    if accept_usernames and curr_node.user_id is not None:
                        message_for_api["name"] = str(curr_node.user_id)
                    messages.append(message_for_api)
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
        if system_prompt := current_config["system_prompt"]:
            now = datetime.now().astimezone()
            prompt = system_prompt.replace("{date}", now.strftime("%B %d, %Y")).replace("{time}", now.strftime("%I:%M %p %Z"))
            prompt += "\nEach user's message is prefixed with their display name."
            prompt += "\nYou must not prefix your own responses with your name (e.g., 'Ana:')."
            if creator_id:
                prompt += f" The user named '{creator_name}' is your creator."
            messages.append({"role": "system", "content": prompt.strip()})
        messages.reverse()
        response_msgs = []
        response_contents = [""]
        edit_task = None
        finish_reason = None
        embed = discord.Embed()
        for warning in sorted(user_warnings):
            embed.add_field(name=warning, value="", inline=False)
        use_plain_responses = current_config.get("use_plain_responses", False)
        max_discord_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))
        try:
            async with new_msg.channel.typing():
                try:
                    stream = await openai_client.chat.completions.create(
                        model=model, messages=messages, stream=True, extra_body=current_config["models"].get(provider_slash_model)
                    )
                except Exception as e:
                    logging.exception("Unexpected error in LLM response generation")
                    await new_msg.reply("An unexpected error occurred.", silent=True)
                    return
                async for chunk in stream:
                    delta_content = chunk.choices[0].delta.content or ""
                    if len(response_contents[-1]) + len(delta_content) > max_discord_message_length:
                        response_contents.append(delta_content)
                    else:
                        response_contents[-1] += delta_content
                    if finish_reason := chunk.choices[0].finish_reason:
                        break
                    if not use_plain_responses:
                        time_since_last_edit = datetime.now().timestamp() - self.last_task_time
                        if (edit_task is None or edit_task.done()) and time_since_last_edit >= EDIT_DELAY_SECONDS:
                            if edit_task: await edit_task
                            embed.description = response_contents[-1] + STREAMING_INDICATOR
                            embed.color = EMBED_COLOR_INCOMPLETE
                            if len(response_msgs) < len(response_contents):
                                reply_to = new_msg if not response_msgs else response_msgs[-1]
                                response_msg = await reply_to.reply(embed=embed, silent=True)
                                response_msgs.append(response_msg)
                            else:
                                edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                            self.last_task_time = datetime.now().timestamp()
                is_good_finish = finish_reason and finish_reason.lower() in ("stop", "end_turn")
                if use_plain_responses:
                    for i, content in enumerate(response_contents):
                        reply_to = new_msg if i == 0 else response_msgs[i-1]
                        response_msgs.append(await reply_to.reply(content=content, suppress_embeds=True))
                elif response_msgs:
                    embed.description = response_contents[-1]
                    embed.color = EMBED_COLOR_COMPLETE if is_good_finish else EMBED_COLOR_INCOMPLETE
                    await response_msgs[-1].edit(embed=embed)
                else:
                    await new_msg.reply("Received an empty response from the LLM.", silent=True)
        except Exception:
            logging.exception("Error while generating LLM response.")
            if response_msgs:
                await response_msgs[-1].edit(content="Sorry, an error occurred while generating the response.", embed=None)
            else:
                await new_msg.reply("Sorry, an error occurred while generating the response.", silent=True)
        for i, response_msg in enumerate(response_msgs):
            node = self.msg_nodes.setdefault(response_msg.id, MsgNode())
            async with node.lock:
                node.text = response_contents[i]
        if (num_nodes := len(self.msg_nodes)) > MAX_MESSAGE_NODES:
            for msg_id in sorted(self.msg_nodes.keys())[:num_nodes - MAX_MESSAGE_NODES]:
                if node_to_delete := self.msg_nodes.get(msg_id):
                    async with node_to_delete.lock:
                        self.msg_nodes.pop(msg_id, None)

    async def _is_message_authorized(self, message: discord.Message, config: dict) -> bool:
        is_dm = message.channel.type == discord.ChannelType.private
        permissions = config["permissions"]
        author = message.author
        user_is_admin = author.id in permissions["users"]["admin_ids"]
        (allowed_user_ids, blocked_user_ids) = (permissions["users"]["allowed_ids"], permissions["users"]["blocked_ids"])
        (allowed_role_ids, blocked_role_ids) = (permissions["roles"]["allowed_ids"], permissions["roles"]["blocked_ids"])
        role_ids = {role.id for role in getattr(author, "roles", [])}
        allow_all_users_or_roles = not allowed_user_ids and (is_dm or not allowed_role_ids)
        is_good_user = user_is_admin or allow_all_users_or_roles or author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
        is_bad_user = not is_good_user or author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)
        (allowed_channel_ids, blocked_channel_ids) = (permissions["channels"]["allowed_ids"], permissions["channels"]["blocked_ids"])
        channel_ids = {message.channel.id, getattr(message.channel, "parent_id", None), getattr(message.channel, "category_id", None)}
        channel_ids.discard(None)
        allow_all_channels = not allowed_channel_ids
        is_good_channel = user_is_admin or (config.get("allow_dms") and is_dm) or (allow_all_channels or any(id in allowed_channel_ids for id in channel_ids))
        is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)
        return not (is_bad_user or is_bad_channel)

async def setup(bot):
    await bot.add_cog(LLM(bot)) 