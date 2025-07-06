import asyncio
import logging
import discord
from discord.ext import commands
from discord import app_commands, Interaction
import io
import wave
import numpy as np
from gtts import gTTS
import tempfile
import os
from typing import Optional, Dict, Any
import whisper
import torch
from datetime import datetime
import discord.ext.voice_recv as voice_recv
import time
from utils.config import get_config

class VoiceLLM(commands.Cog):
    """Enhanced voice cog that integrates with LLM for automatic conversations."""
    
    def __init__(self, bot):
        self.bot = bot
        self.voice_clients = {}  # Store voice clients per guild
        self.audio_queue = {}    # Queue for TTS audio per guild
        self.is_speaking = {}    # Track speaking state per guild
        self.whisper_model = None
        self.llm_cog = None
        self.voice_conversations = {}  # Track active voice conversations
        self.language_settings = {}    # Track language per guild
        self.is_transcribing = {}  # Track Whisper transcription state per guild
        self.transcription_memory = {}  # Store recent transcriptions per guild
        self.setup_whisper()
        
    def setup_whisper(self):
        """Initialize Whisper model for STT."""
        try:
            config = get_config()
            # Safely get the model value from config['voice']['stt']['model']
            model_size = (
                config.get('voice', {})
                      .get('stt', {})
                      .get('model', 'tiny')
            )
            self.whisper_model = whisper.load_model(model_size)
            logging.info(f"Whisper model loaded successfully ({model_size})")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
        
        # Clear any existing audio buffers on startup
        self.clear_audio_buffers_on_startup()
    
    def clear_audio_buffers_on_startup(self):
        """Clear any existing audio buffers when the bot starts up."""
        # Clear any existing audio buffers in voice clients
        for guild_id in list(self.voice_clients.keys()):
            voice_client = self.voice_clients.get(guild_id)
            if voice_client and hasattr(voice_client, '_sink') and voice_client._sink:
                if hasattr(voice_client._sink, 'audio_buffers'):
                    voice_client._sink.audio_buffers.clear()
                    logging.info(f"Cleared audio buffers for guild {guild_id} on startup")
                if hasattr(voice_client._sink, 'last_audio_time'):
                    voice_client._sink.last_audio_time.clear()
                if hasattr(voice_client._sink, 'processing'):
                    voice_client._sink.processing.clear()
        
        # Also clear transcription memory and state
        self.transcription_memory.clear()
        self.is_transcribing.clear()
        
        logging.info("Audio buffers cleared on startup")
    
    def clear_guild_audio_buffers(self, guild_id: int):
        """Clear audio buffers for a specific guild."""
        voice_client = self.voice_clients.get(guild_id)
        if voice_client and hasattr(voice_client, '_sink') and voice_client._sink:
            if hasattr(voice_client._sink, 'audio_buffers'):
                voice_client._sink.audio_buffers.clear()
                logging.info(f"Cleared audio buffers for guild {guild_id}")
            if hasattr(voice_client._sink, 'last_audio_time'):
                voice_client._sink.last_audio_time.clear()
            if hasattr(voice_client._sink, 'processing'):
                voice_client._sink.processing.clear()
        
        # Clear transcription state for this guild
        if guild_id in self.is_transcribing:
            self.is_transcribing[guild_id] = False
        if guild_id in self.transcription_memory:
            del self.transcription_memory[guild_id]
    
    @commands.Cog.listener()
    async def on_ready(self):
        logging.info("VoiceLLM Cog ready")
        # Get reference to LLM cog
        self.llm_cog = self.bot.get_cog('LLM')
    
    @app_commands.command(name="voice", description="Start voice conversation with the bot")
    async def start_voice_conversation(self, interaction: Interaction):
        """Start a voice conversation with automatic STT and TTS."""
        if not interaction.guild:
            await interaction.response.send_message("This command can only be used in a server (not in DMs).")
            return
        member = interaction.guild.get_member(interaction.user.id)
        if not member or not member.voice or not member.voice.channel:
            await interaction.response.send_message("You need to be in a voice channel first!")
            return
        channel = member.voice.channel
        guild_id = interaction.guild.id
        # Check bot permissions
        bot_member = interaction.guild.get_member(self.bot.user.id)
        if not bot_member:
            await interaction.response.send_message("❌ Bot member not found in guild.")
            return
        bot_permissions = channel.permissions_for(bot_member)
        required_permissions = [
            ('connect', bot_permissions.connect),
            ('speak', bot_permissions.speak),
            ('use_voice_activation', bot_permissions.use_voice_activation),
            ('view_channel', bot_permissions.view_channel)
        ]
        missing_permissions = [name for name, has_perm in required_permissions if not has_perm]
        if missing_permissions:
            await interaction.response.send_message(f"❌ Bot lacks required permissions: {', '.join(missing_permissions)}")
            return
        # Check if already connected to a voice channel in this guild
        if guild_id in self.voice_clients:
            await interaction.response.send_message("I'm already in a voice channel! Use `/voice_stop` first.")
            return
        await interaction.response.defer(thinking=True)
        try:
            logging.info(f"Attempting to connect to voice channel: {channel.name}")
            voice_client = await asyncio.wait_for(
                channel.connect(timeout=20.0, self_deaf=False, self_mute=False, cls=voice_recv.VoiceRecvClient),
                timeout=25.0
            )
            logging.info(f"Successfully connected to voice channel: {channel.name}")
            if not voice_client.is_connected():
                raise Exception("Voice client is not connected after connection attempt")
            # Store voice client and setup
            self.voice_clients[guild_id] = voice_client
            self.audio_queue[guild_id] = asyncio.Queue()
            self.is_speaking[guild_id] = False
            self.voice_conversations[guild_id] = {
                'active': True,
                'channel': interaction.channel,
                'last_speech': datetime.now()
            }
            self.language_settings[guild_id] = 'en'
            asyncio.create_task(self.process_audio_queue(guild_id))
            
            # Clear any existing audio buffers for this guild
            self.clear_guild_audio_buffers(guild_id)
            
            # Start listening for audio from all users except the bot
            sink = self.UserAudioSink(self, guild_id)
            voice_client.listen(sink)
            welcome_text = "Hello! I'm ready for voice conversation. Just speak naturally and I'll respond!"
            await self.add_to_tts_queue(guild_id, welcome_text)
            await interaction.followup.send(f"🎤 Voice conversation started in {channel.name}! Just speak naturally.\nUse `/voice_language <code>` to change language (e.g., `/voice_language es` for Spanish)")
        except asyncio.TimeoutError:
            await interaction.followup.send("❌ Voice connection timed out. Please try again.")
        except Exception as e:
            logging.error(f"Failed to start voice conversation: {e}")
            try:
                await interaction.followup.send("❌ Failed to start voice conversation. Please check bot permissions and try again.")
            except Exception as followup_error:
                logging.error(f"Failed to send followup message: {followup_error}")
    
    @app_commands.command(name="voice_language", description="Set the language for voice conversation")
    @app_commands.describe(language_code="Language code (e.g., 'en' for English, 'es' for Spanish)")
    async def set_voice_language(self, interaction: Interaction, language_code: str):
        if not interaction.guild:
            await interaction.response.send_message("This command can only be used in a server (not in DMs).")
            return
        guild_id = interaction.guild.id
        if guild_id not in self.voice_clients:
            await interaction.response.send_message("No active voice conversation! Use `/voice` first.")
            return
        language_codes = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'nl': 'Dutch', 'sv': 'Swedish', 'no': 'Norwegian', 'da': 'Danish',
            'fi': 'Finnish', 'pl': 'Polish', 'tr': 'Turkish', 'he': 'Hebrew'
        }
        if language_code.lower() in language_codes:
            self.language_settings[guild_id] = language_code.lower()
            language_name = language_codes[language_code.lower()]
            await interaction.response.send_message(f"🌍 Language set to {language_name} ({language_code})")
        else:
            await interaction.response.send_message(f"❌ Unsupported language code: {language_code}\nSupported codes: {', '.join(language_codes.keys())}")
    
    @app_commands.command(name="voice_stop", description="Stop voice conversation")
    async def stop_voice_conversation(self, interaction: Interaction):
        if not interaction.guild:
            await interaction.response.send_message("This command can only be used in a server (not in DMs).")
            return
        guild_id = interaction.guild.id
        if guild_id not in self.voice_clients:
            await interaction.response.send_message("No active voice conversation!")
            return
        
        # Clean up voice conversation
        if guild_id in self.voice_conversations:
            self.voice_conversations[guild_id]['active'] = False
            del self.voice_conversations[guild_id]
        
        # Clean up voice client and sink
        voice_client = self.voice_clients[guild_id]
        if hasattr(voice_client, '_sink') and voice_client._sink:
            voice_client._sink.cleanup()
        await voice_client.disconnect()
        del self.voice_clients[guild_id]
        
        # Clean up other data
        if guild_id in self.audio_queue:
            del self.audio_queue[guild_id]
        if guild_id in self.is_speaking:
            del self.is_speaking[guild_id]
        if guild_id in self.language_settings:
            del self.language_settings[guild_id]
        
        # Unload Whisper model and clear memory
        self.unload_whisper_model()
        self.clear_transcription_memory()
        
        await interaction.response.send_message("Voice conversation stopped!")
    
    @app_commands.command(name="voice_speak", description="Make the bot speak text in voice")
    @app_commands.describe(text="The text you want the bot to speak")
    async def speak_text(self, interaction: Interaction, text: str):
        if not interaction.guild:
            await interaction.response.send_message("This command can only be used in a server (not in DMs).")
            return
        guild_id = interaction.guild.id
        if guild_id not in self.voice_clients:
            await interaction.response.send_message("I need to be in a voice channel first! Use `/voice`")
            return
        await self.add_to_tts_queue(guild_id, text)
        await interaction.response.send_message(f"🗣️ Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    async def add_to_tts_queue(self, guild_id: int, text: str):
        """Add text to the TTS queue."""
        if guild_id in self.audio_queue:
            await self.audio_queue[guild_id].put(text)
    
    async def process_audio_queue(self, guild_id: int):
        """Process the TTS audio queue."""
        while guild_id in self.audio_queue:
            try:
                if self.audio_queue[guild_id].empty():
                    await asyncio.sleep(0.1)
                    continue
                    
                text = await self.audio_queue[guild_id].get()
                await self.speak_text_async(guild_id, text)
                
            except Exception as e:
                logging.error(f"Error processing audio queue: {e}")
                await asyncio.sleep(1)
    
    async def speak_text_async(self, guild_id: int, text: str):
        """Convert text to speech and play it."""
        if guild_id not in self.voice_clients or self.is_speaking.get(guild_id, False):
            return
            
        self.is_speaking[guild_id] = True
        voice_client = self.voice_clients[guild_id]
        
        try:
            # Get language setting for this guild
            language = self.language_settings.get(guild_id, 'en')
            
            # Generate TTS audio in the specified language
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                tts.save(temp_file.name)
                temp_path = temp_file.name
            
            # Play the audio
            voice_client.play(discord.FFmpegPCMAudio(temp_path))
            
            # Wait for audio to finish
            while voice_client.is_playing():
                await asyncio.sleep(0.1)
                
            # Clean up temporary file
            os.unlink(temp_path)
            
        except Exception as e:
            logging.error(f"Error in TTS: {e}")
        finally:
            self.is_speaking[guild_id] = False
    
    @commands.Cog.listener()
    async def on_voice_client_error(self, voice_client, error):
        """Handle voice client errors."""
        guild_id = voice_client.guild.id
        logging.error(f"Voice client error in guild {guild_id}: {error}")
        
        # Clean up on error
        if guild_id in self.voice_clients:
            del self.voice_clients[guild_id]
        if guild_id in self.voice_conversations:
            del self.voice_conversations[guild_id]
        if guild_id in self.audio_queue:
            del self.audio_queue[guild_id]
        if guild_id in self.is_speaking:
            del self.is_speaking[guild_id]
        if guild_id in self.language_settings:
            del self.language_settings[guild_id]
    
    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        """Handle voice state updates for automatic STT processing."""
        # Only process if it's a user speaking (not the bot)
        if member.bot:
            return
            
        guild_id = after.channel.guild.id if after.channel else None
        if not guild_id or guild_id not in self.voice_conversations:
            return
            
        # Check if voice conversation is active
        if not self.voice_conversations[guild_id]['active']:
            return
            
        # Check if user is speaking (simplified detection)
        # In a real implementation, you'd use voice activity detection
        if (after.channel and 
            after.channel.guild.id in self.voice_clients and
            self.voice_clients[after.channel.guild.id].channel == after.channel):
            
            # Simulate speech detection (you'd replace this with actual audio capture)
            await self.handle_voice_input(guild_id, member)
    
    async def handle_voice_input(self, guild_id: int, member):
        # This function is now obsolete and will be removed.
        return
    
    async def capture_and_transcribe_speech(self, guild_id: int) -> Optional[str]:
        # This function is now obsolete and will be removed.
        return None
    
    async def process_voice_message(self, guild_id: int, member, text: str):
        """Process a voice message through the LLM and respond with speech."""
        if not self.llm_cog or not self.voice_conversations[guild_id]['active']:
            return
            
        try:
            # Create a simulated message for the LLM
            # In a real implementation, you'd create a proper Discord message object
            simulated_message = self.create_simulated_message(member, text, guild_id)
            
            # Get LLM response (this would need to be adapted from the LLM cog)
            response_text = await self.get_llm_response(simulated_message, text)
            
            if response_text:
                # Speak the response
                await self.add_to_tts_queue(guild_id, response_text)
                
                # Also send to text channel for reference
                if guild_id in self.voice_conversations:
                    channel = self.voice_conversations[guild_id]['channel']
                    await channel.send(f"**{member.display_name}:** {text}\n**Bot:** {response_text}")
                    
        except Exception as e:
            logging.error(f"Error processing voice message: {e}")
    
    def create_simulated_message(self, member, text: str, guild_id: int):
        """Create a simulated Discord message for LLM processing."""
        # This is a simplified approach - in a real implementation,
        # you'd need to properly integrate with the LLM cog's message processing
        return {
            'author': member,
            'content': text,
            'guild': member.guild,
            'channel': self.voice_conversations[guild_id]['channel'] if guild_id in self.voice_conversations else None
        }
    
    async def get_llm_response(self, message, text: str) -> Optional[str]:
        """Get a response from the LLM for voice input."""
        if not self.llm_cog:
            return "LLM cog not available."
            
        try:
            # Get the LLM response using the new method
            response = await self.llm_cog.generate_voice_response(
                user_message=text,
                user_name=message['author'].display_name,
                channel=message['channel']
            )
            return response
        except Exception as e:
            logging.error(f"Error getting LLM response: {e}")
            return "I'm sorry, I encountered an error while processing your message."
    
    @app_commands.command(name="voice_status", description="Check voice conversation status")
    async def voice_status(self, interaction: Interaction):
        if not interaction.guild:
            await interaction.response.send_message("This command can only be used in a server (not in DMs).")
            return
        guild_id = interaction.guild.id
        if guild_id not in self.voice_clients:
            await interaction.response.send_message("No active voice conversation.")
            return
        status = self.voice_conversations.get(guild_id, {})
        is_active = status.get('active', False)
        is_speaking = self.is_speaking.get(guild_id, False)
        language = self.language_settings.get(guild_id, 'en')
        language_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi'
        }
        is_transcribing = self.is_transcribing.get(guild_id, False)
        status_text = f"Voice conversation: {'🟢 Active' if is_active else '🔴 Inactive'}\n"
        status_text += f"Currently speaking: {'🟡 Yes' if is_speaking else '⚪ No'}\n"
        status_text += f"Language: 🌍 {language_names.get(language, language)}\n"
        status_text += f"Whisper model: {'🟢 Loaded' if self.whisper_model else '🔴 Not loaded'}\n"
        status_text += f"Transcribing: {'🟢 Yes' if is_transcribing else '⚪ No'}"
        await interaction.response.send_message(status_text)
    
    @app_commands.command(name="voice_test", description="Test basic voice connection")
    async def voice_test(self, interaction: Interaction):
        if not interaction.guild:
            await interaction.response.send_message("This command can only be used in a server (not in DMs).")
            return
        member = interaction.guild.get_member(interaction.user.id)
        if not member or not member.voice or not member.voice.channel:
            await interaction.response.send_message("You need to be in a voice channel first!")
            return
        channel = member.voice.channel
        guild_id = interaction.guild.id
        await interaction.response.defer(thinking=True)
        try:
            logging.info(f"Testing voice connection to: {channel.name}")
            voice_client = await channel.connect(timeout=10.0)
            if voice_client.is_connected():
                await interaction.followup.send(f"✅ Successfully connected to {channel.name}!")
                await voice_client.disconnect()
                await interaction.followup.send("✅ Voice test completed successfully!")
            else:
                await interaction.followup.send("❌ Connected but voice client reports not connected")
        except Exception as e:
            logging.error(f"Voice test error: {e}")
            await interaction.followup.send(f"❌ Voice test failed: {str(e)}")
    
    @app_commands.command(name="voice_help", description="Show voice commands help")
    async def voice_help(self, interaction: Interaction):
        """Show help for voice commands."""
        help_text = """
🎤 **Voice Commands:**

`/voice` - Start voice conversation with automatic STT/TTS
`/voice_stop` - Stop voice conversation and leave channel
`/voice_speak <text>` - Make bot speak specific text
`/voice_language <code>` - Set TTS language (e.g., `/voice_language es`)
`/voice_status` - Check voice conversation status
`/voice_test` - Test basic voice connection
`/voice_help` - Show this help message

**Language Support:**
- STT (Speech-to-Text): Auto-detects any language
- TTS (Text-to-Speech): Supports multiple languages

**Common Language Codes:**
`en` - English, `es` - Spanish, `fr` - French, `de` - German
`it` - Italian, `pt` - Portuguese, `ru` - Russian, `ja` - Japanese
`ko` - Korean, `zh` - Chinese, `ar` - Arabic, `hi` - Hindi

**How it works:**
1. Use `/voice` to join your voice channel
2. Speak in any language - Whisper will auto-detect
3. Use `/voice_language` to set the bot's response language
4. The bot will respond in the selected language
5. Use `/voice_stop` to end the conversation

**Note:** Full speech recognition requires additional setup for audio capture.
        """
        await interaction.response.send_message(help_text)

    def unload_whisper_model(self):
        """Unload the Whisper model to free memory."""
        if self.whisper_model is not None:
            # Clear the model from memory
            del self.whisper_model
            self.whisper_model = None
            logging.info("Whisper model unloaded from memory")
    
    def clear_transcription_memory(self):
        """Clear any stored transcription data."""
        # Clear transcription memory for all guilds
        self.transcription_memory.clear()
        
        # Clear transcription state for all guilds
        self.is_transcribing.clear()
        
        # Clear any stored audio buffers in sinks
        for guild_id in list(self.voice_clients.keys()):
            voice_client = self.voice_clients.get(guild_id)
            if voice_client and hasattr(voice_client, '_sink') and voice_client._sink:
                if hasattr(voice_client._sink, 'audio_buffers'):
                    voice_client._sink.audio_buffers.clear()
                if hasattr(voice_client._sink, 'last_audio_time'):
                    voice_client._sink.last_audio_time.clear()
                if hasattr(voice_client._sink, 'processing'):
                    voice_client._sink.processing.clear()
        
        logging.info("Transcription memory cleared")
    
    class UserAudioSink(voice_recv.AudioSink):
        def __init__(self, cog, guild_id):
            super().__init__()
            self.cog = cog
            self.guild_id = guild_id
            self.audio_buffers = {}  # user_id -> bytearray
            self.last_audio_time = {}  # user_id -> last received time
            self.processing = set()  # user_ids currently being processed
            self.timeout = 0.8  # seconds of silence to consider end of speech
            self._task = asyncio.create_task(self._monitor_silence())

        def wants_opus(self):
            return False  # We want PCM for Whisper

        def write(self, user, data):
            if user is None or user.bot:
                return
            uid = user.id
            now = time.time()
            if uid not in self.audio_buffers:
                self.audio_buffers[uid] = bytearray()
            self.audio_buffers[uid] += data.pcm
            self.last_audio_time[uid] = now
            print(f"[AudioSink] write called for user: {user} ({uid}), buffer size: {len(self.audio_buffers[uid])}")

        async def _monitor_silence(self):
            while True:
                await asyncio.sleep(0.2)
                now = time.time()
                for uid in list(self.audio_buffers.keys()):
                    if uid in self.processing:
                        continue
                    last_time = self.last_audio_time.get(uid, 0)
                    if now - last_time > self.timeout and len(self.audio_buffers[uid]) > 0:
                        pcm_bytes = bytes(self.audio_buffers[uid])
                        self.audio_buffers[uid] = bytearray()
                        self.processing.add(uid)
                        user = None
                        # Try to get the user object from the voice channel
                        voice_client = self.cog.voice_clients.get(self.guild_id)
                        if voice_client and voice_client.channel:
                            for member in voice_client.channel.members:
                                if member.id == uid:
                                    user = member
                                    break
                        if user is not None:
                            asyncio.create_task(self.process_user_audio(user, pcm_bytes, uid))
                        else:
                            print(f"[AudioSink] Could not find user object for uid {uid}")
                            self.processing.discard(uid)

        async def process_user_audio(self, user, pcm_bytes, uid):
            guild_id = self.guild_id
            self.cog.is_transcribing[guild_id] = True
            import tempfile, wave
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                with wave.open(f, 'wb') as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(2)
                    wf.setframerate(48000)
                    wf.writeframes(pcm_bytes)
                wav_path = f.name
            try:
                print(f"[AudioSink] Transcribing audio for user {user} ({uid}), {len(pcm_bytes)} bytes")
                result = await asyncio.to_thread(self.cog.whisper_model.transcribe, wav_path)
                text = result["text"].strip()
                print(f"[AudioSink] Whisper result for {user}: {text}")
                if text:
                    await self.cog.process_voice_message(self.guild_id, user, text)
            except Exception as e:
                logging.error(f"Error in Whisper transcription: {e}")
            finally:
                import os
                os.unlink(wav_path)
                self.cog.is_transcribing[guild_id] = False
                self.processing.discard(uid)

        def cleanup(self):
            self.audio_buffers.clear()
            self.last_audio_time.clear()
            self.processing.clear()
            if hasattr(self, '_task'):
                self._task.cancel()
            logging.info(f"[AudioSink] Cleanup completed for guild {self.guild_id}")

async def setup(bot):
    await bot.add_cog(VoiceLLM(bot)) 