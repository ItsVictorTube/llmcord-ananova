import asyncio
import logging
import discord
from discord.ext import commands
import io
import wave
import numpy as np
from gtts import gTTS
import tempfile
import os
from typing import Optional
import whisper
import torch

class Voice(commands.Cog):
    """Cog for handling voice interactions with STT and TTS."""
    
    def __init__(self, bot):
        self.bot = bot
        self.voice_clients = {}  # Store voice clients per guild
        self.audio_queue = {}    # Queue for TTS audio per guild
        self.is_speaking = {}    # Track speaking state per guild
        self.whisper_model = None
        self.setup_whisper()
        
    def setup_whisper(self):
        """Initialize Whisper model for STT."""
        try:
            # Use the smallest model for faster processing
            self.whisper_model = whisper.load_model("tiny")
            logging.info("Whisper model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
    
    @commands.Cog.listener()
    async def on_ready(self):
        logging.info("Voice Cog ready")
    
    @commands.command(name="join", description="Join a voice channel")
    async def join_voice(self, ctx):
        """Join the user's voice channel."""
        if not ctx.author.voice:
            await ctx.send("You need to be in a voice channel first!")
            return
            
        channel = ctx.author.voice.channel
        try:
            voice_client = await channel.connect()
            self.voice_clients[ctx.guild.id] = voice_client
            self.audio_queue[ctx.guild.id] = asyncio.Queue()
            self.is_speaking[ctx.guild.id] = False
            
            # Start the audio processing task
            asyncio.create_task(self.process_audio_queue(ctx.guild.id))
            
            await ctx.send(f"Joined {channel.name}!")
        except Exception as e:
            logging.error(f"Failed to join voice channel: {e}")
            await ctx.send("Failed to join the voice channel.")
    
    @commands.command(name="leave", description="Leave the voice channel")
    async def leave_voice(self, ctx):
        """Leave the voice channel."""
        if ctx.guild.id not in self.voice_clients:
            await ctx.send("I'm not in a voice channel!")
            return
            
        voice_client = self.voice_clients[ctx.guild.id]
        await voice_client.disconnect()
        
        # Clean up
        del self.voice_clients[ctx.guild.id]
        if ctx.guild.id in self.audio_queue:
            del self.audio_queue[ctx.guild.id]
        if ctx.guild.id in self.is_speaking:
            del self.is_speaking[ctx.guild.id]
            
        await ctx.send("Left the voice channel!")
    
    @commands.command(name="speak", description="Make the bot speak text")
    async def speak_text(self, ctx, *, text: str):
        """Convert text to speech and play it in voice channel."""
        if ctx.guild.id not in self.voice_clients:
            await ctx.send("I need to be in a voice channel first! Use `/join`")
            return
            
        await self.add_to_tts_queue(ctx.guild.id, text)
        await ctx.send(f"Added to speech queue: {text[:50]}{'...' if len(text) > 50 else ''}")
    
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
            # Generate TTS audio
            tts = gTTS(text=text, lang='en', slow=False)
            
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
    async def on_voice_state_update(self, member, before, after):
        """Handle voice state updates for automatic STT processing."""
        # Only process if it's a user speaking (not the bot)
        if member.bot:
            return
            
        # Check if user started speaking
        if not before.self_mute and after.self_mute:
            return  # User muted themselves
            
        if before.self_deaf and not after.self_deaf:
            return  # User undeafened themselves
            
        # If user is in a channel with the bot and started speaking
        if (after.channel and 
            after.channel.guild.id in self.voice_clients and
            self.voice_clients[after.channel.guild.id].channel == after.channel):
            
            # Start listening for speech
            await self.start_speech_recognition(after.channel.guild.id, member)
    
    async def start_speech_recognition(self, guild_id: int, member):
        """Start listening for speech from a specific member."""
        if guild_id not in self.voice_clients:
            return
            
        voice_client = self.voice_clients[guild_id]
        
        # This is a simplified version - in a real implementation,
        # you'd need to capture audio from the voice client
        # For now, we'll use a placeholder that responds to text commands
        
        # You can extend this by implementing actual audio capture
        # using discord.py's voice receive functionality
        
        logging.info(f"Started listening for speech from {member.display_name}")
    
    async def process_speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """Convert speech audio to text using Whisper."""
        if not self.whisper_model:
            return None
            
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            return result["text"].strip()
            
        except Exception as e:
            logging.error(f"Error in speech recognition: {e}")
            return None
    
    @commands.command(name="listen", description="Toggle speech recognition")
    async def toggle_listening(self, ctx):
        """Toggle speech recognition on/off."""
        if ctx.guild.id not in self.voice_clients:
            await ctx.send("I need to be in a voice channel first! Use `/join`")
            return
            
        # This would toggle speech recognition
        # For now, just acknowledge the command
        await ctx.send("Speech recognition toggled! (Feature in development)")
    
    @commands.command(name="stop", description="Stop current speech")
    async def stop_speech(self, ctx):
        """Stop the current speech."""
        if ctx.guild.id not in self.voice_clients:
            await ctx.send("I'm not in a voice channel!")
            return
            
        voice_client = self.voice_clients[ctx.guild.id]
        if voice_client.is_playing():
            voice_client.stop()
            await ctx.send("Stopped speaking!")
        else:
            await ctx.send("I'm not speaking right now!")

async def setup(bot):
    await bot.add_cog(Voice(bot)) 