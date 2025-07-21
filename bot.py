# bot.py: Main entry point for the modularized Discord LLM bot
import asyncio
import logging
from utils.config import get_config, validate_config
from utils.http_client import httpx_client
from discord.ext import commands
import discord
from cogs.voice_llm import VoiceLLM  # Import VoiceLLM for type hinting
from cogs.llm import LLM # Import LLM for type hinting
from cogs.admin import Admin # Import Admin for type hinting
from typing import cast  # Import cast for type hinting
from datetime import datetime  # Import datetime for mock message
import sys # Import sys for stdout flushing

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# --- CONFIGURATION ---
config = get_config()
validate_config(config)

# --- BOT INITIALIZATION ---
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True  # Required for voice connections
activity = discord.CustomActivity(name=(config.get("status_message", "github.com/jakobdylanc/llmcord"))[:128])
bot = commands.Bot(intents=intents, activity=activity, command_prefix="")

@bot.event
async def on_ready():
    """Called when the bot is ready and connected to Discord."""
    logging.info(f"Bot is ready! Logged in as {bot.user}")
    
    # Sync commands after the bot is ready
    try:
        # Check if there's a test guild configured for instant updates
        test_guild_id = config.get("test_guild_id")
        if test_guild_id:
            guild_obj = discord.Object(id=test_guild_id)
            await bot.tree.sync(guild=guild_obj)
            logging.info(f"Synced commands to test guild: {test_guild_id} (DMs will not have commands)")
        else:
            # Sync globally if no test guild is specified (can take up to an hour)
            await bot.tree.sync()
            logging.info("Synced commands globally (will work in DMs, updates can take up to an hour).")
    except Exception as e:
        logging.error(f"Failed to sync commands: {e}")

# --- LOAD COGS ---
async def load_cogs():
    await bot.load_extension("cogs.llm")
    await bot.load_extension("cogs.admin")
    await bot.load_extension("cogs.voice_llm")
    # Add more cogs as needed

# --- TERMINAL COMMAND HANDLING ---
async def handle_terminal_commands():
    """Handles commands entered directly into the terminal."""
    logging.info("Terminal command mode active. Type 'exit' to quit.")
    while True:
        print("> ", end='') # Print prompt without newline
        sys.stdout.flush() # Flush stdout to ensure prompt appears immediately
        command_line = await asyncio.to_thread(input)  # Use asyncio.to_thread to run input in a non-blocking way
        if command_line.lower() == "exit":
            break

        parts = command_line.split(maxsplit=1)
        command_name = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""

        logging.debug(f"Terminal input: command='{command_name}', args='{args_str}'")

        if command_name == "voice_speak":
            voice_llm_cog = cast(VoiceLLM, bot.get_cog('VoiceLLM')) # Explicitly cast to VoiceLLM
            if voice_llm_cog:
                logging.debug("VoiceLLM cog found.")
                try:
                    # Expecting format: voice_speak <guild_id> <text>
                    speak_args = args_str.split(maxsplit=1)
                    guild_id = int(speak_args[0])
                    text_to_speak = speak_args[1]
                    await voice_llm_cog.speak_text_terminal(guild_id, text_to_speak)
                    logging.info(f"Terminal: Issued voice_speak for guild {guild_id} with text: {text_to_speak[:50]}...")
                except (ValueError, IndexError):
                    print("Usage: voice_speak <guild_id> <text>")
                    logging.error("Invalid usage for voice_speak.")
                except Exception as e:
                    logging.error(f"Error in terminal voice_speak: {e}")
            else:
                print("VoiceLLM cog not loaded.")
                logging.warning("VoiceLLM cog not found when trying to execute voice_speak.")
        elif command_name == "llm":
            llm_cog = cast(LLM, bot.get_cog('LLM')) # Explicitly cast to LLM
            if llm_cog:
                logging.debug("LLM cog found.")
                try:
                    response = await llm_cog._get_llm_response_for_terminal(args_str)
                    print(f"[LLM Response]: {response}")
                    logging.info(f"Terminal: LLM command executed. Response: {response[:50]}...")
                except Exception as e:
                    logging.error(f"Error in terminal LLM command: {e}")
            else:
                print("LLM cog not loaded.")
                logging.warning("LLM cog not found when trying to execute llm command.")
        elif command_name == "model":
            admin_cog = cast(Admin, bot.get_cog('Admin')) # Explicitly cast to Admin
            if admin_cog:
                logging.debug("Admin cog found.")
                try:
                    # For terminal, we'll use a dummy user_id (e.g., 0) or a configurable one
                    # For now, let's assume a dummy user_id for terminal commands.
                    response = await admin_cog._switch_model_internal(args_str, 0) # 0 as dummy user_id
                    print(f"[Model Command]: {response}")
                    logging.info(f"Terminal: Model command executed. Response: {response}")
                except Exception as e:
                    logging.error(f"Error in terminal model command: {e}")
            else:
                print("Admin cog not loaded.")
                logging.warning("Admin cog not found when trying to execute model command.")
        else:
            print(f"Unknown command: {command_name} (Terminal command handling for other commands is not fully implemented yet)")
            logging.info(f"Unknown terminal command received: {command_name}")

# --- MAIN EXECUTION ---
async def main():
    try:
        await load_cogs()
        # Run both the bot and the terminal command handler concurrently
        await asyncio.gather(
            bot.start(config["bot_token"]),
            handle_terminal_commands()
        )
    except discord.LoginFailure:
        logging.critical("Failed to log in. Please check your 'bot_token' in the config file.")
    except Exception:
        logging.critical("An unexpected error occurred during bot startup.", exc_info=True)
    finally:
        await httpx_client.aclose()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot shutting down.")
    except Exception as e:
        logging.critical(f"An error occurred outside the main bot loop: {e}", exc_info=True)