# bot.py: Main entry point for the modularized Discord LLM bot
import asyncio
import logging
from utils.config import get_config, validate_config
from utils.http_client import httpx_client
from discord.ext import commands
import discord

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
bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

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
    await bot.load_extension("cogs.voice")
    await bot.load_extension("cogs.voice_llm")
    # Add more cogs as needed

# --- MAIN EXECUTION ---
async def main():
    try:
        await load_cogs()
        await bot.start(config["bot_token"])
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