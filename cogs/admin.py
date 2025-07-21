import logging
import discord
from discord.ext import commands
from discord import app_commands, Interaction
from utils.config import get_config
from discord import TextChannel, Thread

class Admin(commands.Cog):
    """Cog for admin commands like model switching and purge."""
    def __init__(self, bot):
        self.bot = bot
        self.curr_model = None

    @commands.Cog.listener()
    async def on_ready(self):
        config = get_config()
        self.curr_model = next(iter(config["models"]))
        logging.info("Admin Cog ready. Current model: %s", self.curr_model)

    async def _switch_model_internal(self, model_name: str, user_id: int) -> str:
        """Internal method to switch the current LLM model."""
        config = get_config()
        if user_id not in config["permissions"]["users"]["admin_ids"]:
            return "You don't have permission to change the model."
        
        if model_name not in config["models"]:
            return f"Model `{model_name}` not found in configuration."

        self.curr_model = model_name
        output = f"Model switched to: `{model_name}`"
        logging.info(f"{output} (by user {user_id})")
        return output

    @app_commands.command(name="model", description="View or switch the current LLM")
    async def model_command(self, interaction: Interaction, model: str) -> None:
        if model == self.curr_model:
            await interaction.response.send_message(f"Current model is already: `{self.curr_model}`", ephemeral=True)
            return
        
        response_message = await self._switch_model_internal(model, interaction.user.id)
        await interaction.response.send_message(response_message, ephemeral=True)

    @model_command.autocomplete("model")
    async def model_autocomplete(self, interaction: Interaction, current_input: str):
        config = get_config()
        if current_input == "":
            config = await interaction.client.loop.run_in_executor(None, get_config)
        filtered_models = [m for m in config["models"] if current_input.lower() in m.lower()]
        choices = [
            app_commands.Choice(name=f"○ {m}", value=str(m)) for m in filtered_models if m != self.curr_model and m is not None
        ][:24]
        if self.curr_model and self.curr_model in filtered_models:
            choices.append(app_commands.Choice(name=f"◉ {self.curr_model} (current)", value=str(self.curr_model)))
        return choices

    @app_commands.command(name="purge", description="Admin: Deletes the bot's recent messages in this channel.")
    @app_commands.describe(limit="The number of bot messages to delete (default: 1, max: 100).")
    @app_commands.allowed_contexts(dms=True, guilds=True, private_channels=True)
    async def purge_command(self, interaction: Interaction, limit: int = 1) -> None:
        config = get_config()
        if interaction.user.id not in config["permissions"]["users"]["admin_ids"]:
            await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        limit = max(1, min(100, limit))
        messages_to_delete = []
        channel = interaction.channel
        # Only proceed if channel supports history
        if not isinstance(channel, (TextChannel, Thread, discord.DMChannel, discord.GroupChannel)):
            await interaction.followup.send("This channel type does not support message deletion.")
            return
        try:
            if isinstance(channel, (TextChannel, Thread)):
                async for message in channel.history(limit=limit * 5):
                    if len(messages_to_delete) >= limit:
                        break
                    if message.author.id == self.bot.user.id:  # type: ignore
                        messages_to_delete.append(message)
            if isinstance(channel, (discord.DMChannel, discord.GroupChannel)):
                for msg in messages_to_delete:
                    await msg.delete()
            elif isinstance(channel, (TextChannel, Thread)) and hasattr(channel, "delete_messages"):
                await channel.delete_messages(messages_to_delete)
            else:
                for msg in messages_to_delete:
                    await msg.delete()
            deleted_count = len(messages_to_delete)
            plural = "s" if deleted_count > 1 else ""
            await interaction.followup.send(f"Successfully deleted my last {deleted_count} message{plural}.")
        except discord.Forbidden:
            channel_id = getattr(channel, "id", "unknown")
            logging.warning(f"Missing permissions to delete messages in channel {channel_id}")
            await interaction.followup.send("I lack the 'Manage Messages' permission to delete messages in this channel.")
        except discord.HTTPException as e:
            logging.error(f"Failed to delete messages: {e}")
            await interaction.followup.send("An error occurred. I might not be able to delete messages older than 14 days.")

async def setup(bot):
    await bot.add_cog(Admin(bot))