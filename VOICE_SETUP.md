# Voice Features Setup Guide

This guide will help you set up the voice features for your Discord LLM bot, including Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities.

## Features Added

- **Text-to-Speech (TTS)**: The bot can speak responses using Google's free TTS service
- **Speech-to-Text (STT)**: The bot can listen to voice input using OpenAI's Whisper model
- **Voice Commands**: Easy-to-use commands for voice interactions
- **LLM Integration**: Voice input is processed through your configured LLM

## Commands Available

- `/voice` - Start voice conversation with automatic STT/TTS
- `/voice_stop` - Stop voice conversation and leave channel
- `/voice_speak <text>` - Make bot speak specific text
- `/voice_language <code>` - Set TTS language (e.g., `/voice_language es`)
- `/voice_status` - Check voice conversation status
- `/voice_help` - Show voice commands help

## Language Support

### Speech-to-Text (STT)
- **Automatic Language Detection**: Whisper automatically detects and transcribes any language
- **No Configuration Needed**: Works out of the box with 99+ languages
- **High Accuracy**: Supports languages like English, Spanish, French, German, Chinese, Japanese, Arabic, and many more

### Text-to-Speech (TTS)
- **Multiple Languages**: Supports 20+ languages
- **Easy Configuration**: Use `/voice_language <code>` to change
- **Common Language Codes**:
  - `en` - English, `es` - Spanish, `fr` - French, `de` - German
  - `it` - Italian, `pt` - Portuguese, `ru` - Russian, `ja` - Japanese
  - `ko` - Korean, `zh` - Chinese, `ar` - Arabic, `hi` - Hindi
  - `nl` - Dutch, `sv` - Swedish, `no` - Norwegian, `da` - Danish
  - `fi` - Finnish, `pl` - Polish, `tr` - Turkish, `he` - Hebrew

## Installation

### 1. Install Dependencies

The voice features require additional Python packages. Install them using:

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg

FFmpeg is required for audio processing. Install it based on your operating system:

**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to your system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**CentOS/RHEL:**
```bash
sudo yum install ffmpeg
```

### 3. Verify Installation

Run the bot to check if everything is working:

```bash
python bot.py
```

You should see messages indicating:
- "VoiceLLM Cog ready"
- "Whisper model loaded successfully" (if Whisper loads properly)

## Usage

### Basic Voice Conversation

1. Join a voice channel in Discord
2. Type `/voice` in any text channel
3. The bot will join your voice channel and start listening
4. Speak naturally in any language - Whisper will auto-detect it
5. The bot will respond with voice and text

### Multi-Language Usage

1. **Start voice conversation**: `/voice`
2. **Set response language**: `/voice_language es` (for Spanish)
3. **Speak in any language**: Whisper auto-detects
4. **Bot responds in selected language**: TTS uses the language you set

**Example:**
- You speak in French → Whisper transcribes to French text
- You set `/voice_language fr` → Bot responds in French
- You speak in Japanese → Whisper transcribes to Japanese text  
- You set `/voice_language ja` → Bot responds in Japanese

### Manual TTS

Use `/voice_speak` to make the bot speak specific text:

```
/voice_speak Hello, this is a test message
```

### Check Status

Use `/voice_status` to see the current state of voice features.

## Configuration

### Whisper Model

The bot uses the "tiny" Whisper model by default for faster processing. You can modify this in `cogs/voice_llm.py`:

```python
self.whisper_model = whisper.load_model("tiny")  # Change to "base", "small", "medium", or "large"
```

### TTS Language

The TTS is set to English by default. You can change this in `cogs/voice_llm.py`:

```python
tts = gTTS(text=text, lang='en', slow=False)  # Change 'en' to other language codes
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Make sure FFmpeg is installed and in your PATH
2. **Whisper model fails to load**: Check your internet connection and disk space
3. **TTS not working**: Ensure you have internet access for Google TTS
4. **Bot can't join voice**: Check bot permissions in Discord

### Error Messages

- **"Failed to load Whisper model"**: The model download failed. Check your internet connection.
- **"Failed to join voice channel"**: Bot lacks voice permissions or channel is full.
- **"Error in TTS"**: Google TTS service is unavailable or text is invalid.

### Performance Tips

1. Use the "tiny" Whisper model for faster STT processing
2. Keep voice responses concise for better TTS quality
3. Ensure stable internet connection for TTS generation

## Advanced Setup

### Custom Audio Processing

For advanced users, you can modify the audio processing in `cogs/voice_llm.py`:

- Change audio format and quality
- Implement custom voice activity detection
- Add audio effects or filters

### Integration with Other Services

You can replace Google TTS with other services:

- **Azure Speech Services**: Higher quality but requires API key
- **Amazon Polly**: Good quality with various voices
- **Local TTS**: Use local models for privacy

### Whisper Alternatives

You can replace Whisper with other STT services:

- **Azure Speech Services**: Better accuracy, requires API key
- **Google Speech-to-Text**: Good accuracy, requires API key
- **Local models**: Use local STT for privacy

## Security Considerations

- Voice data is processed locally with Whisper
- TTS requests go to Google's servers
- Consider privacy implications for voice conversations
- Review Discord's voice data policies

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the bot logs for error messages
3. Ensure all dependencies are properly installed
4. Verify Discord bot permissions

## Future Enhancements

Potential improvements for the voice features:

- Real-time voice activity detection
- Multiple voice channels support
- Voice command recognition
- Custom voice models
- Audio recording and playback
- Voice channel moderation features 