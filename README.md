# MurfDebator - AI-Powered Debate Chamber

A sophisticated, real-time debate platform featuring AI opponents, impartial judging, and professional voice interaction. Experience intellectual discourse in a modern, courtroom-inspired interface.

![MurfDebator Demo](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Latest-orange)

## 🎯 Features

### 🎤 Advanced Voice Integration
- **Real-time Voice Input**: Natural speech-to-text using OpenAI Whisper
- **Multiple Input Methods**: Live voice, manual text, or audio file upload
- **Professional Voice Output**: AI-generated speech using Murf AI
- **Enhanced Feedback**: Clear status indicators and error handling

### ⚖️ Intelligent Debate System
- **AI Opponents**: Powered by advanced language models (OpenAI/Groq)
- **Impartial Judging**: AI-based evaluation with detailed scoring
- **Multi-round Debates**: Configurable debate formats (1-5 rounds)
- **Real-time Scoring**: Live feedback and performance metrics
- **Argument Analysis**: Key point extraction and rebuttal detection

### 🎨 Modern User Interface
- **Professional Design**: Elegant courtroom-inspired interface
- **Real-time Feedback**: Instant transcription and processing
- **Multiple Topics**: Pre-configured and custom debate topics
- **Responsive Layout**: Works seamlessly on desktop and mobile
- **Dark Theme**: Eye-friendly interface with modern aesthetics

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Active internet connection
- Microphone (for voice input)
- Speakers/headphones (for voice output)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MurfDebator.git
   cd MurfDebator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   MURF_API_KEY=your_murf_key_here
   GROQ_API_KEY=your_groq_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 🔑 API Configuration

### Required API Keys

#### 1. OpenAI API Key
- **Purpose**: Voice transcription and AI debate responses
- **Get it**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Cost**: Pay-per-use (Whisper: $0.006/minute, GPT: ~$0.01-0.03 per debate)

#### 2. Murf AI API Key
- **Purpose**: Text-to-speech for AI responses
- **Get it**: [Murf.ai](https://murf.ai/)
- **Cost**: Pay-per-use (~$0.01-0.02 per debate)

#### 3. Groq API Key (Optional)
- **Purpose**: Faster AI responses (alternative to OpenAI)
- **Get it**: [Groq Console](https://console.groq.com/)
- **Cost**: Pay-per-use (often cheaper than OpenAI)

### Environment Setup

**Option 1: .env file (Recommended)**
```env
OPENAI_API_KEY=sk-your-openai-key-here
MURF_API_KEY=your-murf-key-here
GROQ_API_KEY=gsk-your-groq-key-here
```

**Option 2: System Environment Variables**
```bash
# Windows
set OPENAI_API_KEY=sk-your-openai-key-here
set MURF_API_KEY=your-murf-key-here
set GROQ_API_KEY=gsk-your-groq-key-here

# macOS/Linux
export OPENAI_API_KEY=sk-your-openai-key-here
export MURF_API_KEY=your-murf-key-here
export GROQ_API_KEY=gsk-your-groq-key-here
```

## 🎮 How to Use

### Starting Your First Debate

1. **Configure Settings**
   - Open the sidebar and verify your API connections
   - Select your preferred debate topic
   - Adjust the number of rounds (1-5)
   - Enable/disable voice features

2. **Begin the Debate**
   - Click "🚀 Start Debate" to initialize
   - The system will assign positions automatically
   - Wait for your turn indicator

3. **Make Your Argument**
   - **Voice Input**: Click "🎤 Voice Input" and speak clearly
   - **Text Input**: Type your argument in the text area
   - Submit when ready (minimum 20 characters)

4. **Follow the Flow**
   - AI will respond with its argument
   - Judge will evaluate both arguments
   - Scores are updated in real-time
   - Continue through all rounds

### Voice Input Tips

- **Clear Speech**: Speak slowly and enunciate clearly
- **Quiet Environment**: Minimize background noise
- **Microphone Check**: Ensure your mic is working
- **Length**: Aim for 30-60 seconds of speech
- **Content**: Structure your argument logically

### Debate Strategies

- **Opening Arguments**: Present your strongest points first
- **Rebuttals**: Address opponent's claims directly
- **Evidence**: Use facts and examples to support claims
- **Clarity**: Keep arguments concise and focused
- **Adaptation**: Respond to judge's feedback

## 🛠️ Troubleshooting

### Common Issues

#### Voice Input Problems
```bash
# Check microphone permissions
# Windows: Settings > Privacy > Microphone
# macOS: System Preferences > Security & Privacy > Microphone
# Linux: Check pulseaudio/alsa configuration
```

#### API Key Issues
```bash
# Verify your keys are correct
# Check API quotas and billing
# Ensure keys are active and not expired
```

#### Installation Problems
```bash
# Update pip
pip install --upgrade pip

# Clear cache and reinstall
pip cache purge
pip install -r requirements.txt --force-reinstall
```

#### Streamlit Issues
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart the application
streamlit run streamlit_app.py
```

### Error Messages

| Error | Solution |
|-------|----------|
| "OpenAI API key not configured" | Add your OpenAI API key to .env file |
| "Microphone not found" | Check microphone permissions and connections |
| "Audio processing failed" | Try shorter audio or check file format |
| "Form submission error" | Ensure argument meets minimum length (20 chars) |
| "Network timeout" | Check internet connection and try again |

## 📊 Performance & Costs

### Estimated Costs per Debate (3 rounds)
- **OpenAI Whisper**: ~$0.01-0.02 (voice transcription)
- **OpenAI GPT**: ~$0.02-0.05 (AI responses)
- **Murf AI**: ~$0.01-0.03 (voice output)
- **Total**: ~$0.04-0.10 per debate

### Performance Tips
- Use Groq API for faster responses
- Disable voice output to reduce costs
- Keep arguments concise
- Use text input for practice sessions

## 🏗️ Technical Architecture

### Core Components

```
MurfDebator/
├── streamlit_app.py          # Main application interface
├── run.py                    # Debate engine and logic
├── debate_integration.py     # API integrations
├── audio_utils.py           # Voice processing utilities
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Key Technologies
- **Frontend**: Streamlit (Python web framework)
- **AI Models**: OpenAI GPT-4, Groq LLM
- **Voice**: OpenAI Whisper (STT), Murf AI (TTS)
- **Styling**: Custom CSS with modern design
- **State Management**: Streamlit Session State

### API Flow
1. **Voice Input** → OpenAI Whisper → Text
2. **Text Input** → Direct processing
3. **AI Response** → OpenAI/Groq → Text
4. **Voice Output** → Murf AI → Speech
5. **Judging** → AI evaluation → Scores

## 🔧 Development

### Local Development Setup

1. **Fork and clone**
   ```bash
   git clone https://github.com/yourusername/MurfDebator.git
   cd MurfDebator
   ```

2. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install black flake8 pytest
   ```

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Code formatting**
   ```bash
   black .
   flake8 .
   ```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Project Structure
```
MurfDebator/
├── streamlit_app.py          # Main UI application
├── run.py                    # Core debate logic
├── debate_integration.py     # External API integrations
├── audio_utils.py           # Audio processing utilities
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── .env.example            # Environment variables template
└── tests/                  # Test files
    ├── test_debate.py
    ├── test_audio.py
    └── test_integration.py
```

## 📈 Roadmap

### Upcoming Features
- [ ] **Multi-language Support**: Debate in different languages
- [ ] **Team Debates**: Multiple participants per side
- [ ] **Debate Templates**: Pre-structured debate formats
- [ ] **Analytics Dashboard**: Detailed performance metrics
- [ ] **Export Features**: Save debates as PDF/Word documents
- [ ] **Mobile App**: Native iOS/Android applications

### Recent Updates
- ✅ **Enhanced UI**: Modern design with improved animations
- ✅ **Voice Integration**: Seamless speech-to-text functionality
- ✅ **Real-time Scoring**: Live performance evaluation
- ✅ **Multi-round Debates**: Configurable debate formats
- ✅ **Error Handling**: Robust error recovery and user feedback

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

### Getting Help
- 📖 **Documentation**: Check this README and inline code comments
- 🐛 **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/MurfDebator/issues)
- 💬 **Discussions**: Join our [GitHub Discussions](https://github.com/yourusername/MurfDebator/discussions)
- 📧 **Email**: Contact us at support@murfdebator.com

### Community
- 🌟 **Star the repo**: Show your support
- 🔄 **Share**: Tell others about MurfDebator
- 💡 **Suggest**: Propose new features
- 🐛 **Report**: Help improve by reporting issues

---

**Made with ❤️ by the MurfDebator Team**

*Experience the future of intellectual discourse with AI-powered debate technology.*