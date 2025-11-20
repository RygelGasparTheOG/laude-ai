# Laude AI

A simple, trainable AI assistant built with Python that uses fuzzy matching and advanced similarity algorithms to provide intelligent responses. Perfect for learning about AI, natural language processing, and building custom chatbots.

## ✨ Features

- **Advanced Fuzzy Matching**: Uses multiple similarity algorithms including Jaccard similarity, Dice coefficient, Levenshtein distance, and n-gram analysis
- **Trainable**: Easily add new question-answer pairs through the web interface
- **Typo Tolerant**: Handles misspellings and variations in user input
- **Web-Based Interface**: Clean, modern UI for chatting and training
- **Persistent Storage**: Model state saved automatically
- **Export/Import**: Download training data as JSON for backup or sharing
- **No Dependencies**: Runs on pure Python with just standard library

## 🚀 Quick Start

### Prerequisites

- Python 3.6 or higher
- No additional packages required!

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/laude-ai.git
cd laude-ai
```

2. Run the server:
```bash
python3 laude.py
```

3. Open your browser and navigate to:
```
http://localhost:8000
```

That's it! Laude is now running and ready to chat.

## 📖 Usage

### Chatting with Laude

Simply type your message in the input box and press Enter or click "Send". Laude will respond based on its training data.

### Training Laude

1. Click the **"Train Model"** button in the header
2. Enter a question or input in the first field
3. Enter the desired response in the second field
4. Click **"Add Training Data"**
5. Click **"Retrain"** to update the model with new knowledge

### Exporting Data

Click the **"Export Data"** button to download your training data as a JSON file. This is useful for:
- Backing up your training data
- Sharing knowledge bases with others
- Version control of your AI's knowledge

## 🏗️ Project Structure

```
laude-ai/
├── laude.py              # Main Python server and AI logic
├── laude_index.html      # Web interface
├── laude_dataset.json    # Training data
├── laude_model.pkl       # Trained model (auto-generated)
└── README.md            # This file
```

## 🧠 How It Works

Laude uses a sophisticated multi-strategy matching system:

1. **Word-Level Matching**: Finds exact word matches between user input and training data
2. **Fuzzy Word Matching**: Identifies similar words using character n-grams
3. **Character N-Gram Analysis**: Catches typos and variations at the character level
4. **Sequence Matching**: Rewards maintaining word order (bigrams)
5. **Multiple Similarity Metrics**: Combines Jaccard similarity, Dice coefficient, and normalized edit distance

The model scores each potential response using a weighted combination of these techniques and returns the best match above a confidence threshold.

## 📊 Default Training Data

Laude comes pre-trained with 500+ responses covering:
- Greetings and casual conversation
- Information about Laude itself
- General knowledge topics
- Emotional support and encouragement
- Self-improvement and motivation
- Technology and internet topics
- Professional development
- Life skills and wisdom
- And much more!

## 🔧 Configuration

You can modify these settings in `laude.py`:

- **PORT**: Change the server port (default: 8000)
- **MODEL_FILE**: Location of the trained model (default: laude_model.pkl)
- **TRAINING_FILE**: Location of training data (default: laude_dataset.json)
- **confidence_threshold**: Minimum score to return a match (default: 7.5)

## 🎯 Use Cases

- **Educational**: Learn about NLP, fuzzy matching, and chatbot development
- **Custom Support Bots**: Train Laude on your specific domain knowledge
- **FAQ Systems**: Build automated question-answering systems
- **Personal Assistant**: Create a personalized AI with knowledge about your interests
- **Prototyping**: Quick chatbot prototypes before building production systems

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

- Add more training data to improve responses
- Implement new matching algorithms
- Enhance the web interface
- Add authentication and multi-user support
- Create additional export formats
- Write tests
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built as a learning project to demonstrate practical NLP techniques
- Inspired by the need for simple, understandable AI systems

## 📧 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Contribute improvements via pull requests

## 🔮 Future Enhancements

- [ ] Multi-user support with separate training data
- [ ] Context awareness for follow-up questions
- [ ] Integration with external APIs
- [ ] Support for multiple languages
- [ ] Voice input/output
- [ ] Mobile-responsive design improvements
- [ ] Docker containerization
- [ ] Advanced analytics and insights

---

**Made with ❤️ for learning and experimentation**
