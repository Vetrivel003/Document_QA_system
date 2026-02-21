# 📚 Document Q&A System

Ask questions about your documents and get instant answers with source citations.

## What Does It Do?

Upload your PDF, TXT, or Word documents, then ask questions in plain English. The system finds relevant information and gives you accurate answers with references to where it found them.

## Features

- 📤 Upload multiple document formats (PDF, TXT, DOCX)
- 💬 Ask questions in natural language
- 🎯 Get accurate answers with source citations
- ⚡ Real-time streaming responses
- 📊 Track your queries and system stats

## Quick Start

### 1. Install Requirements
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the project
git clone <your-repo-url>
cd document-qa-system

# Install dependencies
uv sync
```

### 2. Setup API Key

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key from: https://console.groq.com

### 3. Run the App
```bash
uv run streamlit run app.py
```

Open your browser to `http://localhost:8501`

## How to Use

### Step 1: Upload Documents
1. Go to the "Upload Documents" tab
2. Click "Browse files" or drag & drop your documents
3. Click "Process & Index"
4. Wait for indexing to complete

### Step 2: Ask Questions
1. Go to the "Q&A" tab
2. Type your question in the chat box
3. Press Enter
4. View the answer and click "View sources" for citations

### Example Questions
```
What is the main topic of these documents?
Can you summarize the key points?
What does the document say about [specific topic]?
```

## Project Structure
```
document-qa-system/
├── app.py                 # Main Streamlit application
├── .env                   # Your API keys (create this)
├── pyproject.toml        # Project dependencies
│
├── src/
│   ├── config.py         # Settings
│   ├── document_loader.py # Load PDFs, TXT, DOCX
│   ├── text_processor.py  # Split text into chunks
│   ├── vector_store.py    # Store & search documents
│   └── rag_chain.py       # Answer questions
│
├── tests/                 # Test files
├── data/
│   ├── uploads/          # Your uploaded files go here
│   └── chroma_db/        # Vector database (auto-created)
```

## Settings You Can Adjust

In the sidebar:
- **Documents to Retrieve (k)**: How many text chunks to use (1-10)
  - More = better context but slower
  - Default: 4
- **Temperature**: How creative the answers should be (0.0-1.0)
  - Lower = more factual and precise
  - Higher = more creative
  - Default: 0.1
- **Streaming**: See answers appear word-by-word in real-time

## Troubleshooting

### "No documents indexed"
- Upload documents in the "Upload Documents" tab first

### "Groq API key not found"
- Check that `.env` file exists in project root
- Verify `GROQ_API_KEY=...` is set correctly
- Restart the app

### Blank page when starting
```bash
# Try clearing cache
streamlit cache clear

# Or use a different port
uv run streamlit run app.py --server.port 8502
```

### Slow responses
- Lower the "Documents to Retrieve (k)" setting
- Check your internet connection

## Testing
```bash
# Run all tests
uv run python tests/test_rag_chain.py

# Interactive testing mode
uv run python tests/test_rag_chain.py interactive
```

## Requirements

- Python 3.10 or higher
- Groq API key (free tier available)
- 2GB+ disk space for vector database
- Internet connection (for API calls)

## Tech Stack

- **UI**: Streamlit
- **LLM**: Groq (LLaMA 3.3 70B)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **Framework**: LangChain

## Common Issues

| Problem | Solution |
|---------|----------|
| Import errors | Run `uv sync` again |
| API key errors | Check `.env` file |
| Slow indexing | Upload fewer documents at once |
| Out of memory | Restart the app |

## Tips for Best Results

1. **Upload relevant documents only** - Quality over quantity
2. **Use clear questions** - Be specific about what you want to know
3. **Adjust k value** - Try 3-5 for most questions
4. **Check sources** - Always verify the citations
5. **Keep temperature low** - Use 0.0-0.2 for factual answers

## Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Review the error message in the terminal
3. Try restarting the application

## License

MIT License - Feel free to use and modify

## Author

Built with ❤️ using LangChain, Groq, ChromaDB, and Streamlit