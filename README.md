# CourseMate-AI

A RAG-based AI application powered by Mistral AI that helps users interact with course materials through intelligent document loading, vector search, and conversational AI.

## Features

- Document ingestion from multiple formats (PDF, DOCX, XLSX, PPTX, HTML, Markdown)
- Vector-based semantic search using ChromaDB and sentence-transformers
- LLM-powered Q&A via Mistral AI and LangChain
- FastAPI backend with async support

## Setup

1. Install [uv](https://github.com/astral-sh/uv) if not already installed.
2. Clone the repository and navigate to the project root.
3. Create and activate a virtual environment:
   ```bash
   uv venv --python 3.13
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # macOS/Linux
   ```
4. Install dependencies:
   ```bash
   uv sync
   ```
5. Copy `.env.example` to `.env` and fill in your API keys.

## Usage

```bash
python main.py
```

## License

MIT
