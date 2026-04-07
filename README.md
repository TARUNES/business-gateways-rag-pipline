# Business Gateway RAG API

A FastAPI service that answers questions about JSRS using a simple RAG pipeline.

## Architecture

```
POST /ask  →  Retrieve chunk (FAISS)  →  Generate answer (Ollama)
```

| Layer | Tech |
|---|---|
| API | FastAPI |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (in-memory) |
| LLM | Ollama (local) |

## Requirements

- Python 3.11+
- Poetry (`pip install poetry`) – run as `python -m poetry` if `poetry` isn't on your PATH
- [Ollama](https://ollama.com/) running locally with a model pulled

## Setup

```powershell
# 1. Install dependencies
python -m poetry install

# 2. Copy and edit config (optional)
Copy-Item .env.example .env

# 3. Make sure Ollama is running and a model is available
ollama serve          # in a separate terminal
ollama pull llama3

# 4. Start the server
python -m poetry run uvicorn app.main:app --reload
```

## Usage

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is OCR?\"}"
```

Response:
```js
{
  "answer": "OCR stands for Optical Character Recognition......"
}
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Model name to use |
| `CHUNK_SIZE` | `400` | Document chunk size in characters |
