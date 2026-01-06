# AI Text Intelligence API

Production-ready FastAPI service for sentiment analysis, keyword extraction, summarization, and semantic search powered by Hugging Face, spaCy, Sentence Transformers, and FAISS.

## Tech Stack
- Python 3.11
- FastAPI + Pydantic
- Transformers (`distilbert-base-uncased-finetuned-sst-2-english`, `t5-small`)
- spaCy (`en_core_web_sm`)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS (in-memory vector index)
- Docker

## Quick Start
### Local
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker
```bash
docker build -t ai-text-api .
docker run -p 8000:8000 ai-text-api
```

Swagger UI: http://localhost:8000/docs

## API
### POST /analyze
Request
```json
{ "text": "FastAPI makes building APIs delightful." }
```
Response
```json
{
  "sentiment": "positive",
  "keywords": ["api", "fastapi", "building"]
}
```

### POST /summarize
Request
```json
{ "text": "Long article text..." }
```
Response
```json
{ "summary": "Concise summary text." }
```

### POST /semantic-search
In-memory semantic search over a small default corpus.

Request
```json
{ "query": "How do I deploy apps in containers?" }
```
Response
```json
{ "results": ["Docker simplifies packaging and deploying applications.", "..."] }
```

## Sample cURL
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Transformers make NLP easier."}'

curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI is a modern, fast web framework for building APIs with Python."}'

curl -X POST http://localhost:8000/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is semantic search?"}'
```

## Notes
- Models are warmed on startup to reduce first-request latency.
- FAISS index is in-memory and preloaded with a small default corpus; adapt `DEFAULT_CORPUS` in `app/embeddings.py` for your data.
- For production, consider GPU-enabled images and persistent vector stores for larger corpora.
