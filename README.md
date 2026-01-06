# AI Text Intelligence API

Production-ready FastAPI service for sentiment analysis, keyword extraction, summarization, and semantic search powered by Hugging Face, spaCy, Sentence Transformers, and FAISS. Now with real-time WebSocket support and dynamic corpus management.

## Tech Stack
- Python 3.11
- FastAPI + Pydantic
- Transformers (`distilbert-base-uncased-finetuned-sst-2-english`, `t5-small`)
- spaCy (`en_core_web_sm`)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS (in-memory vector index)
- WebSockets (real-time analysis)
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

## API Endpoints

### REST Endpoints

#### POST /analyze
Analyze sentiment and extract keywords from text.

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

#### POST /summarize
Summarize the provided text using a T5 model.

Request
```json
{ "text": "Long article text here..." }
```
Response
```json
{ "summary": "Concise summary text." }
```

#### POST /semantic-search
Perform semantic search over the in-memory corpus with configurable top_k.

Request
```json
{ 
  "query": "How do I deploy apps in containers?",
  "top_k": 5
}
```
Response
```json
{ "results": ["Docker simplifies packaging and deploying applications.", "..."] }
```

#### POST /corpus/add
Dynamically add texts to the semantic search corpus.

Request
```json
{
  "texts": [
    "Kubernetes orchestrates containerized applications.",
    "Docker containers enable reproducible deployments.",
    "Cloud-native development requires distributed systems knowledge."
  ]
}
```
Response
```json
{
  "size": 8,
  "message": "Added 3 texts to corpus"
}
```

#### GET /corpus/size
Get the current size of the semantic search corpus.

Response
```json
{
  "size": 8,
  "message": "Corpus contains 8 texts"
}
```

### WebSocket Endpoint

#### WS /ws/analyze
Real-time sentiment analysis and keyword extraction over WebSocket.

**Client sends:** Plain text to analyze
**Server responds:** JSON with analysis results and timestamp

**Python Client Example:**
```python
import asyncio
import json
import websockets
from datetime import datetime

async def analyze_stream():
    uri = "ws://localhost:8000/ws/analyze"
    async with websockets.connect(uri) as websocket:
        # Send text for analysis
        texts = [
            "I absolutely love this new feature!",
            "This is terrible and frustrating.",
            "The API works as expected."
        ]
        
        for text in texts:
            print(f"\nSending: {text}")
            await websocket.send(text)
            
            # Receive and display results
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Response: {json.dumps(data, indent=2, default=str)}")
            await asyncio.sleep(1)

asyncio.run(analyze_stream())
```

**JavaScript Client Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analyze');

ws.onopen = () => {
  console.log('Connected to WebSocket');
  
  const texts = [
    'I absolutely love this API!',
    'This is disappointing.',
    'Everything works perfectly.'
  ];
  
  texts.forEach((text, index) => {
    setTimeout(() => {
      ws.send(text);
    }, index * 1000);
  });
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Analysis Result:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket Error:', error);
};

ws.onclose = () => {
  console.log('Connection closed');
};
```

## Sample cURL

### Analyze Sentiment & Extract Keywords
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Transformers make NLP easier."}'
```

### Summarize Text
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI is a modern, fast web framework for building APIs with Python."}'
```

### Semantic Search (Default - top 3 results)
```bash
curl -X POST http://localhost:8000/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is semantic search?"}'
```

### Semantic Search (Custom top_k)
```bash
curl -X POST http://localhost:8000/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"query": "Machine learning", "top_k": 10}'
```

### Add Texts to Corpus
```bash
curl -X POST http://localhost:8000/corpus/add \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Kubernetes is a container orchestration platform.",
      "Docker enables containerization of applications.",
      "Cloud infrastructure requires DevOps expertise."
    ]
  }'
```

### Get Corpus Size
```bash
curl -X GET http://localhost:8000/corpus/size
```

## Features

✅ **Multi-NLP Tasks** - Sentiment analysis, keyword extraction, summarization, semantic search all in one service

✅ **Real-Time Analysis** - WebSocket endpoint for streaming sentiment and keyword analysis

✅ **Dynamic Corpus Management** - Add texts to the semantic search index at runtime without restarting

✅ **Production-Ready** - Error handling, async/await for concurrency, CORS enabled for frontend integration

✅ **Configurable Search** - Control number of results returned via `top_k` parameter in semantic search

✅ **Timestamped Responses** - WebSocket messages include timestamps for audit and debugging

✅ **FAISS-Powered Search** - In-memory semantic search using cosine similarity with Sentence Transformers embeddings

## Notes
- Models are warmed on startup to reduce first-request latency.
- FAISS index is in-memory and preloaded with a default corpus; adapt `DEFAULT_CORPUS` in `app/embeddings.py` for your domain data.
- WebSocket connections accept plain text and respond with JSON containing analysis results.
- The corpus persists for the lifetime of the server process; data is lost on restart (for production, use persistent storage).
- For production deployment:
  - Use GPU-enabled Docker images to accelerate NLP models
  - Consider persistent vector stores (e.g., Redis, Pinecone, Weaviate) for larger corpora
  - Implement authentication/rate limiting as needed
  - Scale horizontally with load balancers
  - Monitor model latency and throughput

## Deployment

### Render
The service is ready for deployment on Render as a Web Service. Push your repository and connect it to Render; it will automatically detect `requirements.txt` and build the container.

### Environment Variables
No additional configuration required—the service starts with sensible defaults. Customize in code as needed.
