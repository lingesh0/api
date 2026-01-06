"""FastAPI entrypoint for the AI-Powered Text Intelligence API."""
from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.embeddings import SemanticSearchEngine, get_semantic_engine
from app.keywords import extract_keywords
from app.schemas import (
    AnalyzeResponse,
    CorpusAddRequest,
    CorpusResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SummaryResponse,
    TextRequest,
    WebSocketMessage,
)
from app.sentiment import analyze_sentiment
from app.summarize import summarize_text


app = FastAPI(
    title="AI Text Intelligence API",
    version="1.0.0",
    description="Production-ready NLP/LLM powered text intelligence service.",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Simple health check endpoint."""

    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: TextRequest) -> AnalyzeResponse:
    """Analyze sentiment and extract keywords from text."""

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    sentiment_task = asyncio.to_thread(analyze_sentiment, text)
    keywords_task = asyncio.to_thread(extract_keywords, text)
    sentiment, keywords = await asyncio.gather(sentiment_task, keywords_task)

    return AnalyzeResponse(sentiment=sentiment, keywords=keywords)


@app.post("/summarize", response_model=SummaryResponse)
async def summarize(payload: TextRequest) -> SummaryResponse:
    """Summarize the provided text using a T5 model."""

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    summary = await asyncio.to_thread(summarize_text, text)
    return SummaryResponse(summary=summary)


@app.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(
    payload: SemanticSearchRequest,
    engine: SemanticSearchEngine = Depends(get_semantic_engine),
) -> SemanticSearchResponse:
    """Perform semantic search over the in-memory corpus."""

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    results = await asyncio.to_thread(engine.search, query, payload.top_k)
    return SemanticSearchResponse(results=results)


@app.post("/corpus/add", response_model=CorpusResponse)
async def add_to_corpus(
    payload: CorpusAddRequest,
    engine: SemanticSearchEngine = Depends(get_semantic_engine),
) -> CorpusResponse:
    """Dynamically add texts to the semantic search corpus."""

    if not payload.texts:
        raise HTTPException(status_code=400, detail="Texts list must not be empty.")

    await asyncio.to_thread(engine.add_text, payload.texts)
    size = await asyncio.to_thread(engine.get_corpus_size)
    return CorpusResponse(size=size, message=f"Added {len(payload.texts)} texts to corpus")


@app.get("/corpus/size", response_model=CorpusResponse)
async def get_corpus_size(
    engine: SemanticSearchEngine = Depends(get_semantic_engine),
) -> CorpusResponse:
    """Get the current size of the semantic search corpus."""

    size = await asyncio.to_thread(engine.get_corpus_size)
    return CorpusResponse(size=size, message=f"Corpus contains {size} texts")


@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time sentiment and keyword analysis."""

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if not data.strip():
                continue

            try:
                # Run NLP tasks in thread pool to avoid blocking
                sentiment_task = asyncio.to_thread(analyze_sentiment, data)
                keywords_task = asyncio.to_thread(extract_keywords, data)
                sentiment, keywords = await asyncio.gather(sentiment_task, keywords_task)

                message = WebSocketMessage(
                    type="analysis",
                    timestamp=datetime.now(),
                    sentiment=sentiment,
                    keywords=keywords,
                )
                await websocket.send_json(message.model_dump(mode="json"))
            except Exception as e:
                error_message = WebSocketMessage(
                    type="error",
                    timestamp=datetime.now(),
                    error=str(e),
                )
                await websocket.send_json(error_message.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
