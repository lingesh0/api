"""FastAPI entrypoint for the AI-Powered Text Intelligence API."""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from app.embeddings import SemanticSearchEngine, get_semantic_engine
from app.keywords import extract_keywords
from app.schemas import (
    AnalyzeResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SummaryResponse,
    TextRequest,
)
from app.sentiment import analyze_sentiment
from app.summarize import summarize_text


app = FastAPI(
    title="AI Text Intelligence API",
    version="1.0.0",
    description="Production-ready NLP/LLM powered text intelligence service.",
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

    results = await asyncio.to_thread(engine.search, query, 3)
    return SemanticSearchResponse(results=results)
