"""Pydantic request and response schemas for the AI Text API."""
from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """Request body containing a single text field."""

    text: str = Field(..., min_length=1, description="Input text to process")


class AnalyzeResponse(BaseModel):
    """Response for the /analyze endpoint."""

    sentiment: Literal["positive", "negative", "neutral"]
    keywords: List[str]


class SummaryResponse(BaseModel):
    """Response for the /summarize endpoint."""

    summary: str


class SemanticSearchRequest(BaseModel):
    """Request for semantic search queries."""

    query: str = Field(..., min_length=1, description="Query text to search against the corpus")
    top_k: int = Field(3, ge=1, le=100, description="Number of top results to return")


class SemanticSearchResponse(BaseModel):
    """Response containing semantic search results."""

    results: List[str]


class CorpusAddRequest(BaseModel):
    """Request to add texts to the semantic search corpus."""

    texts: List[str] = Field(..., min_items=1, description="List of texts to add to corpus")


class CorpusResponse(BaseModel):
    """Response for corpus operations."""

    size: int = Field(..., description="Current size of the corpus")
    message: str


class WebSocketMessage(BaseModel):
    """WebSocket message for real-time text analysis."""

    type: Literal["analysis", "error"]
    timestamp: datetime
    sentiment: Literal["positive", "negative", "neutral"] | None = None
    keywords: List[str] | None = None
    error: str | None = None
