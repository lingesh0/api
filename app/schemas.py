"""Pydantic request and response schemas for the AI Text API."""
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


class SemanticSearchResponse(BaseModel):
    """Response containing semantic search results."""

    results: List[str]
