"""Sentiment analysis utilities using Hugging Face Transformers."""
from functools import lru_cache
from typing import Literal

from transformers import pipeline

SentimentLabel = Literal["positive", "negative", "neutral"]


@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    """Return a cached sentiment-analysis pipeline."""

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


def analyze_sentiment(text: str, neutral_threshold: float = 0.6) -> SentimentLabel:
    """Classify sentiment and return positive, negative, or neutral."""

    analyzer = get_sentiment_analyzer()
    result = analyzer(text)[0]
    label = result["label"].lower()
    score = float(result["score"])

    if score < neutral_threshold:
        return "neutral"
    return "positive" if label == "positive" else "negative"
