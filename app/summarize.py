"""Summarization utilities using a T5 model."""
from functools import lru_cache
from typing import List

from transformers import pipeline


@lru_cache(maxsize=1)
def get_summarizer():
    """Return a cached T5-based summarization pipeline."""

    return pipeline("summarization", model="t5-small")


def _chunk_words(text: str, chunk_size: int) -> List[str]:
    """Split text into word-based chunks to keep summaries accurate."""

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
    return chunks


def summarize_text(
    text: str,
    max_length: int = 120,
    min_length: int = 30,
    chunk_size: int = 400,
) -> str:
    """Generate a concise summary, handling long texts by chunking."""

    summarizer = get_summarizer()
    chunks = _chunk_words(text, chunk_size)
    partial_summaries = []

    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )[0]["summary_text"]
        partial_summaries.append(summary)

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined = " ".join(partial_summaries)
    final = summarizer(
        combined,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )[0]["summary_text"]
    return final
