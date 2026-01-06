"""Keyword extraction powered by spaCy."""
from collections import Counter
from functools import lru_cache
from typing import List

import spacy


@lru_cache(maxsize=1)
def get_spacy_model():
    """Return a cached spaCy English pipeline."""

    return spacy.load("en_core_web_sm")


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """Extract the top keywords using part-of-speech tagging and frequency."""

    nlp = get_spacy_model()
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN"}
        and not token.is_stop
        and token.is_alpha
        and len(token.lemma_) > 2
    ]
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_k)]
