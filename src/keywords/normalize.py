from __future__ import annotations

import re


_WORD_RE = re.compile(r"[^a-z0-9\-\s]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _singularize_word(word: str) -> str:
    if len(word) <= 3:
        return word

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("sses") and len(word) > 5:
        return word[:-2]

    if word.endswith(("xes", "zes", "ches", "shes")) and len(word) > 4:
        return word[:-2]

    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    return word


def normalize_keyword(raw_keyword: str) -> str:
    text = str(raw_keyword).strip().lower()
    text = _WORD_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    words = [_singularize_word(part) for part in text.split(" ") if part]
    normalized = " ".join(words).strip()
    return normalized


def parse_keyword_text(keywords_text: str) -> list[str]:
    normalized = [normalize_keyword(piece) for piece in str(keywords_text).split(",")]
    normalized = [piece for piece in normalized if piece]
    unique_sorted = sorted(set(normalized))
    return unique_sorted


def normalize_keyword_text(keywords_text: str) -> str:
    return ", ".join(parse_keyword_text(keywords_text))
