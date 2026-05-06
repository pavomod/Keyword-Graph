from __future__ import annotations

import re

from src.phase1.keyBERT.normalize import normalize_keyword_text

try:
    from src.phase1.keyBERT.keybert_adapter import extract_keywords_keybert
    _KEYBERT_AVAILABLE = True
except Exception:
    extract_keywords_keybert = None  # type: ignore
    _KEYBERT_AVAILABLE = False

try:
    from src.phase1.gemini.extractor import extract_keywords_gemini

    _GEMINI_AVAILABLE = True
except Exception:
    extract_keywords_gemini = None  # type: ignore
    _GEMINI_AVAILABLE = False

# Maximum keywords: must stay aligned with prepare_data.MAX_KEYWORDS and the prompt rule.
MAX_KEYWORDS = 6

# Stopwords: common grammatical and non-informative terms to exclude from keyword extraction
_STOPWORDS = {
    # Common grammatical stopwords
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
    # Non-informative UI/generic terms
    "home",
    "user",
    "page",
    "system",
    "feature",
    "interface",
    "button",
    "screen",
    "application",
    "window",
    "link",
    "input",
    "field",
    "option",
    "setting",
    "menu",
    "form",
    "dialog",
    "view",
    "panel",
    "message",
    "content",
    "data",
    "item",
    "element",
    "action",
    "function",
    "tool",
}


def predict_keywords(
    inputs: list[str],
    keybert_kwargs: dict | None = None,
    gemini_kwargs: dict | None = None,
    backend: str = "keybert",
) -> tuple[list[str], dict[str, float | int]]:
    """Extract keywords using the selected backend and fall back to a heuristic extractor on failure."""
    predictions: list[str] = []
    model_generated_count = 0
    fallback_generated_count = 0

    keybert_kwargs = keybert_kwargs or {}
    gemini_kwargs = gemini_kwargs or {}
    backend_name = str(backend).strip().lower()

    if backend_name == "gemini":
        if _GEMINI_AVAILABLE:
            for prompt in inputs:
                try:
                    kws = extract_keywords_gemini(prompt, **gemini_kwargs)
                    if kws:
                        joined = ", ".join(kws[:MAX_KEYWORDS])
                        normalized = joined if backend_name == "gemini" else normalize_keyword_text(joined)
                        predictions.append(normalized)
                        model_generated_count += 1
                        continue
                except Exception:
                    pass

                predictions.append(_fallback_keywords_from_prompt(prompt, max_keywords=MAX_KEYWORDS))
                fallback_generated_count += 1
        else:
            for prompt in inputs:
                predictions.append(_fallback_keywords_from_prompt(prompt, max_keywords=MAX_KEYWORDS))
                fallback_generated_count += 1
    elif _KEYBERT_AVAILABLE:
        for prompt in inputs:
            try:
                kws = extract_keywords_keybert(prompt, **keybert_kwargs)
                if kws:
                    joined = ", ".join(kws[:MAX_KEYWORDS])
                    normalized = normalize_keyword_text(joined)
                    predictions.append(normalized)
                    model_generated_count += 1
                    continue
            except Exception:
                pass

            predictions.append(_fallback_keywords_from_prompt(prompt, max_keywords=MAX_KEYWORDS))
            fallback_generated_count += 1
    else:
        for prompt in inputs:
            predictions.append(_fallback_keywords_from_prompt(prompt, max_keywords=MAX_KEYWORDS))
            fallback_generated_count += 1

    total_predictions = len(predictions)
    fallback_ratio = (
        (fallback_generated_count / total_predictions) * 100.0
        if total_predictions
        else 0.0
    )
    print(
        "[diagnostic] keyword_inference "
        f"total={total_predictions} | "
        f"from_keybert={model_generated_count} | "
        f"from_fallback={fallback_generated_count} | "
        f"fallback_ratio={fallback_ratio:.2f}%"
    )

    diagnostics = {
        "total": total_predictions,
        "from_keybert": model_generated_count,
        "from_fallback": fallback_generated_count,
        "fallback_ratio": fallback_ratio,
        "backend": backend_name,
    }

    return predictions, diagnostics


def _fallback_keywords_from_prompt(prompt: str, max_keywords: int = MAX_KEYWORDS) -> str:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", str(prompt).lower())
    filtered = [w for w in words if len(w) > 2 and w not in _STOPWORDS]

    ordered_unique = list(dict.fromkeys(filtered))
    if not ordered_unique:
        relaxed = [w for w in words if len(w) > 1]
        ordered_unique = list(dict.fromkeys(relaxed))
    if not ordered_unique:
        ordered_unique = ["keyword"]

    heuristic = ", ".join(ordered_unique[:max_keywords])
    return normalize_keyword_text(heuristic)