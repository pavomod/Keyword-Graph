import re
from typing import List, Tuple
from keybert import KeyBERT

# Lazy model holder
_kw_model = None
_kw_model_name = None


def _ensure_model(model_name: str | None = None):
    """Initialize KeyBERT model lazily and reuse the instance.

    If a different model_name is provided than the currently loaded one,
    the model is reinitialized. Reinitialization is expensive; prefer
    providing the desired model once at pipeline startup.
    """
    global _kw_model, _kw_model_name
    target = model_name or "BAAI/bge-m3"
    if _kw_model is None or _kw_model_name != target:
        _kw_model = KeyBERT(target)
        _kw_model_name = target


def extract_keywords_keybert(
    text: str,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    top_n: int = 6,
    stop_words: str | None = "english",
    threshold: float = 0.50,
    use_mmr: bool = True,
    diversity: float = 0.7,
    model_name: str | None = None,
) -> List[str]:
    # Text cleanup while preserving the existing filtering logic
    requirement_text = re.sub(
        r"(the system must|students must|user|system)",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()
    
    # Ensure model is loaded (lazy init) and run extraction
    _ensure_model(model_name)
    raw = _kw_model.extract_keywords(
        requirement_text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
        top_n=top_n,
        use_mmr=use_mmr,
        diversity=diversity,
    )

    if not raw:
        return []

    keywords = []
    seen = set()

    for i, (phrase, score) in enumerate(raw):
        clean = phrase.strip().lower()
        if i == 0 or score >= threshold:
            if clean and clean not in seen:
                keywords.append(clean)
                seen.add(clean)
        else:
            continue

    return keywords