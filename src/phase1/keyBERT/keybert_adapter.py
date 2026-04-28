import re
from typing import List, Tuple
from keybert import KeyBERT

# Initialize ONCE
_kw_model = KeyBERT("BAAI/bge-m3")


def extract_requirement_text(prompt: str) -> str:
    """Extract the final user-story requirement from a full pipeline prompt."""
    input_sections = re.split(r"(?im)^Input:\s*", prompt)
    candidate = input_sections[-1] if len(input_sections) > 1 else prompt

    candidate = re.split(r"(?im)^Output:\s*", candidate, maxsplit=1)[0].strip()
    candidate = candidate.splitlines()[0].strip() if candidate else ""

    if candidate:
        match = re.search(r"I want .+? so that .+", candidate, re.DOTALL)
        if match:
            return match.group(0).strip()

    match = re.search(r"I want .+? so that .+?(?=\n|$)", prompt, re.DOTALL)
    if match:
        return match.group(0).strip()

    return prompt


def extract_keywords_keybert(
    text: str,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    top_n: int = 4,
    stop_words: str | None = "english",
    threshold: float = 0.50,
) -> List[str]:
    # Pulizia del testo (mantenendo la tua logica di filtraggio)
    requirement_text = extract_requirement_text(text).replace("my smart home", "").replace("smart home", "").replace("I want", "").strip()
    
    # Estrazione con KeyBERT
    raw = _kw_model.extract_keywords(
        requirement_text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
        top_n=top_n,
        use_mmr=True,
        diversity=0.7,
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