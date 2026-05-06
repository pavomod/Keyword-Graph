from __future__ import annotations

import json
import os
import re
from pathlib import Path

import google.generativeai as genai

from src.phase1.keyBERT.normalize import normalize_keyword


DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompt") / "system_prompt.txt"

MIN_KEYWORDS = 1
MAX_KEYWORDS = 3
MAX_WORDS_PER_KEYWORD = 3


def load_system_prompt(prompt_path: str | None = None) -> str:
    path = Path(prompt_path) if prompt_path else DEFAULT_SYSTEM_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_keywords_gemini(
    text: str,
    top_n: int = MAX_KEYWORDS,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_output_tokens: int = 600,
    api_key: str | None = None,
    system_prompt_path: str | None = None,
) -> list[str]:
    if api_key is None:
        api_key = os.getenv("MODEL_API_KEY")

    if not api_key:
        raise ValueError("API key not provided and MODEL_API_KEY env var not set")

    system_prompt = load_system_prompt(system_prompt_path)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )

    user_prompt = (
        f"REQUIREMENT:\n{text}"
    )

    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        ),
    )

    if not response or not getattr(response, "text", None):
        return []

    response_text = response.text.strip()
    return _parse_keyword_response(response_text, top_n=top_n)


def _parse_keyword_response(response_text: str, top_n: int) -> list[str]:
    if not response_text:
        return []

    stripped = _strip_code_fences(response_text)

    parsed = _load_json_array(stripped)
    if parsed is None:
        parsed = _split_text_items(stripped)

    keywords: list[str] = []
    seen: set[str] = set()

    for item in parsed:
        normalized = normalize_keyword(item)

        if not normalized:
            continue

        normalized = normalized.lower().strip().strip('"\' ')

        if (
            normalized in seen
            or not _is_valid_keyword(normalized)
            or len(normalized) > 40
        ):
            continue

        keywords.append(normalized)
        seen.add(normalized)

        if len(keywords) >= top_n:
            break

    return keywords


def _is_valid_keyword(keyword: str) -> bool:
    words = keyword.split()
    return MIN_KEYWORDS <= len(words) <= MAX_WORDS_PER_KEYWORD


def _strip_code_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _load_json_array(text: str) -> list[str] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, list):
        return [str(item) for item in payload]

    if isinstance(payload, dict):
        for key in ("keywords", "items", "tags", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]

    return None


def _split_text_items(text: str) -> list[str]:
    candidates = re.split(r"[\n,;]+", text)
    items: list[str] = []

    for candidate in candidates:
        cleaned = re.sub(r"^[\s\-*•\d.]+", "", candidate).strip()
        cleaned = cleaned.strip('"\' ')

        if cleaned:
            items.append(cleaned)

    return items