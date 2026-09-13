from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .config import ModelConfig
from .io import atomic_json, dedupe_text, load_json
from .models import JsonModelClient
from .prompts import KEYWORD_SYSTEM_PROMPT, keyword_prompt


PATH_RE = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
SYMBOL_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning)|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|"
    r"[A-Za-z_][A-Za-z0-9_]{2,}(?=\s*\())\b"
)
FORBIDDEN = re.compile(
    r"\b(?:reward|hidden test|golden patch|node[-_ ]?id|instance[-_ ]?id|"
    r"ground[- ]truth|terminal outcome|this experiment)\b",
    re.I,
)
GENERIC = {
    "bug",
    "fix",
    "code",
    "test",
    "tests",
    "repository",
    "trajectory",
    "implementation",
    "software engineering",
}
QUERY_STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "are", "as", "at",
    "be", "because", "been", "before", "but", "by", "can", "could", "did",
    "do", "does", "during", "each", "for", "from", "had", "has", "have", "how",
    "if", "in", "into", "is", "it", "its", "may", "more", "must", "no", "not",
    "of", "on", "only", "or", "other", "our", "should", "so", "some", "such",
    "than", "that", "the", "their", "then", "there", "these", "this", "to", "use",
    "used", "using", "was", "we", "when", "where", "which", "while", "will",
    "with", "would", "you", "your",
}
QUERY_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_text(child)}" for key, child in value.items())
    if isinstance(value, list):
        return "\n".join(_text(child) for child in value)
    return str(value)


def sanitize_keyword(phrase: Any) -> str | None:
    text = " ".join(str(phrase or "").strip().split())
    if not 2 <= len(text) <= 160 or FORBIDDEN.search(text):
        return None
    if text.casefold() in GENERIC:
        return None
    tokens = re.findall(r"[A-Za-z0-9_./:-]+", text)
    if not 1 <= len(tokens) <= 12:
        return None
    return text


def phrase_match(phrase: str, view: Any) -> bool:
    normalized_phrase = re.sub(r"\s+", " ", phrase.casefold()).strip()
    normalized_view = re.sub(r"\s+", " ", _text(view).casefold())
    if normalized_phrase in normalized_view:
        return True
    terms = [
        token
        for token in re.findall(r"[a-z0-9_./:-]+", normalized_phrase)
        if len(token) > 2
    ]
    return bool(terms) and sum(term in normalized_view for term in terms) / len(terms) >= 0.8


def deterministic_candidates(card: dict[str, Any]) -> list[str]:
    public = "\n".join(
        str(card.get(key) or "")
        for key in ("title", "description", "context", "experience")
    )
    title = sanitize_keyword(card.get("title"))
    symbols = [*PATH_RE.findall(public), *SYMBOL_RE.findall(public)]
    references = (card.get("metadata") or {}).get("reference_golden_rubrics") or []
    rubric_titles = [
        item.get("title")
        for item in references
        if isinstance(item, dict) and item.get("title")
    ]
    output = []
    for candidate in dedupe_text([title, *symbols, *rubric_titles]):
        phrase = sanitize_keyword(candidate)
        if phrase is not None:
            output.append(phrase)
    return output


def _stem(token: str) -> str:
    value = token.casefold().strip("./:-_")
    if len(value) > 4 and value.endswith("s"):
        value = value[:-1]
    for suffix in ("ing", "ed", "es"):
        if len(value) > len(suffix) + 3 and value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def query_overlap_candidates(
    card: dict[str, Any], visible_queries: Iterable[Any]
) -> list[str]:
    """Extract public query terms that overlap the card's activation contract.

    Query-derived terms are intentionally a separate candidate source: the final
    deployed bank may select them even when the model did not emit the same phrase.
    """

    card_text = _text(
        {
            key: card.get(key)
            for key in ("title", "description", "context", "experience", "metadata")
        }
    )
    card_stems = {
        _stem(token)
        for token in QUERY_TOKEN_RE.findall(card_text)
        if len(_stem(token)) > 2
    }
    candidates: list[str] = []
    for query in visible_queries:
        query_text = _text(query)
        candidates.extend(PATH_RE.findall(query_text))
        candidates.extend(SYMBOL_RE.findall(query_text))
        for token in QUERY_TOKEN_RE.findall(query_text):
            stem = _stem(token)
            if (
                len(stem) > 2
                and stem not in QUERY_STOPWORDS
                and stem in card_stems
            ):
                candidates.append(token)
    return [
        phrase
        for phrase in (sanitize_keyword(value) for value in dedupe_text(candidates))
        if phrase is not None
    ]


def sanitized_keywords(candidates: Iterable[Any]) -> list[str]:
    """Keep every safe extracted phrase for final joint-replay selection."""

    return [
        phrase
        for phrase in (sanitize_keyword(value) for value in dedupe_text(candidates))
        if phrase is not None
    ]


class KeywordBuilder:
    def __init__(
        self,
        *,
        models: ModelConfig | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.models = models or ModelConfig()
        self.model_kwargs = model_kwargs or {"custom_llm_provider": "openai"}
        self.client = JsonModelClient(
            temperature=self.models.temperature,
            top_p=self.models.top_p,
        )

    async def generate(
        self,
        *,
        card: dict[str, Any],
        positive_views: list[dict[str, Any]] | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        parsed, messages = await self.client.call(
            model=self.models.refiner,
            system=KEYWORD_SYSTEM_PROMPT,
            user=keyword_prompt(card),
            max_tokens=self.models.max_tokens,
            retry_max_tokens=self.models.max_tokens,
            validator=lambda value: isinstance(value.get("keywords"), list),
            model_kwargs=self.model_kwargs,
            route_name="experience_keyword_generation",
        )
        base = deterministic_candidates(card)
        model = sanitized_keywords(parsed.get("keywords") or [])
        generated = dedupe_text([*base, *model])
        query = query_overlap_candidates(card, positive_views or [])
        payload = {
            "schema_version": 1,
            "experience_id": card["experience_id"],
            "base_keywords": base,
            "model_keywords": model,
            "query_keywords": query,
            "keywords": dedupe_text([*generated, *query]),
            "messages": messages,
        }
        if output_path is not None:
            atomic_json(output_path, payload)
        return payload

def materialize_keyword_artifacts(
    *,
    checkpoint: str | Path,
    scope: str,
    records: list[dict[str, Any]],
) -> None:
    root = Path(checkpoint)
    keyword_path = root / "retrieval/keyword_bank.json"
    keyword_payload = load_json(keyword_path)
    for record in records:
        experience_id = record["experience_id"]
        generated = dedupe_text(
            [
                *(record.get("base_keywords") or []),
                *(record.get("model_keywords") or []),
            ]
        )
        selected = dedupe_text(record.get("keywords") or [])
        if not selected:
            raise ValueError(
                f"{scope}:{experience_id}: selected keywords must be nonempty"
            )
        keyword_payload["bank"][scope][experience_id] = {
            "generated_keywords": generated,
            "selected_keywords": selected,
        }
    counts = {
        item_scope: len(keyword_payload["bank"][item_scope])
        for item_scope in ("siblings", "pc")
    }
    keyword_payload["scope_counts"] = counts
    keyword_payload["experience_count"] = sum(counts.values())
    atomic_json(keyword_path, keyword_payload)
