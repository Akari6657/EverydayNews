"""Thread-local near-duplicate cleanup for the V2 story-thread pipeline."""

from __future__ import annotations

import logging
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol, Sequence

from .models import (
    AppConfig,
    Article,
    StoryThread,
    ThreadDedupDiagnostics,
    WithinThreadMerge,
)

LOGGER = logging.getLogger(__name__)
NON_WORD_RE = re.compile(r"[^\w\s]+")
EMBEDDING_CACHE_FILENAME = "embeddings.pkl"


class SupportsEncode(Protocol):
    """Protocol for embedding encoders used in tests and production."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector per input text."""


@dataclass
class EmbeddingCacheEntry:
    """Cached embedding payload with pruning metadata."""

    embedding: list[float]
    updated_at: str


def deduplicate_within_thread_with_diagnostics(
    thread: StoryThread,
    config: AppConfig,
    embedding_cache_path: str | Path | None = None,
    now: datetime | None = None,
    encoder: SupportsEncode | None = None,
) -> tuple[StoryThread, ThreadDedupDiagnostics]:
    """Return a stricter within-thread near-duplicate cleanup plus debug details."""

    diagnostics = ThreadDedupDiagnostics(
        before_articles=len(thread.articles),
        after_articles=len(thread.articles),
    )
    if len(thread.articles) <= 1:
        return thread, diagnostics
    reference_time = now or datetime.now(timezone.utc)
    embedding_cache_file = (
        Path(embedding_cache_path)
        if embedding_cache_path
        else config.root_dir / "cache" / EMBEDDING_CACHE_FILENAME
    )
    priorities = config.source_priorities()
    ordered_articles = _sort_articles_for_clustering(thread.articles, priorities)
    try:
        if config.dedup.method == "embedding":
            canonical_articles, merged_pairs = _deduplicate_within_thread_embedding(
                ordered_articles,
                config,
                reference_time,
                embedding_cache_file,
                encoder,
            )
        else:
            canonical_articles, merged_pairs = _deduplicate_within_thread_difflib(
                ordered_articles,
                config.dedup.within_thread_similarity_threshold,
            )
    except Exception as exc:
        LOGGER.warning("Within-thread dedup failed for '%s': %s", thread.topic, exc)
        return thread, diagnostics
    if len(canonical_articles) == len(thread.articles):
        return thread, diagnostics
    return (
        _rebuild_thread_after_within_dedup(thread, canonical_articles, priorities),
        ThreadDedupDiagnostics(
            before_articles=len(thread.articles),
            after_articles=len(canonical_articles),
            merged_pairs=merged_pairs,
        ),
    )


def _deduplicate_within_thread_embedding(
    articles: list[Article],
    config: AppConfig,
    reference_time: datetime,
    embedding_cache_file: Path,
    encoder: SupportsEncode | None,
) -> tuple[list[Article], list[WithinThreadMerge]]:
    """Collapse only near-identical articles within one story thread."""

    if not articles:
        return [], []
    embeddings = _get_embeddings(
        articles,
        config,
        reference_time,
        embedding_cache_file,
        encoder,
    )
    threshold = config.dedup.within_thread_similarity_threshold
    canonical_articles: list[Article] = []
    canonical_embeddings: list[list[float]] = []
    merged_pairs: list[WithinThreadMerge] = []
    for article, embedding in zip(articles, embeddings):
        best_index: int | None = None
        best_score = -1.0
        for index, existing_embedding in enumerate(canonical_embeddings):
            score = _cosine_similarity(embedding, existing_embedding)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is not None and best_score >= threshold:
            merged_pairs.append(
                WithinThreadMerge(
                    kept_article=canonical_articles[best_index],
                    removed_article=article,
                    similarity=best_score,
                )
            )
            continue
        canonical_articles.append(article)
        canonical_embeddings.append(embedding)
    return canonical_articles, merged_pairs


def _deduplicate_within_thread_difflib(
    articles: list[Article],
    threshold: float,
) -> tuple[list[Article], list[WithinThreadMerge]]:
    """Fallback near-duplicate cleanup using lexical title similarity only."""

    canonical_articles: list[Article] = []
    normalized_titles: list[str] = []
    merged_pairs: list[WithinThreadMerge] = []
    for article in articles:
        normalized = _normalize_title(article.title)
        best_index = _best_difflib_cluster(normalized, normalized_titles, threshold)
        if best_index is None:
            canonical_articles.append(article)
            normalized_titles.append(normalized)
            continue
        score = SequenceMatcher(a=normalized, b=normalized_titles[best_index]).ratio()
        merged_pairs.append(
            WithinThreadMerge(
                kept_article=canonical_articles[best_index],
                removed_article=article,
                similarity=score,
            )
        )
    return canonical_articles, merged_pairs


def _best_difflib_cluster(
    normalized_title: str,
    normalized_titles: list[str],
    threshold: float,
) -> int | None:
    """Return the best lexical cluster match or None."""

    best_index: int | None = None
    best_score = -1.0
    for index, existing in enumerate(normalized_titles):
        score = SequenceMatcher(a=normalized_title, b=existing).ratio()
        if score > best_score:
            best_index = index
            best_score = score
    if best_score >= threshold:
        return best_index
    return None

def _get_embeddings(
    articles: list[Article],
    config: AppConfig,
    reference_time: datetime,
    embedding_cache_file: Path,
    encoder: SupportsEncode | None,
) -> list[list[float]]:
    """Load cached embeddings and encode only missing articles."""

    cache = _prune_embedding_cache(_load_embedding_cache(embedding_cache_file), reference_time)
    if not config.dedup.cache_embeddings:
        cache = {}
    missing_articles = [article for article in articles if article.guid not in cache]
    if missing_articles:
        embedding_encoder = encoder or _build_encoder(config.dedup.model)
        texts = [_embedding_text(article) for article in missing_articles]
        vectors = embedding_encoder.encode(texts)
        for article, vector in zip(missing_articles, vectors):
            cache[article.guid] = EmbeddingCacheEntry(
                embedding=_to_float_list(vector),
                updated_at=reference_time.isoformat(),
            )
    if config.dedup.cache_embeddings:
        _save_embedding_cache(embedding_cache_file, cache)
    return [cache[article.guid].embedding for article in articles]


def _build_encoder(model_name: str) -> SupportsEncode:
    """Lazily construct the sentence-transformers encoder."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embedding_text(article: Article) -> str:
    """Build the text used for semantic deduplication."""

    description = article.description[:200]
    return f"{article.title}\n{description}".strip()


def _sort_articles_for_clustering(
    articles: list[Article],
    priorities: dict[str, int],
) -> list[Article]:
    """Sort by source priority first, then by recency."""

    return sorted(
        articles,
        key=lambda article: (
            _priority(article, priorities),
            -article.published.timestamp(),
        ),
    )

def _rebuild_thread_after_within_dedup(
    thread: StoryThread,
    canonical_articles: list[Article],
    priorities: dict[str, int],
) -> StoryThread:
    """Return a thread with deduplicated article bodies but preserved source coverage."""

    ordered_articles = _sort_articles_for_clustering(canonical_articles, priorities)
    distinct_sources: list[str] = []
    for article in ordered_articles:
        if article.source_name not in distinct_sources:
            distinct_sources.append(article.source_name)
    return StoryThread(
        thread_id=thread.thread_id,
        topic=thread.topic,
        topic_en=thread.topic_en,
        articles=ordered_articles,
        source_names=distinct_sources,
        source_count=len(distinct_sources),
        primary=ordered_articles[0],
        latest_published=thread.latest_published,
        rationale=thread.rationale,
    )


def _priority(article: Article, priorities: dict[str, int]) -> int:
    """Return source priority with a safe default."""

    return priorities.get(article.source_slug, len(priorities))


def _normalize_title(title: str) -> str:
    """Normalize punctuation and spacing for lexical comparison."""

    lowered = title.casefold()
    stripped = NON_WORD_RE.sub(" ", lowered)
    return " ".join(stripped.split())


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two dense vectors."""

    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _to_float_list(vector: Sequence[float] | object) -> list[float]:
    """Convert encoder outputs to a JSON/pickle-friendly float list."""

    if hasattr(vector, "tolist"):
        values = vector.tolist()
        if isinstance(values, list):
            return [float(item) for item in values]
    return [float(item) for item in vector]  # type: ignore[arg-type]

def _load_embedding_cache(cache_path: Path) -> dict[str, EmbeddingCacheEntry]:
    """Load embedding cache from disk."""

    if not cache_path.exists():
        return {}
    try:
        payload = pickle.loads(cache_path.read_bytes())
    except Exception:
        LOGGER.warning("Embedding cache is unreadable, rebuilding: %s", cache_path)
        return {}
    if not isinstance(payload, dict):
        LOGGER.warning("Embedding cache is not a mapping, rebuilding: %s", cache_path)
        return {}
    loaded: dict[str, EmbeddingCacheEntry] = {}
    for guid, entry in payload.items():
        if isinstance(entry, EmbeddingCacheEntry):
            loaded[str(guid)] = entry
            continue
        if isinstance(entry, dict) and "embedding" in entry and "updated_at" in entry:
            loaded[str(guid)] = EmbeddingCacheEntry(
                embedding=_to_float_list(entry["embedding"]),
                updated_at=str(entry["updated_at"]),
            )
    return loaded


def _prune_embedding_cache(
    cache: dict[str, EmbeddingCacheEntry],
    reference_time: datetime,
) -> dict[str, EmbeddingCacheEntry]:
    """Keep only recent embedding cache entries."""

    cutoff = reference_time - timedelta(days=7)
    pruned: dict[str, EmbeddingCacheEntry] = {}
    for guid, entry in cache.items():
        parsed = _parse_cache_time(entry.updated_at)
        if parsed and parsed >= cutoff:
            pruned[guid] = entry
    return pruned


def _save_embedding_cache(cache_path: Path, cache: dict[str, EmbeddingCacheEntry]) -> None:
    """Persist embedding cache to disk."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))


def _parse_cache_time(timestamp: str) -> datetime | None:
    """Parse cached ISO timestamps safely."""

    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
