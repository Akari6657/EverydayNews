"""Deduplicate articles with semantic clustering and cache support."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol, Sequence

from .models import AppConfig, Article, ArticleCluster

LOGGER = logging.getLogger(__name__)
NON_WORD_RE = re.compile(r"[^\w\s]+")
EMBEDDING_CACHE_FILENAME = "embeddings.pkl"
SEEN_CACHE_FILENAME = "seen_guids.json"


class SupportsEncode(Protocol):
    """Protocol for embedding encoders used in tests and production."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector per input text."""


@dataclass
class EmbeddingCacheEntry:
    """Cached embedding payload with pruning metadata."""

    embedding: list[float]
    updated_at: str


def deduplicate(
    articles: list[Article],
    config: AppConfig,
    cache_path: str | Path | None = None,
    embedding_cache_path: str | Path | None = None,
    now: datetime | None = None,
    encoder: SupportsEncode | None = None,
) -> list[ArticleCluster]:
    """Cluster unseen articles and return canonical event groups."""

    reference_time = now or datetime.now(timezone.utc)
    seen_cache_file = Path(cache_path) if cache_path else config.root_dir / "cache" / SEEN_CACHE_FILENAME
    embedding_cache_file = (
        Path(embedding_cache_path)
        if embedding_cache_path
        else config.root_dir / "cache" / EMBEDDING_CACHE_FILENAME
    )
    priorities = config.source_priorities()
    seen_cache = _prune_seen_cache(_load_seen_cache(seen_cache_file), reference_time)
    fresh_articles = [article for article in articles if article.guid not in seen_cache]
    clusters = _cluster_articles(
        fresh_articles,
        config,
        priorities,
        reference_time,
        embedding_cache_file,
        encoder,
    )
    final_clusters = clusters[: config.pipeline.total_articles_for_summary]
    updated_seen_cache = _update_seen_cache(seen_cache, final_clusters, reference_time)
    _save_seen_cache(seen_cache_file, updated_seen_cache)
    return final_clusters


def deduplicate_articles(
    articles: list[Article],
    config: AppConfig,
    cache_path: str | Path | None = None,
    embedding_cache_path: str | Path | None = None,
    now: datetime | None = None,
    encoder: SupportsEncode | None = None,
) -> list[Article]:
    """Return only the primary article from each cluster for v1 compatibility."""

    clusters = deduplicate(
        articles,
        config,
        cache_path=cache_path,
        embedding_cache_path=embedding_cache_path,
        now=now,
        encoder=encoder,
    )
    return [cluster.primary for cluster in clusters]


def _cluster_articles(
    articles: list[Article],
    config: AppConfig,
    priorities: dict[str, int],
    reference_time: datetime,
    embedding_cache_file: Path,
    encoder: SupportsEncode | None,
) -> list[ArticleCluster]:
    """Cluster articles with the configured deduplication method."""

    if config.dedup.method == "embedding":
        try:
            return _dedup_embedding(
                articles,
                config,
                priorities,
                reference_time,
                embedding_cache_file,
                encoder,
            )
        except Exception as exc:
            LOGGER.warning("Embedding dedup failed, falling back to difflib: %s", exc)
    return _dedup_difflib(articles, config, priorities, reference_time)


def _dedup_embedding(
    articles: list[Article],
    config: AppConfig,
    priorities: dict[str, int],
    reference_time: datetime,
    embedding_cache_file: Path,
    encoder: SupportsEncode | None,
) -> list[ArticleCluster]:
    """Cluster semantically similar articles using embeddings."""

    ordered = _sort_articles_for_clustering(articles, priorities)
    if not ordered:
        return []
    embeddings = _get_embeddings(
        ordered,
        config,
        reference_time,
        embedding_cache_file,
        encoder,
    )
    if config.dedup.clustering_algorithm == "dbscan":
        clusters = _cluster_with_dbscan(ordered, embeddings, config, priorities)
    else:
        clusters = _cluster_with_greedy(ordered, embeddings, config)
    return _sort_clusters(clusters, priorities, reference_time)


def _dedup_difflib(
    articles: list[Article],
    config: AppConfig,
    priorities: dict[str, int],
    reference_time: datetime,
) -> list[ArticleCluster]:
    """Cluster articles by lexical title similarity as a fallback."""

    ordered = _sort_articles_for_clustering(articles, priorities)
    clusters: list[ArticleCluster] = []
    normalized_titles: list[str] = []
    for article in ordered:
        normalized = _normalize_title(article.title)
        best_index = _best_difflib_cluster(normalized, normalized_titles, config.dedup.similarity_threshold)
        if best_index is None:
            clusters.append(_make_cluster(article))
            normalized_titles.append(normalized)
            continue
        clusters[best_index].add_duplicate(article)
    return _sort_clusters(clusters, priorities, reference_time)


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


def _cluster_with_greedy(
    articles: list[Article],
    embeddings: list[list[float]],
    config: AppConfig,
) -> list[ArticleCluster]:
    """Greedily assign each article to the nearest existing cluster."""

    clusters: list[ArticleCluster] = []
    primary_embeddings: list[list[float]] = []
    threshold = config.dedup.similarity_threshold
    for article, embedding in zip(articles, embeddings):
        best_index: int | None = None
        best_score = -1.0
        for index, primary_embedding in enumerate(primary_embeddings):
            score = _cosine_similarity(embedding, primary_embedding)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is not None and best_score >= threshold:
            clusters[best_index].add_duplicate(article)
            continue
        clusters.append(_make_cluster(article))
        primary_embeddings.append(embedding)
    return clusters


def _cluster_with_dbscan(
    articles: list[Article],
    embeddings: list[list[float]],
    config: AppConfig,
    priorities: dict[str, int],
) -> list[ArticleCluster]:
    """Cluster articles with DBSCAN using cosine distance."""

    from sklearn.cluster import DBSCAN

    eps = max(0.0, 1.0 - config.dedup.similarity_threshold)
    labels = DBSCAN(metric="cosine", eps=eps, min_samples=1).fit_predict(embeddings)
    grouped: dict[int, list[tuple[int, Article]]] = {}
    for index, (label, article) in enumerate(zip(labels, articles)):
        grouped.setdefault(int(label), []).append((index, article))
    clusters: list[ArticleCluster] = []
    for members in grouped.values():
        ordered_members = sorted(
            members,
            key=lambda item: (
                _priority(item[1], priorities),
                -item[1].published.timestamp(),
                item[0],
            ),
        )
        primary = ordered_members[0][1]
        cluster = _make_cluster(primary)
        for _, article in ordered_members[1:]:
            cluster.add_duplicate(article)
        clusters.append(cluster)
    return clusters


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


def _sort_clusters(
    clusters: list[ArticleCluster],
    priorities: dict[str, int],
    reference_time: datetime,
) -> list[ArticleCluster]:
    """Sort clusters by coverage first, then source priority and recency."""

    return sorted(
        clusters,
        key=lambda cluster: (
            -_cluster_priority(cluster, reference_time),
            _priority(cluster.primary, priorities),
            -cluster.primary.published.timestamp(),
        ),
    )


def _cluster_priority(cluster: ArticleCluster, reference_time: datetime) -> float:
    """Compute a priority score that boosts cross-source coverage."""

    frequency_score = min(cluster.source_count / 3.0, 1.0) * 0.6
    recency_score = _recency_factor(cluster.primary.published, reference_time) * 0.4
    return frequency_score + recency_score


def _recency_factor(published: datetime, reference_time: datetime) -> float:
    """Return a 0-1 recency score within the 24-hour fetch window."""

    age_seconds = max(0.0, (reference_time - published).total_seconds())
    window_seconds = 24 * 60 * 60
    return max(0.0, 1.0 - min(age_seconds, window_seconds) / window_seconds)


def _make_cluster(article: Article) -> ArticleCluster:
    """Create a new cluster for a primary article."""

    digest = hashlib.sha1(article.guid.encode("utf-8")).hexdigest()[:12]
    return ArticleCluster(cluster_id=digest, primary=article)


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


def _load_seen_cache(cache_path: Path) -> dict[str, str]:
    """Load GUID cache from disk."""

    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Cache file is invalid JSON, rebuilding: %s", cache_path)
        return {}
    if not isinstance(payload, dict):
        LOGGER.warning("Cache file is not a mapping, rebuilding: %s", cache_path)
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _prune_seen_cache(cache: dict[str, str], reference_time: datetime) -> dict[str, str]:
    """Keep only GUIDs from the last seven days."""

    cutoff = reference_time - timedelta(days=7)
    pruned: dict[str, str] = {}
    for guid, timestamp in cache.items():
        parsed = _parse_cache_time(timestamp)
        if parsed and parsed >= cutoff:
            pruned[guid] = parsed.isoformat()
    return pruned


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


def _update_seen_cache(
    cache: dict[str, str],
    clusters: list[ArticleCluster],
    reference_time: datetime,
) -> dict[str, str]:
    """Add every article GUID from selected clusters to the seen cache."""

    updated = dict(cache)
    for cluster in clusters:
        for article in cluster.all_articles:
            updated[article.guid] = reference_time.isoformat()
    return _prune_seen_cache(updated, reference_time)


def _save_seen_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """Persist GUID cache to disk."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
