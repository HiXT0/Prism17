"""Dense retrieval of SDG targets for report passages.

The embedding model and FAISS index run locally on CPU. Target embeddings are
cached to disk; passage embeddings are cached per document and model so reruns
do not re-encode the same report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import configured_path, get
from .models import Candidate
from .parse import ParsedDocument, Passage

LOGGER = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}


def _load_sdg_data(config: dict[str, Any]) -> dict[str, Any]:
    data_dir = configured_path(config, "paths", "data_dir")
    target_path = data_dir / "sdg_targets.json"
    with target_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_goals(
    config: dict[str, Any],
    focus_goals: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Load SDG goal metadata, optionally restricted to a set of goals."""
    goals = list(_load_sdg_data(config).get("goals", []))
    if focus_goals:
        focus = {int(goal) for goal in focus_goals}
        goals = [goal for goal in goals if int(goal["goal"]) in focus]
    return goals


def load_targets(
    config: dict[str, Any],
    focus_goals: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Load SDG targets, optionally restricted to a set of goals."""
    targets = list(_load_sdg_data(config).get("targets", []))
    if focus_goals:
        focus = {int(goal) for goal in focus_goals}
        targets = [target for target in targets if int(target["goal"]) in focus]
    if not targets:
        raise ValueError("No SDG targets selected for retrieval")
    return targets


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def _texts_digest(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _embedding_cache_path(
    cache_dir: Path,
    prefix: str,
    model_name: str,
    texts: Sequence[str],
) -> Path:
    return cache_dir / f"{prefix}_{_model_slug(model_name)}_{_texts_digest(texts)}.npy"


def _load_model(model_name: str) -> Any:
    """Load a SentenceTransformer once per process, on CPU."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "sentence-transformers is not installed. Run "
            "`pip install -r requirements.txt`."
        ) from exc

    LOGGER.info("Loading embedding model on CPU: %s", model_name)
    model = SentenceTransformer(model_name, device="cpu")
    _MODEL_CACHE[model_name] = model
    return model


def embed_texts(
    texts: Sequence[str],
    config: dict[str, Any],
    *,
    cache_prefix: str,
    query_prefix: str = "",
) -> np.ndarray:
    """Embed texts with disk caching.

    Embeddings are L2-normalised so FAISS inner product is cosine similarity.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    retrieval_cfg = get(config, "retrieval", default={}) or {}
    model_name = str(retrieval_cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"))
    batch_size = int(retrieval_cfg.get("batch_size", 32))
    cache_dir = configured_path(config, "retrieval", "cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)

    prefixed = [f"{query_prefix}{text}" for text in texts]
    cache_path = _embedding_cache_path(
        cache_dir, cache_prefix, model_name, prefixed
    )
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            if cached.shape[0] == len(texts):
                LOGGER.debug("Loaded embedding cache: %s", cache_path)
                return cached.astype("float32")
            LOGGER.warning("Ignoring stale embedding cache: %s", cache_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not read embedding cache %s: %s", cache_path, exc)

    model = _load_model(model_name)
    LOGGER.info("Embedding %s texts with %s", len(texts), model_name)
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    np.save(cache_path, embeddings)
    return embeddings


def _search_targets(
    query_embeddings: np.ndarray,
    targets: Sequence[dict[str, Any]],
    target_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the top-k target indices and cosine scores for each query."""
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "faiss-cpu is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    if target_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("Target and query embeddings have different dimensions")

    index = faiss.IndexFlatIP(target_embeddings.shape[1])
    index.add(target_embeddings)
    k_effective = min(max(1, int(k)), len(targets))
    return index.search(query_embeddings.astype("float32"), k_effective)


def rank_targets_for_texts(
    texts: Sequence[str],
    config: dict[str, Any],
    *,
    focus_goals: Sequence[int] | None = None,
    top_k: int | None = None,
    cache_prefix: str = "passages",
) -> list[list[dict[str, Any]]]:
    """Rank SDG targets for arbitrary texts, without applying a threshold.

    The result is aligned with ``texts``. Each entry is a list of up to
    ``top_k`` dictionaries with ``target_id``, ``target_goal``,
    ``target_text``, and ``score``. No candidates are dropped, so callers can
    evaluate several thresholds after retrieval (for example, in evaluation
    tooling). Report auditing should use :func:`retrieve_candidates`.
    """
    if not texts:
        return []

    retrieval_cfg = get(config, "retrieval", default={}) or {}
    k = int(top_k if top_k is not None else retrieval_cfg.get("top_k", 3))
    query_prefix = str(retrieval_cfg.get("query_prefix", "") or "")

    targets = load_targets(config, focus_goals)
    target_texts = [str(target["text"]) for target in targets]
    target_embeddings = embed_texts(
        target_texts,
        config,
        cache_prefix="targets",
        query_prefix="",
    )
    query_embeddings = embed_texts(
        list(texts),
        config,
        cache_prefix=cache_prefix,
        query_prefix=query_prefix,
    )
    scores, indices = _search_targets(
        query_embeddings, targets, target_embeddings, k
    )

    ranked_texts: list[list[dict[str, Any]]] = []
    for row_scores, row_indices in zip(scores, indices):
        ranked: list[dict[str, Any]] = []
        for score, target_index in zip(row_scores, row_indices):
            if target_index < 0:
                continue
            target = targets[int(target_index)]
            ranked.append(
                {
                    "target_id": str(target["id"]),
                    "target_goal": int(target["goal"]),
                    "target_text": str(target["text"]),
                    "score": round(float(score), 6),
                }
            )
        ranked_texts.append(ranked)
    return ranked_texts


def retrieve_candidates(
    document: ParsedDocument,
    config: dict[str, Any],
    *,
    focus_goals: Sequence[int] | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    max_passages: int | None = None,
) -> list[Candidate]:
    """Return candidate (passage, SDG target) pairs above the score threshold."""
    retrieval_cfg = get(config, "retrieval", default={}) or {}
    k = int(top_k if top_k is not None else retrieval_cfg.get("top_k", 3))
    min_score = float(
        threshold
        if threshold is not None
        else retrieval_cfg.get("score_threshold", 0.35)
    )
    query_prefix = str(retrieval_cfg.get("query_prefix", "") or "")

    targets = load_targets(config, focus_goals)
    passages: list[Passage] = [
        passage for passage in document.passages if passage.text.strip()
    ]
    if max_passages is not None:
        passages = passages[: max(0, int(max_passages))]
    if not passages:
        return []

    target_texts = [str(target["text"]) for target in targets]
    target_embeddings = embed_texts(
        target_texts,
        config,
        cache_prefix="targets",
        query_prefix="",
    )
    passage_texts = [passage.text for passage in passages]
    passage_embeddings = embed_texts(
        passage_texts,
        config,
        cache_prefix="passages",
        query_prefix=query_prefix,
    )
    scores, indices = _search_targets(
        passage_embeddings, targets, target_embeddings, k
    )

    candidates: list[Candidate] = []
    for passage_index, passage in enumerate(passages):
        for score, target_index in zip(scores[passage_index], indices[passage_index]):
            if target_index < 0:
                continue
            score_value = float(score)
            if score_value < min_score:
                continue
            target = targets[int(target_index)]
            candidates.append(
                Candidate(
                    passage_id=passage.passage_id,
                    page=passage.page,
                    target_id=str(target["id"]),
                    target_goal=int(target["goal"]),
                    target_text=str(target["text"]),
                    score=round(score_value, 6),
                    passage_text=passage.text,
                    section_heading=passage.section_heading,
                    is_table=passage.is_table,
                )
            )

    LOGGER.info(
        "Retrieved %s candidates from %s passages across %s targets",
        len(candidates),
        len(passages),
        len(targets),
    )
    return candidates
