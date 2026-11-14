"""Local LLM judge for SDG disclosure candidates.

The judge runs only through Ollama on localhost. It requests strict JSON,
validates the schema, retries once, and caches every call by
(model + system prompt + user prompt). A second failure becomes an unscored
finding for human review rather than crashing the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import PROJECT_ROOT, configured_path, get
from .models import Candidate, JudgedFinding
from .retrieve import load_targets

LOGGER = logging.getLogger(__name__)

DIMENSION_KEYS = (
    "specificity",
    "measurable_indicator",
    "deadline",
    "baseline",
    "reported_progress",
)
VALID_STATUSES = {"planned", "ongoing", "achieved", "unclear"}
PROMPT_DIR = PROJECT_ROOT / "prompts"

# Version string for the standalone relevance prompt used by evaluation
# tooling. It is part of the cache key, so changing the prompt invalidates old
# judgements instead of silently reusing them.
RELEVANCE_PROMPT_VERSION = "relevance-v1"
SUPPORT_PROMPT_VERSION = "support-v1"

RELEVANCE_SYSTEM_PROMPT = (
    "You are a careful annotator for the UN Sustainable Development Goals. "
    "Decide whether the TEXT addresses the given SDG target. Judge topical "
    "relevance only. Do not assess real-world performance, intent, or whether "
    "the target has been achieved. Return one JSON object with exactly these "
    'keys: "relevant" (boolean), "confidence" (number from 0 to 1), and '
    '"reason" (a short phrase). Do not add markdown or any other text.'
)

SUPPORT_SYSTEM_PROMPT = (
    "You are a careful evidence checker. Decide whether the QUOTE supports the "
    "CLAIM. A quote supports a claim when the claim is a faithful restatement "
    "of information stated in the quote. Do not use outside knowledge and do "
    "not assess real-world performance. Return one JSON object with exactly "
    'these keys: "supported" ("yes", "no", or "unclear"), "confidence" '
    "(number from 0 to 1), and \"reason\" (a short phrase). Do not add markdown "
    "or any other text."
)


class JudgeUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


class JudgeOutputError(ValueError):
    """Raised when a judge response is not valid JSON or schema-conforming."""


def load_prompt_bundle(prompt_dir: Path | None = None) -> str:
    """Load the judge instructions and few-shot examples as one system prompt."""
    directory = prompt_dir or PROMPT_DIR
    system_path = directory / "judge_system.txt"
    examples_path = directory / "judge_examples.txt"
    try:
        system = system_path.read_text(encoding="utf-8").strip()
        examples = examples_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Judge prompt files are missing in {directory}"
        ) from exc
    return f"{system}\n\n{examples}\n"


def build_user_prompt(candidate: Candidate) -> str:
    """Build the per-candidate user prompt."""
    return (
        f"Candidate SDG target:\n"
        f"{candidate.target_id} (Goal {candidate.target_goal}): "
        f"{candidate.target_text}\n\n"
        f"Passage (page {candidate.page}, passage {candidate.passage_id}):\n"
        f"{candidate.passage_text}\n\n"
        "Return the JSON object now."
    )


def build_relevance_prompt(
    *,
    text: str,
    target_id: str,
    target_goal: int,
    target_text: str,
) -> str:
    """Build the per-text relevance prompt used by evaluation tooling."""
    return (
        f"SDG target {target_id} (Goal {target_goal}): {target_text}\n\n"
        f"TEXT:\n{text}\n\n"
        "Return the JSON object now."
    )


@dataclass
class RelevanceJudgement:
    """Binary relevance result for one (text, target) pair."""

    relevant: bool | None
    confidence: float
    reason: str
    status: str
    raw_response: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def validate_relevance_data(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise one relevance JSON object.

    Raises:
        JudgeOutputError: if required fields are missing or malformed.
    """
    relevant = data.get("relevant")
    if not isinstance(relevant, bool):
        raise JudgeOutputError("relevant must be a boolean")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise JudgeOutputError("confidence must be a number") from exc
    confidence = min(1.0, max(0.0, confidence))

    reason = str(data.get("reason", "") or "").strip()
    if len(reason) > 300:
        reason = reason[:300].rstrip()

    return {
        "relevant": relevant,
        "confidence": round(confidence, 4),
        "reason": reason,
    }


def build_support_prompt(*, claim: str, quote: str) -> str:
    """Build the claim/quote support prompt used by Stage 2 (LLM method)."""
    return (
        f"CLAIM:\n{claim}\n\n"
        f"QUOTE:\n{quote}\n\n"
        "Return the JSON object now."
    )


@dataclass
class SupportJudgement:
    """Semantic-support result for one claim/quote pair."""

    supported: str
    confidence: float
    reason: str
    status: str
    raw_response: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def validate_support_data(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise one support JSON object."""
    supported = str(data.get("supported", "") or "").strip().lower()
    if supported not in {"yes", "no", "unclear"}:
        raise JudgeOutputError("supported must be yes, no, or unclear")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise JudgeOutputError("confidence must be a number") from exc
    confidence = min(1.0, max(0.0, confidence))

    reason = str(data.get("reason", "") or "").strip()
    if len(reason) > 300:
        reason = reason[:300].rstrip()

    return {
        "supported": supported,
        "confidence": round(confidence, 4),
        "reason": reason,
    }


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from a model response."""
    if not raw or not raw.strip():
        raise JudgeOutputError("empty response")

    text = raw.strip().lstrip("\ufeff")
    if text.startswith("```"):
        # Remove a Markdown fence if the model ignored the instruction.
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise JudgeOutputError("no JSON object found")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeOutputError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeOutputError("JSON root is not an object")
    return data


def _require_bool(mapping: dict[str, Any], key: str, context: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise JudgeOutputError(f"{context}.{key} must be a boolean")
    return value


def validate_judge_data(
    data: dict[str, Any],
    candidate: Candidate,
    target_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalise one judge JSON object.

    Raises:
        JudgeOutputError: if required fields are missing or malformed.
    """
    relevant = data.get("relevant")
    if not isinstance(relevant, bool):
        raise JudgeOutputError("relevant must be a boolean")

    target_id = str(data.get("sdg_target", "")).strip()
    if not re.fullmatch(r"\d{1,2}\.(?:\d{1,2}|[a-z])[a-z]?", target_id):
        raise JudgeOutputError(f"invalid sdg_target: {target_id!r}")
    if target_ids is not None and target_id not in target_ids:
        raise JudgeOutputError(f"sdg_target is not in the UN target set: {target_id}")
    if target_id != candidate.target_id:
        raise JudgeOutputError(
            f"sdg_target {target_id} does not match candidate target "
            f"{candidate.target_id}"
        )

    goal = int(target_id.split(".", 1)[0])
    if not 1 <= goal <= 17:
        raise JudgeOutputError(f"invalid sdg_goal: {goal}")

    claim = str(data.get("claim", "") or "").strip()
    quote = str(data.get("supporting_quote", "") or "").strip()
    if relevant:
        if not claim:
            raise JudgeOutputError("claim is empty for a relevant finding")
        if not quote:
            raise JudgeOutputError("supporting_quote is empty for a relevant finding")
        if len(quote.split()) < 4:
            raise JudgeOutputError("supporting_quote is too short to verify")

    dimensions_raw = data.get("dimensions")
    if not isinstance(dimensions_raw, dict):
        raise JudgeOutputError("dimensions must be an object")
    dimensions = {
        key: _require_bool(dimensions_raw, key, "dimensions")
        for key in DIMENSION_KEYS
    }

    extracted_raw = data.get("extracted")
    if not isinstance(extracted_raw, dict):
        raise JudgeOutputError("extracted must be an object")
    status = str(extracted_raw.get("status", "unclear") or "unclear").lower()
    if status not in VALID_STATUSES:
        status = "unclear"
    negated = extracted_raw.get("negated", False)
    if isinstance(negated, str) and negated.lower() in {"true", "false"}:
        negated = negated.lower() == "true"
    if not isinstance(negated, bool):
        raise JudgeOutputError("extracted.negated must be a boolean")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise JudgeOutputError("confidence must be a number") from exc
    confidence = min(1.0, max(0.0, confidence))

    return {
        "relevant": relevant,
        "sdg_goal": goal,
        "sdg_target": target_id,
        "claim": claim,
        "supporting_quote": quote,
        "dimensions": dimensions,
        "extracted": {
            "value": str(extracted_raw.get("value", "") or "").strip(),
            "unit": str(extracted_raw.get("unit", "") or "").strip(),
            "year": str(extracted_raw.get("year", "") or "").strip(),
            "status": status,
            "negated": negated,
        },
        "confidence": round(confidence, 4),
    }


def _finding_from_data(
    data: dict[str, Any],
    candidate: Candidate,
    raw_response: str,
) -> JudgedFinding:
    return JudgedFinding(
        candidate=candidate,
        relevant=bool(data["relevant"]),
        sdg_goal=int(data["sdg_goal"]),
        sdg_target=str(data["sdg_target"]),
        claim=str(data["claim"]),
        supporting_quote=str(data["supporting_quote"]),
        dimensions=dict(data["dimensions"]),
        extracted=dict(data["extracted"]),
        confidence=float(data["confidence"]),
        raw_response=raw_response,
    )


def _unscored_finding(
    candidate: Candidate,
    raw_response: str,
    reason: str,
) -> JudgedFinding:
    return JudgedFinding(
        candidate=candidate,
        relevant=False,
        sdg_goal=candidate.target_goal,
        sdg_target=candidate.target_id,
        claim="",
        supporting_quote="",
        dimensions={key: False for key in DIMENSION_KEYS},
        extracted={
            "value": "",
            "unit": "",
            "year": "",
            "status": "unclear",
            "negated": False,
        },
        confidence=0.0,
        raw_response=raw_response,
        scored=False,
        review_reason=reason,
    )


class _OllamaRuntime:
    """Shared local Ollama client handling for strict-JSON callers."""

    def __init__(
        self,
        config: dict[str, Any],
        model: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        llm_cfg = get(config, "llm", default={}) or {}
        self.config = config
        self.model = str(model or llm_cfg.get("model", "qwen2.5:7b-instruct"))
        self.base_url = str(llm_cfg.get("base_url", "http://localhost:11434"))
        self.temperature = float(llm_cfg.get("temperature", 0))
        self.num_ctx = int(llm_cfg.get("num_ctx", 4096))
        self.max_predict = int(llm_cfg.get("max_predict_tokens", 600))
        self.timeout = float(llm_cfg.get("request_timeout_seconds", 300))
        self.retries = max(0, int(llm_cfg.get("retries", 1)))
        self.cache_dir = configured_path(config, "llm", "cache_dir")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._client_injected = client is not None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise JudgeUnavailableError(
                "The ollama Python package is not installed. Run "
                "`pip install -r requirements.txt`."
            ) from exc

        try:
            self._client = ollama.Client(host=self.base_url, timeout=self.timeout)
        except TypeError:  # older ollama clients may not accept timeout
            self._client = ollama.Client(host=self.base_url)
        return self._client

    def ensure_available(self) -> None:
        """Fail fast with an actionable message if Ollama or the model is missing."""
        if self._client_injected:
            return
        try:
            response = self._get_client().list()
        except Exception as exc:  # noqa: BLE001
            raise JudgeUnavailableError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and run "
                f"`ollama pull {self.model}`. Details: {exc}"
            ) from exc

        names: set[str] = set()
        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])
        for item in models or []:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
            else:
                name = getattr(item, "model", None) or getattr(item, "name", None)
            if name:
                names.add(str(name))

        if names and self.model not in names:
            implicit_tag = (
                ":" not in self.model
                and any(name.startswith(f"{self.model}:") for name in names)
            )
            if not implicit_tag:
                raise JudgeUnavailableError(
                    f"Ollama is running but model {self.model!r} is not installed. "
                    f"Run `ollama pull {self.model}`."
                )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.max_predict,
        }
        try:
            response = self._get_client().chat(
                model=self.model,
                messages=messages,
                format="json",
                options=options,
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001 - local runtime failure
            raise JudgeUnavailableError(
                f"Ollama request failed for model {self.model!r}: {exc}"
            ) from exc
        if isinstance(response, dict):
            message = response.get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else ""
        else:  # pragma: no cover - newer ollama client response objects
            message = getattr(response, "message", None)
            content = getattr(message, "content", "") if message is not None else ""
        if not content:
            raise JudgeOutputError("model returned an empty response")
        return str(content)


class OllamaJudge(_OllamaRuntime):
    """Strict-JSON judge backed by a local Ollama model."""

    def __init__(
        self,
        config: dict[str, Any],
        model: str | None = None,
        *,
        client: Any | None = None,
        prompt_dir: Path | None = None,
    ) -> None:
        super().__init__(config, model, client=client)
        self.system_prompt = load_prompt_bundle(prompt_dir)
        self.target_map = {
            str(target["id"]): target for target in load_targets(config)
        }

    def _cache_path(self, user_prompt: str) -> Path:
        digest = hashlib.sha256(
            (
                "judge-v1\n"
                + self.model
                + "\n"
                + self.system_prompt
                + "\n"
                + user_prompt
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def judge(self, candidate: Candidate) -> JudgedFinding:
        """Judge one candidate, with cache and one schema-repair retry."""
        user_prompt = build_user_prompt(candidate)
        cache_path = self._cache_path(user_prompt)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("ok") and isinstance(cached.get("data"), dict):
                    LOGGER.debug("LLM cache hit: %s", cache_path.name)
                    return _finding_from_data(
                        cached["data"], candidate, str(cached.get("raw", ""))
                    )
                if not cached.get("ok"):
                    return _unscored_finding(
                        candidate,
                        str(cached.get("raw", "")),
                        str(cached.get("error", "cached malformed output")),
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Ignoring unreadable LLM cache %s: %s", cache_path, exc)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_raw = ""
        last_error = "unknown error"

        for attempt in range(self.retries + 1):
            try:
                last_raw = self._chat(messages)
                parsed = parse_json_object(last_raw)
                data = validate_judge_data(
                    parsed, candidate, set(self.target_map)
                )
                cache_path.write_text(
                    json.dumps(
                        {"ok": True, "raw": last_raw, "data": data},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return _finding_from_data(data, candidate, last_raw)
            except (JudgeOutputError, ValueError, TypeError) as exc:
                last_error = str(exc)
                LOGGER.warning(
                    "Judge output invalid for %s (attempt %s/%s): %s",
                    candidate.passage_id,
                    attempt + 1,
                    self.retries + 1,
                    last_error,
                )
                if attempt < self.retries:
                    messages.extend(
                        [
                            {"role": "assistant", "content": last_raw},
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was invalid: "
                                    f"{last_error}. Return only one corrected JSON "
                                    "object matching the schema. Do not add markdown."
                                ),
                            },
                        ]
                    )

        reason = f"unscored (needs human review): malformed LLM output: {last_error}"
        cache_path.write_text(
            json.dumps(
                {"ok": False, "raw": last_raw, "error": reason},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return _unscored_finding(candidate, last_raw, reason)


class RelevanceJudge(_OllamaRuntime):
    """Binary relevance judge for evaluation data.

    This is deliberately separate from :class:`OllamaJudge`: it asks only
    whether a text addresses a target, without extracting a claim, quote, or
    rubric dimensions. It shares the local Ollama runtime and caching style.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: str | None = None,
        *,
        client: Any | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(config, model, client=client)
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.cache_dir / "relevance"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str, target_id: str) -> Path:
        digest = hashlib.sha256(
            (
                RELEVANCE_PROMPT_VERSION
                + "\n"
                + self.model
                + "\n"
                + RELEVANCE_SYSTEM_PROMPT
                + "\n"
                + target_id
                + "\n"
                + text
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def judge(
        self,
        text: str,
        *,
        target_id: str,
        target_goal: int,
        target_text: str,
    ) -> RelevanceJudgement:
        """Return a cached binary relevance judgement for one text/target pair."""
        user_prompt = build_relevance_prompt(
            text=text,
            target_id=target_id,
            target_goal=target_goal,
            target_text=target_text,
        )
        cache_path = self._cache_path(text, target_id)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("ok") and isinstance(cached.get("data"), dict):
                    data = cached["data"]
                    return RelevanceJudgement(
                        relevant=bool(data.get("relevant")),
                        confidence=float(data.get("confidence", 0.0)),
                        reason=str(data.get("reason", "")),
                        status="ok",
                        raw_response=str(cached.get("raw", "")),
                    )
                if not cached.get("ok"):
                    return RelevanceJudgement(
                        relevant=None,
                        confidence=0.0,
                        reason="",
                        status="error",
                        raw_response=str(cached.get("raw", "")),
                        error=str(cached.get("error", "cached malformed output")),
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Ignoring unreadable relevance cache %s: %s", cache_path, exc
                )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        last_raw = ""
        last_error = "unknown error"

        for attempt in range(self.retries + 1):
            try:
                last_raw = self._chat(messages)
                parsed = parse_json_object(last_raw)
                data = validate_relevance_data(parsed)
                cache_path.write_text(
                    json.dumps(
                        {"ok": True, "raw": last_raw, "data": data},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return RelevanceJudgement(
                    relevant=bool(data["relevant"]),
                    confidence=float(data["confidence"]),
                    reason=str(data["reason"]),
                    status="ok",
                    raw_response=last_raw,
                )
            except (JudgeOutputError, ValueError, TypeError) as exc:
                last_error = str(exc)
                LOGGER.warning(
                    "Relevance judge output invalid for %s (attempt %s/%s): %s",
                    target_id,
                    attempt + 1,
                    self.retries + 1,
                    last_error,
                )
                if attempt < self.retries:
                    messages.extend(
                        [
                            {"role": "assistant", "content": last_raw},
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was invalid: "
                                    f"{last_error}. Return only one corrected JSON "
                                    "object with the keys relevant, confidence, and "
                                    "reason. Do not add markdown."
                                ),
                            },
                        ]
                    )

        error = f"malformed relevance output: {last_error}"
        cache_path.write_text(
            json.dumps(
                {"ok": False, "raw": last_raw, "error": error},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return RelevanceJudgement(
            relevant=None,
            confidence=0.0,
            reason="",
            status="error",
            raw_response=last_raw,
            error=error,
        )


class SupportJudge(_OllamaRuntime):
    """Claim/quote support judge for Stage 2 semantic verification."""

    def __init__(
        self,
        config: dict[str, Any],
        model: str | None = None,
        *,
        client: Any | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(config, model, client=client)
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.cache_dir / "support"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, claim: str, quote: str) -> Path:
        digest = hashlib.sha256(
            (
                SUPPORT_PROMPT_VERSION
                + "\n"
                + self.model
                + "\n"
                + SUPPORT_SYSTEM_PROMPT
                + "\n"
                + claim
                + "\n"
                + quote
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def judge(self, claim: str, quote: str) -> SupportJudgement:
        """Return a cached support judgement for one claim/quote pair."""
        user_prompt = build_support_prompt(claim=claim, quote=quote)
        cache_path = self._cache_path(claim, quote)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("ok") and isinstance(cached.get("data"), dict):
                    data = cached["data"]
                    return SupportJudgement(
                        supported=str(data.get("supported", "unclear")),
                        confidence=float(data.get("confidence", 0.0)),
                        reason=str(data.get("reason", "")),
                        status="ok",
                        raw_response=str(cached.get("raw", "")),
                    )
                if not cached.get("ok"):
                    return SupportJudgement(
                        supported="unclear",
                        confidence=0.0,
                        reason="",
                        status="error",
                        raw_response=str(cached.get("raw", "")),
                        error=str(cached.get("error", "cached malformed output")),
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Ignoring unreadable support cache %s: %s", cache_path, exc
                )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SUPPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        last_raw = ""
        last_error = "unknown error"

        for attempt in range(self.retries + 1):
            try:
                last_raw = self._chat(messages)
                parsed = parse_json_object(last_raw)
                data = validate_support_data(parsed)
                cache_path.write_text(
                    json.dumps(
                        {"ok": True, "raw": last_raw, "data": data},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return SupportJudgement(
                    supported=str(data["supported"]),
                    confidence=float(data["confidence"]),
                    reason=str(data["reason"]),
                    status="ok",
                    raw_response=last_raw,
                )
            except (JudgeOutputError, ValueError, TypeError) as exc:
                last_error = str(exc)
                LOGGER.warning(
                    "Support judge output invalid (attempt %s/%s): %s",
                    attempt + 1,
                    self.retries + 1,
                    last_error,
                )
                if attempt < self.retries:
                    messages.extend(
                        [
                            {"role": "assistant", "content": last_raw},
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was invalid: "
                                    f"{last_error}. Return only one corrected JSON "
                                    "object with the keys supported, confidence, "
                                    "and reason. Do not add markdown."
                                ),
                            },
                        ]
                    )

        error = f"malformed support output: {last_error}"
        cache_path.write_text(
            json.dumps(
                {"ok": False, "raw": last_raw, "error": error},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return SupportJudgement(
            supported="unclear",
            confidence=0.0,
            reason="",
            status="error",
            raw_response=last_raw,
            error=error,
        )


def judge_candidates(
    candidates: Sequence[Candidate],
    config: dict[str, Any],
    *,
    model: str | None = None,
    judge: OllamaJudge | None = None,
    progress_callback: Callable[[int, int, JudgedFinding], None] | None = None,
) -> list[JudgedFinding]:
    """Judge a sequence of candidates, optionally reporting progress."""
    active_judge = judge or OllamaJudge(config, model=model)
    active_judge.ensure_available()
    findings: list[JudgedFinding] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        finding = active_judge.judge(candidate)
        findings.append(finding)
        if progress_callback is not None:
            progress_callback(index, total, finding)
    return findings
