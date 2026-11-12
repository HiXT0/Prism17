"""Phase 2 CLI pipeline: parse -> retrieve -> judge -> verify -> rubric.

Every scored finding in the console and JSON output has a 1-indexed page number
and a quote verified against the source page. Unverified or malformed findings
are retained in the JSON with an explicit review reason, but they are never
displayed as scored.

Run:
    python -m prism17.pipeline data/reports/example.pdf
    python -m prism17.pipeline data/reports/example.pdf --out outputs/example.audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import get, load_config, resolve_path
from .judge import (
    JudgeUnavailableError,
    OllamaJudge,
    judge_candidates,
)
from .models import (
    Candidate,
    JudgedFinding,
    VerifiedFinding,
    is_not_relevant,
    needs_human_review,
)
from .parse import ParsedDocument, parse_pdf, summarise
from .retrieve import retrieve_candidates
from .rubric import grade_findings
from .semantic import apply_semantic_support
from .verify import verify_findings

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything produced by one audit run."""

    document: ParsedDocument
    candidates: list[Candidate] = field(default_factory=list)
    judged: list[JudgedFinding] = field(default_factory=list)
    findings: list[VerifiedFinding] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def scored(self) -> list[VerifiedFinding]:
        return [finding for finding in self.findings if finding.is_scored]

    @property
    def needs_review(self) -> list[VerifiedFinding]:
        """Findings that failed verification or schema checks.

        Passages the judge correctly marked irrelevant are not review items.
        """
        return [
            finding for finding in self.findings if needs_human_review(finding)
        ]

    @property
    def not_relevant(self) -> list[VerifiedFinding]:
        return [
            finding for finding in self.findings if is_not_relevant(finding)
        ]

    def stats(self) -> dict[str, Any]:
        scored = self.scored
        level_counts = {
            str(level): sum(1 for finding in scored if finding.level == level)
            for level in (1, 2, 3, 4)
        }
        return {
            "pages": self.document.page_count,
            "passages": len(self.document.passages),
            "unreadable_pages": len(self.document.unreadable_pages),
            "candidates": len(self.candidates),
            "judged": len(self.judged),
            "verified": sum(1 for finding in self.findings if finding.quote_verified),
            "scored": len(scored),
            "needs_review": len(self.needs_review),
            "not_relevant": len(self.not_relevant),
            "level_counts": level_counts,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.2",
            "document": {
                "pdf_path": self.document.pdf_path,
                "title": self.document.title,
                "page_count": self.document.page_count,
                "sha256": self.document.metadata.get("sha256"),
            },
            "settings": {
                "model": get(self.config, "llm", "model", default=None),
                "embedding_model": get(
                    self.config, "retrieval", "embedding_model", default=None
                ),
                "focus_sdgs": get(self.config, "report", "focus_sdgs", default=None),
                "semantic_method": get(
                    self.config, "verification", "semantic_method", default=None
                ),
                "semantic_enabled": bool(
                    get(self.config, "verification", "semantic_enabled", default=False)
                ),
            },
            "stats": self.stats(),
            "parse": self.document.to_dict(include_pages=False),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _select_candidates(
    candidates: Sequence[Candidate],
    max_candidates: int | None,
) -> list[Candidate]:
    """Keep the highest-scoring candidates when a safety limit is configured."""
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    if max_candidates is None:
        return ordered
    return ordered[: max(0, int(max_candidates))]


def _progress_printer(
    every: int = 5,
) -> Callable[[int, int, JudgedFinding], None]:
    def report(index: int, total: int, finding: JudgedFinding) -> None:
        if index == 1 or index == total or index % max(1, every) == 0:
            state = "relevant" if finding.relevant else "not relevant"
            if finding.review_reason:
                state = "needs review"
            print(
                f"  judged {index}/{total} (page {finding.candidate.page}, "
                f"{finding.candidate.target_id}: {state})",
                file=sys.stderr,
            )

    return report


def run_pipeline(
    pdf_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    model: str | None = None,
    focus_goals: Sequence[int] | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    max_candidates: int | None = None,
    max_pages: int | None = None,
    extract_tables_enabled: bool | None = None,
    fuzzy_threshold: float | None = None,
    semantic_check: bool | None = None,
    judge: OllamaJudge | None = None,
    progress_callback: Callable[[int, int, JudgedFinding], None] | None = None,
) -> PipelineResult:
    """Run the full Phase 2 audit pipeline on one PDF."""
    started = time.perf_counter()
    cfg = config or load_config()
    if model:
        cfg = {**cfg, "llm": {**(get(cfg, "llm", default={}) or {}), "model": model}}

    document = parse_pdf(
        pdf_path,
        cfg,
        extract_tables_enabled=extract_tables_enabled,
        max_pages=max_pages,
    )
    LOGGER.info("Parsed %s passages", len(document.passages))

    effective_goals = focus_goals
    if effective_goals is None:
        configured_goals = get(cfg, "report", "focus_sdgs", default=None)
        if configured_goals:
            effective_goals = [int(goal) for goal in configured_goals]
    if effective_goals is not None:
        cfg = {
            **cfg,
            "report": {
                **(get(cfg, "report", default={}) or {}),
                "focus_sdgs": list(effective_goals),
            },
        }

    candidates = retrieve_candidates(
        document,
        cfg,
        focus_goals=effective_goals,
        top_k=top_k,
        threshold=min_score,
    )
    if max_candidates is not None:
        limit: int | None = int(max_candidates)
    else:
        configured_limit = get(cfg, "llm", "max_candidates", default=180)
        limit = int(configured_limit) if configured_limit is not None else None
    candidates = _select_candidates(candidates, limit)
    LOGGER.info("Selected %s candidates for judging", len(candidates))

    judged: list[JudgedFinding] = []
    if candidates:
        judged = judge_candidates(
            candidates,
            cfg,
            model=model,
            judge=judge,
            progress_callback=progress_callback,
        )

    findings = verify_findings(judged, document, cfg)
    if fuzzy_threshold is not None:
        # Re-run verification with the CLI override rather than mutating config.
        findings = [
            _reverify(finding, document, fuzzy_threshold) for finding in findings
        ]

    semantic_enabled = (
        bool(get(cfg, "verification", "semantic_enabled", default=False))
        if semantic_check is None
        else bool(semantic_check)
    )
    cfg = {
        **cfg,
        "verification": {
            **(get(cfg, "verification", default={}) or {}),
            "semantic_enabled": semantic_enabled,
        },
    }
    if semantic_enabled and findings:
        findings = apply_semantic_support(findings, cfg)

    findings = grade_findings(findings, cfg)

    elapsed = time.perf_counter() - started
    return PipelineResult(
        document=document,
        candidates=list(candidates),
        judged=judged,
        findings=findings,
        config=cfg,
        elapsed_seconds=elapsed,
    )


def _reverify(
    finding: VerifiedFinding,
    document: ParsedDocument,
    threshold: float,
) -> VerifiedFinding:
    """Re-run stage 1 for a finding with a CLI threshold override."""
    from .verify import verify_finding

    if finding.judged.review_reason is not None:
        return finding
    return verify_finding(finding.judged, document, threshold=threshold)


def _print_finding(index: int, finding: VerifiedFinding) -> None:
    judged = finding.judged
    quote = judged.supporting_quote.strip()
    print(
        f"\n[{index}] Goal {judged.sdg_goal} | target {judged.sdg_target} | "
        f"Level {finding.level} | page {finding.source_page} | "
        f"quote {finding.quote_score:.1f} ({finding.quote_method})"
    )
    print(f"  Claim: {judged.claim}")
    print(f'  Quote: "{quote}"')
    print(f"  Rubric: {finding.level_rationale}")
    if finding.disclosure_gaps:
        print("  Disclosure gaps:")
        for gap in finding.disclosure_gaps:
            print(f"    - {gap}")


def print_report(result: PipelineResult, *, scored_only: bool = False) -> None:
    """Print a neutral, human-readable audit summary."""
    stats = result.stats()
    print(summarise(result.document))
    print(
        f"Candidates judged: {stats['candidates']} | "
        f"quote-verified: {stats['verified']} | "
        f"scored: {stats['scored']} | "
        f"not relevant: {stats['not_relevant']} | "
        f"needs review: {stats['needs_review']}"
    )
    print(
        "Level counts: "
        + ", ".join(
            f"L{level}={count}"
            for level, count in stats["level_counts"].items()
        )
    )

    scored = result.scored
    if scored:
        print(f"\nScored findings ({len(scored)})")
        for index, finding in enumerate(scored, start=1):
            _print_finding(index, finding)
    else:
        print("\nNo scored findings. No finding met both quote verification and rubric requirements.")

    if scored_only:
        return

    review = result.needs_review
    if review:
        print(f"\nNeeds human review ({len(review)})")
        for finding in review:
            judged = finding.judged
            reason = finding.review_reason or "not scored"
            print(
                f"  - page {finding.source_page}, target {judged.sdg_target}: "
                f"{reason}"
            )


def write_output(result: PipelineResult, out_path: str | Path) -> Path:
    """Write the full audit JSON, including unscored findings."""
    path = Path(out_path).expanduser()
    if not path.is_absolute():
        path = resolve_path(path, result.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _parse_goals(values: Sequence[str] | None) -> list[int] | None:
    if not values:
        return None
    goals: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            goal = int(part)
            if not 1 <= goal <= 17:
                raise ValueError(f"SDG goal out of range: {goal}")
            goals.append(goal)
    return goals or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit SDG disclosure in a PDF with local retrieval, a local "
            "Ollama judge, quote verification, and a deterministic rubric."
        )
    )
    parser.add_argument("pdf", help="Path to the PDF report")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--out", default=None, help="Write full audit JSON here")
    parser.add_argument(
        "--report",
        default=None,
        help="Write a Markdown SDG Alignment Report here",
    )
    parser.add_argument(
        "--html-report",
        default=None,
        help="Write a standalone HTML SDG Alignment Report here",
    )
    parser.add_argument("--model", default=None, help="Ollama model name override")
    parser.add_argument(
        "--sdg",
        nargs="+",
        default=None,
        help="Focus on these SDG goals, for example --sdg 8 12 13",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Targets per passage")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Retrieval cosine-similarity threshold",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=None,
        help="Quote fuzzy-match threshold (default: 95)",
    )
    parser.add_argument(
        "--semantic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable Stage 2 semantic-support verification (default: config "
            "verification.semantic_enabled). Use --no-semantic to skip it."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum candidates to judge (default: config llm.max_candidates)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None, help="Parse only the first N pages"
    )
    parser.add_argument(
        "--no-tables", action="store_true", help="Skip pdfplumber table extraction"
    )
    parser.add_argument(
        "--scored-only",
        action="store_true",
        help="Do not print the human-review list",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        goals = _parse_goals(args.sdg)
        config = load_config(args.config)
        result = run_pipeline(
            args.pdf,
            config,
            model=args.model,
            focus_goals=goals,
            top_k=args.top_k,
            min_score=args.min_score,
            max_candidates=args.max_candidates,
            max_pages=args.max_pages,
            extract_tables_enabled=not args.no_tables,
            fuzzy_threshold=args.fuzzy_threshold,
            semantic_check=args.semantic,
            progress_callback=_progress_printer(),
        )
    except JudgeUnavailableError as exc:
        LOGGER.error("%s", exc)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print_report(result, scored_only=args.scored_only)
    if args.out:
        path = write_output(result, args.out)
        print(f"\nWrote audit JSON: {path}")
    if args.report or args.html_report:
        from .report import (
            build_report_data,
            write_html_report,
            write_markdown_report,
        )

        report_data = build_report_data(result)
        if args.report:
            path = write_markdown_report(
                report_data, args.report, result.config
            )
            print(f"\nWrote Markdown report: {path}")
        if args.html_report:
            path = write_html_report(
                report_data, args.html_report, result.config
            )
            print(f"\nWrote HTML report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
