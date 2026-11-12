"""Phase 4 Streamlit UI: run or load an audit, review findings, export.

The UI is a thin layer over the existing pipeline, report, and review modules.
It never scores anything itself: scored findings come from the audit JSON, and
reviewer decisions are stored separately so the rubric result stays auditable.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .config import PROJECT_ROOT, load_config, resolve_path
from .pipeline import PipelineResult, run_pipeline, write_output
from .report import (
    ReportData,
    render_html,
    render_markdown,
    report_data_from_json,
    write_html_report,
    write_markdown_report,
)
from .review import (
    REVIEW_STATUSES,
    ReviewDecision,
    export_rows,
    load_reviews,
    make_decision,
    save_reviews,
    summarise_reviews,
)

DEFAULT_AUDIT = "outputs/sjs.phase2.json"
DEFAULT_PDF = "data/reports/sjs_india_brsr_fy2023_24.pdf"
MODEL_CHOICES = ("qwen2.5:7b-instruct", "llama3.1:8b", "gemma2:9b")

STATUS_HELP = {
    "pending": "Not reviewed yet.",
    "confirmed": "The rubric level is accepted as recorded.",
    "overridden": "The reviewer records a different level; the rubric level "
    "is kept for auditability.",
    "flagged": "The finding needs a second look before it is used.",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _default_review_path(audit_path: str | Path | None) -> Path | None:
    if not audit_path:
        return None
    path = Path(audit_path)
    return path.with_name(f"{path.stem}.reviews.json")


def _level_text(level: Any) -> str:
    return f"Level {int(level)}" if level not in (None, "") else "—"


def _short(text: Any, limit: int = 180) -> str:
    squashed = " ".join(str(text or "").split())
    if len(squashed) <= limit:
        return squashed
    return squashed[: limit - 1].rstrip() + "…"


def _payload_from_upload(uploaded: Any) -> dict[str, Any]:
    return json.loads(uploaded.getvalue().decode("utf-8"))


def _payload_from_path(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _initialise_state() -> None:
    if "payload" in st.session_state:
        return
    st.session_state.payload = None
    st.session_state.audit_label = ""
    st.session_state.audit_path = None
    st.session_state.reviews = {}
    st.session_state.editor_version = 0

    sample = PROJECT_ROOT / DEFAULT_AUDIT
    if sample.exists():
        try:
            _set_audit(_payload_from_path(sample), str(sample), sample)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            st.warning(f"Could not load the sample audit: {exc}")


def _set_audit(
    payload: dict[str, Any],
    label: str,
    audit_path: str | Path | None,
) -> None:
    st.session_state.payload = payload
    st.session_state.audit_label = label
    st.session_state.audit_path = str(audit_path) if audit_path else None
    review_path = _default_review_path(audit_path)
    st.session_state.reviews = (
        load_reviews(review_path) if review_path else {}
    )
    st.session_state.editor_version += 1


def _report_data(config: dict[str, Any]) -> ReportData:
    payload = st.session_state.payload
    if payload is None:
        raise ValueError("No audit loaded")
    data = report_data_from_json(
        payload, config, reviews=dict(st.session_state.reviews)
    )
    company = st.session_state.get("company_override", "").strip()
    title = st.session_state.get("title_override", "").strip()
    if company:
        data.company_name = company
    if title:
        data.report_title = title
    return data


# ---------------------------------------------------------------------------
# Sidebar: load or run an audit
# ---------------------------------------------------------------------------


def _sidebar_config() -> dict[str, Any]:
    config_path = st.sidebar.text_input("Config file", value="config.yaml")
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        st.sidebar.error(str(exc))
        st.stop()
    return config


def _sidebar_source(config: dict[str, Any]) -> None:
    st.sidebar.header("Audit source")
    mode = st.sidebar.radio(
        "Source",
        ("Load audit JSON", "Run new audit"),
        label_visibility="collapsed",
    )

    if mode == "Load audit JSON":
        path_value = st.sidebar.text_input("Audit JSON path", value=DEFAULT_AUDIT)
        uploaded = st.sidebar.file_uploader("Or upload audit JSON", type="json")
        if st.sidebar.button("Load audit", type="primary"):
            try:
                if uploaded is not None:
                    payload = _payload_from_upload(uploaded)
                    _set_audit(payload, uploaded.name, None)
                else:
                    resolved = resolve_path(path_value, config)
                    payload = _payload_from_path(resolved)
                    _set_audit(payload, str(resolved), resolved)
                st.rerun()
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                st.sidebar.error(f"Could not load audit: {exc}")
        return

    _sidebar_run_audit(config)


def _sidebar_run_audit(config: dict[str, Any]) -> None:
    default_goals = [
        int(goal)
        for goal in (
            (config.get("report") or {}).get("focus_sdgs") or [12]
        )
    ]
    goals = st.sidebar.multiselect(
        "Focus SDG goals",
        options=list(range(1, 18)),
        default=default_goals,
    )
    model = st.sidebar.selectbox(
        "Local judge model", MODEL_CHOICES, index=0
    )
    custom_model = st.sidebar.text_input(
        "Custom model (optional)", placeholder="Leave blank to use the list"
    )

    with st.sidebar.expander("Run limits", expanded=False):
        max_pages = st.number_input(
            "Max pages", min_value=1, max_value=1000, value=10, step=1
        )
        max_candidates = st.number_input(
            "Max candidates", min_value=1, max_value=1000, value=12, step=1
        )
        top_k = st.number_input(
            "Targets per passage", min_value=1, max_value=10, value=3, step=1
        )
        min_score = st.slider(
            "Retrieval threshold", 0.0, 0.9, 0.35, step=0.05
        )
        extract_tables = st.checkbox("Extract tables", value=True)

    pdf_upload = st.sidebar.file_uploader("PDF report", type="pdf")
    pdf_path = st.sidebar.text_input("Or PDF path", value=DEFAULT_PDF)
    out_path = st.sidebar.text_input(
        "Audit JSON output", value="outputs/ui.audit.json"
    )

    if not st.sidebar.button("Run audit", type="primary"):
        return

    effective_model = custom_model.strip() or model
    if not goals:
        st.sidebar.error("Select at least one SDG goal.")
        return

    temp_path: Path | None = None
    try:
        if pdf_upload is not None:
            handle = tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            )
            handle.write(pdf_upload.getvalue())
            handle.close()
            temp_path = Path(handle.name)
            source = temp_path
        else:
            source = resolve_path(pdf_path, config)

        progress = st.progress(0.0, text="Parsing and retrieving…")

        def on_progress(index: int, total: int, finding: Any) -> None:
            state = "relevant" if finding.relevant else "not relevant"
            if finding.review_reason:
                state = "needs review"
            progress.progress(
                min(1.0, index / max(1, total)),
                text=f"Judging {index}/{total} — page "
                f"{finding.candidate.page}, {finding.candidate.target_id}: "
                f"{state}",
            )

        with st.spinner("Running the local audit. This can take several minutes."):
            result: PipelineResult = run_pipeline(
                source,
                config,
                model=effective_model,
                focus_goals=goals,
                top_k=int(top_k),
                min_score=float(min_score),
                max_candidates=int(max_candidates),
                max_pages=int(max_pages),
                extract_tables_enabled=bool(extract_tables),
                progress_callback=on_progress,
            )
            audit_file = write_output(result, out_path)
        progress.empty()
        _set_audit(result.to_dict(), str(audit_file), audit_file)
        st.sidebar.success(f"Audit written to {audit_file}")
        st.rerun()
    except Exception as exc:  # noqa: BLE001 - the UI must not crash
        st.sidebar.error(f"Audit failed: {exc}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _sidebar_report_options() -> None:
    st.sidebar.header("Report options")
    st.sidebar.text_input(
        "Company name override",
        key="company_override",
        placeholder="Use the audit document title",
    )
    st.sidebar.text_input(
        "Report title override",
        key="title_override",
        placeholder="SDG Alignment Report",
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _tab_summary(data: ReportData) -> None:
    st.subheader("Summary")
    summary = summarise_reviews(data.scored, data.reviews)
    columns = st.columns(5)
    columns[0].metric("Scored findings", len(data.scored))
    columns[1].metric(
        "Targets with disclosure",
        f"{data.coverage.targets_with_findings} / "
        f"{data.coverage.targets_total}",
    )
    columns[2].metric("Judged candidates", len(data.findings))
    columns[3].metric("Needs human review", len(data.needs_review))
    columns[4].metric(
        "Review pending", summary.pending, help=STATUS_HELP["pending"]
    )

    st.caption(
        "Coverage counts only the candidates retrieved and judged in this run. "
        "A target without a scored finding is reported as not identified, not "
        "as evidence that the company does nothing."
    )

    st.subheader("Coverage by goal")
    if data.coverage.goals:
        rows = [
            {
                "Goal": f"Goal {goal.goal}",
                "Focus area": goal.short_title,
                "Targets with scored disclosure": (
                    f"{goal.covered_target_count} of {goal.targets_total}"
                ),
                "Scored findings": goal.finding_count,
                "Best level": _level_text(goal.best_level),
                "Status": goal.status,
            }
            for goal in data.coverage.goals
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No focus goals were available for coverage accounting.")

    with st.expander("Target-level coverage"):
        target_rows = [
            {
                "Goal": goal.goal,
                "Target": target.target_id,
                "Scored findings": target.finding_count,
                "Best level": _level_text(target.best_level),
                "Pages": ", ".join(str(page) for page in target.pages),
                "Status": target.status,
            }
            for goal in data.coverage.goals
            for target in goal.targets
        ]
        st.dataframe(pd.DataFrame(target_rows), hide_index=True, width="stretch")

    with st.expander("Evidence coverage matrix"):
        st.caption(
            "Absent means the judge did not identify that dimension in the "
            "verified quote. It is a disclosure gap, not proof of inaction. "
            "n/a means the claim could not be assessed and needs review."
        )
        if not data.coverage_matrix.claims:
            st.info("No reviewed claims are available for the matrix.")
        else:
            matrix_rows = [
                {
                    "Target": claim.target_id,
                    "Page": claim.page,
                    "Level": _level_text(claim.level),
                    "Specificity": claim.dimensions.get(
                        "specificity", "not_assessable"
                    ),
                    "Measurable indicator": claim.dimensions.get(
                        "measurable_indicator", "not_assessable"
                    ),
                    "Deadline": claim.dimensions.get("deadline", "not_assessable"),
                    "Baseline": claim.dimensions.get("baseline", "not_assessable"),
                    "Reported progress": claim.dimensions.get(
                        "reported_progress", "not_assessable"
                    ),
                    "Flags": ", ".join(claim.flags) or "none",
                }
                for claim in data.coverage_matrix.claims
            ]
            st.dataframe(
                pd.DataFrame(matrix_rows), hide_index=True, width="stretch"
            )


def _review_dataframe(
    data: ReportData,
    decisions: dict[str, ReviewDecision],
) -> pd.DataFrame:
    rows = export_rows(data.scored, decisions)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["review_level"] = pd.to_numeric(
        frame["review_level"], errors="coerce"
    )
    return frame


def _decisions_from_frame(
    frame: pd.DataFrame,
) -> tuple[dict[str, ReviewDecision], list[str]]:
    decisions: dict[str, ReviewDecision] = {}
    errors: list[str] = []
    for _, row in frame.iterrows():
        key = str(row.get("finding_key") or "").strip()
        if not key:
            continue
        status = str(row.get("review_status") or "pending")
        raw_level = row.get("review_level")
        level = (
            None
            if raw_level is None or pd.isna(raw_level)
            else int(raw_level)
        )
        note = "" if pd.isna(row.get("note")) else str(row.get("note"))
        reviewer = (
            "" if pd.isna(row.get("reviewer")) else str(row.get("reviewer"))
        )
        if status == "overridden" and level is None:
            errors.append(
                f"Target {row.get('target')} on page {row.get('page')}: "
                "choose a reviewer level for an override."
            )
            continue
        decisions[key] = make_decision(
            key,
            status=status,
            level=level,
            note=note,
            reviewer=reviewer,
        )
    return decisions, errors


def _tab_review(data: ReportData, config: dict[str, Any]) -> None:
    st.subheader("Findings review")
    st.caption(
        "The rubric level is never overwritten. An override records a "
        "reviewer level next to it so the original assessment stays auditable."
    )

    if not data.scored:
        st.info(
            "No scored findings. Nothing to review in this audit; check the "
            "Needs review tab for unverified or malformed results."
        )
        return

    review_path = _default_review_path(st.session_state.audit_path)
    frame = _review_dataframe(data, st.session_state.reviews)
    editor_key = (
        f"review_editor_{st.session_state.editor_version}_"
        f"{st.session_state.audit_label}"
    )
    edited = st.data_editor(
        frame,
        key=editor_key,
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        column_order=(
            "target",
            "page",
            "rubric_level",
            "quote",
            "review_status",
            "review_level",
            "reviewer",
            "note",
            "updated_at",
        ),
        column_config={
            "target": st.column_config.TextColumn(
                "Target", disabled=True, width="small"
            ),
            "page": st.column_config.NumberColumn(
                "Page", disabled=True, width="small"
            ),
            "rubric_level": st.column_config.NumberColumn(
                "Rubric level", disabled=True, width="small"
            ),
            "quote": st.column_config.TextColumn(
                "Verified quote", disabled=True, width="large"
            ),
            "review_status": st.column_config.SelectboxColumn(
                "Review status",
                options=list(REVIEW_STATUSES),
                required=True,
                width="small",
                help=" · ".join(
                    f"{status}: {STATUS_HELP[status]}"
                    for status in REVIEW_STATUSES
                ),
            ),
            "review_level": st.column_config.NumberColumn(
                "Reviewer level",
                min_value=1,
                max_value=4,
                step=1,
                width="small",
                help="Required when the status is overridden.",
            ),
            "reviewer": st.column_config.TextColumn(
                "Reviewer", width="small"
            ),
            "note": st.column_config.TextColumn("Note", width="medium"),
            "updated_at": st.column_config.TextColumn(
                "Updated", disabled=True, width="small"
            ),
        },
    )

    save_column, path_column = st.columns([1, 2])
    save_clicked = save_column.button("Save review decisions", type="primary")
    with path_column:
        review_path_value = st.text_input(
            "Review JSON",
            value=str(review_path) if review_path else "outputs/reviews.json",
            label_visibility="collapsed",
        )

    if save_clicked:
        decisions, errors = _decisions_from_frame(edited)
        if errors:
            for message in errors:
                st.error(message)
        else:
            saved = save_reviews(
                resolve_path(review_path_value, config), decisions
            )
            st.session_state.reviews = decisions
            st.session_state.editor_version += 1
            st.success(f"Saved {len(decisions)} review decision(s) to {saved}")
            st.rerun()

    summary = summarise_reviews(data.scored, st.session_state.reviews)
    st.caption(
        f"{summary.confirmed} confirmed · {summary.overridden} overridden · "
        f"{summary.flagged} flagged · {summary.pending} pending"
    )


def _tab_needs_review(data: ReportData) -> None:
    st.subheader("Needs human review")
    if not data.needs_review:
        st.success("No findings need human review in this audit.")
    else:
        st.caption(
            "These findings are not scored: the quote could not be verified "
            "or the judge output did not pass validation."
        )
        rows = [
            {
                "Page": finding.source_page,
                "Target": finding.judged.sdg_target,
                "Reason": finding.review_reason or "not scored",
                "Passage": _short(finding.judged.candidate.passage_text),
            }
            for finding in data.needs_review
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Candidates excluded as not relevant")
    st.write(
        f"{len(data.not_relevant)} candidate(s) were judged not relevant to "
        "their retrieved target and were not scored. This describes retrieval "
        "and judging output, not the company."
    )
    if data.not_relevant:
        with st.expander("Show excluded candidates"):
            rows = [
                {
                    "Page": finding.judged.candidate.page,
                    "Target": finding.judged.candidate.target_id,
                    "Retrieval score": round(
                        finding.judged.candidate.score, 3
                    ),
                    "Passage": _short(finding.judged.candidate.passage_text),
                }
                for finding in data.not_relevant
            ]
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, width="stretch"
            )


def _tab_report(data: ReportData, config: dict[str, Any]) -> None:
    st.subheader("Reviewed report")
    markdown = render_markdown(data)
    html_text = render_html(data)

    download_column, html_column = st.columns(2)
    download_column.download_button(
        "Download Markdown",
        data=markdown,
        file_name="sdg-alignment-report.md",
        mime="text/markdown",
        width="stretch",
    )
    html_column.download_button(
        "Download HTML",
        data=html_text,
        file_name="sdg-alignment-report.html",
        mime="text/html",
        width="stretch",
    )

    if st.button("Save reviewed report to outputs/"):
        base = (
            Path(st.session_state.audit_path).stem
            if st.session_state.audit_path
            else "ui"
        )
        markdown_path = write_markdown_report(
            data, f"outputs/{base}.reviewed.md", config
        )
        html_path = write_html_report(
            data, f"outputs/{base}.reviewed.html", config
        )
        st.success(f"Wrote {markdown_path} and {html_path}")

    with st.expander("Preview report", expanded=True):
        st.markdown(markdown)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Prism17 — SDG Disclosure Audit",
        page_icon="🔎",
        layout="wide",
    )
    _initialise_state()
    config = _sidebar_config()
    _sidebar_source(config)
    _sidebar_report_options()

    st.title("Prism17 — SDG Disclosure Audit")
    st.caption(
        "Assesses disclosure quality, not real-world performance. Every scored "
        "finding keeps a page number and a quote verified against the source "
        "page. A human reviewer should confirm or override findings before use."
    )

    payload = st.session_state.payload
    if payload is None:
        st.info(
            "Load an audit JSON or run a new audit from the sidebar to begin."
        )
        return

    st.write(
        f"**Audit:** `{st.session_state.audit_label}`  \n"
        f"**Review file:** "
        f"`{_default_review_path(st.session_state.audit_path) or 'not saved yet'}`"
    )

    try:
        data = _report_data(config)
    except (KeyError, ValueError) as exc:
        st.error(f"Could not build the report: {exc}")
        return

    summary_tab, review_tab, needs_tab, report_tab = st.tabs(
        ("Summary & coverage", "Findings review", "Needs review", "Report")
    )
    with summary_tab:
        _tab_summary(data)
    with review_tab:
        _tab_review(data, config)
    with needs_tab:
        _tab_needs_review(data)
    with report_tab:
        _tab_report(data, config)


if __name__ == "__main__":  # pragma: no cover - Streamlit entry point
    main()
