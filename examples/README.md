# Sample outputs

These files were generated from the bundled S.J.S. Enterprises BRSR
(`data/reports/sjs_india_brsr_fy2023_24.pdf`) with the local pipeline and
`qwen2.5:7b-instruct`:

- `sjs.parse.json` - parsed pages and passages from `python -m prism17.parse`.
- `sjs.audit.json` - Goal 12 audit from `python -m prism17.pipeline`.
- `sjs.report.md` / `sjs.report.html` - rendered SDG Alignment Report.

They are illustrative development artifacts, not a gold standard or a
benchmark result. Paths in the JSON are repository-relative. The report
describes disclosure quality only; it does not verify real-world performance.
