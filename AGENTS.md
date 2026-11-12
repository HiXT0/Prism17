# AGENTS.md - Prism17 working agreement

Prism17 audits **sustainability disclosure**, not real-world performance. It is
an evidence-grounded prototype for the FIRE 2026 SustAI competition.

## Non-negotiable constraints

- Use only open-source, locally run AI components.
- Never call OpenAI, Anthropic, Google, Azure, Cohere, or any other paid or
  proprietary AI API. Ollama is the only LLM runtime.
- Do not add network calls that send report text to a remote AI service.
- Default LLM: `qwen2.5:7b-instruct`; supported alternatives:
  `llama3.1:8b`, `gemma2:9b`.
- Default embeddings: `BAAI/bge-small-en-v1.5`; vector search: FAISS CPU.
- The project must run on a laptop with no GPU. Slow is acceptable; crashing is
  not.
- Every scored finding must have a page number and a quote verified against the
  source page. A finding without both is never displayed as scored.
- Use neutral language in all user-facing output. Avoid accusatory labels.
  Use "disclosure gap" and "low specificity" for observations.
- The tool assesses disclosure quality. It does not verify real-world impact,
  performance, or intent.
- Handle unreadable pages and malformed LLM output gracefully. Never let either
  crash the pipeline.

## Engineering style

- Python 3.11+, type hints, small functions, comments where the reasoning is
  not obvious.
- Log with the standard `logging` module.
- Prefer a simple working implementation over an ambitious broken one.
- Keep deterministic rubric logic in `prism17/rubric.py`; no LLM calls there.
- Cache expensive work (embeddings, LLM calls) on disk.
- Add tests for every rubric branch and for failure paths.

## Build order

Phase 1: scaffold, config, SDG targets, parser.
Phase 2: retrieval, judging, quote verification stage 1, rubric + tests.
Phase 3: report, coverage, guidance.
Phase 4: Streamlit UI with human review.
Phase 5: semantic verification, evaluation tooling.
Phase 6: comparison and narrow multi-year tracking.
Phase 7: README, rubric documentation, limitations.

Stop after each phase and verify before proceeding.
