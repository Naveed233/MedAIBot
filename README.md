# MedAIBot

Cited question answering over US FDA drug labels for five AstraZeneca medicines (TAGRISSO, LYNPARZA, FARXIGA, IMFINZI, CALQUENCE). It combines hybrid retrieval, a bounded tool-calling agent, checks on every answer, and an evaluation harness that reports where it fails.

> Portfolio project. Not medical advice. Not affiliated with or endorsed by AstraZeneca. The corpus is US FDA labelling, which is not the same as Japanese PMDA-approved labelling.

## What it does

```
question ──> query normalisation ──> agent loop (max 6 steps) ─────────────────────> answer + citations
             (drug aliases incl.      │  tools:                                        + grounding report
              katakana, JA→EN terms)  │   search_label     BM25 + FAISS, fused by RRF  + tool trace
                                      │   get_section      read a whole label section
                                      │   check_live_label live openFDA API call
                                      ▼
                               final_answer ──> checks ──> pass: answer
                                                  │        fail: one retry, then abstain
                                                  ├─ every sentence cites a chunk retrieved in this run
                                                  ├─ every number in the answer appears in the cited text
                                                  └─ judge model: supported / partial / unsupported (heuristic)
```

- **Ingest.** Labels are fetched from openFDA by SPL `set_id` and saved with provenance (request, fetch time, label version, effective date). Chunking follows the label structure: one section, split at numbered subsections ("2.3 Recommended Dosage"), packed by whole sentences to about 900 characters. Subsection fields that openFDA duplicates inside their parent are dropped.
- **Retrieval.** Own BM25 implementation for exact terms (drug names, doses), FAISS inner-product search over `BAAI/bge-small-en-v1.5` embeddings for meaning, and Reciprocal Rank Fusion so the two score scales never need to be compared. Filters by drug and section.
- **Agent.** The model must call a tool on every step and must finish with a structured `final_answer`. Each tool call carries a `reason`, so the trace shows why the agent did what it did. Retrieved text only ever enters the conversation as tool results; instruction-like text in documents is flagged.
- **Checks instead of a confidence score.** An earlier project of mine returned `random.uniform(0.75, 0.98)` as "confidence". This one reports what was actually checked, and abstains when the checks fail twice.

## Results

From `eval/results/` run `20261029T103836Z`, 18 questions with gold chunks (full report: [docs/failure-report.md](docs/failure-report.md)).

| retrieval config | hit@5 | MRR | hit@5 English | hit@5 Japanese |
|---|---|---|---|---|
| BM25 | 0.778 | 0.604 | 0.933 | 0.0 |
| dense | 0.833 | 0.759 | 1.0 | 0.0 |
| hybrid | 0.833 | 0.741 | 1.0 | 0.0 |
| hybrid + normalisation | 1.0 | 0.907 | 1.0 | 1.0 |

The Japanese questions (タグリッソの用量は？) find nothing until the katakana brand name is mapped to TAGRISSO and 用量 to "recommended dosage". The question set is small and was written by reading the labels, so these numbers show failure modes, not reliability.

## Run it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra openai
uv run medaibot fetch      # labels listed in data/sources.json (already committed)
uv run medaibot ingest     # builds data/index, downloads the embedding model once (~64 MB)
uv run medaibot ask "What are the contraindications for TAGRISSO?"
uv run medaibot serve      # http://127.0.0.1:8000, with /health, /ready, /ask and a demo page
uv run medaibot eval       # writes eval/results/*.json and docs/failure-report.md
uv run pytest              # offline: no API keys, no downloads
```

Configuration is read from environment variables or `.env` (see `.env.example`). Without `OPENAI_API_KEY` set and `MEDAIBOT_LLM_PROVIDER=openai`, the agent runs on `offline-rules`, a deterministic stand-in that extracts sentences instead of writing them. The same adapter works with any OpenAI-compatible server (for example a local open-source model) via `MEDAIBOT_OPENAI_BASE_URL`.

## Layout

| path | purpose |
|---|---|
| `src/medaibot/corpus.py` | section-aware chunking with provenance |
| `src/medaibot/index.py` | BM25, FAISS, RRF, filters, versioned save/load |
| `src/medaibot/query.py` | drug alias detection and Japanese term mapping |
| `src/medaibot/tools.py` | typed tools with timeouts and readable errors |
| `src/medaibot/agent.py` | bounded loop, retry, abstention |
| `src/medaibot/grounding.py` | deterministic checks and the judge |
| `src/medaibot/evaluation.py` | metrics by mode, category and language; failure report |
| `src/medaibot/api.py` | FastAPI app; failures return real status codes |
| `tests/` | 60+ offline tests, including injection, outages, timeouts, stale labels |

## Deployment

The Dockerfile builds a slim, non-root image with the index baked in and a health check. CI runs lint and tests, builds the image and smoke-tests `/health`, `/ready` and `/ask`. It has not been deployed. The AWS design I would use is in [docs/decisions.md](docs/decisions.md#aws-path).

## Limitations

- Five labels, one version each. No cross-source contradiction detection beyond comparing the indexed label with the live openFDA record.
- HTML table fields are skipped; tables that openFDA flattens into prose are indexed without their row and column structure.
- The judge is a model and can be wrong. Nothing here is clinical validation.
- Japanese support is text-level query mapping only. There is no speech recognition in this project.
