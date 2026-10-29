# Failure report

Generated from run `20261029T120957Z` (2026-10-29T12:09:57+00:00). Every number below comes from that run's JSON in `eval/results/`.

- Corpus: 1220 chunks from 5 openFDA labels; embedder `BAAI/bge-small-en-v1.5`.
- Questions: 22 written from reading the labels; gold chunks labelled by reading the chunks, not clinically reviewed.
- 22 questions are enough to spot failure modes, not to claim reliability.

## Retrieval (k=5)

`+norm` = drug alias detection and Japanese-to-English term mapping before search, plus a drug filter.

| config | hit@5 all | MRR all | hit@5 en | hit@5 ja |
|---|---|---|---|---|
| bm25 | 0.778 | 0.604 | 0.933 | 0.0 |
| dense | 0.833 | 0.759 | 1.0 | 0.0 |
| hybrid | 0.833 | 0.741 | 1.0 | 0.0 |
| bm25+norm | 0.944 | 0.733 | 0.933 | 1.0 |
| dense+norm | 0.944 | 0.87 | 1.0 | 0.667 |
| hybrid+norm | 1.0 | 0.907 | 1.0 | 1.0 |

### Retrieval misses (hybrid+norm)

- none in this run

## Agent (provider `openai:gpt-5.4-mini`)

| metric | value |
|---|---|
| correct abstain / answer decision | 1.00 (n=21) |
| answer contains expected facts | 0.94 (n=18) |
| cites a gold chunk | 1.00 (n=18) |
| deterministic grounding checks passed | 1.00 (n=19) |
| sentences the judge model marked supported (heuristic) | 1.00 (n=37) |
| injection text not repeated | 1.00 (n=1) |
| latency p50 / p95 | 3524.9 ms / 4629.1 ms |

### Compared with `offline-rules` (run `20261029T103836Z`)

| metric | `openai:gpt-5.4-mini` | `offline-rules` |
|---|---|---|
| correct abstain / answer decision | 1.00 (n=21) | 0.95 (n=21) |
| answer contains expected facts | 0.94 (n=18) | 0.65 (n=17) |
| cites a gold chunk | 1.00 (n=18) | 0.71 (n=17) |
| deterministic grounding checks passed | 1.00 (n=19) | 1.00 (n=17) |
| latency p50 | 3524.9 ms | 10.3 ms |

### Per-question failures

- **q19** (katakana) 'フォシーガは妊娠中に使えますか？': failed keywords_ok. Abstained=False (-). Tools: ['search_label']. Answer: 'フォシーガは、妊娠中には推奨されません。 ラベルでは、動物データに基づき胎児への潜在的リスクがあるため、女性に注意喚起するよう記載されています。 妊婦での限られたデータでは、先天異常や流産の薬剤関連リスクは判断できないとされています。'

## Known limitations (not measured above)

- US FDA labelling is not Japanese PMDA-approved labelling; Japanese questions are answered from US labels.
- `*_table` HTML fields are skipped. Table text that openFDA flattens into the prose fields is indexed, but without row and column structure.
- The judge model is a heuristic, not clinical validation.
- No cross-source contradiction detection beyond comparing the indexed label version with the live one.
