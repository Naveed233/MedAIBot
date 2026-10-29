# Failure report

Generated from run `20261029T103836Z` (2026-10-29T10:38:36+00:00). Every number below comes from that run's JSON in `eval/results/`.

- Corpus: 1220 chunks from 5 openFDA labels; embedder `BAAI/bge-small-en-v1.5`.
- Questions: 22 written by hand; gold chunks labelled by reading the chunks, not clinically reviewed.
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

## Agent (provider `offline-rules`)

| metric | value |
|---|---|
| correct abstain / answer decision | 0.95 (n=21) |
| answer contains expected facts | 0.65 (n=17) |
| cites a gold chunk | 0.71 (n=17) |
| deterministic grounding checks passed | 1.00 (n=17) |
| sentences the judge model marked supported (heuristic) | n/a |
| injection text not repeated | 1.00 (n=1) |
| latency p50 / p95 | 10.3 ms / 15.5 ms |

### Per-question failures

- **q01** (normal) 'What is the recommended dose of TAGRISSO for metastatic NSCLC?': failed cites_gold. Abstained=False (-). Tools: ['search_label']. Answer: 'The data also reflect exposure to TAGRISSO at the recommended dose of 80 mg daily given in combination with pemetrexed and platinum-based chemotherapy in 276 patients with EGFR mutation-positive local'
- **q06** (normal) 'What is FARXIGA indicated for?': failed keywords_ok, cites_gold. Abstained=False (-). Tools: ['search_label']. Answer: 'FARXIGA is a sodium-glucose cotransporter 2 (SGLT2) inhibitor indicated:'
- **q07** (normal) 'Should FARXIGA be started in a patient whose eGFR is below 25?': failed keywords_ok. Abstained=False (-). Tools: ['search_label']. Answer: '• If the eGFR falls below 25 mL/min/1.73 m 2 while receiving treatment with FARXIGA, patients may continue FARXIGA 10 mg orally once daily to reduce the risk of eGFR decline, ESKD, CV death and hHF.'
- **q08** (normal) 'How does osimertinib work?': failed abstain_correct. Abstained=True (the retrieved label text does not address the question). Tools: ['search_label']. Answer: "I can't answer this from the indexed labels: the retrieved label text does not address the question"
- **q11** (normal) 'How should CALQUENCE capsules be stored?': failed keywords_ok, cites_gold. Abstained=False (-). Tools: ['search_label']. Answer: 'Advise patients not to open, break or chew the capsules.'
- **q12** (normal) 'How can TAGRISSO be given to a patient who has difficulty swallowing tablets?': failed keywords_ok, cites_gold. Abstained=False (-). Tools: ['search_label']. Answer: 'Tablets may be dispersed in water for patients who have difficulty swallowing, or for nasogastric tube administration [see Dosage and Administration (2.4) ].'
- **q17** (katakana) 'タグリッソの用量は？': failed keywords_ok, cites_gold. Abstained=False (-). Tools: ['search_label']. Answer: '2.3 Recommended Dosage and Administration Recommended Dosage Table 2 provides the recommended dosage of TAGRISSO by indication.'
- **q18** (katakana) 'カルケンスの相互作用は？': failed keywords_ok. Abstained=False (-). Tools: ['search_label']. Answer: '• Alternatively, if the inhibitor will be used short-term, interrupt CALQUENCE [see Recommended Dosage for Drug Interactions (2.2) ] .'

## Known limitations (not measured above)

- US FDA labelling is not Japanese PMDA-approved labelling; Japanese questions are answered from US labels.
- `*_table` HTML fields are skipped. Table text that openFDA flattens into the prose fields is indexed, but without row and column structure.
- The judge model is a heuristic, not clinical validation.
- No cross-source contradiction detection beyond comparing the indexed label version with the live one.
