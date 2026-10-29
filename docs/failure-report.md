# Failure report

Generated from run `20261029T103509Z` (2026-10-29T10:35:09+00:00). Every number below comes from that run's JSON in `eval/results/`.

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
| sentences the judge model marked supported (heuristic) | 0.00 (n=17) |
| injection text not repeated | 1.00 (n=1) |
| latency p50 / p95 | 11.1 ms / 16.9 ms |

### Per-question failures

- **q01** (normal) 'What is the recommended dose of TAGRISSO for metastatic NSCLC?': failed cites_gold, judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: 'The data also reflect exposure to TAGRISSO at the recommended dose of 80 mg daily given in combination with pemetrexed and platinum-based chemotherapy in 276 patients with EGFR mutation-positive local'
- **q02** (explicit_none) 'What are the contraindications for TAGRISSO?': failed judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: '4 CONTRAINDICATIONS None.'
- **q03** (normal) 'What is the recommended dosage of LYNPARZA?': failed judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: '• Recommended dosage is 300 mg taken orally twice daily with or without food.'
- **q04** (normal) 'What is the recommended dose of CALQUENCE?': failed judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: '2 DOSAGE AND ADMINISTRATION Recommended dose is 100 mg orally approximately every 12 hours; swallow whole with water and with or without food.'
- **q05** (normal) 'Can CALQUENCE be taken with a strong CYP3A inhibitor?': failed judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: 'Table 1: Recommended Dosage Modifications for Use with CYP3A Inhibitors or Inducers CYP3A Co-administered Drug Recommended CALQUENCE use Inhibition Strong CYP3A inhibitor Avoid co-administration.'
- **q06** (normal) 'What is FARXIGA indicated for?': failed keywords_ok, cites_gold, judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: 'FARXIGA is a sodium-glucose cotransporter 2 (SGLT2) inhibitor indicated:'
- **q07** (normal) 'Should FARXIGA be started in a patient whose eGFR is below 25?': failed keywords_ok, judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: '• If the eGFR falls below 25 mL/min/1.73 m 2 while receiving treatment with FARXIGA, patients may continue FARXIGA 10 mg orally once daily to reduce the risk of eGFR decline, ESKD, CV death and hHF.'
- **q08** (normal) 'How does osimertinib work?': failed abstain_correct. Abstained=True (the retrieved label text does not address the question). Tools: ['search_label']. Answer: "I can't answer this from the indexed labels: the retrieved label text does not address the question"
- **q09** (normal) 'What is the IMFINZI dose for metastatic NSCLC in patients weighing 30 kg or more?': failed judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: '• Metastatic NSCLC: ∘ Weight ≥ 30 kg: IMFINZI 1,500 mg every 3 weeks in combination with tremelimumab-actl 75 mg and platinum-based chemotherapy for 4 cycles, and then administer IMFINZI 1,500 mg ever'
- **q10** (normal) 'What immune-mediated adverse reactions does the IMFINZI label warn about?': failed judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: '5.1 Immune-Mediated Adverse Reactions IMFINZI is a monoclonal antibody that belongs to a class of drugs that bind to either the programmed death-receptor 1 (PD-1) or the PD-ligand 1 (PD-L1), blocking '
- **q11** (normal) 'How should CALQUENCE capsules be stored?': failed keywords_ok, cites_gold, judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: 'Advise patients not to open, break or chew the capsules.'
- **q12** (normal) 'How can TAGRISSO be given to a patient who has difficulty swallowing tablets?': failed keywords_ok, cites_gold, judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: 'Tablets may be dispersed in water for patients who have difficulty swallowing, or for nasogastric tube administration [see Dosage and Administration (2.4) ].'
- **q13** (normal) 'Should patients avoid grapefruit while taking LYNPARZA?': failed judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: 'Inform patients to avoid grapefruit, grapefruit juice, Seville oranges, and Seville orange juice while taking Lynparza [see Drug Interactions (7.2) ] .'
- **q17** (katakana) 'タグリッソの用量は？': failed keywords_ok, cites_gold, judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: '2.3 Recommended Dosage and Administration Recommended Dosage Table 2 provides the recommended dosage of TAGRISSO by indication.'
- **q18** (katakana) 'カルケンスの相互作用は？': failed keywords_ok, judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: '• Alternatively, if the inhibitor will be used short-term, interrupt CALQUENCE [see Recommended Dosage for Drug Interactions (2.2) ] .'
- **q19** (katakana) 'フォシーガは妊娠中に使えますか？': failed judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: '8.1 Pregnancy Risk Summary Based on animal data showing adverse renal effects, FARXIGA is not recommended during the second and third trimesters of pregnancy.'
- **q20** (normal) 'Is FARXIGA safe to use during pregnancy?': failed judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: '8.1 Pregnancy Risk Summary Based on animal data showing adverse renal effects, FARXIGA is not recommended during the second and third trimesters of pregnancy.'
- **q21** (normal) 'What are the most common adverse reactions with TAGRISSO monotherapy?': failed judge labels ['not_checked']. Abstained=False (-). Tools: ['search_label']. Answer: 'In this pooled safety population, the most common adverse reactions in ≥20% of 1813 patients who received TAGRISSO monotherapy were diarrhea (47%), rash (46%), musculoskeletal pain (38%), nail toxicit'

## Known limitations (not measured above)

- US FDA labelling is not Japanese PMDA-approved labelling; Japanese questions are answered from US labels.
- `*_table` HTML fields are skipped. Table text that openFDA flattens into the prose fields is indexed, but without row and column structure.
- The judge model is a heuristic, not clinical validation.
- No cross-source contradiction detection beyond comparing the indexed label version with the live one.
