# Design decisions

Short records of what was chosen, what was rejected, and what would change the decision.

## 1. Vector store: FAISS in-process plus own BM25, not a vector database (yet)

**Chosen.** FAISS `IndexFlatIP` for dense search, a small BM25 implementation for keywords, Reciprocal Rank Fusion to merge them, all behind `HybridIndex.search()`.

**Why.** The corpus is about 1,200 chunks. Exact search over all of them takes milliseconds, needs no server, and runs in CI without Docker. Writing BM25 and RRF by hand also means every ranking decision can be explained and tested.

**Rejected for now: Weaviate, Qdrant, OpenSearch, pgvector.** All of them accept externally computed vectors, so "supports custom embeddings" does not separate them. Weaviate has native hybrid search and is named in the job description, but its default keyword tokenizer does not segment Japanese (it needs `kagome_ja`). That matters if the corpus becomes PMDA 添付文書.

**What would change it.** More than roughly 100k chunks, concurrent writers, metadata filters that must run inside the ANN search, or multi-tenant access control. Then: Amazon OpenSearch Serverless (AWS-native, BM25 and k-NN in one place) or Weaviate with `kagome_ja`. The `search()` interface stays the same.

## 2. Hybrid retrieval and query normalisation

**Measured, not assumed.** On the evaluation set, dense alone and hybrid had the same hit@5 (0.833). The large gain came from normalisation: mapping katakana brand names and Japanese section terms to English before searching took Japanese questions from 0.0 to 1.0 hit@5, and hybrid with normalisation had the best MRR (0.907). Numbers: `docs/failure-report.md`.

**Why both signals.** BM25 matches rare exact tokens (drug names, "CYP3A", "80 mg") that embeddings blur. Embeddings match paraphrases ("how does it work" vs "mechanism of action").

## 3. A bounded loop, not a free-running agent

**Chosen.** At most 6 model calls. A tool call is required on every step. The run ends only through `final_answer` (structured, with citations per sentence) or by hitting the limit, which produces an abstention.

**Why.** For label questions the useful plan is short: search, inspect, maybe read the whole section or check the live label, answer. A cap makes cost and latency predictable and makes failures visible in the trace. The `reason` field on every tool call makes the plan auditable.

**Prompt injection.** Retrieved text is passed only as tool results, never in the system prompt. The system prompt is constant (tested). Instruction-like text in a document is flagged in the trace and annotated in the tool result.

## 4. Checks instead of a confidence number

| tier | what it checks | what it proves |
|---|---|---|
| deterministic | each sentence cites ids retrieved in this run; every number in the sentence appears in the cited text; English token overlap | integrity: the answer points at real evidence and does not invent figures |
| judge model | supported / partial / unsupported per sentence, including population and conditions | a heuristic second opinion, not proof |
| live label | indexed label version vs live openFDA record | whether the evidence may be out of date |

A failure gets one retry with the problems listed; a second failure is an abstention. A judge verdict of `partial` counts as a failure: it usually means a dropped population or condition, and in run `20261029T120704Z` a `partial` answer about grapefruit gave inducer advice for an inhibitor. A model outage is an abstention in the CLI and HTTP 503 in the API, never a normal-looking answer.

**Known gap.** The deterministic tier cannot tell that "80 mg" belongs to a different population. The offline baseline shows exactly this on q01. That is what the judge tier is for, and it is still only a heuristic: in run `20261029T120957Z` it marked "not recommended during pregnancy" as supported when the label limits that to the second and third trimesters. The judge is currently the same model as the answerer; a stronger, separate judge model (`MEDAIBOT_JUDGE_MODEL`) is the next thing to try.

## 5. Fine-tuning: not now

**Plan, in order, each step gated on evaluation evidence:**

1. Keep measuring retrieval misses by category. Fix with chunking, normalisation and hybrid weights first; they are cheap and reversible.
2. If misses remain on rare terminology (katakana drug names, abbreviations), fine-tune the embedding model on query/passage pairs generated from the labels, with a held-out set, and compare hit@5 and MRR against the current baseline.
3. LLM fine-tuning only for behaviour (format, refusal style), never to store dosing facts. Structured output via tool calling already covers the format need.

**On RAFT.** RAFT (Zhang et al., 2024) fine-tunes the generator on questions paired with the relevant document plus distractor documents, so it learns to ignore irrelevant context. It is not "fine-tuned embeddings plus a fine-tuned LLM".

**Closest real need in the role.** For the VIVA voice interface, the comparable step is evaluating Whisper on a list of katakana medical terms (character error rate per term), and then adding prompting or post-correction with a term dictionary, like `query.py` does for text.

## 6. Live label data

openFDA can return several records for one product. FARXIGA has a repackager label (A-S Medication Solutions) with a newer effective date than AstraZeneca's own. So labels are pinned by `set_id`, and when the live record differs the agent reports it and does not switch automatically.

TAGRISSO's label has no `openfda` brand metadata, so a brand-name search does not find it. Pinning by `set_id` avoids that too.

## AWS path

Design only. Not deployed.

| concern | choice |
|---|---|
| compute | the existing container on ECS Fargate (or App Runner for a single service), behind an ALB using `/ready` for target health |
| corpus and index | raw labels and built index in versioned S3 objects; the index manifest records embedder and label versions, and a mismatch fails at startup (`IndexMismatch`) |
| secrets | Secrets Manager injected as environment variables; nothing in the image |
| model | Amazon Bedrock behind the same provider interface, in the VPC, so prompts stay inside the account |
| observability | JSON logs to CloudWatch (request id, tool, latency, abstained, token usage); metrics on abstain rate, judge partial/unsupported rate, p95 latency |
| delivery | GitHub Actions: tests, image build, push to ECR, deploy; infrastructure as code with CDK or Terraform |
| evaluation in CI | the offline suite on every push; the model-backed eval nightly, failing the build if hit@5 or abstain accuracy drops below the last release |
