"""Evaluation: retrieval quality per search mode, then end-to-end agent behaviour.

Writes eval/results/<timestamp>.json and regenerates docs/failure-report.md from that run.
"""

import json
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medaibot.config import Settings
from medaibot.drugs import DrugCatalog
from medaibot.index import HybridIndex, Mode
from medaibot.models import AgentAnswer
from medaibot.query import normalise_query

QUESTIONS_PATH = Path("eval/questions.jsonl")
RESULTS_DIR = Path("eval/results")
REPORT_PATH = Path("docs/failure-report.md")
K = 5


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    language: str
    category: str
    drug: str | None
    question: str
    gold_chunks: list[str]
    expect_abstain: bool | None
    answer_keywords: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    labelled_by: str = ""


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    with path.open(encoding="utf-8") as fh:
        return [EvalQuestion(**json.loads(line)) for line in fh if line.strip()]


# ---------------------------------------------------------------- retrieval


def retrieval_rows(
    index: HybridIndex,
    catalog: DrugCatalog,
    questions: Iterable[EvalQuestion],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """One row per (question, mode, normalised?) for questions that have gold chunks."""
    rows: list[dict[str, Any]] = []
    modes: tuple[Mode, ...] = ("bm25", "dense", "hybrid")
    for q in questions:
        if not q.gold_chunks:
            continue
        gold = set(q.gold_chunks)
        nq = normalise_query(q.question, catalog)
        for normalised in (False, True):
            text = nq.text if normalised else q.question
            drug = nq.drug if normalised else None
            for mode in modes:
                started = time.perf_counter()
                hits = index.search(text, k=K, drug=drug, mode=mode, rrf_k=rrf_k)
                ms = (time.perf_counter() - started) * 1000
                ids = [h.chunk.chunk_id for h in hits]
                first = next((i for i, cid in enumerate(ids, start=1) if cid in gold), None)
                rows.append(
                    {
                        "id": q.id,
                        "language": q.language,
                        "category": q.category,
                        "config": f"{mode}{'+norm' if normalised else ''}",
                        "hit": first is not None,
                        "recall": len(gold & set(ids)) / len(gold),
                        "rr": 1.0 / first if first else 0.0,
                        "retrieved": ids,
                        "ms": round(ms, 1),
                    }
                )
    return rows


def summarise_retrieval(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """config -> group ("all", "lang:en", "cat:katakana") -> metric -> value."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for group in ("all", f"lang:{row['language']}", f"cat:{row['category']}"):
            grouped[row["config"]][group].append(row)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for config, groups in grouped.items():
        out[config] = {
            group: {
                "n": len(rs),
                f"hit@{K}": round(sum(r["hit"] for r in rs) / len(rs), 3),
                f"recall@{K}": round(sum(r["recall"] for r in rs) / len(rs), 3),
                "mrr": round(sum(r["rr"] for r in rs) / len(rs), 3),
            }
            for group, rs in groups.items()
        }
    return out


# ---------------------------------------------------------------- agent


def score_answer(q: EvalQuestion, answer: AgentAnswer) -> dict[str, Any]:
    text = answer.answer.lower()
    cited = [c.chunk_id for c in answer.citations]
    row: dict[str, Any] = {
        "id": q.id,
        "language": q.language,
        "category": q.category,
        "question": q.question,
        "abstained": answer.abstained,
        "abstain_reason": answer.abstain_reason,
        "answer": answer.answer,
        "cited": cited,
        "tool_calls": [s.name for s in answer.trace if s.kind == "tool_call"],
        "latency_ms": round(answer.latency_ms, 1),
    }
    if q.expect_abstain is not None:
        row["abstain_correct"] = answer.abstained == q.expect_abstain
    if q.answer_keywords and not answer.abstained:
        row["keywords_ok"] = all(k.lower() in text for k in q.answer_keywords)
    if q.must_not_contain:
        row["no_forbidden_text"] = not any(k.lower() in text for k in q.must_not_contain)
    if q.gold_chunks and not answer.abstained:
        row["cites_gold"] = bool(set(cited) & set(q.gold_chunks))
    if answer.grounding is not None and not answer.abstained:
        row["deterministic_checks_passed"] = answer.grounding.passed
        judged = [s.judge_label for s in answer.grounding.sentences if s.judge_label != "not_checked"]
        if judged:
            row["support_labels"] = judged
    return row


def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [r[key] for r in rows if key in r]
    if not values:
        return None
    return {"n": len(values), "rate": round(sum(bool(v) for v in values) / len(values), 3)}


def summarise_agent(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for group in ("all", f"lang:{row['language']}", f"cat:{row['category']}"):
            groups[group].append(row)
    out: dict[str, Any] = {}
    for group, rs in groups.items():
        latencies = sorted(r["latency_ms"] for r in rs)
        supported = [lab for r in rs for lab in r.get("support_labels", [])]
        out[group] = {
            "n": len(rs),
            "abstain_correct": _rate(rs, "abstain_correct"),
            "keywords_ok": _rate(rs, "keywords_ok"),
            "cites_gold": _rate(rs, "cites_gold"),
            "deterministic_checks_passed": _rate(rs, "deterministic_checks_passed"),
            "no_forbidden_text": _rate(rs, "no_forbidden_text"),
            "judge_supported_sentences": (
                {"n": len(supported), "rate": round(supported.count("supported") / len(supported), 3)}
                if supported
                else None
            ),
            "latency_p50_ms": round(statistics.median(latencies), 1),
            "latency_p95_ms": round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 1),
        }
    return out


# ---------------------------------------------------------------- report


def _fmt(metric: dict[str, float] | None) -> str:
    return "n/a" if metric is None else f"{metric['rate']:.2f} (n={metric['n']})"


def latest_other_provider(result: dict[str, Any], results_dir: Path = RESULTS_DIR) -> dict[str, Any] | None:
    """The most recent saved run whose agent used a different provider, for a side-by-side table."""
    provider = result.get("agent", {}).get("provider")
    if provider is None:
        return None
    for path in sorted(results_dir.glob("*.json"), reverse=True):
        if path.stem == result["run_id"]:
            continue
        other = json.loads(path.read_text(encoding="utf-8"))
        if other.get("agent") and other["agent"]["provider"] != provider:
            return other
    return None


def write_report(result: dict[str, Any], path: Path = REPORT_PATH, baseline: dict[str, Any] | None = None) -> None:
    r_sum = result["retrieval"]["summary"]
    lines = [
        "# Failure report",
        "",
        (
            f"Generated from run `{result['run_id']}` ({result['started_at']}). "
            "Every number below comes from that run's JSON in `eval/results/`."
        ),
        "",
        (
            f"- Corpus: {result['corpus']['n_chunks']} chunks from {len(result['corpus']['label_versions'])} openFDA labels; "
            f"embedder `{result['corpus']['embedder']}`."
        ),
        (
            f"- Questions: {result['n_questions']} written from reading the labels; gold chunks labelled by "
            "reading the chunks, not clinically reviewed."
        ),
        "- 22 questions are enough to spot failure modes, not to claim reliability.",
        "",
        f"## Retrieval (k={K})",
        "",
        "`+norm` = drug alias detection and Japanese-to-English term mapping before search, plus a drug filter.",
        "",
        f"| config | hit@{K} all | MRR all | hit@{K} en | hit@{K} ja |",
        "|---|---|---|---|---|",
    ]
    for config in ("bm25", "dense", "hybrid", "bm25+norm", "dense+norm", "hybrid+norm"):
        groups = r_sum.get(config, {})
        all_, en, ja = groups.get("all", {}), groups.get("lang:en", {}), groups.get("lang:ja", {})
        lines.append(
            f"| {config} | {all_.get(f'hit@{K}', 'n/a')} | {all_.get('mrr', 'n/a')} | "
            f"{en.get(f'hit@{K}', 'n/a')} | {ja.get(f'hit@{K}', 'n/a')} |"
        )

    misses = [r for r in result["retrieval"]["rows"] if r["config"] == "hybrid+norm" and not r["hit"]]
    lines += ["", "### Retrieval misses (hybrid+norm)", ""]
    if misses:
        q_by_id = {q["id"]: q for q in result["questions"]}
        for r in misses:
            q = q_by_id[r["id"]]
            lines.append(
                f"- **{r['id']}** ({r['category']}) {q['question']!r}: gold {q['gold_chunks']}, got {r['retrieved']}"
            )
    else:
        lines.append("- none in this run")

    agent = result.get("agent")
    if agent:
        s = agent["summary"]["all"]
        lines += [
            "",
            f"## Agent (provider `{agent['provider']}`)",
            "",
            "| metric | value |",
            "|---|---|",
            f"| correct abstain / answer decision | {_fmt(s['abstain_correct'])} |",
            f"| answer contains expected facts | {_fmt(s['keywords_ok'])} |",
            f"| cites a gold chunk | {_fmt(s['cites_gold'])} |",
            f"| deterministic grounding checks passed | {_fmt(s['deterministic_checks_passed'])} |",
            f"| sentences the judge model marked supported (heuristic) | {_fmt(s['judge_supported_sentences'])} |",
            f"| injection text not repeated | {_fmt(s['no_forbidden_text'])} |",
            f"| latency p50 / p95 | {s['latency_p50_ms']} ms / {s['latency_p95_ms']} ms |",
            "",
        ]
        if baseline:
            b = baseline["agent"]["summary"]["all"]
            lines += [
                f"### Compared with `{baseline['agent']['provider']}` (run `{baseline['run_id']}`)",
                "",
                f"| metric | `{agent['provider']}` | `{baseline['agent']['provider']}` |",
                "|---|---|---|",
            ]
            for key, label in (
                ("abstain_correct", "correct abstain / answer decision"),
                ("keywords_ok", "answer contains expected facts"),
                ("cites_gold", "cites a gold chunk"),
                ("deterministic_checks_passed", "deterministic grounding checks passed"),
            ):
                lines.append(f"| {label} | {_fmt(s[key])} | {_fmt(b[key])} |")
            lines += [
                f"| latency p50 | {s['latency_p50_ms']} ms | {b['latency_p50_ms']} ms |",
                "",
            ]
        lines += ["### Per-question failures", ""]
        failed = [
            r
            for r in agent["rows"]
            if r.get("abstain_correct") is False
            or r.get("keywords_ok") is False
            or r.get("cites_gold") is False
            or r.get("deterministic_checks_passed") is False
            or r.get("no_forbidden_text") is False
            or any(lab in ("partial", "unsupported") for lab in r.get("support_labels", []))
        ]
        if not failed:
            lines.append("- none in this run")
        for r in failed:
            checks = [
                k
                for k in (
                    "abstain_correct",
                    "keywords_ok",
                    "cites_gold",
                    "deterministic_checks_passed",
                    "no_forbidden_text",
                )
                if r.get(k) is False
            ]
            if any(lab in ("partial", "unsupported") for lab in r.get("support_labels", [])):
                checks.append(f"judge labels {r['support_labels']}")
            lines.append(
                f"- **{r['id']}** ({r['category']}) {r['question']!r}: failed {', '.join(checks)}. "
                f"Abstained={r['abstained']} ({r['abstain_reason'] or '-'}). Tools: {r['tool_calls']}. "
                f"Answer: {r['answer'][:200]!r}"
            )

    lines += [
        "",
        "## Known limitations (not measured above)",
        "",
        "- US FDA labelling is not Japanese PMDA-approved labelling; Japanese questions are answered from US labels.",
        (
            "- `*_table` HTML fields are skipped. Table text that openFDA flattens into the prose fields is indexed, "
            "but without row and column structure."
        ),
        "- The judge model is a heuristic, not clinical validation.",
        "- No cross-source contradiction detection beyond comparing the indexed label version with the live one.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(settings: Settings, retrieval_only: bool = False, limit: int | None = None) -> int:
    from medaibot.runtime import load_runtime

    runtime = load_runtime(settings)
    questions = load_questions()[:limit] if limit else load_questions()
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")

    rows = retrieval_rows(runtime.index, runtime.catalog, questions, rrf_k=settings.rrf_k)
    result: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "n_questions": len(questions),
        "questions": [asdict(q) for q in questions],
        "corpus": {
            "n_chunks": len(runtime.index.chunks),
            "embedder": runtime.index.meta["embedder"],
            "label_versions": runtime.index.meta.get("label_versions", {}),
        },
        "retrieval": {"k": K, "rows": rows, "summary": summarise_retrieval(rows)},
    }

    if not retrieval_only:
        agent = runtime.agent
        agent_rows = []
        for q in questions:
            answer = agent.run(q.question)
            agent_rows.append(score_answer(q, answer))
            print(f"{q.id} abstained={answer.abstained} {answer.latency_ms:.0f} ms")
        result["agent"] = {
            "provider": agent.provider.name,
            "live_openfda": settings.enable_live_openfda,
            "rows": agent_rows,
            "summary": summarise_agent(agent_rows),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run_id}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(result, baseline=latest_other_provider(result))
    print(f"wrote {out} and {REPORT_PATH}")
    for config, groups in result["retrieval"]["summary"].items():
        print(f"{config:12s} {groups['all']}")
    if "agent" in result:
        print(json.dumps(result["agent"]["summary"]["all"], indent=1))
    return 0
