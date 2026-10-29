"""Multi-turn evaluation: does each follow-up get the right kind, drug and standalone question?

It scores the follow-up step only (no search, no answer), so it is cheap and shows whether a
miss comes from reading the conversation or from retrieval. 16 cases, several taken from
failures seen in live testing: enough to catch regressions, not to claim reliability.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medaibot.config import Settings
from medaibot.drugs import DrugCatalog
from medaibot.followup import Resolved, Turn, resolve_follow_up
from medaibot.query import normalise_query
from medaibot.text import is_cjk

CASES_PATH = Path("eval/followups.jsonl")
RESULTS_DIR = Path("eval/results")


@dataclass
class FollowUpCase:
    id: str
    turns: list[dict[str, str]]
    message: str
    kinds: list[str]  # acceptable kinds; "new" also covers a standalone message the model never saw
    drug: str | None = None  # the drug the standalone question must name (None: no drug, or unclear)
    any_terms: list[str] = field(default_factory=list)  # it must contain one of these
    no_terms: list[str] = field(default_factory=list)  # and none of these
    japanese: bool = False  # it must be written in Japanese


def load_cases(path: Path = CASES_PATH) -> list[FollowUpCase]:
    with path.open(encoding="utf-8") as fh:
        return [FollowUpCase(**json.loads(line)) for line in fh if line.strip()]


def score_case(case: FollowUpCase, resolved: Resolved, catalog: DrugCatalog) -> dict[str, Any]:
    kind = resolved.kind or ("new" if not resolved.rewritten else None)
    row: dict[str, Any] = {
        "id": case.id,
        "message": case.message,
        "kind": kind,
        "question": resolved.question,
        "clarify": resolved.clarify,
        "kind_ok": kind in case.kinds,
    }
    if kind == "unclear":
        row["drug_ok"] = None
        row["question_ok"] = bool(resolved.clarify)
    else:
        text = resolved.question.lower()
        row["drug_ok"] = normalise_query(resolved.question, catalog).drug == case.drug
        row["question_ok"] = (
            (not case.any_terms or any(t.lower() in text for t in case.any_terms))
            and not any(t.lower() in text for t in case.no_terms)
            and (not case.japanese or is_cjk(resolved.question))
        )
    row["all_ok"] = row["kind_ok"] and row["drug_ok"] is not False and row["question_ok"]
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> dict[str, float] | None:
        values = [r[key] for r in rows if r[key] is not None]
        return {"n": len(values), "rate": round(sum(values) / len(values), 3)} if values else None

    return {key: rate(key) for key in ("kind_ok", "drug_ok", "question_ok", "all_ok")}


def run_follow_up_evaluation(settings: Settings) -> int:
    from medaibot.runtime import load_runtime

    runtime = load_runtime(settings)
    rewriter = runtime.agent.provider if settings.llm_provider == "openai" else None
    started_at = datetime.now(UTC)
    rows = []
    for case in load_cases():
        started = time.perf_counter()
        resolved = resolve_follow_up(case.message, [Turn(**t) for t in case.turns], runtime.catalog, rewriter)
        row = score_case(case, resolved, runtime.catalog)
        row["ms"] = round((time.perf_counter() - started) * 1000, 1)
        rows.append(row)
        mark = "ok  " if row["all_ok"] else "MISS"
        print(f"{mark} {case.id} {case.message!r} -> {row['kind']}: {row['clarify'] or row['question']!r}")

    result = {
        "run_id": "followups-" + started_at.strftime("%Y%m%dT%H%M%SZ"),
        "started_at": started_at.isoformat(timespec="seconds"),
        "provider": rewriter.name if rewriter else "none (deterministic carry-over only)",
        "cases": [asdict(c) for c in load_cases()],
        "rows": rows,
        "summary": summarise(rows),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{result['run_id']}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(result["summary"], indent=1))
    return 0
