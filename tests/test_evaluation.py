import json

from medaibot.evaluation import latest_other_provider, write_report


def _result(run_id: str, provider: str, rate: float) -> dict:
    metric = {"n": 10, "rate": rate}
    summary = {
        "abstain_correct": metric,
        "keywords_ok": metric,
        "cites_gold": metric,
        "deterministic_checks_passed": metric,
        "judge_supported_sentences": None,
        "no_forbidden_text": None,
        "latency_p50_ms": 10.0,
        "latency_p95_ms": 20.0,
    }
    return {
        "run_id": run_id,
        "started_at": "2026-10-29T00:00:00+00:00",
        "n_questions": 10,
        "questions": [],
        "corpus": {"n_chunks": 5, "embedder": "hashing-256", "label_versions": {"tagrisso": "v36"}},
        "retrieval": {"k": 5, "rows": [], "summary": {}},
        "agent": {"provider": provider, "live_openfda": False, "rows": [], "summary": {"all": summary}},
    }


def test_baseline_is_latest_run_from_another_provider(tmp_path):
    for run_id, provider in (("20261029T100000Z", "offline-rules"), ("20261029T110000Z", "offline-rules")):
        (tmp_path / f"{run_id}.json").write_text(json.dumps(_result(run_id, provider, 0.5)))
    (tmp_path / "20261029T120000Z.json").write_text(json.dumps(_result("20261029T120000Z", "openai:m", 0.9)))

    current = _result("20261029T130000Z", "openai:m", 0.9)
    baseline = latest_other_provider(current, tmp_path)
    assert baseline is not None and baseline["run_id"] == "20261029T110000Z"

    report = tmp_path / "report.md"
    write_report(current, report, baseline=baseline)
    text = report.read_text()
    assert "Compared with `offline-rules` (run `20261029T110000Z`)" in text
    assert "| correct abstain / answer decision | 0.90 (n=10) | 0.50 (n=10) |" in text
