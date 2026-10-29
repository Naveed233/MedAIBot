import json
import time

import httpx
import pytest

from medaibot.openfda import LabelNotFound, OpenFDAClient, OpenFDAError
from medaibot.tools import Toolbox

TAGRISSO_SET_ID = "5e81b4a7-b971-45e1-9c31-29cea8c87ce7"


def _openfda(handler):
    return OpenFDAClient("https://api.fda.gov", transport=httpx.MockTransport(handler), timeout_s=1)


def _json(status, body):
    return lambda request: httpx.Response(status, json=body)


def test_unknown_tool_and_invalid_arguments_are_errors_not_crashes(index, catalog):
    tb = Toolbox(index, catalog)
    assert not tb.run("delete_everything", {}).ok
    bad = tb.run("search_label", {"query": "dose"})  # missing reason
    assert not bad.ok and "reason" in bad.content["error"]


def test_unknown_drug_and_section_return_helpful_errors(index, catalog):
    tb = Toolbox(index, catalog)
    out = tb.run("search_label", {"reason": "r", "query": "dose", "drug": "Keytruda"})
    assert not out.ok and "not in the index" in out.content["error"]
    out = tb.run("search_label", {"reason": "r", "query": "dose", "drug": "tagrisso", "section": "nope"})
    assert not out.ok and "Valid sections" in out.content["error"]


def test_search_returns_citable_evidence(index, catalog):
    out = Toolbox(index, catalog).run(
        "search_label", {"reason": "r", "query": "タグリッソの用量は？", "drug": "タグリッソ"}
    )
    assert out.ok and out.evidence
    assert "TAGRISSO" in out.content["query_used"]
    assert {r["chunk_id"] for r in out.content["results"]} == {c.chunk_id for c in out.evidence}


def test_get_section_reports_a_missing_section(index, catalog):
    out = Toolbox(index, catalog).run("get_section", {"reason": "r", "drug": "tagrisso", "section": "boxed_warning"})
    assert out.ok and out.content["results"] == [] and "has no Boxed Warning" in out.content["note"]


def test_slow_tool_times_out(index, catalog, monkeypatch):
    tb = Toolbox(index, catalog, timeout_s=0.2)

    def slow(args):
        time.sleep(1)

    monkeypatch.setitem(tb._tools, "search_label", (tb._tools["search_label"][0], slow, "slow"))
    out = tb.run("search_label", {"reason": "r", "query": "dose"})
    assert not out.ok and "timed out" in out.content["error"]


def test_live_label_same_version(index, catalog):
    client = _openfda(
        _json(200, {"results": [{"set_id": TAGRISSO_SET_ID, "version": "36", "effective_time": "20240925"}]})
    )
    out = Toolbox(index, catalog, client).run("check_live_label", {"reason": "r", "drug": "TAGRISSO"})
    assert out.ok and out.content["status"] == "same"


def test_live_label_newer_is_reported_but_not_trusted_blindly(index, catalog):
    client = _openfda(
        _json(200, {"results": [{"set_id": TAGRISSO_SET_ID, "version": "40", "effective_time": "20261001"}]})
    )
    out = Toolbox(index, catalog, client).run("check_live_label", {"reason": "r", "drug": "osimertinib"})
    assert out.content["status"] == "live_differs"
    assert "Do not assume the live record is the right one" in out.content["note"]


def test_live_label_for_drug_outside_index_becomes_citable_evidence(index, catalog):
    body = {
        "results": [
            {
                "set_id": "abc",
                "version": "5",
                "effective_time": "20260101",
                "openfda": {"brand_name": ["KEYTRUDA"], "generic_name": ["PEMBROLIZUMAB"]},
                "dosage_and_administration": ["200 mg every 3 weeks."],
            }
        ]
    }
    out = Toolbox(index, catalog, _openfda(_json(200, body))).run(
        "check_live_label", {"reason": "r", "drug": "Keytruda"}
    )
    assert out.content["status"] == "live_only"
    assert [c.chunk_id for c in out.evidence] == ["live:abc:dosage_and_administration"]


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (_json(404, {"error": {"code": "NOT_FOUND"}}), "not_found"),
        (_json(500, {"error": "boom"}), "error"),
    ],
)
def test_openfda_failures_are_reported(index, catalog, handler, expected):
    out = Toolbox(index, catalog, _openfda(handler)).run("check_live_label", {"reason": "r", "drug": "Keytruda"})
    assert out.content["status"] == expected


def test_openfda_timeout_raises_typed_error():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(OpenFDAError):
        _openfda(timeout).label_by_set_id(TAGRISSO_SET_ID)


def test_openfda_empty_results_is_not_found():
    with pytest.raises(LabelNotFound):
        _openfda(_json(200, {"results": []})).find_label("x")


def test_tool_output_is_json_serialisable(index, catalog):
    out = Toolbox(index, catalog).run("get_section", {"reason": "r", "drug": "farxiga", "section": "contraindications"})
    json.dumps(out.content)
