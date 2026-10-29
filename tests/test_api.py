import pytest
from fastapi.testclient import TestClient

from medaibot.api import UsageLimits, create_app
from medaibot.config import Settings
from medaibot.llm.base import LLMError
from medaibot.runtime import Runtime


def _settings(tmp_path, **overrides):
    return Settings(
        _env_file=None,
        llm_provider="offline",
        enable_live_openfda=False,
        index_dir=tmp_path / "missing-index",
        log_json=False,
        **overrides,
    )


def _client(tmp_path, catalog, index, **overrides):
    settings = _settings(tmp_path, **overrides)
    return TestClient(create_app(settings, Runtime(settings=settings, catalog=catalog, index=index)))


@pytest.fixture
def client(tmp_path, catalog, index):
    settings = _settings(tmp_path)
    runtime = Runtime(settings=settings, catalog=catalog, index=index)
    with TestClient(create_app(settings, runtime)) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready").json()
    assert ready["status"] == "ready" and ready["chunks"] > 0


def test_ask_returns_a_cited_answer_with_trace(client):
    r = client.post("/ask", json={"question": "What are the contraindications for TAGRISSO?"})
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is False
    assert body["citations"][0]["chunk_id"] == "tagrisso:contraindications:000"
    assert body["trace"][0]["name"] == "search_label"
    assert "x-request-id" in r.headers


def test_empty_question_is_rejected(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_missing_index_means_not_ready_but_still_alive(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as c:
        assert c.get("/health").status_code == 200
        r = c.get("/ready")
        assert r.status_code == 503 and "no index" in r.json()["detail"]
        assert c.post("/ask", json={"question": "dose?"}).status_code == 503


def test_model_outage_is_a_503_not_a_200_apology(tmp_path, catalog, index):
    class Down:
        name = "down"

        def complete(self, *args, **kwargs):
            raise LLMError("upstream 503")

    settings = _settings(tmp_path)
    runtime = Runtime(settings=settings, catalog=catalog, index=index)
    runtime.agent.provider = Down()
    with TestClient(create_app(settings, runtime)) as c:
        r = c.post("/ask", json={"question": "What is the dose of TAGRISSO?"})
        assert r.status_code == 503


def test_demo_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "MedAIBot" in r.text


QUESTION = {"question": "What are the contraindications for TAGRISSO?"}


def test_access_code_is_required_when_configured(tmp_path, catalog, index):
    with _client(tmp_path, catalog, index, access_code="blue heron") as c:
        assert c.get("/ready").json()["access_code_required"] is True
        assert c.post("/ask", json=QUESTION).status_code == 401
        assert c.post("/ask", json=QUESTION, headers={"x-access-code": "wrong"}).status_code == 401
        assert c.post("/ask", json=QUESTION, headers={"x-access-code": "blue heron"}).status_code == 200


def test_per_minute_limit_counts_wrong_codes_too(tmp_path, catalog, index):
    with _client(tmp_path, catalog, index, access_code="blue heron", rate_limit_per_minute=2) as c:
        assert c.post("/ask", json=QUESTION).status_code == 401
        assert c.post("/ask", json=QUESTION).status_code == 401
        r = c.post("/ask", json=QUESTION, headers={"x-access-code": "blue heron"})
        assert r.status_code == 429 and "wait a minute" in r.json()["detail"]


def test_daily_cap_counts_only_accepted_questions(tmp_path, catalog, index):
    with _client(tmp_path, catalog, index, access_code="blue heron", daily_question_cap=1) as c:
        ok = {"x-access-code": "blue heron"}
        assert c.post("/ask", json=QUESTION).status_code == 401
        assert c.post("/ask", json=QUESTION, headers=ok).status_code == 200
        r = c.post("/ask", json=QUESTION, headers=ok)
        assert r.status_code == 429 and "daily" in r.json()["detail"]


def test_limits_reset_after_a_minute_and_on_a_new_day():
    now = [0.0]
    limits = UsageLimits(per_minute=1, per_day=1, clock=lambda: now[0])
    assert limits.attempt("a") is None and limits.question() is None
    assert limits.attempt("a") is not None and limits.attempt("b") is None
    assert limits.question() is not None
    now[0] = 86_400.0
    assert limits.attempt("a") is None and limits.question() is None


def test_a_follow_up_is_answered_about_the_earlier_drug(client):
    history = [{"question": "What are the contraindications for TAGRISSO?", "answer": "None."}]
    question = "How can it be given to a patient who has difficulty swallowing tablets?"
    r = client.post("/ask", json={"question": question, "history": history})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == question
    assert body["asked_as"] == f"About TAGRISSO: {question}"
    assert all(c["drug"] == "TAGRISSO" for c in body["citations"])


def test_a_question_without_history_is_not_rewritten(client):
    body = client.post("/ask", json=QUESTION).json()
    assert body["asked_as"] is None and body["question"] == QUESTION["question"]


def test_history_is_capped_at_three_turns(client):
    turn = {"question": "What is the dose of TAGRISSO?", "answer": ""}
    assert client.post("/ask", json={"question": "And for children?", "history": [turn] * 4}).status_code == 422
