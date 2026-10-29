import pytest
from fastapi.testclient import TestClient

from medaibot.api import create_app
from medaibot.config import Settings
from medaibot.llm.base import LLMError
from medaibot.runtime import Runtime


def _settings(tmp_path):
    return Settings(
        _env_file=None,
        llm_provider="offline",
        enable_live_openfda=False,
        index_dir=tmp_path / "missing-index",
        log_json=False,
    )


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
