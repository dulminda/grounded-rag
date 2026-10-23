import pytest
from fastapi.testclient import TestClient

from grounded.api.app import create_app
from grounded.pipeline import RAGPipeline
from tests.conftest import FakeLLM


@pytest.fixture
def client(pipeline):
    return TestClient(create_app(pipeline))


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["generation"] is True and body["chunks"] > 0


def test_search(client):
    r = client.post("/v1/search", json={"query": "Which enzyme do statins inhibit?", "k": 2})
    assert r.status_code == 200
    assert r.json()["results"][0]["doc_id"] == "statins"
    assert "x-request-id" in r.headers


def test_answer_and_verify(client):
    r = client.post("/v1/answer", json={"question": "How does aspirin work?"})
    assert r.status_code == 200 and r.json()["answer"]["citations"]
    r = client.post("/v1/verify", json={"claim": "Aspirin inhibits COX-1."})
    assert r.json()["answer"]["verdict"] == "SUPPORT"


def test_validation(client):
    assert client.post("/v1/search", json={"query": ""}).status_code == 422
    assert client.post("/v1/search", json={"query": "x", "k": 1000}).status_code == 422


def test_generation_disabled_without_llm(docs, bm25_config):
    c = TestClient(create_app(RAGPipeline.from_documents(bm25_config, docs)))
    assert c.post("/v1/answer", json={"question": "q"}).status_code == 503


def test_refusal_maps_to_422(docs, bm25_config):
    c = TestClient(create_app(RAGPipeline.from_documents(bm25_config, docs, FakeLLM(refuse=True))))
    r = c.post("/v1/answer", json={"question": "How does aspirin work?"})
    assert r.status_code == 422 and "declined" in r.json()["detail"]
