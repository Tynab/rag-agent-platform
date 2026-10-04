"""Test endpoint rag-api (F6 + F10) và _query_rag của agent-api (F10).

rag-api/app.py trùng tên module `app` với agent-api → load bằng importlib dưới tên
riêng "rag_api_app" (không đụng sys.modules["app"]). Qdrant/embed/LLM được thay bằng
bản giả nên không cần dịch vụ ngoài.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("ingest")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _load_rag_app():
    name = "rag_api_app"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "rag-api" / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


rag_app = _load_rag_app()


class _FakeQdrant:
    def __init__(self, n_hits=3):
        self.n_hits = n_hits

    def collection_exists(self, name):
        return name == rag_app.get_collection_name("crm")

    def get_collection(self, name):
        raise RuntimeError("không cần cho test")

    def query_points(self, **kw):
        pts = [
            types.SimpleNamespace(id=f"id{i}", score=0.9 - i / 10, payload={
                "page_content": f"CHUNK {i}: tenantId là field #1 của mọi compound index",
                "project": "crm", "module": "auth", "source_file": f"f{i}.md", "relative_path": f"auth/f{i}.md",
            })
            for i in range(self.n_hits)
        ]
        return types.SimpleNamespace(points=pts)


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return types.SimpleNamespace(content="câu trả lời từ CHAT_MODEL")


@pytest.fixture()
def rag(monkeypatch):
    llm = _FakeLLM()
    qdrant = _FakeQdrant()
    ingest_calls = []
    monkeypatch.setattr(rag_app, "GRAPH_ENABLED", False)
    monkeypatch.setattr(rag_app, "get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(rag_app, "embed_texts", lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    monkeypatch.setattr(rag_app, "get_chat_model", lambda: llm)
    monkeypatch.setattr(rag_app, "ingest", lambda **kw: ingest_calls.append(kw) or {"status": "ok"})
    return types.SimpleNamespace(client=TestClient(rag_app.app), llm=llm, qdrant=qdrant, ingest_calls=ingest_calls)


# ── F6: tên project không hợp lệ → 400, không chạm filesystem/Qdrant ─────────────

@pytest.mark.parametrize("bad", [".", "..", "/", "../..", "a/b", ".git", ""])
@pytest.mark.parametrize("path", ["/ingest", "/reset-ingest"])
def test_ingest_endpoints_reject_bad_project(rag, path, bad):
    r = rag.client.post(path, json={"project": bad})
    assert r.status_code == 400, r.text
    assert rag.ingest_calls == []


@pytest.mark.parametrize("bad", [".", "..", "/", "a/b", ".git"])
@pytest.mark.parametrize("path", ["/ask", "/retrieve"])
def test_query_endpoints_reject_bad_project(rag, path, bad):
    r = rag.client.post(path, json={"question": "q", "project": bad})
    assert r.status_code == 400, r.text
    assert rag.llm.calls == 0


def test_graph_entities_rejects_bad_project(rag):
    assert rag.client.get("/graph/projects/.git/entities").status_code == 400


def test_valid_project_still_accepted(rag):
    assert rag.client.post("/ingest", json={"project": "crm"}).status_code == 200
    assert rag.ingest_calls == [{"project": "crm", "reset": False}]


# ── F10: /retrieve chỉ truy hồi, không gọi CHAT_MODEL ──────────────────────────

def test_retrieve_returns_raw_chunks_without_llm(rag):
    r = rag.client.post("/retrieve", json={"question": "index?", "project": "crm", "top_k": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert rag.llm.calls == 0
    assert "[SOURCE 1]" in body["context"] and "CHUNK 0: tenantId" in body["context"]
    assert [s["source_file"] for s in body["sources"]] == ["f0.md", "f1.md"]


def test_retrieve_no_hits_and_unknown_project(rag):
    rag.qdrant.n_hits = 0
    r = rag.client.post("/retrieve", json={"question": "q", "project": "crm"})
    assert r.status_code == 200 and r.json() == {"context": "", "sources": []}
    assert rag.client.post("/retrieve", json={"question": "q", "project": "other"}).status_code == 404


def test_ask_still_generates_answer(rag):
    r = rag.client.post("/ask", json={"question": "index?", "project": "crm"})
    assert r.status_code == 200, r.text
    assert r.json()["answer"] == "câu trả lời từ CHAT_MODEL"
    assert rag.llm.calls == 1
    rag.qdrant.n_hits = 0
    r = rag.client.post("/ask", json={"question": "q", "project": "crm"})
    assert r.json()["sources"] == [] and rag.llm.calls == 1  # không hit → không gọi LLM


def test_agent_query_rag_uses_retrieve(rag, monkeypatch):
    workflow = pytest.importorskip("workflow")
    seen = []

    def fake_post(url, json=None, timeout=None):
        seen.append(url)
        return rag.client.post(url.replace("http://rag-api:8090", ""), json=json)

    monkeypatch.setattr(workflow.requests, "post", fake_post)
    ctx = workflow._query_rag("http://rag-api:8090", "BA: CRM", "crm", 3)
    assert seen == ["http://rag-api:8090/retrieve"]
    assert "CHUNK 0: tenantId" in ctx
    assert rag.llm.calls == 0
    # rag-api trả lỗi (vd. 400 project sai) → chuỗi rỗng, không dừng workflow.
    assert workflow._query_rag("http://rag-api:8090", "q", "../x", 3) == ""
