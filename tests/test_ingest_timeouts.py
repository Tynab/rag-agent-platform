"""Timeout thật cho các lời gọi Ollama của rag-api (ingest.py).

langchain-ollama 0.3.x KHÔNG có field `request_timeout` trên ChatOllama — kwarg đó bị
bỏ qua im lặng và httpx client chạy với timeout=None. Vì vậy test kiểm tra timeout
trên httpx client THẬT bên trong ChatOllama (`_client._client.timeout`), không dùng
ChatOllama giả ghi lại kwargs (bản giả vẫn pass dù code truyền kwarg bị bỏ qua).

Ollama và Qdrant được thay bằng httpx.MockTransport + qdrant-client ":memory:".
"""

import json

import pytest

ingest = pytest.importorskip("ingest")
httpx = pytest.importorskip("httpx")
qdrant_client = pytest.importorskip("qdrant_client")


def _sync_timeout(llm):
    return llm._client._client.timeout


def _async_timeout(llm):
    return llm._async_client._client.timeout


# ── make_chat_model: timeout đi qua client_kwargs ────────────────────────────────

def test_make_chat_model_applies_real_httpx_timeout(monkeypatch):
    monkeypatch.setattr(ingest, "OLLAMA_REQUEST_TIMEOUT", 123.0)
    llm = ingest.make_chat_model("qwen3:8b", temperature=0)
    assert _sync_timeout(llm) == httpx.Timeout(123.0)
    assert _async_timeout(llm) == httpx.Timeout(123.0)
    assert llm.model == "qwen3:8b"
    assert llm.temperature == 0
    assert llm.base_url == ingest.OLLAMA_BASE_URL


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_make_chat_model_non_positive_timeout_means_unlimited(monkeypatch, value):
    monkeypatch.setattr(ingest, "OLLAMA_REQUEST_TIMEOUT", value)
    llm = ingest.make_chat_model("m", temperature=0)
    assert _sync_timeout(llm) == httpx.Timeout(None)


def test_make_chat_model_does_not_share_client_kwargs():
    a = ingest.make_chat_model("m", temperature=0)
    b = ingest.make_chat_model("m", temperature=0)
    assert a.client_kwargs is not b.client_kwargs


def test_get_embeddings_dead_code_removed():
    # Không còn caller nào; embed đi qua embed_texts / pipeline httpx /api/embed.
    assert not hasattr(ingest, "get_embeddings")
    assert not hasattr(ingest, "OllamaEmbeddings")


def test_ingest_embed_retries_is_at_least_one():
    assert ingest._INGEST_EMBED_RETRIES >= 1


# ── Pipeline ingest: RAG_TIMEOUT cho /api/embed, OLLAMA_REQUEST_TIMEOUT cho entity ──

class _FakeGraph:
    def setup_schema(self):
        pass

    def save_project_graph(self, project, pairs):
        return len(pairs)

    def delete_stale_chunks(self, project, keep_ids):
        return 0

    def save_entities(self, pid, entities, project):
        pass


@pytest.fixture()
def env(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    proj = raw / "crm"
    proj.mkdir(parents=True)
    (proj / "a.md").write_text("# A\nalpha\n", encoding="utf-8")
    store = qdrant_client.QdrantClient(":memory:")
    async_client_kwargs: list[dict] = []

    class _AsyncAdapter:
        def __init__(self, *_a, **_k):
            pass

        async def upsert(self, **kw):
            return store.upsert(**kw)

        async def close(self):
            pass

    def _embed(request):
        texts = json.loads(request.content)["input"]
        return httpx.Response(200, json={"embeddings": [[1.0, 0.5, 0.25] for _ in texts]})

    real_async_client = httpx.AsyncClient

    def _make_async_client(**kw):
        async_client_kwargs.append(kw)
        return real_async_client(transport=httpx.MockTransport(_embed), **kw)

    monkeypatch.setattr(ingest.httpx, "AsyncClient", _make_async_client)
    monkeypatch.setattr(ingest, "QdrantClient", lambda *a, **k: store)
    monkeypatch.setattr(ingest, "AsyncQdrantClient", _AsyncAdapter)
    monkeypatch.setattr(ingest, "RAW_DATA_DIR", str(raw))
    monkeypatch.setattr(ingest, "GRAPH_ENABLED", False)
    return async_client_kwargs


def test_batch_embed_uses_rag_timeout(env, monkeypatch):
    monkeypatch.setattr(ingest, "_EMBED_HTTP_TIMEOUT", 42.0)
    result = ingest.ingest(project="crm")
    assert result["status"] == "ok" and result["upserted"] == 1
    assert len(env) == 1
    timeout = env[0]["timeout"]
    assert timeout.read == 42.0  # trước đây hardcode 600 → RAG_TIMEOUT không có tác dụng


def test_entity_extraction_llm_has_real_timeout(env, monkeypatch):
    monkeypatch.setattr(ingest, "GRAPH_ENABLED", True)
    monkeypatch.setattr(ingest, "GRAPH_ENTITY_EXTRACTION", True)
    monkeypatch.setattr(ingest, "CHAT_MODEL", "qwen3:8b")
    monkeypatch.setattr(ingest, "_graph_module", _FakeGraph())
    monkeypatch.setattr(ingest, "OLLAMA_REQUEST_TIMEOUT", 321.0)
    seen = []

    def _fake_extract(doc_text, llm):
        seen.append(llm)
        return []

    monkeypatch.setattr(ingest, "_extract_entities_for_document", _fake_extract)
    ingest.ingest(project="crm")

    assert len(seen) == 1
    assert seen[0].model == "qwen3:8b"
    assert _sync_timeout(seen[0]) == httpx.Timeout(321.0)
