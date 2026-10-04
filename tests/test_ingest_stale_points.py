"""Regression test F11 — ingest không-reset phải xóa chunk của file đã sửa/xóa.

Chạy pipeline ingest thật với qdrant-client local mode (":memory:") thay cho Qdrant
server và httpx.MockTransport thay cho Ollama /api/embed — không cần dịch vụ ngoài.
"""

import json

import pytest

ingest = pytest.importorskip("ingest")
httpx = pytest.importorskip("httpx")
qdrant_client = pytest.importorskip("qdrant_client")

COLL = ingest.get_collection_name("crm")


class _FakeGraph:
    """Module graph giả: chỉ ghi lại lời gọi để kiểm tra đồng bộ Neo4j."""

    def __init__(self):
        self.stale_calls: list[tuple[str, set]] = []

    def setup_schema(self):
        pass

    def save_project_graph(self, project, pairs):
        return len(pairs)

    def delete_stale_chunks(self, project, keep_ids):
        self.stale_calls.append((project, set(keep_ids)))
        return 0


@pytest.fixture()
def env(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    proj = raw / "crm"
    proj.mkdir(parents=True)
    store = qdrant_client.QdrantClient(":memory:")
    fail_marker = {"text": None}

    class _AsyncAdapter:
        def __init__(self, *_a, **_k):
            pass

        async def upsert(self, **kw):
            return store.upsert(**kw)

        async def close(self):
            pass

    def _embed(request):
        texts = json.loads(request.content)["input"]
        if fail_marker["text"] and any(fail_marker["text"] in t for t in texts):
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"embeddings": [[float(len(t) % 7 + 1), 1.0, 0.5] for t in texts]})

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        ingest.httpx, "AsyncClient",
        lambda **kw: real_async_client(transport=httpx.MockTransport(_embed), **kw),
    )
    monkeypatch.setattr(ingest, "QdrantClient", lambda *a, **k: store)
    monkeypatch.setattr(ingest, "AsyncQdrantClient", _AsyncAdapter)
    monkeypatch.setattr(ingest, "RAW_DATA_DIR", str(raw))
    monkeypatch.setattr(ingest, "GRAPH_ENABLED", False)
    monkeypatch.setattr(ingest, "_INGEST_EMBED_RETRIES", 1)  # không sleep khi test batch lỗi
    return proj, store, fail_marker


def _index(store):
    pts, _ = store.scroll(COLL, limit=1000, with_payload=True)
    return {str(p.id): (p.payload["relative_path"], p.payload["page_content"]) for p in pts}


def test_reingest_removes_edited_and_deleted_chunks(env):
    proj, store, _ = env
    (proj / "a.md").write_text("# A\nalpha version ONE\n", encoding="utf-8")
    (proj / "b.md").write_text("# B\nbeta will be deleted\n", encoding="utf-8")
    (proj / "c.md").write_text("# C\ngamma unchanged\n", encoding="utf-8")
    ingest.ingest(project="crm")
    before = _index(store)
    c_ids = {pid for pid, (rp, _) in before.items() if rp == "c.md"}
    assert len(before) == 3

    (proj / "a.md").write_text("# A\nalpha version TWO\n", encoding="utf-8")
    (proj / "b.md").unlink()
    result = ingest.ingest(project="crm")
    after = _index(store)

    assert sorted(rp for rp, _ in after.values()) == ["a.md", "c.md"]
    assert all("ONE" not in text for _, text in after.values())
    assert any("TWO" in text for _, text in after.values())
    assert c_ids <= set(after)  # chunk không đổi giữ nguyên point id
    assert result["stale_removed"] == 2
    assert result["points_count"] == 2


def test_failed_batch_skips_cleanup(env):
    proj, store, fail_marker = env
    (proj / "a.md").write_text("# A\nalpha version ONE\n", encoding="utf-8")
    ingest.ingest(project="crm")
    before = _index(store)

    (proj / "a.md").write_text("# A\nalpha version FAIL\n", encoding="utf-8")
    fail_marker["text"] = "FAIL"
    result = ingest.ingest(project="crm")

    assert _index(store) == before  # embed lỗi → không xóa bản cũ
    assert result["stale_removed"] == 0


def test_graph_mirror_called_with_current_ids(env, monkeypatch):
    proj, store, fail_marker = env
    fake = _FakeGraph()
    monkeypatch.setattr(ingest, "GRAPH_ENABLED", True)
    monkeypatch.setattr(ingest, "GRAPH_ENTITY_EXTRACTION", False)
    monkeypatch.setattr(ingest, "_graph_module", fake)

    (proj / "a.md").write_text("# A\nalpha\n", encoding="utf-8")
    (proj / "b.md").write_text("# B\nbeta\n", encoding="utf-8")
    ingest.ingest(project="crm")
    (proj / "b.md").unlink()
    ingest.ingest(project="crm")

    assert len(fake.stale_calls) == 2
    project, keep = fake.stale_calls[-1]
    assert project == "crm" and keep == set(_index(store))

    # reset=True: collection dựng lại từ đầu → không cần bước dọn.
    ingest.ingest(project="crm", reset=True)
    assert len(fake.stale_calls) == 2

    # batch lỗi → Qdrant không dọn → Neo4j cũng không dọn.
    (proj / "a.md").write_text("# A\nFAIL\n", encoding="utf-8")
    fail_marker["text"] = "FAIL"
    ingest.ingest(project="crm")
    assert len(fake.stale_calls) == 2
