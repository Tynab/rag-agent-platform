"""Ingest tăng dần — chỉ embed chunk mới/đã sửa, giữ nguyên chunk không đổi.

Pipeline ingest chạy thật với qdrant-client local mode (":memory:") thay cho Qdrant
server và httpx.MockTransport thay cho Ollama /api/embed. Embedder giả GHI LẠI mọi
request để kiểm tra: lần ingest lại không đổi gì → 0 lời gọi embed; sửa một file → chỉ
embed chunk của file đó; không còn embed "health check" chỉ để biết vector size.
Neo4j được thay bằng graph giả CÓ TRẠNG THÁI (giữ tập Chunk id) để kiểm tra tập giữ lại
khi dọn và việc chỉ ghi/trích xuất chunk mới.
"""

import json

import pytest

ingest = pytest.importorskip("ingest")
httpx = pytest.importorskip("httpx")
qdrant_client = pytest.importorskip("qdrant_client")
models = pytest.importorskip("qdrant_client.http.models")

PROJECT = "crm"
COLL = ingest.get_collection_name(PROJECT)
DIM = 3


class _StatefulGraph:
    """Neo4j giả: giữ Chunk id → source_file và ghi lại mọi lời gọi từ ingest."""

    def __init__(self):
        self.chunks: dict[str, str] = {}
        self.saved: list[list[str]] = []
        self.keep_calls: list[set[str]] = []
        self.entity_chunks: list[str] = []

    def setup_schema(self):
        pass

    def existing_chunk_ids(self, project, ids):
        assert project == PROJECT
        return {i for i in ids if i in self.chunks}

    def save_project_graph(self, project, pairs):
        self.saved.append([pid for _, pid in pairs])
        for chunk, pid in pairs:
            self.chunks[pid] = chunk.metadata["source_file"]
        return len(pairs)

    def delete_stale_chunks(self, project, keep_ids):
        keep = set(keep_ids)
        self.keep_calls.append(keep)
        removed = [pid for pid in self.chunks if pid not in keep]
        for pid in removed:
            del self.chunks[pid]
        return len(removed)

    def save_entities(self, pid, entities, project):
        self.entity_chunks.append(pid)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    proj = raw / PROJECT
    proj.mkdir(parents=True)
    store = qdrant_client.QdrantClient(":memory:")
    state = {
        "requests": [],  # mỗi phần tử = danh sách text của một request /api/embed
        "clients": 0,  # số httpx.AsyncClient đã tạo
        "fail": None,  # text chứa chuỗi này → Ollama trả 500
        "dim": DIM,
    }

    class _AsyncAdapter:
        def __init__(self, *_a, **_k):
            pass

        async def upsert(self, **kw):
            return store.upsert(**kw)

        async def close(self):
            pass

    def _embed(request):
        texts = json.loads(request.content)["input"]
        state["requests"].append(texts)
        if state["fail"] and any(state["fail"] in t for t in texts):
            return httpx.Response(500, json={"error": "boom"})
        vec = [1.0] + [0.5] * (state["dim"] - 1)
        return httpx.Response(200, json={"embeddings": [list(vec) for _ in texts]})

    real_async_client = httpx.AsyncClient

    def _make_async_client(**kw):
        state["clients"] += 1
        return real_async_client(transport=httpx.MockTransport(_embed), **kw)

    monkeypatch.setattr(ingest.httpx, "AsyncClient", _make_async_client)
    monkeypatch.setattr(ingest, "QdrantClient", lambda *a, **k: store)
    monkeypatch.setattr(ingest, "AsyncQdrantClient", _AsyncAdapter)
    monkeypatch.setattr(ingest, "RAW_DATA_DIR", str(raw))
    monkeypatch.setattr(ingest, "GRAPH_ENABLED", False)
    monkeypatch.setattr(ingest, "_INGEST_EMBED_RETRIES", 1)  # không sleep khi test batch lỗi

    def write(name, text):
        (proj / name).write_text(text, encoding="utf-8")

    def embedded():
        """Mọi text đã gửi đi embed (phẳng) rồi xóa log."""
        texts = [t for req in state["requests"] for t in req]
        state["requests"].clear()
        return texts

    def index():
        pts, _ = store.scroll(COLL, limit=1000, with_payload=True)
        return {str(p.id): p.payload["relative_path"] for p in pts}

    return proj, store, state, write, embedded, index


@pytest.fixture()
def graph(env, monkeypatch):
    fake = _StatefulGraph()
    monkeypatch.setattr(ingest, "GRAPH_ENABLED", True)
    monkeypatch.setattr(ingest, "GRAPH_ENTITY_EXTRACTION", True)
    monkeypatch.setattr(ingest, "CHAT_MODEL", "fake-chat")
    monkeypatch.setattr(ingest, "_graph_module", fake)
    extracted: list[str] = []

    def _fake_extract(doc_text, llm):
        extracted.append(doc_text)
        return [{"name": "alpha", "type": "Concept"}, {"name": "beta", "type": "Concept"},
                {"name": "gamma", "type": "Concept"}]

    monkeypatch.setattr(ingest, "_extract_entities_for_document", _fake_extract)
    return fake, extracted


def _seed(write):
    write("a.md", "# A\nalpha version ONE\n")
    write("b.md", "# B\nbeta stays\n")
    write("c.md", "# C\ngamma stays\n")


# ── Qdrant ───────────────────────────────────────────────────────────────────

def test_unchanged_reingest_embeds_nothing_and_removes_nothing(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    first = ingest.ingest(project=PROJECT)
    assert sorted(embedded()) == sorted(["# A\nalpha version ONE", "# B\nbeta stays", "# C\ngamma stays"])
    assert first["upserted"] == 3 and first["skipped_unchanged"] == 0
    before = index()
    clients_before = state["clients"]

    second = ingest.ingest(project=PROJECT)

    assert embedded() == []  # không gọi Ollama lần nào
    assert state["clients"] == clients_before  # không cả tạo HTTP client
    assert second["upserted"] == 0
    assert second["skipped_unchanged"] == 3
    assert second["stale_removed"] == 0
    assert second["points_count"] == 3
    assert second["failed_batches"] == 0 and second["batch_error"] is None
    assert index() == before


def test_editing_one_file_embeds_only_its_chunks(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    embedded()
    before = index()
    old_a = {pid for pid, rp in before.items() if rp == "a.md"}

    write("a.md", "# A\nalpha version TWO\n")
    result = ingest.ingest(project=PROJECT)

    assert embedded() == ["# A\nalpha version TWO"]
    assert result["upserted"] == 1 and result["skipped_unchanged"] == 2
    assert result["stale_removed"] == len(old_a) == 1
    after = index()
    assert not (old_a & set(after))  # chunk cũ của a.md đã bị xóa
    # Chunk của b.md / c.md giữ nguyên point id.
    assert {pid for pid, rp in before.items() if rp != "a.md"} <= set(after)
    assert sorted(after.values()) == ["a.md", "b.md", "c.md"]


def test_deleting_a_file_removes_only_its_points(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    embedded()
    before = index()

    (proj / "b.md").unlink()
    result = ingest.ingest(project=PROJECT)

    assert embedded() == []
    assert result["stale_removed"] == 1 and result["upserted"] == 0
    assert index() == {pid: rp for pid, rp in before.items() if rp != "b.md"}


def test_new_file_embeds_only_the_new_file(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    embedded()

    write("d.md", "# D\ndelta new\n")
    result = ingest.ingest(project=PROJECT)

    assert embedded() == ["# D\ndelta new"]
    assert result["upserted"] == 1 and result["skipped_unchanged"] == 3
    assert result["stale_removed"] == 0 and result["points_count"] == 4


def test_reset_reembeds_everything(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    embedded()
    before = index()

    result = ingest.ingest(project=PROJECT, reset=True)

    assert len(embedded()) == 3  # reset: embed lại toàn bộ như trước
    assert result["upserted"] == 3 and result["skipped_unchanged"] == 0
    assert result["stale_removed"] == 0 and result["reset"] is True
    assert index() == before  # cùng point id (tất định), collection dựng lại


def test_no_health_check_embed_and_size_from_first_real_batch(env):
    proj, store, state, write, embedded, index = env
    state["dim"] = 5
    _seed(write)
    ingest.ingest(project=PROJECT)

    texts = embedded()
    assert "health check" not in texts
    assert len(texts) == 3
    assert ingest.collection_vector_size(store, COLL) == 5


def test_vector_size_read_from_existing_collection(env, monkeypatch):
    proj, store, state, write, embedded, index = env
    store.create_collection(COLL, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
    sizes = []
    real = ingest.collection_vector_size
    monkeypatch.setattr(ingest, "collection_vector_size", lambda c, n: sizes.append(n) or real(c, n))
    write("a.md", "# A\nalpha\n")

    result = ingest.ingest(project=PROJECT)

    assert sizes == [COLL]
    assert embedded() == ["# A\nalpha"]  # chỉ chunk thật, không có "health check"
    assert result["upserted"] == 1 and result["points_count"] == 1


def test_dimension_mismatch_with_existing_collection_reports_clear_error(env):
    proj, store, state, write, embedded, index = env
    store.create_collection(COLL, vectors_config=models.VectorParams(size=7, distance=models.Distance.COSINE))
    write("a.md", "# A\nalpha\n")

    result = ingest.ingest(project=PROJECT)

    assert result["upserted"] == 0 and result["failed_batches"] == 1
    assert "reset-ingest" in result["batch_error"] and "dim=7" in result["batch_error"]
    assert result["stale_removed"] == 0 and result["points_count"] == 0


def test_all_batches_failing_on_fresh_project_does_not_crash(env):
    proj, store, state, write, embedded, index = env
    state["fail"] = "alpha"
    write("a.md", "# A\nalpha\n")

    result = ingest.ingest(project=PROJECT)

    # R1 giữ nguyên: batch lỗi không làm hỏng request; collection chưa được tạo.
    assert result["status"] == "ok"
    assert result["upserted"] == 0 and result["failed_batches"] == 1
    assert "HTTPStatusError" in result["batch_error"]
    assert result["points_count"] == 0
    assert not store.collection_exists(COLL)


def test_concurrent_batches_create_collection_once_even_if_first_fails(env, monkeypatch):
    proj, store, state, write, embedded, index = env
    monkeypatch.setattr(ingest, "UPSERT_BATCH_SIZE", 1)
    monkeypatch.setattr(ingest, "INGEST_EMBED_WORKERS", 3)
    state["fail"] = "alpha"  # a.md là batch đầu tiên (file sắp xếp theo tên)
    _seed(write)

    result = ingest.ingest(project=PROJECT)

    texts = embedded()
    assert "health check" not in texts and len(texts) == 3
    assert result["failed_batches"] == 1 and result["upserted"] == 2
    assert result["points_count"] == 2
    assert sorted(index().values()) == ["b.md", "c.md"]
    assert ingest.collection_vector_size(store, COLL) == DIM

    # Lần sau Ollama ổn → chỉ embed chunk còn thiếu (a.md), không embed lại b/c.
    state["fail"] = None
    result = ingest.ingest(project=PROJECT)
    assert embedded() == ["# A\nalpha version ONE"]
    assert result["upserted"] == 1 and result["skipped_unchanged"] == 2


def test_failed_batch_on_edit_keeps_old_version(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    before = index()

    write("a.md", "# A\nalpha version FAIL\n")
    state["fail"] = "FAIL"
    result = ingest.ingest(project=PROJECT)

    assert result["failed_batches"] == 1 and result["stale_removed"] == 0
    assert index() == before  # embed lỗi → không xóa bản cũ


# ── Neo4j ────────────────────────────────────────────────────────────────────

def test_graph_unchanged_reingest_keeps_all_chunk_nodes(env, graph):
    proj, store, state, write, embedded, index = env
    fake, extracted = graph
    _seed(write)
    ingest.ingest(project=PROJECT)
    all_ids = set(index())
    assert set(fake.chunks) == all_ids
    assert len(extracted) == 3 and sorted(fake.entity_chunks) == sorted(all_ids)
    extracted.clear()
    fake.entity_chunks.clear()

    result = ingest.ingest(project=PROJECT)

    # Tập giữ lại = MỌI chunk hiện tại (không phải danh sách vừa upsert — rỗng ở đây).
    assert fake.keep_calls[-1] == all_ids
    assert set(fake.chunks) == all_ids  # không xóa oan Chunk node của chunk không đổi
    assert fake.saved[-1] == []  # không ghi lại gì
    assert extracted == [] and fake.entity_chunks == []  # không gọi LLM
    assert result["graph_chunks_saved"] == 0


def test_graph_edit_writes_and_extracts_only_new_chunks(env, graph):
    proj, store, state, write, embedded, index = env
    fake, extracted = graph
    _seed(write)
    ingest.ingest(project=PROJECT)
    extracted.clear()
    fake.entity_chunks.clear()

    write("a.md", "# A\nalpha version TWO\n")
    (proj / "c.md").unlink()
    result = ingest.ingest(project=PROJECT)

    now = index()
    new_a = {pid for pid, rp in now.items() if rp == "a.md"}
    assert set(fake.saved[-1]) == new_a
    assert extracted == ["# A\nalpha version TWO"]
    assert set(fake.entity_chunks) == new_a
    assert fake.keep_calls[-1] == set(now)
    assert set(fake.chunks) == set(now)  # a.md cũ + c.md bị dọn, b.md giữ nguyên
    assert result["graph_chunks_saved"] == len(new_a)


def test_graph_enabled_later_builds_full_graph(env, monkeypatch):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)  # GRAPH_ENABLED=False
    embedded()

    fake = _StatefulGraph()
    monkeypatch.setattr(ingest, "GRAPH_ENABLED", True)
    monkeypatch.setattr(ingest, "GRAPH_ENTITY_EXTRACTION", False)
    monkeypatch.setattr(ingest, "_graph_module", fake)
    result = ingest.ingest(project=PROJECT)

    assert embedded() == []  # Qdrant không có gì mới
    assert set(fake.saved[-1]) == set(index())  # nhưng Neo4j trống → dựng đủ graph
    assert set(fake.chunks) == set(index())
    assert result["graph_chunks_saved"] == 3


def test_graph_rewrites_whole_file_to_keep_next_chunk_chain(env, graph, monkeypatch):
    proj, store, state, write, embedded, index = env
    fake, extracted = graph
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 30)
    monkeypatch.setattr(ingest, "CHUNK_OVERLAP", 0)
    paras = ["alpha one two three", "beta four five six", "gamma seven eight"]
    write("long.md", "\n\n".join(paras) + "\n")
    write("other.md", "# O\nunrelated\n")
    ingest.ingest(project=PROJECT)
    long_ids = {pid for pid, rp in index().items() if rp == "long.md"}
    assert len(long_ids) == 3
    extracted.clear()
    fake.entity_chunks.clear()

    write("long.md", "\n\n".join(paras[:2] + ["gamma seven NINE"]) + "\n")
    ingest.ingest(project=PROJECT)

    now = {pid: rp for pid, rp in index().items()}
    new_long = {pid for pid, rp in now.items() if rp == "long.md"}
    changed = new_long - long_ids
    assert len(changed) == 1
    # Ghi lại CẢ file long.md (chuỗi NEXT_CHUNK liền mạch), không đụng other.md...
    assert set(fake.saved[-1]) == new_long
    # ...nhưng chỉ trích xuất thực thể cho chunk mới.
    assert set(fake.entity_chunks) == changed
    assert set(fake.chunks) == set(now)


def test_graph_reset_writes_everything(env, graph):
    proj, store, state, write, embedded, index = env
    fake, extracted = graph
    _seed(write)
    ingest.ingest(project=PROJECT)
    deleted = []
    fake.delete_project_graph = lambda project: (deleted.append(project), fake.chunks.clear())
    extracted.clear()
    keep_calls = len(fake.keep_calls)

    result = ingest.ingest(project=PROJECT, reset=True)

    assert deleted == [PROJECT]
    assert set(fake.saved[-1]) == set(index())
    assert len(extracted) == 3
    assert len(fake.keep_calls) == keep_calls  # reset: không chạy bước dọn
    assert result["graph_chunks_saved"] == 3
