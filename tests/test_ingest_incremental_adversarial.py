"""Ingest tăng dần — test đối kháng (review slice incremental-ingest).

Cùng cách dựng với test_ingest_incremental.py: qdrant-client ":memory:" thay Qdrant
server, httpx.MockTransport thay Ollama /api/embed (GHI LẠI mọi text đã embed) và
Neo4j giả CÓ TRẠNG THÁI. Tập trung vào các đường dễ vỡ mà bộ test gốc chưa chạm:

- reset=True khi Ollama down / EMBEDDING_MODEL mới chưa pull: trước slice, embed
  "health check" lỗi TRƯỚC bước xóa collection; sau khi bỏ health check, reset từng xóa
  collection ngay từ đầu rồi mọi batch lỗi → index trắng (mà response vẫn "ok").
- Ollama trả THIẾU vector so với số text: zip() cắt im lặng → bước dọn F11 xóa bản cũ
  của chunk không có vector mới → index thủng.
- reset sau khi đổi số chiều embedding, reset với batch đầu lỗi, reset không có chunk.
- Đổi tên / chuyển file, phân trang retrieve/scroll/delete (_STALE_PAGE_SIZE nhỏ).
- Neo4j: batch lỗi không dọn và không ghi chunk lỗi; graph trống được dựng lại đủ.
"""

import json

import pytest

ingest = pytest.importorskip("ingest")
httpx = pytest.importorskip("httpx")
qdrant_client = pytest.importorskip("qdrant_client")

PROJECT = "crm"
COLL = ingest.get_collection_name(PROJECT)


class _StatefulGraph:
    """Neo4j giả: giữ Chunk id → source_file và ghi lại mọi lời gọi từ ingest."""

    def __init__(self):
        self.chunks: dict[str, str] = {}
        self.saved: list[list[str]] = []
        self.keep_calls: list[set[str]] = []
        self.deleted_projects: list[str] = []

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
        for pid in [p for p in self.chunks if p not in keep]:
            del self.chunks[pid]
        return 0

    def delete_project_graph(self, project):
        self.deleted_projects.append(project)
        self.chunks.clear()

    def save_entities(self, pid, entities, project):
        pass


@pytest.fixture()
def env(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    proj = raw / PROJECT
    proj.mkdir(parents=True)
    store = qdrant_client.QdrantClient(":memory:")
    state = {
        "requests": [],  # mỗi phần tử = danh sách text của một request /api/embed
        "status": 200,  # != 200 → Ollama trả lỗi cho MỌI request (down / model chưa pull)
        "fail": None,  # text chứa chuỗi này → Ollama trả 500
        "short": False,  # True → trả thiếu một vector so với số text
        "dim": 3,
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
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": "model not found"})
        if state["fail"] and any(state["fail"] in t for t in texts):
            return httpx.Response(500, json={"error": "boom"})
        vecs = [[1.0] + [0.5] * (state["dim"] - 1) for _ in texts]
        if state["short"]:
            vecs = vecs[:-1]
        return httpx.Response(200, json={"embeddings": vecs})

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

    def write(name, text):
        path = proj / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def embedded():
        """Mọi text đã gửi đi embed (phẳng) rồi xóa log."""
        texts = [t for req in state["requests"] for t in req]
        state["requests"].clear()
        return texts

    def index():
        if not store.collection_exists(COLL):
            return {}
        pts, _ = store.scroll(COLL, limit=1000, with_payload=True)
        return {str(p.id): p.payload["relative_path"] for p in pts}

    return proj, store, state, write, embedded, index


@pytest.fixture()
def graph(env, monkeypatch):
    fake = _StatefulGraph()
    monkeypatch.setattr(ingest, "GRAPH_ENABLED", True)
    monkeypatch.setattr(ingest, "GRAPH_ENTITY_EXTRACTION", False)
    monkeypatch.setattr(ingest, "_graph_module", fake)
    return fake


def _seed(write):
    write("a.md", "# A\nalpha version ONE\n")
    write("b.md", "# B\nbeta stays\n")
    write("c.md", "# C\ngamma stays\n")


# ── reset=True ───────────────────────────────────────────────────────────────

def test_reset_with_ollama_down_keeps_existing_index(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    before = index()
    embedded()

    state["status"] = 404  # vd. EMBEDDING_MODEL mới chưa `ollama pull`
    result = ingest.ingest(project=PROJECT, reset=True)

    assert len(embedded()) == 3  # đã thử embed thật (không có "health check")
    assert index() == before  # không batch nào thành công → KHÔNG xóa collection cũ
    assert result["failed_batches"] == 1 and "404" in result["batch_error"]
    assert result["upserted"] == 0 and result["points_count"] == 3
    assert result["reset"] is True


def test_reset_with_malformed_embed_response_keeps_existing_index(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    before = index()

    state["short"] = True
    result = ingest.ingest(project=PROJECT, reset=True)

    assert index() == before
    assert result["failed_batches"] == 1 and "vector" in result["batch_error"]


def test_reset_after_embedding_dim_change_rebuilds_with_new_dim(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    before = index()
    assert ingest.collection_vector_size(store, COLL) == 3

    state["dim"] = 5  # model embedding đổi số chiều
    # Không-reset: lệch dimension → batch lỗi rõ ràng, index cũ giữ nguyên.
    # (Cùng tên model nên point id không đổi → phải sửa file để có chunk cần embed.)
    write("d.md", "# D\ndelta new\n")
    result = ingest.ingest(project=PROJECT)
    assert result["failed_batches"] == 1 and "reset-ingest" in result["batch_error"]
    assert index() == before
    embedded()

    result = ingest.ingest(project=PROJECT, reset=True)

    assert len(embedded()) == 4  # reset embed lại toàn bộ
    assert result["failed_batches"] == 0 and result["batch_error"] is None
    assert result["upserted"] == 4 and result["points_count"] == 4
    assert ingest.collection_vector_size(store, COLL) == 5  # collection dựng lại theo dim mới
    assert set(before) <= set(index())


def test_reset_first_batch_fails_later_batch_drops_and_rebuilds(env, monkeypatch):
    proj, store, state, write, embedded, index = env
    monkeypatch.setattr(ingest, "UPSERT_BATCH_SIZE", 1)
    monkeypatch.setattr(ingest, "INGEST_EMBED_WORKERS", 3)
    _seed(write)
    ingest.ingest(project=PROJECT)

    state["dim"] = 5
    state["fail"] = "alpha"  # batch của a.md lỗi, b/c thành công
    result = ingest.ingest(project=PROJECT, reset=True)

    # Batch thành công đầu tiên xóa collection cũ (dim 3) và tạo lại theo dim 5 —
    # đúng một lần dù 3 batch chạy đồng thời; a.md mất như reset dở dang trước đây.
    assert ingest.collection_vector_size(store, COLL) == 5
    assert sorted(index().values()) == ["b.md", "c.md"]
    assert result["failed_batches"] == 1 and result["upserted"] == 2
    assert result["points_count"] == 2


def test_reset_failed_drop_is_retried_by_next_batch(env, monkeypatch):
    proj, store, state, write, embedded, index = env
    monkeypatch.setattr(ingest, "UPSERT_BATCH_SIZE", 1)
    monkeypatch.setattr(ingest, "INGEST_EMBED_WORKERS", 1)  # thứ tự batch tất định
    _seed(write)
    ingest.ingest(project=PROJECT)

    real_delete = store.delete_collection
    calls = {"n": 0}

    def _flaky_delete(name, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("qdrant busy")
        return real_delete(name, *a, **kw)

    monkeypatch.setattr(store, "delete_collection", _flaky_delete)
    state["dim"] = 5
    result = ingest.ingest(project=PROJECT, reset=True)

    # Batch a.md: xóa lỗi → batch lỗi; batch b.md thử xóa lại rồi tạo collection dim 5
    # (không upsert dim 5 lẫn vào collection dim 3 cũ).
    assert calls["n"] == 2
    assert result["failed_batches"] == 1 and "qdrant busy" in result["batch_error"]
    assert ingest.collection_vector_size(store, COLL) == 5
    assert sorted(index().values()) == ["b.md", "c.md"]


def test_reset_with_no_chunks_still_drops_collection(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    for name in ("a.md", "b.md", "c.md"):
        write(name, "   \n")  # file vẫn còn (documents > 0) nhưng không sinh chunk nào
    embedded()

    result = ingest.ingest(project=PROJECT, reset=True)

    assert embedded() == []
    assert result["chunks"] == 0 and result["points_count"] == 0
    assert index() == {}


# ── Phản hồi embed lỗi định dạng ─────────────────────────────────────────────

def test_short_embed_response_is_a_failed_batch_and_deletes_nothing(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    before = index()

    write("a.md", "# A\nalpha version TWO\n")
    write("b.md", "# B\nbeta TWO\n")
    state["short"] = True  # 2 text → 1 vector
    result = ingest.ingest(project=PROJECT)

    assert result["failed_batches"] == 1 and result["upserted"] == 0
    assert "2 chunk" in result["batch_error"]
    assert result["stale_removed"] == 0
    assert index() == before  # bản cũ của a.md/b.md KHÔNG bị dọn


# ── File nhiều chunk ─────────────────────────────────────────────────────────

_PARAS = ["alpha one two three", "beta four five six", "gamma seven eight"]


def _multi_chunk(monkeypatch, write):
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 30)
    monkeypatch.setattr(ingest, "CHUNK_OVERLAP", 0)
    write("long.md", "\n\n".join(_PARAS) + "\n")
    write("other.md", "# O\nunrelated\n")


def test_editing_last_chunk_of_multi_chunk_file_embeds_only_that_chunk(env, monkeypatch):
    proj, store, state, write, embedded, index = env
    _multi_chunk(monkeypatch, write)
    ingest.ingest(project=PROJECT)
    before = index()
    assert sorted(before.values()) == ["long.md"] * 3 + ["other.md"]
    embedded()

    write("long.md", "\n\n".join(_PARAS[:2] + ["gamma seven NINE"]) + "\n")
    result = ingest.ingest(project=PROJECT)

    assert embedded() == ["gamma seven NINE"]  # chỉ chunk đã sửa
    assert result["upserted"] == 1 and result["skipped_unchanged"] == 3
    assert result["stale_removed"] == 1 and result["points_count"] == 4
    after = index()
    assert len(set(before) - set(after)) == 1  # đúng một point cũ bị thay


def test_prepending_to_multi_chunk_file_reembeds_only_that_file(env, monkeypatch):
    proj, store, state, write, embedded, index = env
    _multi_chunk(monkeypatch, write)
    ingest.ingest(project=PROJECT)
    before = index()
    other_ids = {pid for pid, rp in before.items() if rp == "other.md"}
    embedded()

    write("long.md", "\n\n".join(["zeta new first"] + _PARAS) + "\n")
    result = ingest.ingest(project=PROJECT)

    # chunk_index nằm trong point id → chèn đầu file làm dịch id mọi chunk SAU nó
    # (thiết kế id hiện tại), nhưng KHÔNG đụng tới file khác.
    texts = embedded()
    assert sorted(texts) == sorted(["zeta new first"] + _PARAS)
    assert result["skipped_unchanged"] == 1 and result["stale_removed"] == 3
    after = index()
    assert other_ids <= set(after)
    assert sorted(after.values()) == ["long.md"] * 4 + ["other.md"]


# ── Đổi tên / chuyển file ────────────────────────────────────────────────────

def test_rename_moves_points(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    before = index()
    embedded()

    (proj / "a.md").rename(proj / "z.md")
    result = ingest.ingest(project=PROJECT)

    # relative_path nằm trong point id → file đổi tên embed lại, id cũ bị dọn.
    assert embedded() == ["# A\nalpha version ONE"]
    assert result["upserted"] == 1 and result["skipped_unchanged"] == 2
    assert result["stale_removed"] == 1 and result["points_count"] == 3
    after = index()
    assert sorted(after.values()) == ["b.md", "c.md", "z.md"]
    assert {pid for pid, rp in before.items() if rp != "a.md"} <= set(after)


def test_move_into_module_dir_refreshes_module_payload(env):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)

    (proj / "billing").mkdir()
    (proj / "a.md").rename(proj / "billing" / "a.md")
    result = ingest.ingest(project=PROJECT)

    assert result["upserted"] == 1 and result["stale_removed"] == 1
    pts, _ = store.scroll(COLL, limit=100, with_payload=True)
    modules = {p.payload["relative_path"]: p.payload["module"] for p in pts}
    assert modules == {"billing/a.md": "billing", "b.md": PROJECT, "c.md": PROJECT}


# ── Phân trang retrieve / scroll / delete ────────────────────────────────────

def test_incremental_ingest_across_page_boundaries(env, monkeypatch):
    proj, store, state, write, embedded, index = env
    monkeypatch.setattr(ingest, "_STALE_PAGE_SIZE", 2)
    monkeypatch.setattr(ingest, "UPSERT_BATCH_SIZE", 2)
    retrieve_pages: list[int] = []
    real_retrieve = store.retrieve

    def _spy_retrieve(*a, **kw):
        retrieve_pages.append(len(kw["ids"]))
        return real_retrieve(*a, **kw)

    monkeypatch.setattr(store, "retrieve", _spy_retrieve)
    for i in range(7):
        write(f"f{i}.md", f"# F{i}\nfile number {i}\n")
    ingest.ingest(project=PROJECT)
    before = index()
    assert len(before) == 7
    embedded()
    retrieve_pages.clear()

    # Không đổi gì: 7 id tra theo trang ≤ 2, không embed, không xóa.
    result = ingest.ingest(project=PROJECT)
    assert retrieve_pages == [2, 2, 2, 1]
    assert embedded() == [] and result["skipped_unchanged"] == 7
    assert result["stale_removed"] == 0 and index() == before

    # Sửa 1 file + xóa 3 file: dọn qua nhiều trang scroll/delete, chỉ đúng id cũ.
    write("f6.md", "# F6\nfile number six\n")
    for i in (0, 3, 5):
        (proj / f"f{i}.md").unlink()
    result = ingest.ingest(project=PROJECT)

    assert embedded() == ["# F6\nfile number six"]
    assert result["stale_removed"] == 4 and result["points_count"] == 4
    after = index()
    assert sorted(after.values()) == ["f1.md", "f2.md", "f4.md", "f6.md"]
    kept = {pid for pid, rp in before.items() if rp in ("f1.md", "f2.md", "f4.md")}
    assert kept <= set(after)


# ── Neo4j ────────────────────────────────────────────────────────────────────

def test_graph_failed_batch_keeps_old_chunk_and_skips_failed_chunk(env, graph, monkeypatch):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    before = set(graph.chunks)
    keep_calls = len(graph.keep_calls)

    write("a.md", "# A\nalpha version FAIL\n")
    write("d.md", "# D\ndelta new\n")
    state["fail"] = "FAIL"
    monkeypatch.setattr(ingest, "UPSERT_BATCH_SIZE", 1)  # a.md lỗi, d.md thành công
    result = ingest.ingest(project=PROJECT)

    assert result["failed_batches"] == 1 and result["upserted"] == 1
    assert len(graph.keep_calls) == keep_calls  # có batch lỗi → KHÔNG dọn Neo4j
    d_ids = {pid for pid, rp in index().items() if rp == "d.md"}
    assert graph.chunks.keys() == before | d_ids  # chunk a.md cũ còn, d.md được ghi
    assert graph.saved[-1] == sorted(d_ids)  # chunk a.md mới (lỗi embed) KHÔNG được ghi


def test_failed_reset_keeps_graph_and_next_ingest_rewrites_nothing(env, graph):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    before_graph = dict(graph.chunks)
    embedded()

    state["status"] = 503
    result = ingest.ingest(project=PROJECT, reset=True)
    assert result["failed_batches"] == 1
    assert graph.deleted_projects == []  # Qdrant giữ index cũ → graph cũng giữ nguyên
    assert graph.chunks == before_graph
    assert len(index()) == 3
    assert len(embedded()) == 3  # reset đã thử embed (và lỗi)

    state["status"] = 200
    result = ingest.ingest(project=PROJECT)

    # Không có gì mới ở cả Qdrant lẫn Neo4j → 0 embed, 0 chunk graph ghi lại.
    assert embedded() == []
    assert set(graph.chunks) == set(index())
    assert graph.keep_calls[-1] == set(index())
    assert result["graph_chunks_saved"] == 0 and result["stale_removed"] == 0


def test_partial_reset_rebuilds_graph_from_embedded_chunks_only(env, graph, monkeypatch):
    proj, store, state, write, embedded, index = env
    _seed(write)
    ingest.ingest(project=PROJECT)
    monkeypatch.setattr(ingest, "UPSERT_BATCH_SIZE", 1)
    state["fail"] = "alpha"  # batch của a.md lỗi, b.md/c.md thành công

    result = ingest.ingest(project=PROJECT, reset=True)

    assert result["failed_batches"] == 1
    assert graph.deleted_projects == [PROJECT]  # Qdrant đã dựng lại → graph dựng lại theo
    assert sorted(index().values()) == ["b.md", "c.md"]
    assert set(graph.chunks) == set(index())  # graph khớp đúng các chunk đã có vector
