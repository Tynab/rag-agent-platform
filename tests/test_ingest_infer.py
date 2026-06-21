"""Unit test cho rag-api/ingest.py — suy luận metadata + point id (T2).

Cần dependency của rag-api (langchain, qdrant-client...). Nếu chưa cài, test được
SKIP tự động qua importorskip để suite vẫn xanh ở môi trường tối thiểu.
"""

import pytest

ingest = pytest.importorskip("ingest")


def test_infer_module_flat_vs_nested():
    assert ingest._infer_module("file.md", "crm") == "crm"          # phẳng → project
    assert ingest._infer_module("auth/file.md", "crm") == "auth"    # 1 cấp → thư mục con
    assert ingest._infer_module("auth/sub/file.md", "crm") == "auth"  # sâu → vẫn cấp một


def test_infer_doc_type_underscore_filenames():
    # R5: tên file underscore/prose phải nhận diện đúng (trước đây toàn 'document').
    assert ingest._infer_doc_type("08_API_Specification.md", "crm/08_API_Specification.md") == "api"
    assert ingest._infer_doc_type("04_SRS.md", "crm/04_SRS.md") == "srs"
    assert ingest._infer_doc_type("07_Data_Model_ERD.md", "x") == "data-model"
    assert ingest._infer_doc_type("05_RBAC_Permissions_Matrix.md", "x") == "rbac"
    assert ingest._infer_doc_type("02_PRD.md", "x") == "prd"
    assert ingest._infer_doc_type("16_Glossary.md", "x") == "glossary"
    assert ingest._infer_doc_type("random_notes.md", "x") == "document"


def test_infer_chunk_type():
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    assert ingest._infer_chunk_type(table) == "table"
    assert ingest._infer_chunk_type("Just a normal paragraph of prose.") == "paragraph"


def test_collection_name_double_underscore():
    assert ingest.get_collection_name("crm") == ingest.COLLECTION_NAME + "__crm"


def test_make_point_id_deterministic_and_sensitive():
    from langchain_core.documents import Document

    md = {
        "relative_path": "crm/a.md",
        "loader_index": 0,
        "chunk_index": 1,
        "content_hash": "abc",
        "embedding_model": "m",
    }
    d1 = Document(page_content="x", metadata=dict(md))
    d2 = Document(page_content="x", metadata=dict(md))
    assert ingest.make_point_id(d1) == ingest.make_point_id(d2)  # idempotent

    d3 = Document(page_content="x", metadata={**md, "chunk_index": 2})
    assert ingest.make_point_id(d1) != ingest.make_point_id(d3)  # đổi index → đổi id

    d4 = Document(page_content="x", metadata={**md, "embedding_model": "other"})
    assert ingest.make_point_id(d1) != ingest.make_point_id(d4)  # đổi model → đổi id
