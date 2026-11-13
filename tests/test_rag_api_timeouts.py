"""Timeout thật cho CHAT_MODEL của POST /ask (rag-api/app.py).

app.py trùng tên module `app` với agent-api → load bằng importlib dưới cùng tên
"rag_api_app" như test_rag_api_endpoints. Kiểm tra timeout trên httpx client THẬT bên
trong ChatOllama — `request_timeout=` bị langchain-ollama 0.3.x bỏ qua im lặng.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
ingest = pytest.importorskip("ingest")
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


def test_ask_chat_model_has_real_httpx_timeout(monkeypatch):
    monkeypatch.setattr(rag_app, "_llm", None)
    monkeypatch.setattr(ingest, "OLLAMA_REQUEST_TIMEOUT", 77.0)
    llm = rag_app.get_chat_model()
    assert llm._client._client.timeout == httpx.Timeout(77.0)
    assert llm.model == rag_app.CHAT_MODEL
    assert llm.temperature == 0.1
    assert rag_app.get_chat_model() is llm  # singleton


def test_health_reports_ollama_request_timeout(monkeypatch):
    monkeypatch.setattr(rag_app, "GRAPH_ENABLED", False)
    body = TestClient(rag_app.app).get("/health").json()
    assert body["ollama_request_timeout"] == rag_app.OLLAMA_REQUEST_TIMEOUT
