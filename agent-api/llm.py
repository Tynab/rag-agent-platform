"""
llm.py — Factory tạo ChatOllama dùng chung cho agent-api (timeout thật + trần token)
=====================================================================================

Mô tả
-----
Mọi lần gọi LLM trong workflow.py đi qua make_chat() để đảm bảo cùng một cấu hình:

- Timeout HTTP thật cho client Ollama. ChatOllama (langchain-ollama 0.3.x) KHÔNG có
  field `request_timeout`: kwarg đó bị pydantic bỏ qua im lặng (model_config
  extra="ignore") → httpx client chạy với timeout=None (vô hạn). Timeout phải truyền
  qua `client_kwargs={"timeout": ...}` — được chuyển thẳng vào httpx.Client/AsyncClient
  bên trong ollama.Client.
- Trần token sinh ra (`num_predict`) theo loại lời gọi: planner JSON, sinh từng file,
  báo cáo Markdown một lần. Không có trần, model nhỏ bị lặp sẽ sinh tới khi hết timeout
  và chiếm slot Ollama duy nhất (OLLAMA_MAX_LOADED_MODELS=1) suốt thời gian đó.
- Bật/tắt thinking khi sinh file (FILEGEN_THINK).

Ngữ nghĩa timeout
-----------------
ChatOllama mặc định gọi Ollama ở chế độ stream. Timeout của httpx áp cho từng thao tác
mạng (kết nối, chờ đọc từng chunk) chứ KHÔNG phải tổng thời gian của cả lời gọi:
OLLAMA_REQUEST_TIMEOUT giới hạn thời gian tới token đầu tiên (gồm cả thời gian load /
swap model) và khoảng lặng giữa hai chunk. Một model đang lặp vô tận vẫn đều đặn trả
token nên không bao giờ chạm timeout — vì vậy cần thêm trần num_predict.

Biến môi trường (đều đã truyền qua khối environment: của agent-api)
-------------------------------------------------------------------
    OLLAMA_BASE_URL         URL Ollama mặc định khi caller không truyền base_url.
    OLLAMA_REQUEST_TIMEOUT  Timeout (giây) httpx; <= 0 = không giới hạn.
    PLANNER_NUM_PREDICT     Trần token planner JSON danh sách file (mặc định 2048).
    FILEGEN_NUM_PREDICT     Trần token mỗi lần sinh một file code (mặc định 8192).
    REPORT_NUM_PREDICT      Trần token báo cáo Markdown một lần (mặc định 16384).
                            Với ba biến *_NUM_PREDICT: 0 hoặc âm = không đặt trần.
    FILEGEN_THINK           "true"/"false" (không phân biệt hoa thường) → bật/tắt
                            thinking cho lời gọi sinh file; rỗng/không đặt = để model
                            tự quyết (không truyền `reasoning`).

Hàm xuất khẩu
-------------
    make_chat(model, ...)                 → ChatOllama có timeout thật
    num_predict_for(kind, model, ...)     → int | None (trần token đã tính headroom)
    filegen_reasoning()                   → bool | None (giá trị FILEGEN_THINK)
    is_reasoning_model(model)             → bool
    hit_token_cap(response)               → bool (Ollama dừng vì chạm num_predict)

Các hàm đọc hằng module-level LÚC GỌI (không bind lúc import) để test chỉ cần
monkeypatch module này.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from langchain_ollama import ChatOllama

logger = logging.getLogger("agent-llm")


def _env_int(name: str, default: int) -> int:
    """Đọc biến môi trường số nguyên; không đặt hoặc chuỗi rỗng → *default*.

    Giá trị không phải số nguyên → raise ValueError ngay lúc import (fail fast, giống
    idiom int(os.environ.get(...)) trong workflow.py) thay vì âm thầm dùng default.
    """
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _parse_think(raw: str | None) -> bool | None:
    """Chuẩn hoá FILEGEN_THINK: "true"/"false" (không phân biệt hoa thường) → True/False.

    None / "" → None (không truyền `reasoning`, model tự quyết). Docker Compose truyền
    `${FILEGEN_THINK:-}` nên mặc định biến TỒN TẠI với giá trị rỗng — phải xử lý y như
    chưa đặt. Giá trị khác → cảnh báo và coi như chưa đặt.
    """
    value = (raw or "").strip().lower()
    if not value:
        return None
    if value in ("true", "false"):
        return value == "true"
    logger.warning("FILEGEN_THINK=%r không hợp lệ (chỉ nhận true/false) — bỏ qua.", raw)
    return None


OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
# Timeout (giây) httpx cho mỗi lời gọi Ollama — xem "Ngữ nghĩa timeout" ở docstring module.
# Nên đặt cao hơn RAG_TIMEOUT vì LLM generation (kèm swap model) tốn thời gian hơn embed.
OLLAMA_REQUEST_TIMEOUT: int = _env_int("OLLAMA_REQUEST_TIMEOUT", 1200)

PLANNER_NUM_PREDICT: int = _env_int("PLANNER_NUM_PREDICT", 2048)
FILEGEN_NUM_PREDICT: int = _env_int("FILEGEN_NUM_PREDICT", 8192)
REPORT_NUM_PREDICT: int = _env_int("REPORT_NUM_PREDICT", 16384)
FILEGEN_THINK: bool | None = _parse_think(os.environ.get("FILEGEN_THINK"))

# Frozenset các chuỗi con khớp tên model reasoning (model sinh chain-of-thought
# <think>...</think> trước câu trả lời). Dùng để: (1) workflow._strip_thinking() bóc
# phần suy luận khỏi output; (2) cộng thêm headroom num_predict cho phần suy luận.
REASONING_MODELS: frozenset[str] = frozenset({
    "phi4-mini-reasoning",
    "phi4-reasoning",
    "qwq",
    "deepseek-r1",
    "qwen3",
    "north-mini-code",
    "gemma4",
    "mistral-medium-3.5",
})

# Hệ số headroom cho lời gọi CÓ thinking. Ollama tính token suy luận (<think> hoặc
# trường thinking khi reasoning=True) vào CÙNG ngân sách num_predict với câu trả lời.
# Phần suy luận của reasoning model thường dài ngang hoặc hơn phần trả lời (báo cáo
# BA/SA vài nghìn token có thể kèm chừng ấy token suy luận) → trần gốc được tính cho
# PHẦN TRẢ LỜI và nhân đôi khi model sẽ suy luận, để mặc định không cắt cụt output
# bình thường mà vẫn chặn được model lặp vô tận. Kết quả kẹp ở num_ctx: trần lớn hơn
# context window không có ý nghĩa thực tế.
_THINKING_HEADROOM = 2

CallKind = Literal["planner", "filegen", "report"]


def is_reasoning_model(model: str) -> bool:
    """True nếu tên *model* khớp một chuỗi con trong REASONING_MODELS."""
    name = model.lower()
    return any(m in name for m in REASONING_MODELS)


def filegen_reasoning() -> bool | None:
    """Giá trị `reasoning` cho lời gọi sinh file (FILEGEN_THINK); None = không truyền."""
    return FILEGEN_THINK


def _base_num_predict(kind: CallKind) -> int:
    if kind == "planner":
        return PLANNER_NUM_PREDICT
    if kind == "filegen":
        return FILEGEN_NUM_PREDICT
    return REPORT_NUM_PREDICT


def num_predict_for(
    kind: CallKind,
    model: str,
    *,
    num_ctx: int | None = None,
    reasoning: bool | None = None,
) -> int | None:
    """Trần num_predict cho một lời gọi loại *kind* với *model*.

    - Trần gốc lấy từ PLANNER_/FILEGEN_/REPORT_NUM_PREDICT; <= 0 → None (không trần).
    - Lời gọi sẽ suy luận — reasoning=True, hoặc reasoning=None với reasoning model —
      được nhân _THINKING_HEADROOM. reasoning=False tắt thinking → không cần headroom.
    - Kẹp tối đa *num_ctx* nếu có.
    """
    base = _base_num_predict(kind)
    if base <= 0:
        return None
    thinking = reasoning if reasoning is not None else is_reasoning_model(model)
    cap = base * _THINKING_HEADROOM if thinking else base
    if num_ctx and num_ctx > 0:
        cap = min(cap, num_ctx)
    return cap


def _client_timeout() -> float | None:
    """Timeout httpx từ OLLAMA_REQUEST_TIMEOUT; <= 0 → None (không giới hạn)."""
    return float(OLLAMA_REQUEST_TIMEOUT) if OLLAMA_REQUEST_TIMEOUT > 0 else None


def make_chat(
    model: str,
    *,
    base_url: str | None = None,
    num_ctx: int | None = None,
    temperature: float = 0.1,
    num_predict: int | None = None,
    reasoning: bool | None = None,
) -> ChatOllama:
    """Tạo ChatOllama với timeout httpx thật (client_kwargs) và tuỳ chọn sinh.

    *base_url* None → OLLAMA_BASE_URL. *num_ctx*/*num_predict* None → không truyền
    (dùng mặc định Ollama). *reasoning* None → không truyền (model tự quyết thinking).
    `client_kwargs` là dict MỚI mỗi lần: ChatOllama._set_clients ghi thêm auth header
    vào dict này, dùng chung giữa các instance sẽ rò header từ base_url này sang khác.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url or OLLAMA_BASE_URL,
        "temperature": temperature,
        "client_kwargs": {"timeout": _client_timeout()},
    }
    if num_ctx is not None:
        kwargs["num_ctx"] = num_ctx
    if num_predict is not None:
        kwargs["num_predict"] = num_predict
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    return ChatOllama(**kwargs)


def hit_token_cap(response: object) -> bool:
    """True nếu Ollama dừng sinh vì chạm num_predict (done_reason == "length").

    Dùng getattr vì test double (SimpleNamespace) có thể không có response_metadata.
    """
    meta = getattr(response, "response_metadata", None) or {}
    return isinstance(meta, dict) and meta.get("done_reason") == "length"
