"""
app.py — API điều phối SDLC Agent của rag-agent-platform (cổng 8091)
====================================================================

Mô tả
-----
FastAPI application điều phối toàn bộ quy trình SDLC 15 bước thông qua LangGraph.
Mỗi bước là một AI agent độc lập với model, system prompt và dependency riêng biệt.
Workflow chạy bất đồng bộ (BackgroundTasks) và cập nhật trạng thái real-time.

Danh sách endpoint
------------------
GET  /health
    Trả về trạng thái hoạt động của service, URL Ollama, URL RAG và số lượng agent.
    Dùng bởi Docker healthcheck và giám sát bên ngoài.

GET  /agents
    Liệt kê cấu hình đầy đủ của 15 agent: step_id, name, model, depends_on.

GET  /ui
    Phục vụ giao diện web SDLC Workflow (file static/workflow.html).

POST /agent/{role}
    Chạy đồng bộ một agent đơn lẻ theo role chỉ định.
    Nhận AgentStepRequest (user_input, project, prev_outputs, tech_stack, rag_enabled, rag_top_k).
    Trả về AgentStepResponse (role, name, model, output).
    Hữu ích để test từng agent riêng lẻ hoặc ghép pipeline thủ công.

POST /workflow/run
    Xếp hàng workflow SDLC 15 bước chạy nền, trả về workflow_id ngay lập tức.
    Nhận WorkflowRunRequest (user_input, project, rag_enabled, rag_top_k, tech_stack).
    Trạng thái chuyển đổi: pending → running → completed | failed.

GET  /workflow/{workflow_id}
    Kiểm tra trạng thái hoặc lấy toàn bộ kết quả workflow đã hoàn thành.
    Trả về WorkflowRecord bao gồm step_outputs, completed_steps, artifacts, error.

GET  /workflows
    Liệt kê tối đa 50 workflow gần nhất, sắp xếp theo thời gian tạo mới nhất trước.

GET  /workflow/{workflow_id}/artifacts
    Liệt kê metadata của tất cả file artifact đã được trích xuất và lưu vào disk.
    Không trả về nội dung file — dùng endpoint bên dưới để đọc từng file.

GET  /workflow/{workflow_id}/artifacts/{role}/{path}
    Đọc nội dung một file artifact cụ thể dưới dạng text.
    Thêm query param ?download=1 để tải xuống dưới dạng binary.

Vòng đời workflow
-----------------
    POST /workflow/run
        → WorkflowRecord tạo với status=pending, lưu vào workflow_store
        → BackgroundTasks.add_task(_run_workflow_task)
            → status=running, LangGraph StateGraph.stream() bắt đầu
            → Mỗi node hoàn thành: cập nhật step_outputs, completed_steps, artifacts
            → Sau node cuối cùng (clarifier): chạy Clarifier Regen Loop nếu được bật
            → status=completed (hoặc failed nếu có lỗi không xử lý được)

Clarifier Regen Loop
--------------------
Sau khi workflow hoàn tất thành công, _run_workflow_task phân tích §10 của
Clarifier output để tìm danh sách agent cần re-generate. Với mỗi vòng lặp
(tối đa CLARIFIER_REGEN_LOOPS lần):
    1. Re-run từng agent trong danh sách theo thứ tự WORKFLOW_STEPS
    2. Re-run Clarifier với outputs đã cập nhật
    3. Kiểm tra lại §10 — nếu rỗng thì dừng sớm

Concurrency và thread-safety
-----------------------------
- workflow_store là dict in-memory, được bảo vệ bởi _store_lock (threading.Lock)
  cho mọi thao tác đọc/ghi để tránh race condition.
- get_workflow() dùng double-checked locking: kiểm tra trước lock, kiểm tra lại
  sau lock, đảm bảo LangGraph StateGraph chỉ được compile đúng một lần dù
  nhiều request đến đồng thời lúc khởi động.
- _run_workflow_task chạy trong thread-pool của FastAPI BackgroundTasks.
  WorkflowRecord được cập nhật trực tiếp (pass-by-reference) — không cần re-insert.
- Artifact extraction chạy ngay sau mỗi node hoàn thành (non-fatal: exception
  trong extraction chỉ ghi warning log, không dừng workflow).

Bộ nhớ và giới hạn
------------------
- Tối đa _MAX_STORED_WORKFLOWS (50) workflow trong bộ nhớ. Entry cũ nhất bị xóa
  khi đạt giới hạn.
- user_input được sanitize tại boundary: loại bỏ ký tự điều khiển, cắt tại
  _MAX_INPUT_CHARS (10.000) ký tự.
- Mỗi workflow run được ghi vào data/memory/episodic/workflow_runs.jsonl để
  làm seed dữ liệu lịch sử (episodic memory).
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from artifacts import ARTIFACT_ROLES as _ARTIFACT_ROLES, extract_and_save as _extract_artifacts, list_artifacts as _list_artifacts, read_artifact as _read_artifact
from agents import AGENTS, WORKFLOW_STEPS
from workflow import (
    OLLAMA_BASE_URL,
    RAG_API_URL,
    RAG_TOP_K,
    SDLCState,
    get_workflow,
    run_single_step,
    _parse_clarifier_regen_list,
)


def _require_env(name: str) -> str:
    """Trả về giá trị biến môi trường *name*.

    Raise RuntimeError nếu biến không tồn tại — đảm bảo service không khởi động
    thiếu cấu hình bắt buộc. Các biến tùy chọn nên dùng os.environ.get() trực tiếp.
    """
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set")
    return value


LOG_LEVEL = _require_env("LOG_LEVEL")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("agent-api")

app = FastAPI(title="rag-agent-platform - Bộ điều phối SDLC Agent", version="1.0.0")

_STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
def _validate_models_on_startup() -> None:
    """A4 — cảnh báo (KHÔNG chặn khởi động) nếu model per-role chưa được pull về Ollama.

    Best-effort: nếu Ollama chưa sẵn sàng (depends_on không gate health) thì bỏ qua.
    Mục tiêu là phát hiện sớm model thiếu/sai tên thay vì để node fail giữa workflow —
    bù lại sự bất nhất hiện tại (LOG_LEVEL fail-fast nhưng model thì im lặng fallback).
    """
    import requests  # local import — chỉ cần lúc startup
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        available = {m.get("name", "") for m in resp.json().get("models", [])}
    except Exception as exc:
        logger.warning("A4: bỏ qua kiểm tra model Ollama lúc startup (non-fatal): %s", exc)
        return

    def _present(m: str) -> bool:
        return m in available or f"{m}:latest" in available

    wanted = {cfg.model for cfg in AGENTS.values() if cfg.model}
    missing = sorted(m for m in wanted if not _present(m))
    if missing:
        logger.warning(
            "A4: %d/%d model chưa có trên Ollama — node dùng chúng sẽ fail lúc runtime: %s",
            len(missing), len(wanted), ", ".join(missing),
        )
    else:
        logger.info("A4: tất cả %d model per-role đã sẵn sàng trên Ollama.", len(wanted))


# ── Models Pydantic cho Request và Response của các endpoint ────────────────────────────────────────

class AgentStepRequest(BaseModel):
    user_input: str = Field(..., min_length=1,
                            description="Mục tiêu kinh doanh hoặc context đầu vào cho agent")
    project: str | None = Field(
        None, description="RAG project filter (ví dụ: 'yanlib')")
    extra_context: str | None = Field(
        None, description="Context bổ sung để chèn vào")
    prev_outputs: dict[str, str] | None = Field(
        None, description="Output của các bước trước để chèn vào context (role → text)"
    )
    tech_stack: list[str] | None = Field(
        None, description="Danh sách tech stack bắt buộc. Ví dụ: ['nestjs', 'reactjs', 'mongodb']")
    rag_enabled: bool = Field(
        True, description="Có query RAG knowledge base không")
    rag_top_k: int = Field(RAG_TOP_K, ge=1, le=20)


class AgentStepResponse(BaseModel):
    role: str
    name: str
    model: str
    output: str


class WorkflowRunRequest(BaseModel):
    user_input: str = Field(..., min_length=1,
                            description="Mục tiêu kinh doanh / ý tưởng project để chạy qua SDLC")
    project: str | None = Field(
        None, description="RAG project filter (ví dụ: 'yanlib')")
    rag_enabled: bool = Field(True)
    rag_top_k: int = Field(RAG_TOP_K, ge=1, le=20)
    tech_stack: list[str] | None = Field(
        None,
        description=(
            "Danh sách tech stack bắt buộc (ngôn ngữ, framework, DB, infra). "
            "Ví dụ: ['nestjs', 'reactjs', 'mongodb', 'k8s', 'react native']. "
            "TA Agent sẽ thiết kế kiến trúc dựa trên stack này; các agent khác "
            "cũng sẽ bám sát đúng các công nghệ này."
        ),
    )


class WorkflowStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class WorkflowRecord(BaseModel):
    workflow_id: str
    status: WorkflowStatus
    project: str | None
    user_input: str
    tech_stack: list[str] | None = None
    step_outputs: dict[str, str] = Field(default_factory=dict)
    completed_steps: list[str] = Field(default_factory=list)
    # W6: các bước chạy xong nhưng trả về marker [LỖI ...]. Workflow vẫn 'completed'
    # (stream hoàn tất); đây là nơi phơi bày lỗi từng node thay vì flip cả status.
    failed_steps: list[str] = Field(default_factory=list)
    error: str | None = None
    artifacts: dict[str, list] = Field(default_factory=dict)
    created_at: str
    completed_at: str | None = None


# ── In-memory workflow store — đồng bộ hóa bằng _store_lock ───────────────────
# workflow_store lưu kết quả của tất cả workflow trong RAM. Giới hạn
# _MAX_STORED_WORKFLOWS entry — entry cũ nhất bị xóa khi đạt giới hạn.
# Môi trường production cần Redis hoặc database để persistence.

workflow_store: dict[str, WorkflowRecord] = {}
_store_lock = threading.Lock()

_MAX_STORED_WORKFLOWS = 50
# Số vòng lặp Clarifier re-generation tối đa. 0 = tắt toàn bộ tính năng.
# Tăng lên 2 nếu muốn nhiều vòng tinh chỉnh hơn (tốn thêm thời gian).
_CLARIFIER_REGEN_LOOPS: int = int(os.environ.get("CLARIFIER_REGEN_LOOPS", "1"))
# W1: ngân sách thời gian tổng cho một workflow (giây). 0 = không giới hạn.
# Được kiểm tra giữa các node — node đang chạy không bị hủy giữa chừng, nhưng
# stream dừng sớm sau khi node hiện tại trả về nếu đã vượt ngân sách.
_WORKFLOW_MAX_SECONDS: int = int(os.environ.get("WORKFLOW_MAX_SECONDS", "0"))

# ── Đường dẫn bộ nhớ episodic và giới hạn input ────────────────────────────

MEMORY_DIR: str = os.environ.get("MEMORY_DIR", "/data/memory")
_MAX_INPUT_CHARS: int = 10_000  # Giới hạn 10.000 ký tự để phòng chống context overflow và DoS.


def _sanitize_input(text: str) -> str:
    """Sanitize input người dùng tại system boundary trước khi đưa vào LLM context.

    Hai bước xử lý:
    1. Loại bỏ ký tự điều khiển (control characters) ngoại trừ newline, carriage return
       và tab — ngăn chặn injection qua ANSI escape hay null bytes.
    2. Cắt ngắn tại _MAX_INPUT_CHARS (10.000) nếu input vượt quá — tránh overflow
       context window của model và chống DoS qua input khổng lồ.
    """
    sanitized = "".join(ch for ch in text if ch >= " " or ch in "\n\r\t")
    if len(sanitized) > _MAX_INPUT_CHARS:
        logger.warning(
            "Input truncated from %d to %d chars", len(sanitized), _MAX_INPUT_CHARS
        )
        sanitized = sanitized[:_MAX_INPUT_CHARS]
    return sanitized.strip()


def _log_workflow_run(
    workflow_id: str,
    project: str | None,
    user_input: str,
    completed_steps: list[str],
    status: str,
    error: str | None,
    duration_seconds: float,
) -> None:
    """Ghi thông tin workflow run ra file JSONL để làm dữ liệu lịch sử (episodic memory).

    Mỗi dòng JSONL lưu: workflow_id, project, user_input (500 ký tự đầu),
    danh sách bước đã chạy, số bước, trạng thái, lỗi, thời gian chạy và timestamp.
    Non-fatal: lỗi ghi file chỉ ghi warning log, không làm gán đoạn workflow.
    """
    try:
        log_dir = Path(MEMORY_DIR) / "episodic"
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "workflow_id": workflow_id,
            "project": project,
            "user_input": user_input[:500],
            "completed_steps": completed_steps,
            "steps_count": len(completed_steps),
            "status": status,
            "error": error,
            "duration_seconds": round(duration_seconds, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log_file = log_dir / "workflow_runs.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug("Workflow run logged: %s", workflow_id)
    except Exception as exc:
        logger.warning("Workflow run logging failed (non-fatal): %s", exc)


def _store_workflow(record: WorkflowRecord) -> None:
    """Lưu *record* vào workflow_store, tự động xóa entry cũ nhất khi store đạt giới hạn.

    Thread-safe: dùng _store_lock để đồng bộ hóa ghi vào dict in-memory.
    """
    with _store_lock:
        if len(workflow_store) >= _MAX_STORED_WORKFLOWS:
            # C4: chỉ evict entry đã KẾT THÚC (completed/failed) và cũ nhất.
            # Không bao giờ xóa workflow đang pending/running để tránh mất kết quả
            # của một run đang dở (dict giữ thứ tự chèn → phần tử đầu là cũ nhất).
            evictable = [
                k for k, v in workflow_store.items()
                if v.status in (WorkflowStatus.completed, WorkflowStatus.failed)
            ]
            if evictable:
                del workflow_store[evictable[0]]
            else:
                logger.warning(
                    "workflow_store đạt giới hạn (%d) nhưng mọi entry đang chạy — "
                    "tạm thời vượt giới hạn thay vì drop một run đang dở.",
                    _MAX_STORED_WORKFLOWS,
                )
        workflow_store[record.workflow_id] = record


def _reconstruct_from_disk(workflow_id: str) -> WorkflowRecord | None:
    """W2 — dựng lại WorkflowRecord từ disk khi record đã bị evict khỏi store in-memory.

    Nguồn dữ liệu bền vững: episodic JSONL (metadata: status, project, completed_steps,
    error, timestamp) + artifacts trên disk (_output.md của từng role → step_outputs).
    Không khôi phục được output của role không sinh artifact (planning roles). Trả về
    None nếu không có cả episodic entry lẫn artifact nào trên disk.
    """
    disk_arts = _list_artifacts(workflow_id)
    meta: dict | None = None
    try:
        log_file = Path(MEMORY_DIR) / "episodic" / "workflow_runs.jsonl"
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if entry.get("workflow_id") == workflow_id:
                        meta = entry  # giữ dòng cuối cùng (mới nhất) khớp id
    except Exception as exc:
        logger.warning("W2: đọc episodic log thất bại (non-fatal): %s", exc)

    if not disk_arts and not meta:
        return None

    # Khôi phục step_outputs từ _output.md của từng role có artifact trên disk.
    step_outputs: dict[str, str] = {}
    for role in disk_arts:
        res = _read_artifact(workflow_id, role, "_output.md")
        if res:
            step_outputs[role] = res[0]

    meta = meta or {}
    try:
        status = WorkflowStatus(meta.get("status", "completed"))
    except ValueError:
        status = WorkflowStatus.completed
    now_iso = datetime.now(timezone.utc).isoformat()
    return WorkflowRecord(
        workflow_id=workflow_id,
        status=status,
        project=meta.get("project"),
        user_input=meta.get("user_input", ""),
        step_outputs=step_outputs,
        completed_steps=meta.get("completed_steps", list(disk_arts.keys())),
        error=meta.get("error"),
        artifacts=disk_arts,
        created_at=meta.get("timestamp", now_iso),
        completed_at=meta.get("timestamp"),
    )


# ── Task nền chạy toàn bộ SDLC workflow ────────────────────────────────────────────────────

def _run_workflow_task(workflow_id: str, req: WorkflowRunRequest) -> None:
    """Task nền thực thi toàn bộ SDLC LangGraph workflow và cập nhật WorkflowRecord.

    Được gọi bởi FastAPI BackgroundTasks — chạy trong thread-pool thread riêng biệt.

    Thread-safety:
    - Chỉ một background task tồn tại cho mỗi workflow_id.
    - WorkflowRecord được cập nhật trực tiếp (pass-by-reference) vì dict
      là kiểu tham chiếu — không cần re-insert vào store.
    - Các endpoint đọc (GET /workflow/{id}) chỉ đọc các trường được ghi nguyên tử.
    """
    with _store_lock:
        record = workflow_store.get(workflow_id)
    if record is None:
        return

    record.status = WorkflowStatus.running
    logger.info("Workflow %s started", workflow_id)

    # user_input đã được sanitize lúc enqueue và lưu vào record — tái sử dụng trực tiếp
    # thay vì sanitize lại từ req.user_input để đảm bảo nhất quán.
    sanitized_input = record.user_input

    initial_state: SDLCState = {
        "project": req.project,
        "user_input": sanitized_input,
        "tech_stack": req.tech_stack,
        "rag_enabled": req.rag_enabled,
        "rag_top_k": req.rag_top_k,
        "rag_api_url": RAG_API_URL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "step_outputs": {},
        "completed_steps": [],
        "error": None,
    }

    _start = time.monotonic()
    try:
        final_state: SDLCState = {}  # type: ignore[assignment]
        _node_errors: list[str] = []
        _timed_out = False
        for chunk in get_workflow().stream(initial_state, stream_mode="updates"):
            # chunk = {ten_node: partial_state_dict} với stream_mode="updates".
            # Mỗi chunk cập nhật được ngay lập tức vào record để GET /workflow/{id}
            # phản ánh tiến trình real-time.
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    # Cập nhật real-time để GET /workflow/{id} phản ánh tiến trình
                    if "step_outputs" in node_output:
                        record.step_outputs.update(node_output["step_outputs"])
                        for _role, _out in node_output["step_outputs"].items():
                            # W6: node trả về marker [LỖI ...] → ghi nhận failed_steps,
                            # KHÔNG flip cả workflow sang 'failed' và vẫn cho regen chạy.
                            if _out and _out.startswith("[LỖI"):
                                if _role not in record.failed_steps:
                                    record.failed_steps.append(_role)
                            # Trích xuất artifact ngay sau mỗi coding node thành công.
                            # Non-fatal: lỗi extraction chỉ ghi log, không dừng workflow.
                            elif _role in _ARTIFACT_ROLES and _out:
                                _arts = _extract_artifacts(_role, _out, workflow_id)
                                if _arts:
                                    record.artifacts[_role] = _arts
                    if "completed_steps" in node_output:
                        record.completed_steps = list(
                            dict.fromkeys(
                                record.completed_steps + node_output["completed_steps"]
                            )
                        )
                    if node_output.get("error"):
                        _node_errors.append(str(node_output["error"]))
                    final_state.update(node_output)
            # W1: kiểm tra ngân sách thời gian SAU mỗi node (không mất output node vừa xong).
            if _WORKFLOW_MAX_SECONDS and (time.monotonic() - _start) > _WORKFLOW_MAX_SECONDS:
                _timed_out = True
                logger.warning(
                    "Workflow %s vượt ngân sách %ds — dừng sớm sau %d node.",
                    workflow_id, _WORKFLOW_MAX_SECONDS, len(record.completed_steps),
                )
                break
        # Stream hoàn tất (hoặc dừng do timeout) ⇒ 'completed'. Lỗi từng node nằm trong
        # failed_steps + record.error (tóm tắt); chỉ outer-except mới là 'failed' thật sự.
        if _node_errors:
            record.error = "; ".join(_node_errors[:5])
        if _timed_out:
            record.error = (record.error + " | " if record.error else "") + \
                f"[Timeout: vượt {_WORKFLOW_MAX_SECONDS}s, dừng sớm]"
        record.status = WorkflowStatus.completed

        # ── Clarifier Regen Loop ───────────────────────────────────────────────────────────────────
        # Chỉ chạy khi workflow hoàn thành thành công và CLARIFIER_REGEN_LOOPS > 0.
        # Clarifier phân tích §10 để lấy danh sách agent cần re-gen, re-run từng agent,
        # sau đó re-run Clarifier để đánh giá lại — lặp tối đa CLARIFIER_REGEN_LOOPS lần.
        if record.status == WorkflowStatus.completed and not _timed_out and _CLARIFIER_REGEN_LOOPS > 0:
            _live_outputs: dict[str, str] = dict(record.step_outputs)
            for _loop_idx in range(_CLARIFIER_REGEN_LOOPS):
                regen_roles = _parse_clarifier_regen_list(
                    _live_outputs.get("clarifier", "")
                )
                if not regen_roles:
                    logger.info(
                        "Clarifier regen loop %d/%d: không có agent nào cần re-gen — dừng.",
                        _loop_idx + 1, _CLARIFIER_REGEN_LOOPS,
                    )
                    break

                logger.info(
                    "Clarifier regen loop %d/%d: re-generating roles=%s",
                    _loop_idx + 1, _CLARIFIER_REGEN_LOOPS, regen_roles,
                )
                record.status = WorkflowStatus.running

                # Re-run từng agent được đề xuất theo thứ tự WORKFLOW_STEPS (dependency order).
                for _role in regen_roles:
                    try:
                        _new_output = run_single_step(
                            role=_role,
                            user_input=sanitized_input,
                            project=req.project,
                            prev_outputs=_live_outputs,
                            tech_stack=req.tech_stack,
                            rag_enabled=req.rag_enabled,
                            rag_top_k=req.rag_top_k,
                            ollama_base_url=OLLAMA_BASE_URL,
                            rag_api_url=RAG_API_URL,
                        )
                        _live_outputs[_role] = _new_output
                        record.step_outputs[_role] = _new_output
                        # Re-extract artifact nếu role này có file code — cập nhật artifacts sau re-gen.
                        if (
                            _role in _ARTIFACT_ROLES
                            and _new_output
                            and not _new_output.startswith("[LỖI")
                        ):
                            _arts = _extract_artifacts(_role, _new_output, workflow_id)
                            if _arts:
                                record.artifacts[_role] = _arts
                        logger.info(
                            "Clarifier regen loop %d: %s re-generated (%d chars)",
                            _loop_idx + 1, _role, len(_new_output),
                        )
                    except Exception as _regen_exc:
                        logger.warning(
                            "Clarifier regen loop %d: re-gen '%s' thất bại (non-fatal): %s",
                            _loop_idx + 1, _role, _regen_exc,
                        )

                # Re-run clarifier với outputs đã cập nhật
                try:
                    _new_clarifier = run_single_step(
                        role="clarifier",
                        user_input=sanitized_input,
                        project=req.project,
                        prev_outputs=_live_outputs,
                        tech_stack=req.tech_stack,
                        rag_enabled=req.rag_enabled,
                        rag_top_k=req.rag_top_k,
                        ollama_base_url=OLLAMA_BASE_URL,
                        rag_api_url=RAG_API_URL,
                    )
                    _live_outputs["clarifier"] = _new_clarifier
                    record.step_outputs["clarifier"] = _new_clarifier
                    logger.info(
                        "Clarifier regen loop %d: clarifier re-run xong (%d chars)",
                        _loop_idx + 1, len(_new_clarifier),
                    )
                except Exception as _clarifier_exc:
                    logger.warning(
                        "Clarifier regen loop %d: clarifier re-run thất bại: %s — dừng loop.",
                        _loop_idx + 1, _clarifier_exc,
                    )
                    break

            # W6: lỗi từng node không làm workflow 'failed' — giữ 'completed' sau regen.
            record.status = WorkflowStatus.completed
        # ── Kết thúc Clarifier Regen Loop ─────────────────────────────────────────

    except Exception as exc:
        logger.exception("Workflow %s thất bại", workflow_id)
        record.status = WorkflowStatus.failed
        record.error = str(exc)
    finally:
        record.completed_at = datetime.now(timezone.utc).isoformat()
        _duration = time.monotonic() - _start
        logger.info("Workflow %s hoàn thành với status=%s",
                    workflow_id, record.status)
        _log_workflow_run(
            workflow_id=workflow_id,
            project=req.project,
            user_input=sanitized_input,
            completed_steps=record.completed_steps,
            status=record.status.value,
            error=record.error,
            duration_seconds=_duration,
        )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    """Trả về trạng thái service và cấu hình runtime. Dùng bởi Docker healthcheck."""
    return {
        "status": "ok",
        "ollama_base_url": OLLAMA_BASE_URL,
        "rag_api_url": RAG_API_URL,
        "agents": len(AGENTS),
        "workflow_steps": WORKFLOW_STEPS,
    }


@app.get("/agents")
def list_agents() -> dict[str, Any]:
    """Liệt kê cấu hình tất cả agent: step_id, name, model và chuỗi phụ thuộc."""
    return {
        role: {
            "step_id": cfg.step_id,
            "name": cfg.name,
            "model": cfg.model,
            "depends_on": cfg.depends_on,
        }
        for role, cfg in AGENTS.items()
    }


@app.post("/agent/{role}", response_model=AgentStepResponse)
def run_agent_step(role: str, req: AgentStepRequest) -> AgentStepResponse:
    """
    Chạy đồng bộ một bước agent đơn lẻ.

    Hữu ích để test từng agent riêng lẻ hoặc ghép nối bước thủ công
    khi không cần chạy toàn bộ workflow tuần tự.
    Truyền *prev_outputs* để chèn context từ các bước trước.
    """
    if role not in AGENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Role '{role}' không tồn tại. Hợp lệ: {list(AGENTS.keys())}",
        )

    agent = AGENTS[role]
    logger.info("Bước đơn lẻ: role=%s project=%s", role, req.project)

    try:
        output = run_single_step(
            role=role,
            user_input=_sanitize_input(req.user_input),
            project=req.project,
            extra_context=req.extra_context,
            prev_outputs=req.prev_outputs,
            tech_stack=req.tech_stack,
            rag_enabled=req.rag_enabled,
            rag_top_k=req.rag_top_k,
            ollama_base_url=OLLAMA_BASE_URL,
            rag_api_url=RAG_API_URL,
        )
    except Exception as exc:
        logger.exception("Bước agent thất bại: role=%s", role)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgentStepResponse(
        role=role,
        name=agent.name,
        model=agent.model,
        output=output,
    )


@app.post("/workflow/run")
def start_workflow(req: WorkflowRunRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """
    Gửi workflow SDLC 15 agent để chạy bất đồng bộ.

    Trả về ngay với workflow_id. Poll GET /workflow/{id} để kiểm tra trạng thái.
    Workflow chạy tuần tự: BA → PM → SA → TA → Designer → Team Lead → FE → Mobile → DBA → BE → DA → Tech Lead → Tester → DevSecOps → Clarifier.
    Mỗi bước nhận output đã cắt ngắn của các bước phụ thuộc làm context.
    """
    workflow_id = str(uuid.uuid4())
    # Sanitize trước khi lưu vào record — đảm bảo API response và JSONL log nhất quán.
    sanitized_input = _sanitize_input(req.user_input)
    record = WorkflowRecord(
        workflow_id=workflow_id,
        status=WorkflowStatus.pending,
        project=req.project,
        user_input=sanitized_input,
        tech_stack=req.tech_stack,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _store_workflow(record)
    background_tasks.add_task(_run_workflow_task, workflow_id, req)
    logger.info("Workflow %s đã được xếp hàng", workflow_id)
    return {
        "workflow_id": workflow_id,
        "status": "pending",
        "message": f"Workflow đã được xếp hàng. Poll GET /workflow/{workflow_id} để kiểm tra trạng thái.",
    }


@app.get("/workflow/{workflow_id}", response_model=WorkflowRecord)
def get_workflow_status(workflow_id: str) -> WorkflowRecord:
    """Trả về trạng thái hiện tại của một lần chạy workflow (pending / running / completed / failed)."""
    with _store_lock:
        record = workflow_store.get(workflow_id)
    if record is None:
        # W2: store in-memory bị evict → thử dựng lại từ disk (episodic log + artifacts).
        record = _reconstruct_from_disk(workflow_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' không tìm thấy. Có thể đã bị xóa khỏi store hoặc chưa bắt đầu.",
        )
    return record


@app.get("/workflows")
def list_workflows() -> dict[str, Any]:
    """Liệt kê các workflow gần đây (mới nhất trước, tối đa _MAX_STORED_WORKFLOWS bản ghi)."""
    with _store_lock:
        snapshot = list(workflow_store.values())
    return {
        "count": len(snapshot),
        "workflows": [
            {
                "workflow_id": r.workflow_id,
                "status": r.status,
                "project": r.project,
                "completed_steps": len(r.completed_steps),
                "total_steps": len(WORKFLOW_STEPS),
                "created_at": r.created_at,
                "completed_at": r.completed_at,
            }
            for r in reversed(snapshot)
        ],
    }


@app.get("/workflow/{workflow_id}/artifacts")
def get_workflow_artifacts(workflow_id: str) -> dict[str, Any]:
    """Liệt kê artifacts đã lưu cho workflow (metadata only, không bao gồm nội dung file)."""
    with _store_lock:
        record = workflow_store.get(workflow_id)
    # Merge in-memory artifacts with disk (in case container was restarted/evicted).
    disk_arts = _list_artifacts(workflow_id)
    if record is None and not disk_arts:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' không tìm thấy.",
        )
    # W2: nếu record đã evict nhưng artifact còn trên disk, vẫn trả về được.
    merged = {**disk_arts, **(record.artifacts if record else {})}
    return {"workflow_id": workflow_id, "artifacts": merged}


@app.get("/workflow/{workflow_id}/artifacts/{role}/{path:path}")
def get_artifact_file(
    workflow_id: str,
    role: str,
    path: str,
    download: bool = False,
) -> Any:
    """
    Trả về nội dung file artifact.
    GET ?download=1  → tải về dưới dạng binary.
    GET              → trả về JSON {path, language, content}.
    """
    from fastapi.responses import Response as _Resp

    # Validate role to prevent path traversal — only known artifact roles are valid.
    if role not in _ARTIFACT_ROLES:
        raise HTTPException(status_code=404, detail="File không tìm thấy.")

    result = _read_artifact(workflow_id, role, path)
    if result is None:
        raise HTTPException(status_code=404, detail="File không tìm thấy.")
    content, language = result
    if download:
        # Strip path separators and quote chars to prevent header injection.
        safe_filename = Path(path).name.replace('"', "").replace("\n", "").replace("\r", "")
        return _Resp(
            content=content.encode("utf-8"),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
        )
    return {"path": path, "language": language, "content": content}


@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
def workflow_ui() -> FileResponse:
    """Phục vụ SDLC Workflow UI (single-page application)."""
    return FileResponse(_STATIC_DIR / "workflow.html")
