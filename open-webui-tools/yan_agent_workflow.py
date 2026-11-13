"""
title: rag-agent-platform - SDLC Agent Workflow
author: rag-agent-platform
description: >
    Tool khởi chạy SDLC Agent Workflow 15 bước toàn diện:
    BA → PM → SA → TA → Designer → Team Lead → FE → Mobile → DBA
    → BE → DA → Tech Lead → Tester → DevSecOps → Clarifier.
    Hỗ trợ chạy full workflow (async + polling) hoặc từng agent
    đơn lẻ (sync). Kết quả bao gồm output markdown, file code
    được trích xuất, và Clarifier audit report.
required_open_webui_version: 0.3.0
requirements: requests
version: 2.1.1

Mô tả chi tiết
--------------
Tool là giao tiếp trực tiếp giữa Open WebUI và Agent API của rag-agent-platform (cổng 8091).
Cho phép người dùng khởi động và theo dõi toàn bộ SDLC pipeline AI
trực tiếp từ giao diện chat mà không cần gọi API thủ công.

Valves (biến cấu hình)
----------------------
    agent_api_url      URL của agent-api trong nội bộ Docker network.
                       Mặc định: http://agent-api:8091 (tên service trong compose).
                       Từ bên trong container Open WebUI, localhost /
                       host.docker.internal KHÔNG tới được agent-api (cổng host
                       mặc định chỉ bind 127.0.0.1). Open WebUI chạy ở
                       container/compose khác → nối vào network của stack:
                           docker network connect rag-agent-platform_yan <container>

    timeout            Timeout (giây) cho mỗi single-step agent call (run_agent_step).
                       Mặc định: 3600 giây — ≥ 2 × OLLAMA_REQUEST_TIMEOUT (1200) của
                       agent-api, vì một bước gọi LLM ít nhất 1–2 lần.

    poll_interval      Khoảng cách (giây) giữa các lần poll trạng thái workflow.
                       Mặc định: 15 giây.

    poll_max_attempts  Số lần poll tối đa trước khi timeout.
                       Mặc định: 120 (120 × 15s = 30 phút).
                       Tăng nếu workflow dùng model lớn, chạy lâu hơn.

    rag_enabled        Có query RAG knowledge base cho mỗi agent step không.
                       Mặc định: true. Tắt khi không có tài liệu đã ingest.

    rag_top_k          Số chunk RAG mỗi agent nhận. Mặc định: 5.

    default_project    Project mặc định để lọc RAG. Null = không lọc.

Hàm công khai
--------------
    run_sdlc_workflow(user_input, project, tech_stack) → str
        Khởi chạy full workflow 15 bước. Mặc định trả workflow_id ngay
        (chạy nền); Valve block_until_complete=True thì poll tới khi
        workflow kết thúc (completed / failed / cancelled) rồi trả summary.
        Tham số:
          user_input   Mô tả yêu cầu / mục tiêu kiếnh doanh. Bắt buộc.
          project      Tên project để lọc RAG. Tùy chọn.
          tech_stack   Chuỗi các công nghệ bắt buộc, phân cách bằng dấu phẩy.
                       Ví dụ: "NestJS, React, PostgreSQL, Kubernetes"

    run_agent_step(role, user_input, extra_context, project, tech_stack) → str
        Chạy đồng bộ một agent đơn lẻ, trả về output ngay.
        (extra_context là tham số vị trí thứ 3 — context bổ sung tùy chọn.)
        Hữu ích khi muốn xem nhanh kết quả của một bước cụ thể.
        Ví dụ: role="sa" để xem kiến trúc giải pháp.

    get_workflow_result(workflow_id, role) → str
        Lấy trạng thái / output của một workflow theo ID.
        role tùy chọn: nếu có thì chỉ trả output của role đó.
        Nếu không có role: trả summary toàn bộ workflow (status, vị trí
        hàng đợi khi pending, các bước lỗi failed_steps, preview output).

    cancel_workflow(workflow_id) → str
        Hủy workflow đang pending/running qua POST /workflow/{id}/cancel.
        404 = không tìm thấy; 409 = workflow đã kết thúc, không hủy được.

    list_agent_roles() → str
        Liệt kê tất cả 15 role theo thứ tự thực thi, kèm tên, model
        và danh sách phụ thuộc. Dùng để khám phá cấu hình hiện tại.

Ví dụ sử dụng trong chat
-------------------------
    Chạy SDLC workflow: xây dựng module marketplace B2B, project=yanlib
    Chạy agent sa: thiết kế kiến trúc hệ thống authentication đa tenant
    Lấy kết quả workflow abc123, role=be
    Hủy workflow abc123
    Liệt kê tất cả agent roles
"""

import time

import requests
from pydantic import BaseModel, Field

# Trạng thái KẾT THÚC của workflow (agent-api WorkflowStatus): vòng poll dừng ngay khi
# gặp một trong các giá trị này. pending/running = còn đang chờ/chạy.
_TERMINAL_STATUSES = ("completed", "failed", "cancelled")


def _status_detail_lines(wf: dict) -> list[str]:
    """Các dòng bổ sung cho summary: vị trí hàng đợi (pending), bước lỗi, lỗi, hủy.

    Mọi field đọc bằng .get() — agent-api cũ có thể chưa trả queue_position /
    failed_steps / status cancelled.
    """
    lines: list[str] = []
    status = wf.get("status")
    if status == "pending":
        queue_position = wf.get("queue_position")
        if queue_position is not None:  # 0 là vị trí hợp lệ
            lines.append(f"- **Vị trí hàng đợi:** {queue_position} (đang chờ workflow khác chạy xong)")
        else:
            lines.append("- **Hàng đợi:** đang chờ tới lượt chạy")
    failed_steps = wf.get("failed_steps") or []
    if failed_steps:
        lines.append(f"- **⚠️ Bước lỗi ({len(failed_steps)}):** {', '.join(failed_steps)}")
    if wf.get("error"):
        lines.append(f"- **Error:** {wf['error']}")
    if status == "cancelled":
        lines.append("- **🛑 Đã bị hủy** — các bước chưa chạy sẽ không chạy nữa.")
    return lines


def _response_detail(resp: requests.Response) -> str:
    """`detail` trong body lỗi JSON của FastAPI; fallback text thô."""
    try:
        body = resp.json()
    except ValueError:
        return resp.text
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return resp.text


def _http_error_parts(exc: requests.exceptions.HTTPError) -> tuple[int | None, str]:
    """(status_code, body) của HTTPError. `exc.response` có thể là None (HTTPError raise
    tay không kèm response) → trả (None, str(exc)) thay vì AttributeError."""
    resp = exc.response
    if resp is None:
        return None, str(exc)
    return resp.status_code, resp.text

# Danh sách role hợp lệ theo thứ tự thực thi chuẩn của SDLC pipeline.
# Phải khớp chính xác với WORKFLOW_STEPS trong agent-api/agents.py.
_VALID_ROLES = [
    "ba",
    "pm",
    "sa",
    "ta",
    "designer",
    "tl",
    "fe",
    "mobile",
    "dba",
    "be",
    "da",
    "tech_lead",
    "tester",
    "devsecops",
    "clarifier",
]

_ROLE_DESCRIPTIONS = {
    "ba":           "BA Agent — Business Analysis",
    "pm":           "PM Agent — Project Management & Planning",
    "sa":           "SA Agent — Solution Architecture",
    "ta":           "TA Agent — Technical Architecture",
    "designer":     "Designer Agent — UI/UX Design",
    "tl":           "Team Lead Agent — Engineering Task Planning",
    "fe":           "FE Agent — Frontend Engineering",
    "mobile":       "Mobile Agent — Mobile Engineering",
    "dba":          "DBA Agent — Database Architecture",
    "be":           "BE Agent — Backend Implementation",
    "da":           "DA Agent — Data Analysis & Reporting",
    "tech_lead":    "Tech Lead Agent — Code Review & Standards",
    "tester":       "Tester Agent — Testing & Quality Assurance",
    "devsecops":    "DevSecOps Agent — Infrastructure, CI/CD & Deployment",
    "clarifier":    "Clarifier Agent — Cross-Role Assumption & Gap Review",
}


class Tools:
    class Valves(BaseModel):
        agent_api_url: str = Field(
            default="http://agent-api:8091",
            description=(
                "URL của agent-api service. Mặc định là tên service trong compose "
                "(http://agent-api:8091) — đúng khi Open WebUI chạy CÙNG network `yan` của "
                "stack. Từ trong container, localhost / host.docker.internal KHÔNG tới được "
                "agent-api (cổng host mặc định chỉ bind 127.0.0.1). Open WebUI ở "
                "container/compose khác: `docker network connect rag-agent-platform_yan "
                "<container>` rồi giữ http://agent-api:8091."
            ),
        )
        block_until_complete: bool = Field(
            default=False,
            description=(
                "O1: nếu True, run_sdlc_workflow sẽ poll đồng bộ tới khi xong (có thể "
                "30+ phút, dễ vượt timeout proxy/browser). Mặc định False = trả về "
                "workflow_id ngay rồi dùng get_workflow_result để xem kết quả sau."
            ),
        )
        timeout: int = Field(
            default=3600,
            description=(
                "Timeout cho single-step request run_agent_step (giây). Full workflow dùng "
                "polling. Một bước báo cáo gọi LLM 1–2 lần, mỗi lần tới OLLAMA_REQUEST_TIMEOUT "
                "của agent-api (mặc định 1200s); bước code (be/fe/...) gọi planner + từng file "
                "→ có thể lâu hơn nhiều. Bỏ cuộc sớm thì agent-api vẫn chạy tiếp và giữ slot "
                "Ollama duy nhất. Bước code dài nên dùng run_sdlc_workflow."
            ),
        )
        poll_interval: int = Field(
            default=15,
            description="Khoảng cách giữa các lần poll workflow status (giây)",
        )
        poll_max_attempts: int = Field(
            default=120,
            description="Số lần poll tối đa trước khi trả về timeout (120 × 15s = 30 phút)",
        )
        rag_enabled: bool = Field(
            default=True,
            description="Có query RAG knowledge base cho mỗi agent step không",
        )
        rag_top_k: int = Field(
            default=5,
            ge=1,
            le=20,
            description="Số kết quả RAG trả về cho mỗi agent",
        )
        default_project: str | None = Field(
            default=None,
            description="RAG project mặc định. Để trống để search tất cả projects.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ── Hàm nội bộ ──────────────────────────────────────────────────────

    def _base_url(self) -> str:
        return self.valves.agent_api_url.rstrip("/")

    def _handle_connection_error(self, exc: Exception) -> str:
        return (
            f"❌ Không kết nối được agent-api tại {self._base_url()}.\n"
            f"Kiểm tra service có đang chạy không: `docker ps | grep agent-api`\n"
            f"Open WebUI phải cùng Docker network với agent-api và gọi bằng tên service "
            f"(http://agent-api:8091). Nếu Open WebUI chạy ở container/compose khác, nối nó "
            f"vào network của stack: `docker network connect rag-agent-platform_yan <container>`. "
            f"localhost / host.docker.internal không dùng được vì cổng host chỉ bind 127.0.0.1.\n"
            f"Chi tiết: {exc}"
        )

    def _roles(self) -> list[str]:
        """A3 — danh sách role lấy ĐỘNG từ agent-api /agents (cache theo instance),
        fallback về _VALID_ROLES nếu không gọi được. Tránh phải đồng bộ tay danh sách
        role với agents.WORKFLOW_STEPS ở backend mỗi khi thêm/bớt role.
        """
        cached = getattr(self, "_roles_cache", None)
        if cached:
            return cached
        try:
            resp = requests.get(f"{self._base_url()}/agents", timeout=10)
            resp.raise_for_status()
            agents = resp.json()
            ordered = sorted(agents.keys(), key=lambda r: agents[r].get("step_id", 999))
            if ordered:
                self._roles_cache = ordered
                return ordered
        except Exception:
            pass
        return _VALID_ROLES

    # ── Công cụ ─────────────────────────────────────────────────────────────────

    def run_sdlc_workflow(
        self,
        user_input: str,
        project: str | None = None,
        tech_stack: str | None = None,
    ) -> str:
        """
        Chạy toàn bộ SDLC Workflow 15 bước (BA → PM → SA → TA → Designer → Team Lead → FE → Mobile → DBA → BE → DA → Tech Lead → Tester → DevSecOps → Clarifier).
        Mỗi agent nhận output đã cắt ngắn của các agent phụ thuộc và bổ sung RAG context từ knowledge base.

        :param user_input: Mô tả business goal / feature / project cần phân tích và implement
        :param project: Tên project RAG cần filter (ví dụ: 'yanlib'). Để trống để search tất cả.
        :param tech_stack: Các công nghệ bắt buộc, phân cách bằng dấu phẩy (ví dụ: 'nextjs,nestjs,mongodb'). Để trống nếu không cần chỉ định.
        :return: Tóm tắt kết quả từng step hoặc link để xem chi tiết
        """
        resolved_project = project or self.valves.default_project
        payload: dict = {
            "user_input": user_input,
            "rag_enabled": self.valves.rag_enabled,
            "rag_top_k": self.valves.rag_top_k,
        }
        if resolved_project:
            payload["project"] = resolved_project
        if tech_stack:
            payload["tech_stack"] = [t.strip() for t in tech_stack.split(",") if t.strip()]

        # Submit workflow
        try:
            resp = requests.post(
                f"{self._base_url()}/workflow/run",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError as exc:
            return self._handle_connection_error(exc)
        except requests.exceptions.HTTPError as exc:
            code, body = _http_error_parts(exc)
            return f"❌ agent-api lỗi HTTP {code}: {body}"
        except Exception as exc:
            return f"❌ Lỗi khi submit workflow: {exc}"

        # O4: tránh KeyError nếu backend trả body thiếu workflow_id.
        workflow_id = (data.get("workflow_id") or "").strip()
        if not workflow_id:
            return f"❌ agent-api không trả về workflow_id. Body: {data}"

        roles = self._roles()
        result_lines = [
            "✅ **Workflow đã được khởi chạy**",
            f"- **ID:** `{workflow_id}`",
            f"- **Project:** {resolved_project or 'all'}",
            f"- **Steps:** {' → '.join(roles)}",
        ]
        if data.get("queue_position") is not None:
            result_lines.append(f"- **Vị trí hàng đợi:** {data['queue_position']}")

        # O1: mặc định KHÔNG block. Trả workflow_id ngay để tránh vượt timeout
        # proxy/browser; người dùng dùng get_workflow_result để theo dõi & lấy kết quả.
        # Bật polling đồng bộ bằng Valve block_until_complete=True.
        if not self.valves.block_until_complete:
            result_lines.append(
                f"\n⏳ Đang chạy nền (15–45 phút tùy model). "
                f"Dùng `get_workflow_result('{workflow_id}')` để theo dõi tiến trình & lấy kết quả, "
                f"`cancel_workflow('{workflow_id}')` để hủy."
            )
            return "\n".join(result_lines)

        result_lines.append(
            "\n⏳ Đang chạy... (có thể mất 15–45 phút tùy model và độ phức tạp)"
        )

        # Poll for completion (chỉ khi block_until_complete=True)
        for attempt in range(self.valves.poll_max_attempts):
            time.sleep(self.valves.poll_interval)
            try:
                poll_resp = requests.get(
                    f"{self._base_url()}/workflow/{workflow_id}",
                    timeout=30,
                )
                poll_resp.raise_for_status()
                wf = poll_resp.json()
            except Exception as exc:
                result_lines.append(
                    f"\n⚠️ Poll attempt {attempt + 1} failed: {exc}")
                continue

            status = wf.get("status", "unknown")
            completed = wf.get("completed_steps", [])
            progress = f"📊 Attempt {attempt + 1}: status={status} | completed={completed}"
            if status == "pending" and wf.get("queue_position") is not None:
                progress += f" | queue_position={wf['queue_position']}"
            result_lines.append(progress)

            if status not in _TERMINAL_STATUSES:
                continue

            if status == "completed":
                result_lines.append("\n---\n## ✅ Workflow Hoàn thành\n")
                # W6: workflow 'completed' vẫn có thể có bước trả về marker [LỖI ...].
                failed_steps = wf.get("failed_steps") or []
                if failed_steps:
                    result_lines.append(
                        f"⚠️ **{len(failed_steps)} bước lỗi:** {', '.join(failed_steps)} "
                        "— xem output từng bước bằng get_workflow_result(..., role=...).\n"
                    )
                step_outputs: dict = wf.get("step_outputs", {})
                for role in roles:
                    if role in step_outputs:
                        role_name = _ROLE_DESCRIPTIONS.get(role, role)
                        preview = step_outputs[role][:400].replace("\n", " ")
                        result_lines.append(f"### {role_name}\n{preview}...\n")
                result_lines.append(
                    f"\n💡 Dùng `get_workflow_result('{workflow_id}')` để xem output đầy đủ từng step."
                )
                return "\n".join(result_lines)

            if status == "failed":
                error = wf.get("error") or "unknown error"
                result_lines.append(f"\n❌ **Workflow thất bại:** {error}")
            else:
                # cancelled: dừng poll ngay (trước đây chỉ completed/failed mới dừng nên
                # run đã hủy bị poll tới hết poll_max_attempts rồi báo "Timeout").
                result_lines.append(
                    f"\n🛑 **Workflow đã bị hủy** sau {len(completed)}/{len(roles)} bước. "
                    f"Output các bước đã xong: `get_workflow_result('{workflow_id}')`."
                )
                if wf.get("error"):
                    result_lines.append(f"Chi tiết: {wf['error']}")
            failed_steps = wf.get("failed_steps") or []
            if failed_steps:
                result_lines.append(f"⚠️ Bước lỗi: {', '.join(failed_steps)}")
            return "\n".join(result_lines)

        result_lines.append(
            f"\n⏰ Timeout sau {self.valves.poll_max_attempts} lần poll. "
            f"Dùng `get_workflow_result('{workflow_id}')` để kiểm tra sau."
        )
        return "\n".join(result_lines)

    def run_agent_step(
        self,
        role: str,
        user_input: str,
        extra_context: str | None = None,
        project: str | None = None,
        tech_stack: str | None = None,
    ) -> str:
        """
        Chạy một agent step đơn lẻ (không cần chạy full workflow).
        Dùng để test từng agent hoặc bổ sung output thủ công.

        :param role: Tên agent role. Hợp lệ: ba, pm, sa, ta, designer, tl, fe, mobile, be, dba, da, tech_lead, tester, devsecops, clarifier
        :param user_input: Business goal / context đầu vào cho agent
        :param extra_context: Context bổ sung (ví dụ: output từ step trước dán vào)
        :param project: RAG project filter. Để trống để search tất cả.
        :param tech_stack: Các công nghệ bắt buộc, phân cách bằng dấu phẩy (ví dụ: 'nextjs,nestjs,mongodb').
        :return: Output của agent role được chọn
        """
        valid_roles = self._roles()
        if role not in valid_roles:
            return (
                f"❌ Role không hợp lệ: `{role}`\n"
                f"Các role hợp lệ: {', '.join(valid_roles)}"
            )

        resolved_project = project or self.valves.default_project
        payload: dict = {
            "user_input": user_input,
            "rag_enabled": self.valves.rag_enabled,
            "rag_top_k": self.valves.rag_top_k,
        }
        if resolved_project:
            payload["project"] = resolved_project
        if extra_context:
            payload["extra_context"] = extra_context
        if tech_stack:
            payload["tech_stack"] = [t.strip() for t in tech_stack.split(",") if t.strip()]

        try:
            resp = requests.post(
                f"{self._base_url()}/agent/{role}",
                json=payload,
                timeout=self.valves.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError as exc:
            return self._handle_connection_error(exc)
        except requests.exceptions.Timeout:
            return (
                f"❌ Timeout sau {self.valves.timeout}s. "
                "Tăng timeout trong Valves hoặc dùng run_sdlc_workflow cho async."
            )
        except requests.exceptions.HTTPError as exc:
            code, body = _http_error_parts(exc)
            return f"❌ agent-api lỗi HTTP {code}: {body}"
        except Exception as exc:
            return f"❌ Lỗi: {exc}"

        role_name = data.get("name", role)
        model = data.get("model", "unknown")
        output = data.get("output", "Không có output.")

        return (
            f"## {role_name}\n"
            f"**Model:** `{model}`  |  **Project:** {resolved_project or 'all'}\n\n"
            f"---\n\n{output}"
        )

    def get_workflow_result(
        self,
        workflow_id: str,
        role: str | None = None,
    ) -> str:
        """
        Lấy trạng thái và kết quả của một workflow theo ID (pending / running / completed / failed / cancelled).
        Có thể filter để chỉ xem output của một role cụ thể.

        :param workflow_id: ID của workflow (trả về từ run_sdlc_workflow)
        :param role: Nếu muốn xem output của một step cụ thể (ví dụ: 'ba', 'sa'). Để trống để xem tóm tắt tất cả.
        :return: Kết quả workflow hoặc output của step được chọn
        """
        workflow_id = workflow_id.strip()
        try:
            resp = requests.get(
                f"{self._base_url()}/workflow/{workflow_id}",
                timeout=30,
            )
            resp.raise_for_status()
            wf = resp.json()
        except requests.exceptions.ConnectionError as exc:
            return self._handle_connection_error(exc)
        except requests.exceptions.HTTPError as exc:
            code, body = _http_error_parts(exc)
            if code == 404:
                return f"❌ Workflow `{workflow_id}` không tìm thấy. ID có thể đã hết hạn."
            return f"❌ agent-api lỗi HTTP {code}: {body}"
        except Exception as exc:
            return f"❌ Lỗi: {exc}"

        status = wf.get("status", "unknown")
        completed = wf.get("completed_steps", [])
        step_outputs: dict = wf.get("step_outputs", {})
        roles = self._roles()

        if role:
            if role not in roles:
                return f"❌ Role không hợp lệ: `{role}`. Hợp lệ: {', '.join(roles)}"
            if role not in step_outputs:
                return (
                    f"⚠️ Step `{role}` chưa có output.\n"
                    f"Workflow status: {status} | Completed: {completed}"
                )
            role_name = _ROLE_DESCRIPTIONS.get(role, role)
            if role in (wf.get("failed_steps") or []):
                return f"## {role_name} ⚠️ (bước lỗi)\n\n{step_outputs[role]}"
            return f"## {role_name}\n\n{step_outputs[role]}"

        # Summary of all steps
        lines = [
            f"## Workflow `{workflow_id}`",
            f"- **Status:** {status}",
            f"- **Project:** {wf.get('project') or 'all'}",
            f"- **Completed steps:** {len(completed)}/{len(roles)}",
            f"- **Created:** {wf.get('created_at', 'N/A')}",
            f"- **Completed at:** {wf.get('completed_at', 'N/A')}",
        ]

        lines.extend(_status_detail_lines(wf))

        failed_steps = set(wf.get("failed_steps") or [])
        lines.append("\n---\n### Step Outputs\n")
        for r in roles:
            if r in step_outputs:
                role_name = _ROLE_DESCRIPTIONS.get(r, r)
                marker = " ⚠️ _(lỗi)_" if r in failed_steps else ""
                preview = step_outputs[r][:300].replace("\n", " ")
                lines.append(f"**{role_name}**{marker}\n{preview}...\n")
            else:
                lines.append(
                    f"**{_ROLE_DESCRIPTIONS.get(r, r)}** — _(not run)_\n")

        lines.append(
            f"\n💡 Dùng `get_workflow_result('{workflow_id}', role='ba')` để xem full output từng step."
        )
        if status in ("pending", "running"):
            lines.append(f"💡 Hủy workflow: `cancel_workflow('{workflow_id}')`.")
        return "\n".join(lines)

    def cancel_workflow(self, workflow_id: str) -> str:
        """
        Hủy một workflow đang chờ (pending) hoặc đang chạy (running).
        Output của các bước đã hoàn thành vẫn xem được bằng get_workflow_result.

        :param workflow_id: ID của workflow cần hủy (trả về từ run_sdlc_workflow)
        :return: Kết quả yêu cầu hủy
        """
        workflow_id = workflow_id.strip()
        if not workflow_id:
            return "❌ Thiếu workflow_id."
        try:
            resp = requests.post(
                f"{self._base_url()}/workflow/{workflow_id}/cancel",
                timeout=30,
            )
        except requests.exceptions.ConnectionError as exc:
            return self._handle_connection_error(exc)
        except Exception as exc:
            return f"❌ Lỗi khi hủy workflow: {exc}"

        if resp.status_code == 404:
            return f"❌ Workflow `{workflow_id}` không tìm thấy. ID có thể sai hoặc đã hết hạn."
        if resp.status_code == 409:
            return (
                f"⚠️ Không hủy được workflow `{workflow_id}`: {_response_detail(resp)}\n"
                f"(Workflow đã kết thúc — xem kết quả bằng `get_workflow_result('{workflow_id}')`.)"
            )
        if resp.status_code >= 400:
            return f"❌ agent-api lỗi HTTP {resp.status_code}: {_response_detail(resp)}"

        try:
            data = resp.json()
        except ValueError:
            data = {}
        status = data.get("status", "unknown") if isinstance(data, dict) else "unknown"
        if status == "cancelled":
            head = f"🛑 Workflow `{workflow_id}` đã bị hủy."
        else:
            head = f"🛑 Đã gửi yêu cầu hủy workflow `{workflow_id}` (status hiện tại: **{status}**)."
        return (
            f"{head}\n"
            f"Dùng `get_workflow_result('{workflow_id}')` để kiểm tra trạng thái và output các bước đã xong."
        )

    def list_agent_roles(self) -> str:
        """
        Liệt kê tất cả agent roles hợp lệ, model được dùng, và thứ tự chạy trong SDLC workflow.

        :return: Danh sách agent roles
        """
        try:
            resp = requests.get(f"{self._base_url()}/agents", timeout=15)
            resp.raise_for_status()
            agents = resp.json()
        except requests.exceptions.ConnectionError as exc:
            return self._handle_connection_error(exc)
        except Exception as exc:
            return f"❌ Lỗi: {exc}"

        # A3: thứ tự lấy trực tiếp từ response /agents (theo step_id), fallback hardcode.
        ordered = sorted(agents.keys(), key=lambda r: agents[r].get("step_id", 999)) or _VALID_ROLES
        rows = []
        for role in ordered:
            info = agents.get(role, {})
            rows.append(
                f"| **{info.get('step_id', '?')}** | `{role}` | {info.get('name', role)} | `{info.get('model', '?')}` |"
            )

        return (
            "## SDLC Agent Roles\n\n"
            "| Step | Role | Name | Model |\n"
            "|------|------|------|-------|\n"
            + "\n".join(rows)
        )
