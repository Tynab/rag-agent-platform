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
version: 2.0.0

Mô tả chi tiết
--------------
Tool là giao tiếp trực tiếp giữa Open WebUI và Agent API của rag-agent-platform (cổng 8091).
Cho phép người dùng khởi động và theo dõi toàn bộ SDLC pipeline AI
trực tiếp từ giao diện chat mà không cần gọi API thủ công.

Valves (biến cấu hình)
----------------------
    agent_api_url      URL của agent-api trong nội bộ Docker network.
                       Mặc định: http://agent-api:8091

    timeout            Timeout (giây) cho mỗi single-step agent call.
                       Mặc định: 600 giây.

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
        Khởi chạy full workflow 15 bước, poll đến khi hoàn tất,
        trả về summary kết quả (status, thời gian, output mỗi bước).
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
        Lấy output của một workflow đã chạy xong theo ID.
        role tùy chọn: nếu có thì chỉ trả output của role đó.
        Nếu không có role: trả summary toàn bộ workflow.

    list_agent_roles() → str
        Liệt kê tất cả 15 role theo thứ tự thực thi, kèm tên, model
        và danh sách phụ thuộc. Dùng để khám phá cấu hình hiện tại.

Ví dụ sử dụng trong chat
-------------------------
    Chạy SDLC workflow: xây dựng module marketplace B2B, project=yanlib
    Chạy agent sa: thiết kế kiến trúc hệ thống authentication đa tenant
    Lấy kết quả workflow abc123, role=be
    Liệt kê tất cả agent roles
"""

import time

import requests
from pydantic import BaseModel, Field

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
                "URL của agent-api service. Mặc định là hostname nội bộ Docker network "
                "(http://agent-api:8091) — đúng khi Open WebUI chạy CÙNG compose `yan`. "
                "Nếu Open WebUI chạy ngoài network đó, đổi sang http://localhost:8091 "
                "hoặc http://host.docker.internal:8091."
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
            default=600,
            description="Timeout cho single-step request (giây). Full workflow dùng polling.",
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
            f"Nếu Open WebUI chạy ngoài Docker network `yan`, đổi Valve agent_api_url "
            f"sang http://localhost:8091 hoặc http://host.docker.internal:8091.\n"
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
            return f"❌ agent-api lỗi HTTP {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return f"❌ Lỗi khi submit workflow: {exc}"

        # O4: tránh KeyError nếu backend trả body thiếu workflow_id.
        workflow_id = (data.get("workflow_id") or "").strip()
        if not workflow_id:
            return f"❌ agent-api không trả về workflow_id. Body: {data}"

        roles = self._roles()
        result_lines = [
            f"✅ **Workflow đã được khởi chạy**",
            f"- **ID:** `{workflow_id}`",
            f"- **Project:** {resolved_project or 'all'}",
            f"- **Steps:** {' → '.join(roles)}",
        ]

        # O1: mặc định KHÔNG block. Trả workflow_id ngay để tránh vượt timeout
        # proxy/browser; người dùng dùng get_workflow_result để theo dõi & lấy kết quả.
        # Bật polling đồng bộ bằng Valve block_until_complete=True.
        if not self.valves.block_until_complete:
            result_lines.append(
                f"\n⏳ Đang chạy nền (15–45 phút tùy model). "
                f"Dùng `get_workflow_result('{workflow_id}')` để theo dõi tiến trình & lấy kết quả."
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
            result_lines.append(
                f"📊 Attempt {attempt + 1}: status={status} | completed={completed}"
            )

            if status == "completed":
                result_lines.append("\n---\n## ✅ Workflow Hoàn thành\n")
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
                error = wf.get("error", "unknown error")
                result_lines.append(f"\n❌ **Workflow thất bại:** {error}")
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
            return f"❌ agent-api lỗi HTTP {exc.response.status_code}: {exc.response.text}"
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
        Lấy kết quả của một workflow đã chạy theo ID.
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
            if exc.response.status_code == 404:
                return f"❌ Workflow `{workflow_id}` không tìm thấy. ID có thể đã hết hạn."
            return f"❌ agent-api lỗi HTTP {exc.response.status_code}: {exc.response.text}"
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

        if wf.get("error"):
            lines.append(f"- **Error:** {wf['error']}")

        lines.append("\n---\n### Step Outputs\n")
        for r in roles:
            if r in step_outputs:
                role_name = _ROLE_DESCRIPTIONS.get(r, r)
                preview = step_outputs[r][:300].replace("\n", " ")
                lines.append(f"**{role_name}**\n{preview}...\n")
            else:
                lines.append(
                    f"**{_ROLE_DESCRIPTIONS.get(r, r)}** — _(not run)_\n")

        lines.append(
            f"\n💡 Dùng `get_workflow_result('{workflow_id}', role='ba')` để xem full output từng step."
        )
        return "\n".join(lines)

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
