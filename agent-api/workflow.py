"""
workflow.py — Bộ điều phối SDLC workflow 15 bước dùng LangGraph
================================================================

Mô tả
-----
Module trung tâm xây dựng và vận hành LangGraph StateGraph cho toàn bộ pipeline
SDLC. Mỗi bước (node) trong đồ thị tương ứng một AgentConfig từ agents.py.

Cấu trúc đồ thị (tuần tự nghiêm ngặt)
--------------------------------------
    ba → pm → sa → ta → designer → tl → fe → mobile → dba → be
    → da → tech_lead → tester → devsecops → clarifier → END

Không có nhánh song song. Mỗi node đợi tất cả node phụ thuộc hoàn thành
(thứ tự được đảm bảo bởi chuỗi cạnh tuần tự trong StateGraph).

Vòng đời mỗi node — 3 giai đoạn
---------------------------------
  1. Tổng hợp context (bắt buộc)
       _build_context_parts() thu thập:
       - user_input (mục tiêu kinh doanh gốc từ người dùng)
       - tech_stack bắt buộc nếu được cung cấp
       - output rút gọn từ các agent phụ thuộc (dep_max_chars per dep)
       - gợi ý ngày hiện tại cho planning roles (ba, pm, sa, ta)
       - gợi ý kiểu DB (nosql/sql/both) cho dba và da
       - extra_context tùy chọn

  2. Bổ sung RAG (tùy chọn, khi rag_enabled=True)
       _query_rag() gọi POST /ask đến rag-api với rag_query_hint của agent.
       Kết quả được cắt ngắn và thêm vào context. Lỗi RAG không dừng workflow.

  3. Gọi LLM (bắt buộc)
       Artifact roles (fe, mobile, be, dba, da, tech_lead, devsecops):
           _generate_artifacts_multi_turn() — 2 pha sinh code:
           Pha 1: CODING_PLANNER_MODEL lập kế hoạch danh sách file,
                  tham chiếu TL task board để không bỏ sót task.
           Pha 2: coding model sinh từng file độc lập với context tập trung.
                  Nếu còn TL task chưa cover → follow-up pass sinh thêm file.
           Kết thúc: _build_task_checklist() tạo bảng ✅/⏳ đối chiếu TL tasks.
       Các role còn lại:
           _call_agent() — gọi LLM một lần với toàn bộ context.
           Nếu output < 30 ký tự → retry với context rút gọn 1.500 ký tự cuối.

Quản lý State (LangGraph TypedDict + reducers)
----------------------------------------------
    step_outputs   : Annotated[dict, operator.or_]
        Mỗi node trả về {role: output_text}. LangGraph gộp bằng operator.or_.

    completed_steps: Annotated[list, operator.add]
        Mỗi node nối [role] vào list. LangGraph dùng operator.add.

    Các trường còn lại (project, user_input, tech_stack, error):
        TypedDict thông thường — ghi đè lần cuối thắng.

Giới hạn context window
------------------------
    MAX_PREV_OUTPUT_CHARS (3.000): giới hạn mặc định per dep output.
    _PER_DEP_CHARS: ghi đè per-role — tech_lead: 800, devsecops: 1.000,
    tester: 3.000, clarifier: 1.500.
    OLLAMA_NUM_CTX: context window Ollama, mặc định 32.768 token.

Xử lý reasoning models
-----------------------
    _REASONING_MODELS: frozenset khớp chuỗi con tên model reasoning
    (phi4-mini-reasoning, phi4-reasoning, qwq, deepseek-r1, qwen3,
    north-mini-code, gemma4, mistral-medium-3.5).
    _strip_thinking(): loại bỏ <think>...</think> khỏi output trước khi
    lưu vào step_outputs — chỉ giữ lại phần nội dung thực sự.

Chống vòng lặp output
----------------------
    _deloop(): cắt các dòng lặp liên tiếp quá giới hạn. Phòng ngừa
    coding model nhỏ có xu hướng lặp lại boilerplate vô tận.

TL Task Tracking
----------------
    _extract_tl_tasks(role, context)             → list[dict]
    _find_uncovered_tasks(tl_tasks, file_infos)  → list[dict]
    _build_task_checklist(tl_tasks, file_infos)  → str (markdown table)

Clarifier Regen Loop
--------------------
    _parse_clarifier_regen_list(clarifier_output) → list[str]
    Parse §10 Recommended Re-generation List, chỉ trả về roles thuộc
    _REGEN_ELIGIBLE_ROLES. Logic vòng lặp thực thi trong app.py.

Singleton workflow (thread-safe)
---------------------------------
    _workflow_lock + double-checked locking: StateGraph chỉ compile
    đúng một lần dù nhiều request đến đồng thời lúc khởi động.
    Sau khi compile lần đầu, _workflow được cache module-level.

Hàm xuất khẩu công khai
-----------------------
    get_workflow()                             → CompiledGraph đã compile
    run_single_step(role, ...) → str           → chạy một agent độc lập
    _parse_clarifier_regen_list(text) → list   → parse §10 cho regen loop
"""

import functools
import json as _json
import logging
import os
import re
import threading
from datetime import datetime
from typing import Annotated, TypedDict
import operator

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from agents import AGENTS, MAX_PREV_OUTPUT_CHARS, WORKFLOW_STEPS, AgentConfig

logger = logging.getLogger("agent-workflow")

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
RAG_API_URL: str = os.environ.get("RAG_API_URL", "http://rag-api:8090")
RAG_TOP_K: int = int(os.environ.get("RAG_TOP_K", "5"))
# Kích thước context window (token) gửi cho Ollama mỗi lần gọi LLM.
# Tăng giá trị này nếu model hỗ trợ window lớn hơn (ví dụ: 65536).
OLLAMA_NUM_CTX: int = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "32768"))
# Timeout (giây) chờ rag-api trả về kết quả mỗi lần gọi POST /ask.
RAG_TIMEOUT: int = int(os.environ.get("RAG_TIMEOUT", "600"))
# Timeout (giây) chờ Ollama trả về response cho mỗi lần gọi LLM.
# Nên đặt cao hơn RAG_TIMEOUT vì LLM generation tốn thời gian hơn embed.
OLLAMA_REQUEST_TIMEOUT: int = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "1200"))

# Model nhẹ chuyên lập kế hoạch danh sách file cần sinh — dùng chat model,
# không phải coding model, vì nhiệm vụ là phân tích task rồi xuất JSON array.
CODING_PLANNER_MODEL: str = os.environ.get(
    "CODING_PLANNER_MODEL",
    "granite4.1:8b",
)
# Giới hạn số file tối đa mỗi coding agent được phép sinh ra trong một workflow.
# Tăng giá trị này nếu project phức tạp cần nhiều file hơn.
_MAX_FILES_PER_ROLE: int = int(os.environ.get("MAX_FILES_PER_ROLE", "6"))


# Bộ quy tắc bắt buộc được chèn vào đầu system prompt của mọi agent.
# Đảm bảo hành vi nhất quán: không bịa thông tin, phân loại phát biểu,
# trích dẫn xuyên agent, ưu tiên chi tiết thay vì tóm tắt chung chung.
COMMON_AGENT_RULES: str = """\
Critical rules — apply to every response:
1. Use ONLY the provided User Input, Previous Agent Outputs, Required Tech Stack, and RAG Knowledge Base Context. Do not invent information from outside these sources.
2. Do not invent business requirements, APIs, database fields, UI screens, permissions, SLA rules, cost figures, or infrastructure config unless explicitly marked as [Proposed].
3. Classify important statements with one of these labels:
   - [Confirmed]: directly supported by input or context.
   - [Assumption]: reasonable inference, not explicitly stated — must be flagged.
   - [Open Question]: requires PO/PM/Tech confirmation before proceeding.
   - [Proposed]: suggested implementation option, not yet decided.
4. If information is missing, place it under Open Questions instead of silently filling the gap.
5. Do not repeat large sections from previous agents. Produce only the artifact owned by your role.
6. Keep IDs consistent across artifacts (req ID, user story ID, test case ID).
7. Prefer concise Markdown tables for implementation-ready outputs.
8. If sources are available in RAG context, reference the source file name in a Notes/Source column.
9. If actual source code is not provided, clearly label your output as Design Review, not Code Review. Do not invent file names, line numbers, or PR comments.
10. Complete sections in order. If context budget is exhausted before all sections are done, mark remaining sections as [Deferred — insufficient input] and stop cleanly. Do not produce partial sentences or half-filled tables.
11. LANGUAGE: Respond in English only. Do not write in Vietnamese, even if the user input or project context is in Vietnamese. All section headings, labels, table headers, and prose must be in English.
12. CROSS-AGENT CITATIONS: Whenever a decision, design choice, or data element traces back to a prior agent's output, cite it explicitly using the format "Agent §Section" (e.g., "per BA §3 FR-01", "per SA §3 /api/auth/login", "per TL §4 FE Task #3", "per Designer §5 Screen S-02"). Do NOT silently consume upstream information without citation.
13. INTRA-DOCUMENT LINKAGE: For every section in your output, add a brief note stating which other sections within this document it connects to, using "→ see §N" notation (e.g., "→ see §5 API Integration Map", "→ see §3 Component Breakdown"). This makes the dependency graph within your output explicit.
14. DEPTH OVER SUMMARY: Every section must contain complete, actionable, implementation-ready detail — not a summary or placeholder. Tables must have real data rows derived from the provided context. Bullet points must be specific (names, values, IDs), not generic descriptions. If you find yourself writing a generic statement like "handle errors appropriately", replace it with the exact error codes, HTTP statuses, and recovery actions required.
15. PLATFORM CONVENTIONS ARE BINDING: If the RAG Knowledge Base Context contains project-specific coding conventions, library names, naming rules, architectural patterns (e.g. three-seam pattern, scope=platform rows, Module Federation topology, tenantId-first index invariant), platform-provided shared modules (common-lib, guard decorators, base repositories), or Kafka topic names — treat ALL of these as HARD CONSTRAINTS that OVERRIDE generic best practices. Do not invent a new abstraction when the platform already provides one. Do not use a generic library when the platform mandates a specific one. Do not invent service names, Kafka topic names, route prefixes, port numbers, or decorator names — use only what the RAG context documents. If RAG documents a naming invariant (e.g. "tenantId is field #1 in every compound index"), apply it everywhere without exception.
"""


# ── Trạng thái LangGraph ───────────────────────────────────────────────────────────

class SDLCState(TypedDict):
    """Trạng thái dùng chung được truyền qua các LangGraph node."""

    project: str | None           # Tên project để lọc RAG collection (ví dụ: "yanlib"). Null = tìm tất cả.
    user_input: str               # Mục tiêu kinh doanh hoặc yêu cầu gốc từ người dùng — được truyền qua toàn pipeline.
    rag_enabled: bool             # True = mỗi agent gọi RAG API để lấy context từ knowledge base.
    rag_top_k: int                # Số chunk trả về mỗi lần gọi POST /ask của rag-api.
    rag_api_url: str              # URL của rag-api service (ví dụ: http://rag-api:8090).
    ollama_base_url: str          # URL của Ollama server (ví dụ: http://ollama:11434).

    # LangGraph reducer operator.or_: mỗi node trả về {role: output},
    # LangGraph gộp vào dict tổng bằng cách cập nhật (không ghi đè toàn bộ).
    step_outputs: Annotated[dict[str, str], operator.or_]

    # LangGraph reducer operator.add: mỗi node nối thêm [role] vào list.
    # Kết quả cuối cùng là danh sách tất cả role đã chạy xong theo thứ tự.
    completed_steps: Annotated[list[str], operator.add]

    tech_stack: list[str] | None    # Danh sách công nghệ bắt buộc. Tất cả agent đều nhận danh sách
                                    # này và phải bám sát — không đề xuất công nghệ thay thế.

    error: str | None             # Thông báo lỗi đầu tiên gặp phải trong pipeline. None = không có lỗi.
                                  # Workflow tiếp tục chạy dù có lỗi; lỗi chỉ ghi lại không dừng hẳn.


# ── Hàm tiện ích ───────────────────────────────────────────────────────────────────

def _query_rag(rag_api_url: str, question: str, project: str | None, top_k: int) -> str:
    """
    Gọi endpoint /ask của rag-api và trả về văn bản trả lời.
    Trả về chuỗi rỗng nếu có lỗi (không gây dừng workflow).
    """
    try:
        payload: dict = {"question": question, "top_k": top_k}
        if project:
            payload["project"] = project
        resp = requests.post(
            f"{rag_api_url}/ask",
            json=payload,
            timeout=RAG_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except Exception as exc:
        logger.warning("RAG query failed (non-fatal): %s", exc)
        return ""


def _truncate(text: str, max_chars: int = MAX_PREV_OUTPUT_CHARS) -> str:
    """Cắt ngắn text về tối đa max_chars ký tự.

    Không thêm dấu "..." hay marker cắt để tránh LLM đọc marker đó
    rồi tái tạo lại phần đã bị cắt trong output tiếp theo.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# Frozenset các chuỗi con khớp tên model reasoning. Khi tên model chứa
# một trong các chuỗi này, _strip_thinking() sẽ loại bỏ <think>...</think>
# khỏi output trước khi lưu vào step_outputs.
_REASONING_MODELS: frozenset[str] = frozenset({
    "phi4-mini-reasoning",
    "phi4-reasoning",
    "qwq",
    "deepseek-r1",
    "qwen3",
    "north-mini-code",
    "gemma4",
    "mistral-medium-3.5",
})


def _strip_thinking(text: str) -> str:
    """Loại bỏ tất cả block <think>...</think> từ output của reasoning model.

    Một số model (qwen3, deepseek-r1, qwq...) phát ra chain-of-thought nội bộ
    trong tag <think>. Người dùng không cần thấy phần này — chỉ giữ lại
    phần nội dung thực sự sau khi quá trình suy luận kết thúc.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _get_num_ctx(model: str) -> int:
    """Trả về context window (token) phù hợp với model cụ thể.

    codegemma:2b bị clamp ở 4.096 token để tránh hallucination và vòng lặp
    vô tận — model nhỏ không xử lý tốt context quá dài trong pipeline nhiều bước.
    Các model khác dùng OLLAMA_NUM_CTX toàn cục (mặc định 32.768).
    """
    if "codegemma:2b" in model.lower():
        return min(4096, OLLAMA_NUM_CTX)
    return OLLAMA_NUM_CTX


# Giới hạn ký tự tối đa lấy từ mỗi dep output theo role, để tránh overflow
# context window khi role có nhiều phụ thuộc. Tính toán ước lượng:
#   tech_lead:  5 deps × 800  = 4.000 chars ≈ 1.000 tokens
#   devsecops:  4 deps × 1000 = 4.000 chars ≈ 1.000 tokens
#   tester:     5 deps × 3000 = 15.000 chars ≈ 3.700 tokens
#   clarifier: 14 deps × 1500 = 21.000 chars ≈ 5.250 tokens
_PER_DEP_CHARS: dict[str, int] = {
    "tech_lead":  800,
    "devsecops": 1_000,
    "tester":    3_000,
    "clarifier": 1_500,
}


# Tập hợp các role sử dụng quy trình sinh code 2 pha (lập kế hoạch → sinh từng file).
# "da" được loại trừ khỏi đây vì DA agent chạy qua _call_agent (sinh báo cáo một lần),
# không phải vòng lặp per-file. Tuy nhiên artifacts.py vẫn trích xuất code block
# từ output của DA và lưu vào disk sau khi node hoàn thành.
_ARTIFACT_ROLES: frozenset[str] = frozenset(
    {"fe", "mobile", "be", "dba", "devsecops", "tech_lead"}
)

# Template prompt gửi cho CODING_PLANNER_MODEL để lập kế hoạch danh sách file.
# Model nhận task, tech stack và context tóm tắt, rồi trả về JSON array
# mô tả các file cần sinh: [{filename, description, language}].
_PLAN_PROMPT_TMPL = """\
You are a {role_name}. Based on the task and tech stack, list the source code / config files you will create.
Return ONLY a JSON array (no prose, no markdown wrapper):
[{{"filename": "src/components/Login.tsx", "description": "Login form component", "language": "typescript"}}]
Rules:
- Produce ONLY essential, non-trivial files. Minimum 2, maximum {max_files}. Do NOT pad to the limit — only list files that are truly needed.
- Use relative paths from project root.
- "language" must be a valid code fence name (typescript, python, yaml, sql, dockerfile, etc).
{tl_tasks_section}
{extra}
Task: {user_input}
Tech stack: {tech_stack}
Context summary:
{context_summary}
"""

# Template prompt gửi cho coding model để sinh nội dung một file cụ thể.
# Mỗi file được sinh độc lập với context tập trung (filename + description)
# để tránh model bị phân tán bởi context quá lớn.
_FILE_GEN_PROMPT_TMPL = """\
Write the {language} source file: {filename}
Purpose: {description}
Tech stack: {tech_stack}
{extra}

Strict rules:
- Respond with one triple-backtick code block containing the file content.
- Do not include author/date/version/license/copyright metadata headers.
- Do not repeat identical import lines or boilerplate blocks.
- Match the depth in the guidance above: a SCAFFOLD leaves `// TODO:` markers for feature
  logic; a non-scaffold file must be complete and production-ready. Do not pad either way.
"""


# ── Trích xuất TL task board và tạo checklist hoàn thành cho engineer agents ────

# Ánh xạ từ role engineer sang các từ khóa tiêu đề section trong TL output.
# Dùng để định vị bảng task board tương ứng trong context khi parse.
_TL_ROLE_SECTIONS: dict[str, list[str]] = {
    "fe":     ["FE Task Board", "4. FE Task", "§4 FE", "4. FE"],
    "mobile": ["Mobile Task Board", "5. Mobile Task", "§5 Mobile", "5. Mobile"],
    "be":     ["BE Task Board", "6. BE Task", "§6 BE", "6. BE"],
    "dba":    ["DBA Task Board", "7. DBA Task", "§7 DBA", "7. DBA"],
}


def _extract_tl_tasks(role: str, context: str) -> list[dict]:
    """Phân tích bảng TL task board cho *role* từ chuỗi context tổng hợp.

    Tìm section TL tương ứng với role (dựa trên _TL_ROLE_SECTIONS),
    sau đó quét từng dòng của bảng Markdown để lấy thông tin task.

    Trả về:
        list[dict]: Danh sách task, mỗi phần tử có {id, task, priority}.
        Trả về [] nếu không tìm thấy section hoặc bảng không thể parse.
    """
    patterns = _TL_ROLE_SECTIONS.get(role)
    if not patterns:
        return []

    section_start = -1
    for pat in patterns:
        idx = context.find(pat)
        if idx != -1 and (section_start == -1 or idx < section_start):
            section_start = idx
    if section_start == -1:
        return []

    section_text = context[section_start:]
    # Cắt section text tại heading số tiếp theo (ví dụ: "5. Mobile", "8. Sprint")
    # để tránh đọc nhầm task của section khác.
    stop_match = re.search(r"\n(?:[5-9]|[1-9][0-9])\.\s+[A-Z]", section_text[50:])
    if stop_match:
        section_text = section_text[: 50 + stop_match.start()]

    tasks: list[dict] = []
    for line in section_text.split("\n"):
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        # Bỏ qua dòng kẻ phân cách (---) và dòng tiêu đề bảng Markdown.
        if all(set(c.replace("-", "").replace(":", "").replace(" ", "")) <= {"-", ":", "|"} for c in cells):
            continue
        first = cells[0]
        if first.lower() in ("#", "task", "id", "no") or first.startswith("-"):
            continue
        # Xác định cột nào chứa tên task: nếu cột đầu là số thứ tự thì tên task ở cột 2.
        task_idx = 1 if (first.lstrip("#").strip().isdigit() or first.strip() == "#") else 0
        if task_idx >= len(cells):
            continue
        task_text = cells[task_idx].strip()
        if not task_text or len(task_text) < 4 or task_text.lower() in ("task", "description"):
            continue
        entry: dict = {
            "id": cells[0].lstrip("#").strip() if task_idx == 1 else str(len(tasks) + 1),
            "task": task_text,
            "priority": "—",
        }
        # Lấy mức độ ưu tiên: quét các cột sau tên task để tìm P0/P1/P2/HIGH/MED/LOW.
        for offset in (3, 4, 5):
            if task_idx + offset < len(cells):
                val = cells[task_idx + offset].upper()
                if val in ("P0", "P1", "P2", "HIGH", "MED", "LOW", "H", "M", "L"):
                    entry["priority"] = val
                    break
        tasks.append(entry)
    return tasks


def _find_uncovered_tasks(tl_tasks: list[dict], file_infos: list[dict]) -> list[dict]:
    """Trả về danh sách TL task chưa được cover bởi bất kỳ file nào trong *file_infos*.

    Heuristic xác định coverage: một task được coi là "covered" khi ít nhất
    max(1, len(từ_khóa) // 2) từ khóa dài >= 4 ký tự của task xuất hiện
    trong chuỗi ghép (filename + description) của ít nhất một file đã lên kế hoạch.

    Các task không được cover sẽ được đưa vào follow-up pass để sinh thêm file.
    """
    uncovered: list[dict] = []
    for task in tl_tasks:
        task_text = task.get("task", "").lower()
        keywords = [w for w in re.split(r"\W+", task_text) if len(w) >= 4]
        if not keywords:
            continue
        threshold = max(1, len(keywords) // 2)
        matched = any(
            sum(
                1 for kw in keywords
                if kw in (f.get("filename", "") + " " + f.get("description", "")).lower()
            ) >= threshold
            for f in file_infos
        )
        if not matched:
            uncovered.append(task)
    return uncovered


def _build_task_checklist(tl_tasks: list[dict], file_infos: list[dict]) -> str:
    """Xây dựng bảng Markdown Task Completion Checklist để thêm vào output engineer.

    Mỗi dòng đối chiếu một TL task với kết quả sinh code (3 trạng thái):
    - ✅ Done: task được cover bởi ít nhất một file đã sinh.
    - ⏳ Partial: task chưa có file khớp nhưng đã được đề cập trong output.
    - ❌ Deferred: task bị defer hoặc chưa được xử lý.

    Trả về chuỗi rỗng nếu tl_tasks trống (không có TL task board).
    """
    if not tl_tasks:
        return ""
    lines: list[str] = [
        "\n---",
        "## ✅ Task Completion Checklist (TL Task Board → Implementation)",
        "",
        "| # | Task | Priority | Status | File(s) / Section |",
        "|---|------|----------|--------|-------------------|",
    ]
    for task in tl_tasks:
        task_text = task.get("task", "")
        priority  = task.get("priority", "—")
        tid       = task.get("id", "—")
        task_kws  = [w for w in re.split(r"\W+", task_text.lower()) if len(w) >= 4]
        threshold = max(1, len(task_kws) // 2) if task_kws else 1
        matched_files = [
            f.get("filename", "")
            for f in file_infos
            if task_kws and sum(
                1 for kw in task_kws
                if kw in (f.get("filename", "") + " " + f.get("description", "")).lower()
            ) >= threshold
        ]
        if matched_files:
            status      = "✅ Done"
            files_label = ", ".join(f"`{fn}`" for fn in matched_files[:2])
        else:
            status      = "⏳ Partial"
            files_label = "—"
        task_display = (task_text[:60] + "…") if len(task_text) > 60 else task_text
        lines.append(f"| {tid} | {task_display} | {priority} | {status} | {files_label} |")
    return "\n".join(lines)


# Tập hợp role được phép re-generate trong Clarifier Regen Loop.
# Loại trừ planning roles (ba, pm, sa, ta, designer, tl, tester) để tránh
# thay đổi foundation phân tích nghiệp vụ, và loại clarifier để ngăn vòng lặp vô tận.
_REGEN_ELIGIBLE_ROLES: frozenset[str] = frozenset(
    {"fe", "mobile", "dba", "be", "da", "tech_lead", "devsecops"}
)


def _parse_clarifier_regen_list(clarifier_output: str) -> list[str]:
    """Parse §10 Recommended Re-generation List từ Clarifier output.

    Tìm bảng trong §10, quét cột "Agent Role" để lấy tên role hợp lệ.
    Chỉ trả về các role thuộc _REGEN_ELIGIBLE_ROLES (loại trừ planning roles
    và clarifier để ngăn infinite loop).
    Kết quả được sắp xếp theo thứ tự WORKFLOW_STEPS.
    Trả về list rỗng nếu §10 không có trong output hoặc không có gì để re-gen.
    """
    # Tìm vị trí bắt đầu của §10 trong output Clarifier bằng nhiều pattern khác nhau
    # để xử lý cả trường hợp model viết "10.", "§10", "## 10. Recommended Re-generation".
    section_match = re.search(
        r"(?:10\.|§\s*10\b|#+\s*10\.?\s+Recommended\s+Re.?generation|"
        r"Recommended\s+Re.?generation\s+List)",
        clarifier_output,
        re.IGNORECASE,
    )
    if not section_match:
        return []

    section_text = clarifier_output[section_match.start():]
    # Cắt nội dung tại section tiếp theo (11. hoặc cao hơn) để tránh
    # đọc nhầm tên role từ các section khác trong Clarifier output.
    stop = re.search(r"\n(?:1[1-9]|[2-9][0-9])\.\s+[A-Z]", section_text[30:])
    if stop:
        section_text = section_text[:30 + stop.start()]

    seen: set[str] = set()

    for line in section_text.split("\n"):
        if "|" not in line:
            continue
        cells = [c.strip().lower() for c in line.split("|") if c.strip()]
        if not cells:
            continue
        # Bỏ qua dòng kẻ phân cách Markdown (---) và dòng tiêu đề bảng.
        if all(set(c.replace("-", "").replace(":", "").replace(" ", "")) <= {"-", ":", "|"} for c in cells):
            continue
        # Quét từng cell để tìm tên role hợp lệ bằng word-boundary regex,
        # tránh khớp nhầm tên role con trong tên role khác (ví dụ: "be" trong "devsecops").
        for cell in cells:
            for role in _REGEN_ELIGIBLE_ROLES:
                if role not in seen and re.search(rf"\b{re.escape(role)}\b", cell):
                    seen.add(role)
                    break

    # W4 — fallback prose/bullet: nếu model KHÔNG format §10 thành bảng Markdown
    # (chỉ liệt kê dạng gạch đầu dòng hoặc câu văn), vòng quét theo "|" ở trên sẽ
    # rỗng. Quét lại toàn bộ section text để bắt tên role hợp lệ theo word-boundary,
    # tránh để regen loop âm thầm no-op khi model nhỏ format lỏng.
    if not seen:
        lowered = section_text.lower()
        for role in _REGEN_ELIGIBLE_ROLES:
            if re.search(rf"\b{re.escape(role)}\b", lowered):
                seen.add(role)

    # Trả về theo thứ tự WORKFLOW_STEPS để đảm bảo dependency được tôn trọng:
    # agent phụ thuộc vào agent khác sẽ được re-gen sau agent đó.
    return [r for r in WORKFLOW_STEPS if r in seen]


def _detect_db_type(tech_stack: list[str] | None) -> str:
    """Nhận diện loại cơ sở dữ liệu từ danh sách tech_stack.

    Trả về:
        'nosql'   — chỉ có NoSQL (MongoDB, DynamoDB, Cassandra, Redis...)
        'sql'     — chỉ có SQL (PostgreSQL, MySQL, SQL Server, SQLite...)
        'both'    — có cả SQL và NoSQL
        'unknown' — không nhận diện được hoặc tech_stack trống

    Kết quả được dùng để chọn đúng extra_instruction cho DBA và DA agent.
    """
    if not tech_stack:
        return "unknown"
    _nosql_kw = {"mongodb", "mongo", "nosql", "dynamodb", "cassandra",
                 "redis", "firestore", "couchdb", "elasticsearch"}
    _sql_kw   = {"postgresql", "mysql", "postgres", "sql server",
                 "sqlite", "mssql", "oracle", "mariadb"}
    combined  = " ".join(t.lower() for t in tech_stack)
    has_nosql = any(kw in combined for kw in _nosql_kw)
    has_sql   = any(kw in combined for kw in _sql_kw)
    if has_nosql and has_sql:
        return "both"
    if has_nosql:
        return "nosql"
    if has_sql:
        return "sql"
    return "unknown"


@functools.lru_cache(maxsize=32)
def _fence_pattern(language: str) -> re.Pattern:
    """Compile và cache pattern regex để extract content trong code fence ngôn ngữ chỉ định.

    Dùng @lru_cache để tránh compile lại regex cho cùng ngôn ngữ nhiều lần.
    Pattern khớp cả dạng: ```typescript ... ``` và ```TypeScript ... ```.
    """
    return re.compile(
        rf"```(?:{re.escape(language)}|{re.escape(language.lower())})\s*\n(.*?)(?:\n```\s*$|\n```\s*\Z)",
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )


_FENCE_FALLBACK_RE = re.compile(
    r"```\w*\s*\n(.*?)(?:\n```\s*$|\n```\s*\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)


def _strip_code_fence(content: str, language: str) -> str:
    """Loại bỏ code fence bao ngoài (```lang...```) từ content LLM trả về.

    Cần thiết vì một số model bọc toàn bộ output trong code fence, trong khi
    output đã chứa các fence nội bộ riêng. Loại bỏ fence ngoài để tránh
    markdown bị lồng nhau không hợp lệ khi lưu artifact.
    """
    # Thử extract content giữa ```lang ... ``` bằng pattern đã cache cho ngôn ngữ này.
    for pat in (_fence_pattern(language), _FENCE_FALLBACK_RE):
        m = pat.search(content)
        if m:
            return m.group(1).rstrip()
    # Fallback thủ công: xóa dòng đầu (```lang) và dòng cuối (```) nếu regex không khớp.
    lines = content.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _deloop(text: str, max_repeats: int = 8) -> str:
    """Cắt ngắn các đoạn dòng lặp lại liên tiếp để giảm thiểu lỗi vòng lặp của model.

    Một số coding model nhỏ có xu hướng lặp lại cùng một dòng code (thường là
    import hoặc boilerplate) hàng trăm lần. Hàm này giới hạn tối đa max_repeats
    lần lặp liên tiếp rồi thêm dấu hiệu truncation.
    """
    lines = text.split("\n")
    out: list[str] = []
    prev: str | None = None
    count = 0
    for ln in lines:
        key = ln.rstrip()
        if key and key == prev:
            count += 1
            if count <= max_repeats:
                out.append(ln)
            elif count == max_repeats + 1:
                out.append("// ... (repeated pattern truncated)")
        else:
            prev  = key
            count = 1
            out.append(ln)
    return "\n".join(out)


def _compute_extra_instruction(role: str, tech_stack: list[str] | None) -> str:
    """Sinh câu lệnh bổ sung (extra instruction) đặc thù theo role và tech stack.

    Câu lệnh này được chèn vào planning prompt và file-gen prompt để hướng dẫn
    coding model dùng đúng framework, ngôn ngữ và pattern phù hợp với tech stack.
    Ví dụ: DBA + NoSQL → dùng Mongoose, không dùng SQL DDL.
    """
    if not tech_stack:
        if role == "tech_lead":
            return (
                "Write ARCHITECTURE.md or INTEGRATION.md with technical review, "
                "integration decisions, code standards, and specific code change recommendations."
            )
        return ""

    combined = " ".join(t.lower() for t in tech_stack)
    db_type  = _detect_db_type(tech_stack)

    if role == "dba":
        if db_type == "nosql":
            return (
                "Use MongoDB Mongoose models (TypeScript .ts files). "
                "NO SQL CREATE TABLE or DDL. "
                "Embedded documents and arrays instead of FK/JOINs."
            )
        if db_type == "sql":
            return "Use SQL DDL with CREATE TABLE, FOREIGN KEY references, and CREATE INDEX."
        if db_type == "both":
            return (
                "Create both SQL DDL (schema.sql) AND Mongoose models (.ts). "
                "SQL for relational data, MongoDB for document data."
            )

    # SCAFFOLD mode cho engineer roles (fe/mobile/be): chỉ DỰNG KHUNG codebase —
    # imports, types, signatures, config, wiring — với marker `// TODO: implement`
    # nơi feature logic sẽ được dev "vibe code" trong IDE. KHÔNG viết business logic.
    # (DBA giữ schema thật vì là nền tảng; devsecops giữ infra thật.)
    _scaffold = (
        " — SCAFFOLD ONLY: imports, types/props, signatures, config and route/DB/auth "
        "wiring, with `// TODO: implement` markers where feature logic goes. Do NOT write "
        "business logic or full markup. No license headers."
    )

    if role == "fe":
        if "vue" in combined:
            fw = "Vue 3 component (Composition API, TypeScript, template)"
        elif "angular" in combined:
            fw = "Angular component (TypeScript, decorators, template)"
        elif "svelte" in combined:
            fw = "Svelte component (TypeScript, reactive syntax)"
        else:
            fw = "React/TypeScript component (JSX, hooks, typed props)"
        return "Scaffold a " + fw + _scaffold

    if role == "mobile":
        if "react native" in combined or "expo" in combined:
            fw = "React Native component (TypeScript, JSX, StyleSheet)"
        elif "flutter" in combined or "dart" in combined:
            fw = "Flutter/Dart widget"
        else:
            fw = "Flutter/Dart widget or React Native component"
        return "Scaffold a " + fw + _scaffold

    if role == "be":
        if "fastapi" in combined or (
            "python" in combined and "django" not in combined and "flask" not in combined
        ):
            fw = "FastAPI router + Pydantic models (async)"
        elif "django" in combined:
            fw = "Django/DRF view + serializers"
        elif "flask" in combined:
            fw = "Flask route + request parsing"
        elif "express" in combined and "nest" not in combined:
            fw = "Express.js route + TypeScript types"
        elif "spring" in combined or "java" in combined or "kotlin" in combined:
            fw = "Spring Boot controller + service (annotated)"
        elif "go" in combined or "golang" in combined:
            fw = "Go HTTP handler + struct types"
        else:
            fw = "NestJS controller + service + DTO"
        return "Scaffold a " + fw + _scaffold

    if role == "devsecops":
        return (
            "Write real Kubernetes manifests (Deployment, Service, Ingress, ConfigMap, Secret), "
            "Dockerfile, and CI/CD pipeline YAML with working config. "
            "All sensitive env vars must go in K8s Secret objects referenced via secretKeyRef — "
            "never in plain env: blocks. No license headers."
        )
    if role == "tech_lead":
        return (
            "Write ARCHITECTURE.md or INTEGRATION.md with technical review, "
            "integration decisions, code standards, and specific code change recommendations."
        )
    return ""


def _plan_code_files(
    role: str,
    agent: AgentConfig,
    ollama_base_url: str,
    user_input: str,
    tech_stack: list[str] | None,
    context_summary: str,
    extra_instruction: str = "",
    tl_tasks: list[dict] | None = None,
) -> list[dict]:
    """Gọi CODING_PLANNER_MODEL để lập kế hoạch danh sách file cần sinh.

    Nhận task, tech stack, context tóm tắt và danh sách TL task (nếu có),
    rồi trả về JSON array mô tả các file: [{filename, description, language}].
    Nếu parse JSON thất bại hoặc model không trả về kết quả, trả về [].
    """
    planner = ChatOllama(
        model=CODING_PLANNER_MODEL,
        base_url=ollama_base_url,
        temperature=0.1,
        num_ctx=min(8192, OLLAMA_NUM_CTX),
        request_timeout=float(OLLAMA_REQUEST_TIMEOUT),
    )
    stack_str = ", ".join(tech_stack) if tech_stack else "not specified"
    # Nếu có TL task board, chèn danh sách task vào planning prompt
    # để model biết cần tạo file nào để cover hết task (tối đa 15 task đầu tiên).
    if tl_tasks:
        task_lines = "\n".join(f"  - {t['task']}" for t in tl_tasks[:15])
        tl_tasks_section = (
            "IMPORTANT — Team Lead Task Board: your file list MUST cover all of these tasks:\n"
            + task_lines
        )
    else:
        tl_tasks_section = ""
    prompt = _PLAN_PROMPT_TMPL.format(
        role_name=agent.name,
        max_files=_MAX_FILES_PER_ROLE,
        user_input=user_input[:500],
        tech_stack=stack_str,
        context_summary=context_summary[:1000],
        tl_tasks_section=tl_tasks_section,
        extra=f"Constraint: {extra_instruction}" if extra_instruction else "",
    )
    try:
        resp = planner.invoke([HumanMessage(content=prompt)])
        content = str(resp.content)
        m = re.search(r"\[.*?\]", content, re.DOTALL)
        if m:
            files = _json.loads(m.group(0))
            return [
                f for f in files
                if isinstance(f, dict) and "filename" in f
            ][:_MAX_FILES_PER_ROLE]
    except Exception as exc:
        logger.warning("File planning failed for %s: %s", role, exc)
    return []


def _generate_one_file(
    filename: str,
    description: str,
    language: str,
    agent: AgentConfig,
    ollama_base_url: str,
    tech_stack: list[str] | None,
    extra_instruction: str = "",
    context_snippet: str = "",
) -> str:
    """Sinh nội dung đầy đủ của một file bằng coding model.

    Gọi LLM với prompt tập trung vào filename + description + tech stack.
    Sau khi nhận response, áp dụng _strip_code_fence() và _deloop()
    để làm sạch output trước khi lưu.

    Trả về chuỗi nội dung file (không có code fence bao ngoài).
    Nếu LLM call thất bại, trả về comment lỗi để không làm crash workflow.
    """
    llm = ChatOllama(
        model=agent.model,
        base_url=ollama_base_url,
        temperature=0.1,
        num_ctx=_get_num_ctx(agent.model),
        request_timeout=float(OLLAMA_REQUEST_TIMEOUT),
    )
    stack_str = ", ".join(tech_stack) if tech_stack else "not specified"
    ctx_block  = f"\n\nReview context (previous agents):\n{context_snippet}" if context_snippet else ""
    prompt = _FILE_GEN_PROMPT_TMPL.format(
        filename=filename,
        description=description,
        language=language,
        tech_stack=stack_str,
        extra=extra_instruction,
    ) + ctx_block
    try:
        resp = llm.invoke([
            SystemMessage(content=(
                COMMON_AGENT_RULES
                + "\n\n---\n\n"
                + agent.system_prompt
                + "\n\nWhen generating a single file, output only the requested file content. "
                  "Do not repeat other sections or produce a full project overview."
            )),
            HumanMessage(content=prompt),
        ])
        result = str(resp.content)
        # W3: nếu coding role dùng reasoning model, loại bỏ <think>...</think>
        # TRƯỚC khi lưu — nếu không chain-of-thought bị ghi thẳng vào file artifact.
        if any(m in agent.model.lower() for m in _REASONING_MODELS):
            result = _strip_thinking(result)
        result = _strip_code_fence(result, language)   # remove nested fences
        result = _deloop(result)                        # remove infinite loops
        return result
    except Exception as exc:
        logger.warning("File gen failed for %s (%s): %s", agent.name, filename, exc)
        return f"// Error generating {filename}: {exc}"


def _generate_artifacts_multi_turn(
    role: str,
    agent: AgentConfig,
    ollama_base_url: str,
    context: str,
    user_input: str,
    tech_stack: list[str] | None,
    extra_instruction: str = "",
) -> str:
    """Sinh toàn bộ artifact cho một coding role theo quy trình 2 pha.

    Pha 1 — Lập kế hoạch:
        CODING_PLANNER_MODEL nhận task, tech stack, context tóm tắt và TL task board,
        trả về JSON array danh sách file cần sinh.

    Pha 2 — Sinh từng file:
        Coding model sinh nội dung từng file độc lập với context tập trung.
        Nếu còn TL task chưa được cover sau pha 2 → chạy follow-up pass để lên
        kế hoạch thêm file mới và sinh chúng (bỏ qua file trùng tên).

    Kết thúc:
        Append bảng Task Completion Checklist đối chiếu mọi TL task với kết quả.

    Trả về:
        Chuỗi markdown tổng hợp với ### FILE: sections và checklist cuối.
    """
    context_summary = context[-1200:]

    # Trích xuất danh sách TL task từ context để tracking coverage và lập kế hoạch.
    tl_tasks = _extract_tl_tasks(role, context)
    if tl_tasks:
        logger.info("Role=%s: found %d TL tasks to cover", role, len(tl_tasks))

    # Pha 1: Gọi CODING_PLANNER_MODEL để lập kế hoạch danh sách file.
    files = _plan_code_files(
        role, agent, ollama_base_url, user_input, tech_stack,
        context_summary, extra_instruction, tl_tasks=tl_tasks or None,
    )

    if not files:
        logger.warning("No files planned for %s, fallback to single call", role)
        output = _call_agent(agent, ollama_base_url, context[-2000:])
        if tl_tasks:
            output += _build_task_checklist(tl_tasks, [])
        return output

    logger.info(
        "Planned %d files for %s: %s",
        len(files), role, [f.get("filename") for f in files],
    )

    parts: list[str] = [f"## {agent.name}\n"]
    # Review roles benefit from seeing previous agent outputs
    _review_roles = {"tech_lead", "devsecops"}
    ctx_snippet = context[-1500:] if role in _review_roles else ""

    # Theo dõi tất cả file trên cả hai pha để dedup và tạo checklist chính xác.
    all_files: list[dict] = list(files)

    for file_info in files:
        filename    = file_info.get("filename", "unknown")
        description = file_info.get("description", "")
        language    = file_info.get("language", "text")
        logger.info("Generating file %s for role=%s", filename, role)
        file_content = _generate_one_file(
            filename, description, language, agent, ollama_base_url, tech_stack,
            extra_instruction, ctx_snippet,
        )
        parts.append(f"\n### FILE: {filename}\n```{language}\n{file_content}\n```\n")

    # Pha 2 (follow-up): nếu còn TL task chưa được cover sau pha 1,
    # lên kế hoạch thêm file mới và sinh chúng để đảm bảo coverage đầy đủ.
    if tl_tasks:
        uncovered = _find_uncovered_tasks(tl_tasks, all_files)
        if uncovered:
            logger.info(
                "Role=%s: %d TL tasks uncovered after initial pass — running follow-up loop",
                role, len(uncovered),
            )
            followup_input = (
                user_input
                + "\n\nFOCUS: The following Team Lead tasks were not yet addressed — "
                "implement them now:\n"
                + "\n".join(f"- {t['task']}" for t in uncovered)
            )
            followup_files = _plan_code_files(
                role, agent, ollama_base_url, followup_input, tech_stack,
                context_summary, extra_instruction, tl_tasks=uncovered,
            )
            for file_info in followup_files:
                fname = file_info.get("filename", "")
                # Bỏ qua file đã sinh trong pha 1 để tránh ghi đè.
                if any(f.get("filename") == fname for f in all_files):
                    logger.info("Follow-up skip duplicate file %s for role=%s", fname, role)
                    continue
                description = file_info.get("description", "")
                language    = file_info.get("language", "text")
                logger.info("Follow-up: generating file %s for role=%s", fname, role)
                file_content = _generate_one_file(
                    fname, description, language, agent, ollama_base_url, tech_stack,
                    extra_instruction, ctx_snippet,
                )
                parts.append(f"\n### FILE: {fname}\n```{language}\n{file_content}\n```\n")
                all_files.append(file_info)

    # Thêm bảng Task Completion Checklist vào cuối output để đối chiếu
    # mọi TL task với file đã sinh — ✅ Done, ⏳ Partial, hoặc ❌ Deferred.
    if tl_tasks:
        parts.append(_build_task_checklist(tl_tasks, all_files))

    return "\n".join(parts)


def _call_agent(
    agent: AgentConfig,
    ollama_base_url: str,
    context: str,
) -> str:
    """Gọi LLM một lần với system prompt đầy đủ của agent + context đã tổng hợp.

    Dùng cho tất cả non-artifact roles (ba, pm, sa, ta, designer, tl, da,
    tester, clarifier) — những role sinh báo cáo một lần thay vì per-file loop.

    Nếu model trả về output < 30 ký tự (thường do model nhỏ bị overflow context),
    thực hiện retry với 1.500 ký tự cuối của context để giảm tải.
    """
    llm = ChatOllama(
        model=agent.model,
        base_url=ollama_base_url,
        temperature=0.1,
        num_ctx=_get_num_ctx(agent.model),
        request_timeout=float(OLLAMA_REQUEST_TIMEOUT),
    )
    # Kiểm tra một lần — dùng lại cho cả lần gọi chính và lần retry.
    is_reasoning = any(m in agent.model.lower() for m in _REASONING_MODELS)

    # Build system message once — reused in primary call and retry to avoid duplication.
    sys_msg  = SystemMessage(content=COMMON_AGENT_RULES + "\n\n---\n\n" + agent.system_prompt)
    response = llm.invoke([sys_msg, HumanMessage(content=context)])
    result   = str(response.content)

    # Strip chain-of-thought blocks from reasoning models.
    if is_reasoning:
        result = _strip_thinking(result)

    if len(result.strip()) < 30:
        logger.warning(
            "Agent '%s' (model=%s) trả về output rất ngắn (%d chars) — retry với context rút gọn.",
            agent.name, agent.model, len(result.strip()),
        )
        trimmed = context[-1500:] if len(context) > 1500 else context
        response = llm.invoke([sys_msg, HumanMessage(content=trimmed)])
        result = str(response.content)
        if is_reasoning:
            result = _strip_thinking(result)

    return result


# ── Context builder — shared by _build_node and run_single_step ──────────────────


def _build_context_parts(
    role: str,
    agent: AgentConfig,
    user_input: str,
    step_outputs: dict[str, str],
    tech_stack: list[str] | None,
    dep_max_chars: int,
    rag_enabled: bool = False,
    rag_api_url: str = "",
    project: str | None = None,
    rag_top_k: int = RAG_TOP_K,
    extra_context: str | None = None,
) -> list[str]:
    """Lắp ráp danh sách context-parts cho bất kỳ lần gọi agent nào.

    Đây là hàm trung tâm tập trung logic build context, dùng chung cho cả
    _build_node (workflow chính) và run_single_step (single-step endpoint).
    Đảm bảo hai luồng code luôn tạo context nhất quán — bao gồm:
    - Gợi ý ngày hiện tại cho planning roles (ba, pm, sa, ta) tránh dùng ngày quá khứ.
    - Gợi ý kiểu DB (nosql/sql/both) cho dba/da để chọn đúng dialect schema.
    - Context bổ sung tùy chọn từ caller.
    """
    parts: list[str] = [
        f"## Mục tiêu kinh doanh / Yêu cầu đầu vào\n{user_input}"
    ]

    # Chèn ngày hiện tại cho planning roles để tránh sinh timeline có ngày quá khứ.
    # Chỉ áp dụng cho ba, pm, sa, ta — các role sinh timeline và sprint plan.
    if role in {"pm", "ba", "sa", "ta"}:
        now = datetime.now()
        parts.append(
            f"\n## Ngày hiện tại\n"
            f"{now.strftime('%d/%m/%Y')} (tháng {now.month} năm {now.year}). "
            f"Mọi timeline, sprint, milestone PHẢI bắt đầu từ ngày này trở về sau. "
            f"Không dùng bất kỳ ngày nào trong quá khứ."
        )

    if tech_stack:
        stack_lines = "\n".join(f"- {item}" for item in tech_stack)
        parts.append(
            f"\n## Công nghệ bắt buộc\n"
            f"Các công nghệ sau BẮT BUỘC phải sử dụng. Không đề xuất lựa chọn khác "
            f"trừ khi được yêu cầu rõ ràng.\n{stack_lines}"
        )

    for dep in agent.depends_on:
        if dep in step_outputs:
            dep_name = AGENTS[dep].name
            parts.append(
                f"\n## Kết quả {dep_name}\n{_truncate(step_outputs[dep], dep_max_chars)}"
            )

    if extra_context:
        parts.append(f"\n## Context bổ sung\n{extra_context}")

    if rag_enabled and rag_api_url:
        _hint = agent.rag_query_hint or agent.name
        rag_text = _query_rag(rag_api_url, f"{_hint}: {user_input}", project, rag_top_k)
        if rag_text:
            parts.append(f"\n## Context từ Knowledge Base\n{_truncate(rag_text)}")

    # DB-type hints steer DBA/DA to the correct schema dialect.
    if role in {"dba", "da"} and tech_stack:
        db_type = _detect_db_type(tech_stack)
        if role == "dba":
            if db_type == "nosql":
                parts.append(
                    "\n## DB Schema: NoSQL Only\n"
                    "Tech stack chỉ dùng NoSQL (MongoDB). Tạo Mongoose schema files (.ts). "
                    "KHÔNG tạo SQL DDL hay CREATE TABLE. "
                    "Dùng embedded documents và arrays thay cho JOINs/FK."
                )
            elif db_type == "sql":
                parts.append(
                    "\n## DB Schema: SQL Relational\n"
                    "Tech stack dùng relational DB. Tạo SQL DDL với CREATE TABLE, "
                    "FOREIGN KEY relationships, và CREATE INDEX."
                )
            elif db_type == "both":
                parts.append(
                    "\n## DB Schema: Mixed (SQL + NoSQL)\n"
                    "Tech stack dùng cả SQL và NoSQL. Tạo SQL DDL schema VÀ "
                    "Mongoose models riêng cho từng data store."
                )
        elif role == "da":
            if db_type == "nosql":
                parts.append(
                    "\n## Data Analysis: NoSQL Only\n"
                    "Tech stack chỉ dùng MongoDB NoSQL. Sử dụng aggregation pipeline "
                    "($match, $group, $project, $lookup). "
                    "Nếu cần cross-collection analysis: BE export CSV trước, DA đọc CSV bằng Python/pandas. "
                    "KHÔNG dùng SQL SELECT/FROM/WHERE."
                )
            elif db_type == "sql":
                parts.append(
                    "\n## Data Analysis: SQL\n"
                    "Tech stack dùng relational DB. Phân tích bằng SQL queries với "
                    "GROUP BY, window functions, aggregates, CTEs."
                )

    return parts


# ── Factory tạo node ───────────────────────────────────────────────────────────────

def _build_node(role: str):
    """Factory function tạo LangGraph node cho một role SDLC.

    Sử dụng closure để mỗi node nắm giữ AgentConfig riêng của nó mà không cần
    tra cứu global dict lúc chạy — an toàn hơn và nhanh hơn.

    __name__ được set tường minh thành "{role}_node" để LangGraph hiển thị
    đúng tên role trong trace/debug thay vì tên hàm chung "node_fn".
    """
    agent = AGENTS[role]

    def node_fn(state: SDLCState) -> dict:
        logger.info("Step %d | %s | model=%s",
                    agent.step_id, agent.name, agent.model)

        step_outputs: dict[str, str] = state.get("step_outputs", {})
        tech_stack = state.get("tech_stack")
        # Lấy giới hạn ký tự per-dep cho role này; dùng MAX_PREV_OUTPUT_CHARS nếu không có override.
        dep_max_chars = _PER_DEP_CHARS.get(role, MAX_PREV_OUTPUT_CHARS)

        # 1. Tổng hợp context
        context_parts = _build_context_parts(
            role=role,
            agent=agent,
            user_input=state["user_input"],
            step_outputs=step_outputs,
            tech_stack=tech_stack,
            dep_max_chars=dep_max_chars,
            rag_enabled=state.get("rag_enabled", False),
            rag_api_url=state.get("rag_api_url", ""),
            project=state.get("project"),
            rag_top_k=state.get("rag_top_k", RAG_TOP_K),
        )

        # Tính extra instruction đặc thù theo role và tech stack,
        # chèn vào planning prompt và file-gen prompt.
        extra_instruction = _compute_extra_instruction(role, tech_stack)

        context = "\n".join(context_parts)
        ollama_url = state.get("ollama_base_url", OLLAMA_BASE_URL)

        # 3. Gọi LLM
        error_msg: str | None = None
        try:
            if role in _ARTIFACT_ROLES:
                # Artifact roles: dùng quy trình 2 pha (planner + per-file loop).
                output = _generate_artifacts_multi_turn(
                    role, agent, ollama_url, context,
                    state["user_input"], tech_stack, extra_instruction,
                )
            else:
                output = _call_agent(agent, ollama_url, context)
        except Exception as exc:
            logger.error("Agent '%s' gặp lỗi: %s", role, exc)
            output = f"[LỖI trong {role}] {exc}"
            error_msg = f"Bước '{role}' thất bại: {exc}"

        logger.info("Step %d | %s | output_len=%d",
                    agent.step_id, agent.name, len(output))

        result: dict = {
            "step_outputs": {role: output},
            "completed_steps": [role],
        }
        if error_msg:
            result["error"] = error_msg
        return result

    node_fn.__name__ = f"{role}_node"
    return node_fn


# ── Xây dựng đồ thị ───────────────────────────────────────────────────────────────

def build_workflow():
    """Biên dịch và trả về SDLC LangGraph StateGraph.

    Thêm node theo thứ tự WORKFLOW_STEPS, thêm cạnh tuần tự giữa các node
    liên tiếp để đảm bảo luồng SDLC từ BA đến Clarifier luôn chạy đúng thứ tự.

    Trả về:
        CompiledGraph: LangGraph runnable đã compile, sẵn sàng để stream() hoặc invoke().
    """
    graph: StateGraph = StateGraph(SDLCState)

    for role in WORKFLOW_STEPS:
        graph.add_node(role, _build_node(role))

    graph.set_entry_point(WORKFLOW_STEPS[0])

    for i in range(len(WORKFLOW_STEPS) - 1):
        graph.add_edge(WORKFLOW_STEPS[i], WORKFLOW_STEPS[i + 1])

    graph.add_edge(WORKFLOW_STEPS[-1], END)

    return graph.compile()


# ── Singleton workflow — chỉ compile một lần, cache module-level ────────────────
# Double-checked locking: kiểm tra _workflow trước lock (fast path), kiểm tra lại
# sau lock (safe path). Đảm bảo StateGraph chỉ được compile đúng một lần dù
# nhiều HTTP request đến đồng thời lúc container vừa khởi động.

_workflow = None
_workflow_lock = threading.Lock()


def get_workflow():
    global _workflow
    if _workflow is None:
        with _workflow_lock:
            if _workflow is None:  # kiểm tra lại bên trong lock
                _workflow = build_workflow()
    return _workflow


# ── Chạy từng bước đơn lẻ (dùng bởi endpoint /agent/{role}) ────────────────

def run_single_step(
    role: str,
    user_input: str,
    project: str | None = None,
    extra_context: str | None = None,
    prev_outputs: dict[str, str] | None = None,
    tech_stack: list[str] | None = None,
    rag_enabled: bool = True,
    rag_top_k: int = RAG_TOP_K,
    ollama_base_url: str = OLLAMA_BASE_URL,
    rag_api_url: str = RAG_API_URL,
) -> str:
    """Chạy một agent đơn lẻ mà không cần khởi động toàn bộ workflow.

    Xây dựng context giống hệt như workflow node sẽ thấy:
    - Output rút gọn của các agent phụ thuộc (từ prev_outputs)
    - extra_context tùy chọn
    - RAG context tùy chọn (nếu rag_enabled=True)

    Được dùng bởi:
    - POST /agent/{role}: test một agent đơn lẻ qua API
    - Clarifier Regen Loop: re-run agent cụ thể với outputs đã cập nhật

    Trả về:
        str: Output markdown của agent.
    """
    if role not in AGENTS:
        raise ValueError(
            f"Unknown role '{role}'. Valid roles: {list(AGENTS.keys())}")

    agent = AGENTS[role]
    step_outputs: dict[str, str] = prev_outputs or {}

    # Áp dụng cùng giới hạn ký tự per-dep như workflow node để tránh overflow context.
    dep_max_chars = _PER_DEP_CHARS.get(role, MAX_PREV_OUTPUT_CHARS)
    context_parts = _build_context_parts(
        role=role,
        agent=agent,
        user_input=user_input,
        step_outputs=step_outputs,
        tech_stack=tech_stack,
        dep_max_chars=dep_max_chars,
        rag_enabled=rag_enabled,
        rag_api_url=rag_api_url,
        project=project,
        rag_top_k=rag_top_k,
        extra_context=extra_context,
    )
    context = "\n".join(context_parts)
    if role in _ARTIFACT_ROLES:
        extra_instruction = _compute_extra_instruction(role, tech_stack)
        return _generate_artifacts_multi_turn(
            role,
            agent,
            ollama_base_url,
            context,
            user_input,
            tech_stack,
            extra_instruction,
        )
    return _call_agent(agent, ollama_base_url, context)

