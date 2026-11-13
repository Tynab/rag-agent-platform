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
       - gợi ý kiểu DB (nosql/sql/both) cho dba và da (cả hai đều có gợi ý 'both')
       - extra_context tùy chọn

  2. Bổ sung RAG (tùy chọn, khi rag_enabled=True)
       _query_rag() gọi POST /retrieve đến rag-api với rag_query_hint của agent
       (chỉ truy hồi chunk nguyên văn, không sinh LLM phía rag-api).
       Kết quả được cắt ngắn và thêm vào context. Lỗi RAG không dừng workflow.

  3. Gọi LLM (bắt buộc)
       Artifact roles (fe, mobile, be, dba, da, tech_lead, devsecops):
           _generate_artifacts_multi_turn() — 2 pha sinh code:
           Pha 1: CODING_PLANNER_MODEL lập kế hoạch danh sách file,
                  tham chiếu TOÀN BỘ TL task board để không bỏ sót task
                  (_format_tl_task_lines: cắt chữ từng task theo ngân sách
                  _TL_TASKS_PROMPT_CHARS, không cắt danh sách). Context gửi planner
                  rút gọn bằng _compact_context (đề bài + tech stack + đuôi context).
           Pha 2: coding model sinh từng file độc lập với context tập trung
                  (review roles tech_lead/devsecops nhận thêm context rút gọn
                  bằng _compact_context). Nếu còn TL task chưa cover → follow-up
                  pass sinh thêm file.
           Kết thúc: _build_task_checklist() tạo bảng ✅/⏳ đối chiếu TL tasks.
           Planner không trả kế hoạch → fallback _call_agent() một lần với context
           rút gọn + toàn bộ TL task board.
       Các role còn lại:
           _call_agent() — gọi LLM một lần với toàn bộ context.
           Nếu output < 30 ký tự → retry với context rút gọn 1.500 ký tự
           (_compact_context: giữ đề bài + tech stack ở đầu, thêm đuôi context).

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

Gọi LLM — llm.py
-----------------
    Mọi ChatOllama được tạo qua llm.make_chat(): timeout httpx thật
    (OLLAMA_REQUEST_TIMEOUT qua client_kwargs) + trần num_predict theo loại lời gọi
    (PLANNER_/FILEGEN_/REPORT_NUM_PREDICT, có headroom cho reasoning model) +
    FILEGEN_THINK cho lời gọi sinh file. Xem docstring llm.py.

Xử lý reasoning models
-----------------------
    llm.is_reasoning_model(): khớp chuỗi con tên model reasoning
    (phi4-mini-reasoning, phi4-reasoning, qwq, deepseek-r1, qwen3,
    north-mini-code, gemma4, mistral-medium-3.5).
    _strip_thinking(): loại bỏ <think>...</think> khỏi output trước khi
    lưu vào step_outputs — chỉ giữ lại phần nội dung thực sự (kể cả block
    <think> bị cắt cụt khi chạm trần num_predict).

Lỗi sinh file (artifact roles)
-------------------------------
    _generate_one_file() raise FileGenerationError (lỗi gọi LLM hoặc nội dung rỗng)
    thay vì trả comment lỗi làm nội dung file. File lỗi KHÔNG được ghi thành section
    "### FILE:" và KHÔNG tính là đã cover TL task (follow-up pass được thử lại file
    đó một lần). Output role có thêm section FILE_GEN_ERRORS_HEADER liệt kê từng file
    lỗi + lỗi ngắn gọn. Không file nào sinh được → output bắt đầu "[LỖI trong {role}]"
    (app.py map tiền tố này vào failed_steps) và node ghi "error".

Chống vòng lặp output
----------------------
    _deloop(text, language, filename=...): cắt các dòng lặp liên tiếp quá giới hạn.
    Phòng ngừa coding model nhỏ có xu hướng lặp lại boilerplate vô tận. Marker cắt
    dùng cú pháp comment đúng ngôn ngữ (// · # · -- · /* */ · <!-- -->); JSON và
    ngôn ngữ không rõ cú pháp comment → cắt nhưng KHÔNG chèn marker (file vẫn hợp lệ).

TL Task Tracking
----------------
    _extract_tl_tasks(role, tl_output)           → list[dict]  (parse TL output thô)
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

import json as _json
import logging
import operator
import os
import re
import threading
from datetime import datetime
from typing import Annotated, TypedDict

import requests
from agents import AGENTS, MAX_PREV_OUTPUT_CHARS, WORKFLOW_STEPS, AgentConfig
from artifacts import fence_for, find_fenced_blocks
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from llm import filegen_reasoning, hit_token_cap, is_reasoning_model, make_chat, num_predict_for

logger = logging.getLogger("agent-workflow")

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
RAG_API_URL: str = os.environ.get("RAG_API_URL", "http://rag-api:8090")
RAG_TOP_K: int = int(os.environ.get("RAG_TOP_K", "5"))
# Kích thước context window (token) gửi cho Ollama mỗi lần gọi LLM.
# Tăng giá trị này nếu model hỗ trợ window lớn hơn (ví dụ: 65536).
OLLAMA_NUM_CTX: int = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "32768"))
# Timeout (giây) chờ rag-api trả về kết quả mỗi lần gọi POST /retrieve (F10: chỉ
# embed + search, không sinh LLM → không còn phụ thuộc OLLAMA_REQUEST_TIMEOUT của rag-api).
# Timeout của chính các lời gọi LLM (OLLAMA_REQUEST_TIMEOUT) nằm trong llm.py.
RAG_TIMEOUT: int = int(os.environ.get("RAG_TIMEOUT", "600"))

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
    rag_top_k: int                # Số chunk trả về mỗi lần gọi POST /retrieve của rag-api.
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
    Gọi endpoint /retrieve của rag-api và trả về context thô (chunk nguyên văn + nguồn).
    Trả về chuỗi rỗng nếu có lỗi (không gây dừng workflow).

    F10: trước đây gọi /ask → rag-api chạy trọn một lượt CHAT_MODEL chỉ để sinh câu trả
    lời mà node này lại cắt còn 3000 ký tự; với OLLAMA_MAX_LOADED_MODELS=1 mỗi node phải
    swap thêm CHAT_MODEL, và khi RAG_TIMEOUT < OLLAMA_REQUEST_TIMEOUT của rag-api, lượt
    sinh bị bỏ rơi vẫn chiếm slot Ollama duy nhất. /retrieve chỉ embed + search (không
    gọi LLM) và giữ nguyên văn convention/tên riêng trong knowledge base.
    """
    try:
        payload: dict = {"question": question, "top_k": top_k}
        if project:
            payload["project"] = project
        resp = requests.post(
            f"{rag_api_url}/retrieve",
            json=payload,
            timeout=RAG_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("context", "")
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


# Tiêu đề section do _build_context_parts() sinh ra — PHẢI khớp chuỗi trong hàm đó.
# _compact_context() dùng chúng để tách phần "đề bài" (user input + ngày + tech stack,
# luôn nằm đầu context) khỏi phần thân (output agent phụ thuộc, context bổ sung, RAG).
_CTX_STACK_HDR = "\n## Công nghệ bắt buộc\n"
_CTX_BODY_HDRS: tuple[str, ...] = (
    "\n## Kết quả ",
    "\n## Context bổ sung\n",
    "\n## Context từ Knowledge Base\n",
)


def _compact_context(context: str, max_chars: int) -> str:
    """Rút gọn context về tối đa *max_chars* ký tự nhưng GIỮ phần đề bài ở đầu.

    _build_context_parts() luôn đặt user input, ngày hiện tại và tech stack bắt buộc
    ở đầu context; output agent phụ thuộc / RAG / gợi ý DB nằm phía sau. Cắt kiểu
    context[-N:] làm mất đề bài và tech stack → model sinh lạc đề, sai công nghệ.

    Thứ tự ưu tiên:
      1. Block tech stack (ngắn, bắt buộc).
      2. User input — phần đề bài chiếm tối đa ~60% ngân sách nếu có phần thân.
      3. Đuôi phần thân (output agent gần nhất, RAG, gợi ý DB) lấp phần còn lại.
    """
    if len(context) <= max_chars:
        return context
    body_starts = [i for i in (context.find(h) for h in _CTX_BODY_HDRS) if i != -1]
    head_end = min(body_starts) if body_starts else len(context)
    head, body = context[:head_end], context[head_end:]

    head_budget = max_chars * 3 // 5 if body else max_chars
    if len(head) > head_budget:
        # Đề bài quá dài: giữ nguyên block tech stack, cắt bớt phần user input phía trước.
        stack_idx = head.find(_CTX_STACK_HDR)
        stack = head[stack_idx:][: head_budget // 2] if stack_idx != -1 else ""
        keep = head_budget - len(stack)
        if stack_idx != -1:
            keep = min(keep, stack_idx)
        head = head[:keep] + stack

    tail_budget = max_chars - len(head) - 1
    if tail_budget <= 0 or not body:
        return head[:max_chars]
    return head + "\n" + body[-tail_budget:]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Loại bỏ chain-of-thought <think>...</think> khỏi output của reasoning model.

    Một số model (qwen3, deepseek-r1, qwq...) phát ra chain-of-thought nội bộ
    trong tag <think>. Người dùng không cần thấy phần này — chỉ giữ lại
    phần nội dung thực sự sau khi quá trình suy luận kết thúc.

    Ngoài block đóng đủ cặp, xử lý hai dạng lệch:
    - "</think>" mồ côi (template đã chèn sẵn "<think>" vào prompt nên output chỉ có
      tag đóng) → bỏ mọi thứ tới hết tag đóng CUỐI CÙNG.
    - "<think>" không có tag đóng (chạm trần num_predict giữa lúc suy luận) → bỏ từ
      tag mở tới hết văn bản. Kết quả có thể rỗng — caller coi rỗng là thất bại /
      output quá ngắn, KHÔNG lưu bản nháp suy luận thành file hay báo cáo.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    closes = list(_THINK_CLOSE_RE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opened = _THINK_OPEN_RE.search(text)
    if opened:
        text = text[:opened.start()]
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


# Ngân sách ký tự context (đã _compact_context: đề bài + tech stack + đuôi context)
# gửi planner và gửi kèm mỗi lần sinh file của review roles (tech_lead, devsecops).
_PLAN_CTX_CHARS = 2_000
_REVIEW_CTX_CHARS = 1_500

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
  If the file itself contains ``` code blocks (e.g. a Markdown doc), wrap it in a ```` fence instead.
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

# Nhận diện dòng heading section trong TL output:
#   - Markdown:        "## 5. Mobile Task Board", "### §6 BE Task Board", "## FE Task Board"
#   - Số/đậm, không #: "**5. Mobile Task Board**", "5. Mobile Task Board", "§5 Mobile ..."
#   - Cả dòng in đậm:  "**Mobile Task Board**"
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_MD_HEADING_NUM_RE = re.compile(r"^(?:\*\*|__)?\s*(?:§\s*)?(\d{1,2})(?:[.)]|\s)")
_NUM_HEADING_RE = re.compile(r"^\s{0,3}(?:\*\*|__)?\s*(?:§\s*(\d{1,2})[.)]?|(\d{1,2})[.)])\s+\S")
_BOLD_HEADING_RE = re.compile(r"^\s{0,3}(?:\*\*[^*]+\*\*|__[^_]+__)\s*:?\s*$")
# Level giả cho heading không có '#' — luôn "thấp" hơn mọi heading Markdown.
_PLAIN_HEADING_LEVEL = 7

# Giá trị priority hợp lệ trong TL task board.
_TL_PRIORITIES: frozenset[str] = frozenset({"P0", "P1", "P2", "HIGH", "MED", "LOW", "H", "M", "L"})


def _heading_info(line: str) -> tuple[int, int | None] | None:
    """Phân loại một dòng có phải heading section hay không.

    Trả về (level, số section): level 1-6 cho heading Markdown '#',
    _PLAIN_HEADING_LEVEL cho heading dạng số/in đậm không có '#'.
    Số section là None nếu tiêu đề không đánh số. Trả về None với dòng bảng,
    văn bản thường, danh sách gạch đầu dòng...
    """
    # Dòng chứa '|' là dòng bảng — kể cả bảng GFM không có '|' đầu dòng ("# | Task | ...").
    if "|" in line:
        return None
    m = _MD_HEADING_RE.match(line)
    if m:
        num = _MD_HEADING_NUM_RE.match(m.group(2))
        return len(m.group(1)), (int(num.group(1)) if num else None)
    m = _NUM_HEADING_RE.match(line)
    if m:
        return _PLAIN_HEADING_LEVEL, int(m.group(1) or m.group(2))
    if _BOLD_HEADING_RE.match(line):
        return _PLAIN_HEADING_LEVEL, None
    return None


def _ends_section(start: tuple[int, int | None], cand: tuple[int, int | None]) -> bool:
    """True nếu heading *cand* kết thúc section bắt đầu bởi heading *start*.

    - Heading Markdown kết thúc section khi cùng cấp hoặc cao hơn (số '#' <= start).
      Heading con sâu hơn ("#### Sprint 1") vẫn thuộc section.
    - Heading không có '#' chỉ kết thúc section cũng không có '#'. Nếu section có
      đánh số thì chỉ dừng ở heading có số lớn hơn — tránh dừng ở danh sách đánh số
      "1. ..." hay nhãn in đậm "**Sprint 1**" bên trong section.
    """
    level, num = start
    c_level, c_num = cand
    if c_level < _PLAIN_HEADING_LEVEL:
        return c_level <= level
    if level < _PLAIN_HEADING_LEVEL:
        return False
    if num is not None:
        return c_num is not None and c_num > num
    return True


def _find_tl_section(role: str, text: str) -> str:
    """Trả về đoạn text của TL task board cho *role* (từ heading tới heading kế tiếp).

    Ưu tiên dòng HEADING chứa từ khóa của role — tránh bắt nhầm tham chiếu chéo
    nằm trong bảng khác (ví dụ "→ see §4 FE Task Board" trong §3 Dependency Map).
    Nếu model format lỏng (tiêu đề không phải heading) → fallback dòng đầu tiên
    không phải dòng bảng có chứa từ khóa. Trả về "" nếu không tìm thấy.
    """
    patterns = _TL_ROLE_SECTIONS.get(role)
    if not patterns:
        return ""
    lines = text.split("\n")
    start = next(
        (i for i, ln in enumerate(lines) if _heading_info(ln) and any(p in ln for p in patterns)),
        None,
    )
    if start is None:
        start = next(
            (i for i, ln in enumerate(lines)
             if not ln.lstrip().startswith("|") and any(p in ln for p in patterns)),
            None,
        )
    if start is None:
        return ""

    start_info = _heading_info(lines[start]) or (_PLAIN_HEADING_LEVEL, None)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        cand = _heading_info(lines[j])
        if cand and _ends_section(start_info, cand):
            end = j
            break
    return "\n".join(lines[start:end])


def _split_md_row(line: str) -> list[str]:
    """Tách một dòng bảng Markdown thành các cell — GIỮ cell rỗng để cột không bị lệch."""
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in row.split("|")]


def _is_md_separator(cells: list[str]) -> bool:
    """True nếu dòng là dòng kẻ phân cách bảng Markdown (|---|:--:|)."""
    non_empty = [c.replace(" ", "") for c in cells if c.strip()]
    return bool(non_empty) and all(re.fullmatch(r":?-+:?", c) for c in non_empty)


def _tl_table_columns(header: list[str]) -> tuple[int | None, int | None, int | None]:
    """Xác định (cột id, cột tên task, cột priority) từ dòng tiêu đề bảng TL."""
    names = [c.strip("*_ ").lower() for c in header]
    task_col = next(
        (names.index(n) for n in ("task", "task name", "task title", "title", "task description", "description")
         if n in names),
        None,
    )
    prio_col = next((k for k, n in enumerate(names) if n.startswith("priority")), None)
    id_col = (
        0 if names and task_col != 0 and names[0] in ("#", "id", "no", "no.", "stt", "task id", "task #")
        else None
    )
    return id_col, task_col, prio_col


def _extract_tl_tasks(role: str, context: str) -> list[dict]:
    """Phân tích bảng TL task board cho *role*.

    *context* nên là output THÔ của TL agent (step_outputs["tl"]) — context tổng hợp
    chỉ giữ 3.000 ký tự đầu của TL nên §5–§7 (Mobile/BE/DBA board) thường bị cắt mất.
    Vẫn nhận context tổng hợp để tương thích ngược.

    Section được cắt tại heading cùng cấp tiếp theo (xem _find_tl_section). Trong
    section, dòng tiêu đề bảng (dòng ngay trước dòng kẻ ---) được dùng để xác định
    cột Task / # / Priority thay vì đếm như một task.

    Trả về:
        list[dict]: Danh sách task, mỗi phần tử có {id, task, priority}.
        Trả về [] nếu không tìm thấy section hoặc bảng không thể parse.
    """
    section_text = _find_tl_section(role, context)
    if not section_text:
        return []

    lines = section_text.split("\n")
    # (id_col, task_col, prio_col) của bảng hiện tại — None nếu bảng không có tiêu đề.
    columns: tuple[int | None, int | None, int | None] | None = None
    tasks: list[dict] = []
    for i, line in enumerate(lines):
        if "|" not in line:
            continue
        row = _split_md_row(line)
        if _is_md_separator(row):
            continue
        # Dòng ngay trước dòng kẻ --- là tiêu đề bảng: dùng để map cột, không phải task.
        if i + 1 < len(lines) and "|" in lines[i + 1] and _is_md_separator(_split_md_row(lines[i + 1])):
            columns = _tl_table_columns(row)
            continue

        id_col, task_col, prio_col = columns or (None, None, None)
        if task_col is not None:
            if task_col >= len(row):
                continue
            task_text = row[task_col]
            tid = row[id_col].lstrip("#").strip() if id_col is not None and row[id_col] else ""
            prio_vals = [row[prio_col]] if prio_col is not None and prio_col < len(row) else []
        else:
            # Bảng không có tiêu đề nhận diện được → heuristic cũ theo vị trí cột.
            cells = [c for c in row if c]
            if len(cells) < 2:
                continue
            first = cells[0]
            if first.lower() in ("#", "task", "id", "no") or first.startswith("-"):
                continue
            # Nếu cột đầu là số thứ tự thì tên task ở cột 2.
            task_idx = 1 if first.lstrip("#").strip().isdigit() else 0
            task_text = cells[task_idx]
            tid = first.lstrip("#").strip() if task_idx == 1 else ""
            # Quét các cột sau tên task để tìm P0/P1/P2/HIGH/MED/LOW.
            prio_vals = [cells[task_idx + k] for k in (3, 4, 5) if task_idx + k < len(cells)]

        task_text = task_text.strip()
        if not task_text or len(task_text) < 4 or task_text.lower() in ("task", "description"):
            continue
        priority = "—"
        for val in prio_vals:
            tokens = val.strip("*_ ").upper().split()
            if tokens and tokens[0] in _TL_PRIORITIES:
                priority = tokens[0]
                break
        tasks.append({"id": tid or str(len(tasks) + 1), "task": task_text, "priority": priority})
    return tasks


# Ngân sách ký tự cho danh sách TL task chèn vào prompt planner / fallback single-call.
# Planner chạy num_ctx 8.192 token với PLANNER_NUM_PREDICT mặc định 2.048 (×2 headroom
# khi CODING_PLANNER_MODEL là reasoning model) → prompt còn ~4.000 token; 3.000 ký tự
# task ≈ 800–1.000 token, cộng phần còn lại của prompt (~3.500 ký tự) vẫn dư.
_TL_TASKS_PROMPT_CHARS = 3_000
# Độ rộng sàn khi buộc phải cắt chữ task. Coverage (_find_uncovered_tasks) khớp từ
# khóa của TOÀN BỘ chữ task với filename + description planner trả về → cắt quá ngắn
# khiến planner không thấy từ khóa và task bị coi là chưa cover.
_TL_TASK_MIN_CHARS = 40


def _tl_task_width(texts: list[str], budget: int, overhead: int) -> int:
    """Độ rộng lớn nhất W sao cho Σ min(len(task), W) + overhead mỗi dòng <= *budget*.

    Chỉ task DÀI HƠN W bị cắt — task ngắn giữ nguyên, phần ngân sách chúng không dùng
    nhường cho task dài. Không cần cắt → trả về độ dài task dài nhất. Không W nào vừa
    → _TL_TASK_MIN_CHARS (danh sách có thể vượt budget; xem _format_tl_task_lines).
    """
    def total(width: int) -> int:
        return sum(min(len(t), width) + overhead for t in texts)

    longest = max(len(t) for t in texts)
    if total(longest) <= budget:
        return longest
    lo, hi = _TL_TASK_MIN_CHARS, longest      # tìm nhị phân W lớn nhất còn vừa budget
    if total(lo) > budget:
        return lo
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if total(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _format_tl_task_lines(
    tasks: list[dict], budget: int = _TL_TASKS_PROMPT_CHARS, bullet: str = "  - ",
) -> str:
    """Định dạng TOÀN BỘ *tasks* thành danh sách bullet vừa ngân sách *budget* ký tự.

    Cắt CHỮ của task thay vì cắt DANH SÁCH: tl_tasks[:15] cũ làm board > 15 task luôn
    "chưa hoàn thành" vì checklist và follow-up đối chiếu với mọi task nhưng planner
    chỉ thấy 15 task đầu. Danh sách vừa budget → giữ nguyên mọi task; không vừa → chỉ
    cắt các task dài nhất về cùng một độ rộng (_tl_task_width, đánh dấu "…").

    Board quá lớn (độ rộng chạm sàn _TL_TASK_MIN_CHARS) có thể vượt *budget*; chỉ khi
    vượt gấp đôi — vài trăm task, ngoài khả năng của một lần plan — mới bỏ các task
    cuối và ghi rõ số task bị lược để prompt planner không tràn num_ctx.
    """
    if not tasks:
        return ""
    texts = [" ".join(str(task.get("task", "")).split()) for task in tasks]
    width = _tl_task_width(texts, budget, len(bullet) + 1)
    lines: list[str] = []
    used = 0
    for i, text in enumerate(texts):
        if len(text) > width:
            text = text[: width - 1].rstrip() + "…"
        line = bullet + text
        if used + len(line) + 1 > 2 * budget:
            lines.append(f"{bullet}… (+{len(tasks) - i} more Team Lead tasks not listed)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


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


# Nhận diện tên role trong §10 của Clarifier. Mỗi named group = một role thuộc
# _REGEN_ELIGIBLE_ROLES; alias gồm role id, tên hiển thị và dạng viết đầy đủ.
# Dùng (?<!\w)…(?!\w) thay cho \b để "tech_lead" khớp nguyên cụm và "be" không
# khớp bên trong "devsecops"/"because".
_REGEN_ROLE_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?P<tech_lead>tech[\s_-]?lead)"
    r"|(?P<devsecops>dev[\s-]?sec[\s-]?ops|devops)"
    r"|(?P<fe>front[\s-]?end|fe)"
    r"|(?P<be>back[\s-]?end|be)"
    r"|(?P<mobile>mobile)"
    r"|(?P<dba>dba)"
    r"|(?P<da>data[\s-]analy(?:st|sis)|da)"
    r")(?!\w)",
    re.IGNORECASE,
)
# Từ đệm hay đi kèm tên role trong cell ("BE Agent", "Backend Engineering", "FE và Mobile").
_REGEN_FILLER_RE = re.compile(r"(?<!\w)(?:agents?|roles?|engineer(?:ing)?|team|and|or|và)(?!\w)", re.IGNORECASE)
# Giữa hai tên role chỉ có dấu phân cách danh sách ("be and dba", "FE, BE", "FE/Mobile").
# Bắt buộc có dấu phẩy/gạch chéo/"and"… — chỉ khoảng trắng ("should be FE-owned") không tính.
_REGEN_LIST_SEP_RE = re.compile(r"\s*(?:[,/&+]\s*(?:(?:and|or|và)\s+)?|(?:and|or|và)\s+)", re.IGNORECASE)
# Câu phủ định việc re-gen ("No agent needs to be regenerated", "không cần re-gen").
_REGEN_NEGATION_RE = re.compile(
    r"(?<!\w)(?:no|none|nothing|not|never|không)(?!\w)(?:\W+\w+){0,5}?\W+"
    r"(?:re.?gen\w*|required|necessary|needed|cần)",
    re.IGNORECASE,
)
# Dòng heading §10: tiền tố markdown (#, **, __) + số "10" tuỳ chọn, xét ở ĐẦU DÒNG.
_SEC10_LINE_RE = re.compile(
    r"^\s*(?P<md>#{1,6}|\*\*|__)?\s*(?:§\s*)?(?P<num>10\s*[.):\-–—](?!\d))?\s*[*_]*\s*(?P<rest>.*)$"
)
_REGEN_TITLE_RE = re.compile(r"(?P<rec>Recommended\s+)?Re.?generation\b", re.IGNORECASE)
_TABLE_SEP_CELL_RE = re.compile(r":?-{2,}:?")


def _find_regen_section(clarifier_output: str) -> str | None:
    """Trích nội dung §10 Recommended Re-generation List (không gồm dòng heading).

    Chỉ nhận heading ở ĐẦU DÒNG — trích dẫn giữa câu như "BA §10 RTM" hay
    "→ see §10" ở các section trước bị bỏ qua. Thứ tự ưu tiên (lấy dòng cuối
    cùng của mỗi loại vì §10 là section cuối):
      1. heading (#/**/số 10) có tiêu đề "Re-generation";
      2. dòng trần bắt đầu bằng "Recommended Re-generation";
      3. heading markdown/bold đánh số "10." (model đổi/dịch tiêu đề).
    Section kết thúc ở heading markdown cùng cấp trở lên hoặc section đánh số ≥ 11.
    """
    lines = clarifier_output.splitlines()
    titled: int | None = None
    titled_bare: int | None = None
    numbered: int | None = None
    for i, line in enumerate(lines):
        m = _SEC10_LINE_RE.match(line)
        if not m:
            continue
        heading_like = m.group("md") or m.group("num")
        t = _REGEN_TITLE_RE.match(m.group("rest"))
        if t and heading_like:
            titled = i
        elif t and t.group("rec"):
            titled_bare = i
        elif m.group("md") and m.group("num"):
            numbered = i
    start = next((i for i in (titled, titled_bare, numbered) if i is not None), None)
    if start is None:
        return None

    # _SEC10_LINE_RE luôn khớp (mọi thành phần đều tuỳ chọn) — kiểm tra None chỉ để mypy hài lòng.
    start_m = _SEC10_LINE_RE.match(lines[start])
    md = (start_m.group("md") if start_m else None) or ""
    level = len(md) if md.startswith("#") else 6
    stop_re = re.compile(
        rf"^\s*(?:#{{1,{level}}}\s+\S|(?:#{{1,6}}\s*|\*\*|__)?\s*(?:§\s*)?(?:1[1-9]|[2-9]\d)\s*[.)]\s+[*_]*[A-Z])"
    )
    body: list[str] = []
    for line in lines[start + 1:]:
        if stop_re.match(line):
            break
        body.append(line)
    return "\n".join(body)


def _regen_roles_in(text: str, *, prose: bool = False, role_only: bool = False) -> set[str]:
    """Tìm role được phép re-gen nhắc tới trong một cell bảng hoặc một câu văn.

    - Cell chỉ gồm tên role (+ từ đệm "Agent"/"Engineer", dấu phân cách) như "be",
      "FE, Mobile, DBA", "BE (Backend)" → nhận TẤT CẢ role trong cell.
    - Ngược lại, match "yếu" chỉ được nhận khi có ngữ cảnh role: là từ đầu tiên
      của cell bảng ("be (critical)"), theo sau bởi "agent"/"role", nằm trong
      `backtick`, hoặc đứng cạnh một tên role khác qua dấu phân cách danh sách
      ("the be and dba agents"). Match yếu gồm "be"/"da" viết thường (trùng động
      từ "to be"...) và — trong văn xuôi — dạng mô tả "frontend"/"backend"/"data analyst".
    role_only=True: bỏ qua text không phải cell "chỉ gồm tên role" (dùng cho bảng
    không xác định được cột "Agent Role").
    """
    matches = list(_REGEN_ROLE_RE.finditer(text))
    if not matches:
        return set()
    rest = _REGEN_FILLER_RE.sub(" ", _REGEN_ROLE_RE.sub(" ", text))
    if not re.search(r"[^\W_]", rest):
        return {g for m in matches if (g := m.lastgroup)}
    if role_only:
        return set()

    def _weak(m: re.Match) -> bool:
        word = m.group(0)
        if word.lower() in {"be", "da"}:
            return not word.isupper()
        return prose and m.lastgroup in {"fe", "be", "da"} and len(word) > 3

    def _in_list(a: re.Match, b: re.Match) -> bool:
        return bool(_REGEN_LIST_SEP_RE.fullmatch(text[a.end():b.start()]))

    found: set[str] = set()
    for i, m in enumerate(matches):
        # Mỗi nhánh của _REGEN_ROLE_RE là một named group → lastgroup không bao giờ None.
        role = m.lastgroup
        if role is None:
            continue
        if _weak(m):
            before, after = text[:m.start()], text[m.end():]
            has_ctx = (
                (not prose and not re.search(r"[^\W_]", before))
                or re.match(r"\s*(?:agents?|roles?)(?!\w)", after, re.IGNORECASE)
                or (before.endswith("`") and after.startswith("`"))
                or (i > 0 and _in_list(matches[i - 1], m))
                or (i + 1 < len(matches) and _in_list(m, matches[i + 1]))
            )
            if not has_ctx:
                continue
        found.add(role)
    return found


def _parse_clarifier_regen_list(clarifier_output: str) -> list[str]:
    """Parse §10 Recommended Re-generation List từ Clarifier output.

    Tìm heading §10 thật (đầu dòng), đọc bảng và chỉ quét cột "Agent Role";
    mỗi cell có thể chứa nhiều role ("FE, Mobile"), tất cả đều được lấy.
    Chỉ trả về các role thuộc _REGEN_ELIGIBLE_ROLES (loại trừ planning roles
    và clarifier để ngăn infinite loop).
    Kết quả được sắp xếp theo thứ tự WORKFLOW_STEPS (tất định, không phụ thuộc hash seed).
    Trả về list rỗng nếu §10 không có trong output, bảng rỗng, hoặc không có gì để re-gen.
    """
    section_text = _find_regen_section(clarifier_output)
    if section_text is None:
        return []

    # Gom các dòng bảng liên tiếp thành từng bảng; mỗi dòng → list cell (giữ cell rỗng
    # để đúng chỉ số cột khi đối chiếu với header).
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in section_text.splitlines():
        s = line.strip()
        if s.startswith("|") or s.count("|") >= 2:
            if s.startswith("|"):
                s = s[1:]
            if s.endswith("|"):
                s = s[:-1]
            current.append([c.strip() for c in s.split("|")])
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)

    seen: set[str] = set()
    for rows in tables:
        # Header = dòng ngay trước dòng kẻ phân cách (|---|---|); cột role = cell
        # header đầu tiên chứa "agent"/"role". Không có header → chỉ nhận cell
        # "chỉ gồm tên role" để không đọc nhầm cột Reason/Sections.
        sep_idx = next(
            (i for i, r in enumerate(rows) if all(_TABLE_SEP_CELL_RE.fullmatch(c) or not c for c in r)),
            None,
        )
        role_col: int | None = None
        data_rows = rows
        if sep_idx is not None:
            data_rows = rows[sep_idx + 1:]
            if sep_idx > 0:
                role_col = next(
                    (j for j, c in enumerate(rows[sep_idx - 1])
                     if re.search(r"agent|role|vai\s*trò", c, re.IGNORECASE)),
                    None,
                )
        for cells in data_rows:
            row_roles: set[str] = set()
            if role_col is not None and role_col < len(cells):
                row_roles = _regen_roles_in(cells[role_col])
            if not row_roles:
                # Dòng lệch cột hoặc bảng không header: chỉ nhận cell thuần tên role.
                for cell in cells:
                    row_roles |= _regen_roles_in(cell, role_only=True)
            seen |= row_roles

    # W4 — fallback prose/bullet: CHỈ khi §10 không có bảng nào (model liệt kê dạng
    # gạch đầu dòng/câu văn). Bảng rỗng = "không cần re-gen" theo prompt → không
    # fallback. Quét từng câu, bỏ câu phủ định ("No agent needs to be regenerated").
    # Không tách theo ";" để "No re-generation needed; FE, BE are fine" bị bỏ trọn câu.
    if not tables:
        for sentence in re.split(r"(?<=[.!?])\s+|\n", section_text):
            if sentence.strip() and not _REGEN_NEGATION_RE.search(sentence):
                seen |= _regen_roles_in(sentence, prose=True)

    # Trả về theo thứ tự WORKFLOW_STEPS để đảm bảo dependency được tôn trọng:
    # agent phụ thuộc vào agent khác sẽ được re-gen sau agent đó.
    return [r for r in WORKFLOW_STEPS if r in seen]


def _stack_tokens(tech_stack: list[str] | None) -> set[str]:
    """Chuẩn hoá tech_stack thành tập token để so khớp theo TỪ thay vì chuỗi con.

    - Lowercase, tách mỗi mục theo khoảng trắng và dấu phân cách (, ; / + ( ) | : _ -).
    - Bỏ hậu tố ".js"/"js" và số phiên bản: "NestJS" → "nest", "Vue3" → "vue",
      "python3.12" → "python", "go1.22" → "go", "mysql8" → "mysql".
    - Thêm cặp từ liền kề trong cùng một mục: "react native", "spring boot", "sql server".
    Nhờ vậy "go" không còn khớp nhầm "mongodb"/"argocd"/"django" và "java" không
    khớp "javascript".
    """
    tokens: set[str] = set()
    for item in tech_stack or []:
        words: list[str] = []
        for raw in re.split(r"[\s,;/+()|:_\-]+", str(item).lower()):
            word = re.sub(r"\.?js$", "", raw) if len(raw) > 2 else raw
            word = re.sub(r"@?v?\d+(?:\.\d+)*$", "", word) or word
            if word:
                words.append(word)
        tokens.update(words)
        tokens.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    return tokens


# DB chính (primary store) vs. DB phụ trợ (cache/search). Redis/Elasticsearch/Memcached
# đi kèm một SQL DB là cache/search — KHÔNG biến project thành "both" (sinh Mongoose thừa).
_NOSQL_PRIMARY_KW: frozenset[str] = frozenset(
    {"mongodb", "mongo", "mongoose", "nosql", "dynamodb", "cassandra", "firestore", "couchdb"}
)
_NOSQL_AUX_KW: frozenset[str] = frozenset({"redis", "elasticsearch", "memcached"})
_SQL_KW: frozenset[str] = frozenset(
    {"postgresql", "postgres", "mysql", "sql server", "sqlite", "mssql", "oracle", "mariadb"}
)


def _detect_db_type(tech_stack: list[str] | None) -> str:
    """Nhận diện loại cơ sở dữ liệu từ danh sách tech_stack.

    Trả về:
        'nosql'   — chỉ có NoSQL (MongoDB, DynamoDB, Cassandra...)
        'sql'     — chỉ có SQL (PostgreSQL, MySQL, SQL Server, SQLite...)
        'both'    — có cả SQL và NoSQL chính
        'unknown' — không nhận diện được hoặc tech_stack trống

    Redis/Elasticsearch/Memcached là store phụ trợ (cache/search): chỉ tính là NoSQL
    khi stack KHÔNG có DB chính nào khác (ví dụ stack chỉ có Redis).
    Kết quả được dùng để chọn đúng extra_instruction cho DBA và DA agent.
    """
    if not tech_stack:
        return "unknown"
    tokens    = _stack_tokens(tech_stack)
    has_sql   = bool(tokens & _SQL_KW)
    has_nosql = bool(tokens & _NOSQL_PRIMARY_KW) or (not has_sql and bool(tokens & _NOSQL_AUX_KW))
    if has_nosql and has_sql:
        return "both"
    if has_nosql:
        return "nosql"
    if has_sql:
        return "sql"
    return "unknown"


# Info string của fence được coi là "tài liệu markdown" (lớp bọc hợp lệ của file .md).
_MD_FENCE_LANGS: frozenset[str] = frozenset({"markdown", "md", "mdx"})


def _strip_code_fence(content: str, language: str, filename: str = "") -> str:
    """Bóc code fence BAO NGOÀI (```lang ... ```) mà model bọc quanh nội dung file.

    Chỉ bóc khi fence thực sự là lớp bọc (F3). Dùng find_fenced_blocks() — quét
    fence-aware theo CommonMark, chịu được fence lồng nhau — thay cho regex lazy cũ
    vốn lấy block ĐẦU TIÊN ở bất kỳ đâu và cắt ở ``` đóng đầu tiên:
    - File markdown (language markdown/md hoặc *filename* đuôi .md): fence bên trong
      là NỘI DUNG file → chỉ bóc khi output là MỘT block ```markdown/```md (có thể kèm
      lời dẫn/lời kết) hoặc một block ``` trơn phủ toàn bộ output; ngược lại giữ nguyên.
    - File code: một block duy nhất (kèm hoặc không kèm lời dẫn) → bóc; nhiều block
      (file + ví dụ sử dụng) → ưu tiên block đúng ngôn ngữ, rồi tới block đầu tiên.
    """
    text   = content.strip()
    blocks = find_fenced_blocks(text)
    if not blocks:
        return text
    first = blocks[0]
    lang  = language.strip().lower()
    if lang in _MD_FENCE_LANGS or filename.lower().endswith((".md", ".mdx", ".markdown")):
        whole = len(blocks) == 1 and first.start == 0 and first.end == len(text)
        if len(blocks) == 1 and (first.lang in _MD_FENCE_LANGS or (whole and not first.info)):
            return first.body.lstrip("\r\n").rstrip()
        return text
    chosen = first
    if len(blocks) > 1:
        chosen = next((b for b in blocks if b.lang == lang), first)
    return chosen.body.lstrip("\r\n").rstrip()


def _file_section(filename: str, language: str, content: str) -> str:
    """Ghép một section ### FILE: cho output markdown của coding role.

    Fence bọc ngoài lấy từ fence_for() — dài hơn mọi fence bên trong content — để
    file markdown chứa ```bash không bị artifacts.py/marked.js đóng block giữa chừng (F4).
    """
    fence = fence_for(content)
    return f"\n### FILE: {filename}\n{fence}{language}\n{content}\n{fence}\n"


class FileGenerationError(Exception):
    """Sinh một file thất bại: lỗi gọi LLM (timeout, mất kết nối, model chưa pull...)
    hoặc model không trả về nội dung dùng được (rỗng sau khi bóc <think>/fence).

    Thay cho chuỗi "// Error generating ..." cũ — chuỗi đó từng bị lưu thành nội dung
    file nguồn và bước vẫn được tính là thành công.
    """

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(f"{filename}: {reason}")
        self.filename = filename
        self.reason = reason


# Tiêu đề section liệt kê file sinh lỗi trong output của coding role. HỢP ĐỒNG với UI
# (static/workflow.html) và runtime (app.py): giữ nguyên từng byte — "\u26a0\ufe0f" là
# emoji ⚠️ (U+26A0 + variation selector U+FE0F).
FILE_GEN_ERRORS_HEADER = "\n## \u26a0\ufe0f File generation errors\n"
# Độ dài tối đa của mô tả lỗi mỗi file trong section trên.
_FILE_ERR_MAX_CHARS = 200
_FILE_DIRECTIVE_RE = re.compile(r"#+\s*FILE:", re.IGNORECASE)


def _one_line(text: str, max_chars: int) -> str:
    """Thu text về MỘT dòng an toàn để nhúng vào markdown của output role.

    Gộp khoảng trắng/xuống dòng, thay backtick (không mở được code fence hay inline
    code lệch cặp), vô hiệu directive "### FILE:" (artifacts.py không nhận nhầm thành
    file), rồi cắt còn *max_chars* ký tự.
    """
    flat = " ".join(str(text).split()).replace("`", "'")
    flat = _FILE_DIRECTIVE_RE.sub("FILE:", flat)
    return flat if len(flat) <= max_chars else flat[: max_chars - 1] + "…"


def _short_error(exc: BaseException) -> str:
    """Mô tả lỗi ngắn gọn một dòng: "TênException: thông điệp"."""
    msg = str(exc)
    return _one_line(f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__, _FILE_ERR_MAX_CHARS)


def _file_errors_section(failed: dict[str, str]) -> str:
    """Section liệt kê file sinh lỗi (bắt đầu đúng bằng FILE_GEN_ERRORS_HEADER).

    *failed*: filename → lỗi ngắn. Viết tiếng Anh vì output role là context cho
    agent phía sau (Tech Lead, Tester, Clarifier) — Clarifier có thể đề xuất re-gen.
    """
    lines = [
        "The following planned files could not be generated and were NOT saved as artifacts:",
        "",
    ]
    lines += [
        f"- `{_one_line(name, 200)}` — {_one_line(err, _FILE_ERR_MAX_CHARS)}"
        for name, err in failed.items()
    ]
    return FILE_GEN_ERRORS_HEADER + "\n".join(lines) + "\n"


# Cú pháp comment (mở, đóng) cho marker cắt lặp của _deloop(), tra theo language của
# planner hoặc đuôi file. None = định dạng không có comment (JSON) hoặc không có một cú
# pháp an toàn cho cả file (Vue/Svelte trộn HTML + script, MDX) → cắt lặp nhưng KHÔNG
# chèn marker. Ngôn ngữ không có trong bảng (kể cả "text") cũng không chèn marker.
_CMT_SLASH = ("// ", "")
_CMT_HASH = ("# ", "")
_CMT_DASH = ("-- ", "")
_CMT_CSS = ("/* ", " */")
_CMT_HTML = ("<!-- ", " -->")
_COMMENT_STYLES: dict[str, tuple[str, str] | None] = {
    **dict.fromkeys((
        "typescript", "ts", "tsx", "mts", "cts", "javascript", "js", "jsx", "mjs", "cjs",
        "java", "kotlin", "kt", "kts", "swift", "dart", "go", "golang", "rust", "rs",
        "c", "h", "cpp", "c++", "cc", "hpp", "cs", "csharp", "c#", "scala", "groovy",
        "gradle", "scss", "less", "proto", "protobuf", "prisma", "php",
    ), _CMT_SLASH),
    **dict.fromkeys((
        "python", "py", "yaml", "yml", "bash", "sh", "shell", "zsh", "dockerfile", "docker",
        "toml", "ini", "cfg", "conf", "nginx", "env", "dotenv", "properties", "ruby", "rb",
        "perl", "pl", "r", "hcl", "terraform", "tf", "tfvars", "makefile", "make", "mk",
        "cmake", "powershell", "ps1", "graphql", "gql", "elixir", "ex", "exs",
    ), _CMT_HASH),
    **dict.fromkeys((
        "sql", "psql", "pgsql", "plsql", "mysql", "postgresql", "postgres", "tsql", "sqlite",
        "lua", "haskell", "hs", "elm",
    ), _CMT_DASH),
    "css": _CMT_CSS,
    **dict.fromkeys(("html", "htm", "xml", "svg", "xaml", "xsd", "markdown", "md"), _CMT_HTML),
    **dict.fromkeys(("json", "jsonc", "json5", "jsonl", "ndjson", "ipynb", "vue", "svelte", "mdx"), None),
}
_TRUNCATED_MARKER = "... (repeated pattern truncated)"


def _comment_style(language: str, filename: str = "") -> tuple[str, str] | None:
    """Cú pháp comment cho marker của _deloop: theo *language*, rồi tới tên/đuôi *filename*.

    language của planner là chuỗi tự do (mặc định "text") nên khi không nhận ra thì tra
    theo tên file (Dockerfile, Makefile, .env, đuôi .yml...). File đuôi JSON không bao
    giờ có marker dù language nói gì — JSON không có comment.
    """
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
    ext = name.rsplit(".", 1)[-1] if "." in name.lstrip(".") else ""
    if ext in ("json", "jsonc", "json5", "jsonl", "ndjson", "ipynb"):
        return None
    words = language.strip().lower().split()
    lang = words[0] if words else ""
    if lang in _COMMENT_STYLES:
        return _COMMENT_STYLES[lang]
    if name in ("dockerfile", "makefile") or name.startswith(("dockerfile.", ".env")) \
            or name.endswith(".dockerfile") or name in (".gitignore", ".dockerignore"):
        return _CMT_HASH
    return _COMMENT_STYLES.get(ext)


def _deloop(text: str, language: str = "", max_repeats: int = 8, *, filename: str = "") -> str:
    """Cắt ngắn các đoạn dòng lặp lại liên tiếp để giảm thiểu lỗi vòng lặp của model.

    Một số coding model nhỏ có xu hướng lặp lại cùng một dòng code (thường là
    import hoặc boilerplate) hàng trăm lần. Hàm này giới hạn tối đa max_repeats
    lần lặp liên tiếp rồi thêm một dòng marker truncation.

    Marker là COMMENT đúng cú pháp của *language* (fallback theo *filename*), thụt lề
    như dòng bị lặp: "# ..." cho YAML/Python, "-- ..." cho SQL, "<!-- ... -->" cho
    Markdown/HTML... — marker "//" cố định cũ làm hỏng file YAML/Python/SQL/Markdown.
    JSON / ngôn ngữ không rõ → không chèn marker (chỉ cắt), file vẫn hợp lệ.
    """
    style = _comment_style(language, filename)
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
            elif count == max_repeats + 1 and style is not None:
                indent = ln[: len(ln) - len(ln.lstrip())]
                out.append(f"{indent}{style[0]}{_TRUNCATED_MARKER}{style[1]}")
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

    # So khớp theo token (không theo chuỗi con) — xem _stack_tokens.
    tokens   = _stack_tokens(tech_stack)
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
        if "vue" in tokens:
            fw = "Vue 3 component (Composition API, TypeScript, template)"
        elif "angular" in tokens:
            fw = "Angular component (TypeScript, decorators, template)"
        elif tokens & {"svelte", "sveltekit"}:
            fw = "Svelte component (TypeScript, reactive syntax)"
        else:
            fw = "React/TypeScript component (JSX, hooks, typed props)"
        return "Scaffold a " + fw + _scaffold

    if role == "mobile":
        if tokens & {"react native", "expo"}:
            fw = "React Native component (TypeScript, JSX, StyleSheet)"
        elif tokens & {"flutter", "dart"}:
            fw = "Flutter/Dart widget"
        else:
            fw = "Flutter/Dart widget or React Native component"
        return "Scaffold a " + fw + _scaffold

    if role == "be":
        # Framework ghi rõ thắng ngôn ngữ trần: ["nestjs", "python"] → NestJS
        # (python thường dành cho DA/ML); NestJS trước Express vì Nest chạy trên Express.
        # Chỉ có ngôn ngữ → framework mặc định của ngôn ngữ đó; không gì khớp → NestJS.
        be_rules: tuple[tuple[set[str], str], ...] = (
            ({"fastapi"}, "FastAPI router + Pydantic models (async)"),
            ({"django"}, "Django/DRF view + serializers"),
            ({"flask"}, "Flask route + request parsing"),
            ({"nest"}, "NestJS controller + service + DTO"),
            ({"express"}, "Express.js route + TypeScript types"),
            ({"spring", "springboot"}, "Spring Boot controller + service (annotated)"),
            ({"python"}, "FastAPI router + Pydantic models (async)"),
            ({"java", "kotlin"}, "Spring Boot controller + service (annotated)"),
            ({"go", "golang"}, "Go HTTP handler + struct types"),
        )
        fw = next((f for kws, f in be_rules if tokens & kws), "NestJS controller + service + DTO")
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


_JSON_DECODER = _json.JSONDecoder()


def _extract_json_array(text: str) -> list | None:
    """Trích JSON array (danh sách object) đầu tiên hợp lệ trong output của planner.

    Thử json raw_decode tại từng vị trí '[' — thay cho regex non-greedy \\[.*?\\]
    vốn dừng ở ']' đầu tiên, kể cả ']' nằm trong chuỗi ("app/users/[id]/page.tsx"),
    trong nhãn prose ("[Assumption] ...") hoặc trong mảng lồng của một field.
    Bỏ qua mảng không chứa object nào (ví dụ mảng chuỗi lồng bên trong).
    Trả về None nếu không tìm thấy.
    """
    idx = text.find("[")
    while idx != -1:
        try:
            obj, _end = _JSON_DECODER.raw_decode(text, idx)
        except ValueError:
            obj = None
        if isinstance(obj, list) and any(isinstance(item, dict) for item in obj):
            return obj
        idx = text.find("[", idx + 1)
    return None


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
    planner_ctx = min(8192, OLLAMA_NUM_CTX)
    planner = make_chat(
        CODING_PLANNER_MODEL,
        base_url=ollama_base_url,
        num_ctx=planner_ctx,
        num_predict=num_predict_for("planner", CODING_PLANNER_MODEL, num_ctx=planner_ctx),
    )
    stack_str = ", ".join(tech_stack) if tech_stack else "not specified"
    # Nếu có TL task board, chèn TOÀN BỘ danh sách task vào planning prompt để model biết
    # cần tạo file nào để cover hết task — cắt chữ từng task theo ngân sách ký tự chứ
    # không cắt danh sách (checklist / follow-up đối chiếu với mọi task).
    if tl_tasks:
        tl_tasks_section = (
            "IMPORTANT — Team Lead Task Board: your file list MUST cover all of these tasks:\n"
            + _format_tl_task_lines(tl_tasks)
        )
    else:
        tl_tasks_section = ""
    prompt = _PLAN_PROMPT_TMPL.format(
        role_name=agent.name,
        max_files=_MAX_FILES_PER_ROLE,
        user_input=user_input[:500],
        tech_stack=stack_str,
        # Idempotent khi caller đã rút gọn; context_summary[:1000] cũ cắt mất đuôi context.
        context_summary=_compact_context(context_summary, _PLAN_CTX_CHARS),
        tl_tasks_section=tl_tasks_section,
        extra=f"Constraint: {extra_instruction}" if extra_instruction else "",
    )
    try:
        resp = planner.invoke([HumanMessage(content=prompt)])
        if hit_token_cap(resp):
            logger.warning(
                "Planner %s chạm trần PLANNER_NUM_PREDICT cho role=%s — JSON có thể bị cắt cụt.",
                CODING_PLANNER_MODEL, role,
            )
        # Bỏ <think>...</think> nếu CODING_PLANNER_MODEL là reasoning model — tránh
        # bắt nhầm JSON nháp trong chain-of-thought.
        content = _strip_thinking(str(resp.content))
        files = _extract_json_array(content)
        if files:
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
    Raise FileGenerationError nếu lời gọi LLM lỗi hoặc không còn nội dung dùng được
    — caller KHÔNG được ghi file đó thành artifact.
    """
    num_ctx = _get_num_ctx(agent.model)
    # FILEGEN_THINK chỉ áp cho lời gọi sinh file; None = để model tự quyết thinking.
    reasoning = filegen_reasoning()
    llm = make_chat(
        agent.model,
        base_url=ollama_base_url,
        num_ctx=num_ctx,
        num_predict=num_predict_for("filegen", agent.model, num_ctx=num_ctx, reasoning=reasoning),
        reasoning=reasoning,
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
    except Exception as exc:
        logger.warning("File gen failed for %s (%s): %s", agent.name, filename, exc)
        raise FileGenerationError(filename, _short_error(exc)) from exc
    _warn_if_capped(resp, agent, f"FILEGEN_NUM_PREDICT ({filename})")
    result = str(resp.content)
    # W3: nếu coding role dùng reasoning model, loại bỏ <think>...</think>
    # TRƯỚC khi lưu — nếu không chain-of-thought bị ghi thẳng vào file artifact.
    if is_reasoning_model(agent.model):
        result = _strip_thinking(result)
    result = _strip_code_fence(result, language, filename)   # chỉ bóc fence bọc ngoài
    result = _deloop(result, language, filename=filename)   # cắt vòng lặp, marker đúng ngôn ngữ
    if not result.strip():
        # Rỗng: model không trả gì, hoặc chạm trần num_predict giữa lúc suy luận
        # (_strip_thinking bỏ <think> chưa đóng) — không có gì để lưu.
        reason = "model returned no file content"
        if hit_token_cap(resp):
            reason += " (hit FILEGEN_NUM_PREDICT)"
        logger.warning("File gen empty for %s (%s): %s", agent.name, filename, reason)
        raise FileGenerationError(filename, reason)
    return result


def _generate_artifacts_multi_turn(
    role: str,
    agent: AgentConfig,
    ollama_base_url: str,
    context: str,
    user_input: str,
    tech_stack: list[str] | None,
    extra_instruction: str = "",
    tl_output: str | None = None,
) -> str:
    """Sinh toàn bộ artifact cho một coding role theo quy trình 2 pha.

    *tl_output* là output THÔ của TL agent (step_outputs["tl"]). TL task board được
    parse từ đây thay vì từ *context* — context chỉ giữ 3.000 ký tự đầu của TL nên
    board §5–§7 (Mobile/BE/DBA) thường đã bị cắt. None → fallback parse *context*.

    Pha 1 — Lập kế hoạch:
        CODING_PLANNER_MODEL nhận task, tech stack, context tóm tắt và TL task board,
        trả về JSON array danh sách file cần sinh.

    Pha 2 — Sinh từng file:
        Coding model sinh nội dung từng file độc lập với context tập trung.
        Nếu còn TL task chưa được cover sau pha 2 → chạy follow-up pass để lên
        kế hoạch thêm file mới và sinh chúng (bỏ qua file trùng tên).

    Lỗi từng file (FileGenerationError):
        File lỗi không được ghi thành section ### FILE: và không tính là cover TL task
        (follow-up pass có thể thử lại file đó một lần). Danh sách file còn lỗi sau cả
        hai pha nằm trong section FILE_GEN_ERRORS_HEADER.

    Kết thúc:
        Append bảng Task Completion Checklist đối chiếu mọi TL task với file đã sinh.

    Trả về:
        Chuỗi markdown tổng hợp với ### FILE: sections, section lỗi (nếu có) và
        checklist cuối. Nếu KHÔNG file nào sinh được: chuỗi bắt đầu
        "[LỖI trong {role}]" kèm section lỗi.
    """
    # Giữ đề bài + tech stack ở đầu context và lấp phần còn lại bằng đuôi (output agent
    # gần nhất / RAG / gợi ý DB) — context[-1200:] cũ chỉ còn đuôi, mất đề bài và stack.
    context_summary = _compact_context(context, _PLAN_CTX_CHARS)

    # Trích xuất danh sách TL task (ưu tiên TL output thô) để tracking coverage và lập kế hoạch.
    tl_tasks = _extract_tl_tasks(role, tl_output or context)
    if tl_tasks:
        logger.info("Role=%s: found %d TL tasks to cover", role, len(tl_tasks))

    # Pha 1: Gọi CODING_PLANNER_MODEL để lập kế hoạch danh sách file.
    files = _plan_code_files(
        role, agent, ollama_base_url, user_input, tech_stack,
        context_summary, extra_instruction, tl_tasks=tl_tasks or None,
    )

    if not files:
        logger.warning("No files planned for %s, fallback to single call", role)
        # Context rút gọn nhưng GIỮ đề bài + tech stack (đầu context) và bổ sung TL task
        # board đã parse — context[-2000:] cũ chỉ còn đuôi RAG/output agent gần nhất.
        task_block = ""
        if tl_tasks:
            task_block = "\n\n## TL Task Board — bắt buộc cover\n" + _format_tl_task_lines(
                tl_tasks, bullet="- ",
            )
        fallback_ctx = _compact_context(context, max(1_000, 2_000 - len(task_block))) + task_block
        output = _call_agent(agent, ollama_base_url, fallback_ctx)
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
    ctx_snippet = _compact_context(context, _REVIEW_CTX_CHARS) if role in _review_roles else ""

    # generated: file sinh THÀNH CÔNG — chỉ nhóm này được tính là cover TL task
    # (uncovered-task + checklist). failed: filename → lỗi ngắn, phản ánh trạng thái
    # CUỐI CÙNG (file lỗi ở pha 1 mà follow-up sinh lại được thì gỡ khỏi danh sách).
    # File lỗi không bao giờ được ghi thành section ### FILE: (artifacts.py sẽ lưu nó).
    generated: list[dict] = []
    failed: dict[str, str] = {}

    def _name(file_info: dict) -> str:
        return str(file_info.get("filename") or "unknown")

    def _generate(file_info: dict, phase: str) -> None:
        filename    = _name(file_info)
        # Bỏ qua file đã sinh thành công (planner liệt kê trùng tên / follow-up đề xuất
        # lại) để tránh ghi đè và để một tên file không vừa là FILE vừa nằm trong lỗi.
        if any(_name(f) == filename for f in generated):
            logger.info("%s: skip duplicate file %s for role=%s", phase, filename, role)
            return
        description = file_info.get("description", "")
        language    = file_info.get("language", "text")
        logger.info("%s file %s for role=%s", phase, filename, role)
        try:
            file_content = _generate_one_file(
                filename, description, language, agent, ollama_base_url, tech_stack,
                extra_instruction, ctx_snippet,
            )
        except FileGenerationError as exc:
            failed[filename] = exc.reason
            return
        failed.pop(filename, None)
        parts.append(_file_section(filename, language, file_content))
        generated.append(file_info)

    for file_info in files:
        _generate(file_info, "Generating")

    # Pha 2 (follow-up): nếu còn TL task chưa được cover sau pha 1 (kể cả task của file
    # sinh lỗi), lên kế hoạch thêm file mới và sinh chúng để đảm bảo coverage đầy đủ.
    if tl_tasks:
        uncovered = _find_uncovered_tasks(tl_tasks, generated)
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
            attempted: set[str] = set()
            for file_info in followup_files:
                fname = _name(file_info)
                # File LỖI ở pha 1 được thử lại đúng một lần; file đã sinh thành công
                # bị _generate bỏ qua.
                if fname in attempted:
                    logger.info("Follow-up skip duplicate file %s for role=%s", fname, role)
                    continue
                attempted.add(fname)
                _generate(file_info, "Follow-up: generating")

    if not generated:
        # Không có file nào dùng được → cả bước thất bại. Tiền tố "[LỖI trong {role}]"
        # được app.py map vào failed_steps (và không trích xuất artifact).
        logger.error("Role=%s: tất cả %d file lên kế hoạch đều sinh lỗi", role, len(failed))
        return (
            f"[LỖI trong {role}] Không sinh được file nào "
            f"({len(failed)} file lên kế hoạch đều thất bại).\n"
            + _file_errors_section(failed)
        )

    if failed:
        logger.warning(
            "Role=%s: %d/%d file sinh lỗi: %s",
            role, len(failed), len(failed) + len(generated), list(failed),
        )
        parts.append(_file_errors_section(failed))

    # Thêm bảng Task Completion Checklist vào cuối output để đối chiếu mọi TL task
    # với file đã sinh THÀNH CÔNG — ✅ Done, ⏳ Partial, hoặc ❌ Deferred.
    if tl_tasks:
        parts.append(_build_task_checklist(tl_tasks, generated))

    return "\n".join(parts)


def _warn_if_capped(response: object, agent: AgentConfig, cap_name: str) -> None:
    """Ghi log cảnh báo khi Ollama dừng sinh vì chạm trần num_predict (*cap_name*)."""
    if hit_token_cap(response):
        logger.warning(
            "Agent '%s' (model=%s) chạm trần %s — output có thể bị cắt cụt.",
            agent.name, agent.model, cap_name,
        )


def _call_agent(
    agent: AgentConfig,
    ollama_base_url: str,
    context: str,
) -> str:
    """Gọi LLM một lần với system prompt đầy đủ của agent + context đã tổng hợp.

    Dùng cho tất cả non-artifact roles (ba, pm, sa, ta, designer, tl, da,
    tester, clarifier) — những role sinh báo cáo một lần thay vì per-file loop.

    Nếu model trả về output < 30 ký tự (thường do model nhỏ bị overflow context),
    thực hiện retry với context rút gọn 1.500 ký tự (_compact_context: giữ đề bài +
    tech stack ở đầu, lấp phần còn lại bằng đuôi context) để giảm tải.
    """
    num_ctx = _get_num_ctx(agent.model)
    # Trần REPORT_NUM_PREDICT áp cho cả lần gọi chính lẫn lần retry (cùng instance llm),
    # kể cả khi _call_agent là fallback single-call của artifact role.
    llm = make_chat(
        agent.model,
        base_url=ollama_base_url,
        num_ctx=num_ctx,
        num_predict=num_predict_for("report", agent.model, num_ctx=num_ctx),
    )
    # Kiểm tra một lần — dùng lại cho cả lần gọi chính và lần retry.
    is_reasoning = is_reasoning_model(agent.model)

    # Build system message once — reused in primary call and retry to avoid duplication.
    sys_msg  = SystemMessage(content=COMMON_AGENT_RULES + "\n\n---\n\n" + agent.system_prompt)
    response = llm.invoke([sys_msg, HumanMessage(content=context)])
    _warn_if_capped(response, agent, "REPORT_NUM_PREDICT")
    result   = str(response.content)

    # Strip chain-of-thought blocks from reasoning models.
    if is_reasoning:
        result = _strip_thinking(result)

    if len(result.strip()) < 30:
        logger.warning(
            "Agent '%s' (model=%s) trả về output rất ngắn (%d chars) — retry với context rút gọn.",
            agent.name, agent.model, len(result.strip()),
        )
        trimmed = _compact_context(context, 1_500)
        response = llm.invoke([sys_msg, HumanMessage(content=trimmed)])
        _warn_if_capped(response, agent, "REPORT_NUM_PREDICT")
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
            elif db_type == "both":
                parts.append(
                    "\n## Data Analysis: Mixed (SQL + NoSQL)\n"
                    "Tech stack dùng cả SQL và NoSQL (MongoDB). Phân tích dữ liệu quan hệ bằng "
                    "SQL queries (GROUP BY, window functions, CTEs) và dữ liệu document bằng "
                    "MongoDB aggregation pipeline ($match, $group, $project, $lookup). "
                    "Ghi rõ mỗi query chạy trên data store nào; KHÔNG trộn cú pháp SQL vào "
                    "aggregation pipeline. Cần phân tích chéo hai store: export CSV rồi "
                    "join bằng Python/pandas."
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
                    tl_output=step_outputs.get("tl"),
                )
            else:
                output = _call_agent(agent, ollama_url, context)
        except Exception as exc:
            logger.error("Agent '%s' gặp lỗi: %s", role, exc)
            output = f"[LỖI trong {role}] {exc}"
            error_msg = f"Bước '{role}' thất bại: {exc}"
        error_prefix = f"[LỖI trong {role}]"
        if error_msg is None and output.startswith(error_prefix):
            # Coding role không sinh được file nào — _generate_artifacts_multi_turn tự gắn
            # tiền tố lỗi thay vì raise để giữ section liệt kê lỗi từng file.
            error_msg = f"Bước '{role}' thất bại: {output.splitlines()[0][len(error_prefix):].strip()}"

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
            tl_output=step_outputs.get("tl"),
        )
    return _call_agent(agent, ollama_base_url, context)

