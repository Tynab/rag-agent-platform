# rag-agent-platform

`rag-agent-platform` là nền tảng RAG và SDLC Agent cục bộ, vận hành hoàn toàn **offline** — không cần cloud, không cần API key bên ngoài. Toàn bộ pipeline từ LLM inference, vector search, graph enrichment, RAG, đến tự động hóa quy trình SDLC chạy trong một cụm Docker Compose trên máy local hoặc private server.

**Thành phần cốt lõi:** Ollama · Qdrant · Neo4j · RAG API · Agent API · Open WebUI · Watchtower · Deunhealth

---

## Mục lục

1. [Tổng quan rag-agent-platform](#1-tổng-quan-rag-agent-platform)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [SDLC Workflow — 15 bước](#3-sdlc-workflow--15-bước)
4. [Yêu cầu hệ thống](#4-yêu-cầu-hệ-thống)
5. [Cấu trúc thư mục](#5-cấu-trúc-thư-mục)
6. [Thiết lập từ đầu — hướng dẫn từng bước](#6-thiết-lập-từ-đầu--hướng-dẫn-từng-bước)
7. [Biến môi trường chi tiết](#7-biến-môi-trường-chi-tiết)
8. [Agent API — tài liệu đầy đủ](#8-agent-api--tài-liệu-đầy-đủ)
9. [RAG API — tài liệu đầy đủ](#9-rag-api--tài-liệu-đầy-đủ)
10. [Artifact System](#10-artifact-system)
11. [Clarifier Regen Loop](#11-clarifier-regen-loop)
12. [Episodic Memory Logging](#12-episodic-memory-logging)
13. [Open WebUI — Custom Tools](#13-open-webui--custom-tools)
14. [Neo4j — Graph Enrichment](#14-neo4j--graph-enrichment)
15. [Điều chỉnh model](#15-điều-chỉnh-model)
16. [Vận hành Docker](#16-vận-hành-docker)
17. [Troubleshooting](#17-troubleshooting)
18. [Danh mục URL dịch vụ](#18-danh-mục-url-dịch-vụ)
19. [Phát triển, test & CI](#19-phát-triển-test--ci)

---

## 1. Tổng quan rag-agent-platform

| Service | Image | Port (host) | Bind | Mục đích |
|---|---|---|---|---|
| `ollama` | `ollama/ollama:latest` | 11434 | `127.0.0.1` | LLM inference & embedding — chạy tất cả model AI |
| `qdrant` | `qdrant/qdrant:latest` | 6333 / 6334 | `127.0.0.1` | Vector database — lưu và tìm kiếm embedding (REST / gRPC) |
| `neo4j` | `neo4j:5-community` | 7474 / 7687 | `127.0.0.1` | Graph database — lưu quan hệ thực thể, tăng cường RAG |
| `rag-api` | build `./rag-api` | 8090 | `127.0.0.1` | FastAPI RAG service — ingest tài liệu và hỏi đáp AI |
| `agent-api` | build `./agent-api` | 8091 | `${BIND_ADDR}` | FastAPI SDLC Agent Orchestrator — 15 AI agents phân tích & sinh code |
| `open-webui` | `ghcr.io/open-webui/open-webui` | 8085 (`OPEN_WEBUI_PORT`) | `${BIND_ADDR}` | Giao diện chat kết nối Ollama & Qdrant |
| `watchtower` | `containrrr/watchtower` | 8080 (HTTP API) | `127.0.0.1` | Tự động pull & restart image có label `watchtower.enable=true` |
| `deunhealth` | `qmcgaw/deunhealth` | — | — | Khởi động lại container khi healthcheck thất bại liên tục (`network_mode: none`) |

> **Bảo mật cổng:** không service nào có xác thực mặc định, nên mọi cổng chỉ publish trên `127.0.0.1` (truy cập được từ chính máy chạy Docker). Riêng `agent-api` và Open WebUI bind theo `BIND_ADDR` (mặc định `127.0.0.1`) — xem [Truy cập từ máy khác (LAN)](#truy-cập-từ-máy-khác-lan). Các container gọi nhau qua network nội bộ `yan` (`http://rag-api:8090`, `http://ollama:11434`…), không qua cổng host.

### Phân nhóm model theo chức năng

| Nhóm | Env vars | Vai trò |
|---|---|---|
| Embedding | `EMBEDDING_MODEL` | Chuyển đổi văn bản → vector cho Qdrant |
| Chat RAG | `CHAT_MODEL` | Sinh câu trả lời từ context RAG trong rag-api |
| Planner | `CODING_PLANNER_MODEL` | Lập kế hoạch danh sách file trước khi sinh code |
| Reasoning (phân tích) | `BA_MODEL` `PM_MODEL` `SA_MODEL` `TA_MODEL` `DA_MODEL` | Các agent phân tích nghiệp vụ và kiến trúc |
| Team Lead | `TL_MODEL` | Lập kế hoạch task kỹ thuật cho các engineer |
| Coding (sinh code) | `FE_MODEL` `MOBILE_MODEL` `BE_MODEL` `DBA_MODEL` `TECH_LEAD_MODEL` `DEVSECOPS_MODEL` | Các agent sinh code, schema, manifest |
| Creative / QA | `DESIGNER_MODEL` `TESTER_MODEL` | Thiết kế UI/UX và viết test |
| Clarifier | `CLARIFIER_MODEL` | Kiểm tra toàn bộ output — phát hiện gap, mâu thuẫn, assumption |

> **Lưu ý:** Đổi `EMBEDDING_MODEL` yêu cầu re-ingest toàn bộ tài liệu (`POST /ingest {"reset": true}`). Mọi model khác chỉ cần `docker compose restart agent-api` hoặc `rag-api`.

---

## 2. Kiến trúc hệ thống

```
Người dùng
    │
    ├── Open WebUI (8085) ──────────────────────────────────────────────┐
    │       │                                                           │
    │       ├── yan_knowledge_base.py ──► RAG API (8090)                │
    │       │                               ├── Qdrant (6333)           │
    │       │                               │     └── vector search     │
    │       │                               ├── Neo4j (7687)            │
    │       │                               │     └── graph enrichment  │
    │       │                               └── Ollama (11434)          │
    │       │                                     ├── embed model       │
    │       │                                     └── chat model        │
    │       │                                                           │
    │       └── yan_agent_workflow.py ──► Agent API (8091) ─────────────┘
    │                                        └── LangGraph SDLC Workflow
    │                                              ├── 15 agent nodes
    │                                              ├── Ollama (per-role models)
    │                                              └── RAG API (rag_query_hint)
    │
    └── Trực tiếp qua curl / HTTP client
            ├── Agent API  (8091)
            └── RAG API    (8090)
```

### Kiến trúc Hybrid RAG (rag-api)

```
POST /ask ───┬───► Qdrant  vector search    (cosine similarity, top-k chunks)
             │
             └───► Neo4j   graph search     (entity co-occurrence traversal)
                     │
                     └──── merge & rank ──► Ollama (chat model) ──► trả lời
```

Mỗi request `/ask`:
1. Embed câu hỏi bằng embedding model
2. Tìm top-k chunks trong Qdrant theo cosine similarity
3. Neo4j tìm thêm chunks liên quan qua entity co-occurrence (nếu `GRAPH_ENABLED=true`)
4. Gộp và loại trùng kết quả
5. Đưa vào LLM sinh câu trả lời cuối cùng

### Module-scoped RAG

Khi tổ chức tài liệu theo thư mục con, mỗi chunk được gán `module` tự động:

```
data/raw/{project}/{module}/file.md  →  module = tên thư mục con
data/raw/{project}/file.md           →  module = project (flat file)
```

Ví dụ:
```
data/raw/yanlib/auth/auth-prd.md        →  project=yanlib, module=auth
data/raw/yanlib/billing/schema.md       →  project=yanlib, module=billing
data/raw/yanlib/spec.md                 →  project=yanlib, module=yanlib
```

Dùng filter `module` trong `/ask` để giới hạn search phạm vi nhỏ hơn, tăng độ chính xác:

```bash
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "JWT refresh token flow?", "project": "yanlib", "module": "auth"}' | jq
```

---

## 3. SDLC Workflow — 15 bước

Agent API điều phối một pipeline 15 AI agents chạy tuần tự theo mô hình LangGraph StateGraph. Mỗi agent nhận output rút gọn của các agent phụ thuộc làm context, đồng thời có thể truy vấn RAG knowledge base với `rag_query_hint` riêng để tăng độ chính xác.

```
BA → PM → SA → TA → Designer → Team Lead → FE → Mobile → DBA → BE → DA → Tech Lead → Tester → DevSecOps → Clarifier
```

| Bước | Role | Tên đầy đủ | Model | Phụ thuộc | Đầu ra chính |
|---|---|---|---|---|---|
| 1 | `ba` | Business Analysis | `BA_MODEL` | — | BRD, user stories, RTM, WBS |
| 2 | `pm` | Project Management | `PM_MODEL` | ba | Sprint plan, RAID log, milestone, roadmap |
| 3 | `sa` | Solution Architecture | `SA_MODEL` | ba, pm | C4 diagrams, API contracts, ADRs |
| 4 | `ta` | Technical Architecture | `TA_MODEL` | ba, sa | Tech stack, TDR, trade-off analysis |
| 5 | `designer` | UI/UX Design | `DESIGNER_MODEL` | ba, sa, ta | Wireframes, design system, component spec |
| 6 | `tl` | Team Lead / Task Planning | `TL_MODEL` | ba, sa, ta, designer | Task boards FE/Mobile/BE/DBA theo sprint |
| 7 | `fe` | Frontend Engineering | `FE_MODEL` | ba, sa, ta, designer, tl | React/Vue/Angular code + checklist |
| 8 | `mobile` | Mobile Engineering | `MOBILE_MODEL` | ba, sa, ta, designer, tl | Flutter/React Native code + checklist |
| 9 | `dba` | Database Architecture | `DBA_MODEL` | ba, sa, ta, tl | SQL schema / Mongoose models, indexes + checklist |
| 10 | `be` | Backend Implementation | `BE_MODEL` | ba, sa, ta, fe, mobile, dba, tl | NestJS/FastAPI/Express code + checklist |
| 11 | `da` | Data Analysis & Reporting | `DA_MODEL` | ba, sa, dba | KPI, dashboard spec, SQL/aggregation queries |
| 12 | `tech_lead` | Code Review & Standards | `TECH_LEAD_MODEL` | sa, fe, mobile, be, dba | ARCHITECTURE.md, security review, refactor plan |
| 13 | `tester` | Testing & QA | `TESTER_MODEL` | be, fe, mobile, tech_lead, designer | Test cases, UAT checklist, edge case matrix |
| 14 | `devsecops` | Infrastructure & CI/CD | `DEVSECOPS_MODEL` | sa, ta, tech_lead, tester | Dockerfile, K8s manifests, CI/CD pipeline YAML |
| 15 | `clarifier` | Cross-Role Gap Review | `CLARIFIER_MODEL` | tất cả 14 agent trước | Audit report, contradiction log, regen list |

### Cơ chế sinh code (Artifact Roles)

Các agent `fe`, `mobile`, `be`, `dba`, `tech_lead`, `devsecops` (role `mode="codegen"` — xem [Role registry](#role-registry--cấu-hình-hành-vi-theo-role)) sử dụng quy trình sinh code **2 pha**. `da` là role báo cáo (gọi LLM một lần) nhưng code block trong output vẫn được trích thành artifact.

**Pha 1 — Lập kế hoạch file:**
- `CODING_PLANNER_MODEL` (model nhẹ, ví dụ `granite4.1:8b`) nhận **toàn bộ** TL task board của role. Board dài hơn ngân sách ~3.000 ký tự thì chỉ cắt bớt chữ của các task dài nhất — không task nào bị bỏ
- Context gửi planner (~2.000 ký tự) giữ đề bài + tech stack ở đầu, cộng phần đuôi context
- Sinh JSON array các file cần tạo: `[{"filename": "src/auth/login.tsx", "description": "...", "language": "typescript"}]`
- Tối đa `MAX_FILES_PER_ROLE` file mỗi role

**Pha 2 — Sinh từng file:**
- Mỗi file được sinh độc lập bằng coding model với context tập trung. `tech_lead`/`devsecops` (review role) nhận thêm context rút gọn ~1.500 ký tự (đề bài + tech stack + đuôi output các agent trước)
- Nếu còn task TL chưa được cover → **pha follow-up**: lên kế hoạch thêm file và sinh tiếp. Pass follow-up có thể thử lại file lỗi một lần

**Giới hạn của mỗi lời gọi LLM:** timeout httpx thật (`OLLAMA_REQUEST_TIMEOUT`) và trần token `num_predict` theo loại lời gọi (`PLANNER_NUM_PREDICT` / `FILEGEN_NUM_PREDICT` / `REPORT_NUM_PREDICT`, nhân ×2 khi model suy luận) — xem [Nhóm Agent API](#nhóm-agent-api). Thinking khi sinh file bật/tắt bằng `FILEGEN_THINK`. Khi chạm trần, agent-api ghi log cảnh báo `chạm trần ...`.

**File sinh lỗi:** file lỗi (timeout, lỗi Ollama, model trả nội dung rỗng hoặc chạm trần giữa lúc suy luận) **không** được lưu thành artifact. Output của role có thêm section `## ⚠️ File generation errors` liệt kê từng file lỗi kèm lỗi ngắn gọn, và các task TL tương ứng hiện ⏳ trong checklist. Nếu **không file nào** sinh được, bước bị đánh dấu lỗi: output bắt đầu bằng `[LỖI trong {role}]` và role nằm trong `failed_steps` của workflow.

**Task Completion Checklist:**
Cuối output của mỗi engineer agent luôn có bảng checklist đối chiếu từng task TL với file đã sinh:

| # | Task | Priority | Status | File(s) / Section |
|---|------|----------|--------|-------------------|
| 1 | Setup project scaffolding | P0 | ✅ Done | `src/main.tsx`, `vite.config.ts` |
| 2 | Implement login form | P0 | ✅ Done | `src/pages/Login.tsx` |
| 3 | API integration layer | P1 | ⏳ Addressed in output | — |

### Clarifier Regen Loop

Sau khi workflow hoàn tất, Clarifier phân tích §10 "Recommended Re-generation List" của chính nó. Nếu có agent nào cần re-generate:
1. Re-run từng agent được chỉ định (theo thứ tự WORKFLOW_STEPS) với output đã cập nhật
2. Re-run Clarifier lại để đánh giá kết quả
3. Lặp tối đa `CLARIFIER_REGEN_LOOPS` lần (mặc định = 1)

Tắt tính năng: `CLARIFIER_REGEN_LOOPS=0` trong `.env`.

### Role registry — cấu hình hành vi theo role

`AgentConfig` trong `agent-api/agents.py` là **nguồn sự thật duy nhất** cho hành vi của từng role. Các bảng trong `workflow.py`/`artifacts.py` (role có artifact, role được re-gen, section TL board, giới hạn context…) được dẫn xuất tự động từ `AGENTS` lúc khởi động — không sửa bảng trực tiếp.

| Field | Ý nghĩa | Role hiện tại |
|---|---|---|
| `mode` | `"codegen"`: sinh code 2 pha (planner lập danh sách file → sinh từng file); `"report"` (mặc định): gọi LLM một lần | codegen: fe, mobile, be, dba, tech_lead, devsecops |
| `extract_artifacts` | Trích code block trong output ra file (tab Artifacts, tải ZIP) | fe, mobile, be, dba, da, tech_lead, devsecops |
| `regen_eligible` | Được chạy lại trong [Clarifier Regen Loop](#11-clarifier-regen-loop) (§10 của Clarifier) | fe, mobile, be, dba, da, tech_lead, devsecops |
| `tl_sections` | Từ khóa heading TL task board dành cho role | fe, mobile, be, dba |
| `per_dep_chars` | Số ký tự tối đa lấy từ output mỗi agent phụ thuộc (mặc định 3.000) | tech_lead 800, devsecops 1.000, tester 3.000, clarifier 1.500 |
| `wants_date_hint` | Chèn ngày hiện tại vào context (timeline/sprint không dùng ngày quá khứ) | ba, pm, sa, ta |
| `wants_db_hint` | Chèn gợi ý kiểu DB (SQL/NoSQL/cả hai) suy từ tech stack. Tech stack có cả SQL lẫn NoSQL (vd. PostgreSQL + MongoDB) → cả DBA và DA nhận gợi ý cho DB hỗn hợp | dba, da |
| `review_context` | Mỗi lần sinh file kèm context rút gọn của các agent trước | tech_lead, devsecops |

**Thêm role mới:**
1. Khai báo `MODEL_XXX = os.environ.get("XXX_MODEL", "<default>")`, thêm `XXX_MODEL` vào `.env` và khối `environment:` của agent-api trong `docker-compose.yml`.
2. Thêm `AgentConfig` vào `AGENTS` (step_id, role, name, model, depends_on, rag_query_hint, system_prompt) và đặt các field hành vi ở bảng trên — mặc định là role `"report"` không có hành vi đặc biệt.
3. Chèn role vào `WORKFLOW_STEPS` (và danh sách step trong `open-webui-tools/yan_agent_workflow.py`).
4. Nếu `regen_eligible=True`: thêm alias tên role vào `_REGEN_ROLE_RE` trong `workflow.py`. Nếu `wants_db_hint=True`: thêm câu chữ gợi ý DB cho role trong `_build_context_parts`.
5. Chạy `python -m pytest tests/test_role_registry*.py` — test kiểm tra bảng dẫn xuất và các bất biến (codegen ⇒ trích artifact, TL board/review chỉ cho codegen, planning roles không re-gen...).

---

## 4. Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Ghi chú |
|---|---|---|
| Docker Engine | ≥ 24.x | Docker Desktop (Windows/Mac) hoặc Docker Engine + Compose v2 (Linux) |
| RAM | ≥ 16 GB | Model 7B cần ~8 GB; model 35B cần ~24 GB với quantization |
| Dung lượng ổ cứng | ≥ 50 GB | Model + Docker images + data |
| GPU (tùy chọn) | NVIDIA với CUDA | Cải thiện tốc độ inference đáng kể; driver + NVIDIA Container Toolkit |
| CPU | ≥ 8 cores | Dùng khi không có GPU hoặc chạy song song nhiều model nhỏ |

**Định cỡ model theo tài nguyên:**

| RAM / VRAM | Khuyến nghị |
|---|---|
| 16 GB | Model 7B–8B (quantized Q4) cho tất cả roles |
| 32 GB | Model 14B–24B cho reasoning; model 7B cho coding |
| 64 GB+ | Model 35B+ cho reasoning; model 14B–32B cho coding |

---

## 5. Cấu trúc thư mục

```
rag-agent-platform/
├── .env                              # Nguồn cấu hình duy nhất (single source of truth)
├── docker-compose.yml                # Định nghĩa toàn bộ dịch vụ của rag-agent-platform
│
├── data/
│   ├── raw/                          # Tài liệu nguồn cho RAG
│   │   ├── {project}/                → Qdrant collection: yan_raw_docs__{project}
│   │   │   ├── {module}/             → chunk.module = {module}
│   │   │   │   └── *.md / *.pdf ...  → tài liệu được ingest
│   │   │   └── *.md                  → chunk.module = {project} (flat)
│   │   └── ...
│   ├── memory/
│   │   ├── workflows.sqlite3         → Trạng thái mọi workflow (WORKFLOW_DB_PATH, SQLite WAL)
│   │   └── episodic/
│   │       └── workflow_runs.jsonl   → Log tự động mỗi workflow run (JSONL)
│   └── artifacts/
│       └── {workflow_id}/
│           └── {role}/
│               ├── _output.md        → Toàn bộ markdown output của agent
│               ├── src/Login.tsx     → File code được trích xuất
│               └── ...
│
├── rag-api/
│   ├── app.py                        → FastAPI endpoints: /ingest /ask /projects /graph/*
│   ├── ingest.py                     → Pipeline: load → split → embed → upsert Qdrant + Neo4j
│   ├── graph.py                      → Neo4j GraphRAG: entity extraction & co-occurrence
│   ├── requirements.txt
│   └── Dockerfile
│
├── agent-api/
│   ├── app.py                        → FastAPI SDLC orchestrator: /workflow/run /cancel /agent/{role} /artifacts, hàng đợi workflow
│   ├── agents.py                     → 15 AgentConfig: model, system_prompt, depends_on, rag_query_hint + hành vi per-role (role registry)
│   ├── workflow.py                   → LangGraph StateGraph: node factory, context builder, artifact generator
│   ├── artifacts.py                  → Trích xuất file code từ markdown output, lưu vào disk
│   ├── llm.py                        → Factory ChatOllama: timeout httpx thật, trần num_predict, FILEGEN_THINK
│   ├── store.py                      → Lưu trạng thái workflow vào SQLite + khôi phục lúc khởi động
│   ├── requirements.txt
│   ├── Dockerfile
│   └── static/                       → Phục vụ tại /ui (workflow.html) và /ui/static/<path>
│       ├── workflow.html             → SDLC Workflow UI (single-page app)
│       └── vendor/                   → marked, DOMPurify, highlight.js, JSZip (+ LICENSE) — UI chạy offline
│
├── open-webui-tools/
│   ├── yan_knowledge_base.py         → Open WebUI tool: query RAG knowledge base
│   └── yan_agent_workflow.py         → Open WebUI tool: chạy / theo dõi / hủy SDLC workflow hoặc agent đơn lẻ
│
├── tests/                            → pytest (chạy không cần Docker) — xem mục 19
├── pyproject.toml                    → Cấu hình pytest, ruff, mypy
└── .github/workflows/ci.yml          → CI: ruff + mypy + pytest (Python 3.11 / 3.12)
```

**Định dạng file được hỗ trợ khi ingest:**
`.md` `.txt` `.pdf` `.docx` `.csv` `.json` `.jsonl` `.yaml` `.yml` `.xml` `.html` `.py` `.js` `.ts` `.go` `.rs` `.java` `.cs` `.sh` `.sql` `.log`

---

## 6. Thiết lập từ đầu — hướng dẫn từng bước

### Bước 1 — Chuẩn bị file `.env`

Điền các biến bắt buộc vào `.env`:

```env
# ─── Truy cập mạng ───────────────────────────────────────────────────────────
TZ=Asia/Ho_Chi_Minh
# 0.0.0.0 = mở agent-api UI + Open WebUI cho LAN (khi đó cần WEBUI_AUTH=true)
BIND_ADDR=127.0.0.1

# ─── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*
OLLAMA_CONTEXT_LENGTH=32768
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=5m
OLLAMA_REQUEST_TIMEOUT=1200

# ─── Models ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL=qwen3-embedding:8b
CHAT_MODEL=qwen3.6:35b
CODING_PLANNER_MODEL=granite4.1:8b

BA_MODEL=qwen3.6:35b
PM_MODEL=qwen3.6:35b
SA_MODEL=qwen3.6:35b
TA_MODEL=qwen3.6:35b
DA_MODEL=qwen3.6:35b

TL_MODEL=north-mini-code-1.0:q4_K_M

FE_MODEL=north-mini-code-1.0:q4_K_M
MOBILE_MODEL=north-mini-code-1.0:q4_K_M
BE_MODEL=north-mini-code-1.0:q4_K_M
DBA_MODEL=north-mini-code-1.0:q4_K_M
TECH_LEAD_MODEL=north-mini-code-1.0:q4_K_M
DEVSECOPS_MODEL=north-mini-code-1.0:q4_K_M

DESIGNER_MODEL=gemma4:31b
TESTER_MODEL=qwen3.6:35b
CLARIFIER_MODEL=qwen3.6:35b

# ─── RAG ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://ollama:11434
RAG_API_URL=http://rag-api:8090
RAG_TOP_K=5
RAG_TIMEOUT=600
RAG_LOG_LEVEL=INFO
RAW_DATA_DIR=/data/raw
COLLECTION_NAME=yan_raw_docs
QDRANT_URL=http://qdrant:6333
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
UPSERT_BATCH_SIZE=64
INGEST_EMBED_WORKERS=2
# INGEST_EMBED_RETRIES=3

# ─── Neo4j ───────────────────────────────────────────────────────────────────
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=changeme_in_production
NEO4J_DATABASE=neo4j
GRAPH_ENABLED=true
GRAPH_ENTITY_EXTRACTION=false

# ─── Open WebUI ──────────────────────────────────────────────────────────────
OPEN_WEBUI_PORT=8085
WEBUI_AUTH=false
WEBUI_NAME=rag-agent-platform
VECTOR_DB=qdrant
QDRANT_PREFER_GRPC=false
ENABLE_QDRANT_MULTITENANCY_MODE=false
QDRANT_COLLECTION_PREFIX=yan_raw_docs
RAG_EMBEDDING_ENGINE=ollama

# ─── Watchtower ──────────────────────────────────────────────────────────────
WATCHTOWER_HTTP_API_TOKEN=your_secure_token_here
# 6 trường (giây phút giờ ngày tháng thứ), tính theo TZ ở trên → 04:00 hằng ngày
WATCHTOWER_SCHEDULE=0 0 4 * * *

# ─── Agent API (tùy chọn — đã có mặc định trong docker-compose.yml) ──────────
MAX_FILES_PER_ROLE=6
CLARIFIER_REGEN_LOOPS=1
# WORKFLOW_MAX_SECONDS=0
# WORKFLOW_CONCURRENCY=1
# WORKFLOW_DB_PATH=/data/memory/workflows.sqlite3
# PLANNER_NUM_PREDICT=2048
# FILEGEN_NUM_PREDICT=8192
# REPORT_NUM_PREDICT=16384
# FILEGEN_THINK=
```

### Bước 2 — Khởi động rag-agent-platform

```bash
docker compose up -d
```

Lần đầu sẽ pull image (~2–5 GB). Theo dõi tiến trình:

```bash
docker compose logs -f
```

Chờ đến khi tất cả container ở trạng thái `healthy`:

```bash
docker compose ps
```

> `ollama`, `qdrant`, `neo4j`, `rag-api`, `agent-api` đều có healthcheck. Thứ tự khởi động: `rag-api` chờ `ollama` + `qdrant` **healthy**; `agent-api` chờ thêm `rag-api` healthy; Open WebUI chờ `ollama` + `qdrant` healthy. `neo4j` cố ý chỉ là `service_started` với `rag-api` (graph là tùy chọn, rag-api degrade graceful) — lần đầu Neo4j tải plugin APOC nên có thể mất ~1 phút mới `healthy`.

### Bước 3 — Pull models Ollama

Pull tất cả model đang khai báo trong `.env` (tự động loại trùng):

```bash
grep -E '^(EMBEDDING_MODEL|CHAT_MODEL|CODING_PLANNER_MODEL|BA_MODEL|PM_MODEL|SA_MODEL|TA_MODEL|DA_MODEL|TL_MODEL|FE_MODEL|MOBILE_MODEL|BE_MODEL|DBA_MODEL|TECH_LEAD_MODEL|DEVSECOPS_MODEL|TESTER_MODEL|DESIGNER_MODEL|CLARIFIER_MODEL)=' .env \
  | cut -d= -f2 \
  | sort -u \
  | xargs -I {} docker exec ollama ollama pull "{}"
```

Xác nhận:

```bash
docker exec ollama ollama list
```

### Bước 4 — Tổ chức tài liệu RAG

Đặt tài liệu vào thư mục con theo cấu trúc `data/raw/{project}/{module}/`:

```bash
mkdir -p data/raw/myproject/auth
mkdir -p data/raw/myproject/billing
mkdir -p data/raw/myproject/api

cp docs/auth-prd.md         data/raw/myproject/auth/
cp docs/billing-schema.md   data/raw/myproject/billing/
cp docs/api-contracts.md    data/raw/myproject/api/
cp docs/overview.md         data/raw/myproject/
```

Cấu trúc kết quả:

```
data/raw/myproject/
├── auth/
│   └── auth-prd.md          → module=auth
├── billing/
│   └── billing-schema.md    → module=billing
├── api/
│   └── api-contracts.md     → module=api
└── overview.md              → module=myproject (flat)
```

### Bước 5 — Ingest tài liệu

```bash
# Ingest tất cả project
curl -s -X POST http://localhost:8090/ingest | jq

# Hoặc ingest từng project
curl -s -X POST http://localhost:8090/ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "myproject"}' | jq
```

Kết quả thành công (ingest tất cả project — rút gọn; field đầy đủ xem [POST /ingest](#post-ingest)):

```json
{
  "status": "ok",
  "projects_ingested": ["myproject"],
  "total_documents": 4,
  "total_chunks": 142,
  "total_upserted": 142,
  "total_skipped_unchanged": 0,
  "reset": false,
  "details": {
    "myproject": {
      "status": "ok",
      "chunks": 142,
      "upserted": 142,
      "skipped_unchanged": 0,
      "failed_batches": 0,
      "batch_error": null,
      "graph_chunks_saved": 142,
      "stale_removed": 0
    }
  }
}
```

Ingest là **tăng dần**: chạy lại khi tài liệu không đổi trả `upserted: 0`, `skipped_unchanged: 142` và không gọi Ollama lần nào.

### Bước 6 — Kiểm tra hệ thống

```bash
# Kiểm tra RAG API
curl -s http://localhost:8090/health | jq

# Kiểm tra Agent API
curl -s http://localhost:8091/health | jq

# Test RAG query
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Auth flow hoạt động như thế nào?", "project": "myproject"}' | jq .answer

# Xem danh sách agent và model đang dùng
curl -s http://localhost:8091/agents | jq
```

### Bước 7 — Chạy SDLC Workflow đầu tiên

```bash
# Submit workflow
curl -s -X POST http://localhost:8091/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Xây dựng hệ thống đăng nhập và quản lý phiên cho SaaS platform",
    "project": "myproject",
    "rag_enabled": true,
    "rag_top_k": 5,
    "tech_stack": [
      "Backend: NestJS (Node.js, TypeScript)",
      "Frontend: React + TypeScript + TanStack Query",
      "Mobile: React Native",
      "Database: PostgreSQL",
      "Cache: Redis",
      "Infra: Docker, Kubernetes"
    ]
  }' | jq
```

```json
{
  "workflow_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "queue_position": null,
  "message": "Workflow đã được xếp hàng. Poll GET /workflow/a1b2c3d4-... để kiểm tra trạng thái."
}
```

```bash
# Poll trạng thái (workflow thường mất 15–60 phút tùy model)
watch -n 30 'curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq .status'

# Hoặc mở giao diện theo dõi trên trình duyệt
#   http://localhost:8091/ui?id=a1b2c3d4-e5f6-7890-abcd-ef1234567890

# Hủy nếu cần (bước đang chạy chạy nốt rồi dừng)
curl -s -X POST http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890/cancel | jq
```

> Mặc định chỉ một workflow chạy tại một thời điểm (`WORKFLOW_CONCURRENCY=1`); workflow gửi thêm nằm ở `pending` với `queue_position` = vị trí trong hàng đợi — xem [Hàng đợi workflow](#hàng-đợi-workflow-workflow_concurrency).

### Bước 8 — Thiết lập Open WebUI Tools (tùy chọn)

1. Mở [http://localhost:8085](http://localhost:8085)
2. Vào **Admin Panel → Tools → "+"**
3. Dán nội dung `open-webui-tools/yan_knowledge_base.py` → **Save**
4. Lặp lại cho `open-webui-tools/yan_agent_workflow.py`
5. Bật tool trong **Model Settings → Tools**

> **Tool không tự cập nhật:** Open WebUI lưu bản tool được dán tay, nên sau khi pull code mới phải dán lại cả hai file (hiện là `yan_knowledge_base` 1.2.0 và `yan_agent_workflow` 2.1.1). Giá trị Valve đã lưu từ bản cũ (vd. `timeout` 120, URL `localhost` / `host.docker.internal`) **giữ nguyên** tới khi sửa tay trong **Valves** — xem [mục 13](#13-open-webui--custom-tools).

---

## 7. Biến môi trường chi tiết

> **Lưu ý nguồn chuẩn (D6):** File `.env` đang chạy là **nguồn sự thật duy nhất**. Các giá trị ví dụ trong tài liệu này (model, `RAG_TOP_K`, `CHUNK_OVERLAP`, `UPSERT_BATCH_SIZE`…) mang tính minh hoạ và **có thể khác** `.env` thực tế — khi lệch nhau, hãy tin `.env`. Xem nhanh cấu hình runtime bằng `curl -s localhost:8090/health | jq` và `curl -s localhost:8091/agents | jq`.
>
> **Thư mục runtime (D8):** Trước lần `docker compose up` đầu tiên, tạo sẵn các thư mục mount để tránh Docker tự tạo (có thể bị root sở hữu trên Linux):
> ```bash
> mkdir -p ./data/raw ./data/memory/episodic ./data/artifacts
> ```
>
> **Wiring biến (C1):** compose **không** dùng `env_file:` — mỗi service chỉ nhận đúng các key trong khối `environment:` của nó. Thêm biến mới vào code thì phải thêm `NAME: ${NAME:-<default>}` vào `environment:` của service tương ứng; `tests/test_config_wiring.py` quét code agent-api/rag-api và fail nếu thiếu, đồng thời kiểm tra mọi `${VAR}` không có default trong compose đều có trong `.env`.

### Nhóm truy cập mạng & chung

| Biến | Mặc định | Mô tả |
|---|---|---|
| `BIND_ADDR` | `127.0.0.1` | Địa chỉ host publish cổng của `agent-api` (8091) và Open WebUI (`OPEN_WEBUI_PORT`). `0.0.0.0` = mở cho LAN — khi đó cần `WEBUI_AUTH=true`. Hạ tầng (ollama, qdrant, neo4j, rag-api, watchtower) luôn chỉ bind `127.0.0.1` |
| `TZ` | `Asia/Ho_Chi_Minh` | Múi giờ cho `deunhealth` và `watchtower` (quyết định giờ chạy `WATCHTOWER_SCHEDULE`; thiếu TZ = UTC) |
| `OPEN_WEBUI_PORT` | `8085` | Cổng Open WebUI (cả trong container lẫn trên host) |
| `WEBUI_AUTH` | `false` | Bật đăng nhập cho Open WebUI — **bắt buộc `true` nếu `BIND_ADDR=0.0.0.0`** |

### Nhóm Ollama

| Biến | Mặc định | Mô tả |
|---|---|---|
| `OLLAMA_HOST` | `0.0.0.0` | Interface Ollama lắng nghe |
| `OLLAMA_ORIGINS` | `*` | CORS origins cho Ollama API |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | Context window (tokens) cho mọi LLM call trong agent-api |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Số model giữ đồng thời trong VRAM/RAM. Workflow được thiết kế cho **một slot**: giữ `1` cùng `WORKFLOW_CONCURRENCY=1` |
| `OLLAMA_NUM_PARALLEL` | `1` | Số request Ollama xử lý song song |
| `OLLAMA_KEEP_ALIVE` | `5m` | Thời gian giữ model trong bộ nhớ sau request cuối |
| `OLLAMA_REQUEST_TIMEOUT` | `1200` (agent-api) / `600` (rag-api) | Timeout (giây) httpx cho mỗi lần gọi LLM — agent-api: mọi agent step; rag-api: chat model của `/ask` **và** LLM trích xuất thực thể lúc ingest (`GRAPH_ENTITY_EXTRACTION=true`). ChatOllama gọi Ollama ở chế độ stream nên đây là giới hạn thời gian tới **token đầu tiên** (gồm cả load/swap model và xử lý prompt) và **khoảng lặng giữa hai chunk**, không phải tổng thời gian sinh. `<= 0` = không giới hạn. Timeout này nay mới thực sự được áp dụng (trước đây ChatOllama 0.3.x bỏ qua kwarg `request_timeout` nên timeout là vô hạn) — máy chậm/chạy CPU có thể phải tăng giá trị |
| `OLLAMA_FLASH_ATTENTION` | `1` | Bật Flash Attention (tăng tốc trên GPU hỗ trợ) |

### Nhóm Models

| Biến | Mô tả |
|---|---|
| `EMBEDDING_MODEL` | Model embedding cho Qdrant |
| `CHAT_MODEL` | Model chat cho rag-api `/ask` |
| `CODING_PLANNER_MODEL` | Model nhẹ lập kế hoạch danh sách file |
| `BA_MODEL` | BA agent — phân tích nghiệp vụ |
| `PM_MODEL` | PM agent — quản lý dự án |
| `SA_MODEL` | SA agent — kiến trúc giải pháp |
| `TA_MODEL` | TA agent — kiến trúc kỹ thuật |
| `DA_MODEL` | DA agent — phân tích dữ liệu |
| `TL_MODEL` | Team Lead agent — lập kế hoạch task |
| `FE_MODEL` | FE agent — frontend code |
| `MOBILE_MODEL` | Mobile agent — mobile code |
| `BE_MODEL` | BE agent — backend code |
| `DBA_MODEL` | DBA agent — database schema |
| `TECH_LEAD_MODEL` | Tech Lead agent — code review |
| `DEVSECOPS_MODEL` | DevSecOps agent — infra / CI-CD |
| `DESIGNER_MODEL` | Designer agent — UI/UX |
| `TESTER_MODEL` | Tester agent — test cases |
| `CLARIFIER_MODEL` | Clarifier agent — gap analysis |

### Nhóm RAG API

| Biến | Mặc định | Mô tả |
|---|---|---|
| `RAW_DATA_DIR` | `/data/raw` | Thư mục chứa tài liệu cần ingest |
| `QDRANT_URL` | `http://qdrant:6333` | URL Qdrant REST API |
| `COLLECTION_NAME` | `yan_raw_docs` | Tiền tố tên collection: `yan_raw_docs__{project}` |
| `CHUNK_SIZE` | `1000` | Kích thước chunk (ký tự) khi chia tài liệu |
| `CHUNK_OVERLAP` | `200` | Overlap giữa các chunk kề nhau |
| `UPSERT_BATCH_SIZE` | `64` | Số vector upsert vào Qdrant mỗi batch |
| `RAG_TOP_K` | `5` | Số chunk trả về mỗi query (default) |
| `INGEST_EMBED_WORKERS` | `1` | Số request embed asyncio đồng thời khi ingest |
| `INGEST_EMBED_RETRIES` | `3` | Số lần retry khi embed một batch lúc ingest lỗi (OOM / HTTP tạm thời) |
| `RAG_TIMEOUT` | `600` | Ở rag-api: timeout (giây) mỗi lần gọi Ollama `/api/embed` — cả embed câu hỏi (`/ask`, `/retrieve`) lẫn từng batch lúc ingest (trước đây ingest dùng cứng 600 s). Cùng biến với agent-api, xem bên dưới |
| `GRAPH_ENABLED` | `true` | Bật Neo4j graph enrichment (`false` = rag-api không chạm tới Neo4j) |
| `GRAPH_ENTITY_EXTRACTION` | `false` | LLM extract entity khi ingest (chậm hơn nhưng tăng độ phong phú graph). Chỉ áp cho chunk **mới** — bật sau lần ingest đầu thì chạy `/reset-ingest` để trích xuất cho chunk đã có |

### Nhóm Agent API

| Biến | Mặc định | Mô tả |
|---|---|---|
| `RAG_TIMEOUT` | `600` | Ở agent-api: timeout (giây) cho mỗi lần gọi RAG `/retrieve` từ agent (chỉ embed + search) |
| `MAX_FILES_PER_ROLE` | `6` | Số file tối đa mỗi coding agent sinh trong 1 workflow |
| `CLARIFIER_REGEN_LOOPS` | `1` | Số vòng lặp Clarifier re-generation (0 = tắt) |
| `WORKFLOW_MAX_SECONDS` | `0` | Ngân sách thời gian tổng của một workflow (giây), kiểm tra giữa các node. `0` = không giới hạn |
| `WORKFLOW_CONCURRENCY` | `1` | Số workflow chạy đồng thời; workflow khác nằm `pending` trong hàng đợi FIFO. Giữ `1` (Ollama chỉ có một slot, một model nạp sẵn) — xem [Hàng đợi workflow](#hàng-đợi-workflow-workflow_concurrency) |
| `WORKFLOW_DB_PATH` | `/data/memory/workflows.sqlite3` | File SQLite lưu trạng thái workflow — nằm trong volume `./data/memory` nên còn sau restart / rebuild. Xem [Lưu trạng thái workflow](#lưu-trạng-thái-workflow-sqlite-và-khôi-phục-sau-restart) |
| `PLANNER_NUM_PREDICT` | `2048` | Trần token (`num_predict`) cho planner JSON danh sách file. Ghi chú chung cho ba biến `*_NUM_PREDICT` ở dòng `REPORT_NUM_PREDICT` |
| `FILEGEN_NUM_PREDICT` | `8192` | Trần token cho mỗi lần sinh một file code |
| `REPORT_NUM_PREDICT` | `16384` | Trần token cho báo cáo Markdown (gọi LLM một lần) của mỗi agent. **Ba biến `*_NUM_PREDICT`:** giá trị env là ngân sách cho **phần trả lời**. Lời gọi có thinking (model thuộc danh sách reasoning: `qwen3*`, `north-mini-code*`, `gemma4*`, `deepseek-r1*`, `qwq*`, `phi4-reasoning*`…, hoặc `FILEGEN_THINK=true`) được nhân ×2 vì token suy luận tính chung ngân sách `num_predict`; kết quả kẹp tối đa ở context window (`OLLAMA_CONTEXT_LENGTH`). `0` hoặc âm = không đặt trần. Với model mặc định (mọi agent model đều là reasoning) và `OLLAMA_CONTEXT_LENGTH=32768`, trần thực tế là 16384 token mỗi file và 32768 mỗi báo cáo; chỉ planner (`granite4.1:8b`) dùng đúng 2048. Chạm trần → log cảnh báo `chạm trần ...` |
| `FILEGEN_THINK` | *(rỗng)* | Bật/tắt thinking khi sinh file code. Chỉ nhận `true`/`false` (không phân biệt hoa thường); giá trị khác bị bỏ qua kèm log cảnh báo. Rỗng = theo mặc định của model. Chỉ áp cho lời gọi sinh từng file (không áp cho planner hay báo cáo). Model không hỗ trợ thinking mà đặt `true` thì mọi file sẽ báo lỗi |
| `MEMORY_DIR` / `ARTIFACT_DIR` | `/data/memory` / `/data/artifacts` | Cố định trong compose, khớp volume `./data/memory` và `./data/artifacts` |

### Nhóm Neo4j

| Biến | Mặc định | Mô tả |
|---|---|---|
| `NEO4J_URI` | `bolt://neo4j:7687` | Bolt URI kết nối Neo4j |
| `NEO4J_USERNAME` | `neo4j` | Username Neo4j |
| `NEO4J_PASSWORD` | — | Password Neo4j — **phải đổi trước khi deploy** |
| `NEO4J_DATABASE` | `neo4j` | Tên database |

---

## 8. Agent API — tài liệu đầy đủ

**Base URL:** `http://localhost:8091`

### GET /health

Kiểm tra trạng thái service và cấu hình runtime.

```bash
curl -s http://localhost:8091/health | jq
```

```json
{
  "status": "ok",
  "ollama_base_url": "http://ollama:11434",
  "rag_api_url": "http://rag-api:8090",
  "agents": 15,
  "workflow_steps": ["ba","pm","sa","ta","designer","tl","fe","mobile","dba","be","da","tech_lead","tester","devsecops","clarifier"]
}
```

### GET /agents

Liệt kê cấu hình tất cả agent: step_id, name, model, depends_on.

```bash
curl -s http://localhost:8091/agents | jq
```

```json
{
  "ba": { "step_id": 1, "name": "BA Agent — Business Analysis", "model": "qwen3.6:35b", "depends_on": [] },
  "pm": { "step_id": 2, "name": "PM Agent — Project Management", "model": "qwen3.6:35b", "depends_on": ["ba"] }
}
```

### GET /ui

Giao diện web để theo dõi và tương tác với SDLC workflow. `GET /ui` (và `/ui/`) trả trang `workflow.html`; mọi file trong `agent-api/static/` được phục vụ tại `/ui/static/<đường dẫn>` (thư viện JS vendor: `/ui/static/vendor/<file>`) — UI không tải gì từ CDN.

```
http://localhost:8091/ui
http://localhost:8091/ui?id=<workflow_id>     # mở thẳng một workflow
```

Chi tiết tính năng: [Giao diện theo dõi workflow (/ui)](#giao-diện-theo-dõi-workflow-ui).

### POST /agent/{role}

Chạy đồng bộ một agent đơn lẻ. Hữu ích để test từng role hoặc ghép nối thủ công.

| HTTP | Khi nào |
|---|---|
| `200` | Bước chạy xong. Với coding role (`fe`, `be`, ...) không sinh được file nào, `output` bắt đầu bằng `[LỖI trong <role>]` kèm section `## ⚠️ File generation errors` liệt kê lỗi từng file — kiểm tra tiền tố này |
| `404` | `role` không tồn tại |
| `422` | Body không hợp lệ (vd. `user_input` rỗng) |
| `500` | Lỗi LLM (timeout, mất kết nối, model chưa pull), hoặc output rỗng sau cả lần retry — ở agent báo cáo, và ở coding role khi planner không lập được kế hoạch file nên rơi về một lời gọi đơn — `detail` chứa lý do |

**Request body:**

| Trường | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `user_input` | string | ✅ | Mục tiêu hoặc context đầu vào |
| `project` | string\|null | | RAG project filter |
| `extra_context` | string\|null | | Context bổ sung thêm vào prompt |
| `prev_outputs` | object\|null | | Output của các agent trước `{"role": "text"}` |
| `tech_stack` | string[]\|null | | Danh sách tech stack bắt buộc |
| `rag_enabled` | bool | | Có query RAG không (mặc định: true) |
| `rag_top_k` | int | | Số kết quả RAG (mặc định: env `RAG_TOP_K`) |

**Ví dụ — chạy BA agent:**

```bash
curl -s -X POST http://localhost:8091/agent/ba \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Xây dựng hệ thống thanh toán cho marketplace",
    "project": "myproject",
    "rag_enabled": true,
    "rag_top_k": 5
  }' | jq .output
```

**Ví dụ — chạy FE agent với tech_stack cụ thể:**

```bash
curl -s -X POST http://localhost:8091/agent/fe \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Xây dựng trang quản lý đơn hàng cho admin",
    "tech_stack": ["React", "TypeScript", "TanStack Query", "Zustand", "Tailwind CSS"],
    "rag_enabled": false
  }' | jq .output
```

**Ví dụ — chạy SA agent với output BA đã có:**

```bash
BA_OUT=$(curl -s -X POST http://localhost:8091/agent/ba \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Xây dựng hệ thống thanh toán", "rag_enabled": false}' | jq -r .output)

curl -s -X POST http://localhost:8091/agent/sa \
  -H "Content-Type: application/json" \
  -d "{
    \"user_input\": \"Xây dựng hệ thống thanh toán\",
    \"prev_outputs\": {\"ba\": $(echo \"$BA_OUT\" | jq -R -s .)},
    \"tech_stack\": [\"NestJS\", \"PostgreSQL\", \"Redis\", \"Kubernetes\"]
  }" | jq .output
```

**Response:**

```json
{
  "role": "fe",
  "name": "FE Agent — Frontend Engineering",
  "model": "north-mini-code-1.0:q4_K_M",
  "output": "## FE Agent\n\n### FILE: src/pages/OrderManagement.tsx\n```typescript\n..."
}
```

### POST /workflow/run

Xếp hàng SDLC workflow 15 bước chạy nền (async). Trả về `workflow_id` và `queue_position` ngay lập tức. Workflow chạy khi tới lượt trong [hàng đợi](#hàng-đợi-workflow-workflow_concurrency).

**Request body:**

| Trường | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `user_input` | string | ✅ | Mục tiêu kinh doanh / ý tưởng project |
| `project` | string\|null | | RAG project filter |
| `rag_enabled` | bool | | Có query RAG mỗi bước không (mặc định: true) |
| `rag_top_k` | int | | Số kết quả RAG mỗi agent |
| `tech_stack` | string[]\|null | | Danh sách tech stack bắt buộc — binding cho tất cả agents |

**Ví dụ — workflow cơ bản:**

```bash
curl -s -X POST http://localhost:8091/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Xây dựng module quản lý tenant và billing cho SaaS platform đa khách hàng",
    "project": "myproject",
    "rag_enabled": true,
    "rag_top_k": 5
  }' | jq
```

**Ví dụ — workflow với tech_stack đầy đủ:**

```bash
curl -s -X POST http://localhost:8091/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Xây dựng hệ thống checkout và thanh toán cho marketplace B2B",
    "project": "marketplace",
    "rag_enabled": true,
    "rag_top_k": 5,
    "tech_stack": [
      "Backend: NestJS (Node.js, TypeScript)",
      "Frontend: React 18 + TypeScript + TanStack Query v5 + Zustand",
      "Mobile: React Native (Expo)",
      "Database: PostgreSQL 16",
      "Cache: Redis 7",
      "Message Queue: RabbitMQ",
      "Search: Elasticsearch",
      "Infra: Docker, Kubernetes (AWS EKS)",
      "CI/CD: GitHub Actions"
    ]
  }' | jq
```

**Response:**

```json
{
  "workflow_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "queue_position": 2,
  "message": "Workflow đã được xếp hàng. Poll GET /workflow/a1b2c3d4-... để kiểm tra trạng thái."
}
```

`queue_position`: vị trí trong hàng đợi, đếm từ 1 (1 = chạy kế tiếp); `null` khi workflow chạy ngay (còn slot trống).

### GET /workflow/{workflow_id}

Kiểm tra trạng thái hoặc lấy kết quả đã hoàn thành. Trạng thái đọc từ SQLite nên phản ánh tiến trình thật (ghi lại sau mỗi agent) và vẫn đọc được sau khi agent-api restart.

```bash
# Xem status
curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq .status

# Xem output của bước BA
curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq '.step_outputs.ba'

# Xem output của Clarifier (kết quả kiểm tra cuối)
curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq '.step_outputs.clarifier'

# Xem danh sách bước đã hoàn thành
curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq '.completed_steps'

# Kiểm tra lỗi nếu có
curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq '.error'

# Các bước lỗi (output bắt đầu bằng [LỖI ...])
curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq '.failed_steps'

# Vị trí hàng đợi / trạng thái hủy
curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq '{status, queue_position, cancel_requested}'
```

**Các trường chính của response:**

| Trường | Mô tả |
|---|---|
| `status` | `pending` \| `running` \| `completed` \| `failed` \| `cancelled` |
| `step_outputs` | `{role: markdown output}` của các bước đã chạy |
| `completed_steps` | Danh sách role đã chạy xong (theo thứ tự) |
| `failed_steps` | Role chạy xong nhưng output là lỗi (`[LỖI trong {role}]...`) |
| `error` | Tóm tắt lỗi (bước lỗi, timeout, bị hủy, bị gián đoạn) hoặc `null` |
| `artifacts` | `{role: [metadata file]}` — giống `GET /workflow/{id}/artifacts` |
| `queue_position` | Vị trí trong hàng đợi khi `pending` (1 = chạy kế tiếp); `null` khi không còn xếp hàng (sắp chạy, đang chạy hoặc đã kết thúc) |
| `cancel_requested` | `true` khi đã gửi yêu cầu hủy cho run đang chạy — workflow dừng sau bước hiện tại |
| `created_at` / `completed_at` | ISO-8601 UTC |

**Trạng thái workflow:** `pending → running → completed | failed | cancelled`

| Status | Ý nghĩa |
|---|---|
| `pending` | Đã xếp hàng, chưa bắt đầu (xem `queue_position`) |
| `running` | LangGraph đang thực thi các bước (bao gồm cả Clarifier regen loop) |
| `completed` | Hoàn tất toàn bộ + Clarifier regen loop (có thể kèm một vài bước lỗi trong `failed_steps`) |
| `failed` | Lỗi không xử lý được — kiểm tra trường `error`. Các thông báo cố định: `Bị gián đoạn do agent-api khởi động lại` (run đang chạy/xếp hàng lúc restart), `Không đọc được bản ghi workflow từ store — workflow không được chạy` (record trong SQLite hỏng/không đọc được lúc worker nhận slot — không node nào chạy), `Workflow dừng bất thường do lỗi hệ thống (<ExceptionType>)` (worker thoát bất thường; hậu tố là tên loại exception nếu có) |
| `cancelled` | Bị hủy qua `POST /workflow/{id}/cancel` — `error = "Đã hủy theo yêu cầu người dùng"`; output các bước đã xong vẫn giữ nguyên |

> Nếu một bước gặp lỗi LLM (timeout, model chưa pull, coding agent không sinh được file nào, agent báo cáo trả về output **rỗng** sau cả lần retry...), hệ thống ghi `[LỖI trong <role>]...` vào `step_outputs`, thêm role vào `failed_steps` và tóm tắt vào trường `error`, nhưng **workflow vẫn tiếp tục** các bước kế tiếp. Trong Clarifier regen loop: bước regen lỗi **giữ nguyên** output + artifact của lần chạy trước; bước từng lỗi mà regen thành công được gỡ khỏi `failed_steps` và khỏi tóm tắt trong `error`.

### POST /workflow/{workflow_id}/cancel

Hủy một workflow chưa kết thúc.

```bash
curl -s -X POST http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890/cancel | jq
```

| Trạng thái lúc hủy | Kết quả | Response |
|---|---|---|
| `pending` (đang chờ) | Bị hủy ngay, không bao giờ chạy | 200 `{"workflow_id": "...", "status": "cancelled"}` |
| `running` | Bước (agent) hiện tại **chạy nốt** — không thể ngắt giữa chừng một lần gọi LLM — rồi workflow dừng với `status = "cancelled"`. Trong lúc chờ, `GET /workflow/{id}` trả `cancel_requested: true` | 200 `{"workflow_id": "...", "status": "running"}` |
| Không tồn tại | — | 404 |
| Đã kết thúc (`completed` / `failed` / `cancelled`) | — | 409 |

Workflow bị hủy có `status = "cancelled"` và `error = "Đã hủy theo yêu cầu người dùng"`; output của các bước đã xong vẫn được giữ và xem được bình thường (kể cả artifact). Yêu cầu hủy đã được chấp nhận (200) thì kết quả cuối luôn là `cancelled`.

### GET /workflows

Liệt kê workflow gần nhất (mới nhất trước) — mặc định 50, thêm `?limit=N` (1–500) để xem nhiều hơn. `completed_steps` ở đây là **số** bước đã xong.

```bash
curl -s http://localhost:8091/workflows | jq '.workflows[] | {workflow_id, status, completed_steps, created_at, completed_at}'

# 200 workflow gần nhất
curl -s "http://localhost:8091/workflows?limit=200" | jq .count
```

```json
{
  "count": 1,
  "workflows": [
    {
      "workflow_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "status": "completed",
      "project": "myproject",
      "completed_steps": 15,
      "total_steps": 15,
      "created_at": "2026-06-06T10:00:00+00:00",
      "completed_at": "2026-06-06T10:30:42+00:00"
    }
  ]
}
```

### GET /workflow/{workflow_id}/artifacts

Liệt kê các file code đã được trích xuất và lưu vào disk.

```bash
curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890/artifacts | jq
```

```json
{
  "workflow_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "artifacts": {
    "fe": [
      {"path": "fe/_output.md", "filename": "_output.md", "language": "markdown", "size": 18234},
      {"path": "fe/src/pages/Login.tsx", "filename": "src/pages/Login.tsx", "language": "typescript", "size": 2341}
    ],
    "be": [
      {"path": "be/src/auth/auth.controller.ts", "filename": "src/auth/auth.controller.ts", "language": "typescript", "size": 3102}
    ]
  }
}
```

`filename` là đường dẫn tương đối trong thư mục role — dùng cho endpoint đọc file bên dưới. `language` là ngôn ngữ ghi trên code block lúc trích xuất (workflow chỉ còn trên disk thì suy từ đuôi file, vd. `md`, `tsx`).

### GET /workflow/{workflow_id}/artifacts/{role}/{path}

Đọc nội dung một file artifact cụ thể.

```bash
# Đọc nội dung file
curl -s "http://localhost:8091/workflow/a1b2c3d4.../artifacts/fe/src/pages/Login.tsx"

# Tải xuống dưới dạng binary
curl -O "http://localhost:8091/workflow/a1b2c3d4.../artifacts/fe/src/pages/Login.tsx?download=1"

# Đọc toàn bộ markdown output của một agent
curl -s "http://localhost:8091/workflow/a1b2c3d4.../artifacts/be/_output.md"
```

### Hàng đợi workflow (`WORKFLOW_CONCURRENCY`)

- agent-api chạy tối đa `WORKFLOW_CONCURRENCY` workflow cùng lúc (mặc định `1` — nên giữ `1` vì Ollama chỉ có một slot và một model nạp sẵn; chạy song song chỉ làm model bị swap liên tục). Workflow gửi thêm nằm ở trạng thái `pending` trong hàng đợi **FIFO** và chạy lần lượt.
- `queue_position` (trong response của `POST /workflow/run` và `GET /workflow/{id}`): vị trí trong hàng đợi, đếm từ 1 (1 = chạy kế tiếp). Giá trị là `null` khi workflow không còn xếp hàng (sắp chạy, đang chạy hoặc đã kết thúc).
- Hủy workflow đang chờ hoặc đang chạy: [`POST /workflow/{workflow_id}/cancel`](#post-workflowworkflow_idcancel).
- Hàng đợi và cờ hủy chỉ nằm trong bộ nhớ: khi agent-api khởi động lại, mọi workflow còn `pending`/`running` (kể cả workflow đang chờ hủy) chuyển sang `failed` với lỗi `"Bị gián đoạn do agent-api khởi động lại"` — hãy gửi lại nếu cần.

### Lưu trạng thái workflow (SQLite) và khôi phục sau restart

- agent-api lưu trạng thái mọi workflow vào SQLite tại `WORKFLOW_DB_PATH` (mặc định `/data/memory/workflows.sqlite3`, nằm trong volume `./data/memory` nên còn sau khi container restart hoặc được build lại). Mỗi bản ghi lưu dạng JSON đầy đủ, kèm cột index `workflow_id`, `status`, `created_at`. DB chạy WAL mode.
- Giữ tối đa **500** workflow mới nhất. Bản ghi đã kết thúc (`completed` / `failed` / `cancelled`) cũ nhất bị xóa trước; run đang `pending`/`running` không bao giờ bị xóa.
- Tiến trình được ghi lại sau mỗi agent hoàn thành, nên `GET /workflow/{id}` phản ánh tiến trình thật và vẫn đọc được sau restart.
- Nếu agent-api khởi động lại khi còn workflow đang chạy hoặc đang chờ, lúc startup các workflow đó chuyển sang `failed` với lỗi `"Bị gián đoạn do agent-api khởi động lại"`. Workflow đã kết thúc không bị thay đổi. Hiện **chưa hỗ trợ chạy tiếp (resume)** workflow bị gián đoạn: hãy chạy lại.
- Workflow cũ không có trong DB (chạy trước khi có SQLite, hoặc đã bị xóa do giới hạn 500) vẫn xem được nhờ dựng lại từ `data/artifacts` và `data/memory/episodic/workflow_runs.jsonl` (chỉ khôi phục được output của role có artifact). Run có artifact nhưng không có dòng hoàn tất trong JSONL được báo là `failed` (bị gián đoạn).
- Chỉ chạy agent-api với **1 process** (uvicorn 1 worker, như Dockerfile): bước khôi phục lúc startup coi mọi run đang chạy là của process cũ đã dừng.
- Nếu không mở được file SQLite (sai quyền, file hỏng) thì agent-api dừng ngay khi khởi động và ghi log lỗi. Sửa quyền hoặc xóa file (mất lịch sử workflow trong DB; artifact trên disk vẫn còn) rồi khởi động lại.

```bash
# Xem nhanh DB (cần sqlite3 trên host)
sqlite3 data/memory/workflows.sqlite3 \
  "SELECT workflow_id, status, created_at FROM workflows ORDER BY created_at DESC LIMIT 10;"
```

### Giao diện theo dõi workflow (/ui)

**Trạng thái, hủy và kết nối:**

- **Bước lỗi (✗ đỏ):** bước có output bắt đầu bằng `[LỖI …]` (hoặc nằm trong `failed_steps` khi chưa có output) hiện dấu ✗ đỏ trên sơ đồ 15 bước và trên tab; mở tab sẽ thấy thông báo "Bước này thất bại" ở đầu, chi tiết lỗi ngay bên dưới. Thanh trạng thái ghi thêm "· ✗ N lỗi" cạnh số bước. Workflow vẫn có thể "Completed" khi một vài bước lỗi.
- **Một số file không sinh được (⚠ vàng):** coding role sinh được một phần file — output có mục "⚠️ File generation errors" — hiện ⚠ trên tab và dấu ✓ màu vàng trên sơ đồ, kèm thông báo dẫn tới danh sách file lỗi. Các file còn lại vẫn dùng được.
- **Hàng đợi:** workflow đang chờ hiện "Đang chờ trong hàng đợi (vị trí N)" — vị trí 1 là chạy kế tiếp.
- **Hủy workflow:** nút "✕ Hủy workflow" hiện khi workflow `pending`/`running`. Sau khi xác nhận: run đang chờ bị hủy ngay; run đang chạy hiện "Đang hủy — sẽ dừng sau bước hiện tại" (bước đang chạy chạy nốt rồi dừng) và chuyển sang trạng thái **Cancelled** (badge xám). Workflow đã kết thúc thì không hủy được nữa.
- **Mất kết nối tạm thời:** khi agent-api khởi động lại hoặc mạng chập chờn (lỗi 5xx / không kết nối được), trang **không** xóa dữ liệu đang xem: hiện banner vàng "Mất kết nối tới agent-api…", tự thử lại sau 2.5s → 5s → 10s → 20s → tối đa 30s (hoặc bấm "Thử lại ngay"), banner tự ẩn khi kết nối lại. Chỉ khi workflow không tồn tại (404) trang mới dừng theo dõi và báo lỗi. Khi chuyển sang workflow khác, trang không bao giờ hiện dữ liệu của workflow trước trong lúc workflow mới chưa tải xong.
- **Tab ẩn:** khi tab trình duyệt bị ẩn, trang tạm dừng poll để đỡ tải agent-api; quay lại tab là cập nhật ngay.

**Điều hướng và hiển thị:**

- **Back/Forward:** URL `/ui?id=<workflow_id>` luôn khớp với workflow đang theo dõi. Mỗi lần mở một workflow khác thì có thêm một mục lịch sử; Back/Forward của trình duyệt chuyển qua lại giữa các workflow (workflow vừa rời đi sẽ ngừng poll). Back về `/ui` (không có `?id=`) thì quay lại màn hình trống. Bấm Track lại cùng ID hoặc tải lại trang không tạo mục lịch sử trùng.
- **Recent:** mục của hôm nay chỉ hiện giờ, mục của ngày khác hiện ngày + giờ. Di chuột lên mục để xem thời điểm đầy đủ.
- **Copy ID / Copy code chạy cả trên http thường:** Clipboard API của trình duyệt chỉ có trong secure context (https hoặc localhost). Khi mở UI qua `http://<ip>:8091`, UI tự chuyển sang cách sao chép cũ (`document.execCommand('copy')`). Mọi phản hồi (sao chép, hủy workflow, tải zip) hiện dưới dạng toast ở góc dưới phải, trình đọc màn hình cũng đọc được. Nếu không sao chép được, toast hiện luôn Workflow ID để chép tay.
- **Bảng và output dài:** ô bảng tự xuống dòng (cột đầu chỉ giữ một dòng khi mọi ô của cột đều ngắn). Bảng chỉ cuộn ngang trong khung riêng khi có quá nhiều cột, bảng cao cuộn dọc với hàng tiêu đề dính trên. Thanh tab các bước dính ngay dưới header khi cuộn output dài. Bước đang chạy tự được cuộn vào khung của dải tiến trình. Trên điện thoại, thanh trạng thái thu gọn và vẫn có nút hủy.
- **Khối code:** chỉ khối có ghi ngôn ngữ mà highlight.js hỗ trợ mới được tô màu; khối không ghi ngôn ngữ (sơ đồ ASCII, cây thư mục, log) không bao giờ bị tô. Việc tô màu chỉ chạy khi khối sắp hiện trên màn hình. Một số tên ngôn ngữ được đổi trước khi tô (bảng `HL_ALIASES` trong `workflow.html`): `tf`/`hcl`/`terraform`/`mermaid`/`text` → không tô; `conf` → nginx; `Dockerfile` → dockerfile; `Jenkinsfile`/`gradle` → groovy; `env` → ini; `shell`/`zsh` → bash. Mỗi khối code (và trình xem file artifact) có nút **Copy**.

**Chạy offline — thư viện giao diện đi kèm repo:**

Giao diện `/ui` không tải gì từ Internet/CDN. Các thư viện JS/CSS nằm sẵn trong `agent-api/static/vendor/`, được phục vụ qua `/ui/static/vendor/<file>`. Phiên bản ghi ngay trong tên file, mỗi thư viện kèm file LICENSE:

| Thư viện | File | Vai trò |
|---|---|---|
| marked 11.2.0 | `marked-11.2.0.min.js` | Render markdown |
| DOMPurify 3.4.17 | `dompurify-3.4.17.min.js` | Lọc HTML (giấy phép Apache-2.0 hoặc MPL-2.0) |
| highlight.js 11.12.0 (gói `@highlightjs/cdn-assets`) | `highlight-11.12.0.min.js` + `highlight-11.12.0-github-dark.min.css` | Tô màu code |
| highlight.js grammar lẻ | `highlight-11.12.0-<ngôn ngữ>.min.js`: dockerfile, nginx, dart, groovy, properties | Lấy nguyên bản từ `npm pack @highlightjs/cdn-assets@11.12.0`, dùng chung `highlight-11.12.0.LICENSE`, phải nạp **sau** `highlight-11.12.0.min.js` |
| JSZip 3.10.2 | `jszip-3.10.2.min.js` | Chỉ nạp khi bấm "Download All" lần đầu |

Nếu marked/DOMPurify không tải được (file thiếu hoặc bị chặn), trang vẫn poll bình thường và hiện output của từng bước dạng văn bản thô đã escape, kèm dòng cảnh báo "Không định dạng được markdown".

Nâng cấp một thư viện:
1. `npm pack <gói>@<phiên bản>`, giải nén, chép đúng file `.min.js`/`.css` và LICENSE vào `agent-api/static/vendor/`, đổi tên thành `<lib>-<x.y.z>…`
2. Sửa đường dẫn `/ui/static/vendor/...` trong `workflow.html`
3. Xóa file `.js`/`.css` của bản cũ
4. Chạy `python -m pytest tests/test_workflow_ui.py`. Test kiểm tra: không còn CDN, mọi file được tham chiếu đều tồn tại, banner khớp phiên bản trong tên file, có LICENSE, không sót file `.js`/`.css` cũ.

Thêm ngôn ngữ cho highlight.js: chép `languages/<lang>.min.js` cùng phiên bản theo đúng quy tắc đặt tên `highlight-11.12.0-<lang>.min.js`, rồi thêm thẻ `<script>` sau `highlight-11.12.0.min.js` (test kiểm tra tên file, LICENSE, banner phiên bản và việc file được dùng).

---

## 9. RAG API — tài liệu đầy đủ

**Base URL:** `http://localhost:8090`

### GET /health

```bash
curl -s http://localhost:8090/health | jq
```

```json
{
  "status": "ok",
  "ollama_base_url": "http://ollama:11434",
  "qdrant_url": "http://qdrant:6333",
  "collection_prefix": "yan_raw_docs",
  "embedding_model": "qwen3-embedding:8b",
  "chat_model": "qwen3.6:35b",
  "rag_top_k": 5,
  "ollama_request_timeout": 1200.0,
  "graph": {
    "enabled": true,
    "neo4j_uri": "bolt://neo4j:7687",
    "entity_extraction": false,
    "neo4j_connected": true
  }
}
```

`ollama_request_timeout`: timeout (giây) mỗi lần gọi `CHAT_MODEL` — dùng để căn Valve `timeout` của tool [`yan_knowledge_base`](#tool-1-yan_knowledge_basepy--hỏi-đáp-tài-liệu-nội-bộ) (phải lớn hơn giá trị này).

### GET /projects

Liệt kê tất cả project và trạng thái index.

```bash
curl -s http://localhost:8090/projects | jq
```

```json
{
  "raw_data_dir": "/data/raw",
  "projects": {
    "myproject": {
      "collection": "yan_raw_docs__myproject",
      "indexed": true,
      "points_count": 287
    },
    "marketplace": {
      "collection": "yan_raw_docs__marketplace",
      "indexed": false,
      "points_count": null
    }
  }
}
```

### POST /ingest

Ingest tài liệu vào Qdrant (+ Neo4j nếu `GRAPH_ENABLED=true`).

**Ingest tăng dần:** `point_id` tất định theo (`relative_path`, `loader_index`, `chunk_index`, `content_hash`, `embedding_model`). Ingest không-reset chỉ embed + upsert chunk **mới/đã sửa**; chunk không đổi được bỏ qua (`skipped_unchanged`) — chạy lại khi không có gì đổi thì không gọi Ollama lần nào. Chunk của file đã sửa/xóa/đổi tên bị dọn khỏi Qdrant và Neo4j (`stale_removed`). Không còn lời gọi embed "health check": vector size đọc từ collection có sẵn, hoặc lấy từ batch embed thật đầu tiên.

> Payload của chunk không đổi **không** được ghi lại (doc_type/module suy luận, `created_at` giữ giá trị lần embed đầu). Sau khi đổi luật suy luận metadata hoặc `RAW_DATA_DIR`, chạy `/reset-ingest` để làm mới.
>
> Trích xuất thực thể (`GRAPH_ENTITY_EXTRACTION`) cũng chỉ chạy cho chunk **mới** trong Neo4j: bật `GRAPH_ENTITY_EXTRACTION` sau lần ingest đầu, hoặc muốn thử lại sau một lần trích xuất không ra thực thể nào → chạy `/reset-ingest`.

**Request body:**

| Trường | Mô tả |
|---|---|
| `project` | Chỉ ingest một project cụ thể. Bỏ trống để ingest tất cả. |
| `reset` | `true` = embed lại toàn bộ và dựng lại collection. Collection Qdrant cũ chỉ bị xóa khi batch embed thật **đầu tiên** thành công — Ollama không phản hồi hoặc `EMBEDDING_MODEL` mới chưa pull thì index vector cũ được giữ nguyên (response có `failed_batches` + `batch_error`). |

```bash
# Ingest tất cả projects
curl -s -X POST http://localhost:8090/ingest | jq

# Ingest một project cụ thể
curl -s -X POST http://localhost:8090/ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "myproject"}' | jq

# Reset và ingest lại một project
curl -s -X POST http://localhost:8090/ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "myproject", "reset": true}' | jq
```

**Response — ingest một project** (trả về phẳng):

```json
{
  "project": "myproject",
  "collection": "yan_raw_docs__myproject",
  "status": "ok",
  "documents": 24,
  "chunks": 287,
  "upserted": 3,
  "skipped_unchanged": 284,
  "failed_batches": 0,
  "batch_error": null,
  "points_count": 287,
  "graph_chunks_saved": 12,
  "stale_removed": 2,
  "idempotent": true,
  "reset": false
}
```

**Response — ingest tất cả project:** `status`, `projects_ingested`, `total_documents`, `total_chunks`, `total_upserted`, `total_skipped_unchanged`, `reset` và `details.{project}` (mỗi project có đúng các trường như trên).

| Trường (mỗi project) | Mô tả |
|---|---|
| `documents` / `chunks` | Số tài liệu đọc được / số chunk sau khi chia |
| `upserted` | Số chunk vừa embed + upsert ở lần này (bằng 0 khi không có gì đổi) |
| `skipped_unchanged` | Số chunk đã có sẵn trong Qdrant, không embed lại |
| `stale_removed` | Số point cũ (chunk của file đã sửa/xóa) đã xóa |
| `failed_batches` / `batch_error` | Số batch embed lỗi (đã bỏ qua sau `INGEST_EMBED_RETRIES` lần thử) và lỗi đầu tiên, ví dụ Ollama không phản hồi hoặc lệch dimension ("Chạy POST /reset-ingest") |
| `points_count` | Số point trong collection |
| `graph_chunks_saved` | Số Chunk ghi vào Neo4j ở lần này (bằng 0 khi không có gì đổi; file có chunk mới thì cả các chunk không đổi của file đó cũng được ghi lại để giữ chuỗi `NEXT_CHUNK`) |

Project không có file hỗ trợ trả `"status": "empty"`.

### POST /reset-ingest

Embed lại và dựng lại collection từ đầu (tương đương `POST /ingest` với `"reset": true`) — cho tất cả project, hoặc một project nếu truyền `project`. Collection Qdrant cũ chỉ bị xóa khi batch embed thật đầu tiên thành công, nên chạy lúc Ollama đang lỗi không làm mất index vector. Graph Neo4j của project theo cùng quy tắc: chỉ bị xóa (rồi dựng lại từ các chunk đã có vector) khi Qdrant thật sự đã dựng lại; không batch embed nào thành công (Ollama down, `EMBEDDING_MODEL` chưa pull) thì cả index Qdrant lẫn graph cũ giữ nguyên. Lưu ý: nếu embed thành công nhưng **upsert** vào Qdrant lỗi ở mọi batch, collection đã bị dựng lại nên cả Qdrant lẫn graph đều trống — kiểm tra `failed_batches` / `batch_error` và `points_count` trong response.

```bash
# Reset và ingest lại toàn bộ
curl -s -X POST http://localhost:8090/reset-ingest | jq

# Reset một project cụ thể
curl -s -X POST http://localhost:8090/reset-ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "myproject"}' | jq
```

### POST /ask

Hỏi đáp RAG: embed câu hỏi → tìm kiếm Qdrant + Neo4j → sinh câu trả lời bằng LLM.

**Request body:**

| Trường | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `question` | string | ✅ | Câu hỏi |
| `project` | string\|null | | Lọc theo project (null = search tất cả) |
| `module` | string\|null | | Lọc theo module trong project |
| `top_k` | int\|null | | Số chunk trả về (null = dùng env `RAG_TOP_K`) |

```bash
# Query toàn bộ (tất cả collections)
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Mô tả auth flow?"}' | jq

# Query theo project
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Schema billing như thế nào?", "project": "myproject"}' | jq

# Query theo project + module (chính xác nhất)
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "JWT refresh token flow?", "project": "myproject", "module": "auth"}' | jq

# Tùy chỉnh số chunk trả về
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Kiến trúc marketplace?", "project": "myproject", "top_k": 10}' | jq

# Chỉ lấy phần câu trả lời
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Database schema là gì?", "project": "myproject", "module": "billing"}' \
  | jq .answer
```

**Response:**

```json
{
  "answer": "Schema billing bao gồm...",
  "sources": [
    {
      "score": 0.91,
      "project": "myproject",
      "module": "billing",
      "doc_type": "schema",
      "chunk_type": "table",
      "source_file": "billing-schema.md",
      "relative_path": "myproject/billing/billing-schema.md",
      "file_type": "md",
      "chunk_index": 3,
      "preview": "| Field | Type | Description |..."
    }
  ]
}
```

**Metadata của `sources[]`:**

| Trường | Mô tả |
|---|---|
| `score` | Cosine similarity (0–1) |
| `project` | Tên project |
| `module` | Tên module (thư mục con) |
| `doc_type` | Loại tài liệu: `prd`, `schema`, `api`, `architecture`, `spec`, `other` |
| `chunk_type` | Loại chunk: `paragraph`, `table`, `code`, `heading` |
| `source_file` | Tên file gốc |
| `relative_path` | Đường dẫn tương đối trong `data/raw/` |
| `chunk_index` | Số thứ tự chunk trong file |
| `preview` | 200 ký tự đầu của chunk |

### POST /retrieve

Chỉ truy hồi (embed → Qdrant → Neo4j), **không** gọi `CHAT_MODEL`. Body giống `/ask`.
Trả về `context` (chunk nguyên văn theo định dạng `[SOURCE n]` / `[GRAPH CONTEXT n]`,
đúng chuỗi mà `/ask` đưa cho LLM) và `sources`. agent-api dùng endpoint này cho mọi
node workflow nên không phải swap sang `CHAT_MODEL` ở mỗi bước.

```bash
curl -s -X POST http://localhost:8090/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "JWT refresh token flow?", "project": "myproject", "top_k": 4}' | jq -r .context
```

> Tên `project` (mọi endpoint) phải là một tên thư mục đơn giản trong `data/raw/`: chữ, số,
> `_`, `.`, `-`, ký tự đầu là chữ/số. Tên sai quy tắc (`.`, `..`, `/`, `a/b`...) → HTTP 400.

### Qdrant — kiểm tra trực tiếp

```bash
# Liệt kê tất cả collections
curl -s http://localhost:6333/collections | jq

# Chi tiết một collection
curl -s http://localhost:6333/collections/yan_raw_docs__myproject | jq

# Đếm số points trong collection
curl -s -X POST http://localhost:6333/collections/yan_raw_docs__myproject/points/count \
  -H "Content-Type: application/json" \
  -d '{"exact": true}' | jq

# Xóa collection thủ công
curl -s -X DELETE http://localhost:6333/collections/yan_raw_docs__myproject | jq
```

### Ollama — kiểm tra trực tiếp

```bash
# Danh sách model đã pull
curl -s http://localhost:11434/api/tags | jq .models[].name

# Kiểm tra Ollama đang hoạt động
curl -s http://localhost:11434/

# Chat trực tiếp (không qua RAG)
curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6:35b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }' | jq .message.content
```

---

## 10. Artifact System

Sau khi mỗi coding agent hoàn thành, output markdown được tự động quét để trích xuất file code và lưu vào disk.

**Roles có artifact:** `fe`, `mobile`, `be`, `dba`, `da`, `tech_lead`, `devsecops` (field `extract_artifacts` trong [role registry](#role-registry--cấu-hình-hành-vi-theo-role))

**Layout thư mục artifact:**

```
/data/artifacts/{workflow_id}/{role}/
├── _output.md              ← Toàn bộ markdown output của agent (luôn có, ghi đầu tiên)
├── src/pages/Login.tsx     ← File được trích xuất từ ### FILE: directive
├── src/hooks/useAuth.ts
├── schema.sql
└── Dockerfile
```

Mỗi lần trích xuất artifact của một role (lần chạy đầu hoặc khi Clarifier re-gen role đó), thư mục `/data/artifacts/{workflow_id}/{role}/` được **dọn sạch trước khi ghi**: file của lần sinh trước mà lần này không còn sinh ra sẽ không còn trong danh sách artifact. Việc xóa chỉ diễn ra bên trong đúng thư mục role — không theo symlink, và từ chối nếu thư mục role nằm ngoài `ARTIFACT_DIR/<workflow>/<role>`.

File mà coding agent **không sinh được** (timeout, lỗi Ollama, nội dung rỗng…) không được lưu thành artifact — xem section `## ⚠️ File generation errors` trong `_output.md` của role ([Cơ chế sinh code](#cơ-chế-sinh-code-artifact-roles)).

**Cơ chế nhận diện file trong markdown:**

| Ưu tiên | Pattern nhận diện | Ví dụ |
|---|---|---|
| 1 (cao nhất) | `### FILE: path/to/file.ext` + code block ngay sau. Đường dẫn bọc backtick hoặc in đậm được bóc trang trí (`` ### FILE: `src/a.ts` ``, `### FILE: **src/a.ts**` → lưu `src/a.ts`) | `### FILE: src/auth/login.tsx` |
| 2 | Dòng comment đầu code block | `// filename: src/Login.tsx` |
| 3 | Bold/heading ngay trước code block | `**src/Login.tsx**` |
| 4 (fallback) | Đặt tên theo ngôn ngữ (6 ký tự đầu) + index. Dockerfile: block đầu là `Dockerfile`, các block sau `Dockerfile.02`, `Dockerfile.03`… (`docker build -f Dockerfile.02 .`). Tên tự động không bao giờ trùng file đã lưu | `yaml-01.yml`, `python-02.py`, `typesc-03.ts` |

**Chống lặp output của model nhỏ:** khi một dòng lặp quá 8 lần liên tiếp, phần lặp bị cắt và chèn một dòng marker `... (repeated pattern truncated)` viết bằng comment đúng cú pháp của file: `//` (TS/JS/Java/Go...), `#` (Python/YAML/shell/Dockerfile), `--` (SQL), `/* */` (CSS), `<!-- -->` (Markdown/HTML/XML). File JSON và định dạng không rõ cú pháp comment chỉ bị cắt, không chèn marker, nên file vẫn hợp lệ.

**Truy cập artifacts qua API:**

```bash
# Liệt kê tất cả artifacts của workflow
curl -s http://localhost:8091/workflow/{id}/artifacts | jq

# Đọc nội dung file
curl -s "http://localhost:8091/workflow/{id}/artifacts/fe/src/pages/Login.tsx"

# Tải xuống binary
curl -O "http://localhost:8091/workflow/{id}/artifacts/be/src/auth/auth.service.ts?download=1"

# Đọc raw output markdown của agent
curl -s "http://localhost:8091/workflow/{id}/artifacts/be/_output.md"
```

---

## 11. Clarifier Regen Loop

### Cơ chế hoạt động

1. **Workflow hoàn thành** → Clarifier agent sinh §10 "Recommended Re-generation List"
2. **Parse danh sách** → `_parse_clarifier_regen_list()` quét bảng §10, tìm tên role trong cột "Agent Role"
3. **Re-run từng agent** → mỗi role trong danh sách được chạy lại qua `run_single_step()` với `prev_outputs` đầy đủ nhất (bao gồm output mới của các agent đã re-gen trước đó trong cùng loop)
4. **Re-run Clarifier** → Clarifier được chạy lại với outputs đã cập nhật để đánh giá lại
5. **Lặp** → nếu Clarifier mới vẫn đề xuất re-gen → lặp tiếp, tối đa `CLARIFIER_REGEN_LOOPS` lần

### Roles có thể được re-gen

`fe`, `mobile`, `dba`, `be`, `da`, `tech_lead`, `devsecops` (field `regen_eligible` trong [role registry](#role-registry--cấu-hình-hành-vi-theo-role)). Role re-gen **thành công** thì thư mục artifact của role đó được dọn sạch và trích xuất lại (và role được gỡ khỏi `failed_steps` / `error` nếu lần đầu lỗi); re-gen lỗi (exception hoặc output `[LỖI trong <role>]`) giữ nguyên output và artifact cũ.

Các planning roles (`ba`, `pm`, `sa`, `ta`, `designer`, `tl`, `tester`) không nằm trong danh sách eligible để tránh thay đổi foundation của toàn pipeline.

### Cấu hình

```env
CLARIFIER_REGEN_LOOPS=1   # số vòng lặp tối đa
CLARIFIER_REGEN_LOOPS=0   # tắt hoàn toàn tính năng
CLARIFIER_REGEN_LOOPS=2   # tăng nếu muốn nhiều vòng tinh chỉnh hơn
```

### Theo dõi qua log

```bash
docker compose logs -f agent-api | grep "Clarifier regen"
```

Output mẫu:
```
Clarifier regen loop 1/1: re-generating roles=['be', 'dba']
Clarifier regen loop 1: be re-generated (15432 chars)
Clarifier regen loop 1: dba re-generated (8921 chars)
Clarifier regen loop 1: clarifier re-run xong (12340 chars)
```

---

## 12. Episodic Memory Logging

Mỗi lần workflow kết thúc (`completed`, `failed` hoặc `cancelled` — đúng một dòng mỗi run), agent-api tự động ghi log vào file dưới đây. Ngoại lệ: run bị gián đoạn do agent-api khởi động lại (chuyển sang `failed` lúc startup) **không** có dòng log — xem trạng thái của chúng qua `GET /workflow/{id}` hoặc `GET /workflows`.

```
data/memory/episodic/workflow_runs.jsonl
```

Mỗi dòng là một JSON object:

```json
{
  "workflow_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "project": "myproject",
  "user_input": "Xây dựng hệ thống thanh toán...",
  "completed_steps": ["ba","pm","sa","ta","designer","tl","fe","mobile","dba","be","da","tech_lead","tester","devsecops","clarifier"],
  "steps_count": 15,
  "status": "completed",
  "error": null,
  "duration_seconds": 1842.5,
  "timestamp": "2026-06-06T10:30:00+00:00"
}
```

**Truy vấn log:**

```bash
# Xem 10 workflow gần nhất
tail -n 10 data/memory/episodic/workflow_runs.jsonl | jq .

# Lọc workflow thất bại
cat data/memory/episodic/workflow_runs.jsonl | jq 'select(.status == "failed")'

# Thống kê thời gian trung bình (giây)
cat data/memory/episodic/workflow_runs.jsonl | jq '[.duration_seconds] | add/length'

# Lọc theo project
cat data/memory/episodic/workflow_runs.jsonl | jq 'select(.project == "myproject")'

# Workflow nào chạy không đủ 15 bước
cat data/memory/episodic/workflow_runs.jsonl | jq 'select(.steps_count < 15)'
```

---

## 13. Open WebUI — Custom Tools

### Cài đặt

1. Mở [http://localhost:8085](http://localhost:8085)
2. **Admin Panel → Tools → "+" (Add Tool)**
3. Dán nội dung file tool, nhấn **Save**
4. Bật tool trong **Model Settings → Tools** cho model đang dùng

> **Cập nhật tool:** tool được dán tay vào Admin Panel nên **không** tự cập nhật khi pull code mới — dán lại cả hai file (hiện là `yan_knowledge_base` **1.2.0** và `yan_agent_workflow` **2.1.1**). Giá trị Valve đã lưu từ bản cũ (vd. `timeout` 120, URL `localhost` / `host.docker.internal`) giữ nguyên tới khi sửa tay trong **Valves**.
>
> **URL theo tên service:** Valve URL mặc định là tên service trong compose (`http://rag-api:8090`, `http://agent-api:8091`) — đúng khi Open WebUI chạy cùng network `yan` của stack. `localhost` / `host.docker.internal` **không** dùng được từ trong container vì cổng host chỉ bind `127.0.0.1`. Open WebUI chạy ở container/compose khác thì nối vào network của stack rồi giữ nguyên URL theo tên service:
> ```bash
> docker network connect rag-agent-platform_yan <container>
> ```

### Tool 1: `yan_knowledge_base.py` — Hỏi đáp tài liệu nội bộ

**Mục đích:** Query RAG API từ giao diện chat Open WebUI — tìm thông tin trong PRD, spec, architecture docs đã ingest.

**Valves (cấu hình):**

| Valve | Mặc định | Mô tả |
|---|---|---|
| `rag_api_url` | `http://rag-api:8090` | URL rag-api — tên service trong compose. `localhost` / `host.docker.internal` không dùng được từ trong container vì cổng host chỉ bind `127.0.0.1` |
| `timeout` | `1260` | Timeout chờ `/ask` (giây). Phải ≥ `OLLAMA_REQUEST_TIMEOUT` của rag-api (1200 trong `.env`; xem `curl -s localhost:8090/health \| jq .ollama_request_timeout`) + thời gian embed/truy hồi. Tool bỏ cuộc sớm **không** hủy generation: Ollama vẫn sinh tiếp và chiếm slot duy nhất (`OLLAMA_MAX_LOADED_MODELS=1`). Với profile `.env` nặng hơn (`OLLAMA_REQUEST_TIMEOUT` 1800–3600), nâng Valve tương ứng |
| `top_k` | `null` | Số chunk trả về (null = dùng env `RAG_TOP_K`) |
| `default_project` | `null` | Project mặc định (null = search tất cả) |
| `default_module` | `null` | Module mặc định (null = toàn project) |

**Ví dụ sử dụng trong chat:**
```
Tìm trong knowledge base: auth flow của yanlib hoạt động như thế nào?
```

```
Tìm thông tin về billing schema trong project yanlib, module billing
```

### Tool 2: `yan_agent_workflow.py` — Chạy SDLC Workflow

**Mục đích:** Khởi chạy, theo dõi và hủy SDLC workflow, hoặc chạy từng agent đơn lẻ trực tiếp từ giao diện chat.

**Valves (cấu hình):**

| Valve | Mặc định | Mô tả |
|---|---|---|
| `agent_api_url` | `http://agent-api:8091` | URL agent-api — tên service trong compose (xem lưu ý ở [Cài đặt](#cài-đặt)) |
| `block_until_complete` | `false` | `false` = `run_sdlc_workflow` trả `workflow_id` ngay; `true` = poll đồng bộ tới khi kết thúc (có thể 30+ phút, dễ vượt timeout proxy/browser) |
| `timeout` | `3600` | Timeout cho single-step request `run_agent_step` (giây) — mức ước lượng, **không** phải giới hạn trên: `OLLAMA_REQUEST_TIMEOUT` (1200) chỉ chặn thời gian tới token đầu và khoảng lặng giữa hai chunk, không chặn tổng thời gian một lời gọi (xem [mục 7](#7-biến-môi-trường-chi-tiết)); một bước báo cáo gọi LLM 1–2 lần, bước code gọi planner + từng file nên có thể lâu hơn — dùng `run_sdlc_workflow` cho bước code dài. Với profile `.env` nặng hơn (`OLLAMA_REQUEST_TIMEOUT` 1800–3600, model reasoning), nâng Valve tương ứng. Tool bỏ cuộc sớm thì agent-api vẫn chạy tiếp và giữ slot Ollama |
| `poll_interval` | `15` | Khoảng cách poll workflow status (giây) |
| `poll_max_attempts` | `120` | Số poll tối đa trước khi timeout (120 × 15s = 30 phút) |
| `rag_enabled` | `true` | Query RAG mỗi agent step |
| `rag_top_k` | `5` | Số kết quả RAG mỗi agent |
| `default_project` | `null` | Project mặc định |

**Functions:**

| Function | Mô tả |
|---|---|
| `run_sdlc_workflow(user_input, project, tech_stack)` | Gửi workflow 15 bước. Mặc định trả `workflow_id` ngay (chạy nền); Valve `block_until_complete=true` thì poll tới khi kết thúc (`completed` / `failed` / `cancelled`) |
| `run_agent_step(role, user_input, project, tech_stack)` | Chạy 1 agent đơn lẻ (sync) |
| `get_workflow_result(workflow_id, role)` | Lấy output theo workflow ID, lọc theo role. Summary gồm status, vị trí hàng đợi khi `pending`, các bước lỗi (`failed_steps`), trạng thái hủy |
| `cancel_workflow(workflow_id)` | Hủy workflow đang `pending`/`running` (`POST /workflow/{id}/cancel`). 404 = không tìm thấy, 409 = đã kết thúc |
| `list_agent_roles()` | Liệt kê tất cả roles, models, thứ tự chạy |

**Ví dụ sử dụng trong chat:**
```
Chạy SDLC workflow: xây dựng module marketplace cho ứng dụng B2B, project=myproject
```

```
Chạy agent sa với yêu cầu: thiết kế kiến trúc hệ thống authentication đa tenant
```

```
Xem kết quả workflow abc123
```

```
Hủy workflow abc123
```

---

## 14. Neo4j — Graph Enrichment

### Kích hoạt Graph

```env
GRAPH_ENABLED=true
GRAPH_ENTITY_EXTRACTION=false
```

```bash
docker compose restart rag-api
```

> Vừa bật `GRAPH_ENABLED`: lần `POST /ingest` kế tiếp tự dựng graph cho mọi chunk chưa có trong Neo4j (không embed lại). Vừa bật `GRAPH_ENTITY_EXTRACTION` sau khi đã ingest: trích xuất thực thể chỉ chạy cho chunk **mới**, nên chạy `POST /reset-ingest` để trích xuất cho toàn bộ chunk hiện có.

### Kiểm tra trạng thái

```bash
curl -s http://localhost:8090/graph/status | jq
```

```json
{
  "enabled": true,
  "connected": true,
  "neo4j_uri": "bolt://neo4j:7687",
  "entity_extraction": false,
  "stats": {
    "projects": 2,
    "documents": 24,
    "chunks": 892,
    "entities": 0
  }
}
```

### Xem entities (khi `GRAPH_ENTITY_EXTRACTION=true`)

```bash
curl -s http://localhost:8090/graph/projects/myproject/entities | jq
```

### Neo4j Browser — Cypher Queries

Mở [http://localhost:7474](http://localhost:7474) (user: `neo4j`, password: giá trị `NEO4J_PASSWORD` trong `.env`).

```cypher
// Xem toàn bộ graph của một project
MATCH path = (:Project {name:"myproject"})-[:HAS_DOCUMENT]->(:Document)-[:HAS_CHUNK]->(:Chunk)
RETURN path LIMIT 50;

// Thống kê node counts
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC;

// Top entities được đề cập nhiều nhất (cần GRAPH_ENTITY_EXTRACTION=true)
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN e.name, e.type, count(c) AS mentions
ORDER BY mentions DESC LIMIT 20;

// Chunks đề cập một entity cụ thể
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name:"BillingPlan"})
RETURN c.source_file, c.chunk_index, c.text LIMIT 10;

// Entities co-occur (xuất hiện cùng nhau trong chunks)
MATCH (e1:Entity)-[:CO_OCCURS_WITH]-(e2:Entity)
RETURN e1.name, e2.name LIMIT 30;

// Xóa toàn bộ graph (reset thủ công)
MATCH (n) DETACH DELETE n;
```

---

## 15. Điều chỉnh model

Tất cả model được quản lý trong `.env`. Sau khi đổi model, chỉ cần restart service — **không cần rebuild image**.

```bash
# Đổi model reasoning — ví dụ nâng BA_MODEL
# 1. Sửa trong .env: BA_MODEL=qwen3.6:35b

# 2. Pull model mới
docker exec ollama ollama pull qwen3.6:35b

# 3. Restart agent-api
docker compose restart agent-api
```

```bash
# Đổi CHAT_MODEL (rag-api)
docker compose restart rag-api

# Đổi EMBEDDING_MODEL — bắt buộc re-ingest toàn bộ
docker compose restart rag-api
curl -s -X POST http://localhost:8090/reset-ingest | jq
curl -s -X POST http://localhost:8090/ingest | jq
```

### Quản lý model Ollama

```bash
# Pull model cụ thể
docker exec ollama ollama pull qwen3.6:35b

# Xem tất cả model đã pull
docker exec ollama ollama list

# Xóa model không cần thiết
docker exec ollama ollama rm old-model:tag
```

### Pull toàn bộ model từ `.env`

```bash
grep -E '^(EMBEDDING_MODEL|CHAT_MODEL|CODING_PLANNER_MODEL|BA_MODEL|PM_MODEL|SA_MODEL|TA_MODEL|DA_MODEL|TL_MODEL|FE_MODEL|MOBILE_MODEL|BE_MODEL|DBA_MODEL|TECH_LEAD_MODEL|DEVSECOPS_MODEL|TESTER_MODEL|DESIGNER_MODEL|CLARIFIER_MODEL)=' .env \
  | cut -d= -f2 \
  | sort -u \
  | xargs -I {} docker exec ollama ollama pull "{}"
```

### Cấu hình cho phần cứng yếu (16 GB RAM)

```env
EMBEDDING_MODEL=qwen3-embedding:0.6b
CHAT_MODEL=qwen3.5:9b
BA_MODEL=qwen3.5:9b
PM_MODEL=qwen3.5:9b
SA_MODEL=qwen3.5:9b
TA_MODEL=qwen3.5:9b
DA_MODEL=qwen3.5:9b
TL_MODEL=qwen3.5:9b
FE_MODEL=qwen3.5:9b
MOBILE_MODEL=qwen3.5:9b
BE_MODEL=qwen3.5:9b
DBA_MODEL=qwen3.5:9b
TECH_LEAD_MODEL=qwen3.5:9b
DEVSECOPS_MODEL=qwen3.5:9b
DESIGNER_MODEL=qwen3.5:9b
TESTER_MODEL=qwen3.5:9b
CLARIFIER_MODEL=qwen3.5:9b
CODING_PLANNER_MODEL=granite4.1:3b
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1
OLLAMA_CONTEXT_LENGTH=8192
MAX_FILES_PER_ROLE=3
CLARIFIER_REGEN_LOOPS=0
```

---

## 16. Vận hành Docker

### Khởi động & dừng

```bash
# Khởi động toàn bộ rag-agent-platform
docker compose up -d

# Dừng tất cả (giữ volumes)
docker compose down

# Dừng và xóa volumes — MẤT TOÀN BỘ DỮ LIỆU Ollama + Qdrant + Neo4j
docker compose down -v
```

### Rebuild sau khi cập nhật mã nguồn

```bash
# Rebuild agent-api
docker compose up -d --build agent-api

# Rebuild rag-api
docker compose up -d --build rag-api

# Rebuild cả hai
docker compose up -d --build agent-api rag-api
```

> Restart/rebuild agent-api **không** mất lịch sử workflow (SQLite trong `./data/memory`), nhưng workflow đang chạy hoặc đang chờ sẽ chuyển sang `failed` ("Bị gián đoạn do agent-api khởi động lại") và phải chạy lại — kiểm tra `curl -s localhost:8091/workflows | jq '.workflows[] | select(.status == "running" or .status == "pending")'` trước khi rebuild.

### Theo dõi log

```bash
# Tất cả services
docker compose logs -f

# Chỉ agent-api
docker compose logs -f agent-api

# 100 dòng gần nhất
docker compose logs --tail=100 agent-api

# Grep lỗi
docker compose logs agent-api | grep -i error
```

### Quản lý container

```bash
# Restart một service
docker compose restart agent-api

# Xem trạng thái tất cả container
docker compose ps

# Theo dõi tài nguyên CPU/RAM
docker stats

# Vào shell container
docker exec -it agent-api bash
docker exec -it ollama bash
```

### Watchtower — Auto-update

```bash
# Trigger update thủ công
curl -s -H "Authorization: Bearer ${WATCHTOWER_HTTP_API_TOKEN}" \
  http://localhost:8080/v1/update

# Xem metrics
curl -s -H "Authorization: Bearer ${WATCHTOWER_HTTP_API_TOKEN}" \
  http://localhost:8080/v1/metrics
```

> Watchtower chỉ cập nhật container có label `com.centurylinklabs.watchtower.enable=true` (`WATCHTOWER_LABEL_ENABLE=true`) — hiện là `open-webui` và `deunhealth`. Lịch `WATCHTOWER_SCHEDULE` (mặc định `0 0 4 * * *`) tính theo `TZ` truyền vào container watchtower.
>
> - `ollama`, `qdrant`, `neo4j` là service **stateful** → `watchtower.enable=false` để không bị recreate giữa lúc workflow/ingest đang chạy. Cập nhật thủ công khi rảnh: `docker compose pull ollama qdrant neo4j && docker compose up -d ollama qdrant neo4j`.
> - `rag-api` và `agent-api` dùng local build với `watchtower.enable=false` — dùng `docker compose up -d --build` khi cần cập nhật.

### Truy cập từ máy khác (LAN)

Mặc định mọi cổng chỉ bind `127.0.0.1` — chỉ truy cập được từ chính máy chạy Docker. Để máy khác trong LAN mở **Agent API UI** (`:8091/ui`) và **Open WebUI** (`:8085`):

1. Trong `.env`: đặt `BIND_ADDR=0.0.0.0` **và** `WEBUI_AUTH=true` (Open WebUI không có đăng nhập khi `WEBUI_AUTH=false` — ai vào được cổng là dùng được toàn bộ model/tool).
2. `docker compose up -d agent-api open-webui` để áp dụng binding mới.
3. Mở firewall host cho cổng 8091 / `OPEN_WEBUI_PORT` nếu cần, chỉ trong mạng tin cậy.

> ⚠ `agent-api` **chưa có xác thực**: ai truy cập được cổng 8091 đều chạy được workflow và tải artifact. Chỉ mở `BIND_ADDR=0.0.0.0` trong LAN tin cậy, hoặc đặt sau reverse proxy có auth.
>
> `ollama`, `qdrant`, `neo4j`, `rag-api` và watchtower API **luôn** chỉ bind `127.0.0.1` (không theo `BIND_ADDR`). Từ máy khác hãy dùng SSH tunnel, ví dụ `ssh -L 8090:127.0.0.1:8090 user@host`. Container khác trên host cũng **không** tới được các cổng này qua `host.docker.internal` — hãy join network `rag-agent-platform_yan` và gọi bằng tên service (`http://rag-api:8090`).

### Cập nhật tài liệu RAG

```bash
# 1. Thêm/sửa file
cp new-spec.md data/raw/myproject/auth/

# 2. Ingest lại (tăng dần — chỉ embed chunk mới/đã sửa, dọn chunk của file đã sửa/xóa)
curl -s -X POST http://localhost:8090/ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "myproject"}' | jq '{upserted, skipped_unchanged, stale_removed}'

#    Ingest tất cả project: jq .details.myproject.upserted

# 3. Kiểm tra
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "nội dung spec mới?", "project": "myproject", "module": "auth"}' | jq .answer
```

---

## 17. Troubleshooting

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `rag-api` / `agent-api` crash khi start | Thiếu env var bắt buộc | `docker compose logs rag-api` — tìm `RuntimeError` |
| `/ask` trả về 404 "chưa được ingest" | Chưa chạy `/ingest` | `curl -X POST localhost:8090/ingest -d '{"project":"..."}'` |
| `/ingest` trả về `"status": "empty"` | Không có subfolder trong `data/raw/` | Tạo subfolder và đặt file vào đó |
| `/ask` với `module` không trả kết quả | Module sai tên hoặc chưa ingest | Tên module = tên thư mục con chính xác |
| Ingest chậm | `INGEST_EMBED_WORKERS` quá thấp | Tăng bằng giá trị `OLLAMA_NUM_PARALLEL`; restart `rag-api` |
| Embedding timeout | Model chưa được pull, hoặc embed batch lớn trên máy chậm | Pull lại `EMBEDDING_MODEL`; tăng `RAG_TIMEOUT` (rag-api) |
| Ingest trả `failed_batches` > 0 / có `batch_error` | Ollama không phản hồi, embed lỗi sau `INGEST_EMBED_RETRIES` lần thử, hoặc lệch dimension (đổi `EMBEDDING_MODEL`) | Đọc `batch_error`; lệch dimension → `POST /reset-ingest`. Chunk lỗi chưa có vector, ingest lại sẽ embed bù |
| SDLC workflow `status=failed` | Xem field `error` | `curl .../workflow/{id} \| jq .error` |
| Workflow `failed` với lỗi "Bị gián đoạn do agent-api khởi động lại" | agent-api restart/rebuild khi workflow đang chạy hoặc đang chờ | Gửi lại workflow (chưa hỗ trợ chạy tiếp). Output các bước đã xong vẫn xem được |
| Workflow đứng ở `pending` lâu | Đang có workflow khác chạy (`WORKFLOW_CONCURRENCY=1`) | Xem `queue_position`; chờ hoặc hủy workflow đang chạy (`POST /workflow/{id}/cancel`) |
| Đã hủy nhưng workflow vẫn `running` | Bước hiện tại chạy nốt (không ngắt giữa lời gọi LLM) | Chờ bước hiện tại xong — `cancel_requested: true`, sau đó `status=cancelled` |
| agent-api dừng ngay khi khởi động, log lỗi SQLite | Không mở được `WORKFLOW_DB_PATH` (sai quyền, file hỏng) | Sửa quyền `./data/memory`, hoặc xóa file DB (mất lịch sử trong DB; artifact vẫn còn) rồi khởi động lại |
| Step output bắt đầu bằng `[LỖI` | LLM timeout, model chưa pull, hoặc coding agent không sinh được file nào — xem section `⚠️ File generation errors` | Kiểm tra model; tăng `OLLAMA_REQUEST_TIMEOUT`; xem `jq .failed_steps` |
| Output có section `⚠️ File generation errors` | Một số file lỗi: timeout, lỗi Ollama, model trả rỗng hoặc chạm trần `num_predict` giữa lúc suy luận | File lỗi không được lưu; tăng `OLLAMA_REQUEST_TIMEOUT` / `FILEGEN_NUM_PREDICT`; đặt `FILEGEN_THINK=false` nếu model suy luận quá dài |
| Bước LLM bị timeout sau khi cập nhật (trước đây chạy được) | `OLLAMA_REQUEST_TIMEOUT` nay mới thực sự được áp dụng (trước đây timeout vô hạn) | Máy chậm / chạy CPU: tăng `OLLAMA_REQUEST_TIMEOUT` (giới hạn thời gian tới token đầu tiên và khoảng lặng giữa hai chunk) |
| Log agent-api có cảnh báo `chạm trần ...` | Output bị cắt ở trần `num_predict` | Tăng `PLANNER_NUM_PREDICT` / `FILEGEN_NUM_PREDICT` / `REPORT_NUM_PREDICT` (≤ `OLLAMA_CONTEXT_LENGTH`) |
| UI hiện banner vàng "Mất kết nối tới agent-api…" | agent-api đang restart hoặc mạng chập chờn (5xx / không kết nối được) | UI tự thử lại (tối đa 30s một lần) và giữ dữ liệu đang xem; `docker compose ps agent-api` |
| UI hiện "Không định dạng được markdown" | Không tải được `marked`/`DOMPurify` từ `/ui/static/vendor/` | Kiểm tra file trong `agent-api/static/vendor/` rồi rebuild agent-api; output vẫn hiện dạng văn bản thô |
| Output chứa `<think>...</think>` thô | Model reasoning chưa nhận diện | Tên model phải chứa: `phi4-mini-reasoning`, `phi4-reasoning`, `qwq`, `deepseek-r1`, `qwen3`, `north-mini-code`, `gemma4` hoặc `mistral-medium-3.5` |
| `user_input` bị cắt ngắn | Input > 10.000 ký tự | Chia nhỏ input |
| `workflow_runs.jsonl` không được tạo | Volume chưa mount | Kiểm tra `./data/memory:/data/memory` trong docker-compose |
| Artifacts không được lưu | Volume chưa mount | Kiểm tra `./data/artifacts:/data/artifacts` trong docker-compose |
| Open WebUI không gọi được rag-api | URL dùng `localhost` / `host.docker.internal` thay vì hostname nội bộ (rag-api chỉ bind `127.0.0.1` trên host), hoặc Open WebUI không cùng network | Valve `rag_api_url` = `http://rag-api:8090`; Open WebUI ở container/compose khác: `docker network connect rag-agent-platform_yan <container>` |
| Open WebUI không gọi được agent-api | URL dùng `localhost` / `host.docker.internal`, hoặc Open WebUI không cùng network | Valve `agent_api_url` = `http://agent-api:8091`; Open WebUI ở container/compose khác: `docker network connect rag-agent-platform_yan <container>` |
| Tool `yan_knowledge_base` báo timeout dù rag-api vẫn đang trả lời | Valve `timeout` cũ (120) nhỏ hơn `OLLAMA_REQUEST_TIMEOUT` của rag-api — Ollama vẫn sinh tiếp và chiếm slot | Dán lại tool bản mới và đặt Valve `timeout` ≥ `curl -s localhost:8090/health \| jq .ollama_request_timeout` + thời gian truy hồi (mặc định 1260) |
| Máy khác trong LAN không mở được UI | Cổng chỉ bind `127.0.0.1` (mặc định) | Đặt `BIND_ADDR=0.0.0.0` + `WEBUI_AUTH=true` — xem [Truy cập từ máy khác (LAN)](#truy-cập-từ-máy-khác-lan) |
| `curl localhost:6333` / `:8090` từ máy khác bị từ chối | Hạ tầng luôn chỉ bind `127.0.0.1` | Chạy curl trên chính host hoặc dùng SSH tunnel |
| Neo4j không kết nối | Container chưa sẵn sàng (lần đầu tải APOC) | Chờ `docker compose ps` báo neo4j `healthy`; `docker compose logs neo4j` |
| `dependency failed to start: container qdrant is unhealthy` | Qdrant chưa mở cổng 6333 sau ~1 phút | `docker compose logs qdrant` (thường do volume `qdrant_storage` hỏng / thiếu quyền) |
| Biến trong `.env` không có tác dụng | Biến chưa được khai báo trong `environment:` của service | Thêm vào `docker-compose.yml`; chạy `pytest tests/test_config_wiring.py` để kiểm tra |
| `/graph/status` → `connected: false` | Sai `NEO4J_PASSWORD` | `.env` phải khớp với lần đầu Neo4j init |
| `graph_chunks_saved: 0` sau ingest | `GRAPH_ENABLED=false`, hoặc không có chunk nào mới/đã sửa (ingest tăng dần không ghi lại Chunk đã có trong Neo4j) | Bình thường nếu `upserted: 0` và `skipped_unchanged` > 0; nếu không, đổi `GRAPH_ENABLED=true` + rebuild `rag-api` |
| Workflow mất > 60 phút | Model nặng hoặc tài nguyên thiếu | Giảm song song; dùng model nhỏ hơn; đặt `WORKFLOW_MAX_SECONDS` để dừng sớm |
| `GET /workflow/{id}` trả 404 | Sai ID, hoặc workflow đã bị xóa khỏi SQLite (giữ 500 workflow mới nhất) mà không còn artifact/dòng JSONL nào trên disk | Kiểm tra ID qua `GET /workflows?limit=500`; workflow cũ có artifact vẫn xem được (dựng lại từ disk) |
| Clarifier regen không hoạt động | `CLARIFIER_REGEN_LOOPS=0` | Kiểm tra env var; xem log `grep "Clarifier regen"` |
| Task Completion Checklist thiếu | TL output không có bảng task board | Kiểm tra TL agent đã chạy thành công và sinh §4/§5/§6/§7 |

---

## 18. Danh mục URL dịch vụ

> Các URL dưới đây chỉ mở được **trên chính máy chạy Docker** (bind `127.0.0.1`). Open WebUI và Agent API mở được từ LAN khi `BIND_ADDR=0.0.0.0` — xem [Truy cập từ máy khác (LAN)](#truy-cập-từ-máy-khác-lan).

| Service | URL | Mô tả |
|---|---|---|
| **Open WebUI** | http://localhost:8085 | Giao diện chat chính |
| **RAG API** | http://localhost:8090 | FastAPI root |
| **RAG API Swagger** | http://localhost:8090/docs | Swagger UI — thử API trực tiếp |
| **RAG API ReDoc** | http://localhost:8090/redoc | ReDoc documentation |
| **Agent API** | http://localhost:8091 | SDLC Orchestrator root |
| **Agent API UI** | http://localhost:8091/ui | SDLC Workflow web interface |
| **Agent API Swagger** | http://localhost:8091/docs | Swagger UI — thử API trực tiếp |
| **Ollama API** | http://localhost:11434 | LLM inference API |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | Vector DB dashboard |
| **Qdrant REST** | http://localhost:6333 | Qdrant REST API |
| **Qdrant gRPC** | localhost:6334 | Qdrant gRPC endpoint |
| **Neo4j Browser** | http://localhost:7474 | Graph DB browser UI |
| **Neo4j Bolt** | bolt://localhost:7687 | Neo4j driver connection |
| **Watchtower** | http://localhost:8080 *(cần `WATCHTOWER_HTTP_API_TOKEN`)* | HTTP API: `/v1/update`, `/v1/metrics` |
| **Deunhealth** | *(không publish cổng)* | Health watchdog — `network_mode: none`, xem `docker compose logs deunhealth` |

---

## 19. Phát triển, test & CI

Test, lint và type check chạy **không cần Docker** (Ollama/Qdrant/Neo4j đều được mock). Dùng Python 3.11 và cài dependency của cả hai service:

```bash
python -m pip install pytest ruff mypy==2.4.1
python -m pip install -r rag-api/requirements.txt -r agent-api/requirements.txt
```

| Bước | Lệnh | Cấu hình |
|---|---|---|
| Unit test | `python -m pytest -q` | `[tool.pytest.ini_options]` trong `pyproject.toml` (thư mục `tests/`) |
| Lint | `ruff check rag-api agent-api open-webui-tools tests` | `[tool.ruff]` |
| Type check | Vòng lặp bên dưới | `[tool.mypy]` (Python 3.11, `ignore_missing_imports`, `explicit_package_bases`, `warn_unused_ignores`) |

`agent-api/app.py` và `rag-api/app.py` trùng tên module `app`, nên mypy chạy **riêng từng thư mục service** (import phẳng như trong container) — CI chạy đúng lệnh này:

```bash
for d in agent-api rag-api open-webui-tools; do
  (cd "$d" && python -m mypy --config-file ../pyproject.toml *.py)
done
```

**CI** (`.github/workflows/ci.yml`) chạy trên mỗi push vào `main`/`develop` và mỗi pull request, với Python 3.11 và 3.12: cài tooling (`mypy==2.4.1`) + dependency của rag-api/agent-api → `ruff` → `mypy` (cả 3 thư mục, báo hết lỗi rồi mới fail) → `pytest`.

Một số test đáng chú ý khi sửa code:

| Test | Kiểm tra |
|---|---|
| `tests/test_config_wiring.py` | Mọi env var code đọc đều được truyền qua khối `environment:` của service trong `docker-compose.yml` |
| `tests/test_role_registry*.py` | Bảng hành vi dẫn xuất từ `AgentConfig` và các bất biến của role registry |
| `tests/test_workflow_queue*.py` | Hàng đợi FIFO, `WORKFLOW_CONCURRENCY`, hủy workflow và các race hủy/hoàn tất |
| `tests/test_workflow_store.py` | SQLite store, retention 500 bản ghi, khôi phục sau restart |
| `tests/test_agent_app.py` | Endpoint agent-api (`/health`, `/agents`, `/agent/{role}`, `/workflow/*`, `/workflows`, artifacts, `/ui`, `/ui/static`) + recovery lúc khởi động |
| `tests/test_rag_api_endpoints.py` | Endpoint rag-api (`/ingest`, `/reset-ingest`, `/ask`, `/retrieve`, `/projects`, `/graph/status`, entities) và `_query_rag` của agent-api |
| `tests/test_workflow_ui.py` | UI `/ui`: không CDN, file vendor + LICENSE + phiên bản, trạng thái/poll (phần chạy JS của trang cần `node`; tự skip nếu không có) |
| `tests/test_ingest_incremental*.py` | Ingest tăng dần, `skipped_unchanged`, `stale_removed`, reset an toàn khi Ollama lỗi |
