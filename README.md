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

---

## 1. Tổng quan rag-agent-platform

| Service | Image | Port (host) | Mục đích |
|---|---|---|---|
| `ollama` | `ollama/ollama:latest` | 11434 | LLM inference & embedding — chạy tất cả model AI |
| `qdrant` | `qdrant/qdrant:latest` | 6333 / 6334 | Vector database — lưu và tìm kiếm embedding (REST / gRPC) |
| `neo4j` | `neo4j:5-community` | 7474 / 7687 | Graph database — lưu quan hệ thực thể, tăng cường RAG |
| `rag-api` | build `./rag-api` | 8090 | FastAPI RAG service — ingest tài liệu và hỏi đáp AI |
| `agent-api` | build `./agent-api` | 8091 | FastAPI SDLC Agent Orchestrator — 15 AI agents phân tích & sinh code |
| `open-webui` | `ghcr.io/open-webui/open-webui` | 8085 | Giao diện chat kết nối Ollama & Qdrant |
| `watchtower` | `containrrr/watchtower` | — | Tự động pull & restart image mới nhất |
| `deunhealth` | `qmcgaw/deunhealth` | 9999 | Khởi động lại container khi healthcheck thất bại liên tục |

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

Các agent `fe`, `mobile`, `be`, `dba`, `da`, `tech_lead`, `devsecops` sử dụng quy trình sinh code **2 pha**:

**Pha 1 — Lập kế hoạch file:**
- `CODING_PLANNER_MODEL` (model nhẹ, ví dụ `granite4.1:8b`) nhận danh sách task từ TL output
- Sinh JSON array các file cần tạo: `[{"filename": "src/auth/login.tsx", "description": "...", "language": "typescript"}]`
- Tối đa `MAX_FILES_PER_ROLE` file mỗi role

**Pha 2 — Sinh từng file:**
- Mỗi file được sinh độc lập bằng coding model với context tập trung
- Nếu còn task TL chưa được cover → **pha follow-up**: lên kế hoạch thêm file và sinh tiếp

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
│   ├── app.py                        → FastAPI SDLC orchestrator: /workflow/run /agent/{role} /artifacts
│   ├── agents.py                     → 15 AgentConfig: model, system_prompt, depends_on, rag_query_hint
│   ├── workflow.py                   → LangGraph StateGraph: node factory, context builder, artifact generator
│   ├── artifacts.py                  → Trích xuất file code từ markdown output, lưu vào disk
│   ├── requirements.txt
│   ├── Dockerfile
│   └── static/
│       └── workflow.html             → SDLC Workflow UI (single-page app)
│
└── open-webui-tools/
    ├── yan_knowledge_base.py         → Open WebUI tool: query RAG knowledge base
    └── yan_agent_workflow.py         → Open WebUI tool: chạy SDLC workflow hoặc agent đơn lẻ
```

**Định dạng file được hỗ trợ khi ingest:**
`.md` `.txt` `.pdf` `.docx` `.csv` `.json` `.jsonl` `.yaml` `.yml` `.xml` `.html` `.py` `.js` `.ts` `.go` `.rs` `.java` `.cs` `.sh` `.sql` `.log`

---

## 6. Thiết lập từ đầu — hướng dẫn từng bước

### Bước 1 — Chuẩn bị file `.env`

Điền các biến bắt buộc vào `.env`:

```env
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
WATCHTOWER_SCHEDULE=0 * * * * *

# ─── Agent API ───────────────────────────────────────────────────────────────
MAX_FILES_PER_ROLE=6
CLARIFIER_REGEN_LOOPS=1
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

Kết quả thành công:

```json
{
  "status": "ok",
  "projects": {
    "myproject": {
      "upserted": 142,
      "skipped": 0,
      "errors": 0,
      "graph_chunks_saved": 142
    }
  }
}
```

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
  "message": "Workflow đã được xếp hàng. Poll GET /workflow/a1b2c3d4-... để kiểm tra trạng thái."
}
```

```bash
# Poll trạng thái (workflow thường mất 15–60 phút tùy model)
watch -n 30 'curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq .status'
```

### Bước 8 — Thiết lập Open WebUI Tools (tùy chọn)

1. Mở [http://localhost:8085](http://localhost:8085)
2. Vào **Admin Panel → Tools → "+"**
3. Dán nội dung `open-webui-tools/yan_knowledge_base.py` → **Save**
4. Lặp lại cho `open-webui-tools/yan_agent_workflow.py`
5. Bật tool trong **Model Settings → Tools**

---

## 7. Biến môi trường chi tiết

> **Lưu ý nguồn chuẩn (D6):** File `.env` đang chạy là **nguồn sự thật duy nhất**. Các giá trị ví dụ trong tài liệu này (model, `RAG_TOP_K`, `CHUNK_OVERLAP`, `UPSERT_BATCH_SIZE`…) mang tính minh hoạ và **có thể khác** `.env` thực tế — khi lệch nhau, hãy tin `.env`. Xem nhanh cấu hình runtime bằng `curl -s localhost:8090/health | jq` và `curl -s localhost:8091/agents | jq`.
>
> **Thư mục runtime (D8):** Trước lần `docker compose up` đầu tiên, tạo sẵn các thư mục mount để tránh Docker tự tạo (có thể bị root sở hữu trên Linux):
> ```bash
> mkdir -p ./data/raw ./data/memory/episodic ./data/artifacts
> ```

### Nhóm Ollama

| Biến | Mặc định | Mô tả |
|---|---|---|
| `OLLAMA_HOST` | `0.0.0.0` | Interface Ollama lắng nghe |
| `OLLAMA_ORIGINS` | `*` | CORS origins cho Ollama API |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | Context window (tokens) cho mọi LLM call trong agent-api |
| `OLLAMA_MAX_LOADED_MODELS` | `2` | Số model giữ đồng thời trong VRAM/RAM |
| `OLLAMA_NUM_PARALLEL` | `2` | Số request Ollama xử lý song song |
| `OLLAMA_KEEP_ALIVE` | `5m` | Thời gian giữ model trong bộ nhớ sau request cuối |
| `OLLAMA_REQUEST_TIMEOUT` | `1200` | Timeout (giây) cho mỗi lần gọi LLM trong agent-api |
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
| `GRAPH_ENABLED` | `false` | Bật Neo4j graph enrichment |
| `GRAPH_ENTITY_EXTRACTION` | `false` | LLM extract entity khi ingest (chậm hơn nhưng tăng độ phong phú graph) |

### Nhóm Agent API

| Biến | Mặc định | Mô tả |
|---|---|---|
| `RAG_TIMEOUT` | `600` | Timeout (giây) cho mỗi lần gọi RAG `/ask` từ agent |
| `MAX_FILES_PER_ROLE` | `6` | Số file tối đa mỗi coding agent sinh trong 1 workflow |
| `CLARIFIER_REGEN_LOOPS` | `1` | Số vòng lặp Clarifier re-generation (0 = tắt) |

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

Giao diện web để theo dõi và tương tác với SDLC workflow.

```
http://localhost:8091/ui
```

### POST /agent/{role}

Chạy đồng bộ một agent đơn lẻ. Hữu ích để test từng role hoặc ghép nối thủ công.

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

Gửi SDLC workflow 15 bước chạy nền (async). Trả về `workflow_id` ngay lập tức.

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
  "message": "Workflow đã được xếp hàng. Poll GET /workflow/a1b2c3d4-... để kiểm tra trạng thái."
}
```

### GET /workflow/{workflow_id}

Kiểm tra trạng thái hoặc lấy kết quả đã hoàn thành.

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
```

**Trạng thái workflow:**

| Status | Ý nghĩa |
|---|---|
| `pending` | Đã xếp hàng, chưa bắt đầu |
| `running` | LangGraph đang thực thi các bước (bao gồm cả Clarifier regen loop) |
| `completed` | Hoàn tất toàn bộ + Clarifier regen loop |
| `failed` | Lỗi không xử lý được — kiểm tra trường `error` |

> Nếu một bước gặp lỗi LLM (timeout, model chưa pull...), hệ thống ghi `[LỖI trong <role>]...` vào `step_outputs` và trường `error`, nhưng **workflow vẫn tiếp tục** các bước kế tiếp.

### GET /workflows

Liệt kê 50 workflow gần nhất.

```bash
curl -s http://localhost:8091/workflows | jq '[.[] | {workflow_id, status, created_at, completed_at}]'
```

### GET /workflow/{workflow_id}/artifacts

Liệt kê các file code đã được trích xuất và lưu vào disk.

```bash
curl -s http://localhost:8091/workflow/a1b2c3d4-e5f6-7890-abcd-ef1234567890/artifacts | jq
```

```json
{
  "fe": [
    {"role": "fe", "path": "src/pages/Login.tsx", "filename": "Login.tsx", "size_bytes": 2341},
    {"role": "fe", "path": "src/hooks/useAuth.ts", "filename": "useAuth.ts", "size_bytes": 876}
  ],
  "be": [
    {"role": "be", "path": "src/auth/auth.controller.ts", "filename": "auth.controller.ts", "size_bytes": 3102}
  ]
}
```

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
  "graph": {
    "enabled": true,
    "neo4j_uri": "bolt://neo4j:7687",
    "entity_extraction": false,
    "neo4j_connected": true
  }
}
```

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

Ingest tài liệu vào Qdrant (+ Neo4j nếu `GRAPH_ENABLED=true`). Idempotent — chạy lại không tạo bản sao.

| Trường | Mô tả |
|---|---|
| `project` | Chỉ ingest một project cụ thể. Bỏ trống để ingest tất cả. |
| `reset` | `true` = xóa collection cũ trước rồi ingest lại. |

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

**Response:**

```json
{
  "status": "ok",
  "projects": {
    "myproject": {
      "upserted": 287,
      "skipped": 0,
      "errors": 0,
      "graph_chunks_saved": 287
    }
  }
}
```

### POST /reset-ingest

Xóa tất cả collection rồi ingest lại toàn bộ từ đầu.

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

**Roles có artifact:** `fe`, `mobile`, `be`, `dba`, `da`, `tech_lead`, `devsecops`

**Layout thư mục artifact:**

```
/data/artifacts/{workflow_id}/{role}/
├── _output.md              ← Toàn bộ markdown output của agent (luôn có)
├── src/pages/Login.tsx     ← File được trích xuất từ ### FILE: directive
├── src/hooks/useAuth.ts
├── schema.sql
└── Dockerfile
```

**Cơ chế nhận diện file trong markdown:**

| Ưu tiên | Pattern nhận diện | Ví dụ |
|---|---|---|
| 1 (cao nhất) | `### FILE: path/to/file.ext` + code block ngay sau | `### FILE: src/auth/login.tsx` |
| 2 | Dòng comment đầu code block | `// filename: src/Login.tsx` |
| 3 | Bold/heading ngay trước code block | `**src/Login.tsx**` |
| 4 (fallback) | Đặt tên theo ngôn ngữ + index | `typescript-01.ts` |

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

`fe`, `mobile`, `dba`, `be`, `da`, `tech_lead`, `devsecops`

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

Mỗi lần workflow hoàn thành (thành công hoặc thất bại), agent-api tự động ghi log vào:

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

### Tool 1: `yan_knowledge_base.py` — Hỏi đáp tài liệu nội bộ

**Mục đích:** Query RAG API từ giao diện chat Open WebUI — tìm thông tin trong PRD, spec, architecture docs đã ingest.

**Valves (cấu hình):**

| Valve | Mặc định | Mô tả |
|---|---|---|
| `rag_api_url` | `http://rag-api:8090` | URL rag-api trong Docker network |
| `timeout` | `120` | Timeout request (giây) |
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

**Mục đích:** Khởi chạy toàn bộ SDLC workflow hoặc từng agent đơn lẻ trực tiếp từ giao diện chat.

**Valves (cấu hình):**

| Valve | Mặc định | Mô tả |
|---|---|---|
| `agent_api_url` | `http://agent-api:8091` | URL agent-api trong Docker network |
| `timeout` | `600` | Timeout cho single-step request (giây) |
| `poll_interval` | `15` | Khoảng cách poll workflow status (giây) |
| `poll_max_attempts` | `120` | Số poll tối đa trước khi timeout (120 × 15s = 30 phút) |
| `rag_enabled` | `true` | Query RAG mỗi agent step |
| `rag_top_k` | `5` | Số kết quả RAG mỗi agent |
| `default_project` | `null` | Project mặc định |

**Functions:**

| Function | Mô tả |
|---|---|
| `run_sdlc_workflow(user_input, project, tech_stack)` | Chạy đầy đủ 15 bước, poll đến khi xong, trả về summary |
| `run_agent_step(role, user_input, project, tech_stack)` | Chạy 1 agent đơn lẻ (sync) |
| `get_workflow_result(workflow_id, role)` | Lấy output theo workflow ID, lọc theo role |
| `list_agent_roles()` | Liệt kê tất cả roles, models, thứ tự chạy |

**Ví dụ sử dụng trong chat:**
```
Chạy SDLC workflow: xây dựng module marketplace cho ứng dụng B2B, project=myproject
```

```
Chạy agent sa với yêu cầu: thiết kế kiến trúc hệ thống authentication đa tenant
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

> `rag-api` và `agent-api` dùng local build với `watchtower.enable=false` — Watchtower không tự cập nhật chúng. Dùng `docker compose up -d --build` khi cần cập nhật.

### Cập nhật tài liệu RAG

```bash
# 1. Thêm/sửa file
cp new-spec.md data/raw/myproject/auth/

# 2. Ingest lại (idempotent — file cũ không bị duplicate)
curl -s -X POST http://localhost:8090/ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "myproject"}' | jq .projects.myproject.upserted

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
| Embedding timeout | Model chưa được pull | Pull lại `EMBEDDING_MODEL` |
| SDLC workflow `status=failed` | Xem field `error` | `curl .../workflow/{id} | jq .error` |
| Step output bắt đầu bằng `[LỖI` | LLM timeout hoặc model chưa pull | Kiểm tra model; tăng `OLLAMA_REQUEST_TIMEOUT` |
| Output chứa `<think>...</think>` thô | Model reasoning chưa nhận diện | Tên model phải chứa: `phi4-mini-reasoning`, `phi4-reasoning`, `qwq`, `deepseek-r1`, `qwen3`, `north-mini-code`, `gemma4` hoặc `mistral-medium-3.5` |
| `user_input` bị cắt ngắn | Input > 10.000 ký tự | Chia nhỏ input |
| `workflow_runs.jsonl` không được tạo | Volume chưa mount | Kiểm tra `./data/memory:/data/memory` trong docker-compose |
| Artifacts không được lưu | Volume chưa mount | Kiểm tra `./data/artifacts:/data/artifacts` trong docker-compose |
| Open WebUI không gọi được rag-api | URL dùng `localhost` thay vì hostname nội bộ | Valve `rag_api_url` = `http://rag-api:8090` |
| Open WebUI không gọi được agent-api | URL dùng `localhost` | Valve `agent_api_url` = `http://agent-api:8091` |
| Neo4j không kết nối | Container chưa sẵn sàng | Đợi ~30s; `docker compose logs neo4j` |
| `/graph/status` → `connected: false` | Sai `NEO4J_PASSWORD` | `.env` phải khớp với lần đầu Neo4j init |
| `graph_chunks_saved: 0` sau ingest | `GRAPH_ENABLED=false` | Đổi thành `true` + rebuild `rag-api` |
| Workflow mất > 60 phút | Model nặng hoặc tài nguyên thiếu | Giảm song song; dùng model nhỏ hơn |
| `workflow_id` not found sau vài giờ | In-memory store bị evict (tối đa 50) | Lưu output ngay; tăng `_MAX_STORED_WORKFLOWS` trong `app.py` |
| Clarifier regen không hoạt động | `CLARIFIER_REGEN_LOOPS=0` | Kiểm tra env var; xem log `grep "Clarifier regen"` |
| Task Completion Checklist thiếu | TL output không có bảng task board | Kiểm tra TL agent đã chạy thành công và sinh §4/§5/§6/§7 |

---

## 18. Danh mục URL dịch vụ

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
| **Deunhealth** | http://localhost:9999 | Health watchdog |
| **Watchtower** | http://localhost:8080 *(nội bộ)* | Auto-update metrics |
