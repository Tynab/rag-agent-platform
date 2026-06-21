"""
title: rag-agent-platform - Kho tri thức
author: rag-agent-platform
description: >
    Tool hỏi đáp tài liệu nội bộ qua RAG API. Cho phép chạt trực tiếp
    với các PRD, spec, kiến trúc, schema đã được ingest vào Qdrant.
    Hỗ trợ lọc kết quả theo project và module để tăng độ chính xác retrieval.
required_open_webui_version: 0.3.0
requirements: requests
version: 1.1.0

Mô tả chi tiết
--------------
Tool này là giao tiếp trực tiếp giữa Open WebUI và RAG API của rag-agent-platform (cổng 8090).
Khi được gọi, tool gửi câu hỏi của người dùng tới endpoint POST /ask của
rag-api, nhận kết quả RAG hybrid (Qdrant vector search + Neo4j graph enrichment
+ Ollama LLM), và trả về câu trả lời kèm danh sách file nguồn.

Valves (biến cấu hình)
----------------------
    rag_api_url      URL của rag-api trong nội bộ Docker network.
                     Mặc định: http://rag-api:8090
                     Khi gọi từ ngoài container: http://localhost:8090

    timeout          Timeout tính bằng giây cho mỗi request.
                     Mặc định: 120 giây.

    top_k            Số chunk kết quả trả về. Null = dùng giá trị RAG_TOP_K
                     trong env của rag-api (mặc định 5).

    default_project  Tên project mặc định nếu người dùng không chỉ định.
                     Null = tìm kiếm trên tất cả collection.

    default_module   Tên module mặc định. Null = toàn bộ project.
                     Ví dụ: auth, billing, marketplace.

Hàm công khai
--------------
    ask_internal_docs(question, project, module) → str
        Tham số:
          question  Câu hỏi cần trả lời. Không được rỗng.
          project   Tên project cần query. Null = search tất cả project.
          module    Lọc theo module trong project. Null = toàn bộ project.
        Trả về:
          Câu trả lời dưới dạng text, kèm danh sách file nguồn
          và điểm similarity score.

Ví dụ sử dụng trong chat
-------------------------
    Tìm trong knowledge base: JWT refresh token flow hoạt động thế nào?
    Hỏi về billing schema của dự án yanlib, module billing
    Kiến trúc marketplace là gì? (project=marketplace)
"""

import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        rag_api_url: str = Field(
            default="http://rag-api:8090",
            description=(
                "URL của rag-api service. Mặc định hostname nội bộ Docker network "
                "(http://rag-api:8090) — đúng khi Open WebUI chạy cùng compose `yan`. "
                "Nếu chạy ngoài network đó: http://localhost:8090 hoặc "
                "http://host.docker.internal:8090."
            ),
        )
        timeout: int = Field(
            default=120,
            description="Timeout cho mỗi request (giây)",
        )
        top_k: int | None = Field(
            default=None,
            ge=1,
            le=20,
            description="Số kết quả RAG trả về. Để trống dùng default của rag-api (RAG_TOP_K env).",
        )
        default_project: str | None = Field(
            default=None,
            description="Project mặc định. Để trống để search tất cả projects.",
        )
        default_module: str | None = Field(
            default=None,
            description="Module mặc định để lọc chunk (ví dụ: auth, billing, marketplace). Để trống để search toàn bộ project.",
        )

    def __init__(self):
        self.valves = self.Valves()

    def ask_internal_docs(self, question: str, project: str | None = None, module: str | None = None) -> str:
        """Hỏi đáp với tài liệu nội bộ của dự án qua RAG API.

        Dùng khi cần tìm thông tin trong các tài liệu kỹ thuật đã ingest:
        PRD, spec, kiến trúc, schema, auth, billing, marketplace...

        Tham số:
            question: Câu hỏi cần trả lời. Không được rỗng.
            project:  Tên project cần query (ví dụ: yanlib). Null = search tất cả.
            module:   Lọc theo module trong project (ví dụ: auth, billing). Null = toàn project.

        Trả về:
            Câu trả lời kèm danh sách file nguồn và điểm similarity score.
        """
        resolved_project = project or self.valves.default_project
        resolved_module = module or self.valves.default_module
        payload: dict = {"question": question}
        if resolved_project:
            payload["project"] = resolved_project
        if resolved_module:
            payload["module"] = resolved_module
        if self.valves.top_k is not None:
            payload["top_k"] = self.valves.top_k

        try:
            resp = requests.post(
                f"{self.valves.rag_api_url}/ask",
                json=payload,
                timeout=self.valves.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            return (
                f"❌ Không kết nối được rag-api tại {self.valves.rag_api_url}. "
                "Kiểm tra service đang chạy (`docker ps | grep rag-api`). Nếu Open WebUI "
                "chạy ngoài Docker network `yan`, đổi Valve rag_api_url sang "
                "http://localhost:8090 hoặc http://host.docker.internal:8090."
            )
        except requests.exceptions.Timeout:
            return f"❌ rag-api timeout sau {self.valves.timeout}s. Thử lại hoặc tăng timeout trong Valves."
        except requests.exceptions.HTTPError as exc:
            return f"❌ rag-api trả lỗi HTTP {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return f"❌ Lỗi không xác định: {exc}"

        answer = data.get("answer", "Không có câu trả lời.")
        sources = data.get("sources", [])

        if sources:
            seen: set = set()
            unique_files = [
                s["source_file"]
                for s in sources
                # type: ignore[func-returns-value]
                if s.get("source_file") and not (s["source_file"] in seen or seen.add(s["source_file"]))
            ]
            if unique_files:
                answer += "\n\n---\n**Nguồn:** " + " · ".join(unique_files)
            # Hiển thị module/doc_type nếu có để người dùng biết context đến từ đâu
            modules = {s.get("module") for s in sources if s.get("module")}
            if modules:
                answer += "\n**Module:** " + ", ".join(sorted(modules))

        return answer
