
from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from agents.knower import build_knower
from agents.mathy import build_mathy
from agents.researcher import build_researcher
from agents.writer import build_writer
from config import config
from graph import build_direct, build_graph, build_supervisor
from logger import reset_steps

# Estado global compartilhado entre requests
runtime: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[server] Carregando modelo Ollama: {config.ollama.model}")
    print("[server] Subindo MCP server via stdio...")

    client = MultiServerMCPClient(
        {
            "switchboard": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "cwd": str(config.project_root),
            }
        }
    )
    tools = await client.get_tools()
    print(f"[server] Tools carregadas do MCP: {[t.name for t in tools]}")

    researcher = build_researcher(tools, config.ollama.researcher_model)
    writer = build_writer(tools, config.ollama.writer_model)
    mathy = build_mathy(tools, config.ollama.mathy_model)
    knower = build_knower(tools, config.ollama.knower_model)
    supervisor = build_supervisor(config.ollama.supervisor_model)
    direct = build_direct(config.ollama.direct_model)

    async with AsyncSqliteSaver.from_conn_string(config.memory_db) as checkpointer:
        graph = build_graph(researcher, writer, mathy, supervisor, direct, knower, checkpointer)
        runtime["graph"] = graph
        runtime["mcp"] = client
        print("[server] Pronto. Endpoints: /chat /chat/stream /upload /health")
        try:
            yield
        finally:
            runtime.clear()
            print("[server] Encerrando...")


app = FastAPI(title="Agentrix Switchboard", lifespan=lifespan)

# CORS aberto para facilitar dev local (Expo Web / device fisico).
# Em producao, restrinja a allow_origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class Attachment(BaseModel):
    id: str
    name: str
    uri: Optional[str] = None
    mimeType: Optional[str] = None
    size: Optional[int] = None
    base64: Optional[str] = None


class ChatBody(BaseModel):
    conversationId: str
    messages: list[ChatMessage]
    attachments: Optional[list[Attachment]] = None


def _get_graph():
    g = runtime.get("graph")
    if g is None:
        raise HTTPException(status_code=503, detail="Graph ainda nao esta pronto.")
    return g


def _build_user_message(body: ChatBody) -> HumanMessage:
    """
    Monta o HumanMessage a partir do payload. O qwen2.5 nao e multimodal,
    entao por ora apenas adicionamos uma nota com os nomes dos anexos.
    """
    last_text = body.messages[-1].content if body.messages else ""
    if body.attachments:
        names = ", ".join(a.name for a in body.attachments)
        suffix = f"\n\n[Anexos enviados pelo usuario: {names}]"
        last_text = (last_text + suffix) if last_text else suffix.strip()
    return HumanMessage(content=last_text)


def _config_for(body: ChatBody) -> dict:
    return {
        "configurable": {"thread_id": body.conversationId},
        "recursion_limit": config.recursion_limit,
    }



@app.get("/health")
async def health():
    return {
        "ok": True,
        "graph_ready": "graph" in runtime,
        "model": config.ollama.model,
    }


@app.post("/chat")
async def chat(body: ChatBody):
    graph = _get_graph()
    reset_steps()
    result = await graph.ainvoke(
        {"messages": [_build_user_message(body)]},
        config=_config_for(body),
    )
    last = result["messages"][-1]
    return {"reply": last.content}


@app.post("/chat/stream")
async def chat_stream(body: ChatBody, request: Request):
    graph = _get_graph()
    reset_steps()
    run_config = _config_for(body)  # nao usar `config` (shadow do modulo)
    user_msg = _build_user_message(body)

    async def event_source() -> AsyncIterator[bytes]:
        try:
            # stream_mode="messages" emite tuplas (chunk_de_mensagem, metadata)
            # com `metadata["langgraph_node"]` indicando o no que esta falando.
            # Filtramos o supervisor (saida structured) e propagamos so o resto.
            async for chunk, metadata in graph.astream(
                {"messages": [user_msg]},
                config=run_config,
                stream_mode="messages",
            ):
                if await request.is_disconnected():
                    break

                node = (metadata or {}).get("langgraph_node")
                if node == "supervisor":
                    # supervisor usa structured output (RouterDecision); nao vaza pro usuario
                    continue

                # chunk pode ser AIMessageChunk ou ToolMessage; pegamos so texto
                delta = getattr(chunk, "content", "") or ""
                if not isinstance(delta, str):
                    try:
                        delta = "".join(
                            p.get("text", "") for p in delta if isinstance(p, dict)
                        )
                    except Exception:
                        delta = str(delta)

                if delta:
                    payload = json.dumps({"delta": delta}, ensure_ascii=False)
                    yield f"data: {payload}\n\n".encode("utf-8")

            yield b"data: [DONE]\n\n"
        except asyncio.CancelledError:
            # cliente desconectou
            raise
        except Exception as exc:  # noqa: BLE001
            err = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"data: {err}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # desabilita buffering em nginx
            "Connection": "keep-alive",
        },
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Compat: o react-chat-ui usa /upload pra anexos genericos.
    Apenas le e retorna metadata. Nao indexa.
    """
    size = 0
    while chunk := await file.read(1024 * 64):
        size += len(chunk)
    return {
        "id": file.filename or "upload",
        "filename": file.filename,
        "content_type": file.content_type,
        "size": size,
        "note": "Use /documents/upload para indexar no RAG.",
    }


@app.post("/documents/upload")
async def documents_upload(file: UploadFile = File(...)):
    """
    Upload de documentos que devem ENTRAR no RAG (uploads persistentes).

    Aceita: .pdf, .docx, .txt, .md (texto extraido e indexado);
            imagens sao salvas mas nao indexadas (qwen2.5 nao e multimodal).

    Retorna: { id, filename, kind, chunks, size, note? }
    """
    import uuid as _uuid
    from pathlib import Path

    from extract import extract
    from rag import get_rag

    data = await file.read()
    size = len(data)

    # 1) Salva o original em uploads/ (mesmo que nao seja indexado)
    uploads_dir = Path(__file__).parent / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    doc_id = _uuid.uuid4().hex
    safe_name = (file.filename or doc_id).replace("/", "_").replace("\\", "_")
    saved_path = uploads_dir / f"{doc_id}__{safe_name}"
    saved_path.write_bytes(data)

    # 2) Extrai texto conforme tipo
    extracted = extract(data, file.filename or "", file.content_type)

    # 3) Indexa no RAG se houver texto
    chunks = 0
    note = extracted.note
    if extracted.text.strip():
        try:
            rag = get_rag()
            chunks = rag.index_text(
                extracted.text,
                source=f"upload://{safe_name}",
                kind="upload",
                extra_metadata={
                    "doc_id": doc_id,
                    "filename": safe_name,
                    "content_type": file.content_type or "",
                },
            )
        except Exception as e:
            note = f"Texto extraido mas indexacao falhou: {e}"

    return {
        "id": doc_id,
        "filename": file.filename,
        "saved_as": str(saved_path.name),
        "kind": extracted.kind,
        "content_type": file.content_type,
        "size": size,
        "chunks": chunks,
        "note": note,
    }


@app.get("/documents")
async def list_documents():
    """Lista as fontes ja indexadas no RAG (debug)."""
    from rag import get_rag

    try:
        rag = get_rag()
        return {"count": rag.count(), "sources": rag.sources()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Erro lendo RAG: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=config.port,
        reload=config.reload,
    )
