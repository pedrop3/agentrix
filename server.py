
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from agents.compare_agent import build_compare_agent
from agents.faq_agent import build_faq_agent
from agents.web_agent import build_web_agent
from logger import _tool_event_queue
from agents.mathy import build_mathy
from agents.researcher import build_researcher
from agents.writer import build_writer
from config import config
from graph import build_direct, build_graph, build_supervisor
from logger import reset_steps, turn_summary

# ─── LangSmith tracing ────────────────────────────────────────────────────────
if config.langsmith.enabled and config.langsmith.api_key:
    import os
    os.environ.setdefault("LANGSMITH_TRACING",  "true")
    os.environ.setdefault("LANGSMITH_API_KEY",   config.langsmith.api_key)
    os.environ.setdefault("LANGSMITH_PROJECT",   config.langsmith.project)
    os.environ.setdefault("LANGSMITH_ENDPOINT",  config.langsmith.endpoint)
    print(f"[langsmith] Tracing activo → projecto '{config.langsmith.project}'")
else:
    print("[langsmith] Tracing desactivado (LANGSMITH_TRACING=false ou sem API key)")

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
    faq = build_faq_agent(tools)
    compare = build_compare_agent(tools)
    web = build_web_agent(tools)
    supervisor = build_supervisor(config.ollama.supervisor_model)
    direct = build_direct(config.ollama.direct_model)

    async with AsyncSqliteSaver.from_conn_string(config.memory_db) as checkpointer:
        graph = build_graph(researcher, writer, mathy, supervisor, direct, faq, compare, web, checkpointer)
        runtime["graph"] = graph
        runtime["mcp"] = client
        print("[server] Pronto. Agentes: faq · compare · web · researcher · writer · mathy · direct")
        print("[server] Endpoints: /chat /chat/stream /chat/resume /upload /health")
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


class ResumeBody(BaseModel):
    conversationId: str
    value: str  # label of the option selected by the user


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
    cfg: dict = {
        "configurable": {"thread_id": body.conversationId},
        "recursion_limit": config.recursion_limit,
    }
    # Metadados visíveis no LangSmith UI por cada run
    if config.langsmith.enabled and config.langsmith.api_key:
        cfg["run_name"] = f"turn/{body.conversationId[:8]}"
        cfg["tags"]     = ["agentrix", config.web.domain]
        cfg["metadata"] = {
            "conversation_id": body.conversationId,
            "domain":          config.web.domain,
        }
    return cfg



def _sse(data: dict) -> bytes:
    """Encode a dict as a Server-Sent Event data line (AG-UI format)."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


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
    try:
        result = await graph.ainvoke(
            {"messages": [_build_user_message(body)]},
            config=_config_for(body),
        )
        last = result["messages"][-1]
        return {"reply": last.content}
    finally:
        print(turn_summary())


def _streaming_response(generator: AsyncIterator[bytes]) -> StreamingResponse:
    """Wrap an async bytes generator into a StreamingResponse with SSE headers."""
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _run_graph_stream(
    graph,
    graph_input,
    run_config: dict,
    conversation_id: str,
    request: Optional[Request] = None,
):
    """Shared SSE generator for both /chat/stream and /chat/resume.

    Streams AG-UI events from a LangGraph run. After the astream loop ends,
    checks the graph state for a pending interrupt and emits an INTERRUPT event
    so the frontend can render interactive option cards.
    """
    run_id = str(uuid.uuid4())
    current_msg_id: str | None = None

    tool_q: asyncio.Queue = asyncio.Queue()
    ctx_token = _tool_event_queue.set(tool_q)

    def _drain_tool_queue():
        events = []
        while not tool_q.empty():
            events.append(tool_q.get_nowait())
        return events

    yield _sse({"type": "RUN_STARTED", "runId": run_id, "threadId": conversation_id})

    try:
        async for chunk, metadata in graph.astream(
            graph_input,
            config=run_config,
            stream_mode="messages",
        ):
            if request and await request.is_disconnected():
                break

            node = (metadata or {}).get("langgraph_node")
            if node in ("supervisor", "clarify"):
                continue

            # Always flush tool events before any text check so rag_search
            # badges (from faq_agent) appear even when we suppress the text.
            for ev in _drain_tool_queue():
                yield _sse(ev)

            # Suppress FAQ_MISS routing signals — internal signal from faq_agent
            # that the cache was empty. The supervisor re-routes to web_agent;
            # the user should never see this intermediate "FAQ_MISS: ..." text.
            raw_preview = getattr(chunk, "content", "") or ""
            if isinstance(raw_preview, str) and raw_preview.startswith("FAQ_MISS"):
                continue

            raw = getattr(chunk, "content", "") or ""
            if not isinstance(raw, str):
                try:
                    delta = "".join(
                        p.get("text", "")
                        for p in raw
                        if isinstance(p, dict) and p.get("type") != "thinking"
                    )
                except Exception:
                    delta = ""
            else:
                delta = raw

            if delta:
                if current_msg_id is None:
                    current_msg_id = str(uuid.uuid4())
                    yield _sse({
                        "type": "TEXT_MESSAGE_START",
                        "messageId": current_msg_id,
                        "role": "assistant",
                    })
                yield _sse({
                    "type": "TEXT_MESSAGE_DELTA",
                    "messageId": current_msg_id,
                    "delta": delta,
                })

        for ev in _drain_tool_queue():
            yield _sse(ev)

        if current_msg_id:
            yield _sse({"type": "TEXT_MESSAGE_END", "messageId": current_msg_id})

        # ── Check for pending interrupt (e.g. clarify node paused the graph) ──
        try:
            state = await graph.aget_state(run_config)
            all_interrupts = [
                intr.value
                for task in (state.tasks or [])
                for intr in (getattr(task, "interrupts", None) or [])
            ]
            if all_interrupts:
                intr_val = all_interrupts[0]
                print(f"[server] interrupt detected: {intr_val}")
                yield _sse({"type": "INTERRUPT", **intr_val})
        except Exception as exc:  # noqa: BLE001
            print(f"[server] aget_state error: {exc}")

        yield _sse({"type": "RUN_FINISHED", "runId": run_id, "threadId": conversation_id})

    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        for ev in _drain_tool_queue():
            yield _sse(ev)
        if current_msg_id:
            yield _sse({"type": "TEXT_MESSAGE_END", "messageId": current_msg_id})
        yield _sse({"type": "RUN_ERROR", "runId": run_id, "message": str(exc)})
    finally:
        _tool_event_queue.reset(ctx_token)
        print(turn_summary())


@app.post("/chat/stream")
async def chat_stream(body: ChatBody, request: Request):
    graph = _get_graph()
    reset_steps()
    run_config = _config_for(body)
    user_msg = _build_user_message(body)

    return _streaming_response(
        _run_graph_stream(graph, {"messages": [user_msg]}, run_config, body.conversationId, request)
    )


@app.post("/chat/resume")
async def chat_resume(body: ResumeBody, request: Request):
    """Resume a graph that was paused at an interrupt (e.g. clarify node).

    The frontend sends the value selected by the user; we pass it to the graph
    via Command(resume=value) so the clarify_node can receive it and continue.
    """
    from langgraph.types import Command

    graph = _get_graph()
    reset_steps()
    run_config = _config_for(body)

    return _streaming_response(
        _run_graph_stream(graph, Command(resume=body.value), run_config, body.conversationId, request)
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


@app.get("/documents/stats")
async def rag_stats():
    """Diagnostico: contagem por kind, entidades, mentions. Use pra entender
    o estado do RAG sem precisar passar por agent."""
    from rag import get_rag

    try:
        rag = get_rag()
        return rag.stats()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Erro lendo stats: {e}")


@app.delete("/documents/{kind}")
async def delete_documents_by_kind(kind: str):
    """Apaga TODOS os chunks de um kind. Use com cuidado.

    Caso de uso: `DELETE /documents/search_results` pra limpar snippets
    de DDG antigos que estavam confundindo o RAG-first.
    """
    from rag import get_rag

    try:
        rag = get_rag()
        removed = rag.delete_by_kind(kind)
        return {"kind": kind, "removed": removed}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Erro removendo kind={kind}: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=config.port,
        reload=config.reload,
    )
