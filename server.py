
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from agents.mathy import build_mathy
from agents.researcher import build_researcher
from agents.writer import build_writer
from graph import build_direct, build_graph, build_supervisor
from logger import reset_steps

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
PROJECT_ROOT = Path(__file__).parent
MEMORY_DB = str(PROJECT_ROOT / "memory.db")
RECURSION_LIMIT = int(os.getenv("AGENTRIX_RECURSION_LIMIT", "10"))

# Estado global compartilhado entre requests
runtime: dict = {}



@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[server] Carregando modelo Ollama: {MODEL}")
    print("[server] Subindo MCP server via stdio...")

    client = MultiServerMCPClient(
        {
            "switchboard": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "cwd": str(PROJECT_ROOT),
            }
        }
    )
    tools = await client.get_tools()
    print(f"[server] Tools carregadas do MCP: {[t.name for t in tools]}")

    researcher = build_researcher(tools, MODEL)
    writer = build_writer(tools, MODEL)
    mathy = build_mathy(tools, MODEL)
    supervisor = build_supervisor(MODEL)
    direct = build_direct(MODEL)

    async with AsyncSqliteSaver.from_conn_string(MEMORY_DB) as checkpointer:
        graph = build_graph(researcher, writer, mathy, supervisor, direct, checkpointer)
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
        "recursion_limit": RECURSION_LIMIT,
    }



@app.get("/health")
async def health():
    return {"ok": True, "graph_ready": "graph" in runtime, "model": MODEL}


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
    config = _config_for(body)
    user_msg = _build_user_message(body)

    async def event_source() -> AsyncIterator[bytes]:
        try:
            # stream_mode="messages" emite tuplas (chunk_de_mensagem, metadata)
            # com `metadata["langgraph_node"]` indicando o no que esta falando.
            # Filtramos o supervisor (saida structured) e propagamos so o resto.
            async for chunk, metadata in graph.astream(
                {"messages": [user_msg]},
                config=config,
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

    size = 0
    while chunk := await file.read(1024 * 64):
        size += len(chunk)
    return {
        "id": file.filename or "upload",
        "filename": file.filename,
        "content_type": file.content_type,
        "size": size,
        "note": "Upload recebido mas ignorado: o modelo atual nao processa imagens/arquivos.",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
