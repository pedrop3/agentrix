"""
Supervisor + StateGraph.

O supervisor olha o estado da conversa e decide o proximo passo:
researcher / writer / mathy / direct / END.

Cada sub-agent, depois de rodar, volta pro supervisor — que pode encerrar
ou rotear pra outro agent.
"""
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from config import config
from logger import LLMLogger, ToolOnlyLogger, get_logger

DIRECT_HISTORY_LIMIT = 6


def _inject_no_think(messages: list) -> list:
    """
    Anexa ' /no_think' ao final da ULTIMA HumanMessage da lista (qwen3 so
    respeita o switch quando ele esta na mensagem mais recente). Nao mexe
    nas outras mensagens. Funciona pra qualquer modelo (pra modelos que
    nao sao qwen3 vira so um sufixo inocuo).
    """
    if not messages:
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if isinstance(m, HumanMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            if "/no_think" in content:
                break  # ja tem
            out[i] = HumanMessage(content=f"{content} /no_think", name=getattr(m, "name", None))
            break
    return out


# ---------------------------------------------------------------------------
# Schema da decisao do supervisor (structured output)
# ---------------------------------------------------------------------------
class RouterDecision(BaseModel):
    """Routing decision."""

    next: Literal["researcher", "writer", "mathy", "knower", "direct", "END"] = Field(
        description="Which agent should act next, or END if the task is complete."
    )
    reason: str = Field(description="Brief reason for the decision (one sentence).")


def _supervisor_prompt() -> str:
    """Prompt do supervisor com dominio/nome de exibicao vindos do config."""
    web = config.web
    return f"""/no_think
You are a supervisor routing user requests to specialized agents.

Available agents:
- "researcher": searches the user's own saved notes (writer-created)
- "writer":     saves new notes to disk
- "mathy":      performs math calculations
- "knower":     answers questions using public {web.domain} content
                (web search + page fetch + RAG over indexed pages/uploads)
                Topic: {web.display_name}
- "direct":     conversational/contextual questions — NO math, NO {web.display_name} knowledge

Decision rules:
- If the message contains ANY arithmetic or numbers to compute -> mathy (MANDATORY)
- If the user asks to SAVE / CREATE / WRITE a note -> writer
- If the user asks to FIND something they previously saved -> researcher
- If the question is about {web.display_name} (its products, fees, FAQs, public
  documents) OR about content from documents the user uploaded -> knower
- If the question is conversational/greeting/context-only (no math, no {web.display_name}) -> direct
- If the previous agent message already completed the user's task -> END

CRITICAL: NEVER send math to "direct" or "knower". Even trivial expressions
like "2+2" MUST go to "mathy".

You MUST return valid JSON with fields {{"next": "...", "reason": "..."}}.
"""


# mantido por compat — modulos que importam SUPERVISOR_PROMPT continuam funcionando
SUPERVISOR_PROMPT = _supervisor_prompt()

DIRECT_PROMPT = """
You are a helpful assistant. Answer the user's question directly and concisely
using the conversation history as context. Do not use any tools.
Reply in 1-3 short sentences unless asked for more."""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_supervisor(model_name: str = None):
    """Returns a model that produces a RouterDecision via structured output."""
    name = model_name or config.ollama.supervisor_model
    # model = ChatOllama(
    #     model=name,
    #     temperature=config.ollama.temperature,
    #     num_ctx=config.ollama.num_ctx,
    #     keep_alive="24h",  # mantem modelo carregado entre requests
    #     callbacks=[LLMLogger("supervisor")],
    # )
    model = ChatGoogleGenerativeAI(
        model=config.gemini.model,
        google_api_key=config.gemini.api_key,
        temperature=config.gemini.temperature,
        callbacks=[LLMLogger("supervisor")],
    )

    return model.with_structured_output(RouterDecision)


def build_direct(model_name: str = None):
    """Returns a plain LLM for direct conversational answers."""
    name = model_name or config.ollama.direct_model
    return ChatGoogleGenerativeAI(
        model=config.gemini.model,
        google_api_key=config.gemini.api_key,
        temperature=config.gemini.temperature,
        callbacks=[LLMLogger("supervisor")],
    )


class SupervisorState(MessagesState):
    """Estado compartilhado: mensagens + proximo destino decidido."""

    next: str


def build_graph(researcher, writer, mathy, supervisor, direct, knower, checkpointer=None):
    def supervisor_node(state: SupervisorState):
        last = state["messages"][-1]
        if getattr(last, "name", None) in ("knower", "direct"):
            print("[supervisor] -> END  (agent already responded)")
            return {"next": "END"}
        # Pra rotear, o supervisor so precisa da MENSAGEM ATUAL do usuario.
        # Mandar todo o historico aqui = prompt enorme e routing igualmente correto.
        # Encolhe drasticamente o prompt do supervisor em conversas longas.
        messages = _inject_no_think([SystemMessage(content=SUPERVISOR_PROMPT), last])
        decision: RouterDecision = supervisor.invoke(messages)
        print(f"[supervisor] -> {decision.next}  ({decision.reason})")
        return {"next": decision.next}

    async def direct_node(state: SupervisorState):
        # Limita historico aos ultimos N mensagens (preserva contexto curto, evita
        # arrastar respostas longas anteriores que inflam o prompt).
        history = state["messages"][-DIRECT_HISTORY_LIMIT:]
        messages = _inject_no_think([SystemMessage(content=DIRECT_PROMPT), *history])
        response = await direct.ainvoke(messages)
        return {"messages": [AIMessage(content=response.content, name="direct")]}

    def _wrap(sub_agent, name: str, isolate: bool = False):
        """Wrap a sub-agent as a graph node.

        isolate=True: pass ONLY the last HumanMessage to the agent.
        Use for retrieval agents (knower) to prevent conversation history from
        a previous turn about topic A from biasing answers about topic B.

        Tool-callback fix: LangGraph's ToolNode does not inherit callbacks
        that are attached statically to the LLM model.  We pass a
        ToolOnlyLogger via the invocation config so on_tool_start /
        on_tool_end fire correctly inside create_react_agent, while all
        LLM-level events continue to come from the model's own callback.
        """
        # Resolve once at build time — loggers are registered at import time.
        _parent_logger = get_logger(name)
        _tool_cb = ToolOnlyLogger(_parent_logger) if _parent_logger else None

        async def node(state: SupervisorState):
            if isolate:
                # Retrieval agents must search fresh for each question.
                # Sending the full history causes the LLM to hallucinate by
                # reusing answers from previous turns about different topics.
                last_human = next(
                    (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
                    state["messages"][-1],
                )
                messages = [last_human]
            else:
                messages = state["messages"]

            # Pass ToolOnlyLogger in the invocation config so LangGraph
            # propagates it to ToolNode (and every other child node).
            invoke_cfg = {"callbacks": [_tool_cb]} if _tool_cb else {}
            result = await sub_agent.ainvoke({"messages": messages}, invoke_cfg or None)
            last = result["messages"][-1]
            return {
                "messages": [AIMessage(content=last.content, name=name)],
            }

        return node

    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("direct", direct_node)
    workflow.add_node("researcher", _wrap(researcher, "researcher"))
    workflow.add_node("writer", _wrap(writer, "writer"))
    workflow.add_node("mathy", _wrap(mathy, "mathy"))
    workflow.add_node("knower", _wrap(knower, "knower", isolate=True))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {
            "researcher": "researcher",
            "writer": "writer",
            "mathy": "mathy",
            "knower": "knower",
            "direct": "direct",
            "END": END,
        },
    )
    workflow.add_edge("direct", "supervisor")
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("mathy", "supervisor")
    workflow.add_edge("knower", "supervisor")

    return workflow.compile(checkpointer=checkpointer)
