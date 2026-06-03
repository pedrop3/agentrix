"""
Supervisor + StateGraph.

O supervisor olha o estado da conversa e decide o proximo passo:
researcher / writer / mathy / direct / END.

Cada sub-agent, depois de rodar, volta pro supervisor — que pode encerrar
ou rotear pra outro agent.
"""
from typing import Literal, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt as lg_interrupt
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

    next: Literal[
        "researcher", "writer", "mathy",
        "faq", "compare", "web",
        "direct", "clarify", "out_of_scope", "crisis", "END"
    ] = Field(
        description=(
            "Which agent should act next, or END if the task is complete. "
            "Use 'clarify' only when a product/service is ambiguous. "
            "Use 'out_of_scope' when the topic has no connection to banking/finance. "
            "Use 'crisis' when the user expresses self-harm, suicide, or severe distress."
        )
    )
    reason: str = Field(description="Brief reason for the decision (one sentence).")
    clarification_question: Optional[str] = Field(
        default=None,
        description="Only when next=clarify: short question to ask the user.",
    )
    clarification_options: Optional[list[str]] = Field(
        default=None,
        description="Only when next=clarify: 2-4 option strings the user can choose from.",
    )


def _supervisor_prompt() -> str:
    """Prompt do supervisor com dominio/nome de exibicao vindos do config."""
    web = config.web
    return f"""/no_think
You are a supervisor routing user requests to specialized agents for {web.display_name}, a financial services assistant.

## In-scope topics
Banking, finance, and {web.display_name} products: accounts, cards, loans, investments, insurance, fees,
transfers, payments, exchange rates, interest rates. Also: saving notes, math calculations, greetings,
follow-up questions about a previous answer, and clarification requests.

## Out-of-scope topics (examples)
Programming, technology, science, history, sports, entertainment, cooking, health (non-financial),
geography, general knowledge — anything with NO connection to banking, finance, or {web.display_name}.

Available agents:
- "researcher": searches the user's own saved notes (writer-created)
- "writer":     saves new notes to disk
- "mathy":      performs math calculations
- "faq":        FAST — queries local RAG cache only for {web.display_name} questions.
                Route here FIRST for single-product/FAQ questions about {web.display_name}.
                If faq responds with "FAQ_MISS:" → route to "web" next.
                If faq provided a complete answer → END.
- "compare":    side-by-side comparison of 2+ {web.display_name} products/services.
                Uses the Knowledge Graph (fees, requirements, benefits).
                Route here for questions with "diferença", "comparar", "vs", "entre X e Y",
                "qual é melhor", "qual tem menor taxa", etc.
- "web":        fetches fresh content from {web.domain} web.
                Route here when faq returned "FAQ_MISS:", or when live data is needed,
                or for general {web.display_name} questions that are NOT comparisons.
- "direct":     conversational/contextual — greetings, follow-ups, clarifications.
                NO math, NO {web.display_name} product knowledge.
- "clarify":    ask the user to disambiguate when a {web.display_name} product/service
                is ambiguous at the TYPE level OR the VARIANT level:
                  • TYPE unclear  → "cartão" (credit? debit? prepaid?),
                                    "conta" (à ordem? poupança? jovem?),
                                    "seguro" (vida? automóvel? habitação?)
                  • VARIANT unclear → "cartão de crédito" (Standard? Gold? Platinum?)
                → set clarification_question + clarification_options (2-4 items).
                Do NOT clarify if the user already named a specific product.
- "out_of_scope": the question has NO connection to banking, finance, or {web.display_name}.
                  Use this when the topic is completely outside the domain defined above.
                  Do NOT use for vague questions — give them the benefit of the doubt.

Decision rules (in order):
0. CRISIS FIRST: User expresses thoughts of self-harm, suicide, hopelessness, or severe
   emotional distress (e.g. "vou me matar", "não quero viver", "quero desaparecer") → "crisis"
   This rule overrides ALL others. Never route crisis to direct, web, or any other agent.
1. ANY arithmetic or numbers to compute → "mathy" (MANDATORY, even trivial like "2+2")
2. SAVE / CREATE / WRITE a note → "writer"
3. FIND something previously saved → "researcher"
4. Topic is clearly outside banking/finance (e.g. "what is OOP in Java",
   "who won the World Cup", "recipe for pasta") → "out_of_scope"
5. Message already contains "(opção selecionada pelo utilizador:...)" (user went through
   clarification) → do NOT clarify again. Route directly to "compare", "faq", or "web".
6. Product TYPE is ambiguous (e.g. "cartão", "conta", "seguro" without type) → "clarify"
   EXCEPTION: if the question is a comparison/recommendation ("qual melhor", "comparar",
   "diferença") with ambiguous type → still clarify TYPE only, then compare will handle variants.
7. Comparison between 2+ {web.display_name} products → "compare"
8. Single {web.display_name} product/FAQ question → "faq" first
   - If last agent was "faq" AND its reply starts with "FAQ_MISS:" → "web"
   - If last agent was "faq" AND it provided a real answer → END
9. Product VARIANT is ambiguous AND the question is NOT a comparison/recommendation → "clarify"
   NOTE: For comparison/recommendation questions, skip variant clarify — "compare" handles all variants.
10. General {web.display_name} question (not a comparison, not a single product FAQ) → "web"
11. Conversational / greeting / follow-up → "direct"
12. Previous terminal agent already completed the task → END

CRITICAL: NEVER send math to "direct", "faq", "compare", or "web".
CRITICAL: When in doubt about scope, prefer "direct" over "out_of_scope".
CRITICAL: Rule 0 (crisis) is absolute and cannot be overridden.

You MUST return valid JSON:
{{"next": "...", "reason": "...", "clarification_question": null, "clarification_options": null}}
When next=clarify, fill clarification_question and clarification_options.
"""



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
    # Set by supervisor_node when routing to clarify; read by clarify_node.
    clarify_question: Optional[str]
    clarify_options: Optional[list]


OUT_OF_SCOPE_MSG = (
    "Desculpe, sou um assistente especializado em serviços financeiros "
    "e não consigo ajudar com esse tema. "
    "Posso responder perguntas sobre contas, cartões, créditos, investimentos "
    "e outros produtos financeiros. Em que posso ajudar?"
)

CRISIS_MSG = (
    "Percebo que estás a passar por um momento muito difícil. "
    "A tua segurança é a prioridade acima de tudo.\n\n"
    "Se precisares de apoio imediato, podes contactar:\n"
    "• **SOS Voz Amiga** — 213 544 545 (24h)\n"
    "• **Linha de Saúde 24** — 808 24 24 24\n"
    "• **Emergência** — 112\n\n"
    "Não estás sozinho/a. Há pessoas prontas para ouvir."
)


def build_graph(researcher, writer, mathy, supervisor, direct, faq, compare, web, checkpointer=None):
    # Agents that always complete their task — supervisor skips LLM for these.
    # "faq" is intentionally NOT in this set: it can return FAQ_MISS, which
    # triggers a re-route to "web" via the supervisor LLM.
    _TERMINAL = {"web", "compare", "direct"}

    def supervisor_node(state: SupervisorState):
        last = state["messages"][-1]
        if getattr(last, "name", None) in _TERMINAL:
            print("[supervisor] -> END  (terminal agent already responded)")
            return {"next": "END"}
        # Pra rotear, o supervisor so precisa da MENSAGEM ATUAL do usuario.
        # Mandar todo o historico aqui = prompt enorme e routing igualmente correto.
        # Encolhe drasticamente o prompt do supervisor em conversas longas.
        messages = _inject_no_think([SystemMessage(content=SUPERVISOR_PROMPT), last])
        decision: RouterDecision = supervisor.invoke(messages)
        print(f"[supervisor] -> {decision.next}  ({decision.reason})")

        if decision.next == "clarify":
            options = [
                {"id": str(i), "label": opt}
                for i, opt in enumerate(decision.clarification_options or [])
            ]
            return {
                "next": "clarify",
                "clarify_question": decision.clarification_question or "Por favor, seja mais específico:",
                "clarify_options": options,
            }

        return {"next": decision.next}

    def clarify_node(state: SupervisorState):
        """Pausa o grafo para pedir ao utilizador que escolha uma opção.

        On first call: lg_interrupt() raises GraphInterrupt → graph checkpoints.
        On resume: lg_interrupt() returns the value from Command(resume=value).
        The node is re-run from the top on resume; state fields are preserved
        from the checkpoint, so clarify_question / clarify_options are still set.
        """
        question = (state.get("clarify_question") or "Por favor, seja mais específico:")
        options = state.get("clarify_options") or []

        selected: str = lg_interrupt({"question": question, "options": options})

        print(f"[clarify] resumed with: {selected!r}")

        # Append clarification to the last user message so knower receives
        # an unambiguous question (knower uses isolate=True → last HumanMessage).
        original = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        if isinstance(original, list):
            original = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in original
            )
        clarified = HumanMessage(
            content=f"{original} (opção selecionada pelo utilizador: {selected})"
        )
        return {
            "messages": [clarified],
            "next": "faq",  # supervisor will re-evaluate; faq is the sane default
            "clarify_question": None,
            "clarify_options": None,
        }

    async def direct_node(state: SupervisorState):
        # Limita historico aos ultimos N mensagens (preserva contexto curto, evita
        # arrastar respostas longas anteriores que inflam o prompt).
        history = state["messages"][-DIRECT_HISTORY_LIMIT:]
        messages = _inject_no_think([SystemMessage(content=DIRECT_PROMPT), *history])
        response = await direct.ainvoke(messages)

        # Flatten Gemini list-content to plain string.
        content = response.content
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )

        # IMPORTANT: preserve the original message ID so LangGraph recognises this
        # as the same message that was already streamed token-by-token during ainvoke.
        # Creating a new AIMessage (new ID) causes LangGraph to emit it a second time
        # → the full text appears duplicated in the client.
        out = response.model_copy(update={"content": content, "name": "direct"})
        return {"messages": [out]}

    def _wrap(sub_agent, name: str, isolate: bool = False):
        """Wrap a sub-agent as a graph node.

        isolate=True: pass ONLY the last HumanMessage to the agent.
        Used for domain agents (faq, compare, web) to prevent conversation
        history from a previous turn about topic A from biasing answers about topic B.

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

    def out_of_scope_node(state: SupervisorState):
        """Resposta fixa para perguntas fora do domínio bancário/financeiro."""
        print("[out_of_scope] pergunta fora do domínio — resposta canned")
        return {
            "messages": [AIMessage(content=OUT_OF_SCOPE_MSG, name="direct")],
        }

    def crisis_node(state: SupervisorState):
        """Resposta de crise — nunca usa LLM, resposta canned com recursos de apoio."""
        print("[crisis] conteúdo de crise detectado — resposta de segurança")
        return {
            "messages": [AIMessage(content=CRISIS_MSG, name="direct")],
        }

    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("direct", direct_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("crisis", crisis_node)
    workflow.add_node("researcher", _wrap(researcher, "researcher"))
    workflow.add_node("writer", _wrap(writer, "writer"))
    workflow.add_node("mathy", _wrap(mathy, "mathy"))
    # Domain-knowledge triad — all isolated (no conversation history bleed)
    workflow.add_node("faq",     _wrap(faq,     "faq",     isolate=True))
    workflow.add_node("compare", _wrap(compare, "compare", isolate=True))
    workflow.add_node("web",     _wrap(web,     "web",     isolate=True))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {
            "researcher":    "researcher",
            "writer":        "writer",
            "mathy":         "mathy",
            "faq":           "faq",
            "compare":       "compare",
            "web":           "web",
            "direct":        "direct",
            "clarify":       "clarify",
            "out_of_scope":  "out_of_scope",
            "crisis":        "crisis",
            "END":           END,
        },
    )
    # All agents loop back to supervisor; supervisor decides END or next agent.
    # out_of_scope uses name="direct" so supervisor short-circuits to END.
    for node in ("direct", "clarify", "out_of_scope", "crisis",
                 "researcher", "writer", "mathy", "faq", "compare", "web"):
        workflow.add_edge(node, "supervisor")

    return workflow.compile(checkpointer=checkpointer)
