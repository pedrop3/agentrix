"""
Supervisor + StateGraph.

O supervisor olha o estado da conversa e decide o proximo passo:
researcher / writer / mathy / END.

Cada sub-agent, depois de rodar, volta pro supervisor — que pode encerrar
ou roteear pra outro agent.
"""
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schema da decisao do supervisor (structured output)
# ---------------------------------------------------------------------------
class RouterDecision(BaseModel):
    """Routing decision."""

    next: Literal["researcher", "writer", "mathy", "END"] = Field(
        description="Which agent should act next, or END if the task is complete."
    )
    reason: str = Field(description="Brief reason for the decision (one sentence).")


SUPERVISOR_PROMPT = """You are a supervisor routing user requests to specialized agents.

Available agents:
- "researcher": searches existing saved notes for information
- "writer": saves new notes to disk
- "mathy": performs math calculations

Decision rules:
- If the user asks to FIND or LOOK UP existing information -> researcher
- If the user asks to SAVE, CREATE, or WRITE a note -> writer
- If the user asks for a CALCULATION -> mathy
- If the previous agent message already completed the user's task -> END

You MUST return valid JSON with fields {"next": "...", "reason": "..."}.
"""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_supervisor(model_name: str = "qwen2.5:7b"):
    """Returns a model that produces a RouterDecision via structured output."""
    model = ChatOllama(model=model_name, temperature=0, num_ctx=8192)
    return model.with_structured_output(RouterDecision)


class SupervisorState(MessagesState):
    """Estado compartilhado: mensagens + proximo destino decidido."""

    next: str


def build_graph(researcher, writer, mathy, supervisor):
    def supervisor_node(state: SupervisorState):
        last = state["messages"][-1]
        if getattr(last, "name", None) in ("researcher", "writer", "mathy"):
            print("[supervisor] -> END  (agent already responded)")
            return {"next": "END"}
        messages = [SystemMessage(content=SUPERVISOR_PROMPT), *state["messages"]]
        decision: RouterDecision = supervisor.invoke(messages)
        print(f"[supervisor] -> {decision.next}  ({decision.reason})")
        return {"next": decision.next}

    def _wrap(sub_agent, name: str):
        async def node(state: SupervisorState):
            result = await sub_agent.ainvoke({"messages": state["messages"]})
            last = result["messages"][-1]
            return {
                "messages": [AIMessage(content=last.content, name=name)],
            }

        return node

    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", _wrap(researcher, "researcher"))
    workflow.add_node("writer", _wrap(writer, "writer"))
    workflow.add_node("mathy", _wrap(mathy, "mathy"))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {
            "researcher": "researcher",
            "writer": "writer",
            "mathy": "mathy",
            "END": END,
        },
    )
    # Cada sub-agent volta pro supervisor decidir se acabou ou nao
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("mathy", "supervisor")

    return workflow.compile()
