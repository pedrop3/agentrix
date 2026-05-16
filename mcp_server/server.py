"""
MCP server que expoe 3 tools: search_notes, save_note, calc.

Roda standalone: python -m mcp_server.server
Ou e iniciado automaticamente pelo MultiServerMCPClient em main.py
(via transport stdio).
"""
import ast
import operator
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("switchboard-tools")

# Pasta onde as notas vivem (criada na primeira execucao)
NOTES_DIR = Path(__file__).parent / "notes"
NOTES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Tool 1: buscar notas
# ---------------------------------------------------------------------------
@mcp.tool()
def search_notes(query: str) -> str:
    """Search saved notes by keyword in title or content.

    Returns matching notes with title and a content snippet.
    Use this to find information the user previously saved.
    """
    q = query.lower().strip()
    results = []
    for note in NOTES_DIR.glob("*.md"):
        content = note.read_text(encoding="utf-8")
        if q in content.lower() or q in note.stem.lower():
            snippet = content[:300] + ("..." if len(content) > 300 else "")
            results.append(f"## {note.stem}\n{snippet}")

    if not results:
        return f"No notes found matching '{query}'."
    return "\n\n---\n\n".join(results)


# ---------------------------------------------------------------------------
# Tool 2: salvar nota
# ---------------------------------------------------------------------------
@mcp.tool()
def save_note(title: str, content: str) -> str:
    """Save a note to disk with the given title and markdown content.

    Use this when the user wants to save, write, or create a new note.
    """
    safe = "".join(c for c in title if c.isalnum() or c in " -_").strip()
    if not safe:
        return "Error: title must contain at least one alphanumeric character."
    note_path = NOTES_DIR / f"{safe}.md"
    note_path.write_text(content, encoding="utf-8")
    return f"Note saved at {note_path.name} ({len(content)} chars)."


# ---------------------------------------------------------------------------
# Tool 3: calculadora segura (sem eval)
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


@mcp.tool()
def calc(expression: str) -> str:
    """Evaluate a math expression safely.

    Supports +, -, *, /, **, %, //, and parentheses.
    Example: '23 * 47 + 100' returns '23 * 47 + 100 = 1181'.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


if __name__ == "__main__":
    # Stdio transport: o cliente (langchain-mcp-adapters) sobe esse processo
    mcp.run(transport="stdio")
