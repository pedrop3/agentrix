# Switchboard

POC para validar três conceitos juntos:
- **MCP server** que expõe tools
- **Sub-agents** especializados, cada um com um subconjunto das tools
- **Router (supervisor)** que decide qual sub-agent invocar

Stack: LangGraph + LangChain + Ollama + MCP.

## Setup

1. Instale o Ollama e baixe o modelo:

   ```
   ollama pull qwen2.5:7b
   ```

2. Crie um virtualenv e instale as dependências:

   ```
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copie `.env.example` para `.env` e ajuste se quiser.

4. Rode o smoke test pra garantir que tool calling no Ollama tá funcionando:

   ```
   python test_ollama.py
   ```

   Se ver "Tool calling works.", pode prosseguir. Se ver "Model did NOT use the tool", troque o modelo (tente `llama3.1:8b`).

5. Rode o app:

   ```
   python main.py
   ```

## Servidor HTTP (FastAPI + SSE)

Pra plugar uma UI (ex.: `react-chat-ui`) no switchboard, suba o servidor:

```
python server.py
# ou:
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

- `POST /chat` — resposta JSON única `{ "reply": "..." }`
- `POST /chat/stream` — Server-Sent Events com `data: {"delta": "..."}` por token, terminando em `data: [DONE]`
- `POST /upload` — multipart (placeholder; qwen2.5 não é multimodal)
- `GET  /health` — status do servidor

Payload esperado em `/chat` e `/chat/stream`:

```json
{
  "conversationId": "uuid",
  "messages": [{ "role": "user", "content": "..." }],
  "attachments": []
}
```

O `conversationId` é mapeado pro `thread_id` do `AsyncSqliteSaver`, então cada conversa do chat UI vira uma thread persistida no `memory.db`.

Teste rápido com `curl`:

```
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"conversationId":"demo","messages":[{"role":"user","content":"quanto e 2+2?"}]}'
```

## Arquitetura

```
        User
         |
         v
   +-----------+
   | Supervisor|  (LLM decide: researcher / writer / mathy / END)
   +-----------+
     |    |    |
     v    v    v
researcher writer mathy
     \    |    /
      \   |   /
       \  |  /
        v v v
     +---------+
     |   MCP   |  (search_notes, save_note, calc)
     +---------+
```

## O desafio em 3 níveis

**Nível 1 — MCP server (1h)**
Roda `python -m mcp_server.server` e valida que o `langchain-mcp-adapters` consegue listar as 3 tools (`search_notes`, `save_note`, `calc`).

**Nível 2 — Sub-agents (1-2h)**
Testa cada agent isolado. Comece pelo `mathy` (mais simples: "quanto é 23 * 47?").

**Nível 3 — Router/Supervisor (1h)**
Roda o fluxo completo via `main.py`. Abra o LangSmith pra ver os traces.

## Exemplos pra testar

- `"salve uma nota sobre minha reuniao de amanha: discutir orcamento Q3, contratar designer, revisar roadmap"`
- `"o que eu salvei sobre reunioes?"`
- `"calcule 23 * 47 + 100"`

## Checkpoint de aprendizado

Quando terminar, você deveria conseguir responder:

1. Por que MCP é melhor do que passar funções Python direto pro agent?
2. Qual a diferença prática entre uma tool e um sub-agent?
3. Quando vale a pena um router determinístico (código) vs. um router via LLM?
