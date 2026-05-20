# Switchboard

POC para validar três conceitos juntos:
- **MCP server** que expõe tools
- **Sub-agents** especializados, cada um com um subconjunto das tools
- **Router (supervisor)** que decide qual sub-agent invocar

Stack: LangGraph + LangChain + Ollama + MCP.

## Setup

1. Instale o Ollama e baixe o modelo:

   ```
   ollama pull qwen3:14b
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
- `POST /upload` — multipart genérico (compat com `react-chat-ui`); apenas recebe, não indexa
- `POST /documents/upload` — multipart para PDF / DOCX / TXT / MD: extrai texto e **indexa no RAG**
- `GET  /documents` — lista as fontes já indexadas
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

## RAG + Knowledge Graph (Neo4j) + agent knower

O agent **knower** responde perguntas usando:

- **Conteúdo indexado** (chunks de páginas web + uploads do usuário) com **busca vetorial** via embeddings Ollama (`nomic-embed-text`).
- **Knowledge graph** com entidades de banking (Product, Fee, Requirement, Benefit, Channel, Customer) extraídas dos chunks.

**Tudo no Neo4j 5+**: vector index nativo + grafo na mesma instância.

### Setup (uma vez)

1. **Subir o Neo4j** via Docker Compose:

   ```bash
   docker compose up -d
   ```

   Browser de admin: http://localhost:7474 (login `neo4j` / `agentrix-dev-pwd`, ou o que estiver no `.env`).

2. **Baixar o modelo de embeddings**:

   ```bash
   ollama pull nomic-embed-text
   ```

3. **Copiar o `.env`**:

   ```bash
   cp .env.example .env
   ```

4. **Instalar Python deps**:

   ```bash
   pip install -r requirements.txt
   ```

### Fluxo de resposta do knower

1. Se a pergunta é **comparativa** ou pede atributos de um produto específico → tenta `kg_search_entity` + `kg_neighbors` (resposta vem do grafo).
2. Senão (ou se grafo vazio pra esse tópico) → `rag_search` (busca semântica nos chunks).
3. Se vazio → `web_search` → `web_fetch(url)` (baixa, indexa e responde).
4. `kg_extract` (custoso) só roda quando explicitamente útil pra construir grafo sobre o tema.

> O domínio que o knower consulta é configurável via `.env` (`WEB_DOMAIN`, `WEB_DISPLAY_NAME`, `WEB_LANGUAGE`, etc.). Troque essas vars pra apontar o agent pra outro contexto sem mexer em código.

### Tools MCP

| Tool | Para que serve |
| --- | --- |
| `search_notes` / `save_note` | Notas locais (researcher/writer) |
| `calc` | Math (mathy) |
| `web_search(query)` | DDG restrito a `site:<WEB_DOMAIN>` (RAG-first) |
| `web_fetch(url)` | Baixa HTML/PDF do domínio e indexa no Neo4j |
| `rag_search(query, k)` | Busca vetorial (chunks) |
| `rag_sources()` | Lista fontes indexadas |
| `kg_extract(source, limit)` | LLM extrai entidades pros chunks pendentes |
| `kg_search_entity(name)` | Acha entidades por substring |
| `kg_neighbors(name)` | Vizinhos 1-hop de uma entidade no grafo |

### Schema do grafo

```
(:Source {url, title, kind})-[:CONTAINS]->(:Chunk {id, text, embedding, kind, source})
(:Chunk)-[:MENTIONS]->(:Product | :Fee | :Requirement | :Benefit | :Channel | :Customer)

(:Product)-[:HAS_FEE]->(:Fee)
(:Product)-[:REQUIRES]->(:Requirement)
(:Product)-[:OFFERS]->(:Benefit)
(:Product)-[:AVAILABLE_IN]->(:Channel)
(:Product)-[:FOR_SEGMENT]->(:Customer)
```

Vector index: `chunk_embedding_index` em `:Chunk(embedding)`, cosine, 768 dim.

**Limitações conhecidas do scraping**

- Só dá pra acessar páginas **públicas**. Netbanco e qualquer área autenticada estão fora.
- O site pode estar atrás de WAF (Cloudflare/Akamai) e bloquear bots. Se um fetch retornar `403`, é normal — o agent informa e tenta outra URL.
- Páginas heavy-JS (SPAs) só retornam o HTML estático; conteúdo renderizado por JavaScript não aparece.

### Upload de documentos pro RAG

```
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@./meu_extrato.pdf"
```

Retorna `{ id, filename, kind, chunks, size, ... }`. Daí em diante o knower pode responder perguntas sobre esse conteúdo.

Tipos aceitos: `.pdf` (via pypdf), `.docx` (via python-docx), `.txt` / `.md`. Imagens são salvas em `uploads/` mas não indexadas (sem OCR / sem multimodal).

Ver o que está indexado:

```
curl http://localhost:8000/documents
```

## Arquitetura

```
                       User
                        |
                        v
                  +-----------+
                  | Supervisor|  (researcher / writer / mathy / knower / direct / END)
                  +-----------+
                /    |     |     |     \
               v     v     v     v      v
        researcher writer mathy knower  direct
              \    |     |     |      /
               \   |     |     |     /
                v  v     v     v    v
              +--------------------------+
              |        MCP tools         |
              | search_notes / save_note |
              |           calc           |
              |   web_fetch              |
              |        rag_search        |
              |  kg_search_entity        |
              |       kg_neighbors       |
              |        kg_extract        |
              +--------------------------+
                       |
                       v
              +---------------------------+
              |        Neo4j 5+           |
              |   Chunks (vector index)   |
              |   + Knowledge Graph       |
              | Product/Fee/Requirement/  |
              | Benefit/Channel/Customer  |
              +---------------------------+
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
