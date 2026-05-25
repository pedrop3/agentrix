from config import config
from mcp_server.server import web_search,web_fetch
from web_fetcher import search_site
from rag import get_rag


# def test_web_search_real():
#
#
#     assert config.web.domain, "https://www.santander.pt"
#
#     query = "quais os critérios para abrir uma conta teens ?"
#
#     result = web_fetch(query)
#
#     # 1. deve retornar algo
#     assert result is not None
#     assert len(result) > 0
#
#     # 2. deve conter referência ao domínio
#     assert config.web.domain in result or "http" in result
#
#     print(result)
#
#     # 3. opcional: verificar se indexou no RAG
#     rag = get_rag()
#     stats = rag.stats()
#
#     assert stats is not None


def test_web_fetcher_real():


    assert config.web.domain, "https://www.google.pt"


    query = "quais os critérios para abrir uma conta ?"

    result = search_site(query)

    # 1. deve retornar algo
    assert result is not None


    # 2. deve conter referência ao domínio


    print(result)

    # 3. opcional: verificar se indexou no RAG
    rag = get_rag()
    stats = rag.stats()

    assert stats is not None