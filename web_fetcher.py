"""
Fetch e busca restritos a UM dominio configuravel.

O dominio, hosts permitidos, URL base e nome de exibicao vem do config
(WEB_DOMAIN / WEB_BASE_URL / WEB_ALLOWED_HOSTS / WEB_DISPLAY_NAME no .env).

Funcoes:
- search_site(query):  busca no DuckDuckGo (HTML) com `site:<config.web.domain>`
- fetch_url(url):      baixa HTML ou PDF e devolve texto limpo
- get_sitemap_urls():  le `<base>/sitemap.xml` e retorna URLs descobertas
- find_pdfs_on_page(): extrai links PDF de uma pagina
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config import config

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Safari/605.1.15"
)
TIMEOUT = httpx.Timeout(20.0, connect=10.0)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class Page:
    url: str
    title: str
    text: str
    kind: str  # "html" | "pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _allowed_hosts() -> set[str]:
    return {h.lower() for h in config.web.allowed_hosts}


def _is_allowed(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host in _allowed_hosts()


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.6",
            "Accept": (
                "text/html,application/xhtml+xml,application/pdf,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
        },
        timeout=TIMEOUT,
        follow_redirects=True,
    )


# ---------------------------------------------------------------------------
# Search (DuckDuckGo HTML)
# ---------------------------------------------------------------------------
def search_site(query: str, max_results: int = 8) -> list[SearchResult]:
    """Busca no DDG limitada a `site:<config.web.domain>`. Sem chave de API."""
    q = f"site:{config.web.domain} {query}".strip()
    url = "https://html.duckduckgo.com/html/"
    with _client() as c:
        resp = c.post(url, data={"q": q})
        resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    results: list[SearchResult] = []
    for a in soup.select("a.result__a"):
        href = a.get("href") or ""
        m = re.search(r"uddg=([^&]+)", href)
        target = unquote(m.group(1)) if m else href
        if not _is_allowed(target):
            continue
        title = a.get_text(" ", strip=True)
        snippet_tag = (
            a.find_parent("div", class_="result").find("a", class_="result__snippet")
            if a.find_parent("div", class_="result")
            else None
        )
        snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
        results.append(SearchResult(title=title, url=target, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


# ---------------------------------------------------------------------------
# Fetch (HTML ou PDF)
# ---------------------------------------------------------------------------
def fetch_url(url: str) -> Page:
    """Baixa uma URL dentro do dominio permitido e devolve texto limpo."""


    with _client() as c:
        resp = c.get(url)
        resp.raise_for_status()
    content_type = (resp.headers.get("content-type") or "").lower()

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        text = _extract_pdf_text(resp.content)
        title = url.rsplit("/", 1)[-1] or url
        return Page(url=url, title=title, text=_clean(text), kind="pdf")

    return _parse_html(url, resp.text)


def _parse_html(url: str, html: str) -> Page:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav"]):
        tag.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url
    main = soup.find("main") or soup.body or soup
    text = main.get_text("\n", strip=True)
    return Page(url=url, title=title, text=_clean(text), kind="html")


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------
def get_sitemap_urls(limit: int = 500) -> list[str]:
    """Le sitemap.xml (e indices de sitemaps) do dominio configurado."""
    sitemap_url = f"{config.web.base_url.rstrip('/')}/sitemap.xml"
    urls: list[str] = []
    seen: set[str] = set()

    def _consume(xml_text: str) -> None:
        soup = BeautifulSoup(xml_text, "xml")
        for sm in soup.find_all("sitemap"):
            loc = sm.find("loc")
            if loc and loc.text and loc.text not in seen:
                seen.add(loc.text)
                try:
                    with _client() as c:
                        r = c.get(loc.text)
                        r.raise_for_status()
                    _consume(r.text)
                except Exception:
                    continue
        for u in soup.find_all("url"):
            loc = u.find("loc")
            if not loc or not loc.text:
                continue
            if not _is_allowed(loc.text):
                continue
            urls.append(loc.text)
            if len(urls) >= limit:
                return

    try:
        with _client() as c:
            r = c.get(sitemap_url)
            r.raise_for_status()
        _consume(r.text)
    except Exception:
        return []

    seen2: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u in seen2:
            continue
        seen2.add(u)
        out.append(u)
    return out[:limit]


# ---------------------------------------------------------------------------
# PDFs detectados em uma pagina HTML
# ---------------------------------------------------------------------------
def find_pdfs_on_page(url: str) -> list[str]:
    if not _is_allowed(url):
        return []
    try:
        with _client() as c:
            r = c.get(url)
            r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        out: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(url, href)
            if _is_allowed(full) and full.lower().endswith(".pdf"):
                out.append(full)
        return list(dict.fromkeys(out))
    except Exception:
        return []
