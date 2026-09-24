from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

MAX_REDIRECTS = 5
MAX_BODY_BYTES = 5_000_000

# Clash/Surge "fake-ip" mode resolves every domain into 198.18.0.0/15 and routes it through the
# local proxy. The range is reserved for benchmarking and never hosts a real internal service,
# so allowing it keeps web_fetch usable for those users without opening a path inward.
_FAKE_IP_RANGE = ipaddress.ip_network("198.18.0.0/15")


class NetworkPolicyError(Exception):
    """A URL was refused before any request was sent to it.

    Deliberately not a ValueError: address parsing raises ValueError, and a handler meant for
    "not an IP literal" must never be able to swallow a policy refusal.
    """


async def fetch_url(
    url: str,
    max_length: int = 10_000,
    timeout: float = 15.0,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    # Redirects are followed by hand so every hop is checked. With follow_redirects=True a
    # public URL could answer "302 -> http://127.0.0.1/admin" and httpx would go there.
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=False, transport=transport
    ) as client:
        current = url
        for _hop in range(MAX_REDIRECTS + 1):
            _validate_public_url(current)
            async with client.stream(
                "GET", current, headers={"user-agent": "AgentCLI-Python/0.1.0"}
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise httpx.HTTPStatusError(
                            "redirect without a Location header",
                            request=response.request,
                            response=response,
                        )
                    current = urljoin(str(response.url), location)
                    continue
                response.raise_for_status()
                body = await _read_capped(response)
                content_type = response.headers.get("content-type", "")
                raw = body.decode(response.encoding or "utf-8", errors="replace")
                break
        else:
            raise NetworkPolicyError(f"too many redirects (more than {MAX_REDIRECTS})")

    if "html" not in content_type:
        return _truncate(raw, max_length) or "(empty page)"
    page = extract_page(raw)
    if _needs_javascript(raw, page.text):
        return (
            "[This page builds its content with JavaScript, so a plain fetch did not get "
            "the article text. If browser tools are available, load them in one call: "
            'load_tools(server="chrome-devtools", names=["list_pages", "navigate_page", '
            '"take_snapshot", "click"]), then list_pages for the page id, navigate_page, '
            "take_snapshot. For a login wall or bot check, use the same call with "
            'server="chrome-visible" and ask the user to sign in or complete any CAPTCHA '
            "in that window themselves; never try to solve or evade such checks. If no "
            "browser is available or both fail, use another source.]\n"
            f"Visible text: {page.text[:300] or '(none)'}"
        )
    text = f"{page.title}\n\n{page.text}" if page.title else page.text
    return _truncate(text, max_length) or "(empty page)"


async def _read_capped(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _truncate(text: str, max_length: int) -> str:
    if len(text) > max_length:
        return text[:max_length] + "\n... [truncated]"
    return text


# ---------------------------------------------------------------------------
# Main-content extraction
# ---------------------------------------------------------------------------

# Never content: code, styling, and page chrome repeated on every page of a site.
_SKIP_TAGS = {
    "script", "style", "noscript", "template", "svg", "canvas", "iframe",
    "nav", "header", "footer", "aside", "form", "button", "select",
}  # fmt: skip
_MAIN_TAGS = {"article", "main"}
_BLOCK_TAGS = {
    "p", "div", "section", "br", "li", "ul", "ol", "tr", "table", "pre", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6", "article", "main", "dd", "dt",
}  # fmt: skip
_VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "source", "wbr"}


class _Page:
    def __init__(self, title: str, text: str):
        self.title = title
        self.text = text


class _ContentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.main_depth = 0
        self.in_title = False
        self.title: list[str] = []
        self.body: list[str] = []
        self.main: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _VOID_TAGS:
            self._break(tag)
            return
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
        elif tag == "title":
            self.in_title = True
        elif tag in _MAIN_TAGS or dict(attrs).get("role") == "main":
            self.main_depth += 1
        self._break(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag == "title":
            self.in_title = False
        elif tag in _MAIN_TAGS:
            self.main_depth = max(0, self.main_depth - 1)
        self._break(tag)

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title.append(data)
            return
        if self.skip_depth:
            return
        self.body.append(data)
        if self.main_depth:
            self.main.append(data)

    def _break(self, tag: str) -> None:
        if tag in _BLOCK_TAGS and not self.skip_depth:
            self.body.append("\n")
            if self.main_depth:
                self.main.append("\n")


def extract_page(raw_html: str) -> _Page:
    """Title plus the readable text of a page, preferring its <article>/<main> region."""

    parser = _ContentParser()
    try:
        parser.feed(raw_html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML still yields whatever was parsed
        pass
    main = _clean(parser.main)
    body = _clean(parser.body)
    # A tiny <main> is usually a wrapper around a widget, not the article.
    text = main if len(main) >= 200 else body
    return _Page(title=_clean(parser.title).replace("\n", " "), text=text)


def extract_text_from_html(raw_html: str) -> str:
    return extract_page(raw_html).text


def _clean(parts: list[str]) -> str:
    text = "".join(parts)
    lines = (re.sub(r"[ \t\r\f\v ]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


_JS_REQUIRED_MARKERS = ("enable javascript", "需要允许该网站执行 javascript", "请开启javascript")


def _needs_javascript(raw_html: str, text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _JS_REQUIRED_MARKERS):
        return True
    # A big page that yields next to no text is a client-rendered shell or a bot check: the
    # bytes are script, not content.
    if len(text) >= 200:
        return False
    return len(raw_html) > 5_000 or raw_html.lower().count("<script") >= 3


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


def _validate_public_url(url: str) -> None:
    """Refuse URLs that point at this machine or a private network (SSRF).

    Every address the host resolves to must be public: an attacker-controlled domain can
    return several records, and the client may connect to any of them.
    """

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise NetworkPolicyError("only http/https URLs are allowed")
    host = parsed.hostname
    if not host:
        raise NetworkPolicyError("URL must include a hostname")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        _reject_non_public(literal, host)
        return

    try:
        infos = socket.getaddrinfo(host, parsed.port or None)
    except socket.gaierror as exc:
        raise NetworkPolicyError(f"cannot resolve host: {host}") from exc
    for info in infos:
        _reject_non_public(ipaddress.ip_address(info[4][0].split("%")[0]), host)


def _reject_non_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, host: str) -> None:
    # ::ffff:127.0.0.1 is 127.0.0.1 in disguise.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if isinstance(ip, ipaddress.IPv4Address) and ip in _FAKE_IP_RANGE:
        return
    # An allowlist: anything that is not globally routable is refused, which covers loopback,
    # private ranges, link-local (cloud metadata at 169.254.169.254), 0.0.0.0, carrier NAT,
    # and reserved space, without having to enumerate them.
    if not ip.is_global or ip.is_multicast:
        raise NetworkPolicyError(f"{host} resolves to {ip}, which is not a public internet address")
