from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from agentcli.config import load_config
from agentcli.tools.base import ToolContext
from agentcli.tools.builtins import _web_fetch
from agentcli.web.fetch import (
    NetworkPolicyError,
    _needs_javascript,
    _validate_public_url,
    extract_page,
    fetch_url,
)


def _fake_dns(monkeypatch, table: dict[str, str]):
    def getaddrinfo(host, port, *args, **kwargs):
        if host not in table:
            raise socket.gaierror(f"unknown host {host}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://192.168.1.1/admin",
        "http://10.0.0.5/",
        "http://169.254.169.254/latest/meta-data/",  # cloud instance metadata
        "http://0.0.0.0/",
        "http://100.64.0.1/",  # carrier-grade NAT
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
        "file:///etc/passwd",
    ],
)
def test_private_and_local_targets_are_refused(url):
    with pytest.raises(NetworkPolicyError):
        _validate_public_url(url)


def test_hostnames_are_checked_by_what_they_resolve_to(monkeypatch):
    _fake_dns(
        monkeypatch,
        {
            "intranet.example": "10.1.2.3",
            "news.example": "93.184.216.34",
            "clash.example": "198.18.0.7",
        },
    )

    with pytest.raises(NetworkPolicyError, match="10.1.2.3"):
        _validate_public_url("http://intranet.example/")
    _validate_public_url("https://news.example/article")
    # Clash/Surge fake-ip answers stay usable.
    _validate_public_url("https://clash.example/")


def test_redirect_into_private_network_is_refused(monkeypatch):
    _fake_dns(monkeypatch, {"evil.example": "93.184.216.34"})
    visited: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        visited.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1:8080/admin"})

    with pytest.raises(NetworkPolicyError):
        asyncio.run(fetch_url("http://evil.example/", transport=httpx.MockTransport(handler)))
    # The private target was never requested.
    assert visited == ["http://evil.example/"]


def test_safe_redirects_are_followed(monkeypatch):
    _fake_dns(monkeypatch, {"a.example": "93.184.216.34", "b.example": "93.184.216.35"})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example":
            return httpx.Response(301, headers={"location": "https://b.example/final"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><title>Final</title><body><p>" + "arrived " * 40 + "</p></body></html>",
        )

    text = asyncio.run(fetch_url("http://a.example/", transport=httpx.MockTransport(handler)))

    assert text.startswith("Final")
    assert "arrived" in text


def test_redirect_loops_stop(monkeypatch):
    _fake_dns(monkeypatch, {"loop.example": "93.184.216.34"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/again"})

    with pytest.raises(NetworkPolicyError, match="too many redirects"):
        asyncio.run(fetch_url("http://loop.example/", transport=httpx.MockTransport(handler)))


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


def test_article_text_is_kept_and_page_chrome_dropped():
    html = (
        """
    <html><head><title>DeepSeek V4 评测</title><style>.x{}</style></head><body>
      <header>网站 Logo 登录 注册</header>
      <nav>首页 产品 解决方案 定价 文档</nav>
      <article>
        <h1>DeepSeek V4 评测</h1>
        <p>第一段："""
        + "正文内容。" * 30
        + """</p>
        <p>第二段：结论 &amp; 建议。</p>
      </article>
      <aside>热门文章 推荐阅读</aside>
      <footer>版权所有 备案号</footer>
      <script>track()</script>
    </body></html>
    """
    )

    page = extract_page(html)

    assert page.title == "DeepSeek V4 评测"
    assert "第一段" in page.text and "结论 & 建议" in page.text
    for chrome in ("登录", "解决方案", "推荐阅读", "备案号", "track()"):
        assert chrome not in page.text
    # Paragraphs stay on separate lines instead of one run-on string.
    assert "\n第二段" in page.text


def test_pages_without_article_fall_back_to_body_minus_chrome():
    html = "<body><nav>菜单</nav><div><p>" + "正文" * 150 + "</p></div><footer>页脚</footer></body>"

    text = extract_page(html).text

    assert text.startswith("正文") and "菜单" not in text and "页脚" not in text


# ---------------------------------------------------------------------------
# Error reporting and JS-only pages
# ---------------------------------------------------------------------------


def test_fetch_error_always_names_the_failure(tmp_path, monkeypatch):
    async def time_out(*_args, **_kwargs):
        raise httpx.ReadTimeout("")  # stringifies to "", as real timeouts often do

    monkeypatch.setattr("agentcli.tools.builtins.fetch_url", time_out)
    context = ToolContext(cwd=str(tmp_path), config=load_config(project_root=tmp_path))

    result = asyncio.run(_web_fetch({"url": "https://example.com"}, context))

    assert result.is_error
    assert result.content == "Fetch error: ReadTimeout"


def test_client_rendered_pages_are_flagged():
    shell = "<html><script>a</script><script>b</script><script>c</script><div id=app></div>"
    assert _needs_javascript(shell, "")
    assert _needs_javascript("<html></html>", "今日头条 您需要允许该网站执行 JavaScript")
    # Bot-check page: empty body, one huge script.
    assert _needs_javascript("<html><body></body><script>" + "x" * 70_000 + "</script>", "")
    assert not _needs_javascript("<html><body>short</body></html>", "short")

    article = "<html><script>x</script><p>" + "正文内容。" * 100 + "</p></html>"
    assert not _needs_javascript(article, "正文内容。" * 100)
