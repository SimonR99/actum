"""Small local web-fetch helper for laptop companion use."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str):
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def fetch_text(url: str, timeout_s: float = 10.0, max_chars: int = 6000) -> dict[str, Any]:
    """Fetch a web page and return readable text.

    This is intentionally modest: it supports direct HTTP(S) fetches for the
    local agent. Search engines, authenticated services, and richer browsing
    should be attached through configured MCP servers.
    """

    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Unsupported URL scheme. Use http:// or https://.")
    if not parsed.netloc:
        raise ValueError("URL must include a host.")

    limit = max(200, min(int(max_chars), 12000))
    request = Request(
        parsed.geturl(),
        headers={
            "User-Agent": "actum-agent/0.1 (+https://github.com/local/actum)",
            "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.5",
        },
    )

    with urlopen(request, timeout=float(timeout_s)) as response:
        raw = response.read(limit * 4)
        content_type = response.headers.get("content-type", "")
        encoding = response.headers.get_content_charset() or "utf-8"
        status = getattr(response, "status", 200)
        final_url = response.geturl()

    decoded = raw.decode(encoding, errors="replace")
    if "html" in content_type.lower():
        extractor = _TextExtractor()
        extractor.feed(decoded)
        text = extractor.text()
    else:
        text = decoded

    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return {
        "url": parsed.geturl(),
        "final_url": final_url,
        "status": status,
        "content_type": content_type,
        "text": text[:limit],
        "truncated": len(text) > limit or len(raw) >= limit * 4,
    }
