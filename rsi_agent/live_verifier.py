"""Read a current HTTP page from an operator-approved origin."""

import hashlib
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


def origin(url):
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None or parsed.fragment):
        raise ValueError("Use an HTTP(S) URL without credentials or fragment")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname.lower(), port


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Page verification does not follow redirects")


def fetch_page(url, allowed_origins, *, timeout, max_bytes):
    if origin(url) not in {origin(item) for item in allowed_origins}:
        raise ValueError("Page origin is outside the operator's permission policy")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    request = Request(url, headers={"Cache-Control": "no-cache", "Accept": "text/html"})
    started = time.monotonic()
    chunks = []
    size = 0
    with opener.open(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError("Page returned an unsuccessful HTTP status")
        while True:
            chunk = response.read(min(65536, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes or time.monotonic() - started > timeout:
                raise ValueError("Page verification exceeded its response budget")
        raw = b"".join(chunks)
        charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset), {"url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                                "bytes": size, "observed_at": time.time(),
                                "scope": "live_http_response_dom"}
