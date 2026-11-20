"""Observe a JavaScript-rendered DOM in a fresh Chromium context."""

import hashlib
import json
import sys
import time

from .live_verifier import origin


def observe(rule, allowed_origins, timeout_seconds):
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as BrowserTimeout
    except ImportError as error:
        raise RuntimeError("Install the browser extra and playwright Chromium") from error
    allowed = {origin(value) for value in allowed_origins}
    if origin(rule["url"]) not in allowed:
        raise ValueError("Browser page origin is not permitted")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, chromium_sandbox=True)
        try:
            context = browser.new_context(service_workers="block", accept_downloads=False)
            context.set_default_timeout(timeout_seconds * 1000)

            def route(request):
                try:
                    permitted = origin(request.request.url) in allowed
                except ValueError:
                    permitted = False
                if permitted and request.request.method in {"GET", "HEAD"}:
                    request.continue_()
                else:
                    request.abort()

            context.route("**/*", route)
            context.route_web_socket("**/*", lambda socket: socket.close())
            page = context.new_page()
            response = page.goto(rule["url"], wait_until="domcontentloaded")
            if response is None or response.status != 200 or origin(page.url) not in allowed:
                raise ValueError("Browser navigation did not reach an approved successful page")
            element = page.locator(rule["selector"])
            passed = False
            try:
                from playwright.sync_api import expect
                expect(element).to_have_count(1, timeout=timeout_seconds * 1000)
                for name, expected in rule.get("attributes", {}).items():
                    expect(element).to_have_attribute(name, expected, timeout=timeout_seconds * 1000)
                if "text" in rule:
                    expect(element).to_have_text(rule["text"], timeout=timeout_seconds * 1000)
                passed = True
            except (AssertionError, BrowserTimeout):
                pass
            dom = page.content().encode()
            return {"passed": passed, "evidence": {
                "kind": "browser", "scope": "rendered_chromium_dom", "url": page.url,
                "selector": rule["selector"], "observed_at": time.time(),
                "dom_sha256": hashlib.sha256(dom).hexdigest(),
                "text": element.inner_text()[:2000] if element.count() == 1 else None}}
        finally:
            browser.close()


def main():
    request = json.load(sys.stdin)
    print(json.dumps(observe(request["rule"], request["origins"], request["timeout"])))


if __name__ == "__main__":
    main()
