#!/usr/bin/env python3
"""Drive the real portal in a real browser and report what happened.

The four cases in section 0 of docs/LIVE_TEST_PLAN.md all have the same false
pass: the page renders perfectly with a dead script. Only a browser that sends
a question and waits for the answer can tell the difference, and only one that
watches the console can say WHY it did not come.

Run it on the box the service runs on:

    pip install playwright && playwright install chromium
    export QB_EMAIL='tester@example.com' QB_PASSWORD='...'
    python deploy/live_smoke.py http://20.63.92.31:8001 Emco_test \
        --ask "total sales by warehouse"

Credentials come from the environment, never the command line, so they do not
land in shell history. Nothing is written to the workspace: it signs in, asks,
reads, and writes its own report to ./live_smoke/.

Output: live_smoke/report.json plus a screenshot per step. Send both.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

OUT = Path("live_smoke")
DEFAULT_QUESTION = "total sales by warehouse"

# The chat page's own ids. #skeletonBubble and #answerProgressElapsed are the
# two the shadowed-translator defect took out, so they are the evidence that
# the fix is live rather than merely merged.
SEL_INPUT = "#input"
SEL_SEND = "#sendBtn"
SEL_THREAD = "#thread"
SEL_SKELETON = "#skeletonBubble"
SEL_ELAPSED = "#answerProgressElapsed"
SEL_BOT_BUBBLE = "#thread .msg-bubble[data-raw]"
# The send button is enabled only by ws.onopen, so "not disabled" IS the proof
# that the websocket connected. A chat page whose socket never opens renders
# perfectly and accepts nothing -- L0-1's false pass exactly.
SEL_SEND_READY = "#sendBtn:not([disabled])"


def shot(page, name: str) -> str:
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="e.g. http://20.63.92.31:8001")
    ap.add_argument("account_id", help="the workspace's account id")
    ap.add_argument("--ask", default=DEFAULT_QUESTION)
    ap.add_argument("--wait", type=int, default=180,
                    help="seconds to wait for the answer (default 180)")
    ap.add_argument("--lang", default="", choices=["", "en", "fr"],
                    help="set the portal language cookie before loading")
    ap.add_argument("--chromium", default=os.getenv("QB_CHROMIUM", ""),
                    help="path to a Chromium/Chrome binary, when `playwright "
                         "install` is not an option (env: QB_CHROMIUM)")
    args = ap.parse_args()

    email = os.getenv("QB_EMAIL", "")
    password = os.getenv("QB_PASSWORD", "")
    if not email or not password:
        print("Set QB_EMAIL and QB_PASSWORD in the environment.", file=sys.stderr)
        return 2

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed:\n"
              "  pip install playwright && playwright install chromium",
              file=sys.stderr)
        return 2

    OUT.mkdir(exist_ok=True)
    base = args.base.rstrip("/")
    report: dict = {"base": base, "account_id": args.account_id,
                    "question": args.ask, "steps": [], "console": [],
                    "page_errors": [], "failed_requests": []}

    def step(name: str, ok: bool, detail: str = "", **extra) -> None:
        report["steps"].append(dict(name=name, ok=bool(ok), detail=detail, **extra))
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    with sync_playwright() as p:
        launch: dict = {}
        if args.chromium:
            launch["executable_path"] = args.chromium
        browser = p.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 1000},
                                      locale="fr-FR" if args.lang == "fr" else "en-GB")
        if args.lang:
            context.add_cookies([{"name": "qb_lang", "value": args.lang,
                                  "url": base}])
        page = context.new_page()

        page.on("console", lambda m: report["console"].append(
            {"type": m.type, "text": m.text[:400]}) if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: report["page_errors"].append(str(e)[:500]))
        page.on("requestfailed", lambda r: report["failed_requests"].append(
            {"url": r.url[:200], "failure": (r.failure or "")[:200]}))

        try:
            # ── sign in ──────────────────────────────────────────────────
            page.goto(f"{base}/portal/login", wait_until="domcontentloaded",
                      timeout=60000)
            step("login page loads", True, page.title())
            report["login_shot"] = shot(page, "01-login")

            page.fill("#login-account", args.account_id)
            page.fill("#login-email", email)
            page.fill("#login-password", password)
            page.click(".login-submit")
            page.wait_for_load_state("domcontentloaded", timeout=60000)

            signed_in = "/portal/login" not in page.url
            step("signed in", signed_in, page.url)
            if not signed_in:
                error = page.locator(".alert, .form-error, .error").first
                step("login rejected", False,
                     error.inner_text()[:200] if error.count() else "no message shown")
                report["login_error_shot"] = shot(page, "02-login-error")
                return finish(report, browser, 1)

            # ── the chat page ────────────────────────────────────────────
            page.goto(f"{base}/portal/chat", wait_until="domcontentloaded",
                      timeout=60000)

            # The admin gate, checked before the composer: without it the page
            # renders a lock screen and every chat case is untestable here.
            body = page.inner_text("body")[:4000]
            if "not enabled" in body.lower() and page.locator(SEL_INPUT).count() == 0:
                step("internal chat is enabled for this workspace", False,
                     "the portal shows the lock screen — turn on Chat UI in "
                     "Client Settings, then re-run")
                report["chat_shot"] = shot(page, "03-chat-disabled")
                return finish(report, browser, 1)

            try:
                page.wait_for_selector(SEL_INPUT, timeout=30000)
            except Exception:
                step("chat page renders a composer", False,
                     "no #input — the page is not the chat surface")
                report["chat_shot"] = shot(page, "03-chat-no-composer")
                return finish(report, browser, 1)
            report["chat_shot"] = shot(page, "03-chat-at-rest")
            step("chat page renders a composer", True)

            # A dead script leaves the shell looking perfect. This is the
            # distinguishing check: a function the page's own script defines.
            wired = page.evaluate(
                "() => typeof window.sendMessage === 'function'"
                " && typeof window.t === 'function'")
            step("the page's script is live", bool(wired),
                 "sendMessage and the translator are both defined" if wired
                 else "the script did not finish — the shell is all you have")

            # ── the websocket ────────────────────────────────────────────
            try:
                page.wait_for_selector(SEL_SEND_READY, timeout=45000)
                step("websocket connected", True, "send button enabled")
            except Exception:
                state = page.locator("#connectionState, .connection-state").first
                step("websocket connected", False,
                     "the send button never enabled — /ws/chat did not open"
                     + (f" ({state.inner_text()[:120]})" if state.count() else ""))
                report["chat_shot"] = shot(page, "03b-no-socket")
                return finish(report, browser, 1)

            # ── ask ──────────────────────────────────────────────────────
            page.fill(SEL_INPUT, args.ask)
            started = time.time()
            page.click(SEL_SEND)

            # The skeleton and the elapsed timer are what the shadowed
            # translator killed. Short timeout: they appear on the first frame.
            try:
                page.wait_for_selector(SEL_SKELETON, timeout=15000)
                step("skeleton bubble appears", True)
            except Exception:
                step("skeleton bubble appears", False,
                     "no #skeletonBubble — the first status frame threw")
            elapsed_live = page.locator(SEL_ELAPSED).count() > 0
            step("elapsed timer started", elapsed_live,
                 "" if elapsed_live else "#answerProgressElapsed missing")
            report["asking_shot"] = shot(page, "04-asking")

            # ── the answer ───────────────────────────────────────────────
            answered = False
            try:
                page.wait_for_selector(SEL_BOT_BUBBLE, timeout=args.wait * 1000)
                answered = True
            except Exception:
                pass
            took = round(time.time() - started, 1)
            report["answer_seconds"] = took
            step("an answer came back", answered,
                 f"{took}s" if answered else f"nothing after {took}s")
            report["answer_shot"] = shot(page, "05-answer")

            if answered:
                bubble = page.locator(SEL_BOT_BUBBLE).last
                text = bubble.inner_text()[:1500]
                report["answer_text"] = text
                step("the answer has content", bool(text.strip()),
                     f"{len(text)} chars")
                # An error bubble IS an answer arriving, which is what L0-1
                # asks. Reporting it as a plain pass would be this script's own
                # false pass, so it gets its own line.
                looks_wrong = text.lstrip().startswith(("⚠", "❌")) or any(
                    phrase in text for phrase in
                    ("Contact your administrator", "cannot generate",
                     "no database", "No database"))
                step("the answer is not an error", not looks_wrong,
                     text.strip().splitlines()[0][:160] if looks_wrong
                     else "reads as a real answer")

            # ── the false pass ───────────────────────────────────────────
            fatal = [e for e in report["page_errors"]
                     if re.search(r"SyntaxError|TypeError|ReferenceError", e)]
            step("no fatal script error", not fatal,
                 "; ".join(fatal[:2]) if fatal else "console clean")

        except Exception as exc:      # noqa: BLE001
            step("run completed", False, f"{type(exc).__name__}: {exc}"[:300])
            try:
                report["crash_shot"] = shot(page, "99-crash")
            except Exception:
                pass
            return finish(report, browser, 1)

        failures = [s for s in report["steps"] if not s["ok"]]
        return finish(report, browser, 1 if failures else 0)


def finish(report: dict, browser, code: int) -> int:
    browser.close()
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport: {OUT / 'report.json'}")
    print(f"screenshots: {OUT}/*.png")
    if report["page_errors"]:
        print(f"\n{len(report['page_errors'])} page error(s) — first:")
        print("  " + report["page_errors"][0][:300])
    return code


if __name__ == "__main__":
    sys.exit(main())
