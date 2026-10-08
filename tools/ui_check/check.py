"""The UI check: the product in a real browser, at desktop and phone width, on invented data.

    python tools/ui_check/check.py [--out DIR] [--port 8790] [--server-python PATH] [--chromium PATH]

It starts tools/ui_check/serve.py (a fresh copy of the product on invented data), signs in to the
reader portal and the admin console in headless Chromium, asks the chat a set of questions that
between them draw every kind of answer, and opens the other pages. Each page and each answer is
captured at 1440 x 900 and at 390 x 844 into DIR (default ui-check/), with report.json beside them.

It fails (exit 1) on what a person would see as broken:
  - a console error or an uncaught exception on any page;
  - a page that scrolls sideways (the document wider than the window);
  - tiles that overlap (dashboard tiles, KPI tiles, anything marked data-check-tile);
  - a button, field or select smaller than 44px on a phone;
  - a question that gets no answer.

It needs Playwright (pip install playwright) for the browser; the server runs under
--server-python, which needs the product's own requirements. The screenshots are the reference a
UI change is reviewed against: run it before and after, and look at both.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

QUESTIONS = [
    "What was net sales in April 2026?",
    "Net sales by store in the first half of 2026",
    "Monthly net sales since January 2025",
    "Net sales by region by month in 2026",
    "Compare net sales by store in April against March",
    "Net sales, cost and gross amount in 2025",
    "Top 5 products by net sales in 2026",
    "Why did net sales drop in April?",
    "List the stores",
    "What data do you have?",
]
VIEWPORTS = {"desktop": (1440, 900), "phone": (390, 844)}
READER = ("acct-retail", "reader@example.com", "reader-pass-123")
ADMIN_PASSWORD = "ui-check-admin-pass-1"
TILES = ".grid-stack-item, .kpi-tile, [data-check-tile]"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def _wait_up(url: str, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:  # noqa: S310 - a local address the check started
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(1)
    raise SystemExit(f"The local server did not come up at {url} within {seconds:.0f}s.")


class Page:
    """One browser page with what it logs collected, and the checks run on what it shows."""

    def __init__(self, browser, name: str, size: tuple[int, int], out: Path, problems: list[dict]):
        self.name, self.out, self.problems = name, out, problems
        mobile = size[0] < 600
        self.page = browser.new_context(viewport={"width": size[0], "height": size[1]}, is_mobile=mobile,
                                        has_touch=mobile).new_page()
        self.errors: list[str] = []
        # A failed request is logged with its address; the console's own line for it says only "404".
        self.page.on("console", lambda m: self.errors.append(m.text[:300])
                     if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
        self.page.on("pageerror", lambda e: self.errors.append(f"uncaught: {e}"[:300]))
        self.page.on("response", lambda r: self.errors.append(f"HTTP {r.status} {r.request.method} {r.url}"[:300])
                     if r.status >= 400 else None)
        # A Playwright screenshot drops Chromium's touch emulation, so a phone page
        # measured after one reads as a mouse page ((pointer: coarse) false). It is
        # put back after every shot.
        self._cdp = self.page.context.new_cdp_session(self.page) if mobile else None

    def _touch_again(self) -> None:
        if self._cdp is not None:
            self._cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 5})

    def shot(self, label: str, *, element=None, full: bool = False) -> dict:
        path = self.out / f"{self.name}_{label}.png"
        try:
            if element is not None:
                element.scroll_into_view_if_needed(timeout=5000)
                element.screenshot(path=str(path), timeout=15000)
            else:
                self.page.screenshot(path=str(path), full_page=full)
        except Exception:  # noqa: BLE001 - an element that will not shoot is still checked below
            self.page.screenshot(path=str(path))
        self._touch_again()
        found = self.check(label)
        return {"shot": path.name, **found}

    def check(self, label: str) -> dict:
        where = f"{self.name} {label}"
        overflow = self.page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
        overlaps = self.page.evaluate(
            """sel => {
                 const boxes = [...document.querySelectorAll(sel)]
                   .filter(e => e.offsetParent !== null)
                   .map(e => [e, e.getBoundingClientRect()]);
                 const hits = [];
                 for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
                   const [, a] = boxes[i], [, b] = boxes[j];
                   const w = Math.min(a.right, b.right) - Math.max(a.left, b.left);
                   const h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
                   if (w > 2 && h > 2 && !boxes[i][0].contains(boxes[j][0]) && !boxes[j][0].contains(boxes[i][0]))
                     hits.push([boxes[i][0].className, boxes[j][0].className]);
                 }
                 return hits.slice(0, 5);
               }""", TILES)
        small = self.page.evaluate(
            """() => window.innerWidth >= 600 ? [] : [...document.querySelectorAll(
                   'a[href], button, input:not([type=hidden]):not([type=checkbox]):not([type=radio]), select, textarea, '
                   + '[role=button], [role=tab]')]
                 .filter(e => e.offsetParent !== null)
                 .map(e => [e, e.getBoundingClientRect()])
                 .filter(([, r]) => r.width > 0 && r.height > 0 && r.right > 0 && r.left < window.innerWidth)
                 .filter(([, r]) => r.height < 44 || r.width < 44)
                 .filter(([e]) => !e.closest('.chart-canvas, [data-chart], table'))
                 .map(([e, r]) => `${e.tagName.toLowerCase()}.${(e.className || '').toString().split(' ')[0]}`
                                  + ` ${Math.round(r.width)}x${Math.round(r.height)}`)
                 .slice(0, 40)""")
        errors, self.errors = self.errors, []
        for e in errors:
            self.problems.append({"where": where, "problem": "console error", "detail": e})
        if overflow > 1:
            self.problems.append({"where": where, "problem": "scrolls sideways", "detail": f"{overflow}px wider"})
        for a, b in overlaps:
            self.problems.append({"where": where, "problem": "tiles overlap", "detail": f"{a} / {b}"})
        # A control a fingertip cannot hit: a button, a field or a select under 44px on a phone.
        # Links inside a sentence are excepted, as WCAG does; checkboxes are measured by their label.
        for target in small:
            kind = target.split(".", 1)[0]
            if kind in ("button", "select", "input", "textarea") or (kind == "a" and "btn" in target.split(" ")[0]):
                self.problems.append({"where": where, "problem": "small tap target", "detail": target})
        # Tap targets under 44px on a phone, reported (not failed) so a pass can follow them.
        return {"errors": errors, "overflow_px": overflow, "overlaps": overlaps, "small_targets": small}


def _pin(page, card, problems: list[dict], view: str) -> None:
    """Add the answer to a new dashboard, as a reader would: the dashboard page then shows its tile."""
    button = card.locator('[data-pin="chart"]')
    if not button.count():
        problems.append({"where": f"{view} chat", "problem": "no Add to dashboard", "detail": "comparison answer"})
        return
    button.first.click()
    page.click("#dashboardNewMode")
    page.fill("#dashboardNewName", "Store comparison")
    page.click("#dashboardPickerSubmit")
    try:
        page.wait_for_function("() => !document.getElementById('dashboardPickerBackdrop').classList.contains('open')",
                               timeout=15000)
    except Exception:  # noqa: BLE001
        problems.append({"where": f"{view} chat", "problem": "pin did not finish",
                         "detail": page.inner_text("#dashboardPickerError")[:200]})


def run(base: str, out: Path, chromium: str | None) -> tuple[list[dict], list[dict]]:
    from playwright.sync_api import sync_playwright

    problems: list[dict] = []
    report: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(**({"executable_path": chromium} if chromium else {}))
        for view, size in VIEWPORTS.items():
            p = Page(browser, view, size, out, problems)
            page = p.page
            page.goto(base + "/portal/login", wait_until="networkidle")
            report.append({"page": "portal sign-in", "view": view, **p.shot("00_portal_login")})
            page.fill("#login-account", READER[0])
            page.fill("#login-email", READER[1])
            page.fill("#login-password", READER[2])
            page.click("button.login-submit")
            page.wait_for_load_state("networkidle")
            page.goto(base + "/portal/chat", wait_until="networkidle")
            page.wait_for_timeout(1200)
            report.append({"page": "chat", "view": view, **p.shot("01_chat_start")})
            for i, question in enumerate(QUESTIONS, start=2):
                before = page.locator(".msg.msg-bot").count()
                page.fill("#input", question)
                page.keyboard.press("Enter")
                if i == 2:
                    # The pending answer, with the working wave beside its line.
                    try:
                        page.wait_for_selector("#answerProgressBrand", timeout=5000)
                        report.append({"page": "pending answer", "view": view,
                                       **p.shot("02a_pending", element=page.locator("#skeletonBubble"))})
                    except Exception:  # noqa: BLE001 - an answer quicker than the capture
                        pass
                try:
                    page.wait_for_function("n => document.querySelectorAll('.msg.msg-bot').length > n",
                                           arg=before, timeout=90000)
                except Exception:  # noqa: BLE001
                    problems.append({"where": f"{view} chat", "problem": "no answer", "detail": question})
                    continue
                page.wait_for_timeout(2500)
                card = page.locator(".msg.msg-bot").last
                report.append({"page": "answer", "view": view, "question": question,
                               **p.shot(f"{i:02d}_{_slug(question)}", element=card)})
                if question.startswith("Compare") and view == "desktop":
                    _pin(page, card, problems, view)
                if question.startswith("Monthly") and card.locator("[data-open-artifact]").count():
                    # The larger view: a side panel on a desktop, a sheet from the bottom on a phone.
                    card.locator("[data-open-artifact]").first.click()
                    page.wait_for_timeout(900)
                    report.append({"page": "the larger view", "view": view, **p.shot(f"{i:02d}b_expanded")})
                    page.keyboard.press("Escape")
                    page.evaluate("() => window.closeArtifactPane && window.closeArtifactPane()")
                    page.wait_for_timeout(400)
            report.append({"page": "chat, whole conversation", "view": view, **p.shot("90_chat_full", full=True)})
            for path in ("/portal/dashboard", "/portal/kb"):
                page.goto(base + path, wait_until="networkidle")
                page.wait_for_timeout(1500)
                report.append({"page": path, "view": view, **p.shot("95" + path.replace("/", "_"), full=True)})
                pinned = page.locator("a", has_text="Store comparison")
                if path == "/portal/dashboard" and pinned.count():
                    pinned.first.click()
                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(2500)
                    report.append({"page": "a pinned dashboard", "view": view, **p.shot("96_dashboard_open", full=True)})

            a = Page(browser, view, size, out, problems)
            a.page.goto(base + "/admin/login", wait_until="networkidle")
            report.append({"page": "admin sign-in", "view": view, **a.shot("50_admin_login")})
            a.page.fill("input[name=password]", ADMIN_PASSWORD)
            a.page.click("button[type=submit]")
            a.page.wait_for_load_state("networkidle")
            workspace = f"/admin/clients/{READER[0]}"
            for path in ("/admin", workspace, workspace + "/setup", workspace + "/graph", workspace + "/settings",
                         workspace + "/compliance", workspace + "/diagnostics", "/admin/system"):
                a.page.goto(base + path, wait_until="networkidle")
                a.page.wait_for_timeout(800)
                report.append({"page": path, "view": view, **a.shot("51" + path.replace("/", "_"), full=True)})
        browser.close()
    return report, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="ui-check")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--server-python", default=sys.executable)
    ap.add_argument("--chromium", default=None, help="a Chromium binary, when Playwright's own is not installed")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    log = (out / "server.log").open("w")
    server = subprocess.Popen([args.server_python, str(HERE / "serve.py"), str(out / "run"), str(args.port)],
                              cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{args.port}"
    try:
        _wait_up(base + "/portal/login", 240)
        report, problems = run(base, out, args.chromium)
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
    (out / "report.json").write_text(json.dumps({"problems": problems, "pages": report}, indent=1))
    print(f"{len(report)} captures in {out}")
    for p in problems:
        print(f"  {p['where']}: {p['problem']} - {p['detail']}")
    print("UI check: " + ("passed" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
