"""
tests/portal_render.py

Render any portal template the way a browser is served it.

Not a test module. Two things make a bare ``jinja2.Environment`` the wrong tool
for these pages now:

  * portal_base.html calls ``t()`` and ``i18n_catalogue``, which arrive from
    portal.routes._language_context -- a Jinja2Templates *context processor*.
    A hand-built environment has no processors, so the shell raises rather than
    renders.
  * The catalogue is injected into the page as JSON, so every English string is
    present in the source whatever the page displays. ``visible()`` drops the
    <script> blocks so an absence assertion still means something.

The context comes from the PRODUCTION processor, so a fixture here cannot drift
from what is actually served.
"""

from __future__ import annotations

import html
import re


class _URL:
    def __init__(self, path):
        self.path = path


class Req:
    """The slice of Request the portal templates and _language_context read."""

    query_params: dict = {}

    def __init__(self, path="/portal/dashboard", lang=None):
        self.url = _URL(path)
        self.cookies = {"qb_lang": lang} if lang else {}
        self.headers = {}


def language_context(request=None, lang=None) -> dict:
    import portal.routes as pr
    return pr._language_context(request or Req(lang=lang))


def unescaped(markup: str) -> str:
    """The markup with HTML entities resolved.

    Jinja autoescapes, so a French string carrying an apostrophe reaches the
    page as "n&#39;a pas" -- correct output that no assertion written in French
    will match. Resolving entities lets a test say what the reader sees.
    """
    return html.unescape(markup)


def visible(markup: str) -> str:
    """What the reader actually sees: no <script> blocks, no entities.

    Dropping the scripts is what makes an absence assertion mean anything --
    the catalogue is injected into the page as JSON, so every English string is
    present in the source whatever the page displays.
    """
    return unescaped(re.sub(r"<script\b.*?</script>", " ", markup, flags=re.S | re.I))


def render(template_name: str, *, lang=None, path="/portal/dashboard", **context) -> str:
    """Render `template_name` through the portal's own Jinja environment.

    Undefined is swapped for ChainableUndefined so a test supplying a partial
    context still exercises the shell, which is what these renders are for.
    """
    from jinja2 import ChainableUndefined

    import portal.routes as pr

    request = context.pop("request", None) or Req(path, lang)
    env = pr.templates.env
    previous, env.undefined = env.undefined, ChainableUndefined
    try:
        return env.get_template(template_name).render(
            request=request, **context, **language_context(request),
        )
    finally:
        env.undefined = previous
