#!/usr/bin/env python3
"""Is this workspace actually in a state where the live test plan means anything?

Run this on the server, before working through docs/LIVE_TEST_PLAN.md:

    python -m deploy.preflight_live                 # list workspaces
    python -m deploy.preflight_live <account_id>    # report on one

It reads and prints; it changes nothing. The point is to separate "this case
failed" from "this case could never have passed here", which is the difference
between a finding and a wasted afternoon. Half the plan's cases need something
the workspace may not have -- a second subject area, a value index, more than
one warehouse connection -- and a tester who does not know that records a
failure against working code.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, WARN, GAP = "ok  ", "warn", "GAP "


def line(mark: str, label: str, detail: str = "") -> None:
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))


def workspaces() -> None:
    import store

    rows = store.list_clients()
    if not rows:
        print("No workspaces registered.")
        return
    print(f"{len(rows)} workspace(s):\n")
    for row in rows:
        print(f"  {row['account_id']}   {row.get('state','?'):<16} "
              f"{row.get('client_name') or ''}")
    print("\nRun again with an account id for the full report.")


def report(account_id: str) -> int:
    import store

    client = store.get_client(account_id)
    if not client:
        print(f"No workspace with account id {account_id!r}.")
        return 2

    print(f"\n{client.get('client_name') or account_id}  ({account_id})")
    print("=" * 66)

    gaps = 0

    # ── 1 · the basics every case needs ──────────────────────────────────
    print("\nFoundation")
    state = str(client.get("state") or "")
    line(OK if state == "READY" else GAP, f"state = {state or 'unset'}",
         "" if state == "READY" else "discovery/KB build has not finished")
    gaps += state != "READY"

    cstate = store.get_client_state(account_id) or {}
    schema_dir = str(cstate.get("schema_dir") or "")
    has_schema = bool(schema_dir and (Path(schema_dir) / "_schema.json").is_file())
    line(OK if has_schema else GAP, "discovered schema",
         schema_dir or "no schema_dir on the client state")
    gaps += not has_schema

    # ── 2 · vocabulary ───────────────────────────────────────────────────
    print("\nVocabulary")
    from core.vocab_packs import (_client_pack_ids, _detected_pack_ids,
                                  list_available_packs, vocab_for_account)

    chosen = _client_pack_ids(account_id)
    detected = _detected_pack_ids(account_id) if not chosen else []
    kinds = {m["pack_id"]: m.get("pack_kind", "erp") for m in list_available_packs()}
    active = chosen or detected
    erp = [p for p in active if kinds.get(p) != "industry"]
    industry = [p for p in active if kinds.get(p) == "industry"]

    line(OK if erp else WARN, "ERP pack",
         ", ".join(erp) + ("  (auto-detected)" if detected else "") if erp
         else "none — the builtin vocabulary only")
    line(OK if industry else WARN, "industry pack",
         ", ".join(industry) if industry
         else "none — sections 11.4's cases (L11-19..L11-23) need one selected")

    vocab = vocab_for_account(account_id)
    line(OK, "merged vocabulary",
         f"{len(vocab.abbreviations)} abbreviations, "
         f"{len(vocab.direct_aliases)} aliased columns, "
         f"{len(vocab.column_dict)} dictionary codes")

    terms = store.list_table_descriptions(account_id) or {}
    columns_with_terms = sum(len(e.get("column_synonym_map") or {}) for e in terms.values())
    line(OK if columns_with_terms else WARN, "admin column terms",
         f"{columns_with_terms} column(s) across {len(terms)} table(s)"
         if columns_with_terms else "none yet — L11-14/L11-15 need at least one")

    # ── 3 · the graph ────────────────────────────────────────────────────
    print("\nGraph")
    entities = store.list_entities(account_id, active_only=False)
    rels = store.list_relationships(account_id, active_only=False)
    facts = [e for e in entities if str(e.get("entity_type") or "").lower() == "fact"]
    line(OK if entities else GAP, "entities", f"{len(entities)} ({len(facts)} fact)")
    gaps += not entities
    line(OK if rels else GAP, "relationships", str(len(rels)))
    gaps += not rels

    broken = [r for r in rels if str(r.get("validation_status") or "") == "broken"]
    line(WARN if broken else OK, "flagged unverified",
         f"{len(broken)} — expected if you have imported a mapping document"
         if broken else "none")

    def _conditions(row) -> list:
        raw = row.get("join_conditions")
        if isinstance(raw, str):
            import json
            try:
                raw = json.loads(raw or "[]")
            except ValueError:
                return []
        return list(raw or [])

    composite = [r for r in rels if _conditions(r)]
    line(OK if composite else WARN, "composite joins",
         f"{len(composite)} — L11-5/L11-10 test the round trip" if composite
         else "none — L11-5's false pass says the grouping is untested without one")

    pairs: dict[tuple, int] = {}
    for r in rels:
        key = (str(r.get("from_entity")), str(r.get("to_entity")))
        pairs[key] = pairs.get(key, 0) + 1
    roleplaying = [k for k, n in pairs.items() if n > 1]
    line(OK if roleplaying else WARN, "role-playing joins",
         f"{len(roleplaying)} entity pair(s) joined more than once — L11-11"
         if roleplaying else "none — L11-11 cannot be run here")

    try:
        from core.graph_health import check_graph_health
        health = check_graph_health(account_id)
        errors = [i for i in (health.issues or [])
                  if str(getattr(i, "severity", "") or i.get("severity", "")) == "ERROR"]
        line(OK if not errors else WARN, "health",
             f"score {health.score}, {len(errors)} error(s), "
             f"{len(health.issues or [])} issue(s) total")
    except Exception as exc:      # noqa: BLE001
        line(WARN, "health", f"could not run: {exc}")

    # ── 4 · what the harder sections need ────────────────────────────────
    print("\nFor the later sections")
    try:
        domains = store.list_domains(account_id, active_only=False)
    except Exception:
        domains = []
    line(OK if len(domains) >= 2 else WARN, "subject areas", f"{len(domains)}"
         + ("" if len(domains) >= 2
            else " — section 1 needs two whose synonyms overlap, over DIFFERENT facts"))

    try:
        from store.source_store import list_client_sources
        sources = list_client_sources(account_id)
    except Exception:
        sources = []
    line(OK if len(sources) >= 2 else WARN, "connections", f"{len(sources)}"
         + ("" if len(sources) >= 2 else " — L10-19..L10-22 need a second one"))

    try:
        import core.value_index as vi
        idx = vi._index_path(account_id)
        if idx.is_file():
            rows = vi.sample_values_by_column(account_id)
            line(OK if rows else WARN, "value index",
                 f"{len(rows)} column(s) indexed — drafting (10.1) reads this")
        else:
            line(WARN, "value index",
                 "not built — L10-1..L10-8 cannot produce a draft")
    except Exception as exc:      # noqa: BLE001
        line(WARN, "value index", f"could not read: {exc}")

    try:
        with store.get_db() as conn:
            asked = conn.execute(
                "SELECT COUNT(*) FROM query_log WHERE account_id=?", (account_id,)
            ).fetchone()[0]
    except Exception:
        asked = 0
    line(OK if asked >= 20 else WARN, "questions logged", f"{asked}"
         + ("" if asked >= 20
            else " — the vocabulary drafter mines these; ask some before section 10"))

    try:
        users = store.list_users(account_id)
    except Exception:
        users = []
    line(OK if len(users) >= 2 else WARN, "portal users", f"{len(users)}"
         + ("" if len(users) >= 2
            else " — L1-3 and L9-x need one RESTRICTED and one unrestricted"))

    # ── 5 · retrieval ────────────────────────────────────────────────────
    print("\nRetrieval")
    try:
        from core.vector_store import _qdrant
        _qdrant()
        line(OK, "qdrant", "reachable")
    except Exception as exc:      # noqa: BLE001
        line(GAP, "qdrant", f"{type(exc).__name__} — every question will fail")
        gaps += 1

    print("\n" + "=" * 66)
    if gaps:
        print(f"{gaps} blocking gap(s). Fix those before recording any case:\n"
              "a case that fails because the workspace lacks a prerequisite is\n"
              "not a finding.")
    else:
        print("No blocking gaps. Start at section 0 of docs/LIVE_TEST_PLAN.md —\n"
              "four regressions, under two minutes each.")
    print("Anything marked 'warn' means a section of the plan cannot be run here,\n"
          "not that something is broken.")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(report(sys.argv[1]) if len(sys.argv) > 1 else (workspaces() or 0))
