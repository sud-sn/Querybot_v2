"""
CEO / Business-user quality test — PROFITABILITY schema.
Run: /home/chatbotadmin/Querybot_v2/venv/bin/python3 /tmp/test_bot_ceo.py
"""
import sys, asyncio, json, time
sys.path.insert(0, '/home/chatbotadmin/Querybot_v2')
sys.stdout.reconfigure(encoding='utf-8')
import os; os.chdir('/home/chatbotadmin/Querybot_v2')

ACCOUNT = 'Web_UI'
SCHEMA  = 'PROFITABILITY'

QUESTIONS = [
    # ── Revenue & Margin ──────────────────────────────────────────────────────
    {"q": "What is our total revenue this year?",                         "tag": "P01 Revenue total"},
    {"q": "What is total revenue by profit center?",                      "tag": "P02 Revenue by profit center"},
    {"q": "Which are our top 10 customers by revenue?",                   "tag": "P03 Top customers revenue"},
    {"q": "What is our gross margin by customer type?",                   "tag": "P04 Margin by customer type"},
    {"q": "What is our gross margin percentage this year?",               "tag": "P05 Margin percentage"},
    {"q": "Which profit center has the highest revenue?",                 "tag": "P06 Best profit center"},
    {"q": "What is our total COGS this year?",                            "tag": "P07 Total COGS"},
    # ── Inventory ────────────────────────────────────────────────────────────
    {"q": "What is the current on-hand inventory quantity by warehouse?", "tag": "P08 Inventory by warehouse"},
    {"q": "Which warehouse has the most stock on hand?",                  "tag": "P09 Highest stock warehouse"},
    {"q": "What is total on-hand inventory across all warehouses?",       "tag": "P10 Total inventory"},
    # ── Purchasing ───────────────────────────────────────────────────────────
    {"q": "How many purchase orders were received last month?",           "tag": "P11 PO count last month"},
    {"q": "What is total purchase orders received by warehouse?",         "tag": "P12 POs by warehouse"},
    # ── Customer & Operations ─────────────────────────────────────────────────
    {"q": "How many customers did we invoice this year?",                 "tag": "P13 Customer count"},
    {"q": "What is our revenue trend by month this year?",                "tag": "P14 Monthly revenue trend"},
]

import store

class _Event:
    def __init__(self):
        self.user_id      = 'test_ceo'
        self.channel      = 'test'
        self.schema_hint  = SCHEMA
        self.conversation_history = []
        self.event_id     = 'test'

class _Adapter:
    def __init__(self):
        self.messages = []; self.sql = ''; self.rows = []; self.error = ''
    async def send_message(self, event, text, **kw):
        self.messages.append(str(text)[:600])
    async def send_chart(self, *a, **kw): pass
    async def send_results(self, event, question, rows, sql, **kw):
        self.sql = sql or ''; self.rows = rows or []
    async def send_error(self, event, text, **kw):
        self.error = str(text)[:400]
    async def send_stage(self, *a, **kw): pass
    def add_to_history(self, **kw): pass

async def ask(q):
    from core.query_pipeline import handle_query
    event   = _Event()
    adapter = _Adapter()
    portal_user = {'account_id': ACCOUNT, 'id': None, 'role': 'admin', 'allowed_tables': None}
    t0 = time.time()
    try:
        await handle_query(ACCOUNT, event, adapter, q['q'], portal_user)
    except Exception as exc:
        adapter.error = f'EXCEPTION: {exc}'
    elapsed = round((time.time() - t0) * 1000)

    if adapter.error:
        status = 'FAIL'; detail = adapter.error[:300]
    elif adapter.rows:
        status = 'PASS'; detail = f'{len(adapter.rows)} row(s)'
    elif adapter.messages:
        txt = ' '.join(adapter.messages)
        low = txt.lower()
        if any(w in low for w in ['error','fail','unable','cannot','no table','invalid','no database','no result']):
            status = 'FAIL'; detail = txt[:300]
        elif any(w in low for w in ['clarif','which schema','which one','ambiguous']):
            status = 'CLARIFY'; detail = txt[:300]
        else:
            status = 'PASS'; detail = txt[:300]
    else:
        status = 'PASS'; detail = '(empty result set)'

    sample = []
    for row in (adapter.rows or [])[:3]:
        d = dict(row) if hasattr(row,'keys') else row
        sample.append({k: v for k, v in list(d.items())[:5]})

    return {
        'tag': q['tag'], 'question': q['q'], 'status': status,
        'detail': detail, 'sql': (' '.join((adapter.sql or '').split()))[:300],
        'sample': sample, 'ms': elapsed,
    }

async def main():
    print('=' * 72)
    print(f'QueryBot v2 — CEO Quality Test | Account: {ACCOUNT} | Schema: {SCHEMA}')
    print(f'{len(QUESTIONS)} questions')
    print('=' * 72)

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f'\n[{i:02d}/{len(QUESTIONS)}] {q["tag"]}')
        print(f'  Q: {q["q"]}')
        r = await ask(q)
        results.append(r)
        icon = {'PASS': '✓', 'FAIL': '✗', 'CLARIFY': '?'}.get(r['status'], '?')
        print(f'  {icon} {r["status"]} ({r["ms"]}ms)')
        if r['sql']:
            print(f'  SQL: {r["sql"][:200]}')
        if r['sample']:
            print(f'  Data: {r["sample"][:2]}')
        if r['status'] != 'PASS':
            print(f'  Detail: {r["detail"][:250]}')

    passed  = sum(1 for r in results if r['status'] == 'PASS')
    failed  = sum(1 for r in results if r['status'] == 'FAIL')
    clarify = sum(1 for r in results if r['status'] == 'CLARIFY')
    total   = len(results)
    avg_ms  = int(sum(r['ms'] for r in results) / total)

    print('\n' + '=' * 72)
    print('RESULTS SUMMARY')
    print('=' * 72)
    print(f'  PASS   : {passed}/{total}  ({round(passed/total*100)}%)')
    print(f'  FAIL   : {failed}/{total}')
    print(f'  CLARIFY: {clarify}/{total}')
    print(f'  Avg ms : {avg_ms}ms')
    if failed or clarify:
        print('\nNon-PASS items:')
        for r in results:
            if r['status'] != 'PASS':
                print(f'  [{r["status"]}] {r["tag"]}: {r["detail"][:200]}')

    print('\n## JSON_RESULTS_START')
    print(json.dumps(results, indent=2, default=str))
    print('## JSON_RESULTS_END')

asyncio.run(main())
