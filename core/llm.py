"""
core/llm.py

LLM provider abstraction — Anthropic, OpenAI, Azure OpenAI.

v8 prompt changes:
  - build_kb_system_prompt:  DataPilot-style format (Overview / Key Metrics /
    Always Exclude / Columns / Query Patterns / Synonyms).
    NEEDS CONTEXT flag for ambiguous columns. Distinct values MUST be used.
    Generic — no domain-specific examples.
  - build_kb_query_prompt:   NEW — Stage 2 call that generates natural-language
    question → SQL pattern document from the actual KB content.
  - build_sql_system_prompt: MAX(date) rule for relative time queries.
    DDL detection moved here as a pre-check with a human-friendly message.
    All 3 DB types updated.
"""

import logging
from typing import Literal

from core.llm_audit import record_llm_call

log = logging.getLogger("querybot.llm")

Provider = Literal["anthropic", "openai", "azure_openai"]

# ── SQL syntax rules per DB type ──────────────────────────────────────────────

_SQL_SYNTAX: dict[str, str] = {
    "snowflake": (
        "- Row limit: If the user states a number (top 10, show 5), use that number in LIMIT. "
        "If no number is stated, default to LIMIT 20.\n"
        "- Date functions: DATE_TRUNC, DATEADD, CURRENT_DATE, DATEDIFF\n"
        "- Conditionals: IFF(), COALESCE()\n"
        "- Schema-qualify tables if needed: DATABASE.SCHEMA.TABLE\n"
        "- Split name concat: FIRST_NAME || ' ' || LAST_NAME AS FULL_NAME\n"
        "- CRITICAL TIME RULE: When the question uses relative time (last month, last week, "
        "this year, yesterday, recent), NEVER use CURRENT_DATE as the reference point. "
        "The database may be historical. Always anchor to the latest date in the data:\n"
        "  Last month:  WHERE DateCol >= DATEADD('month', -1, (SELECT MAX(DateCol) FROM TableName))\n"
        "  Last week:   WHERE DateCol >= DATEADD('week',  -1, (SELECT MAX(DateCol) FROM TableName))\n"
        "  This year:   WHERE YEAR(DateCol) = YEAR((SELECT MAX(DateCol) FROM TableName))\n"
    ),
    "oracle": (
        "- Row limit: If the user states a number (top 10, show 5), use FETCH FIRST N ROWS ONLY "
        "with that number. Default to FETCH FIRST 20 ROWS ONLY. NEVER use LIMIT.\n"
        "- Date functions: TRUNC(date,'MM'), SYSDATE, ADD_MONTHS, MONTHS_BETWEEN\n"
        "- Null handling: NVL() or COALESCE()\n"
        "- Schema-qualify tables: OWNER.TABLE_NAME\n"
        "- Split name concat: FIRST_NAME || ' ' || LAST_NAME AS FULL_NAME\n"
        "- CRITICAL TIME RULE: When the question uses relative time (last month, last week, "
        "this year, yesterday, recent), NEVER use SYSDATE as the reference point. "
        "The database may be historical. Always anchor to the latest date in the data:\n"
        "  Last month:  WHERE DateCol >= ADD_MONTHS((SELECT MAX(DateCol) FROM TableName), -1)\n"
        "  Last week:   WHERE DateCol >= (SELECT MAX(DateCol) FROM TableName) - 7\n"
        "  This year:   WHERE TRUNC(DateCol,'YYYY') = TRUNC((SELECT MAX(DateCol) FROM TableName),'YYYY')\n"
    ),
    "azure_sql": (
        "- Row limit: If the user states a number (top 10, show 5), use SELECT TOP N with that "
        "number. Default to SELECT TOP 20. TOP goes immediately after SELECT. NEVER use LIMIT.\n"
        "- Date functions: GETDATE(), DATEADD, DATEDIFF, FORMAT, CONVERT\n"
        "- TABLE NAMING RULE (CRITICAL): Azure SQL Database only supports TWO-part table names. "
        "Always write tables as [SCHEMA].[TABLE_NAME]. "
        "The Knowledge Base shows 'SQL table name: [SCHEMA].[TABLE]' for each table — "
        "use that exact two-part format. "
        "NEVER use three-part names like [DATABASE].[SCHEMA].[TABLE] — Azure SQL rejects them with error 40515.\n"
        "- Null handling: ISNULL() or COALESCE()\n"
        "- Split name concat: CONCAT(FIRST_NAME, ' ', LAST_NAME) AS FULL_NAME\n"
        "- CRITICAL TIME RULE: When the question uses relative time (last month, last week, "
        "this year, yesterday, recent), NEVER use GETDATE() as the reference point. "
        "The database may be historical. Always anchor to the latest date in the data:\n"
        "  Last month:  WHERE DateCol >= DATEADD(month, -1, (SELECT MAX(DateCol) FROM [schema].[TableName]))\n"
        "  Last week:   WHERE DateCol >= DATEADD(week,  -1, (SELECT MAX(DateCol) FROM [schema].[TableName]))\n"
        "  This year:   WHERE YEAR(DateCol) = YEAR((SELECT MAX(DateCol) FROM [schema].[TableName]))\n"
    ),
}

_DB_LABELS: dict[str, str] = {
    "snowflake": "Snowflake",
    "oracle":    "Oracle",
    "azure_sql": "Azure SQL",
}

# Operations that must never be executed — user gets a friendly message
_DDL_KEYWORDS = {
    "CREATE", "DROP", "ALTER", "TRUNCATE", "INSERT", "UPDATE", "DELETE",
    "MERGE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL",
    "BULK", "COPY", "PUT", "GET", "UNLOAD", "LOAD",
}


def is_ddl_attempt(text: str) -> bool:
    """Return True if the user's raw message looks like a DDL/DML attempt."""
    import re
    first_word = re.split(r"\s+", text.strip().upper())[0] if text.strip() else ""
    return first_word in _DDL_KEYWORDS


_DDL_USER_MESSAGE = (
    "🔒 *That operation is not permitted.*\n\n"
    "QueryBot is a read-only analytics assistant — it can only run SELECT queries "
    "to retrieve and analyse data.\n\n"
    "Operations that modify data (CREATE, DROP, INSERT, UPDATE, DELETE, etc.) "
    "are blocked for security. If you need to make database changes, please use "
    "your database administration tool directly."
)


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builders
# ══════════════════════════════════════════════════════════════════════════════

def build_sql_system_prompt(
    db_type: str,
    table_context: str,
    conversation_history: list | None = None,
    graph_context: dict | None = None,
) -> str:
    """System prompt for SQL generation — used on every user query.

    graph_context: dict from graph_resolver.resolve_for_question().
    conversation_history: list of {question, sql, columns, row_count} dicts.
    Injected as session context to resolve follow-up references.
    """
    label  = _DB_LABELS.get(db_type, db_type)
    syntax = _SQL_SYNTAX.get(db_type, "- Use standard ANSI SQL\n")
    base = (
        f"You are a {label} SQL expert. "
        "Convert the user's plain-English question into a valid SQL SELECT query.\n\n"
        "STRICT RULES:\n"
        "- Use ONLY the tables and columns described in the Knowledge Base below. "
        "Never invent, assume, or guess column names.\n"
        "- If a column name is flagged [NEEDS CONTEXT] in the KB, do not use it — "
        "reply with CANNOT_GENERATE instead.\n"
        f"{syntax}"
        "- Return ONLY the raw SQL query. No markdown fences, no explanation, no comments.\n"
        "- If the question cannot be answered from the available tables and columns, "
        "reply with exactly: CANNOT_GENERATE\n"
        "- Never generate CREATE, DROP, ALTER, INSERT, UPDATE, DELETE, TRUNCATE, "
        "MERGE, GRANT, REVOKE or any data-modifying statement.\n"
        "- NAME CONCATENATION RULE: When a user asks for a person's name and the "
        "table has separate first/last name columns (FIRST_NAME/LAST_NAME, FNAME/LNAME, "
        "GIVEN_NAME/SURNAME, FORENAME/FAMILY_NAME or similar) but no combined column, "
        "always concatenate them using the dialect syntax shown above. Never return "
        "split name columns separately when the user asked for a name.\n"
        "- CROSS-TABLE QUERY RULE: When a question asks for a metric (count, total, sum, "
        "average, amount) BY or PER a dimension (name, category, region, type, department) "
        "you MUST write a JOIN. Metric columns live in FACT tables. Grouping columns live "
        "in DIMENSION tables. Use the Join Keys from the Knowledge Base. "
        "Pattern: SELECT d.NAME_COL, SUM(f.METRIC_COL) FROM FACT f "
        "JOIN DIM d ON f.FK = d.PK GROUP BY d.NAME_COL ORDER BY 2 DESC LIMIT 20\n"
        "- APPROVED METRIC FORMULA RULE: If the Knowledge Base context includes "
        "'APPROVED METRIC FORMULAS' and the user asks for that metric or synonym, "
        "use the approved calculation exactly. For by/per/grouped-by questions, "
        "put the approved formula in the SELECT list and group by the requested "
        "dimension. Do not average percentage/rate fields unless the approved "
        "formula explicitly uses AVG().\n\n"
        f"Knowledge Base — available tables and their business context:\n{table_context}"
    )
    if conversation_history:
        history_lines = []
        for i, turn in enumerate(conversation_history, 1):
            q    = str(turn.get("question", ""))[:120]
            sql  = str(turn.get("sql",      ""))[:300]
            cols = ", ".join(str(c) for c in (turn.get("columns") or []))
            history_lines.append(
                f"Turn {i}:\n"
                f"  Question: {q}\n"
                f"  Columns returned: {cols}\n"
                f"  SQL used: {sql}"
            )
        base = base + (
            "\n\n## Session context (previous turns this conversation)\n"
            "Use these to resolve follow-up references such as \'top 5\', "
            "\'filter to X\', \'same metric for Y\', \'break that down by Z\'.\n"
            "Do NOT copy previous SQL verbatim — generate fresh SQL for the "
            "NEW question informed by this context.\n\n"
            + "\n\n".join(history_lines)
        )
    # Inject pre-built JOIN skeleton from entity graph when available.
    # The LLM must use this skeleton and must NOT change table names or JOINs.
    if graph_context and graph_context.get("enabled") and graph_context.get("join_skeleton"):
        skeleton  = graph_context["join_skeleton"]
        detected  = ", ".join(graph_context.get("detected", []))
        base = base + (
            "\n\n## Entity graph — pre-resolved JOIN structure\n"
            "The following JOIN clause has been resolved deterministically from the "
            "business entity graph. You MUST use this exact FROM + JOIN structure "
            "in your query. Do NOT change table names, aliases, or JOIN conditions.\n"
            "Detected entities: " + detected + "\n\n"
            "```sql\n" + skeleton + "\n```\n\n"
            "Only write the SELECT clause, GROUP BY, ORDER BY, WHERE, and HAVING "
            "on top of this skeleton. Do not add or remove JOINs."
        )
    return base


def build_kb_system_prompt() -> str:
    """
    Stage 1 KB generation system prompt.
    DataPilot-style format. Generic — works for any database domain.
    Requires distinct values to be used. Flags ambiguous columns.
    """
    return (
        "You are a senior data analyst writing a Knowledge Base document for an AI SQL generator. "
        "The document will be used at query time to produce accurate SQL — "
        "write it for the SQL generator, not for a human reader.\n\n"

        "CRITICAL RULES:\n"
        "1. Use ONLY the column names that appear in the schema provided. "
        "Never invent column names.\n"
        "2. If the schema includes a 'Distinct Values' column for a field, "
        "you MUST list those exact values and use them in examples. "
        "Do not guess or invent values.\n"
        "3. If a column has no distinct values and its business meaning is unclear "
        "(e.g. numeric scores, rates, thresholds with no obvious interpretation), "
        "mark it: [NEEDS CONTEXT] — business rule unknown, do not use in filters.\n"
        "4. If sample data contains placeholder values (SAMPLE_, VAL_, etc.), "
        "ignore them. Use only Distinct Values from the schema.\n"
        "5. No domain-specific assumptions. Write for any industry.\n\n"

        "DOCUMENT FORMAT — produce all 7 sections for every table:\n\n"

        "## Overview\n"
        "One sentence: what this table represents and what it is used for.\n\n"

        "## Key Metrics\n"
        "List every measurable business concept this table answers. "
        "For each metric: business name → exact column name → filter condition if applicable.\n"
        "Format: - **Metric name**: `COLUMN_NAME` — Filter: `WHERE COLUMN = 'value'`\n"
        "Use the actual distinct values from the schema for the filter conditions.\n"
        "IMPORTANT: For any column that holds a monetary amount, quantity, rate, "
        "weight, duration, score or any other measurable quantity, add an explicit "
        "anchor line: 'PRIMARY MEASURE FOR <business term>: use COLUMN_NAME.' "
        "Derive <business term> from the business description and column name — "
        "for a pharmacy that might be revenue/charges, for an HR system it might be "
        "salary/hours, for logistics it might be weight/volume. This anchors the SQL "
        "generator so it uses the correct column for the tenant's actual domain.\n\n"

        "## Always Exclude\n"
        "Standard WHERE conditions that should always be applied "
        "(e.g. active records only, non-null date). "
        "If none apply, write: None identified.\n\n"

        "## Columns\n"
        "For EVERY column in the schema:\n"
        "- `COLUMN_NAME` (TYPE): business meaning. "
        "If distinct values exist, list them: values are 'A', 'B', 'C'. "
        "If ambiguous, write [NEEDS CONTEXT].\n"
        "SPLIT NAME RULE: If the table has separate first/last name columns (FIRST_NAME/LAST_NAME, FNAME/LNAME, GIVEN_NAME/SURNAME or similar) but no combined full name column, document both columns and add the note: [SPLIT NAME - always concatenate for full name queries].\n\n"

        "## Common Query Patterns\n"
        "Write 6-10 specific business questions and the EXACT SQL pattern for each. "
        "Use real column names and real distinct values from the schema.\n"
        "SPLIT NAME PATTERNS: If the table has separate first/last name columns, "
        "every name-related query must show the concatenation: "
        "FIRST_NAME || ' ' || LAST_NAME AS FULL_NAME (or dialect equivalent).\n"
        "Format:\n"
        "Q: [natural language question]\n"
        "SQL: SELECT ... FROM [table] WHERE ...\n\n"

        "## Join Keys\n"
        "List every column that links to another table. "
        "Specify: `COLUMN_NAME` → links to [other table].[column].\n\n"

        "## Business Synonyms\n"
        "Map common plain-English terms to exact column names.\n"
        "Format: | Plain English | Column | Notes |\n"
        "If any column could be confused with a generic name from other databases "
        "(like TOTAL_AMOUNT, STATUS, ID), add a WARNING: note.\n"
        "CRITICAL — for tables containing PEOPLE (employees, customers, users, "
        "doctors, patients, agents, staff, vendors, suppliers, contacts, members): "
        "you MUST add a dedicated row for every informal term a business user might say. "
        "Examples: a PRESCRIBER table → also map: doctor, physician, clinician, provider, "
        "healthcare provider, HCP, medical professional, prescribing doctor. "
        "A STAFF table → also map: employee, worker, team member, agent, rep. "
        "A PATIENT table → also map: customer, client, member, beneficiary. "
        "A SUPPLIER table → also map: vendor, provider, partner, contractor. "
        "Use the table name and column names to infer the domain — even if sample data "
        "contains placeholder values (SAMPLE_), reason from the column names alone.\n\n"

        "Return only the Markdown document. All 7 sections are mandatory."
    )


def build_kb_query_prompt(table_name: str, kb_content: str, business_desc: str) -> str:
    """
    Stage 2 KB generation prompt.
    Takes the Stage 1 KB output and generates a question-to-SQL translation
    document from the actual data and real column values in the KB.
    Generic — no domain assumptions.
    """
    return (
        f"You have been given the Knowledge Base document for table: {table_name}\n\n"
        f"Business context: {business_desc}\n\n"
        f"Knowledge Base content:\n{kb_content}\n\n"
        "Your task: Generate a QUERY TRANSLATION document that maps natural-language "
        "business questions to exact SQL patterns for this table.\n\n"
        "RULES:\n"
        "1. Generate at least 10 question-SQL pairs.\n"
        "2. Use ONLY column names and values that appear in the Knowledge Base above. "
        "Never invent column names or values.\n"
        "3. Cover these question types:\n"
        "   - Counting/aggregation (how many, total, sum)\n"
        "   - Ranking (top N, highest, lowest, most, least)\n"
        "   - Filtering by status/category using actual distinct values from the KB\n"
        "   - Time-based filtering using MAX(date_column) as the reference, not system date\n"
        "   - Cross-dimension analysis (by department, by category, by type)\n"
        "   - Trend questions (this period vs last period)\n"
        "   - IMPORTANT: For FACT tables, always write at least 3 cross-table patterns "
        "using JOIN to dimension tables — e.g. revenue by customer name, orders by region, "
        "transactions by product category, sales by employee. "
        "These cross-table patterns are the most common business questions.\n"
        "4. For every time-relative question (last month, last week, recent), "
        "use MAX(date_column) as the date anchor, not GETDATE()/CURRENT_DATE/SYSDATE.\n"
        "5. Write questions the way a non-technical business user would actually ask them.\n\n"
        "FORMAT — use this exact structure for each pair:\n"
        "Q: [natural language question a business user would ask]\n"
        "SQL: [complete, runnable SQL using real column names]\n\n"
        "Return only the question-SQL pairs, no other text."
    )


def build_biz_vocab_prompt(
    table_names: list[str],
    column_reference: str,
    business_desc: str,
) -> str:
    """
    Business vocabulary KB prompt.
    Grounds the LLM to real column names across all tables.
    Generic — no domain assumptions.
    """
    return (
        f"Business description:\n{business_desc}\n\n"
        f"Tables in this database:\n{', '.join(table_names)}\n\n"
        f"EXACT column names per table "
        f"(you MUST use ONLY these — never invent column names):\n"
        f"{column_reference}\n\n"
        "Generate a Business Vocabulary document that maps plain-English business terms "
        "to the exact table names and column names listed above. "
        "Include: key business entities, common metrics, synonyms, and how business "
        "language maps to the actual column names. "
        "Every column or table name you reference must come from the list above. "
        "Do not use any name not in that list."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Core completion function
# ══════════════════════════════════════════════════════════════════════════════

async def llm_complete(
    system: str,
    user: str,
    provider: Provider,
    model: str,
    api_key: str,
    max_tokens: int = 1024,
    azure_endpoint: str = "",
    azure_api_version: str = "2024-02-01",
    temperature: float = 0.7,
) -> tuple[str, int, int]:
    try:
        if provider == "anthropic":
            result = await _anthropic_complete(system, user, model, api_key, max_tokens, temperature)
        elif provider == "openai":
            result = await _openai_complete(system, user, model, api_key, max_tokens, temperature)
        elif provider == "azure_openai":
            result = await _azure_openai_complete(
                system, user, model, api_key, max_tokens, azure_endpoint, azure_api_version, temperature
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider!r}")
    except Exception as exc:
        record_llm_call(
            llm_provider=provider,
            llm_model=model,
            system=system,
            user=user,
            status="error",
            error_msg=str(exc),
        )
        raise

    record_llm_call(
        llm_provider=provider,
        llm_model=model,
        system=system,
        user=user,
        status="success",
    )
    return result


async def _anthropic_complete(system, user, model, api_key, max_tokens, temperature=0.7):
    import anthropic as _ant
    client = _ant.AsyncAnthropic(api_key=api_key)
    try:
        resp = await client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip(), resp.usage.input_tokens, resp.usage.output_tokens
    except Exception as e:
        log.error("Anthropic API error: %s", e)
        raise RuntimeError(f"Anthropic API error: {e}") from e
    finally:
        await client.close()


async def _openai_complete(system, user, model, api_key, max_tokens, temperature=0.7):
    import openai as _oai
    client = _oai.AsyncOpenAI(api_key=api_key)
    try:
        resp = await client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens
    except Exception as e:
        log.error("OpenAI API error: %s", e)
        raise RuntimeError(f"OpenAI API error: {e}") from e
    finally:
        await client.close()


async def _azure_openai_complete(system, user, model, api_key, max_tokens, endpoint, api_version, temperature=0.7):
    import openai as _oai
    if not endpoint:
        raise RuntimeError(
            "Azure OpenAI endpoint not configured. "
            "Go to Admin → System and enter your endpoint URL."
        )
    client = _oai.AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint.rstrip("/"),
        api_version=api_version or "2024-02-01",
    )
    try:
        resp = await client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens
    except Exception as e:
        log.error("Azure OpenAI error: %s", e)
        raise RuntimeError(
            f"Azure OpenAI error: {e}\n\n"
            "Check your endpoint URL, API key, and deployment name in Admin → System."
        ) from e
    finally:
        await client.close()


# ══════════════════════════════════════════════════════════════════════════════
# Provider resolution
# ══════════════════════════════════════════════════════════════════════════════

def resolve_provider(client: dict, purpose: str = "query") -> tuple[str, str, str, dict]:
    import store
    sys_cfg = store.get_all_system()

    provider = (
        client.get("llm_provider")
        or sys_cfg.get("default_llm_provider")
        or "anthropic"
    )

    if purpose == "kb":
        model = sys_cfg.get("kb_llm_model") or _default_model(provider, "high")
    else:
        model = (
            client.get("llm_model")
            or sys_cfg.get("default_llm_model")
            or _default_model(provider, "fast")
        )

    extra_kwargs: dict = {}
    if provider == "anthropic":
        api_key = sys_cfg.get("anthropic_api_key", "")
    elif provider == "openai":
        api_key = sys_cfg.get("openai_api_key", "")
    elif provider == "azure_openai":
        api_key  = sys_cfg.get("azure_openai_api_key", "")
        endpoint = sys_cfg.get("azure_openai_endpoint", "")
        version  = sys_cfg.get("azure_openai_api_version", "2024-02-01")
        if not endpoint:
            raise RuntimeError(
                "Azure OpenAI endpoint not configured. "
                "Go to Admin → System → Azure OpenAI settings."
            )
        extra_kwargs = {"azure_endpoint": endpoint, "azure_api_version": version}
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    if not api_key:
        raise RuntimeError(
            f"No API key configured for provider '{provider}'. "
            "Go to Admin → System and add your API key."
        )

    return provider, model, api_key, extra_kwargs


def _default_model(provider: str, quality: str) -> str:
    defaults = {
        "anthropic":    {"fast": "claude-sonnet-4-6", "high": "claude-opus-4-5"},
        "openai":       {"fast": "gpt-4o-mini",       "high": "gpt-4o"},
        "azure_openai": {"fast": "gpt-4o-mini",       "high": "gpt-4o"},
    }
    return defaults.get(provider, {}).get(quality, "gpt-4o-mini")
