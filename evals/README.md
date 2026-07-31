# SQL accuracy evaluation

QueryBot's SQL accuracy target is measured on returned business results, not
only on SQL text. The production-path evaluator uses the same retrieval,
semantic planning, graph resolution, date handling, validation, repair,
policy, execution, and result-protection path as the portal.

## Targets

- First-pass result correctness: at least 85% on the locked staging suite.
- Correct after clarification or constrained repair: at least 95%.
- Safety, ACL, and destructive-query cases: 100% blocked as expected.
- No critical category below 80% (metrics, joins, dates, filters, rankings).

Do not count the deterministic `evals.business_user` suites as LLM accuracy.
They verify validators and result semantics with fixed SQL and are the lower
level regression baseline.

## Golden case format

Prefer expected result rows whenever stable staging data is available:

```yaml
cases:
  - id: revenue_by_month
    question: Show monthly net revenue for 2026.
    expected_tables: [FINANCE.INVOICE_FACT]
    expected_result:
      rows:
        - {month: "2026-01", net_revenue: 125000.00}
        - {month: "2026-02", net_revenue: 131500.00}
      columns: [month, net_revenue]
      order_matters: true
      numeric_tolerance: 0.01
    min_score: 0.85
```

Row-count-only assertions are also supported with `expected_row_count`,
`min_row_count`, and `max_row_count`. SQL substring assertions remain useful
diagnostics but should not substitute for expected values.

## Run the production path

The client must be configured with a staging database, READY KB, model
provider, semantic contract, and applicable user/table policy.

Run the credential-safe preflight first:

```powershell
python -m evals.readiness --client CLIENT_ID --schema SCHEMA_NAME
```

It checks the database assignment, model configuration, schema/KB artifacts,
KB quality, semantic contract, embedding runtime, Qdrant reachability, suite
size, and result-assertion coverage without printing credentials.

```powershell
python -m evals.run `
  --client CLIENT_ID `
  --schema SCHEMA_NAME `
  --cases evals/clients/CLIENT_ID/SCHEMA_NAME/golden_questions.yaml `
  --generate `
  --execute
```

Both flags are intentional. `--generate --execute` uses the full production
pipeline and performs live read-only staging queries, including normal usage,
trace, audit, and policy controls. `--generate` without `--execute` retains the
legacy prompt-only mode for offline diagnostics.

## Staging-suite composition

Use at least 100 locked questions per client domain, with a separate holdout
set. Include simple aggregates, multi-table joins, fact-to-fact cases,
approved metrics, contextual dates, top-N/window functions, anti-joins,
null/zero behavior, ambiguous terms, follow-ups, ACL denials, and destructive
requests. Store the case file and the semantic-contract version with every
reported run so an 85% result is reproducible.
