"""A question the plan cannot express is answered by one query the AI writes from the approved model, checked by
code before it runs (core2/direct.py, the safety net).

"What share of the posts with a score over 5 are owned by members over 65?" needs a shape no plan holds, and was
refused. The AI now writes the query, from the tables, links and metrics the reader may use and nothing else, and
code checks it: one read-only SELECT; only approved tables and columns; tables joined only on their approved links;
no figure added up over rows a join repeats; no function that reads files or runs commands. Each table's default
filters and the row limit are put on by code. The answer is marked as a direct query, says how it was worked out in
code's words, and shows the query to creators only. A query that breaks a rule never runs, and the refusal stands.
It is off in every workspace until an admin turns it on.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json
import random

import duckdb
import pytest

from core2 import direct as DQ
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.model.schema import ColumnFilter
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def forum():
    rng = random.Random(23)
    con = duckdb.connect()
    con.execute("CREATE TABLE users (Id INTEGER, DisplayName VARCHAR, Age INTEGER, Reputation INTEGER)")
    con.execute("""CREATE TABLE posts (Id INTEGER, OwnerUserId INTEGER, LastEditorUserId INTEGER, Title VARCHAR,
                   Score INTEGER, ViewCount INTEGER, CreationDate DATE)""")
    con.execute("CREATE TABLE votes (Id INTEGER, PostId INTEGER, UserId INTEGER, VoteTypeId INTEGER)")
    con.executemany("INSERT INTO users VALUES (?,?,?,?)",
                    [(i, f"Member {i:03d}", rng.randint(18, 80), rng.randint(1, 5000)) for i in range(1, 301)])
    con.executemany("INSERT INTO posts VALUES (?,?,?,?,?,?,?)",
                    [(i, rng.randint(1, 300), rng.randint(1, 300), f"Notes on topic {i}", rng.randint(-3, 40),
                      rng.randint(5, 9000), dt.date(2010, 1, 1) + dt.timedelta(days=i * 3)) for i in range(1, 501)])
    con.executemany("INSERT INTO votes VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 500), rng.randint(1, 300), rng.randint(1, 5)) for i in range(1, 701)])
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=[]), client_id="eval-forum-direct",
                        options=BuildOptions(workers=1))

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'"{model.tables[column.table].name}"').rows]

    return model, warehouse, build_index(model, fetch)


def _view(model, allowed=None):
    return DQ.approved(model, "duckdb", allowed)


def _key(model, table):
    return next(k for k, t in model.tables.items() if t.name == table)


def _column(model, table, name):
    return next(c for c in model.columns.values() if c.name == name and model.tables[c.table].name == table)


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) AS posts FROM posts p JOIN users u ON p.OwnerUserId = u.Id WHERE u.Age > 65",
    "SELECT u.DisplayName AS who, SUM(p.ViewCount) AS views FROM posts p JOIN users u ON p.OwnerUserId = u.Id "
    "GROUP BY u.DisplayName ORDER BY views DESC",
    "SELECT AVG(x.n) AS posts_each FROM (SELECT p.OwnerUserId AS uid, COUNT(*) AS n FROM posts p "
    "GROUP BY p.OwnerUserId) x",
    "SELECT COUNT(DISTINCT b.uid) AS members FROM (SELECT p.OwnerUserId AS uid FROM posts p WHERE p.Score > 10) b",
    "SELECT COUNT(*) AS members FROM users u WHERE EXISTS (SELECT 1 FROM posts p WHERE p.OwnerUserId = u.Id)",
    "SELECT u.DisplayName AS who, x.n AS posts FROM users u JOIN (SELECT p.OwnerUserId AS uid, COUNT(*) AS n "
    "FROM posts p GROUP BY p.OwnerUserId) x ON x.uid = u.Id",
])
def test_a_query_on_the_approved_model_passes(forum, sql):
    checked = DQ.check(sql, _view(forum[0]))
    assert "LIMIT 5000" in checked.tree.sql(dialect="duckdb")


@pytest.mark.parametrize("sql, why", [
    ("DELETE FROM users", "only SELECT"),
    ("SELECT 1 AS a; SELECT 2 AS b", "one statement"),
    ("SELECT u.Id FROM users u JOIN votes v ON u.Id = v.Id", "not an approved link"),
    ("SELECT COUNT(*) FROM users u, votes v WHERE u.Id = v.UserId", "JOIN ... ON"),
    ("SELECT COUNT(*) FROM users u CROSS JOIN votes v", "JOIN ... ON"),
    ("SELECT COUNT(*) FROM posts p JOIN users u ON p.OwnerUserId = u.Id OR p.LastEditorUserId = u.Id", "OR"),
    ("SELECT COUNT(*) FROM posts p JOIN users u ON p.ViewCount > u.Reputation", "no approved link"),
    ("SELECT SUM(u.Reputation) AS rep FROM posts p JOIN users u ON p.OwnerUserId = u.Id", "repeat"),
    ("SELECT SUM(CASE WHEN p.Score > 5 THEN u.Reputation ELSE 0 END) AS rep FROM posts p "
     "JOIN users u ON p.OwnerUserId = u.Id", "repeat"),
    ("SELECT * FROM users", "name each column"),
    ("SELECT u.* FROM users u", "name each column"),
    ("SELECT x.a FROM read_csv('/etc/passwd') x", "not allowed"),
    ("SELECT u.Password FROM users u", "not an approved column"),
    ("SELECT Age FROM posts p JOIN users u ON p.OwnerUserId = u.Id", "with its table's alias"),
    ("SELECT t.table_name FROM information_schema.tables t", "not one of the approved tables"),
    ("SELECT u.DisplayName FROM users u WHERE u.DisplayName = '⟨v9⟩'", "name the question did not hold"),
])
def test_a_query_that_breaks_a_rule_is_refused_with_the_rule(forum, sql, why):
    with pytest.raises(DQ.NotSafe) as refused:
        DQ.check(sql, _view(forum[0]))
    assert why in str(refused.value), str(refused.value)


def test_only_what_the_reader_may_see_is_approved(forum):
    model = forum[0]
    sql = "SELECT AVG(u.Reputation) AS rep FROM users u"
    hidden = model.model_copy(deep=True)
    _column(hidden, "users", "Reputation").hidden = True
    with pytest.raises(DQ.NotSafe, match="not an approved column"):
        DQ.check(sql, _view(hidden))
    detail = model.model_copy(deep=True)
    _column(detail, "users", "Reputation").personal = "detail"
    with pytest.raises(DQ.NotSafe, match="not an approved column"):
        DQ.check(sql, _view(detail))
    assert "Reputation" not in DQ.schema_text(_view(detail), values_allowed=True)
    # A table the reader may not use is no table of the query, and none of its links are offered.
    posts_only = _view(model, {_key(model, "posts")})
    with pytest.raises(DQ.NotSafe, match="not one of the approved tables"):
        DQ.check("SELECT COUNT(*) FROM posts p JOIN users u ON p.OwnerUserId = u.Id", posts_only)
    assert "users" not in DQ.schema_text(posts_only, values_allowed=True)


def test_each_table_is_read_through_its_default_filters(forum):
    model, warehouse, _ = forum
    kept = model.model_copy(deep=True)
    kept.tables[_key(kept, "posts")].default_filters = [
        ColumnFilter(column=_column(kept, "posts", "Score").key, op="gte", values=[0])]
    checked = DQ.check("SELECT COUNT(*) AS posts FROM posts p", _view(kept))
    got = warehouse.query(checked.tree.sql(dialect="duckdb")).rows[0][0]
    assert got == warehouse.query("SELECT COUNT(*) FROM posts WHERE Score >= 0").rows[0][0] < 500
    assert any("left out" in n.lower() or "score" in n.lower() for n in DQ.left_out_notes(_view(kept), checked))


# ── no way round the checks: the reader's question reaches the AI, so a reader can ask for any SQL ──

@pytest.fixture(scope="module")
def guarded(forum):
    """The forum with a hidden figure (reputation), a personal detail (age), a sensitive one (views), and rows each
    table leaves out by default."""
    model = forum[0].model_copy(deep=True)
    _column(model, "users", "Reputation").hidden = True
    _column(model, "users", "Age").personal = "detail"
    _column(model, "posts", "ViewCount").sensitivity = "confidential"
    model.tables[_key(model, "users")].default_filters = [
        ColumnFilter(column=_column(model, "users", "Id").key, op="gt", values=[0])]
    model.tables[_key(model, "posts")].default_filters = [
        ColumnFilter(column=_column(model, "posts", "Score").key, op="gte", values=[-10])]
    return model


@pytest.mark.parametrize("dialect, sql", [
    # a WITH named like a table, read as the table
    ("duckdb", "WITH users AS (SELECT u.Id AS Id FROM users u) SELECT * FROM users"),
    ("duckdb", "WITH users AS (SELECT p.Title AS Reputation FROM posts p) SELECT x.Reputation AS r FROM users x"),
    # an output's name standing for a column it is not
    ("duckdb", "SELECT u.DisplayName AS Reputation, Reputation AS leaked FROM users u"),
    ("tsql", "SELECT u.DisplayName AS Reputation, Reputation AS leaked FROM users u"),
    ("duckdb", "SELECT u.Id AS Reputation, SUM(Reputation) AS s FROM users u GROUP BY u.Id"),
    ("duckdb", "SELECT u.Id AS Reputation FROM users u GROUP BY u.Id HAVING MAX(Reputation) > 4000"),
    ("duckdb", "SELECT u.Id AS Age FROM users u WHERE Age > 65"),
    ("postgres", "SELECT p.Title AS ViewCount, ViewCount AS v FROM posts p"),
    # columns chosen when the query runs
    ("duckdb", "SELECT COLUMNS('Rep.*') FROM users u"),
    ("duckdb", "SELECT COLUMNS(c -> c LIKE 'Rep%') FROM users u"),
    ("snowflake", "SELECT IDENTIFIER('REPUTATION') AS r FROM users u"),
    # locks, sequences, functions anyone may define
    ("postgres", "SELECT v.Id AS i FROM votes v FOR UPDATE"),
    ("tsql", "SELECT v.Id AS i FROM votes v WITH (UPDLOCK, HOLDLOCK)"),
    ("postgres", "SELECT setval('seq', 1) AS s FROM votes v"),
    ("postgres", "SELECT nextval('seq') AS s FROM votes v"),
    ("postgres", "SELECT public.purge_rows(v.Id) AS x FROM votes v"),
    # added up over rows a join repeats, through a subquery or another aggregate
    ("duckdb", "SELECT SUM(x.n) AS total FROM (SELECT u.Id AS n FROM posts p JOIN users u ON p.OwnerUserId = u.Id) x"),
    ("duckdb", "SELECT SUM(x.id) AS t FROM posts p JOIN (SELECT u.Id AS id FROM users u) x ON p.OwnerUserId = x.id"),
    ("duckdb", "SELECT fsum(u.Id) AS s FROM posts p JOIN users u ON p.OwnerUserId = u.Id"),
    ("duckdb", "SELECT mean(u.Id) AS s FROM posts p JOIN users u ON p.OwnerUserId = u.Id"),
    ("duckdb", "SELECT list_sum(LIST(u.Id)) AS s FROM posts p JOIN users u ON p.OwnerUserId = u.Id"),
    ("duckdb", "SELECT STRING_AGG(u.DisplayName, ',') AS s FROM posts p JOIN users u ON p.OwnerUserId = u.Id"),
    ("duckdb", "SELECT COUNT(*) AS n FROM posts p JOIN users u ON p.OwnerUserId = u.Id "
               "JOIN votes v ON v.UserId = u.Id"),
    # two tables matched outside JOIN ... ON on no link
    ("duckdb", "SELECT COUNT(*) AS n FROM users u WHERE EXISTS (SELECT 1 FROM votes v WHERE v.Id = u.Id)"),
    ("duckdb", "SELECT COUNT(*) AS n FROM users u WHERE u.Id IN (SELECT v.Id FROM votes v)"),
    ("duckdb", "SELECT u.Id AS i, (SELECT MAX(v.VoteTypeId) FROM votes v WHERE v.Id = u.Id) AS m FROM users u"),
    ("duckdb", "SELECT COUNT(*) AS n FROM users u JOIN (SELECT v.UserId AS a, v.Id AS b FROM votes v) x(b, a) "
               "ON x.a = u.Id"),
    # past versions of a table, a limit that is not a number of rows, two queries in one
    ("snowflake", "SELECT v.Id AS i FROM votes AT(OFFSET => -3600) v"),
    ("tsql", "SELECT v.Id AS i FROM votes FOR SYSTEM_TIME ALL AS v"),
    ("tsql", "SELECT TOP 100 PERCENT v.Id AS i FROM votes v"),
    ("tsql", "SELECT TOP 10 WITH TIES v.Id AS i FROM votes v ORDER BY v.Id"),
    ("duckdb", "SELECT u.Id AS i FROM users u UNION SELECT v.Id FROM votes v"),
])
def test_no_query_gets_round_the_checks(guarded, dialect, sql):
    with pytest.raises(DQ.NotSafe):
        DQ.check(sql, DQ.approved(guarded, dialect))


def test_an_outputs_name_is_read_as_what_it_is(guarded, forum):
    warehouse = forum[1]
    view = DQ.approved(guarded, "duckdb")
    # Named like a hidden figure, ordered by it: ordered by what the output is, the member's name.
    checked = DQ.check("SELECT u.DisplayName AS Reputation FROM users u ORDER BY Reputation DESC LIMIT 3", view)
    sql = checked.tree.sql(dialect="duckdb")
    assert "ORDER BY dq_row" not in sql and "Reputation DESC" not in sql and "DisplayName DESC" in sql, sql
    assert [r[0] for r in warehouse.query(sql).rows] == [
        r[0] for r in warehouse.query("SELECT DisplayName FROM users ORDER BY DisplayName DESC LIMIT 3").rows]
    ranked = DQ.check("SELECT u.DisplayName AS who, COUNT(*) AS n FROM posts p JOIN users u ON p.OwnerUserId = u.Id "
                      "GROUP BY who ORDER BY n DESC", view)
    assert warehouse.query(ranked.tree.sql(dialect="duckdb")).rows


def test_a_list_of_a_subquerys_values_on_a_link_passes(guarded, forum):
    view = DQ.approved(guarded, "duckdb")
    checked = DQ.check("SELECT COUNT(*) AS n FROM posts p WHERE p.OwnerUserId IN "
                       "(SELECT u.Id FROM users u WHERE u.DisplayName LIKE 'Member 1%')", view)
    assert forum[1].query(checked.tree.sql(dialect="duckdb")).rows[0][0] > 0
    # A figure the subquery works out is a value, not a match of two tables.
    DQ.check("SELECT COUNT(*) AS n FROM posts p WHERE p.Score IN (SELECT MAX(q.Score) FROM posts q)", view)


def test_each_table_runs_as_the_approved_one_in_full(guarded, forum):
    view = DQ.approved(guarded, "duckdb")
    sql = DQ.check("SELECT v.VoteTypeId AS kind FROM votes v", view).tree.sql(dialect="duckdb")
    assert view.tables[_key(guarded, "votes")].sql in sql, sql
    # Read through its default filters, a table gives its approved columns by name: never *, which a classified
    # workspace refuses, and never a column the query could not name.
    sql = DQ.check("SELECT u.DisplayName AS who FROM users u", view).tree.sql(dialect="duckdb")
    assert "*" not in sql and "Reputation" not in sql and "Age" not in sql, sql
    assert forum[1].query(sql).rows


def test_each_column_runs_as_it_is_stored(guarded):
    # Matched in any case, written as stored: a quoted "displayname" in Snowflake is another column than DisplayName.
    sql = DQ.check('SELECT u."displayname" AS d FROM "users" u', DQ.approved(guarded, "snowflake")).tree \
        .sql(dialect="snowflake")
    assert 'u."DisplayName"' in sql and '"displayname"' not in sql, sql
    # Two columns whose names differ only by case: neither is approved, so a name in any case means one column.
    twin = guarded.model_copy(deep=True)
    name = _column(twin, "users", "DisplayName")
    twin.columns["users.displayname_lower"] = name.model_copy(update={"key": "users.displayname_lower",
                                                                      "name": "displayname", "hidden": True})
    with pytest.raises(DQ.NotSafe, match="not an approved column"):
        DQ.check('SELECT u."DisplayName" AS d FROM users u', DQ.approved(twin, "snowflake"))


def test_default_filters_are_written_as_the_compiler_writes_them(forum):
    model, warehouse, _ = forum
    title = _column(model, "posts", "Title")
    for op, values, where in [
        ("contains", ["TOPIC 1"], "upper(Title) LIKE '%TOPIC 1%'"),
        ("starts_with", ["notes on topic 2"], "upper(Title) LIKE 'NOTES ON TOPIC 2%'"),
        ("between", [10, 20], "Score BETWEEN 10 AND 20"),
    ]:
        kept = model.model_copy(deep=True)
        column = title if op != "between" else _column(kept, "posts", "Score")
        kept.tables[_key(kept, "posts")].default_filters = [ColumnFilter(column=column.key, op=op, values=values)]
        checked = DQ.check("SELECT COUNT(*) AS n FROM posts p", _view(kept))
        assert warehouse.query(checked.tree.sql(dialect="duckdb")).rows[0][0] == \
            warehouse.query(f"SELECT COUNT(*) FROM posts WHERE {where}").rows[0][0], op
    # A name on a field whose values are never read is matched in either case, as the compiler matches it.
    unread = model.model_copy(deep=True)
    unread.columns[title.key].values_allowed = False
    unread.tables[_key(unread, "posts")].default_filters = [
        ColumnFilter(column=title.key, op="ne", values=["notes on topic 7"])]
    checked = DQ.check("SELECT COUNT(*) AS n FROM posts p", _view(unread))
    assert warehouse.query(checked.tree.sql(dialect="duckdb")).rows[0][0] == 499


def test_the_lines_of_a_document_left_out_are_left_out_with_it(forum):
    model, warehouse, _ = forum
    kept = model.model_copy(deep=True)
    posts = kept.tables[_key(kept, "posts")]
    posts.kind = "fact"
    posts.default_filters = [ColumnFilter(column=_column(kept, "posts", "Score").key, op="gte", values=[20])]
    checked = DQ.check("SELECT COUNT(*) AS n FROM votes v", _view(kept))
    assert warehouse.query(checked.tree.sql(dialect="duckdb")).rows[0][0] == warehouse.query(
        "SELECT COUNT(*) FROM votes v WHERE EXISTS (SELECT 1 FROM posts p WHERE p.Id = v.PostId AND p.Score >= 20)"
    ).rows[0][0]
    assert any("score" in n.lower() for n in DQ.left_out_notes(_view(kept), checked))


def test_a_name_given_as_its_placeholder_is_put_back(forum):
    checked = DQ.check("SELECT COUNT(*) AS posts FROM posts p WHERE p.Title = '⟨v1⟩'", _view(forum[0]),
                       values={"⟨v1⟩": "Notes on topic 7"})
    assert "'Notes on topic 7'" in checked.tree.sql(dialect="duckdb")


def test_the_switch_is_off_until_an_admin_turns_it_on():
    assert DQ.enabled({}) is False
    assert DQ.enabled(None) is False
    assert DQ.enabled({DQ.SWITCH: True}) is True
    assert DQ.enabled({DQ.SWITCH: False}) is False
    assert DQ.enabled({DQ.SWITCH: "false"}) is False


# ── the whole chat path ──

QUESTION = "What share of the posts with a score over 5 are owned by members over 65?"
SQL = ("SELECT 100.0 * SUM(CASE WHEN u.Age > 65 THEN 1 ELSE 0 END) / COUNT(*) AS share_of_posts "
       "FROM posts p JOIN users u ON p.OwnerUserId = u.Id WHERE p.Score > 5")


class Recorded:
    """The planner says the plan cannot hold it; asked for a query, the AI writes ``sql``."""

    def __init__(self, sql=SQL, parts=None):
        self.sql = sql
        self.parts = parts if parts is not None else [
            {"words": "share of the posts", "in": "share_of_posts"},
            {"words": "owned by members over 65", "in": "u.Age > 65"},
            {"words": "with a score over 5", "in": "p.Score > 5"}]
        self.sent: list[tuple[str, str]] = []

    def __call__(self, stable, tail):
        self.sent.append((stable, tail))
        if "You write one SQL query" in stable:
            return json.dumps({"sql": self.sql, "parts": self.parts})
        return json.dumps({"kind": "unsupported", "notes": ["a share of rows by a condition on another table"],
                           "asked": [{"words": "share of the posts", "kind": "figure"},
                                     {"words": "owned by members over 65", "kind": "condition"},
                                     {"words": "with a score over 5", "kind": "condition"}]})


class Counting:
    def __init__(self, inner):
        self.inner, self.ran = inner, []

    @property
    def dialect(self):
        return self.inner.dialect

    @property
    def db_type(self):
        return self.inner.db_type

    def query(self, sql, *, max_rows=None):
        self.ran.append(sql)
        return self.inner.query(sql, max_rows=max_rows)


def _ask(forum, ai, *, direct=True, sees_query=False):
    model, warehouse, index = forum
    counted = Counting(warehouse)
    services = Services(model=model, warehouse=counted, complete=ai, index=index, today=TODAY, direct=direct,
                        sees_query=sees_query)
    return answer_question(QUESTION, services, Session()), counted.ran


def test_the_question_is_answered_by_the_checked_query(forum):
    ai = Recorded()
    payload, ran = _ask(forum, ai)
    hand = forum[1].query("SELECT 100.0 * SUM(CASE WHEN u.Age > 65 THEN 1 ELSE 0 END) / COUNT(*) FROM posts p "
                          "JOIN users u ON p.OwnerUserId = u.Id WHERE p.Score > 5").rows[0][0]
    assert payload.get("direct_query") is True, payload["answer"]
    assert round(float(payload["kpi"]["value"]), 6) == round(float(hand), 6)
    assert any(b.get("kind") == "direct_query" for b in payload["answer"]["badges"])
    assert "checked before it ran" in payload["answer"]["scope_note"]
    # The AI was given the approved model, never a row of the data.
    stable = next(s for s, _ in ai.sent if "You write one SQL query" in s)
    assert "posts.OwnerUserId = users.Id" in stable and "Member 0" not in stable
    # Only a creator sees the query.
    assert payload["trust"]["sql"] == ""
    shown, _ = _ask(forum, Recorded(), sees_query=True)
    assert "JOIN" in shown["trust"]["sql"]


def test_a_question_about_business_in_general_is_never_a_query(forum):
    model, warehouse, index = forum
    ai = Recorded()
    sorter = lambda stable, tail: json.dumps({"kind": "general_knowledge", "answer": "EBITDA is ..."})  # noqa: E731
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY, direct=True,
                        beyond=sorter)
    payload = answer_question("What does EBITDA mean?", services, Session())
    assert not payload.get("direct_query")
    assert not any("You write one SQL query" in s for s, _ in ai.sent)


def test_with_the_switch_off_the_refusal_stands(forum):
    payload, ran = _ask(forum, Recorded(), direct=False)
    assert not payload.get("direct_query")
    assert not any("users" in sql and "posts" in sql for sql in ran)


def test_a_query_that_breaks_a_rule_never_runs(forum):
    bad = Recorded("SELECT SUM(u.Reputation) AS rep FROM posts p JOIN users u ON p.OwnerUserId = u.Id")
    payload, ran = _ask(forum, bad)
    assert not payload.get("direct_query")
    assert not any("Reputation" in sql for sql in ran)


def test_a_query_that_leaves_out_a_part_of_the_request_is_not_shown(forum):
    # A part the planner listed that the query does not give ("with a score over 5" left out)...
    left_out = Recorded("SELECT 100.0 * SUM(CASE WHEN u.Age > 65 THEN 1 ELSE 0 END) / COUNT(*) AS share_of_posts "
                        "FROM posts p JOIN users u ON p.OwnerUserId = u.Id",
                        parts=[{"words": "share of the posts", "in": "share_of_posts"},
                               {"words": "owned by members over 65", "in": "u.Age > 65"}])
    assert not _ask(forum, left_out)[0].get("direct_query")
    # ...or one the AI says it gives and the query does not hold.
    claimed = Recorded(parts=[{"words": "share of the posts", "in": "share_of_posts"},
                              {"words": "owned by members over 65", "in": "u.Age > 65"},
                              {"words": "with a score over 5", "in": "the output column"}])
    assert not _ask(forum, claimed)[0].get("direct_query")


def test_a_direct_query_cannot_be_put_on_a_dashboard(forum):
    # A tile is drawn again by its plan, with no AI; a direct query has none, so no pin token is given and the
    # answer has no Add to dashboard (and so no schedule).
    from gateway.core2_bridge import _pin

    payload, _ = _ask(forum, Recorded(), sees_query=True)
    assert payload.get("direct_query") and payload["trust"]["sql"] and payload["data"] is not None
    assert _pin("acct-direct", {"id": 7}, QUESTION, payload) == ""
