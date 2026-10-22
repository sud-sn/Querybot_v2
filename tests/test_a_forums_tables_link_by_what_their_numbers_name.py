"""A forum's tables link by what their numbers name, and its questions are read as their words say.

Found on a public text-to-SQL benchmark's forum data, invented here:

* Every table numbers its own rows 1, 2, 3 under a plain "Id". Two such numbers hold the same values without one
  pointing at the other, and Learn linked the users to the votes by them (and the posts to the badges), so "who
  owns the post titled ..." was answered through a vote. A plain Id links nothing unless the database declares it.
* A post's history type ("PostHistoryTypeId", 1 to 30) was linked to the posts as a role, "history type post".
* A post's owner's name, kept again on the post and empty on nearly every one ("OwnerDisplayName"), was what
  "posts by Harvey" was read by, and found none. The owner's name is the user's, through the link.
* "Posts owned by an elder user": the age can be reached as the post's owner or its last editor, and the question
  says which. It was asked anyway.
* A title quoted with its question mark ('... is doing?') was filtered without it, and found no post.
* "User 23853 gave a comment to a post ..." was answered in French: "comment" is a French word too.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json
import random

import duckdb
import pytest

from core2.answer import beyond
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.ir import Plan
from core2.plan.values import build_index
from core2.service import Services, Session, _quoted_as_written, _reading_said, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)
TITLE = "Why does my model drift over time?"


def _forum(*, copy_filled: bool = False, badge: bool = False):
    rng = random.Random(11)
    con = duckdb.connect()
    con.execute("CREATE TABLE users (Id INTEGER, DisplayName VARCHAR, Age INTEGER, Reputation INTEGER)")
    con.execute("""CREATE TABLE posts (Id INTEGER, OwnerUserId INTEGER, LastEditorUserId INTEGER,
                   OwnerDisplayName VARCHAR, Title VARCHAR, Score INTEGER, ViewCount INTEGER, CreationDate DATE)""")
    if badge:
        con.execute("ALTER TABLE posts ADD COLUMN BadgeDisplayName VARCHAR")
    con.execute("CREATE TABLE votes (Id INTEGER, PostId INTEGER, UserId INTEGER, VoteTypeId INTEGER)")
    con.execute("CREATE TABLE postHistory (Id INTEGER, PostId INTEGER, PostHistoryTypeId INTEGER, UserId INTEGER)")
    first = ["Ada", "Bram", "Cleo", "Dov", "Esme", "Finn", "Gus", "Hana", "Ivo", "Juno", "Kai", "Lena"]
    last = ["Marsh", "Okafor", "Pryce", "Quill", "Rasmussen", "Sato", "Tamm", "Ueda", "Voss", "Wren"]
    users = [(i, f"{first[i % 12]} {last[i % 10]} {i}", rng.randint(18, 80), rng.randint(1, 5000))
             for i in range(1, 301)]
    con.executemany("INSERT INTO users VALUES (?,?,?,?)", users)
    names = {u[0]: u[1] for u in users}
    for i in range(1, 501):
        owner, editor = rng.randint(1, 300), rng.randint(1, 300)
        copy = names[owner] if copy_filled or rng.random() < 0.03 else None
        title = TITLE if i == 77 else f"How do I read question {i}?" if i % 2 else f"Notes on topic {i}"
        con.execute(f"INSERT INTO posts VALUES (?,?,?,?,?,?,?,?{',?' if badge else ''})",
                    [i, owner, editor, copy, title, rng.randint(-3, 40), rng.randint(5, 9000),
                     dt.date(2010, 1, 1) + dt.timedelta(days=i * 3)]
                    + ([f"Badge {i % 7}" if rng.random() < 0.03 else None] if badge else []))
    con.executemany("INSERT INTO votes VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 500), rng.randint(1, 300), rng.randint(1, 5)) for i in range(1, 701)])
    con.executemany("INSERT INTO postHistory VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 500), rng.randint(1, 30), rng.randint(1, 300)) for i in range(1, 801)])
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=[]), client_id="eval-forum",
                        options=BuildOptions(workers=1))

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'"{model.tables[column.table].name}"').rows]

    return model, warehouse, build_index(model, fetch)


@pytest.fixture(scope="module")
def forum():
    return _forum()


def _links(model):
    name = lambda key: f"{model.tables[model.columns[key].table].name}.{model.columns[key].name}"  # noqa: E731
    return {(name(j.from_columns[0]), name(j.to_columns[0])) for j in model.joins.values()}


def test_tables_numbered_by_a_plain_id_are_not_linked_by_it(forum):
    links = _links(forum[0])
    assert {("posts.OwnerUserId", "users.Id"), ("posts.LastEditorUserId", "users.Id"), ("votes.PostId", "posts.Id"),
            ("votes.UserId", "users.Id"), ("postHistory.PostId", "posts.Id")} <= links, links
    assert not any(a.endswith(".Id") and b.endswith(".Id") for a, b in links), links


def test_a_plain_id_repeated_on_its_table_is_the_number_it_holds():
    # A lab result's ID is its patient's: repeated on the results, one patient to many. Not the results' own row
    # number, so it links to the patients by it, as the tables' own row numbers never do.
    rng = random.Random(4)
    con = duckdb.connect()
    con.execute("CREATE TABLE Patient (ID INTEGER, SEX VARCHAR, Diagnosis VARCHAR)")
    con.execute("CREATE TABLE Laboratory (ID INTEGER, Date DATE, GOT INTEGER, GPT INTEGER)")
    con.execute("CREATE TABLE Ward (ID INTEGER, Name VARCHAR)")
    con.executemany("INSERT INTO Patient VALUES (?,?,?)",
                    [(1000 + i * 7, rng.choice("MF"), rng.choice(["SLE", "RA", "PSS"])) for i in range(400)])
    con.executemany("INSERT INTO Laboratory VALUES (?,?,?,?)",
                    [(1000 + rng.randint(0, 399) * 7, dt.date(1995, 1, 1) + dt.timedelta(days=i), rng.randint(5, 90),
                      rng.randint(5, 90)) for i in range(3000)])
    con.executemany("INSERT INTO Ward VALUES (?,?)", [(1000 + i * 7, f"Ward {i}") for i in range(40)])
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=[]), client_id="eval-lab",
                        options=BuildOptions(workers=1))
    links = _links(model)
    assert ("Laboratory.ID", "Patient.ID") in links, links
    assert ("Ward.ID", "Patient.ID") not in links and ("Patient.ID", "Ward.ID") not in links, links


def test_a_kind_named_after_a_table_is_no_link_to_it(forum):
    assert not any(a == "postHistory.PostHistoryTypeId" for a, _ in _links(forum[0]))


def test_a_name_kept_again_on_a_few_rows_is_read_from_what_it_names(forum):
    model = forum[0]
    columns = {model.columns[a.column].name for a in model.attributes.values()}
    assert "DisplayName" in columns and "OwnerDisplayName" not in columns
    # Kept on every post, the copy is a field of its own like any other.
    filled = _forum(copy_filled=True)[0]
    assert "OwnerDisplayName" in {filled.columns[a.column].name for a in filled.attributes.values()}
    # A name on a few rows beside no link of its own words (a post's badge, not its owner) is a field of its own.
    badged = _forum(badge=True)[0]
    kept = {badged.columns[a.column].name for a in badged.attributes.values()}
    assert "BadgeDisplayName" in kept and "OwnerDisplayName" not in kept, kept


class Recorded:
    def __init__(self, *plans):
        self.plans = [json.dumps({"kind": "query", **p}) for p in plans]
        self.sent: list[str] = []

    def __call__(self, stable, tail):
        self.sent.append(tail)
        return self.plans.pop(0)


def _slug(model, table, column):
    return next(a.slug for a in model.attributes.values()
                if model.columns[a.column].name == column and model.tables[model.columns[a.column].table].name == table)


def _measure(model, table):
    return next(m.slug for m in model.measures.values()
                if model.tables[m.table].name == table and m.slug.startswith("number_of"))


def test_the_link_the_question_names_is_followed_without_asking(forum):
    model, warehouse, index = forum
    assert _reading_said("posts owned by an elder user", ["Last editor user", "Owner user"]) == "Owner user"
    assert _reading_said("posts last edited by an elder user", ["Last editor user", "Owner user"]) == \
        "Last editor user"
    assert _reading_said("posts by an elder user", ["Last editor user", "Owner user"]) is None
    age = _slug(model, "users", "Age")
    plan = {"intent": "value", "measures": [_measure(model, "posts")],
            "filters": [{"field": age, "op": "gt", "values": [65]},
                        {"field": _slug(model, "posts", "Score"), "op": "gte", "values": [20]}]}
    ai = Recorded(plan, plan)
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY)
    payload = answer_question("Among the posts owned by an elder user (age over 65), how many have a score of 20 "
                              "or more?", services, Session())
    assert payload.get("kind") != "clarify" and payload.get("clarify") is None, payload.get("answer")
    hand = warehouse.query("SELECT COUNT(*) FROM posts p JOIN users u ON u.Id = p.OwnerUserId "
                           "WHERE u.Age > 65 AND p.Score >= 20").rows[0][0]
    assert round(float(payload["kpi"]["value"])) == hand


def test_a_title_quoted_with_its_question_mark_is_found(forum):
    model, warehouse, index = forum
    title = _slug(model, "posts", "Title")
    question = f"How many views has the post titled '{TITLE}'?"
    planned = Plan.model_validate({"kind": "query", "intent": "value", "measures": ["number_of_posts"],
                                   "filters": [{"field": title, "op": "eq", "values": [TITLE.rstrip("?")]}]})
    # The titles are read: the one stored is taken.
    assert title in index.attributes
    assert _quoted_as_written(planned, question, model, index).filters[0].values == [TITLE]
    # Not read (a free-text field): either way is matched.
    unread = index.only(set(index.attributes) - {title})
    kept = _quoted_as_written(planned, question, model, unread)
    assert kept.filters[0].op == "in" and set(kept.filters[0].values) == {TITLE, TITLE.rstrip("?")}
    # Unquoted, nothing changes.
    assert _quoted_as_written(planned, question.replace("'", ""), model, index) == planned
    ai = Recorded({"intent": "value", "measures": ["number_of_posts"],
                   "filters": [{"field": title, "op": "eq", "values": [TITLE.rstrip("?")]}]})
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY)
    payload = answer_question(f"How many posts are titled '{TITLE}'?", services, Session())
    assert payload.get("kpi") and round(float(payload["kpi"]["value"])) == 1, payload.get("answer")


@pytest.mark.parametrize("question, fr", [
    ("User No.23853 gave a comment to a post at 9:08:18 on 2013/7/12, was that post well-finished?", False),
    ("What is the comment count by month?", False),
    ("how do I share a dashboard?", False),
    ("Comment puis-je partager un tableau ?", True),
    ('Quelles sont les ventes de "The Corner Shop" ?', True),
    ("Combien de clients ont acheté du café ?", True),
    ("ventes par magasin", True),
    ("Combien de ventes pour The Corner Shop and Co ?", True),
    ("The total sales by store in 2025", False),
    ("WHAT IS THE COMMENT COUNT BY MONTH?", False),
])
def test_a_question_is_answered_in_the_language_it_is_written_in(question, fr):
    assert beyond.french(question) is fr
