"""
A question's own words name its measure and its period, never a member.

Three English questions their French twins answered went wrong, each on a
word:

  * "Units sold by month" -- the value index matched "sold" to a product
    group whose name begins with it, so the question was taken to name that
    group and left to the model. A registered measure's own words are never a
    member.
  * "Unités vendues au premier semestre 2022" -- with the measure's words out
    of the way, "premier" matched a product group called "... Premier". A
    word of the stated period's own phrase is the period's; "le mois le plus
    récent" offered "plus" and "récent" the same way, and French question and
    time words are never members either.
  * "How many units did we sell" matched no metric -- "sell" is not "sold" --
    and was asked which dataset it meant. A verb is one word in any tense.
    The value index read it the same way: the measure's words were kept out
    of its candidates as the metric spells them, so "sold" was and "sell" was
    not. Where a few party codes held the letters ("GRUSSELL", "NRUSSELL"),
    "sell" was taken to name a party the index could not find, and the
    governed compilers left the question to the model.

And "deliveries and physical inventory counts by month in 2022" was answered
with month-end inventory value, a metric it reached through the word "month":
a grain or a window says how the answer is cut, never which measure.

A synthetic tenant (tests/answer_harness.py) whose item groups include a
"SOLDER" and a "PREMIER FITTINGS" with no items; the warehouse and the model
are the only stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("words")) as built:
        yield built


@pytest.fixture
def codes_holding_sell(warehouse):
    """Four party codes the value index holds with "SELL" inside them, as a
    warehouse's own party list can, taken out again after the test: the
    tenant is shared by every module that asks it a question."""
    import sqlite3

    from core.value_index import _index_path, normalize_value

    codes = ("GRUSSELL", "SRUSSELL", "NRUSSELL", "PRUSSELLE")
    path = _index_path(harness.ACCOUNT)
    with sqlite3.connect(path) as conn:
        (table,) = conn.execute(
            "SELECT DISTINCT table_fqn FROM column_value WHERE column_name = 'PTY_CD'").fetchone()
        conn.executemany(
            "INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
            "VALUES (?, 'PTY_CD', 'Party Code', ?, ?)",
            [(table, code, normalize_value(code)) for code in codes])
    try:
        yield codes
    finally:
        with sqlite3.connect(path) as conn:
            conn.executemany("DELETE FROM column_value WHERE column_name = 'PTY_CD' AND value = ?",
                             [(code,) for code in codes])


@pytest.fixture
def members_named_with_a_measures_verb(warehouse):
    """Warehouses named with a registered measure's verb in another form --
    "Units sold", "Purchased quantity" -- as a warehouse's own list can name
    them, one of them with no other word but "group", taken out of the value
    index again after the test."""
    import sqlite3

    from core.value_index import _index_path, normalize_value

    names = ("SELLING FLOOR", "PURCHASING DEPOT", "BUYING GROUP")
    path = _index_path(harness.ACCOUNT)
    with sqlite3.connect(path) as conn:
        (table, business_name) = conn.execute(
            "SELECT DISTINCT table_fqn, business_name FROM column_value WHERE column_name = 'WHS_DSC'").fetchone()
        conn.executemany(
            "INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
            "VALUES (?, 'WHS_DSC', ?, ?, ?)",
            [(table, business_name, name, normalize_value(name)) for name in names])
    try:
        yield names
    finally:
        with sqlite3.connect(path) as conn:
            conn.executemany("DELETE FROM column_value WHERE column_name = 'WHS_DSC' AND value = ?",
                             [(name,) for name in names])


@pytest.fixture
def members_named_round_a_measures_verb(warehouse):
    """Warehouses whose names end with a registered measure's verb in another
    form, or hold it with words on both sides -- "Group purchasing", "North
    East Selling Floor" -- taken out of the value index again after the test."""
    import sqlite3

    from core.value_index import _index_path, normalize_value

    names = ("GROUP PURCHASING", "MAIN PURCHASING", "NORTH EAST SELLING FLOOR")
    path = _index_path(harness.ACCOUNT)
    with sqlite3.connect(path) as conn:
        (table, business_name) = conn.execute(
            "SELECT DISTINCT table_fqn, business_name FROM column_value WHERE column_name = 'WHS_DSC'").fetchone()
        conn.executemany(
            "INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
            "VALUES (?, 'WHS_DSC', ?, ?, ?)",
            [(table, business_name, name, normalize_value(name)) for name in names])
    try:
        yield names
    finally:
        with sqlite3.connect(path) as conn:
            conn.executemany("DELETE FROM column_value WHERE column_name = 'WHS_DSC' AND value = ?",
                             [(name,) for name in names])


@pytest.fixture
def selling_floor_north(warehouse, members_named_with_a_measures_verb):
    """A warehouse whose name holds another's ("Selling Floor North"), taken out
    of the value index again after the test."""
    import sqlite3

    from core.value_index import _index_path, normalize_value

    path = _index_path(harness.ACCOUNT)
    with sqlite3.connect(path) as conn:
        (table, business_name) = conn.execute(
            "SELECT DISTINCT table_fqn, business_name FROM column_value WHERE column_name = 'WHS_DSC'").fetchone()
        conn.execute("INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
                     "VALUES (?, 'WHS_DSC', ?, 'SELLING FLOOR NORTH', ?)",
                     (table, business_name, normalize_value("SELLING FLOOR NORTH")))
    try:
        yield "SELLING FLOOR NORTH"
    finally:
        with sqlite3.connect(path) as conn:
            conn.execute("DELETE FROM column_value WHERE column_name = 'WHS_DSC' AND value = 'SELLING FLOOR NORTH'")


def _sold(first: int, last: int, by_month: bool = False) -> dict:
    """Units sold per unit (and month) over yyyymm first..last, month rows only."""
    totals: dict = {}
    for _whs, item, period, sold, _bought, _receipts, _cost in harness.MOVES:
        if first <= period <= last:
            key = (period, harness.ITEMS[item][3]) if by_month else harness.ITEMS[item][3]
            totals[key] = totals.get(key, 0) + sold
    return totals


def _rows(answer: dict, measure: str) -> list[dict]:
    (answered,) = [run for run in answer["executed"] if f"AS {measure}" in run["sql"]]
    return answered["rows"]


class TestTheProductAnswersThem:

    def test_the_measures_word_is_not_a_product_group(self, warehouse):
        answer = harness.ask(warehouse, "Units sold by month in 2025")
        assert answer["model_wrote_sql"] is False
        assert {(int(str(row["PERIOD"])[:7].replace("-", "")), row["UNT_OF_MSR"]): row["UNITS_SOLD"]
                for row in _rows(answer, "UNITS_SOLD")} == pytest.approx(_sold(202501, 202512, by_month=True))

    def test_the_periods_word_is_not_a_product_group(self, warehouse):
        answer = harness.ask(warehouse, "Unités vendues au premier semestre 2025", "fr")
        assert answer["model_wrote_sql"] is False
        assert {row["UNT_OF_MSR"]: row["UNITS_SOLD"] for row in _rows(answer, "UNITS_SOLD")} == pytest.approx(
            _sold(202501, 202506))

    def test_did_we_sell_is_units_sold(self, warehouse):
        answer = harness.ask(warehouse, "How many units did we sell in 2025?")
        assert answer["model_wrote_sql"] is False
        assert {row["UNT_OF_MSR"]: row["UNITS_SOLD"] for row in _rows(answer, "UNITS_SOLD")} == pytest.approx(
            _sold(202501, 202512))

    def test_did_we_sell_where_party_codes_hold_the_letters(self, warehouse, codes_holding_sell):
        answer = harness.ask(warehouse, "How many units did we sell in 2025?")
        assert answer["model_wrote_sql"] is False
        assert {row["UNT_OF_MSR"]: row["UNITS_SOLD"] for row in _rows(answer, "UNITS_SOLD")} == pytest.approx(
            _sold(202501, 202512))


def _measure_forms() -> dict:
    """The measures' word forms as the pipeline passes them. Read at call time:
    on a tree without them each test fails on what the resolver does rather
    than on an import."""
    from core import value_resolver

    build = getattr(value_resolver, "build_measure_forms", None)
    return {"measure_forms": build(harness.ACCOUNT)} if build else {}


def _resolved(question: str) -> dict:
    """The question's members, read as the pipeline reads them."""
    from core.value_resolver import build_known_terms, build_vocabulary_words, resolve_literals

    return resolve_literals(
        harness.ACCOUNT, question, known_terms=build_known_terms(harness.ACCOUNT, None),
        vocabulary=build_vocabulary_words(harness.ACCOUNT, None), **_measure_forms())


class TestAMembersNameMayHoldTheVerb:
    """A measure's verb in another form is never a member on its own, but a
    member's name may hold it: "the selling floor", "the purchasing depot".
    Kept out of every run of words, as the measure's own words are, the member
    was lost and the answer covered every warehouse."""

    @pytest.mark.parametrize("question,member", [
        ("How many units did we sell on the selling floor?", "SELLING FLOOR"),
        ("Stock on hand at the selling floor", "SELLING FLOOR"),
        ("Units purchased by the purchasing depot", "PURCHASING DEPOT"),
        # Named with capitals, a name all of whose words are a measure's or the
        # product's own: taken as vocabulary, the warehouse was dropped.
        ("Stock on hand at Buying Group", "BUYING GROUP"),
        ("Units sold at Buying Group in 2025", "BUYING GROUP"),
        # Written as it is spoken: the name is a run of words the question
        # writes whole -- "group" is a known term, "only" is no part of the
        # name, "selling floor January" is longer than a name's own fuzzy
        # match reaches -- and the verb alone is looked up whole.
        ("Stock on hand at the buying group", "BUYING GROUP"),
        ("Buying group stock on hand", "BUYING GROUP"),
        ("Units sold at the buying group in 2025", "BUYING GROUP"),
        ("Stock on hand, selling floor only", "SELLING FLOOR"),
        ("What did the purchasing depot buy in 2025?", "PURCHASING DEPOT"),
        ("Units sold on the selling floor in January 2025", "SELLING FLOOR"),
        ("Units sold on the selling floor in September 2025", "SELLING FLOOR"),
        ("Units sold on the selling floor in 2025 vs 2024", "SELLING FLOOR"),
        ("Unités vendues au selling floor au premier semestre 2025", "SELLING FLOOR"),
    ])
    def test_the_member_is_named(self, warehouse, members_named_with_a_measures_verb, question, member):
        assert member in {item.get("value") for item in _resolved(question)["verified"]}

    def test_two_members_named_in_one_question_are_both_named(self, warehouse, members_named_with_a_measures_verb):
        named = {item.get("value") for item in _resolved("Compare the selling floor and the buying group")["verified"]}
        assert {"SELLING FLOOR", "BUYING GROUP"} <= named

    @pytest.mark.parametrize("question,named", [
        ("Stock on hand at the selling floor north", {"SELLING FLOOR NORTH"}),
        ("Stock on hand at the selling floor", {"SELLING FLOOR"}),
    ])
    def test_the_longer_name_is_the_one_the_question_writes(
            self, warehouse, members_named_with_a_measures_verb, question, named):
        import sqlite3

        from core.value_index import _index_path, normalize_value

        path = _index_path(harness.ACCOUNT)
        with sqlite3.connect(path) as conn:
            (table, business_name) = conn.execute(
                "SELECT DISTINCT table_fqn, business_name FROM column_value WHERE column_name = 'WHS_DSC'").fetchone()
            conn.execute("INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
                         "VALUES (?, 'WHS_DSC', ?, 'SELLING FLOOR NORTH', ?)",
                         (table, business_name, normalize_value("SELLING FLOOR NORTH")))
        try:
            assert {item.get("value") for item in _resolved(question)["verified"]} == named
        finally:
            with sqlite3.connect(path) as conn:
                conn.execute("DELETE FROM column_value WHERE column_name = 'WHS_DSC' AND value = 'SELLING FLOOR NORTH'")

    @pytest.mark.parametrize("question,named", [
        ("Stock on hand at the selling floor and the selling floor north",
         {"SELLING FLOOR", "SELLING FLOOR NORTH"}),
        ("Compare stock on hand at the selling floor and the selling floor north",
         {"SELLING FLOOR", "SELLING FLOOR NORTH"}),
        ("Stock on hand at the selling floor north and the selling floor",
         {"SELLING FLOOR", "SELLING FLOOR NORTH"}),
    ])
    def test_a_name_inside_another_is_named_too_where_the_question_writes_it_apart(
            self, warehouse, selling_floor_north, question, named):
        assert {item.get("value") for item in _resolved(question)["verified"]} == named

    def test_both_are_handed_on_where_the_question_names_both(self, warehouse, selling_floor_north):
        # Two members of one column are the SQL writer's to filter on: answered
        # for the longer name alone, the governed answer left the shorter out.
        answer = harness.ask(warehouse, "Stock on hand at the selling floor and the selling floor north")
        assert answer["model_wrote_sql"] is True
        prompt = answer["prompts"][0]
        assert "SELLING FLOOR NORTH" in prompt and "'SELLING FLOOR'" in prompt.replace("SELLING FLOOR NORTH", "")

    @pytest.mark.parametrize("question,member", [
        # Named whole by the words round the verb only.
        ("Stock on hand at the buying group", "BUYING GROUP"),
        # Named whole by a run of words of its own as well: the member is
        # found once, not once by each.
        ("Stock on hand at the selling floor", "SELLING FLOOR"),
    ])
    def test_a_name_two_columns_keep_is_named_in_both(
            self, warehouse, members_named_with_a_measures_verb, question, member):
        import sqlite3

        from core.value_index import _index_path, normalize_value

        path = _index_path(harness.ACCOUNT)
        with sqlite3.connect(path) as conn:
            (table, business_name) = conn.execute(
                "SELECT DISTINCT table_fqn, business_name FROM column_value WHERE column_name = 'ITM_STK_STS_DSC'"
            ).fetchone()
            conn.execute("INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
                         "VALUES (?, 'ITM_STK_STS_DSC', ?, ?, ?)", (table, business_name, member, normalize_value(member)))
        try:
            resolved = _resolved(question)
            assert [(item["value"], len(item["columns"])) for item in resolved["several"]] == [(member, 2)]
            assert member not in {item.get("value") for item in resolved["verified"]}
        finally:
            with sqlite3.connect(path) as conn:
                conn.execute("DELETE FROM column_value WHERE column_name = 'ITM_STK_STS_DSC' AND value = ?", (member,))

    def test_the_member_is_named_by_the_words_the_question_writes(
            self, warehouse, members_named_with_a_measures_verb):
        (item,) = [item for item in _resolved("Stock on hand at the buying group")["verified"]
                   if item.get("value") == "BUYING GROUP"]
        assert item["phrase"] == "buying group"

    @pytest.mark.parametrize("question,member", [
        ("Stock on hand at the buying group", "BUYING GROUP"),
        ("Units sold at the buying group in 2025", "BUYING GROUP"),
        ("Units sold on the selling floor in January 2025", "SELLING FLOOR"),
    ])
    def test_the_answer_is_filtered_on_the_member(
            self, warehouse, members_named_with_a_measures_verb, question, member):
        answer = harness.ask(warehouse, question)
        assert answer["model_wrote_sql"] is False
        assert any(member in executed["sql"] for executed in answer["executed"])

    @pytest.mark.parametrize("question", [
        "Stock on hand at the selling floor",
        # Beside the year the run of words is "selling floor 2025", and the
        # member is found only with the year left off.
        "How many units did we sell on the selling floor in 2025?",
    ])
    def test_the_answer_is_the_members(self, warehouse, members_named_with_a_measures_verb, question):
        answer = harness.ask(warehouse, question)
        assert answer["model_wrote_sql"] is False
        assert any("SELLING FLOOR" in executed["sql"] for executed in answer["executed"])


class TestAMemberWhoseNameEndsWithTheVerb:
    """A name may end with the measure's verb ("Group purchasing") or hold it
    with words on both sides ("North East Selling Floor"): the run of words the
    question writes is looked up whole, wherever the verb stands in it. Looked
    up only as a run that goes on past the verb, "Stock on hand at group
    purchasing" lost the member, and the answer covered every warehouse."""

    @pytest.mark.parametrize("question,member", [
        ("Stock on hand at group purchasing", "GROUP PURCHASING"),
        ("Units bought by group purchasing in 2025", "GROUP PURCHASING"),
        ("Stock on hand at main purchasing", "MAIN PURCHASING"),
        ("Stock on hand at the group purchasing warehouse", "GROUP PURCHASING"),
        ("Stock en main chez group purchasing", "GROUP PURCHASING"),
        # Words on both sides of the verb, four in all.
        ("Stock on hand at the north east selling floor", "NORTH EAST SELLING FLOOR"),
        ("Units sold at the north east selling floor in 2025", "NORTH EAST SELLING FLOOR"),
    ])
    def test_the_member_is_named(self, warehouse, members_named_round_a_measures_verb, question, member):
        assert member in {item.get("value") for item in _resolved(question)["verified"]}

    def test_the_answer_is_filtered_on_the_member(self, warehouse, members_named_round_a_measures_verb):
        answer = harness.ask(warehouse, "Stock on hand at group purchasing")
        assert answer["model_wrote_sql"] is False
        assert any("GROUP PURCHASING" in executed["sql"] for executed in answer["executed"])

    def test_a_name_of_more_words_than_a_run_holds_is_not_one(self, warehouse, members_named_round_a_measures_verb):
        # Five words: no run of four reaches it, and it is named by no other.
        found = {item.get("value") for item in _resolved("Stock on hand at the far north east selling floor")["verified"]}
        assert "NORTH EAST SELLING FLOOR" in found  # its own four words, the fifth left out
        assert "FAR NORTH EAST SELLING FLOOR" not in found


@pytest.fixture
def statuses_a_clause_reads_as(warehouse):
    """Item stock statuses the index holds that read as what a question says is
    done -- "Do not sell", "Available to sell" -- taken out of the value index
    again after the test."""
    import sqlite3

    from core.value_index import _index_path, normalize_value

    names = ("DO NOT SELL", "AVAILABLE TO SELL")
    path = _index_path(harness.ACCOUNT)
    with sqlite3.connect(path) as conn:
        (table, business_name) = conn.execute(
            "SELECT DISTINCT table_fqn, business_name FROM column_value WHERE column_name = 'ITM_STK_STS_DSC'"
        ).fetchone()
        conn.executemany(
            "INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
            "VALUES (?, 'ITM_STK_STS_DSC', ?, ?, ?)",
            [(table, business_name, name, normalize_value(name)) for name in names])
    try:
        yield names
    finally:
        with sqlite3.connect(path) as conn:
            conn.executemany("DELETE FROM column_value WHERE column_name = 'ITM_STK_STS_DSC' AND value = ?",
                             [(name,) for name in names])


def _found(resolved: dict) -> set:
    """Every value the resolver handed on, in whatever bucket."""
    values = set()
    for bucket in ("verified", "narrowed", "several"):
        values |= {item.get("value") for item in resolved.get(bucket) or []}
    for item in resolved.get("in_lists") or []:
        values |= set(item.get("values") or [])
    for item in resolved.get("clarify") or []:
        values |= {option.get("value") for option in item.get("options") or []}
    return values


class TestAClauseIsNoMember:
    """A run that ends with a measure's verb and holds the question's grammar
    before it -- "do not", "available to", "we" -- says what is done, not who
    or what it is done to, and is not looked up as a name: "What items do not
    sell at North Depot?" names North Depot, and no status DO NOT SELL."""

    @pytest.mark.parametrize("question", [
        "What items do not sell at North Depot?",
        "Which products do not sell well in winter?",
        "Items we do not sell anymore",
        "How many units are available to sell at North Depot?",
        "Which items don't sell at North Depot?",
        "Which items will we not sell at North Depot?",
    ])
    def test_a_status_written_as_a_clause_is_no_member(self, warehouse, statuses_a_clause_reads_as, question):
        assert not {"DO NOT SELL", "AVAILABLE TO SELL"} & _found(_resolved(question))

    def test_a_member_named_beside_the_clause_is_still_named(self, warehouse, statuses_a_clause_reads_as):
        assert "NORTH DEPOT" in _found(_resolved("What items do not sell at North Depot?"))

    @pytest.mark.parametrize("question,status", [
        ("Units whose status is Do Not Sell", "DO NOT SELL"),
        ("Units whose status is Available To Sell", "AVAILABLE TO SELL"),
    ])
    def test_a_status_the_question_writes_as_a_name_is_the_member(
            self, warehouse, statuses_a_clause_reads_as, question, status):
        assert status in _found(_resolved(question))


def _bare_index(tmp_path, *values) -> str:
    """A value index holding these values of one column, and nothing else."""
    import sqlite3

    from core.value_index import normalize_value

    folder = tmp_path / "acct"
    folder.mkdir()
    conn = sqlite3.connect(folder / "value_index.sqlite")
    conn.execute("CREATE TABLE column_value (table_fqn TEXT NOT NULL, column_name TEXT NOT NULL, "
                 "business_name TEXT NOT NULL DEFAULT '', value TEXT NOT NULL, value_norm TEXT NOT NULL)")
    conn.executemany("INSERT INTO column_value VALUES ('T', 'NAME', '', ?, ?)",
                     [(value, normalize_value(value)) for value in values])
    conn.commit()
    conn.close()
    return str(tmp_path)


class TestTheRunsOfWordsRoundAVerb:
    """What a run may hold, where it may start, and what the question is asked
    once for: the rules the synthetic tenant's questions do not reach, for a
    question's first four phrases are all the resolver reads and the verb is
    often beyond them."""

    @staticmethod
    def _around(tmp_path, question, *values, word="selling"):
        from core.value_resolver import _names_written_around

        base = _bare_index(tmp_path, *values)
        return [run for run, _ in _names_written_around("acct", question, word, None, base)]

    @pytest.mark.parametrize("name", [
        "GROUP SELLING", "MAIN GROUP SELLING", "WEST MAIN GROUP SELLING",
        "SELLING FLOOR", "SELLING FLOOR NORTH", "MAIN SELLING FLOOR", "NORTH EAST SELLING FLOOR",
        "GROUP SELLING FLOOR"])
    def test_a_run_starts_up_to_three_words_before_the_verb_and_holds_four(self, tmp_path, name):
        assert self._around(tmp_path, f"stock on hand at {name.lower()} today", name) == [name.lower()]

    def test_a_name_of_five_words_is_not_one(self, tmp_path):
        found = self._around(tmp_path, "stock on hand at far north east selling floor",
                             "FAR NORTH EAST SELLING FLOOR", "NORTH EAST SELLING FLOOR")
        assert found == ["north east selling floor"]

    def test_a_name_inside_another_at_another_start_is_no_member_of_its_own(self, tmp_path):
        # "buying group" is inside "central buying group east" and starts later and ends sooner.
        assert self._around(tmp_path, "stock on hand at central buying group east",
                            "CENTRAL BUYING GROUP EAST", "BUYING GROUP", word="buying") == ["central buying group east"]

    def test_a_name_written_apart_from_another_is_named_beside_it(self, tmp_path):
        found = self._around(tmp_path, "the selling floor and the selling floor north",
                             "SELLING FLOOR", "SELLING FLOOR NORTH")
        assert sorted(found) == ["selling floor", "selling floor north"]

    def test_grammar_inside_a_name_that_goes_on_past_the_verb_is_no_clause(self, tmp_path):
        # "to" before the verb says what is done only where the run ends with the verb.
        (tmp_path / "goes-on").mkdir()
        (tmp_path / "ends").mkdir()
        assert self._around(tmp_path / "goes-on", "stock on hand at point to selling floor",
                            "POINT TO SELLING FLOOR") == ["point to selling floor"]
        assert self._around(tmp_path / "ends", "stock on hand at point to selling", "POINT TO SELLING") == []

    @pytest.mark.parametrize("grammar", [
        "do", "does", "did", "don", "doesn", "didn", "t", "not", "to", "we", "you", "they", "i", "can",
        "cannot", "will", "would", "are", "is", "be", "been", "being"])
    def test_each_word_of_the_questions_grammar_makes_a_run_that_ends_with_the_verb_no_name(self, tmp_path, grammar):
        assert self._around(tmp_path, f"stock on hand at {grammar} selling", f"{grammar.upper()} SELLING") == []
        # ... whatever stands before the grammar, where the run still ends with the verb.
        (tmp_path / "named").mkdir()
        assert self._around(tmp_path / "named", f"stock on hand at main {grammar} selling",
                            f"MAIN {grammar.upper()} SELLING") == []

    def test_grammar_anywhere_before_the_verb_makes_a_run_that_ends_with_it_no_name(self, tmp_path):
        # Not only the word beside it: "we really sell" is what is done.
        assert self._around(tmp_path, "units we really sell", "WE REALLY SELL", word="sell") == []

    @pytest.mark.parametrize("name,question", [
        ("NO SELL", "stock on hand of no sell items"),
        ("NO SELL", "units in no sell status by warehouse"),
        ("NO BUYING", "stock on hand of no buying items"),
        ("NO PURCHASING", "units in no purchasing status"),
    ])
    def test_a_status_that_starts_with_no_is_a_name(self, tmp_path, name, question):
        assert self._around(tmp_path, question, name, word=name.split()[-1].lower()) == [name.lower()]

    def test_no_longer_is_no_status(self, tmp_path):
        assert self._around(tmp_path, "which items do we no longer sell", "NO SELL", word="sell") == []

    def test_an_article_is_no_grammar(self, tmp_path):
        assert self._around(tmp_path, "stock on hand at the selling", "THE SELLING") == ["the selling"]

    @pytest.mark.parametrize("word", [
        "a", "an", "on", "for", "in", "at", "by", "or", "and", "of", "with", "from", "as", "ok", "pre", "best", "top", "all",
        "new", "old", "main", "north", "east", "nothing"])
    def test_any_other_word_before_the_verb_is_part_of_a_name(self, tmp_path, word):
        """The grammar set is narrow: the words round a verb that say what is done, and no other."""
        assert self._around(tmp_path, f"stock on hand at {word} selling", f"{word.upper()} SELLING") == [f"{word} selling"]

    def test_a_name_inside_another_that_ends_where_it_ends_is_no_member_of_its_own(self, tmp_path):
        assert self._around(tmp_path, "stock on hand at north east selling floor",
                            "NORTH EAST SELLING FLOOR", "EAST SELLING FLOOR") == ["north east selling floor"]

    def test_a_name_written_twice_is_named_once(self, tmp_path):
        assert self._around(tmp_path, "the selling floor versus the selling floor again",
                            "SELLING FLOOR") == ["selling floor"]

    def test_no_text_is_asked_of_the_index_twice(self, tmp_path, monkeypatch):
        from core import value_resolver

        base = _bare_index(tmp_path, "SELLING FLOOR", "NORTH DEPOT")
        asked = []
        real = value_resolver.lookup_exact

        def counted(account_id, text, allowed_tables=None, base_dir="clients"):
            asked.append(text)
            return real(account_id, text, allowed_tables, base_dir=base_dir)

        monkeypatch.setattr(value_resolver, "lookup_exact", counted)
        resolved = value_resolver.resolve_literals(
            "acct", "How many units did we sell and selling at the selling floor, sold in the selling floor?",
            base_dir=base, known_terms={"units", "sold"}, vocabulary={"units"}, measure_forms={"sell", "selling"})
        assert asked and len(asked) == len(set(asked))
        assert {item["value"] for item in resolved["verified"]} == {"SELLING FLOOR"}

    def test_nor_the_same_text_in_another_case(self, tmp_path, monkeypatch):
        from core import value_resolver

        base = _bare_index(tmp_path, "SELLING FLOOR", "NORTH DEPOT")
        asked = []
        real = value_resolver.lookup_exact

        def counted(account_id, text, allowed_tables=None, base_dir="clients"):
            asked.append(text)
            return real(account_id, text, allowed_tables, base_dir=base_dir)

        monkeypatch.setattr(value_resolver, "lookup_exact", counted)
        resolved = value_resolver.resolve_literals(
            "acct", "Selling floor stock on hand, as at the selling floor", base_dir=base,
            known_terms={"stock", "hand"}, vocabulary={"stock"}, measure_forms={"sell", "selling"})
        assert asked and len(asked) == len({text.casefold() for text in asked})
        assert {item["value"] for item in resolved["verified"]} == {"SELLING FLOOR"}


class TestAMemberCalledByAMeasuresWord:
    """A location called PURCHASING, beside the metric "Units purchased": the
    word "purchasing" is another form of the measure's, and is looked up whole
    and never by its letters. A member called so is still found."""

    @pytest.fixture
    def purchasing(self, warehouse):
        import sqlite3

        from core.value_index import _index_path, normalize_value

        path = _index_path(harness.ACCOUNT)
        with sqlite3.connect(path) as conn:
            (table, business_name) = conn.execute(
                "SELECT DISTINCT table_fqn, business_name FROM column_value WHERE column_name = 'WHS_DSC'").fetchone()
            conn.execute(
                "INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
                "VALUES (?, 'WHS_DSC', ?, 'PURCHASING', ?)", (table, business_name, normalize_value("PURCHASING")))
        try:
            yield
        finally:
            with sqlite3.connect(path) as conn:
                conn.execute("DELETE FROM column_value WHERE column_name = 'WHS_DSC' AND value = 'PURCHASING'")

    @pytest.mark.parametrize("question", [
        "Stock on hand at Purchasing", "Stock on hand in PURCHASING", "Stock on hand where warehouse is Purchasing",
        "How much stock does Purchasing hold?", "Stock on hand at Main and Purchasing"])
    def test_written_as_a_name_it_is_the_member(self, purchasing, question):
        assert "PURCHASING" in {item.get("value") for item in _resolved(question)["verified"]}

    def test_a_name_is_found_whole_or_not_at_all(self, codes_holding_sell):
        resolved = _resolved("How Many Units Did We Sell in 2025?")
        found = {str(item.get("value")) for bucket in ("verified", "narrowed", "in_lists", "clarify")
                 for item in resolved.get(bucket) or []}
        assert not found & set(codes_holding_sell)


class TestAHeadingInCapitals:
    """A question is often typed as a report is titled. Read as names, the
    measure words of "Total Available Quantity by Warehouse" filtered by a
    stock status AVAILABLE -- half the rows, and not a word said -- and those
    of "Units Sold by Warehouse" by an item group SOLD. A measure's own words
    are never a member, whatever their capitals."""

    _MEMBERS = (("ITM_STK_STS_DSC", "AVAILABLE"), ("ITM_STK_STS_DSC", "STOCK"), ("ITM_STK_STS_DSC", "SOLD"),
                ("ITM_STK_STS_DSC", "RECEIPTS"), ("WHS_DSC", "MAIN"), ("WHS_DSC", "MAIN STREET"),
                ("WHS_DSC", "PURCHASING"))

    @pytest.fixture
    def members(self, warehouse):
        import sqlite3

        from core.value_index import _index_path, normalize_value

        path = _index_path(harness.ACCOUNT)
        with sqlite3.connect(path) as conn:
            for column, value in self._MEMBERS:
                (table, business_name) = conn.execute(
                    "SELECT DISTINCT table_fqn, business_name FROM column_value WHERE column_name = ?",
                    (column,)).fetchone()
                conn.execute(
                    "INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
                    "VALUES (?, ?, ?, ?, ?)", (table, column, business_name, value, normalize_value(value)))
        try:
            yield
        finally:
            with sqlite3.connect(path) as conn:
                for column, value in self._MEMBERS:
                    conn.execute("DELETE FROM column_value WHERE column_name = ? AND value = ?", (column, value))

    @pytest.mark.parametrize("question", [
        "Total Available Quantity by Warehouse", "TOTAL AVAILABLE QUANTITY BY WAREHOUSE",
        "What is our Stock on Hand?", "Total Stock On Hand by Warehouse", "Units Sold by Warehouse",
        "How many Units Sold in 2025?", "STOCK EN MAIN PAR ENTREPOT", "Stock en Main par Entrepôt",
        "Units In Stock by Warehouse", "UNITS IN STOCK BY WAREHOUSE", "Units in Stock by Warehouse",
        "What are our total Units In Stock?", "Quantity In Stock by Warehouse",
        "What is the inventory value for Stock?", "What is the inventory value for the Stock?",
        "What is the total for Stock?", "Show me the monthly trend for Stock",
        "WHAT IS THE INVENTORY VALUE FOR STOCK?", "What is the value of total Stock?",
        "What is the value of all our Stock?", "Show the change in total Stock by month",
        "How many units are in our Stock?", "Quelle est la valeur pour le Stock ?",
        "Stock on hand in North Main"])
    def test_its_measure_words_are_no_member(self, members, question):
        resolved = _resolved(question)
        found = {str(item.get("value")) for bucket in ("verified", "several", "narrowed", "in_lists", "clarify")
                 for item in resolved.get(bucket) or []}
        assert not found & {value for _, value in self._MEMBERS}

    @pytest.mark.parametrize("question,named", [
        ("Stock on hand in Main Street", {"MAIN STREET"}),
        ("Stock on hand at the Main Street warehouse", {"MAIN STREET"}),
        ("Stock on hand at North Depot and Purchasing", {"NORTH DEPOT", "PURCHASING"}),
    ])
    def test_a_member_a_question_names_is_still_found(self, members, question, named):
        assert {item.get("value") for item in _resolved(question)["verified"]} >= named


class TestWhatIsNeverAMember:

    @staticmethod
    def _candidates(question: str) -> set[str]:
        from core.value_resolver import build_known_terms, build_vocabulary_words, extract_candidate_phrases

        return {phrase.lower() for phrase in extract_candidate_phrases(
            question, build_known_terms(harness.ACCOUNT, None), build_vocabulary_words(harness.ACCOUNT, None),
            **_measure_forms())}

    def test_a_registered_measures_words(self, warehouse):
        found = self._candidates("Units sold by month in 2025")
        assert not {"units", "sold"} & found and not any("sold" in phrase for phrase in found)

    @pytest.mark.parametrize("question", [
        "How many units did we sell in 2025?", "units we are selling", "units the branch sells"])
    def test_a_registered_measures_verb_in_any_tense(self, warehouse, codes_holding_sell, question):
        # Looked up whole and never by its letters, so no party whose code
        # holds them is a member; part of a member's name it may be (the
        # selling floor, TestAMembersNameMayHoldTheVerb).
        from core.value_resolver import build_known_terms, build_vocabulary_words, extract_candidate_phrases

        whole_only: list[str] = []
        extract_candidate_phrases(
            question, build_known_terms(harness.ACCOUNT, None), build_vocabulary_words(harness.ACCOUNT, None),
            **_measure_forms(), exact_only=whole_only)
        assert {"sell", "selling", "sells"} & {word.lower() for word in whole_only}
        assert not any(_resolved(question).values())

    @pytest.mark.parametrize("question", ["Stock le plus récent", "Ventes du mois le plus récent"])
    def test_french_time_words(self, warehouse, question):
        assert not any(set(phrase.split()) & {"plus", "récent", "mois"} for phrase in self._candidates(question))

    def test_a_member_is_still_one(self, warehouse):
        assert "brass elbow" in self._candidates("Units sold for BRASS ELBOW in 2025")


class TestTheStatedPeriodsWords:

    @staticmethod
    def _kept(phrases: list[str], question: str, reader_question: str = "") -> list[str]:
        from core.query_pipeline import _without_the_stated_period

        resolved = {"verified": [{"phrase": phrase} for phrase in phrases]}
        return [item["phrase"] for item in _without_the_stated_period(
            resolved, question, reader_question)["verified"]]

    def test_a_word_of_the_period_phrase_is_the_periods(self):
        assert self._kept(["premier", "Premier Fittings"], "unites vendues to premier semestre 2025",
                          "Unités vendues au premier semestre 2025") == ["Premier Fittings"]

    def test_read_on_the_readers_own_words_too(self):
        # "mars" is the reader's; the canonical question says "March".
        assert self._kept(["mars"], "unites vendues en March 2025", "Unités vendues en mars 2025") == []

    def test_a_question_with_no_stated_period_keeps_them(self):
        assert self._kept(["premier"], "units sold in the last 3 months") == ["premier"]


_METRICS = [
    {"name": "Units sold", "synonyms": "units sold, quantity sold", "base_table": "MART.PRD_FCT",
     "formula_type": "expression", "sql_template": "SUM(SLD_QTY)"},
    {"name": "Purchased quantity", "synonyms": "units purchased, quantity purchased", "base_table": "MART.PRD_FCT",
     "formula_type": "expression", "sql_template": "SUM(PCH_QTY)"},
    {"name": "Inventory value", "synonyms": "inventory value, stock value", "base_table": "MART.DLY_FCT",
     "formula_type": "expression", "sql_template": "SUM(ON_HND_QTY * ITM_CST)"},
    {"name": "Month-end inventory value", "synonyms": "month-end inventory value, month end inventory",
     "base_table": "MART.PRD_FCT", "formula_type": "expression", "sql_template": "SUM(CUR_ON_HND_QTY * ITM_CST)"},
    {"name": "Month-to-date revenue", "synonyms": "revenue this month", "base_table": "MART.SLS_FCT",
     "formula_type": "expression", "sql_template": "SUM(NET_AMT)"},
    {"name": "Revenue", "synonyms": "revenue, sales amount", "base_table": "MART.SLS_FCT",
     "formula_type": "expression", "sql_template": "SUM(NET_AMT)"},
]


def _scoped(question: str) -> list[str]:
    from core.metric_scope import resolve_metric_scope

    return [metric["name"] for metric in resolve_metric_scope(_METRICS, question, None).metrics]


class TestWhichMetricTheQuestionNames:

    @pytest.mark.parametrize("question", [
        "How many units did we sell in 2025?", "units we are selling", "units the store sells"])
    def test_a_verb_in_any_tense(self, question):
        assert _scoped(question) == ["Units sold"]

    def test_bought_is_purchased(self):
        assert _scoped("units bought last quarter") == ["Purchased quantity"]

    def test_a_grain_is_not_evidence(self):
        # "month" cut the answer by month; it did not name month-end value.
        assert set(_scoped("deliveries and physical inventory counts by month in 2022")) == {
            "Inventory value", "Month-end inventory value"}

    def test_a_window_is_not_evidence(self):
        assert set(_scoped("inventory counts last month")) == {"Inventory value", "Month-end inventory value"}

    def test_a_metric_named_for_its_period_still_is(self):
        assert _scoped("what is the month-end inventory value?")[0] == "Month-end inventory value"

    def test_a_phrase_authored_with_its_window_still_matches(self):
        assert _scoped("revenue this month")[0] == "Month-to-date revenue"

    def test_the_prompts_candidate_list_reads_them_the_same_way(self):
        from store.config_store import _score_metric_for_question

        units_sold, _bought, _inventory, month_end, _mtd, _revenue = _METRICS
        assert _score_metric_for_question(units_sold, "How many units did we sell in 2025?") > (
            _score_metric_for_question(units_sold, "How many units in 2025?"))
        assert _score_metric_for_question(month_end, "physical inventory counts by month in 2022") == (
            _score_metric_for_question(month_end, "physical inventory counts in 2022"))
