"""
A validated plan is reused only under the release that made it.

A question's validated SQL is kept, and the same question asked again may be
answered by running it without planning it anew. Reuse was keyed on the
question, the tables, the database and the semantic contract, but not on the
code that wrote the SQL. On a made-up retailer's warehouse, "online sales in
Canada in 2025" had once been answered with every country's sales; the release
that stops a member being left out declined to compile it, and the pipeline
then reused the old plan and answered with every country's sales again. A fix
reached no question already asked, for as long as the contract stood still.

Every answer now records the release that planned it (core/release.py: the
code that plans an answer, read byte for byte), and a plan is reused only under
that release.

tests/star_harness.py keeps Canada as a sales territory.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("plan-release")) as built:
        yield built


def _total() -> float:
    return sum(line[7] * star.PRODUCTS[line[4]][4] for line in star.orders())


def _stored_plan_for(question: str, asked_as: str, release: str) -> None:
    """The plan the warehouse's total was answered with, kept as the plan of
    ``question`` by ``release`` -- what an earlier release that left the
    member out would have kept."""
    import store

    with store.get_db() as conn:
        row = conn.execute(
            "SELECT id, question_id FROM query_log WHERE account_id=? AND question=? ORDER BY id DESC LIMIT 1",
            (star.ACCOUNT, asked_as),
        ).fetchone()
        conn.execute("UPDATE query_log SET question=? WHERE id=?", (question, row["id"]))
        conn.execute("UPDATE answer_trace SET code_release=? WHERE account_id=? AND question_id=?",
                     (release, star.ACCOUNT, row["question_id"]))


class TestTheProductAnswers:

    def test_every_answer_records_its_release(self, warehouse):
        import store
        from core.release import code_release

        star.ask(warehouse, "Total sales amount")
        with store.get_db() as conn:
            row = conn.execute(
                "SELECT a.code_release FROM query_log q JOIN answer_trace a ON a.question_id=q.question_id "
                "WHERE q.account_id=? AND q.question=? ORDER BY q.id DESC LIMIT 1",
                (star.ACCOUNT, "Total sales amount"),
            ).fetchone()
        assert row["code_release"] == code_release()

    def test_a_plan_an_earlier_release_made_is_not_reused(self, warehouse):
        answer = star.ask(warehouse, "Total sales amount")
        assert [list(row.values()) for row in answer["rows"]] == [[_total()]]
        _stored_plan_for("Sales amount in Canada", "Total sales amount", "an-earlier-release")

        answer = star.ask(warehouse, "Sales amount in Canada")
        # Planned anew: Canada is a member, so the SQL writer is asked.
        assert answer["model_wrote_sql"] is True
        assert answer["rows"] == []

    def test_a_plan_this_release_made_is_reused(self, warehouse):
        from core.release import code_release

        star.ask(warehouse, "Total sales amount")
        _stored_plan_for("Sales amount in Canada", "Total sales amount", code_release())

        answer = star.ask(warehouse, "Sales amount in Canada")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[_total()]]


class TestTheRelease:

    @staticmethod
    def _tree(root):
        for folder, name, text in (("core", "planner.py", "PLAN = 1\n"), ("store", "keep.py", "KEEP = 1\n"),
                                   ("packs", "star.json", "{}"), ("portal", "page.html", "<p>")):
            (root / folder).mkdir(parents=True, exist_ok=True)
            (root / folder / name).write_text(text, encoding="utf-8")

    @pytest.mark.parametrize("changed", ["core/planner.py", "store/keep.py", "packs/star.json"])
    def test_the_code_that_plans_an_answer(self, tmp_path, changed):
        from core.release import release_of

        self._tree(tmp_path)
        before = release_of(tmp_path)
        assert release_of(tmp_path) == before
        (tmp_path / changed).write_text((tmp_path / changed).read_text(encoding="utf-8") + " ", encoding="utf-8")
        assert release_of(tmp_path) != before

    def test_a_page_is_not_the_code_that_plans(self, tmp_path):
        from core.release import release_of

        self._tree(tmp_path)
        before = release_of(tmp_path)
        (tmp_path / "portal" / "page.html").write_text("<p>changed", encoding="utf-8")
        assert release_of(tmp_path) == before

    def test_a_file_moved_is_another_release(self, tmp_path):
        from core.release import release_of

        self._tree(tmp_path)
        before = release_of(tmp_path)
        (tmp_path / "core" / "planner.py").rename(tmp_path / "core" / "planner2.py")
        assert release_of(tmp_path) != before
