"""
A ranking is read from the end the question asks for.

"Which warehouse has the lowest stock on hand?" was compiled with the same
ORDER BY ... DESC as "highest", and its answer card, which always led with the
largest row, named the warehouse with the most. "Bottom 2 items" could not be
compiled at all -- the validator refused the descending order it was given --
and went to the model. And "which item has the most stock on hand?" was not a
ranking: "most" is not "highest", so the answer was an unordered breakdown.

Now a superlative after "which <thing> has" is a ranking in English and French
("quel entrepôt a le moins de stock"), the compiler sorts from the end the
question asks for -- an explicit Top/Bottom-N's direction, else its words --
and the card names the lowest when the lowest was asked for. "At least 100"
and "the most recent month" are not rankings.

A synthetic tenant (tests/answer_harness.py); the warehouse and the model are
the only stand-ins.
"""

from __future__ import annotations

import re

import pytest

from core import analytical_intent, i18n
from core.analytical_intent import plan_analytical_intent
from core.question_normalizer import canonical_question
from tests import answer_harness as harness


def _without_clause_openers(question: str) -> str:
    """The same words with none that opens a clause, so the verb is all that says
    it is a change: "Rank items whose stock went ..." is "Rank items' stock went
    ...", "Which items went ..." is "Items went ..."."""
    question = re.sub(r"\s+whose\s+", "' ", question)
    return re.sub(r"^Which\s+", "", question)


# Read at call time, so each test fails on a tree without them rather than the
# module failing to import.
def asks_for_ranking(question: str) -> bool:
    return analytical_intent.asks_for_ranking(question)


def ranking_direction(question: str) -> str:
    return analytical_intent.ranking_direction(question)


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("rankings")) as built:
        yield built


class TestWhichEndIsAskedFor:

    @pytest.mark.parametrize("question,direction", [
        ("Which item has the most stock on hand?", "descending"),
        ("Which warehouse has the least stock on hand?", "ascending"),
        ("Which warehouse has the lowest stock on hand?", "ascending"),
        ("what item has the fewest units", "ascending"),
        ("Bottom 2 items by stock on hand", "ascending"),
        ("which item group has the highest inventory value", "descending"),
        ("Top 5 warehouses with at least 100 units", "descending"),
    ])
    def test_in_english(self, question, direction):
        assert asks_for_ranking(question) and ranking_direction(question) == direction
        assert plan_analytical_intent(question).intent == "ranking"

    @pytest.mark.parametrize("question,direction", [
        ("Sales by region from highest to lowest", "descending"),
        ("Stock by unit of measure, largest to smallest", "descending"),
        ("Sales by region from lowest to highest", "ascending"),
        ("Ventes par région du plus grand au plus petit", "descending"),
        ("Ventes par région du plus petit au plus grand", "ascending"),
        ("Ventes par région du plus bas au plus élevé", "ascending"),
        ("Ventes par région de la plus grande à la plus petite", "descending"),
        ("Ventes par région de la plus élevée à la plus basse", "descending"),
        ("Sales by region from low to high", "ascending"),
        ("Sales by region in ascending order", "ascending"),
        ("Ventes par région par ordre croissant", "ascending"),
        ("Ventes par région par ordre décroissant", "descending"),
        # The end asked for decides which rows are read, whatever order they
        # are listed in.
        ("Which warehouses have the lowest stock value, listed highest to lowest?", "ascending"),
    ])
    def test_an_order_written_out_reads_from_the_end_it_names_first(self, question, direction):
        """Read for its "lowest", "from highest to lowest" was headed by its
        lowest row."""
        assert ranking_direction(question) == direction

    @pytest.mark.parametrize("question,direction", [
        ("Show the top 10 customers by revenue, from highest to lowest", "descending"),
        ("Which warehouse has the highest stock, listed from highest to lowest?", "descending"),
        ("Items with the most stock, from most to least", "descending"),
        ("Stock by warehouse from highest to lowest as of the most recent snapshot", "descending"),
    ])
    def test_the_end_it_names_first_is_the_ranking_s_own_end_too(self, question, direction):
        """A top N or a "most" says the end the order starts from, so it does
        not turn the order round: read for its "lowest", "top 10 customers,
        from highest to lowest" was headed by its lowest row."""
        assert ranking_direction(question) == direction

    @pytest.mark.parametrize("question,direction", [
        ("Bottom 3 warehouses by stock, from highest to lowest", "ascending"),
        ("Which warehouses have the lowest stock value, listed highest to lowest?", "ascending"),
        ("Sales by region for our top 10 products, from lowest to highest", "ascending"),
        ("Rank warehouses by inventory value from lowest to highest for our top suppliers", "ascending"),
        ("Stock of our best sellers by warehouse, lowest to highest", "ascending"),
        ("Stock by item against maximum stock level, lowest to highest", "ascending"),
    ])
    def test_the_opposite_end_named_elsewhere_leaves_the_question_read_as_before(self, question, direction):
        """Where the rest of the question names the opposite end of a ranking,
        which end the order is read from is left as it always was (the low end
        wherever a low end is named), not turned around."""
        assert ranking_direction(question) == direction

    @pytest.mark.parametrize("question,direction", [
        # A low word that is a condition is no ranking's end: the question
        # asks for no ranking, and its order is the one written.
        ("Stock by warehouse from high to low with a minimum of 10 units", "descending"),
        ("Stock by warehouse in descending order for the least expensive items", "descending"),
        ("Stock by warehouse largest to smallest, for items at their minimum level", "descending"),
        ("Stock by warehouse most to least, for our fewest-order customers", "descending"),
        ("Valeur du stock par entrepôt par ordre décroissant pour les articles les moins vendus", "descending"),
        ("Valeur du stock par entrepôt du plus élevé au plus bas pour les articles les moins chers", "descending"),
    ])
    def test_a_low_word_in_a_condition_does_not_turn_an_order_round(self, question, direction):
        """Read as main reads them, "the least expensive items" made "from
        high to low" a ranking from the lowest end: the card was headed by
        its lowest row."""
        assert not asks_for_ranking(question)
        assert ranking_direction(question) == direction

    @pytest.mark.parametrize("question", [
        "Rank warehouses by inventory value from low season to high season",
        "Rank warehouses by inventory value, top shelf to bottom shelf",
        "Rank items by sales of items that went from low to high demand",
        "Rank items by sales of transfers from low stock to high stock sites",
        "Rank customers who upgraded from low to high tier by revenue",
        "Rank items by sales from low cost to high cost suppliers",
        "Rank suppliers by spend on items that moved from low to high priority",
        "Rank items in ascending demand by stock",
        "Which items moved from low to high stock this year?",
        "Sales for the low-to-high price band by region",
        "Stock by warehouse from low to high value items",
        "Show items in an ascending trend by warehouse",
    ])
    def test_ends_that_describe_something_else_are_no_order(self, question):
        """"From low season to high season" and "in ascending demand" name
        two ends and an ascent, and no order the rows are listed in: read as
        one, they turned the ranking round."""
        assert not analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == ("ascending" if analytical_intent._LOW_END_RE.search(question)
                                               else "descending")

    @pytest.mark.parametrize("question", [
        # A change the rows went through, closed by a word that would close an order.
        "Which items went from low to high in 2025?",
        "Rank warehouses whose stock went from low to high during the year",
        "Rank warehouses whose stock moved from low to high over the quarter by sales",
        "Rank warehouses whose prices went from low to high since January",
        "Rank warehouses whose stock swung from low to high and back",
        "Rank warehouses whose stock climbed from low to high between Q1 and Q2",
        "Rank warehouses whose stock jumped from low to high after the promotion",
        "Rank warehouses for items that went from low to high",
        "Rank warehouses whose stock went from low to high for three months",
        "Rank warehouses whose stock rose from low to high at the end of the year",
        "Rank warehouses whose inventory value went from low to high in 2025",
        "Rank warehouses whose stock fell from high to low in 2025",
        "Which items changed from high to low in 2025?",
    ])
    def test_a_change_the_rows_went_through_is_no_order(self, question):
        """"Went from low to high in 2025" says what the stock did, not the
        order the rows are listed in: read as one, the ranking was turned
        round and its card headed by its lowest row."""
        for text in (question, _without_clause_openers(question)):
            assert not analytical_intent.writes_an_order(text), text
            assert ranking_direction(text) == ("ascending" if analytical_intent._LOW_END_RE.search(text)
                                               else "descending")

    @pytest.mark.parametrize("verb", """
        went gone go goes going came coming comes got gotten move moved moves moving rose rise rises rising risen
        fell fall falls falling fallen jump jumped jumps jumping climb climbed climbs climbing drop dropped drops
        dropping swung swing swings swinging grew grow grows growing grown shift shifted shifts shifting change
        changed changes changing increase increased increases increasing decrease decreased decreases decreasing
        slid slide slides sliding dip dipped dips dipping surged surges surging spiked spikes spiking turn turned
        turns turning switch switched switches switching upgraded downgraded upgrade upgrades upgrading downgrade
        downgrades downgrading come get gets getting soar soared soars soaring plunged
        plunges plunging decline declined declines declining improved improves improving worsened worsens
        worsening recovered recovers recovering rebounded rebounds rebounding doubled doubles doubling tripled
        triples tripling halved halves halving varied varies varying ranged ranges ranging trended trends trending
        evolved evolves evolving ramped ramps ramping escalated escalates escalating flipped flips flipping
        promoted promotes promoting demoted demotes demoting crept creep creeps creeping sank sunk sinks sinking
        slipped slips slipping shot shoots shooting leapt leaped leap leaps leaping surge spike plunge improve
        worsen recover rebound double triple halve vary range trend evolve ramp escalate flip promote demote
        sink slip shoot""".split())
    def test_a_verb_of_movement_in_any_form_makes_two_plain_ends_no_order(self, verb):
        # Right before the ends, where any -ed or -ing word is read as a change too; and with a phrase between, where
        # only the verb itself says so.
        for question in (f"Inventory value {verb} from low to high in 2025", f"Inventory value {verb} in 2025 from low to high"):
            assert not analytical_intent.writes_an_order(question), question
            assert ranking_direction(question) == ("ascending" if analytical_intent._LOW_END_RE.search(question)
                                                   else "descending")

    @pytest.mark.parametrize("between,is_an_order", [
        ("", False), ("up ", False), ("up sharply ", False), ("up sharply again ", False),
        ("up sharply in Q1 ", False), ("steadily all the way ", False), ("up and down ", False),
        ("up (sharply) ", False), ("up - sharply - ", False), ("up 50% ", False), ("quarter-on-quarter ", False),
        # A semicolon, a colon, "!", "?" or the full stop that ends a sentence ends the sentence the verb is in.
        ("up; ", True), ("up: ", True), ("up. ", True), ("up! ", True), ("up? ", True),
        # A comma does not: the change goes on after it. Nor does a full stop or a colon inside a figure or an abbreviation.
        ("up, ", False), ("up, in 2025, ", False), ("up sharply, quarter on quarter, ", False), ("up 2.5% ", False),
        ("up approx. 10% ", False), ("up vs. last year ", False), ("up 1,000 units ", False), ("up from 10:00 ", False),
        ("up e.g. in Q1 ", False),
        # A colon ends the sentence unless it sits between two digits: after a figure, or before one with a word in front.
        ("up 10: ", True), ("up:10 ", True)])
    def test_a_verb_of_movement_holds_the_ends_anywhere_before_them_in_its_sentence(self, between, is_an_order):
        question = f"Rank warehouses' stock went {between}from low to high"
        assert analytical_intent.writes_an_order(question) is is_an_order

    @pytest.mark.parametrize("abbreviation", """
        vs Vs VS approx Approx etc incl excl dept Dept inc Inc co Co ltd Ltd corp Corp st St ST mt Mt ft cf avg est qty
        wk yr mr Mr mrs ms dr Dr jan Jan feb mar apr jun jul aug sep Sep sept Sept oct nov dec Dec""".split())
    def test_the_full_stop_of_an_abbreviation_ends_no_sentence(self, abbreviation):
        """A capital after it is no new sentence: "rose approx. EUR 100", "went up vs. Q1", "at St. Louis" still
        describe the change the ends follow."""
        question = f"Rank warehouses whose inventory value rose {abbreviation}. Q1 from low to high"
        for text in (question, _without_clause_openers(question)):
            assert not analytical_intent.writes_an_order(text), text

    @pytest.mark.parametrize("question", [
        "Which warehouses' inventory value went up in the U.S. Midwest from low to high?",
        "Which items' stock rose for J. Smith from low to high?",
        "Which items rose in Plant B. Depot from low to high?",
        "Which items rose e.g. North Depot from low to high?",
        "Rank items that grew in the U.K. Wire division from low to high",
        "Which items rose approx. EUR 100 from low to high?",
        "Which items rose per Dept. Finance from low to high?",
        "Which items' stock went up at St. Louis from low to high?",
        "Which items rose vs. Last Year from low to high?",
    ])
    def test_the_full_stop_of_an_initial_or_an_abbreviation_before_a_name_ends_no_sentence(self, question):
        assert not analytical_intent.writes_an_order(question)

    @pytest.mark.parametrize("question", [
        "Which items went up in March. Show stock from low to high",
        "Which items went up in 2025. Show stock from low to high",
        # A word that ends in an abbreviation's letters, or begins with them, is none.
        "Which items went up in cost. Show stock from low to high",
        "Which items went up in the east. Show stock from low to high",
        "Which items went up at the post. Show stock from low to high",
        "Which items went up in Stock. Show stock from low to high",
        "Which items went up in the market. Show stock from low to high",
        "Which items went up this decade. Show stock from low to high",
        "Which items went up. Show stock from low to high",
        "Which items went up.  Show stock from low to high",
        "Which items went up.\nShow stock from low to high",
    ])
    def test_a_full_stop_that_ends_a_sentence_leaves_the_ends_to_the_next(self, question):
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "ascending"

    @pytest.mark.parametrize("stop", [";", "!", "?", ":"])
    def test_only_a_full_stop_is_the_full_stop_of_an_abbreviation_or_an_initial(self, stop):
        assert analytical_intent.writes_an_order(f"Which items rose in plan B{stop} Show stock from low to high")
        assert analytical_intent.writes_an_order(f"Which items rose vs{stop} Show stock from low to high")

    def test_a_full_stop_with_no_space_after_it_ends_no_sentence(self):
        """A version, a file name or a missing space: nothing says the change is over."""
        assert not analytical_intent.writes_an_order("Which items went up in Q1.Show stock from low to high")

    def test_a_full_stop_before_a_lower_case_word_ends_no_sentence(self):
        """Typed in lower case, a new sentence cannot be told from "approx. ten units" or "inc. ltd"."""
        assert not analytical_intent.writes_an_order("Which items went up in Q1. show stock from low to high")

    @pytest.mark.parametrize("question", [
        # The sentence the ends are in is the last, whatever the ones before it say.
        "Show items. Which went up. Sort them from low to high",
        "Which items went up! Which fell? Show stock from low to high",
        "Show items; which went up. Show stock from low to high",
        "Which items went up. Which fell. Which moved. Show stock from low to high",
    ])
    def test_the_sentence_the_ends_are_in_is_the_last_one(self, question):
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "ascending"

    def test_a_full_stop_after_a_digit_ends_a_sentence(self):
        """"Region 5." ends in a figure, not in a letter that could be an initial."""
        assert analytical_intent.writes_an_order("Which items went up in region 5. Show stock from low to high")

    @pytest.mark.parametrize("question", [
        # Any other past tense right before the ends, or before a particle or an adverb of manner.
        "Rank warehouses whose stock transitioned from low to high in 2025",
        "Rank items whose demand progressed from low to high in 2025",
        "Which items' prices fluctuated from low to high?",
        "Rank items whose priority was raised from low to high in 2025",
        "Rank items whose priority was bumped from low to high in 2025",
        "Rank items whose priority was reclassified from low to high in 2025",
        "Rank items whose stock ticked up from low to high in 2025",
        "Rank items whose stock edged up sharply from low to high in 2025",
        "Rank items whose stock picked back up from low to high in 2025",
    ])
    def test_any_other_past_tense_before_the_ends_is_a_change_too(self, question):
        for text in (question, _without_clause_openers(question)):
            assert not analytical_intent.writes_an_order(text), text
            assert ranking_direction(text) == "descending"

    @pytest.mark.parametrize("word", [
        "that", "which", "whose", "who", "whom", "where", "when", "what", "That", "WHICH",
        "quel", "quels", "quelle", "quelles", "dont", "qui", "que", "lesquels", "lesquelles", "o\u00f9"])
    def test_a_clause_a_relative_or_interrogative_word_opens_describes_a_change(self, word):
        """The clause has a verb of its own, and the ends say what it did: whatever the verb is -- the list of
        verbs of movement is closed, the words that open a clause are not."""
        question = f"Show items {word} we hold from low to high"
        assert not analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "descending"

    @pytest.mark.parametrize("question", [
        "Show equipment stock from low to high", "Show the queue length from low to high",
        "Show whereabouts from low to high", "Show somewhat late items from low to high",
        "Show the quiet warehouses from low to high", "Show queued items from low to high",
        "Show thatched roofs from low to high", "Show whomever we hold from low to high",
        "Show whichever we hold from low to high"])
    def test_a_word_that_only_holds_a_clause_openers_letters_opens_none(self, question):
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "ascending"

    def test_the_same_question_with_no_such_word_is_an_order(self):
        assert analytical_intent.writes_an_order("Show items we hold from low to high")
        assert ranking_direction("Show items we hold from low to high") == "ascending"

    @pytest.mark.parametrize("question", [
        # A present tense, an irregular past or an -ed past far from the ends, where the verb is none of the listed.
        "Which items transition from low to high?", "Which items' prices fluctuate from low to high?",
        "Which items' demand accelerates from low to high?", "Which items' risk levels are reclassified from low to high?",
        "Which items' sales take off from low to high?", "Which items' stock builds from low to high?",
        "Rank items whose priority was set from low to high in 2025", "Which items' ratings sprang from low to high?",
        "Which items' prices sped up from low to high?", "Which items' priority was put from low to high?",
        "Which items' priority was raised in 2025 from low to high?",
        "Rank items whose tier migrated in March from low to high",
        "Which items' prices rocketed in Q1 from low to high?",
        "Which items' demand accelerated very sharply from low to high?",
        "Which items' ratings were revised upward from low to high?",
        "Which items' ratings were bumped all the way from low to high?",
        # A break character inside the change.
        "Which items rose, in 2025, from low to high?", "Which items' prices went up 2.5% from low to high?",
        "Which items' prices rose approx. 10% from low to high?", "Which items rose vs. last year from low to high?",
        "Which items' stock went up 1,000 units from low to high?", "Which items rose from 10:00 from low to high?",
        "Rank warehouses whose stock went, in 2025, from low to high",
        # Top and bottom.
        "Which items fell, in 2025, from top to bottom?", "Which items dropped by a lot from top to bottom?",
        "Which items' priority was set from top to bottom?",
        # French typed, English ends.
        "Quels articles sont pass\u00e9s de low \u00e0 high en 2025 ?", "Classez les articles dont le stock a augment\u00e9 from low to high",
        "Quels articles ont grimp\u00e9 from low to high ?", "Articles dont le prix est mont\u00e9 from low to high",
        "Quels articles ont chang\u00e9 de rang from low to high ?",
    ])
    def test_a_change_in_any_verb_is_no_order(self, question):
        assert not analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == ("ascending" if analytical_intent._LOW_END_RE.search(question)
                                               else "descending")

    @pytest.mark.parametrize("question", [
        # A participle right before the ends, with or without a particle or an adverb of manner.
        "Show stock progressing from low to high", "Show stock picking up from low to high",
        "Show stock ticking up sharply from low to high", "Rank items transitioning from low to high",
        "Show prices fluctuating from low to high in 2025", "Show demand accelerating steadily from low to high",
        "Show stock raised from low to high", "Show stock picked back up from low to high",
    ])
    def test_a_participle_or_a_past_right_before_the_ends_is_a_change(self, question):
        assert not analytical_intent.writes_an_order(question)

    @pytest.mark.parametrize("particle", """
        up down back over off out again around fast hard sharply steadily gradually quickly slowly rapidly dramatically
        significantly substantially considerably strongly markedly massively drastically abruptly suddenly swiftly
        smoothly slightly modestly moderately aggressively notably""".split())
    def test_each_particle_after_a_past_tense_is_part_of_the_change(self, particle):
        assert not analytical_intent.writes_an_order(f"Show stock ticked {particle} from low to high")

    def test_any_number_of_particles_after_a_past_tense_is_part_of_the_change(self):
        assert not analytical_intent.writes_an_order("Show stock ticked up sharply again slowly back down from low to high")

    @pytest.mark.parametrize("question", [
        "Show inventory value by warehouse with stock accelerating fast from low to high",
        "Show inventory value by warehouse, stock ticked up rapidly from low to high",
        "Show stock ticked up significantly from low to high",
        "Show stock progressing dramatically from low to high",
    ])
    def test_an_adverb_of_manner_after_a_past_tense_or_a_participle_is_part_of_the_change(self, question):
        assert not analytical_intent.writes_an_order(question)

    @pytest.mark.parametrize("word", "daily weekly monthly quarterly yearly annually hourly early only".split())
    def test_an_adverb_of_time_or_degree_after_a_past_tense_is_none_of_the_change(self, word):
        question = f"Show items shipped {word} from low to high"
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "ascending"

    @pytest.mark.parametrize("word", """
        used aged fled sped bled tied died lied owed iced bring sting cling fling sling wring being doing using
        """.split())
    def test_a_word_of_four_or_five_letters_ending_like_a_participle_is_read_as_one(self, word):
        assert not analytical_intent.writes_an_order(f"Show stock {word} from low to high")

    @pytest.mark.parametrize("word", "red bed wed fed led ring king wing sing ping".split())
    def test_a_shorter_word_ending_like_a_participle_is_none(self, word):
        question = f"Show the {word} from low to high"
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "ascending"

    @pytest.mark.parametrize("question", [
        "Rank items whose stock went up sharply in Q1 from low to high",
        "Rank items whose demand climbed steadily all the way from low to high in 2025",
        "Rank items whose demand moved up and down from low to high in 2025",
        "Rank items whose stock went up in the first quarter from low to high",
        "Rank items whose stock rose again and again from low to high",
        "Rank items whose stock fell from high to low and then from low to high in 2025",
        "Rank items whose stock went up (sharply) from low to high",
        "Rank items whose stock went up - sharply - from low to high",
        "Rank items whose stock went up\u2014sharply\u2014from low to high",
        "Rank items whose stock went quarter-on-quarter from low to high",
        "Rank items whose stock went up 50% from low to high",
        "Rank items whose stock rose year-over-year from low to high",
        "Show every warehouse in the northern region of the whole country, including the main depots, "
        "whose stock went from low to high",
    ])
    def test_a_change_described_at_length_is_no_order(self, question):
        for text in (question, _without_clause_openers(question)):
            assert not analytical_intent.writes_an_order(text), text

    @pytest.mark.parametrize("word", ["sorted", "ordered", "ranked", "listed", "arranged", "presented", "displayed", "shown"])
    def test_each_word_that_sorts_right_before_the_ends_makes_them_an_order(self, word):
        question = f"Which items changed {word} from low to high"
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "ascending"

    @pytest.mark.parametrize("question", [
        # A sort word that is not right before the ends says nothing of them.
        "Sorted items that went from low to high in 2025",
        "Show items that changed sorted up from low to high",
        "Which items did we list that went from low to high",
        # "Orders" here is a noun, not a sort.
        "Show items that went up in orders from low to high",
        # A word that holds one of them is none.
        "Show items that changed unsorted from low to high", "Which items moved misordered from low to high",
        "Show items that changed unranked from low to high", "Which items went up relisted from low to high",
        "Show items that changed reshown from low to high",
    ])
    def test_a_word_that_sorts_elsewhere_makes_no_order_of_a_change(self, question):
        assert not analytical_intent.writes_an_order(question)

    @pytest.mark.parametrize("question,direction", [
        ("Show items that changed and sort them from low to high", "ascending"),
        ("Show items that changed then sort from low to high", "ascending"),
        ("Show items that changed also rank them from low to high", "ascending"),
        ("Show items that changed, please sort them from low to high", "ascending"),
        ("Show items that changed and rank them from high to low", "descending"),
        ("Show items that changed then rank from low to high", "ascending"),
        ("Show items that changed then list from low to high", "ascending"),
        ("Show items that changed then arrange from low to high", "ascending"),
        ("Show items that changed then order them from low to high", "ascending"),
        ("Show items that changed and sort from high to low", "descending"),
        ("Show items that changed AND SORT them from low to high", "ascending"),
        ("Show items that changed Then Rank them from high to low", "descending"),
    ])
    def test_a_sort_asked_for_after_a_change_word_makes_them_an_order(self, question, direction):
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == direction

    @pytest.mark.parametrize("question", [
        "Show items that changed and sort them all from low to high",
        "Show items that changed and sort it from low to high",
        "Show items that changed and sort these from low to high",
        "Show items that changed and sort those from low to high",
        "Show items that changed and sort all of them from low to high",
        "Show items that changed and sort the results from low to high",
        "Show items that changed and rank the result from low to high",
        "Show items that changed and rank the rows from low to high",
        "Show items that changed then list the output from low to high",
        "Show items that changed then arrange the table from low to high",
        "Show items that changed and order the data from low to high",
        "Show items that changed and sort the list from low to high",
        "Show items that changed and sort results from low to high",
        "Show items that changed and rank data from low to high",
        "Show items that changed and list output from low to high",
        "Show items that changed and sort them, from low to high",
        "Show items that changed, then sort the results , from low to high",
        "Please list items reclassified and sort them from low to high",
    ])
    def test_a_sort_of_the_rows_asked_for_right_before_the_ends_makes_them_an_order(self, question):
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "ascending"

    @pytest.mark.parametrize("question", [
        # A noun that begins as a sort verb does, after "and", sorts nothing.
        "Rank warehouses that grew inventory value and order volume from low to high",
        "Which warehouses grew inventory value and order count from low to high?",
        "Which warehouses increased inventory value and order volume from low to high?",
        "Show items that changed in price and list price from low to high",
        "Show items that changed and rank score from low to high",
        "Show items that changed and sort key from low to high",
        "Show items that changed and arrange fee from low to high",
        # "and" and "then" are words: a noun that ends in them does not ask for a sort.
        "Show items that changed in demand order from low to high",
        "Which items changed their brand rank from low to high?",
        # A sort asked for, but not right before the ends.
        "Show items that changed and sort them by name, as they grew from low to high",
        "Show items that changed and sort them as they moved from low to high",
        "Show items that changed and list items that grew from low to high",
        # A sort verb before a participle that says what changed.
        "Please list items reclassified from low to high",
        "List items reclassified from low to high",
        "Please list inventory value by warehouse reclassified from low to high",
        "Then list items upgraded from low to high",
        "Also rank warehouses migrated from low to high",
        "Please sort and list items reclassified from low to high",
    ])
    def test_a_sort_asked_for_anywhere_but_right_before_the_ends_makes_no_order_of_a_change(self, question):
        assert not analytical_intent.writes_an_order(question)

    @pytest.mark.parametrize("question", [
        # A key says nothing: the words that describe a change are read as what they say.
        "Rank warehouses that grew in 2025 by sales from low to high",
        "Rank items that changed by stock value from low to high",
        "Sort items by price change from low to high",
        "Rank items by increase from low to high",
        "Show stock changes by warehouse from low to high",
        "Rank warehouses whose inventory value grew by month from low to high",
        "Which warehouses' stock grew by month from low to high?",
        "Which items moved by quarter from low to high?",
        "Which items' demand shifted by season from low to high?",
        "Which items' prices went up by a lot from low to high?",
        "Which items were upgraded by the buyer from low to high?",
        # A sort word that is a noun or a participle sorts nothing: only a sort asked for, "and sort them", does.
        "Which items climbed the ranking from low to high?",
        "Which items moved up the list from low to high?",
        "Which items changed ranking from low to high?",
        "Which items went up in rank from low to high?",
        "Which items changed order priority from low to high?",
        "Which items changed their list price from low to high?",
        "Which customers increased order volume from low to high?",
        "Which items moved up the order queue from low to high?",
        "Which items dropped in the listing from high to low?",
        "Which items went down the sort order from high to low?",
        "Show items that changed the sorting from low to high",
        "Show items that changed a ranking from low to high",
        "Show items that changed sorting from low to high", "Show items that changed in order from low to high",
        "Which items rose in order from low to high?",
    ])
    def test_a_key_or_a_noun_that_sorts_makes_no_order_of_a_change(self, question):
        assert not analytical_intent.writes_an_order(question)

    @pytest.mark.parametrize("question", [
        # A sort asked for before the last change word says nothing of what it describes.
        "Show items that changed and sort them as they fell from low to high",
        "Rank items that went up and then fell from low to high",
        # Nor does a noun that begins as a sort verb does.
        "Show items that went up and lists from low to high", "Show items that changed and orders from low to high",
        "Show warehouses that grew and ranks from low to high", "Show items that changed and sorts from low to high",
        "Show items that moved and arranges from low to high", "Show items that changed and sorting from low to high",
        "Show items that changed and ranking from low to high", "Show items that changed and listing from low to high",
        "Show items that changed and ordering from low to high",
    ])
    def test_only_a_sort_after_the_last_change_word_makes_an_order(self, question):
        assert not analytical_intent.writes_an_order(question)

    @pytest.mark.parametrize("question", [
        # "By" before a change is no key: the change is what is said.
        "Rank regions by sales that climbed from low to high between Q1 and Q2",
        "Rank warehouses by stock that went from low to high for three months",
        "Rank items by demand which rose from low to high in 2025",
    ])
    def test_by_before_a_change_is_no_key(self, question):
        assert not analytical_intent.writes_an_order(question)

    @pytest.mark.parametrize("question,direction", [
        # A verb is a word: nothing is read inside another word, in any case.
        ("Sort returns from low to high", "ascending"),
        ("Rank the snapshot from low to high", "ascending"),
        ("Show items shipped a year ago from low to high", "ascending"),
        ("Sort the selected items from low to high", "ascending"),
        ("Show the last hundred from low to high", "ascending"),
        ("Show speed from low to high", "ascending"),
        ("Show the need from low to high", "ascending"),
        ("Show the stock value from low to high", "ascending"),
        ("List the price from low to high", "ascending"),
        ("Sort the changeovers from low to high", "ascending"),
        ("Sort the dropouts from low to high", "ascending"),
        ("Sort the goods from low to high", "ascending"),
        ("Rank items by speed from low to high", "ascending"),
    ])
    def test_a_word_that_only_holds_a_verb_is_none(self, question, direction):
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == direction

    def test_the_verb_is_read_in_any_case(self):
        assert not analytical_intent.writes_an_order("Rank Items' Stock Went From Low To High")
        assert not analytical_intent.writes_an_order("RANK ITEMS' STOCK TRANSITIONED FROM LOW TO HIGH")
        assert not analytical_intent.writes_an_order("Rank Items Whose Stock Rose From Low To High")
        assert not analytical_intent.writes_an_order("RANK ITEMS WHOSE STOCK TRANSITIONED FROM LOW TO HIGH")
        assert not analytical_intent.writes_an_order("SHOW STOCK PROGRESSING FROM LOW TO HIGH")

    @pytest.mark.parametrize("question,direction", [
        # A word that sorts, between the verb and the ends, says what they are.
        ("Show items that changed sorted from low to high", "ascending"),
        ("Show items that moved ranked from high to low", "descending"),
        ("Which items went up listed from low to high", "ascending"),
        ("Items that rose, ranked from low to high", "ascending"),
        ("Which items went up, then sorted from low to high", "ascending"),
        ("Show items that changed, SORTED from low to high", "ascending"),
        ("Show items that moved Ranked from high to low", "descending"),
    ])
    def test_a_word_that_sorts_makes_them_an_order_again(self, question, direction):
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == direction

    @pytest.mark.parametrize("question", [
        "Stock by warehouse that changed from highest to lowest",
        "Which items went from best to worst revenue in 2025",
        "Warehouses that moved from highest to lowest stock"])
    def test_superlative_ends_are_an_order_after_a_verb_of_movement_too(self, question):
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "descending"

    @pytest.mark.parametrize("question,direction", [
        ("Quels articles ont augment\u00e9 du plus bas au plus \u00e9lev\u00e9 ?", "ascending"),
        ("Quels articles dont le stock a augment\u00e9 du plus bas au plus \u00e9lev\u00e9 en 2025", "ascending"),
        ("Quels articles ont chang\u00e9 du plus \u00e9lev\u00e9 au plus bas ?", "descending")])
    def test_a_french_change_with_superlative_ends_is_an_order_as_typed_and_as_canonicalised(self, question, direction):
        """"Du plus bas au plus \u00e9lev\u00e9" is a superlative end, an order wherever it stands: only plain ends are a
        change after a verb of movement."""
        for text in (question, canonical_question(question, "fr")):
            assert analytical_intent.writes_an_order(text), text
            assert ranking_direction(text) == direction

    def test_an_order_later_in_the_question_is_read_after_a_change_that_is_not_one(self):
        question = "Items that went from low to high in 2025; rank them from high to low"
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "descending"

    @pytest.mark.parametrize("question,direction", [
        # An order still, where no verb of movement stands just before it.
        ("Which items changed? List them from low to high", "ascending"),
        ("Items that moved last quarter, sorted from low to high", "ascending"),
        ("Stock went down; show warehouses from low to high", "ascending"),
        ("Stock went down! Show warehouses from low to high", "ascending"),
        ("Stock went down. Show warehouses from low to high", "ascending"),
        ("Show warehouses by stock, from low to high", "ascending"),
        ("Sales by region from high to low in 2025", "descending"),
    ])
    def test_an_order_after_other_words_is_still_read(self, question, direction):
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == direction

    @pytest.mark.parametrize("question,direction", [
        ("Sales by region from highest to lowest revenue", "descending"),
        ("Sales by region from best to worst customers", "descending"),
        ("Sales by region from lowest to highest revenue", "ascending"),
        ("Sales by region from low to high, 2025", "ascending"),
        ("Sales by region from high to low (2025)", "descending"),
        ("Sales by region from high to low - 2025", "descending"),
        ("Sales by region from low to high for 2025 products", "ascending"),
        ("Sales by region from high to low as of the latest snapshot", "descending"),
        ("Sales by region, low to high", "ascending"),
        # Where the phrase stands twice, the order it writes is read.
        ("Sales by region from low season to high season, from highest to lowest", "descending"),
    ])
    def test_an_order_is_read_where_it_closes_or_its_ends_are_superlatives(self, question, direction):
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == direction

    @pytest.mark.parametrize("word", [
        "for", "in", "by", "with", "among", "amongst", "on", "at", "per", "and", "then", "where", "that", "whose",
        "of", "as", "since", "until", "during", "over", "across", "within", "between", "against", "based", "using",
        "after", "before", "from", "pour", "par", "avec", "dans", "sur", "et", "puis", "de", "des", "du", "en",
        "selon", "depuis", "entre", "chez", "lors"])
    def test_two_plain_ends_are_an_order_before_a_word_that_opens_a_clause(self, word):
        question = f"Sales by region from low to high {word} the year"
        assert analytical_intent.writes_an_order(question)
        assert ranking_direction(question) == "ascending"

    @pytest.mark.parametrize("question", [
        "Which warehouses have increasing stock?", "Show warehouses with increasing stock value",
        "Quels articles ont une demande croissante ?", "Quels entrepôts ont un stock croissant ?",
        "Which store sells the most croissant?"])
    def test_a_trend_is_no_order(self, question):
        """"Increasing" and "croissant" say a trend, not an order: read as
        one, a question about growing stock was headed by its lowest row --
        and canonicalised, "une demande croissante" is "a demand ascending"."""
        for text in (question, canonical_question(question, "fr")):
            assert ranking_direction(text) == "descending", text
            assert not analytical_intent.writes_an_order(text), text

    @pytest.mark.parametrize("question,direction", [
        ("Sales by region, highest-to-lowest", "descending"),
        ("Sales by region from highest sales to lowest", "descending"),
        ("Stock by warehouse from best to worst", "descending"),
        ("Stock by warehouse from top to bottom", "descending"),
        ("Stock by warehouse in increasing order", "ascending"),
        ("Stock by warehouse in decreasing order", "descending"),
        ("Ventes par région de la plus forte à la plus faible", "descending"),
        ("Ventes par région du moins élevé au plus élevé", "ascending"),
    ])
    def test_an_order_in_other_words(self, question, direction):
        assert ranking_direction(question) == direction

    @pytest.mark.parametrize("question,direction", [
        ("Valeur du stock par entrepôt du plus bas au plus élevé", "ascending"),
        ("Valeur du stock par entrepôt du plus élevé au plus bas", "descending"),
        ("Ventes par région de la plus grande à la plus petite", "descending"),
        ("Ventes par région de la plus forte à la plus faible", "descending"),
        ("Ventes par région de la plus élevée à la plus basse", "descending"),
        ("Ventes par région des plus grandes aux plus petites", "descending"),
        ("Ventes par région du moins élevé au plus élevé", "ascending"),
    ])
    def test_a_french_order_as_the_compiler_reads_it(self, question, direction):
        """The compiler reads the canonical question: "du plus bas au plus
        élevé" arrives as "of plus bas to highest", and was sorted highest
        first."""
        for text in (question, canonical_question(question, "fr")):
            assert ranking_direction(text) == direction, text

    @pytest.mark.parametrize("question,direction", [
        ("Quel article a le plus de stock en main ?", "descending"),
        ("Quel entrepôt a le moins de stock en main ?", "ascending"),
        ("Quel entrepôt a le stock le plus bas ?", "ascending"),
    ])
    def test_in_french_as_typed_and_as_canonicalised(self, question, direction):
        for text in (question, canonical_question(question, "fr")):
            assert asks_for_ranking(text) and ranking_direction(text) == direction

    @pytest.mark.parametrize("question", [
        "which warehouses have at least 100 units",
        "units sold in the most recent month",
        "at most 5 items per order",
        "Quel est le stock le plus récent ?",
    ])
    def test_what_is_not_a_ranking(self, question):
        assert not asks_for_ranking(question)
        assert not asks_for_ranking(canonical_question(question, "fr"))


def _answered(answer: dict) -> dict:
    (run,) = [run for run in answer["executed"] if "AS STOCK_ON_HAND" in run["sql"]]
    return run


def _stock_by(position: int) -> dict:
    """Stock on hand at the newest snapshot per item or warehouse name."""
    totals: dict = {}
    for whs, item, _buyer, _created, on_hand, _allocated, _cost in harness.STOCK:
        name = harness.ITEMS[item][1] if position == 1 else harness.WAREHOUSES[whs][1]
        totals[name] = totals.get(name, 0) + on_hand
    return totals


class TestTheProductSortsFromThatEnd:

    @pytest.mark.parametrize("question,lang", [
        ("Which warehouse has the lowest stock on hand?", "en"),
        ("Quel entrepôt a le moins de stock en main ?", "fr"),
    ])
    def test_the_lowest_first(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        run = _answered(answer)
        assert "STOCK_ON_HAND ASC" in run["sql"]
        values = [row["STOCK_ON_HAND"] for row in run["rows"]]
        assert values == sorted(values)

    def test_the_most_first(self, warehouse):
        answer = harness.ask(warehouse, "Which item has the most stock on hand?")
        assert answer["model_wrote_sql"] is False
        run = _answered(answer)
        assert "STOCK_ON_HAND DESC" in run["sql"]
        assert run["rows"][0]["ITEM"] == max(_stock_by(1).items(), key=lambda pair: pair[1])[0]

    def test_a_bottom_n_is_compiled(self, warehouse):
        answer = harness.ask(warehouse, "Bottom 2 items by stock on hand")
        assert answer["model_wrote_sql"] is False
        assert [row["ITEM"] for row in _answered(answer)["rows"]] == [
            name for name, _ in sorted(_stock_by(1).items(), key=lambda pair: pair[1])[:2]]

    def test_its_card_names_the_lowest(self, warehouse):
        answer = harness.ask(warehouse, "Bottom 1 item by stock on hand")
        (card,) = [payload["answer"] for kind, payload in answer["replies"]
                   if isinstance(payload, dict) and payload.get("answer")]
        lowest = min(_stock_by(1).items(), key=lambda pair: pair[1])[0]
        assert card["headline"].startswith("Lowest-ranked result: " + lowest)
        assert "lowest row" in card["comparison"]


class TestTheCardNamesTheEndAskedFor:

    ROWS = [{"WAREHOUSE": "NORTH DEPOT", "STOCK": 900}, {"WAREHOUSE": "SOUTH DEPOT", "STOCK": 350},
            {"WAREHOUSE": "EAST DEPOT", "STOCK": 120}]

    @staticmethod
    def _answer(question: str, lang: str = "en") -> dict:
        from core.response_builder import build_answer, infer_result_scope

        token = i18n.activate_language(lang)
        try:
            rows = TestTheCardNamesTheEndAskedFor.ROWS
            return build_answer(rows, question, infer_result_scope(rows, question, mode="ranking"))
        finally:
            i18n.deactivate_language(token)

    def test_the_lowest(self):
        card = self._answer("Which warehouse has the lowest stock?")
        assert card["headline"] == "EAST DEPOT is lowest at 120."
        assert card["comparison"] == "230 below the next result"

    def test_the_lowest_listed_highest_first(self):
        # "NORTH DEPOT leads at 900." answered it.
        card = self._answer("Which warehouses have the lowest stock value, listed highest to lowest?")
        assert card["headline"] == "EAST DEPOT is lowest at 120."

    def test_the_lowest_in_french(self):
        card = self._answer("Quel entrepôt a le moins de stock ?", "fr")
        assert card["headline"].startswith("EAST DEPOT est le plus bas")

    @pytest.mark.parametrize("question", [
        # The article and the adjective agree with their noun, and the
        # superlative can come late: the query read these smallest first
        # while the card named the largest row as the leader.
        "Quelle catégorie de produits a eu les ventes les plus faibles en 2025 ?",
        "Quels entrepôts ont les stocks les plus bas ?",
        "Quel entrepôt a la valeur la plus basse ?",
        "Quels entrepôts ont les stocks les plus petits ?",
    ])
    def test_the_lowest_in_french_whatever_the_agreement(self, question):
        card = self._answer(question, "fr")
        assert card["headline"].startswith("EAST DEPOT est le plus bas")

    def test_the_highest_in_french_is_unchanged(self):
        card = self._answer("Quels entrepôts ont les stocks les plus élevés ?", "fr")
        assert card["headline"].startswith("NORTH DEPOT arrive en tête")

    def test_the_highest_is_unchanged(self):
        card = self._answer("Which warehouse has the highest stock?")
        assert card["headline"] == "NORTH DEPOT leads at 900."
        assert card["comparison"] == "550 above the next result"

    @pytest.mark.parametrize("question,lang,headline", [
        # An order written out with no ranking word is still the order read:
        # both were headed by the largest row.
        ("Stock by warehouse from low to high", "en", "EAST DEPOT is lowest at 120."),
        ("Stock par entrepôt du plus bas au plus élevé", "fr", "EAST DEPOT est le plus bas"),
        ("Stock par entrepôt par ordre croissant", "fr", "EAST DEPOT est le plus bas"),
        ("Stock by warehouse from highest to lowest", "en", "NORTH DEPOT leads at 900."),
        ("Stock par entrepôt du plus élevé au plus bas", "fr", "NORTH DEPOT arrive en tête"),
        # A low word in a condition is no ranking's end.
        ("Stock by warehouse from high to low with a minimum of 10 units", "en", "NORTH DEPOT leads at 900."),
        ("Stock by warehouse in descending order for the least expensive items", "en",
         "NORTH DEPOT leads at 900."),
        ("Valeur du stock par entrepôt par ordre décroissant pour les articles les moins vendus", "fr",
         "NORTH DEPOT arrive en tête"),
        # Two ends that describe something else are no order.
        ("Stock by warehouse from low to high value items", "en", "NORTH DEPOT leads at 900."),
        ("Show warehouses in an ascending trend", "en", "NORTH DEPOT leads at 900."),
    ])
    def test_the_order_written_out(self, question, lang, headline):
        assert self._answer(question, lang)["headline"].startswith(headline)
