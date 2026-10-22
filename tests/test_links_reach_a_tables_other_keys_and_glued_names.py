"""Links to a table's other keys, to a table named in one glued word, and to a number kept as text.

Found on a public text-to-SQL benchmark's card, superhero and football data, invented here, where Learn
missed the links most of its questions follow:

* A card's UUID, unique beside the card's row number, is what its rulings and legal formats hold. A unique text
  column was a target only when its last word is a key word (ID, CODE), and "uuid" is none.
* A set's translations hold the set's code (setCode, for sets.code) on a fifth of the sets: too few for the values
  alone to vouch for it, and the name was read as the set's row number, not its code.
* A hero's powers hold hero_id and power_id, for the superheroes and the superpowers: tables named in one glued
  word whose last part is what the column names. Only names agree that way, and whole numbers meet by chance, so
  the links wait for the admin's word (proposed: used where nothing better links).
* A player's attributes hold the player's number (player_api_id) as a whole number, and the players keep it as
  text. Only a code kept as text pointing at a number key was looked for.
* A match's home and away teams (home_team_api_id) are the teams' own number (team_api_id), unique beside their row
  number: a role before the other key's full name.

What stays unlinked, as seven reviews of these rules found it did not:
* a UUID named alike on many tables (an application's database keeps one on every table, some backfilled with
  gaps): each names its own table's rows, and testing every pair used up the tests the real links needed; nor
  UUIDs named for a role (created by, owner) or for nothing in particular (an audit entry's subject, of many tables);
* another table's key reached by a role: a vendor's or a supplier's external number is no customer's, whatever the
  role word, unless the key names its table (a team's TEAM_API_ID);
* a glued name read anywhere but on a table pairing the things its name says (an order number is the orders' or the
  order headers', never a work order's; a wish list's item is no cart item), or on a table named in two words (an
  order's item is no SHOP_CARTITEM's) or by a two-letter word (an ad is no lead);
* a card's UUID kept again on a table of its prices; a name padded to one length, which is no identifier;
* a number against a same-named code kept as text anywhere but on the table it names, and only where nothing else is
  named for the column: five status numbers against a lookup's or a help desk's status codes, a department's code
  against a cost centre's, a customer number against its copy on a CRM cross-reference, a master copy, another
  schema's copy or a staging copy kept as text (each took the orders' link to the customers), nor a load's staging
  table at all (a library's book copies and a factory's raw materials are no copies);
* of two proposed links, a staging or raw copy of the table holding more of the values (it keeps the deleted rows);
* a profile's copy of its user's uuid, unique on the profiles: the user's, not the profiles' own.
And a number pointing at codes kept as text is told when '01' and '1' both answer to it, a column named with no
Latin letter at all is read without fault, and a glued name stays proposed whatever the warehouse's queries say (the
answers that used it are among them).

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import random
import string
import uuid

import duckdb
import pytest

from core2.bootstrap import joins
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import Inventory, from_duckdb
from core2.warehouse.runner import DuckDBWarehouse


def _model(con: duckdb.DuckDBPyConnection, client: str, **options):
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse, declared_fks=[]), client_id=client,
                       options=BuildOptions(workers=1, **options))


def _links(con: duckdb.DuckDBPyConnection, client: str, **options) -> dict[tuple[str, str], tuple[str, str | None]]:
    model = _model(con, client, **options)
    name = lambda key: f"{model.tables[model.columns[key].table].name}.{model.columns[key].name}"  # noqa: E731
    return {(name(j.from_columns[0]), name(j.to_columns[0])): (j.trust, j.role) for j in model.joins.values()}


def _uuids(rng: random.Random, n: int) -> list[str]:
    return [str(uuid.UUID(int=rng.getrandbits(128))) for _ in range(n)]


@pytest.fixture(scope="module")
def cards():
    rng = random.Random(5)
    con = duckdb.connect()
    con.execute("CREATE TABLE sets (id INTEGER, code VARCHAR, name VARCHAR, released DATE)")
    codes: list[str] = []
    while len(codes) < 300:
        code = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.randint(3, 5)))
        if code not in codes:
            codes.append(code)
    con.executemany("INSERT INTO sets VALUES (?,?,?,?)",
                    [(i + 1, c, f"Edition {c.title()} {i}", dt.date(1995, 1, 1) + dt.timedelta(days=i * 23))
                     for i, c in enumerate(codes)])
    con.execute("CREATE TABLE cards (id INTEGER, uuid VARCHAR, name VARCHAR, artist VARCHAR, setCode VARCHAR, "
                "manaValue INTEGER)")
    ids = [str(uuid.UUID(int=rng.getrandbits(128))) for _ in range(3000)]
    con.executemany("INSERT INTO cards VALUES (?,?,?,?,?,?)",
                    [(i + 1, u, f"Card {i}", f"Artist {i % 140}", codes[i % 300], rng.randint(0, 9))
                     for i, u in enumerate(ids)])
    con.execute("CREATE TABLE rulings (id INTEGER, uuid VARCHAR, ruled DATE, text VARCHAR)")
    ruled = rng.sample(ids, 900)
    con.executemany("INSERT INTO rulings VALUES (?,?,?,?)",
                    [(i + 1, ruled[i % 900], dt.date(2004, 1, 1) + dt.timedelta(days=i), f"Ruling {i} " + "x" * (i % 50))
                     for i in range(1500)])
    con.execute("CREATE TABLE legalities (id INTEGER, uuid VARCHAR, format VARCHAR, status VARCHAR)")
    con.executemany("INSERT INTO legalities VALUES (?,?,?,?)",
                    [(n + 1, u, f, rng.choice(["Legal", "Banned", "Restricted"]))
                     for n, (u, f) in enumerate((u, f) for u in ids[:2000] for f in ("modern", "legacy", "vintage"))])
    # Translations for one set in five, eight languages each.
    con.execute("CREATE TABLE set_translations (id INTEGER, setCode VARCHAR, language VARCHAR, translation VARCHAR)")
    languages = ["French", "German", "Italian", "Spanish", "Japanese", "Korean", "Russian", "Portuguese"]
    con.executemany("INSERT INTO set_translations VALUES (?,?,?,?)",
                    [(n + 1, c, lang, f"{c} ({lang})")
                     for n, (c, lang) in enumerate((c, lang) for c in codes[::5] for lang in languages)])
    return _links(con, "eval-cards")


def test_a_unique_identifier_beside_the_row_number_is_linked_to(cards):
    for table in ("rulings", "legalities"):
        assert cards.get((f"{table}.uuid", "cards.uuid"), ("",))[0] == "verified", cards


def test_a_tables_code_named_with_the_table_is_linked_to(cards):
    assert cards.get(("set_translations.setCode", "sets.code"), ("",))[0] == "verified", cards
    assert ("cards.setCode", "sets.code") in cards, cards
    assert not any(a.endswith(".setCode") and b == "sets.id" for a, b in cards), cards


@pytest.fixture(scope="module")
def heroes():
    rng = random.Random(8)
    con = duckdb.connect()
    con.execute("CREATE TABLE superhero (id INTEGER, superhero_name VARCHAR, full_name VARCHAR, height_cm INTEGER)")
    con.executemany("INSERT INTO superhero VALUES (?,?,?,?)",
                    [(i, f"Hero {i}", f"Person {i} Doe", rng.randint(150, 230)) for i in range(1, 601)])
    con.execute("CREATE TABLE superpower (id INTEGER, power_name VARCHAR)")
    con.executemany("INSERT INTO superpower VALUES (?,?)", [(i, f"Power {i}") for i in range(1, 151)])
    con.execute("CREATE TABLE hero_power (hero_id INTEGER, power_id INTEGER)")
    pairs = {(h, p) for h in rng.sample(range(1, 601), 520) for p in rng.sample(range(1, 151), rng.randint(1, 12))}
    pairs |= {(1 + p % 600, p) for p in range(1, 151)}       # every power held by someone
    con.executemany("INSERT INTO hero_power VALUES (?,?)", sorted(pairs))
    con.execute("CREATE TABLE attribute (id INTEGER, attribute_name VARCHAR)")
    con.executemany("INSERT INTO attribute VALUES (?,?)",
                    list(enumerate(["Intelligence", "Strength", "Speed", "Durability", "Power", "Combat"], 1)))
    con.execute("CREATE TABLE hero_attribute (hero_id INTEGER, attribute_id INTEGER, attribute_value INTEGER)")
    con.executemany("INSERT INTO hero_attribute VALUES (?,?,?)",
                    [(h, a, rng.randint(1, 100)) for h in rng.sample(range(1, 601), 500) for a in range(1, 7)])
    return _links(con, "eval-heroes")


@pytest.mark.parametrize("column, target, trust", [
    # Named only by the end of a glued name: found, and left for the admin to confirm.
    ("hero_power.hero_id", "superhero.id", "proposed"),
    ("hero_power.power_id", "superpower.id", "proposed"),
    ("hero_attribute.hero_id", "superhero.id", "proposed"),
    ("hero_attribute.attribute_id", "attribute.id", "verified"),
])
def test_a_table_named_in_one_glued_word_is_linked_by_its_last_part(heroes, column, target, trust):
    assert heroes.get((column, target), ("",))[0] == trust, heroes


def test_a_glued_name_gives_way_to_a_table_named_by_the_word_itself():
    # Ten orders in a hundred purged, so the orders' link is only proposed; the work orders' numbers run 1 to 4000 and
    # hold every shipment's order number. ORDER_ID still names the orders.
    rng = random.Random(4)
    con = duckdb.connect()
    alive = sorted(rng.sample(range(1, 3001), 2700))
    con.execute("CREATE TABLE orders (id INTEGER, order_no VARCHAR, customer VARCHAR, ordered DATE)")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)",
                    [(o, f"SO-{o:06d}", f"Customer {o % 300}", dt.date(2022, 1, 1) + dt.timedelta(days=o % 700))
                     for o in alive])
    con.execute("CREATE TABLE workorder (id INTEGER, wonum VARCHAR, description VARCHAR, opened DATE)")
    con.executemany("INSERT INTO workorder VALUES (?,?,?,?)",
                    [(w, f"WO{w:07d}", f"Fix asset {w} " + "x" * (w % 9), dt.date(2021, 1, 1) + dt.timedelta(days=w % 900))
                     for w in range(1, 4001)])
    con.execute("CREATE TABLE shipments (id INTEGER, order_id INTEGER, carrier VARCHAR, weight_kg DECIMAL(8,2))")
    con.executemany("INSERT INTO shipments VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 3000), rng.choice(["Road", "Air", "Sea"]), rng.randint(10, 5000) / 100)
                     for i in range(1, 5001)])
    links = _links(con, "eval-workorders")
    assert ("shipments.order_id", "orders.id") in links and ("shipments.order_id", "workorder.id") not in links, links


def test_two_tables_own_identifiers_of_one_name_are_not_tested_against_each_other():
    # Every table of an application's database keeps a unique uuid beside its row number. Testing every pair of
    # them, at the top of the list by name, used up the tests the real links needed.
    rng = random.Random(3)
    con = duckdb.connect()
    parents = {"customers": 400, "products": 300, "stores": 80, "suppliers": 120}
    for table, n in parents.items():
        con.execute(f"CREATE TABLE {table} (id INTEGER, uuid VARCHAR, name VARCHAR)")
        con.executemany(f"INSERT INTO {table} VALUES (?,?,?)",
                        [(i, u, f"{table[:-1].title()} {i} {'x' * (i % 7)}") for i, u in enumerate(_uuids(rng, n), 1)])
    names_ = list(parents)
    expected = set()
    for c in range(8):
        one, two = names_[c % 4], names_[(c + 1 + c // 4) % 4]
        con.execute(f"CREATE TABLE events_{c} (id INTEGER, uuid VARCHAR, {one[:-1]}_id INTEGER, {two[:-1]}_id INTEGER, "
                    "amount DECIMAL(10,2))")
        con.executemany(f"INSERT INTO events_{c} VALUES (?,?,?,?,?)",
                        [(i, u, rng.randint(1, parents[one]), rng.randint(1, parents[two]), rng.randint(100, 99999) / 100)
                         for i, u in enumerate(_uuids(rng, 500), 1)])
        expected |= {(f"events_{c}.{one[:-1]}_id", f"{one}.id"), (f"events_{c}.{two[:-1]}_id", f"{two}.id")}
    links = _links(con, "eval-app", max_join_tests=60)
    assert expected <= set(links), expected - set(links)
    assert not any(a.endswith(".uuid") for a, _ in links), links


def test_a_role_naming_another_tables_rows_points_there():
    # Customers and vendors each keep an external number beside their row number; a vendor invoice's
    # vendor_external_id is the vendor's (a tenth of the vendors since deleted), never the customer's, however
    # densely the customers' numbers run.
    rng = random.Random(11)
    con = duckdb.connect()
    con.execute("CREATE TABLE customers (id INTEGER, legacy_id INTEGER, external_id INTEGER, name VARCHAR)")
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)",
                    [(i, i + 100, e, f"Customer {i}") for i, e in enumerate(rng.sample(range(1, 3001), 3000), 1)])
    alive = sorted(rng.sample(range(1, 401), 360))
    con.execute("CREATE TABLE vendors (id INTEGER, external_id INTEGER, name VARCHAR)")
    con.executemany("INSERT INTO vendors VALUES (?,?,?)", [(i, e, f"Vendor {e}") for i, e in enumerate(alive, 1)])
    con.execute("CREATE TABLE vendor_invoices (id INTEGER, vendor_external_id INTEGER, vendor_legacy_id INTEGER, "
                "amount DECIMAL(12,2), invoiced DATE)")
    invoices = []
    for i in range(1, 4001):
        e = rng.randint(1, 400)
        invoices.append((i, e, 101 + e * 7 % 800, rng.randint(1000, 900000) / 100,
                         dt.date(2023, 1, 1) + dt.timedelta(days=i % 600)))
    con.executemany("INSERT INTO vendor_invoices VALUES (?,?,?,?,?)", invoices)
    links = _links(con, "eval-vendors")
    assert not any(a.startswith("vendor_invoices.") and b.startswith("customers.") for a, b in links), links
    assert ("vendor_invoices.vendor_external_id", "vendors.external_id") in links, links


def test_a_cards_identifier_kept_again_on_its_prices_is_the_cards():
    rng = random.Random(9)
    con = duckdb.connect()
    ids = _uuids(rng, 3000)
    con.execute("CREATE TABLE cards (id INTEGER, uuid VARCHAR, name VARCHAR, artist VARCHAR)")
    con.executemany("INSERT INTO cards VALUES (?,?,?,?)", [(i, u, f"Card {i}", f"Artist {i % 90}") for i, u in enumerate(ids, 1)])
    con.execute("CREATE TABLE card_prices (id INTEGER, card_uuid VARCHAR, price DECIMAL(10,2), currency VARCHAR)")
    con.executemany("INSERT INTO card_prices VALUES (?,?,?,?)",
                    [(i, u, rng.randint(5, 9000) / 100, "USD") for i, u in enumerate(ids, 1)])
    con.execute("CREATE TABLE rulings (id INTEGER, card_uuid VARCHAR, ruled DATE, text VARCHAR)")
    ruled = rng.sample(ids, 1200)
    con.executemany("INSERT INTO rulings VALUES (?,?,?,?)",
                    [(i, ruled[i % 1200], dt.date(2005, 1, 1) + dt.timedelta(days=i % 400), f"Ruling {i} " + "y" * (i % 30))
                     for i in range(2000)])
    links = _links(con, "eval-card-prices")
    assert ("rulings.card_uuid", "cards.uuid") in links and ("card_prices.card_uuid", "cards.uuid") in links, links
    assert not any(b == "card_prices.card_uuid" or a == "cards.uuid" for a, b in links), links


def test_a_number_answered_by_two_codes_kept_as_text_is_told():
    # The players' numbers kept as text, two of them written with a leading zero beside the same number without
    # one: player 7 finds both '07' and '7', and totals through the link would count its rows twice.
    rng = random.Random(17)
    con = duckdb.connect()
    con.execute("CREATE TABLE Player (id VARCHAR, player_api_id VARCHAR, player_name VARCHAR)")
    numbers = rng.sample(range(10, 90_000), 600)
    codes = [str(n) for n in numbers] + ["0" + str(numbers[0]), "0" + str(numbers[1])]
    con.executemany("INSERT INTO Player VALUES (?,?,?)", [(str(i), c, f"Player {i}") for i, c in enumerate(codes, 1)])
    con.execute("CREATE TABLE Player_Attributes (id INTEGER, player_api_id INTEGER, date DATE, overall_rating INTEGER)")
    con.executemany("INSERT INTO Player_Attributes VALUES (?,?,?,?)",
                    [(i, numbers[i % 600], dt.date(2010, 1, 1) + dt.timedelta(days=i % 900), rng.randint(40, 95))
                     for i in range(4000)])
    model = _model(con, "eval-padded")
    name = lambda key: f"{model.tables[model.columns[key].table].name}.{model.columns[key].name}"  # noqa: E731
    link = next(j for j in model.joins.values()
                if (name(j.from_columns[0]), name(j.to_columns[0])) == ("Player_Attributes.player_api_id",
                                                                        "Player.player_api_id"))
    assert not link.to_unique and link.max_fanout == 2, link


@pytest.fixture(scope="module")
def football():
    rng = random.Random(13)
    con = duckdb.connect()
    # The players' numbers kept as text, the attributes' as whole numbers.
    con.execute("CREATE TABLE Player (id VARCHAR, player_api_id VARCHAR, player_name VARCHAR, height VARCHAR)")
    numbers = rng.sample(range(30_000, 750_000), 800)
    con.executemany("INSERT INTO Player VALUES (?,?,?,?)",
                    [(str(i + 1), str(n), f"Player {i}", str(rng.randint(160, 200))) for i, n in enumerate(numbers)])
    con.execute("CREATE TABLE Player_Attributes (id INTEGER, player_api_id INTEGER, date DATE, overall_rating INTEGER)")
    con.executemany("INSERT INTO Player_Attributes VALUES (?,?,?,?)",
                    [(i + 1, numbers[i % 800], dt.date(2008, 1, 1) + dt.timedelta(days=i % 2900), rng.randint(40, 95))
                     for i in range(6000)])
    con.execute("CREATE TABLE Team (id INTEGER, team_api_id INTEGER, team_long_name VARCHAR, team_short_name VARCHAR)")
    teams = rng.sample(range(1_600, 11_000), 120)
    con.executemany("INSERT INTO Team VALUES (?,?,?,?)",
                    [(i * 37 + 1, t, f"Club {i}", f"C{i:02}") for i, t in enumerate(teams)])
    con.execute("CREATE TABLE Match (id INTEGER, season VARCHAR, match_api_id INTEGER, home_team_api_id INTEGER, "
                "away_team_api_id INTEGER, home_team_goal INTEGER, away_team_goal INTEGER)")
    games = []
    for i in range(3000):
        home, away = rng.sample(teams, 2)
        games.append((i + 1, f"{2008 + i % 8}/{2009 + i % 8}", 480_000 + i * 3, home, away, rng.randint(0, 5),
                      rng.randint(0, 5)))
    con.executemany("INSERT INTO Match VALUES (?,?,?,?,?,?,?)", games)
    return _links(con, "eval-football")


def test_a_number_points_at_the_same_number_kept_as_text(football):
    assert football.get(("Player_Attributes.player_api_id", "Player.player_api_id"), ("",))[0] == "verified", football
    assert ("Player_Attributes.player_api_id", "Player.id") not in football, football


@pytest.mark.parametrize("column, role", [("Match.home_team_api_id", "Home team"),
                                          ("Match.away_team_api_id", "Away team")])
def test_a_role_before_another_keys_full_name_is_linked_to_it(football, column, role):
    trust, named = football.get((column, "Team.team_api_id"), ("", None))
    assert trust == "verified" and named == role, football


def test_a_role_naming_no_table_reaches_no_other_tables_key():
    # The supplier and the payee are the vendors' (a tenth of them deleted); the customers' external numbers run
    # densely 1 to 3000 and hold them all. Neither role word names a table, and EXTERNAL_ID names none either.
    rng = random.Random(11)
    con = duckdb.connect()
    con.execute("CREATE TABLE customers (id INTEGER, external_id INTEGER, name VARCHAR)")
    con.executemany("INSERT INTO customers VALUES (?,?,?)",
                    [(i, e, f"Customer {i}") for i, e in enumerate(rng.sample(range(1, 3001), 3000), 1)])
    alive = sorted(rng.sample(range(1, 401), 360))
    con.execute("CREATE TABLE vendors (id INTEGER, external_id INTEGER, name VARCHAR)")
    con.executemany("INSERT INTO vendors VALUES (?,?,?)", [(i, e, f"Vendor {e}") for i, e in enumerate(alive, 1)])
    con.execute("CREATE TABLE ap_invoices (id INTEGER, supplier_external_id INTEGER, payee_external_id INTEGER, "
                "amount DECIMAL(12,2), invoiced DATE)")
    con.executemany("INSERT INTO ap_invoices VALUES (?,?,?,?,?)",
                    [(i, e, e, rng.randint(1000, 900000) / 100, dt.date(2023, 1, 1) + dt.timedelta(days=i % 600))
                     for i, e in ((i, rng.randint(1, 400)) for i in range(1, 4001))])
    links = _links(con, "eval-payables")
    assert not any(a.startswith("ap_invoices.") and b == "customers.external_id" for a, b in links), links


def test_a_glued_name_is_read_only_on_a_table_named_for_what_it_holds():
    rng = random.Random(4)
    con = duckdb.connect()
    # A wish list's items are products; the cart items' own numbers run 1 to 1000.
    con.execute("CREATE TABLE shop_product (id INTEGER, title VARCHAR, price DECIMAL(10,2))")
    con.executemany("INSERT INTO shop_product VALUES (?,?,?)",
                    [(i, f"Product {i} " + "x" * (i % 7), rng.randint(100, 9999) / 100) for i in range(1, 801)])
    con.execute("CREATE TABLE shop_cartitem (id INTEGER, product_id INTEGER, quantity INTEGER, added DATE)")
    con.executemany("INSERT INTO shop_cartitem VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 800), rng.randint(1, 5), dt.date(2023, 1, 1) + dt.timedelta(days=i % 300))
                     for i in range(1, 1001)])
    wished = rng.sample(range(1, 801), 600)
    con.execute("CREATE TABLE shop_wishlistentry (id INTEGER, item_id INTEGER, created DATE, note VARCHAR)")
    con.executemany("INSERT INTO shop_wishlistentry VALUES (?,?,?,?)",
                    [(i, rng.choice(wished), dt.date(2023, 1, 1) + dt.timedelta(days=i % 300), f"Note {i % 13}")
                     for i in range(1, 5001)])
    # The orders kept as order headers; the work orders' numbers run 1 to 4000.
    con.execute("CREATE TABLE order_header (id INTEGER, order_no VARCHAR, customer VARCHAR, ordered DATE)")
    con.executemany("INSERT INTO order_header VALUES (?,?,?,?)",
                    [(o, f"SO-{o:06d}", f"Customer {o % 300}", dt.date(2022, 1, 1) + dt.timedelta(days=o % 700))
                     for o in range(1, 3001)])
    con.execute("CREATE TABLE workorder (id INTEGER, wonum VARCHAR, description VARCHAR, opened DATE)")
    con.executemany("INSERT INTO workorder VALUES (?,?,?,?)",
                    [(w, f"WO{w:07d}", f"Fix asset {w} " + "x" * (w % 9), dt.date(2021, 1, 1) + dt.timedelta(days=w % 900))
                     for w in range(1, 4001)])
    con.execute("CREATE TABLE shipments (id INTEGER, order_id INTEGER, carrier VARCHAR, weight_kg DECIMAL(8,2))")
    con.executemany("INSERT INTO shipments VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 3000), rng.choice(["Road", "Air", "Sea"]), rng.randint(10, 5000) / 100)
                     for i in range(1, 5001)])
    links = _links(con, "eval-glued")
    assert ("shop_wishlistentry.item_id", "shop_cartitem.id") not in links, links
    assert ("shipments.order_id", "workorder.id") not in links, links


def test_uuids_named_for_a_role_leave_the_real_links_tested():
    # Every table keeps a uuid; each event says who created, updated and owns it by the users' uuid. "Created by" names
    # no table, so those columns are no candidates against every table's uuid, and the bill-to links are tested.
    rng = random.Random(3)
    con = duckdb.connect()
    users = _uuids(rng, 200)
    con.execute("CREATE TABLE users (id INTEGER, uuid VARCHAR, login VARCHAR)")
    con.executemany("INSERT INTO users VALUES (?,?,?)", [(i, u, f"user{i}{'x' * (i % 5)}") for i, u in enumerate(users, 1)])
    con.execute("CREATE TABLE customers (id INTEGER, uuid VARCHAR, name VARCHAR)")
    con.executemany("INSERT INTO customers VALUES (?,?,?)",
                    [(i, u, f"Customer {i} {'y' * (i % 6)}") for i, u in enumerate(_uuids(rng, 400), 1)])
    for m in range(10):
        con.execute(f"CREATE TABLE lookup_{m} (id INTEGER, uuid VARCHAR, label VARCHAR)")
        con.executemany(f"INSERT INTO lookup_{m} VALUES (?,?,?)",
                        [(i, u, f"Label {m} {i} {'z' * (i % 4)}") for i, u in enumerate(_uuids(rng, 40 + m), 1)])
    expected = set()
    for c in range(6):
        con.execute(f"CREATE TABLE events_{c} (id INTEGER, uuid VARCHAR, created_by_uuid VARCHAR, updated_by_uuid VARCHAR, "
                    "owner_uuid VARCHAR, bill_to_customer_id INTEGER, amount DECIMAL(10,2))")
        con.executemany(f"INSERT INTO events_{c} VALUES (?,?,?,?,?,?,?)",
                        [(i, u, rng.choice(users), rng.choice(users), rng.choice(users), rng.randint(1, 400),
                          rng.randint(100, 99999) / 100) for i, u in enumerate(_uuids(rng, 500), 1)])
        expected.add((f"events_{c}.bill_to_customer_id", "customers.id"))
    links = _links(con, "eval-audit-uuids", max_join_tests=40)
    assert expected <= set(links), expected - set(links)
    assert not any(a.endswith(("_by_uuid", "owner_uuid")) for a, _ in links), links


def test_an_audit_entrys_subject_of_many_tables_is_no_link():
    rng = random.Random(5)
    con = duckdb.connect()
    customers, orders = _uuids(rng, 500), _uuids(rng, 2000)
    con.execute("CREATE TABLE customers (id INTEGER, uuid VARCHAR, name VARCHAR)")
    con.executemany("INSERT INTO customers VALUES (?,?,?)",
                    [(i, u, f"Customer {i} {'q' * (i % 5)}") for i, u in enumerate(customers, 1)])
    con.execute("CREATE TABLE orders (id INTEGER, uuid VARCHAR, customer_id INTEGER, amount DECIMAL(10,2))")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)",
                    [(i, u, rng.randint(1, 500), rng.randint(100, 99999) / 100) for i, u in enumerate(orders, 1)])
    con.execute("CREATE TABLE audit_log (id INTEGER, subject_uuid VARCHAR, action VARCHAR, logged DATE)")
    con.executemany("INSERT INTO audit_log VALUES (?,?,?,?)",
                    [(i, rng.choice(orders) if rng.random() < 0.7 else rng.choice(customers),
                      rng.choice(["create", "update", "delete"]), dt.date(2024, 1, 1) + dt.timedelta(days=i % 200))
                     for i in range(1, 6001)])
    links = _links(con, "eval-audit")
    assert not any(a == "audit_log.subject_uuid" for a, _ in links), links


def test_prefixed_tables_own_uuids_are_not_tested_against_each_other(monkeypatch):
    tested = []
    real = joins._test

    def counted(warehouse, inventory, finding, profiles, placeholders):
        tested.append((finding.from_column, finding.to_column))
        return real(warehouse, inventory, finding, profiles, placeholders)

    monkeypatch.setattr(joins, "_test", counted)
    rng = random.Random(2)
    con = duckdb.connect()
    for table, prefix, n in [("customers", "cu", 300), ("products", "pr", 200), ("stores", "st", 60),
                             ("suppliers", "sp", 90), ("regions", "rg", 40), ("carriers", "ca", 30)]:
        con.execute(f"CREATE TABLE {table} ({prefix}_{table[:4]}key INTEGER, {prefix}_uuid VARCHAR, {prefix}_name VARCHAR)")
        con.executemany(f"INSERT INTO {table} VALUES (?,?,?)",
                        [(i, u, f"{table} {i} {'k' * (i % 6)}") for i, u in enumerate(_uuids(rng, n), 1)])
    _model(con, "eval-prefixed")
    assert not [t for t in tested if t[0].endswith("_uuid") and t[1].endswith("_uuid")], tested


def test_status_numbers_and_a_padded_name_are_no_links():
    rng = random.Random(7)
    con = duckdb.connect()
    # Five order status numbers, and a help desk's eight status codes kept as text: no order status table.
    con.execute("CREATE TABLE ticket_status (status_code VARCHAR, label VARCHAR)")
    con.executemany("INSERT INTO ticket_status VALUES (?,?)", [(str(i), label) for i, label in enumerate(
        ["New", "Assigned", "In progress", "Waiting on customer", "Escalated", "Resolved", "Closed", "Reopened"], 1)])
    con.execute("CREATE TABLE tickets (id INTEGER, status_code VARCHAR, opened DATE, subject VARCHAR)")
    con.executemany("INSERT INTO tickets VALUES (?,?,?,?)",
                    [(i, str(rng.randint(1, 8)), dt.date(2023, 1, 1) + dt.timedelta(days=i % 400), f"Subject {i}")
                     for i in range(1, 1501)])
    con.execute("CREATE TABLE orders (id INTEGER, status_code INTEGER, ordered DATE, amount DECIMAL(10,2))")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 5), dt.date(2023, 1, 1) + dt.timedelta(days=i % 500), rng.randint(100, 99999) / 100)
                     for i in range(1, 4001)])
    # A product's name padded to forty characters, as a CHAR column reads on some warehouses, kept on its lines.
    names_ = [f"Product {i} {'abcdefghij'[i % 10] * (i % 9)}".ljust(40) for i in range(1, 501)]
    con.execute("CREATE TABLE products (id INTEGER, product_name VARCHAR, price DECIMAL(10,2))")
    con.executemany("INSERT INTO products VALUES (?,?,?)",
                    [(i, n, rng.randint(100, 9999) / 100) for i, n in enumerate(names_, 1)])
    con.execute("CREATE TABLE order_lines (id INTEGER, order_id INTEGER, product_id INTEGER, product_name VARCHAR, "
                "qty INTEGER)")
    lines = []
    for i in range(1, 6001):
        product = rng.randint(1, 500)
        lines.append((i, rng.randint(1, 4000), product, names_[product - 1], rng.randint(1, 9)))
    con.executemany("INSERT INTO order_lines VALUES (?,?,?,?,?)", lines)
    links = _links(con, "eval-status")
    assert ("orders.status_code", "ticket_status.status_code") not in links, links
    assert ("order_lines.product_name", "products.product_name") not in links, links
    assert ("order_lines.product_id", "products.id") in links and ("order_lines.order_id", "orders.id") in links, links


def test_a_uuid_named_alike_on_many_tables_is_no_target():
    # Some older tables had their uuid added later and backfilled for most rows: not unique there, it named every
    # other table's uuid alike, and those pairs took every test before the real links.
    rng = random.Random(3)
    con = duckdb.connect()
    sizes = {}
    for p in range(24):
        table = f"entity{p:03d}s" if p >= 4 else ["customers", "products", "stores", "suppliers"][p]
        sizes[table] = rng.randint(40, 400)
        con.execute(f"CREATE TABLE {table} (id INTEGER, uuid VARCHAR, name VARCHAR)")
        con.executemany(f"INSERT INTO {table} VALUES (?,?,?)",
                        [(i, u, f"{table} {i} {'x' * (i % 7)}") for i, u in enumerate(_uuids(rng, sizes[table]), 1)])
    for m in range(4):
        con.execute(f"CREATE TABLE legacy_note{m} (id INTEGER, uuid VARCHAR, body VARCHAR)")
        con.executemany(f"INSERT INTO legacy_note{m} VALUES (?,?,?)",
                        [(i, u if rng.random() < 0.9 else None, f"Note {i} {'n' * (i % 9)}")
                         for i, u in enumerate(_uuids(rng, 300), 1)])
    four, expected = ["customers", "products", "stores", "suppliers"], set()
    for c in range(4):
        one, two = four[c], four[(c + 1) % 4]
        con.execute(f"CREATE TABLE events_{c} (id INTEGER, uuid VARCHAR, {one[:-1]}_id INTEGER, {two[:-1]}_id INTEGER, "
                    "amount DECIMAL(10,2))")
        con.executemany(f"INSERT INTO events_{c} VALUES (?,?,?,?,?)",
                        [(i, u, rng.randint(1, sizes[one]), rng.randint(1, sizes[two]), rng.randint(100, 99999) / 100)
                         for i, u in enumerate(_uuids(rng, 500), 1)])
        expected |= {(f"events_{c}.{one[:-1]}_id", f"{one}.id"), (f"events_{c}.{two[:-1]}_id", f"{two}.id")}
    links = _links(con, "eval-backfilled", max_join_tests=40)
    assert expected <= set(links), expected - set(links)


def test_a_wish_lists_items_are_no_cart_items():
    rng = random.Random(4)
    con = duckdb.connect()
    con.execute("CREATE TABLE shop_product (id INTEGER, title VARCHAR, price DECIMAL(10,2))")
    con.executemany("INSERT INTO shop_product VALUES (?,?,?)",
                    [(i, f"Product {i} " + "x" * (i % 7), rng.randint(100, 9999) / 100) for i in range(1, 801)])
    con.execute("CREATE TABLE shop_cartitem (id INTEGER, product_id INTEGER, quantity INTEGER, added DATE)")
    con.executemany("INSERT INTO shop_cartitem VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 800), rng.randint(1, 5), dt.date(2023, 1, 1) + dt.timedelta(days=i % 300))
                     for i in range(1, 1001)])
    wished = rng.sample(range(1, 801), 600)
    con.execute("CREATE TABLE wishlist_items (id INTEGER, item_id INTEGER, created DATE, note VARCHAR)")
    con.executemany("INSERT INTO wishlist_items VALUES (?,?,?,?)",
                    [(i, rng.choice(wished), dt.date(2023, 1, 1) + dt.timedelta(days=i % 300), f"Note {i % 13}")
                     for i in range(1, 5001)])
    links = _links(con, "eval-wishlist")
    assert ("wishlist_items.item_id", "shop_cartitem.id") not in links, links


def test_a_bridges_glued_name_gives_way_to_a_table_named_by_the_word():
    # An order's items pair orders and items; the orders are kept as order headers (a tenth of them purged), and the
    # work orders' numbers run densely past every order's. No name ties ORDER_ID to the headers, and still it is no
    # work order's: another table is named by the word.
    rng = random.Random(14)
    con = duckdb.connect()
    alive = sorted(rng.sample(range(1, 3001), 2700))
    con.execute("CREATE TABLE order_header (id INTEGER, order_no VARCHAR, ordered DATE)")
    con.executemany("INSERT INTO order_header VALUES (?,?,?)",
                    [(o, f"SO-{o:06d}", dt.date(2022, 1, 1) + dt.timedelta(days=o % 700)) for o in alive])
    con.execute("CREATE TABLE workorder (id INTEGER, wonum VARCHAR, opened DATE)")
    con.executemany("INSERT INTO workorder VALUES (?,?,?)",
                    [(w, f"WO{w:07d}", dt.date(2021, 1, 1) + dt.timedelta(days=w % 900)) for w in range(1, 4001)])
    con.execute("CREATE TABLE items (id INTEGER, item_name VARCHAR)")
    con.executemany("INSERT INTO items VALUES (?,?)", [(i, f"Item {i} {'i' * (i % 5)}") for i in range(1, 301)])
    con.execute("CREATE TABLE order_item (order_id INTEGER, item_id INTEGER, qty INTEGER)")
    con.executemany("INSERT INTO order_item VALUES (?,?,?)",
                    sorted({(rng.randint(1, 3000), rng.randint(1, 300), rng.randint(1, 9)) for _ in range(6000)}))
    links = _links(con, "eval-order-items")
    assert ("order_item.order_id", "workorder.id") not in links, links
    assert ("order_item.item_id", "items.id") in links, links


def test_a_number_points_at_no_code_of_another_table_or_name():
    rng = random.Random(6)
    con = duckdb.connect()
    # Each master keeps its own code beside its row number: the departments' as numbers, the cost centres' as text.
    con.execute("CREATE TABLE departments (id INTEGER, code INTEGER, name VARCHAR)")
    con.executemany("INSERT INTO departments VALUES (?,?,?)",
                    [(i, c, f"Department {c} {'d' * (c % 5)}") for i, c in enumerate(rng.sample(range(1, 61), 60), 1)])
    con.execute("CREATE TABLE cost_centres (id INTEGER, code VARCHAR, name VARCHAR)")
    con.executemany("INSERT INTO cost_centres VALUES (?,?,?)",
                    [(i, str(c), f"Cost centre {c} {'c' * (c % 7)}")
                     for i, c in enumerate(rng.sample(range(1, 501), 500), 1)])
    # The customers' numbers kept again as text on a CRM cross-reference, under the orders' column's name.
    con.execute("CREATE TABLE customers (id INTEGER, name VARCHAR, segment VARCHAR)")
    con.executemany("INSERT INTO customers VALUES (?,?,?)",
                    [(i, f"Customer {i} {'c' * (i % 6)}", rng.choice("ABC")) for i in range(1, 1201)])
    con.execute("CREATE TABLE customer_xref (customer_id VARCHAR, crm_ref VARCHAR, synced DATE)")
    con.executemany("INSERT INTO customer_xref VALUES (?,?,?)",
                    [(str(i), f"CRM-{rng.randint(10**6, 10**7)}", dt.date(2024, 1, 1) + dt.timedelta(days=i % 90))
                     for i in range(1, 1201)])
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DECIMAL(10,2), ordered DATE)")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 1200), rng.randint(100, 99999) / 100,
                      dt.date(2023, 1, 1) + dt.timedelta(days=i % 500)) for i in range(1, 6001)])
    # A customer's number beside its code, the codes kept as text and in another order than the numbers.
    con.execute("CREATE TABLE cus (id INTEGER, cus_cd VARCHAR, cus_nm VARCHAR)")
    codes = rng.sample(range(1, 201), 200)
    con.executemany("INSERT INTO cus VALUES (?,?,?)", [(i, str(c), f"Customer {i}") for i, c in enumerate(codes, 1)])
    con.execute("CREATE TABLE cus_visits (id INTEGER, cus_key INTEGER, visited DATE)")
    con.executemany("INSERT INTO cus_visits VALUES (?,?,?)",
                    [(i, rng.randint(1, 200), dt.date(2024, 1, 1) + dt.timedelta(days=i % 300)) for i in range(1, 2001)])
    links = _links(con, "eval-codes")
    assert ("departments.code", "cost_centres.code") not in links, links
    assert ("orders.customer_id", "customers.id") in links, links
    assert ("cus_visits.cus_key", "cus.cus_cd") not in links, links


def test_a_staging_copy_never_wins_for_holding_more_of_the_values():
    # The customers less the deleted ones, beside the staging copy keeping them all; orders also name customers created
    # since the last load, so both links are only proposed.
    rng = random.Random(12)
    con = duckdb.connect()
    staged = list(range(1, 2901))
    alive = sorted(rng.sample(staged, 2600))
    con.execute("CREATE TABLE customers (id INTEGER, name VARCHAR, segment VARCHAR, created DATE)")
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)",
                    [(c, f"Customer {c} {'q' * (c % 6)}", rng.choice(["SMB", "Mid", "Ent"]),
                      dt.date(2020, 1, 1) + dt.timedelta(days=c % 900)) for c in alive])
    con.execute("CREATE TABLE stg_customers (id INTEGER, name VARCHAR, segment VARCHAR, is_deleted BOOLEAN, loaded DATE)")
    con.executemany("INSERT INTO stg_customers VALUES (?,?,?,?,?)",
                    [(c, f"Customer {c} {'q' * (c % 6)}", rng.choice(["SMB", "Mid", "Ent"]), c not in set(alive),
                      dt.date(2024, 1, 1)) for c in staged])
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DECIMAL(10,2), ordered DATE)")
    pool = rng.sample(range(1, 3001), 2000)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)",
                    [(i, rng.choice(pool), rng.randint(100, 99999) / 100, dt.date(2023, 1, 1) + dt.timedelta(days=i % 600))
                     for i in range(1, 8001)])
    links = _links(con, "eval-staging")
    assert ("orders.customer_id", "customers.id") in links, links


def test_a_column_named_with_no_latin_letter_is_read():
    rng = random.Random(1)
    con = duckdb.connect()
    con.execute('CREATE TABLE kunden (id INTEGER, "識別子" VARCHAR, name VARCHAR)')
    con.executemany("INSERT INTO kunden VALUES (?,?,?)",
                    [(i, u, f"Kunde {i} {'k' * (i % 5)}") for i, u in enumerate(_uuids(rng, 300), 1)])
    con.execute("CREATE TABLE bestellungen (id INTEGER, kunden_id INTEGER, betrag DECIMAL(10,2))")
    con.executemany("INSERT INTO bestellungen VALUES (?,?,?)",
                    [(i, rng.randint(1, 300), rng.randint(100, 9999) / 100) for i in range(1, 2001)])
    assert ("bestellungen.kunden_id", "kunden.id") in _links(con, "eval-unlettered")


def test_a_uuid_is_reached_by_name_alone():
    # The cards alone keep a unique uuid; an audit entry's subject is a card on most rows and something else on the
    # rest. Its values fit well enough, but no name says the cards: an identifier is pointed at by name.
    rng = random.Random(19)
    con = duckdb.connect()
    cards = _uuids(rng, 3000)
    con.execute("CREATE TABLE cards (id INTEGER, uuid VARCHAR, name VARCHAR)")
    con.executemany("INSERT INTO cards VALUES (?,?,?)", [(i, u, f"Card {i} {'c' * (i % 5)}") for i, u in enumerate(cards, 1)])
    subjects = rng.sample(cards, 1400) + _uuids(rng, 600)
    con.execute("CREATE TABLE audit_log (id INTEGER, subject_uuid VARCHAR, action VARCHAR, logged DATE)")
    con.executemany("INSERT INTO audit_log VALUES (?,?,?,?)",
                    [(i, rng.choice(subjects), rng.choice(["create", "update"]), dt.date(2024, 1, 1) + dt.timedelta(days=i % 200))
                     for i in range(1, 6001)])
    con.execute("CREATE TABLE rulings (id INTEGER, uuid VARCHAR, ruled DATE)")
    con.executemany("INSERT INTO rulings VALUES (?,?,?)",
                    [(i, rng.choice(cards), dt.date(2005, 1, 1) + dt.timedelta(days=i % 400)) for i in range(1, 2001)])
    links = _links(con, "eval-subject")
    assert ("rulings.uuid", "cards.uuid") in links, links
    assert not any(a == "audit_log.subject_uuid" for a, _ in links), links



def _customers_orders(con, rng, n=1500):
    con.execute("CREATE TABLE customers (id INTEGER, name VARCHAR, segment VARCHAR, created DATE)")
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)",
                    [(i, f"Customer {i} {'q' * (i % 6)}", rng.choice(["SMB", "Mid", "Ent"]),
                      dt.date(2020, 1, 1) + dt.timedelta(days=i % 900)) for i in range(1, n + 1)])
    con.execute("CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, status_code INTEGER, amount DECIMAL(10,2))")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)",
                    [(i, rng.randint(1, n), rng.randint(1, 5), rng.randint(100, 99999) / 100) for i in range(1, 6001)])


@pytest.mark.parametrize("copy", ["customer_master", "stg_customers", "crm.customers"])
def test_a_customer_number_kept_again_as_text_never_takes_the_orders_link(copy):
    rng = random.Random(23)
    con = duckdb.connect()
    _customers_orders(con, rng)
    if copy.startswith("crm."):
        con.execute("CREATE SCHEMA crm")
    con.execute(f"CREATE TABLE {copy} (customer_id VARCHAR, crm_ref VARCHAR, synced DATE)")
    con.executemany(f"INSERT INTO {copy} VALUES (?,?,?)",
                    [(str(i), f"CRM-{rng.randint(10**6, 10**7)}", dt.date(2024, 1, 1)) for i in range(1, 1501)])
    # A lookup's status codes kept as text, and five order status numbers.
    con.execute("CREATE TABLE lkp_status (status_code VARCHAR, label VARCHAR)")
    con.executemany("INSERT INTO lkp_status VALUES (?,?)",
                    [(str(i), label) for i, label in enumerate(["New", "Open", "Held", "Late", "Shut", "Lost", "Won", "Void"], 1)])
    warehouse = DuckDBWarehouse(con)
    parts = [from_duckdb(warehouse, schema=schema, declared_fks=[]) for schema in ("main", "crm")[:1 + copy.startswith("crm.")]]
    model = build_model(warehouse, Inventory(tables={k: t for i in parts for k, t in i.tables.items()}, foreign_keys=[]),
                        client_id="eval-copies", options=BuildOptions(workers=1))
    name = lambda key: f"{model.tables[model.columns[key].table].name}.{model.columns[key].name}"  # noqa: E731
    links = {(name(j.from_columns[0]), name(j.to_columns[0])) for j in model.joins.values()}
    assert ("orders.customer_id", "customers.id") in links, links
    assert not any(a == "orders.status_code" for a, _ in links), links


def test_a_raw_copy_holding_more_codes_never_wins():
    # The live catalogue, a sixth of its SKUs removed, beside the raw load keeping every SKU ever loaded; order lines
    # name some SKUs created since, so both links are only proposed.
    rng = random.Random(4)
    con = duckdb.connect()
    skus = [f"SK{n:08d}" for n in rng.sample(range(10**7, 10**8), 3000)]
    live = skus[:2500]
    con.execute("CREATE TABLE products (id INTEGER, sku VARCHAR, title VARCHAR, category VARCHAR)")
    con.executemany("INSERT INTO products VALUES (?,?,?,?)",
                    [(i, sku, f"Product {i} {'p' * (i % 7)}", rng.choice("ABCDEFG")) for i, sku in enumerate(live, 1)])
    con.execute("CREATE TABLE products_raw (sku VARCHAR, title VARCHAR, category VARCHAR, loaded DATE)")
    con.executemany("INSERT INTO products_raw VALUES (?,?,?,?)",
                    [(sku, f"Product {i} {'p' * (i % 7)}", rng.choice("ABCDEFG"), dt.date(2024, 1, 1))
                     for i, sku in enumerate(skus, 1)])
    new = [f"SK{n:08d}" for n in rng.sample(range(10**7, 10**8), 300)]
    pool = live[:1700] + skus[2500:] + new[:200]
    con.execute("CREATE TABLE order_lines (id INTEGER, sku VARCHAR, qty INTEGER, amount DECIMAL(10,2))")
    con.executemany("INSERT INTO order_lines VALUES (?,?,?,?)",
                    [(i, rng.choice(pool), rng.randint(1, 9), rng.randint(100, 9999) / 100) for i in range(1, 9001)])
    links = _links(con, "eval-raw-skus")
    assert ("order_lines.sku", "products.sku") in links, links


def test_a_profiles_copy_of_its_users_uuid_is_no_target():
    rng = random.Random(5)
    con = duckdb.connect()
    users = _uuids(rng, 1200)
    con.execute("CREATE TABLE accounts_user (id INTEGER, uuid VARCHAR, email VARCHAR, joined DATE)")
    con.executemany("INSERT INTO accounts_user VALUES (?,?,?,?)",
                    [(i, u, f"user{i}@example.com", dt.date(2023, 1, 1) + dt.timedelta(days=i % 400))
                     for i, u in enumerate(users, 1)])
    con.execute("CREATE TABLE profiles_profile (id INTEGER, user_uuid VARCHAR, bio VARCHAR, country VARCHAR)")
    con.executemany("INSERT INTO profiles_profile VALUES (?,?,?,?)",
                    [(i, u, f"Bio {i} {'b' * (i % 9)}", rng.choice(["FR", "DE", "US"])) for i, u in enumerate(users, 1)])
    con.execute("CREATE TABLE orders_order (id INTEGER, user_uuid VARCHAR, total DECIMAL(10,2), placed DATE)")
    con.executemany("INSERT INTO orders_order VALUES (?,?,?,?)",
                    [(i, rng.choice(users), rng.randint(100, 99999) / 100, dt.date(2023, 1, 1) + dt.timedelta(days=i % 300))
                     for i in range(1, 7001)])
    links = _links(con, "eval-profiles")
    assert ("orders_order.user_uuid", "profiles_profile.user_uuid") not in links, links


def test_a_bridge_with_its_own_number_still_reads_its_glued_names():
    rng = random.Random(8)
    con = duckdb.connect()
    con.execute("CREATE TABLE superhero (id INTEGER, superhero_name VARCHAR, height_cm INTEGER)")
    con.executemany("INSERT INTO superhero VALUES (?,?,?)",
                    [(i, f"Hero {i}", rng.randint(150, 230)) for i in range(1, 601)])
    con.execute("CREATE TABLE superpower (id INTEGER, power_name VARCHAR)")
    con.executemany("INSERT INTO superpower VALUES (?,?)", [(i, f"Power {i}") for i in range(1, 151)])
    pairs = sorted({(h, p) for h in rng.sample(range(1, 601), 520) for p in rng.sample(range(1, 151), rng.randint(1, 12))}
                   | {(1 + p % 600, p) for p in range(1, 151)})
    con.execute("CREATE TABLE hero_power (id INTEGER, hero_id INTEGER, power_id INTEGER)")
    con.executemany("INSERT INTO hero_power VALUES (?,?,?)", [(i, h, p) for i, (h, p) in enumerate(pairs, 1)])
    links = _links(con, "eval-bridge-own-id")
    assert links.get(("hero_power.hero_id", "superhero.id"), ("",))[0] == "proposed", links
    assert links.get(("hero_power.power_id", "superpower.id"), ("",))[0] == "proposed", links


def test_a_table_named_with_no_latin_letter_stops_nothing():
    rng = random.Random(1)
    con = duckdb.connect()
    con.execute('CREATE TABLE "товары" (id INTEGER, name VARCHAR, price DECIMAL(10,2))')
    con.executemany('INSERT INTO "товары" VALUES (?,?,?)',
                    [(i, f"Товар {i} {'x' * (i % 5)}", rng.randint(100, 9999) / 100) for i in range(1, 501)])
    con.execute("CREATE TABLE orders (id INTEGER, customer VARCHAR, ordered DATE)")
    con.executemany("INSERT INTO orders VALUES (?,?,?)",
                    [(i, f"C{i % 70}", dt.date(2023, 1, 1) + dt.timedelta(days=i % 300)) for i in range(1, 2001)])
    con.execute("CREATE TABLE order_item (order_id INTEGER, item_id INTEGER, qty INTEGER)")
    con.executemany("INSERT INTO order_item VALUES (?,?,?)",
                    sorted({(rng.randint(1, 2000), rng.randint(1, 500), rng.randint(1, 9)) for _ in range(6000)}))
    assert ("order_item.order_id", "orders.id") in _links(con, "eval-cyrillic")


def test_a_facts_number_reaches_its_dimensions_key_kept_as_text():
    # A warehouse loaded the customer dimension from files, its key as text; the facts keep it as a number.
    rng = random.Random(12)
    con = duckdb.connect()
    con.execute("CREATE TABLE dim_customer (customer_id VARCHAR, customer_name VARCHAR, segment VARCHAR)")
    con.executemany("INSERT INTO dim_customer VALUES (?,?,?)",
                    [(str(i), f"Customer {i} {'c' * (i % 6)}", rng.choice(["SMB", "Mid", "Ent"])) for i in range(1, 801)])
    con.execute("CREATE TABLE fact_sales (sale_id INTEGER, customer_id INTEGER, sold DATE, amount DECIMAL(10,2))")
    con.executemany("INSERT INTO fact_sales VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 800), dt.date(2023, 1, 1) + dt.timedelta(days=i % 600), rng.randint(100, 99999) / 100)
                     for i in range(1, 5001)])
    assert ("fact_sales.customer_id", "dim_customer.customer_id") in _links(con, "eval-dim-text-key")


def test_a_uuid_held_with_gaps_on_many_tables_leaves_the_named_keys_tested():
    # One table's uuid, unique there; thirty older tables keep a uuid of the same name backfilled with gaps. Each is
    # tested against it, after the named keys: first, they would use up the cap.
    rng = random.Random(6)
    con = duckdb.connect()
    con.execute("CREATE TABLE customers (id INTEGER, uuid VARCHAR, name VARCHAR)")
    con.executemany("INSERT INTO customers VALUES (?,?,?)",
                    [(i, u, f"Customer {i} {'q' * (i % 6)}") for i, u in enumerate(_uuids(rng, 900), 1)])
    for p in range(30):
        con.execute(f"CREATE TABLE archive_{p:02d} (id INTEGER, uuid VARCHAR, body VARCHAR)")
        con.executemany(f"INSERT INTO archive_{p:02d} VALUES (?,?,?)",
                        [(i, u if rng.random() < 0.9 else None, f"Entry {i} {'e' * (i % 8)}")
                         for i, u in enumerate(_uuids(rng, 200), 1)])
    con.execute("CREATE TABLE products (id INTEGER, title VARCHAR, price DECIMAL(10,2))")
    con.executemany("INSERT INTO products VALUES (?,?,?)",
                    [(i, f"Product {i} {'p' * (i % 7)}", rng.randint(100, 9999) / 100) for i in range(1, 401)])
    expected = set()
    for c in range(4):
        con.execute(f"CREATE TABLE orders_{c} (id INTEGER, customer_id INTEGER, product_id INTEGER, amount DECIMAL(10,2))")
        con.executemany(f"INSERT INTO orders_{c} VALUES (?,?,?,?)",
                        [(i, rng.randint(1, 900), rng.randint(1, 400), rng.randint(100, 99999) / 100)
                         for i in range(1, 801)])
        expected |= {(f"orders_{c}.customer_id", "customers.id"), (f"orders_{c}.product_id", "products.id")}
    links = _links(con, "eval-uuid-cap", max_join_tests=24)
    assert expected <= set(links), expected - set(links)


@pytest.mark.parametrize("key", ["card_uuid", "CardGuid"])
def test_a_uuid_named_with_its_table_is_that_tables_key(key):
    rng = random.Random(21)
    con = duckdb.connect()
    cards = _uuids(rng, 2500)
    con.execute(f"CREATE TABLE cards (id INTEGER, {key} VARCHAR, name VARCHAR)")
    con.executemany("INSERT INTO cards VALUES (?,?,?)", [(i, u, f"Card {i} {'c' * (i % 5)}") for i, u in enumerate(cards, 1)])
    con.execute(f"CREATE TABLE rulings (id INTEGER, {key} VARCHAR, ruled DATE, text VARCHAR)")
    con.executemany("INSERT INTO rulings VALUES (?,?,?,?)",
                    [(i, rng.choice(cards), dt.date(2005, 1, 1) + dt.timedelta(days=i % 400), f"Ruling {i}")
                     for i in range(1, 2001)])
    assert (f"rulings.{key}", f"cards.{key}") in _links(con, "eval-card-uuid")


@pytest.mark.parametrize("customers, copy", [("retail_customers", "stg_customer"), ("clients", "stg_customer")])
def test_a_number_kept_again_as_text_on_a_load_never_takes_the_orders_link(customers, copy):
    # The customers are kept under another name: found by the orders' values (RETAIL_CUSTOMERS shares the word), or by
    # nothing at all (CLIENTS, for the AI to read). A load's staging table keeps the same numbers as text.
    rng = random.Random(29)
    con = duckdb.connect()
    alive = sorted(rng.sample(range(1, 2201), 1600))
    con.execute(f"CREATE TABLE {customers} (id INTEGER, name VARCHAR, segment VARCHAR, created DATE)")
    con.executemany(f"INSERT INTO {customers} VALUES (?,?,?,?)",
                    [(i, f"Customer {i} {'q' * (i % 6)}", rng.choice(["SMB", "Mid", "Ent"]),
                      dt.date(2020, 1, 1) + dt.timedelta(days=i % 900)) for i in alive])
    con.execute(f"CREATE TABLE {copy} (customer_id VARCHAR, crm_ref VARCHAR, loaded DATE)")
    con.executemany(f"INSERT INTO {copy} VALUES (?,?,?)",
                    [(str(i), f"CRM-{rng.randint(10**6, 10**7)}", dt.date(2024, 1, 1)) for i in alive])
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DECIMAL(10,2), ordered DATE)")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)",
                    [(i, rng.choice(alive), rng.randint(100, 99999) / 100, dt.date(2023, 1, 1) + dt.timedelta(days=i % 500))
                     for i in range(1, 6001)])
    links = _links(con, f"eval-load-{copy}")
    assert ("orders.customer_id", f"{copy}.customer_id") not in links, links
    if customers == "retail_customers":
        assert ("orders.customer_id", "retail_customers.id") in links, links


def test_a_glued_name_is_one_word_ending_in_three_letters_or_more():
    # An order's items are the products: no table is named for the item, but an application's cart items (two words)
    # and a CRM's leads (whose end is an ad's) hold numbers in the same ranges.
    rng = random.Random(4)
    con = duckdb.connect()
    con.execute("CREATE TABLE orders (id INTEGER, customer VARCHAR, ordered DATE)")
    con.executemany("INSERT INTO orders VALUES (?,?,?)",
                    [(i, f"C{i % 70}", dt.date(2023, 1, 1) + dt.timedelta(days=i % 300)) for i in range(1, 2001)])
    pids = sorted(rng.sample(range(1, 1001), 500))
    con.execute("CREATE TABLE products (id INTEGER, title VARCHAR, price DECIMAL(10,2))")
    con.executemany("INSERT INTO products VALUES (?,?,?)",
                    [(i, f"Product {i} {'x' * (i % 7)}", rng.randint(100, 9999) / 100) for i in pids])
    con.execute("CREATE TABLE shop_cartitem (id INTEGER, product_id INTEGER, quantity INTEGER, added DATE)")
    con.executemany("INSERT INTO shop_cartitem VALUES (?,?,?,?)",
                    [(i, rng.choice(pids), rng.randint(1, 5), dt.date(2023, 1, 1) + dt.timedelta(days=i % 300))
                     for i in range(1, 1001)])
    con.execute("CREATE TABLE order_item (order_id INTEGER, item_id INTEGER, qty INTEGER)")
    con.executemany("INSERT INTO order_item VALUES (?,?,?)",
                    sorted({(rng.randint(1, 2000), rng.choice(pids), rng.randint(1, 9)) for _ in range(6000)}))
    ad_ids = sorted(rng.sample(range(1, 2401), 900))
    con.execute("CREATE TABLE ads (id INTEGER, headline VARCHAR, budget DECIMAL(10,2))")
    con.executemany("INSERT INTO ads VALUES (?,?,?)",
                    [(i, f"Headline {i} {'h' * (i % 6)}", rng.randint(1000, 90000) / 100) for i in ad_ids])
    con.execute("CREATE TABLE keywords (id INTEGER, phrase VARCHAR)")
    con.executemany("INSERT INTO keywords VALUES (?,?)", [(i, f"phrase {i} {'k' * (i % 5)}") for i in range(1, 301)])
    con.execute("CREATE TABLE ad_keyword (ad_id INTEGER, keyword_id INTEGER, bid DECIMAL(8,2))")
    con.executemany("INSERT INTO ad_keyword VALUES (?,?,?)",
                    sorted({(rng.choice(ad_ids), rng.randint(1, 300), rng.randint(10, 500) / 100) for _ in range(5000)}))
    con.execute("CREATE TABLE leads (id INTEGER, email VARCHAR, created DATE)")
    con.executemany("INSERT INTO leads VALUES (?,?,?)",
                    [(i, f"lead{i}@example.com", dt.date(2023, 1, 1) + dt.timedelta(days=i % 500)) for i in range(1, 2501)])
    links = _links(con, "eval-glued-words")
    assert ("order_item.item_id", "shop_cartitem.id") not in links, links
    assert ("ad_keyword.ad_id", "leads.id") not in links, links
    assert ("order_item.order_id", "orders.id") in links and ("ad_keyword.keyword_id", "keywords.id") in links, links


def test_the_warehouses_own_queries_never_vouch_for_a_glued_name():
    # The warehouse's queries include the answers that used the proposed link: they cannot confirm it.
    rng = random.Random(8)
    con = duckdb.connect()
    con.execute("CREATE TABLE superhero (id INTEGER, superhero_name VARCHAR, height_cm INTEGER)")
    con.executemany("INSERT INTO superhero VALUES (?,?,?)", [(i, f"Hero {i}", rng.randint(150, 230)) for i in range(1, 601)])
    con.execute("CREATE TABLE superpower (id INTEGER, power_name VARCHAR)")
    con.executemany("INSERT INTO superpower VALUES (?,?)", [(i, f"Power {i}") for i in range(1, 151)])
    pairs = {(h, p) for h in rng.sample(range(1, 601), 520) for p in rng.sample(range(1, 151), rng.randint(1, 12))}
    con.execute("CREATE TABLE hero_power (hero_id INTEGER, power_id INTEGER)")
    con.executemany("INSERT INTO hero_power VALUES (?,?)", sorted(pairs | {(1 + p % 600, p) for p in range(1, 151)}))
    queries = ["SELECT s.superhero_name, COUNT(*) FROM hero_power hp JOIN superhero s ON hp.hero_id = s.id GROUP BY 1",
               "SELECT AVG(s.height_cm) FROM hero_power hp JOIN superhero s ON s.id = hp.hero_id"]
    links = _links(con, "eval-glued-history", query_history=lambda warehouse: queries)
    assert links.get(("hero_power.hero_id", "superhero.id"), ("",))[0] == "proposed", links
    assert links.get(("hero_power.power_id", "superpower.id"), ("",))[0] == "proposed", links




def test_of_two_tables_holding_a_uuid_once_each_the_one_it_names_wins():
    rng = random.Random(13)
    con = duckdb.connect()
    devices = _uuids(rng, 400)
    con.execute("CREATE TABLE DeviceGuidMap (Id INTEGER, DeviceGuid VARCHAR, Label VARCHAR)")
    con.executemany("INSERT INTO DeviceGuidMap VALUES (?,?,?)",
                    [(i, u, f"Dev {i} {'d' * (i % 3)}") for i, u in enumerate(devices, 1)])
    con.execute("CREATE TABLE Device (Id INTEGER, DeviceGuid VARCHAR, Model VARCHAR)")
    con.executemany("INSERT INTO Device VALUES (?,?,?)", [(i, u, f"M{i % 9}") for i, u in enumerate(devices, 1)])
    con.execute("CREATE TABLE Telemetry (Id INTEGER, DeviceGuid VARCHAR, Reading DOUBLE)")
    con.executemany("INSERT INTO Telemetry VALUES (?,?,?)",
                    [(i, rng.choice(devices), rng.random() * 100) for i in range(1, 3001)])
    links = _links(con, "eval-device-guid")
    assert ("Telemetry.DeviceGuid", "Device.DeviceGuid") in links, links


@pytest.mark.parametrize("target, source, key", [("book_copies", "loans", "book_copy_id"),
                                                 ("raw_materials", "bom_lines", "raw_material_id"),
                                                 ("landing_pages", "web_sessions", "landing_page_id"),
                                                 ("archives", "documents", "archive_id"),
                                                 ("saved_searches", "search_alerts", "saved_search_id"),
                                                 ("old_age_pensions", "pension_payments", "old_age_pension_id")])
def test_a_business_table_named_with_a_copy_word_keeps_its_number_kept_as_text(target, source, key):
    # A library's book copies, a factory's raw materials: their own numbers, kept as text, pointed at as numbers.
    rng = random.Random(len(target))
    con = duckdb.connect()
    numbers = sorted(rng.sample(range(1000, 99999), 400))
    con.execute(f"CREATE TABLE {target} ({key} VARCHAR, label VARCHAR, note VARCHAR, created DATE)")
    con.executemany(f"INSERT INTO {target} VALUES (?,?,?,?)",
                    [(str(v), f"Label {v} {'x' * (v % 7)}", rng.choice("abcd"), dt.date(2023, 1, 1) + dt.timedelta(days=v % 300))
                     for v in numbers])
    con.execute(f"CREATE TABLE {source} (id INTEGER, {key} INTEGER, qty INTEGER, amount DECIMAL(10,2))")
    con.executemany(f"INSERT INTO {source} VALUES (?,?,?,?)",
                    [(i, rng.choice(numbers), rng.randint(1, 9), rng.randint(100, 99999) / 100) for i in range(1, 3001)])
    assert (f"{source}.{key}", f"{target}.{key}") in _links(con, f"eval-copy-word-{target}")


@pytest.mark.parametrize("competitors", [300, 250])
def test_a_number_kept_as_text_under_its_own_name_beats_a_chance_match_on_values(competitors):
    # The chain's stores keep their codes as text; a market study numbers competitor stores 1, 2, 3 (sharing the word,
    # and every code, or nearly): the codes' own name says which.
    rng = random.Random(1)
    con = duckdb.connect()
    codes = sorted(rng.sample(range(1, 301), 80))
    con.execute("CREATE TABLE stores (store_code VARCHAR, store_name VARCHAR, city VARCHAR)")
    con.executemany("INSERT INTO stores VALUES (?,?,?)",
                    [(str(c), f"Store {c} {'s' * (c % 7)}", rng.choice(["Lyon", "Paris", "Nice"])) for c in codes])
    con.execute("CREATE TABLE sales (sale_id INTEGER, store_code INTEGER, amount DECIMAL(10,2), qty INTEGER)")
    con.executemany("INSERT INTO sales VALUES (?,?,?,?)",
                    [(i, rng.choice(codes), rng.randint(100, 99999) / 100, rng.randint(1, 9)) for i in range(1, 4001)])
    con.execute("CREATE TABLE competitor_stores (id INTEGER, store_name VARCHAR, chain VARCHAR, sqm INTEGER)")
    con.executemany("INSERT INTO competitor_stores VALUES (?,?,?,?)",
                    [(i, f"Competitor {i} {'c' * (i % 5)}", rng.choice(["North", "South"]), rng.randint(200, 5000))
                     for i in range(1, competitors + 1)])
    links = _links(con, f"eval-competitors-{competitors}")
    assert ("sales.store_code", "stores.store_code") in links, links


def test_a_hashtags_tags_are_read_from_its_glued_name():
    rng = random.Random(2)
    con = duckdb.connect()
    con.execute("CREATE TABLE posts (id INTEGER, body VARCHAR, posted DATE, likes INTEGER)")
    con.executemany("INSERT INTO posts VALUES (?,?,?,?)",
                    [(i, f"Post {i} {'b' * (i % 9)}", dt.date(2024, 1, 1) + dt.timedelta(days=i % 200), rng.randint(0, 900))
                     for i in range(1, 1201)])
    con.execute("CREATE TABLE hashtags (id INTEGER, name VARCHAR)")
    con.executemany("INSERT INTO hashtags VALUES (?,?)", [(i, f"#tag{i}{'z' * (i % 4)}") for i in range(1, 301)])
    con.execute("CREATE TABLE post_tag (post_id INTEGER, tag_id INTEGER)")
    con.executemany("INSERT INTO post_tag VALUES (?,?)",
                    sorted({(rng.randint(1, 1200), rng.randint(1, 300)) for _ in range(4000)}))
    links = _links(con, "eval-hashtags")
    assert links.get(("post_tag.tag_id", "hashtags.id"), ("",))[0] == "proposed", links


def test_a_bridge_named_for_two_things_is_no_fuller_name_for_a_code():
    # Each product names its main category's code; a bridge lists the other categories, keeping no code of its own.
    rng = random.Random(5)
    con = duckdb.connect()
    con.execute("CREATE TABLE categories (id INTEGER, code INTEGER, name VARCHAR)")
    con.executemany("INSERT INTO categories VALUES (?,?,?)",
                    [(i, 4000 + i * 13, f"Category {i} {'g' * (i % 5)}") for i in range(1, 121)])
    con.execute("CREATE TABLE products (id INTEGER, name VARCHAR, product_category_code INTEGER, price DECIMAL(8,2))")
    con.executemany("INSERT INTO products VALUES (?,?,?,?)",
                    [(i, f"Product {i} {'p' * (i % 9)}", 4000 + rng.randint(1, 120) * 13, rng.randint(100, 9999) / 100)
                     for i in range(1, 2001)])
    con.execute("CREATE TABLE product_categories (product_id INTEGER, category_id INTEGER)")
    con.executemany("INSERT INTO product_categories VALUES (?,?)",
                    sorted({(rng.randint(1, 2000), rng.randint(1, 120)) for _ in range(3000)}))
    assert ("products.product_category_code", "categories.code") in _links(con, "eval-category-bridge")


def test_the_warehouses_queries_say_which_way_two_uuid_tables_link():
    # One row each, both keyed by the card's uuid: only the queries say the prices are the cards'.
    rng = random.Random(3)
    con = duckdb.connect()
    cards = _uuids(rng, 900)
    con.execute("CREATE TABLE cards (card_uuid VARCHAR, name VARCHAR, rarity VARCHAR)")
    con.executemany("INSERT INTO cards VALUES (?,?,?)",
                    [(u, f"Card {i} {'c' * (i % 7)}", rng.choice(["common", "rare"])) for i, u in enumerate(cards)])
    con.execute("CREATE TABLE card_prices (uuid VARCHAR, price DECIMAL(8,2), currency VARCHAR)")
    con.executemany("INSERT INTO card_prices VALUES (?,?,?)",
                    [(u, rng.randint(10, 9999) / 100, rng.choice(["EUR", "USD"])) for u in cards[:850]])
    queries = ["SELECT c.name, p.price FROM cards c JOIN card_prices p ON p.uuid = c.card_uuid",
               "SELECT AVG(p.price) FROM card_prices p JOIN cards c ON c.card_uuid = p.uuid WHERE c.rarity = 'rare'"]
    links = _links(con, "eval-uuid-history", query_history=lambda warehouse: queries)
    assert ("card_prices.uuid", "cards.card_uuid") in links, links


@pytest.mark.parametrize("glued, word", [("ebooks", "book"), ("emails", "mail")])
def test_a_glued_name_after_one_letter_is_read(glued, word):
    rng = random.Random(3)
    con = duckdb.connect()
    con.execute(f"CREATE TABLE {glued} (id INTEGER, title VARCHAR, created DATE)")
    con.executemany(f"INSERT INTO {glued} VALUES (?,?,?)",
                    [(i, f"Title {i} {'t' * (i % 9)}", dt.date(2020, 1, 1) + dt.timedelta(days=i % 900)) for i in range(1, 1501)])
    con.execute("CREATE TABLE authors (id INTEGER, name VARCHAR, country VARCHAR)")
    con.executemany("INSERT INTO authors VALUES (?,?,?)",
                    [(i, f"Author {i} {'a' * (i % 6)}", rng.choice(["FR", "UK", "US"])) for i in range(1, 401)])
    pairs = sorted({(rng.randint(1, 400), b) for b in range(1, 1501)} | {(rng.randint(1, 400), rng.randint(1, 1500))
                                                                       for _ in range(300)})
    con.execute(f"CREATE TABLE author_{word} (author_id INTEGER, {word}_id INTEGER)")
    con.executemany(f"INSERT INTO author_{word} VALUES (?,?)", pairs)
    links = _links(con, f"eval-glued-{glued}")
    assert links.get((f"author_{word}.{word}_id", f"{glued}.id"), ("",))[0] == "proposed", links
