"""A number is a link only when something more than its values says so.

The accuracy benchmark (evals/core2/benchmark.py) showed Learn inventing links: a visit's minutes
(30..299) sat inside the customers' keys, line and fill numbers (1..4) inside a six-row partners
table, "rental duration" was read as pointing at rentals, "quantity prescribed" at the prescribers,
and a device's uplink at a backup copy of the devices. A question that follows such a link answers
with the wrong numbers, and says nothing.

So: a name's leftover words must end in a key word ("rental duration" is a figure about rentals),
a count word makes a figure ("quantity prescribed"); a line number in a two-column key points
nowhere; values alone must use most of a small table, a run of numbers must reach the target's
newest key, and a partial match must use nearly all of its members; a backup copy is never a target.
What must still be found is found: every item on every day of a balance, a playlist of a track, a
link whose few strays are customers not loaded yet, and a device's uplink to another device.

Invented data only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, learn
from core2.bootstrap.inventory import from_duckdb
from core2.bootstrap.joins import name_score
from core2.warehouse.runner import DuckDBWarehouse


def _learn(con: duckdb.DuckDBPyConnection):
    warehouse = DuckDBWarehouse(con)
    inventory = from_duckdb(warehouse)
    found = learn(warehouse, inventory, BuildOptions(workers=1))
    name_of = {key: t.name for key, t in inventory.tables.items()}
    links = {(name_of[j.from_table], j.from_column): (name_of[j.to_table], j.trust) for j in found.joins
             if j.trust != "rejected"}
    keys = {name_of[key]: k for key, k in found.keys.items()}
    return links, keys, found


@pytest.mark.parametrize("column,table,key,points", [
    ("rental_duration", "rental", "rental_id", False),        # how long a rental lasts
    ("visits_included", "visits", "visit_id", False),         # how many visits a plan holds
    ("QTY_PRSC", "PRSC", "PRSC_KEY", False),                  # quantity prescribed, beside the prescriber
    ("total_customers", "customers", "customer_id", False),
    ("ABC_CLASS_VOLUME_KEY", "ABC_CLASS", "ABC_CLASS_KEY", True),
    ("bill_to_customer_id", "customers", "customer_id", True),
    ("ship_to_customer", "customers", "customer_id", True),
    ("customer_id", "customers", "customer_id", True),
])
def test_what_the_names_say(column, table, key, points):
    assert (name_score(column, table, key)[0] > 0) is points


@pytest.fixture(scope="module")
def opaque():
    """Names that say nothing (T01.C02) and no declared keys: only the values speak."""
    con = duckdb.connect()
    # Customers 1..900, partners 1..6.
    con.execute("CREATE TABLE T01 (C01 INTEGER, C02 VARCHAR)")
    con.execute("INSERT INTO T01 SELECT i, 'Customer ' || i FROM range(1, 901) t(i)")
    con.execute("CREATE TABLE T02 (C01 INTEGER, C02 VARCHAR)")
    con.execute("INSERT INTO T02 SELECT i, 'Partner ' || i FROM range(1, 7) t(i)")
    # Booking lines: booking number and line 1..4 (bookings differ in lines), a customer, an amount.
    con.execute("CREATE TABLE T03 (C01 VARCHAR, C02 INTEGER, C03 INTEGER, C04 DECIMAL(10,2))")
    con.execute("INSERT INTO T03 SELECT 'BK' || (100000 + b), l, (b * 7919) % 900 + 1, (b * l) % 97 + 10.5 "
                "FROM range(1, 3001) a(b), range(1, 5) c(l) WHERE l <= b % 4 + 1")
    # Visits: an id, minutes 30..299, a customer, the visits a plan includes (1, 2 or 4).
    con.execute("CREATE TABLE T04 (C01 INTEGER, C02 INTEGER, C03 INTEGER, C04 INTEGER)")
    con.execute("INSERT INTO T04 SELECT i, 30 + i % 270, (i * 104729) % 900 + 1, [1, 2, 4][i % 3 + 1] "
                "FROM range(1, 5001) t(i)")
    # Returns: customers 1..880 and 40 not loaded yet (901..940): a real link with strays.
    con.execute("CREATE TABLE T05 (C01 INTEGER, C02 INTEGER, C03 DECIMAL(10,2))")
    con.execute("INSERT INTO T05 SELECT i, CASE WHEN i % 25 = 0 THEN 901 + i % 40 ELSE i % 880 + 1 END, i % 50 + 1.5 "
                "FROM range(1, 4001) t(i)")
    # A third of the customers and 20 numbers found nowhere: the values only overlap.
    con.execute("CREATE TABLE T07 (C01 INTEGER, C02 INTEGER, C03 DECIMAL(10,2))")
    con.execute("INSERT INTO T07 SELECT i, CASE WHEN i % 40 = 0 THEN 2000 + i % 20 ELSE (i * 3) % 900 + 1 END, "
                "i % 30 + 2.5 FROM range(1, 3001) t(i)")
    # Partner referrals using every partner: a link to a small table, by values alone.
    con.execute("CREATE TABLE T06 (C01 INTEGER, C02 INTEGER, C03 DECIMAL(10,2))")
    con.execute("INSERT INTO T06 SELECT i, i % 6 + 1, i % 40 + 0.5 FROM range(1, 2001) t(i)")
    return _learn(con)


def test_a_line_number_in_a_two_column_key_points_nowhere(opaque):
    links, keys, _ = opaque
    assert keys["T03"].line_numbers == ["C02"]
    assert ("T03", "C02") not in links, "lines 1..4 sit inside the six partners' keys by chance"
    assert links[("T03", "C03")][0] == "T01"


def test_minutes_and_counts_are_not_links(opaque):
    links, _, _ = opaque
    assert ("T04", "C02") not in links, "minutes 30..299 are a run of numbers inside the customers' keys"
    assert ("T04", "C04") not in links, "1, 2 and 4 use half of a six-row table"
    assert links[("T04", "C03")][0] == "T01"


def test_a_link_whose_strays_are_members_not_loaded_yet_is_kept(opaque):
    links, _, _ = opaque
    assert links[("T05", "C02")] == ("T01", "proposed")


def test_a_partial_match_on_a_few_members_is_not_a_link(opaque):
    links, _, _ = opaque
    assert ("T07", "C02") not in links


def test_a_small_table_used_in_full_is_linked_by_values(opaque):
    links, _, _ = opaque
    assert links[("T06", "C02")][0] == "T02"


def test_every_item_on_every_day_is_a_list_of_items_not_lines():
    con = duckdb.connect()
    con.execute("CREATE TABLE items (item_id INTEGER, item_name VARCHAR)")
    con.execute("INSERT INTO items SELECT i, 'Item ' || i FROM range(1, 21) t(i)")
    con.execute("CREATE TABLE daily_balances (day_key INTEGER, item_id INTEGER, on_hand INTEGER)")
    con.execute("INSERT INTO daily_balances SELECT 20260101 + d, i, (d * i) % 300 "
                "FROM range(0, 60) a(d), range(1, 21) b(i)")
    links, keys, _ = _learn(con)
    assert keys["daily_balances"].line_numbers == []
    assert links[("daily_balances", "item_id")][0] == "items"


def test_a_tracks_playlists_are_not_its_lines():
    """Playlist 1 holds every track, so each track's playlists start at 1; they do not run without a gap."""
    con = duckdb.connect()
    con.execute("CREATE TABLE playlist (playlist_id INTEGER, playlist_name VARCHAR)")
    con.execute("INSERT INTO playlist SELECT i, 'Playlist ' || i FROM range(1, 19) t(i)")
    con.execute("CREATE TABLE track (track_id INTEGER, track_name VARCHAR)")
    con.execute("INSERT INTO track SELECT i, 'Track ' || i FROM range(1, 501) t(i)")
    con.execute("CREATE TABLE playlist_track (playlist_id INTEGER, track_id INTEGER)")
    con.execute("INSERT INTO playlist_track SELECT p, t FROM range(1, 19) a(p), range(1, 501) b(t) "
                "WHERE p = 1 OR (p IN (5, 8, 13) AND t % p = 0)")
    links, keys, _ = _learn(con)
    assert keys["playlist_track"].line_numbers == []
    assert links[("playlist_track", "playlist_id")][0] == "playlist"


def test_a_fill_number_beside_the_fills_date_numbers_the_fills():
    """The prescription and the fill's date were found unique first; the fill number still runs within it."""
    con = duckdb.connect()
    con.execute("CREATE TABLE payers (payer_id INTEGER, payer_name VARCHAR)")
    con.execute("INSERT INTO payers SELECT i, 'Payer ' || i FROM range(1, 7) t(i)")
    con.execute("CREATE TABLE fills (rx VARCHAR, fill_date DATE, fill_number INTEGER, quantity INTEGER)")
    con.execute("INSERT INTO fills SELECT 'RX' || r, DATE '2025-01-01' + (r % 60 + f * 30)::INTEGER, f, 30 "
                "FROM range(1, 1501) a(r), range(1, 5) b(f) WHERE f <= r % 4 + 1")
    links, keys, _ = _learn(con)
    assert keys["fills"].line_numbers == ["fill_number"]
    assert ["rx", "fill_number"] in keys["fills"].alternate_keys
    assert ("fills", "fill_number") not in links


def test_a_backup_copy_is_never_the_target_and_an_uplink_is_a_device():
    con = duckdb.connect()
    con.execute("CREATE TABLE devices (device_id INTEGER, hostname VARCHAR, uplink_device_id INTEGER)")
    con.execute("INSERT INTO devices SELECT i, 'sw' || i, CASE WHEN i > 8 THEN (i % 8) + 1 END FROM range(1, 154) t(i)")
    con.execute("CREATE TABLE devices_bak AS SELECT * FROM devices WHERE device_id <= 142")
    con.execute("CREATE TABLE interfaces (interface_id INTEGER, device_id INTEGER, interface_name VARCHAR)")
    con.execute("INSERT INTO interfaces SELECT i, (i - 1) // 4 + 1, 'Eth1/' || ((i - 1) % 4 + 1) FROM range(1, 569) t(i)")
    links, _, _ = _learn(con)
    assert links[("interfaces", "device_id")][0] == "devices"
    assert links[("devices", "uplink_device_id")][0] == "devices"
    assert not any(target == "devices_bak" for target, _ in links.values())


def test_readings_with_no_link_still_have_their_counters():
    """An interface's counters every 15 minutes, where no link says what each reading is of: the column
    that, with the time, identifies a reading is found, counts nothing, and the counters run within it."""
    con = duckdb.connect()
    # Octets (unique at every reading, so read as the key) and errors, which rise every third reading.
    con.execute("CREATE TABLE T01 (C01 TIMESTAMP, C02 INTEGER, C03 BIGINT, C04 INTEGER)")
    con.execute("INSERT INTO T01 SELECT TIMESTAMP '2026-03-01 00:00:00' + INTERVAL (s * 15) MINUTE, i, "
                "i * 1000000 + s * (i * 937 + 11), i * 100 + s // 3 FROM range(0, 200) a(s), range(1, 41) b(i)")
    _, keys, found = _learn(con)
    measures = {m.column: m for m in found.measures}
    assert ["C02", "C01"] in keys["T01"].alternate_keys
    assert "C02" not in measures, "what each reading is of counts nothing"
    assert (measures["C04"].agg, measures["C04"].additivity) == ("max", "non_additive")
