"""Public sample warehouses, scored by the benchmark next to the synthetic ones.

The synthetic domains plant what we already know to look for. These three were designed by
other people for other purposes, which is their worth: a music store (Chinook), a trading
company (Northwind) and a DVD rental chain (Sakila). Their data is not kept in the repository.
It is fetched once, checked against the hash written here, and cached outside the repository:

    python -m evals.core2.public fetch          # into ~/.cache/querybot/benchmark (or $QUERYBOT_BENCHMARK_DATA)
    python -m evals.core2.benchmark             # scores them with every other domain once cached

Each database is read as published (SQLite), its tables and columns named in snake case
(InvoiceLine.UnitPrice -> invoice_line.unit_price), then put through the same naming styles
as the synthetic domains. The truth below is written by hand from the schema and the rows:
every link, date, metric and label in it was checked against the data.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import sqlite3
import ssl
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from evals.core2.framework import DateTruth, Domain, JoinTruth, MeasureTruth, TableDef, Truth


@dataclass(frozen=True)
class Source:
    url: str
    sha256: str
    file: str
    description: str


SOURCES = {
    "chinook": Source(
        "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/"
        "Chinook_Sqlite.sqlite",
        "7651ba378ac2fcd0dfc3c66fb101f7a7eed3ba39a612ec642b96e20702061f15", "chinook.sqlite",
        "A digital music store: invoices and their lines, tracks, albums, artists, playlists, staff."),
    "northwind": Source(
        "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db",
        "2f4f5c68dfcd33ba27373eae48c7a4869800c68095ee0f9f0da494f83382a877", "northwind.sqlite",
        "A trading company: orders and order lines, products, suppliers, shippers, employees and territories."),
    "sakila": Source(
        "https://raw.githubusercontent.com/bradleygrant/sakila-sqlite3/main/sakila_master.db",
        "88c91a4a1a6b61f9d3f35904c0a173c887b25e73f20c3c2fdb073818c06f4268", "sakila.sqlite",
        "A DVD rental chain: rentals, payments, the copies in each store, films, actors, customers, staff."),
}

# Columns left out as read: pictures, and a password column a warehouse would never expose.
_LEFT_OUT = {"sakila": {"staff.password", "staff.picture"}}


def cache_dir() -> Path:
    return Path(os.environ.get("QUERYBOT_BENCHMARK_DATA") or Path.home() / ".cache" / "querybot" / "benchmark")


def cached() -> list[str]:
    """The public warehouses fetched into the cache."""
    return [name for name, source in SOURCES.items() if (cache_dir() / source.file).is_file()]


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(names: list[str] | None = None) -> list[Path]:
    """Download each warehouse once, refusing any file whose hash is not the one written here."""
    cache_dir().mkdir(parents=True, exist_ok=True)
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    context = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    out = []
    for name in names or list(SOURCES):
        source = SOURCES[name]
        target = cache_dir() / source.file
        if target.is_file() and _digest(target) == source.sha256:
            out.append(target)
            continue
        with tempfile.NamedTemporaryFile(dir=cache_dir(), delete=False) as tmp, \
                urllib.request.urlopen(source.url, context=context, timeout=120) as response:  # noqa: S310
            while block := response.read(1 << 20):
                tmp.write(block)
        got = _digest(Path(tmp.name))
        if got != source.sha256:
            os.unlink(tmp.name)
            raise ValueError(f"{name}: the download's hash is {got}, not the {source.sha256} it was checked as")
        os.replace(tmp.name, target)
        out.append(target)
    return out


# ── reading a SQLite database as a domain's tables ───────────────────────────


def snake(name: str) -> str:
    """InvoiceLine -> invoice_line, CustomerID -> customer_id, "Order Details" -> order_details."""
    name = re.sub(r"[\s\-]+", "_", name.strip())
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return re.sub(r"_+", "_", name).lower()


def _duck_type(declared: str, values: pd.Series) -> str | None:
    """The warehouse type a SQLite column's declared type stands for (None: left out)."""
    t = declared.upper().strip()
    if t == "BLOB":
        return None
    if "INT" in t:
        return "INTEGER"
    if m := re.match(r"(?:NUMERIC|DECIMAL)\s*\((\d+)\s*,\s*(\d+)\)", t):
        return f"DECIMAL({m.group(1)},{m.group(2)})"
    if t in ("NUMERIC", "DECIMAL"):
        numbers = pd.to_numeric(values, errors="coerce").dropna()
        return "INTEGER" if len(numbers) and (numbers == numbers.round()).all() else "DECIMAL(18,4)"
    if t in ("REAL", "FLOAT", "DOUBLE"):
        return "DOUBLE"
    if t in ("DATETIME", "TIMESTAMP"):
        return "TIMESTAMP"
    if t == "DATE":
        return "DATE"
    return "VARCHAR"


def _typed(values: pd.Series, duck: str) -> pd.Series:
    if duck == "INTEGER":
        return pd.to_numeric(values, errors="raise").astype("Int64")
    if duck.startswith("DECIMAL") or duck == "DOUBLE":
        return pd.to_numeric(values, errors="raise").astype("float64")
    if duck == "TIMESTAMP":
        return pd.to_datetime(values, format="mixed")
    if duck == "DATE":
        return pd.to_datetime(values, format="mixed").dt.date
    return values.map(lambda v: None if v is None or v != v else str(v))


def read_sqlite(path: Path, *, left_out: set[str] = frozenset()) -> dict[str, tuple[pd.DataFrame, dict[str, str], list[str]]]:
    """Every table with rows: logical name -> (rows, column types, primary key), in snake case."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        out = {}
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for table in names:
            info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
            frame = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
            if frame.empty:
                continue
            logical = snake(table)
            data, types = {}, {}
            for _, column, declared, _, _, _ in info:
                name = snake(column)
                if f"{logical}.{name}" in left_out:
                    continue
                duck = _duck_type(declared or "", frame[column])
                if duck is None:
                    continue
                data[name] = _typed(frame[column], duck)
                types[name] = duck
            key = [snake(r[1]) for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
            out[logical] = (pd.DataFrame(data), types, key)
        return out
    finally:
        con.close()


# ── the truth, by hand ───────────────────────────────────────────────────────


def _j(from_ref: str, to_ref: str, role: str | None = None) -> JoinTruth:
    (ft, fc), (tt, tc) = from_ref.split("."), to_ref.split(".")
    return JoinTruth(ft, fc, tt, tc, role=role)


def _count(table: str, name: str) -> MeasureTruth:
    return MeasureTruth(table, None, "count", "additive", "count", name)


def _audit(tables: list[str]) -> list[DateTruth]:
    return [DateTruth(t, "last_update", "Last update", "audit", False) for t in tables]


TRUTH = {
    "chinook": {
        "today": dt.date(2026, 1, 1),
        "kinds": {"invoice": "fact", "invoice_line": "fact", "playlist_track": "bridge"},
        "joins": [
            _j("album.artist_id", "artist.artist_id"),
            _j("customer.support_rep_id", "employee.employee_id", "Support rep"),
            _j("employee.reports_to", "employee.employee_id", "Reports to"),
            _j("invoice.customer_id", "customer.customer_id"),
            _j("invoice_line.invoice_id", "invoice.invoice_id"),
            _j("invoice_line.track_id", "track.track_id"),
            _j("playlist_track.playlist_id", "playlist.playlist_id"),
            _j("playlist_track.track_id", "track.track_id"),
            _j("track.album_id", "album.album_id"),
            _j("track.media_type_id", "media_type.media_type_id"),
            _j("track.genre_id", "genre.genre_id"),
        ],
        "dates": [
            DateTruth("invoice", "invoice_date", "Invoice date", "event", False, default=True),
            DateTruth("employee", "hire_date", "Hire date", "event", False, default=True),
            DateTruth("employee", "birth_date", "Birth date", "event", False),
        ],
        "measures": [
            _count("invoice", "Invoices"),
            MeasureTruth("invoice", "total", "sum", "additive", "currency", "Invoice total"),
            _count("invoice_line", "Invoice lines"),
            MeasureTruth("invoice_line", "quantity", "sum", "additive", "integer", "Quantity"),
            MeasureTruth("invoice_line", "unit_price", "avg", "non_additive", "currency", "Unit price"),
        ],
        "not_measures": ["invoice.customer_id", "invoice_line.invoice_id", "invoice_line.track_id",
                         "playlist_track.playlist_id", "playlist_track.track_id"],
        "labels": {"artist": "name", "album": "title", "genre": "name", "media_type": "name"},
    },
    "northwind": {
        "today": dt.date(2023, 11, 1),
        "kinds": {"orders": "fact", "order_details": "fact", "employee_territories": "bridge"},
        "joins": [
            _j("employees.reports_to", "employees.employee_id", "Reports to"),
            _j("employee_territories.employee_id", "employees.employee_id"),
            _j("employee_territories.territory_id", "territories.territory_id"),
            _j("order_details.order_id", "orders.order_id"),
            _j("order_details.product_id", "products.product_id"),
            _j("orders.customer_id", "customers.customer_id"),
            _j("orders.employee_id", "employees.employee_id"),
            _j("orders.ship_via", "shippers.shipper_id", "Shipper"),
            _j("products.supplier_id", "suppliers.supplier_id"),
            _j("products.category_id", "categories.category_id"),
            _j("territories.region_id", "regions.region_id"),
        ],
        "dates": [
            DateTruth("orders", "order_date", "Order date", "event", False, default=True),
            DateTruth("orders", "required_date", "Required date", "due", False),
            DateTruth("orders", "shipped_date", "Shipped date", "event", False),
            DateTruth("employees", "hire_date", "Hire date", "event", False, default=True),
            DateTruth("employees", "birth_date", "Birth date", "event", False),
        ],
        "measures": [
            _count("orders", "Orders"),
            MeasureTruth("orders", "freight", "sum", "additive", "currency", "Freight"),
            _count("order_details", "Order lines"),
            MeasureTruth("order_details", "quantity", "sum", "additive", "integer", "Quantity"),
            MeasureTruth("order_details", "unit_price", "avg", "non_additive", "currency", "Unit price"),
            MeasureTruth("order_details", "discount", "avg", "non_additive", "percent", "Discount"),
        ],
        "not_measures": ["orders.employee_id", "orders.ship_via", "order_details.order_id",
                         "order_details.product_id", "employee_territories.employee_id"],
        "labels": {"categories": "category_name", "products": "product_name", "shippers": "company_name",
                   "suppliers": "company_name", "regions": "region_description"},
    },
    "sakila": {
        "today": dt.date(2006, 3, 1),
        "kinds": {"rental": "fact", "payment": "fact", "film_actor": "bridge", "film_category": "bridge"},
        "joins": [
            _j("address.city_id", "city.city_id"),
            _j("city.country_id", "country.country_id"),
            _j("customer.address_id", "address.address_id"),
            _j("customer.store_id", "store.store_id"),
            _j("film.language_id", "language.language_id"),
            _j("film_actor.actor_id", "actor.actor_id"),
            _j("film_actor.film_id", "film.film_id"),
            _j("film_category.film_id", "film.film_id"),
            _j("film_category.category_id", "category.category_id"),
            _j("inventory.film_id", "film.film_id"),
            _j("inventory.store_id", "store.store_id"),
            _j("payment.customer_id", "customer.customer_id"),
            _j("payment.staff_id", "staff.staff_id"),
            _j("payment.rental_id", "rental.rental_id"),
            _j("rental.inventory_id", "inventory.inventory_id"),
            _j("rental.customer_id", "customer.customer_id"),
            _j("rental.staff_id", "staff.staff_id"),
            _j("staff.address_id", "address.address_id"),
            _j("staff.store_id", "store.store_id"),
            _j("store.address_id", "address.address_id"),
            _j("store.manager_staff_id", "staff.staff_id", "Manager"),
        ],
        "dates": [
            DateTruth("rental", "rental_date", "Rental date", "event", False, default=True),
            DateTruth("rental", "return_date", "Return date", "event", False),
            DateTruth("payment", "payment_date", "Payment date", "event", False, default=True),
            # every customer row was created within one second of the others: a load stamp, whatever its name
            DateTruth("customer", "create_date", "Created", "audit", False),
            *_audit(["actor", "address", "category", "city", "country", "customer", "film", "film_actor",
                     "film_category", "inventory", "language", "payment", "rental", "staff", "store"]),
        ],
        "measures": [
            _count("rental", "Rentals"),
            _count("payment", "Payments"),
            MeasureTruth("payment", "amount", "sum", "additive", "currency", "Amount paid"),
        ],
        "not_measures": ["payment.customer_id", "payment.staff_id", "payment.rental_id", "rental.inventory_id",
                         "rental.customer_id", "rental.staff_id"],
        "labels": {"category": "name", "language": "name", "country": "country", "film": "title"},
    },
}


def build(name: str) -> Domain:
    """A cached public warehouse as a domain, with its hand-written truth."""
    source = SOURCES[name]
    path = cache_dir() / source.file
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not fetched: python -m evals.core2.public fetch {name}")
    got = _digest(path)
    if got != source.sha256:
        raise ValueError(f"{path} is not the file the truth was written for (hash {got})")
    spec = TRUTH[name]
    tables = []
    for logical, (frame, types, key) in read_sqlite(path, left_out=_LEFT_OUT.get(name, set())).items():
        kind = spec["kinds"].get(logical, "dimension")
        tables.append(TableDef(logical, kind, frame, types, key))
    truth = Truth(
        kinds={t.name: t.kind for t in tables}, primary_keys={t.name: t.primary_key for t in tables},
        joins=spec["joins"], calendar=None, dates=spec["dates"], measures=spec["measures"],
        labels=spec["labels"], not_measures=spec["not_measures"])
    return Domain(name=name, tables=tables, truth=truth, today=spec["today"], description=source.description)


def main(argv: list[str]) -> int:
    if argv[:1] == ["fetch"]:
        for path in fetch(argv[1:] or None):
            print(path)
        return 0
    if argv[:1] == ["list"]:
        for name, source in SOURCES.items():
            print(f"{name:10} {'cached' if name in cached() else 'not fetched':12} {source.description}")
        return 0
    print("usage: python -m evals.core2.public fetch [name ...] | list", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
