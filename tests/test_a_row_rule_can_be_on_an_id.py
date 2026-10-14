"""A row rule can be on an ID column: a member's own key, or the key a table's rows point at it with.

Row access offered only the names, codes and categories the new core learned: a doctor's revenue rows could be
limited by the doctor's name or code, never by DOCTOR_ID, which is often the only field the people and the rows
share. The ID columns are now offered too:

* a member's own key ("Customer › ID") narrows the members and every table linked to them;
* the key a table's rows point at a member with ("Order › Customer ID") narrows those rows, never the list of
  members;
* an ID kept as a number is compared as one: a value that is no number is refused when the rule is saved, and a
  person's own value that is no number matches no row, where compared as written it failed every question.

Invented data only: the shop of the row access tests, twenty customers and six hundred orders.
"""

from __future__ import annotations

from tests.test_a_row_rule_reaches_every_table_that_links_to_it import _call, _rule, shop  # noqa: F401

CUSTOMERS = "SELECT COUNT(*) AS n FROM customers"


def _fields(shop) -> dict[str, dict]:
    from admin import row_access
    from core2.bootstrap.service import load_model

    model = load_model(shop.account, shop.store.get_client(shop.account)["db_config_id"])
    return {f["label"]: f for f in row_access.fields_of(model)}


def _save(shop, body: dict) -> dict:
    from admin import row_access

    return _call(row_access.row_rule_save, f"/admin/clients/{shop.account}/row-access/api/rules", shop.account,
                 body=body)


def test_the_id_columns_are_offered(shop):
    fields = _fields(shop)
    assert fields["Customer › ID"]["field"] == "customer_id" and fields["Customer › ID"]["numeric"]
    assert fields["Order › Customer ID"]["table"].endswith("ORDERS")
    # Not a table's own row number, nor an amount.
    assert not {f["field"] for f in fields.values()} & {"order_id", "order_amount"}
    # The names, codes and categories stay.
    assert {"Customer", "Customer › Segment", "Order › Region"} <= set(fields)


def test_a_rule_on_a_members_id_narrows_the_members_and_their_rows(shop):
    ada = shop.person("Ada")
    field = _fields(shop)["Customer › ID"]
    saved = _save(shop, {"who": "user", "whom": str(ada["id"]), "table": field["table"], "field": "customer_id",
                         "sees": "values", "values": "1, 4"})
    assert saved["ok"], saved
    assert shop.total(ada) == shop.truth("c.customer_id IN (1, 4)") < shop.truth()
    assert shop.total(ada, CUSTOMERS) == 2
    assert shop.total(shop.person("Ben")) == shop.truth(), "the rule is Ada's alone"


def test_a_rule_on_the_id_rows_point_with_narrows_those_rows_alone(shop):
    ada = shop.person("Ada")
    field = _fields(shop)["Order › Customer ID"]
    assert _save(shop, {"who": "user", "whom": str(ada["id"]), "table": field["table"], "field": "customer_id",
                        "sees": "values", "values": ["1", "4"]})["ok"]
    assert shop.total(ada) == shop.truth("c.customer_id IN (1, 4)")
    assert shop.total(ada, CUSTOMERS) == 20, "the list of customers is not the orders' to narrow"


def test_an_id_that_is_no_number_is_refused_when_the_rule_is_saved(shop):
    field = _fields(shop)["Customer › ID"]
    refused = _save(shop, {"who": "everyone", "table": field["table"], "field": "customer_id", "sees": "values",
                           "values": "1, C-4"})
    assert not refused.get("ok") and "Customer › ID holds numbers" in str(refused) and "C-4" in str(refused)
    assert shop.store.list_row_rules(shop.account) == []


def test_a_persons_own_id_is_read_as_a_number_and_one_that_is_not_matches_no_row(shop):
    field = _fields(shop)["Customer › ID"]
    assert _save(shop, {"who": "role", "whom": "analyst", "table": field["table"], "field": "customer_id",
                        "sees": "own", "key": "customer"})["ok"]
    ada, ben, cy = shop.person("Ada"), shop.person("Ben"), shop.person("Cy")
    shop.store.set_user_attributes(ada["id"], {"customer": "4"})
    shop.store.set_user_attributes(ben["id"], {"customer": "C-4"})        # typed wrong: no row, and no failure
    assert shop.total(shop.store.get_user(ada["id"])) == shop.truth("c.customer_id = 4")
    assert shop.total(shop.store.get_user(ben["id"])) == 0
    assert shop.total(cy) == 0, "no value of their own: none of the rows"


def test_a_rule_written_before_on_a_category_is_unchanged(shop):
    """No "numeric" mark on a text field: its values are compared as written."""
    ada = shop.person("Ada")
    field = _fields(shop)["Customer › Segment"]
    assert not field["numeric"]
    assert _save(shop, {"who": "user", "whom": str(ada["id"]), "table": field["table"], "field": "segment",
                        "sees": "values", "values": "Gold"})["ok"]
    kept = shop.store.list_row_rules(shop.account)[0]["condition"]
    assert kept["value"] == ["Gold"] and "numeric" not in kept
    assert shop.total(ada) == shop.truth("c.segment = 'Gold'")
