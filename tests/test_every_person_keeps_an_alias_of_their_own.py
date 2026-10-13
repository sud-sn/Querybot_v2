"""Every person keeps an alias of their own, and so does every identifier.

A name a reader is not cleared to see becomes a stable alias, "Patient K-3F2A9C1D": a letter and a code from a keyed
hash of the name. The code was 3 hex digits (26 x 16^3 aliases, about 10^5), so in a table of a few thousand patients
dozens of pairs of people shared one: their rows read alike and a chart keyed by the label added them together. The
masked-names test caught it once in a few hundred runs (39 aliases for 40 people). An identifier's code was 4 digits
(16^4 values) for numbers that run to millions.

Now a name's code has 8 digits and an identifier's 12, still stable for the same value and still unlinkable without
the key.
"""

from __future__ import annotations

import re

import pytest

from core.compliance import result_guard


@pytest.fixture(autouse=True)
def _fixed_key(monkeypatch):
    monkeypatch.setenv("PII_PSEUDONYM_SECRET", "a-fixed-key-for-this-test-only")


def test_twenty_thousand_people_keep_twenty_thousand_aliases():
    names = [f"Person {i} Surname{i % 997}" for i in range(20_000)]
    aliases = {result_guard._mask(n, "safe_alias_name", "acct-test", source="PATIENTS.PATIENT_NAME") for n in names}
    assert len(aliases) == len(names)


def test_a_hundred_thousand_identifiers_keep_their_own():
    ids = [f"RX{i:08d}" for i in range(100_000)]
    aliases = {result_guard._mask(i, "safe_alias_identifier", "acct-test", source="RX.RX_NUMBER") for i in ids}
    assert len(aliases) == len(ids)


def test_the_alias_reads_as_before_and_stays_the_same():
    alias = result_guard._mask("Ana Stone", "safe_alias_name", "acct-test", source="PATIENTS.PATIENT_NAME")
    assert re.fullmatch(r"Patient [A-Z]-[0-9A-F]{8}", alias), alias
    assert alias == result_guard._mask("Ana Stone", "safe_alias_name", "acct-test", source="PATIENTS.PATIENT_NAME")
    doctor = result_guard._mask("Sam Hill", "safe_alias_name", "acct-test", source="RX.PRESCRIBER_NAME")
    assert re.fullmatch(r"Dr\. [A-Z]-[0-9A-F]{8}", doctor), doctor
    rx = result_guard._mask("RX00001234", "safe_alias_identifier", "acct-test", source="RX.RX_NUMBER")
    assert re.fullmatch(r"RX-[0-9A-F]{12}", rx), rx


def test_another_workspace_sees_another_alias():
    one = result_guard._mask("Ana Stone", "safe_alias_name", "acct-one", source="PATIENTS.PATIENT_NAME")
    two = result_guard._mask("Ana Stone", "safe_alias_name", "acct-two", source="PATIENTS.PATIENT_NAME")
    assert one != two
