"""People's data: what never reaches the AI, and what is never shown at all.

A pharmacy's patients, a shop's customers, a company's staff: their names may be shown to the readers
of an answer (they work with these people), but never sent to the AI, and never put in the list of
names the AI matches a question against. Their contact details, birth dates and national ID numbers
are never shown, listed or filtered on.

Read from what columns are called and from what their values look like (the profiler counts emails,
phone numbers, street addresses and national ID numbers in the warehouse, so a column called C07
holding emails is found too). A company's name ("practice name", "payer name") is not a person's.
An admin can release any of it from the Knowledge base page.
"""

from __future__ import annotations

from core2.bootstrap import names
from core2.bootstrap.inventory import InvTable
from core2.bootstrap.profiler import TableProfile

PII = "pii"                # never shown, listed, filtered on or sent
NAME = "name"              # a person's name: shown in answers, never sent to the AI

SHAPES = {"email", "phone", "street", "national_id"}
# Each word as written and as warehouses abbreviate it (EML, PHN, STRT_ADDR). Listed, never guessed: a loose
# match reads "strategy" as a street and "add-on" as an address.
CONTACT_WORDS = {"address", "addr", "adr", "addrs", "street", "strt", "phone", "phn", "telephone", "tel", "tele",
                 "fax", "email", "eml", "emailaddress", "emailaddr"}
BIRTH_WORDS = {"birth", "brth", "bth", "dob", "born", "birthday", "birthdate"}
ID_WORDS = {"ssn", "passport", "ssnumber"}
_NAME = {"name", "nm"}
FIRST_WORDS = {"first", "fst", "frst", "given", "forename"}
LAST_WORDS = {"last", "lst", "surname", "family"}
_FIRST_ALONE = {"fname", "firstname", "forename", "givenname"}
_LAST_ALONE = {"lname", "lastname", "surname", "familyname"}
_MIDDLE = {"middle", "mname", "middlename"}
# Who a table is about, when it is about people: its plain name column is then a person's name.
PERSON_WORDS = {"patient", "ptnt", "person", "people", "employee", "emp", "empl", "staff", "tchn", "cntc", "cntct", "rep", "salesrep", "salesperson", "agent",
                "doctor", "physician", "prescriber", "member", "contact", "user", "manager", "technician",
                "pharmacist", "nurse", "driver", "student", "teacher", "worker", "associate", "guardian",
                "beneficiary", "caregiver", "clinician", "therapist", "dentist", "surgeon", "tenant", "guest",
                "passenger", "volunteer", "candidate", "applicant", "subscriber", "insured"}
# A name of something other than a person, even in a table about people: "practice name", "plan name".
_NAME_EXTRA = {"full", "display", "preferred", "nick", "nickname", "legal", "known", "as"}
ORG_WORDS = {"practice", "company", "business", "org", "organization", "organisation", "firm", "store", "shop",
             "pharmacy", "hospital", "clinic", "plan", "payer", "product", "item", "account", "brand", "vendor",
             "supplier", "agency", "group", "facility", "site", "location", "branch", "center", "centre", "office",
             "department", "dept", "team", "carrier", "service", "school", "university", "employer", "insurer",
             "file", "host", "server", "database", "table", "report"}


def _tokens(name: str) -> set[str]:
    return set() if names.opaque(name) else set(names.tokens(name))


def _like(words: set[str], vocabulary: set[str]) -> bool:
    """A word of the vocabulary, as written or as listed abbreviated (FST for first, PHN for phone)."""
    return bool(words & vocabulary)


def is_first_name(name: str) -> bool:
    t = _tokens(name)
    return bool(t & _FIRST_ALONE or t & _NAME and _like(t, FIRST_WORDS))


def is_last_name(name: str) -> bool:
    t = _tokens(name)
    return bool(t & _LAST_ALONE or t & _NAME and _like(t, LAST_WORDS))


def is_contact(name: str, pattern: str | None) -> bool:
    """An address, a phone number, an email, a national ID: by its name or by its values."""
    t = _tokens(name)
    return pattern in SHAPES or _like(t, CONTACT_WORDS) or bool(t & ID_WORDS) or _says(name, "social", "security") \
        or _says(name, "national", "id")


def _says(name: str, *words: str) -> bool:
    return set(words) <= _tokens(name)


def about_people(table: InvTable) -> bool:
    """A table about people: it names them by first and last name, or says who they are."""
    columns = [c.name for c in table.columns if c.data_type == "text"]
    if any(is_first_name(c) for c in columns) and any(is_last_name(c) for c in columns):
        return True
    return _like(_tokens(table.name), PERSON_WORDS)


def read_table(table: InvTable, profile: TableProfile) -> dict[str, str]:
    """Column name -> PII or NAME, for the columns of ``table`` that hold people's data."""
    people = about_people(table)
    subject = set(names.core_table(table.name)) | _tokens(table.name)
    out: dict[str, str] = {}
    for column in table.columns:
        p = profile.columns.get(column.name)
        words = _tokens(column.name)
        if column.data_type == "text" and is_contact(column.name, p.pattern if p else None):
            out[column.name] = PII
        elif column.data_type in ("date", "timestamp") and _like(words, BIRTH_WORDS):
            out[column.name] = PII
        elif column.data_type == "text" and (is_first_name(column.name) or is_last_name(column.name)) and (
                people or words & (_FIRST_ALONE | _LAST_ALONE | FIRST_WORDS | LAST_WORDS)):
            # An abbreviation (FST_NM) only beside its other half: PRC_LST_NM alone is a price list's name.
            out[column.name] = NAME
        elif column.data_type == "text" and words & _NAME and (words & _MIDDLE or "full" in words):
            out[column.name] = NAME
        elif column.data_type == "text" and words & _NAME and not _like(words, ORG_WORDS) \
                and (_like(words, PERSON_WORDS) or people and not words - _NAME - subject - _NAME_EXTRA):
            out[column.name] = NAME     # "patient name" in a patients table, a partner's "contact name" anywhere
    return out
