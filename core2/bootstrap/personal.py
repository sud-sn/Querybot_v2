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

import re

from core2.bootstrap import names
from core2.bootstrap.inventory import InvTable
from core2.bootstrap.profiler import TableProfile

PII = "pii"                # never shown, listed, filtered on or sent
NAME = "name"              # a person's name: shown in answers, never sent to the AI
RECORD = "record"          # a record's number about a person (an Rx number): shown and linked, never listed or sent

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
    return bool(t & _FIRST_ALONE or t & _NAME and _like(t, FIRST_WORDS) or t == {"first"})


def is_last_name(name: str) -> bool:
    t = _tokens(name)
    return bool(t & _LAST_ALONE or t & _NAME and _like(t, LAST_WORDS) or t == {"last"})


# A person's other names, written alone in a table about people (MIDDLE, MAIDEN).
_OTHER_NAMES = {"middle", "maiden", "mname", "middlename", "maidenname"}
# The numbers HIPAA's Safe Harbor names that a pharmacy or a clinic keeps: a medical record number (MRN), a
# health-plan beneficiary number (Medicare's MBI, the older HICN), a patient's own number or account number.
NUMBER_WORDS = {"number", "no", "num", "nbr", "nmbr"}
_MEDICAL_RECORD = {"mrn", "mbi", "hicn"}
_BENEFICIARY = {"beneficiary", "medicare", "medicaid"}
_PATIENT = {"patient", "ptnt", "pat"}
# A prescription's number (RX_NUMBER, PRESCRIPTION_NO, RXNBR, ORIG_RX_NUMBER): Safe Harbor counts it as an identifier.
_PRESCRIPTION = {"rx", "prescription", "script"}
_RX_GLUED = {"rxno", "rxnbr", "rxnum", "rxnumber", "rxid"}
_RX_QUALIFIERS = {"orig", "original", "pharmacy", "new", "old", "prior", "previous", "parent", "source", "refill"}
# A driver's licence number, written alone or with a number word (DRIVERS, DL_NO, LICENSE_NUMBER).
_LICENCE = {"drivers", "driver", "licence", "license", "lic", "dl"}
# Where someone lives, as coordinates (LAT, LON) in a table about people.
_COORDINATES = {"lat", "lon", "lng", "long", "latitude", "longitude"}


def is_contact(name: str, pattern: str | None) -> bool:
    """An address, a phone number, an email, a national ID: by its name or by its values."""
    t = _tokens(name)
    return pattern in SHAPES or _like(t, CONTACT_WORDS) or bool(t & ID_WORDS) or _says(name, "social", "security") \
        or _says(name, "national", "id")


def _says(name: str, *words: str) -> bool:
    return set(words) <= _tokens(name)


def is_health_record(name: str, people: bool = False) -> bool:
    """A patient's medical record number, health-plan beneficiary number, or own number: a personal detail.
    ``people``: the table is about people, where PT_NO is a patient's number (elsewhere a part's or a point's)."""
    t = _tokens(name)
    if t & _MEDICAL_RECORD or {"medical", "record"} <= t or {"med", "rec"} <= t or {"medical", "rec"} <= t:
        return True
    if "mr" in t and t & NUMBER_WORDS and t <= {"mr"} | NUMBER_WORDS:
        return True             # MR_NO, MRNumber: a medical record number
    if t & _BENEFICIARY and t & (NUMBER_WORDS | {"id", "identifier"}):
        return True
    # A patient's number or account number, never the patient_id a warehouse keys its rows by.
    patient = _PATIENT | ({"pt"} if people else set())
    return bool(t & patient and t & NUMBER_WORDS and t <= patient | NUMBER_WORDS | {"account", "acct"})


def is_prescription_number(name: str) -> bool:
    """A prescription's number (RX_NUMBER, RX_ID, PRESCRIPTION_NO, RXNBR, ORIG_RX_NUMBER); never a column called
    just RX (whether a sale was a prescription's), nor the PRESCRIPTION_ID a warehouse keys its rows by."""
    t = _tokens(name)
    if t & _RX_GLUED and t <= _RX_GLUED | _RX_QUALIFIERS:
        return True
    return bool(t & _PRESCRIPTION and t <= _PRESCRIPTION | NUMBER_WORDS | {"id"} | _RX_QUALIFIERS
                and (t & NUMBER_WORDS or "rx" in t and "id" in t))


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
        elif column.data_type in ("text", "integer") and is_health_record(column.name, people):
            out[column.name] = PII      # an MRN, a Medicare number, a patient's own number
        elif column.data_type in ("text", "integer") and is_prescription_number(column.name):
            out[column.name] = RECORD   # an Rx number: the key a fill and a claim link by, never listed
        elif column.data_type in ("date", "timestamp") and _like(words, BIRTH_WORDS):
            out[column.name] = PII
        elif column.data_type == "text" and (is_first_name(column.name) or is_last_name(column.name)) and (
                people or words & (_FIRST_ALONE | _LAST_ALONE | FIRST_WORDS | LAST_WORDS)):
            # An abbreviation (FST_NM) only beside its other half: PRC_LST_NM alone is a price list's name.
            out[column.name] = NAME
        elif people and column.data_type == "text" and words and words <= _OTHER_NAMES:
            out[column.name] = NAME     # MAIDEN, MIDDLE beside FIRST and LAST
        elif people and column.data_type == "text" and words & _LICENCE and words <= _LICENCE | {
                "no", "num", "number", "nbr", "id"}:
            out[column.name] = PII      # a patient's DRIVERS: their licence number
        elif people and words and words <= _COORDINATES and column.data_type in ("decimal", "float", "integer"):
            out[column.name] = PII      # a patient's LAT and LON: where they live
        elif column.data_type == "text" and words & _NAME and (words & _MIDDLE or "full" in words):
            out[column.name] = NAME
        elif column.data_type == "text" and words & _NAME and not _like(words, ORG_WORDS) \
                and (_like(words, PERSON_WORDS) or people and not words - _NAME - subject - _NAME_EXTRA):
            out[column.name] = NAME     # "patient name" in a patients table, a partner's "contact name" anywhere
    return out


# The words a business's or a team's own name is made of: what it is ("Harbor Trading", "Quarry Supply Co",
# "Kestrel Clinics") or which part of a company it is ("Facilities Desk", "Capital Projects"). None is a common
# surname: "Power", "Law" and "Hall" are left out.
ORGANISATION_NAME_WORDS = frozenset("""
inc incorporated llc llp lp ltd limited plc co corp corporation company companies gmbh ag sa sarl sas bv nv pty pvt
group holdings enterprises ventures partners partnership capital industries industrial manufacturing works
market markets mart store stores shop shops outlet outlets supermarket grocer grocers grocery foods food provisions
goods trading traders trade wholesale wholesalers retail retailers distribution distributors distributor supply
supplies suppliers imports exports logistics freight transport shipping
labs laboratories systems solutions services technologies technology tech software digital media design designs
studio studios analytics consulting consultants advisors advisory agency legal energy motors automotive
health healthcare clinic clinics hospital hospitals pharmacy pharma medical dental bank banking financial finance
insurance realty properties construction builders engineering electric electronics farms farm dairy bakery cafe
restaurant restaurants kitchen hotel hotels travel airlines academy university college institute foundation
association club center centre centers centres office department dept team teams desk unit division operations
projects procurement purchasing buying sourcing spend maintenance facilities
home homes living housewares kitchenware kitchens cookware hardware mercantile outfitters boutique emporium depot
bistro brasserie inn suites resort lodge catering gifts promotions incentives rewards branding marketplace mall
deals quincaillerie comptoir cuisine culinaire maison auberge articles
""".split())
# What only a person's name has: a title before it, a degree or a generation after a comma ("Dr. Ana Ruiz",
# "Ana Ruiz, MD"), or a surname written first ("Ruiz, Ana").
_PERSON_TITLE = re.compile(r"^\s*(?:dr|mr|mrs|ms|miss|mx|prof|sir|madam)\b\.?\s", re.IGNORECASE)
_PERSON_SUFFIX = re.compile(r",\s*(?:md|do|pa|pa-c|np|rn|lpn|pharmd|rph|dds|dmd|dvm|od|phd|jr|sr|ii|iii|iv)\.?\s*$",
                            re.IGNORECASE)
_SURNAME_FIRST = re.compile(r"^\s*[A-Z][a-z'’\-]+,\s*[A-Z][a-z'’\-]+\s*$")


def organisations_named(values: list[str]) -> bool:
    """Do these values (a sample of one column's) name businesses or teams, not people? Only when there are at
    least three, nine in ten or more carry a business's or a team's word (or an "&"), and none reads as a
    person's name. Read in the app: no value is sent anywhere."""
    seen = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if len(seen) < 3:
        return False
    if any(_PERSON_TITLE.search(v) or _PERSON_SUFFIX.search(v) or _SURNAME_FIRST.match(v) for v in seen):
        return False
    named = sum("&" in v or bool({names.singular(w) for w in names.tokens(v)} & _ORGANISATION_SINGULAR) for v in seen)
    return named >= 0.9 * len(seen)


_ORGANISATION_SINGULAR = frozenset(names.singular(w) for w in ORGANISATION_NAME_WORDS)
