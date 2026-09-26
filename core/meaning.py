"""
core/meaning.py
───────────────
One answer to "what does this column mean?", and where each part of it came from.

A column's business name, its description, how it is used and the words
readers use for it were kept in seven stores -- field_overrides.json, approved
semantic feedback, the semantic model's approved fields, the join graph's
confirmed column properties, confirmed business meanings, the table setup's
column terms, the database's own comment -- beside what the vocabulary and the
naming rules read from the name and what the knowledge base's model wrote.
Each reader picked its own winner. The Semantic Layer page read KB prose, then
approved feedback, then overrides, and never the join graph or the model's
approvals; the count-target label put a machine expansion of the name ahead of
the meaning an admin had approved; and confirming a column in the graph wrote
its synonyms into the Semantic Layer as its meaning.

Here every store states a Claim -- attribute, value, source, evidence -- and
one rule resolves them:

    admin  >  database  >  vocabulary  >  rule  >  ai

An admin's decision wins; then what the database's owner wrote on the column;
then a reading the tenant's vocabulary (packs, dictionaries) gives the name;
then what the naming rules make of it; the knowledge base's generated prose
last. Inside a tier the more specific claim wins: a column's own override
before a suggestion an admin approved, before the knowledge base's copy of
that approval, before a column term shared by name. Synonyms are the union,
in that order. Every resolved value keeps its source and evidence, so a page
can say where a meaning came from and an answer can be traced to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

TIERS = ("admin", "database", "vocabulary", "rule", "ai")
_RANK = {tier: len(TIERS) - index for index, tier in enumerate(TIERS)}

# One scale for every source: how far a reader may rely on the value without
# anyone checking it. An admin's decision is the reference; a database comment
# is the warehouse owner's own words; a vocabulary reading is a governed
# dictionary; a naming rule is a guess from the spelling; generated prose is a
# proposal.
CONFIDENCE = {"admin": 100, "database": 90, "vocabulary": 85, "rule": 70, "ai": 60}

SINGLE_VALUED = ("label", "description", "use_case")


@dataclass(frozen=True)
class Claim:
    """One store's statement about one attribute of one column."""

    attribute: str                  # "label" | "description" | "use_case" | "synonyms"
    value: Any                      # a string, or a sequence of terms for synonyms
    source: str                     # one of TIERS
    evidence: str                   # where it was read, in words an admin recognises
    specificity: int = 0            # within a tier, the more specific claim wins

    def __post_init__(self) -> None:
        if self.source not in _RANK:
            raise ValueError(f"unknown source of meaning {self.source!r}")


@dataclass(frozen=True)
class Resolved:
    value: Any
    source: str
    evidence: str
    confidence: int
    confirmed: bool


def resolve(claims: Iterable[Claim]) -> dict[str, Resolved]:
    """The winning value of each attribute, and all the synonyms, in order."""
    best: dict[str, Claim] = {}
    synonyms: list[tuple[str, Claim]] = []
    for claim in claims:
        if claim.attribute == "synonyms":
            values = claim.value if isinstance(claim.value, (list, tuple, set)) else [claim.value]
            synonyms.extend((str(term).strip(), claim) for term in values if str(term or "").strip())
            continue
        if not str(claim.value or "").strip():
            continue
        current = best.get(claim.attribute)
        if current is None or (_RANK[claim.source], claim.specificity) > (_RANK[current.source], current.specificity):
            best[claim.attribute] = claim

    resolved = {
        attribute: Resolved(str(claim.value).strip(), claim.source, claim.evidence,
                            CONFIDENCE[claim.source], claim.source == "admin")
        for attribute, claim in best.items()
    }
    if synonyms:
        ordered = sorted(synonyms, key=lambda item: (-_RANK[item[1].source], -item[1].specificity))
        seen: set[str] = set()
        terms: list[str] = []
        for term, _claim in ordered:
            key = " ".join(term.lower().split())
            if key not in seen:
                seen.add(key)
                terms.append(term)
        top = ordered[0][1]
        resolved["synonyms"] = Resolved(tuple(terms), top.source, top.evidence,
                                        CONFIDENCE[top.source], top.source == "admin")
    return resolved


def _terms(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple, set)):
        return [str(term).strip() for term in raw if str(term or "").strip()]
    return [term.strip() for term in re.split(r"[,;\n]", str(raw or "")) if term.strip()]


def _first_clause(text: Any) -> str:
    return re.split(r"[.;\n]", str(text or "").strip(), maxsplit=1)[0].strip()[:90]


def column_claims(
    *,
    column: str,
    override: dict | None = None,
    approved_feedback: dict | None = None,
    model_field: dict | None = None,
    graph_property: dict | None = None,
    business_meaning: dict | None = None,
    column_terms: Iterable[str] | None = None,
    db_comment: str = "",
    kb_field: dict | None = None,
    expansion: tuple[str, str] | None = None,
) -> list[Claim]:
    """Every claim the stores make about one column. Each argument is that
    store's record as the store returns it; a store with nothing to say about
    the column is None or empty.

    ``kb_field`` is what the knowledge base's markdown says: its meaning and
    use (an admin's approved edit when ``approved``, generated prose
    otherwise) and the synonyms its model wrote. ``expansion`` is (expanded
    name, "vocabulary" | "rule") -- what the enrichment read from the name,
    and on what authority.
    """
    claims: list[Claim] = []
    if override:
        evidence = "admin override"
        claims += [Claim("description", override.get("meaning"), "admin", evidence, 50),
                   Claim("use_case", override.get("use_case"), "admin", evidence, 50),
                   Claim("synonyms", _terms(override.get("synonyms")), "admin", evidence, 50)]
    if approved_feedback:
        evidence = "approved suggestion"
        claims += [Claim("description", approved_feedback.get("suggested_meaning"), "admin", evidence, 40),
                   Claim("use_case", approved_feedback.get("suggested_use_case"), "admin", evidence, 40),
                   Claim("synonyms", _terms(approved_feedback.get("suggested_synonyms")), "admin", evidence, 40)]
    if kb_field:
        # The markdown carries an approval as a copy of the decision (the
        # "Admin-approved Semantic Layer edit" marker, or a legacy comment),
        # so it ranks under the records it was copied from. Its synonyms are
        # the knowledge-base model's either way: an approval marks the row.
        if kb_field.get("approved"):
            tier, evidence, specificity = "admin", "approved edit in the knowledge base", 35
        else:
            tier, evidence, specificity = "ai", "knowledge base (generated)", 0
        claims += [Claim("description", kb_field.get("meaning"), tier, evidence, specificity),
                   Claim("use_case", kb_field.get("use_case"), tier, evidence, specificity)]
        # A synonym list an admin saved on the field is the whole list: the
        # editor shows the generated terms, and one the admin took out must
        # stay out.
        if not (_terms((override or {}).get("synonyms"))
                or _terms((approved_feedback or {}).get("suggested_synonyms"))):
            claims.append(Claim("synonyms", _terms(kb_field.get("synonyms")), "ai", "knowledge base (generated)"))
    if model_field and str(model_field.get("status") or "") == "approved":
        evidence = "approved in the semantic model"
        claims += [Claim("description", model_field.get("approved_meaning"), "admin", evidence, 30),
                   Claim("use_case", model_field.get("approved_use_case"), "admin", evidence, 30),
                   # The model names an approved field by its approved meaning
                   # (its first business candidate); as a label it yields to a
                   # name an admin gave the column outright.
                   Claim("label", _first_clause(model_field.get("approved_meaning")), "admin", evidence, 1)]
    if graph_property and str(graph_property.get("status") or "") == "confirmed":
        evidence = "confirmed in the join graph"
        claims += [Claim("label", graph_property.get("display_name"), "admin", evidence, 20),
                   Claim("synonyms", _terms(graph_property.get("synonyms")), "admin", evidence, 20)]
    if business_meaning:
        # What the admin decided, never the proposal: the store keeps a
        # decided reading and synonyms only while a meaning is confirmed, and
        # a confirmation with the synonyms cleared keeps none.
        evidence = "confirmed business meaning"
        claims += [Claim("label", business_meaning.get("decided_reading"), "admin", evidence, 10),
                   Claim("synonyms", _terms(business_meaning.get("decided_synonyms")), "admin", evidence, 10)]
    if column_terms:
        claims.append(Claim("synonyms", _terms(list(column_terms)), "admin", "the table's column terms", 5))
    if str(db_comment or "").strip():
        claims.append(Claim("description", " ".join(str(db_comment).split()), "database", "database comment"))
    if expansion and str(expansion[0] or "").strip():
        tier = expansion[1] if expansion[1] in ("vocabulary", "rule") else "rule"
        claims.append(Claim("label", expansion[0], tier,
                            "the tenant's vocabulary" if tier == "vocabulary" else "naming rules"))
    return claims


def column_meaning(**stores: Any) -> dict[str, Resolved]:
    """resolve(column_claims(**stores))."""
    return resolve(column_claims(**stores))
