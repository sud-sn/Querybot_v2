"""
core/question_normalizer.py

Turn a French question into the product's own canonical English analytics
phrasing, so the detectors can read it.

Why this exists
───────────────
The product understands a question by running hand-written ENGLISH regexes over
the raw text, and every one of them runs before any model sees the question. A
French question therefore reaches SQL generation with almost no intent detected
-- and none of it errors. Measured on five questions and their French
equivalents: 5 analytical intents and 5 semantic flags in English against 1 and
1 in French, the single survivor being the cognate "contribution".

The consequence is worse than a missing feature. "Montre-moi les 10 meilleurs
clients par marge" loses ``wants_top_n``, so no row limit is ever requested and
the reader is handed every customer, narrated as the top 10. A portal that
answers the wrong question fluently is worse than one that answers the right
question in the wrong language.

What this is NOT
────────────────
Not a translator. The output does not have to read well; it has to fire the
right detectors. "les 10 meilleurs clients par marge" becomes "les 10 best
clients by margin", which is not a sentence anyone would write and is exactly
enough for detect_top_n_intent to see "10 best" and for the field planner to
see "margin".

Not a replacement for the reader's text either. The canonical form is an ADDED
field: the French the reader typed still goes to the chart title, the dashboard
tile, the trace and the audit log. That distinction is the whole design -- the
alternative fills a French user's dashboard with English tile names.

Why not French twins for every detector vocabulary
──────────────────────────────────────────────────
Measured at ~1,173 distinct English lexical items across 484 regex patterns and
101 vocabulary sets in 28+ modules. It is also the failure mode this codebase
documents against itself: core/multi_period.py notes "a documented habit of
growing parallel detectors for the same concept and letting them drift", and
core/llm.py records two ENGLISH vocabularies for one concept drifting by 13
phrasings and silently inverting an anti-join. And it would still not work:
core/date_roles.py normalises with ``[^a-z0-9]+`` which SHREDS accented French
("année" -> "ann e") rather than merely failing to match it, and BM25 strips
every non-``[A-Za-z0-9_]`` character while the embedder is English-only --
measured token overlap between a French question and the English KB: zero.

One canonicalising pass at the front door leaves all 484 patterns unedited.
"""

from __future__ import annotations

import re
import unicodedata

# Languages this module can canonicalise. Anything else is returned unchanged,
# which is what makes calling it unconditionally safe.
CANONICALISABLE = ("fr",)


def _fold(text: str) -> str:
    """Lowercase and strip accents, for MATCHING only.

    The replacement text is always the canonical English, so nothing accented
    survives into the output by this path. Folding is what lets one entry match
    "l'année dernière", "l'annee derniere" and "L'Année Dernière" alike --
    accents are the first thing a hurried typist drops.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


# ── The lexicon ──────────────────────────────────────────────────────────────
#
# Written as French -> canonical English, and applied LONGEST FIRST so a phrase
# always beats its own words. Every entry is here because it changes what a
# detector sees; this is not a French dictionary and should not grow into one.
#
# Multi-word entries exist where a word-by-word map would be wrong. "chiffre
# d'affaires" is three words for one measure; "écart budgétaire" is the pair
# core/budget_vs_actual.py looks for.

_LEXICON: dict[str, str] = {
    # ── Measures ─────────────────────────────────────────────────────────────
    "chiffre d'affaires": "revenue",
    "chiffre d affaires": "revenue",
    # NOT "ca": folded, the abbreviation for chiffre d'affaires and the pronoun
    # "ça" are the same two letters, and "qu'est-ce que ça donne" would become
    # "what does revenue give". Case would tell them apart, and matching runs on
    # folded lowercase text precisely so accents cannot be relied on either.
    "ventes": "sales",
    "vente": "sales",
    "marge brute": "gross margin",
    "marge nette": "net margin",
    "marge": "margin",
    "benefice": "profit",
    "benefices": "profit",
    "resultat net": "net income",
    "couts": "cost",
    "cout": "cost",
    "depenses": "spend",
    "depense": "spend",
    "montant": "amount",
    "quantite": "quantity",
    "quantites": "quantity",
    "effectif": "headcount",
    "effectifs": "headcount",
    "commandes": "orders",
    "commande": "orders",
    "factures": "invoices",
    "facture": "invoice",
    "stock": "inventory",
    "stocks": "inventory",
    "prix": "price",
    "remise": "discount",
    "remises": "discount",

    # ── Dimensions ───────────────────────────────────────────────────────────
    "clients": "customers",
    "client": "customer",
    "produits": "products",
    "produit": "product",
    "pays": "country",
    "secteur": "segment",
    "secteurs": "segments",
    "categorie": "category",
    "categories": "categories",
    "fournisseurs": "suppliers",
    "fournisseur": "supplier",
    "employes": "employees",
    "employe": "employee",
    "vendeurs": "sales reps",
    "magasins": "stores",
    "magasin": "store",
    "entrepot": "warehouse",
    "entrepots": "warehouses",

    # ── Analytical intent ────────────────────────────────────────────────────
    "ecart budgetaire": "budget variance",
    "ecart au budget": "budget variance",
    "ecart par rapport au budget": "budget variance",
    "realise": "actual",
    "reel": "actual",
    "comparer": "compare",
    "comparaison": "comparison",
    "par rapport a": "versus",
    "par rapport au": "versus",
    "par rapport aux": "versus",
    "contre": "versus",
    "evolution": "trend",
    "tendance": "trend",
    "prevision": "forecast",
    "previsions": "forecast",
    "prevoir": "forecast",
    "predire": "predict",
    "repartition": "breakdown",
    "ventilation": "breakdown",
    "part de marche": "market share",
    "part du total": "share of total",
    "part": "share",
    "pourcentage": "percentage",
    "contribution": "contribution",
    "valeurs aberrantes": "outliers",
    "valeur aberrante": "outlier",
    "anomalie": "anomaly",
    "inhabituel": "unusual",
    "inhabituelle": "unusual",
    "inhabituels": "unusual",
    "inhabituelles": "unusual",
    "trouve": "find",
    "trouver": "find",
    "cherche": "find",
    "detecte": "detect",
    "identifie": "identify",
    "pic": "spike",
    "pics": "spikes",
    "ecart type": "standard deviation",
    "correlation": "correlation",
    "correle": "correlated",
    "entonnoir": "funnel",
    "tunnel de conversion": "conversion funnel",
    "cohorte": "cohort",
    "cohortes": "cohorts",
    "tableau croise dynamique": "pivot table",
    "tableau croise": "pivot table",
    "histogramme": "histogram",
    "boite a moustaches": "box plot",
    "et si": "what if",
    "simulation": "what if scenario",
    "moyenne mobile": "rolling average",
    "moyenne glissante": "rolling average",
    "total cumule": "running total",
    "cumul": "running total",
    "classement": "ranking",
    "rang": "rank",
    "croissance": "growth",
    "taux de croissance": "growth rate",
    "moyenne": "average",
    "mediane": "median",
    "somme": "sum",
    "nombre de": "count of",
    "combien": "how much",

    # ── Superlatives, which are what carry Top-N ─────────────────────────────
    # detect_top_n_intent matches "<n> best" as readily as "top <n>", and
    # French puts the number first -- "les 10 meilleurs" becomes "les 10 best",
    # which that detector already reads. No new pattern is needed there.
    "meilleurs": "best",
    "meilleures": "best",
    "meilleur": "best",
    "meilleure": "best",
    "pires": "worst",
    "pire": "worst",
    "plus eleves": "highest",
    "plus eleve": "highest",
    "plus elevees": "highest",
    "plus elevee": "highest",
    "plus faibles": "lowest",
    "plus faible": "lowest",
    "plus gros": "biggest",
    "plus grands": "largest",
    "plus grand": "largest",
    "plus petits": "smallest",
    "plus petit": "smallest",
    "premiers": "top",
    "premieres": "top",
    "derniers": "bottom",

    # ── Time ─────────────────────────────────────────────────────────────────
    "depuis le debut de l'annee": "year to date",
    "depuis le debut de l annee": "year to date",
    "cumul annuel": "year to date",
    "annee derniere": "last year",
    "l'annee derniere": "last year",
    "annee precedente": "previous year",
    "cette annee": "this year",
    "mois dernier": "last month",
    "le mois dernier": "last month",
    "mois precedent": "previous month",
    "ce mois-ci": "this month",
    "ce mois": "this month",
    "semaine derniere": "last week",
    "la semaine derniere": "last week",
    "cette semaine": "this week",
    "trimestre dernier": "last quarter",
    "trimestre precedent": "previous quarter",
    "ce trimestre": "this quarter",
    "aujourd'hui": "today",
    "aujourd hui": "today",
    "hier": "yesterday",
    "demain": "tomorrow",
    "exercice fiscal": "fiscal year",
    "annee fiscale": "fiscal year",
    "exercice": "fiscal year",
    "annees": "years",
    "annee": "year",
    "trimestres": "quarters",
    "trimestre": "quarter",
    "mois": "month",
    "semaines": "weeks",
    "semaine": "week",
    "jours": "days",
    "jour": "day",
    "janvier": "January",
    "fevrier": "February",
    "mars": "March",
    "avril": "April",
    "mai": "May",
    "juin": "June",
    "juillet": "July",
    "aout": "August",
    "septembre": "September",
    "octobre": "October",
    "novembre": "November",
    "decembre": "December",

    # ── The function words the detectors actually read ───────────────────────
    # Deliberately few. "de" is not here: it lives inside customer names
    # ("Banque de France"), and no detector pattern needs it that a phrase
    # entry above does not already cover.
    "par rapport": "versus",
    "entre": "between",
    "les": "the",
    "le": "the",
    "la": "the",
    "une": "a",
    "un": "a",
    # "of" is load-bearing: "share of total", "breakdown of revenue by product"
    # and "distribution of" are all anchored on it. The cost is that a customer
    # name carrying "de" outside quotes reaches RETRIEVAL as "Banque of France"
    # -- BM25 still matches "Banque" and "France", and value resolution reads
    # the reader's own text, not this one.
    "de la": "of",
    "des": "of",
    "du": "of",
    "de": "of",
    "aux": "to",
    "au": "to",
    "pour": "for",
    "sur": "over",
    "dans": "in",
    "avec": "with",
    "et": "and",
    "pour chaque": "for each",
    "chaque": "each",
    "par": "by",
    "pourquoi": "why",
    "quelle est la raison": "what is the reason",
    "qu'est-ce qui a cause": "what caused",
    "qu est ce qui a cause": "what caused",
    "qui a cause": "what caused",
    "qu'est-ce que": "what",
    "qu est ce que": "what",
    "qu'est-ce qui": "what",
    "qu est ce qui": "what",
    "est-ce que": "does",
    # "contribute", not "contributed": core/contribution_analysis.py matches
    # `contribut(?:ion|e|es|ing)?` -- the past participle is the one form that
    # pattern does NOT read, so the obvious translation is the one that fails.
    "contribue": "contribute",
    "contribuent": "contribute",
    "contribuer": "contribute",
    "baisse": "decreased",
    "baisser": "decrease",
    "augmente": "increased",
    "augmenter": "increase",
    "hausse": "increase",
    "montre": "show",
    "montre-moi": "show me",
    "montre moi": "show me",
    "affiche": "show",
    "affiche-moi": "show me",
    "donne-moi": "give me",
    "liste": "list",
    "quels sont": "what are",
    "quelles sont": "what are",
    "quel est": "what is",
    "quelle est": "what is",
    "quels": "which",
    "quelles": "which",
    "quel": "which",
    "quelle": "which",
    "prochains": "next",
    "prochain": "next",
    "prochaines": "next",
    "prochaine": "next",
    "dernier": "last",
    "derniere": "last",
    "dernieres": "last",
}

# Entries are matched on the FOLDED text, so the keys are folded once here
# rather than at every call. Sorted longest first so a phrase always beats its
# own words -- without this, "chiffre d'affaires" would be eaten by "ca".
_ENTRIES: tuple[tuple[str, str], ...] = tuple(sorted(
    ((_fold(source), target) for source, target in _LEXICON.items()),
    key=lambda pair: (-len(pair[0]), pair[0]),
))

_LEXICON_RE = re.compile(
    r"(?<![0-9a-z])(?:" + "|".join(re.escape(key) for key, _ in _ENTRIES) + r")(?![0-9a-z])"
)
_REPLACEMENTS = dict(_ENTRIES)

# Rules that carry a number, which a flat lexicon cannot express.
# Applied BEFORE the lexicon, on folded text, so their English output is not
# then re-matched by it.
_UNITS = {
    "mois": "months", "semaine": "weeks", "semaines": "weeks",
    "jour": "days", "jours": "days", "annee": "years", "annees": "years",
    "an": "years", "ans": "years", "trimestre": "quarters",
    "trimestres": "quarters",
}
_UNIT_ALT = "|".join(sorted(_UNITS, key=len, reverse=True))

_NUMERIC_RULES: tuple[tuple[re.Pattern[str], object], ...] = (
    # "les 6 derniers mois" -> "the last 6 months". The determiner is
    # TRANSLATED, not dropped: analyze_query_intent's time-series pattern is
    # `over\s+the\s+last\s+\d+`, so "over last 30 days" misses it by one word
    # and the reader gets a flat aggregate instead of a series. It is also not
    # left in place -- "ces last 12 months" keeps a French word in the middle
    # of the phrase the detector reads.
    (re.compile(rf"\b([lc]es?\s+)?(\d+)\s+derniers?\s+({_UNIT_ALT})\b"),
     lambda m: f"{'the ' if m.group(1) else ''}last {m.group(2)} {_UNITS[m.group(3)]}"),
    # "les 3 prochains mois" -> "the next 3 months"
    (re.compile(rf"\b([lc]es?\s+)?(\d+)\s+prochains?\s+({_UNIT_ALT})\b"),
     lambda m: f"{'the ' if m.group(1) else ''}next {m.group(2)} {_UNITS[m.group(3)]}"),
    # "sur 3 mois" -> "over 3 months". French "mois" is invariant, so the
    # English plural cannot come from the lexicon -- only the number knows.
    (re.compile(rf"\b(\d+)\s+({_UNIT_ALT})\b"),
     lambda m: f"{m.group(1)} {_UNITS[m.group(2)]}"),
    # "T1 2025" / "T1" -> "Q1 2025" / "Q1". core/multi_period.py's quarter
    # pattern is anchored on the letter Q.
    (re.compile(r"\bt([1-4])\b(?=\s*(?:20\d{2})?)"), lambda m: f"Q{m.group(1)}"),
    # "le 3e trimestre" -> "Q3"
    (re.compile(r"\b(?:le\s+)?([1-4])\s*(?:e|er|eme|ere)?\s+trimestre\b"),
     lambda m: f"Q{m.group(1)}"),
)


_ELISION_RE = re.compile(r"(?<![0-9a-z])(?:l|d|n|s|c|j|m|t|qu)'(?=[a-z])")


# A single quote is a value delimiter only when it opens at a word boundary and
# closes at one. Without that, French elision breaks the pairing: in "chiffre
# d'affaires pour 'Banque de France'" a naive `'[^']*'` pairs the apostrophe in
# "d'affaires" with the opening quote and calls everything between them a
# customer value -- so the real value goes unprotected and half the question
# goes untranslated. Double quotes and guillemets have no such ambiguity.
_QUOTED_RE = re.compile(
    r"\"[^\"]*\"|«[^»]*»|(?<![0-9a-z])'[^']*?'(?![0-9a-z])"
)


def _spans_to_skip(text: str) -> list[tuple[int, int]]:
    """Quoted spans, which hold customer values rather than French.

    A tenant with a customer called "Marge" or a product called "Premier"
    would otherwise have its own data rewritten on the way to retrieval.
    """
    return [m.span() for m in _QUOTED_RE.finditer(text)]


def canonicalise(text: str) -> str:
    """The canonical English form of a French question.

    Applied to the FOLDED text: accents never reach the output, because every
    replacement is English and every unmatched run is copied from the folded
    source. That costs the accents on words the lexicon does not know -- which
    is the right trade, since core/date_roles.py shreds them anyway and the
    reader's own text is preserved separately.
    """
    original = str(text or "")
    if not original.strip():
        return original

    folded = _fold(original)
    protected = _spans_to_skip(folded)

    def _outside_quotes(start: int, end: int) -> bool:
        return not any(a <= start and end <= b for a, b in protected)

    for pattern, build in _NUMERIC_RULES:
        folded = pattern.sub(
            lambda m: build(m) if _outside_quotes(*m.span()) else m.group(0),
            folded,
        )
        protected = _spans_to_skip(folded)

    folded = _LEXICON_RE.sub(
        lambda m: (_REPLACEMENTS[m.group(0)] if _outside_quotes(*m.span())
                   else m.group(0)),
        folded,
    )

    # The elided article the lexicon leaves behind: "l'évolution" matches
    # "évolution" and comes out "l'trend". Run AFTER the lexicon, so entries
    # that contain an apostrophe of their own ("aujourd'hui", "chiffre
    # d'affaires") are already consumed. A possessive is safe -- "John's" has
    # no word boundary before the "s".
    protected = _spans_to_skip(folded)
    return _ELISION_RE.sub(
        lambda m: "" if _outside_quotes(*m.span()) else m.group(0), folded,
    )


def canonical_question(text: str, lang: str | None) -> str:
    """The text the detectors should read, for a reader in ``lang``.

    English is returned untouched and unfolded -- every existing tenant runs
    the exact bytes it ran before, which is what makes this safe to call
    unconditionally at the top of the pipeline.
    """
    if not text or str(lang or "en").strip().lower()[:2] not in CANONICALISABLE:
        return text
    return canonicalise(text)
