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


# The typographic apostrophe is not an exotic character in French -- it is the
# DEFAULT. A French (AZERTY) layout, macOS, Word, Google Docs and every mobile
# keyboard all produce U+2019 for the elision apostrophe, so "chiffre
# d’affaires" is what a reader actually types and "chiffre d'affaires" is the
# variant. Folded without this, the two share not one lexicon key: the measure
# entry misses, "aujourd’hui" misses, and "depuis le début de l’année" loses
# its whole window -- detect_temporal_window then returns {} and the answer is
# anchored on the SERVER CLOCK instead of the data. Silent, and confident.
#
# Every character here is an apostrophe someone's editor substitutes: the two
# curly quotes, the modifier letter, the prime, the fullwidth form, and the
# acute/grave a typist reaches for when the layout fights them. All fold to the
# ASCII apostrophe the lexicon keys are written with.
_APOSTROPHE_FOLD = str.maketrans(
    {ch: "'" for ch in "\u2018\u2019\u02bc\u2032\uff07\u00b4\u0060"}
)


def _fold(text: str) -> str:
    """Lowercase, strip accents and normalise apostrophes, for MATCHING only.

    The replacement text is always the canonical English, so nothing accented
    survives into the output by this path. Folding is what lets one entry match
    "l'année dernière", "l’annee derniere" and "L'Année Dernière" alike --
    accents and the shape of the apostrophe are the first two things a hurried
    typist or an autocorrecting editor changes.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.translate(_APOSTROPHE_FOLD)


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
    # A phrase, not two words: "superieur" alone would also fire inside
    # "chiffre superieur a 100", where "above average" would be a lie.
    "superieur a la moyenne": "above average",
    "superieure a la moyenne": "above average",
    "au-dessus de la moyenne": "above average",
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
    #
    # THE CURRENT PERIOD, which had no entry at all. Measured: "du mois en
    # cours", "de l'annee en cours", "du mois courant", "du mois actuel" and
    # every quarter/week sibling all reached detect_temporal_window with no
    # window -- and a question with no window gets no governed business date,
    # so generation falls through to the dialect's own recipes and anchors on
    # the SERVER CLOCK. On a mart loaded nightly that is a silently wrong
    # number; on a stale demo mart it is an empty result reported as a fact.
    #
    # Every key here anchors a CALENDAR UNIT. That is deliberate and it is the
    # whole reason "en cours" is safe to read: on its own it means "in
    # progress", and a distributor's questions are full of it -- "commandes en
    # cours" is open orders, "travaux en cours" is WIP. Neither contains a
    # unit, so neither is touched.
    "mois en cours": "this month",
    "annee en cours": "this year",
    "trimestre en cours": "this quarter",
    "semaine en cours": "this week",
    "en cours d'annee": "this year",
    "mois courant": "this month",
    "annee courante": "this year",
    "trimestre courant": "this quarter",
    "semaine courante": "this week",
    "mois actuel": "this month",
    "annee actuelle": "this year",
    "trimestre actuel": "this quarter",
    "semaine actuelle": "this week",

    # TO-DATE. detect_temporal_window already reads "month to date" and its
    # siblings and maps each onto the matching this_* kind; French had only the
    # year forms, so "mois a date" and "cumul mensuel" carried no window while
    # "cumul annuel" did. "a date" is written folded -- "à date" reaches the
    # lexicon with the accent already stripped.
    "depuis le debut de l'annee": "year to date",
    "depuis le debut de l annee": "year to date",
    "depuis le debut du mois": "month to date",
    "depuis le debut du trimestre": "quarter to date",
    "depuis le debut de la semaine": "week to date",
    "cumul annuel": "year to date",
    "cumul mensuel": "month to date",
    "cumul trimestriel": "quarter to date",
    "cumul hebdomadaire": "week to date",
    "annee a date": "year to date",
    "mois a date": "month to date",
    "trimestre a date": "quarter to date",
    "semaine a date": "week to date",
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
    # The verbs core/result_renderer.py's "here is what you CAN ask" hint
    # puts in front of a reader. Infinitives, because the noun forms are
    # ambiguous: "classe" is also a category and "filtre" also a filter, and
    # either would rewrite a column name in a real question.
    "classer": "rank",
    "filtrer": "filter",
    "lister": "list",
    "ventiler": "break down",
    # The imperative too. The chart's click-to-drill writes its question into
    # the composer, so a French reader sends "Ventile ceci pour X" -- and the
    # refinement classifier in core/conversation_state.py matches on
    # "break this down". Without this line that click is classified as a fresh
    # question and the reader silently loses the result they were drilling
    # into. The English phrasing has always matched by construction; this is
    # the same guarantee for French.
    "ventile": "break down",
    "ceci": "this",
    "cela": "that",
    "afficher": "show",
    "montre": "show",
    "montre-moi": "show me",
    "montre moi": "show me",
    "affiche": "show",
    "affiche-moi": "show me",
    "donne-moi": "give me",
    "liste": "list",
    "lignes": "rows",
    "ligne": "row",
    "enregistrements": "records",
    "enregistrement": "record",
    "derriere": "behind",
    # Demonstratives. "ce mois-ci" and its siblings are longer entries, and
    # the lexicon is applied longest-first, so those still win.
    "ce": "this",
    "cet": "this",
    "cette": "this",
    "ces": "these",
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

# A count is as often written as a word as a digit -- "les six derniers mois" is
# ordinary business French. Read ONLY in the count position below (directly
# before derniers/prochains or a unit), which is what makes "neuf" safe: on its
# own it means "new", and "un produit neuf" is a new product, not nine of them.
# "un"/"une" are deliberately absent: they are already lexicon entries meaning
# the English article, and "un client" must stay "a client".
_COUNT_WORDS = {
    "deux": "2", "trois": "3", "quatre": "4", "cinq": "5", "six": "6",
    "sept": "7", "huit": "8", "neuf": "9", "dix": "10", "onze": "11",
    "douze": "12", "quinze": "15", "dix-huit": "18", "vingt": "20",
    "vingt-quatre": "24", "trente": "30", "soixante": "60",
    "quatre-vingt-dix": "90",
}
# Longest first so "quatre-vingt-dix" is not read as "quatre", and digits first
# so a digit is never partially eaten by a word alternative.
_COUNT_ALT = r"\d+|" + "|".join(
    re.escape(word) for word in sorted(_COUNT_WORDS, key=len, reverse=True))


def _count(token: str) -> str:
    return _COUNT_WORDS.get(token, token)


# The superlatives the lexicon already knows. A count word sitting directly in
# front of one of these is a count and nothing else -- French adjectives do not
# stack that way, so "neuf meilleurs" cannot be read as "new best".
_RANK_AHEAD = "|".join((
    r"meilleur(?:es?|s)?",
    r"pires?",
    r"premier(?:es?|s)?",
    r"dernier(?:es?|s)?",
    r"plus\s+(?:eleve|faible|gros|grand|petit)(?:e?s)?",
    r"moins\s+(?:eleve|faible|gros|grand|petit)(?:e?s)?",
))


_NUMERIC_RULES: tuple[tuple[re.Pattern[str], object], ...] = (
    # "les 6 derniers mois" -> "the last 6 months". The determiner is
    # TRANSLATED, not dropped: analyze_query_intent's time-series pattern is
    # `over\s+the\s+last\s+\d+`, so "over last 30 days" misses it by one word
    # and the reader gets a flat aggregate instead of a series. It is also not
    # left in place -- "ces last 12 months" keeps a French word in the middle
    # of the phrase the detector reads.
    #
    # `dernier(?:es?|s)?` covers all four inflections. `derniers?` covered two,
    # and the two it missed are the FEMININE ones -- which is not an edge case,
    # because semaine and annee are feminine nouns. Measured: "les 4 dernieres
    # semaines" and "les 2 dernieres annees" fell through to the flat lexicon,
    # came out as "the 4 last weeks", and matched no window pattern at all.
    (re.compile(
        rf"\b([lc]es?\s+)?({_COUNT_ALT})\s+dernier(?:es?|s)?\s+({_UNIT_ALT})\b"),
     lambda m: f"{'the ' if m.group(1) else ''}last "
               f"{_count(m.group(2))} {_UNITS[m.group(3)]}"),
    # "les 3 prochains mois" -> "the next 3 months"
    (re.compile(
        rf"\b([lc]es?\s+)?({_COUNT_ALT})\s+prochain(?:es?|s)?\s+({_UNIT_ALT})\b"),
     lambda m: f"{'the ' if m.group(1) else ''}next "
               f"{_count(m.group(2))} {_UNITS[m.group(3)]}"),
    # "sur 3 mois" -> "over 3 months". French "mois" is invariant, so the
    # English plural cannot come from the lexicon -- only the number knows.
    (re.compile(rf"\b({_COUNT_ALT})\s+({_UNIT_ALT})\b"),
     lambda m: f"{_count(m.group(1))} {_UNITS[m.group(2)]}"),
    # "les cinq meilleurs clients" -> "les 5 meilleurs clients", which the
    # lexicon then finishes into "the 5 best customers". Only the COUNT is
    # rewritten, by lookahead, so the superlative is still the lexicon's to
    # translate.
    #
    # This is the asymmetry that made it worth doing: detect_top_n_intent reads
    # "top five customers" and "top 5 customers" alike, so ENGLISH was never
    # exposed here. French was. "les cinq meilleurs clients" produced no
    # TopNIntent at all, which means no row limit was ever requested and the
    # reader was handed every customer, narrated as the top five -- this
    # module's own headline example, in the one spelling it still missed.
    (re.compile(rf"\b({_COUNT_ALT})(?=\s+(?:{_RANK_AHEAD}))"),
     lambda m: _count(m.group(1))),
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
