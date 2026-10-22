"""Replies to what the data cannot answer: the product itself, business in general, anything else, and data the
workspace does not hold.

A question the catalog cannot answer used to get one sentence ("I cannot answer that from this data: ..."). It is
now answered as a person would answer it:

* a question about Solviz itself ("how do I share a dashboard?", "comment télécharger le tableau ?") is answered
  in code from what the product does, before any AI;
* a question about business in general ("what is a good inventory turnover?", "what does EBITDA mean?") is
  answered from the AI's general knowledge, labelled "General knowledge, not from your data", where the admin
  allows it (on unless switched off);
* a question about something else entirely (the weather, a joke) gets a friendly decline, worded a few ways, and
  questions it can answer;
* a question about the business's own figures that its data does not hold says so, says what the data does hold,
  offers the nearest questions, and a button that asks the admin for that data.

The AI sorts the last three, after the plan said the catalog cannot answer: one more call, on refusals only. It is
given the question as the planner was (scrubbed where the workspace scrubs), why the plan stopped, and the names of
the reader's tables, never a row or a value. Everything is said in the reader's language.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger("querybot.core2")

GENERAL = "general_knowledge"
PRODUCT = "product_help"
OFF_TOPIC = "off_topic"
NOT_HERE = "data_not_here"
LABEL = {"en": "General knowledge, not from your data", "fr": "Connaissances générales, pas vos données"}
MAX_WORDS = 130


# ── Solviz itself ────────────────────────────────────────────────────────────

# A question about how to do something, opening the question: never a request to do it ("show sales as a line" is
# answered, not explained), and the product's own verb right after it ("can I get the results for each store?" and
# "can we see our market share?" are questions about the data).
_HOW = (r"(?:how\s+(?:do|can|could|should)\s+(?:i|we)|how\s+to|where\s+(?:do|can)\s+(?:i|we)|can\s+(?:i|we)|"
        r"is\s+there\s+a\s+way\s+to|is\s+it\s+possible\s+to|"
        r"comment|o[uù]\s+(?:puis-je|peut-on)|"
        r"puis-je|peut-on|est-il\s+possible\s+de)")
# What a dashboard, an answer or a report is called: a topic word means the product only beside one ("market share by
# region", "export sales by country" and "email opens by month" are figures).
_DET = (r"(?:my|the|this|that|a|an|our|your|these|those|mon|ma|mes|le|la|les|l['’]|ce|cet|cette|ces|un|une|des|"
        r"notre|nos|votre|vos)")
_THING = (r"(?:dashboards?|boards?|reports?|answers?|results?|tables?|charts?|graphs?|this|that|it|them|tableaux?\s+de"
          r"\s+bord|tableaux?|rapports?|réponses?|résultats?|graphiques?|ceci|cela)")
_TOPICS: list[tuple[str, str, str, str]] = [
    # (key, words after a how-question, English, French)
    ("pin", r"(?:pin|add|save|put|keep)\b[^?]{0,40}\b(?:dashboard|board)|(?:ajouter|épingler|enregistrer|mettre)"
            r"\b[^?]{0,40}\btableau\s+de\s+bord",
     "Under an answer's chart, press Add to dashboard: pick one of your dashboards or create a new one, and give "
     "the tile a name. The tile stays live: it is counted again each time the dashboard opens.",
     "Sous le graphique d'une réponse, appuyez sur Ajouter au tableau de bord : choisissez l'un de vos tableaux de "
     "bord ou créez-en un, et nommez la tuile. La tuile reste à jour : elle est recalculée à chaque ouverture."),
    ("dashboard", r"(?:create|make|build|start|set\s+up)\b[^?]{0,30}\b(?:dashboard|board)|"
                  r"(?:créer|faire|construire)\b[^?]{0,30}\btableau\s+de\s+bord",
     "Ask for it in the chat, as you would say it: “create a dashboard of net sales by store and month”, or "
     "“create a dashboard using this” with an answer on screen. You can also add answers to it one by one "
     "with Add to dashboard.",
     "Demandez-le dans la conversation, comme vous le diriez : « crée un tableau de bord des ventes par magasin "
     "et par mois », ou « crée un tableau de bord avec ceci » quand une réponse est à l'écran. Vous "
     "pouvez aussi y ajouter des réponses une à une avec Ajouter au tableau de bord."),
    ("share", r"(?:share|partager)\s+(?:" + _DET + r"\s+)?" + _THING + r"\b",
     "Open the dashboard and press Share: add people or groups by name, or share it with everyone in the workspace. "
     "Each person sees only the data they may see.",
     "Ouvrez le tableau de bord et appuyez sur Partager : ajoutez des personnes ou des groupes par leur nom, ou "
     "partagez-le avec tout l'espace de travail. Chacun ne voit que les données auxquelles il a accès."),
    ("schedule", r"(?:e-?mail|send|get|receive|envoyer|recevoir)\s+(?:" + _DET + r"\s+)?" + _THING
                 + r"\b[^?]{0,30}\b(?:e-?mail(?:ed)?|every\s+(?:day|week|month|morning|monday|tuesday|wednesday|"
                 r"thursday|friday|saturday|sunday)|daily|weekly|monthly|by\s+e-?mail|courriel|par\s+mail|"
                 r"chaque\s+(?:jour|semaine|mois|matin|lundi|mardi|mercredi|jeudi|vendredi)|tous\s+les\s+(?:jours|"
                 r"lundis|mois|matins))\b|(?:schedule|subscribe\s+to|programmer|planifier|s['’]abonner\s+à)\s+(?:"
                 + _DET + r"\s+)?" + _THING + r"\b|(?:subscribe|s['’]abonner)\s*[?.!]*\s*$",
     "Open the dashboard, press More actions and choose Subscribe: pick when it is sent and to whom. It arrives by "
     "email with the dashboard in the message and as a PDF.",
     "Ouvrez le tableau de bord, appuyez sur Plus d'actions puis S'abonner : choisissez quand et à qui il est "
     "envoyé. Il arrive par courriel, le tableau de bord dans le message et en PDF."),
    ("alert", r"(?:set\s+up|set|create|add|make|get)\s+(?:an?\s+)?(?:alerts?|notifications?)\b|(?:get|be)\s+"
              r"(?:alerted|notified|warned)\b|(?:notify|warn|alert)\s+me\b|tell\s+me\s+when\b|(?:créer|mettre|"
              r"ajouter|recevoir)\s+(?:une\s+)?(?:alerte|notification)s?\b|être\s+(?:prévenu|averti|alerté)e?s?\b|"
              r"me\s+(?:prévenir|notifier)\b|m['’](?:avertir|alerter)\b",
     "On a dashboard tile, press Alert and say when to be told (above or below a value, or a change). It is checked "
     "every minute, and you are told in Notifications and by email.",
     "Sur une tuile de tableau de bord, appuyez sur Alerte et dites quand être prévenu (au-dessus ou en dessous "
     "d'une valeur, ou un changement). Elle est vérifiée chaque minute, et vous êtes prévenu dans les Notifications "
     "et par courriel."),
    ("download", r"(?:download|export|save|télécharger|exporter|enregistrer)\s+(?:" + _DET + r"\s+)?(?:" + _THING
                 + r"|data|rows|données|lignes)\b|(?:download|export|exporter|télécharger)\b[^?]{0,30}\b(?:to|in|into"
                 r"|as|en|vers)\s+(?:excel|csv)\b|(?:download|télécharger)\s*[?.!]*\s*$",
     "Under an answer's table, press Download as CSV: every row of the answer is saved, not only those on screen.",
     "Sous le tableau d'une réponse, appuyez sur Télécharger en CSV : toutes les lignes de la réponse sont "
     "enregistrées, pas seulement celles à l'écran."),
    ("new_question", r"(?:start|ask|open)\b[^?]{0,20}\bnew\s+(?:question|chat|conversation)|(?:poser|commencer|"
                     r"démarrer|ouvrir|lancer)\b[^?]{0,20}\bnouvelle\s+(?:question|conversation)",
     "Press New chat for a fresh start. When a question is read as following the answer above and you meant a new "
     "one, press Ask as a new question under it.",
     "Appuyez sur Nouvelle conversation pour repartir de zéro. Quand une question est lue comme la suite de la "
     "réponse au-dessus alors que vous vouliez une nouvelle question, appuyez sur Poser comme nouvelle question."),
    ("always", r"(?:always|toujours)\s+(?:use|leave|exclude|include|show|keep|count|utiliser|exclure|inclure|"
               r"garder|afficher|compter)\b|(?:set|make|keep|change|choose)\b[^?]{0,30}\b(?:by\s+default|default|"
               r"every\s+time)\b|(?:définir|régler|mettre|choisir)\b[^?]{0,30}\bpar\s+défaut\b",
     "Say it in the chat: “from now on use the last 2 months” or “always leave out 'Wholesale'”. "
     "It applies to every answer in the conversation until you say “stop using the last 2 months”.",
     "Dites-le dans la conversation : « à partir de maintenant, utilise les 2 derniers mois » ou « "
     "exclus toujours « Grossiste » ». Cela s'applique à chaque réponse jusqu'à ce que vous disiez « "
     "arrête d'utiliser les 2 derniers mois »."),
    ("metric", r"(?:new|own|my|custom)\s+(?:metric|figure|measure|kpi)|(?:define|add)\b[^?]{0,20}\b(?:metric|figure|"
               r"measure|kpi)|(?:créer|ajouter|définir)\b[^?]{0,20}\bindicateur",
     "Write it in the chat as name = how it is counted (“Average basket = net sales divided by the number of "
     "orders”) and it is worked out for this conversation. To keep it for everyone, ask your admin with the "
     "button under the answer, or from the Data guide.",
     "Écrivez-le dans la conversation sous la forme nom = calcul (« Panier moyen = ventes nettes divisées par "
     "le nombre de commandes ») : il est calculé pour cette conversation. Pour le garder pour tous, demandez-le "
     "à votre administrateur avec le bouton sous la réponse, ou depuis le Guide des données."),
    ("chart", r"(?:change|switch)\s+(?:the\s+)?(?:chart|graph)(?:\s+type)?\b|(?:changer|modifier)\s+(?:le\s+)?"
              r"(?:type\s+de\s+)?graphique",
     "Say how you want it drawn (“as a line”, “as a table”, “as a map”), or use the "
     "chart type buttons above the chart. How lines and colours look is set in your Profile, under Charts.",
     "Dites comment le dessiner (« en courbe », « en tableau », « en carte »), ou "
     "utilisez les boutons de type au-dessus du graphique. L'apparence des courbes et des couleurs se règle dans "
     "votre Profil, sous Graphiques."),
    ("password", r"(?:change|reset|update|set)\s+(?:my\s+|the\s+)?password\b|sign\s+out\b|log\s*out\b|"
                 r"(?:turn\s+on|switch\s+to|use)\s+dark\s+mode\b|(?:changer|réinitialiser|modifier)\s+(?:mon\s+|le\s+)?"
                 r"mot\s+de\s+passe\b|me\s+déconnecter\b|(?:passer\s+en|activer\s+le)\s+mode\s+sombre\b",
     "Open your Profile from the menu with your name: your password and devices are under Security, and how your "
     "charts look under Charts.",
     "Ouvrez votre Profil depuis le menu à votre nom : votre mot de passe et vos appareils sont sous Sécurité, et "
     "l'apparence de vos graphiques sous Graphiques."),
]
_OPENING = r"^\s*(?:(?:ok(?:ay)?|so|and|hi|hello|please|bonjour|alors|et)\s*,?\s+)*"
_ON_THE_WAY = r"(?:(?:also|just|please|easily|quickly|aussi|facilement)\s+)*"
_TOPIC_RES = [(key, re.compile(rf"{_OPENING}{_HOW}\s+{_ON_THE_WAY}(?:{words})", re.IGNORECASE), en, fr)
              for key, words, en, fr in _TOPICS]
_WHAT_ARE_YOU = re.compile(
    r"^\s*(?:what\s+(?:can|do)\s+you\s+do|what\s+are\s+you|who\s+are\s+you|what\s+is\s+(?:solviz|this\s+(?:app|tool))|"
    r"how\s+does\s+(?:this|solviz)\s+work|help|aide|que\s+(?:peux|sais)-tu\s+faire|qui\s+es-tu|"
    r"qu'est-ce\s+que\s+(?:solviz|cet\s+outil))\s*[?!.]*\s*$", re.IGNORECASE)
_ABOUT_YOU = ("I answer questions about your business from your own data: totals, breakdowns, trends, rankings, "
              "comparisons and why a figure moved, each with the numbers it was worked out from. Ask as you would "
              "ask an analyst (“net sales by store this year against last year”), follow up on an answer, "
              "add it to a dashboard, or ask “what data do you have?” to see what there is.",
              "Je réponds aux questions sur votre activité à partir de vos propres données : totaux, répartitions, "
              "tendances, classements, comparaisons et raisons d'une variation, chaque fois avec les chiffres "
              "utilisés. Posez la question comme à un analyste (« ventes nettes par magasin cette année contre "
              "l'an dernier »), enchaînez sur une réponse, ajoutez-la à un tableau de bord, ou demandez « "
              "quelles données avez-vous ? » pour voir ce qu'il y a.")
_FRENCH = re.compile(r"\b(?:comment|puis-je|peut-on|est-il|je|tu|vous|tableau|données|quelles?|quels?|pourquoi|"
                     r"combien|aide|où|ou\s+puis|télécharger|partager|qu'est-ce|peux|sais|que|faire|quel|quelle|"
                     r"fera|demain|temps|ventes|magasins?)\b", re.IGNORECASE)


_QUOTES = re.compile(r""""[^"\n]*"|“[^”\n]*”|«[^»\n]*»|(?<![\w])'[^'\n]*'(?![\w])""")
# Words only English writes, to set against the French ones: "comment" and "temps" are English words too ("gave a
# comment to a post").
_ENGLISH = re.compile(r"\b(?:the|is|was|were|are|what|which|who|how|of|to|and|with|for|that|this|did|does|my|by|"
                      r"from|in|at|show|list|give)\b", re.IGNORECASE)


def french(question: str) -> bool:
    """Is the question written in French? More of its words French than English; names in quotes count for
    neither (« ventes de "The Corner Shop" »), nor an English word written with a capital inside the sentence: it
    is a name's (« ventes pour The Corner Shop and Co »)."""
    text = _QUOTES.sub(" ", question or "")
    found = len(_FRENCH.findall(text))
    english = sum(1 for m in _ENGLISH.finditer(text)
                  if m.group(0).islower() or not text[:m.start()].strip() or text[:m.start()].rstrip()[-1] in ".?!:;")
    return found > 0 and found > english


def about_solviz(*, lang: str) -> str:
    """What Solviz does, for a question about it no topic above answers."""
    return _ABOUT_YOU[1 if lang == "fr" else 0]


def product_help(question: str) -> tuple[str, str] | None:
    """The answer to a question about Solviz itself, in the reader's language, with its topic; None for any other."""
    fr = french(question)
    if _WHAT_ARE_YOU.match(question or ""):
        return _ABOUT_YOU[1 if fr else 0], "about"
    for key, pattern, en, fr_text in _TOPIC_RES:
        if pattern.search(question or ""):
            return (fr_text if fr else en), key
    return None


# ── "What does X mean?" ─────────────────────────────────────────────────────

# "What does EBITDA mean?", "what is a churn rate?", "define net revenue", "que signifie EBITDA ?": the thing asked
# about, when the plan named none ("what is the data about?" asks about the data as a whole).
_EN = r"(?:an?\s+|the\s+|our\s+|my\s+|your\s+|their\s+|its\s+)?"
_FR = r"(?:une?\s+|le\s+|la\s+|les\s+|l['’]|notre\s+|nos\s+|votre\s+|vos\s+|mon\s+|ma\s+|mes\s+)?"
_MEANING = re.compile(
    rf"^\s*(?:what\s+(?:does|do)\s+{_EN}(?P<a>.+?)\s+(?:mean|stand\s+for)|"
    rf"what\s+(?:is|are)\s+{_EN}(?P<b>.+?)|define\s+{_EN}(?P<c>.+?)|"
    rf"(?:the\s+)?(?:meaning|definition)\s+of\s+{_EN}(?P<d>.+?)|"
    rf"qu['’]est-ce\s+qu(?:e|['’])\s*{_FR}(?P<e>.+?)|"
    rf"que\s+(?:signifie|veut\s+dire)\s+{_FR}(?P<f>.+?))\s*[?.!]*\s*$", re.IGNORECASE)
# What the data holds, asked of the data as a whole ("what is in the database?", "what are the KPIs?",
# "qu'est-ce que je peux demander ?"): its overview, never the meaning of a word.
_THE_DATA = re.compile(r"\b(?:data|dataset|database|db|données|tables?|fields?|columns?|champs?|colonnes?|metrics?|"
                       r"measures?|kpis?|indicateurs?|available|disponibles?|covered|cover|here|there|ask|demander|"
                       r"avez|have|this|that|it|ça|cela|ceci)\b", re.IGNORECASE)


def meaning_asked(question: str) -> str:
    """What a "what does X mean?" question asks about, or "" for one about the data as a whole."""
    found = _MEANING.match(question or "")
    thing = next((g for g in found.groups() if g), "") if found else ""
    return "" if not thing or _THE_DATA.search(thing) else thing.strip()


# ── Anything else ────────────────────────────────────────────────────────────

_DECLINES = {
    "en": ["That one is outside what I can help with: I answer questions about your business from your data. "
           "Here are a few you could try.",
           "I can't help with that, I'm afraid; I only know your business through its data. These are questions I "
           "can answer.",
           "That's beyond me: I work from your data, and only your data. Try one of these instead.",
           "I'll leave that one to someone better placed: my job is your data. Here is what I can look at."],
    "fr": ["Cette question sort de ce que je sais faire : je réponds sur votre activité à partir de vos données. "
           "Voici quelques questions à essayer.",
           "Je ne peux pas vous aider sur ce point ; je ne connais votre activité que par ses données. Voici des "
           "questions auxquelles je sais répondre.",
           "C'est hors de ma portée : je travaille à partir de vos données, et seulement d'elles. Essayez plutôt "
           "l'une de celles-ci.",
           "Je laisse cette question à quelqu'un de mieux placé : mon domaine, ce sont vos données. Voici ce que je "
           "peux regarder."],
}


def decline(question: str, *, lang: str) -> str:
    """A friendly decline, worded one of a few ways (the same question is always declined the same way)."""
    ways = _DECLINES["fr" if lang == "fr" else "en"]
    pick = int(hashlib.sha256((question or "").strip().casefold().encode()).hexdigest(), 16) % len(ways)
    return ways[pick]


_QUALIFIERS = {"monthly", "annual", "yearly", "weekly", "daily", "average", "avg", "total", "net", "gross"}


def held(names: list[str], said: str) -> list[str]:
    """The data's names of two words or more that the words name ("monthly recurring revenue" in "monthly recurring
    revenue by product", by its initials "MRR", or without its leading qualifier, "recurring revenue"): what the data
    holds, though the question could not be answered from it."""
    words = " " + " ".join(re.findall(r"[a-z0-9]+", (said or "").casefold())) + " "
    out = []
    for name in names:
        parts = [w for w in re.findall(r"[a-z0-9]+", name.casefold()) if w not in ("of", "the", "a", "an")]
        if len(parts) < 2 or name in out:
            continue
        ways = [" ".join(parts)]
        if len(parts) >= 3 and parts[0] in _QUALIFIERS:
            ways.append(" ".join(parts[1:]))
        initials = "".join(w[0] for w in parts)
        if len(initials) >= 3:
            ways.append(initials)
        if any(f" {way} " in words for way in ways):
            out.append(name)
    return out


def not_here(wanted: str, why: str, holds: str, *, lang: str, has: list[str] | None = None) -> tuple[str, str]:
    """What the data does not hold, and what it does: the reply's sentence and its note. Where the words name what
    the data does hold (``has``), it is never said not to hold it: the answer could not be worked out from it."""
    if has:
        names = ", ".join(has)
        if lang == "fr":
            return (f"Ces données contiennent {names}, mais je n'ai pas pu en tirer la réponse : {why}",
                    "Essayez de poser la question autrement, ou demandez à votre administrateur d'en faire un indicateur.")
        return (f"This data holds {names}, but I could not work the answer out from it: {why}",
                "Try asking another way, or ask your admin to make it a metric.")
    if lang == "fr":
        lead = f"Ces données ne couvrent pas ce sujet : {wanted}." if wanted else \
            f"Je ne peux pas répondre à cela avec ces données : {why}"
        return lead, (f"Elles contiennent {holds}. Vous pouvez demander ces données à votre administrateur."
                      if holds else "Vous pouvez demander ces données à votre administrateur.")
    lead = f"Your data does not hold {wanted}." if wanted else f"I cannot answer that from this data: {why}"
    return lead, (f"It holds {holds}. You can ask your admin to add this data." if holds
                  else "You can ask your admin to add this data.")


def general_off(*, lang: str) -> str:
    """What is said to a general business question where the admin has switched general knowledge off."""
    if lang == "fr":
        return ("C'est une question sur les affaires en général, et je ne réponds qu'à partir de vos données. Voici "
                "ce que je peux regarder.")
    return ("That's a question about business in general, and I answer from your data only. Here is what I can "
            "look at.")


# ── The AI sorting a refusal ─────────────────────────────────────────────────

SYSTEM = """A business reader asked a question that their company's data cannot answer. Sort it.

You are given the question, why the data cannot answer it, and the names of the tables the reader can use.
Reply with one JSON object: {"kind": "...", "wanted": "...", "answer": "..."}.

kind is one of:
- "general_knowledge": a question about business, finance, operations, sales, supply chain or analysis in general
  that does not depend on this company's own figures ("what is a good inventory turnover for a distributor?",
  "what does EBITDA mean?", "how do analysts usually measure churn?"). Put the answer in answer: two to four plain
  sentences of general knowledge, in {language}. Say what is usual or how it is generally done; never state or
  guess a figure about this company, and never say what its data shows.
- "data_not_here": a question about this company's own figures or records that its tables do not hold (a question
  on employees when the tables hold sales). Put in wanted the data it needs, in a few words, as a reader would name
  it, in {language} ("employee headcount"). Leave answer empty.
- "product": a question about how to use this analytics tool itself (dashboards, sharing, exporting, alerts,
  charts, settings, what it can do). Leave wanted and answer empty.
- "off_topic": anything else (the weather, sport, news, jokes, personal advice, writing code). Leave wanted and
  answer empty.

No markdown, no lists, no quotation marks inside the answer."""


@dataclass
class Sorted:
    kind: str
    wanted: str = ""
    answer: str = ""
    notes: list[str] = field(default_factory=list)
    unsorted: bool = False      # no AI sorted it (none, or it failed or replied otherwise): "not in the data" stands


_FROM_THE_DATA = re.compile(
    r"\b(?:your|the|this)\s+(?:data|figures|numbers|records|tables?)\s+(?:shows?|says|indicates?|suggests?|holds?|"
    r"contains?)\b|\b(?:in|from)\s+your\s+(?:data|figures|numbers|records)\b|"
    r"\b(?:according\s+to|based\s+on|looking\s+at|judging\s+by|going\s+by|from)\s+(?:your|our)\b|"
    r"\b(?:your|our)\s+(?:own\s+)?(?:\w+\s+){0,2}(?:is|are|was|were|grew|fell|rose|dropped|stands?|stood|came\s+to|"
    r"reached|totals?|totalled|turns?|turned)\s+(?:at\s+|about\s+|around\s+|by\s+|to\s+)?[$€£]?\d|"
    r"\b(?:d['’]après|selon|au\s+vu\s+de)\s+(?:vos|votre|nos|notre)\b|\b(?:vos|nos)\s+(?:données|chiffres)\b|"
    r"\b(?:votre|vos|notre|nos)\s+(?:\w+\s+){0,2}(?:est|sont|était|étaient|a\s+(?:augmenté|baissé|atteint)|atteint)"
    r"\s+(?:de\s+|d['’]environ\s+|à\s+)?\d", re.IGNORECASE)


def _general(text: str, tables: list[str] | None = None) -> str:
    """A general-knowledge answer as it may be shown: plain sentences of a reasonable length, never claiming to come
    from the reader's data or stating its figures ("looking at your order lines, net sales grew 12%"); "" when it is
    not one."""
    said = " ".join(str(text or "").split()).strip().strip('"')
    if not said or len(said.split()) > MAX_WORDS or said.startswith(("#", "-", "*", "{", "[")) or "```" in said:
        return ""
    own = [re.escape(t.casefold()) for t in tables or [] if t]
    yours = re.compile(rf"\b(?:your|our)\s+(?:{'|'.join(own)})\w*\b[^.]*\d", re.IGNORECASE) if own else None
    if _FROM_THE_DATA.search(said) or (yours is not None and yours.search(said)):
        log.info("A general-knowledge answer was not shown: it says it comes from the reader's data")
        return ""
    return said


def material(question: str, why: str, tables: list[str]) -> str:
    """What the AI is given: the question, why the plan stopped, and the reader's tables by name."""
    return "\n".join([f"Question: {question}", f"Why the data cannot answer it: {why or 'not given'}",
                      f"The reader's tables: {', '.join(tables) or 'none'}"])


def sort(question: str, why: str, tables: list[str], complete: Callable[[str, str], str] | None, *,
         lang: str) -> Sorted:
    """The refusal sorted by the workspace's AI; a plain "not in the data" where there is none to ask, or when it fails
    or replies otherwise."""
    if complete is None:
        return Sorted(NOT_HERE, unsorted=True)
    try:
        reply = complete(SYSTEM.replace("{language}", "French" if lang == "fr" else "English"),
                         material(question, why, tables))
    except Exception as exc:  # noqa: BLE001 - the refusal stands as it is
        log.warning("The workspace's AI could not sort a question the data cannot answer: %s", exc)
        return Sorted(NOT_HERE, unsorted=True)
    text = str(reply or "").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        found = json.loads(text[start:end + 1]) if 0 <= start < end else None
    except ValueError:
        found = None
    if not isinstance(found, dict):
        log.warning("The sort of a question the data cannot answer held no JSON object: kept as not in the data")
        return Sorted(NOT_HERE, unsorted=True)
    kind = str(found.get("kind") or "")
    if kind == GENERAL:
        answer = _general(str(found.get("answer") or ""), tables)
        return Sorted(GENERAL, answer=answer) if answer else Sorted(NOT_HERE)
    if kind == OFF_TOPIC:
        return Sorted(OFF_TOPIC)
    if kind in ("product", PRODUCT):
        return Sorted(PRODUCT)
    wanted = " ".join(str(found.get("wanted") or "").split()).strip(" .")[:80]
    return Sorted(NOT_HERE, wanted=wanted)


_WORD = re.compile(r"[a-zà-ÿ]{4,}", re.IGNORECASE)


def nearest(question: str, wanted: str, examples: list[str], most: int = 3) -> list[str]:
    """The questions to try nearest to one the data cannot answer: those sharing its words first, in their order."""
    words = {w.casefold()[:5] for w in _WORD.findall(f"{question} {wanted}")}
    scored = sorted(enumerate(examples), key=lambda e: (-len(words & {w.casefold()[:5] for w in _WORD.findall(e[1])}),
                                                         e[0]))
    return [q for _, q in scored][:most]
