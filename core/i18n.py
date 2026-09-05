"""
core/i18n.py
────────────
One message catalogue, delivered three ways: to Python prose producers, to Jinja
markup, and to the browser's inline scripts.

Why a catalogue and not gettext
───────────────────────────────
The toolchain is the only real argument for gettext, and it does not apply here.
``xgettext`` cannot extract an f-string, and 426 of this product's ~1,300
user-facing strings are f-strings (263 of them interpolating mid-sentence), so a
gettext project would begin with exactly the same manual rewrite of every
producer that this needs -- and then add a .po/.mo compile step to a deploy that
ships plain files. ``babel`` is also not a dependency here.

Its locale model is wrong for this server besides. ``gettext.install()`` and
``locale.setlocale`` are process- or thread-global, and this is an async FastAPI
app with two users in different languages in flight on one event loop. That
class of problem is already solved in this codebase with a ContextVar --
core/vocab_packs.py, core/llm_audit.py, core/pipeline_trace.py -- so this module
copies that shape rather than inventing a fourth one.

Why post-hoc translation of the payload cannot work
───────────────────────────────────────────────────
``sanitize_response_text_fields`` in core/response_builder.py already walks the
whole outgoing payload keyed by field name and looks like the perfect single
hook. By the time the payload reaches it the field holds

    "Acme SA leads at 1,250,000."

-- copy, tenant entity name and a locale-formatted number fused into one opaque
string. There is no way to separate the translatable third, and an LLM pass over
it would rewrite tenant data values. Message ids have to be introduced at the
producer.

Rules this module enforces
──────────────────────────
1. **Whole sentences, never fragments.** A caller that glues clauses together
   and then sentence-cases the result cannot be translated: French word order
   and agreement do not survive it. Register one id per whole sentence, with a
   separate id for each shape.
2. **Named placeholders only.** ``{count}``, never ``{0}`` or ``{}``. Word order
   changes between languages; positional placeholders silently swap arguments.
3. **English is the fallback for everything.** An unknown id, an unknown
   language, or a missing keyword all degrade to something readable rather than
   raising. A missing translation is a cosmetic bug; an exception in an answer
   path is an outage.
4. **Opt-in per call site.** Nothing is scanned or auto-extracted, so SQL, log
   lines, audit records, column names and tenant data values are excluded by
   construction -- they are only translated if someone rewrites a producer to
   call ``t()``.
5. **Sentinels are refused.** Some strings look like copy and are actually
   protocol: ``"redacted segment"`` is produced by core/insight.py and compared
   BY EQUALITY in two places, so translating it makes the comparison fail and
   the real label reaches the user. ``FORBIDDEN_VALUES`` below is checked by a
   test rather than trusted to reviewer memory.
"""

from __future__ import annotations

import re
from contextvars import ContextVar

DEFAULT_LANGUAGE = "en"

# Kept in step with store.user_store.SUPPORTED_LANGUAGES; asserted by a test
# rather than imported, because store imports are heavier than this module wants
# and a drift between the two is exactly what the test is for.
SUPPORTED_LANGUAGES = ("en", "fr")

# Endonyms -- each language's name IN that language. Deliberately not catalogue
# entries: the whole point of a language switcher is that someone who cannot
# read the current language can still find their own, so "Français" must read
# "Français" on an English page too. Translating these would defeat the control.
LANGUAGE_NAMES: dict[str, str] = {"en": "English", "fr": "Français"}

# Strings that must never appear as a catalogue value in ANY language, because
# something compares them by equality:
#   "redacted segment"  core/insight.py::_display_label, compared at
#                       core/response_builder.py::_safe_category_label and
#                       core/insight.py; a translated sentinel fails the
#                       comparison and the unredacted label is rendered.
#   "[REDACTED]"        core/compliance/result_guard.py::_mask output, matched
#                       by the _MASKED_MARKERS sets in core/result_planner.py,
#                       core/result_commands.py and core/forecast_gate.py.
#   "CANNOT_GENERATE"   the model's refusal token, compared in core/validator.py.
FORBIDDEN_VALUES = ("redacted segment", "[REDACTED]", "CANNOT_GENERATE")

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


# ══════════════════════════════════════════════════════════════════════════════
# The catalogue
# ══════════════════════════════════════════════════════════════════════════════
#
# msg_id -> {language: template}. Ids are namespaced by surface so the browser
# can be sent one subtree rather than the whole catalogue.
#
#   ui.*        portal chrome (Jinja markup and inline scripts)
#   answer.*    the deterministic answer surface
#   caveat.*    coverage caveats
#
MESSAGES: dict[str, dict[str, str]] = {
    # ── The add-to-dashboard picker ──────────────────────────────────────────
    "ui.pin.title": {
        "en": "Add to dashboard",
        "fr": "Ajouter au tableau de bord",
    },
    "ui.pin.subtitle": {
        "en": "Choose where this live, governed result should appear.",
        "fr": "Choisissez où ce résultat gouverné et actualisé doit apparaître.",
    },
    "ui.pin.mode_existing": {
        "en": "Existing dashboard",
        "fr": "Tableau de bord existant",
    },
    "ui.pin.mode_new": {"en": "Create new", "fr": "Créer un tableau de bord"},
    "ui.pin.search_placeholder": {
        "en": "Search your dashboards",
        "fr": "Rechercher dans vos tableaux de bord",
    },
    "ui.pin.loading": {
        "en": "Loading dashboards…",
        "fr": "Chargement des tableaux de bord…",
    },
    "ui.pin.none_yet": {
        "en": "You have no dashboards yet. Create one to pin this result.",
        "fr": "Vous n'avez pas encore de tableau de bord. Créez-en un pour épingler ce résultat.",
    },
    "ui.pin.no_match": {
        "en": "No dashboard matches “{query}”.",
        "fr": "Aucun tableau de bord ne correspond à « {query} ».",
    },
    "ui.pin.name_label": {"en": "Dashboard name", "fr": "Nom du tableau de bord"},
    "ui.pin.name_placeholder": {
        "en": "e.g. Pharmacy performance",
        "fr": "ex. Performance pharmacie",
    },
    "ui.pin.description_label": {"en": "Description", "fr": "Description"},
    "ui.pin.optional": {"en": "(optional)", "fr": "(facultatif)"},
    "ui.pin.description_placeholder": {
        "en": "What this dashboard is used for",
        "fr": "À quoi sert ce tableau de bord",
    },
    "ui.pin.visibility_label": {"en": "Visibility", "fr": "Visibilité"},
    "ui.pin.visibility_personal": {"en": "Personal", "fr": "Personnel"},
    "ui.pin.visibility_team": {"en": "Team draft", "fr": "Brouillon d'équipe"},
    "ui.pin.adding_what": {
        "en": "Adding: {title}",
        "fr": "À ajouter : {title}",
    },
    "ui.pin.submit": {"en": "Add chart", "fr": "Ajouter le graphique"},
    "ui.pin.submit_new": {
        "en": "Create and add",
        "fr": "Créer et ajouter",
    },
    "ui.pin.submitting": {"en": "Adding…", "fr": "Ajout en cours…"},
    "ui.pin.cancel": {"en": "Cancel", "fr": "Annuler"},
    "ui.pin.close": {"en": "Close", "fr": "Fermer"},
    "ui.pin.added": {
        "en": "Added to {dashboard}.",
        "fr": "Ajouté à {dashboard}.",
    },
    "ui.pin.err_name_required": {
        "en": "Give the new dashboard a name.",
        "fr": "Donnez un nom au nouveau tableau de bord.",
    },
    "ui.pin.err_pick_one": {
        "en": "Choose a dashboard, or create a new one.",
        "fr": "Choisissez un tableau de bord ou créez-en un.",
    },
    "ui.pin.err_expired": {
        "en": "This result can no longer be pinned. Run the question again and retry.",
        "fr": "Ce résultat ne peut plus être épinglé. Relancez la question puis réessayez.",
    },
    "ui.pin.err_forbidden": {
        "en": "You do not have access to that dashboard.",
        "fr": "Vous n'avez pas accès à ce tableau de bord.",
    },
    "ui.pin.err_generic": {
        "en": "The chart could not be added, and nothing was changed. "
              "Run the question again to retry.",
        "fr": "Le graphique n'a pas pu être ajouté et rien n'a été modifié. "
              "Relancez la question pour réessayer.",
    },
    "ui.pin.err_offline": {
        "en": "Could not reach the server. Nothing was changed — check your "
              "connection and try again.",
        "fr": "Serveur injoignable. Rien n'a été modifié — vérifiez votre "
              "connexion et réessayez.",
    },
    "ui.pin.err_load": {
        "en": "Your dashboards could not be loaded. You can still create a new one.",
        "fr": "Vos tableaux de bord n'ont pas pu être chargés. Vous pouvez tout de "
              "même en créer un.",
    },
    "ui.pin.err_session": {
        "en": "Your session ended. Sign in again to add this result.",
        "fr": "Votre session a expiré. Reconnectez-vous pour ajouter ce résultat.",
    },
    "ui.pin.already_added": {
        "en": "Already added to {dashboard}. Run the question again to add it a second time.",
        "fr": "Déjà ajouté à {dashboard}. Relancez la question pour l'ajouter une seconde fois.",
    },
    # ── The dashboard page ───────────────────────────────────────────────────
    "ui.dash.kicker": {"en": "Dashboard artifact", "fr": "Tableau de bord"},
    "ui.dash.version": {"en": "Version {version}", "fr": "Version {version}"},
    "ui.dash.refresh": {"en": "{schedule} refresh", "fr": "Actualisation {schedule}"},
    "ui.dash.read_only": {"en": "Read only", "fr": "Lecture seule"},
    "ui.dash.following": {"en": "Following · {cadence}", "fr": "Abonné · {cadence}"},
    "ui.dash.unfollow_hint": {
        "en": "Stop following this dashboard",
        "fr": "Ne plus suivre ce tableau de bord",
    },
    "ui.dash.subscribe": {"en": "Subscribe", "fr": "S'abonner"},
    "ui.dash.cadence_label": {
        "en": "Dashboard subscription cadence",
        "fr": "Fréquence de l'abonnement au tableau de bord",
    },
    "ui.dash.chat_with": {"en": "Chat with dashboard", "fr": "Discuter du tableau de bord"},
    "ui.dash.unpublished_title": {
        "en": "Your teammates cannot see this dashboard.",
        "fr": "Vos collègues ne voient pas ce tableau de bord.",
    },
    "ui.dash.unpublished_body": {
        "en": "It has unpublished changes, and a team dashboard is only visible "
              "to others while it is published.",
        "fr": "Il contient des modifications non publiées, et un tableau de bord "
              "d'équipe n'est visible par les autres que lorsqu'il est publié.",
    },
    "ui.dash.publish": {"en": "Publish", "fr": "Publier"},
    "ui.dash.tabs_label": {"en": "Dashboard tabs", "fr": "Onglets du tableau de bord"},
    "ui.dash.filter_placeholder": {"en": "Filter {field}", "fr": "Filtrer {field}"},
    "ui.dash.apply_filters": {"en": "Apply filters", "fr": "Appliquer les filtres"},
    "ui.dash.clear": {"en": "Clear", "fr": "Effacer"},
    "ui.dash.tab_visuals": {"en": "{tab} visuals", "fr": "Visuels : {tab}"},
    "ui.dash.drag_hint": {
        "en": "Drag or resize to edit placement",
        "fr": "Faites glisser ou redimensionnez pour modifier la disposition",
    },
    "ui.dash.live_data": {"en": "Live governed data", "fr": "Données gouvernées en direct"},
    "ui.dash.rename_hint": {"en": "Double-click to rename", "fr": "Double-cliquez pour renommer"},
    "ui.dash.drag_handle_title": {
        "en": "Drag to reorder or resize from the card corner",
        "fr": "Faites glisser pour réordonner, ou redimensionnez par le coin de la carte",
    },
    "ui.dash.drag_handle_label": {
        "en": "Drag chart to reorder",
        "fr": "Déplacer le graphique pour le réordonner",
    },
    "ui.dash.showing_of": {
        "en": "Showing {shown} of {total} rows",
        "fr": "Affichage de {shown} sur {total} lignes",
    },
    "ui.dash.cache": {"en": "Protected cache {at}", "fr": "Cache protégé {at}"},
    "ui.dash.live_refresh": {"en": "Live governed refresh", "fr": "Actualisation gouvernée en direct"},
    "ui.dash.expand": {"en": "Expand", "fr": "Agrandir"},
    "ui.dash.expand_title": {"en": "Expand chart", "fr": "Agrandir le graphique"},
    "ui.dash.remove": {"en": "Remove", "fr": "Retirer"},
    "ui.dash.chart_failed": {"en": "Chart could not refresh", "fr": "Le graphique n'a pas pu être actualisé"},
    "ui.dash.no_data": {"en": "No data returned for this query.", "fr": "Aucune donnée renvoyée pour cette requête."},
    "ui.dash.empty_title": {
        "en": "No visuals in this dashboard yet",
        "fr": "Aucun visuel dans ce tableau de bord",
    },
    "ui.dash.empty_body": {
        "en": "Ask QueryBot for an analysis, then use “Add to dashboard” and "
              "select this dashboard.",
        "fr": "Demandez une analyse à QueryBot, puis utilisez « Ajouter au "
              "tableau de bord » et choisissez ce tableau de bord.",
    },
    "ui.dash.browse_semantic": {"en": "Browse Semantic Layer", "fr": "Parcourir la couche sémantique"},
    "ui.dash.open_chat": {"en": "Open Chat", "fr": "Ouvrir le chat"},
    "ui.dash.no_dashboards_title": {"en": "No dashboards yet", "fr": "Aucun tableau de bord"},
    "ui.dash.no_dashboards_body": {
        "en": "Create an analysis in Chat and add the resulting KPI, chart, or "
              "table to a new named dashboard.",
        "fr": "Créez une analyse dans le chat puis ajoutez le KPI, le graphique "
              "ou le tableau obtenu à un nouveau tableau de bord nommé.",
    },
    "ui.dash.sources": {"en": "Data sources · {count}", "fr": "Sources de données · {count}"},
    "ui.dash.no_sources": {"en": "No reusable sources yet", "fr": "Aucune source réutilisable"},
    "ui.dash.history": {"en": "Revision history · {count}", "fr": "Historique des révisions · {count}"},
    "ui.dash.restore": {"en": "Restore", "fr": "Restaurer"},
    "ui.dash.current": {"en": "Current", "fr": "Version actuelle"},
    "ui.dash.published_team": {"en": "Published team dashboard", "fr": "Tableau de bord d'équipe publié"},
    "ui.dash.published_team_body": {
        "en": "You can view and filter this dashboard. Only its owner can edit "
              "or restore it.",
        "fr": "Vous pouvez consulter et filtrer ce tableau de bord. Seul son "
              "propriétaire peut le modifier ou le restaurer.",
    },
    "ui.dash.provenance_label": {"en": "Dashboard provenance", "fr": "Provenance du tableau de bord"},
    "ui.dash.workspace_drawer": {
        "en": "Workspace usage and table access",
        "fr": "Utilisation de l'espace de travail et accès aux tables",
    },
    "ui.dash.title": {"en": "Dashboards", "fr": "Tableaux de bord"},
    "ui.dash.subtitle": {
        "en": "Organize governed KPIs, charts, and tables for the decisions your "
              "team follows.",
        "fr": "Organisez les KPI, graphiques et tableaux gouvernés qui guident "
              "les décisions de votre équipe.",
    },
    "ui.dash.library_title": {"en": "Your dashboard artifacts", "fr": "Vos tableaux de bord"},
    "ui.dash.library_count": {"en": "{count} saved", "fr": "{count} enregistrés"},
    "ui.dash.shared_with_team": {"en": "Shared with team", "fr": "Partagé avec l'équipe"},
    "ui.dash.welcome": {"en": "Welcome to QueryBot, {name}", "fr": "Bienvenue sur QueryBot, {name}"},
    "ui.dash.welcome_body": {
        "en": "Your account is ready. Add governed results to named dashboards "
              "and refresh them with live data.",
        "fr": "Votre compte est prêt. Ajoutez des résultats gouvernés à des "
              "tableaux de bord nommés et actualisez-les avec des données en direct.",
    },
    "ui.dash.open_chat_arrow": {"en": "Open chat →", "fr": "Ouvrir le chat →"},
    # ── Workspace usage ──────────────────────────────────────────────────────
    "ui.dash.kpi_visuals": {"en": "Dashboard visuals", "fr": "Visuels du tableau de bord"},
    "ui.dash.kpi_queries": {"en": "Queries this month", "fr": "Requêtes ce mois-ci"},
    "ui.dash.kpi_query_limit": {"en": "Query limit", "fr": "Limite de requêtes"},
    "ui.dash.kpi_left": {"en": "{count} left", "fr": "{count} restantes"},
    "ui.dash.kpi_used": {"en": "{used} / {limit} used", "fr": "{used} / {limit} utilisées"},
    "ui.dash.kpi_tokens": {"en": "Remaining tokens", "fr": "Jetons restants"},
    "ui.dash.kpi_no_cap": {"en": "No monthly token cap", "fr": "Aucun plafond mensuel de jetons"},
    "ui.dash.kpi_group": {"en": "Group", "fr": "Groupe"},
    "ui.dash.kpi_tables": {"en": "Tables access", "fr": "Accès aux tables"},
    "ui.dash.kpi_all": {"en": "All", "fr": "Toutes"},
    "ui.dash.table_access": {"en": "Your table access ({count} tables)", "fr": "Vos accès aux tables ({count} tables)"},
    "ui.dash.view_semantic": {"en": "View Semantic Layer →", "fr": "Voir la couche sémantique →"},
    "ui.dash.no_group": {
        "en": "You haven't been assigned to a group yet. Contact your "
              "administrator to get table access.",
        "fr": "Vous n'êtes pas encore affecté à un groupe. Contactez votre "
              "administrateur pour obtenir l'accès aux tables.",
    },
    "ui.dash.limit_reached": {
        "en": "Monthly query limit reached: {used} / {limit} used. Ask your "
              "administrator to increase the workspace limit.",
        "fr": "Limite mensuelle de requêtes atteinte : {used} / {limit} "
              "utilisées. Demandez à votre administrateur d'augmenter la limite.",
    },
    "ui.dash.limit_warning": {
        "en": "Query usage is above 80%: {used} / {limit} used this month.",
        "fr": "L'utilisation dépasse 80 % : {used} / {limit} utilisées ce mois-ci.",
    },
    # ── Counts. French makes ZERO singular, English makes it plural, so these
    # cannot be one template with an inline `{% if n != 1 %}s{% endif %}` --
    # which is what the page had, and which renders "0 visuels" in French.
    "ui.dash.rows.one": {"en": "{count} row", "fr": "{count} ligne"},
    "ui.dash.rows.other": {"en": "{count} rows", "fr": "{count} lignes"},
    "ui.dash.visuals.one": {"en": "{count} visual", "fr": "{count} visuel"},
    "ui.dash.visuals.other": {"en": "{count} visuals", "fr": "{count} visuels"},
    # ── Server enums. Rendered with |capitalize before, which cannot translate
    # and mangles anything the database happens to store in another case.
    "ui.enum.status.draft": {"en": "Draft", "fr": "Brouillon"},
    "ui.enum.status.published": {"en": "Published", "fr": "Publié"},
    "ui.enum.visibility.personal": {"en": "Personal", "fr": "Personnel"},
    "ui.enum.visibility.team": {"en": "Team", "fr": "Équipe"},
    "ui.enum.cadence.daily": {"en": "Daily", "fr": "Quotidien"},
    "ui.enum.cadence.weekly": {"en": "Weekly", "fr": "Hebdomadaire"},
    "ui.enum.cadence.monthly": {"en": "Monthly", "fr": "Mensuel"},
    "ui.enum.schedule.manual": {"en": "Manual", "fr": "Manuelle"},
    "ui.enum.schedule.daily": {"en": "Daily", "fr": "Quotidienne"},
    "ui.enum.schedule.weekly": {"en": "Weekly", "fr": "Hebdomadaire"},
    "ui.enum.schedule.monthly": {"en": "Monthly", "fr": "Mensuelle"},
    # Chart-type badges. KPI stays KPI -- it is an international acronym, and
    # French finance uses it untranslated -- but "table" and "chart" are words.
    "ui.enum.charttype.kpi": {"en": "KPI", "fr": "KPI"},
    "ui.enum.charttype.table": {"en": "Table", "fr": "Tableau"},
    "ui.enum.charttype.chart": {"en": "Chart", "fr": "Graphique"},
    "ui.enum.charttype.bar": {"en": "Bar", "fr": "Barres"},
    "ui.enum.charttype.line": {"en": "Line", "fr": "Courbe"},
    "ui.enum.charttype.area": {"en": "Area", "fr": "Aires"},
    "ui.enum.charttype.pie": {"en": "Pie", "fr": "Secteurs"},
    "ui.enum.charttype.donut": {"en": "Donut", "fr": "Anneau"},
    "ui.enum.charttype.scatter": {"en": "Scatter", "fr": "Nuage de points"},
    "ui.enum.role.admin": {"en": "Admin", "fr": "Administrateur"},
    "ui.enum.role.analyst": {"en": "Analyst", "fr": "Analyste"},
    "ui.enum.source.governed_query": {"en": "Governed query", "fr": "Requête gouvernée"},
    "ui.enum.source.metric": {"en": "Metric", "fr": "Métrique"},
    "ui.enum.source.report": {"en": "Report", "fr": "Rapport"},

    # ── The portal shell ─────────────────────────────────────────────────────
    "ui.shell.nav_label": {"en": "Portal navigation", "fr": "Navigation du portail"},
    "ui.shell.nav_primary": {"en": "Primary", "fr": "Principale"},
    "ui.shell.home": {"en": "QueryBot dashboard", "fr": "Tableau de bord QueryBot"},
    "ui.shell.collapse": {"en": "Collapse sidebar", "fr": "Réduire le menu latéral"},
    "ui.shell.expand": {"en": "Expand sidebar", "fr": "Développer le menu latéral"},
    "ui.shell.open_nav": {"en": "Open navigation", "fr": "Ouvrir la navigation"},
    "ui.shell.close_nav": {"en": "Close navigation", "fr": "Fermer la navigation"},
    "ui.shell.new_thread": {"en": "New Thread", "fr": "Nouvelle conversation"},
    "ui.shell.chat": {"en": "Chat", "fr": "Chat"},
    "ui.shell.dashboard": {"en": "Dashboard", "fr": "Tableau de bord"},
    "ui.shell.semantic_layer": {"en": "Semantic Layer", "fr": "Couche sémantique"},
    "ui.shell.notifications": {"en": "Notifications", "fr": "Notifications"},
    "ui.shell.no_group": {"en": "No group", "fr": "Aucun groupe"},
    "ui.shell.account_actions": {"en": "Account actions", "fr": "Actions du compte"},
    "ui.shell.settings": {"en": "Settings", "fr": "Paramètres"},
    "ui.shell.logout": {"en": "Logout", "fr": "Déconnexion"},
    # The house confirm dialog. Its defaults are set in JS, so they need the
    # catalogue on the browser side too.
    "ui.shell.confirm_title": {"en": "Are you sure?", "fr": "Confirmer l'action ?"},
    "ui.shell.cancel": {"en": "Cancel", "fr": "Annuler"},
    "ui.shell.confirm": {"en": "Confirm", "fr": "Confirmer"},
    # The live toast. The status sentence was assembled by concatenation --
    # `(column || 'Field') + ' was ' + statusText + ' by admin.'` -- which
    # cannot be translated as fragments: French puts the participle after the
    # auxiliary and agrees it with the subject. One whole sentence per outcome.
    "ui.shell.toast_semantic": {"en": "Semantic Layer updated", "fr": "Couche sémantique mise à jour"},
    "ui.shell.toast_approved_title": {
        "en": "Semantic Layer change approved",
        "fr": "Modification de la couche sémantique approuvée",
    },
    "ui.shell.toast_rejected_title": {
        "en": "Semantic Layer change rejected",
        "fr": "Modification de la couche sémantique refusée",
    },
    "ui.shell.toast_approved_body": {
        "en": "{field} was approved by an administrator.",
        "fr": "{field} a été approuvé par un administrateur.",
    },
    "ui.shell.toast_rejected_body": {
        "en": "{field} was rejected by an administrator.",
        "fr": "{field} a été refusé par un administrateur.",
    },
    "ui.shell.limit_reached_title": {
        "en": "Monthly query limit reached",
        "fr": "Limite mensuelle de requêtes atteinte",
    },
    "ui.shell.limit_warning_title": {
        "en": "Monthly query limit warning",
        "fr": "Alerte de limite mensuelle de requêtes",
    },
    "ui.shell.limit_title": {
        "en": "Monthly query limit",
        "fr": "Limite mensuelle de requêtes",
    },
    "ui.shell.limit_reached_body.one": {
        "en": "{count}/{limit} query used this month. Ask your admin to increase the limit.",
        "fr": "{count}/{limit} requête utilisée ce mois-ci. Demandez à votre administrateur d'augmenter la limite.",
    },
    "ui.shell.limit_reached_body.other": {
        "en": "{count}/{limit} queries used this month. Ask your admin to increase the limit.",
        "fr": "{count}/{limit} requêtes utilisées ce mois-ci. Demandez à votre administrateur d'augmenter la limite.",
    },
    "ui.shell.limit_warning_body.one": {
        "en": "{count}/{limit} query used this month. Your workspace is above 80% of its limit.",
        "fr": "{count}/{limit} requête utilisée ce mois-ci. Votre espace de travail dépasse 80 % de sa limite.",
    },
    "ui.shell.limit_warning_body.other": {
        "en": "{count}/{limit} queries used this month. Your workspace is above 80% of its limit.",
        "fr": "{count}/{limit} requêtes utilisées ce mois-ci. Votre espace de travail dépasse 80 % de sa limite.",
    },
    "ui.shell.limit_remaining_body.one": {
        "en": "{count} query remaining this month.",
        "fr": "{count} requête restante ce mois-ci.",
    },
    "ui.shell.limit_remaining_body.other": {
        "en": "{count} queries remaining this month.",
        "fr": "{count} requêtes restantes ce mois-ci.",
    },
    "ui.shell.toast_field": {"en": "Field", "fr": "Champ"},
    "ui.shell.toast_system": {"en": "System Notice", "fr": "Message système"},
    "ui.shell.toast_system_body": {"en": "System update", "fr": "Mise à jour du système"},
    "ui.shell.toast_query_limit": {"en": "Monthly query limit", "fr": "Limite mensuelle de requêtes"},

    # ── The answer card's deterministic sentences ────────────────────────────
    #
    # These are not chrome. When a question is not causal enough for the
    # narration model to run, these sentences ARE the answer the reader gets,
    # so an English card under a French portal is the product failing to answer
    # in the reader's language rather than a cosmetic gap.
    #
    # Whole sentences, one per outcome, rather than a stem plus an adverb.
    # English builds "{measure} rose 12.3% from Q1 to Q2" by joining a verb to a
    # percentage; French conjugates ("a augmenté de 12,3 %") and there is no
    # seam in the middle to translate.

    # The scope badge and its note.
    "answer.scope.returned": {"en": "Returned result", "fr": "Résultat renvoyé"},
    "answer.scope.returned_note": {
        "en": "This reflects the rows returned by the query.",
        "fr": "Ceci correspond aux lignes renvoyées par la requête.",
    },
    "answer.scope.top_one": {"en": "Top result only", "fr": "Meilleur résultat uniquement"},
    "answer.scope.top_one_note": {
        "en": "This result is based on the top-ranked row only, not the full distribution.",
        "fr": "Ce résultat repose uniquement sur la ligne la mieux classée, et non sur la distribution complète.",
    },
    "answer.scope.top_n": {"en": "Top {n} only", "fr": "{n} premiers uniquement"},
    "answer.scope.top_n_note": {
        "en": "This result is based only on the top {n} returned rows.",
        "fr": "Ce résultat repose uniquement sur les {n} premières lignes renvoyées.",
    },
    "answer.scope.full_distribution": {"en": "Full distribution", "fr": "Distribution complète"},
    "answer.scope.full_distribution_note": {
        "en": "This result reflects the full returned distribution.",
        "fr": "Ce résultat reflète la distribution complète renvoyée.",
    },
    "answer.scope.full_series": {"en": "Full series", "fr": "Série complète"},
    "answer.scope.full_series_note": {
        "en": "This result reflects the full returned time series.",
        "fr": "Ce résultat reflète la série temporelle complète renvoyée.",
    },
    "answer.scope.preview": {"en": "Preview", "fr": "Aperçu"},
    "answer.scope.preview_note": {
        "en": "This result is a preview because the returned rows are capped for display.",
        "fr": "Ce résultat est un aperçu : le nombre de lignes affichées est plafonné.",
    },
    "answer.scope.filtered_subset": {"en": "Filtered subset", "fr": "Sous-ensemble filtré"},
    "answer.scope.filtered_subset_note": {
        "en": "This result reflects a filtered subset defined by the query conditions.",
        "fr": "Ce résultat correspond à un sous-ensemble filtré par les conditions de la requête.",
    },
    "answer.scope.slice_note": {
        "en": "Interpret this as a returned slice rather than a complete picture.",
        "fr": "À interpréter comme un extrait renvoyé, et non comme une vue complète.",
    },

    # Counts. French takes the singular at zero, which the inline
    # `{'s' if n != 1 else ''}` these replace could not express.
    "answer.rows.one": {"en": "{count} row", "fr": "{count} ligne"},
    "answer.rows.other": {"en": "{count} rows", "fr": "{count} lignes"},
    "answer.across_results.one": {"en": "Across {count} result", "fr": "Sur {count} résultat"},
    "answer.across_results.other": {"en": "Across {count} results", "fr": "Sur {count} résultats"},

    # Nothing matched.
    "answer.no_match_headline": {
        "en": "No matching data was found for this question.",
        "fr": "Aucune donnée correspondante n'a été trouvée pour cette question.",
    },
    "answer.no_match_hint": {
        "en": "Try adjusting the filters or time range.",
        "fr": "Essayez d'ajuster les filtres ou la période.",
    },

    # A successful aggregate over no facts -- an empty analytical result, not
    # a failure, so the copy stays calm in both languages.
    "answer.no_data": {"en": "No data", "fr": "Aucune donnée"},
    "answer.no_metric_headline": {
        "en": "No {metric} data was found for {target}.",
        "fr": "Aucune donnée de {metric} n'a été trouvée pour {target}.",
    },
    "answer.no_metric_comparison": {
        "en": "The query completed successfully, but no metric value was returned.",
        "fr": "La requête a abouti, mais aucune valeur de mesure n'a été renvoyée.",
    },
    "answer.no_metric_note": {
        "en": "There are no matching {metric} values for {target}.",
        "fr": "Il n'existe aucune valeur de {metric} correspondant à {target}.",
    },
    "answer.target_period": {"en": "the requested period", "fr": "la période demandée"},
    "answer.target_filters": {"en": "the current filters", "fr": "les filtres actuels"},

    # An aggregate that matched rows but every value was NULL.
    "answer.null_metric_headline": {
        "en": "{metric}: {value} because all matched values are missing.",
        "fr": "{metric} : {value}, car toutes les valeurs correspondantes sont manquantes.",
    },
    "answer.null_metric_comparison.one": {
        "en": "{count} matching record, 0 non-null {metric} values",
        "fr": "{count} enregistrement correspondant, 0 valeur de {metric} non nulle",
    },
    "answer.null_metric_comparison.other": {
        "en": "{count} matching records, 0 non-null {metric} values",
        "fr": "{count} enregistrements correspondants, 0 valeur de {metric} non nulle",
    },
    "answer.null_metric_badge": {
        "en": "Missing metric values", "fr": "Valeurs de mesure manquantes",
    },
    "answer.null_metric_note.one": {
        "en": "The filter matched {count} record, but the requested metric column had no non-null values in it.",
        "fr": "Le filtre a retenu {count} enregistrement, mais la colonne de mesure demandée n'y contenait aucune valeur non nulle.",
    },
    "answer.null_metric_note.other": {
        "en": "The filter matched {count} records, but the requested metric column had no non-null values in those records.",
        "fr": "Le filtre a retenu {count} enregistrements, mais la colonne de mesure demandée n'y contenait aucune valeur non nulle.",
    },

    # A single scalar. French puts a space before the colon, which is why this
    # is a message and not an f-string join.
    "answer.label_value": {"en": "{label}: {value}", "fr": "{label} : {value}"},
    "answer.single_value": {"en": "Single-value result", "fr": "Résultat à valeur unique"},

    # A named-period comparison. Six ids rather than a verb slot: French
    # conjugates the movement and agrees it with the measure.
    "answer.period_rose": {
        "en": "{measure} rose {pct} from {old} to {new}",
        "fr": "{measure} a augmenté de {pct} entre {old} et {new}",
    },
    "answer.period_fell": {
        "en": "{measure} fell {pct} from {old} to {new}",
        "fr": "{measure} a diminué de {pct} entre {old} et {new}",
    },
    "answer.period_flat": {
        "en": "{measure} was flat from {old} to {new}",
        "fr": "{measure} est resté stable entre {old} et {new}",
    },
    "answer.period_rose_unquantified": {
        "en": "{measure} rose from {old} to {new}",
        "fr": "{measure} a augmenté entre {old} et {new}",
    },
    "answer.period_fell_unquantified": {
        "en": "{measure} fell from {old} to {new}",
        "fr": "{measure} a diminué entre {old} et {new}",
    },
    "answer.period_mover": {
        "en": "{sentence}; {mover} moved the most, {change}",
        "fr": "{sentence} ; c'est {mover} qui a le plus varié, {change}",
    },
    "answer.period_mover_unnamed": {
        "en": "{sentence}; the largest single move was {change}",
        "fr": "{sentence} ; la plus forte variation isolée est de {change}",
    },
    "answer.period_versus": {
        "en": "{pct} versus {old}", "fr": "{pct} par rapport à {old}",
    },
    "answer.period_compared": {
        "en": "compared with {old}", "fr": "par rapport à {old}",
    },
    "answer.total": {"en": "Total", "fr": "Total"},

    # A time series.
    "answer.series_close": {
        "en": "{label} closed at {value}.", "fr": "{label} a terminé à {value}.",
    },
    "answer.trend_up": {
        "en": "Trend is up versus {value} at the start",
        "fr": "Tendance à la hausse par rapport à {value} au départ",
    },
    "answer.trend_down": {
        "en": "Trend is down versus {value} at the start",
        "fr": "Tendance à la baisse par rapport à {value} au départ",
    },
    "answer.trend_flat": {
        "en": "Trend is flat versus {value} at the start",
        "fr": "Tendance stable par rapport à {value} au départ",
    },
    "answer.latest_period": {"en": "Latest period", "fr": "Dernière période"},

    # A ranking.
    "answer.leads": {
        "en": "{label} leads at {value}.", "fr": "{label} arrive en tête avec {value}.",
    },
    "answer.top_ranked": {
        "en": "Top-ranked result: {label} at {value}.",
        "fr": "Résultat le mieux classé : {label}, avec {value}.",
    },
    "answer.leading_row_only": {
        "en": "This card shows only the leading row",
        "fr": "Cette carte n'affiche que la première ligne",
    },
    "answer.above_next": {
        "en": "{delta} above the next result",
        "fr": "{delta} de plus que le résultat suivant",
    },
    "answer.top_result": {"en": "Top result", "fr": "Meilleur résultat"},

    # Numbers with no label column to hang them on.
    "answer.returned_rows.one": {
        "en": "Returned {count} row for {question}.",
        "fr": "{count} ligne renvoyée pour {question}.",
    },
    "answer.returned_rows.other": {
        "en": "Returned {count} rows for {question}.",
        "fr": "{count} lignes renvoyées pour {question}.",
    },
    "answer.range": {"en": "Range {low} to {high}", "fr": "Plage de {low} à {high}"},
    "answer.this_query": {"en": "this query", "fr": "cette requête"},

    # A list of names.
    "answer.found_results.one": {
        "en": "Found {count} result for: {question}",
        "fr": "{count} résultat trouvé pour : {question}",
    },
    "answer.found_results.other": {
        "en": "Found {count} results for: {question}",
        "fr": "{count} résultats trouvés pour : {question}",
    },
    "answer.more_items": {"en": "+{count} more", "fr": "+{count} autres"},
    "answer.review_records": {
        "en": "Review the records below", "fr": "Consultez les enregistrements ci-dessous",
    },
    "answer.your_query": {"en": "your query", "fr": "votre requête"},

    # ── The notes under the answer card ──────────────────────────────────────
    #
    # _build_insight_summary, _build_anomaly_callouts and _build_decision_signal
    # are pure statistics -- no model call -- and the chat page merges all three
    # into one .answer-notes list. They are the reader's second sentence after
    # the headline, so they follow it into the reader's language.

    "answer.value": {"en": "Value", "fr": "Valeur"},
    "answer.entries": {"en": "entries", "fr": "entrées"},
    "answer.groups": {"en": "groups", "fr": "groupes"},
    "answer.returned_period": {"en": "the returned period", "fr": "la période renvoyée"},
    "answer.first_period": {"en": "the first period", "fr": "la première période"},
    "answer.second_period": {"en": "the second period", "fr": "la deuxième période"},

    # Movement suffixes. The sentence goes in whole and comes out whole for the
    # same reason the headline's does: French conjugates the direction.
    "answer.note.up": {"en": "{sentence} - up {pct}.", "fr": "{sentence} — en hausse de {pct}."},
    "answer.note.down": {"en": "{sentence} - down {pct}.", "fr": "{sentence} — en baisse de {pct}."},
    "answer.note.unchanged": {"en": "{sentence} - unchanged.", "fr": "{sentence} — inchangé."},

    "answer.note.null_metric.one": {
        "en": "{count} record matched, but {metric} is missing for it.",
        "fr": "{count} enregistrement correspondait, mais {metric} y est absent.",
    },
    "answer.note.null_metric.other": {
        "en": "{count} records matched, but {metric} is missing for every matched row.",
        "fr": "{count} enregistrements correspondaient, mais {metric} est absent pour chacun d'eux.",
    },
    "answer.note.single_value": {"en": "{label}: {value}.", "fr": "{label} : {value}."},
    "answer.note.period_versus": {
        "en": "{measure} was {current} in {current_period} versus {previous} in {previous_period}",
        "fr": "{measure} s'élevait à {current} en {current_period} contre {previous} en {previous_period}",
    },
    "answer.note.single_observation": {
        "en": "{measure} was {value} in {period}.",
        "fr": "{measure} s'élevait à {value} en {period}.",
    },
    "answer.note.changed_from": {
        "en": "{measure} changed from {first} in {first_period} to {last} in {last_period}",
        "fr": "{measure} est passé de {first} en {first_period} à {last} en {last_period}",
    },
    "answer.note.trended_up": {
        "en": "{measure} trended up {pct} from {first} to {last}.",
        "fr": "{measure} a progressé de {pct} entre {first} et {last}.",
    },
    "answer.note.trended_down": {
        "en": "{measure} trended down {pct} from {first} to {last}.",
        "fr": "{measure} a reculé de {pct} entre {first} et {last}.",
    },
    "answer.note.trended_flat": {
        "en": "{measure} trended flat {pct} from {first} to {last}.",
        "fr": "{measure} est resté stable ({pct}) entre {first} et {last}.",
    },
    "answer.note.remained_up": {
        "en": "{measure} remained up between {first} and {last}.",
        "fr": "{measure} est resté orienté à la hausse entre {first} et {last}.",
    },
    "answer.note.remained_down": {
        "en": "{measure} remained down between {first} and {last}.",
        "fr": "{measure} est resté orienté à la baisse entre {first} et {last}.",
    },
    "answer.note.remained_flat": {
        "en": "{measure} remained flat between {first} and {last}.",
        "fr": "{measure} est resté stable entre {first} et {last}.",
    },
    "answer.note.peak": {
        "en": "{sentence} Peak: {value} in {period}.",
        "fr": "{sentence} Pic : {value} en {period}.",
    },
    "answer.note.leader_share": {
        "en": " ({pct}% of total)", "fr": " ({pct} % du total)",
    },
    "answer.note.leads_across": {
        "en": "{leader} leads at {value}{share} across {count} {label}.",
        "fr": "{leader} arrive en tête avec {value}{share}, sur {count} {label}.",
    },
    "answer.note.range_summary.one": {
        "en": "{count} record — {measure} ranges {low} to {high}, avg {avg}.",
        "fr": "{count} enregistrement — {measure} varie de {low} à {high}, moyenne {avg}.",
    },
    "answer.note.range_summary.other": {
        "en": "{count} records — {measure} ranges {low} to {high}, avg {avg}.",
        "fr": "{count} enregistrements — {measure} varie de {low} à {high}, moyenne {avg}.",
    },

    # The statistical callouts.
    "answer.callout.biggest_drop": {
        "en": "Biggest drop: {old} → {new} ({pct})",
        "fr": "Plus forte baisse : {old} → {new} ({pct})",
    },
    "answer.callout.biggest_gain": {
        "en": "Biggest gain: {old} → {new} ({pct})",
        "fr": "Plus forte hausse : {old} → {new} ({pct})",
    },
    "answer.callout.decline_streak.one": {
        "en": "{count} consecutive period of decline",
        "fr": "{count} période de baisse consécutive",
    },
    "answer.callout.decline_streak.other": {
        "en": "{count} consecutive periods of decline",
        "fr": "{count} périodes de baisse consécutives",
    },
    "answer.callout.concentration": {
        "en": "Top 3 entries account for {pct}% of total — highly concentrated",
        "fr": "Les 3 premières entrées représentent {pct} % du total — forte concentration",
    },
    "answer.callout.dominance": {
        "en": "{label} holds {pct}% of the total",
        "fr": "{label} détient {pct} % du total",
    },
    "answer.callout.outlier": {
        "en": "Outlier in {column}: max {high} vs avg {avg}",
        "fr": "Valeur aberrante dans {column} : max {high} contre moyenne {avg}",
    },

    # The "so what" line.
    "answer.signal.concentration": {
        "en": "Top entries drive {pct}% of the total — concentration risk if any one is lost.",
        "fr": "Les premières entrées représentent {pct} % du total — risque de concentration si l'une d'elles est perdue.",
    },
    "answer.signal.dominance": {
        "en": "{leader} alone holds {pct}% of the total — a single point of dependency.",
        "fr": "{leader} détient à lui seul {pct} % du total — un point de dépendance unique.",
    },
    "answer.signal.spread": {
        "en": "Volume is spread across the field — no single entry exceeds {pct}%; broadly diversified.",
        "fr": "Le volume est réparti sur l'ensemble — aucune entrée ne dépasse {pct} % ; largement diversifié.",
    },
    "answer.signal.decline_pct": {
        "en": "Sustained downward trend ({pct} overall) — worth investigating before it compounds.",
        "fr": "Tendance baissière durable ({pct} au total) — à examiner avant que cela ne s'aggrave.",
    },
    "answer.signal.decline": {
        "en": "Sustained downward trend — worth investigating before it compounds.",
        "fr": "Tendance baissière durable — à examiner avant que cela ne s'aggrave.",
    },
    "answer.signal.growth": {
        "en": "Momentum is building ({pct} overall) — confirm it is sustainable, not a one-off spike.",
        "fr": "La dynamique s'installe ({pct} au total) — vérifiez qu'elle est durable et non un pic isolé.",
    },
    "answer.signal.stable": {
        "en": "Metric is holding steady over the period — no urgent action indicated.",
        "fr": "L'indicateur reste stable sur la période — aucune action urgente n'est requise.",
    },
    "answer.signal.outlier": {
        "en": "One or more values sit well above normal — review for data quality or a genuine signal before acting.",
        "fr": "Une ou plusieurs valeurs sortent nettement de l'ordinaire — vérifiez la qualité des données ou la réalité du signal avant d'agir.",
    },
    "answer.signal.single": {
        "en": "{comparison} — factor this into the decision.",
        "fr": "{comparison} — à prendre en compte dans la décision.",
    },

    # The named-period comparison note. The grew/shrank counts agree with their
    # own numbers in French, so each is built by plural() and dropped into the
    # opening as a finished clause.
    "answer.period.grew.one": {"en": "{count} grew", "fr": "{count} a augmenté"},
    "answer.period.grew.other": {"en": "{count} grew", "fr": "{count} ont augmenté"},
    "answer.period.shrank.one": {"en": "{count} shrank", "fr": "{count} a diminué"},
    "answer.period.shrank.other": {"en": "{count} shrank", "fr": "{count} ont diminué"},
    "answer.period.opening": {
        "en": "Across {count} {label}, {grew} and {shrank} between {old} and {new}.",
        "fr": "Sur {count} {label}, {grew} et {shrank} entre {old} et {new}.",
    },
    "answer.period.share": {
        "en": ", {pct}% of the net change", "fr": ", soit {pct} % de la variation nette",
    },
    "answer.period.added_most": {
        "en": "{who} added the most ({change}{share})",
        "fr": "c'est {who} qui a le plus progressé ({change}{share})",
    },
    "answer.period.largest_increase": {
        "en": "the largest increase was {change}{share}",
        "fr": "la plus forte hausse est de {change}{share}",
    },
    "answer.period.fell_most": {
        "en": "{who} fell the most ({change})",
        "fr": "c'est {who} qui a le plus reculé ({change})",
    },
    "answer.period.largest_decrease": {
        "en": "the largest decrease was {change}",
        "fr": "la plus forte baisse est de {change}",
    },
    "answer.period.none_moved": {
        "en": "{opening} No category moved between the two periods.",
        "fr": "{opening} Aucune catégorie n'a bougé entre les deux périodes.",
    },

    # ── The action chips under an answer, and the card each one opens ────────
    #
    # The chip ID routes the action and is never translated; only the label and
    # the hover hint are copy. Same for the analysis card's action key.

    "chip.compare": {"en": "Compare periods", "fr": "Comparer les périodes"},
    "chip.compare_hint": {
        "en": "{pct} overall change", "fr": "{pct} de variation globale",
    },
    "chip.diagnose_drop": {"en": "Why the drop?", "fr": "Pourquoi cette baisse ?"},
    "chip.diagnose_rise": {"en": "Why the rise?", "fr": "Pourquoi cette hausse ?"},
    "chip.diagnose_drop_hint": {
        "en": "{pct} drop — identify what drove this",
        "fr": "baisse de {pct} — identifier ce qui l'a provoquée",
    },
    "chip.diagnose_rise_hint": {
        "en": "{pct} rise — identify what drove this",
        "fr": "hausse de {pct} — identifier ce qui l'a provoquée",
    },
    "chip.compare_prior": {"en": "vs prior period", "fr": "vs période précédente"},
    "chip.compare_prior_hint": {
        "en": "Fetch the same metric for the previous cycle",
        "fr": "Récupérer le même indicateur pour le cycle précédent",
    },
    "chip.contribution": {
        "en": "Show % contribution", "fr": "Afficher la contribution en %",
    },
    "chip.contribution_hint": {
        "en": "{leader} holds {pct}% of total",
        "fr": "{leader} détient {pct} % du total",
    },
    "chip.drill_dim": {"en": "Break down by {name}", "fr": "Détailler par {name}"},
    "chip.drill_dim_hint": {
        "en": "Add {name} dimension to this result",
        "fr": "Ajouter la dimension {name} à ce résultat",
    },
    "chip.download_csv": {"en": "Download CSV", "fr": "Télécharger le CSV"},
    "chip.download_csv_hint.one": {
        "en": "{count} row ready to export", "fr": "{count} ligne prête à exporter",
    },
    "chip.download_csv_hint.other": {
        "en": "{count} rows ready to export", "fr": "{count} lignes prêtes à exporter",
    },

    # The titles on the card a chip opens.
    "analysis.title.explain": {"en": "Result explanation", "fr": "Explication du résultat"},
    "analysis.title.analyze": {"en": "Trend analysis", "fr": "Analyse de tendance"},
    "analysis.title.compare": {"en": "Compare periods", "fr": "Comparaison de périodes"},
    "analysis.title.predict": {"en": "Predict next period", "fr": "Prévision de la période suivante"},
    "analysis.title.why": {"en": "Why this pattern?", "fr": "Pourquoi ce schéma ?"},
    "analysis.title.decide": {"en": "Recommended next step", "fr": "Prochaine étape recommandée"},
    "analysis.title.default": {"en": "Analysis", "fr": "Analyse"},
    "analysis.failed_headline": {
        "en": "Analysis could not be completed.",
        "fr": "L'analyse n'a pas pu être menée à son terme.",
    },
    "analysis.failed_body": {
        "en": "The insight engine encountered an error: {detail}",
        "fr": "Le moteur d'analyse a rencontré une erreur : {detail}",
    },
    "analysis.failed_next": {
        "en": "Try rephrasing your question or running a more specific query.",
        "fr": "Essayez de reformuler votre question ou de poser une question plus précise.",
    },

    # ── The live stage trail ─────────────────────────────────────────────────
    #
    # Pushed from the pipeline as the answer is built, and shown by the chat
    # page. The pipeline activates the reader's language for the whole answer,
    # so these resolve there; portal_chat.html's STATUS_FALLBACK reads the same
    # ids, which also collapses a drift -- the browser's copy of
    # "executing_query" said "Executing the query against your connected
    # database" while the server said "Executing the SQL against your connected
    # data source". Same stage, two sentences, neither wrong.
    #
    # The STAGE KEY is the wire value and is never translated: the page picks
    # the mark's animation state from it.

    "stage.authorization.label": {"en": "Checking access", "fr": "Vérification des accès"},
    "stage.authorization.detail": {
        "en": "Verifying workspace access and available data.",
        "fr": "Vérification de vos accès et des données disponibles.",
    },
    "stage.analysing_results.label": {"en": "Analysing results", "fr": "Analyse des résultats"},
    "stage.analysing_results.detail": {
        "en": "Running a governed analysis on the previously returned data.",
        "fr": "Analyse gouvernée des données déjà renvoyées.",
    },
    "stage.building_trend.label": {"en": "Building the trend", "fr": "Construction de la tendance"},
    "stage.building_trend.detail": {
        "en": "Re-running the previous answer's governed query, grouped by its approved business date.",
        "fr": "Réexécution de la requête gouvernée précédente, regroupée par sa date métier approuvée.",
    },
    "stage.metric_registry.label": {"en": "Using known metric", "fr": "Utilisation d'un indicateur connu"},
    "stage.metric_registry.detail": {
        "en": "Found a trusted metric definition for this question.",
        "fr": "Une définition d'indicateur approuvée a été trouvée pour cette question.",
    },
    "stage.metric_query.label": {"en": "Running query", "fr": "Exécution de la requête"},
    "stage.metric_query.detail": {
        "en": "Executing the trusted metric query against your database.",
        "fr": "Exécution de la requête d'indicateur approuvée sur votre base de données.",
    },
    "stage.retrieving_context.label": {"en": "Understanding your data", "fr": "Compréhension de vos données"},
    "stage.retrieving_context.detail": {
        "en": "Retrieving the most relevant schema, examples, and business context.",
        "fr": "Récupération du schéma, des exemples et du contexte métier les plus pertinents.",
    },
    "stage.compiling_sql.label": {"en": "Compiling governed query", "fr": "Compilation de la requête gouvernée"},
    "stage.compiling_sql.detail": {
        "en": "Using the approved metric formula and business date mapping.",
        "fr": "Utilisation de la formule d'indicateur et du mappage de date métier approuvés.",
    },
    "stage.reusing_sql.label": {"en": "Using validated query", "fr": "Réutilisation d'une requête validée"},
    "stage.reusing_sql.detail": {
        "en": "Revalidating a successful governed query plan for this workspace.",
        "fr": "Revalidation d'un plan de requête gouvernée déjà réussi pour cet espace de travail.",
    },
    "stage.generating_sql.label": {"en": "Generating query", "fr": "Génération de la requête"},
    "stage.generating_sql.detail": {
        "en": "Translating the business question into SQL.",
        "fr": "Traduction de la question métier en SQL.",
    },
    "stage.recovering_sql.label": {"en": "Resolving query plan", "fr": "Résolution du plan de requête"},
    "stage.recovering_sql.detail": {
        "en": "Retrying once with the approved tables, fields, dates, and joins.",
        "fr": "Nouvelle tentative avec les tables, champs, dates et jointures approuvés.",
    },
    "stage.validating_sql.label": {"en": "Checking query safety", "fr": "Contrôle de sûreté de la requête"},
    "stage.validating_sql.detail": {
        "en": "Verifying table access, structure, and execution safety.",
        "fr": "Vérification des accès aux tables, de la structure et de la sûreté d'exécution.",
    },
    "stage.executing_query.label": {"en": "Running query", "fr": "Exécution de la requête"},
    "stage.executing_query.detail": {
        "en": "Executing the SQL against your connected data source.",
        "fr": "Exécution du SQL sur votre source de données connectée.",
    },
    "stage.repairing_query.label": {"en": "Repairing query", "fr": "Correction de la requête"},
    "stage.repairing_query.detail": {
        "en": "Fixing a validation or execution issue before retrying.",
        "fr": "Correction d'un problème de validation ou d'exécution avant nouvelle tentative.",
    },
    "stage.completing_repair.label": {"en": "Completing query repair", "fr": "Finalisation de la correction"},
    "stage.completing_repair.detail": {
        "en": "The first correction exposed another validation issue; applying one bounded follow-up repair.",
        "fr": "La première correction en a révélé une autre ; une seule correction complémentaire est appliquée.",
    },
    "stage.retrying_query.label": {"en": "Retrying query", "fr": "Nouvelle tentative"},
    "stage.retrying_query.detail": {
        "en": "Running the corrected query against your data.",
        "fr": "Exécution de la requête corrigée sur vos données.",
    },
    "stage.formatting_results.label": {"en": "Preparing results", "fr": "Préparation des résultats"},
    "stage.formatting_results.detail": {
        "en": "Formatting the answer and any chart for display.",
        "fr": "Mise en forme de la réponse et du graphique éventuel.",
    },
    "stage.chart_ready.label": {"en": "Building chart", "fr": "Construction du graphique"},
    "stage.chart_ready.detail": {
        "en": "Rendering an interactive chart for this answer.",
        "fr": "Rendu d'un graphique interactif pour cette réponse.",
    },

    # ── Live connection and run status, shown by the chat page ───────────────
    # The STATE KEY is the wire value in every case; only the label is copy.
    "ui.chat.connected": {"en": "Connected", "fr": "Connecté"},
    "ui.chat.reconnecting": {"en": "Reconnecting…", "fr": "Reconnexion…"},
    "ui.chat.restoring_session": {"en": "Restoring live session", "fr": "Rétablissement de la session en direct"},
    "ui.chat.session_active": {"en": "Live session active", "fr": "Session en direct active"},
    "ui.chat.connection_issue": {"en": "Connection issue", "fr": "Problème de connexion"},
    "ui.chat.recovering_session": {"en": "Trying to recover live session", "fr": "Tentative de rétablissement de la session"},
    "ui.chat.connection_lost_retry": {
        "en": "Connection lost. Retrying in {seconds}s",
        "fr": "Connexion perdue. Nouvelle tentative dans {seconds} s",
    },
    "ui.chat.draft_ready": {"en": "Draft ready to send", "fr": "Brouillon prêt à envoyer"},
    "ui.chat.retry_when_connected": {"en": "Retry when connected", "fr": "Réessayer une fois reconnecté"},
    "ui.chat.interrupted_retry": {
        "en": "Interrupted · Retry when connected",
        "fr": "Interrompu · Réessayer une fois reconnecté",
    },
    "ui.chat.need_clarification": {"en": "Need clarification…", "fr": "Précision nécessaire…"},

    # Shown when a status frame carries no stage the page recognises.
    "ui.chat.stage_generic": {"en": "Working on your answer", "fr": "Traitement de votre réponse"},
    "ui.chat.stage_generic_detail": {
        "en": "Preparing a trusted response.", "fr": "Préparation d'une réponse fiable.",
    },

    # The governed-agent run badge.
    "ui.chat.run.running": {"en": "Governed agent", "fr": "Agent gouverné"},
    "ui.chat.run.completed": {"en": "Governed answer", "fr": "Réponse gouvernée"},
    "ui.chat.run.waiting_for_user": {"en": "Waiting for your input", "fr": "En attente de votre réponse"},
    "ui.chat.run.blocked": {"en": "Blocked by policy", "fr": "Bloqué par la politique"},
    "ui.chat.run.failed": {"en": "Run failed", "fr": "Exécution échouée"},
    "ui.chat.run.cancelled": {"en": "Run cancelled", "fr": "Exécution annulée"},
    "ui.chat.run.read_only": {"en": "read only", "fr": "lecture seule"},

    # The composer mascot's status labels. Short and executive in tone: no
    # exclamations, no emoji, in either language.
    "ui.chat.mascot.idle": {"en": "Ready", "fr": "Prêt"},
    "ui.chat.mascot.focus": {"en": "Listening", "fr": "À l'écoute"},
    "ui.chat.mascot.typing": {"en": "Attending", "fr": "Attention"},
    "ui.chat.mascot.thinking": {"en": "Considering", "fr": "Réflexion"},
    "ui.chat.mascot.send": {"en": "Processing", "fr": "Traitement"},
    "ui.chat.mascot.ready": {"en": "Response ready", "fr": "Réponse prête"},
    "ui.chat.mascot.error": {"en": "Connection issue", "fr": "Problème de connexion"},

    # ── Chat toasts, drafts, message actions and feedback ────────────────────
    "ui.chat.toast.copied": {"en": "Copied to clipboard", "fr": "Copié dans le presse-papiers"},
    "ui.chat.toast.copy_failed": {"en": "Copy failed", "fr": "Échec de la copie"},
    "ui.chat.toast.result_copied": {"en": "Result copied", "fr": "Résultat copié"},
    "ui.chat.toast.analysis_copied": {"en": "Analysis copied", "fr": "Analyse copiée"},
    "ui.chat.toast.chart_copied": {"en": "Chart summary copied", "fr": "Résumé du graphique copié"},
    "ui.chat.toast.copied_to_composer": {
        "en": "Message copied to composer", "fr": "Message copié dans la zone de saisie",
    },
    "ui.chat.toast.csv_started": {"en": "CSV download started", "fr": "Téléchargement du CSV lancé"},
    "ui.chat.toast.session_not_ready": {"en": "Live session is not ready", "fr": "La session en direct n'est pas prête"},
    "ui.chat.toast.stop_before_retry": {
        "en": "Stop the current answer before retrying",
        "fr": "Arrêtez la réponse en cours avant de réessayer",
    },
    "ui.chat.toast.stop_before_send": {
        "en": "Stop the current answer before sending another message",
        "fr": "Arrêtez la réponse en cours avant d'envoyer un autre message",
    },
    "ui.chat.toast.thread_unavailable": {"en": "Thread is unavailable", "fr": "Conversation indisponible"},
    "ui.chat.toast.feedback_saved": {"en": "Feedback saved", "fr": "Avis enregistré"},
    "ui.chat.toast.feedback_failed": {"en": "Could not save feedback", "fr": "Impossible d'enregistrer l'avis"},
    "ui.chat.toast.limit_updated": {"en": "Monthly query limit updated.", "fr": "Limite mensuelle de requêtes mise à jour."},
    "ui.chat.toast.offline_saved": {
        "en": "You are offline. Your message is saved and ready to send after reconnection.",
        "fr": "Vous êtes hors ligne. Votre message est enregistré et sera envoyé après la reconnexion.",
    },
    "ui.chat.toast.send_failed_retry": {
        "en": "The message could not be sent. Use Retry when the connection returns.",
        "fr": "Le message n'a pas pu être envoyé. Utilisez Réessayer une fois la connexion rétablie.",
    },
    "ui.chat.toast.send_failed_available": {
        "en": "The message could not be sent. Your request is still available to retry.",
        "fr": "Le message n'a pas pu être envoyé. Votre demande reste disponible pour une nouvelle tentative.",
    },
    "ui.chat.toast.retry_when_live": {
        "en": "The message is ready to retry when the live connection returns.",
        "fr": "Le message pourra être renvoyé dès le retour de la connexion en direct.",
    },

    # The chat's own system lines.
    "ui.chat.system.session_unavailable": {
        "en": "Chat session unavailable. Refresh the page or sign in again.",
        "fr": "Session de chat indisponible. Actualisez la page ou reconnectez-vous.",
    },
    "ui.chat.system.csv_no_download": {
        "en": "The CSV was prepared, but the browser could not start the download.",
        "fr": "Le CSV a été préparé, mais le navigateur n'a pas pu démarrer le téléchargement.",
    },
    "ui.chat.system.action_timed_out": {
        "en": "This action did not finish in time. Please retry it.",
        "fr": "Cette action ne s'est pas terminée à temps. Veuillez réessayer.",
    },

    # Drafts kept in the composer.
    "ui.chat.draft.saved": {"en": "Draft saved", "fr": "Brouillon enregistré"},
    "ui.chat.draft.kept_offline": {"en": "Draft kept while offline", "fr": "Brouillon conservé hors ligne"},
    "ui.chat.draft.kept_reconnecting": {
        "en": "Draft kept while reconnecting", "fr": "Brouillon conservé pendant la reconnexion",
    },
    "ui.chat.draft.restored": {"en": "Draft restored", "fr": "Brouillon restauré"},
    "ui.chat.draft.trimmed": {
        "en": "Pasted text was trimmed to {limit} characters.",
        "fr": "Le texte collé a été réduit à {limit} caractères.",
    },

    # Buttons on a message.
    "ui.chat.action.edit": {"en": "Edit", "fr": "Modifier"},
    "ui.chat.action.retry": {"en": "Retry", "fr": "Réessayer"},
    "ui.chat.action.stop": {"en": "Stop generating", "fr": "Arrêter la génération"},

    # The thumbs-up / thumbs-down panel.
    "ui.chat.feedback.up": {"en": "Yes, correct answer", "fr": "Oui, réponse correcte"},
    "ui.chat.feedback.down": {"en": "No, something was wrong", "fr": "Non, quelque chose n'allait pas"},
    "ui.chat.feedback.not_helpful": {"en": "Not helpful", "fr": "Pas utile"},
    "ui.chat.feedback.comments": {
        "en": "Additional comments (optional)", "fr": "Commentaires supplémentaires (facultatif)",
    },
    "ui.chat.feedback.thanks": {"en": "Thanks!", "fr": "Merci !"},

    # ── The answer card as the browser draws it ──────────────────────────────
    "ui.chat.card.answer_ready": {"en": "Answer ready", "fr": "Réponse prête"},
    "ui.chat.card.copy": {"en": "Copy", "fr": "Copier"},
    "ui.chat.card.copied": {"en": "Copied!", "fr": "Copié !"},
    "ui.chat.card.next_step": {"en": "Next step: ", "fr": "Étape suivante : "},

    # The "How this answer was produced" disclosure.
    "ui.chat.trust.summary": {
        "en": "How this answer was produced", "fr": "Comment cette réponse a été produite",
    },
    "ui.chat.trust.open": {
        "en": "See how this answer was produced",
        "fr": "Voir comment cette réponse a été produite",
    },
    "ui.chat.trust.rows": {"en": "Rows", "fr": "Lignes"},
    "ui.chat.trust.runtime": {"en": "Runtime", "fr": "Durée"},
    "ui.chat.trust.data_source": {"en": "Data source", "fr": "Source de données"},
    "ui.chat.trust.schema_mode": {"en": "Schema mode", "fr": "Mode de schéma"},
    "ui.chat.trust.result_scope": {"en": "Result scope", "fr": "Portée du résultat"},
    "ui.chat.trust.execution": {"en": "Execution", "fr": "Exécution"},
    "ui.chat.trust.all_schemas": {"en": "All allowed schemas", "fr": "Tous les schémas autorisés"},
    "ui.chat.trust.schema_named": {"en": "{schema} schema", "fr": "schéma {schema}"},
    "ui.chat.trust.planner_metadata": {"en": "Metadata-only planner", "fr": "Planificateur sur métadonnées seules"},
    "ui.chat.trust.planner_none": {"en": "No LLM call", "fr": "Aucun appel au modèle"},
    "ui.chat.trust.bounded": {"en": "Bounded execution", "fr": "Exécution bornée"},
    "ui.chat.trust.no_sql": {
        "en": "No SQL query was executed for this result.",
        "fr": "Aucune requête SQL n'a été exécutée pour ce résultat.",
    },
    "ui.chat.trust.business_date": {"en": "Business date", "fr": "Date métier"},
    "ui.chat.trust.grain": {"en": "{grain} grain", "fr": "granularité {grain}"},
    "ui.chat.trust.calendar_grain": {"en": "calendar grain", "fr": "granularité calendaire"},
    "ui.chat.trust.inferred": {"en": "inferred encoded field", "fr": "champ encodé déduit"},
    "ui.chat.trust.governed": {"en": "governed date context", "fr": "contexte de date gouverné"},

    # The result table's own controls.
    "ui.chat.table.filter": {"en": "Filter rows…", "fr": "Filtrer les lignes…"},
    "ui.chat.table.filter_label": {"en": "Filter table rows", "fr": "Filtrer les lignes du tableau"},
    "ui.chat.table.download_csv": {"en": "Download as CSV", "fr": "Télécharger en CSV"},

    # The chart card.
    "ui.chat.chart.result": {"en": "Chart result", "fr": "Graphique"},
    "ui.chat.chart.expand": {"en": "Expand chart", "fr": "Agrandir le graphique"},
    "ui.chat.chart.pin": {"en": "Add to dashboard", "fr": "Ajouter au tableau de bord"},
    "ui.chat.chart.copy_summary": {"en": "Copy summary", "fr": "Copier le résumé"},
    "ui.chat.chart.rows": {"en": "{count} rows · {duration}", "fr": "{count} lignes · {duration}"},
    "ui.chat.chart.value_range": {"en": "Value range", "fr": "Plage de valeurs"},
    "ui.chat.chart.biggest_drop": {"en": "Biggest drop", "fr": "Plus forte baisse"},
    "ui.chat.chart.biggest_gain": {"en": "Biggest gain", "fr": "Plus forte hausse"},
    "ui.chat.chart.no_value_column": {"en": "No value column to plot", "fr": "Aucune colonne de valeurs à tracer"},
    "ui.chat.chart.no_rows": {"en": "No rows to plot", "fr": "Aucune ligne à tracer"},

    # The artifact pane's own copy.
    "ui.chat.artifact.result": {"en": "Analysis result", "fr": "Résultat d'analyse"},
    "ui.chat.artifact.isolated": {"en": "Isolated result analysis", "fr": "Analyse de résultat isolée"},
    "ui.chat.artifact.governed_note": {
        "en": "Governed result from the current conversation",
        "fr": "Résultat gouverné issu de la conversation en cours",
    },
    "ui.chat.artifact.single_value": {"en": "Single-value result", "fr": "Résultat à valeur unique"},
    "ui.chat.artifact.value": {"en": "Value", "fr": "Valeur"},
    "ui.chat.artifact.default_visual": {"en": "Dashboard visual", "fr": "Visuel de tableau de bord"},
    "ui.chat.artifact.dashboard_updated": {"en": "Dashboard updated", "fr": "Tableau de bord mis à jour"},
    "ui.chat.artifact.dashboard_note": {
        "en": "This artifact uses saved governed SQL and refreshes through your existing access policies.",
        "fr": "Cet élément utilise le SQL gouverné enregistré et s'actualise selon vos politiques d'accès existantes.",
    },
    "ui.chat.artifact.open_dashboard": {"en": "Open dashboard", "fr": "Ouvrir le tableau de bord"},
    "ui.chat.artifact.untitled": {"en": "Untitled dashboard", "fr": "Tableau de bord sans titre"},

    # The follow-up composer under a result.
    "ui.chat.followup.open": {"en": "Ask about this result", "fr": "Poser une question sur ce résultat"},
    "ui.chat.followup.placeholder": {
        "en": "e.g. what is the avg? who is above average? show top 5…",
        "fr": "ex. quelle est la moyenne ? qui est au-dessus ? afficher les 5 premiers…",
    },
    "ui.chat.followup.hint": {
        "en": "Ask about the result above. If your follow-up needs a metric or dimension missing from these rows, QueryBot can fall back to the database using the original answer context.",
        "fr": "Posez votre question sur le résultat ci-dessus. Si votre question porte sur un indicateur ou une dimension absents de ces lignes, QueryBot peut revenir à la base en s'appuyant sur le contexte de la réponse d'origine.",
    },

    # The feedback reasons. The VALUE is the wire enum and is never
    # translated -- store/learning_store.py groups on it.
    "ui.chat.reason.other": {"en": "Why was this wrong? (optional)", "fr": "Pourquoi était-ce incorrect ? (facultatif)"},
    "ui.chat.reason.wrong_metric": {"en": "Wrong metric", "fr": "Mauvais indicateur"},
    "ui.chat.reason.wrong_dimension": {"en": "Wrong dimension / grouping", "fr": "Mauvaise dimension / mauvais regroupement"},
    "ui.chat.reason.wrong_filter": {"en": "Wrong filter / date range", "fr": "Mauvais filtre / mauvaise période"},
    "ui.chat.reason.wrong_join": {"en": "Wrong join / relationship", "fr": "Mauvaise jointure / mauvaise relation"},
    "ui.chat.reason.wrong_data": {"en": "Wrong data / values", "fr": "Mauvaises données / mauvaises valeurs"},
    "ui.chat.reason.incomplete": {"en": "Incomplete answer", "fr": "Réponse incomplète"},
    "ui.chat.reason.confusing": {"en": "Confusing or unclear", "fr": "Confus ou peu clair"},
    "ui.chat.reason.expected_data_missing": {"en": "Expected data missing", "fr": "Données attendues manquantes"},
    "ui.chat.feedback.submit": {"en": "Submit feedback", "fr": "Envoyer l'avis"},

    # The proof strip inside the disclosure.
    "ui.chat.trust.sql_used": {"en": "SQL used", "fr": "SQL utilisé"},
    "ui.chat.trust.sources": {"en": "Sources", "fr": "Sources"},
    "ui.chat.trust.source": {"en": "source", "fr": "source"},
    "ui.chat.trust.no_db_query": {"en": "No database query", "fr": "Aucune requête à la base"},
    "ui.chat.trust.from_result": {
        "en": "Worked only from result {id}", "fr": "Travail effectué uniquement à partir du résultat {id}",
    },
    "ui.chat.trust.child_tasks.one": {"en": "{count} child task", "fr": "{count} tâche enfant"},
    "ui.chat.trust.child_tasks.other": {"en": "{count} child tasks", "fr": "{count} tâches enfants"},
    "ui.chat.trust.validated_code": {"en": "Validated code {hash}", "fr": "Code validé {hash}"},
    "ui.chat.trust.ast_nodes": {
        "en": "{count} AST nodes · planner input: {input}",
        "fr": "{count} nœuds AST · entrée du planificateur : {input}",
    },
    "ui.chat.trust.planner_none_detail": {
        "en": "0 result rows sent to any model",
        "fr": "0 ligne de résultat transmise à un modèle",
    },
    "ui.chat.trust.planner_metadata_detail": {
        "en": "Only column metadata and sanitized intent were sent; 0 rows and 0 sample values",
        "fr": "Seules les métadonnées de colonnes et l'intention nettoyée ont été transmises ; 0 ligne et 0 valeur d'exemple",
    },
    "ui.chat.trust.none": {"en": "none", "fr": "aucune"},

    # The dashboard confirmation strip under an answer.
    "ui.chat.confirm.updated": {"en": "{name} updated", "fr": "{name} mis à jour"},
    "ui.chat.confirm.saved_as": {
        "en": "Saved as a governed {kind} · {status}",
        "fr": "Enregistré comme {kind} gouverné · {status}",
    },
    "ui.chat.confirm.dashboard": {"en": "Dashboard", "fr": "Tableau de bord"},
    "ui.chat.confirm.visual": {"en": "visual", "fr": "visuel"},

    # ── Browser tab titles ───────────────────────────────────────────────────
    # Chrome the reader sees on every page, and the one string that is still in
    # front of them when the tab is in the background.
    "ui.title.suffix": {"en": "QueryBot Portal", "fr": "Portail QueryBot"},
    "ui.title.sign_in": {"en": "Sign In", "fr": "Connexion"},
    "ui.title.register": {"en": "Register", "fr": "Créer un compte"},
    "ui.title.change_password": {"en": "Change Password", "fr": "Changer le mot de passe"},
    "ui.title.notifications": {"en": "My Notifications", "fr": "Mes notifications"},
    "ui.title.new_report": {"en": "New Report", "fr": "Nouveau rapport"},

    # ── The chat page ────────────────────────────────────────────────────────
    #
    # Where a reader spends the session. Note the two "plain English" strings:
    # the phrase means "plain language", not "the English language", so the
    # French is "en langage courant" -- translating it literally would tell a
    # French customer to ask their questions in English.

    # The thread panel in the sidebar.
    "ui.chat.recent_threads": {"en": "Recent Threads", "fr": "Conversations récentes"},
    "ui.chat.refresh_threads": {"en": "Refresh threads", "fr": "Actualiser les conversations"},
    "ui.chat.search_threads": {"en": "Search threads", "fr": "Rechercher une conversation"},
    "ui.chat.clear_search": {"en": "Clear thread search", "fr": "Effacer la recherche"},
    "ui.chat.history_label": {"en": "Recent query history", "fr": "Historique des requêtes récentes"},
    "ui.chat.your_conversations": {"en": "Your conversations", "fr": "Vos conversations"},
    "ui.chat.close_history": {"en": "Close history", "fr": "Fermer l'historique"},
    "ui.chat.loading_threads": {"en": "Loading recent threads...", "fr": "Chargement des conversations récentes..."},
    "ui.chat.current_workspace": {"en": "Current workspace", "fr": "Espace de travail actuel"},
    "ui.chat.workspace": {"en": "Workspace", "fr": "Espace de travail"},
    "ui.chat.your_workspace": {"en": "Your workspace", "fr": "Votre espace de travail"},

    # Chat turned off for the workspace.
    "ui.chat.disabled_title": {
        "en": "Internal Chat not enabled", "fr": "Chat interne non activé",
    },
    "ui.chat.disabled_body": {
        "en": "Your administrator has not enabled the Internal Chat UI for your account.",
        "fr": "Votre administrateur n'a pas activé le chat interne pour votre compte.",
    },
    "ui.chat.back_to_dashboard": {
        "en": "Back to dashboard", "fr": "Retour au tableau de bord",
    },

    # The hero panel.
    "ui.chat.hero_title": {
        "en": "Ask your data in plain English",
        "fr": "Interrogez vos données en langage courant",
    },
    "ui.chat.hero_subtitle": {
        "en": "Query live business data, get structured answers, and pin useful charts to your dashboard without leaving the conversation.",
        "fr": "Interrogez vos données métier en direct, obtenez des réponses structurées et épinglez les graphiques utiles à votre tableau de bord sans quitter la conversation.",
    },
    "ui.chat.pill_workspace": {"en": "Workspace · {name}", "fr": "Espace de travail · {name}"},
    "ui.chat.pill_role": {"en": "Role · {role}", "fr": "Rôle · {role}"},
    "ui.chat.pill_group": {"en": "Group · {group}", "fr": "Groupe · {group}"},
    "ui.chat.not_assigned": {"en": "Not assigned", "fr": "Non attribué"},
    "ui.chat.tokens_title": {
        "en": "Input {input} · Output {output}", "fr": "Entrée {input} · Sortie {output}",
    },
    "ui.chat.tokens_pill": {
        "en": "Tokens this month · {total}", "fr": "Jetons ce mois-ci · {total}",
    },
    "ui.chat.queries_title": {
        "en": "{used} / {limit} queries used this month",
        "fr": "{used} / {limit} requêtes utilisées ce mois-ci",
    },
    "ui.chat.queries_left": {"en": "Queries left · {count}", "fr": "Requêtes restantes · {count}"},
    "ui.chat.hide_hero": {"en": "Hide hero panel", "fr": "Masquer le panneau d'accueil"},
    "ui.chat.dismiss_hero": {"en": "Dismiss hero panel", "fr": "Fermer le panneau d'accueil"},
    "ui.chat.show_info": {"en": "Show workspace info", "fr": "Afficher les informations de l'espace"},
    "ui.chat.show_info_panel": {
        "en": "Show workspace info panel",
        "fr": "Afficher le panneau d'informations de l'espace de travail",
    },
    "ui.chat.info": {"en": "Info", "fr": "Infos"},

    # The conversation shell.
    "ui.chat.shell_title": {"en": "QueryBot live analyst", "fr": "Analyste QueryBot en direct"},
    "ui.chat.shell_subtitle": {
        "en": "Interactive answers, chart previews, and governed dashboard workflows.",
        "fr": "Réponses interactives, aperçus de graphiques et flux de tableau de bord gouvernés.",
    },
    "ui.chat.connecting": {"en": "Connecting…", "fr": "Connexion…"},
    "ui.chat.waiting_session": {"en": "Waiting for live session", "fr": "En attente de la session en direct"},
    "ui.chat.view_history": {"en": "View recent query history", "fr": "Voir l'historique des requêtes récentes"},
    "ui.chat.history": {"en": "History", "fr": "Historique"},

    # The schema selector.
    "ui.chat.schema": {"en": "Schema", "fr": "Schéma"},
    "ui.chat.all": {"en": "All", "fr": "Tous"},
    "ui.chat.all_schemas": {"en": "All schemas", "fr": "Tous les schémas"},
    "ui.chat.table_count.one": {"en": "{count} table", "fr": "{count} table"},
    "ui.chat.table_count.other": {"en": "{count} tables", "fr": "{count} tables"},
    "ui.chat.multi_schema_title": {"en": "Multi-schema mode.", "fr": "Mode multi-schéma."},
    "ui.chat.multi_schema_body": {
        "en": "QueryBot will search all schemas you can access and may ask for clarification if more than one schema matches.",
        "fr": "QueryBot cherchera dans tous les schémas auxquels vous avez accès et pourra demander une précision si plusieurs correspondent.",
    },

    # The thread itself.
    "ui.chat.system_line": {
        "en": "QueryBot Internal Chat · {workspace}",
        "fr": "Chat interne QueryBot · {workspace}",
    },
    "ui.chat.welcome_title": {"en": "How can I help you today?", "fr": "Comment puis-je vous aider aujourd'hui ?"},
    "ui.chat.welcome_body": {
        "en": "Ask a business question in plain English and QueryBot will turn your governed data into a clear, traceable answer.",
        "fr": "Posez une question métier en langage courant et QueryBot transformera vos données gouvernées en une réponse claire et traçable.",
    },
    "ui.chat.thinking": {"en": "Thinking…", "fr": "Réflexion…"},
    "ui.chat.ready": {"en": "Ready", "fr": "Prêt"},

    # Suggestions.
    "ui.chat.suggestions_label": {
        "en": "Start with a workspace question",
        "fr": "Commencez par une question sur votre espace de travail",
    },
    "ui.chat.shuffle": {"en": "Show different suggestions", "fr": "Afficher d'autres suggestions"},
    "ui.chat.refresh": {"en": "Refresh", "fr": "Actualiser"},
    "ui.chat.workspace_insight": {"en": "Workspace insight", "fr": "Analyse de l'espace"},

    # The composer.
    "ui.chat.composer_placeholder": {
        "en": "Ask anything about your data…",
        "fr": "Posez n'importe quelle question sur vos données…",
    },
    "ui.chat.send": {"en": "Send message", "fr": "Envoyer le message"},
    "ui.chat.hint_all_schemas": {
        "en": "All schemas active. Select a schema when you want a stricter answer.",
        "fr": "Tous les schémas sont actifs. Sélectionnez-en un pour obtenir une réponse plus stricte.",
    },
    "ui.chat.hint_enter": {
        "en": "Press Enter to send · Shift + Enter for a new line",
        "fr": "Entrée pour envoyer · Maj + Entrée pour aller à la ligne",
    },

    # The artifact pane.
    "ui.chat.artifact_label": {"en": "Analysis artifact", "fr": "Panneau d'analyse"},
    "ui.chat.artifact_title": {"en": "Result preview", "fr": "Aperçu du résultat"},
    "ui.chat.close_artifact": {"en": "Close artifact", "fr": "Fermer le panneau"},
    "ui.chat.close": {"en": "Close", "fr": "Fermer"},
    "ui.chat.artifact_empty": {
        "en": "Charts, KPI values, and result tables open here while the explanation stays in the conversation.",
        "fr": "Les graphiques, les indicateurs et les tableaux de résultats s'ouvrent ici, tandis que l'explication reste dans la conversation.",
    },

    # ── The language switcher ────────────────────────────────────────────────
    "ui.lang.label": {"en": "Language", "fr": "Langue"},
    "ui.lang.en": {"en": "English", "fr": "Anglais"},
    "ui.lang.fr": {"en": "French", "fr": "Français"},
    "ui.lang.switch_to": {"en": "Switch to {language}", "fr": "Passer en {language}"},
    "ui.lang.current": {"en": "Current language: {language}", "fr": "Langue actuelle : {language}"},
    "ui.lang.failed": {
        "en": "The language could not be changed. Please try again.",
        "fr": "La langue n'a pas pu être changée. Veuillez réessayer.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Lookup
# ══════════════════════════════════════════════════════════════════════════════

_ACTIVE: ContextVar[str] = ContextVar("querybot_active_language", default=DEFAULT_LANGUAGE)


def normalise_language(value) -> str:
    """A supported language tag, or "en". Accepts "fr-FR" and "FR"."""
    tag = str(value or "").strip().lower().replace("_", "-").split("-")[0]
    return tag if tag in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def get_active_language() -> str:
    return _ACTIVE.get()


def activate_language(lang: str):
    """Set the language for this context; returns a token for deactivate.

    Does NOT survive ``loop.run_in_executor``. core/vocab_packs.py's docstring
    documents that, and core/query_pipeline.py records a real incident where a
    consumer read the default and degraded silently. Anything reached through an
    executor takes the language explicitly instead.
    """
    return _ACTIVE.set(normalise_language(lang))


def deactivate_language(token) -> None:
    try:
        _ACTIVE.reset(token)
    except Exception:
        pass


def lookup(msg_id: str, lang: str | None = None) -> str:
    """The raw template for an id, un-formatted.

    Falls back English, then to the id itself. Returning the id rather than ""
    is deliberate: a missing string shows up as ``ui.pin.title`` on the page,
    which is obvious in review, where an empty string looks like a layout bug.
    """
    entry = MESSAGES.get(msg_id)
    if not entry:
        return msg_id
    tag = normalise_language(lang if lang is not None else get_active_language())
    return entry.get(tag) or entry.get(DEFAULT_LANGUAGE) or msg_id


def t(msg_id: str, /, lang: str | None = None, **kw) -> str:
    """Translate and interpolate.

    ``lang`` is keyword-only in practice and normally omitted -- the ContextVar
    carries it. Pass it explicitly on any path that crosses an executor.

    A missing keyword leaves its placeholder visibly in the output rather than
    raising. This runs inside answer construction, and a KeyError there costs
    the user their whole answer to save a word.
    """
    template = lookup(msg_id, lang)
    if not kw:
        return template
    try:
        return template.format(**kw)
    except (KeyError, IndexError, ValueError):
        out = template
        for name, value in kw.items():
            out = out.replace("{" + name + "}", str(value))
        return out


def translator_for(lang: str):
    """A ``t`` bound to one language, for Jinja and for any executor-side caller.

    Templates render outside the pipeline's activation, so they must not depend
    on the ContextVar.
    """
    tag = normalise_language(lang)

    def _bound(msg_id: str, **kw) -> str:
        return t(msg_id, lang=tag, **kw)

    return _bound


def catalogue_for(lang: str, prefix: str = "") -> dict[str, str]:
    """A flat ``{msg_id: template}`` for one language, for JSON injection.

    The browser gets this as ``const I18N = {{ i18n_catalogue|tojson }}`` and
    does its own ``{name}`` interpolation, so the inline scripts and the markup
    read from one source instead of drifting -- which is what happened to
    ``_fmtNum``, and to the stage labels, already.
    """
    tag = normalise_language(lang)
    return {
        msg_id: (entry.get(tag) or entry.get(DEFAULT_LANGUAGE) or msg_id)
        for msg_id, entry in MESSAGES.items()
        if not prefix or msg_id.startswith(prefix)
    }


def plural(msg_id_stem: str, count, lang: str | None = None, **kw) -> str:
    """Pick the ``.one`` or ``.other`` form of a count message.

    English and French disagree about zero, and the page had the rule inline as
    ``{% if n != 1 %}s{% endif %}`` -- which is right for English and renders
    "0 visuels" in French, where zero takes the singular. So the rule lives
    here, once, and the two forms are separate catalogue entries rather than a
    stem with a bolted-on "s".

      en: 0 rows   1 row    2 rows
      fr: 0 ligne  1 ligne  2 lignes

    Deliberately only two forms. Neither language in this build needs more, and
    a general CLDR plural engine would be scaffolding for a case that does not
    exist yet -- add it with the language that needs it.
    """
    tag = normalise_language(lang if lang is not None else get_active_language())
    try:
        number = abs(float(count))
    except (TypeError, ValueError):
        number = 2.0
    if tag == "fr":
        one = number < 2          # French: 0 and 1 both take the singular
    else:
        one = number == 1
    return t(f"{msg_id_stem}.{'one' if one else 'other'}", lang=tag,
             count=count, **kw)


def enum_label(group: str, value, lang: str | None = None) -> str:
    """A translated label for a server enum -- status, visibility, cadence.

    The templates rendered these with ``|capitalize``, which cannot translate
    and also mangles whatever case the database happens to hold. An unknown
    value falls back to the prettified raw value rather than an id, because
    these come from the database and a new one must not render as
    "ui.enum.status.archived" on a customer's screen.
    """
    key = str(value or "").strip().lower().replace(" ", "_")
    if not key:
        return ""
    msg_id = f"ui.enum.{group}.{key}"
    if msg_id in MESSAGES:
        return t(msg_id, lang=lang)
    return key.replace("_", " ").capitalize()


def placeholders(msg_id: str) -> set[str]:
    """The named placeholders in an id's English template."""
    return set(_PLACEHOLDER_RE.findall(lookup(msg_id, DEFAULT_LANGUAGE)))
