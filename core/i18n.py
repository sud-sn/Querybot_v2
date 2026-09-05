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

    # ── The zero-latency analysis card ───────────────────────────────────────
    #
    # build_analysis_response, used when no model call is available. Its titles
    # live under analysis.title.* beside the model path's, so the two cards
    # cannot end up calling the same action two different things.
    #
    # The scope phrase is the reason this needed more than translation.
    # The card said `scope.get("badge").lower()` mid-sentence -- "This result
    # shows full distribution." French cannot make an inline noun phrase by
    # lowercasing a title-case badge: it needs the article, and the article
    # carries the gender. So each scope has an INLINE form of its own,
    # alongside the badge and the note it already had.
    "answer.scope.returned.inline": {
        "en": "the returned rows", "fr": "les lignes renvoyées",
    },
    "answer.scope.top_one.inline": {
        "en": "the top-ranked row only", "fr": "uniquement la ligne la mieux classée",
    },
    "answer.scope.top_n.inline": {
        "en": "the top {n} rows only", "fr": "uniquement les {n} premières lignes",
    },
    "answer.scope.full_distribution.inline": {
        "en": "the full returned distribution", "fr": "la distribution complète renvoyée",
    },
    "answer.scope.full_series.inline": {
        "en": "the full returned series", "fr": "la série complète renvoyée",
    },
    "answer.scope.preview.inline": {
        "en": "a capped preview of the rows", "fr": "un aperçu plafonné des lignes",
    },
    "answer.scope.filtered_subset.inline": {
        "en": "a filtered subset of the rows", "fr": "un sous-ensemble filtré des lignes",
    },

    "analysis.title.explain_result": {"en": "Result explanation", "fr": "Explication du résultat"},
    "analysis.title.detailed": {"en": "Detailed analysis", "fr": "Analyse détaillée"},
    "analysis.title.comparison": {"en": "Comparison view", "fr": "Vue comparative"},
    "analysis.title.framing": {"en": "Business framing", "fr": "Mise en perspective métier"},
    "analysis.title.forecast": {"en": "Forecast", "fr": "Prévision"},
    "analysis.title.next_step": {"en": "Recommended next step", "fr": "Prochaine étape recommandée"},
    "analysis.title.unavailable": {
        "en": "Not available for this workspace",
        "fr": "Indisponible pour cet espace de travail",
    },
    "analysis.regulated_body": {
        "en": "This workspace is configured for a regulated industry. To keep protected data from ever reaching the AI model, the assistant only writes SQL queries here — it doesn't generate follow-up analysis, explanations, or comparisons from results.",
        "fr": "Cet espace de travail est configuré pour un secteur réglementé. Pour qu'aucune donnée protégée n'atteigne le modèle, l'assistant se limite ici à écrire des requêtes SQL — il ne produit ni analyse complémentaire, ni explication, ni comparaison à partir des résultats.",
    },

    # Explain.
    "analysis.explain.series": {
        "en": "This result shows {scope}. The latest returned period is {period} at {value}.",
        "fr": "Ce résultat porte sur {scope}. La dernière période renvoyée est {period}, à {value}.",
    },
    "analysis.explain.direction_up": {
        "en": "Overall direction across the returned series: up ({pct})",
        "fr": "Orientation générale de la série renvoyée : à la hausse ({pct})",
    },
    "analysis.explain.direction_down": {
        "en": "Overall direction across the returned series: down ({pct})",
        "fr": "Orientation générale de la série renvoyée : à la baisse ({pct})",
    },
    "analysis.explain.direction_flat": {
        "en": "Overall direction across the returned series: flat ({pct})",
        "fr": "Orientation générale de la série renvoyée : stable ({pct})",
    },
    "analysis.explain.ranking": {
        "en": "This result shows {scope}. {leader} ranks first at {value}.",
        "fr": "Ce résultat porte sur {scope}. {leader} arrive en première position, à {value}.",
    },
    "analysis.explain.runner_up": {
        "en": "{sentence} The next highest returned result is {runner_up} at {value}.",
        "fr": "{sentence} Le résultat suivant est {runner_up}, à {value}.",
    },
    "analysis.explain.numeric.one": {
        "en": "The result contains {count} numeric row with values ranging from {low} to {high}.",
        "fr": "Le résultat contient {count} ligne numérique, avec des valeurs allant de {low} à {high}.",
    },
    "analysis.explain.numeric.other": {
        "en": "The result contains {count} numeric rows with values ranging from {low} to {high}.",
        "fr": "Le résultat contient {count} lignes numériques, avec des valeurs allant de {low} à {high}.",
    },
    "analysis.explain.concise": {
        "en": "This result is already concise and does not require deeper interpretation without an additional breakdown.",
        "fr": "Ce résultat est déjà concis et ne demande pas d'interprétation plus poussée sans une ventilation supplémentaire.",
    },
    "analysis.latest_period": {"en": "the latest period", "fr": "la dernière période"},
    "analysis.first_period": {"en": "the first period", "fr": "la première période"},

    # Analyze.
    "analysis.detail.series": {
        "en": "The returned time series varies between {low} and {high}, with an average of {average}.",
        "fr": "La série temporelle renvoyée varie entre {low} et {high}, pour une moyenne de {average}.",
    },
    "analysis.detail.avg_step": {"en": "Average step change: {value}", "fr": "Variation moyenne par pas : {value}"},
    "analysis.detail.volatility": {
        "en": "Observed volatility per step: {value}", "fr": "Volatilité observée par pas : {value}",
    },
    "analysis.detail.concentrated": {
        "en": "The ranking is concentrated: the top three returned categories account for {pct} of the total.",
        "fr": "Le classement est concentré : les trois premières catégories renvoyées représentent {pct} du total.",
    },
    "analysis.detail.distribution": {
        "en": "The ranking pattern should be read as a distribution, not just a winner.",
        "fr": "Ce classement se lit comme une distribution, pas seulement comme un vainqueur.",
    },
    "analysis.detail.category_count": {
        "en": "Category count in returned result: {count}",
        "fr": "Nombre de catégories dans le résultat renvoyé : {count}",
    },
    "analysis.detail.spread_range": {
        "en": "Spread from highest to lowest returned value: {value}",
        "fr": "Écart entre la valeur renvoyée la plus élevée et la plus faible : {value}",
    },
    "analysis.detail.std_dev": {
        "en": "Standard deviation across returned values: {value}",
        "fr": "Écart type des valeurs renvoyées : {value}",
    },
    "analysis.detail.numeric.one": {
        "en": "The numeric values average {average} across {count} row.",
        "fr": "Les valeurs numériques s'établissent en moyenne à {average} sur {count} ligne.",
    },
    "analysis.detail.numeric.other": {
        "en": "The numeric values average {average} across {count} rows.",
        "fr": "Les valeurs numériques s'établissent en moyenne à {average} sur {count} lignes.",
    },
    "analysis.detail.spread": {"en": "Spread: {value}", "fr": "Écart : {value}"},
    "analysis.detail.median": {"en": "Median: {value}", "fr": "Médiane : {value}"},
    "analysis.detail.not_enough": {
        "en": "There is not enough structure in this result for a richer analysis without a more specific breakdown.",
        "fr": "Ce résultat n'a pas assez de structure pour une analyse plus riche sans une ventilation plus précise.",
    },

    # Compare.
    "analysis.compare.series": {
        "en": "{last_period} is {last_value} versus {first_value} in {first_period}.",
        "fr": "{last_period} s'établit à {last_value}, contre {first_value} en {first_period}.",
    },
    "analysis.compare.pct_change": {
        "en": "Percent change across returned periods: {pct}",
        "fr": "Variation en pourcentage entre les périodes renvoyées : {pct}",
    },
    "analysis.compare.leader": {
        "en": "{leader} is ahead of {runner_up} by {gap}.",
        "fr": "{leader} devance {runner_up} de {gap}.",
    },
    "analysis.compare.leader_share": {
        "en": "Leader share of returned total: {pct}",
        "fr": "Part du premier dans le total renvoyé : {pct}",
    },
    "analysis.compare.only_one": {
        "en": "{leader} is the only comparable returned category, so there is no runner-up to compare.",
        "fr": "{leader} est la seule catégorie comparable renvoyée : il n'y a pas de second à comparer.",
    },
    "analysis.compare.not_comparable": {
        "en": "There is not enough comparable structure in this result for a comparison.",
        "fr": "Ce résultat n'a pas assez de structure comparable pour établir une comparaison.",
    },
    "analysis.compare.not_yet": {
        "en": "This result does not yet have enough comparable structure for a useful comparison card.",
        "fr": "Ce résultat n'a pas encore assez de structure comparable pour une carte de comparaison utile.",
    },

    # Why.
    "analysis.why.caveat": {
        "en": "This framing is based on the returned result shape, not on inferred root causes.",
        "fr": "Cette mise en perspective repose sur la forme du résultat renvoyé, non sur des causes déduites.",
    },
    "analysis.why.unstable_base": {
        "en": "The direction is visible, but the starting point is too close to zero for a stable percentage comparison.",
        "fr": "L'orientation est visible, mais le point de départ est trop proche de zéro pour une comparaison en pourcentage fiable.",
    },
    "analysis.why.higher": {
        "en": "This leaves the latest period {pct} higher than the starting period, which is useful for judging whether performance is improving or deteriorating over time.",
        "fr": "La dernière période se situe ainsi {pct} au-dessus de la période de départ, ce qui aide à juger si la performance s'améliore ou se dégrade dans le temps.",
    },
    "analysis.why.lower": {
        "en": "This leaves the latest period {pct} lower than the starting period, which is useful for judging whether performance is improving or deteriorating over time.",
        "fr": "La dernière période se situe ainsi {pct} en dessous de la période de départ, ce qui aide à juger si la performance s'améliore ou se dégrade dans le temps.",
    },
    "analysis.why.flat": {
        "en": "This leaves the latest period level with the starting period, which is useful for judging whether performance is improving or deteriorating over time.",
        "fr": "La dernière période se situe ainsi au même niveau que la période de départ, ce qui aide à juger si la performance s'améliore ou se dégrade dans le temps.",
    },
    "analysis.why.gap": {
        "en": "The leading category is ahead by {gap}, so performance is concentrated rather than evenly distributed across categories.",
        "fr": "La première catégorie devance les autres de {gap} : la performance est concentrée plutôt que répartie uniformément.",
    },
    "analysis.why.leader_only": {
        "en": "This identifies the leading category directly, which helps focus follow-up analysis on where performance is strongest or weakest.",
        "fr": "Cela identifie directement la catégorie de tête, ce qui aide à cibler l'analyse suivante là où la performance est la plus forte ou la plus faible.",
    },
    "analysis.why.spread": {
        "en": "The spread between the minimum and maximum values shows whether the result is tightly grouped or highly variable.",
        "fr": "L'écart entre la valeur minimale et la valeur maximale indique si le résultat est resserré ou très variable.",
    },
    "analysis.why.empty": {
        "en": "No impact can be inferred because the result set is empty under the current filters.",
        "fr": "Aucun impact ne peut être déduit : le résultat est vide avec les filtres actuels.",
    },
    "analysis.why.starting_point": {
        "en": "This result is best used as a starting point for a more targeted follow-up question.",
        "fr": "Ce résultat sert surtout de point de départ à une question de suivi plus ciblée.",
    },

    # Predict.
    "analysis.predict.projection": {
        "en": "A simple trend projection puts the next period near {value}.",
        "fr": "Une simple projection de tendance situe la période suivante autour de {value}.",
    },
    "analysis.predict.confidence_low": {
        "en": "This is a low-confidence directional estimate based only on the returned series, not a full forecasting model.",
        "fr": "Il s'agit d'une estimation directionnelle peu fiable, fondée uniquement sur la série renvoyée et non sur un vrai modèle de prévision.",
    },
    "analysis.predict.confidence_medium": {
        "en": "This is a medium-confidence directional estimate based only on the returned series, not a full forecasting model.",
        "fr": "Il s'agit d'une estimation directionnelle moyennement fiable, fondée uniquement sur la série renvoyée et non sur un vrai modèle de prévision.",
    },
    "analysis.predict.confidence_moderate": {
        "en": "This is a moderate-confidence directional estimate based only on the returned series, not a full forecasting model.",
        "fr": "Il s'agit d'une estimation directionnelle assez fiable, fondée uniquement sur la série renvoyée et non sur un vrai modèle de prévision.",
    },
    "analysis.predict.last_observed": {
        "en": "Last observed period: {period} at {value}",
        "fr": "Dernière période observée : {period}, à {value}",
    },
    "analysis.predict.step_used": {
        "en": "Average step change used in projection: {value}",
        "fr": "Variation moyenne par pas utilisée dans la projection : {value}",
    },
    "analysis.predict.needs_series": {
        "en": "Prediction is only available when the result contains a clear time series with at least three periods.",
        "fr": "La prévision n'est disponible que si le résultat contient une série temporelle claire d'au moins trois périodes.",
    },

    # Decide.
    "analysis.decide.starting_point": {
        "en": "This result is a useful starting point. Before acting, confirm the figures against a second cut of the data.",
        "fr": "Ce résultat est un bon point de départ. Avant d'agir, confirmez les chiffres avec une seconde lecture des données.",
    },
    "analysis.decide.finding": {
        "en": "Finding: based only on the returned result, not external context.",
        "fr": "Constat : fondé uniquement sur le résultat renvoyé, sans contexte extérieur.",
    },
    "analysis.decide.caveat": {
        "en": "Caveat: this is an advisory observation, not a directive.",
        "fr": "Réserve : il s'agit d'une observation indicative, non d'une directive.",
    },
    "analysis.decide.based_on": {"en": "Based on the returned rows.", "fr": "D'après les lignes renvoyées."},
    "analysis.decide.next_step": {
        "en": "Re-run with a narrower filter or a second time window to verify before acting.",
        "fr": "Relancez avec un filtre plus étroit ou une seconde fenêtre temporelle pour vérifier avant d'agir.",
    },
    "analysis.unsupported": {
        "en": "This follow-up action is not supported for the current result.",
        "fr": "Cette action de suivi n'est pas prise en charge pour ce résultat.",
    },

    # ── Coverage caveats ─────────────────────────────────────────────────────
    #
    # The "silent gap" notes: the answer looks complete and something was
    # quietly left out of it. They are assembled in core/result_renderer.py
    # from four modules, and they are the sentences a reader most needs to be
    # able to READ -- an English caveat under a French answer is a warning the
    # person it is for cannot act on.
    #
    # Whole sentences throughout. Each refusal names its own reason, so there
    # is no stem-plus-clause to share; and where a count decides the wording,
    # the two forms are separate entries because French takes the singular at
    # zero as well as one.

    # The truncated result.
    "caveat.truncated": {
        "en": "Showing the first {count} rows — the full result is larger. Distribution statistics (median, quartiles, histogram bins, correlation) are not shown, because computing them over a partial result would give a misleading answer. Narrow the question with a filter or a shorter date range to see them.",
        "fr": "Affichage des {count} premières lignes — le résultat complet est plus grand. Les statistiques de distribution (médiane, quartiles, classes d'histogramme, corrélation) ne sont pas affichées : les calculer sur un résultat partiel donnerait une réponse trompeuse. Restreignez la question par un filtre ou une période plus courte pour les obtenir.",
    },

    # ── Time grains, named ───────────────────────────────────────────────────
    #
    # Four call sites built these as f"{grain}s" -- English pluralisation
    # bolted onto a wire token. "mois" has no plural s, and French takes the
    # singular at zero as well as one, so the rule has to be the catalogue's.
    # Read through i18n.grain_label(); an unrecognised grain (a tenant's own
    # temporal_grain) passes through untranslated, because it is data.
    "caveat.grain.day.one": {"en": "day", "fr": "jour"},
    "caveat.grain.day.other": {"en": "days", "fr": "jours"},
    "caveat.grain.week.one": {"en": "week", "fr": "semaine"},
    "caveat.grain.week.other": {"en": "weeks", "fr": "semaines"},
    "caveat.grain.month.one": {"en": "month", "fr": "mois"},
    "caveat.grain.month.other": {"en": "months", "fr": "mois"},
    "caveat.grain.quarter.one": {"en": "quarter", "fr": "trimestre"},
    "caveat.grain.quarter.other": {"en": "quarters", "fr": "trimestres"},
    "caveat.grain.year.one": {"en": "year", "fr": "année"},
    "caveat.grain.year.other": {"en": "years", "fr": "années"},
    "caveat.grain.period.one": {"en": "period", "fr": "période"},
    "caveat.grain.period.other": {"en": "periods", "fr": "périodes"},

    # ── Why a forecast was refused, or what was clamped about one that ran ───
    "caveat.forecast.policy_blocked": {
        "en": "I did not project future periods: this result's policy does not allow a derived visual of these values.",
        "fr": "Je n'ai pas projeté de périodes futures : la politique appliquée à ce résultat n'autorise pas de visuel dérivé de ces valeurs.",
    },
    "caveat.forecast.no_temporal_axis": {
        "en": "I did not project future periods: this result has no column I can read as a time axis.",
        "fr": "Je n'ai pas projeté de périodes futures : ce résultat n'a aucune colonne lisible comme axe temporel.",
    },
    "caveat.forecast.no_measure": {
        "en": "I did not project future periods: this result has no numeric column to project.",
        "fr": "Je n'ai pas projeté de périodes futures : ce résultat n'a aucune colonne numérique à projeter.",
    },
    "caveat.forecast.masked_series": {
        "en": "I did not project future periods: some values in this result are masked, so a projection would be fitted to an incomplete series.",
        "fr": "Je n'ai pas projeté de périodes futures : certaines valeurs de ce résultat sont masquées, une projection serait donc ajustée sur une série incomplète.",
    },
    "caveat.forecast.truncated_result": {
        "en": "I did not project future periods: this result was truncated, so the series is only the first part of the data.",
        "fr": "Je n'ai pas projeté de périodes futures : ce résultat a été tronqué, la série ne couvre donc que le début des données.",
    },
    "caveat.forecast.multi_series": {
        "en": "I did not project future periods: this result is broken down by {column}, so it holds several series rather than one.",
        "fr": "Je n'ai pas projeté de périodes futures : ce résultat est ventilé par {column}, il contient donc plusieurs séries et non une seule.",
    },
    "caveat.forecast.too_short": {
        "en": "I did not project future periods: a reliable projection needs at least {minimum} periods and this result has {count}.",
        "fr": "Je n'ai pas projeté de périodes futures : une projection fiable demande au moins {minimum} périodes, et ce résultat en compte {count}.",
    },
    "caveat.forecast.too_short_grain": {
        "en": "I did not project future periods: this series has {count} {periods} and a reliable projection needs at least {minimum}.",
        "fr": "Je n'ai pas projeté de périodes futures : cette série compte {count} {periods} et une projection fiable en demande au moins {minimum}.",
    },
    "caveat.forecast.unordered_series": {
        "en": "I did not project future periods: the periods are not in chronological order, so the trend cannot be read from them.",
        "fr": "Je n'ai pas projeté de périodes futures : les périodes ne sont pas dans l'ordre chronologique, la tendance ne peut donc pas en être lue.",
    },
    "caveat.forecast.irregular_cadence": {
        "en": "I did not project future periods: the periods are not evenly spaced, so there is no consistent step to project forward.",
        "fr": "Je n'ai pas projeté de périodes futures : les périodes ne sont pas régulièrement espacées, il n'y a donc pas de pas constant à prolonger.",
    },
    # "{missing} of {expected} months are missing" makes the verb agree with
    # the wrong number -- "1 of 24 days are missing" -- and French additionally
    # has to agree the noun. Both languages say it impersonally instead.
    "caveat.forecast.gaps_in_series": {
        "en": "I did not project future periods: this series is missing {missing} of the {expected} {periods} it should cover.",
        "fr": "Je n'ai pas projeté de périodes futures : il manque {missing} des {expected} {periods} que cette série devrait couvrir.",
    },
    "caveat.forecast.constant_series": {
        "en": "I did not project future periods: this series does not vary, so a projection would just repeat the same number.",
        "fr": "Je n'ai pas projeté de périodes futures : cette série ne varie pas, une projection ne ferait que répéter le même nombre.",
    },
    "caveat.forecast.poor_fit": {
        "en": "I did not project future periods: the trend line explains only {r2}% of the movement in this series and back-testing it against the periods I already have was {mape}% out, so a projection would not mean much.",
        "fr": "Je n'ai pas projeté de périodes futures : la droite de tendance n'explique que {r2} % des variations de cette série, et son test rétrospectif sur les périodes déjà connues s'écarte de {mape} % — une projection n'aurait guère de sens.",
    },
    "caveat.forecast.capped_horizon": {
        "en": "I projected {capped} {periods} rather than {asked}: beyond about half the length of the history a projection is guesswork.",
        "fr": "J'ai projeté {capped} {periods} au lieu de {asked} : au-delà de la moitié environ de l'historique, une projection relève de la conjecture.",
    },
    # "this data is {grain}ly" builds an English adverb out of a wire token,
    # and the French adjective would have to agree with two different nouns in
    # the same sentence ("ces données", "la projection"). Both languages name
    # the grain instead of deriving a word from it.
    "caveat.forecast.grain_mismatch": {
        "en": "You asked by {asked_grain}, but this data is recorded by {grain}, so the projection is by {grain}.",
        "fr": "Vous avez demandé par {asked_grain}, mais ces données sont enregistrées par {grain} ; la projection est donc par {grain}.",
    },
    "caveat.forecast.model_fallback": {
        "en": "I projected these periods with a {model} model; the {preferred} model was not available here.",
        "fr": "J'ai projeté ces périodes avec un modèle {model} ; le modèle {preferred} n'était pas disponible ici.",
    },

    # ── The date-range coverage gap ──────────────────────────────────────────
    #
    # The English sentence puts the subject first ("Revenue records were
    # available on ..."), which French cannot do without an article, so the
    # subject has a sentence-initial form of its own. English also built the
    # window as the compound "6-month period"; both languages now count the
    # grain instead ("period of 6 months"), because a placeholder that exists
    # in one language and not the other is the one shape a translation cannot
    # be checked for.
    "caveat.dates.grain_recorded": {
        "en": "This source is recorded by {grain}. The most recent data it holds is dated {through}, and the result covers the requested period up to that point.",
        "fr": "Cette source est enregistrée par {grain}. La donnée la plus récente qu'elle contient est datée du {through}, et le résultat couvre la période demandée jusque-là.",
    },
    "caveat.dates.metric_sparse": {
        "en": "You asked for the last {requested} days. Records existed on {actual} {date_label}, but {metric} was nonzero on only {active} {active_label}, through {through}. The result reflects the available metric values.",
        "fr": "Vous avez demandé les {requested} derniers jours. Des enregistrements existaient sur {actual} {date_label}, mais {metric} n'affichait une valeur non nulle que sur {active} {active_label}, jusqu'au {through}. Le résultat reflète les valeurs disponibles de l'indicateur.",
    },
    "caveat.dates.days_sparse": {
        "en": "You asked for the last {requested} days, but {subject} were found on only {actual} {date_label} ({actual} {active_label} with data), through {through}. The result reflects the available data.",
        "fr": "Vous avez demandé les {requested} derniers jours, mais {subject} n'ont été trouvés que sur {actual} {date_label} ({actual} {active_label} avec des données), jusqu'au {through}. Le résultat reflète les données disponibles.",
    },
    "caveat.dates.period_sparse": {
        "en": "{subject_lead} were available on {actual} distinct {date_label} within the requested period of {requested} {units}, through {through}. The result reflects those available records.",
        "fr": "{subject_lead} étaient disponibles sur un total de {actual} {date_label} dans la période de {requested} {units} demandée, jusqu'au {through}. Le résultat reflète ces enregistrements disponibles.",
    },
    "caveat.dates.default_metric": {"en": "the selected metric", "fr": "l'indicateur sélectionné"},
    "caveat.dates.subject.named": {"en": "{metric} records", "fr": "des enregistrements de {metric}"},
    "caveat.dates.subject.named_lead": {"en": "{metric} records", "fr": "Les enregistrements de {metric}"},
    "caveat.dates.subject.generic": {"en": "records", "fr": "des enregistrements"},
    "caveat.dates.subject.generic_lead": {"en": "Records", "fr": "Les enregistrements"},
    "caveat.dates.business_date.one": {"en": "business date", "fr": "date métier"},
    "caveat.dates.business_date.other": {"en": "business dates", "fr": "dates métier"},
    "caveat.dates.role_date.one": {"en": "{role} date", "fr": "date « {role} »"},
    "caveat.dates.role_date.other": {"en": "{role} dates", "fr": "dates « {role} »"},

    # ── The lossy join ───────────────────────────────────────────────────────
    # The percentage came from a profiling run at some past admin click, and
    # the age of that measurement is part of the claim -- so it is part of the
    # sentence, not a suffix bolted onto one.
    "caveat.join.undated": {
        "en": "The join from {source} to {target} was measured as excluding about {rate}% of rows with no match. That measurement is undated, so it may not reflect the current data — some rows may not be counted.",
        "fr": "La jointure de {source} vers {target} a été mesurée comme excluant environ {rate} % des lignes sans correspondance. Cette mesure n'est pas datée : elle peut ne plus refléter les données actuelles — certaines lignes peuvent ne pas être comptées.",
    },
    "caveat.join.stale": {
        "en": "The join from {source} to {target} excluded about {rate}% of rows with no match when it was last profiled, {when}. Re-profile the relationship to confirm the current figure — some rows may not be counted.",
        "fr": "La jointure de {source} vers {target} excluait environ {rate} % des lignes sans correspondance lors de son dernier profilage, {when}. Reprofilez la relation pour confirmer le chiffre actuel — certaines lignes peuvent ne pas être comptées.",
    },
    "caveat.join.measured": {
        "en": "The join from {source} to {target} excludes about {rate}% of rows with no match (measured {when}) — some data may not be counted.",
        "fr": "La jointure de {source} vers {target} exclut environ {rate} % des lignes sans correspondance (mesuré {when}) — certaines données peuvent ne pas être comptées.",
    },
    "caveat.join.default_source": {"en": "the source table", "fr": "la table source"},
    "caveat.join.default_target": {"en": "the joined table", "fr": "la table jointe"},
    "caveat.join.today": {"en": "today", "fr": "aujourd'hui"},
    "caveat.join.days_ago.one": {"en": "{count} day ago", "fr": "il y a {count} jour"},
    "caveat.join.days_ago.other": {"en": "{count} days ago", "fr": "il y a {count} jours"},

    # ── The named-period comparison that came back incomplete ────────────────
    "caveat.period.missing_columns": {
        "en": "The result did not come back with a column for {missing}, so the change shown below is between {oldest} and {newest} only.",
        "fr": "Le résultat n'est pas revenu avec une colonne pour {missing} ; la variation ci-dessous ne porte donc qu'entre {oldest} et {newest}.",
    },
    "caveat.period.truncated": {
        "en": "The result stopped at its row cap, so I did not compute the change between periods or each category's share of it -- both would be statistics over the first rows only, not the whole result.",
        "fr": "Le résultat s'est arrêté à son plafond de lignes ; je n'ai donc calculé ni la variation entre périodes ni la part de chaque catégorie — les deux ne porteraient que sur les premières lignes, pas sur l'ensemble du résultat.",
    },
    "caveat.period.cancelling": {
        "en": "Gains and losses across these categories almost cancel out, so each category's share of the net change would be misleading. I left that column empty and kept the per-category changes themselves.",
        "fr": "Les hausses et les baisses de ces catégories se compensent presque ; la part de chaque catégorie dans la variation nette serait donc trompeuse. J'ai laissé cette colonne vide et conservé les variations par catégorie.",
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

    # ── The diagnostic card ──────────────────────────────────────────────────
    #
    # These are DISPLAY labels only. The markers the page scans for in the
    # server's message -- "Most likely reason:", "SQL tried:" and the rest --
    # stay English and are NOT in this catalogue: core/answer_formatter.py
    # writes them, the same text degrades to plain Teams and Zoom messages, and
    # a card that silently stops parsing renders as raw text with no signal
    # that anything went wrong. Wire format on one side, copy on the other.
    "ui.chat.diag.query_failed": {"en": "Query failed", "fr": "Échec de la requête"},
    "ui.chat.diag.validation": {"en": "Validation issue", "fr": "Problème de validation"},
    "ui.chat.diag.no_rows": {"en": "No rows returned", "fr": "Aucune ligne renvoyée"},
    "ui.chat.diag.no_match": {"en": "No matching data was found.", "fr": "Aucune donnée correspondante n'a été trouvée."},
    "ui.chat.diag.confidence": {"en": "Confidence", "fr": "Confiance"},
    "ui.chat.diag.reason": {"en": "Most likely reason", "fr": "Raison la plus probable"},
    "ui.chat.diag.next_step": {"en": "Suggested next step", "fr": "Prochaine étape suggérée"},
    "ui.chat.diag.why": {"en": "Why", "fr": "Pourquoi"},
    "ui.chat.diag.technical": {"en": "Technical details", "fr": "Détails techniques"},
    "ui.chat.diag.sql_tried": {"en": "SQL tried", "fr": "SQL tenté"},

    # ── The clarification and metric-draft cards ─────────────────────────────
    "ui.chat.clar.composed": {"en": "I composed a calculation for this.", "fr": "J'ai composé un calcul pour cela."},
    "ui.chat.clar.untitled": {"en": "Untitled", "fr": "Sans titre"},
    "ui.chat.clar.uses": {"en": "Uses {tables}", "fr": "Utilise {tables}"},
    "ui.chat.clar.promote": {"en": "Ask an admin to save this", "fr": "Demander à un administrateur de l'enregistrer"},
    "ui.chat.clar.discard": {"en": "Just for now", "fr": "Seulement pour cette fois"},
    "ui.chat.clar.sent_for_review": {"en": "Sent for review", "fr": "Envoyé pour validation"},
    "ui.chat.clar.kept_local": {"en": "Kept to this chat", "fr": "Conservé dans cette conversation"},
    "ui.chat.clar.choose_one": {
        "en": "Please choose one option so I can continue.",
        "fr": "Veuillez choisir une option pour que je puisse continuer.",
    },
    "ui.chat.clar.add_detail": {
        "en": "Please add the detail I need so I can continue.",
        "fr": "Veuillez ajouter la précision dont j'ai besoin pour continuer.",
    },
    "ui.chat.clar.add_detail_label": {
        "en": "Add the detail QueryBot needs to continue",
        "fr": "Ajoutez la précision dont QueryBot a besoin pour continuer",
    },
    "ui.chat.clar.suggested_dates": {"en": "Suggested business dates", "fr": "Dates métier suggérées"},
    "ui.chat.clar.your_answer": {"en": "Your answer", "fr": "Votre réponse"},
    "ui.chat.clar.continue": {"en": "Continue", "fr": "Continuer"},
    "ui.chat.clar.search_dates": {"en": "Search by business date name", "fr": "Rechercher par nom de date métier"},
    "ui.chat.clar.date_example": {"en": "For example: invoice date", "fr": "Par exemple : date de facture"},
    "ui.chat.clar.needs_answer": {
        "en": "Add an answer so QueryBot can continue",
        "fr": "Ajoutez une réponse pour que QueryBot puisse continuer",
    },
    "ui.chat.clar.offline": {
        "en": "Connection lost. Your clarification is still here; send it after reconnection.",
        "fr": "Connexion perdue. Votre précision est conservée ; envoyez-la après la reconnexion.",
    },
    "ui.chat.clar.busy": {
        "en": "QueryBot is still finishing the current step; applying your clarification now.",
        "fr": "QueryBot termine l'étape en cours ; votre précision est appliquée maintenant.",
    },
    "ui.chat.clar.resolving": {"en": "Resolving clarification", "fr": "Prise en compte de la précision"},
    "ui.chat.clar.resolving_detail": {
        "en": "Applying your clarification and continuing the governed query.",
        "fr": "Application de votre précision et poursuite de la requête gouvernée.",
    },
    "ui.chat.clar.answer_when_connected": {"en": "Answer again when connected", "fr": "Répondez à nouveau une fois reconnecté"},
    "ui.chat.clar.requested": {"en": "Clarification requested", "fr": "Précision demandée"},

    # ── Sections and buttons on an answer ────────────────────────────────────
    "ui.chat.card.key_insights": {"en": "Key insights", "fr": "Points clés"},
    "ui.chat.card.follow_up": {"en": "Follow-up", "fr": "Question de suivi"},
    "ui.chat.card.helpful": {"en": "Was this helpful?", "fr": "Cette réponse vous a-t-elle été utile ?"},
    "ui.chat.card.copy_answer": {"en": "Copy answer", "fr": "Copier la réponse"},
    "ui.chat.card.copy_result": {"en": "Copy result", "fr": "Copier le résultat"},
    "ui.chat.card.copy_analysis": {"en": "Copy analysis", "fr": "Copier l'analyse"},
    "ui.chat.card.next_step_label": {"en": "Next step", "fr": "Étape suivante"},
    "ui.chat.card.rows_first": {"en": "Returned rows first", "fr": "D'abord les lignes renvoyées"},
    "ui.chat.card.db_fallback": {"en": "Database fallback if needed", "fr": "Retour à la base si nécessaire"},
    "ui.chat.card.duckdb": {
        "en": "Executed in the governed DuckDB session cache",
        "fr": "Exécuté dans le cache de session DuckDB gouverné",
    },

    # ── Message delivery states ──────────────────────────────────────────────
    "ui.chat.delivery.sending": {"en": "Sending", "fr": "Envoi"},
    "ui.chat.delivery.working": {"en": "Working", "fr": "En cours"},
    "ui.chat.delivery.complete": {"en": "Answered", "fr": "Répondu"},
    "ui.chat.delivery.interrupted": {"en": "Interrupted", "fr": "Interrompu"},
    "ui.chat.delivery.needs_attention": {"en": "Needs attention", "fr": "À vérifier"},
    "ui.chat.delivery.waiting": {"en": "Waiting for connection", "fr": "En attente de connexion"},
    "ui.chat.delivery.stopped": {"en": "Stopped", "fr": "Arrêté"},
    "ui.chat.delivery.retrying": {"en": "Retrying analysis", "fr": "Nouvelle analyse en cours"},
    "ui.chat.delivery.retrying_detail": {
        "en": "Resending your request through the governed query pipeline.",
        "fr": "Renvoi de votre demande dans le pipeline de requêtes gouverné.",
    },
    "ui.chat.delivery.starting": {"en": "Starting analysis", "fr": "Démarrage de l'analyse"},
    "ui.chat.delivery.starting_detail": {
        "en": "Sending your request to the backend.",
        "fr": "Envoi de votre demande au serveur.",
    },
    "ui.chat.delivery.understanding": {"en": "Understanding your question", "fr": "Compréhension de votre question"},
    "ui.chat.delivery.understanding_detail": {
        "en": "Checking access to your workspace and data.",
        "fr": "Vérification de l'accès à votre espace de travail et à vos données.",
    },
    "ui.chat.delivery.processing": {"en": "Processing your request…", "fr": "Traitement de votre demande…"},
    "ui.chat.delivery.composer_working": {
        "en": "Working on your answer…", "fr": "Élaboration de votre réponse…",
    },

    # ── Errors the browser raises on its own ─────────────────────────────────
    "ui.chat.err.chart_library": {
        "en": "Chart library failed to load. Refresh the page and try again.",
        "fr": "La bibliothèque de graphiques n'a pas pu être chargée. Actualisez la page et réessayez.",
    },
    "ui.chat.err.chart_render": {"en": "Unable to render chart.", "fr": "Impossible d'afficher le graphique."},
    "ui.chat.err.db_waking": {
        "en": "Database is waking up — please try again in a moment",
        "fr": "La base de données démarre — réessayez dans un instant",
    },
    "ui.chat.err.connection": {
        "en": "Connection issue — please try again",
        "fr": "Problème de connexion — veuillez réessayer",
    },
    "ui.chat.err.generic_rephrase": {
        "en": "Something went wrong — try rephrasing or retry",
        "fr": "Une erreur s'est produite — reformulez ou réessayez",
    },
    "ui.chat.err.generic_retry": {
        "en": "Something went wrong — please retry",
        "fr": "Une erreur s'est produite — veuillez réessayer",
    },
    "ui.chat.err.generic": {"en": "Something went wrong.", "fr": "Une erreur s'est produite."},
    "ui.chat.err.clarify_format": {"en": "Please clarify the format.", "fr": "Veuillez préciser le format."},
    "ui.chat.err.not_enabled": {
        "en": "Chat is not enabled for this workspace.",
        "fr": "Le chat n'est pas activé pour cet espace de travail.",
    },
    "ui.chat.err.session_invalid": {
        "en": "Your session is invalid or no longer authorized.",
        "fr": "Votre session est invalide ou n'est plus autorisée.",
    },
    "ui.chat.err.action_failed": {"en": "Action failed", "fr": "Action échouée"},
    "ui.chat.err.action_incomplete": {"en": "Action could not be completed.", "fr": "L'action n'a pas pu être menée à son terme."},
    "ui.chat.err.action_accepted": {
        "en": "The governed action was accepted and is being completed.",
        "fr": "L'action gouvernée a été acceptée et est en cours d'exécution.",
    },
    "ui.chat.err.no_rows_matched": {
        "en": "No rows matched that query in the current result.",
        "fr": "Aucune ligne ne correspond à cette question dans le résultat actuel.",
    },
    "ui.chat.err.connection_not_ready": {"en": "Connection not ready. Please wait.", "fr": "Connexion non prête. Veuillez patienter."},
    "ui.chat.err.returned_only": {
        "en": "Using the returned result only — no new query is being run.",
        "fr": "Utilisation du seul résultat renvoyé — aucune nouvelle requête n'est exécutée.",
    },
    "ui.chat.action.finishing": {"en": "Finishing...", "fr": "Finalisation..."},

    # ── The schema lock, mirrored from the markup by updateSchemaModeCopy ────
    "ui.chat.schema_locked_title": {"en": "{schema} schema locked.", "fr": "Schéma {schema} verrouillé."},
    "ui.chat.schema_locked_body": {
        "en": "QueryBot will retrieve context and generate SQL only from this selected schema.",
        "fr": "QueryBot n'utilisera le contexte et ne générera du SQL qu'à partir de ce schéma.",
    },
    "ui.chat.schema_locked_hint": {
        "en": "Using {schema} schema. Press Enter to send, Shift + Enter for a new line.",
        "fr": "Schéma {schema} actif. Entrée pour envoyer, Maj + Entrée pour aller à la ligne.",
    },
    "ui.chat.schema_locked_placeholder": {
        "en": "Ask anything about {schema} data…",
        "fr": "Posez n'importe quelle question sur les données {schema}…",
    },
    "ui.chat.schema_named": {"en": "{schema} schema", "fr": "schéma {schema}"},

    # ── The last of the answer card ──────────────────────────────────────────
    "ui.chat.card.select": {"en": "Select", "fr": "Choisir"},
    "ui.chat.card.visual": {"en": "Visual", "fr": "Visuel"},
    "ui.chat.card.analysis": {"en": "Analysis", "fr": "Analyse"},

    "ui.chat.clar.help": {
        "en": "Press Enter to continue · Shift + Enter for a new line",
        "fr": "Entrée pour continuer · Maj + Entrée pour aller à la ligne",
    },

    # ── The history panel ────────────────────────────────────────────────────
    "ui.chat.hist.refreshing": {"en": "Refreshing threads...", "fr": "Actualisation des conversations..."},
    "ui.chat.hist.none_yet": {"en": "No threads yet.", "fr": "Aucune conversation pour l'instant."},
    "ui.chat.hist.empty": {
        "en": "Your recent questions will appear here.",
        "fr": "Vos questions récentes apparaîtront ici.",
    },
    "ui.chat.hist.failed": {"en": "Could not load threads.", "fr": "Impossible de charger les conversations."},
    "ui.chat.hist.try_again": {"en": "Try again", "fr": "Réessayer"},
    "ui.chat.hist.no_match": {"en": "No queries matching \"{term}\"", "fr": "Aucune question ne correspond à « {term} »"},

    # The history list's date groups.
    "ui.chat.hist.today": {"en": "Today", "fr": "Aujourd'hui"},
    "ui.chat.hist.yesterday": {"en": "Yesterday", "fr": "Hier"},
    "ui.chat.hist.last_week": {"en": "Previous 7 days", "fr": "7 derniers jours"},
    "ui.chat.hist.older": {"en": "Older", "fr": "Plus ancien"},

    # The forecast chart's series. These are the legend AND the seriesName the
    # tooltip filters on, so both sides read the same constant.
    "ui.chat.chart.actual": {"en": "Actual", "fr": "Réel"},
    "ui.chat.chart.forecast": {"en": "Forecast", "fr": "Prévision"},
    "ui.chat.chart.interval": {"en": "95% interval", "fr": "Intervalle à 95 %"},

    # ── The chart palette picker ─────────────────────────────────────────────
    "ui.chat.chart.palette": {"en": "Palette", "fr": "Palette"},
    "ui.chat.chart.outlier": {"en": "Outlier", "fr": "Valeur aberrante"},

    # ── The sign-in page ─────────────────────────────────────────────────────
    # Pre-authentication, so the language comes from Accept-Language rather
    # than the reader's stored preference -- a French browser gets a French
    # login screen before there is any account to read a setting from.
    "ui.auth.tagline": {
        "en": "Secure data intelligence workspace",
        "fr": "Espace sécurisé d'intelligence des données",
    },
    "ui.auth.account_id": {"en": "Account ID", "fr": "Identifiant du compte"},
    "ui.auth.account_id_placeholder": {"en": "e.g. acme-corp", "fr": "ex. acme-corp"},
    "ui.auth.account_id_note": {
        "en": "Ask your administrator for your Account ID",
        "fr": "Demandez votre identifiant de compte à votre administrateur",
    },
    "ui.auth.email": {"en": "Email", "fr": "E-mail"},
    "ui.auth.password": {"en": "Password", "fr": "Mot de passe"},
    "ui.auth.email_placeholder": {"en": "you@company.com", "fr": "vous@entreprise.com"},
    "ui.auth.show_password": {"en": "Show password", "fr": "Afficher le mot de passe"},
    "ui.auth.hide_password": {"en": "Hide password", "fr": "Masquer le mot de passe"},
    "ui.auth.toggle_password": {"en": "Show / hide password", "fr": "Afficher / masquer le mot de passe"},
    "ui.auth.sign_in": {"en": "Sign in", "fr": "Se connecter"},
    "ui.auth.first_time": {
        "en": "First time? Message your organisation's QueryBot to get a registration link.",
        "fr": "Première visite ? Écrivez au QueryBot de votre organisation pour obtenir un lien d'inscription.",
    },

    # ── Registration ─────────────────────────────────────────────────────────
    "ui.auth.setting_up_for": {
        "en": "Setting up your account for {client}",
        "fr": "Configuration de votre compte pour {client}",
    },
    "ui.auth.link_expired_help": {
        "en": "Message your QueryBot in Zoom to receive a new registration link.",
        "fr": "Écrivez à votre QueryBot dans Zoom pour recevoir un nouveau lien d'inscription.",
    },
    "ui.auth.create_account_title": {"en": "Create your account", "fr": "Créez votre compte"},
    "ui.auth.one_time_link": {"en": "One-time registration link", "fr": "Lien d'inscription à usage unique"},
    "ui.auth.one_time_link_note": {
        "en": "This link expires in 48 hours and can only be used once.",
        "fr": "Ce lien expire dans 48 heures et ne peut être utilisé qu'une seule fois.",
    },
    "ui.auth.full_name": {"en": "Full name", "fr": "Nom complet"},
    "ui.auth.full_name_placeholder": {"en": "Jane Smith", "fr": "Marie Dupont"},
    "ui.auth.work_email": {"en": "Work email", "fr": "E-mail professionnel"},
    "ui.auth.work_email_placeholder": {"en": "jane@company.com", "fr": "marie@entreprise.com"},
    "ui.auth.min_characters": {"en": "Min 8 characters", "fr": "8 caractères minimum"},
    "ui.auth.confirm_password": {"en": "Confirm password", "fr": "Confirmez le mot de passe"},
    "ui.auth.repeat_password": {"en": "Repeat password", "fr": "Répétez le mot de passe"},
    "ui.auth.create_account": {"en": "Create account", "fr": "Créer le compte"},
    "ui.auth.after_registering": {
        "en": "After registering, your administrator will assign you to a group so you can start querying data.",
        "fr": "Après votre inscription, votre administrateur vous affectera à un groupe pour que vous puissiez interroger les données.",
    },

    # ── Changing a password ──────────────────────────────────────────────────
    "ui.auth.set_password_title": {"en": "Set your password", "fr": "Définissez votre mot de passe"},
    "ui.auth.set_password_body": {
        "en": "You must set a new password before continuing.",
        "fr": "Vous devez définir un nouveau mot de passe avant de continuer.",
    },
    "ui.auth.change_password_title": {"en": "Change password", "fr": "Changer le mot de passe"},
    "ui.auth.change_password_body": {
        "en": "Update your account password.",
        "fr": "Mettez à jour le mot de passe de votre compte.",
    },
    "ui.auth.temporary_password": {
        "en": "Your account was created with a temporary password. Please set a new one now.",
        "fr": "Votre compte a été créé avec un mot de passe temporaire. Veuillez en définir un nouveau maintenant.",
    },
    "ui.auth.current_password": {"en": "Current password", "fr": "Mot de passe actuel"},
    "ui.auth.current_password_placeholder": {"en": "Your current password", "fr": "Votre mot de passe actuel"},
    "ui.auth.new_password": {"en": "New password", "fr": "Nouveau mot de passe"},
    "ui.auth.confirm_new_password": {"en": "Confirm new password", "fr": "Confirmez le nouveau mot de passe"},
    "ui.auth.repeat_new_password": {"en": "Repeat new password", "fr": "Répétez le nouveau mot de passe"},
    "ui.auth.set_password": {"en": "Set password", "fr": "Définir le mot de passe"},

    # ── Pinning a chart from a link ──────────────────────────────────────────
    "ui.pinpage.title": {"en": "Add chart to a dashboard", "fr": "Ajouter le graphique à un tableau de bord"},
    "ui.pinpage.subtitle": {
        "en": "Choose a named dashboard or create one for this live, governed result.",
        "fr": "Choisissez un tableau de bord existant ou créez-en un pour ce résultat gouverné en direct.",
    },
    "ui.pinpage.chart_title": {"en": "Chart title", "fr": "Titre du graphique"},
    "ui.pinpage.chart_title_placeholder": {
        "en": "e.g. Total revenue this month", "fr": "ex. Chiffre d'affaires total ce mois-ci",
    },
    "ui.pinpage.existing": {"en": "Existing dashboard", "fr": "Tableau de bord existant"},
    "ui.pinpage.create_new": {"en": "Create a new dashboard", "fr": "Créer un tableau de bord"},
    "ui.pinpage.option.one": {"en": "{name} · {count} visual", "fr": "{name} · {count} visuel"},
    "ui.pinpage.option.other": {"en": "{name} · {count} visuals", "fr": "{name} · {count} visuels"},
    "ui.pinpage.new_name": {"en": "New dashboard name", "fr": "Nom du nouveau tableau de bord"},
    "ui.pinpage.new_name_only": {
        "en": "New dashboard name (only when creating new)",
        "fr": "Nom du nouveau tableau de bord (uniquement à la création)",
    },
    "ui.pinpage.new_name_placeholder": {"en": "e.g. Pharmacy performance", "fr": "ex. Performance pharmacie"},
    "ui.pinpage.default_name": {"en": "My Dashboard", "fr": "Mon tableau de bord"},
    "ui.pinpage.original_question": {"en": "Original question", "fr": "Question d'origine"},
    "ui.pinpage.sql_note": {
        "en": "SQL query (runs live on refresh)",
        "fr": "Requête SQL (exécutée en direct à l'actualisation)",
    },

    # ── The new-report form ──────────────────────────────────────────────────
    "ui.report.title": {"en": "New report", "fr": "Nouveau rapport"},
    # The braces around {name} are literal: the sentence shows the reader the
    # shape of the question, it is not interpolated. t() leaves an unsupplied
    # placeholder in place, so this is safe whether or not kwargs are ever
    # passed to this id.
    "ui.report.intro": {
        "en": "Group a few metrics you check often into a named report you can ask for later (\"what's my {name} report?\") or subscribe to for a scheduled digest. Each metric still respects your own table access.",
        "fr": "Regroupez les indicateurs que vous consultez souvent dans un rapport nommé que vous pourrez demander plus tard (« quel est mon rapport {name} ? ») ou recevoir sur abonnement. Chaque indicateur respecte toujours vos propres accès aux tables.",
    },
    "ui.report.name": {"en": "Report name", "fr": "Nom du rapport"},
    "ui.report.name_placeholder": {"en": "e.g. My Territory Snapshot", "fr": "ex. Synthèse de mon secteur"},
    "ui.report.description": {"en": "Description", "fr": "Description"},
    "ui.report.description_placeholder": {"en": "What this report covers", "fr": "Ce que couvre ce rapport"},
    "ui.report.metrics": {"en": "Metrics to include", "fr": "Indicateurs à inclure"},
    "ui.report.no_metrics": {
        "en": "No metrics are available to you yet — ask your admin to grant table access or add metrics to the registry.",
        "fr": "Aucun indicateur ne vous est encore accessible — demandez à votre administrateur d'accorder l'accès aux tables ou d'ajouter des indicateurs au registre.",
    },
    "ui.report.create": {"en": "Create report", "fr": "Créer le rapport"},

    # ── The notifications page ───────────────────────────────────────────────
    "ui.notif.title": {"en": "My Notifications", "fr": "Mes notifications"},
    "ui.notif.intro": {
        "en": "Alerts you've set up on results, and reports you can subscribe to for a scheduled digest.",
        "fr": "Les alertes que vous avez créées sur des résultats, et les rapports auxquels vous pouvez vous abonner pour un envoi programmé.",
    },
    "ui.notif.saved": {"en": "Saved.", "fr": "Enregistré."},
    "ui.notif.your_alerts": {"en": "Your alerts", "fr": "Vos alertes"},
    # No plural forms: "min" is invariant in both languages, so a .one/.other
    # split here would be two identical sentences pretending to be a rule.
    "ui.notif.checks_every": {
        "en": "checks every {count} min", "fr": "vérifie toutes les {count} min",
    },
    "ui.notif.last_checked": {"en": "last checked {when}", "fr": "dernière vérification {when}"},
    "ui.notif.last_sent": {"en": "last sent {when}", "fr": "dernier envoi {when}"},
    "ui.notif.delete": {"en": "Delete", "fr": "Supprimer"},
    "ui.notif.delete_alert_title": {"en": "Delete alert?", "fr": "Supprimer l'alerte ?"},
    "ui.notif.delete_undone": {"en": "This cannot be undone.", "fr": "Cette action est irréversible."},
    "ui.notif.delete_report_title": {"en": "Delete report?", "fr": "Supprimer le rapport ?"},
    "ui.notif.delete_report_body": {
        "en": "This cannot be undone. Subscriptions to it will stop.",
        "fr": "Cette action est irréversible. Les abonnements à ce rapport prendront fin.",
    },
    "ui.notif.no_alerts": {
        "en": "No alerts set up yet — ask a question in Chat, then use \"Alert me on changes\" on the result.",
        "fr": "Aucune alerte pour l'instant — posez une question dans le chat, puis utilisez « M'alerter en cas de changement » sur le résultat.",
    },
    "ui.notif.my_reports": {"en": "My reports", "fr": "Mes rapports"},
    "ui.notif.new_report": {"en": "New report", "fr": "Nouveau rapport"},
    "ui.notif.describe_in_chat": {
        "en": "Or describe it in chat — try \"build a report with net revenue and top customers, scheduled every Monday at 9am\".",
        "fr": "Ou décrivez-le dans le chat — essayez « crée un rapport avec le chiffre d'affaires net et les meilleurs clients, programmé chaque lundi à 9 h ».",
    },
    "ui.notif.no_my_reports": {
        "en": "You haven't created any reports yet — group a few metrics you check often into one you can ask for or subscribe to.",
        "fr": "Vous n'avez encore créé aucun rapport — regroupez les indicateurs que vous consultez souvent dans un rapport que vous pourrez demander ou recevoir sur abonnement.",
    },
    "ui.notif.subscriptions": {"en": "Report subscriptions", "fr": "Abonnements aux rapports"},
    "ui.notif.unsubscribe": {"en": "Unsubscribe", "fr": "Se désabonner"},
    "ui.notif.subscribe": {"en": "Subscribe", "fr": "S'abonner"},
    "ui.notif.subscribed": {"en": "Subscribed", "fr": "Abonné"},
    "ui.notif.on_day": {"en": "(day {day})", "fr": "(jour {day})"},
    "ui.notif.at_hour": {"en": "at {hour}:00", "fr": "à {hour} h 00"},
    "ui.notif.day_label": {"en": "Day (weekly only)", "fr": "Jour (hebdomadaire uniquement)"},
    "ui.notif.hour_label": {"en": "Hour (server local time)", "fr": "Heure (heure locale du serveur)"},
    "ui.notif.no_reports": {
        "en": "No reports have been set up for this account yet — ask your admin to create one.",
        "fr": "Aucun rapport n'a encore été créé pour ce compte — demandez à votre administrateur d'en créer un.",
    },

    # Alert and subscription states. The VALUE is the wire enum in each case.
    "ui.enum.alertstatus.active": {"en": "active", "fr": "active"},
    "ui.enum.alertstatus.paused": {"en": "paused", "fr": "en pause"},
    "ui.enum.condition.change_pct": {"en": "changes by", "fr": "varie de"},
    "ui.enum.condition.above": {"en": "goes above", "fr": "dépasse"},
    "ui.enum.condition.below": {"en": "falls below", "fr": "descend sous"},

    # Weekday abbreviations for the schedule picker. The VALUE is the index.
    "ui.notif.day.0": {"en": "Mon", "fr": "Lun"},
    "ui.notif.day.1": {"en": "Tue", "fr": "Mar"},
    "ui.notif.day.2": {"en": "Wed", "fr": "Mer"},
    "ui.notif.day.3": {"en": "Thu", "fr": "Jeu"},
    "ui.notif.day.4": {"en": "Fri", "fr": "Ven"},
    "ui.notif.day.5": {"en": "Sat", "fr": "Sam"},
    "ui.notif.day.6": {"en": "Sun", "fr": "Dim"},

    # ── The Semantic Layer page ──────────────────────────────────────────────
    "ui.kb.intro": {
        "en": "Review the field meanings QueryBot uses for your assigned tables. If something is wrong, submit a correction for admin approval instead of changing the live Semantic Layer directly.",
        "fr": "Vérifiez la signification des champs que QueryBot utilise pour les tables qui vous sont attribuées. Si quelque chose est incorrect, soumettez une correction à la validation de l'administrateur plutôt que de modifier directement la couche sémantique en production.",
    },
    "ui.kb.pending.one": {"en": "{count} pending review", "fr": "{count} correction en attente"},
    "ui.kb.pending.other": {"en": "{count} pending reviews", "fr": "{count} corrections en attente"},
    "ui.kb.back_to_chat": {"en": "Back to chat", "fr": "Retour au chat"},
    "ui.kb.saved": {
        "en": "Your correction was sent to the admin review queue. It will update the Semantic Layer only after approval.",
        "fr": "Votre correction a été envoyée à la file de validation de l'administrateur. Elle ne mettra à jour la couche sémantique qu'après approbation.",
    },
    "ui.kb.empty_submission": {
        "en": "Add a suggested meaning, suggested use, or comment before submitting.",
        "fr": "Ajoutez une signification, un usage ou un commentaire avant d'envoyer.",
    },
    "ui.kb.no_metadata": {
        "en": "No Semantic Layer metadata is available yet, or you have not been assigned to a table group.",
        "fr": "Aucune métadonnée de couche sémantique n'est encore disponible, ou vous n'avez pas été affecté à un groupe de tables.",
    },
    "ui.kb.schema_selector": {"en": "Schema selector", "fr": "Sélecteur de schéma"},
    "ui.kb.search": {"en": "Search tables and fields…", "fr": "Rechercher des tables et des champs…"},
    "ui.kb.no_match": {
        "en": "No tables or fields match {query}. Try a shorter term or clear the search.",
        "fr": "Aucune table ni aucun champ ne correspond à {query}. Essayez un terme plus court ou effacez la recherche.",
    },
    "ui.kb.matched.tables.one": {"en": "{count} table", "fr": "{count} table"},
    "ui.kb.matched.tables.other": {"en": "{count} tables", "fr": "{count} tables"},
    "ui.kb.matched.fields.one": {"en": "{count} field matched", "fr": "{count} champ trouvé"},
    "ui.kb.matched.fields.other": {"en": "{count} fields matched", "fr": "{count} champs trouvés"},
    "ui.kb.tables_in": {"en": "Tables in {schema}", "fr": "Tables dans {schema}"},
    "ui.kb.field_count.one": {"en": "{count} field", "fr": "{count} champ"},
    "ui.kb.field_count.other": {"en": "{count} fields", "fr": "{count} champs"},
    "ui.kb.default_overview": {
        "en": "Field-level business metadata extracted from the approved KB.",
        "fr": "Métadonnées métier au niveau des champs, extraites de la base de connaissances approuvée.",
    },
    "ui.kb.pass": {"en": "{score}% pass", "fr": "{score} % de réussite"},

    # The field table's headings.
    "ui.kb.col.field": {"en": "Field", "fr": "Champ"},
    "ui.kb.col.meaning": {"en": "What this field is", "fr": "Ce qu'est ce champ"},
    "ui.kb.col.use_case": {"en": "What it is used for", "fr": "À quoi il sert"},
    "ui.kb.col.terms": {"en": "Business terms", "fr": "Termes métier"},
    "ui.kb.col.confidence": {"en": "Confidence", "fr": "Confiance"},
    "ui.kb.col.feedback": {"en": "Feedback", "fr": "Retour"},

    "ui.kb.type_unknown": {"en": "type unknown", "fr": "type inconnu"},
    "ui.kb.nullable": {"en": "nullable {value}", "fr": "nullable {value}"},
    "ui.kb.values": {"en": "Values: {values}", "fr": "Valeurs : {values}"},
    "ui.kb.badge.pending": {"en": "pending review", "fr": "en attente de validation"},
    "ui.kb.badge.approved": {"en": "admin approved", "fr": "validé par l'administrateur"},
    "ui.kb.badge.needs_context": {"en": "needs context", "fr": "contexte manquant"},
    "ui.kb.no_terms": {"en": "No business terms yet", "fr": "Aucun terme métier pour l'instant"},

    # The correction form.
    "ui.kb.suggest_edit": {"en": "Suggest edit", "fr": "Proposer une correction"},
    "ui.kb.close_edit": {"en": "Close edit", "fr": "Fermer"},
    "ui.kb.suggested_meaning": {"en": "Suggested field meaning", "fr": "Signification proposée"},
    "ui.kb.suggested_meaning_placeholder": {
        "en": "What should this field mean?", "fr": "Que devrait signifier ce champ ?",
    },
    "ui.kb.suggested_use": {"en": "Suggested use case", "fr": "Usage proposé"},
    "ui.kb.suggested_use_placeholder": {
        "en": "How should QueryBot use this field?",
        "fr": "Comment QueryBot devrait-il utiliser ce champ ?",
    },
    "ui.kb.terms_placeholder": {
        "en": "Comma-separated, e.g. purchase quantity, number of items purchased",
        "fr": "Séparés par des virgules, ex. quantité achetée, nombre d'articles achetés",
    },
    "ui.kb.terms_hint": {
        "en": "Other words a question might use for this field — helps QueryBot match phrasing it wouldn't otherwise recognize.",
        "fr": "D'autres mots qu'une question pourrait employer pour ce champ — cela aide QueryBot à reconnaître des formulations qu'il ignorerait autrement.",
    },
    "ui.kb.comment": {"en": "Comment for admin", "fr": "Commentaire pour l'administrateur"},
    "ui.kb.comment_placeholder": {"en": "Why should this be changed?", "fr": "Pourquoi faut-il modifier cela ?"},
    "ui.kb.approval_note": {
        "en": "This creates an admin approval request. The live Semantic Layer is unchanged until approved.",
        "fr": "Cela crée une demande de validation. La couche sémantique en production reste inchangée jusqu'à l'approbation.",
    },
    "ui.kb.send_for_approval": {"en": "Send for approval", "fr": "Envoyer pour validation"},

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

    # ══════════════════════════════════════════════════════════════════════════
    # reply.*  --  what the chat socket SAYS
    # ══════════════════════════════════════════════════════════════════════════
    #
    # ui.* is the page's own markup; these are the assistant's own turns,
    # written by gateway/webhooks.py and core/drill_dimension.py and pushed
    # down the socket as `content`, `suggestion`, `title` and `body`. They are
    # the only part of the conversation the reader did not write themselves,
    # so an English one under a French answer reads as a system that half
    # understood them.
    #
    # The refusals matter most: "this is turned off by the data policy" is the
    # sentence that decides whether someone files a ticket or gives up.

    # ── Connecting ──────────────────────────────────────────────────────────
    "reply.session.connected": {
        "en": "Connected as {name}. Ask me anything about your data.",
        "fr": "Connecté en tant que {name}. Posez-moi toutes vos questions sur vos données.",
    },

    # ── When something went wrong ───────────────────────────────────────────
    "reply.error.answer_failed": {
        "en": "Something went wrong while preparing your answer — please try asking again.",
        "fr": "Une erreur s'est produite pendant la préparation de votre réponse — veuillez reposer la question.",
    },
    "reply.error.generic": {
        "en": "Something went wrong. Please try again.",
        "fr": "Une erreur s'est produite. Veuillez réessayer.",
    },
    "reply.error.connection": {
        "en": "Connection error. Please refresh and try again.",
        "fr": "Erreur de connexion. Actualisez la page et réessayez.",
    },
    "reply.error.unreadable_message": {
        "en": "I could not read that message. Please send the question again.",
        "fr": "Je n'ai pas pu lire ce message. Veuillez renvoyer la question.",
    },
    "reply.error.unreadable_question": {
        "en": "I could not read that question. Please type it again.",
        "fr": "Je n'ai pas pu lire cette question. Veuillez la saisir à nouveau.",
    },
    "reply.error.empty_question": {
        "en": "Please type a question.",
        "fr": "Veuillez saisir une question.",
    },
    "reply.query.stopped": {"en": "Query stopped.", "fr": "Requête arrêtée."},

    # ── Acting on the result already on screen ──────────────────────────────
    # These say what did NOT leave the workspace, so they are the sentences a
    # governance reviewer reads over someone's shoulder.
    "reply.result.no_llm_used": {
        "en": "No LLM or database query was used.",
        "fr": "Aucun appel à un LLM ni à la base de données n'a été effectué.",
    },
    "reply.result.no_values_sent": {
        "en": "No result values were sent to an LLM.",
        "fr": "Aucune valeur du résultat n'a été envoyée à un LLM.",
    },
    "reply.result.stopped_locally": {
        "en": "The request was stopped locally. No cached rows, sample values, or bound literals were sent to the LLM or source database.",
        "fr": "La demande a été arrêtée localement. Aucune ligne en cache, valeur d'exemple ou littéral lié n'a été envoyé au LLM ni à la base source.",
    },
    "reply.result.update_failed": {
        "en": "I could not update the cached result. Please run the business question again.",
        "fr": "Je n'ai pas pu mettre à jour le résultat en cache. Veuillez reposer la question métier.",
    },
    "reply.result.unsafe_operation": {
        "en": "I could not safely apply that operation to the cached result. Use an exact result column name or a row number.",
        "fr": "Je n'ai pas pu appliquer cette opération au résultat en cache en toute sécurité. Utilisez un nom de colonne exact du résultat ou un numéro de ligne.",
    },
    "reply.result.none_cached": {
        "en": "No cached result found. Please run a query first.",
        "fr": "Aucun résultat en cache. Veuillez d'abord exécuter une requête.",
    },
    "reply.result.followup_failed": {
        "en": "The cached result could not be updated.",
        "fr": "Le résultat en cache n'a pas pu être mis à jour.",
    },
    "reply.result.filter_failed": {
        "en": "The filtered view could not be created. Please retry.",
        "fr": "La vue filtrée n'a pas pu être créée. Veuillez réessayer.",
    },
    "reply.result.expired_analysis": {
        "en": "That result is no longer available for analysis. Run the question again to create a fresh governed result.",
        "fr": "Ce résultat n'est plus disponible pour analyse. Reposez la question pour créer un résultat gouverné à jour.",
    },
    "reply.result.expired_action": {
        "en": "That result is no longer available for this action. Run the question again to create a fresh governed result.",
        "fr": "Ce résultat n'est plus disponible pour cette action. Reposez la question pour créer un résultat gouverné à jour.",
    },
    "reply.result.querying_db": {
        "en": "Querying your database for a complete answer…",
        "fr": "Interrogation de votre base de données pour une réponse complète…",
    },
    "reply.result.which_value": {
        "en": "Which result value did you mean?",
        "fr": "De quelle valeur du résultat parliez-vous ?",
    },

    # ── A metric worked out on the fly ──────────────────────────────────────
    "reply.metric.empty_filter": {
        "en": "I can build that, but I had to guess which value of **{columns}** you mean and the one I tried matches no rows — so the number would have been calculated over nothing.\n\nTell me the value and I'll rebuild it, for example: \"{example} is Y\".",
        "fr": "Je peux le construire, mais j'ai dû deviner de quelle valeur de **{columns}** vous parliez, et celle que j'ai essayée ne correspond à aucune ligne — le chiffre aurait donc été calculé sur rien.\n\nIndiquez-moi la valeur et je le reconstruis, par exemple : « {example} est Y ».",
    },
    "reply.metric.session_only": {
        "en": "I worked out **{name}** and used it to answer you. It applies to this conversation only — ask to save it and an admin can make it available to everyone.",
        "fr": "J'ai établi **{name}** et l'ai utilisé pour vous répondre. Cela ne vaut que pour cette conversation — demandez à l'enregistrer et un administrateur pourra le rendre disponible pour tout le monde.",
    },
    "reply.draft.gone": {
        "en": "That draft is no longer available.",
        "fr": "Ce brouillon n'est plus disponible.",
    },
    # {status} is the stored status value, not copy: the same token the admin
    # queue filters on.
    "reply.draft.already": {
        "en": "That draft was already {status}.",
        "fr": "Ce brouillon était déjà « {status} ».",
    },
    "reply.draft.sent": {
        "en": "Sent. An admin will review '{name}' before it becomes available to everyone — you can keep using it here in the meantime.",
        "fr": "Envoyé. Un administrateur examinera « {name} » avant qu'il ne devienne disponible pour tout le monde — vous pouvez continuer à l'utiliser ici en attendant.",
    },
    "reply.draft.send_failed": {
        "en": "That request could not be sent.",
        "fr": "Cette demande n'a pas pu être envoyée.",
    },

    # ── Reports ─────────────────────────────────────────────────────────────
    "reply.report.no_metrics": {
        "en": "There are no metrics available to you yet -- ask your admin to add some to the metric registry first.",
        "fr": "Aucun indicateur ne vous est encore accessible — demandez à votre administrateur d'en ajouter au registre des indicateurs.",
    },
    "reply.report.plan_failed": {
        "en": "Could not build a report from that -- try naming the metrics explicitly.",
        "fr": "Impossible de construire un rapport à partir de cela — essayez de nommer les indicateurs explicitement.",
    },
    "reply.report.created_title": {
        "en": "Report \"{name}\" created",
        "fr": "Rapport « {name} » créé",
    },
    "reply.report.created_body.one": {
        "en": "I've created **{name}** with {count} metric.",
        "fr": "J'ai créé **{name}** avec {count} indicateur.",
    },
    "reply.report.created_body.other": {
        "en": "I've created **{name}** with {count} metrics.",
        "fr": "J'ai créé **{name}** avec {count} indicateurs.",
    },
    "reply.report.no_schedule": {
        "en": "No schedule was requested -- ask any time to add one.",
        "fr": "Aucune planification n'a été demandée — demandez-en une à tout moment.",
    },
    "reply.report.weekly_schedule": {
        "en": "Delivered every {day} at {hour}.",
        "fr": "Envoyé chaque {day} à {hour}.",
    },
    "reply.report.daily_schedule": {
        "en": "Delivered daily at {hour}.",
        "fr": "Envoyé chaque jour à {hour}.",
    },
    "reply.report.metrics_bullet": {
        "en": "Metrics: {names}",
        "fr": "Indicateurs : {names}",
    },
    "reply.report.build_failed": {
        "en": "Could not build that report right now.",
        "fr": "Ce rapport n'a pas pu être construit pour le moment.",
    },
    "reply.report.skipped": {
        "en": "No worries — skipping today's reports.",
        "fr": "Pas de souci — je passe les rapports du jour.",
    },
    "reply.weekday.0": {"en": "Monday", "fr": "lundi"},
    "reply.weekday.1": {"en": "Tuesday", "fr": "mardi"},
    "reply.weekday.2": {"en": "Wednesday", "fr": "mercredi"},
    "reply.weekday.3": {"en": "Thursday", "fr": "jeudi"},
    "reply.weekday.4": {"en": "Friday", "fr": "vendredi"},
    "reply.weekday.5": {"en": "Saturday", "fr": "samedi"},
    "reply.weekday.6": {"en": "Sunday", "fr": "dimanche"},

    # ── Dashboards ──────────────────────────────────────────────────────────
    "reply.dash.read_only": {
        "en": "This is a published team dashboard owned by another user, so it is read-only. You can ask questions about the data, but only the owner can change the artifact.",
        "fr": "Ce tableau de bord d'équipe publié appartient à un autre utilisateur : il est en lecture seule. Vous pouvez poser des questions sur les données, mais seul le propriétaire peut modifier l'artefact.",
    },
    "reply.dash.no_restore": {
        "en": "There is no dashboard here to restore yet.",
        "fr": "Il n'y a pas encore de tableau de bord à restaurer ici.",
    },
    "reply.dash.version_missing": {
        "en": "Version {version} is not available for this dashboard.",
        "fr": "La version {version} n'est pas disponible pour ce tableau de bord.",
    },
    "reply.dash.restored_title": {
        "en": "Restored \"{name}\" from version {version}",
        "fr": "« {name} » restauré depuis la version {version}",
    },
    "reply.dash.restored_body": {
        "en": "The restore created a new draft checkpoint, so the full history is still available.",
        "fr": "La restauration a créé un nouveau point de sauvegarde en brouillon : l'historique complet reste disponible.",
    },
    "reply.dash.need_dashboard_schedule": {
        "en": "Create a dashboard before setting its refresh schedule.",
        "fr": "Créez un tableau de bord avant de définir sa planification d'actualisation.",
    },
    "reply.dash.schedule_title": {
        "en": "{name} will refresh {schedule}",
        "fr": "{name} s'actualisera {schedule}",
    },
    "reply.dash.schedule_body": {
        "en": "Scheduled refreshes run as the dashboard owner through current ACL, semantic, validation, and compliance controls. Released rows are encrypted and expire at the policy cache TTL.",
        "fr": "Les actualisations planifiées s'exécutent au nom du propriétaire du tableau de bord, à travers les contrôles ACL, sémantiques, de validation et de conformité en vigueur. Les lignes diffusées sont chiffrées et expirent à la durée de vie du cache définie par la politique.",
    },
    "reply.dash.need_dashboard_filter": {
        "en": "Create a dashboard before adding filters.",
        "fr": "Créez un tableau de bord avant d'ajouter des filtres.",
    },
    "reply.dash.filter_title": {
        "en": "Added a {field} filter to \"{name}\"",
        "fr": "Filtre sur {field} ajouté à « {name} »",
    },
    "reply.dash.filter_body": {
        "en": "The control is applied only to dashboard sources that return a matching field.",
        "fr": "Le contrôle ne s'applique qu'aux sources du tableau de bord qui renvoient un champ correspondant.",
    },
    "reply.dash.need_dashboard_tab": {
        "en": "Create a dashboard before adding tabs.",
        "fr": "Créez un tableau de bord avant d'ajouter des onglets.",
    },
    "reply.dash.tab_title": {
        "en": "Added the {tab} tab to \"{name}\"",
        "fr": "Onglet {tab} ajouté à « {name} »",
    },
    "reply.dash.tab_body": {
        "en": "Ask me to add a new visual to this dashboard and name the tab to place it there.",
        "fr": "Demandez-moi d'ajouter un nouveau visuel à ce tableau de bord en nommant l'onglet pour l'y placer.",
    },
    "reply.dash.need_dashboard_share": {
        "en": "Create a dashboard before sharing it.",
        "fr": "Créez un tableau de bord avant de le partager.",
    },
    "reply.dash.published_title": {
        "en": "Published \"{name}\" to your workspace team",
        "fr": "« {name} » publié auprès de votre équipe",
    },
    "reply.dash.published_body": {
        "en": "Workspace users can view and filter it under their own current data access. Only the owner can edit or restore the artifact.",
        "fr": "Les utilisateurs de l'espace de travail peuvent le consulter et le filtrer selon leurs propres droits d'accès aux données. Seul le propriétaire peut modifier ou restaurer l'artefact.",
    },
    "reply.dash.no_rename": {
        "en": "There is no dashboard in this thread to rename yet.",
        "fr": "Il n'y a pas encore de tableau de bord à renommer dans ce fil.",
    },
    "reply.dash.renamed_title": {
        "en": "Dashboard renamed to \"{name}\"",
        "fr": "Tableau de bord renommé « {name} »",
    },
    "reply.dash.no_publish": {
        "en": "There is no dashboard in this thread to publish yet.",
        "fr": "Il n'y a pas encore de tableau de bord à publier dans ce fil.",
    },
    "reply.dash.publish_title": {
        "en": "Dashboard \"{name}\" published",
        "fr": "Tableau de bord « {name} » publié",
    },
    "reply.dash.no_update": {
        "en": "There is no dashboard in this thread to update yet.",
        "fr": "Il n'y a pas encore de tableau de bord à mettre à jour dans ce fil.",
    },
    "reply.dash.no_visual": {
        "en": "That dashboard does not have a visual to update yet.",
        "fr": "Ce tableau de bord n'a pas encore de visuel à mettre à jour.",
    },
    "reply.dash.visual_changed_title": {
        "en": "Changed the latest visual in \"{name}\" to {chart_type}",
        "fr": "Dernier visuel de « {name} » changé en {chart_type}",
    },
    "reply.dash.need_dashboard_visual": {
        "en": "Create a dashboard first, then I can add new governed visuals to it.",
        "fr": "Créez d'abord un tableau de bord, et je pourrai y ajouter de nouveaux visuels gouvernés.",
    },
    "reply.dash.rerun_for_chooser": {
        "en": "Run the result again so I can open the dashboard chooser for it.",
        "fr": "Réexécutez le résultat pour que je puisse ouvrir le sélecteur de tableau de bord.",
    },
    "reply.dash.default_visual_title": {
        "en": "Dashboard visual",
        "fr": "Visuel de tableau de bord",
    },
    "reply.dash.none_yet": {
        "en": "There is no dashboard in this thread yet. Say \"create a dashboard from this result\" first.",
        "fr": "Il n'y a pas encore de tableau de bord dans ce fil. Dites d'abord « crée un tableau de bord à partir de ce résultat ».",
    },
    "reply.dash.added_title": {
        "en": "Added this result to \"{name}\"",
        "fr": "Ce résultat a été ajouté à « {name} »",
    },
    "reply.dash.building_title": {
        "en": "Building \"{name}\"",
        "fr": "Construction de « {name} »",
    },
    "reply.dash.building_body.one": {
        "en": "I'll run {count} governed data task and assemble the successful result in the artifact pane.",
        "fr": "Je vais exécuter {count} tâche de données gouvernée et assembler le résultat obtenu dans le volet des artefacts.",
    },
    "reply.dash.building_body.other": {
        "en": "I'll run {count} governed data tasks and assemble the successful results in the artifact pane.",
        "fr": "Je vais exécuter {count} tâches de données gouvernées et assembler les résultats obtenus dans le volet des artefacts.",
    },
    "reply.dash.building_step": {
        "en": "Building visual {index} of {total}",
        "fr": "Construction du visuel {index} sur {total}",
    },
    "reply.dash.built_title": {
        "en": "Built {completed} of {total} visuals for \"{name}\"",
        "fr": "{completed} visuels sur {total} construits pour « {name} »",
    },
    "reply.dash.built_body": {
        "en": "Open the artifact to review the live charts, data sources, controls, and revision history.",
        "fr": "Ouvrez l'artefact pour examiner les graphiques en direct, les sources de données, les contrôles et l'historique des révisions.",
    },
    "reply.dash.what_to_track": {
        "en": "What should the dashboard track? For example, say \"create a dashboard showing monthly revenue by region\".",
        "fr": "Que doit suivre le tableau de bord ? Dites par exemple « crée un tableau de bord du chiffre d'affaires mensuel par région ».",
    },
    "reply.dash.created_title": {
        "en": "Dashboard \"{name}\" created",
        "fr": "Tableau de bord « {name} » créé",
    },
    "reply.dash.update_failed": {
        "en": "I could not update that dashboard right now.",
        "fr": "Je n'ai pas pu mettre à jour ce tableau de bord pour le moment.",
    },

    # ── Explaining what was run ─────────────────────────────────────────────
    "reply.explain.title": {
        "en": "Here's exactly what I ran",
        "fr": "Voici exactement ce que j'ai exécuté",
    },
    "reply.explain.body": {
        "en": "I can't see how your number was calculated, so I can't explain the gap directly -- but here's my exact definition. Try one of these to see if it closes the difference:",
        "fr": "Je ne peux pas voir comment votre chiffre a été calculé, je ne peux donc pas expliquer l'écart directement — mais voici ma définition exacte. Essayez l'une de ces pistes pour voir si elle comble la différence :",
    },

    # ── Governed Python analysis ────────────────────────────────────────────
    "reply.analysis.default_title": {
        "en": "Analysis work",
        "fr": "Travail d'analyse",
    },
    "reply.analysis.custom_python_title": {
        "en": "Custom Python analysis",
        "fr": "Analyse Python personnalisée",
    },
    "reply.analysis.need_result": {
        "en": "Run a data question first, then ask me to analyze that result.",
        "fr": "Posez d'abord une question sur les données, puis demandez-moi d'analyser ce résultat.",
    },
    "reply.analysis.python_disabled": {
        "en": "Governed Python analysis is disabled for this workspace. An administrator can enable it in Client settings → Agent Analysis.",
        "fr": "L'analyse Python gouvernée est désactivée pour cet espace de travail. Un administrateur peut l'activer dans Paramètres client → Analyse par agent.",
    },
    "reply.analysis.no_pasted_source": {
        "en": "This workspace allows governed Python plans but not pasted source. Ask for the calculation in plain English, or have an administrator enable user-submitted Python.",
        "fr": "Cet espace de travail autorise les plans Python gouvernés mais pas le code collé. Formulez le calcul en langage courant, ou demandez à un administrateur d'activer le Python soumis par les utilisateurs.",
    },
    "reply.analysis.need_precision": {
        "en": "I need a more precise calculation and output shape before I run Python.",
        "fr": "J'ai besoin d'un calcul et d'un format de sortie plus précis avant d'exécuter du Python.",
    },
    "reply.analysis.completed_title": {
        "en": "{title} completed",
        "fr": "{title} terminé",
    },
    "reply.analysis.completed_planner": {
        "en": "I analyzed only the governed rows already returned to this conversation. A metadata-only planner produced the validated calculation; zero result rows or sample values were sent to the model.",
        "fr": "Je n'ai analysé que les lignes gouvernées déjà renvoyées dans cette conversation. Un planificateur travaillant uniquement sur les métadonnées a produit le calcul validé ; aucune ligne de résultat ni valeur d'exemple n'a été envoyée au modèle.",
    },
    "reply.analysis.completed_local": {
        "en": "I analyzed only the governed rows already returned to this conversation. No database query or model call was made for these calculations.",
        "fr": "Je n'ai analysé que les lignes gouvernées déjà renvoyées dans cette conversation. Aucune requête à la base de données ni appel au modèle n'a été effectué pour ces calculs.",
    },
    "reply.analysis.child_tasks": {
        "en": "{completed} of {total} child tasks completed in isolated, time-bounded workers.",
        "fr": "{completed} sous-tâches sur {total} terminées dans des workers isolés et limités dans le temps.",
    },
    "reply.analysis.failed": {
        "en": "I could not complete the governed result analysis.",
        "fr": "Je n'ai pas pu mener à bien l'analyse gouvernée du résultat.",
    },

    # The isolated worker's own result card. `operation` is a wire token --
    # profile, outliers, correlation, trend, python -- so the row it counts is
    # named per token rather than title-cased out of it.
    "reply.analysis.done_headline": {
        "en": "{operation} analysis completed.",
        "fr": "Analyse — {operation} : terminée.",
    },
    # The five operations, named. `operation` itself is the wire token the
    # priority table and the subtask records are keyed on, so it is never
    # title-cased into a word -- the word is looked up.
    "reply.analysis.op.profile": {"en": "Profile", "fr": "Profil"},
    "reply.analysis.op.outliers": {"en": "Outliers", "fr": "Valeurs aberrantes"},
    "reply.analysis.op.correlation": {"en": "Correlation", "fr": "Corrélation"},
    "reply.analysis.op.trend": {"en": "Trend", "fr": "Tendance"},
    "reply.analysis.op.python": {"en": "Python", "fr": "Python"},
    "reply.analysis.op.default": {"en": "Analysis", "fr": "Analyse"},
    # The unnamed operation, whose headline would otherwise read "Analysis
    # analysis completed."
    "reply.analysis.done_headline_generic": {
        "en": "Analysis completed.",
        "fr": "Analyse terminée.",
    },
    "reply.analysis.chart_title": {
        "en": "{operation} analysis",
        "fr": "Analyse — {operation}",
    },
    "reply.analysis.row.profile.one": {"en": "profile row", "fr": "ligne de profil"},
    "reply.analysis.row.profile.other": {"en": "profile rows", "fr": "lignes de profil"},
    "reply.analysis.row.outliers.one": {"en": "potential outlier", "fr": "valeur aberrante potentielle"},
    "reply.analysis.row.outliers.other": {"en": "potential outliers", "fr": "valeurs aberrantes potentielles"},
    "reply.analysis.row.correlation.one": {"en": "correlation pair", "fr": "paire de corrélation"},
    "reply.analysis.row.correlation.other": {"en": "correlation pairs", "fr": "paires de corrélation"},
    "reply.analysis.row.trend.one": {"en": "trend row", "fr": "ligne de tendance"},
    "reply.analysis.row.trend.other": {"en": "trend rows", "fr": "lignes de tendance"},
    "reply.analysis.row.python.one": {"en": "derived row", "fr": "ligne dérivée"},
    "reply.analysis.row.python.other": {"en": "derived rows", "fr": "lignes dérivées"},
    "reply.analysis.row.default.one": {"en": "analysis row", "fr": "ligne d'analyse"},
    "reply.analysis.row.default.other": {"en": "analysis rows", "fr": "lignes d'analyse"},
    "reply.analysis.short_value": {"en": "{count} {label}", "fr": "{count} {label}"},
    "reply.analysis.based_on.one": {
        "en": "Based on {count} returned row",
        "fr": "D'après {count} ligne renvoyée",
    },
    "reply.analysis.based_on.other": {
        "en": "Based on {count} returned rows",
        "fr": "D'après {count} lignes renvoyées",
    },
    "reply.analysis.isolated_note": {
        "en": "Calculated in an isolated worker without a new database query.",
        "fr": "Calculé dans un worker isolé, sans nouvelle requête à la base de données.",
    },
    "reply.analysis.released_note.one": {
        "en": "Based on {count} released row.",
        "fr": "D'après {count} ligne diffusée.",
    },
    "reply.analysis.released_note.other": {
        "en": "Based on {count} released rows.",
        "fr": "D'après {count} lignes diffusées.",
    },
    "reply.analysis.partial_failure": {
        "en": "Could not complete: {operations}",
        "fr": "N'a pas pu être terminé : {operations}",
    },
    "reply.analysis.stage_label": {
        "en": "Analyzing the governed result",
        "fr": "Analyse du résultat gouverné",
    },
    "reply.analysis.stage_detail.one": {
        "en": "Running {count} bounded child task in isolated workers",
        "fr": "Exécution de {count} sous-tâche bornée dans des workers isolés",
    },
    "reply.analysis.stage_detail.other": {
        "en": "Running {count} bounded child tasks in isolated workers",
        "fr": "Exécution de {count} sous-tâches bornées dans des workers isolés",
    },
    "reply.analysis.stage_running": {
        "en": "Running isolated analysis tasks",
        "fr": "Exécution des tâches d'analyse isolées",
    },

    # ── The result-chat panel beside a result ───────────────────────────────
    "reply.result_chat.db_note": {
        "en": "Answer required a full database query.",
        "fr": "La réponse a nécessité une requête complète à la base de données.",
    },
    "reply.result_chat.local_note": {
        "en": "Computed locally from the cached result. No result values were sent to the model.",
        "fr": "Calculé localement à partir du résultat en cache. Aucune valeur du résultat n'a été envoyée au modèle.",
    },
    "reply.result_chat.blocked_detail": {
        "en": "The request was stopped locally. No cached rows, sample values, source SQL, or bound literals were sent to the model.",
        "fr": "La demande a été arrêtée localement. Aucune ligne en cache, valeur d'exemple, requête SQL source ou littéral lié n'a été envoyé au modèle.",
    },
    "reply.result_chat.retry_detail": {
        "en": "Run the business question again or use an exact result column.",
        "fr": "Reposez la question métier ou utilisez un nom de colonne exact du résultat.",
    },

    # ── The prior-period chip ───────────────────────────────────────────────
    "reply.prior.failed_headline": {
        "en": "Could not complete the prior period comparison.",
        "fr": "La comparaison avec la période précédente n'a pas pu être effectuée.",
    },

    # ── The dashboard work planner ──────────────────────────────────────────
    "reply.dash.need_detail": {
        "en": "I can build that dashboard, but I need the exact visuals, time range, and grouping you want before I run several queries.",
        "fr": "Je peux construire ce tableau de bord, mais j'ai besoin des visuels, de la période et du regroupement exacts que vous voulez avant de lancer plusieurs requêtes.",
    },

    # ── Clarifications ──────────────────────────────────────────────────────
    "reply.clarify.expired": {
        "en": "That clarification is no longer active. Please ask the question again.",
        "fr": "Cette demande de précision n'est plus active. Veuillez reposer la question.",
    },
    "reply.clarify.superseded": {
        "en": "That clarification belongs to an older step and is no longer active. Please answer the newest clarification card.",
        "fr": "Cette demande de précision porte sur une étape antérieure et n'est plus active. Veuillez répondre à la carte de précision la plus récente.",
    },
    "reply.clarify.restate": {
        "en": "Understood. Please restate the new business question, and I'll answer it from the governed data source.",
        "fr": "Entendu. Reformulez la nouvelle question métier et j'y répondrai à partir de la source de données gouvernée.",
    },
    "reply.clarify.display_failed": {
        "en": "I could not apply that display choice. Please try again.",
        "fr": "Je n'ai pas pu appliquer ce choix d'affichage. Veuillez réessayer.",
    },
    "reply.clarify.choose_option": {
        "en": "Please choose one of the available clarification options.",
        "fr": "Veuillez choisir l'une des options de précision proposées.",
    },
    "reply.clarify.type_answer": {
        "en": "Please type your clarification.",
        "fr": "Veuillez saisir votre précision.",
    },
    "reply.clarify.apply_failed": {
        "en": "I hit an error while applying that clarification. Please try again.",
        "fr": "J'ai rencontré une erreur en appliquant cette précision. Veuillez réessayer.",
    },
    "reply.clarify.choose_one": {
        "en": "Please choose one option.",
        "fr": "Veuillez choisir une option.",
    },
    "reply.clarify.choose_one_to_continue": {
        "en": "Please choose one option so I can continue.",
        "fr": "Veuillez choisir une option pour que je puisse continuer.",
    },
    "reply.clarify.ambiguous_business_date": {
        "en": "I couldn't match that business date unambiguously. Choose a suggested business date or type a more specific business name.",
        "fr": "Je n'ai pas pu identifier cette date métier sans ambiguïté. Choisissez une date métier proposée ou saisissez un nom métier plus précis.",
    },
    "reply.clarify.previous_result": {
        "en": "Are you referring to the previous result?",
        "fr": "Faites-vous référence au résultat précédent ?",
    },
    "reply.clarify.use_previous": {
        "en": "Yes — use the previous result",
        "fr": "Oui — utiliser le résultat précédent",
    },
    "reply.clarify.new_question": {
        "en": "No — this is a new question",
        "fr": "Non — c'est une nouvelle question",
    },

    # ── The chips under a result ────────────────────────────────────────────
    "reply.prior.title": {
        "en": "Prior period comparison",
        "fr": "Comparaison avec la période précédente",
    },
    "reply.prior.failed_body": {
        "en": "An unexpected error occurred while preparing the prior period. Try asking the comparison directly in your question.",
        "fr": "Une erreur inattendue s'est produite lors de la préparation de la période précédente. Essayez de demander la comparaison directement dans votre question.",
    },
    "reply.prior.next_step": {
        "en": "Ask: \"Show [metric] for [period A] vs [period B]\"",
        "fr": "Demandez : « Affiche [indicateur] pour [période A] par rapport à [période B] »",
    },
    "reply.contribution.failed": {
        "en": "Could not compute contribution share.",
        "fr": "La part de contribution n'a pas pu être calculée.",
    },
    "reply.contribution.share_failed": {
        "en": "Could not compute the % share breakdown.",
        "fr": "La ventilation en pourcentage n'a pas pu être calculée.",
    },
    "reply.outliers.none": {
        "en": "No outliers found in this result.",
        "fr": "Aucune valeur aberrante trouvée dans ce résultat.",
    },
    "reply.outliers.filter_failed": {
        "en": "Could not filter outliers from this result.",
        "fr": "Les valeurs aberrantes n'ont pas pu être filtrées de ce résultat.",
    },
    "reply.csv.blocked": {
        "en": "Export is blocked by the workspace data policy.",
        "fr": "L'export est bloqué par la politique de données de l'espace de travail.",
    },
    "reply.csv.failed": {
        "en": "Could not generate CSV from this result.",
        "fr": "Le CSV n'a pas pu être généré à partir de ce résultat.",
    },
    "reply.alert.blocked": {
        "en": "Alerts are blocked by the workspace data policy.",
        "fr": "Les alertes sont bloquées par la politique de données de l'espace de travail.",
    },
    "reply.alert.no_metric": {
        "en": "Could not identify a numeric metric to monitor. Ask for a specific KPI result first.",
        "fr": "Aucun indicateur numérique à surveiller n'a pu être identifié. Demandez d'abord un résultat de KPI précis.",
    },
    "reply.alert.created_title": {"en": "Alert created", "fr": "Alerte créée"},
    "reply.alert.created_body": {
        "en": "I'll monitor **{metric}** (baseline: {baseline}) and flag it when the value changes by more than {threshold}%.",
        "fr": "Je surveillerai **{metric}** (référence : {baseline}) et le signalerai lorsque la valeur variera de plus de {threshold} %.",
    },
    "reply.alert.created_secondary": {
        "en": "Alert ID: {id} — use this ID to check the current value against the baseline at any time.",
        "fr": "Identifiant d'alerte : {id} — utilisez-le pour comparer la valeur actuelle à la référence à tout moment.",
    },
    "reply.alert.bullet_metric": {"en": "Metric: {metric}", "fr": "Indicateur : {metric}"},
    "reply.alert.bullet_baseline": {"en": "Baseline: {baseline}", "fr": "Référence : {baseline}"},
    "reply.alert.bullet_trigger": {
        "en": "Trigger: change > {threshold}%",
        "fr": "Déclencheur : variation > {threshold} %",
    },
    # "change_pct" is the stored condition token, not copy.
    "reply.alert.bullet_condition": {
        "en": "Condition: {condition}",
        "fr": "Condition : {condition}",
    },
    "reply.alert.next_step": {
        "en": "Ask \"Check alert {id}\" to compare the current value to this baseline.",
        "fr": "Demandez « Vérifie l'alerte {id} » pour comparer la valeur actuelle à cette référence.",
    },
    "reply.alert.failed": {
        "en": "Could not create the alert.",
        "fr": "L'alerte n'a pas pu être créée.",
    },
    "reply.rootcause.title": {
        "en": "Root cause analysis",
        "fr": "Analyse des causes profondes",
    },
    "reply.rootcause.failed_body": {
        "en": "I could not run the breakdown automatically. Try asking directly: \"Why did this value change?\" or \"Break it down by [dimension]\".",
        "fr": "Je n'ai pas pu exécuter la ventilation automatiquement. Demandez directement : « Pourquoi cette valeur a-t-elle changé ? » ou « Ventile-la par [dimension] ».",
    },
    "reply.plan.explain_hint": {
        "en": "Sure -- ask me a question and I'll explain my plan before running it, e.g. \"explain your plan: what was net revenue for last 7 days\".",
        "fr": "Bien sûr — posez-moi une question et j'expliquerai mon plan avant de l'exécuter, par exemple « explique ton plan : quel était le chiffre d'affaires net sur les 7 derniers jours ».",
    },
    "reply.plan.preview_suffix": {
        "en": "{summary} Say \"go ahead\" to run it, or tell me what to change.",
        "fr": "{summary} Dites « vas-y » pour l'exécuter, ou indiquez-moi ce qu'il faut changer.",
    },

    # ── Breaking a result down by a dimension ────────────────────────────────
    # {dimension} is the dimension's own name from the semantic model, which is
    # the tenant's word and stays as it is.
    "reply.drill.title": {
        "en": "Break down by {dimension}",
        "fr": "Ventiler par {dimension}",
    },
    "reply.drill.suggestion_default": {
        "en": "Try asking: \"Show [metric] broken down by {dimension}\"",
        "fr": "Essayez de demander : « Affiche [indicateur] ventilé par {dimension} »",
    },
    "reply.drill.not_in_model": {
        "en": "Dimension '{dimension}' was not found in the semantic model.",
        "fr": "La dimension « {dimension} » est introuvable dans le modèle sémantique.",
    },
    "reply.drill.not_in_model_suggestion": {
        "en": "Try asking: \"Break down by {dimension}\" in a new question.",
        "fr": "Posez une nouvelle question : « Ventile par {dimension} ».",
    },
    "reply.drill.not_joinable": {
        "en": "The '{dimension}' dimension is not safely joinable to the current result.",
        "fr": "La dimension « {dimension} » ne peut pas être jointe en toute sécurité au résultat actuel.",
    },
    "reply.drill.not_joinable_suggestion": {
        "en": "Ask a new question that explicitly requests a {dimension} breakdown.",
        "fr": "Posez une nouvelle question demandant explicitement une ventilation par {dimension}.",
    },
    "reply.drill.invalid_sql": {
        "en": "The rewritten query failed validation.",
        "fr": "La requête réécrite n'a pas passé la validation.",
    },
    "reply.drill.invalid_sql_suggestion": {
        "en": "Try asking: \"Show [metric] by {dimension}\" directly.",
        "fr": "Demandez directement : « Affiche [indicateur] par {dimension} ».",
    },
    "reply.drill.validation_error": {
        "en": "Validation error while preparing the drill-down query.",
        "fr": "Erreur de validation lors de la préparation de la requête de ventilation.",
    },
    "reply.drill.execution_failed": {
        "en": "The drill-down query failed to execute: {error}",
        "fr": "L'exécution de la requête de ventilation a échoué : {error}",
    },
    "reply.drill.no_data": {
        "en": "The '{dimension}' breakdown returned no data.",
        "fr": "La ventilation par « {dimension} » n'a renvoyé aucune donnée.",
    },
    "reply.drill.no_data_suggestion": {
        "en": "The dimension may not have data for the current filter period.",
        "fr": "Cette dimension n'a peut-être pas de données pour la période filtrée actuelle.",
    },
    "reply.drill.llm_blocked": {
        "en": "Breaking a result down with AI is turned off for this workspace by the data policy.",
        "fr": "La ventilation d'un résultat par l'IA est désactivée pour cet espace de travail par la politique de données.",
    },
    "reply.drill.llm_blocked_suggestion": {
        "en": "Ask \"Break down by {dimension}\" as a new question — that runs as a governed query instead.",
        "fr": "Posez « Ventile par {dimension} » comme nouvelle question : elle s'exécutera alors comme une requête gouvernée.",
    },
    "reply.drill.failed": {
        "en": "Could not complete the drill-down.",
        "fr": "La ventilation n'a pas pu être effectuée.",
    },
    "reply.drill.failed_suggestion": {
        "en": "Try asking: \"Break down by {dimension}\" directly.",
        "fr": "Demandez directement : « Ventile par {dimension} ».",
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
    except Exception:  # noqa: BLE001
        # Anything that will not coerce takes the plural. Deliberately broad,
        # and for the same reason t() leaves an unsupplied placeholder in
        # place: a count that arrives missing is a bug, but a page that 500s
        # because of it costs the reader everything else on it. Jinja's
        # Undefined raises UndefinedError rather than TypeError, so a template
        # whose route forgot one key is exactly this case.
        number = 2.0
    if tag == "fr":
        one = number < 2          # French: 0 and 1 both take the singular
    else:
        one = number == 1
    return t(f"{msg_id_stem}.{'one' if one else 'other'}", lang=tag,
             count=count, **kw)


def grain_label(grain, count=2, lang: str | None = None) -> str:
    """The name of a time grain, in the number a count calls for.

    Four modules wrote this as ``f"{grain}s"``. ``grain`` is a wire token --
    "day", "week", "month" -- and suffixing it was already wrong in English
    (core/forecast_gate.py clamps a horizon to 1 and said "1 months"); in
    French "mois" has no plural s and zero takes the singular. An unrecognised
    grain comes back verbatim rather than as a missing-id token, because a
    tenant's own ``temporal_grain`` is data, not copy.
    """
    key = str(grain or "").strip().lower()
    stem = f"caveat.grain.{key}"
    if f"{stem}.one" not in MESSAGES:
        return key
    return plural(stem, count, lang=lang)


def format_count(value, lang: str | None = None) -> str:
    """A whole number with the thousands separator its language groups by.

    ``f"{n:,}"`` is English. French groups with a narrow no-break space and
    reads a comma as the decimal point, so "1,234 lignes" is one and a bit
    rather than a thousand -- an off-by-a-thousand in a caveat about how much
    of the result the reader is being shown.
    """
    try:
        grouped = f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)
    tag = normalise_language(lang if lang is not None else get_active_language())
    return grouped.replace(",", "\u202f") if tag == "fr" else grouped


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
