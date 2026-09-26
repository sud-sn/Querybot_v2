/* ============================================================
   QueryBot UI -- one dialog and one toast, for both consoles.

   The admin console and the portal each carried their own
   confirm dialog and between them three toast systems; the
   portal chat called window.qbToast, which only the admin page
   defined, so its warnings never showed. The admin toast wrote
   its message as HTML, and callers pass server error text.

   qbConfirm(opts) -> Promise
     opts: {title, body, confirm, cancel, variant, input, onConfirm}
       variant: 'danger' (default) | 'primary' | 'warning'
       input:   {placeholder, value} -- collects one line of text
     Resolves true (or the typed value) on confirm, false on
     cancel, Esc or a click outside; onConfirm(value) runs first.
     A native <dialog> shown modally: the browser keeps focus
     inside it, and focus returns to whatever opened it.

   <button data-confirm="Delete Ada?" data-confirm-title="..."
           data-confirm-label="..." data-confirm-variant="...">
     Asks before the button acts; on confirm it is clicked again,
     so it submits exactly as it would have. The message is an
     attribute the page escapes, never part of a handler.

   qbToast.notify({title, body, tone, action, duration, onDismiss})
     tone:   'info' | 'success' | 'warning' | 'danger'
     action: {label, href} or {label, onClick}
     duration: ms before it leaves; 0 keeps it until dismissed.
   qbToast.show(message, tone, duration), .success/.error/.warning/.info
     The older one-line form.

   qbTabs(tablist, {onChange}) -> {select(name)}
     <div role="tablist" data-qb-tabs [data-qb-tabs-hash] aria-label="...">
       <button role="tab" data-tab="profile" id="tab-profile"
               aria-controls="panel-profile" aria-selected="true">
     <section role="tabpanel" id="panel-profile" aria-labelledby="tab-profile">
     One tab component: the selected tab is aria-selected and the
     only one in the Tab order; arrow keys, Home and End move
     between tabs; unselected panels are hidden. With
     data-qb-tabs-hash the tab follows location.hash, so a refresh,
     a bookmark or a link lands on it. [data-qb-tabs] lists are set
     up on load; a script that builds one calls qbTabs itself.
     onChange(name, tab, panel) runs when the reader changes tab.

   All text is set as text, never as markup. Labels come from
   window.qbT when the page has it (the portal, in French too).
   ============================================================ */
(function (global) {
  'use strict';
  var doc = global.document;

  function label(key, fallback) {
    try {
      if (typeof global.qbT === 'function') {
        var value = global.qbT(key);
        if (value && value !== key) return value;
      }
    } catch (_) { /* a missing catalogue falls back to English */ }
    return fallback;
  }

  function icon(name, size) {
    return typeof global.qbIcon === 'function' ? global.qbIcon(name, size) : '';
  }

  function later(fn, ms) { return global.setTimeout(fn, ms); }

  /* ── Toasts ─────────────────────────────────────────────── */
  var TONE_ICONS = { info: 'info', success: 'check-circle', warning: 'alert-triangle', danger: 'x-circle' };
  var TONE_ALIASES = { error: 'danger', blocked: 'danger' };
  var TONE_MS = { info: 5000, success: 4000, warning: 7000, danger: 8000 };
  var region = null;

  function toastRegion() {
    if (region && region.parentNode) return region;
    region = doc.createElement('div');
    region.className = 'qb-toasts';
    region.setAttribute('role', 'status');
    region.setAttribute('aria-live', 'polite');
    doc.body.appendChild(region);
    return region;
  }

  function notify(opts) {
    opts = opts || {};
    var tone = TONE_ALIASES[opts.tone] || opts.tone;
    if (!TONE_ICONS[tone]) tone = 'info';

    var el = doc.createElement('div');
    el.className = 'qb-toast qb-toast--' + tone;
    if (tone === 'danger') el.setAttribute('role', 'alert');

    var mark = doc.createElement('span');
    mark.className = 'qb-toast-icon';
    mark.innerHTML = icon(TONE_ICONS[tone], 16);
    el.appendChild(mark);

    var text = doc.createElement('div');
    text.className = 'qb-toast-text';
    if (opts.title) {
      var title = doc.createElement('strong');
      title.className = 'qb-toast-title';
      title.textContent = String(opts.title);
      text.appendChild(title);
    }
    if (opts.body) {
      var body = doc.createElement('span');
      body.className = 'qb-toast-body';
      body.textContent = String(opts.body);
      text.appendChild(body);
    }
    el.appendChild(text);

    var gone = false;
    var timer = null;
    function dismiss() {
      if (gone) return;
      gone = true;
      global.clearTimeout(timer);
      el.classList.remove('is-in');
      later(function () {
        if (el.parentNode) el.parentNode.removeChild(el);
        if (typeof opts.onDismiss === 'function') opts.onDismiss();
      }, 180);
    }

    var action = opts.action;
    if (action && action.label) {
      var link;
      if (action.href) {
        link = doc.createElement('a');
        link.setAttribute('href', String(action.href));
      } else {
        link = doc.createElement('button');
        link.setAttribute('type', 'button');
        link.addEventListener('click', function () {
          try { if (typeof action.onClick === 'function') action.onClick(); } finally { dismiss(); }
        });
      }
      link.className = 'qb-toast-action';
      link.textContent = String(action.label);
      text.appendChild(link);
    }

    var close = doc.createElement('button');
    close.setAttribute('type', 'button');
    close.className = 'qb-toast-close';
    close.setAttribute('aria-label', label('ui.shell.dismiss', 'Dismiss'));
    close.innerHTML = icon('x', 14);
    close.addEventListener('click', dismiss);
    el.appendChild(close);

    var ms = opts.duration != null ? Number(opts.duration) : TONE_MS[tone];
    function arm(delay) {
      if (!(ms > 0)) return;
      global.clearTimeout(timer);
      timer = later(dismiss, delay);
    }
    // Reading a toast holds it: pointer or keyboard focus on it stops the clock.
    el.addEventListener('mouseenter', function () { global.clearTimeout(timer); });
    el.addEventListener('mouseleave', function () { arm(2000); });
    el.addEventListener('focusin', function () { global.clearTimeout(timer); });

    toastRegion().appendChild(el);
    later(function () { el.classList.add('is-in'); }, 0);
    arm(ms);
    return { element: el, dismiss: dismiss };
  }

  function show(message, tone, duration) {
    return notify({ body: message, tone: tone, duration: duration }).element;
  }

  global.qbToast = {
    notify: notify,
    show: show,
    success: function (m, d) { return show(m, 'success', d); },
    error: function (m, d) { return show(m, 'danger', d); },
    warning: function (m, d) { return show(m, 'warning', d); },
    info: function (m, d) { return show(m, 'info', d); }
  };

  /* ── The dialog ─────────────────────────────────────────── */
  var VARIANTS = ['danger', 'primary', 'warning'];
  var current = null;

  function qbConfirm(opts) {
    opts = opts || {};
    // One question at a time: a new one answers the open one with "no".
    if (current) current.cancel();

    var opener = doc.activeElement;
    var dialog = doc.createElement('dialog');
    dialog.className = 'qb-dialog';
    dialog.setAttribute('aria-labelledby', 'qbDialogTitle');

    var form = doc.createElement('form');
    form.setAttribute('method', 'dialog');
    form.className = 'qb-dialog-form';

    var title = doc.createElement('h2');
    title.className = 'qb-dialog-title';
    title.id = 'qbDialogTitle';
    title.textContent = String(opts.title || label('ui.shell.confirm_title', 'Are you sure?'));
    form.appendChild(title);

    if (opts.body) {
      var body = doc.createElement('p');
      body.className = 'qb-dialog-body';
      body.id = 'qbDialogBody';
      body.textContent = String(opts.body);
      form.appendChild(body);
      dialog.setAttribute('aria-describedby', 'qbDialogBody');
    }

    var input = null;
    if (opts.input) {
      input = doc.createElement('input');
      input.setAttribute('type', 'text');
      input.className = 'form-input qb-dialog-input';
      input.setAttribute('aria-labelledby', 'qbDialogTitle');
      input.placeholder = String(opts.input.placeholder || '');
      input.value = String(opts.input.value || '');
      form.appendChild(input);
    }

    var footer = doc.createElement('div');
    footer.className = 'qb-dialog-footer';
    var cancel = doc.createElement('button');
    cancel.setAttribute('type', 'button');
    cancel.className = 'btn btn-secondary';
    cancel.textContent = String(opts.cancel || label('ui.shell.cancel', 'Cancel'));
    var confirm = doc.createElement('button');
    confirm.setAttribute('type', 'submit');
    confirm.setAttribute('value', 'confirm');
    confirm.className = 'btn btn-' + (VARIANTS.indexOf(opts.variant) >= 0 ? opts.variant : 'danger');
    confirm.textContent = String(opts.confirm || label('ui.shell.confirm', 'Confirm'));
    footer.appendChild(cancel);
    footer.appendChild(confirm);
    form.appendChild(footer);
    dialog.appendChild(form);

    var settle;
    var promise = new Promise(function (resolve) { settle = resolve; });
    var done = false;
    function finish(confirmed) {
      if (done) return;
      done = true;
      if (current && current.dialog === dialog) current = null;
      var value = input ? input.value : undefined;
      if (dialog.open) dialog.close();
      if (dialog.parentNode) dialog.parentNode.removeChild(dialog);
      if (opener && typeof opener.focus === 'function' && opener.isConnected !== false) opener.focus();
      if (confirmed && typeof opts.onConfirm === 'function') opts.onConfirm(value);
      settle(confirmed ? (input ? value : true) : false);
    }

    form.addEventListener('submit', function (e) { e.preventDefault(); finish(true); });
    cancel.addEventListener('click', function () { finish(false); });
    // Esc: the browser fires "cancel" and would close the dialog itself.
    dialog.addEventListener('cancel', function (e) { e.preventDefault(); finish(false); });
    // A click on the backdrop lands on the <dialog> element itself.
    dialog.addEventListener('click', function (e) { if (e.target === dialog) finish(false); });

    doc.body.appendChild(dialog);
    current = { dialog: dialog, cancel: function () { finish(false); } };
    dialog.showModal();
    if (input) { input.focus(); input.select(); } else { confirm.focus(); }
    return promise;
  }

  global.qbConfirm = qbConfirm;

  doc.addEventListener('click', function (e) {
    var el = e.target && e.target.closest ? e.target.closest('[data-confirm]') : null;
    if (!el) return;
    if (el.dataset.confirmed === '1') { delete el.dataset.confirmed; return; }
    e.preventDefault();
    e.stopPropagation();
    qbConfirm({
      title: el.dataset.confirmTitle,
      body: el.dataset.confirm,
      confirm: el.dataset.confirmLabel,
      variant: el.dataset.confirmVariant,
      onConfirm: function () { el.dataset.confirmed = '1'; el.click(); }
    });
  }, true);

  /* ── Tabs ──────────────────────────────────────────────── */
  function tabsOf(list) {
    return Array.prototype.filter.call(list.querySelectorAll('[role="tab"]'), function (tab) {
      return tab.closest('[role="tablist"]') === list;
    });
  }

  function panelOf(tab) {
    var id = tab.getAttribute('aria-controls');
    return id ? doc.getElementById(id) : null;
  }

  function nameOf(tab) { return tab.getAttribute('data-tab') || tab.id; }

  function qbTabs(list, opts) {
    if (!list) return null;
    if (list.__qbTabs) return list.__qbTabs;
    opts = opts || {};
    list.setAttribute('role', 'tablist');
    var followHash = list.hasAttribute('data-qb-tabs-hash');

    function byName(name) {
      return tabsOf(list).filter(function (tab) { return nameOf(tab) === name; })[0] || null;
    }

    function select(tab, how) {
      var tabs = tabsOf(list);
      if (tabs.indexOf(tab) < 0) return false;
      tabs.forEach(function (each) {
        var on = each === tab;
        each.setAttribute('aria-selected', on ? 'true' : 'false');
        each.setAttribute('tabindex', on ? '0' : '-1');
        each.classList[on ? 'add' : 'remove']('is-active');
        var panel = panelOf(each);
        if (panel) {
          panel.hidden = !on;
          panel.classList[on ? 'add' : 'remove']('is-active');
        }
      });
      if (how === 'key') tab.focus();
      if (how === 'init') return true;
      var name = nameOf(tab);
      if (followHash && how !== 'hash' && global.location.hash !== '#' + name) {
        global.history.replaceState(null, '', '#' + name);
      }
      if (typeof opts.onChange === 'function') opts.onChange(name, tab, panelOf(tab));
      return true;
    }

    list.addEventListener('click', function (e) {
      var tab = e.target && e.target.closest ? e.target.closest('[role="tab"]') : null;
      if (tab) select(tab, 'click');
    });
    list.addEventListener('keydown', function (e) {
      var tabs = tabsOf(list);
      var at = tabs.indexOf(doc.activeElement);
      if (at < 0) return;
      var to = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: tabs.length - 1 }[e.key];
      if (to === undefined) return;
      e.preventDefault();
      select(tabs[(to + tabs.length) % tabs.length], 'key');
    });

    var api = {
      list: list,
      select: function (name) { var tab = byName(name); return tab ? select(tab, 'api') : false; }
    };
    list.__qbTabs = api;

    var tabs = tabsOf(list);
    var first = (followHash && byName(String(global.location.hash || '').slice(1)))
      || tabs.filter(function (tab) { return tab.getAttribute('aria-selected') === 'true'; })[0]
      || tabs[0];
    if (first) select(first, 'init');
    if (followHash) {
      global.addEventListener('hashchange', function () {
        var tab = byName(String(global.location.hash || '').slice(1));
        if (tab) select(tab, 'hash');
      });
    }
    return api;
  }

  global.qbTabs = qbTabs;

  function setUpTabs() {
    Array.prototype.forEach.call(doc.querySelectorAll('[data-qb-tabs]'), function (list) { qbTabs(list); });
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', setUpTabs);
  else setUpTabs();
})(window);
