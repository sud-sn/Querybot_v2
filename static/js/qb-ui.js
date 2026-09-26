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

   qbSelect(select) / <select data-qb-select>
     A searchable list over a native <select>, which stays in the
     form and keeps its name, value and change event, so the page
     posts and reacts exactly as before. Typing filters the options
     (case and accents ignored); arrows move, Enter picks, Esc
     puts the choice back. Options a script replaces, a value it
     sets and the select's disabled state are followed.

   <button data-qb-reveal="id [id]"> shows or hides those password
   fields (.input-reveal-wrap in base.css).

   Submitting a form (both consoles, every form but
   data-no-loading) marks the pressed button aria-busy, which .btn
   draws as a spinner, and holds a second submit until the page
   leaves (or 15 s). A form a page's own script stops is left alone.

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

  /* ── Searchable select ─────────────────────────────────── */
  var selectCount = 0;

  function fold(text) {
    var out = String(text || '').toLowerCase();
    try { out = out.normalize('NFD').replace(/[\u0300-\u036f]/g, ''); } catch (_) { /* no normalize: exact */ }
    return out;
  }

  function optionsOf(select) {
    return Array.prototype.map.call(select.querySelectorAll('option'), function (opt) {
      var group = opt.parentNode && opt.parentNode.tagName === 'OPTGROUP' ? opt.parentNode.getAttribute('label') : '';
      // An option's text is read as the browser shows it: runs of white
      // space (a template's line breaks) collapse, the ends are trimmed.
      var text = String(opt.textContent || '').replace(/\s+/g, ' ').trim();
      return { el: opt, value: opt.getAttribute('value') !== null ? opt.getAttribute('value') : text,
               label: text, group: group || '', disabled: opt.hasAttribute('disabled') };
    });
  }

  function fireChange(select) {
    var ev;
    try { ev = new global.Event('change', { bubbles: true }); } catch (_) { ev = { type: 'change', bubbles: true }; }
    select.dispatchEvent(ev);
  }

  function qbSelect(select) {
    if (!select) return null;
    if (select.__qbSelect) return select.__qbSelect;
    var id = 'qb-select-' + (++selectCount);

    var wrap = doc.createElement('div');
    wrap.className = 'qb-select';
    var input = doc.createElement('input');
    input.setAttribute('type', 'text');
    input.setAttribute('role', 'combobox');
    input.setAttribute('aria-autocomplete', 'list');
    input.setAttribute('aria-expanded', 'false');
    input.setAttribute('aria-controls', id + '-list');
    input.setAttribute('autocomplete', 'off');
    input.className = 'form-input qb-select-input';
    input.id = id;
    var list = doc.createElement('ul');
    list.id = id + '-list';
    list.className = 'qb-select-list';
    list.setAttribute('role', 'listbox');
    list.hidden = true;

    // What named the select names what is typed in: its <label for>, the
    // <label> just before it, or its aria-label.
    var labelEl = select.id && doc.querySelector ? doc.querySelector('label[for="' + select.id + '"]') : null;
    var before = select.previousElementSibling;
    if (!labelEl && before && before.tagName === 'LABEL') labelEl = before;
    if (labelEl) {
      if (!labelEl.id) labelEl.id = id + '-label';
      input.setAttribute('aria-labelledby', labelEl.id);
      if (labelEl.getAttribute('for') === select.id) labelEl.setAttribute('for', id);
    } else if (select.getAttribute('aria-label')) {
      input.setAttribute('aria-label', select.getAttribute('aria-label'));
    }

    var chevron = doc.createElement('span');
    chevron.className = 'qb-select-chevron';
    chevron.setAttribute('aria-hidden', 'true');
    chevron.innerHTML = icon('chevron-down', 16);

    select.parentNode.insertBefore(wrap, select);
    wrap.appendChild(select);
    wrap.appendChild(input);
    wrap.appendChild(list);
    wrap.appendChild(chevron);
    select.classList.add('qb-select-native');
    select.setAttribute('tabindex', '-1');
    select.setAttribute('aria-hidden', 'true');

    var shown = [];
    var active = -1;

    function current() {
      var value = select.value;
      return optionsOf(select).filter(function (o) { return o.value === value; })[0] || null;
    }
    function placeholder() {
      var blank = optionsOf(select).filter(function (o) { return o.value === ''; })[0];
      return blank ? blank.label : null;
    }
    function sync() {
      var picked = current();
      input.value = picked && picked.value !== '' ? picked.label : '';
      input.placeholder = placeholder() || '';
      input.disabled = !!select.disabled;
    }
    function close(restore) {
      list.hidden = true;
      input.setAttribute('aria-expanded', 'false');
      input.removeAttribute('aria-activedescendant');
      if (restore) sync();
    }
    function highlight(index) {
      active = index;
      shown.forEach(function (row, i) {
        row.li.setAttribute('aria-selected', i === index ? 'true' : 'false');
        row.li.classList[i === index ? 'add' : 'remove']('is-active');
      });
      if (index >= 0 && shown[index]) input.setAttribute('aria-activedescendant', shown[index].li.id);
      else input.removeAttribute('aria-activedescendant');
    }
    function render(query) {
      while (list.children.length) list.removeChild(list.children[0]);
      var q = fold(query);
      var lastGroup = null;
      shown = [];
      optionsOf(select).forEach(function (o) {
        if (o.value === '' || o.disabled) return;
        if (q && fold(o.label).indexOf(q) < 0 && fold(o.group).indexOf(q) < 0) return;
        if (o.group && o.group !== lastGroup) {
          var head = doc.createElement('li');
          head.className = 'qb-select-group';
          head.setAttribute('role', 'presentation');
          head.textContent = o.group;
          list.appendChild(head);
          lastGroup = o.group;
        }
        var li = doc.createElement('li');
        li.id = id + '-opt-' + shown.length;
        li.className = 'qb-select-option';
        li.setAttribute('role', 'option');
        li.textContent = o.label;
        var row = { li: li, option: o };
        li.addEventListener('mousedown', function (e) { e.preventDefault(); pick(row.option); });
        list.appendChild(li);
        shown.push(row);
      });
      if (!shown.length) {
        var empty = doc.createElement('li');
        empty.className = 'qb-select-empty';
        empty.setAttribute('role', 'presentation');
        empty.textContent = label('ui.shell.no_matches', 'No matches');
        list.appendChild(empty);
      }
      var picked = current();
      var at = -1;
      shown.forEach(function (row, i) { if (picked && row.option.value === picked.value) at = i; });
      highlight(at >= 0 ? at : (shown.length ? 0 : -1));
    }
    function open(query) {
      if (select.disabled) return;
      render(query);
      list.hidden = false;
      input.setAttribute('aria-expanded', 'true');
    }
    function pick(option) {
      var changed = select.value !== option.value;
      select.value = option.value;
      close(true);
      if (changed) fireChange(select);
    }

    input.addEventListener('focus', function () { if (input.select) input.select(); });
    input.addEventListener('click', function () { if (list.hidden) open(''); });
    input.addEventListener('input', function () { open(input.value); });
    input.addEventListener('blur', function () {
      // Emptying the box clears a choice the form does not require.
      if (!select.required && !String(input.value).trim() && select.value !== '' && placeholder() !== null) {
        close(false);
        pick({ value: '' });
        return;
      }
      close(true);
    });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (list.hidden) { open(''); return; }
        if (!shown.length) return;
        var step = e.key === 'ArrowDown' ? 1 : -1;
        highlight((active + step + shown.length) % shown.length);
      } else if (e.key === 'Enter') {
        if (list.hidden) return;
        e.preventDefault();
        if (active >= 0 && shown[active]) pick(shown[active].option);
      } else if (e.key === 'Escape') {
        if (list.hidden) return;
        e.preventDefault();
        close(true);
      }
    });

    // Follow what the page's own script does to the select.
    select.addEventListener('change', sync);
    if (select.form) select.form.addEventListener('reset', function () { later(sync, 0); });
    if (typeof global.MutationObserver === 'function') {
      new global.MutationObserver(sync).observe(select, { childList: true, subtree: true, attributes: true,
                                                          attributeFilter: ['disabled'] });
    }
    try {
      var proto = Object.getPrototypeOf(select);
      var desc = Object.getOwnPropertyDescriptor(proto, 'value');
      if (desc && desc.set) {
        Object.defineProperty(select, 'value', {
          configurable: true,
          get: function () { return desc.get.call(this); },
          set: function (v) { desc.set.call(this, v); sync(); }
        });
      }
    } catch (_) { /* a value set by script is then shown on the next change */ }

    sync();
    var api = { input: input, select: select, sync: sync, open: open, close: close };
    select.__qbSelect = api;
    return api;
  }

  global.qbSelect = qbSelect;

  /* ── Show a password ───────────────────────────────────── */
  // <button data-qb-reveal="id [id]"> shows or hides the named password
  // fields together (a new password and its confirmation), and says which.
  doc.addEventListener('click', function (e) {
    var btn = e.target && e.target.closest ? e.target.closest('[data-qb-reveal]') : null;
    if (!btn) return;
    var inputs = String(btn.getAttribute('data-qb-reveal') || '').split(/[\s,]+/)
      .map(function (id) { return id ? doc.getElementById(id) : null; })
      .filter(function (input) { return input; });
    if (!inputs.length) return;
    var show = inputs[0].type === 'password';
    inputs.forEach(function (input) { input.type = show ? 'text' : 'password'; });
    btn.setAttribute('aria-pressed', show ? 'true' : 'false');
    // A button may say what it shows (a key, a URL); a password is the default.
    btn.setAttribute('aria-label', show
      ? (btn.getAttribute('data-label-hide') || label('ui.auth.hide_password', 'Hide password'))
      : (btn.getAttribute('data-label-show') || label('ui.auth.show_password', 'Show password')));
    btn.innerHTML = icon(show ? 'eye-off' : 'eye', 16);
  });

  /* ── A submitted form's button says it is working ─────────── */
  var BUSY_MS = 15000;
  var busyNow = [];

  function settle(form, btn) {
    form.removeAttribute('data-qb-submitting');
    if (btn) btn.removeAttribute('aria-busy');
    busyNow = busyNow.filter(function (pair) { return pair[0] !== form; });
  }

  doc.addEventListener('submit', function (e) {
    var form = e.target;
    if (!form || !form.getAttribute || form.hasAttribute('data-no-loading')) return;
    // A second press while the first is in flight sends nothing.
    if (form.getAttribute('data-qb-submitting') === '1') { e.preventDefault(); return; }
    // The button pressed, not the form's first: a form with Approve and
    // Reject must not spin the other one.
    var btn = e.submitter && e.submitter.form === form ? e.submitter
      : form.querySelector('[type="submit"], button:not([type])');
    if (btn && btn.hasAttribute('data-no-loading')) btn = null;
    // Decided once every listener has run: a form a page's script stopped
    // (a check that failed, a request it sends itself) is not in flight.
    // The button is not disabled: a disabled submitter drops its
    // name=value from what the form sends.
    later(function () {
      if (e.defaultPrevented) return;
      form.setAttribute('data-qb-submitting', '1');
      if (btn) btn.setAttribute('aria-busy', 'true');
      busyNow.push([form, btn]);
      later(function () { settle(form, btn); }, BUSY_MS);
    }, 0);
  });
  // Back to a page the browser kept in memory: nothing on it is in flight.
  global.addEventListener('pageshow', function (e) {
    if (!e || !e.persisted) return;
    busyNow.slice().forEach(function (pair) { settle(pair[0], pair[1]); });
  });

  function setUp() {
    Array.prototype.forEach.call(doc.querySelectorAll('[data-qb-tabs]'), function (list) { qbTabs(list); });
    Array.prototype.forEach.call(doc.querySelectorAll('select[data-qb-select]'), function (select) { qbSelect(select); });
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', setUp);
  else setUp();
})(window);
