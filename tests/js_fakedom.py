"""
tests/js_fakedom.py

A browser, as far as static/js/qb-ui.js touches one, for dukpy.

Not a test module. The component code runs unmodified; only document, window,
timers and elements are stand-ins. Promises settle synchronously so a test can
read what a dialog resolved to within one evaluation (dukpy does not run the
job queue before evaljs returns). runTimers() fires what setTimeout scheduled.
toastsNow() and dialogNow() read what is on the page; make(tag, attrs, parent)
builds an element; goToHash(h) changes location.hash as a link would; El.find/byClass search
an element's subtree; El.fire(type) dispatches to its listeners.
"""

FAKE_DOM = r"""
function SyncPromise(executor) {
  var self = this; self.done = false; self.value = undefined; self.cbs = [];
  executor(function (v) { if (self.done) return; self.done = true; self.value = v;
                          self.cbs.forEach(function (cb) { cb(v); }); });
}
SyncPromise.prototype.then = function (cb) { if (this.done) cb(this.value); else this.cbs.push(cb); return this; };
var Promise = SyncPromise;

function El(tag) {
  var self = this;
  this.tagName = tag.toUpperCase(); this.children = []; this.attributes = {}; this.listeners = {};
  this.parentNode = null; this.className = ''; this._text = ''; this._html = ''; this.dataset = {};
  this.id = ''; this.value = ''; this.placeholder = ''; this.open = false; this.clicks = 0;
  this.classList = {
    add: function (c) { var s = self.className.split(' ').filter(Boolean); if (s.indexOf(c) < 0) s.push(c); self.className = s.join(' '); },
    remove: function (c) { self.className = self.className.split(' ').filter(function (x) { return x && x !== c; }).join(' '); },
    contains: function (c) { return self.className.split(' ').indexOf(c) >= 0; }
  };
}
El.prototype.appendChild = function (c) { c.parentNode = this; this.children.push(c); return c; };
El.prototype.removeChild = function (c) { this.children.splice(this.children.indexOf(c), 1); c.parentNode = null; };
El.prototype.setAttribute = function (k, v) { this.attributes[k] = String(v); };
El.prototype.getAttribute = function (k) { return this.attributes.hasOwnProperty(k) ? this.attributes[k] : null; };
El.prototype.addEventListener = function (t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); };
El.prototype.fire = function (t, extra) {
  var ev = extra || {}; ev.type = t; ev.target = ev.target || this; ev.defaultPrevented = false;
  ev.preventDefault = function () { ev.defaultPrevented = true; }; ev.stopPropagation = function () {};
  (this.listeners[t] || []).slice().forEach(function (f) { f(ev); }); return ev;
};
El.prototype.focus = function () { document.activeElement = this; };
El.prototype.select = function () {};
El.prototype.click = function () { this.clicks += 1; };
El.prototype.showModal = function () { this.open = true; };
El.prototype.close = function () { this.open = false; };
El.prototype.find = function (pred) {
  if (pred(this)) return this;
  for (var i = 0; i < this.children.length; i++) { var hit = this.children[i].find(pred); if (hit) return hit; }
  return null;
};
El.prototype.byClass = function (c) { return this.find(function (e) { return e.classList.contains(c); }); };
El.prototype.hasAttribute = function (k) { return this.attributes.hasOwnProperty(k); };
// Selectors: a tag, .class, [attr] and [attr="value"], compounded (no combinators).
El.prototype.matches = function (sel) {
  var el = this, rest = sel.replace(/\[([\w-]+)(?:="([^"]*)")?\]/g, function (_, k, v) {
    if (!el.hasAttribute(k) && !(k === 'id' && el.id)) { el._miss = true; }
    else if (v !== undefined && el.getAttribute(k) !== v) { el._miss = true; }
    return '';
  });
  var miss = !!el._miss; el._miss = false;
  if (miss) return false;
  var classes = rest.split('.'), tag = classes.shift();
  if (tag && tag.toUpperCase() !== el.tagName) return false;
  return classes.every(function (c) { return !c || el.classList.contains(c); });
};
El.prototype.closest = function (sel) {
  for (var node = this; node && node.matches; node = node.parentNode) { if (node.matches(sel)) return node; }
  return null;
};
El.prototype.querySelectorAll = function (sel) {
  var out = [];
  (function walk(node) { node.children.forEach(function (c) { if (c.matches(sel)) out.push(c); walk(c); }); })(this);
  return out;
};
El.prototype.querySelector = function (sel) { return this.querySelectorAll(sel)[0] || null; };
El.prototype.dispatchEvent = function (ev) { this.fire(ev.type, ev); return true; };
Object.defineProperty(El.prototype, 'textContent', {
  get: function () { return this._text + this.children.map(function (c) { return c.textContent; }).join(''); },
  set: function (v) { this._text = String(v); this.children = []; }
});
Object.defineProperty(El.prototype, 'innerHTML', {
  get: function () { return this._html; }, set: function (v) { this._html = String(v); }
});

var docListeners = {};
var document = { body: new El('body'), activeElement: null,
  createElement: function (t) { return new El(t); },
  addEventListener: function (t, f) { (docListeners[t] = docListeners[t] || []).push(f); },
  querySelectorAll: function (sel) { return document.body.querySelectorAll(sel); },
  getElementById: function (id) { return document.body.find(function (e) { return e.id === id; }); } };
var timers = [];
var winListeners = {};
var window = { document: document,
  setTimeout: function (f, ms) { timers.push({f: f, ms: ms}); return timers.length; },
  clearTimeout: function (id) { if (id && timers[id - 1]) timers[id - 1].cleared = true; },
  qbIcon: function (n, s) { return '<svg data-icon="' + n + '"></svg>'; },
  location: { hash: '' },
  history: { replaceState: function (s, t, url) { window.location.hash = String(url); } },
  addEventListener: function (t, f) { (winListeners[t] = winListeners[t] || []).push(f); } };
function goToHash(h) { window.location.hash = h; (winListeners.hashchange || []).forEach(function (f) { f({}); }); }
function make(tag, attrs, parent) {
  var el = new El(tag);
  Object.keys(attrs || {}).forEach(function (k) { if (k === 'id') el.id = attrs[k]; else el.setAttribute(k, attrs[k]); });
  (parent || document.body).appendChild(el);
  return el;
}
function runTimers() {
  for (var round = 0; round < 10; round++) {
    var due = timers.filter(function (t) { return !t.ran && !t.cleared; });
    if (!due.length) return;
    due.forEach(function (t) { t.ran = true; t.f(); });
  }
}
function toastsNow() { var r = document.body.byClass('qb-toasts'); return r ? r.children : []; }
function dialogNow() { return document.body.find(function (e) { return e.tagName === 'DIALOG'; }); }
"""
