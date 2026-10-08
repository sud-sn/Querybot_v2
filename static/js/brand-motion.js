(function () {
  'use strict';

  // The mark's one motion: while an answer is worked out its bars rise and
  // fall (data-state="thinking" and the other working states); anything else
  // leaves it still. Callers set the state through here.
  function resolve(target) {
    if (!target) return null;
    return typeof target === 'string' ? document.getElementById(target) : target;
  }

  function setState(target, state, label) {
    var node = resolve(target);
    if (!node) return;
    node.dataset.state = state || 'idle';
    if (label != null) {
      var labelNode = node.querySelector('[data-brand-label]');
      if (labelNode) labelNode.textContent = label;
    }
  }

  window.QBBrandMotion = { setState: setState };

  // A sign-in form says it is working on its own button, not with the brand:
  // the button is held busy so a second press does not post the form twice.
  document.addEventListener('submit', function (event) {
    var form = event.target;
    if (!form || !form.matches || !form.matches('form[data-busy-on-submit]')) return;
    var button = form.querySelector('[type="submit"]');
    if (!button) return;
    button.setAttribute('aria-busy', 'true');
    button.disabled = true;
  }, true);
  // Back to the page from the browser's history: the form is usable again.
  window.addEventListener('pageshow', function () {
    document.querySelectorAll('form[data-busy-on-submit] [type="submit"][aria-busy="true"]').forEach(function (button) {
      button.removeAttribute('aria-busy');
      button.disabled = false;
    });
  });
})();
