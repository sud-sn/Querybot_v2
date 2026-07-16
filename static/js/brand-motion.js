(function () {
  'use strict';

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

  document.addEventListener('submit', function (event) {
    var form = event.target;
    if (!form || !form.matches || !form.matches('form[data-brand-loading]')) return;
    var targetId = form.getAttribute('data-brand-target');
    var target = targetId ? document.getElementById(targetId) : form.querySelector('.qb-brand-motion');
    setState(target, 'loading', form.getAttribute('data-brand-label') || 'Signing in securely');
    document.body.classList.add('qb-auth-submitting');
  }, true);
})();
