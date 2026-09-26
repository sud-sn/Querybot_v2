/* qbIcon(name, size, cls): one icon from the sprite, as markup, for page
   scripts that build HTML. Templates use the ic() macro (icons.html), which
   renders the same element; tools/build_icon_sprite.py lists the names.
   A name outside [a-z0-9-] draws nothing rather than reaching the markup. */
(function (global) {
  'use strict';
  var SPRITE = '/static/icons/qb-icons.svg?v=20260926-lucide-1';
  function qbIcon(name, size, cls) {
    name = String(name || '');
    if (!/^[a-z0-9-]+$/.test(name)) return '';
    size = Number(size) || 16;
    var extra = cls && /^[\w -]+$/.test(cls) ? ' ' + cls : '';
    return '<svg class="qb-icon' + extra + '" width="' + size + '" height="' + size +
      '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"' +
      ' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">' +
      '<use href="' + SPRITE + '#' + name + '"/></svg>';
  }
  global.qbIcon = qbIcon;
})(window);
