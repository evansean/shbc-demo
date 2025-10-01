(function () {
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter') return;
    const target = e.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains('inp')) return;

    e.preventDefault();
    const card = target.closest('.form-card') || document;
    const inputs = Array.from(card.querySelectorAll('.inp')).filter(el => !el.disabled);
    const idx = inputs.indexOf(target);
    if (idx === -1) return;
    if (idx < inputs.length - 1) {
      const next = inputs[idx + 1];
      next.focus();
      if (typeof next.select === 'function') next.select();
      return;
    }
    const submit = card.querySelector('.buttons .btn:not(.btn-secondary)');
    if (submit) submit.focus();
  }, true);
})();
