(function () {
  document.addEventListener('click', function (e) {
    const btn = e.target.closest('.btn.btn-secondary'); // your Clear button
    if (!btn) return;
    const card = btn.closest('.form-card');
    if (!card) return;
    const inputs = card.querySelectorAll('.inp');
    inputs.forEach(inp => { inp.value = ''; inp.dispatchEvent(new Event('input', {bubbles: true})); });
  }, true);
})();
