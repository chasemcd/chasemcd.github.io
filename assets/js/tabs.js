document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.tab-link').forEach(function(tab) {
    tab.addEventListener('click', function() {
      var tabGroup = this.closest('.tabs');
      tabGroup.querySelectorAll('.tab-link').forEach(function(t) { t.classList.remove('active'); });
      tabGroup.querySelectorAll('.tab-content').forEach(function(c) { c.classList.remove('active'); });
      this.classList.add('active');
      document.getElementById(this.getAttribute('data-tab')).classList.add('active');
    });
  });
});
