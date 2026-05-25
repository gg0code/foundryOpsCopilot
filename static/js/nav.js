// Mobile navigation toggle — drives the hamburger menu on small screens.
// Shared across all pages; no-ops on desktop where the toggle is hidden via CSS.
(function () {
    'use strict';

    var toggle = document.querySelector('.nav-toggle');
    var nav = document.getElementById('primary-nav');
    if (!toggle || !nav) return;

    function closeMenu() {
        nav.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
    }

    toggle.addEventListener('click', function () {
        var isOpen = nav.classList.toggle('open');
        toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    });

    // Tapping a link should navigate normally and collapse the menu.
    nav.querySelectorAll('a').forEach(function (link) {
        link.addEventListener('click', closeMenu);
    });

    // Reset state when the viewport grows back past the mobile breakpoint.
    window.addEventListener('resize', function () {
        if (window.innerWidth > 860) closeMenu();
    });

    // Escape closes the open menu.
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') closeMenu();
    });
})();
