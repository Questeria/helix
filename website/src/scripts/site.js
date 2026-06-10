// =====================================================================
// Shared site interactions: scroll-aware nav, reveal-on-scroll,
// mobile nav toggle. Loaded on every page.
// =====================================================================

// ---------- Nav adapts to the section it currently overlaps ----------
(function () {
  const nav = document.querySelector(".nav");
  if (!nav) return;
  const sections = [...document.querySelectorAll("section, footer")];
  if (!sections.length) return;

  let rafId = null;
  function update() {
    rafId = null;
    const navBottom = 80; // nav height + a hair
    let current = sections[0];
    for (const s of sections) {
      const rect = s.getBoundingClientRect();
      if (rect.top <= navBottom) current = s;
    }
    nav.classList.toggle("nav-dark", current.classList.contains("dark"));
  }
  window.addEventListener(
    "scroll",
    () => {
      if (rafId) return;
      rafId = requestAnimationFrame(update);
    },
    { passive: true }
  );
  update();
})();

// ---------- Mobile nav toggle ----------
(function () {
  const toggle = document.querySelector(".nav-toggle");
  const links = document.getElementById("nav-links");
  if (!toggle || !links) return;
  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  });
  // Close on link click + on Escape
  links.addEventListener("click", (e) => {
    if (e.target.closest("a")) {
      links.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && links.classList.contains("open")) {
      links.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
      toggle.focus();
    }
  });
})();

// ---------- Reveal-on-scroll ----------
(function () {
  const els = document.querySelectorAll(".reveal");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced || !("IntersectionObserver" in window)) {
    els.forEach((el) => el.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
  );
  els.forEach((el) => io.observe(el));
})();
