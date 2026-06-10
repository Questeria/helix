// =====================================================================
// Home hero: byte ribbon + lazy 3D helix mount (with graceful fallback).
// The heavy three.js module is only fetched when the hero is on screen
// and the user hasn't asked for reduced motion.
// =====================================================================

// ---------- Helix code ribbon — scrolling snippets of source ----------
(function () {
  const track = document.getElementById("byte-track");
  if (!track) return;
  const LINES = [
    "fn matmul(a: Tensor[f32], b: Tensor[f32]) -> Tensor[f32]",
    "let grad = backward(loss, [w1, w2, b])",
    "let g = grad_rev_all(loss)(2.0_f64, 3.0_f64)",
    "tile(x, [128, 64]) |> matmul(w) |> relu",
    "Quote { fn step(x: bf16) -> bf16 { x * lr } }",
    "Splice($body) into modify fn forward",
    "impl Layer for RMSNorm { fn forward(&self, x) { ... } }",
    "let logits = embed(tok) |> stack(blocks) |> unembed",
    "@pure fn rmsnorm(x: Tensor[f32]) -> Tensor[f32]",
    "kernel matmul_tiled<TILE=128>(a, b, out) { ... }",
    "for i in 0..steps { opt.step(grad); zero_grad() }",
    "module Transformer { layers: [Block; N_LAYERS] }",
    "let x = x + attn(rmsnorm(x))",
    "fn rope(q: Tensor, k: Tensor) -> (Tensor, Tensor)",
    "@inline fn softmax(x) { exp(x) / sum(exp(x)) }",
    "trait Optim { fn step(&mut self, p: &mut Param) }",
    "match dtype { F16 => ..., BF16 => ..., F32 => ... }",
    "grad(f)(x)  //  forward-mode, symbolic",
  ];
  const TAGS = ["// helix", "// kernel", "// grad", "// ptx", "// macro", "// trait", "// module"];
  let s = "";
  for (let i = 0; i < 60; i++) {
    if (i % 5 === 0) s += `<span class="b-tag">${TAGS[Math.floor(Math.random() * TAGS.length)]}</span>`;
    s += `<span>${LINES[Math.floor(Math.random() * LINES.length)].replace(/&/g, "&amp;").replace(/</g, "&lt;")}</span>`;
  }
  track.innerHTML = s + s; // duplicate for seamless loop
})();

// ---------- Mount 3D helix (lazy, fallback-safe) ----------
(function () {
  const slot = document.getElementById("hero-3d");
  if (!slot) return;

  const fallbackTemplate = document.getElementById("hero-3d-fallback-tpl");
  function showFallback() {
    if (fallbackTemplate) {
      slot.innerHTML = "";
      slot.appendChild(fallbackTemplate.content.cloneNode(true));
    }
  }

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced) {
    // Static brand mark instead of a spinning WebGL scene.
    showFallback();
    return;
  }

  let handle = null;
  let contextLost = false;

  async function mount() {
    try {
      const { mountHelix } = await import("./helix-logo.js");
      slot.innerHTML = "";
      contextLost = false;
      handle = mountHelix(slot, { autoRotate: true });
      handle.setTheme(true); // hero is always the dark carbon scene
      const canvas = slot.querySelector("canvas");
      if (canvas) {
        canvas.addEventListener("webglcontextlost", (e) => {
          e.preventDefault();
          contextLost = true;
        });
        canvas.addEventListener("webglcontextrestored", () => {
          contextLost = false;
        });
      }
    } catch (err) {
      console.error("Helix 3D failed to mount:", err);
      showFallback();
    }
  }

  function glDead() {
    const canvas = slot.querySelector("canvas");
    if (!canvas) return true;
    if (contextLost) return true;
    try {
      const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
      return !!gl && gl.isContextLost();
    } catch (_) {
      return false;
    }
  }

  function remount() {
    try { if (handle) handle.dispose(); } catch (_) { /* no-op */ }
    handle = null;
    mount();
  }

  // Browsers may kill the WebGL context while the tab is backgrounded, and
  // back/forward-cache restores can revive the page with a dead canvas.
  // Whenever the page becomes visible again, remount if the context is gone.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && handle && glDead()) remount();
  });
  window.addEventListener("pageshow", (e) => {
    if (handle && (e.persisted || glDead())) remount();
  });

  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          io.disconnect();
          mount();
        }
      },
      { rootMargin: "200px" }
    );
    io.observe(slot);
  } else {
    mount();
  }
})();

// ---------- Hero SVG scene generator (grid, hex wave, mountains, stars) ----
(function () {
  const ns = "http://www.w3.org/2000/svg";
  const groundG = document.querySelector("#ground-lines");
  const starsG = document.querySelector("#stars");
  const mtnsG = document.querySelector("#mountains");
  if (!groundG || !starsG || !mtnsG) return;
  const VPX = 800, HY = 540, BOT = 900;

  function line(parent, cls, x1, y1, x2, y2) {
    const ln = document.createElementNS(ns, "line");
    ln.setAttribute("class", cls);
    ln.setAttribute("x1", x1); ln.setAttribute("y1", y1);
    ln.setAttribute("x2", x2); ln.setAttribute("y2", y2);
    parent.appendChild(ln);
  }
  function curvedBand(parent, cls, yCenter, bow) {
    const path = document.createElementNS(ns, "path");
    path.setAttribute("class", cls);
    const yEdge = yCenter + bow;
    const yCtrl = yCenter - bow;
    path.setAttribute("d", `M -200 ${yEdge} Q 800 ${yCtrl} 1800 ${yEdge}`);
    path.setAttribute("fill", "none");
    parent.appendChild(path);
  }

  // ---------- Ground grid ----------
  const HXL = 540, HXR = 1060;
  const FXL = -18000, FXR = 19600;
  const COLS = 240;
  for (let i = 0; i <= COLS; i++) {
    const t = i / COLS;
    const xf = FXL + (FXR - FXL) * t;
    const xh = HXL + (HXR - HXL) * t;
    const cls = i % 8 === 0 ? "ln-b" : "ln";
    line(groundG, cls, xf, BOT + 100, xh, HY);
  }
  const ROWS = 24;
  const MAX_BOW = 26;
  for (let k = 1; k <= ROWS; k++) {
    const t = Math.pow(k / ROWS, 0.5);
    const yCenter = BOT - (BOT - HY) * t;
    const bow = MAX_BOW * t;
    const cls = k % 3 === 0 ? "ln-b" : "ln";
    curvedBand(groundG, cls, yCenter, bow);
  }

  // ---------- hex0 wave grid ----------
  const hexG = document.querySelector("#hex-grid");
  if (hexG) {
    const PER_TABLE = {
      92: 80.0, 93: 76.5, 94: 74.0, 95: 75.0, 96: 72.0,
      97: 78.5, 98: 71.0, 99: 70.0, 100: 68.0, 101: 67.5,
      102: 64.5, 103: 67.0, 104: 64.0, 105: 64.5, 106: 64.5,
      107: 60.0, 108: 57.5, 109: 56.0, 110: 54.5, 111: 54.5,
      112: 50.5, 113: 50.0, 114: 44.5, 115: 37.5,
      116: 33.5, 117: 24.5, 118: 18.0, 119: 8.5,
      120: 0.0,
      121: -8.5, 122: -18.0, 123: -24.5, 124: -33.5,
      125: -37.5, 126: -44.5, 127: -50.0, 128: -50.5,
      129: -54.5, 130: -54.5, 131: -56.0, 132: -57.5, 133: -60.0,
      134: -64.5, 135: -64.5, 136: -64.0, 137: -67.0, 138: -64.5,
      139: -67.5, 140: -68.0, 141: -70.0, 142: -71.0, 143: -78.5,
      144: -72.0, 145: -75.0, 146: -74.0, 147: -76.5, 148: -80.0,
    };
    function tiltFor(col) {
      if (col in PER_TABLE) return PER_TABLE[col];
      return col < 120 ? 82 : -82;
    }
    const HEX_POOL = ["48","89","C7","BA","CD","80","EB","C3","B0","3C","0F","05","7F","45","4C","46","B8","01","00","BB","31","C0","E8","5D","FF","D2","83","EC","C1","E9","C6","D1","F8","74","75","FE"];
    let hseed = 7;
    const hrand = () => { hseed = (hseed * 9301 + 49297) % 233280; return hseed / 233280; };

    for (let i = 0; i < COLS; i++) {
      const t = i / COLS;
      const xf = FXL + (FXR - FXL) * t;
      const xh = HXL + (HXR - HXL) * t;
      for (let k = 1; k <= ROWS; k++) {
        const tb = Math.pow(k / ROWS, 0.5);
        const yC = BOT - (BOT - HY) * tb;
        const bow = MAX_BOW * tb;
        const yEdge = yC + bow;
        const YA = BOT + 100;
        const dy = HY - YA;
        let d = (yC - YA) / dy;
        let x = xf + (xh - xf) * d;
        let u = (x + 200) / 2000;
        for (let j = 0; j < 2; j++) {
          const yBand = yEdge - 4 * bow * u * (1 - u);
          d = (yBand - YA) / dy;
          x = xf + (xh - xf) * d;
          u = (x + 200) / 2000;
        }
        if (x < -20 || x > 1620) continue;
        const yAt = yEdge - 4 * bow * u * (1 - u);
        const text = document.createElementNS(ns, "text");
        text.setAttribute("class", "hex-byte");
        text.setAttribute("x", x.toFixed(2));
        text.setAttribute("y", yAt.toFixed(2));
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("dominant-baseline", "central");
        text.setAttribute("font-size", (18 * Math.pow(1 - tb * tb, 1.3) + 1.5).toFixed(2));
        // Apply the hand-tuned per-column tilts exactly as authored in the
        // design handoff (design_files/index.html applyTilt uses tiltFor directly).
        const tilt = tiltFor(i);
        text.setAttribute(
          "transform",
          `translate(${x.toFixed(2)} ${yAt.toFixed(2)}) ` +
            `scale(1 0.42) rotate(${tilt.toFixed(2)}) ` +
            `translate(${(-x).toFixed(2)} ${(-yAt).toFixed(2)})`
        );
        text.textContent = HEX_POOL[Math.floor(hrand() * HEX_POOL.length)];
        const phase = tb * 1.5 + (i / COLS) * 0.8 + hrand() * 0.2;
        text.style.animationDelay = (-phase * 4).toFixed(2) + "s";
        hexG.appendChild(text);
      }
    }
  }

  // ---------- Mountains ----------
  let seed = 7;
  const rand = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };

  const CURVE_AMP = 32;
  function horizonY(x) {
    const u = (x - 800) / 800;
    return HY + CURVE_AMP * (u * u);
  }

  function ridgePoints(xStart, xEnd, baseFn, amp, roughness) {
    const yS = baseFn(xStart), yE = baseFn(xEnd);
    let pts = [
      [xStart, yS + 4],
      [xStart + (xEnd - xStart) * 0.18, baseFn(xStart + (xEnd - xStart) * 0.18) - amp * 0.6],
      [xStart + (xEnd - xStart) * 0.42, baseFn(xStart + (xEnd - xStart) * 0.42) - amp * 0.95],
      [xStart + (xEnd - xStart) * 0.66, baseFn(xStart + (xEnd - xStart) * 0.66) - amp * 0.55],
      [xStart + (xEnd - xStart) * 0.84, baseFn(xStart + (xEnd - xStart) * 0.84) - amp * 0.78],
      [xEnd, yE + 4],
    ];
    for (let pass = 0; pass < 5; pass++) {
      const next = [pts[0]];
      const disp = amp * roughness * Math.pow(0.55, pass);
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], b = pts[i + 1];
        next.push([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + (rand() - 0.5) * disp], b);
      }
      pts = next;
    }
    for (const p of pts) {
      const floor = baseFn(p[0]) - amp;
      if (p[1] < floor) p[1] = floor + rand() * 4;
    }
    return pts;
  }

  function ridgeFn(pts) {
    pts = pts.slice().sort((a, b) => a[0] - b[0]);
    return function (x) {
      for (let i = 0; i < pts.length - 1; i++) {
        if (x >= pts[i][0] && x <= pts[i + 1][0]) {
          const u = (x - pts[i][0]) / (pts[i + 1][0] - pts[i][0] || 1);
          return pts[i][1] + (pts[i + 1][1] - pts[i][1]) * u;
        }
      }
      return pts[pts.length - 1][1];
    };
  }

  function drawRange(opts) {
    const { xStart, xEnd, yBase, amp, roughness, stroke, sw, opacity, fillStep, fillOpacity } = opts;
    const baseFn = typeof yBase === "function" ? yBase : () => yBase;
    const pts = ridgePoints(xStart, xEnd, baseFn, amp, roughness);
    const fn = ridgeFn(pts);

    for (let x = xStart; x <= xEnd; x += fillStep) {
      const yTop = fn(x);
      const yBot = baseFn(x);
      if (yTop >= yBot) continue;
      const ln = document.createElementNS(ns, "line");
      ln.setAttribute("x1", x.toFixed(1)); ln.setAttribute("y1", (yBot + 1).toFixed(1));
      ln.setAttribute("x2", x.toFixed(1)); ln.setAttribute("y2", yTop.toFixed(1));
      ln.setAttribute("stroke", stroke);
      ln.setAttribute("stroke-width", sw * 0.7);
      ln.setAttribute("opacity", fillOpacity);
      mtnsG.appendChild(ln);
    }
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      const ln = document.createElementNS(ns, "line");
      ln.setAttribute("x1", a[0].toFixed(1)); ln.setAttribute("y1", a[1].toFixed(1));
      ln.setAttribute("x2", b[0].toFixed(1)); ln.setAttribute("y2", b[1].toFixed(1));
      ln.setAttribute("stroke", stroke);
      ln.setAttribute("stroke-width", sw);
      ln.setAttribute("opacity", opacity);
      mtnsG.appendChild(ln);
    }
  }

  seed = 11; drawRange({ xStart: -40, xEnd: 560, yBase: horizonY, amp: 26, roughness: 0.55, stroke: "oklch(58% 0.10 280)", sw: 0.8, opacity: 0.55, fillStep: 4, fillOpacity: 0.32 });
  seed = 19; drawRange({ xStart: 1040, xEnd: 1640, yBase: horizonY, amp: 26, roughness: 0.55, stroke: "oklch(58% 0.10 280)", sw: 0.8, opacity: 0.55, fillStep: 4, fillOpacity: 0.32 });
  seed = 29; drawRange({ xStart: -40, xEnd: 520, yBase: horizonY, amp: 38, roughness: 0.65, stroke: "oklch(66% 0.13 285)", sw: 0.9, opacity: 0.7, fillStep: 3, fillOpacity: 0.45 });
  seed = 47; drawRange({ xStart: 1080, xEnd: 1640, yBase: horizonY, amp: 38, roughness: 0.65, stroke: "oklch(66% 0.13 285)", sw: 0.9, opacity: 0.7, fillStep: 3, fillOpacity: 0.45 });
  seed = 71; drawRange({ xStart: -40, xEnd: 460, yBase: horizonY, amp: 54, roughness: 0.72, stroke: "oklch(74% 0.15 287)", sw: 1.1, opacity: 0.85, fillStep: 2, fillOpacity: 0.6 });
  seed = 97; drawRange({ xStart: 1140, xEnd: 1640, yBase: horizonY, amp: 54, roughness: 0.72, stroke: "oklch(74% 0.15 287)", sw: 1.1, opacity: 0.85, fillStep: 2, fillOpacity: 0.6 });

  // ---------- Stars ----------
  seed = 3;
  for (let i = 0; i < 110; i++) {
    const x = rand() * 1600;
    const y = rand() * 460 + 10;
    const r = rand() * 0.9 + 0.35;
    const o = rand() * 0.55 + 0.2;
    const c = document.createElementNS(ns, "circle");
    c.setAttribute("cx", x); c.setAttribute("cy", y);
    c.setAttribute("r", r); c.setAttribute("opacity", o);
    starsG.appendChild(c);
  }

  // ---------- Mobile framing ----------
  // On narrow screens the 1600x900 scene is recropped to a portrait window
  // and the black hole is moved into the visible sky, so the machine-code
  // grid and the black hole stay behind the rotating helix.
  const svg = groundG.ownerSVGElement;
  const bh = document.getElementById("bh");
  const mobile = window.matchMedia("(max-width: 880px)");
  function frameScene() {
    if (mobile.matches) {
      svg.setAttribute("viewBox", "440 40 720 860");
      if (bh) bh.setAttribute("transform", "translate(865 462) scale(0.62)");
    } else {
      svg.setAttribute("viewBox", "0 0 1600 900");
      if (bh) bh.setAttribute("transform", "translate(1240 110)");
    }
  }
  frameScene();
  if (mobile.addEventListener) mobile.addEventListener("change", frameScene);
})();

// ---------- Black hole: gravitationally lensed orbiting bodies ----------
// Bodies orbit in the accretion-disc plane. While a body passes BEHIND the
// hole, its light is bent around the photon sphere (point-mass lens
// approximation): the primary image is displaced outward toward the Einstein
// radius, stretched tangentially into an arc, and a fainter counter-image
// appears on the opposite side. In front, bodies pass undistorted.
(function () {
  const backG = document.getElementById("bh-back-bodies");
  const frontG = document.getElementById("bh-front-bodies");
  if (!backG || !frontG) return;
  const ns = "http://www.w3.org/2000/svg";

  const TILT = (-14 * Math.PI) / 180; // disc tilt (matches the SVG groups)
  const COS_T = Math.cos(TILT), SIN_T = Math.sin(TILT);
  const THETA_E = 46;   // Einstein radius, px (just outside the photon ring)
  const MAX_STRETCH = 3.2;

  const BODIES = [
    { a: 78,  b: 22, period: 14, r: 3.2, fill: "url(#bhPlanetA)",     phase: 0.0 },
    { a: 118, b: 34, period: 22, r: 2.6, fill: "url(#bhPlanetB)",     phase: 2.1 },
    { a: 158, b: 22, period: 34, r: 3.8, fill: "url(#bhPlanetC)",     phase: 4.2 },
    { a: 54,  b: 16, period: 9,  r: 0.9, fill: "oklch(96% 0.06 80)",  phase: 1.1 },
    { a: 100, b: 28, period: 18, r: 0.8, fill: "oklch(96% 0.04 220)", phase: 3.3 },
  ];

  function mkEllipse(fill, parent) {
    const e = document.createElementNS(ns, "ellipse");
    e.setAttribute("fill", fill);
    parent.appendChild(e);
    return e;
  }

  const nodes = BODIES.map((cfg) => ({
    cfg,
    img: mkEllipse(cfg.fill, frontG),     // primary image (reparented as needed)
    ghost: mkEllipse(cfg.fill, backG),    // lensed counter-image (back only)
  }));

  function place(t) {
    for (const { cfg, img, ghost } of nodes) {
      const phi = (t / cfg.period) * 2 * Math.PI + cfg.phase;
      // Disc-plane position (orbit ellipse already encodes the projection)
      const xd = cfg.a * Math.cos(phi);
      const yd = cfg.b * Math.sin(phi);
      // Rotate into the black hole's screen frame (disc is tilted -14°)
      const xs = xd * COS_T - yd * SIN_T;
      const ys = xd * SIN_T + yd * COS_T;
      const behind = yd < 0 ? Math.min(1, -yd / cfg.b) : 0; // 0 → in front/at limb

      if (behind === 0) {
        if (img.parentNode !== frontG) frontG.appendChild(img);
        img.setAttribute("cx", xs.toFixed(2));
        img.setAttribute("cy", ys.toFixed(2));
        img.setAttribute("rx", cfg.r);
        img.setAttribute("ry", cfg.r);
        img.removeAttribute("transform");
        img.setAttribute("opacity", "1");
        ghost.setAttribute("opacity", "0");
        continue;
      }

      // --- point-mass lens: beta -> theta_plus = (beta + sqrt(beta^2 + 4*thetaE^2)) / 2
      if (img.parentNode !== backG) backG.appendChild(img);
      const beta = Math.max(6, Math.hypot(xs, ys));
      const thetaP = 0.5 * (beta + Math.sqrt(beta * beta + 4 * THETA_E * THETA_E));
      const dApp = beta + behind * (thetaP - beta);   // blend in the deflection
      const ux = xs / beta, uy = ys / beta;
      const px = ux * dApp, py = uy * dApp;

      // tangential stretch (magnification) — the image smears into an arc
      const mu = 1 + behind * (Math.min(MAX_STRETCH, thetaP / beta) - 1);
      const ang = (Math.atan2(py, px) * 180) / Math.PI + 90; // tangent direction
      img.setAttribute("cx", "0");
      img.setAttribute("cy", "0");
      img.setAttribute("rx", (cfg.r * mu).toFixed(2));
      img.setAttribute("ry", (cfg.r / Math.sqrt(mu)).toFixed(2));
      img.setAttribute("transform", `translate(${px.toFixed(2)} ${py.toFixed(2)}) rotate(${ang.toFixed(1)})`);
      img.setAttribute("opacity", (0.8 + 0.2 * behind).toFixed(2));

      // counter-image: opposite side, inside the Einstein ring, demagnified
      const dGhost = Math.max(40, (THETA_E * THETA_E) / thetaP);
      const gAng = ang + 180;
      ghost.setAttribute("cx", "0");
      ghost.setAttribute("cy", "0");
      ghost.setAttribute("rx", (cfg.r * 0.8 * Math.min(2, mu)).toFixed(2));
      ghost.setAttribute("ry", (cfg.r * 0.5).toFixed(2));
      ghost.setAttribute("transform", `translate(${(-ux * dGhost).toFixed(2)} ${(-uy * dGhost).toFixed(2)}) rotate(${gAng.toFixed(1)})`);
      ghost.setAttribute("opacity", (0.45 * behind * behind).toFixed(2));
    }
  }

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced) {
    place(0); // static composition, no animation
    return;
  }

  let running = false;
  let rafId = 0;
  const t0 = performance.now();
  function tick(now) {
    if (!running) return;
    place((now - t0) / 1000);
    rafId = requestAnimationFrame(tick);
  }
  function setRunning(on) {
    if (on === running) return;
    running = on;
    if (on) rafId = requestAnimationFrame(tick);
    else cancelAnimationFrame(rafId);
  }

  const svg = backG.ownerSVGElement;
  if ("IntersectionObserver" in window && svg) {
    const io = new IntersectionObserver(
      (entries) => setRunning(entries.some((e) => e.isIntersecting)),
      { rootMargin: "60px" }
    );
    io.observe(svg);
  } else {
    setRunning(true);
  }
})();
