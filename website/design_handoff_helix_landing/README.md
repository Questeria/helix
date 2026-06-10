# Handoff: Helix Landing Page

## Overview
This is the marketing landing page for **Helix** — a from-scratch, ML-native systems language whose compiler bootstraps itself from 120 bytes of hand-encoded x86-64 machine code. The page sells three ideas: it's open by commitment, it's bootstrapped from raw binary, and autodiff/tiles/GPU codegen are language features rather than libraries. It is a single, long-scroll page culminating in a "try the playground" call to action.

The signature visual is a **3D double helix** (one carbon strand, one iridescent silicon strand) floating in a synthwave-style perspective room, rendered live in the hero.

## About the Design Files
The files in `design_files/` are **design references created in HTML/CSS/JS** — a working prototype that shows the intended look, motion, and behavior. They are **not** meant to be shipped verbatim as production code.

The task is to **recreate this design in your target codebase's environment** (React, Vue, Svelte, Astro, plain static site, etc.), using its established component patterns, styling system, and build tooling. If no environment exists yet, pick the framework most appropriate for a marketing site (a static-site generator like Astro or Next.js static export is a strong fit, since the page is mostly content with two self-contained interactive widgets) and implement the design there.

The one part that should be ported **as-is** is the Three.js 3D helix (`helix-logo.js`) — it's a self-contained ES module that mounts into a DOM node and is hard to meaningfully "rewrite." Treat it as a black-box widget you wrap in a component.

## Fidelity
**High-fidelity.** This is a pixel-considered design with final colors (OKLCH), typography (Geist + JetBrains Mono), spacing, motion, and interactions. Recreate the UI faithfully using your codebase's libraries and patterns. All exact values are documented below; the source files are the ground truth where this README is silent.

---

## Tech & Dependencies
- **Fonts** (Google Fonts): `Geist` (weights 300–700) for display & body; `JetBrains Mono` (400/500/600) for code, eyebrows, labels, and the byte ribbon.
- **Three.js** `0.160.1` (ES module via import map) — for the 3D helix. Also uses `RoomEnvironment` from three's examples for the environment map.
- No CSS framework, no build step in the prototype. All styling is hand-written CSS using custom properties.
- The page uses an **import map** to resolve `three`. In a bundled environment, install `three@0.160.1` from npm instead and import normally.

---

## Design Tokens

All colors are authored in **OKLCH**. Defined as CSS custom properties on `:root` in `styles.css`.

### Color — Light ("Wafer") theme (default / page base)
| Token | Value | Use |
|---|---|---|
| `--paper` | `oklch(98% 0.005 90)` | Page background (warm off-white) |
| `--paper-2` | `oklch(96% 0.006 90)` | Elevated surface |
| `--paper-3` | `oklch(93% 0.007 90)` | Deep surface |
| `--ink` | `oklch(15% 0.008 270)` | Primary text |
| `--ink-2` | `oklch(28% 0.008 270)` | Muted text |
| `--ink-3` | `oklch(45% 0.008 270)` | Soft text |
| `--hairline` | `oklch(15% 0.008 270 / 0.10)` | Hairline border |
| `--hairline-2` | `oklch(15% 0.008 270 / 0.16)` | Stronger border |

### Color — Dark ("Carbon") theme (applied per-section via `.dark` class)
| Token | Value | Use |
|---|---|---|
| `--carbon` | `oklch(10% 0.005 270)` | Dark section background |
| `--carbon-2` | `oklch(14% 0.006 270)` | Elevated dark surface |
| `--carbon-3` | `oklch(20% 0.007 270)` | Deep dark surface |
| `--wafer` | `oklch(98% 0 0)` | Text on dark |
| `--wafer-2` | `oklch(85% 0 0)` | Muted text on dark |
| `--wafer-3` | `oklch(65% 0 0)` | Soft text on dark |

### Color — Accent ("Silicon iridescence")
| Token | Value |
|---|---|
| `--violet` | `oklch(60% 0.18 290)` |
| `--violet-soft` | `oklch(78% 0.10 290)` |
| `--magenta` | `oklch(65% 0.20 340)` |
| `--cyan` | `oklch(72% 0.14 220)` |

The hero headline's "120 bytes" uses a gradient of `violet → magenta → cyan`. Selection highlight is `--violet`.

### Theming model
The page is **not** a global light/dark toggle. Instead, **alternating sections** flip to the dark theme by adding `class="dark"` to the `<section>`. The `.dark` rule re-maps the surface/text/line tokens to the carbon palette. Dark sections in order: Hero (01), Bootstrap chain (03), Code gallery (05), CTA (07). Light sections: Pillars (02), Autodiff (04), Numbers (06), Footer.

The sticky nav watches scroll and adds `.nav-dark` to itself whenever the section under it is dark (see "Interactions").

### Typography
- `--display` / `--body`: `"Geist", "Inter Tight", system-ui, sans-serif`
- `--mono`: `"JetBrains Mono", "IBM Plex Mono", ui-monospace, monospace`
- Body base: `17px / 1.55`, `font-feature-settings: "ss01","ss02","cv11"`, antialiased.
- Display headings use tight tracking (`-0.025em` to `-0.04em`) and `text-wrap: balance`.

Key type sizes (all clamp() fluid):
| Element | Size | Weight | Tracking | Line-height |
|---|---|---|---|---|
| Hero headline | `clamp(44px, 6.4vw, 92px)` | 500 | -0.035em | 0.96 |
| Section title | `clamp(32px, 3.6vw, 52px)` | 500 | -0.025em | 1.04 |
| Feature/CTA h2 | `clamp(32–40px, …, 52–80px)` | 500 | -0.025 to -0.035em | 1.0–1.04 |
| Number value | `clamp(48px, 6vw, 84px)` | 400 | -0.04em | 0.95 |
| Eyebrow | 12px mono, uppercase, `0.14em`, with a 6px violet dot `::before` |
| Body / lede | 15–20px | 400 | — | 1.5–1.55 |

### Spacing & Rhythm
| Token | Value |
|---|---|
| `--maxw` | `1240px` (content max width via `.wrap`) |
| `--gutter` | `clamp(20px, 4vw, 48px)` |
| `--section-y` | `clamp(80px, 12vw, 160px)` (vertical section padding) |

### Radii & Shadows
- Pills / buttons / nav-cta: `border-radius: 999px`
- Cards (pillars container, numbers grid): `20px`; code cards: `16px`; chain container: `24px`; small chips/insets: `6–12px`
- Card shadow (code cards): `0 30px 60px -30px color-mix(in oklch, var(--ink) 25%, transparent)` (pure black at -30px in dark)
- Nav: `backdrop-filter: blur(14px) saturate(140%)`, translucent bg via `color-mix`. On the dark hero it switches to a solid `oklch(4% 0.018 268)` with no blur.

### Motion
- Buttons: `transform: translateY(-1px)` on hover; arrow glyph slides `translateX(3px)`. Easing `cubic-bezier(.2,.8,.2,1)`, ~200ms.
- Reveal-on-scroll: elements with `.reveal` start at `opacity:0; translateY(16px)` and transition to visible (`.in`) over 700ms with `cubic-bezier(.2,.7,.2,1)`.
- The 3D helix auto-rotates and responds to pointer parallax.

---

## Screens / Sections

The page is one vertical scroll. Each `<section>` carries a `data-screen-label` (e.g. `01 Hero`). The shared layout primitive is `.wrap` (max-width 1240px, centered, gutter padding). Section headers use a flex `.section-head` row: left = eyebrow + `.section-title`; right = `.section-lede` paragraph (max ~38ch, muted).

### 0. Nav (sticky)
- Sticky top bar, height 64px, `z-index: 50`, hairline bottom border, blurred translucent background.
- Left: brand — a 22px inline-SVG helix mark (three stroked quadratic curves + two small circles) next to the wordmark "Helix" (Geist 600, 17px).
- Right: text links — "Why Helix", "Bootstrap", "Autodiff", "Code" — plus a pill CTA "Playground →" (`.nav-cta`, 1px border, rounds to `999px`).
- Adds `.nav-dark` class when overlapping a dark section (solid carbon bg, light text).

### 1. Hero — `.hero.dark` (`01 Hero`)
The most complex section. A dark, full-bleed synthwave scene with the 3D helix and the headline.

**Layout:** `.hero-grid` is a 2-col grid `1.1fr 0.9fr`, gap `clamp(32px,5vw,80px)`, vertically centered. Collapses to 1 column below 880px. Left = text; right = 3D helix slot.

**Left — `.hero-text`:**
- Eyebrow: `v0.7 · self-hosting · apache 2.0`
- Headline (`.hero-headline`): "A compiler that builds itself from **120 bytes** of hex." — "120 bytes" is wrapped in `<em>` rendered in JetBrains Mono with the violet→magenta→cyan gradient text clip.
- Sub (`.hero-sub`): one paragraph, max 36ch, near-white with a layered dark text-shadow so it stays legible over the gradient.
- Button row: primary pill "Try the playground →" (filled with `--fg`, inverts; hover turns violet) + ghost pill "See the bootstrap".

**Right — `.hero-3d`** (`#hero-3d`): aspect-ratio `1 / 1.5`, max-width 480px, transparent. The Three.js helix mounts here. **If WebGL fails, `landing.js` injects a `[ helix · 3D unavailable ]` mono fallback** — preserve a graceful fallback in your port.

**Background scene** — a full-bleed inline `<svg class="room" viewBox="0 0 1600 900">` behind the grid (`z-index: 0`), built partly in markup and partly generated by an inline `<script>`:
- A single full-canvas vertical gradient (`#backdrop`) going deep-night → bright violet at the horizon → deep-night, with a soft elliptical bloom (`#horizonBloom`).
- A small **animated black hole** in the upper-right (event horizon, radial accretion disc, and 5 orbiting bodies driven by SVG `<animateMotion>` along elliptical `<path>` orbits, split into front/back halves via clip paths so bodies appear to pass behind it).
- A **synthwave perspective floor grid**: ~240 converging vertical lines + 24 curved horizontal cross-bands (power-law spaced, gently bowed for an "Earth-curve" horizon), generated in JS.
- A **field of `hex0` bytes** (e.g. `48`, `89`, `C7`…) sitting at grid intersections in JetBrains Mono, each pulsing opacity (`@keyframes hexpulse`) with a per-tile animation-delay so a wave sweeps across the floor. Each byte is rotated/skewed to sit in the floor plane.
- Procedurally generated **mountain silhouettes** (fractal midpoint-displacement ridgelines with vertical-hatch interior fill) flanking left & right.
- A starfield (~110 dots) and a soft horizon glow arc.
- A scrolling **byte ribbon** at the bottom (`.byte-ribbon`): a marquee of Helix source-code fragments in mono, masked to fade at both edges, animating `translateX` over 220s, duplicated for a seamless loop. Content generated by `landing.js`.

> **Porting note:** The SVG scene is decorative and heavy. It's acceptable to port it faithfully (copy the SVG + its generator script into a component) or, if your perf budget is tight, to replace it with a static rendering / simplified version — but the violet-horizon synthwave mood and the 3D helix are essential to the brand and must be kept. There are leftover dev-only tuning panels referenced in CSS (`.tilt-panel`, `.rainbow-tuner`, `.per-byte-*`) and a hidden `.cyber-room` block — these are **not present in the live markup** and should **not** be ported. Ignore them.

### 2. Pillars — `#why` (`02 Pillars`, light)
- Section head: eyebrow "Three commitments", title "A foundation, not another framework.", lede on the right.
- `.pillars`: a 3-col grid with 1px gaps over a `--line` background (so the gaps read as hairline dividers), 1px border, `border-radius: 20px`, clipped. Collapses to 1 col < 880px.
- Each `.pillar` (min-height 320px, generous padding): a number tag top-right (`01/02/03`, mono), a 40px rounded icon tile (inline SVG, 20px stroked glyph), an h3 title, a paragraph, and a footer row of mono "chips" (bordered pills).
  1. **Open by commitment.** — Apache 2.0 / CC-BY 4.0 / CC0 chips.
  2. **Bootstrapped from binary.** — "120 B → 50 KB", "0 deps".
  3. **ML-first language design.** — "grad / grad_rev_all", "tile<f32, [N,N], REG>".
- Hover: pillar bg lifts to `--bg-elev`.

### 3. Bootstrap chain — `#bootstrap` (`03 Bootstrap chain`, dark)
An **interactive** widget. Section head: eyebrow "The bootstrap chain", title "Seven links. One byte you have to trust."
- `.chain` container (`--bg-elev`, 1px border, radius 24px). Inside: a horizontal connector line (`.chain-line`, a gradient with a violet midpoint), a 7-column track of nodes (`#chain-track`), and a detail panel (`#chain-detail`).
- The 7 nodes and their data are defined in `landing.js` (array `NODES`): `hex0` (120 B) → `hex1` (~700 B) → `M0` (~3 KB) → `M1` (~8 KB) → `M2-Planet` (~30 KB) → `kovc-bs` (~80 KB) → `kovc` (~50 KB). Each has a tag, title, description, and a `hex`/source snippet.
- Each node renders as a button: a 14px dot on the connector line, a mono name, and a mono size. Below the track, the detail panel shows the selected node's tag, title, description, and a `<pre class="chain-hex">` code/hex block (2-col on desktop).
- **Interaction:** hovering **or** clicking a node selects it (`.active`) and re-renders the detail panel. Node 0 (`hex0`) is active by default. Active/hover dot turns violet, scales 1.25, gets a soft ring glow.
- Track is 7-col on desktop, 2-col < 880px.

### 4. Autodiff feature — `#autodiff` (`04 Autodiff`, light)
- `.feature-grid`: 2-col `0.95fr 1.05fr`, centered, large gap. 1-col < 880px.
- **Left — code card** (`.code-card`): a faux editor window. Title bar (`.code-head`) with 3 traffic-light dots and a filename "autodiff.hx · grad". `<pre class="code-body">` with **hand-marked syntax highlighting** via spans (`.kw` keyword=violet, `.ty` type=cyan, `.num` number=magenta, `.fn` function, `.cmt` comment=soft italic). Footer (`.code-foot`) shows a command `kovc loss.hx -O2` and an output `→ 7.0_f64` (output in violet).
- **Right — `.feature-text`:** eyebrow "Autodiff, in the language", h2 "Gradients are a keyword, not a library.", a paragraph, and a `.feature-list` (4 items, each a `<b><code>` token + a description span, separated by hairline rules): `grad(f)(x)`, `grad_rev_all(f)(...)`, `@checkpoint`, `@kernel`.

### 5. Code gallery — `#code` (`05 Code gallery`, dark)
- Section head: eyebrow "In Helix, this compiles", title "A real systems language. With math notation."
- `.gallery`: 3-col grid of `.code-card`s (same editor-window styling as above, smaller code font 13px), 1-col < 880px. Three samples: `fib.hx` (→ 55), `matmul.hx` (→ 4.0_f32), `reflect.hx` (→ 42). Each card has a head (dots + filename), highlighted code body, and a foot (caption + violet output).

### 6. Numbers — `.numbers-section` (`06 Numbers`, light)
- Section head: eyebrow "By the numbers", title "Hard numbers, every one of them grounded."
- `.numbers`: 4-col grid (2-col < 880px), same 1px-gap-over-line hairline-divider treatment as pillars, radius 20px. Eight `.num-cell`s (min-height 180px), each: a big display number (`.num-val`) with a small mono superscript `.unit`, and an uppercase mono `.num-label` below. Values: `120 bytes`, `3,000+`, `23 disclosed`, `9 passes`, `0 deps`, `12 types`, `39 stages`, `~50 KB`.

### 7. CTA — `.cta.dark` (`07 CTA`, dark)
- Centered block: eyebrow "Carbon meets silicon", huge h2 "Compile something honest today." (`clamp(40px,5.6vw,80px)`, max 16ch), a paragraph, and a centered button row (primary "Open the playground →" + ghost "Read the spec").

### 8. Footer (light)
- `.foot-grid`: 4 columns `1.4fr repeat(3, 1fr)` (2-col < 880px). Col 1: brand mark + one-line description. Cols 2–4: link lists under headings "Language", "Build", "Project".
- `.foot-fineprint`: a top-bordered row, space-between, mono — "Apache 2.0 · CC-BY 4.0 · CC0 weights" and "kovc v0.7 · 2026-05-09".

---

## Interactions & Behavior (summary)
All interaction logic lives in `landing.js` (plus the hero's inline `<script>` for the SVG scene, and `helix-logo.js` for the 3D).

1. **Scroll-aware nav** — On scroll, finds the section currently under the nav (top ≤ 80px) and toggles `.nav-dark` to match its theme. Throttled with `requestAnimationFrame`.
2. **Reveal-on-scroll** — `IntersectionObserver` (threshold 0.12, `rootMargin: 0px 0px -40px 0px`) adds `.in` to `.reveal` elements once, then unobserves. Falls back to showing everything if `IntersectionObserver` is unavailable.
3. **Bootstrap chain** — Builds 7 node buttons from the `NODES` array; hover/click renders the detail panel; node 0 active on load.
4. **Byte ribbon** — Builds a randomized marquee string from `LINES` (Helix source fragments) + `TAGS`, duplicated for a seamless CSS loop.
5. **3D helix mount** — Dynamically `import("./helix-logo.js")` and calls `mountHelix(slot, { autoRotate: true })`. Stores theme/rainbow/sheen setters on `window`. Wraps in try/catch with the text fallback. (The `#sheen-tuner` panel it looks for is a dev tool not present in markup — that block is a no-op and can be dropped.)

### State to recreate
- Bootstrap chain: `selectedIndex` (0–6), default 0; updated on hover/click.
- Nav: `isDark` boolean derived from scroll position.
- Reveal: per-element "has appeared" (one-shot).
- These are trivial local component state in any framework — no global store or data fetching needed. The page is fully static content.

---

## The 3D Helix (`helix-logo.js`) — port as-is

A self-contained ES module exporting `mountHelix(container, opts)`. It builds a Three.js scene with:
- Two `TubeGeometry` strands along catmull-rom helix curves (1.5 turns, radius 1.2, height 6.6), 180° out of phase: **strand A = carbon** (matte dark `MeshPhysicalMaterial`), **strand B = silicon** (metallic, clearcoat, **iridescence** + a custom `onBeforeCompile` shader injection that paints seven world-fixed rainbow "spots" as glow).
- 13 **rungs**, each half carbon / half silicon, with a glowing seam ferrule + additive halo sprite that cycles hue over time.
- A metallic **pedestal** base with a pulsing violet glow ring.
- `RoomEnvironment` PMREM env-map (essential for the metal to reflect).
- Pointer parallax (helix spins on Y, root tilts on X), constant slow auto-rotation, `ResizeObserver` for responsive sizing.
- Returns a handle: `{ setTheme(isDark), setRainbow(v), setSheen(params), dispose() }`.

**Integration contract:** give it a sized container (`aspect-ratio: 1/1.5`, transparent bg). Call `mountHelix(el, { autoRotate: true })` after mount; call `handle.dispose()` on unmount to free GL resources (the prototype doesn't, but a component-based app should). Import `three@0.160.1` from your bundler instead of the CDN import map, and adjust the `RoomEnvironment` import path to `three/examples/jsm/environments/RoomEnvironment.js`.

---

## Assets
- **No raster image assets.** Everything is code: inline SVG, CSS, generated canvas textures (inside the Three.js module), and Google Fonts. There are no logos/photos to hand off — the helix mark is inline SVG and the hero centerpiece is the live 3D render.

## Files (in `design_files/`)
| File | What it is |
|---|---|
| `index.html` | Full page markup + the hero's inline SVG-scene generator script. Sections are clearly comment-delimited. |
| `styles.css` | All styling and tokens. **Note:** contains dead CSS for removed dev tuning panels (`.tilt-panel`, `.rainbow-tuner`, `.per-byte-*`, `.cyber-*`) — ignore/skip these when porting. |
| `landing.js` | Page interactions: scroll-nav, reveal observer, bootstrap-chain widget (incl. all node copy/data), byte ribbon, and the 3D mount. |
| `helix-logo.js` | Self-contained Three.js 3D double-helix module. Port as-is. |

## Recommended port order
1. Scaffold the page shell, tokens (port the OKLCH custom properties verbatim), fonts, and `.wrap`/section rhythm.
2. Build the static content sections top-to-bottom (Pillars → Autodiff → Code gallery → Numbers → CTA → Footer) — these are pure layout/typography.
3. Add the two interactive widgets: bootstrap chain and reveal-on-scroll.
4. Wrap `helix-logo.js` as a client-only component and mount it in the hero.
5. Port the hero SVG scene last (heaviest, most decorative); keep a non-WebGL fallback for the helix.
