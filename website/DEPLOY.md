# Deploying the Helix website

The site is an [Astro](https://astro.build) static build in `website/`. Output is plain
HTML/CSS/JS in `website/dist/` — any static host works. The build also copies the repo's
committed demo artifacts (`demo/*.html`, `demo/captured_run.js`) verbatim into `dist/demo/`
so the "watch the real run" / "see the proof" links resolve.

## Build locally

```bash
cd website
npm install
npm run build        # -> website/dist/
npm run preview      # serve dist/ locally to check it
```

Canonical / Open Graph URLs default to the production origin **https://299bytes.com**
(set in `astro.config.mjs`). Override only for a staging origin:

```bash
SITE_URL=https://staging.example npm run build
```

## Cloudflare Pages (recommended)

Create a Pages project connected to the GitHub repo with:

| Setting                | Value           |
| ---------------------- | --------------- |
| Production branch      | `main`          |
| Root directory         | `website`       |
| Build command          | `npm run build` |
| Build output directory | `dist`          |
| Environment variable   | none needed (canonical defaults to `https://299bytes.com`; set `SITE_URL` only to override) |

Then add **299bytes.com** as a Custom Domain on the Pages project (Pages → your project →
Custom domains). If the domain is in the same Cloudflare account, DNS is wired automatically.

Cloudflare clones the full repo, so the build's `../demo` copy step finds the demo
artifacts. Every push redeploys; preview deployments are created per-branch.

## GitHub Pages (alternative)

GitHub Pages serves project sites under `https://<user>.github.io/<repo>/`, which needs a
base path. Build with:

```bash
SITE_URL=https://questeria.github.io npx astro build --base /helix/ && node scripts/copy-demo.mjs
```

then publish `website/dist/` to the `gh-pages` branch (e.g. with `npx gh-pages -d dist`).
A custom domain (CNAME) avoids the base-path complication entirely — if you use one,
build with plain `npm run build` and `SITE_URL=https://<your-domain>`.

## Owner checklist before going live

- [x] Production domain **299bytes.com** is the default in `astro.config.mjs` (canonical + Open Graph resolve to it; verified in `dist/`).
- [ ] The nav/footer "demo" links point at the copied `/demo/` artifacts. If you would
      rather host the live chat demo elsewhere, update the links on `/verify/` and in
      `src/components/{Nav,Footer}.astro`.
- [ ] `public/og.png` is the social-share card — regenerate if the headline changes.
- [ ] Optional: add a sitemap (e.g. `@astrojs/sitemap`) once the final domain is fixed.

## What the site never does

The site is static marketing. It states that the models were verified (per the committed
gates) but never claims to run or prove anything live; the demo pages it links are
committed replays of real captured runs and label themselves as such.
