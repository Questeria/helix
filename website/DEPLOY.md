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

Set `SITE_URL` to your production origin so canonical / Open Graph URLs are absolute:

```bash
SITE_URL=https://your-domain.example npm run build
```

## Cloudflare Pages (recommended)

Create a Pages project connected to the GitHub repo with:

| Setting                | Value           |
| ---------------------- | --------------- |
| Production branch      | `main` (or `cowork/website` while reviewing) |
| Root directory         | `website`       |
| Build command          | `npm run build` |
| Build output directory | `dist`          |
| Environment variable   | `SITE_URL=https://<your-domain>` |

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

- [ ] Set the real domain via `SITE_URL` (canonical + Open Graph URLs depend on it).
- [ ] The nav/footer "demo" links point at the copied `/demo/` artifacts. If you would
      rather host the live chat demo elsewhere, update the links on `/verify/` and in
      `src/components/{Nav,Footer}.astro`.
- [ ] `public/og.png` is the social-share card — regenerate if the headline changes.
- [ ] Optional: add a sitemap (e.g. `@astrojs/sitemap`) once the final domain is fixed.

## What the site never does

The site is static marketing. It states that the models were verified (per the committed
gates) but never claims to run or prove anything live; the demo pages it links are
committed replays of real captured runs and label themselves as such.
