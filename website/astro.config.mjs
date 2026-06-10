// @ts-check
import { defineConfig } from "astro/config";

// Set SITE_URL at build time to your production origin so canonical / Open
// Graph URLs are absolute, e.g.:
//   SITE_URL=https://helix.example.com npm run build
// Defaults to the Cloudflare Pages preview origin placeholder.
const SITE_URL = process.env.SITE_URL || "https://helix-website.pages.dev";

export default defineConfig({
  site: SITE_URL,
  trailingSlash: "ignore",
  build: {
    format: "directory",
  },
  vite: {
    build: {
      // three.js is only loaded lazily on the home page; keep it in its own chunk.
      chunkSizeWarningLimit: 900,
    },
  },
});
