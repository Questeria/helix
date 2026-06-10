// @ts-check
import { defineConfig } from "astro/config";

// Production origin for absolute canonical / Open Graph URLs.
// Override at build time if needed:  SITE_URL=https://staging.example npm run build
const SITE_URL = process.env.SITE_URL || "https://299bytes.com";

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
