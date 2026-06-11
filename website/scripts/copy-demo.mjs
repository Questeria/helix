// Copies the repo's committed, self-contained demo artifacts (demo/*.html +
// demo/captured_run.js) into the built site at /demo/ so the "watch the real
// run" / "see the proof" links resolve on the deployed site.
//
// The demo pages are SEPARATE, dependency-free trust artifacts. They are
// copied verbatim — never rewritten, never folded into the site framework.
import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoDemo = join(here, "..", "..", "demo");
const outDemo = join(here, "..", "dist", "demo");

if (!existsSync(repoDemo)) {
  console.warn(
    "[copy-demo] ../demo not found — skipping (demo links will 404 in this build)."
  );
  process.exit(0);
}

mkdirSync(outDemo, { recursive: true });
let n = 0;
for (const f of readdirSync(repoDemo)) {
  if (f.endsWith(".html") || f.endsWith(".js")) {
    cpSync(join(repoDemo, f), join(outDemo, f));
    n++;
  }
}
console.log(`[copy-demo] copied ${n} demo artifact(s) verbatim into dist/demo/`);
