// Copies the repo's committed, self-contained demo artifacts (demo/*.html +
// demo/captured_run.js) into the built site at /demo/ so the "watch the real
// run" / "see the proof" links resolve on the deployed site — AND packs the
// same artifacts, plus an honest README, into dist/helix-demo.zip so visitors
// can download the whole demo and run it offline on their own machine.
//
// The demo pages are SEPARATE, dependency-free trust artifacts. They are
// copied verbatim — never rewritten, never folded into the site framework.
import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import AdmZip from "adm-zip";

const here = dirname(fileURLToPath(import.meta.url));
const repoDemo = join(here, "..", "..", "demo");
const outDemo = join(here, "..", "dist", "demo");
const distRoot = join(here, "..", "dist");

if (!existsSync(repoDemo)) {
  console.warn(
    "[copy-demo] ../demo not found — skipping (demo links will 404 in this build)."
  );
  process.exit(0);
}

// ---- 1. copy the artifacts verbatim into dist/demo/ (served at /demo/) ----
mkdirSync(outDemo, { recursive: true });
const artifacts = readdirSync(repoDemo).filter(
  (f) => f.endsWith(".html") || f.endsWith(".js")
);
for (const f of artifacts) cpSync(join(repoDemo, f), join(outDemo, f));
console.log(`[copy-demo] copied ${artifacts.length} demo artifact(s) verbatim into dist/demo/`);

// ---- 2. pack the SAME artifacts + an honest README into dist/helix-demo.zip ----
// The offline bundle has no AI backend, so it is honest about what it can do:
// the Guided Run REPLAYS a real recorded GPT-2-XL run (committed as data); a
// typed prompt with no backend is clearly badged PREVIEW (simulated). This
// README states that plainly and points to the live-on-your-own-GPU path.
const README = `HELIX x GPT-2  --  OFFLINE DEMO
===============================

What this is
------------
A self-contained copy of the Helix demo. Every file in this folder is plain
HTML and JavaScript: no build step, no installer, no network connection, no
tracking, no accounts. Open it, use it offline, and read the source if you
like -- that openness is the whole point of the project.

How to open
-----------
Double-click            landing.html            -- the "start here" page.
Then use the tabs across the top to move between the pages. Everything runs
straight from your file system.

If your browser blocks local files from loading each other, serve the folder
over a tiny local web server instead (any one of these):

    python -m http.server 8000
    npx serve .
    php -S localhost:8000

...then open  http://localhost:8000/landing.html

What is real here, and what is not
----------------------------------
This offline copy has NO AI backend bundled with it, so it cannot run a model
live. It is honest about that everywhere:

  REPLAY   The Guided Run (journey.html) replays a REAL, recorded run of
           GPT-2-XL (1.5 billion parameters) executing on Helix-compiled GPU
           kernels. The tokens, the per-layer trace, and the 25/25
           verification are the genuine recorded output, committed as data in
           captured_run.js. The badge reads "REPLAY".

  PREVIEW  If you type your own prompt with no backend running, the page
           animates the run's STRUCTURE using simulated numbers, clearly
           badged "PREVIEW -- simulated numbers". Those tokens are NOT from a
           real model. The demo is built so a mock can never be mistaken for a
           real run.

Run it live on your own GPU
---------------------------
To watch a model generate live -- on a toolchain compiled from a 299-byte
hand-typed seed, with no Python anywhere in it -- clone the public repository
and run one script. You will need an NVIDIA GPU, WSL2, and CUDA. Step-by-step
instructions:

    https://299bytes.com/run-local/

Source for everything in this bundle:
    https://github.com/Questeria/helix   (folder: demo/)

More
----
Website .......... https://299bytes.com
The trust chain .. https://299bytes.com/trust-chain/
Contact .......... ajdemarco10@gmail.com

This bundle contains no telemetry and phones home to nobody.
`;

const TOP = "helix-demo";
const zip = new AdmZip();
for (const f of artifacts) zip.addLocalFile(join(repoDemo, f), TOP);
zip.addFile(`${TOP}/README.txt`, Buffer.from(README, "utf8"));
zip.writeZip(join(distRoot, "helix-demo.zip"));
console.log(
  `[copy-demo] packed ${artifacts.length} artifact(s) + README into dist/helix-demo.zip (top folder: ${TOP}/)`
);
