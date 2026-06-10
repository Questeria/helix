// =====================================================================
// Bootstrap-chain interactive widget — the REAL nine-rung ladder.
// Every name, size, and SHA prefix below traces to committed files:
//   stage0/README.md (ladder status table), stage0/*/[name].bin,
//   stage0/helixc-bootstrap/seed.sha256, docs/TRUST_CHAIN_CLOSED.md.
// =====================================================================

const NODES = [
  {
    id: "hex0", name: "hex0", size: "299 B",
    tag: "01 / The trust root",
    title: "Hand-authored x86-64",
    desc: "299 hand-authored bytes (stage0/hex0/hex0.bin) — a raw ELF that reads hex characters on stdin and writes their byte values to stdout, skipping whitespace and comments. Every byte is annotated and auditable one at a time. This is the only program in the chain you must trust by reading.",
    hex: "; stage0/hex0/hex0.hex — annotated byte-by-byte source\n7F 45 4C 46 02 01 01 00  ; ELF magic, 64-bit, LSB\n...\n; 299 bytes total — sha256 cc1d1741…\n; frozen root: any change is a user-flag event",
  },
  {
    id: "hex1", name: "hex1", size: "622 B",
    tag: "02 / Adds labels",
    title: "Single-character labels",
    desc: "The first abstraction. hex1 understands single-character labels and relative references, so addresses no longer have to be counted by hand. Built by hex0 from annotated hex source; 622 bytes (sha256 c264a212…).",
    hex: "; built by hex0 → produces hex1\n; adds :a label definitions and\n; relative references to them\n:a\n  48 89 E5\n  E8 !a   ; call back-reference",
  },
  {
    id: "hex2", name: "hex2", size: "1,519 B",
    tag: "03 / Long labels + linker",
    title: "Labels and absolute addresses",
    desc: "hex2 adds long labels and absolute addresses — it acts as the chain's linker. Built by hex1; 1,519 bytes (sha256 6c69c7e6…).",
    hex: "; built by hex1 → produces hex2\n:_start\n  E8 %read_input   ; absolute address of\n                   ; a long-named label\n:read_input\n  ...",
  },
  {
    id: "catm", name: "catm", size: "299 B",
    tag: "04 / Concatenation",
    title: "File concatenation",
    desc: "catm OUT in1 in2 … — concatenates files, replacing any reliance on cat or shell redirection inside the build. Built by hex2; 299 bytes (sha256 911d19bf…).",
    hex: "; catm OUT in1 in2 ...\n; replaces `cat a b > out` so the\n; bootstrap never depends on the\n; shell for artifact assembly",
  },
  {
    id: "M0", name: "M0", size: "1,684 B",
    tag: "05 / Macro assembler",
    title: "Mnemonics get names",
    desc: "The first text-mode assembler: M1-syntax assembly (mnemonics, named registers, macros) down to hex2. Built by catm + hex2; 1,684 bytes (sha256 db97dff1…).",
    hex: "DEFINE push_rbp 55\nDEFINE mov_rbp,rsp 4889E5\n; M0 is the smallest program where\n; 'mov rbp, rsp' has a name.",
  },
  {
    id: "cc_amd64", name: "cc_amd64", size: "17,976 B",
    tag: "06 / Minimal C",
    title: "From assembly to C",
    desc: "A minimal C compiler — enough of a C subset to compile the next rung, emitting M1 assembly. Built by M0; 17,976 bytes (sha256 ea0054d1…).",
    hex: "/* cc_amd64: C subset → M1 asm */\nint main(int argc, char **argv) {\n    if (argc < 2) return 1;\n    return compile(argv[1]);\n}",
  },
  {
    id: "M2", name: "M2-Planet", size: "200,561 B",
    tag: "07 / Full C compiler",
    title: "The last vendored rung",
    desc: "A self-hosting C compiler, vendored at a pinned commit and built by cc_amd64 — the last third-party rung in the ladder. 200,561 bytes (sha256 724b9e2d…).",
    hex: "/* M2-Planet — vendored @ 761c2af5 */\n/* built by cc_amd64; compiles the\n   M2 C-subset the seed is written in */",
  },
  {
    id: "seed", name: "seed", size: "62,467 B",
    tag: "08 / The bridge to Helix",
    title: "The C-subset bootstrap compiler",
    desc: "seed.c — original work, written in the M2-Planet C subset. Built by M2-Planet, it compiles the Helix compiler sources. Re-derives byte-identically to the pinned sha256 9837db12…; an independent gcc build of the same step produces a byte-identical K1 (the diverse double-compile).",
    hex: "$ sha256sum seed.bin\n9837db12752a22159ca75a533910bc0d…\n\n; every reproduction deletes the\n; committed binary and rebuilds it\n; from the rung below",
  },
  {
    id: "kovc", name: "kovc", size: "698,392 B",
    tag: "09 / Self-hosted",
    title: "kovc, written in Helix",
    desc: "The Helix compiler — lexer, parser, and x86-64-ELF + PTX code generator, written in Helix (helixc/bootstrap/{lexer,parser,kovc}.hx). It compiles its own source: seed → K1 → K2 → K3 → K4 with K2 == K3 == K4 byte-identical (fixpoint 0992dddd…). The chain closes.",
    hex: "seed → K1 → K2 → K3 → K4\nK2 == K3 == K4   byte-identical\nfixpoint sha256  0992dddd…\n\nfn main(argc: i32, argv: **u8) -> i32 {\n    emit_elf(codegen(parse(lex(src))))\n}",
  },
];

const track = document.getElementById("chain-track");
const detail = document.getElementById("chain-detail");

if (track && detail) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

  function render(idx) {
    track
      .querySelectorAll(".chain-node")
      .forEach((n, i) => {
        n.classList.toggle("active", i === idx);
        n.setAttribute("aria-pressed", i === idx ? "true" : "false");
      });
    const n = NODES[idx];
    detail.innerHTML = `
      <div>
        <span class="chain-tag">${esc(n.tag)}</span>
        <h4>${esc(n.title)}</h4>
        <p>${esc(n.desc)}</p>
      </div>
      <pre class="chain-hex"><span class="hl">${esc(n.name)}</span> &nbsp; ${esc(n.size)}\n\n${esc(n.hex)}</pre>
    `;
  }

  NODES.forEach((n, i) => {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "chain-node" + (i === 0 ? " active" : "");
    el.setAttribute("aria-label", `Inspect rung ${i + 1}: ${n.name}`);
    el.innerHTML = `
      <div class="chain-dot"></div>
      <div class="chain-name">${esc(n.name)}</div>
      <div class="chain-size">${esc(n.size)}</div>
    `;
    el.addEventListener("click", () => render(i));
    el.addEventListener("mouseenter", () => render(i));
    el.addEventListener("focus", () => render(i));
    track.appendChild(el);
  });

  render(0);
}
