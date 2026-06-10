// =====================================================================
// Landing page interactions
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
  window.addEventListener("scroll", () => {
    if (rafId) return;
    rafId = requestAnimationFrame(update);
  }, { passive: true });
  update();
})();

// ---------- Reveal-on-scroll ----------
(function () {
  const els = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    els.forEach((el) => el.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add("in");
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
  els.forEach((el) => io.observe(el));
})();

// ---------- Bootstrap-chain interactive detail ----------
(function () {
  const NODES = [
    {
      id: "hex0", name: "hex0", size: "120 B",
      tag: "01 / The trust root",
      title: "Hand-encoded x86-64",
      desc: "120 bytes of raw machine code. Reads hex digits from stdin, writes their byte values to stdout. The only program in the chain you must trust.",
      hex: "31 C0 B8 03 00 00 00 BF 00 00 00 00 BE 00 10 60 00 BA 02 00 00 00 0F 05 48 89 C2 48 85 C0 7E 1A B8 01 00 00 00 BF 01 00 00 00 BE 00 10 60 00 0F 05 31 FF B8 3C 00 00 00 0F 05 ...",
    },
    {
      id: "hex1", name: "hex1", size: "~700 B",
      tag: "02 / Adds labels",
      title: "Symbolic references",
      desc: "First abstraction. Recognizes labels and back-references. Eliminates the need to count addresses by hand.",
      hex: "; defined by hex0 → produces hex1\n:start\n  48 89 E5\n  48 83 EC 40\n  E8 .resolve_label\n  48 89 C7\n  ...",
    },
    {
      id: "M0", name: "M0", size: "~3 KB",
      tag: "03 / Minimal assembler",
      title: "Macro assembler, v0",
      desc: "First text-mode assembler. Mnemonics, basic macros, fixed-width instructions. Compiled by hex1.",
      hex: "DEFINE PUSH_RBP    55\nDEFINE MOV_RBP_RSP 48 89 E5\nDEFINE SUB_RSP    48 83 EC\n; M0 is the smallest program where 'mov rbp, rsp' has a name.",
    },
    {
      id: "M1", name: "M1", size: "~8 KB",
      tag: "04 / Full assembler",
      title: "Macro assembler, complete",
      desc: "Conditional macros, directives, sections, relocations. The last assembler you'll need before C.",
      hex: ".section .text\n.global _start\n_start:\n    push  %rbp\n    mov   %rsp, %rbp\n    sub   $0x40, %rsp\n    call  read_input\n    ...",
    },
    {
      id: "M2", name: "M2-Planet", size: "~30 KB",
      tag: "05 / ANSI C subset",
      title: "From assembly to C",
      desc: "Tiny C compiler — enough ANSI C to compile the next link. Compiled by M1.",
      hex: "int main(int argc, char **argv) {\n    if (argc < 2) return 1;\n    return compile(argv[1]);\n}",
    },
    {
      id: "kovc-bs", name: "kovc-bs", size: "~80 KB",
      tag: "06 / Helix in C",
      title: "kovc-bootstrap",
      desc: "The Helix compiler, written in C. Lexer, parser, IR, x86-64 codegen — enough to compile the self-hosted Helix sources.",
      hex: "// kovc-bootstrap.c\n#include \"lex.h\"\n#include \"parse.h\"\n#include \"ir.h\"\n#include \"codegen_x86.h\"\nint main(int argc, char **argv) { ... }",
    },
    {
      id: "kovc", name: "kovc", size: "~50 KB",
      tag: "07 / Self-hosted",
      title: "kovc, in Helix",
      desc: "The compiler compiles itself. Same source, byte-identical output. The chain closes — and Helix is free of every dependency below it.",
      hex: "fn main(argc: i32, argv: *const *const u8) -> i32 {\n    let src = read_file(argv[1]);\n    let ast = parse(lex(src));\n    let ir  = lower(monomorphize(ast));\n    emit_elf(codegen_x86(ir))\n}",
    },
  ];

  const track  = document.getElementById("chain-track");
  const detail = document.getElementById("chain-detail");
  if (!track || !detail) return;

  function render(idx) {
    track.querySelectorAll(".chain-node").forEach((n, i) => n.classList.toggle("active", i === idx));
    const n = NODES[idx];
    detail.innerHTML = `
      <div>
        <span class="chain-tag">${n.tag}</span>
        <h4>${n.title}</h4>
        <p>${n.desc}</p>
      </div>
      <pre class="chain-hex"><span class="hl">${n.name}</span> &nbsp; ${n.size}\n\n${n.hex}</pre>
    `;
  }

  NODES.forEach((n, i) => {
    const el = document.createElement("button");
    el.className = "chain-node" + (i === 0 ? " active" : "");
    el.innerHTML = `
      <div class="chain-dot"></div>
      <div class="chain-name">${n.name}</div>
      <div class="chain-size">${n.size}</div>
    `;
    el.addEventListener("click", () => render(i));
    el.addEventListener("mouseenter", () => render(i));
    track.appendChild(el);
  });

  render(0);
})();

// ---------- Helix code ribbon — scrolling snippets of source ----------
(function () {
  const track = document.getElementById("byte-track");
  if (!track) return;
  // Real-looking fragments of Helix source. Mix of decls, exprs, attrs.
  const LINES = [
    "fn matmul(a: Tensor[f32], b: Tensor[f32]) -> Tensor[f32]",
    "let grad = backward(loss, [w1, w2, b])",
    "@checkpoint fn block(x) { norm(x) |> attn |> mlp }",
    "tile(x, [128, 64]) |> matmul(w) |> relu",
    "Quote { fn step(x: bf16) -> bf16 { x * lr } }",
    "Splice($body) into modify fn forward",
    "impl Layer for RMSNorm { fn forward(&self, x) { ... } }",
    "let logits = embed(tok) |> stack(blocks) |> unembed",
    "@grad let loss = cross_entropy(logits, target)",
    "kernel matmul_tiled<TILE=128>(a, b, out) { ... }",
    "for i in 0..steps { opt.step(grad); zero_grad() }",
    "module Transformer { layers: [Block; N_LAYERS] }",
    "let x = x + attn(rmsnorm(x))",
    "fn rope(q: Tensor, k: Tensor) -> (Tensor, Tensor)",
    "@inline fn softmax(x) { exp(x) / sum(exp(x)) }",
    "trait Optim { fn step(&mut self, p: &mut Param) }",
    "let model = Transformer::new(cfg).to(device)",
    "match dtype { F16 => ..., BF16 => ..., F32 => ... }",
  ];
  const TAGS = ["// helix", "// kernel", "// grad", "// jit", "// macro", "// trait", "// module"];
  let s = "";
  for (let i = 0; i < 60; i++) {
    if (i % 5 === 0) s += `<span class="b-tag">${TAGS[Math.floor(Math.random() * TAGS.length)]}</span>`;
    s += `<span>${LINES[Math.floor(Math.random() * LINES.length)]}</span>`;
  }
  // duplicate for seamless loop
  track.innerHTML = s + s;
})();

// ---------- Mount 3D helix ----------
(async function () {
  const slot = document.getElementById("hero-3d");
  if (!slot) return;

  try {
    const { mountHelix } = await import("./helix-logo.js");
    const handle = mountHelix(slot, { autoRotate: true });
    window.__helixSetTheme = handle.setTheme;
    handle.setTheme(document.documentElement.dataset.theme === "dark");
    window.__helixSetRainbow = handle.setRainbow;
    window.__helixSetSheen   = handle.setSheen;
    // Wire the multi-control sheen tuner panel.
    const tuner = document.getElementById("sheen-tuner");
    if (tuner) {
      const fmt = (k, v) => {
        if (k === "thicknessMin" || k === "thicknessMax" || k === "cycles") return String(Math.round(v));
        return v.toFixed(2);
      };
      const labels = tuner.querySelectorAll("label[data-k]");
      labels.forEach((lab) => {
        const k = lab.dataset.k;
        const inp = lab.querySelector("input[type=range]");
        const val = lab.querySelector(".v");
        const apply = () => {
          const v = parseFloat(inp.value);
          val.textContent = fmt(k, v);
          handle.setSheen({ [k]: v });
        };
        inp.addEventListener("input", apply);
      });
      // Push initial values from sliders into the material so the panel
      // and material are guaranteed in sync.
      const seed = {};
      labels.forEach((lab) => {
        seed[lab.dataset.k] = parseFloat(lab.querySelector("input").value);
      });
      handle.setSheen(seed);
      // Bake button — dumps current values as JSON that I can paste into
      // siliconMat defaults to make them permanent.
      const bakeBtn = document.getElementById("sheen-bake");
      const bakeOut = document.getElementById("sheen-bake-out");
      if (bakeBtn && bakeOut) {
        bakeBtn.addEventListener("click", () => {
          const out = {};
          tuner.querySelectorAll("label[data-k]").forEach((lab) => {
            out[lab.dataset.k] = parseFloat(lab.querySelector("input").value);
          });
          const json = JSON.stringify(out, null, 2);
          bakeOut.value = json;
          if (navigator.clipboard) navigator.clipboard.writeText(json).catch(() => {});
          console.log("SHEEN BAKED:", out);
        });
      }
    }
  } catch (err) {
    console.error("Helix logo failed to mount:", err);
    slot.innerHTML = '<div style="display:grid;place-items:center;height:100%;color:var(--fg-soft);font-family:var(--mono);font-size:12px;">[ helix · 3D unavailable ]</div>';
  }
})();
