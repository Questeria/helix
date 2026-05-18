# Verifying Helix from a fresh Claude session

This document is a self-contained prompt you can paste into a **fresh
Claude conversation** (one with no Kovostov / Helix context) to verify
the Helix compiler end-to-end. Every check is mechanically observable.

Heavy emphasis on Stage 52 (modal-origin taint-tracking) since that
is the most recent surface and the closest to AI-safety semantics.

---

## Paste-this prompt

```
You're going to verify a programming language called Helix that lives at
https://github.com/Questeria/helix.git (also at C:/Projects/Kovostov-Native
on the local machine if available).

Helix is a Phase-0 compiler for an AGI-oriented language with semantic
types (Known/Believed/Goal/Uncertain modal, spatial frames, temporal
kinds, causal). Recent work shipped modal-origin taint-tracking — the
compiler should block any Uncertain→Known launder, including indirect
ones across let-bindings, match arms, if/else, while loops, name
aliases, PatBind, PatOr, Call-form scrutinees, and guard expressions.

Run these 4 checks in order. Report PASS/FAIL with diagnostic snippets.

============================================================
CHECK 1 — Self-host cascade (sanity)
============================================================
cd C:/Projects/Kovostov-Native
python scripts/selfhost_cascade.py

EXPECT: "cascade: PASS G2..G11 are byte-identical sha=a6f1ee44..." +
"smoke: PASS final generation compiled and ran all smoke programs" +
4 lines of "smoke <name>: exit=42".

============================================================
CHECK 2 — Stage 40 modal test sweep (the Stage 52 work)
============================================================
python -m pytest helixc/tests/test_stage40_modal.py --tb=line

EXPECT: 113+ tests, ALL pass. If any test_stage52_* or
test_stage53_* test fails, copy the assertion message.

============================================================
CHECK 3 — Adversarial AI-safety probe (write .hx + compile)
============================================================
Write this file to /tmp/launder_attacks.hx:

fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(7);

    // ATTACK 1: inline launder
    let k1: Known<i32> = into_known(from_uncertain(u));

    // ATTACK 2: let-binding bypass
    let r2: i32 = from_uncertain(u);
    let k2: Known<i32> = into_known(r2);

    // ATTACK 3: let-alias bypass (Stage 52 gate-6 CRITICAL-2)
    let r3: i32 = from_uncertain(u);
    let s3: i32 = r3;
    let k3: Known<i32> = into_known(s3);

    // ATTACK 4: match-arm PatBind (gate-4 HIGH-1)
    let r4: i32 = from_uncertain(u);
    let k4: i32 = match r4 { x => from_known(into_known(x)) };

    // ATTACK 5: Call-form match scrutinee (gate-6 CRITICAL-1)
    let k5: i32 = match from_uncertain(u) {
        x => from_known(into_known(x))
    };

    // ATTACK 6: if-branch install
    let mut r6: i32 = 0;
    if true { r6 = from_uncertain(u); };
    let k6: Known<i32> = into_known(r6);

    // ATTACK 7: while-body install
    let mut r7: i32 = 0;
    let mut i7: i32 = 0;
    while i7 < 1 { r7 = from_uncertain(u); i7 = i7 + 1; };
    let k7: Known<i32> = into_known(r7);

    // ATTACK 8: guard expression (gate-5 HIGH-1)
    let r8: i32 = from_uncertain(u);
    let k8: i32 = match r8 {
        x if from_known(into_known(x)) > 0 => 1,
        _ => 0
    };

    // ATTACK 9: PatOr-of-same-PatBind (gate-6 CRITICAL-3)
    let r9: i32 = from_uncertain(u);
    let k9: i32 = match r9 {
        x | x => from_known(into_known(x)),
        _ => 0
    };

    0
}

Run:
python -c "import sys; sys.path.insert(0, '.'); from helixc.frontend.parser import parse; from helixc.frontend.typecheck import typecheck; errs = typecheck(parse(open(r'/tmp/launder_attacks.hx').read(), include_stdlib=True)); launder = [e for e in errs if 'launder' in str(e)]; print(f'launder errors: {len(launder)}'); [print(' ', str(e)[:140]) for e in launder]"

EXPECT: AT LEAST 9 "launder" errors, each mentioning "Uncertain" and
"Known". If fewer than 9, identify which attack slipped through.

============================================================
CHECK 4 — Legitimate epistemic upgrade compiles clean
============================================================
Write to /tmp/legit.hx:

fn main() -> i32 {
    // Believed -> Known via the audited confirm()
    let b: Believed<i32> = into_believed(42);
    let k: Known<i32> = confirm(b);
    from_known(k)
}

Then:
python -c "import sys; sys.path.insert(0, '.'); from helixc.frontend.parser import parse; from helixc.frontend.typecheck import typecheck; errs = typecheck(parse(open(r'/tmp/legit.hx').read(), include_stdlib=True)); print(f'errors: {len(errs)}')"

EXPECT: 0 errors.

============================================================
REPORTING
============================================================
For each check, report PASS or FAIL plus a one-line summary. If
CHECK 3 finds fewer than 9 launders, that's a regression in the
AI-safety property — paste the exact attack patterns that slipped
through.
```

---

## What each check verifies

- **CHECK 1** — the self-host cascade verifies the compiler can compile
  itself, twice, byte-identical. If this breaks, the rest is suspect.
- **CHECK 2** — runs the 113+ modal-origin tests we've been hardening
  across gates 1-13 of Stage 52 + Stage 53 Inc 1+2. Every "ATTACK"
  in CHECK 3 has a
  corresponding regression pin here.
- **CHECK 3** — the AGI epistemic-safety property in action. Each
  attack is a known laundering pattern closed during Stages 40 / 52.
  If Claude reports `launder errors: 9`, the entire taint-tracking
  surface is working end-to-end.
- **CHECK 4** — verifies the legitimate path (`confirm()` is the
  audited Believed→Known upgrade). If it errors, we have a
  false-positive — even worse than a silent miscompile because it
  blocks real programs.

## Once Stage 53 lands

Add a CHECK 5 with a helper-function indirection:

```
fn launder(x: i32) -> Known<i32> { into_known(x) }
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let k: Known<i32> = launder(r);   // Stage 53 must catch this
    from_known(k)
}
```

Currently silent; Stage 53 (inter-procedural taint propagation via
`_fn_modal_return_kind` per the explorer blueprint) will close it.
