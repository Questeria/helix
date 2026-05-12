# Audit Stage 28.9 cycle 98 — Silent failures

Scope: HEAD `1ff41ff` (Stage 28.9 cycle 97 fix-sweep landed in commit `3b065d2`).

## Audit verdict

**PASS — 0 findings at confidence ≥ 75%.**

## Cycle-97 fix verification

Both cycle-96 HIGH findings are verified in source at HEAD:

1. **C96-1 silent-failures fix (backend float-type classifier)** —
   `helixc/backend/x86_64.py:1000-1014` `_is_float_type` now includes
   `fp8`, `mxfp4`, `nvfp4`, `ternary` in addition to `f16`, `bf16`,
   `f32`, `f64`. `_check_float_supported` at lines 1033-1053 now also
   rejects the four quantized suffixes with the message *"x86_64
   backend supports only f32 and f64 currently; got '<name>'…"*. The
   check fires during `compile()` slot pre-allocation (lines 927, 931,
   935) for all block params, op results, and fn params — every
   typed-by-value path is fenced before any op-level emission. The
   silent integer-ABI fall-through documented in cycle-96 cannot reach
   `CONST_FLOAT`, `RETURN`, or float-arith arms.

2. **C96-1 type-design fix (A.Loop block orphan)** —
   `helixc/ir/lower_ast.py:1909-1910` both `header_blk` and `body_blk`
   are now constructed via `self.builder.append_block()` (matching the
   For/While arms at 1813/1873), so the blocks are appended to
   `current_fn.blocks` and are visible to slot pre-allocation, label
   emission, and BR-target validation in the x86_64 backend. The
   regression test `test_c96_loop_blocks_appended_to_fn_blocks` in
   `test_ir.py` asserts ≥3 blocks in `main` and every BR target lives
   in `fn.blocks`.

## Rotated fresh areas

### helixc/frontend/lexer.py — edge-case suffix tokenisation

`_lex_number` at lines 269-354 reviewed. Suffix-parsing block at
lines 328-347 correctly rewinds on unrecognized suffixes, on EOF
after the underscore, and on non-alpha after the underscore. The
recognized-suffix set at 338-341 matches `_FLOAT_PRIM_NAMES` /
typecheck. Exponent path (317-324) and `.<digit>` float trigger
(312) properly handle the gating conditions.

One latent class noted but BELOW threshold: hex literals followed by
`_<suffix-starting-with-hex-digit>` (`_b…`, `_f…`, `_a…`, etc.) get
consumed into the hex digit run because the hex-digit set
`"0123456789abcdef"` includes `b`/`f` and the loop also accepts
`_` (line 276). Example: `0x1_f32` lexes as `INT 7986` (`0x1f32`)
with no suffix, rather than `INT 1` with `f32` suffix. The same
applies to `0xff_b32`, `0xf_e8`, etc. However: (a) the only
recognized suffixes starting with hex digits are the float-domain
ones (`bf16`, `f16`, `f32`, `f64`) — semantically applying a float
suffix to a hex literal is implausible; (b) all the integer
suffixes (`i*`, `u*`, `isize`, `usize`) start with non-hex letters,
so `0xFF_u8` separates cleanly into `INT 255` + `KW_U8` (which the
parser then rejects with a clear *"expected ';' or '}'"* diagnostic).
Confidence that this is a meaningful silent failure in real code
≈ 50-60%, below the 75% bar. Flagged here for future cycles in case
a hex+float-suffix idiom enters the spec.

### helixc/ir/passes/fdce.py — FFI-only fn reachability

`fdce_module` enumerates call edges via `OpKind.CALL`,
`OpKind.MODIFY.verifier_fn`, and `OpKind.QUOTE.ast_pretty`
identifier scan, with roots `entry_fn`, `is_pub`, and `kernel`
attrs. `FFI_CALL` is NOT enumerated as a call edge, and `is_extern`
is NOT a root marker. Verified that this is benign: extern fn
entries exist in `module.functions` only for typecheck / call-site
resolution; their bodies are not emitted (`_lower_fn_body` at
`lower_ast.py:463` early-returns for `is_extern`); the backend
loop at `x86_64.py:3014` explicitly skips `is_extern` fns; and the
GOT-indirect call at the FFI_CALL site (line 1803,
`call_qword_ptr_rip_rel_ffi(target)`) registers the dynamic-linker
symbol directly via `self.b.dyn.add_import(symbol)` (line 487),
needing no surviving `module.functions[<extern>]` entry. So fdce
dropping an unreferenced extern fn declaration has no observable
effect on emitted ELF. Not a silent failure.

### helixc/backend/x86_64.py — CONST_FLOAT emission paths

`CONST_FLOAT` arm at lines 1186-1202 branches on `_is_f64_type`. For
f64, packs 8 bytes via two 32-bit moves (lo then hi at slot, slot+4).
For non-f64 (f32 with cycle-97 fences enforcing `_check_float_supported`
already ran), packs as 4 bytes via single 32-bit move. Byte order:
`struct.pack("<d", value)` and `struct.pack("<f", value)` — both
little-endian, matching x86_64 native byte order. Slot offsets:
`slot` and `slot+4` for f64 — `_alloc_slot` allocates 8-byte slots
for f64 (verified via `_is_f64_type` check elsewhere in the slot
allocator), so the `slot+4` write does not overflow. With cycle-97
fences in place, fp8/mxfp4/nvfp4/ternary cannot reach this arm. No
silent miscompile path observed.

## Summary

0 findings at confidence ≥ 75%.

PASS.

## No edits

This audit ran in strict read-only mode. No source files were edited.
Only this audit document was written.
