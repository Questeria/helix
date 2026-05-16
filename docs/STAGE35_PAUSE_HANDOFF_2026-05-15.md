# Helix Stage 35 Pause Handoff - 2026-05-15 restart 21

User requested a pause to restart computer.

Resume status: completed. The dirty restart-21 fix sweep described below was
finished after reboot, verified, recorded in the restart-21 audit document, and
is intended to be committed as "Fix Stage 35 twenty-first restart findings".
This file remains as the reboot continuity record.

Repo: C:\Projects\Kovostov-Native
Current branch: main
Last pushed clean commit before dirty work: c6273a9 Fix Stage 35 twentieth restart findings
Stage: 35 AI/ML Capability Push
Clean gates: 0/3
Restart 21 result so far: FAIL, in fix sweep, not committed.

Verified before pause:
- From c6273a9 restart 21 smoke passed: stdlib parser 16/16, Stage35 reverse AD smoke 8/8, 2D/accessor smoke 5/5, emit_ptx smoke 11/11, direct PTX smoke 20/20, docs stale scan clean under the narrower scan, py_compile check/ptx clean.
- Broader restart 21 verification passed: tensor/accessor 45, CLI 156, PTX 72, reverse AD 29, collection 2,258.
- After partial fix edits: py_compile helixc/frontend/autodiff.py and autodiff_reverse.py passed; parser passed for tensor.hx, nn.hx, autodiff_reverse.hx.

Audit findings to fix:
Lane A found four issues:
1. P2: 2D metadata enforced only by direct accessors; higher-level 2D/matrix/NN helpers trust caller rows/cols and can read/write adjacent allocations.
2. P2: rev_backward can partially mutate adjoints before discovering an older corrupt tape entry.
3. P2: __gelu is listed as AD-known but missing forward and reverse chain rules.
4. P3: BCE public losses are runtime-tested but not AD-compatible because __clamp remains opaque; bce_loss_scalar used min/max rather than one AD-known helper.
Lane C replacement found docs drift:
1. Stale 2,254 test count remains in README.md, QUICKSTART.md, helix_website/stats_and_facts.md, helix_website/HELIX_REFERENCE.md. Latest collection is 2,258.
2. HELIX_REFERENCE.md still has a 1000+ tests tree comment.
3. HELIX_REFERENCE.md/api_contracts.ts still have 120-byte/120 B visuals/data; actual hex0.bin is 299 bytes.
Lane B replacement was closed while still running; no PTX findings received.

Dirty partial edits at pause:
- helixc/stdlib/tensor.hx: added t2d_shape_ok and shape checks to many 2D helpers. Parser passed, but runtime tests not yet run after this edit.
- helixc/stdlib/nn.hx: added t2d_shape_ok checks to NN matrix helpers and changed bce_loss_scalar to call __bce. Parser passed, runtime tests not yet run.
- helixc/stdlib/autodiff_reverse.hx: added pre-validation pass in rev_backward before adjoint mutation. Parser passed, runtime tests not yet run.
- helixc/frontend/autodiff.py: partial forward AD additions for __bce and __gelu. py_compile passed. Reverse AD __gelu still NOT implemented yet. BCE chain rule tests still need to be added and run.

Next steps after reboot:
1. Inspect git status and diff; do not discard dirty edits.
2. Finish reverse-mode __gelu in helixc/frontend/autodiff_reverse.py.
3. Add tests:
   - tensor helper shape mismatch does not read guard: tf2d_row_sum/col_sum/matvec/matmul/transpose or at least a representative set.
   - rev_backward corrupt older op leaves all adjoints zero.
   - grad/grad_rev through __gelu at x=0 returns about 0.5 (multiply by 84 -> 42).
   - grad_rev through __bce at p=0.5, y=1 gives derivative about -2 (times -21 -> 42), and bce_loss_scalar remains runtime stable.
4. Fix docs drift from Lane C to 2,258 and 299 bytes.
5. Run focused parser/py_compile, targeted tests, then broad tensor/CLI/PTX/reverse AD/collection checks.
6. Write docs/audit-stage35-clean-gate1-twenty-first-restart-failed-2026-05-15.md and append Increment 40 to docs/stage35-progress-2026-05-15.md.
7. Commit as "Fix Stage 35 twenty-first restart findings", push, Telegram update, then start restart 22.
