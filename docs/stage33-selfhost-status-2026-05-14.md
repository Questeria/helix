# Stage 33 Self-Host Status - 2026-05-14

Purpose: return from Stage 32 speed work to the central independence goal:
Helix should compile Helix, repeatedly and byte-identically, until the Python
compiler can become a historical reference instead of a required dependency.

## Baseline

Prior Stage 30 release-hardening evidence:

- `python scripts\selfhost_cascade.py --generations 10 --keep`
- Result: PASS
- G2..G11 stable SHA-256:
  `5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408`
- G2..G11 stable size: `277899` bytes
- Final-generation smoke cases: literal, call, and loop all returned `42`

Fresh Stage 33 baseline:

- `python scripts\selfhost_cascade.py --generations 3`
  - Result: PASS
  - G2..G4 stable SHA-256:
    `5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408`
  - G2..G4 stable size: `277899` bytes
  - Final-generation smoke cases: literal, call, and loop all returned `42`

## Slice 1 - Machine-Readable Cascade Reports

`scripts/selfhost_cascade.py` now accepts `--json-out <path>` and writes a
machine-readable cascade report with:

- schema id: `helix.selfhost_cascade.v0`
- seed compiler size and SHA-256
- every self-host generation's size and SHA-256
- stable/unstable decision
- stable hash and size when stable
- final-generation smoke results

Validation:

- `python -m pytest -q helixc\tests\test_selfhost_cascade.py`
  - Result: `3 passed`
- `python -m pytest -q helixc\tests\test_selfhost_cascade.py helixc\tests\test_stage32_select_tests.py`
  - Result: `14 passed`
- `python scripts\stage31_validate.py --mode focused --skip-snapshot scripts\selfhost_cascade.py helixc\tests\test_selfhost_cascade.py`
  - Result: `rc=0`
- `python scripts\selfhost_cascade.py --generations 3 --json-out .stage33-logs\selfhost-cascade-g3.json`
  - Result: PASS
  - JSON report confirms `stable: true`

## Completed Follow-Up

The planned stricter cascade gate is now in place through Slice 2 and Slice 3.

## Slice 2 - Cascade Report Validator

`scripts/selfhost_cascade_validate.py` validates a cascade report and fails
closed when:

- the schema is wrong
- `stable` is not true
- the stable hash or stable size is malformed
- any self-host generation drifts from the stable hash or size
- a generation exit low byte does not match the stable size low byte
- literal, call, or loop smoke evidence is missing or does not return `42`

The validator also supports `--expect-stable-sha` so a release gate can pin a
known compiler fixed point.

Validation:

- `python -m pytest -q helixc\tests\test_selfhost_cascade_validate.py helixc\tests\test_selfhost_cascade.py helixc\tests\test_stage32_select_tests.py`
  - Result: `21 passed`
- `python scripts\selfhost_cascade_validate.py .stage33-logs\selfhost-cascade-g3.json --min-generations 3 --expect-stable-sha 5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408`
  - Result: `selfhost-cascade-validate: ok`
- `python scripts\stage31_validate.py --mode focused --skip-snapshot scripts\selfhost_cascade_validate.py helixc\tests\test_selfhost_cascade_validate.py scripts\stage32_select_tests.py helixc\tests\test_stage32_select_tests.py`
  - Result: `rc=0`

## Slice 3 - One-Command Self-Host Gate

`scripts/stage33_selfhost_gate.py` runs the cascade and then validates the JSON
report in one command. This is the command future self-host parity work should
use before commit:

```powershell
python scripts\stage33_selfhost_gate.py --generations 3 --expect-stable-sha 5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408
```

The default report path is `.stage33-logs/selfhost-cascade-latest.json`, which
is ignored as generated evidence. The default prefix is the canonical
`/tmp/helix_cascade` prefix used by the Stage 30 baseline, because the driver
embeds its input/output paths and those path strings affect the stable binary
hash.

Validation:

- `python -m pytest -q helixc\tests\test_stage33_selfhost_gate.py helixc\tests\test_selfhost_cascade_validate.py helixc\tests\test_selfhost_cascade.py helixc\tests\test_stage32_select_tests.py`
  - Result: `26 passed`
- `python scripts\stage31_validate.py --mode focused --skip-snapshot scripts\stage33_selfhost_gate.py helixc\tests\test_stage33_selfhost_gate.py`
  - Result: `rc=0`
- `python scripts\stage33_selfhost_gate.py --generations 3 --expect-stable-sha 5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408`
  - Result: `rc=0`
  - G2..G4 stable SHA-256:
    `5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408`
  - Validator result: `selfhost-cascade-validate: ok`

## Slice 4 - Ten-Generation Release Gate

The Stage 33 one-command gate now has fresh 10-generation evidence matching
the Stage 30 fixed point. This directly exercises the longer cascade the user
requested before moving past the self-host proof layer.

Validation:

- `python scripts\stage33_selfhost_gate.py --generations 10 --expect-stable-sha 5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408 --json-out .stage33-logs\selfhost-cascade-g10.json`
  - Result: `rc=0`
  - G2..G11 stable SHA-256:
    `5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408`
  - G2..G11 stable size: `277899` bytes
  - Final-generation smoke cases: literal, call, and loop all returned `42`
  - Validator result: `selfhost-cascade-validate: ok`

## Slice 5 - Deprecated Message Metadata

The bootstrap parser now preserves `@deprecated("message")` string-literal
payloads instead of reducing every deprecated marker to a bare flag. Since the
bootstrap AST has no Python-style attribute list, the message body range is
stored on `AST_FN_DECL` slots 12/13:

- slot 12: deprecated message byte start
- slot 13: deprecated message byte length

The warning renderer still emits the existing deprecated-call diagnostic shape.
This slice only makes the metadata available in the Helix bootstrap compiler
without opening a broader diagnostics-rendering change.

Validation:

- `python -m pytest -q helixc\tests\test_codegen.py::test_bootstrap_kovc_deprecated_message_attr_preserved`
  - Result: `1 passed`
- `python -m pytest -q helixc\tests\test_deprecated.py helixc\tests\test_codegen.py::test_bootstrap_kovc_deprecated_pass_warning_does_not_trap helixc\tests\test_codegen.py::test_bootstrap_kovc_dep_tab_overflow_emits_28702 helixc\tests\test_codegen.py::test_bootstrap_kovc_deprecated_message_attr_preserved helixc\tests\test_stage33_selfhost_gate.py`
  - Result: `36 passed`
- `python -m pytest -q helixc\tests\test_parser.py helixc\tests\test_codegen.py::test_bootstrap_kovc_deprecated_pass_warning_does_not_trap helixc\tests\test_codegen.py::test_bootstrap_kovc_dep_tab_overflow_emits_28702 helixc\tests\test_codegen.py::test_bootstrap_kovc_deprecated_message_attr_preserved`
  - Result: `68 passed`
- `python scripts\stage33_selfhost_gate.py --generations 3 --json-out .stage33-logs\selfhost-cascade-deprecated-msg-g3.json`
  - Result: `rc=0`
  - G2..G4 stable SHA-256:
    `3da2bf2338eadc53e933246adc05698f10dcc001def71ac252dd58bb77421454`
  - G2..G4 stable size: `279185` bytes
  - Final-generation smoke cases: literal, call, and loop all returned `42`
  - Validator result: `selfhost-cascade-validate: ok`
- `python scripts\stage33_selfhost_gate.py --generations 10 --expect-stable-sha 3da2bf2338eadc53e933246adc05698f10dcc001def71ac252dd58bb77421454 --json-out .stage33-logs\selfhost-cascade-deprecated-msg-g10.json`
  - Result: `rc=0`
  - G2..G11 stable SHA-256:
    `3da2bf2338eadc53e933246adc05698f10dcc001def71ac252dd58bb77421454`
  - G2..G11 stable size: `279185` bytes
  - Final-generation smoke cases: literal, call, and loop all returned `42`
  - Validator result: `selfhost-cascade-validate: ok`

Focused validation note:

- `python scripts\stage31_validate.py --mode focused --skip-snapshot helixc\bootstrap\parser.hx helixc\bootstrap\kovc.hx helixc\tests\test_codegen.py`
  - Result: timed out after 15 minutes while selecting/running broad focused coverage.
  - Recovery: replaced with bounded direct parser/deprecated suites plus fresh
    3-generation and 10-generation Stage 33 self-host gates.

## Slice 6 - Autotune Validation Metadata

The bootstrap parser now captures summary metadata for `@kernel` and
`@autotune(...)` attributes on `AST_FN_DECL`:

- slot 14: `is_kernel`
- slot 15: `is_autotune`
- slot 16: deduped autotune variant product, saturated at `17`
- slot 17: autotune parse-error flag for malformed, empty, or missing params

`kovc.hx` now has a bootstrap-side `autotune_pass` that runs before codegen and
emits severity-2 diagnostics for:

- `27001`: variant product exceeds the Phase-0 cap of `16`
- `27002`: `@autotune` is present without `@kernel`
- `27003`: malformed/no-param/empty-list autotune args

Full variant generation and runtime dispatch remain out of scope for this
slice.

Validation:

- `python -m pytest -q helixc\tests\test_codegen.py::test_bootstrap_kovc_autotune_clean_metadata_at_cap helixc\tests\test_codegen.py::test_bootstrap_kovc_autotune_validation_diagnostics helixc\tests\test_codegen.py::test_bootstrap_kovc_autotune_error_traps_in_codegen`
  - Result: `3 passed`
- `python -m pytest -q helixc\tests\test_autotune.py helixc\tests\test_effect_check.py::test_c20_t2_autotune_attr_no_spurious_19002 helixc\tests\test_effect_check.py::test_c20_t2_combo_attrs_no_spurious_19002 helixc\tests\test_codegen.py::test_bootstrap_kovc_autotune_clean_metadata_at_cap helixc\tests\test_codegen.py::test_bootstrap_kovc_autotune_validation_diagnostics helixc\tests\test_codegen.py::test_bootstrap_kovc_autotune_error_traps_in_codegen helixc\tests\test_codegen.py::test_bootstrap_kovc_deprecated_message_attr_preserved`
  - Result: `30 passed`
- `python -m pytest -q helixc\tests\test_parser.py helixc\tests\test_autotune.py helixc\tests\test_codegen.py::test_bootstrap_kovc_autotune_clean_metadata_at_cap helixc\tests\test_codegen.py::test_bootstrap_kovc_autotune_validation_diagnostics helixc\tests\test_codegen.py::test_bootstrap_kovc_autotune_error_traps_in_codegen`
  - Result: `92 passed`
- `python scripts\stage33_selfhost_gate.py --generations 3 --json-out .stage33-logs\selfhost-cascade-autotune-g3.json`
  - Result: `rc=0`
  - G2..G4 stable SHA-256:
    `0ebfb630c34092fe06ab28ad4f1022f6f03666f4d0cbdfa9559cf8af3aafee81`
  - G2..G4 stable size: `285186` bytes
  - Final-generation smoke cases: literal, call, and loop all returned `42`
  - Validator result: `selfhost-cascade-validate: ok`
- `python scripts\stage33_selfhost_gate.py --generations 10 --expect-stable-sha 0ebfb630c34092fe06ab28ad4f1022f6f03666f4d0cbdfa9559cf8af3aafee81 --json-out .stage33-logs\selfhost-cascade-autotune-g10.json`
  - Result: `rc=0`
  - G2..G11 stable SHA-256:
    `0ebfb630c34092fe06ab28ad4f1022f6f03666f4d0cbdfa9559cf8af3aafee81`
  - G2..G11 stable size: `285186` bytes
  - Final-generation smoke cases: literal, call, and loop all returned `42`
  - Validator result: `selfhost-cascade-validate: ok`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: `rc=0`

## Next

The next Stage 33 slice should either thread preserved deprecated/autotune
metadata into richer diagnostic rendering, or take the next parser-safe
Python-only behavior from the Stage 33 candidate list. Full autotune variant
generation remains larger than the current slice size and should stay gated
behind another self-host proof.
