"""
helixc/frontend/autotune.py — Stage 27: Triton-style autotune.

A `@autotune(BLOCK_SIZE: [16, 32, 64], NUM_WARPS: [4, 8])` attribute
on a `@kernel` fn declares a sweep over kernel-launch parameters. At
compile time, one kernel variant is generated per (BLOCK_SIZE,
NUM_WARPS) tuple in the cross product. The long-term design is runtime
measurement plus cached dispatch to the fastest variant; Phase-0 only
records and validates the static sweep specification.

Phase-0 scope:
  * Parser captures `@autotune(K: [v1, v2, ...])` via attrs as
    "autotune" + "autotune:K=v1,v2,...".
  * `parse_autotune_attrs(fn) -> (dict[str, list[int]], list[str])`
    reads the attrs back into a (param, values) dict AND collects
    diagnostics for malformed attrs (Audit 28.8 A12). The pre-fix
    `except ValueError: continue` silently dropped the entire key on
    a non-int value (e.g. `@autotune(BS: [16, "fast", 32])`) — fix
    surfaces the typo via diagnostics.
  * `autotune_variants(params) -> list[dict]` Cartesian-product the
    parameter dict into one config-dict per variant.
    Audit 28.8 A12: values per key are now dedup'd before product so
    `@autotune(X: [1, 1, 2])` doesn't emit two same-named variants.
  * Trap 27001 reserved: variant product > 16 (cap).
  * `validate_autotune(fn)` returns diagnostic strings on cap-violation,
    @autotune without @kernel, malformed attrs.

Runtime variant-dispatch is bootstrap-side (kovc.hx will emit a
dispatch table); this module provides the static spec.

License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional
from itertools import product

from . import ast_nodes as A


TRAP_AUTOTUNE_OVERSIZED = 27001
MAX_VARIANT_PRODUCT = 16


def parse_autotune_attrs(fn: A.FnDecl) -> tuple[dict[str, list[int]], list[str]]:
    """Decode the attrs list back into ({KEY: [v1, v2, ...]}, diags).

    Convention: parser emits "autotune" once, then one
    "autotune:KEY=v1,v2,..." per key.

    Audit 28.8 A12: pre-fix, malformed attrs (missing `=`, non-int
    values) were silently `continue`-d. Now each malformed attr
    produces a diagnostic string. Callers that don't care about the
    diagnostics can take the first element of the tuple."""
    out: dict[str, list[int]] = {}
    diags: list[str] = []
    for a in fn.attrs:
        if not a.startswith("autotune:"):
            continue
        body = a[len("autotune:"):]
        if "=" not in body:
            diags.append(
                f"@autotune on fn {fn.name!r}: malformed attr {a!r} — "
                f"expected `KEY=v1,v2,...` (got no `=`)"
            )
            continue
        key, vals_str = body.split("=", 1)
        if not key:
            diags.append(
                f"@autotune on fn {fn.name!r}: malformed attr {a!r} — "
                f"empty key before `=`"
            )
            continue
        raw_vals = [v for v in vals_str.split(",") if v]
        parsed: list[int] = []
        for v in raw_vals:
            try:
                parsed.append(int(v))
            except ValueError:
                diags.append(
                    f"@autotune on fn {fn.name!r}: parameter {key!r} "
                    f"value {v!r} is not an integer (must be a "
                    f"compile-time constant)"
                )
        # Audit 28.8 A12: dedup at the per-key level so [1, 1, 2] becomes
        # [1, 2] before Cartesian product (otherwise variant_count == 3
        # and two variants would mangle to the SAME name, with the
        # second registration silently overwriting the first).
        seen: set[int] = set()
        deduped: list[int] = []
        for v in parsed:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        out[key] = deduped
    return out, diags


# Backwards-compatible alias for callers that only want the dict and
# don't care about diagnostics. Old API was `dict` only; the dual-return
# version above is the new canonical form. Keep this thin wrapper so
# existing call sites don't break.
def parse_autotune_attrs_dict(fn: A.FnDecl) -> dict[str, list[int]]:
    """Legacy/convenience wrapper around parse_autotune_attrs that
    drops the diagnostic list. Prefer the tuple-returning version when
    surfacing user-visible errors."""
    params, _diags = parse_autotune_attrs(fn)
    return params


def autotune_variants(params: dict[str, list[int]]) -> list[dict[str, int]]:
    """Cartesian product of the parameter dict.

    Example:
      params = {"BLOCK_SIZE": [16, 32], "NUM_WARPS": [4, 8]}
      -> [
        {"BLOCK_SIZE": 16, "NUM_WARPS": 4},
        {"BLOCK_SIZE": 16, "NUM_WARPS": 8},
        {"BLOCK_SIZE": 32, "NUM_WARPS": 4},
        {"BLOCK_SIZE": 32, "NUM_WARPS": 8},
      ]

    Audit 28.8 A12: per-key dedup is applied so duplicate values
    in a list (typo like `[1, 1, 2]`) don't generate same-named
    variants."""
    if not params:
        return []
    keys = sorted(params.keys())
    # Per-key dedup preserves first occurrence (matches
    # parse_autotune_attrs's dedup contract).
    val_lists: list[list[int]] = []
    for k in keys:
        seen: set[int] = set()
        deduped: list[int] = []
        for v in params[k]:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        val_lists.append(deduped)
    out: list[dict[str, int]] = []
    for combo in product(*val_lists):
        out.append(dict(zip(keys, combo)))
    return out


def variant_count(params: dict[str, list[int]]) -> int:
    """Number of variants in the Cartesian product (post-dedup, per
    Audit 28.8 A12)."""
    if not params:
        return 0
    n = 1
    for k in params:
        # Match autotune_variants's dedup behavior so the count
        # reflects emitted variants, not raw attr values.
        n *= len(set(params[k]))
    return n


def has_autotune(fn: A.FnDecl) -> bool:
    return "autotune" in fn.attrs


def has_kernel(fn: A.FnDecl) -> bool:
    return "kernel" in fn.attrs


def mangled_variant_name(fn_name: str, cfg: dict[str, int]) -> str:
    """Construct a mangled variant fn name from the base name + config.

    Example: matmul + {BLOCK_SIZE:32, NUM_WARPS:4}
    -> 'matmul__autotune_BLOCK_SIZE_32_NUM_WARPS_4'
    """
    parts = [fn_name, "_autotune"]
    for k in sorted(cfg.keys()):
        parts.append(f"{k}_{cfg[k]}")
    return "_".join(parts)


def validate_autotune(fn: A.FnDecl) -> list[str]:
    """Sanity checks for @autotune. Returns diagnostic strings.

    Rules:
      * @autotune requires @kernel on the same fn
      * Variant cross-product capped at MAX_VARIANT_PRODUCT = 16
        (trap 27001 reservation)
      * Each parameter list must be non-empty
      * Each parameter value must parse as integer (Audit 28.8 A12)

    Audit 28.8 A12 forwarded fixes:
      * Diagnostic from `parse_autotune_attrs` (malformed attrs) flow
        into the returned list, so a `@autotune(B: [16, "fast"])`
        surfaces `value 'fast' is not an integer` instead of silently
        producing `no parameters parsed` (which used to mask the real
        cause).
    """
    diags: list[str] = []
    if not has_autotune(fn):
        return diags
    if not has_kernel(fn):
        diags.append(
            f"@autotune on fn {fn.name!r}: also requires @kernel "
            f"(autotune only applies to GPU kernel decls)"
        )
    params, parse_diags = parse_autotune_attrs(fn)
    diags.extend(parse_diags)
    if not params:
        diags.append(
            f"@autotune on fn {fn.name!r}: no parameters parsed "
            f"(expected `K: [v1, v2, ...]` form)"
        )
        return diags
    for k, vs in params.items():
        if not vs:
            diags.append(
                f"@autotune on fn {fn.name!r}: parameter {k!r} has "
                f"empty value list"
            )
    n = variant_count(params)
    if n > MAX_VARIANT_PRODUCT:
        diags.append(
            f"@autotune on fn {fn.name!r}: variant product {n} exceeds "
            f"Phase-0 cap of {MAX_VARIANT_PRODUCT} (trap 27001)"
        )
    return diags


def collect_autotuned_fns(prog: A.Program) -> list[A.FnDecl]:
    """All top-level fn decls with @autotune."""
    return [it for it in prog.items
            if isinstance(it, A.FnDecl) and has_autotune(it)]


def validate_autotune_prog(prog: A.Program) -> list[str]:
    """Audit 28.8 A12: program-level entry point so check.py can run
    `validate_autotune` over every autotuned fn at once. Returns the
    flat list of diagnostics across all such fns. Empty list means
    clean."""
    diags: list[str] = []
    for fn in collect_autotuned_fns(prog):
        diags.extend(validate_autotune(fn))
    return diags
