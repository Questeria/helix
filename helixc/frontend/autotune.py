"""
helixc/frontend/autotune.py — Stage 27: Triton-style autotune.

A `@autotune(BLOCK_SIZE: [16, 32, 64], NUM_WARPS: [4, 8])` attribute
on a `@kernel` fn declares a sweep over kernel-launch parameters. At
compile time, one kernel variant is generated per (BLOCK_SIZE,
NUM_WARPS) tuple in the cross product. At runtime, the first call
times each variant and records the fastest; subsequent calls jump to
the fastest variant.

Phase-0 scope:
  * Parser captures `@autotune(K: [v1, v2, ...])` via attrs as
    "autotune" + "autotune:K=v1,v2,...".
  * `parse_autotune_attrs(fn) -> dict[str, list[int]]` reads the
    attrs back into a (param, values) dict.
  * `autotune_variants(params) -> list[dict]` Cartesian-product the
    parameter dict into one config-dict per variant.
  * Trap 27001 reserved: variant product > 16 (cap).
  * `validate_autotune(fn)` returns diagnostic strings on cap-violation
    or @autotune without @kernel.

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


def parse_autotune_attrs(fn: A.FnDecl) -> dict[str, list[int]]:
    """Decode the attrs list back into {KEY: [v1, v2, ...]}.

    Convention: parser emits "autotune" once, then one
    "autotune:KEY=v1,v2,..." per key."""
    out: dict[str, list[int]] = {}
    for a in fn.attrs:
        if a.startswith("autotune:"):
            body = a[len("autotune:"):]
            if "=" not in body:
                continue
            key, vals_str = body.split("=", 1)
            try:
                out[key] = [int(v) for v in vals_str.split(",") if v]
            except ValueError:
                continue
    return out


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
    """
    if not params:
        return []
    keys = list(params.keys())
    val_lists = [params[k] for k in keys]
    out: list[dict[str, int]] = []
    for combo in product(*val_lists):
        out.append(dict(zip(keys, combo)))
    return out


def variant_count(params: dict[str, list[int]]) -> int:
    """Number of variants in the Cartesian product."""
    if not params:
        return 0
    n = 1
    for k in params:
        n *= len(params[k])
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
    """
    diags: list[str] = []
    if not has_autotune(fn):
        return diags
    if not has_kernel(fn):
        diags.append(
            f"@autotune on fn {fn.name!r}: also requires @kernel "
            f"(autotune only applies to GPU kernel decls)"
        )
    params = parse_autotune_attrs(fn)
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
