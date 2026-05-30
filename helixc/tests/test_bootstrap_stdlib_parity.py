"""Bootstrap stdlib-parity corpus.

Mirrors test_bootstrap_autodiff_parity.py: a data-driven corpus of small
stdlib-using programs, each run through BOTH the Python reference compiler
(compile_and_run, which auto-includes the stdlib via include_stdlib=True)
AND the Helix-native bootstrap compiler, asserting behavioral parity.

The bootstrap (unlike Python) does NOT auto-include the stdlib, so the
needed Helix-source stdlib module(s) are PREPENDED to the bootstrap source
(MODULE_DEPS + _stdlib_prefix). The bootstrap CAN compile the stdlib
modules; this corpus measures that it compiles stdlib-USING programs to the
same result as Python. (The eventual "bootstrap auto-includes stdlib on its
own" — so production programs need no manual prefix — is a separate K3/
driver concern tracked in project_helix_status.)

Items move from KNOWN_PARITY_GAPS to passing as bootstrap stdlib-compilation
bugs are fixed. Programs return a plain i32 (no float-return / float-cast
fragility); many are designed to return 42.
"""

import os

import pytest

# Stdlib modules to PREPEND for each corpus module-label (the bootstrap has
# no auto-include + no fn-DCE, so a label's deps must all be present). The
# label is the corpus's first tuple field; the values are helixc/stdlib/*.hx
# basenames (sans extension), prepended in order.
MODULE_DEPS: dict[str, list[str]] = {
    "option": ["option"],
    "result": ["result"],
    "vec": ["vec"],
    "iter": ["iterators"],
    "hashmap": ["tensor", "hashmap"],   # hashmap ops call into tensor.hx
    "string": ["string"],
    "nn": ["tensor", "nn"],             # nn fns reference tensor.hx helpers
    "math": ["transcendentals"],        # math i32 helpers live in transcendentals.hx
}


# ============================================================================
# PARITY CORPUS  (module, name, src, expected_rc) — all independently verified
# 2026-05-30: BOOT == PY == expected, except the KNOWN_PARITY_GAPS below.
# ============================================================================
PARITY_CORPUS: list[tuple[str, str, str, int]] = [
    # ---- option (option.hx; enum Option { Some(i32), None }) ----
    ("option", "unwrap",
     "fn main() -> i32 { let o = Option::Some(42); option_unwrap_or(o, 0) }", 42),
    ("option", "max",
     "fn main() -> i32 { let a = Option::Some(3); let b = Option::Some(9); option_max(a, b) }", 9),
    ("option", "sum",
     "fn main() -> i32 { let a = Option::Some(7); let b = Option::None; option_sum(a, b) }", 7),

    # ---- result (result.hx; enum Result { Ok(i32), Err(i32) }) ----
    ("result", "unwrap",
     "fn main() -> i32 { let r = Result::Ok(42); result_unwrap_or(r, 0) }", 42),
    ("result", "err_code",
     "fn main() -> i32 { let r = Result::Err(7); result_err_code_or(r, 0) }", 7),
    ("result", "is_ok_false",
     "fn main() -> i32 { let r = Result::Err(5); result_is_ok(r) }", 0),

    # ---- vec (vec.hx; carry-pair Vec<i32>: arena start + threaded count) ----
    ("vec", "push_sum",
     "fn main() -> i32 { let s = vec_new(); let c0 = vec_push(s, 0, 5); "
     "let c1 = vec_push(s, c0, 7); let c2 = vec_push(s, c1, 30); vec_sum(s, c2) }", 42),
    ("vec", "max",
     "fn main() -> i32 { let s = vec_new(); let c0 = vec_push(s, 0, 12); "
     "let c1 = vec_push(s, c0, 30); let c2 = vec_push(s, c1, 7); vec_max(s, c2) }", 30),
    ("vec", "index_of",
     "fn main() -> i32 { let s = vec_new(); let c0 = vec_push(s, 0, 11); "
     "let c1 = vec_push(s, c0, 22); let c2 = vec_push(s, c1, 33); vec_index_of(s, c2, 33) }", 2),

    # ---- iterators (iterators.hx; range_to_vec + folds over arena vecs) ----
    ("iter", "fold_sum",
     "fn main() -> i32 { let s = range_to_vec(0, 5); vec_fold_op(s, 5, 0, 0) }", 10),
    ("iter", "dot",
     "fn main() -> i32 { let s = range_to_vec(1, 4); vec_dot(s, s, 3) }", 14),
    ("iter", "count_gt",
     "fn main() -> i32 { let s = range_to_vec(0, 10); vec_count_gt(s, 10, 6) }", 3),

    # ---- hashmap (hashmap.hx + tensor.hx) ----
    ("hashmap", "put_get",
     "fn main() -> i32 { let h = hashmap_new(8); hashmap_put(h, 8, 3, 42); hashmap_get(h, 8, 3, 0) }", 42),
    ("hashmap", "size",
     "fn main() -> i32 { let h = hashmap_new(8); hashmap_put(h, 8, 1, 10); "
     "hashmap_put(h, 8, 2, 20); hashmap_size(h, 8) }", 2),

    # ---- string (string.hx; carry-pair byte string, 1 byte/arena slot) ----
    ("string", "to_int",
     "fn main() -> i32 { let s = string_new(); let n1 = string_push(s, 0, 52); "
     "let n2 = string_push(s, n1, 50); string_to_int(s, n2) }", 42),
    ("string", "count_byte",
     "fn main() -> i32 { let s = string_new(); let n1 = string_push(s, 0, 65); "
     "let n2 = string_push(s, n1, 66); let n3 = string_push(s, n2, 65); "
     "string_count_byte(s, n3, 65) }", 2),

    # ---- nn (nn.hx + tensor.hx; pure scalar helpers) ----
    ("nn", "sgd_step_scalar",
     "fn main() -> i32 { sgd_step_scalar(50, 4, 2) }", 42),
    ("nn", "lin_reg_grad_b",
     "fn main() -> i32 { lin_reg_grad_b(3, 1, 2, 6) }", 2),

    # ---- math (transcendentals.hx; i32 helpers) ----
    ("math", "max_i32",
     "fn main() -> i32 { __max_i32(42, 7) }", 42),
    ("math", "clamp_i32",
     "fn main() -> i32 { __clamp_i32(100, 0, 42) }", 42),
    ("math", "abs_i32",
     "fn main() -> i32 { __abs_i32(0 - 42) }", 42),

    # ==== Expansion 2026-05-30 (harvested from test_codegen.py validated
    #      tests; each independently BOOT==PY==expected verified). ====
    ("option", "or_zero",
     "fn main() -> i32 { let a = Option::Some(42); let b = Option::None; option_or_zero(a) + option_or_zero(b) }", 42),
    ("option", "or_neg",
     "fn main() -> i32 { let a = Option::Some(43); let b = Option::None; option_or_neg(a) + option_or_neg(b) }", 42),
    ("option", "eq_some",
     "fn main() -> i32 { let a = Option::Some(42); let b = Option::Some(99); let c = Option::None; let n_match = option_eq_some(a, 42); let n_no = option_eq_some(b, 42); let n_none = option_eq_some(c, 42); n_match * 42 + n_no + n_none }", 42),
    ("option", "min",
     "fn main() -> i32 { let a = Option::Some(22); let b = Option::Some(20); let none = Option::None; let m1 = option_min(a, b); let m2 = option_min(a, none); m1 + m2 }", 42),
    ("result", "or_zero",
     "fn main() -> i32 { let a = Result::Ok(42); let b = Result::Err(99); result_or_zero(a) + result_or_zero(b) }", 42),
    ("result", "eq_ok",
     "fn main() -> i32 { let a = Result::Ok(42); let b = Result::Ok(99); let c = Result::Err(7); let n_match = result_eq_ok(a, 42); let n_no = result_eq_ok(b, 42); let n_err = result_eq_ok(c, 42); n_match * 42 + n_no + n_err }", 42),
    ("vec", "contains",
     "fn main() -> i32 { let s = vec_new(); let c0 = vec_push(s, 0, 11); let c1 = vec_push(s, c0, 22); let c2 = vec_push(s, c1, 33); let hit = vec_contains(s, c2, 22); let miss = vec_contains(s, c2, 99); (hit - miss) * 42 }", 42),
    # vec_abs_sum + vec_sum_squares TRAP (rc132) under the bootstrap while
    # Python returns 42 -- a real bootstrap codegen gap in these two vec fns
    # (vec_contains/sum/max/push/index_of all work). KNOWN_PARITY_GAPS below.
    ("vec", "abs_sum",
     "fn main() -> i32 { let v = vec_new(); let n0 = vec_push(v, 0, 5); let n1 = vec_push(v, n0, -3); let n2 = vec_push(v, n1, -8); let n3 = vec_push(v, n2, 1); vec_abs_sum(v, n3) * 2 + 8 }", 42),
    ("vec", "sum_squares",
     "fn main() -> i32 { let v = vec_new(); let n0 = vec_push(v, 0, 1); let n1 = vec_push(v, n0, 2); let n2 = vec_push(v, n1, 3); let n3 = vec_push(v, n2, 4); vec_sum_squares(v, n3) + 12 }", 42),
    ("string", "from_to_int_roundtrip",
     "fn main() -> i32 { let start = __arena_len(); let n = string_from_int(123); let v = string_to_int(start, n); if v == 123 { 42 } else { 1 } }", 42),
    ("string", "last_index_of",
     "fn main() -> i32 { let s = string_new(); let n0 = string_push(s, 0, 97); let n1 = string_push(s, n0, 98); let n2 = string_push(s, n1, 99); let n3 = string_push(s, n2, 98); let n4 = string_push(s, n3, 97); let last_b = string_last_index_of(s, n4, 98); let last_z = string_last_index_of(s, n4, 122); if last_b == 3 { if last_z == 0 - 1 { 42 } else { 1 } } else { 2 } }", 42),
    ("math", "min_max_clamp",
     "fn main() -> i32 { let a = __min_i32(5, 3); let b = __max_i32(5, 3); let c = __clamp_i32(100, 0, 10); a + b + c }", 18),
    ("math", "sign_i32",
     "fn main() -> i32 { let a = __sign_i32(7); let b = __sign_i32(0 - 5); let c = __sign_i32(0); a * 40 - b * 2 + c }", 42),
]


# ============================================================================
# KNOWN PARITY GAPS — bootstrap stdlib-compilation divergences (xfail).
# Remove entries as the underlying bootstrap bug is fixed.
# ============================================================================
KNOWN_PARITY_GAPS: set[tuple[str, str]] = {
    # vec_abs_sum / vec_sum_squares: bootstrap SIGILLs (rc132) while Python
    # returns the correct sum. The other vec fns (contains/sum/max/push/
    # index_of) all work, so it's a codegen gap specific to these two
    # (likely an op they use -- abs or i32 square/saturation -- that the
    # bootstrap mishandles in that context). Found 2026-05-30 corpus
    # expansion; root-cause + fix is a follow-up chunk.
    ("vec", "abs_sum"),
    ("vec", "sum_squares"),
    #
    # FIXED 2026-05-30: option/sum was the symptom of a deep nested-match
    # codegen bug. A nested match in ANY arm of a match re-init'd the SHARED
    # match_state region (kovc.hx single fail_state+end_table at bn_state+
    # 84..117), orphaning the enclosing match's already-recorded merge-jump
    # -> the parent arm's jmp kept its placeholder (jmp +0) and fell through
    # into the next arm, re-executing it (option_sum(Some,Some) -> the 2nd
    # payload; (Some,None) -> 0). Fixed in emit_one_match_arm by saving/
    # restoring the 34-slot match_state region across each arm body (it now
    # nests via the call stack); single/sequential matches unaffected. Was a
    # real CPU-language gap too (the 277-case corpus didn't cover it).
}


_STDLIB_CACHE: dict[str, str] = {}


def _stdlib_prefix(modules: list[str]) -> str:
    """Concatenate the given helixc/stdlib/<module>.hx sources (cached).

    The bootstrap does not auto-include the stdlib (Python does, via
    parse(include_stdlib=True)), so a stdlib-using program must carry the
    Helix-source definitions. Generalizes the autodiff harness's
    _autodiff_stdlib_prefix to an arbitrary module set.
    """
    helixc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = []
    for m in modules:
        if m not in _STDLIB_CACHE:
            with open(os.path.join(helixc_dir, "stdlib", m + ".hx"),
                      encoding="utf-8") as f:
                _STDLIB_CACHE[m] = f.read()
        parts.append(_STDLIB_CACHE[m])
    return "\n\n".join(parts)


@pytest.mark.parametrize(
    "module,name,src,expected_rc",
    PARITY_CORPUS,
    ids=[f"{c[0]}/{c[1]}" for c in PARITY_CORPUS],
)
def test_stdlib_parity(module: str, name: str, src: str, expected_rc: int):
    """Run one corpus entry through Python (sanity) + the bootstrap; assert
    behavioral parity. The bootstrap source gets the module's stdlib deps
    prepended (Python auto-includes the stdlib itself).

    Bootstrap retry: 3 retries on any mismatch (WSL cold-start flakes are
    transient; real gaps are deterministic).
    """
    from helixc.tests.test_codegen import (
        _kovc_self_host_compile_and_run as bootstrap_compile,
        compile_and_run as python_compile,
    )

    # Python sanity check: the corpus expected_rc must match the reference.
    python_rc = python_compile(src)
    assert python_rc == expected_rc, (
        f"[{module}/{name}] Python helixc rc={python_rc}, expected="
        f"{expected_rc}. Corpus expected_rc is wrong or Python regressed."
    )

    # Bootstrap check with the needed stdlib module(s) prepended.
    boot_src = _stdlib_prefix(MODULE_DEPS[module]) + "\n\n" + src
    bootstrap_rc = bootstrap_compile(f"stdlib_{module}_{name}", boot_src)
    tries = 0
    while bootstrap_rc != expected_rc and tries < 3:
        tries += 1
        bootstrap_rc = bootstrap_compile(f"stdlib_{module}_{name}_r{tries}", boot_src)

    if (module, name) in KNOWN_PARITY_GAPS and bootstrap_rc != expected_rc:
        pytest.xfail(
            f"known stdlib parity gap [{module}/{name}]: "
            f"expected={expected_rc}, bootstrap={bootstrap_rc}"
        )

    assert bootstrap_rc == expected_rc, (
        f"[{module}/{name}] STDLIB PARITY GAP: "
        f"expected={expected_rc}, bootstrap={bootstrap_rc}."
    )
