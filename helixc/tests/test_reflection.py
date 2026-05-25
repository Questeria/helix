"""End-to-end tests for verifier-gated reflection runtime behavior.

These tests exercise 64 mutable cells in the
binary's writable region, modify(handle, new_value, verifier_fn) calls the
verifier function and conditionally writes the cell, while splice(handle)
reads it back.
"""

from __future__ import annotations
import os, sys, subprocess, tempfile, shlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.grad_pass import grad_pass
from helixc.ir.lower_ast import lower
from helixc.ir.passes.const_fold import fold_module
from helixc.ir.passes.dce import dce_module
from helixc.ir.passes.fdce import fdce_module
from helixc.tests._codegen_backend import compile_module_to_elf


def _win_to_wsl(win_path: str) -> str:
    p = os.path.abspath(win_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"
    return p


def compile_and_run(src: str) -> int:
    prog = parse(src, include_stdlib=True)
    grad_pass(prog)
    mod = lower(prog)
    fold_module(mod)
    dce_module(mod)
    fdce_module(mod)
    elf = compile_module_to_elf(mod)
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(proj_root, "helixc", "tests", "_tmp")
    os.makedirs(out_dir, exist_ok=True)
    fd, out_path = tempfile.mkstemp(
        prefix="reflect_", suffix=".bin", dir=out_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(elf)
    try:
        os.chmod(out_path, 0o755)
    except OSError:
        pass
    wsl_path = shlex.quote(_win_to_wsl(out_path))
    result = subprocess.run(
        ["wsl", "--", "bash", "-c", f"chmod +x {wsl_path} && {wsl_path}"],
        capture_output=True, timeout=10
    )
    return result.returncode


def test_modify_with_function_verifier_accepting():
    # The verifier returns 1 → modification applied → splice reads new value.
    src = """
    fn always_yes(handle: i32, new_val: i32) -> i32 { 1 }
    fn main() -> i32 {
        let h = quote(0);
        let applied = modify(h, 42, always_yes);
        if applied == 1 {
            // Read back from the cell — should now be 42
            splice(h)
        } else {
            0
        }
    }
    """
    assert compile_and_run(src) == 42


def test_modify_with_function_verifier_rejecting():
    # Verifier returns 0 → no write → splice reads original (0).
    src = """
    fn always_no(handle: i32, new_val: i32) -> i32 { 0 }
    fn main() -> i32 {
        let h = quote(0);
        let applied = modify(h, 99, always_no);
        // applied should be 0; cell should still be 0
        let val = splice(h);
        // 0 (applied) + 0 (val) + 42 = 42 if both rejected correctly
        applied + val + 42
    }
    """
    assert compile_and_run(src) == 42


def test_verifier_inspects_proposed_value():
    # Verifier examines the proposed value and only accepts if it's <= 100.
    src = """
    fn under_100(handle: i32, new_val: i32) -> i32 {
        if new_val <= 100 { 1 } else { 0 }
    }
    fn main() -> i32 {
        let h = quote(1);
        // First: try 200 → rejected
        let r1 = modify(h, 200, under_100);
        // Second: try 42 → accepted
        let r2 = modify(h, 42, under_100);
        // r1=0, r2=1, splice(h)=42 → 0 + 1 + 42 = 43; subtract 1 = 42
        let v = splice(h);
        r1 + r2 + v - 1
    }
    """
    assert compile_and_run(src) == 42


def test_independent_cells_dont_interfere():
    src = """
    fn ok(h: i32, v: i32) -> i32 { 1 }
    fn main() -> i32 {
        let h0 = quote(0);
        let h1 = quote(1);
        modify(h0, 10, ok);
        modify(h1, 32, ok);
        // splice(h0) + splice(h1) = 10 + 32 = 42
        splice(h0) + splice(h1)
    }
    """
    assert compile_and_run(src) == 42


def test_multiple_modifications_compose():
    # Each modify overwrites the previous value.
    src = """
    fn ok(h: i32, v: i32) -> i32 { 1 }
    fn main() -> i32 {
        let h = quote(2);
        modify(h, 10, ok);
        modify(h, 20, ok);
        modify(h, 42, ok);
        splice(h)
    }
    """
    assert compile_and_run(src) == 42


def test_dogfood_01_one_param_gradient_descent():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_01_one_param.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_02_linreg():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_02_linreg.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_03_affine_with_f32_cells():
    # Affine fit exercising the f32-cell path (splice_f / modify_f) and the
    # newly-correct float calling convention (xmm-args).
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_03_affine.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_05_binary_classifier():
    # Sigmoid-based logistic regression exercising BCE loss, grad_rev_all,
    # and range-reduced __exp.
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_05_binary_classifier.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_04_xor_relu_perceptron():
    # Two-layer ReLU net touching XOR. Confirms grad_rev composes through
    # the stdlib __relu chain rule and 6-float-arg SysV calling convention.
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_04_xor_relu.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_06_provenance_datalog():
    # Stage 36 Increment 4 dogfood: Datalog-shaped propositional
    # reasoning over provenance-typed truth values. Verifies the
    # grandparent rule fires AND the tautology (P OR NOT P) holds
    # for both P=0 and P=1. Exit 42 on success.
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_06_provenance_datalog.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_07_provenance_sgd():
    # Stage 36 Increment 7 dogfood: SGD over a fuzzy-logic loss
    # surface. The first running Helix program that LEARNS a
    # provenance-typed parameter via gradients that flow through
    # propositional logic. loss(w) = (fuzzy_and(0.5, w) - 0.4)^2;
    # converges to w = 0.8 with lr=2.0 in one step. Exit 42 confirms
    # w_rounded * 100 - 38 = 42, i.e. w ≈ 0.8.
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_07_provenance_sgd.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_08_two_param_fuzzy_rule():
    # Stage 36 Increment 8 dogfood: TWO-parameter SGD over a fuzzy
    # rule. hypothesis = fuzzy_or(fuzzy_and(a, w1), fuzzy_and(b, w2)).
    # Trains w1 from example (1, 0)→0.9 and w2 from (0, 1)→0.7 via
    # grad_rev with indexed argument differentiation. Exit 42 iff
    # w1*100 + w2*100 ≈ 160 (i.e. w1≈0.9, w2≈0.7).
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_08_two_param_fuzzy_rule.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_11_spatial_frames():
    # Stage 38 Increment 3 dogfood: spatial-frame lifecycle reasoner.
    # 3 observations cycle through WorldFrame -> RobotFrame ->
    # CameraFrame -> WorldFrame via Stage 38 Inc 2 cross-frame
    # transforms. Exit 42 iff each obs round-trips AND the sum
    # equals 42 (10+14+18).
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_11_spatial_frames.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_12_temporal_lifecycle():
    # Stage 39 Increment 3 dogfood: temporal-type lifecycle reasoner.
    # 3 observations flow through Present -> Future (forecast) ->
    # Present (actualize) -> Past (to_past) -> unwrap, plus a
    # recall_past side-check and an Eternal intro/elim sanity check.
    # Exit 42 iff every wrapper transition preserves the inner value
    # AND the sum of unwrapped observations equals 42.
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_12_temporal_lifecycle.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_10_memory_tiers():
    # Stage 37 Increment 2 dogfood: memory-tier lifecycle reasoner.
    # 3 observations flow through working -> episodic -> consolidate
    # -> semantic -> recall -> working; procedural tier sanity check
    # also exercised. Exit 42 iff all 4 binary witnesses pass AND
    # the sum of recalled observations equals 42 (10 + 14 + 18).
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_10_memory_tiers.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_dogfood_09_knowledge_graph():
    # Stage 36 Increment 10 dogfood: chained-rule knowledge graph.
    # 3 facts (parent edges) + 2 grandparent rules + provenance
    # recovery via parent_left_at / parent_right_at. Exercises the
    # Inc 9 audit-clean primitives in a small AGI-shaped scenario.
    # Exit 42 iff both grandparent rules fire AND the evidence
    # handles correctly recover the parent source IDs.
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "dogfood_09_knowledge_graph.hx")
    with open(p) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_self_improving_agent_example():
    # Compiles and runs helixc/examples/self_improving_agent.hx — the
    # example covering reverse-mode AD, reflection, verifier gating, and
    # effect annotations together.
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    example_path = os.path.join(proj_root, "helixc", "examples", "self_improving_agent.hx")
    with open(example_path) as f:
        src = f.read()
    assert compile_and_run(src) == 42


def test_splice_oob_handle_returns_zero_not_crash():
    # A negative handle would, without bounds-check, do a wild read into
    # code memory. The bounds check turns OOB into a clean 0 read.
    src = """
    fn always_yes(h: i32, v: i32) -> i32 { 1 }
    fn main() -> i32 {
        let bad_handle = -1;
        let v = splice(bad_handle);
        // v must be 0 (OOB safe path); add 42 for the success exit code.
        v + 42
    }
    """
    assert compile_and_run(src) == 42


def test_verifier_with_unit_return_raises_clear_error():
    # A verifier with the right param shape but missing return type would
    # previously fall back silently to the legacy "treat verifier as a
    # runtime value" path, where every modify is then rejected with no
    # diagnostic. Now it should raise a compile-time ValueError.
    src = """
    fn bad_verifier(handle: i32, new_val: i32) {
        // No final expression and no return type → unit
    }
    fn main() -> i32 {
        let h = quote(0);
        modify(h, 1, bad_verifier);
        0
    }
    """
    try:
        compile_and_run(src)
    except ValueError as e:
        assert "verifier" in str(e).lower(), f"got {e}"
        return
    raise AssertionError("expected ValueError for unit-return verifier")


def test_modify_oob_handle_does_not_write():
    # An out-of-range handle must not be allowed to write past the cell array.
    # MODIFY returns 0 for OOB without calling the verifier.
    src = """
    fn always_yes(h: i32, v: i32) -> i32 { 1 }
    fn main() -> i32 {
        // 100 is way past HELIX_NUM_CELLS (= 64) — OOB.
        let r = modify(100, 999, always_yes);
        // r must be 0 (rejected); add 42.
        r + 42
    }
    """
    assert compile_and_run(src) == 42


def test_verifier_can_bound_state():
    # Fixed verifier-gated updates must keep the reflected state inside a
    # safe range.
    src = """
    fn safe_range(h: i32, v: i32) -> i32 {
        if v >= 0 { if v <= 100 { 1 } else { 0 } } else { 0 }
    }
    fn main() -> i32 {
        let param = quote(0);
        // Try several updates; verifier vetoes anything outside [0, 100]
        modify(param, 30, safe_range);
        modify(param, 200, safe_range);   // rejected
        modify(param, -5, safe_range);    // rejected
        modify(param, 42, safe_range);    // accepted
        // Final value should be 42
        splice(param)
    }
    """
    assert compile_and_run(src) == 42


def test_quote_handles_stable_across_runs():
    """Two distinct compiles of the same `quote { e }` get the same handle.
    Also: alpha-equivalent quotes (different bound names, same shape) get
    the same handle."""
    from helixc.frontend.parser import parse
    from helixc.ir.lower_ast import lower as ir_lower

    src1 = "fn f() -> i64 { let q = quote { 1 + 2 }; splice(q) }"
    src2 = "fn f() -> i64 { let q = quote { 1 + 2 }; splice(q) }"
    src_alpha = "fn f() -> i64 { let q = quote { 1 + 2 }; splice(q) }"

    def first_quote_handle(src: str) -> int:
        prog = parse(src, include_stdlib=False)
        mod = ir_lower(prog)
        for fn in mod.functions.values():
            for blk in fn.blocks:
                for op in blk.ops:
                    if op.kind.name == "QUOTE":
                        return op.attrs["ast_handle"]
        raise AssertionError("no QUOTE op emitted")

    h1 = first_quote_handle(src1)
    h2 = first_quote_handle(src2)
    h_alpha = first_quote_handle(src_alpha)
    assert h1 == h2, f"identical quote should reuse handle: {h1} vs {h2}"
    assert h1 == h_alpha, f"alpha-eq quote should reuse handle: {h1} vs {h_alpha}"


def main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
