"""End-to-end tests for verifier-gated self-modification.

These tests exercise the real reflection runtime: 64 mutable cells in the
binary's writable region, modify(handle, new_value, verifier_fn) calls the
verifier function and conditionally writes the cell, splice(handle) reads
it back. This is the unique-to-Helix AGI primitive — no other AI language
exposes verifier-gated state mutation as a first-class feature.
"""

from __future__ import annotations
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.grad_pass import grad_pass
from helixc.ir.lower_ast import lower
from helixc.ir.passes.const_fold import fold_module
from helixc.ir.passes.dce import dce_module
from helixc.ir.passes.fdce import fdce_module
from helixc.backend.x86_64 import compile_module_to_elf


def compile_and_run(src: str) -> int:
    prog = parse(src)
    grad_pass(prog)
    mod = lower(prog)
    fold_module(mod)
    dce_module(mod)
    fdce_module(mod)
    elf = compile_module_to_elf(mod)
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(proj_root, "helixc", "tests", "_tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reflect.bin")
    with open(out_path, "wb") as f:
        f.write(elf)
    try:
        os.chmod(out_path, 0o755)
    except OSError:
        pass
    rel = os.path.relpath(out_path, proj_root).replace("\\", "/")
    wsl_path = f"/mnt/c/Projects/Kovostov-Native/{rel}"
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


def test_flagship_self_improving_agent_example():
    # Compiles and runs helixc/examples/self_improving_agent.hx — the
    # flagship demo composing reverse-mode AD + reflection + verifier
    # gating + effect annotations all in one program.
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
    # An agent learns by gradient descent; verifier ensures the state
    # never exceeds a safe range. This is the AGI demo in miniature.
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
