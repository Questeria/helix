"""Gate-2 audit probes — temporary."""
import os
import subprocess
import tempfile

from helixc.frontend.parser import parse
from helixc.frontend.typecheck import typecheck
from helixc.ir.lower_ast import lower
from helixc.backend.x86_64 import compile_module_to_elf


def run_src(name, src):
    print(f"=== {name} ===")
    try:
        prog = parse(src, include_stdlib=True)
    except Exception as e:
        print(" parse exception:", type(e).__name__, e)
        return
    errs = typecheck(prog)
    if errs:
        print(" typecheck errors:")
        for e in errs[:5]:
            print("   ", e)
        return
    try:
        elf = compile_module_to_elf(lower(prog))
    except Exception as e:
        print(" lower/codegen exception:", type(e).__name__, e)
        return
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(elf)
        p = f.name
    os.chmod(p, 0o755)
    abs_p = p.replace("\\", "/").replace("C:", "/mnt/c")
    r = subprocess.run(
        ["wsl", "--", "bash", "-c", f"chmod +x {abs_p} && {abs_p}"],
        capture_output=True, timeout=30,
    )
    print(" rc=", r.returncode)
    err = r.stderr.decode("utf-8", errors="replace").strip()
    if err:
        print(" stderr=", err[:300])
    try:
        os.unlink(p)
    except OSError:
        pass


# Q-INC15: `?` inside unwrap_ok(...). Stage 49 Inc 4 documented `?` only
# in top-level statement positions. What about inside another call?
run_src("Q-INC15: unwrap_ok(safe_div(10,0)?) typecheck behavior", """
fn safe_div(a: i32, b: i32) -> Result<i32, i32> {
    if b == 0 { Err(1) } else { Ok(a / b) }
}
fn driver() -> Result<i32, i32> {
    let v = unwrap_ok(safe_div(10, 0)?);
    Ok(v)
}
fn main() -> i32 {
    let r = driver();
    if is_err(r) { 50 } else { unwrap_ok(r) }
}
""")

# COMPOSE-1: let v = make_err()?; ... — `?` early-returns the Err.
run_src("COMPOSE-1: make_err()? early-returns", """
fn make_err() -> Result<i32, i32> { Err(7) }
fn driver() -> Result<i32, i32> {
    let v = make_err()?;
    Ok(v + 100)
}
fn main() -> i32 {
    let r = driver();
    if is_err(r) { 90 } else { unwrap_ok(r) }
}
""")

# MAP-1: unwrap_ok(map_ok(Err, 99)) — must panic (Err passes through).
run_src("MAP-1: unwrap_ok(map_ok(Err, 99)) panics", """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(1);
    unwrap_ok(map_ok(r, 99))
}
""")

# IS-1: is_ok(make_err()) returns false (no panic).
run_src("IS-1: is_ok(make_err()) returns 0", """
fn make_err() -> Result<i32, i32> { Err(99) }
fn main() -> i32 {
    if is_ok(make_err()) { 1 } else { 0 }
}
""")

# NEG-PAYLOAD: unwrap_ok(Err(-1)) — does panic message report wrong-arm?
run_src("NEG-PAYLOAD: unwrap_ok on Err(-1)", """
fn make_err() -> Result<i32, i32> { Err(-1) }
fn main() -> i32 {
    unwrap_ok(make_err())
}
""")

# RECUR-1: recursive Result fn with `?`.
run_src("RECUR-1: recursive Result with `?`", """
fn count_down(n: i32) -> Result<i32, i32> {
    if n <= 0 {
        Ok(0)
    } else {
        let prev = count_down(n - 1)?;
        Ok(prev + 1)
    }
}
fn main() -> i32 {
    let r = count_down(20);
    unwrap_ok(r)
}
""")

# F32-PAYLOAD: Result<f32,i32> + unwrap_ok.
run_src("F32-PAYLOAD: Result<f32,i32> unwrap_ok", """
fn make_ok() -> Result<f32, i32> {
    let v: f32 = 3.14;
    Ok(v)
}
fn main() -> i32 {
    let v: f32 = unwrap_ok(make_ok());
    v as i32
}
""")

# TWO-UNWRAP: two unwrap_ok of same Result name in one expression.
run_src("TWO-UNWRAP: unwrap_ok(r) + unwrap_ok(r)", """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(21);
    unwrap_ok(r) + unwrap_ok(r)
}
""")

# MUL-Q: two `?` in one expression.
run_src("MUL-Q: two `?` in one expression", """
fn make_ok() -> Result<i32, i32> { Ok(11) }
fn driver() -> Result<i32, i32> {
    Ok(make_ok()? + make_ok()?)
}
fn main() -> i32 {
    let r = driver();
    unwrap_ok(r)
}
""")

# Q-CASCADE: chained `?` early-return cascade.
run_src("Q-CASCADE: chained `?`", """
fn make_err() -> Result<i32, i32> { Err(42) }
fn inner() -> Result<i32, i32> {
    let v = make_err()?;
    Ok(v + 1)
}
fn outer() -> Result<i32, i32> {
    let v = inner()?;
    Ok(v * 2)
}
fn main() -> i32 {
    let r = outer();
    if is_err(r) { 7 } else { unwrap_ok(r) }
}
""")

# UNWRAP-CALL: unwrap_ok on a Call (not a Name) — make sure Inc 1.5
# wraps Call operands correctly (gate-1 F2 root cause was Call operands
# bypassing static-provenance — now Inc 1.5 protects them at runtime).
run_src("UNWRAP-CALL: unwrap_ok(make_err()) ok-mismatch", """
fn make_err() -> Result<i32, i32> { Err(13) }
fn main() -> i32 {
    unwrap_ok(make_err())
}
""")

# OK-WITH-OPT: confirm const-fold doesn't mistakenly fold the tag-check.
# RESULT_TAG / RESULT_PACK are NOT in const_fold's foldable set, so even
# `unwrap_ok(Ok(7))` should still emit the runtime branch.
run_src("OPT-NO-FOLD: -O optimization, unwrap_ok(Ok(7))", """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    unwrap_ok(r)
}
""")

# INC15-IN-DEAD-CODE: `if false { unwrap_ok(make_err()) }` — does
# the dead branch's TRAP affect anything?
run_src("DEAD-PANIC: unwrap_ok in dead-branch", """
fn make_err() -> Result<i32, i32> { Err(99) }
fn main() -> i32 {
    let cond: bool = false;
    if cond { unwrap_ok(make_err()) } else { 42 }
}
""")
