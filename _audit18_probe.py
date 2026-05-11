"""Cycle 18 audit-C adversarial probes. Read-only diagnostic; safe to delete."""
import sys, os, subprocess, hashlib
sys.path.insert(0, '.')
from helixc.frontend.parser import parse
from helixc.frontend.flatten_modules import flatten_modules
from helixc.frontend.flatten_impls import flatten_impls
from helixc.frontend.monomorphize import monomorphize
from helixc.frontend.grad_pass import grad_pass
from helixc.frontend.typecheck import typecheck
from helixc.ir.lower_ast import lower
from helixc.ir.passes.const_fold import fold_module
from helixc.ir.passes.cse import cse_module
from helixc.ir.passes.dce import dce_module
from helixc.ir.passes.fdce import fdce_module
from helixc.backend.x86_64 import compile_module_to_elf


def cr(src, label, opt=True):
    prog = parse(src, include_stdlib=True)
    flatten_modules(prog); flatten_impls(prog); monomorphize(prog); grad_pass(prog)
    errs = typecheck(prog)
    hard = [e for e in errs if not (hasattr(e,'is_warning') and e.is_warning)]
    print(f'[{label}] typecheck hard:', len(hard))
    for e in hard:
        print(f'  ERR: {e}')
    try:
        mod = lower(prog)
    except Exception as e:
        print(f'[{label}] lower FAILED: {type(e).__name__}: {e}')
        return None
    if opt:
        fold_module(mod); cse_module(mod); dce_module(mod); fdce_module(mod)
    try:
        elf = compile_module_to_elf(mod)
    except Exception as e:
        print(f'[{label}] codegen FAILED: {type(e).__name__}: {e}')
        return None
    h = hashlib.sha256(elf).hexdigest()[:12]
    out_dir = 'helixc/tests/_tmp'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'audit18_{h}.bin')
    with open(out_path, 'wb') as f:
        f.write(elf)
    abs_path = os.path.abspath(out_path).replace(os.sep, '/')
    if len(abs_path) >= 2 and abs_path[1] == ':':
        wsl_path = '/mnt/' + abs_path[0].lower() + abs_path[2:]
    else:
        wsl_path = abs_path
    r = subprocess.run(
        ['wsl', '--', 'bash', '-c', f'chmod +x {wsl_path} && {wsl_path}'],
        capture_output=True, timeout=20,
    )
    out = r.stdout.decode(errors='replace').strip()
    err = r.stderr.decode(errors='replace').strip()
    print(f'[{label}] elf_size={len(elf)} rc={r.returncode} stdout={out!r}')
    if err:
        print(f'[{label}] stderr={err!r}')
    return r


# Probe A: nested array - xs[0] when xs: [[i32;3];2]
src_a = '''
fn main() -> i32 {
    let xs = [[10, 20, 30], [40, 50, 60]];
    xs[0]
}
'''
cr(src_a, 'A: xs[0] nested no-opt', opt=False)
cr(src_a, 'A2: xs[0] nested -O', opt=True)

# Probe B: outer index 1
src_b = '''
fn main() -> i32 {
    let xs = [[10, 20, 30], [40, 50, 60]];
    xs[1]
}
'''
cr(src_b, 'B: xs[1] nested', opt=False)

# Probe C: nested index xs[0][1]
src_c = '''
fn main() -> i32 {
    let xs = [[10, 20, 30], [40, 50, 60]];
    xs[0][1]
}
'''
cr(src_c, 'C: xs[0][1] nested', opt=False)

# Probe D: nested with sum
src_d = '''
fn main() -> i32 {
    let xs = [[1, 2, 3], [4, 5, 6]];
    let mut s = 0;
    let mut i = 0;
    while i < 2 {
        let mut j = 0;
        while j < 3 {
            s = s + xs[i][j];
            j = j + 1;
        }
        i = i + 1;
    }
    s
}
'''
cr(src_d, 'D: sum nested 2x3 (expect 21)', opt=False)

# Probe E: simple 1D array sanity (control)
src_e = '''
fn main() -> i32 {
    let xs = [10, 20, 30];
    xs[1]
}
'''
cr(src_e, 'E: 1D xs[1] (expect 20)', opt=False)

# Probe F: nested array, struct-style (array of struct)
src_f = '''
struct P { x: i32, y: i32 }

fn main() -> i32 {
    let ps = [P { x: 7, y: 8 }, P { x: 9, y: 10 }];
    ps[1].x
}
'''
cr(src_f, 'F: array-of-struct (expect 9)', opt=False)

# Probe G: nested f64 (intersection with C16-1)
src_g = '''
fn main() -> i32 {
    let xs = [[1.0_f64, 2.0_f64], [3.0_f64, 4.0_f64]];
    xs[0];
    0
}
'''
cr(src_g, 'G: nested f64', opt=False)

# Probe H: triple-nested i32 — [[i32;3];4]; xs[2][1]
src_h = '''
fn main() -> i32 {
    let xs = [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]];
    xs[2][1]
}
'''
cr(src_h, 'H: 4x3 xs[2][1] (expect 80)', opt=False)
