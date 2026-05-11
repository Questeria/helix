"""Cycle 18 audit-C round 2: user-requested adversarial probes."""
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
    print(f'[{label}] tc_hard={len(hard)}', end=' ')
    for e in hard:
        print(f'ERR={e!r}', end=' ')
    try:
        mod = lower(prog)
    except Exception as e:
        print(f'lower FAIL: {type(e).__name__}: {e}')
        return None
    if opt:
        fold_module(mod); cse_module(mod); dce_module(mod); fdce_module(mod)
    try:
        elf = compile_module_to_elf(mod)
    except Exception as e:
        print(f'codegen FAIL: {type(e).__name__}: {e}')
        return None
    h = hashlib.sha256(elf).hexdigest()[:12]
    out_dir = 'helixc/tests/_tmp'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'audit18_v2_{h}.bin')
    with open(out_path, 'wb') as f:
        f.write(elf)
    abs_path = os.path.abspath(out_path).replace(os.sep, '/')
    wsl_path = '/mnt/' + abs_path[0].lower() + abs_path[2:]
    r = subprocess.run(
        ['wsl', '--', 'bash', '-c', f'chmod +x {wsl_path} && {wsl_path}'],
        capture_output=True, timeout=20,
    )
    print(f'rc={r.returncode}')
    return r


# User-requested adversarial probes:
# 1. [[i32; 3]; 4] with -O2
src_a = '''
fn main() -> i32 {
    let xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
    xs[0]
}
'''
cr(src_a, '1a: [[i32;3];4] xs[0] no-opt', opt=False)
cr(src_a, '1b: [[i32;3];4] xs[0] -O', opt=True)

src_a2 = '''
fn main() -> i32 {
    let xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
    xs[3]
}
'''
cr(src_a2, '1c: [[i32;3];4] xs[3] no-opt', opt=False)
cr(src_a2, '1d: [[i32;3];4] xs[3] -O', opt=True)

# 2. Struct-of-array (a struct whose field is an array)
src_b = '''
struct Row { vals: i32, n: i32 }
fn main() -> i32 {
    let r = Row { vals: 42, n: 3 };
    r.vals
}
'''
cr(src_b, '2a: struct-with-scalar (control)', opt=False)

# Surface syntax for struct-with-array-field
src_b2 = '''
struct Box { xs: [i32; 3] }
fn main() -> i32 {
    let b = Box { xs: [10, 20, 30] };
    b.xs[1]
}
'''
cr(src_b2, '2b: struct-of-array Box{xs:[i32;3]}', opt=False)

# 3. Array-of-struct with -O2
src_c = '''
struct P { x: i32, y: i32 }
fn main() -> i32 {
    let ps = [P { x: 7, y: 8 }, P { x: 9, y: 10 }];
    ps[1].x
}
'''
cr(src_c, '3a: array-of-struct ps[1].x no-opt (expect 9)', opt=False)
cr(src_c, '3b: array-of-struct ps[1].x -O (expect 9)', opt=True)

# Control: bare 1D
src_d = '''
fn main() -> i32 {
    let xs = [10, 20, 30, 40];
    xs[2]
}
'''
cr(src_d, '4: control 1D xs[2] (expect 30)', opt=False)

# Bonus: does nested ARRAY of f64 hit C16-1 trap or silently miscompile?
src_e = '''
fn main() -> i32 {
    let xs = [[1.0_f64, 2.5_f64], [3.5_f64, 4.5_f64]];
    xs[0];
    0
}
'''
cr(src_e, '5: nested f64 - does C16-1 trap fire?', opt=False)
