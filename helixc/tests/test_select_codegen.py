"""Regression tests for the SELECT op's branch-displacement encoding.

The SELECT op compiles to:
  mov eax, [cond]
  test eax, eax
  je SKIP_A
    mov eax, [a]
  jmp END
  SKIP_A: mov eax, [b]
  END: mov [res], eax

The mov instructions are 3 bytes when the slot fits in disp8 (-128..127) and
6+ bytes when the slot needs disp32. The old implementation hard-coded the
je/jmp byte offsets assuming disp8 — silently miscompiling whenever a frame
was big enough to push slots past -128.

These tests build TIR by hand (the parser doesn't emit SELECT yet) and
exercise both the small-frame and large-frame paths.
"""

from __future__ import annotations
import os, sys, subprocess, tempfile, shlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.ir import tir
from helixc.backend.x86_64 import compile_module_to_elf


def _win_to_wsl(win_path: str) -> str:
    p = os.path.abspath(win_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"
    return p


def _make_module_with_select(n_padding_slots: int) -> tir.Module:
    """Build a module with: fn main() -> i32 = select(1, 7, 99) + n_padding 0s.
    The padding adds dead SSA values to fatten the frame, pushing the SELECT
    operand slots past disp8 range when n_padding is large enough.
    """
    mod = tir.Module()
    builder_id = [0]
    builder_blk = [0]

    def vid():
        v = builder_id[0]
        builder_id[0] += 1
        return v

    def bid():
        v = builder_blk[0]
        builder_blk[0] += 1
        return v

    i32 = tir.TIRScalar("i32")
    mod.next_value_id = 0
    mod.next_block_id = 0

    # Build entry block
    blk = tir.Block(id=bid())
    ops: list[tir.Op] = []

    # n_padding ALLOC_VAR (each gets a slot but is unused)
    for i in range(n_padding_slots):
        v = tir.Value(id=vid(), ty=i32, name_hint=f"pad_{i}")
        ops.append(tir.Op(kind=tir.OpKind.ALLOC_VAR, operands=[],
                          results=[v], attrs={"name": f"pad_{i}"}))

    # Constants for cond=1, a=7, b=99
    v_cond = tir.Value(id=vid(), ty=i32)
    v_a = tir.Value(id=vid(), ty=i32)
    v_b = tir.Value(id=vid(), ty=i32)
    ops.append(tir.Op(kind=tir.OpKind.CONST_INT, operands=[],
                      results=[v_cond], attrs={"value": 1}))
    ops.append(tir.Op(kind=tir.OpKind.CONST_INT, operands=[],
                      results=[v_a], attrs={"value": 7}))
    ops.append(tir.Op(kind=tir.OpKind.CONST_INT, operands=[],
                      results=[v_b], attrs={"value": 99}))

    # SELECT
    v_res = tir.Value(id=vid(), ty=i32)
    ops.append(tir.Op(kind=tir.OpKind.SELECT,
                      operands=[v_cond, v_a, v_b],
                      results=[v_res]))

    # Return v_res
    ops.append(tir.Op(kind=tir.OpKind.RETURN, operands=[v_res], results=[]))

    blk.ops = ops
    mod.next_value_id = builder_id[0]
    mod.next_block_id = builder_blk[0]

    fn = tir.FnIR(name="main", params=[], return_ty=i32, blocks=[blk])
    mod.functions["main"] = fn
    return mod


def _run_elf(elf_bytes: bytes) -> int:
    """Drop the ELF to disk, run via WSL, return exit code."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(proj_root, "helixc", "tests", "_tmp")
    os.makedirs(out_dir, exist_ok=True)
    fd, out_path = tempfile.mkstemp(
        prefix="select_", suffix=".bin", dir=out_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(elf_bytes)
    try:
        os.chmod(out_path, 0o755)
    except OSError:
        pass
    wsl_path = shlex.quote(_win_to_wsl(out_path))
    proc = subprocess.run(
        ["wsl", "--", "bash", "-c", f"chmod +x {wsl_path} && {wsl_path}"],
        capture_output=True, timeout=10
    )
    return proc.returncode


def test_select_small_frame():
    # Few slots — SELECT operands fit in disp8.
    mod = _make_module_with_select(n_padding_slots=2)
    elf = compile_module_to_elf(mod)
    assert _run_elf(elf) == 7   # cond=1 => result = a = 7


def test_select_large_frame_disp32():
    # Many padding slots push SELECT operands past -128 byte offset, forcing
    # the loads to use disp32 encoding (6 bytes vs 3). The old hard-coded
    # je +5 / jmp +3 placed the branch landing in the middle of an
    # instruction. With the patched-displacement fix this should work.
    mod = _make_module_with_select(n_padding_slots=40)  # 40*8 = 320 bytes of padding
    elf = compile_module_to_elf(mod)
    assert _run_elf(elf) == 7


def test_select_picks_b_when_cond_zero():
    # cond=0 path. Build manually with cond=0.
    mod = tir.Module()
    i32 = tir.TIRScalar("i32")
    blk = tir.Block(id=0)

    v_cond = tir.Value(id=0, ty=i32)
    v_a = tir.Value(id=1, ty=i32)
    v_b = tir.Value(id=2, ty=i32)
    v_res = tir.Value(id=3, ty=i32)

    blk.ops = [
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_cond],
               attrs={"value": 0}),
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_a],
               attrs={"value": 7}),
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_b],
               attrs={"value": 23}),
        tir.Op(kind=tir.OpKind.SELECT, operands=[v_cond, v_a, v_b],
               results=[v_res]),
        tir.Op(kind=tir.OpKind.RETURN, operands=[v_res], results=[]),
    ]
    mod.functions["main"] = tir.FnIR(name="main", params=[],
                                     return_ty=i32, blocks=[blk])
    mod.next_value_id = 4
    mod.next_block_id = 1
    elf = compile_module_to_elf(mod)
    assert _run_elf(elf) == 23   # cond=0 => result = b = 23


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
