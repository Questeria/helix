"""
helixc/backend/llvm_ir.py — textual LLVM IR backend (v3.0 Phase D).

v3.0 replaces the hand-rolled x86_64 ELF emitter with a backend that
emits textual LLVM IR for the LLVM toolchain (`opt` + `llc`) to consume.
Per the v3.0 migration strategy (docs/V3_PLAN.md) this is ADDITIVE: it
consumes the same host IR — a `tir.Module` — that
`helixc/backend/x86_64.py::compile_module_to_elf` consumes, and
`x86_64.py` is left completely untouched until the Stage 221 cutover.

Supported so far (Stages 200, 202-204):
  - module header + target triple
  - a `define` for each function (integer params + integer/void return)
  - integer constants (CONST_INT, materialized as inline literals)
  - integer add / sub / mul; `ret`
  - control flow (Stage 202): multi-block functions — every tir block
    becomes a labelled LLVM basic block; BR / COND_BR become LLVM
    `br`; a block's tir parameters become LLVM `phi` nodes collecting
    the matching branch argument from each predecessor.
  - scalar op set (Stage 203): the six integer comparisons (`icmp`,
    signed or unsigned per operand dtype), SELECT, NEG; plus the
    unsigned integer dtypes (u8/u16/u32/u64/usize) and isize.
  - division / remainder and the bitwise op set (Stage 203 cont.):
    DIV / MOD (signed `sdiv`/`srem` or unsigned `udiv`/`urem` per
    operand dtype), the sign-agnostic AND / OR / XOR and left shift
    SHL, the right shift SHR (arithmetic `ashr` or logical `lshr` per
    operand dtype), and the unary bitwise NOT.
  - mutable local variables and stack arrays (Stage 204): ALLOC_VAR /
    LOAD_VAR / STORE_VAR and ALLOC_ARRAY / LOAD_ELEM / STORE_ELEM
    become LLVM `alloca` / `load` / `store` (plus `getelementptr` for
    an array element's address) — every `alloca` is hoisted to the
    entry block; loads, stores and GEPs use opaque pointers (`ptr`).
  - direct + FFI function calls (Stage 205): CALL and FFI_CALL become
    an LLVM `call` — a value call (`%vN = call <ty> @callee(...)`) or
    a void call, arguments passed positionally as typed operands. An
    FFI_CALL additionally emits a module-scope `declare` for its
    extern target.
  - Result<T,E> packed-tag intrinsics (Stage 206): RESULT_PACK /
    RESULT_TAG / RESULT_PAYLOAD become integer `zext`/`shl`/`or`,
    `lshr`+`trunc`, and `trunc` — a Result is one i64 with the tag in
    the high 32 bits and the payload in the low 32.
  - panic (Stage 206): TRAP lowers to `write(2, msg, len)` of the
    `panic[<id>]: <text>` message to stderr, `exit(<id> & 0xFF)`, and
    `unreachable` — the message is a private module-scope string
    constant; `write` / `exit` are declared externs.
  - string-literal access (Stage 206): STR_PTR lowers to `ptrtoint`
    of the literal's module-scope constant; STR_BYTE to a
    bounds-checked indexed `load i8` (an out-of-range index yields 0,
    with no out-of-bounds read — see the TRAP-shared string globals).
  - string output (Stage 206): a `print_str` PRINT lowers to
    `write(1, msg, len)` of a module-scope string constant — the i64
    byte count truncated to the op's i32 result.
  - integer output (Stage 206-R): a `print_int` PRINT lowers to a
    call to the internal helper `@__helix_print_int(i32)` (an i32 ->
    ASCII decimal conversion plus `write(1, buf, len)`); the helper's
    body is emitted exactly once per module via the `_HELPER_FUNCTIONS`
    registry.
  - file output (Stage 206-R): a `write_file` PRINT lowers to the
    libc sequence `open(path, O_WRONLY|O_CREAT|O_TRUNC, 0644) ->
    write(fd, content, len) -> close(fd)`. The op's i32 result is
    `nwritten < 0 ? nwritten : 0` — matches x86_64.py's "negative on
    failure, 0 on success" contract.
  - file input (Stage 206-R): a `read_file_to_arena` PRINT lowers
    to a call to `@__helix_read_file_to_arena` (a six-block helper
    that opens path O_RDONLY, reads up to BUF_SIZE bytes into a
    stack buffer, traps via `@llvm.trap()` on truncation
    (`nread == BUF_SIZE` sentinel — matches x86's `ud2`), and pushes
    each byte to the shared arena via `__helix_arena_push`). The
    helper carries `helper_deps=("__helix_arena_push",)` so the
    transitive dependency on the arena global is auto-registered
    through `_register_helper_function`'s recursive walk.
  - arena ops (Stage 206-R): `ARENA_PUSH` / `ARENA_GET` / `ARENA_SET`
    / `ARENA_LEN` / `ARENA_PUSH_PAIR` / `ARENA_PUSH_TRIPLE` all lower
    to calls to internal helpers (each its own three-block bounds-
    checked routine, except ARENA_LEN which is a single load) that
    share the module-scope arena global `@__helix_arena_base` (a
    `[CAP+1 x i32]` BSS buffer with the i32 cursor at slot 0 and user
    data in slots 1..CAP). The multi-slot pushes (PAIR / TRIPLE) are
    atomic-or-none: on overflow neither / none of the writes happen
    AND the cursor does not advance. The `_MODULE_GLOBALS` registry
    mirrors `_HELPER_FUNCTIONS` for shared module-scope state; a
    helper that touches a global lists it in its `module_globals`
    and the global is emitted exactly once per module.
  - trace ops (Stage 206-R): `TRACE_ENTRY` / `TRACE_EXIT` lower to a
    call to the void-returning helper `@__helix_trace_event(i32 fn_id,
    i32 kind)` (the first void-returning helper in the registry).
    The helper appends a (fn_id, kind) event to the ring buffer
    `@__helix_trace_buf` ([2*CAP x i32]) at cursor `@__helix_trace_count`
    if there's room; full buffer silently drops the event (fail-
    closed, no allocation/syscall — matches x86). fn_ids are
    interned per-module by `_intern_trace_fn_ids` in module-walk
    order — `emit_module` builds the table BEFORE constructing the
    per-fn emitters and shares it across all of them.
  - AGI metaprogramming (Stage 206-R): `QUOTE` materialises a
    compile-time reflection-cell handle in [0, NUM_CELLS) as a
    pure `add i32 0, <handle>` (the `ast_handle` attr mod
    NUM_CELLS); `SPLICE` reads cell[handle] via the bounds-checked
    `@__helix_splice(i32) -> i32` helper (3 blocks, returns 0 on
    OOB); `MODIFY` bounds-checks handle, calls a user-supplied
    verifier(handle, new_value), and conditionally stores via the
    `@__helix_modify(i32, i32, ptr) -> i32` helper (4 blocks,
    returns 1 on accepted-store, 0 on OOB or verifier-reject). Both
    SPLICE / MODIFY share the `@__helix_state_base = [NUM_CELLS x
    i64]` cell array (matches x86's `__helix_state_base` BSS
    region). Only `value_kind == "i32"` is lowered today; f32 / f64
    variants raise loudly (Stage 207 polymorphic-helper follow-up).
    `REFLECT_HASH` is a placeholder in TIR with no x86 lowering
    either — it lands in the LLVM catchall fail-closed, matching
    x86's NotImplementedError.

Anything outside that set — structs, floats, the wider op surface —
is REJECTED with a loud `LLVMEmitError`,
never emitted wrong. Those land in later stages. A mock-validation path
(`mock_validate_ll`) checks the emitted `.ll` text shape without needing
an LLVM toolchain, mirroring `gpu_ci.py`'s mock path; real
`llvm-as`/`opt`/`llc` dispatch (`llvm_toolchain.py`) is Stage 201.

License: Apache 2.0
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, NamedTuple, Optional

from ..ir import tir


# The host target. x86_64.py emits a Linux x86-64 ELF; the LLVM path
# targets the same triple so the Stage 207 parity harness compares like
# for like.
LLVM_TARGET_TRIPLE = "x86_64-unknown-linux-gnu"


class LLVMEmitError(Exception):
    """The host IR contains a construct the LLVM backend does not
    cover. Raised loudly so an unsupported op can never be silently
    dropped or mis-emitted — the v3.0 "additive, parity-gated"
    discipline (docs/V3_PLAN.md): a partial backend fails closed, it
    never produces wrong IR. The supported op set is the one this
    module's docstring enumerates."""


# tir scalar-integer dtype name -> LLVM integer type. LLVM integer
# types are sign-agnostic — `u32` and `i32` are both the LLVM type
# `i32`; signedness is per-instruction (icmp slt vs ult). The unsigned
# dtypes live in `_UNSIGNED_INT_DTYPES`, which carries the sign for
# that choice.
_LLVM_INT_TYPES: dict[str, str] = {
    "bool": "i1",
    "i8": "i8", "u8": "i8",
    "i16": "i16", "u16": "i16",
    "i32": "i32", "u32": "i32",
    "i64": "i64", "u64": "i64",
    "isize": "i64", "usize": "i64",
    # `char` is also a TIRScalar integer dtype, but its bit width is not
    # yet pinned for the LLVM path — consciously deferred. A char-typed
    # function loudly raises LLVMEmitError here until a later stage
    # fixes the width (fail-closed, not a silent miss).
}

# Helix unsigned integer dtypes — each shares an LLVM integer type with
# its signed counterpart but selects an unsigned predicate for `icmp`.
_UNSIGNED_INT_DTYPES: frozenset[str] = frozenset({
    "u8", "u16", "u32", "u64", "usize",
})

# v3.1 step 4 — TIR float dtype -> LLVM float type. Currently used
# only by the `__helix_splice_f32/f64` + `__helix_modify_f32/f64`
# polymorphic helpers' op-handler validation (the rest of the LLVM
# backend's docstring at the top of this file says "floats and
# structs are later stages" — full float-arithmetic support remains
# a larger v3.1+ chunk that adds `fadd`/`fsub`/etc. and threads
# float types through `_llvm_return_type`).
_LLVM_FLOAT_TYPES: dict[str, str] = {
    "f32": "float",
    "f64": "double",
}

# v3.1 step 4 audit-fix HIGH-1 — single source of truth for the
# SPLICE/MODIFY polymorphic dispatch tables. Hoisted to module scope so
# the validation set (`value_kind in _SPLICE_DISPATCH`) and the lookup
# (`_SPLICE_DISPATCH[value_kind]`) cannot drift. The op handler used to
# duplicate the validation tuple `("i32","f32","f64")` alongside the
# inline dict literal — adding a new value_kind to one without the
# other surfaced as a bare KeyError in production (silent-failure-
# hunter LOW-1 + type-design-analyzer HIGH-1).
#
# SPLICE: value_kind -> (helper_name, expected_llvm_result_ty,
#                        llvm_call_ret_ty)
#   helper_name: registry key in `_HELPER_FUNCTIONS`
#   expected_llvm_result_ty: what the result's LLVM type-text MUST be
#   llvm_call_ret_ty: the LLVM return-type text in the call site
# All three are the same string today (i32/i32/i32, float/float/float,
# double/double/double) but kept as a triple in case a future
# value_kind decouples (e.g. an i64 splice that casts to i32 result).
_SPLICE_DISPATCH: Mapping[str, tuple[str, str, str]] = MappingProxyType({
    "i32": ("__helix_splice", "i32", "i32"),
    "f32": ("__helix_splice_f32", "float", "float"),
    "f64": ("__helix_splice_f64", "double", "double"),
})

# MODIFY: value_kind -> (helper_name, new_value_llvm_ty). Result is
# ALWAYS i32 (the accepted-or-not flag) — independent of value_kind.
_MODIFY_DISPATCH: Mapping[str, tuple[str, str]] = MappingProxyType({
    "i32": ("__helix_modify", "i32"),
    "f32": ("__helix_modify_f32", "float"),
    "f64": ("__helix_modify_f64", "double"),
})

# tir comparison OpKind -> (signed predicate, unsigned predicate) for
# LLVM `icmp`. eq / ne are sign-agnostic; the ordered comparisons are
# not, so the predicate is chosen by the operand dtype's signedness.
_LLVM_ICMP_PREDS: dict[tir.OpKind, tuple[str, str]] = {
    tir.OpKind.CMP_EQ: ("eq", "eq"),
    tir.OpKind.CMP_NE: ("ne", "ne"),
    tir.OpKind.CMP_LT: ("slt", "ult"),
    tir.OpKind.CMP_LE: ("sle", "ule"),
    tir.OpKind.CMP_GT: ("sgt", "ugt"),
    tir.OpKind.CMP_GE: ("sge", "uge"),
}

# tir integer binary OpKind -> LLVM instruction mnemonic, for the
# SIGN-AGNOSTIC binary ops: LLVM's `add`/`and`/`shl`/... are a single
# instruction whose result does not depend on whether the Helix
# operands are signed or unsigned. The shift-RIGHT and the division /
# remainder ops are NOT sign-agnostic — they live in
# `_LLVM_SIGNED_BINOPS`.
_LLVM_SCALAR_BINOPS: dict[tir.OpKind, str] = {
    tir.OpKind.ADD: "add",
    tir.OpKind.SUB: "sub",
    tir.OpKind.MUL: "mul",
    tir.OpKind.BIT_AND: "and",
    tir.OpKind.BIT_OR: "or",
    tir.OpKind.BIT_XOR: "xor",
    tir.OpKind.SHL: "shl",
}

# tir integer binary OpKind -> (signed mnemonic, unsigned mnemonic).
# Unlike `_LLVM_SCALAR_BINOPS`, each of these LLVM instructions comes
# in a signed and an unsigned form, chosen by the signedness of the
# Helix operand dtype:
#   - DIV -> `sdiv` / `udiv`, MOD -> `srem` / `urem`;
#   - SHR (shift right) -> arithmetic `ashr` (sign-extends the vacated
#     high bits) / logical `lshr` (zero-fills them).
# SHL (shift left) is sign-agnostic and stays in `_LLVM_SCALAR_BINOPS`.
_LLVM_SIGNED_BINOPS: dict[tir.OpKind, tuple[str, str]] = {
    tir.OpKind.DIV: ("sdiv", "udiv"),
    tir.OpKind.MOD: ("srem", "urem"),
    tir.OpKind.SHR: ("ashr", "lshr"),
}

def _check_binop_table_disjoint() -> None:
    """A binary OpKind must live in exactly one of `_LLVM_SCALAR_BINOPS`
    / `_LLVM_SIGNED_BINOPS` — an op in both would silently take the
    sign-agnostic form and lose its signed/unsigned distinction. This
    is a module-load guard: an explicit `raise`, not a bare `assert`
    (which `python -O` would strip, silently disabling the check).
    Mirrors `llvm_toolchain.py`'s `_check_llvm_toolchain_drift`."""
    overlap = _LLVM_SCALAR_BINOPS.keys() & _LLVM_SIGNED_BINOPS.keys()
    if overlap:
        raise AssertionError(
            f"helixc.backend.llvm_ir: tir.OpKind(s) "
            f"{sorted(k.name for k in overlap)} appear in both "
            f"_LLVM_SCALAR_BINOPS and _LLVM_SIGNED_BINOPS — an op in "
            f"both silently loses its signed/unsigned distinction"
        )


_check_binop_table_disjoint()

# LLVM's unquoted global/local identifier grammar.
_LLVM_BARE_IDENT = re.compile(r"[-a-zA-Z$._][-a-zA-Z$._0-9]*\Z")

# A double-quoted span — a quoted `@"..."` identifier or the target-
# triple string. `_llvm_global_name` hex-escapes `"` inside a quoted
# name, so a span never contains a literal `"`; `mock_validate_ll`
# masks these spans before counting braces (a `}` inside a quoted
# name is not a structural brace).
_QUOTED_SPAN = re.compile(r'"[^"]*"')


def _llvm_int_type(ty: tir.TIRType, *, ctx: str) -> str:
    """Map a TIR scalar integer type to its LLVM type string. Raises
    LLVMEmitError for any non-integer-scalar type (floats, tensors,
    tuples, unit) — all outside this backend's covered subset."""
    if not isinstance(ty, tir.TIRScalar):
        raise LLVMEmitError(
            f"{ctx}: the LLVM backend emits only scalar integer "
            f"types, got {type(ty).__name__}"
        )
    llvm = _LLVM_INT_TYPES.get(ty.name)
    if llvm is None:
        raise LLVMEmitError(
            f"{ctx}: the LLVM backend does not emit dtype "
            f"{ty.name!r} (supported: {sorted(_LLVM_INT_TYPES)})"
        )
    return llvm


def _llvm_return_type(ty: tir.TIRType, *, ctx: str) -> str:
    """Like `_llvm_int_type`, but a function return type may also be
    the unit type `()` — which maps to LLVM `void`."""
    if isinstance(ty, tir.TIRUnit):
        return "void"
    return _llvm_int_type(ty, ctx=ctx)


def _llvm_global_name(name: str) -> str:
    """Render a function name as an LLVM global symbol. A name that fits
    LLVM's unquoted-identifier grammar is emitted bare (`@main`);
    anything else (e.g. a monomorphized generic name carrying `<`, `>`,
    `,` or spaces) is emitted in LLVM's quoted form `@"..."` with `"`
    and `\\` hex-escaped — so an out-of-grammar name yields valid IR
    instead of silently-malformed text."""
    if _LLVM_BARE_IDENT.match(name):
        return "@" + name
    escaped = name.replace("\\", "\\5C").replace('"', "\\22")
    return f'@"{escaped}"'


def _llvm_cstring(data: bytes) -> str:
    """Render raw bytes as an LLVM `c"..."` string-constant body. A
    printable-ASCII byte passes through literally — except `"` and
    `\\`, which would end / escape the literal — and every other byte
    becomes `\\XX`, LLVM's two-hex-digit (uppercase) escape."""
    out: list[str] = []
    for byte in data:
        if 0x20 <= byte <= 0x7E and byte not in (0x22, 0x5C):
            out.append(chr(byte))
        else:
            out.append(f"\\{byte:02X}")
    return 'c"' + "".join(out) + '"'


def _is_unsigned_int(ty: tir.TIRType) -> bool:
    """True for a Helix unsigned-integer scalar dtype. Drives the
    signed-vs-unsigned LLVM instruction choice (`icmp slt` vs `ult`,
    `sdiv` vs `udiv`, `ashr` vs `lshr`)."""
    return (isinstance(ty, tir.TIRScalar)
            and ty.name in _UNSIGNED_INT_DTYPES)


def _require_same_signedness(a: tir.Value, b: tir.Value, *,
                             ctx: str) -> None:
    """Raise `LLVMEmitError` when two integer values disagree on Helix
    signedness.

    The LLVM-type equality check (`_llvm_int_type(a) ==
    _llvm_int_type(b)`) cannot catch this: `i32` and `u32` are the
    SAME LLVM type. But for an op whose LLVM instruction is *chosen
    by* signedness — `sdiv` vs `udiv`, `srem` vs `urem`, `ashr` vs
    `lshr`, the ordered `icmp` predicates — a mixed signed/unsigned
    combination makes that choice ambiguous. Silently picking one
    interpretation would emit IR that can disagree with the x86_64
    backend on the same program. Failing closed here keeps the v3.0
    'additive, parity-gated' discipline: the LLVM backend never
    silently resolves an ill-specified construct.

    `a` and `b` are the two values that must agree — for DIV / MOD the
    two operands, for SHR the shifted value and the result. (A shift
    COUNT is never passed here — a count's signedness never affects
    the result — and `eq` / `ne` are sign-agnostic, so neither
    reaches this guard.)"""
    if _is_unsigned_int(a.ty) != _is_unsigned_int(b.ty):
        a_name = a.ty.name if isinstance(a.ty, tir.TIRScalar) else a.ty
        b_name = b.ty.name if isinstance(b.ty, tir.TIRScalar) else b.ty
        raise LLVMEmitError(
            f"{ctx}: values disagree on signedness ({a_name} vs "
            f"{b_name}) — this op's LLVM instruction is selected by "
            f"signedness, so a mixed signed/unsigned combination is "
            f"ambiguous; insert an explicit cast so the integer "
            f"dtypes agree")


# --------------------------------------------------------------------------
# 206-R helper functions: small internal runtime emitted as module-scope
# `define internal` blocks. The first user is `PRINT.print_int` (an i32-
# to-ASCII conversion + sys_write that is too unwieldy to inline at every
# call site). Each spec carries (1) the full LLVM `define internal ...`
# text emitted into the module exactly once and (2) the FFI declares the
# helper transitively needs (so an op calling the helper does not have to
# re-state them at the call site). All helper names live under the
# reserved `__helix_` prefix so they cannot collide with user-defined
# Helix function names; `emit_module` enforces that with an explicit
# collision check.
# --------------------------------------------------------------------------
class _FFIDeclareSpec(NamedTuple):
    """One module-scope `declare` line a helper function depends on.
    Named (vs. a bare 4-tuple) so positional-order swaps are caught
    at edit time rather than as wrong-symbol declares at emit time.

    Fields:
      target: the FFI symbol's bare name (e.g. "write") — the key into
              `_FnEmitter.ffi_declares` for dedup.
      callee: the LLVM-globalized name (e.g. "@write") — appears verbatim
              in the emitted `declare` line.
      ret_ty: the LLVM return-type string (e.g. "i64").
      arg_tys: the LLVM argument-type strings (e.g. ("i32", "ptr", "i64")).
    """
    target: str
    callee: str
    ret_ty: str
    arg_tys: tuple[str, ...]


# Arena cap — must agree with `helixc/backend/x86_64.py::HELIX_ARENA_CAP`
# (the Stage 207 parity gate compares both backends against this single
# value; a drift here would silently change the arena overflow point).
# Slot layout: i32 cursor at slot 0, user data in slots 1..CAP
# (inclusive), so the LLVM global needs `CAP + 1` slots total.
_HELIX_ARENA_CAP = 2097152

# Threshold formula for an N-slot atomic push: `cursor >= CAP - (N - 1)`
# is overflow. PUSH (N=1) uses CAP, PAIR (N=2) uses CAP - 1, TRIPLE
# (N=3) uses CAP - 2. A future `__helix_arena_push_quad` (N=4) would
# use CAP - 3. The formula is encoded inline in each push helper
# rather than centralised — `mock_validate_ll` and the helper-text-
# pinning tests grep the literal threshold, which would have to know
# the formula too if it were factored out.

# Shared module-globals tuple — every arena helper depends on the
# same `__helix_arena_base` global, so reference one constant rather
# than typing the name six times (eliminates the typo surface; a
# new arena helper added in a future chunk gets the dependency
# right by reference).
_HELIX_ARENA_GLOBALS: tuple[str, ...] = ("__helix_arena_base",)


# Trace ring-buffer capacity — must agree with
# `helixc/backend/x86_64.py::HELIX_TRACE_CAP`. Each trace event is
# (i32 fn_id, i32 kind) — 8 bytes — so the buffer is `2 * CAP` i32
# slots. Cursor (`@__helix_trace_count`) tracks how many events
# have been appended; when it reaches CAP, subsequent events are
# silently dropped (Phase-0 fail-closed: no allocation, no syscall,
# no blocking — matches x86 line 4404-4407).
_HELIX_TRACE_CAP = 1024


# `read_file_to_arena` stack-buffer size. 1 MiB matches
# `x86_64.py::BUF_SIZE` in the same op (line ~3456) — sized to
# accommodate the bootstrap source (lexer + parser + kovc, ~111 KB)
# with ~4x headroom for tokens + AST. If the file is larger than
# this, `read(2)` returns exactly this many bytes and the helper
# TRAPS (truncation sentinel — mirrors x86's `ud2`). Stage 207
# parity requires both backends to use the same buffer size so they
# trap on the same input.
_HELIX_READ_FILE_BUF_SIZE = 0x100000


# Reflection-cell count — must match `x86_64.py::HELIX_NUM_CELLS`.
# Each cell is i64 (`HELIX_CELL_SIZE = 8` on x86). QUOTE materialises
# a handle in [0, NUM_CELLS) at compile time (via `ast_handle %
# NUM_CELLS`); SPLICE loads from `@__helix_state_base[handle]`;
# MODIFY conditionally stores after a user-supplied verifier
# function approves. A handle outside [0, NUM_CELLS) returns 0 from
# SPLICE / fails MODIFY (matches x86's bounds-check semantics).
_HELIX_NUM_CELLS = 64

# Shared module-globals tuple for the reflection cells (parallel to
# `_HELIX_ARENA_GLOBALS` / `_HELIX_TRACE_GLOBALS`).
_HELIX_STATE_GLOBALS: tuple[str, ...] = ("__helix_state_base",)


# The PRINT op's `_kind` attribute names a sub-operation; this is
# the closed set the LLVM backend currently lowers. A single source
# of truth across the dispatch check (`if print_kind not in ...`)
# and the error-message text, so the two can never drift apart when
# the next PRINT sub-kind lands (a new entry here updates both
# sites simultaneously).
_SUPPORTED_PRINT_KINDS: frozenset[str] = frozenset({
    "print_str", "print_int", "write_file", "read_file_to_arena",
})


# Shared module-globals tuple for the trace helper (parallel to
# `_HELIX_ARENA_GLOBALS`).
_HELIX_TRACE_GLOBALS: tuple[str, ...] = (
    "__helix_trace_count", "__helix_trace_buf")


@dataclass(frozen=True, slots=True)
class _ModuleGlobalSpec:
    """One module-scope global variable a helper function depends on
    (e.g. the arena buffer `@__helix_arena_base`). Emitted exactly
    once per module via the `_MODULE_GLOBALS` registry when at least
    one registered helper declares the dependency.

    Frozen + slots + final — house pattern.

    Fields:
      name: the LLVM global symbol's bare name (e.g.
            "__helix_arena_base"). Reserved `__helix_` prefix so a
            user symbol cannot collide (enforced at emit time).
      definition: the full LLVM line (e.g.
            `@__helix_arena_base = internal global [2097153 x i32]
            zeroinitializer, align 4`).
    """
    name: str
    definition: str

    def __init_subclass__(cls, **kwargs) -> None:
        raise TypeError(
            "_ModuleGlobalSpec is final; subclassing could bypass "
            "the invariants")

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"_ModuleGlobalSpec: name must be a non-empty string "
                f"(got {type(self.name).__name__})")
        if not self.name.startswith("__helix_"):
            raise ValueError(
                f"_ModuleGlobalSpec: name {self.name!r} must use the "
                f"reserved `__helix_` prefix so user symbols cannot "
                f"collide")
        if not isinstance(self.definition, str) or not self.definition.strip():
            raise ValueError(
                f"_ModuleGlobalSpec: definition must be a non-empty "
                f"string (got {type(self.definition).__name__})")
        # The definition text must declare the registered name in
        # the canonical `@<name> = ...` form on its own line — drift
        # here means a future patch could register one name but
        # define a different global. Tightened from substring to
        # per-line so a `;`-comment mentioning `@<name> =` cannot
        # satisfy the guard (matches the parallel tightening on
        # `_check_helper_function_table`'s `define ` check).
        expected_prefix = f"@{self.name} ="
        if not any(ln.lstrip().startswith(expected_prefix)
                   for ln in self.definition.splitlines()):
            raise ValueError(
                f"_ModuleGlobalSpec: definition does not contain a "
                f"line starting with {expected_prefix!r} — registry "
                f"name and global declaration disagree (got "
                f"{self.definition[:80]!r}...)")


@dataclass(frozen=True, slots=True)
class _HelperFunctionSpec:
    """One internal helper function. `definition` is the full
    `define internal ...` LLVM text (multi-line); `ret_ty` is the
    helper's LLVM return-type string (e.g. "i32", "void"), used by
    `_check_helper_function_table` to cross-check the call-site
    emission shape — a mismatch between call-site `call <ret_ty>
    @<name>(...)` and the helper's `define internal <ret_ty>` would
    produce malformed IR that `mock_validate_ll` does not catch;
    `ffi_declares` is a tuple of `_FFIDeclareSpec` entries the
    helper's body calls (these go through the same
    `_register_ffi_declare` plumbing so a helper-needed extern
    dedups against an FFI_CALL-needed extern instead of double-
    declaring); `module_globals` is a tuple of `__helix_*` module-
    global names this helper references (each is looked up in
    `_MODULE_GLOBALS` and the global is emitted exactly once per
    module).

    Frozen + slots + final (subclass-guarded) — matches the house
    pattern for cross-cutting backend result types (see `Backend` in
    `helixc/backend/backend_registry.py` and `ParityResult` in
    `helixc/ir/mlir/parity.py`). `__post_init__` raises `ValueError`
    on invariant violation (same contract as the rest of the family).

    `helper_deps` lists other `__helix_*` helpers this helper's body
    calls transitively. `_register_helper_function` walks `helper_deps`
    recursively so a caller that registers only the leaf helper still
    pulls every transitively-needed helper (and its globals + FFI
    declares) into the emitted module. Cycles are broken by the early
    `if name in self.helper_functions: return` short-circuit.
    """
    definition: str
    ret_ty: str
    ffi_declares: tuple[_FFIDeclareSpec, ...]
    module_globals: tuple[str, ...] = ()
    helper_deps: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        raise TypeError(
            "_HelperFunctionSpec is final; subclassing could bypass "
            "the invariants")

    def __post_init__(self) -> None:
        if not isinstance(self.definition, str) or not self.definition.strip():
            raise ValueError(
                f"_HelperFunctionSpec: definition must be a non-empty "
                f"string (got {type(self.definition).__name__})")
        # Tightened from a substring check to a per-line check: at
        # least one stripped line must START with `define ` so a
        # `define` mention in a `;`-comment cannot satisfy the guard.
        if not any(ln.lstrip().startswith("define ")
                   for ln in self.definition.splitlines()):
            raise ValueError(
                f"_HelperFunctionSpec: definition must contain a "
                f"line starting with `define ` "
                f"(got {self.definition[:60]!r}...)")
        if not isinstance(self.ret_ty, str) or not self.ret_ty.strip():
            raise ValueError(
                f"_HelperFunctionSpec: ret_ty must be a non-empty "
                f"string (got {type(self.ret_ty).__name__})")
        # Cross-check the advertised ret_ty against the helper's
        # `define internal <ret_ty>` line — drift here would let a
        # call-site emit `call i32 @<name>(...)` against a `define
        # internal void` helper, producing IR that `mock_validate_ll`
        # does not detect but `llvm-as` rejects.
        expected_define_prefix = f"define internal {self.ret_ty} "
        if not any(ln.lstrip().startswith(expected_define_prefix)
                   for ln in self.definition.splitlines()):
            raise ValueError(
                f"_HelperFunctionSpec: ret_ty {self.ret_ty!r} does "
                f"not match any `define internal <ret_ty> ...` line "
                f"in the definition text (expected a line starting "
                f"with {expected_define_prefix!r})")
        if not isinstance(self.ffi_declares, tuple):
            raise ValueError(
                f"_HelperFunctionSpec: ffi_declares must be a tuple "
                f"(got {type(self.ffi_declares).__name__})")
        for entry in self.ffi_declares:
            if not isinstance(entry, _FFIDeclareSpec):
                raise ValueError(
                    f"_HelperFunctionSpec: ffi_declares entry must be "
                    f"a _FFIDeclareSpec, got {type(entry).__name__} "
                    f"({entry!r})")
            if (not isinstance(entry.target, str)
                    or not isinstance(entry.callee, str)
                    or not isinstance(entry.ret_ty, str)
                    or not isinstance(entry.arg_tys, tuple)
                    or not all(isinstance(x, str)
                               for x in entry.arg_tys)):
                raise ValueError(
                    f"_HelperFunctionSpec: ffi_declares entry has "
                    f"wrong field types (got {entry!r})")
        if not isinstance(self.module_globals, tuple):
            raise ValueError(
                f"_HelperFunctionSpec: module_globals must be a "
                f"tuple (got {type(self.module_globals).__name__})")
        for entry in self.module_globals:
            if not isinstance(entry, str) or not entry.startswith("__helix_"):
                raise ValueError(
                    f"_HelperFunctionSpec: module_globals entry "
                    f"{entry!r} must be a `__helix_*` string")
        # `module_globals` is a tuple but used set-like (dedup
        # downstream via `self.module_globals.add`). Reject duplicate
        # entries at the spec level so a typo in the registry that
        # lists `("__helix_arena_base", "__helix_arena_base")`
        # surfaces at module load. Mirrors `Backend.required_dialects`
        # in `helixc/backend/backend_registry.py`.
        if len(self.module_globals) != len(set(self.module_globals)):
            raise ValueError(
                f"_HelperFunctionSpec: module_globals has duplicates "
                f"({self.module_globals})")
        if not isinstance(self.helper_deps, tuple):
            raise ValueError(
                f"_HelperFunctionSpec: helper_deps must be a tuple "
                f"(got {type(self.helper_deps).__name__})")
        for entry in self.helper_deps:
            if not isinstance(entry, str) or not entry.startswith("__helix_"):
                raise ValueError(
                    f"_HelperFunctionSpec: helper_deps entry "
                    f"{entry!r} must be a `__helix_*` string")
        if len(self.helper_deps) != len(set(self.helper_deps)):
            raise ValueError(
                f"_HelperFunctionSpec: helper_deps has duplicates "
                f"({self.helper_deps})")


# `__helix_print_int(i32) -> i32`: convert an i32 to a base-10 ASCII
# string and write it to stdout via `write(1, buf, len)`; return the
# byte count (the i64 syscall return truncated to i32, matching the
# print_str PRINT contract).
#
# Mirrors `x86_64.py::print_int` (line ~4315) — the SAME absolute-value
# strategy (compute `0 - v` as i32, treat the result as unsigned for the
# digit loop). This means `INT_MIN` rolls over to itself when negated,
# is then interpreted as the unsigned value 2147483648 by `udiv`/`urem`,
# producing the digits "2147483648"; the `is_neg` flag is true for
# `INT_MIN` so a '-' is prepended, yielding "-2147483648" — the correct
# decimal text for `INT_MIN` and bit-for-bit parity with x86_64.py.
#
# The body is five basic blocks (entry / loop / after_loop /
# prepend_sign / do_write); the loop's two phi nodes carry the digit
# pointer (walks down from `buf+16`) and the running value (starts at
# `abs`, becomes `quot` on each iteration). It is a do-while loop so
# the input `0` still writes one '0' digit, matching x86_64.py's
# first-iteration-always-runs structure. The stack buffer is 16 bytes
# — large enough for an `i32`'s ten decimal digits plus a sign byte,
# with three bytes of headroom. `internal` linkage prevents external
# collisions; the `__helix_` prefix prevents in-module collisions
# (enforced by `emit_module`).
_HELIX_PRINT_INT_HELPER = """\
define internal i32 @__helix_print_int(i32 %v) {
entry:
  %buf = alloca [16 x i8], align 1
  %end_ptr = getelementptr inbounds [16 x i8], ptr %buf, i64 0, i64 16
  %is_neg = icmp slt i32 %v, 0
  %neg_v = sub i32 0, %v
  %abs = select i1 %is_neg, i32 %neg_v, i32 %v
  br label %loop

loop:
  %ptr_cur = phi ptr [ %end_ptr, %entry ], [ %ptr_next, %loop ]
  %val_cur = phi i32 [ %abs, %entry ], [ %quot, %loop ]
  %ptr_next = getelementptr inbounds i8, ptr %ptr_cur, i64 -1
  %quot = udiv i32 %val_cur, 10
  %rem = urem i32 %val_cur, 10
  %rem8 = trunc i32 %rem to i8
  %digit = add i8 %rem8, 48
  store i8 %digit, ptr %ptr_next, align 1
  %not_done = icmp ne i32 %quot, 0
  br i1 %not_done, label %loop, label %after_loop

after_loop:
  br i1 %is_neg, label %prepend_sign, label %do_write

prepend_sign:
  %sign_ptr = getelementptr inbounds i8, ptr %ptr_next, i64 -1
  store i8 45, ptr %sign_ptr, align 1
  br label %do_write

do_write:
  %start_ptr = phi ptr [ %ptr_next, %after_loop ], [ %sign_ptr, %prepend_sign ]
  %end_i64 = ptrtoint ptr %end_ptr to i64
  %start_i64 = ptrtoint ptr %start_ptr to i64
  %len = sub i64 %end_i64, %start_i64
  %nwritten = call i64 @write(i32 1, ptr %start_ptr, i64 %len)
  %result = trunc i64 %nwritten to i32
  ret i32 %result
}"""


# `@__helix_arena_base`: the shared arena buffer. Slot 0 is the i32
# cursor; slots 1..CAP are user data. Sized to match
# `x86_64.py::HELIX_ARENA_CAP` so both backends overflow at the same
# slot (a Stage 207 parity requirement). `internal` linkage so the
# symbol is link-local; `zeroinitializer` puts it in BSS (no on-disk
# size cost — only virtual-memory).
_HELIX_ARENA_GLOBAL_DEF = (
    f"@__helix_arena_base = internal global "
    f"[{_HELIX_ARENA_CAP + 1} x i32] zeroinitializer, align 4")


# Trace globals — `@__helix_trace_count` is the i32 cursor (number of
# events appended), `@__helix_trace_buf` is the ring buffer. Each
# event is two consecutive i32s (fn_id at +0, kind at +4), so the
# buffer is laid out as `[2 * CAP x i32]` and the helper indexes
# `count * 2` for fn_id, `count * 2 + 1` for kind. Layout matches
# x86_64.py's `(HELIX_TRACE_CAP * 8) bytes` BSS region (line ~4925
# of x86_64.py) — the Stage 207 parity gate compares observable
# event-buffer state after a traced run, so the byte layout must
# agree exactly.
_HELIX_TRACE_COUNT_GLOBAL_DEF = (
    "@__helix_trace_count = internal global i32 0, align 4")

_HELIX_TRACE_BUF_GLOBAL_DEF = (
    f"@__helix_trace_buf = internal global "
    f"[{_HELIX_TRACE_CAP * 2} x i32] zeroinitializer, align 4")


# Reflection-cell state: `[NUM_CELLS x i64] zeroinitializer`. Each
# cell is 8 bytes (matches `x86_64.py::HELIX_CELL_SIZE = 8`). SPLICE
# reads one cell; MODIFY conditionally writes one (after the user's
# verifier function approves). x86's layout is the same array, in
# the binary's writable region (line 4903-4907 of x86_64.py).
_HELIX_STATE_BASE_GLOBAL_DEF = (
    f"@__helix_state_base = internal global "
    f"[{_HELIX_NUM_CELLS} x i64] zeroinitializer, align 8")

# `__helix_arena_push(i32) -> i32`: push one i32 value to the arena;
# return the new slot's index, or -1 on overflow. Mirrors
# `x86_64.py::ARENA_PUSH` (line ~3135):
#   load cursor; if cursor >= CAP -> return -1; else
#   store value at slot (cursor + 1); cursor++; return old cursor.
#
# Three basic blocks (entry / in_bounds / exit). On overflow, entry
# branches directly to exit — the "overflow" path is folded into
# entry's else branch (no separate `overflow:` block), so the phi at
# exit reads `[ -1, %entry ]` (not `[ -1, %overflow ]`). The arena
# global's layout is `[CAP+1 x i32]`: slot 0 is the cursor, slots
# 1..CAP are user data, so slot N's byte offset from the base pointer
# is `(N + 1) * 4` — we GEP with `i64 (cursor + 1)` after a sext.
# `exit` is a phi (-1 from entry, old cursor from in_bounds) so the
# function has exactly one return.
_HELIX_ARENA_PUSH_HELPER = f"""\
define internal i32 @__helix_arena_push(i32 %value) {{
entry:
  %cursor = load i32, ptr @__helix_arena_base, align 4
  %ovfl = icmp uge i32 %cursor, {_HELIX_ARENA_CAP}
  br i1 %ovfl, label %exit, label %in_bounds

in_bounds:
  %cursor_plus_one = add i32 %cursor, 1
  %slot_idx_i64 = sext i32 %cursor_plus_one to i64
  %slot_ptr = getelementptr inbounds i32, ptr @__helix_arena_base, i64 %slot_idx_i64
  store i32 %value, ptr %slot_ptr, align 4
  store i32 %cursor_plus_one, ptr @__helix_arena_base, align 4
  br label %exit

exit:
  %result = phi i32 [ -1, %entry ], [ %cursor, %in_bounds ]
  ret i32 %result
}}"""


# `__helix_arena_get(i32 idx) -> i32`: return arena[idx + 1] (slot 0
# is the cursor; user data lives in slots 1..CAP). Out-of-bounds
# (negative when interpreted unsigned, or >= CAP) returns 0 — defined
# behaviour matching x86_64.py::ARENA_GET (line ~3261). The `icmp uge`
# catches both negative indices (which are large unsigned values) and
# values >= CAP. Three blocks (entry / in_bounds / exit).
_HELIX_ARENA_GET_HELPER = f"""\
define internal i32 @__helix_arena_get(i32 %idx) {{
entry:
  %ovfl = icmp uge i32 %idx, {_HELIX_ARENA_CAP}
  br i1 %ovfl, label %exit, label %in_bounds

in_bounds:
  %idx_plus_one = add i32 %idx, 1
  %idx_i64 = sext i32 %idx_plus_one to i64
  %slot_ptr = getelementptr inbounds i32, ptr @__helix_arena_base, i64 %idx_i64
  %loaded = load i32, ptr %slot_ptr, align 4
  br label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %loaded, %in_bounds ]
  ret i32 %result
}}"""


# `__helix_arena_set(i32 idx, i32 value) -> i32`: store value at
# arena[idx + 1]. Out-of-bounds silently no-ops (matches
# x86_64.py::ARENA_SET line ~3284 — "out-of-bounds set silently
# no-ops"). Always returns 0 — TIR says ARENA_SET has no result, but
# x86_64.py tolerates a result slot and writes 0 into it; this helper
# matches that contract so the op handler can either bind the return
# (when op.results) or discard it (when no results). Three blocks
# (entry / in_bounds / exit).
_HELIX_ARENA_SET_HELPER = f"""\
define internal i32 @__helix_arena_set(i32 %idx, i32 %value) {{
entry:
  %ovfl = icmp uge i32 %idx, {_HELIX_ARENA_CAP}
  br i1 %ovfl, label %exit, label %in_bounds

in_bounds:
  %idx_plus_one = add i32 %idx, 1
  %idx_i64 = sext i32 %idx_plus_one to i64
  %slot_ptr = getelementptr inbounds i32, ptr @__helix_arena_base, i64 %idx_i64
  store i32 %value, ptr %slot_ptr, align 4
  br label %exit

exit:
  ret i32 0
}}"""


# `__helix_arena_len() -> i32`: return the current cursor (slot 0 of
# the arena). One-block helper — the same single load x86_64.py
# emits inline (line ~3334), but wrapped in a helper for symmetry
# with the other arena ops (every arena op routes through a helper,
# the helper is the only thing that touches the global).
_HELIX_ARENA_LEN_HELPER = """\
define internal i32 @__helix_arena_len() {
entry:
  %result = load i32, ptr @__helix_arena_base, align 4
  ret i32 %result
}"""


# `__helix_arena_push_pair(i32 left, i32 right) -> i32`: ATOMIC two-
# slot push. Writes `left` at arena[cursor+1] and `right` at
# arena[cursor+2], advances cursor by 2, returns the OLD cursor (=
# slot index of left). Overflow when cursor would land both writes
# outside data slots: `cursor + 1 >= CAP` ⇔ `cursor >= CAP - 1`
# (i.e., CAP - 1 leaves room for slot CAP-1 and slot CAP at write
# offsets, both of which are inside the [CAP+1]-slot array). On
# overflow, neither write happens AND the cursor does NOT advance —
# atomic-or-none, mirroring `x86_64.py::ARENA_PUSH_PAIR` (line
# ~3170). Returns -1 on overflow.
#
# Three blocks (entry / in_bounds / exit). The phi at exit reads
# `[-1, %entry]` (overflow path is folded into entry's else branch,
# same shape as `__helix_arena_push`).
_HELIX_ARENA_PUSH_PAIR_HELPER = f"""\
define internal i32 @__helix_arena_push_pair(i32 %left, i32 %right) {{
entry:
  %cursor = load i32, ptr @__helix_arena_base, align 4
  %ovfl = icmp uge i32 %cursor, {_HELIX_ARENA_CAP - 1}
  br i1 %ovfl, label %exit, label %in_bounds

in_bounds:
  %cursor_plus_one = add i32 %cursor, 1
  %cursor_plus_two = add i32 %cursor, 2
  %left_idx_i64 = sext i32 %cursor_plus_one to i64
  %right_idx_i64 = sext i32 %cursor_plus_two to i64
  %left_ptr = getelementptr inbounds i32, ptr @__helix_arena_base, i64 %left_idx_i64
  %right_ptr = getelementptr inbounds i32, ptr @__helix_arena_base, i64 %right_idx_i64
  store i32 %left, ptr %left_ptr, align 4
  store i32 %right, ptr %right_ptr, align 4
  store i32 %cursor_plus_two, ptr @__helix_arena_base, align 4
  br label %exit

exit:
  %result = phi i32 [ -1, %entry ], [ %cursor, %in_bounds ]
  ret i32 %result
}}"""


# `__helix_arena_push_triple(i32 left, i32 middle, i32 right) -> i32`:
# ATOMIC three-slot push. Mirror of PUSH_PAIR with one extra slot.
# Writes at cursor+1 / cursor+2 / cursor+3; advances cursor by 3;
# returns the OLD cursor (slot index of left). Overflow threshold is
# `cursor + 2 >= CAP` ⇔ `cursor >= CAP - 2` (so all three target
# slots fit in [CAP+1] data). Atomic-or-none: on overflow none of
# the writes happen and the cursor stays put. Mirrors
# `x86_64.py::ARENA_PUSH_TRIPLE` (line ~3214). Returns -1 on overflow.
_HELIX_ARENA_PUSH_TRIPLE_HELPER = f"""\
define internal i32 @__helix_arena_push_triple(i32 %left, i32 %middle, i32 %right) {{
entry:
  %cursor = load i32, ptr @__helix_arena_base, align 4
  %ovfl = icmp uge i32 %cursor, {_HELIX_ARENA_CAP - 2}
  br i1 %ovfl, label %exit, label %in_bounds

in_bounds:
  %cursor_plus_one = add i32 %cursor, 1
  %cursor_plus_two = add i32 %cursor, 2
  %cursor_plus_three = add i32 %cursor, 3
  %left_idx_i64 = sext i32 %cursor_plus_one to i64
  %middle_idx_i64 = sext i32 %cursor_plus_two to i64
  %right_idx_i64 = sext i32 %cursor_plus_three to i64
  %left_ptr = getelementptr inbounds i32, ptr @__helix_arena_base, i64 %left_idx_i64
  %middle_ptr = getelementptr inbounds i32, ptr @__helix_arena_base, i64 %middle_idx_i64
  %right_ptr = getelementptr inbounds i32, ptr @__helix_arena_base, i64 %right_idx_i64
  store i32 %left, ptr %left_ptr, align 4
  store i32 %middle, ptr %middle_ptr, align 4
  store i32 %right, ptr %right_ptr, align 4
  store i32 %cursor_plus_three, ptr @__helix_arena_base, align 4
  br label %exit

exit:
  %result = phi i32 [ -1, %entry ], [ %cursor, %in_bounds ]
  ret i32 %result
}}"""


# `__helix_trace_event(i32 fn_id, i32 kind) -> void`: append one
# trace event to the ring buffer. Three blocks (entry / store /
# skip). If `count >= CAP`, skip silently (full buffer fail-closed,
# matches x86 comment "no allocation, no syscall"). Otherwise store
# fn_id at `__helix_trace_buf[count * 2]`, kind at `[count * 2 + 1]`,
# and increment `__helix_trace_count`. The helper returns void --
# the first void-returning entry in `_HELPER_FUNCTIONS`. Mirrors
# x86_64.py::_emit_trace_event (line ~1005).
_HELIX_TRACE_EVENT_HELPER = f"""\
define internal void @__helix_trace_event(i32 %fn_id, i32 %kind) {{
entry:
  %count = load i32, ptr @__helix_trace_count, align 4
  %full = icmp uge i32 %count, {_HELIX_TRACE_CAP}
  br i1 %full, label %skip, label %store

store:
  %count_i64 = sext i32 %count to i64
  %fn_id_idx = shl i64 %count_i64, 1
  %kind_idx = add i64 %fn_id_idx, 1
  %fn_id_ptr = getelementptr inbounds i32, ptr @__helix_trace_buf, i64 %fn_id_idx
  %kind_ptr = getelementptr inbounds i32, ptr @__helix_trace_buf, i64 %kind_idx
  store i32 %fn_id, ptr %fn_id_ptr, align 4
  store i32 %kind, ptr %kind_ptr, align 4
  %new_count = add i32 %count, 1
  store i32 %new_count, ptr @__helix_trace_count, align 4
  br label %skip

skip:
  ret void
}}"""


# `__helix_read_file_to_arena(ptr path) -> i32`: open `path` read-
# only, read up to BUF_SIZE bytes into a stack buffer, push each
# byte (as i32) into the arena via `__helix_arena_push`, return the
# number of bytes read (clamped to 0 on a negative read return).
# Mirrors `x86_64.py::read_file_to_arena` (line ~3423).
#
# TRUNCATION SENTINEL: if `read` returns exactly BUF_SIZE, the file
# either filled the buffer exactly OR was larger and got truncated.
# The helper cannot distinguish, so it TRAPS via `@llvm.trap()`
# (lowers to `ud2` / SIGILL on x86_64, matching x86's literal `ud2`
# at line 3500). The build fails loudly rather than silently
# producing a corrupt arena state — the original cascade-bug
# (bootstrap source crept up to 261 KB of a 256 KB buffer, silent
# truncation produced a bad K2, SIGILL at runtime far downstream —
# see x86_64.py's BUG-mitigation comments at line 3486-3496) is
# exactly the failure mode this guards against.
#
# FALSE-POSITIVE WINDOW: a file of EXACTLY BUF_SIZE bytes traps
# even though it was not truncated — `read` cannot signal "file
# is exactly this size" without a follow-up zero-byte read. The
# trap is the conservative choice (lose a legitimate edge-case
# file to a build failure; do not silently produce a corrupt
# arena from a truncated file). Mirrors x86_64.py line 3494-3500.
#
# Six basic blocks (entry / trap / sign_check / loop_header /
# loop_body / exit). The per-byte push loop calls
# `__helix_arena_push` for each byte and discards the result (a
# full arena returns -1 from push but the loop continues — matches
# x86's "rcx increments regardless of push success" at line 3537).
#
# NOTE (Stage 207 parity): four cross-backend contract gaps are
# inherited verbatim from x86_64.py — both backends are mutually
# consistent, and these are NOT silent failures introduced by the
# LLVM path. Documented here for the Stage 207 parity gate:
#
#   1. `open` failure (path not found, permission denied, etc.) is
#      propagated indirectly: a -1 fd flows into `read(fd=-1, ...)`
#      which returns -EBADF, which the sign-clamp drives to 0 —
#      the user-visible result is "0 bytes read" rather than the
#      real `open` errno.
#   2. `read` failure (EINTR, EIO, etc.) is sign-clamped to 0 —
#      same shape as #1 (real errno lost).
#   3. `close(fd)` failure (EIO from delayed flush, EBADF from
#      prior bug) silently discarded — `%close_ret` is bound but
#      never observed. Matches x86's "no error check after
#      sys_close".
#   4. `__helix_arena_push` failure mid-loop (arena full -> -1
#      returns) is dropped on the floor; the helper returns bytes-
#      READ, NOT bytes-PUSHED. A full arena scenario silently
#      under-reports actual push count. Matches x86's
#      "rcx increments regardless of push success" at line 3537.
#
# All four are conscious tradeoffs that match x86's contract; the
# Stage 207 parity gate decides whether to tighten any of them in
# a coordinated way.
_HELIX_READ_FILE_TO_ARENA_HELPER = f"""\
define internal i32 @__helix_read_file_to_arena(ptr %path) {{
entry:
  %buf = alloca [{_HELIX_READ_FILE_BUF_SIZE} x i8], align 1
  %fd = call i32 @open(ptr %path, i32 0, i32 0)
  %nread = call i64 @read(i32 %fd, ptr %buf, i64 {_HELIX_READ_FILE_BUF_SIZE})
  %close_ret = call i32 @close(i32 %fd)
  %was_full = icmp eq i64 %nread, {_HELIX_READ_FILE_BUF_SIZE}
  br i1 %was_full, label %trap, label %sign_check

trap:
  call void @llvm.trap()
  unreachable

sign_check:
  %nread_i32 = trunc i64 %nread to i32
  %is_neg = icmp slt i32 %nread_i32, 0
  %nread_clamped = select i1 %is_neg, i32 0, i32 %nread_i32
  br label %loop_header

loop_header:
  %i = phi i32 [ 0, %sign_check ], [ %i_next, %loop_body ]
  %done = icmp uge i32 %i, %nread_clamped
  br i1 %done, label %exit, label %loop_body

loop_body:
  %i_i64 = sext i32 %i to i64
  %byte_ptr = getelementptr inbounds [{_HELIX_READ_FILE_BUF_SIZE} x i8], ptr %buf, i64 0, i64 %i_i64
  %byte = load i8, ptr %byte_ptr, align 1
  %byte_i32 = zext i8 %byte to i32
  %push_ret = call i32 @__helix_arena_push(i32 %byte_i32)
  %i_next = add i32 %i, 1
  br label %loop_header

exit:
  ret i32 %nread_clamped
}}"""


# `__helix_splice(i32 handle) -> i32`: load cell[handle] as i32
# (truncated from i64). Out-of-bounds handle (negative when
# interpreted signed, or >= NUM_CELLS) returns 0. Mirrors
# `x86_64.py::SPLICE` (line ~4477) for the default i32 value_kind.
# Three blocks (entry / load / exit). The phi at exit returns 0
# from entry (the OOB path) and the truncated load from load.
#
# NOTE (Stage 207 parity): x86 also handles `value_kind == "f64"`
# at the same op site (returns the full i64 instead of truncating);
# the LLVM helper only handles the i32 default. The op handler
# rejects non-i32 value_kind explicitly — supporting f32/f64 would
# require either a polymorphic helper or per-kind variants and is
# deferred to a follow-up chunk.
_HELIX_SPLICE_HELPER = f"""\
define internal i32 @__helix_splice(i32 %handle) {{
entry:
  %neg = icmp slt i32 %handle, 0
  %big = icmp sge i32 %handle, {_HELIX_NUM_CELLS}
  %oob = or i1 %neg, %big
  br i1 %oob, label %exit, label %load

load:
  %handle_i64 = sext i32 %handle to i64
  %slot_ptr = getelementptr inbounds i64, ptr @__helix_state_base, i64 %handle_i64
  %loaded = load i64, ptr %slot_ptr, align 8
  %trunc = trunc i64 %loaded to i32
  br label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %trunc, %load ]
  ret i32 %result
}}"""


# `__helix_modify(i32 handle, i32 new_value, ptr verifier) -> i32`:
# bounds-check handle; call verifier(handle, new_value); if the
# verifier returns non-zero AND the handle is in bounds, store
# new_value (sign-extended to i64) into cell[handle] and return 1;
# else return 0. Mirrors `x86_64.py::MODIFY` (line ~4529) for the
# default i32 value_kind.
#
# The verifier is a user-supplied function — its name is hard-coded
# at the LLVM call site as a function pointer (`ptr @<verifier>`).
# The helper itself is verifier-agnostic; the call goes through the
# `%verifier` argument so one helper definition serves every MODIFY
# call site in the module.
#
# Four blocks (entry / verify / apply / exit). On bounds failure,
# entry branches DIRECTLY to exit (skipping verifier + store);
# matches x86's "OOB never reaches verifier" semantics (line 4554-
# 4565). Exit phi: 0 from entry (OOB), 0 from verify (rejected),
# 1 from apply (accepted-store).
_HELIX_MODIFY_HELPER = f"""\
define internal i32 @__helix_modify(i32 %handle, i32 %new_value, ptr %verifier) {{
entry:
  %neg = icmp slt i32 %handle, 0
  %big = icmp sge i32 %handle, {_HELIX_NUM_CELLS}
  %oob = or i1 %neg, %big
  br i1 %oob, label %exit, label %verify

verify:
  %accepted = call i32 %verifier(i32 %handle, i32 %new_value)
  %ok = icmp ne i32 %accepted, 0
  br i1 %ok, label %apply, label %exit

apply:
  %handle_i64 = sext i32 %handle to i64
  %slot_ptr = getelementptr inbounds i64, ptr @__helix_state_base, i64 %handle_i64
  %value_i64 = sext i32 %new_value to i64
  store i64 %value_i64, ptr %slot_ptr, align 8
  br label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ 0, %verify ], [ 1, %apply ]
  ret i32 %result
}}"""


# v3.1 step 4 — f32 / f64 polymorphic SPLICE / MODIFY helpers.
# x86_64.py::SPLICE/MODIFY (lines 4523-4527, 4571-4609) handle the
# three value_kinds at the same op site by switching ABIs:
#   - i32 splice -> load i64, return low 32 bits (current
#     `__helix_splice`).
#   - f32 splice -> load low 32 bits of cell, bitcast to float (the
#     cell's upper 32 bits are zero per the f32 modify path).
#   - f64 splice -> load full 64-bit cell, bitcast to double.
#   - i32 modify -> verifier(i32, i32) -> i32; sext value to i64, store.
#   - f32 modify -> verifier(i32, float) -> i32; bitcast float to
#     i32, zext to i64, store. Cell upper 32 bits = 0.
#   - f64 modify -> verifier(i32, double) -> i32; bitcast double to
#     i64, store.
#
# Each kind gets its own helper because LLVM's `define internal
# <T>` is mono-typed; a polymorphic `__helix_splice` would need
# either separate definitions per overload OR an i64-typed return
# the caller bitcasts (which complicates the call-site SSA shape
# and loses LLVM's type-system check). Separate helpers keep each
# call site clean: the op handler picks the right one by
# value_kind.
_HELIX_SPLICE_F32_HELPER = f"""\
define internal float @__helix_splice_f32(i32 %handle) {{
entry:
  %neg = icmp slt i32 %handle, 0
  %big = icmp sge i32 %handle, {_HELIX_NUM_CELLS}
  %oob = or i1 %neg, %big
  br i1 %oob, label %exit, label %load

load:
  %handle_i64 = sext i32 %handle to i64
  %slot_ptr = getelementptr inbounds i64, ptr @__helix_state_base, i64 %handle_i64
  %loaded = load i32, ptr %slot_ptr, align 4
  %as_float = bitcast i32 %loaded to float
  br label %exit

exit:
  %result = phi float [ 0.000000e+00, %entry ], [ %as_float, %load ]
  ret float %result
}}"""


_HELIX_SPLICE_F64_HELPER = f"""\
define internal double @__helix_splice_f64(i32 %handle) {{
entry:
  %neg = icmp slt i32 %handle, 0
  %big = icmp sge i32 %handle, {_HELIX_NUM_CELLS}
  %oob = or i1 %neg, %big
  br i1 %oob, label %exit, label %load

load:
  %handle_i64 = sext i32 %handle to i64
  %slot_ptr = getelementptr inbounds i64, ptr @__helix_state_base, i64 %handle_i64
  %loaded = load i64, ptr %slot_ptr, align 8
  %as_double = bitcast i64 %loaded to double
  br label %exit

exit:
  %result = phi double [ 0.000000e+00, %entry ], [ %as_double, %load ]
  ret double %result
}}"""


_HELIX_MODIFY_F32_HELPER = f"""\
define internal i32 @__helix_modify_f32(i32 %handle, float %new_value, ptr %verifier) {{
entry:
  %neg = icmp slt i32 %handle, 0
  %big = icmp sge i32 %handle, {_HELIX_NUM_CELLS}
  %oob = or i1 %neg, %big
  br i1 %oob, label %exit, label %verify

verify:
  %accepted = call i32 %verifier(i32 %handle, float %new_value)
  %ok = icmp ne i32 %accepted, 0
  br i1 %ok, label %apply, label %exit

apply:
  %handle_i64 = sext i32 %handle to i64
  %slot_ptr = getelementptr inbounds i64, ptr @__helix_state_base, i64 %handle_i64
  %value_i32 = bitcast float %new_value to i32
  %value_i64 = zext i32 %value_i32 to i64
  store i64 %value_i64, ptr %slot_ptr, align 8
  br label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ 0, %verify ], [ 1, %apply ]
  ret i32 %result
}}"""


_HELIX_MODIFY_F64_HELPER = f"""\
define internal i32 @__helix_modify_f64(i32 %handle, double %new_value, ptr %verifier) {{
entry:
  %neg = icmp slt i32 %handle, 0
  %big = icmp sge i32 %handle, {_HELIX_NUM_CELLS}
  %oob = or i1 %neg, %big
  br i1 %oob, label %exit, label %verify

verify:
  %accepted = call i32 %verifier(i32 %handle, double %new_value)
  %ok = icmp ne i32 %accepted, 0
  br i1 %ok, label %apply, label %exit

apply:
  %handle_i64 = sext i32 %handle to i64
  %slot_ptr = getelementptr inbounds i64, ptr @__helix_state_base, i64 %handle_i64
  %value_i64 = bitcast double %new_value to i64
  store i64 %value_i64, ptr %slot_ptr, align 8
  br label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ 0, %verify ], [ 1, %apply ]
  ret i32 %result
}}"""


# Private authority dict — mutated only at module init. The public
# `_HELPER_FUNCTIONS` is a `MappingProxyType` view (immutable from
# outside the module), mirroring the MLIR-side authority pattern at
# `helixc/ir/mlir/backends.py`. Tests that need to mutate the registry
# (drift-guard probes, etc.) must do so through the private name and
# restore on teardown.
_HELPER_FUNCTIONS_AUTHORITY: dict[str, _HelperFunctionSpec] = {
    "__helix_print_int": _HelperFunctionSpec(
        definition=_HELIX_PRINT_INT_HELPER,
        ret_ty="i32",
        ffi_declares=(
            _FFIDeclareSpec(
                target="write",
                callee="@write",
                ret_ty="i64",
                arg_tys=("i32", "ptr", "i64"),
            ),
        ),
    ),
    "__helix_arena_push": _HelperFunctionSpec(
        definition=_HELIX_ARENA_PUSH_HELPER,
        ret_ty="i32",
        ffi_declares=(),
        module_globals=_HELIX_ARENA_GLOBALS,
    ),
    "__helix_arena_get": _HelperFunctionSpec(
        definition=_HELIX_ARENA_GET_HELPER,
        ret_ty="i32",
        ffi_declares=(),
        module_globals=_HELIX_ARENA_GLOBALS,
    ),
    "__helix_arena_set": _HelperFunctionSpec(
        definition=_HELIX_ARENA_SET_HELPER,
        ret_ty="i32",
        ffi_declares=(),
        module_globals=_HELIX_ARENA_GLOBALS,
    ),
    "__helix_arena_len": _HelperFunctionSpec(
        definition=_HELIX_ARENA_LEN_HELPER,
        ret_ty="i32",
        ffi_declares=(),
        module_globals=_HELIX_ARENA_GLOBALS,
    ),
    "__helix_arena_push_pair": _HelperFunctionSpec(
        definition=_HELIX_ARENA_PUSH_PAIR_HELPER,
        ret_ty="i32",
        ffi_declares=(),
        module_globals=_HELIX_ARENA_GLOBALS,
    ),
    "__helix_arena_push_triple": _HelperFunctionSpec(
        definition=_HELIX_ARENA_PUSH_TRIPLE_HELPER,
        ret_ty="i32",
        ffi_declares=(),
        module_globals=_HELIX_ARENA_GLOBALS,
    ),
    "__helix_trace_event": _HelperFunctionSpec(
        definition=_HELIX_TRACE_EVENT_HELPER,
        ret_ty="void",
        ffi_declares=(),
        module_globals=_HELIX_TRACE_GLOBALS,
    ),
    "__helix_read_file_to_arena": _HelperFunctionSpec(
        definition=_HELIX_READ_FILE_TO_ARENA_HELPER,
        ret_ty="i32",
        ffi_declares=(
            _FFIDeclareSpec(
                target="open", callee="@open", ret_ty="i32",
                arg_tys=("ptr", "i32", "i32"),
            ),
            _FFIDeclareSpec(
                target="read", callee="@read", ret_ty="i64",
                arg_tys=("i32", "ptr", "i64"),
            ),
            _FFIDeclareSpec(
                target="close", callee="@close", ret_ty="i32",
                arg_tys=("i32",),
            ),
            _FFIDeclareSpec(
                target="llvm.trap", callee="@llvm.trap",
                ret_ty="void", arg_tys=(),
            ),
        ),
        module_globals=(),
        helper_deps=("__helix_arena_push",),
    ),
    "__helix_splice": _HelperFunctionSpec(
        definition=_HELIX_SPLICE_HELPER,
        ret_ty="i32",
        ffi_declares=(),
        module_globals=_HELIX_STATE_GLOBALS,
    ),
    "__helix_modify": _HelperFunctionSpec(
        definition=_HELIX_MODIFY_HELPER,
        ret_ty="i32",
        ffi_declares=(),
        module_globals=_HELIX_STATE_GLOBALS,
    ),
    "__helix_splice_f32": _HelperFunctionSpec(
        definition=_HELIX_SPLICE_F32_HELPER,
        ret_ty="float",
        ffi_declares=(),
        module_globals=_HELIX_STATE_GLOBALS,
    ),
    "__helix_splice_f64": _HelperFunctionSpec(
        definition=_HELIX_SPLICE_F64_HELPER,
        ret_ty="double",
        ffi_declares=(),
        module_globals=_HELIX_STATE_GLOBALS,
    ),
    "__helix_modify_f32": _HelperFunctionSpec(
        definition=_HELIX_MODIFY_F32_HELPER,
        ret_ty="i32",
        ffi_declares=(),
        module_globals=_HELIX_STATE_GLOBALS,
    ),
    "__helix_modify_f64": _HelperFunctionSpec(
        definition=_HELIX_MODIFY_F64_HELPER,
        ret_ty="i32",
        ffi_declares=(),
        module_globals=_HELIX_STATE_GLOBALS,
    ),
}

_HELPER_FUNCTIONS: Mapping[str, _HelperFunctionSpec] = MappingProxyType(
    _HELPER_FUNCTIONS_AUTHORITY)


# Module-scope globals registry — same private-AUTHORITY + public-proxy
# pattern as `_HELPER_FUNCTIONS`. A helper declares its module-global
# dependencies in `module_globals`; `emit_module` collects the union
# across helpers used in the module and emits each global exactly
# once. The `__helix_` prefix is reserved (same collision-protection
# semantics as the helper functions).
_MODULE_GLOBALS_AUTHORITY: dict[str, _ModuleGlobalSpec] = {
    "__helix_arena_base": _ModuleGlobalSpec(
        name="__helix_arena_base",
        definition=_HELIX_ARENA_GLOBAL_DEF,
    ),
    "__helix_trace_count": _ModuleGlobalSpec(
        name="__helix_trace_count",
        definition=_HELIX_TRACE_COUNT_GLOBAL_DEF,
    ),
    "__helix_trace_buf": _ModuleGlobalSpec(
        name="__helix_trace_buf",
        definition=_HELIX_TRACE_BUF_GLOBAL_DEF,
    ),
    "__helix_state_base": _ModuleGlobalSpec(
        name="__helix_state_base",
        definition=_HELIX_STATE_BASE_GLOBAL_DEF,
    ),
}

_MODULE_GLOBALS: Mapping[str, _ModuleGlobalSpec] = MappingProxyType(
    _MODULE_GLOBALS_AUTHORITY)


def _check_module_global_table() -> None:
    """Drift guard for `_MODULE_GLOBALS_AUTHORITY`. Fails loudly at
    module load. Three invariants:

      1. Each entry's registry key matches its `_ModuleGlobalSpec.name`
         (a typo would let the registry resolve a different name than
         the helper actually references in its body text).
      2. Every helper's `module_globals` dependency name resolves in
         `_MODULE_GLOBALS_AUTHORITY` (otherwise the helper would
         register a global that is never emitted, and the call site
         would link to nothing).
      3. The `__helix_*` namespace is shared between helpers and
         module-globals — no name appears in BOTH registries
         simultaneously. Same-name in both would emit BOTH a
         `define internal i32 @__helix_X(...)` AND a
         `@__helix_X = internal global ...` line — `llvm-as` rejects
         this with "redefinition of @X", and `mock_validate_ll`
         does NOT detect it.
    """
    for key, spec in _MODULE_GLOBALS_AUTHORITY.items():
        if key != spec.name:
            raise AssertionError(
                f"helixc.backend.llvm_ir: module-global key {key!r} "
                f"does not match spec.name {spec.name!r}")
    # Invariant 2 — every helper's module-global dependency resolves.
    for hname, hspec in _HELPER_FUNCTIONS_AUTHORITY.items():
        for global_name in hspec.module_globals:
            if global_name not in _MODULE_GLOBALS_AUTHORITY:
                raise AssertionError(
                    f"helixc.backend.llvm_ir: helper {hname!r} "
                    f"declares module-global dependency on "
                    f"{global_name!r} but no such global is "
                    f"registered (known: "
                    f"{sorted(_MODULE_GLOBALS_AUTHORITY)})")
    # Invariant 3 — helper-name vs module-global-name collision.
    overlap = (set(_HELPER_FUNCTIONS_AUTHORITY)
               & set(_MODULE_GLOBALS_AUTHORITY))
    if overlap:
        raise AssertionError(
            f"helixc.backend.llvm_ir: name(s) {sorted(overlap)} "
            f"appear in BOTH `_HELPER_FUNCTIONS_AUTHORITY` and "
            f"`_MODULE_GLOBALS_AUTHORITY` — every `__helix_*` name "
            f"must be unique across the two registries (emitting "
            f"BOTH a `define internal @X(...)` and a "
            f"`@X = ... global ...` line produces malformed LLVM IR)")


def _strip_llvm_comment(line: str) -> str:
    """Strip a trailing `; ...` LLVM comment from a line. Used by the
    drift-guard cross-checks so a `;`-comment mentioning a call
    pattern cannot satisfy the body-vs-registry coherence check
    (audit-fix HIGH-2: pre-tightening, a comment-only mention of
    `call i32 @__helix_arena_push(` would falsely satisfy the
    helper_deps body check). LLVM textual IR uses `;` for line
    comments. None of the helper texts in this file use `";"` inside
    quoted identifiers, so a plain `find(";")` is safe."""
    semi = line.find(";")
    return line if semi < 0 else line[:semi]


def _check_helper_function_table() -> None:
    """Drift guard: each helper's definition text must contain a
    `define internal ...` line declaring the same name as its
    registry key, must contain a `ret ` instruction (so a degenerate
    no-terminator helper is rejected at module load), every callee
    mentioned in its `ffi_declares` must appear in a `call` line
    whose return type matches the declared `ret_ty`, and the
    helper_deps graph must be acyclic.

    Catches:
      - typo'd helper name (`__helix_print_int` key vs.
        `@__helix_printint` define) — the call site would link to a
        nonexistent symbol;
      - helper text with no terminator — `mock_validate_ll` would
        catch this at first use, but the registry guard catches it at
        module load before any test runs;
      - helper text drift away from its declared FFI signature
        (e.g. body says `call i32 @write(...)` but declare says
        `i64 @write(...)`) — produces declare/use signature mismatch
        that `mock_validate_ll` does not detect;
      - helper_deps cycle (A depends on B, B depends on A, or any
        cycle through them) — `_register_helper_function` cannot
        break a true cycle even with its idempotency check, since
        the visited marker is set AFTER recursion. Module-load
        detection produces an actionable diagnostic naming the
        cycle, vs. the raw `RecursionError` Python would otherwise
        leak.
    Fails loudly at module load.
    """
    for name, spec in _HELPER_FUNCTIONS_AUTHORITY.items():
        if not name.startswith("__helix_"):
            raise AssertionError(
                f"helixc.backend.llvm_ir: helper {name!r} is missing "
                f"the reserved `__helix_` prefix — required so user "
                f"function names cannot collide with the helper")
        # Tightened from substring to per-line: a `define internal` line
        # mentioning `@<name>(` somewhere in the helper text is required
        # — a `@<name>(` inside a `; comment` no longer satisfies the
        # guard.
        signature_marker = f"@{name}("
        define_lines = [
            ln for ln in spec.definition.splitlines()
            if ln.lstrip().startswith("define internal ")
            and signature_marker in ln
        ]
        if not define_lines:
            raise AssertionError(
                f"helixc.backend.llvm_ir: helper {name!r} registry "
                f"key does not match a `define internal ...` line in "
                f"its definition text (expected a line starting with "
                f"`define internal ` and containing "
                f"{signature_marker!r})")
        if not any(ln.lstrip().startswith("ret ")
                   for ln in spec.definition.splitlines()):
            raise AssertionError(
                f"helixc.backend.llvm_ir: helper {name!r} definition "
                f"contains no `ret` instruction — a function body "
                f"with no return is malformed LLVM IR")
        # Cross-check: every FFI callee declared as a dependency must
        # appear in at least one `call <ret_ty> <callee>(...)` line in
        # the helper body. A helper that registers `write` as a
        # declare but never calls it (or calls a different callee)
        # has drifted between its FFI registration and its body — the
        # emit-time path would silently emit an unused declare.
        # `_strip_llvm_comment` ensures a `;`-comment mentioning the
        # call pattern does not falsely satisfy the check.
        uncomment_lines = [
            _strip_llvm_comment(ln)
            for ln in spec.definition.splitlines()
        ]
        for entry in spec.ffi_declares:
            call_pattern = f"call {entry.ret_ty} {entry.callee}("
            if not any(call_pattern in ln for ln in uncomment_lines):
                raise AssertionError(
                    f"helixc.backend.llvm_ir: helper {name!r} "
                    f"declares FFI dependency on {entry.callee} "
                    f"(ret_ty={entry.ret_ty!r}) but its body does "
                    f"not contain a matching `call` line — registry "
                    f"and body have drifted")
        # Cross-check: every helper_dep listed must resolve in the
        # registry AND appear in the body as a `call <dep.ret_ty>
        # @<dep>(...)` line — same drift discipline as the FFI
        # check above. Same `;`-comment exclusion.
        for dep_name in spec.helper_deps:
            dep_spec = _HELPER_FUNCTIONS_AUTHORITY.get(dep_name)
            if dep_spec is None:
                raise AssertionError(
                    f"helixc.backend.llvm_ir: helper {name!r} "
                    f"declares helper-dep on {dep_name!r} but no "
                    f"such helper is registered (known: "
                    f"{sorted(_HELPER_FUNCTIONS_AUTHORITY)})")
            dep_call_pattern = (
                f"call {dep_spec.ret_ty} @{dep_name}(")
            if not any(dep_call_pattern in ln
                       for ln in uncomment_lines):
                raise AssertionError(
                    f"helixc.backend.llvm_ir: helper {name!r} "
                    f"declares helper-dep on {dep_name!r} "
                    f"(ret_ty={dep_spec.ret_ty!r}) but its body "
                    f"does not contain a matching `call` line — "
                    f"registry and body have drifted")
    # Cycle detection on the helper_deps DAG (DFS with grey/black
    # marker set). Audit-fix HIGH-1: `_register_helper_function`'s
    # idempotency check cannot break a true cycle because the visited
    # marker is added AFTER the recursive `helper_deps` loop.
    # Detecting at module load produces an actionable diagnostic
    # rather than a `RecursionError` at first use.
    _GREY, _BLACK = 1, 2
    state: dict[str, int] = {}

    def _visit(name: str, path: list[str]) -> None:
        if state.get(name) == _BLACK:
            return
        if state.get(name) == _GREY:
            cycle_start = path.index(name)
            cycle = path[cycle_start:] + [name]
            raise AssertionError(
                f"helixc.backend.llvm_ir: helper_deps cycle "
                f"detected: {' -> '.join(cycle)}. A cycle cannot "
                f"be broken by `_register_helper_function`'s "
                f"idempotency check (the visited marker is added "
                f"after the recursive walk).")
        state[name] = _GREY
        for dep in _HELPER_FUNCTIONS_AUTHORITY[name].helper_deps:
            _visit(dep, path + [name])
        state[name] = _BLACK

    for helper_name in _HELPER_FUNCTIONS_AUTHORITY:
        _visit(helper_name, [])


_check_helper_function_table()
_check_module_global_table()


class _FnEmitter:
    """Emits the LLVM IR for one `tir.FnIR`.

    Stage 200 handled a single straight-line block; Stage 202 adds
    control flow — every tir block becomes a labelled LLVM basic
    block, tir BR / COND_BR become LLVM `br`, and a block's tir
    parameters become LLVM `phi` nodes collecting the matching branch
    argument from each predecessor.

    `operand` maps a tir SSA value id to its LLVM operand text — an
    inline integer literal (a CONST_INT) or an `%vN` register. It is
    fully populated by `_prepass` BEFORE any block is emitted, so a
    loop-header phi can reference a value defined later on the
    back-edge (LLVM textual IR permits the forward reference).

    `var_slots` / `array_slots` map each mutable local / stack array's
    name to its `alloca` pointer register and type (also populated by
    `_prepass`); the allocas are hoisted to the entry block.

    `ffi_declares` collects, during emission, the module-scope
    `declare` each FFI_CALL needs; `string_globals` likewise collects
    the read-only string constants a TRAP needs; `emit_module` emits
    both deduped sets."""

    def __init__(self, fn: tir.FnIR, *,
                 trace_fn_ids: Optional[Mapping[str, int]] = None):
        self.fn = fn
        # Per-module trace-event fn_name -> i32 id table, populated by
        # `_intern_trace_fn_ids` in `emit_module` before any emitter
        # runs. `None` means no module-level interning was done (used
        # by single-function callers like `emit_function`); a TRACE
        # op in that case fails closed at emit time with an explicit
        # diagnostic. The table is SHARED across all `_FnEmitter`
        # instances in one `emit_module` call so a fn_name appearing
        # in one function's TRACE op gets the same id as the same
        # fn_name in another function. Typed as `Mapping` (read-only
        # view) so a per-op handler cannot accidentally mutate the
        # shared table.
        self.trace_fn_ids: Optional[Mapping[str, int]] = trace_fn_ids
        # tir.Value.id -> LLVM operand text
        self.operand: dict[int, str] = {}
        # mutable-local name -> (alloca register, LLVM cell type),
        # populated by `_prepass`; see `_register_alloc_var`.
        self.var_slots: dict[str, tuple[str, str]] = {}
        # stack-array name -> (alloca register, LLVM element type,
        # length), populated by `_prepass`; see `_register_alloc_array`.
        self.array_slots: dict[str, tuple[str, str, int]] = {}
        # counter for the `%gep.N` element-address registers that a
        # LOAD_ELEM / STORE_ELEM emits.
        self.gep_count: int = 0
        # counter for `%trace_keepalive.N` bitcast registers that
        # TRACE_EXIT emits to force a real LLVM use of its optional
        # operand (mirrors x86's always-load liveness semantics).
        self.trace_keepalive_count: int = 0
        # extern symbol -> its module-scope `declare` line, filled as
        # FFI_CALLs (and TRAP's write/exit) are emitted; `emit_module`
        # collects + dedups these.
        self.ffi_declares: dict[str, str] = {}
        # content-addressed global name -> its `... = constant ...`
        # line, filled as TRAP panic messages are emitted; collected +
        # deduped by `emit_module`.
        self.string_globals: dict[str, str] = {}
        # internal helper-function names this function depends on (e.g.
        # `__helix_print_int` for a print_int PRINT). `emit_module`
        # collects the union across all emitters and emits each helper's
        # `define internal ...` text exactly once. See `_HELPER_FUNCTIONS`
        # for the registry.
        self.helper_functions: set[str] = set()
        # module-scope global names this function (transitively, via
        # the helpers it registered) depends on. `emit_module` collects
        # the union across emitters and emits each global exactly once.
        # See `_MODULE_GLOBALS` for the registry.
        self.module_globals: set[str] = set()

    @staticmethod
    def _block_label(block_id: int) -> str:
        return f"bb{block_id}"

    def _ref(self, v: tir.Value) -> str:
        """LLVM operand text for a tir value. Every value the function
        defines is registered by `_prepass`; a missing one is genuinely
        undefined (defined by no op, parameter, or block parameter)."""
        text = self.operand.get(v.id)
        if text is None:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: value v{v.id} is referenced "
                f"but is defined by no op, function parameter, or block "
                f"parameter"
            )
        return text

    def _prepass(self) -> None:
        """Register the LLVM operand text of every value the function
        defines — function params, block params, op results — before
        emission, so a forward reference (a loop-header phi citing a
        back-edge value) resolves. CONST_INT results become inline
        literals; everything else becomes an `%vN` register."""
        for p in self.fn.params:
            self.operand[p.id] = f"%v{p.id}"
        for block in self.fn.blocks:
            for p in block.params:
                self.operand[p.id] = f"%v{p.id}"
            for op in block.ops:
                if op.kind == tir.OpKind.CONST_INT:
                    self._register_const_int(op)
                elif op.kind == tir.OpKind.ALLOC_VAR:
                    self._register_alloc_var(op)
                elif op.kind == tir.OpKind.ALLOC_ARRAY:
                    self._register_alloc_array(op)
                elif op.kind == tir.OpKind.TRAP:
                    # TRAP defines no LLVM value — it ends in
                    # `unreachable`. A result it carries is SSA
                    # bookkeeping only; leaving it unregistered makes a
                    # stray reference to it fail closed in `_ref`
                    # rather than emit a dangling `%vN`.
                    pass
                else:
                    for r in op.results:
                        # A unit-typed result (a void CALL) is not a
                        # materialized LLVM value — it gets no register.
                        if not isinstance(r.ty, tir.TIRUnit):
                            self.operand[r.id] = f"%v{r.id}"

    def _register_const_int(self, op: tir.Op) -> None:
        """Validate a CONST_INT and record its inline-literal operand."""
        if len(op.results) != 1:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: CONST_INT must have exactly "
                f"one result, got {len(op.results)}"
            )
        result = op.results[0]
        value = op.attrs.get("value")
        # `type(value) is int`, NOT isinstance — a Python `bool` is an
        # int subclass and would `str()` to "True"/"False", emitting
        # malformed IR. A real boolean constant is a CONST_BOOL op.
        if type(value) is not int:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: CONST_INT op needs an "
                f"integer 'value' attr (got {value!r}: "
                f"{type(value).__name__})"
            )
        _llvm_int_type(
            result.ty, ctx=f"function {self.fn.name!r} CONST_INT")
        self.operand[result.id] = str(value)

    def _alloc_op_name(self, op: tir.Op, op_label: str) -> str:
        """Validate the shape shared by ALLOC_VAR / ALLOC_ARRAY — no
        result, no operands, a non-empty string `name` attr that is
        not already declared (in either slot table) in this function —
        and return that name. lower_ast mangles shadowed locals to
        unique names, so a duplicate slot name is malformed IR."""
        if op.results:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {op_label} produces no "
                f"result, got {len(op.results)}")
        if op.operands:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {op_label} takes no "
                f"operands, got {len(op.operands)}")
        name = op.attrs.get("name")
        if not isinstance(name, str) or not name:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {op_label} needs a "
                f"non-empty string 'name' attr (got {name!r})")
        if name in self.var_slots or name in self.array_slots:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: slot name {name!r} is "
                f"declared more than once — every ALLOC_VAR / "
                f"ALLOC_ARRAY name must be unique within a function")
        return name

    def _register_alloc_var(self, op: tir.Op) -> None:
        """Validate an ALLOC_VAR and register its stack slot — a
        counter-named `%slot.N` `alloca` pointer plus the cell's LLVM
        type. The `alloca` itself is emitted, hoisted to the entry
        block, by `_emit_allocas`; LOAD_VAR / STORE_VAR resolve the
        slot by name via `_lookup_slot`."""
        name = self._alloc_op_name(op, "ALLOC_VAR")
        llvm_ty = _llvm_int_type(
            op.attrs.get("dtype"),
            ctx=f"function {self.fn.name!r} ALLOC_VAR {name!r} dtype")
        register = f"%slot.{len(self.var_slots)}"
        self.var_slots[name] = (register, llvm_ty)

    def _register_alloc_array(self, op: tir.Op) -> None:
        """Validate an ALLOC_ARRAY and register its stack slot — a
        counter-named `%arr.N` `alloca` pointer, the LLVM element
        type, and the length. The array-typed `alloca` is emitted,
        hoisted to the entry block, by `_emit_allocas`; LOAD_ELEM /
        STORE_ELEM resolve the slot by name via `_lookup_slot` and
        index it with a `getelementptr`."""
        name = self._alloc_op_name(op, "ALLOC_ARRAY")
        elem_ty = _llvm_int_type(
            op.attrs.get("dtype"),
            ctx=f"function {self.fn.name!r} ALLOC_ARRAY {name!r} dtype")
        length = op.attrs.get("length")
        # `type(...) is int`, not isinstance — a Python bool is an int
        # subclass and `[True x i32]` is malformed IR.
        if type(length) is not int or length < 1:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: ALLOC_ARRAY {name!r} needs "
                f"a positive integer 'length' attr (got {length!r})")
        register = f"%arr.{len(self.array_slots)}"
        self.array_slots[name] = (register, elem_ty, length)

    def _lookup_slot(self, op: tir.Op, label: str, table: dict,
                     declarer: str) -> tuple:
        """Resolve a memory op's `name` attr against `table` —
        `var_slots` for LOAD_VAR / STORE_VAR, `array_slots` for
        LOAD_ELEM / STORE_ELEM. Raises `LLVMEmitError` if the name is
        missing / not a string, or names no `declarer`-declared
        variable in this function. Returns the slot tuple."""
        name = op.attrs.get("name")
        if not isinstance(name, str) or not name:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {label} needs a non-empty "
                f"string 'name' attr (got {name!r})")
        slot = table.get(name)
        if slot is None:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {label} references "
                f"variable {name!r}, which no {declarer} in this "
                f"function declares")
        return slot

    def _compute_predecessors(
            self) -> "dict[int, list[tuple[tir.Block, tir.Op]]]":
        """Map each block id -> the list of (predecessor block, branch
        op) edges into it. A BR is one edge (carrying its branch args);
        a COND_BR is one edge per DISTINCT target (carrying no args)."""
        block_ids = {b.id for b in self.fn.blocks}
        preds: "dict[int, list[tuple[tir.Block, tir.Op]]]" = {
            b.id: [] for b in self.fn.blocks}
        for block in self.fn.blocks:
            for op in block.ops:
                if op.kind == tir.OpKind.BR:
                    tgt = op.attrs.get("target_block")
                    if tgt not in block_ids:
                        raise LLVMEmitError(
                            f"function {self.fn.name!r}: BR in bb"
                            f"{block.id} targets unknown block {tgt!r}")
                    preds[tgt].append((block, op))
                elif op.kind == tir.OpKind.COND_BR:
                    seen: set = set()
                    for key in ("true_block", "false_block"):
                        tgt = op.attrs.get(key)
                        if tgt not in block_ids:
                            raise LLVMEmitError(
                                f"function {self.fn.name!r}: COND_BR in "
                                f"bb{block.id} {key} is unknown block "
                                f"{tgt!r}")
                        if tgt not in seen:
                            seen.add(tgt)
                            preds[tgt].append((block, op))
        return preds

    def emit(self) -> str:
        fn = self.fn
        if not fn.blocks:
            raise LLVMEmitError(f"function {fn.name!r} has no blocks")
        ret_ty = _llvm_return_type(
            fn.return_ty, ctx=f"function {fn.name!r} return type")
        param_decls: list[str] = []
        for p in fn.params:
            p_ty = _llvm_int_type(
                p.ty, ctx=f"function {fn.name!r} parameter")
            param_decls.append(f"{p_ty} %v{p.id}")
        self._prepass()
        preds = self._compute_predecessors()
        entry = fn.blocks[0]
        if preds.get(entry.id):
            raise LLVMEmitError(
                f"function {fn.name!r}: the entry block bb{entry.id} is "
                f"a branch target — LLVM forbids branching to a "
                f"function's entry block")
        if entry.params:
            raise LLVMEmitError(
                f"function {fn.name!r}: the entry block bb{entry.id} "
                f"carries block parameters — the entry block receives "
                f"the function parameters, not phi inputs")
        lines: list[str] = [
            f"define {ret_ty} {_llvm_global_name(fn.name)}"
            f"({', '.join(param_decls)}) {{"
        ]
        for i, block in enumerate(fn.blocks):
            lines.extend(
                self._emit_block(block, preds, is_entry=(i == 0)))
        lines.append("}")
        return "\n".join(lines)

    def _emit_allocas(self) -> list[str]:
        """The `alloca` instruction for every mutable local and stack
        array, in ALLOC_VAR / ALLOC_ARRAY encounter order
        (deterministic — a Stage 207 parity prerequisite). `align` is
        omitted, so LLVM uses each type's ABI-natural alignment."""
        lines = [f"  {register} = alloca {llvm_ty}"
                 for register, llvm_ty in self.var_slots.values()]
        lines += [
            f"  {register} = alloca [{length} x {elem_ty}]"
            for register, elem_ty, length in self.array_slots.values()]
        return lines

    def _emit_block(
            self, block: tir.Block,
            preds: "dict[int, list[tuple[tir.Block, tir.Op]]]",
            *, is_entry: bool) -> list[str]:
        lines = [f"{self._block_label(block.id)}:"]
        if is_entry:
            # Every mutable local's `alloca` is hoisted to the top of
            # the entry block: the LLVM convention, and — the entry
            # block dominating every other — it lets a LOAD_VAR /
            # STORE_VAR in any block reference the slot pointer.
            lines.extend(self._emit_allocas())
        lines.extend(self._emit_phis(block, preds.get(block.id, [])))
        saw_terminator = False
        for op in block.ops:
            if saw_terminator:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: op {op.kind.value} in "
                    f"bb{block.id} follows the block terminator — "
                    f"unreachable code")
            text = self._emit_op(op)
            if text is not None:
                # `_emit_op` may return several newline-joined
                # instruction lines (a LOAD_ELEM / STORE_ELEM lowers to
                # a GEP plus a load / store); indent each.
                lines.extend(f"  {ln}" for ln in text.split("\n"))
            if op.kind in (tir.OpKind.RETURN, tir.OpKind.BR,
                           tir.OpKind.COND_BR, tir.OpKind.TRAP):
                saw_terminator = True
        if not saw_terminator:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: block bb{block.id} has no "
                f"terminator — every LLVM basic block must end with "
                f"RETURN, BR, COND_BR, or TRAP")
        return lines

    def _emit_phis(
            self, block: tir.Block,
            block_preds: "list[tuple[tir.Block, tir.Op]]") -> list[str]:
        """Emit a `phi` for each tir block parameter, collecting the
        matching branch argument from every predecessor."""
        if not block.params:
            return []
        if not block_preds:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: block bb{block.id} has "
                f"{len(block.params)} block parameter(s) but no "
                f"predecessors — a phi node needs at least one incoming "
                f"edge")
        # An LLVM phi requires exactly one incoming edge per DISTINCT
        # predecessor block — a duplicate predecessor label is malformed
        # IR. Unreachable today (a block has one terminator, so it
        # cannot branch to one target twice), but the guard keeps the
        # phi emitter sound if a future multi-edge construct lands.
        pred_ids = [pb.id for pb, _ in block_preds]
        if len(pred_ids) != len(set(pred_ids)):
            raise LLVMEmitError(
                f"function {self.fn.name!r}: block bb{block.id} has a "
                f"duplicate predecessor (block ids {sorted(pred_ids)}) — "
                f"a phi node needs exactly one incoming edge per "
                f"predecessor block")
        lines: list[str] = []
        for i, param in enumerate(block.params):
            pty = _llvm_int_type(
                param.ty,
                ctx=f"function {self.fn.name!r} bb{block.id} param {i}")
            incomings: list[str] = []
            for pred_block, br_op in block_preds:
                if br_op.kind != tir.OpKind.BR:
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: block bb{block.id} "
                        f"has block parameters, but predecessor bb"
                        f"{pred_block.id} reaches it via "
                        f"{br_op.kind.value}, which carries no branch "
                        f"arguments — only BR can supply phi inputs")
                if len(br_op.operands) != len(block.params):
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: BR from bb"
                        f"{pred_block.id} to bb{block.id} passes "
                        f"{len(br_op.operands)} argument(s) but the "
                        f"target block has {len(block.params)} "
                        f"parameter(s)")
                arg = br_op.operands[i]
                # The incoming value's LLVM type must equal the phi
                # node's (the block parameter's) type — otherwise the
                # emitted `phi {pty} [ %v..., ... ]` references a
                # wrong-width register: malformed IR. Every other
                # operand-consuming op type-checks; the phi path must
                # too.
                arg_ty = _llvm_int_type(
                    arg.ty,
                    ctx=(f"function {self.fn.name!r} bb{block.id} param "
                         f"{i} incoming from bb{pred_block.id}"))
                if arg_ty != pty:
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: BR from bb"
                        f"{pred_block.id} to bb{block.id} passes an "
                        f"argument of LLVM type {arg_ty} for block "
                        f"parameter {i} (type {pty}) — a phi incoming "
                        f"must match the parameter type")
                incomings.append(
                    f"[ {self._ref(arg)}, "
                    f"%{self._block_label(pred_block.id)} ]")
            lines.append(
                f"  %v{param.id} = phi {pty} {', '.join(incomings)}")
        return lines

    def _emit_gep(self, index: tir.Value, array_reg: str,
                  elem_ty: str, length: int) -> tuple[str, str]:
        """Emit a `getelementptr` for the address of one stack-array
        element. Returns `(gep_instruction, address_register)`.

        The element index may be any integer-typed value — LLVM
        permits a mixed-width GEP index. `inbounds` is deliberately
        omitted: the LLVM backend does not assume the index is
        bounds-checked (whether Helix bounds-checks array access is a
        Stage 207 parity decision), and a plain `getelementptr` is
        well-defined pointer arithmetic regardless."""
        idx_ty = _llvm_int_type(
            index.ty, ctx=f"function {self.fn.name!r} array index")
        addr = f"%gep.{self.gep_count}"
        self.gep_count += 1
        gep = (f"{addr} = getelementptr [{length} x {elem_ty}], "
               f"ptr {array_reg}, i64 0, {idx_ty} {self._ref(index)}")
        return gep, addr

    def _next_keepalive_idx(self) -> int:
        """Allocate a fresh `%trace_keepalive.N` index — used by the
        TRACE_EXIT handler to force an LLVM-level use of the
        optional operand. N increments per TRACE_EXIT with an
        operand within this function, so concurrent TRACE_EXITs
        get distinct keepalive register names."""
        idx = self.trace_keepalive_count
        self.trace_keepalive_count += 1
        return idx

    def _validate_path_attr(self, kind_label: str,
                            path: object) -> str:
        """Validate a user-supplied filesystem-path attr: must be a
        non-NUL-containing string. Shared by `write_file` and
        `read_file_to_arena` PRINT sub-kinds — both call `open(2)`
        which reads the path as a C-string and stops at the first
        NUL byte, so an embedded NUL would SILENTLY truncate the
        target (a path `"/tmp/a\\0/etc/passwd"` would open
        `/tmp/a`). Fail closed.

        `kind_label` is the user-facing op name (`"write_file"` /
        `"read_file_to_arena"`) used in the diagnostic. Returns the
        validated string."""
        if not isinstance(path, str):
            raise LLVMEmitError(
                f"function {self.fn.name!r}: a {kind_label} PRINT "
                f"needs a string 'path' attr (got "
                f"{type(path).__name__})")
        if "\x00" in path:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: a {kind_label} PRINT "
                f"'path' attr contains an embedded NUL byte at "
                f"position {path.index(chr(0))} — open() reads a "
                f"C-string, so a NUL would silently truncate the "
                f"path")
        return path

    def _emit_call(self, op: tir.Op) -> str:
        """Lower a CALL or FFI_CALL to an LLVM `call`. The two share
        every emission detail — a value call `%vN = call <ty> @t(...)`
        or a void call — and differ only in that an FFI_CALL targets
        an extern symbol, so it ALSO registers a module-scope
        `declare` (collected by `emit_module`) via
        `_register_ffi_declare`."""
        is_ffi = op.kind == tir.OpKind.FFI_CALL
        label = "FFI_CALL" if is_ffi else "CALL"
        target = op.attrs.get("target")
        if not isinstance(target, str) or not target:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {label} needs a non-empty "
                f"string 'target' attr (got {target!r})")
        if len(op.results) > 1:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {label} to {target!r} has "
                f"{len(op.results)} results — an LLVM call produces at "
                f"most one value")
        arg_tys: list[str] = []
        args: list[str] = []
        for i, arg in enumerate(op.operands):
            arg_ty = _llvm_int_type(
                arg.ty,
                ctx=f"function {self.fn.name!r} {label} {target!r} "
                    f"argument {i}")
            arg_tys.append(arg_ty)
            args.append(f"{arg_ty} {self._ref(arg)}")
        callee = _llvm_global_name(target)
        arglist = ", ".join(args)
        # A call with no result, or a unit-typed result, is a void
        # call — `()` is not a materialized LLVM value.
        if not op.results or isinstance(op.results[0].ty, tir.TIRUnit):
            ret_ty = "void"
            call_line = f"call void {callee}({arglist})"
        else:
            result = op.results[0]
            ret_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} {label} {target!r} "
                    f"result")
            call_line = (f"%v{result.id} = call {ret_ty} "
                         f"{callee}({arglist})")
        if is_ffi:
            self._register_ffi_declare(target, callee, ret_ty, arg_tys)
        return call_line

    def _register_ffi_declare(self, target: str, callee: str,
                              ret_ty: str,
                              arg_tys: list[str]) -> None:
        """Record the module-scope `declare` an FFI_CALL needs. An
        extern symbol called more than once must agree on its
        signature — a mismatch is malformed IR, so fail closed.
        `emit_module` collects these across functions and emits the
        deduped declares."""
        decl = f"declare {ret_ty} {callee}({', '.join(arg_tys)})"
        existing = self.ffi_declares.get(target)
        if existing is not None and existing != decl:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: FFI symbol {target!r} is "
                f"called with two different signatures — {existing!r} "
                f"vs {decl!r}")
        self.ffi_declares[target] = decl

    def _register_helper_function(self, name: str) -> None:
        """Record that this function calls the internal helper `name`.

        Looks the helper up in `_HELPER_FUNCTIONS` and records each of
        its FFI declares through `_register_ffi_declare` (so a helper's
        extern dedups against an FFI_CALL extern that names the same
        symbol — there is only one `declare` per symbol in the emitted
        module). The helper's `define internal ...` text itself is
        emitted exactly once by `emit_module` across the union of
        per-function helper sets.

        Raises `LLVMEmitError` (not `KeyError`) when `name` is not
        registered — keeps the emit-time fault surface consistent with
        every other "this op needs X but X is missing" path in the
        backend."""
        # Idempotency: if this helper is already registered, return
        # early. This is NOT a cycle break — the visited marker
        # (`self.helper_functions.add(name)`) only fires AFTER the
        # recursive `helper_deps` walk completes, so a true
        # registry-level cycle would still recurse to RecursionError.
        # `_check_helper_function_table` runs DFS-based cycle
        # detection at module load (audit-fix HIGH-1) so a cyclic
        # registry never reaches this code path. The idempotency
        # check here serves diamond dependencies (A->B, A->C, B->C
        # registers C once) and double-registration from caller
        # code, both of which are benign.
        if name in self.helper_functions:
            return
        spec = _HELPER_FUNCTIONS.get(name)
        if spec is None:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: internal helper {name!r} "
                f"is not registered (known helpers: "
                f"{sorted(_HELPER_FUNCTIONS)})")
        # Transactional: register transitive helper deps FIRST (each
        # one recursively pulls in its own FFI / globals / deps),
        # then this helper's FFI declares, then its module-globals,
        # then mark this helper as needed. If any step fails, the
        # deps stay registered (harmless — they're emitted but their
        # caller never makes it into the module) and THIS helper is
        # left out of `helper_functions` so emit_module won't emit
        # its `define`.
        for dep_name in spec.helper_deps:
            self._register_helper_function(dep_name)
        for entry in spec.ffi_declares:
            self._register_ffi_declare(
                entry.target, entry.callee, entry.ret_ty,
                list(entry.arg_tys))
        for global_name in spec.module_globals:
            # `_check_module_global_table` already enforced at module
            # load that every helper's `module_globals` name resolves
            # in `_MODULE_GLOBALS`; a miss here would mean a runtime
            # registry mutation moved the global out from under the
            # helper (e.g. a test that pokes
            # `_MODULE_GLOBALS_AUTHORITY` without restoring it).
            if global_name not in _MODULE_GLOBALS:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: helper {name!r} "
                    f"references module-global {global_name!r} but "
                    f"it is not registered (known globals: "
                    f"{sorted(_MODULE_GLOBALS)})")
            self.module_globals.add(global_name)
        self.helper_functions.add(name)

    def _register_string(self, data: bytes) -> tuple[str, int]:
        """Register a read-only string constant — a TRAP panic
        message, or a STR_PTR / STR_BYTE string literal. Returns
        `(global_name, byte_length)`. The global is content-addressed
        — its name is a hash of the bytes — so two identical strings
        dedup to one module global and the name is stable across
        functions (`emit_module` collects + dedups the per-function
        `string_globals`)."""
        name = f"@.helix.str.{hashlib.sha256(data).hexdigest()[:16]}"
        if name not in self.string_globals:
            self.string_globals[name] = (
                f"{name} = private unnamed_addr constant "
                f"[{len(data)} x i8] {_llvm_cstring(data)}")
        return name, len(data)

    def _result_unpack(self, op: tir.Op, label: str) -> tuple[str, int]:
        """Validate a RESULT_TAG / RESULT_PAYLOAD op — exactly one i64
        operand (the packed Result) and one i32 result — and return
        `(packed_operand_ref, result_id)`. A packed Result is one i64;
        its tag and payload are each i32 (the Stage 49 convention)."""
        if len(op.results) != 1:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {label} must have exactly "
                f"one result, got {len(op.results)}")
        if len(op.operands) != 1:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {label} expects exactly "
                f"one operand (the packed Result), got "
                f"{len(op.operands)}")
        packed = op.operands[0]
        packed_ty = _llvm_int_type(
            packed.ty,
            ctx=f"function {self.fn.name!r} {label} operand")
        if packed_ty != "i64":
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {label} operand has LLVM "
                f"type {packed_ty}, but a packed Result is an i64")
        result = op.results[0]
        res_ty = _llvm_int_type(
            result.ty, ctx=f"function {self.fn.name!r} {label} result")
        if res_ty != "i32":
            raise LLVMEmitError(
                f"function {self.fn.name!r}: {label} result has LLVM "
                f"type {res_ty}, but a Result tag/payload is an i32")
        return self._ref(packed), result.id

    def _emit_op(self, op: tir.Op) -> Optional[str]:
        """Emit one op's LLVM instruction text — `None` when the op
        materializes no instruction (CONST_INT, ALLOC_VAR,
        ALLOC_ARRAY), otherwise the instruction line. An op that lowers
        to several LLVM instructions (LOAD_ELEM, STORE_ELEM) returns
        them as a newline-joined block, un-indented — `_emit_block`
        indents each line."""
        kind = op.kind
        if kind == tir.OpKind.CONST_INT:
            return None  # registered as an inline literal by _prepass
        if kind in _LLVM_SCALAR_BINOPS or kind in _LLVM_SIGNED_BINOPS:
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: {kind.value} must have "
                    f"exactly one result, got {len(op.results)}"
                )
            if len(op.operands) != 2:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: {kind.value} expects "
                    f"two operands, got {len(op.operands)}"
                )
            result = op.results[0]
            ctx = f"function {self.fn.name!r} {kind.value}"
            ty = _llvm_int_type(result.ty, ctx=ctx)
            # LLVM requires a binary op's two operands and its result to
            # share one type. The host IR does not structurally
            # guarantee that, so verify it — a mismatch would otherwise
            # silently emit malformed IR (an `add i32` referencing an
            # i64 register). This also covers the shifts: LLVM requires
            # a shift's value and shift-amount operands to share a type.
            for i, operand in enumerate(op.operands):
                operand_ty = _llvm_int_type(
                    operand.ty, ctx=f"{ctx} operand {i}")
                if operand_ty != ty:
                    raise LLVMEmitError(
                        f"{ctx}: operand {i} has LLVM type {operand_ty} "
                        f"but the result is {ty} — LLVM binary ops "
                        f"require matching operand and result types"
                    )
            lhs = self._ref(op.operands[0])
            rhs = self._ref(op.operands[1])
            if kind in _LLVM_SCALAR_BINOPS:
                mnemonic = _LLVM_SCALAR_BINOPS[kind]
            else:
                # DIV / MOD / SHR each have a signed and an unsigned
                # LLVM form, chosen by signedness. Fail closed when the
                # signedness-relevant values disagree — a mixed
                # signed/unsigned combination makes the choice
                # ambiguous and, left unchecked, could silently diverge
                # from x86_64.py:
                #   - DIV / MOD: the two operands must agree (`sdiv` vs
                #     `udiv`, `srem` vs `urem`);
                #   - SHR: the shifted VALUE (operand 0) and the result
                #     must agree (`ashr` vs `lshr`). The shift COUNT
                #     (operand 1) is exempt — a count's signedness
                #     never affects the result.
                if kind in (tir.OpKind.DIV, tir.OpKind.MOD):
                    _require_same_signedness(
                        op.operands[0], op.operands[1], ctx=ctx)
                elif kind == tir.OpKind.SHR:
                    _require_same_signedness(
                        op.operands[0], result, ctx=ctx)
                signed_mn, unsigned_mn = _LLVM_SIGNED_BINOPS[kind]
                mnemonic = (unsigned_mn
                            if _is_unsigned_int(op.operands[0].ty)
                            else signed_mn)
            # NOTE (Stage 207 parity): the integer binops are emitted in
            # their plain LLVM form, leaving three UB questions to the
            # Stage 207 parity gate against x86_64.py:
            #   - add/sub/mul carry no `nsw`/`nuw` (they wrap, two's-
            #     complement);
            #   - sdiv/udiv/srem/urem are UB on a zero divisor, and
            #     `sdiv INT_MIN, -1` overflows — x86_64.py's hardware
            #     `div` faults (#DE) on those, so the paths may diverge;
            #   - shl/lshr/ashr yield `poison` when the shift amount is
            #     >= the type width, where x86 masks the count.
            # Emitting the plain ops is correct for in-range inputs; the
            # parity gate decides whether explicit guards are needed.
            return f"%v{result.id} = {mnemonic} {ty} {lhs}, {rhs}"
        if kind == tir.OpKind.RETURN:
            if not op.operands:
                return "ret void"
            if len(op.operands) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: RETURN expects zero or "
                    f"one operands, got {len(op.operands)}"
                )
            v = op.operands[0]
            ty = _llvm_int_type(
                v.ty, ctx=f"function {self.fn.name!r} RETURN value")
            return f"ret {ty} {self._ref(v)}"
        if kind == tir.OpKind.BR:
            # target validity already checked by _compute_predecessors.
            tgt = op.attrs.get("target_block")
            return f"br label %{self._block_label(tgt)}"
        if kind == tir.OpKind.COND_BR:
            if len(op.operands) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: COND_BR expects exactly "
                    f"one operand (the condition), got "
                    f"{len(op.operands)}")
            cond = op.operands[0]
            cond_ty = _llvm_int_type(
                cond.ty,
                ctx=f"function {self.fn.name!r} COND_BR condition")
            if cond_ty != "i1":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: COND_BR condition has "
                    f"LLVM type {cond_ty}, but LLVM `br` requires an i1 "
                    f"(bool) condition")
            # Both keys were validated present by _compute_predecessors;
            # `.get` keeps this consistent with the BR path above.
            true_lbl = self._block_label(op.attrs.get("true_block"))
            false_lbl = self._block_label(op.attrs.get("false_block"))
            return (f"br i1 {self._ref(cond)}, label %{true_lbl}, "
                    f"label %{false_lbl}")
        if kind in _LLVM_ICMP_PREDS:
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: {kind.value} must have "
                    f"exactly one result, got {len(op.results)}")
            if len(op.operands) != 2:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: {kind.value} expects "
                    f"two operands, got {len(op.operands)}")
            result = op.results[0]
            ctx = f"function {self.fn.name!r} {kind.value}"
            # A comparison's result is a bool — LLVM i1.
            res_ty = _llvm_int_type(result.ty, ctx=f"{ctx} result")
            if res_ty != "i1":
                raise LLVMEmitError(
                    f"{ctx}: result has LLVM type {res_ty}, but a "
                    f"comparison produces a bool (i1)")
            a, b = op.operands
            a_ty = _llvm_int_type(a.ty, ctx=f"{ctx} operand 0")
            b_ty = _llvm_int_type(b.ty, ctx=f"{ctx} operand 1")
            if a_ty != b_ty:
                raise LLVMEmitError(
                    f"{ctx}: operands have mismatched LLVM types "
                    f"{a_ty} and {b_ty} — a comparison's two operands "
                    f"must share one type")
            signed_pred, unsigned_pred = _LLVM_ICMP_PREDS[kind]
            # An ORDERED comparison (signed_pred != unsigned_pred, e.g.
            # slt vs ult) is chosen by operand signedness, so a mixed
            # signed/unsigned operand pair makes the predicate
            # ambiguous — fail closed. `eq` / `ne` are sign-agnostic
            # (both predicates equal) and stay exempt.
            if signed_pred != unsigned_pred:
                _require_same_signedness(a, b, ctx=ctx)
            # The icmp predicate's signedness follows the OPERAND dtype
            # (a Helix u8/u16/u32/u64/usize operand -> unsigned).
            pred = (unsigned_pred if _is_unsigned_int(a.ty)
                    else signed_pred)
            return (f"%v{result.id} = icmp {pred} {a_ty} "
                    f"{self._ref(a)}, {self._ref(b)}")
        if kind == tir.OpKind.SELECT:
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: SELECT must have exactly "
                    f"one result, got {len(op.results)}")
            if len(op.operands) != 3:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: SELECT expects three "
                    f"operands (cond, then, else), got "
                    f"{len(op.operands)}")
            result = op.results[0]
            ctx = f"function {self.fn.name!r} SELECT"
            cond, t_val, f_val = op.operands
            cond_ty = _llvm_int_type(cond.ty, ctx=f"{ctx} condition")
            if cond_ty != "i1":
                raise LLVMEmitError(
                    f"{ctx}: condition has LLVM type {cond_ty}, but "
                    f"LLVM `select` requires an i1 (bool) condition")
            ty = _llvm_int_type(result.ty, ctx=ctx)
            for i, arm in ((1, t_val), (2, f_val)):
                arm_ty = _llvm_int_type(arm.ty, ctx=f"{ctx} operand {i}")
                if arm_ty != ty:
                    raise LLVMEmitError(
                        f"{ctx}: operand {i} has LLVM type {arm_ty} but "
                        f"the result is {ty} — the two SELECT arms and "
                        f"the result must share one type")
            return (f"%v{result.id} = select i1 {self._ref(cond)}, "
                    f"{ty} {self._ref(t_val)}, {ty} {self._ref(f_val)}")
        if kind in (tir.OpKind.NEG, tir.OpKind.BIT_NOT):
            # The two unary integer ops. LLVM has a dedicated
            # instruction for neither, so each is emitted in its
            # canonical two-operand form; the arity / type checks are
            # identical, so they share one branch.
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: {kind.value} must have "
                    f"exactly one result, got {len(op.results)}")
            if len(op.operands) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: {kind.value} expects "
                    f"one operand, got {len(op.operands)}")
            result = op.results[0]
            ctx = f"function {self.fn.name!r} {kind.value}"
            ty = _llvm_int_type(result.ty, ctx=ctx)
            operand = op.operands[0]
            operand_ty = _llvm_int_type(operand.ty, ctx=f"{ctx} operand")
            if operand_ty != ty:
                raise LLVMEmitError(
                    f"{ctx}: operand has LLVM type {operand_ty} but the "
                    f"result is {ty} — they must share one type")
            operand_ref = self._ref(operand)
            if kind == tir.OpKind.NEG:
                # LLVM has no integer-negate instruction; `sub <ty> 0,
                # x` is the canonical form (two's-complement, wrapping).
                return f"%v{result.id} = sub {ty} 0, {operand_ref}"
            # BIT_NOT: LLVM has no bitwise-NOT instruction either;
            # `xor <ty> x, -1` is the canonical form — `-1` is all-ones
            # at every integer width, so the xor flips every bit.
            return f"%v{result.id} = xor {ty} {operand_ref}, -1"
        if kind == tir.OpKind.ALLOC_VAR:
            # The `alloca` was hoisted to the entry block by
            # `_emit_allocas` (the slot was registered in `_prepass`).
            # The ALLOC_VAR op itself materializes no instruction at
            # its original position — like a CONST_INT.
            return None
        if kind == tir.OpKind.LOAD_VAR:
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: LOAD_VAR must have "
                    f"exactly one result, got {len(op.results)}")
            if op.operands:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: LOAD_VAR takes no "
                    f"operands, got {len(op.operands)}")
            register, slot_ty = self._lookup_slot(
                op, "LOAD_VAR", self.var_slots, "ALLOC_VAR")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} LOAD_VAR result")
            if res_ty != slot_ty:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: LOAD_VAR result has "
                    f"LLVM type {res_ty} but the variable's cell was "
                    f"allocated as {slot_ty} — a load must read the "
                    f"type the cell holds")
            return f"%v{result.id} = load {slot_ty}, ptr {register}"
        if kind == tir.OpKind.STORE_VAR:
            if op.results:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STORE_VAR produces no "
                    f"result, got {len(op.results)}")
            if len(op.operands) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STORE_VAR expects "
                    f"exactly one operand (the value), got "
                    f"{len(op.operands)}")
            register, slot_ty = self._lookup_slot(
                op, "STORE_VAR", self.var_slots, "ALLOC_VAR")
            value = op.operands[0]
            val_ty = _llvm_int_type(
                value.ty,
                ctx=f"function {self.fn.name!r} STORE_VAR value")
            if val_ty != slot_ty:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STORE_VAR value has "
                    f"LLVM type {val_ty} but the variable's cell was "
                    f"allocated as {slot_ty} — a store must write the "
                    f"type the cell holds")
            return f"store {slot_ty} {self._ref(value)}, ptr {register}"
        if kind == tir.OpKind.ALLOC_ARRAY:
            # The array-typed `alloca` was hoisted to the entry block
            # by `_emit_allocas` (registered in `_prepass`). Like
            # ALLOC_VAR, the op materializes no instruction here.
            return None
        if kind == tir.OpKind.LOAD_ELEM:
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: LOAD_ELEM must have "
                    f"exactly one result, got {len(op.results)}")
            if len(op.operands) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: LOAD_ELEM expects "
                    f"exactly one operand (the index), got "
                    f"{len(op.operands)}")
            register, elem_ty, length = self._lookup_slot(
                op, "LOAD_ELEM", self.array_slots, "ALLOC_ARRAY")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} LOAD_ELEM result")
            if res_ty != elem_ty:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: LOAD_ELEM result has "
                    f"LLVM type {res_ty} but the array's elements are "
                    f"{elem_ty} — a load must read the element type")
            gep, addr = self._emit_gep(
                op.operands[0], register, elem_ty, length)
            return f"{gep}\n%v{result.id} = load {elem_ty}, ptr {addr}"
        if kind == tir.OpKind.STORE_ELEM:
            if op.results:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STORE_ELEM produces no "
                    f"result, got {len(op.results)}")
            if len(op.operands) != 2:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STORE_ELEM expects "
                    f"exactly two operands (index, value), got "
                    f"{len(op.operands)}")
            register, elem_ty, length = self._lookup_slot(
                op, "STORE_ELEM", self.array_slots, "ALLOC_ARRAY")
            value = op.operands[1]
            val_ty = _llvm_int_type(
                value.ty,
                ctx=f"function {self.fn.name!r} STORE_ELEM value")
            if val_ty != elem_ty:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STORE_ELEM value has "
                    f"LLVM type {val_ty} but the array's elements are "
                    f"{elem_ty} — a store must write the element type")
            gep, addr = self._emit_gep(
                op.operands[0], register, elem_ty, length)
            return (f"{gep}\nstore {elem_ty} {self._ref(value)}, "
                    f"ptr {addr}")
        if kind in (tir.OpKind.CALL, tir.OpKind.FFI_CALL):
            return self._emit_call(op)
        if kind == tir.OpKind.RESULT_PACK:
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: RESULT_PACK must have "
                    f"exactly one result, got {len(op.results)}")
            if len(op.operands) != 2:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: RESULT_PACK expects "
                    f"exactly two operands (tag, payload), got "
                    f"{len(op.operands)}")
            tag, payload = op.operands
            tag_ty = _llvm_int_type(
                tag.ty,
                ctx=f"function {self.fn.name!r} RESULT_PACK tag")
            payload_ty = _llvm_int_type(
                payload.ty,
                ctx=f"function {self.fn.name!r} RESULT_PACK payload")
            if tag_ty != "i32" or payload_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: RESULT_PACK tag/payload "
                    f"have LLVM types {tag_ty}/{payload_ty}, but both "
                    f"must be i32")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} RESULT_PACK result")
            if res_ty != "i64":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: RESULT_PACK result has "
                    f"LLVM type {res_ty}, but a packed Result is an i64")
            rid = result.id
            # packed = (zext tag) << 32 | (zext payload). `zext`
            # zero-fills the high half, so a zext'd i32 payload already
            # equals `payload & 0xFFFFFFFF` as an i64 — no mask needed.
            t0, t1, t2 = f"%v{rid}.t0", f"%v{rid}.t1", f"%v{rid}.t2"
            return (f"{t0} = zext i32 {self._ref(tag)} to i64\n"
                    f"{t1} = shl i64 {t0}, 32\n"
                    f"{t2} = zext i32 {self._ref(payload)} to i64\n"
                    f"%v{rid} = or i64 {t1}, {t2}")
        if kind == tir.OpKind.RESULT_TAG:
            packed_ref, rid = self._result_unpack(op, "RESULT_TAG")
            # tag = packed >> 32 (logical shift), narrowed to i32.
            tmp = f"%v{rid}.t0"
            return (f"{tmp} = lshr i64 {packed_ref}, 32\n"
                    f"%v{rid} = trunc i64 {tmp} to i32")
        if kind == tir.OpKind.RESULT_PAYLOAD:
            packed_ref, rid = self._result_unpack(op, "RESULT_PAYLOAD")
            # payload = the low 32 bits of the packed i64.
            return f"%v{rid} = trunc i64 {packed_ref} to i32"
        if kind == tir.OpKind.TRAP:
            if op.operands:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: TRAP takes no "
                    f"operands, got {len(op.operands)}")
            text = op.attrs.get("text", "")
            if not isinstance(text, str):
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: TRAP needs a string "
                    f"'text' attr (got {type(text).__name__})")
            trap_id = op.attrs.get("trap_id", 28501)
            if type(trap_id) is not int:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: TRAP needs an integer "
                    f"'trap_id' attr (got {trap_id!r})")
            # The panic message — rendered byte-identically to
            # x86_64.py so the Stage 207 parity gate sees the same
            # stderr: `panic[<id>]: <text>` followed by a newline.
            message = f"panic[{trap_id}]: {text}\n".encode("utf-8")
            str_name, str_len = self._register_string(message)
            self._register_ffi_declare(
                "write", "@write", "i64", ["i32", "ptr", "i64"])
            self._register_ffi_declare(
                "exit", "@exit", "void", ["i32"])
            # write(2, msg, len); exit(trap_id & 0xFF); unreachable —
            # the exit status is the low byte of the trap id, matching
            # x86_64.py. `unreachable` is the block terminator: a TRAP
            # never returns. A TRAP may carry a result for SSA
            # bookkeeping; it is unreachable and unreferenced, so
            # nothing defines it (its `%vN` never reaches the output).
            return (f"call i64 @write(i32 2, ptr {str_name}, "
                    f"i64 {str_len})\n"
                    f"call void @exit(i32 {trap_id & 0xFF})\n"
                    f"unreachable")
        if kind == tir.OpKind.STR_PTR:
            if op.operands:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STR_PTR takes no "
                    f"operands, got {len(op.operands)}")
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STR_PTR must have "
                    f"exactly one result, got {len(op.results)}")
            text = op.attrs.get("text", "")
            if not isinstance(text, str):
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STR_PTR needs a "
                    f"string 'text' attr (got {type(text).__name__})")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} STR_PTR result")
            if res_ty != "i64":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STR_PTR result has "
                    f"LLVM type {res_ty}, but a string pointer is a "
                    f"u64 (i64)")
            str_name, _ = self._register_string(text.encode("utf-8"))
            # The result is the literal's address as an integer.
            return f"%v{result.id} = ptrtoint ptr {str_name} to i64"
        if kind == tir.OpKind.STR_BYTE:
            if len(op.operands) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STR_BYTE expects "
                    f"exactly one operand (the index), got "
                    f"{len(op.operands)}")
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STR_BYTE must have "
                    f"exactly one result, got {len(op.results)}")
            text = op.attrs.get("text", "")
            if not isinstance(text, str):
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STR_BYTE needs a "
                    f"string 'text' attr (got {type(text).__name__})")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} STR_BYTE result")
            if res_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: STR_BYTE result has "
                    f"LLVM type {res_ty}, but a string byte is an i32")
            index = op.operands[0]
            idx_ty = _llvm_int_type(
                index.ty,
                ctx=f"function {self.fn.name!r} STR_BYTE index")
            data = text.encode("utf-8")
            n = len(data)
            # The byte-access global is the literal + one NUL pad, so
            # the bounds-clamped GEP index (0 when out of range) always
            # lands on a valid byte — even for an empty literal. The
            # bounds check is against the REAL length `n`: an
            # out-of-range index yields 0, matching x86_64.py, and no
            # out-of-bounds memory is ever read.
            str_name, padded_len = self._register_string(
                data + b"\x00")
            rid = result.id
            idx = self._ref(index)
            t0, t1, t2, t3, t4 = (
                f"%v{rid}.t0", f"%v{rid}.t1", f"%v{rid}.t2",
                f"%v{rid}.t3", f"%v{rid}.t4")
            return (
                f"{t0} = icmp ult {idx_ty} {idx}, {n}\n"
                f"{t1} = select i1 {t0}, {idx_ty} {idx}, "
                f"{idx_ty} 0\n"
                f"{t2} = getelementptr [{padded_len} x i8], ptr "
                f"{str_name}, i64 0, {idx_ty} {t1}\n"
                f"{t3} = load i8, ptr {t2}\n"
                f"{t4} = zext i8 {t3} to i32\n"
                f"%v{rid} = select i1 {t0}, i32 {t4}, i32 0")
        if kind == tir.OpKind.ARENA_PUSH:
            # `arena.push` pushes one i32 value into the module-scope
            # arena buffer and returns the new slot's index (or -1 on
            # overflow). The bounds-check + conditional store + cursor-
            # increment is a 4-block LLVM helper — too unwieldy to
            # inline at every call site — so this op lowers to a `call`
            # to `@__helix_arena_push`. The helper transitively pulls
            # in the `@__helix_arena_base` module-global via its
            # `module_globals` entry.
            if len(op.operands) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH takes one "
                    f"operand (the i32 value to push), got "
                    f"{len(op.operands)}")
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH must have "
                    f"exactly one result, got {len(op.results)}")
            value = op.operands[0]
            value_ty = _llvm_int_type(
                value.ty,
                ctx=f"function {self.fn.name!r} ARENA_PUSH operand")
            if value_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH operand "
                    f"has LLVM type {value_ty}, but the helper "
                    f"`__helix_arena_push` takes an i32")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} ARENA_PUSH result")
            if res_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH result "
                    f"has LLVM type {res_ty}, but the arena slot "
                    f"index is an i32")
            self._register_helper_function("__helix_arena_push")
            return (f"%v{result.id} = call i32 "
                    f"@__helix_arena_push(i32 {self._ref(value)})")
        if kind == tir.OpKind.ARENA_GET:
            # `arena.get` reads arena[idx + 1]. Out-of-bounds returns
            # 0 (the helper handles the bounds check); one i32 operand
            # (index), one i32 result (the loaded value).
            if len(op.operands) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_GET takes one "
                    f"operand (the i32 index), got "
                    f"{len(op.operands)}")
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_GET must have "
                    f"exactly one result, got {len(op.results)}")
            idx = op.operands[0]
            idx_ty = _llvm_int_type(
                idx.ty,
                ctx=f"function {self.fn.name!r} ARENA_GET operand")
            if idx_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_GET operand has "
                    f"LLVM type {idx_ty}, but the helper "
                    f"`__helix_arena_get` takes an i32 index")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} ARENA_GET result")
            if res_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_GET result has "
                    f"LLVM type {res_ty}, but the arena cell is an "
                    f"i32")
            self._register_helper_function("__helix_arena_get")
            return (f"%v{result.id} = call i32 "
                    f"@__helix_arena_get(i32 {self._ref(idx)})")
        if kind == tir.OpKind.ARENA_SET:
            # `arena.set` writes value at arena[idx + 1]. Out-of-bounds
            # silently no-ops. Two i32 operands (index, value); TIR
            # says "no result" but x86_64.py tolerates a result slot
            # (and writes 0 into it) — this handler matches that
            # tolerance. The helper always returns i32 0, which the
            # op handler either binds (if op.results) or discards.
            if len(op.operands) != 2:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_SET takes two "
                    f"operands (i32 index, i32 value), got "
                    f"{len(op.operands)}")
            if len(op.results) > 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_SET expects "
                    f"zero or one results, got {len(op.results)}")
            idx, value = op.operands
            idx_ty = _llvm_int_type(
                idx.ty,
                ctx=f"function {self.fn.name!r} ARENA_SET index")
            if idx_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_SET index has "
                    f"LLVM type {idx_ty}, but the helper "
                    f"`__helix_arena_set` takes an i32 index")
            value_ty = _llvm_int_type(
                value.ty,
                ctx=f"function {self.fn.name!r} ARENA_SET value")
            if value_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_SET value has "
                    f"LLVM type {value_ty}, but the helper "
                    f"`__helix_arena_set` takes an i32 value")
            self._register_helper_function("__helix_arena_set")
            call_args = (f"i32 {self._ref(idx)}, "
                         f"i32 {self._ref(value)}")
            if op.results:
                result = op.results[0]
                res_ty = _llvm_int_type(
                    result.ty,
                    ctx=f"function {self.fn.name!r} ARENA_SET result")
                if res_ty != "i32":
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: ARENA_SET result "
                        f"has LLVM type {res_ty}, but the helper "
                        f"returns i32 (always 0)")
                return (f"%v{result.id} = call i32 "
                        f"@__helix_arena_set({call_args})")
            # No result — discard the helper's return.
            return f"call i32 @__helix_arena_set({call_args})"
        if kind == tir.OpKind.ARENA_LEN:
            # `arena.len` returns the cursor (the i32 at slot 0).
            # Zero operands, one i32 result.
            if op.operands:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_LEN takes no "
                    f"operands, got {len(op.operands)}")
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_LEN must have "
                    f"exactly one result, got {len(op.results)}")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} ARENA_LEN result")
            if res_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_LEN result has "
                    f"LLVM type {res_ty}, but the arena cursor is an "
                    f"i32")
            self._register_helper_function("__helix_arena_len")
            return f"%v{result.id} = call i32 @__helix_arena_len()"
        if kind == tir.OpKind.ARENA_PUSH_PAIR:
            # `arena.push_pair` ATOMICALLY writes two i32 values into
            # consecutive arena slots and returns the slot index of
            # `left` (= old cursor). On overflow neither write
            # happens AND the cursor does not advance — atomic-or-none.
            if len(op.operands) != 2:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH_PAIR takes "
                    f"two operands (i32 left, i32 right), got "
                    f"{len(op.operands)}")
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH_PAIR must "
                    f"have exactly one result, got {len(op.results)}")
            left, right = op.operands
            left_ty = _llvm_int_type(
                left.ty,
                ctx=f"function {self.fn.name!r} ARENA_PUSH_PAIR left")
            if left_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH_PAIR left "
                    f"operand has LLVM type {left_ty}, but the helper "
                    f"`__helix_arena_push_pair` takes an i32")
            right_ty = _llvm_int_type(
                right.ty,
                ctx=f"function {self.fn.name!r} ARENA_PUSH_PAIR right")
            if right_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH_PAIR right "
                    f"operand has LLVM type {right_ty}, but the helper "
                    f"`__helix_arena_push_pair` takes an i32")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} ARENA_PUSH_PAIR result")
            if res_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH_PAIR result "
                    f"has LLVM type {res_ty}, but the arena slot index "
                    f"is an i32")
            self._register_helper_function("__helix_arena_push_pair")
            return (f"%v{result.id} = call i32 "
                    f"@__helix_arena_push_pair(i32 {self._ref(left)}, "
                    f"i32 {self._ref(right)})")
        if kind == tir.OpKind.ARENA_PUSH_TRIPLE:
            # `arena.push_triple` is the three-slot counterpart of
            # PUSH_PAIR with the same atomic-or-none contract.
            if len(op.operands) != 3:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH_TRIPLE "
                    f"takes three operands (i32 left, i32 middle, "
                    f"i32 right), got {len(op.operands)}")
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH_TRIPLE "
                    f"must have exactly one result, got "
                    f"{len(op.results)}")
            left, middle, right = op.operands
            for label, operand in (("left", left), ("middle", middle),
                                   ("right", right)):
                operand_ty = _llvm_int_type(
                    operand.ty,
                    ctx=(f"function {self.fn.name!r} ARENA_PUSH_TRIPLE "
                         f"{label}"))
                if operand_ty != "i32":
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: ARENA_PUSH_TRIPLE "
                        f"{label} operand has LLVM type {operand_ty}, "
                        f"but the helper `__helix_arena_push_triple` "
                        f"takes an i32")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} ARENA_PUSH_TRIPLE result")
            if res_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: ARENA_PUSH_TRIPLE "
                    f"result has LLVM type {res_ty}, but the arena "
                    f"slot index is an i32")
            self._register_helper_function("__helix_arena_push_triple")
            return (f"%v{result.id} = call i32 "
                    f"@__helix_arena_push_triple("
                    f"i32 {self._ref(left)}, "
                    f"i32 {self._ref(middle)}, "
                    f"i32 {self._ref(right)})")
        if kind == tir.OpKind.QUOTE:
            # `quote` returns a stable cell handle in [0, NUM_CELLS)
            # — the handle is derived at compile time from the
            # `ast_handle` attr (mod NUM_CELLS, mirroring
            # x86_64.py::QUOTE line 4473). Pure inline emission: a
            # single `add i32 0, <handle>` materialises the constant
            # into the result register; LLVM's instruction combiner
            # folds it away after isel. No operands; one i32 result.
            #
            # The cell array `@__helix_state_base` is NOT registered
            # here — QUOTE only emits a compile-time constant, it
            # does not touch the cell array. SPLICE / MODIFY pull
            # the global in transitively via their `module_globals`.
            if op.operands:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: QUOTE takes no "
                    f"operands, got {len(op.operands)}")
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: QUOTE must have "
                    f"exactly one result, got {len(op.results)}")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} QUOTE result")
            if res_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: QUOTE result has "
                    f"LLVM type {res_ty}, but a reflection-cell "
                    f"handle is an i32")
            ast_handle = op.attrs.get("ast_handle", 0)
            # `type(...) is int` rejects bool (which is an int
            # subclass — `isinstance(True, int)` is True). Matches
            # the CONST_INT discipline: a bool ast_handle wraps to
            # 0/1 silently, but should fail loudly so a buggy
            # frontend / hand-built tir.Module is visible. Audit-
            # fix MEDIUM-1.
            if type(ast_handle) is not int:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: QUOTE needs an int "
                    f"'ast_handle' attr (got "
                    f"{type(ast_handle).__name__})")
            # Python `%` with non-negative divisor always returns
            # non-negative — a negative ast_handle wraps into
            # [0, NUM_CELLS) (matches x86's
            # `int(...) % HELIX_NUM_CELLS` at line 4473). The wrap
            # is intentional: the front-end computes a content-
            # addressed hash that may legally be any int.
            handle = ast_handle % _HELIX_NUM_CELLS
            return f"%v{result.id} = add i32 0, {handle}"
        if kind == tir.OpKind.SPLICE:
            # `splice` loads cell[handle] and returns it as i32 / f32
            # / f64 depending on `value_kind`. One i32 operand, one
            # result matching the value_kind. v3.1 step 4 added
            # polymorphic f32/f64 dispatch — pre-v3.1 only `i32` was
            # supported (the f32/f64 variants raised).
            if len(op.operands) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: SPLICE takes one "
                    f"operand (the i32 handle), got "
                    f"{len(op.operands)}")
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: SPLICE must have "
                    f"exactly one result, got {len(op.results)}")
            value_kind = op.attrs.get("value_kind", "i32")
            # Validation + lookup go through `_SPLICE_DISPATCH` so the
            # known set and the dispatch entries cannot drift (audit-
            # fix HIGH-1).
            splice_entry = _SPLICE_DISPATCH.get(value_kind)
            if splice_entry is None:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: SPLICE value_kind "
                    f"{value_kind!r} is not supported (known: "
                    f"{sorted(_SPLICE_DISPATCH)})")
            helper_name, expected_ty, call_ret_ty = splice_entry
            handle = op.operands[0]
            handle_ty = _llvm_int_type(
                handle.ty,
                ctx=f"function {self.fn.name!r} SPLICE handle")
            if handle_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: SPLICE handle has "
                    f"LLVM type {handle_ty}, but every SPLICE helper "
                    f"takes an i32 handle")
            result = op.results[0]
            # Validate result type matches value_kind. f32/f64 result
            # types are validated against `_LLVM_FLOAT_TYPES`; i32
            # against the existing int-type check.
            if value_kind == "i32":
                actual_ty = _llvm_int_type(
                    result.ty,
                    ctx=f"function {self.fn.name!r} SPLICE result")
            else:
                # Float scalar — must be the same TIRScalar name as
                # value_kind for the bitcast inside the helper to
                # produce a value of the expected type.
                if (not isinstance(result.ty, tir.TIRScalar)
                        or result.ty.name != value_kind):
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: SPLICE "
                        f"value_kind {value_kind!r} requires a "
                        f"matching {value_kind} result type (got "
                        f"{result.ty!r})")
                actual_ty = _LLVM_FLOAT_TYPES[value_kind]
            if actual_ty != expected_ty:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: SPLICE result has "
                    f"LLVM type {actual_ty}, but value_kind "
                    f"{value_kind!r} yields {expected_ty}")
            self._register_helper_function(helper_name)
            return (f"%v{result.id} = call {call_ret_ty} "
                    f"@{helper_name}(i32 {self._ref(handle)})")
        if kind == tir.OpKind.MODIFY:
            # `modify` bounds-checks the handle, calls a user-
            # supplied verifier(handle, new_value), and stores
            # new_value into cell[handle] if accepted. Two i32
            # operands (handle, new_value), one i32 result (1 on
            # accepted-store, 0 on OOB or verifier-reject).
            #
            # The verifier name lives in `op.attrs["verifier_fn"]`
            # and is passed as a function pointer at the call site —
            # one shared helper, many verifier-specific call sites.
            # If the verifier function is not defined in the module
            # at link time LLVM rejects; this op handler does not
            # cross-check (the per-fn emitter has no module-level
            # view). NOTE (Stage 207 parity): the verifier's
            # (i32, i32) -> i32 ABI is not cross-checked at emit
            # time either — a wrong-arity verifier produces ABI-
            # mismatch UB at runtime; x86_64.py's `call_rel32` has
            # the same gap. The Stage 207 parity gate is the
            # decision-maker for an emit-time cross-check (would
            # need module-level visibility plumbed into _FnEmitter).
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: MODIFY must have "
                    f"exactly one result, got {len(op.results)}")
            value_kind = op.attrs.get("value_kind", "i32")
            # Validation goes through `_MODIFY_DISPATCH` (audit-fix
            # HIGH-1) so the known set and the dispatch entries cannot
            # drift. Lookup happens later in the canonical-form branch.
            if value_kind not in _MODIFY_DISPATCH:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: MODIFY value_kind "
                    f"{value_kind!r} is not supported (known: "
                    f"{sorted(_MODIFY_DISPATCH)})")
            verifier_fn = op.attrs.get("verifier_fn")
            result = op.results[0]
            # MODIFY's result is always i32 (the accepted-or-not
            # flag) — independent of value_kind.
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} MODIFY result")
            if res_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: MODIFY result has "
                    f"LLVM type {res_ty}, but the accepted-or-not "
                    f"flag is an i32")
            # Legacy fallback (audit-fix HIGH-1, x86 parity): when
            # `verifier_fn` is missing, the op semantics degrade to
            # "is operand[2] truthy?" — no bounds check, no actual
            # cell store, just a 0/1 flag. x86_64.py::MODIFY lines
            # 4538-4551 handles this fallback for legacy frontend
            # emissions (e.g. the `modify(h, v, runtime_expr)` form
            # that lower_ast emits when no verifier function is in
            # scope). Without this branch, programs that exercise
            # the legacy form would compile on x86 but fail on LLVM,
            # producing a real Stage 207 parity divergence.
            if verifier_fn is None:
                if len(op.operands) >= 3:
                    legacy = op.operands[2]
                    legacy_ty = _llvm_int_type(
                        legacy.ty,
                        ctx=(f"function {self.fn.name!r} MODIFY "
                             f"legacy verifier operand"))
                    if legacy_ty != "i32":
                        raise LLVMEmitError(
                            f"function {self.fn.name!r}: MODIFY "
                            f"legacy verifier operand has LLVM "
                            f"type {legacy_ty}, but the truthy "
                            f"check expects i32")
                    rid = result.id
                    is_ne = f"%v{rid}.is_ne"
                    return (
                        f"{is_ne} = icmp ne i32 "
                        f"{self._ref(legacy)}, 0\n"
                        f"%v{rid} = zext i1 {is_ne} to i32")
                # Legacy fallback with <3 operands: result = 0
                # (matches x86 line 4549-4550).
                return f"%v{result.id} = add i32 0, 0"
            # Canonical form: 2 operands + non-empty verifier_fn
            # string attr.
            if not isinstance(verifier_fn, str) or not verifier_fn:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: MODIFY needs a "
                    f"non-empty 'verifier_fn' string attr naming "
                    f"the verifier function (got "
                    f"{type(verifier_fn).__name__} {verifier_fn!r})")
            if len(op.operands) != 2:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: canonical MODIFY "
                    f"(with verifier_fn) takes two operands (i32 "
                    f"handle, i32 new_value), got "
                    f"{len(op.operands)}")
            handle, new_value = op.operands
            handle_ty = _llvm_int_type(
                handle.ty,
                ctx=f"function {self.fn.name!r} MODIFY handle")
            if handle_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: MODIFY handle has "
                    f"LLVM type {handle_ty}, but every MODIFY helper "
                    f"takes an i32 handle")
            # Dispatch by value_kind to the right helper + new_value
            # LLVM type. v3.1 step 4. `value_kind` was validated
            # against `_MODIFY_DISPATCH.keys()` above, so the lookup
            # cannot KeyError.
            helper_name, new_value_llvm_ty = (
                _MODIFY_DISPATCH[value_kind])
            if value_kind == "i32":
                new_value_ty = _llvm_int_type(
                    new_value.ty,
                    ctx=f"function {self.fn.name!r} MODIFY new_value")
                if new_value_ty != "i32":
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: MODIFY new_value "
                        f"has LLVM type {new_value_ty}, but the i32 "
                        f"value_kind takes an i32 new_value")
            else:
                # f32 / f64 — validate new_value's TIRScalar matches
                # value_kind so the helper's signature aligns.
                if (not isinstance(new_value.ty, tir.TIRScalar)
                        or new_value.ty.name != value_kind):
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: MODIFY "
                        f"value_kind {value_kind!r} requires a "
                        f"matching {value_kind} new_value type (got "
                        f"{new_value.ty!r})")
            self._register_helper_function(helper_name)
            verifier_global = _llvm_global_name(verifier_fn)
            return (f"%v{result.id} = call i32 "
                    f"@{helper_name}(i32 {self._ref(handle)}, "
                    f"{new_value_llvm_ty} {self._ref(new_value)}, "
                    f"ptr {verifier_global})")
        if kind in (tir.OpKind.TRACE_ENTRY, tir.OpKind.TRACE_EXIT):
            # `trace.entry` / `trace.exit` append a (fn_id, kind=0/1)
            # event to the trace ring buffer via the void-returning
            # `__helix_trace_event` helper. The fn_id is the per-
            # module interned i32 for the `fn_name` attr (table built
            # by `_intern_trace_fn_ids` in `emit_module`). When the
            # buffer is full the event is silently dropped (matches
            # x86_64.py's "no allocation, no syscall" contract).
            #
            # TRACE_EXIT optionally takes one operand (the return
            # value being returned); on x86 this triggers an extra
            # `mov eax, [slot]` load that keeps the SSA value alive
            # past the trace call.
            #
            # NOTE (Stage 207 parity): LLVM's SSA def-use chains do
            # NOT keep a value alive when it has zero uses — if the
            # TRACE_EXIT operand is the SOLE use of some SSA value,
            # an LLVM DCE pass can drop the value's computation,
            # diverging from x86's always-load. The handler emits a
            # no-op `bitcast` to force the operand into a real use
            # so the LLVM IR mirrors x86's observable load. The
            # bitcast result is discarded; LLVM's instruction
            # combiner will fold it away after register allocation
            # but the SSA-level use keeps the operand's def alive
            # through every optimization pass that respects use lists.
            if kind == tir.OpKind.TRACE_ENTRY and op.operands:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: TRACE_ENTRY takes no "
                    f"operands, got {len(op.operands)}")
            if kind == tir.OpKind.TRACE_EXIT and len(op.operands) > 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: TRACE_EXIT expects "
                    f"zero or one operands (the optional return value "
                    f"for liveness), got {len(op.operands)}")
            if op.results:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: {kind.value} has no "
                    f"result (void), but got {len(op.results)}")
            fn_name = op.attrs.get("fn_name", "")
            if not isinstance(fn_name, str) or not fn_name:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: {kind.value} needs a "
                    f"non-empty 'fn_name' string attr (got "
                    f"{type(fn_name).__name__} {fn_name!r})")
            if self.trace_fn_ids is None:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: {kind.value} requires "
                    f"a module-level fn-id interning table — emit this "
                    f"module via `emit_module(...)` (which builds the "
                    f"table) rather than `emit_function(...)`")
            fn_id = self.trace_fn_ids.get(fn_name)
            if fn_id is None:
                # Two possible causes — name both, since a developer
                # writing a custom caller will see this before any
                # documented `emit_module` flow could mutate state.
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: {kind.value} "
                    f"references fn_name {fn_name!r} but it is not in "
                    f"the trace-fn-id table (table has "
                    f"{len(self.trace_fn_ids)} entries). Possible "
                    f"causes: (a) the caller constructed _FnEmitter "
                    f"directly with a hand-built trace_fn_ids that "
                    f"omits this fn_name — use `emit_module(...)` to "
                    f"auto-build the table; (b) the module was "
                    f"mutated between `_intern_trace_fn_ids` and "
                    f"`emit()` (concurrent mutation)")
            event_kind = 0 if kind == tir.OpKind.TRACE_ENTRY else 1
            self._register_helper_function("__helix_trace_event")
            lines = []
            # Force a real LLVM-IR use of the TRACE_EXIT operand to
            # mirror x86's `mov eax, [slot]` load (see the Stage 207
            # parity note above). `bitcast i32 X to i32` is the
            # cheapest possible use — LLVM treats it as a no-op
            # value but it appears in the use-list of X's def, so
            # DCE cannot remove the def while the bitcast lives.
            if (kind == tir.OpKind.TRACE_EXIT
                    and len(op.operands) == 1):
                operand = op.operands[0]
                operand_ty = _llvm_int_type(
                    operand.ty,
                    ctx=f"function {self.fn.name!r} TRACE_EXIT operand")
                # `%vN.trace_keepalive` namespaces the bitcast so it
                # never collides with other generated names.
                # `_op_index` is the position of this op within the
                # function — stable across re-emits.
                idx = self._next_keepalive_idx()
                lines.append(
                    f"%trace_keepalive.{idx} = bitcast "
                    f"{operand_ty} {self._ref(operand)} to {operand_ty}")
            lines.append(
                f"call void @__helix_trace_event("
                f"i32 {fn_id}, i32 {event_kind})")
            return "\n".join(lines)
        if kind == tir.OpKind.PRINT:
            print_kind = op.attrs.get("_kind", "print_str")
            if print_kind not in _SUPPORTED_PRINT_KINDS:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: PRINT _kind "
                    f"{print_kind!r} is not yet emitted by the LLVM "
                    f"backend (supported: "
                    f"{sorted(_SUPPORTED_PRINT_KINDS)}. "
                    f"TRACE_*, ARENA_*, QUOTE/SPLICE/MODIFY/REFLECT_HASH "
                    f"lower via their own OpKinds, NOT via PRINT)")
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: PRINT must have "
                    f"exactly one result, got {len(op.results)}")
            result = op.results[0]
            res_ty = _llvm_int_type(
                result.ty,
                ctx=f"function {self.fn.name!r} PRINT result")
            if res_ty != "i32":
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: PRINT result has LLVM "
                    f"type {res_ty}, but a PRINT yields an i32 (the "
                    f"byte count)")
            if print_kind == "write_file":
                # `write_file` takes NO operands and two string attrs
                # — `path` (the file path) and `content` (the bytes to
                # write). Lowers to `open(path, O_WRONLY|O_CREAT|O_TRUNC,
                # 0644) -> write(fd, content, len) -> close(fd)`, the
                # exact sequence x86_64.py emits via direct syscalls
                # (we go through libc here; the LLVM target triple is
                # `x86_64-unknown-linux-gnu` so the libc constants line
                # up with the syscall numbers). The op's i32 result is
                # `nwritten < 0 ? nwritten : 0` — matches x86_64.py's
                # "return negative on failure, 0 on success" contract.
                # Inline (no helper) because the sequence is short
                # enough that a per-call-site lowering reads cleaner
                # than a separate helper would.
                if op.operands:
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: a write_file "
                        f"PRINT takes no operands, got "
                        f"{len(op.operands)}")
                content = op.attrs.get("content")
                if not isinstance(content, str):
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: a write_file "
                        f"PRINT needs a string 'content' attr (got "
                        f"{type(content).__name__})")
                path = self._validate_path_attr(
                    "write_file", op.attrs.get("path"))
                # The path goes to `open()` which takes a C-string —
                # register it with a trailing NUL. The content is raw
                # bytes; `write` takes (ptr, len) so no terminator is
                # needed (content with embedded NULs is preserved).
                path_data = path.encode("utf-8") + b"\x00"
                content_data = content.encode("utf-8")
                path_name, _path_len = self._register_string(path_data)
                content_name, content_len = self._register_string(
                    content_data)
                # libc declarations. `mode_t` is `unsigned int` on
                # Linux x86-64; LLVM's signless-integer types make the
                # signed/unsigned distinction per-instruction, so we
                # declare both `flags` and `mode` as i32.
                self._register_ffi_declare(
                    "open", "@open", "i32", ["ptr", "i32", "i32"])
                self._register_ffi_declare(
                    "write", "@write", "i64", ["i32", "ptr", "i64"])
                self._register_ffi_declare(
                    "close", "@close", "i32", ["i32"])
                # O_WRONLY=1 | O_CREAT=64 (0o100) | O_TRUNC=512 (0o1000)
                # = 577 (0x241). Mode 0o644 = 420 (0x1A4). The flag
                # bit values are stable between the Linux kernel ABI
                # (which x86_64.py uses via direct syscalls) and the
                # glibc / musl `open()` wrappers (which the LLVM path
                # invokes here).
                #
                # NOTE (Stage 207 parity): three cross-backend
                # contract gaps are inherited verbatim from x86_64.py
                # to keep the two backends bit-for-bit observable-
                # equivalent — they are NOT silent failures
                # introduced by the LLVM path:
                #
                #   1. `open` failure is propagated indirectly: if
                #      open returns -1, the program does
                #      `write(-1, ...)` -> -EBADF, and the user-
                #      visible result is -EBADF, NOT the real errno
                #      from open (ENOENT, EACCES, EROFS, ...).
                #   2. Short writes (0 < nwritten < content_len) are
                #      reported as `nwritten == 0` -> success, even
                #      though `content_len - nwritten` bytes were
                #      dropped.
                #   3. `close(fd)` failure (EIO from a delayed flush
                #      on NFS, EBADF from a double-close) is silently
                #      discarded — the LLVM register `%vN.close`
                #      captures the return for naming clarity but the
                #      value flows nowhere, matching x86_64.py's
                #      `pop rcx (= fd, discarded)`.
                #
                # Both backends are mutually consistent on each
                # point; the Stage 207 parity gate decides whether to
                # tighten any of them in a coordinated way (e.g.
                # wrap `write` in an EINTR / short-write loop, surface
                # `open` errors before `write`, propagate `close`
                # errors). Until then: documented contract gaps, not
                # silent bugs.
                rid = result.id
                fd = f"%v{rid}.fd"
                nwritten = f"%v{rid}.nwritten"
                # close's return is intentionally discarded — see the
                # Stage 207 parity note above.
                close_ret = f"%v{rid}.close"
                nw32 = f"%v{rid}.nw32"
                is_neg = f"%v{rid}.is_neg"
                return (
                    f"{fd} = call i32 @open(ptr {path_name}, "
                    f"i32 577, i32 420)\n"
                    f"{nwritten} = call i64 @write(i32 {fd}, ptr "
                    f"{content_name}, i64 {content_len})\n"
                    f"{close_ret} = call i32 @close(i32 {fd})\n"
                    f"{nw32} = trunc i64 {nwritten} to i32\n"
                    f"{is_neg} = icmp slt i32 {nw32}, 0\n"
                    f"%v{rid} = select i1 {is_neg}, i32 {nw32}, i32 0")
            if print_kind == "read_file_to_arena":
                # `read_file_to_arena` takes NO operands and one
                # string attr `path`. Opens the path O_RDONLY, reads
                # up to BUF_SIZE bytes into a stack buffer, pushes
                # each byte (as i32) into the shared arena via
                # `__helix_arena_push`, returns the byte count read
                # (clamped to 0 on a negative read return).
                #
                # TRUNCATION SENTINEL: if read returns exactly
                # BUF_SIZE, the helper traps via `@llvm.trap()`
                # (matches x86's `ud2` at line 3500 — the build must
                # fail loudly, not silently produce a corrupt arena
                # state from a truncated source).
                #
                # The arena IS the result destination — there's no
                # arena-pointer operand because the arena is module-
                # global state (`@__helix_arena_base`). The helper's
                # transitive `helper_deps=("__helix_arena_push",)`
                # auto-registers the arena helper + global through
                # the existing `_register_helper_function` chain.
                if op.operands:
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: a "
                        f"read_file_to_arena PRINT takes no "
                        f"operands, got {len(op.operands)}")
                path = self._validate_path_attr(
                    "read_file_to_arena", op.attrs.get("path"))
                # Register the NUL-terminated path string as a
                # module-scope constant (content-addressed, dedups
                # against any other path with the same bytes).
                path_data = path.encode("utf-8") + b"\x00"
                path_name, _path_len = self._register_string(
                    path_data)
                # Pull in the helper (which transitively pulls in
                # arena_push + arena_base + open/read/close/llvm.trap
                # via its `helper_deps` + `ffi_declares`).
                self._register_helper_function(
                    "__helix_read_file_to_arena")
                return (f"%v{result.id} = call i32 "
                        f"@__helix_read_file_to_arena(ptr "
                        f"{path_name})")
            if print_kind == "print_int":
                # `print_int` takes ONE i32 operand (the value to print)
                # and emits a call to the `__helix_print_int` internal
                # helper — the digit-conversion loop is too unwieldy to
                # inline at every print_int call site. The helper does
                # the `write(1, buf, len)` syscall itself; this op just
                # forwards its i32 result (the byte count).
                if len(op.operands) != 1:
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: a print_int PRINT "
                        f"takes one operand (the i32 value), got "
                        f"{len(op.operands)}")
                value = op.operands[0]
                value_ty = _llvm_int_type(
                    value.ty,
                    ctx=f"function {self.fn.name!r} print_int operand")
                if value_ty != "i32":
                    raise LLVMEmitError(
                        f"function {self.fn.name!r}: print_int operand "
                        f"has LLVM type {value_ty}, but the helper "
                        f"`__helix_print_int` takes an i32")
                self._register_helper_function("__helix_print_int")
                return (f"%v{result.id} = call i32 "
                        f"@__helix_print_int(i32 {self._ref(value)})")
            # print_str: a string attr (no operands) + `write(1, msg,
            # len)` of a module-scope constant.
            if op.operands:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: a print_str PRINT "
                    f"takes no operands, got {len(op.operands)}")
            text = op.attrs.get("text", "")
            if not isinstance(text, str):
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: a print_str PRINT "
                    f"needs a string 'text' attr (got "
                    f"{type(text).__name__})")
            str_name, str_len = self._register_string(
                text.encode("utf-8"))
            self._register_ffi_declare(
                "write", "@write", "i64", ["i32", "ptr", "i64"])
            # write(1, msg, len) — fd 1 is stdout. `write` returns an
            # i64 byte count; PRINT's result is that count truncated
            # to i32, matching x86_64.py (which stores `eax`).
            rid = result.id
            return (f"%v{rid}.t0 = call i64 @write(i32 1, ptr "
                    f"{str_name}, i64 {str_len})\n"
                    f"%v{rid} = trunc i64 %v{rid}.t0 to i32")
        raise LLVMEmitError(
            f"function {self.fn.name!r}: the LLVM backend does not yet "
            f"emit op {kind.value} (supported: CONST_INT; the integer "
            f"arithmetic ADD/SUB/MUL/DIV/MOD; the bitwise AND/OR/XOR/NOT "
            f"and shifts SHL/SHR; the six comparisons; SELECT; NEG; "
            f"the mutable locals ALLOC_VAR/LOAD_VAR/STORE_VAR; the stack "
            f"arrays ALLOC_ARRAY/LOAD_ELEM/STORE_ELEM; direct + FFI "
            f"calls; the Result intrinsics RESULT_PACK/TAG/PAYLOAD; "
            f"TRAP; STR_PTR; STR_BYTE; print_str / print_int / "
            f"write_file / read_file_to_arena PRINT; "
            f"ARENA_PUSH / GET / SET / LEN / PUSH_PAIR / PUSH_TRIPLE; "
            f"TRACE_ENTRY / TRACE_EXIT; QUOTE / SPLICE / MODIFY; "
            f"RETURN; BR; COND_BR — REFLECT_HASH + float / struct "
            f"support are later stages)"
        )


def emit_function(fn: tir.FnIR) -> str:
    """Emit the textual LLVM IR `define` for one host-IR function.
    Raises `LLVMEmitError` for any construct outside the supported set
    (see this module's docstring for the current supported op set).

    This is a single-function FRAGMENT for inspecting one function in
    isolation — it carries no module `target triple`, and an
    FFI_CALL's module-scope `declare` is emitted only by
    `emit_module`. Use `emit_module` for a complete,
    standalone-valid module."""
    return _FnEmitter(fn).emit()


def _intern_trace_fn_ids(module: tir.Module) -> dict[str, int]:
    """Walk a module and assign a stable i32 id to each `fn_name`
    referenced by a TRACE_ENTRY/TRACE_EXIT op. Returns a name->id
    dict where ids are 0, 1, 2... in first-encounter order.

    Iteration: `module.functions` insertion order, then
    `fn.blocks` order, then `block.ops` order. This produces a
    deterministic id table — same module -> same ids — which is a
    parity-gate prerequisite (the x86_64 backend uses the same
    insertion-order interning).

    Skips `is_extern` functions (they have no body to scan). A
    fn_name that is not a non-empty string is left for the per-op
    handler to reject loudly — this pre-pass intentionally avoids
    raising so a single malformed TRACE op does not block
    interning of every other op in the module.
    """
    fn_ids: dict[str, int] = {}
    for fn in module.functions.values():
        if fn.attrs.get("is_extern"):
            continue
        if fn.attrs.get("kernel"):
            # Mirrors `emit_module`'s kernel rejection — a kernel
            # function never produces LLVM IR (raised at line ~2670),
            # so its TRACE ops would otherwise pollute the fn-id
            # table with names that will never appear in emitted
            # output. Latent today (the kernel rejection runs after
            # this pre-pass), defensive against any future relaxation.
            continue
        for block in fn.blocks:
            for op in block.ops:
                if op.kind not in (
                        tir.OpKind.TRACE_ENTRY, tir.OpKind.TRACE_EXIT):
                    continue
                fn_name = op.attrs.get("fn_name", "")
                if not isinstance(fn_name, str) or not fn_name:
                    # Per-op handler will reject this loudly with a
                    # better diagnostic; skip rather than raise so
                    # this pre-pass stays total over the module.
                    continue
                if fn_name not in fn_ids:
                    fn_ids[fn_name] = len(fn_ids)
    return fn_ids


def emit_module(module: tir.Module) -> str:
    """Emit a complete textual LLVM IR module from a host `tir.Module`.

    Additive v3.0 Phase-D backend — consumes the same IR that
    `x86_64.py::compile_module_to_elf` consumes and emits LLVM IR the
    toolchain accepts (real `opt`/`llc` dispatch is `llvm_toolchain`).
    Functions are emitted in `module.functions` insertion order so the
    output is deterministic (a Stage 207 parity prerequisite).

    Mirrors `compile_module_to_elf`'s function filter: an `is_extern`
    ("extern C") function is a body-less DECLARATION — it gets no
    `define` (its module-scope `declare` is emitted by the FFI_CALL
    that references it); a `@kernel` function is a GPU kernel, outside
    this host CPU backend's scope, and is rejected with a loud
    `LLVMEmitError`. A module-scope `declare` is emitted for each
    extern symbol an FFI_CALL targets."""
    # Filter exactly as x86_64.py::compile_module_to_elf does: skip
    # body-less `is_extern` declarations (handing one to
    # `_FnEmitter.emit()` raises a misleading "block has no
    # terminator"), and reject a `@kernel` function loudly — the LLVM
    # backend emits host CPU IR only, it does not lower GPU kernels.
    # Pre-pass: build the trace-fn-id interning table by walking
    # every TRACE_ENTRY / TRACE_EXIT op in every non-extern function
    # (in `module.functions` insertion order, then per-block, then
    # per-op order). This guarantees fn_id assignment is deterministic
    # — same module input -> same id table -> same emitted constants.
    # The table is shared across every `_FnEmitter` so a fn_name
    # appearing in multiple functions' TRACE ops resolves to one
    # consistent i32. Mirrors x86_64.py's per-module
    # `_trace_fn_ids` semantics (shared across the single
    # `compile_module_to_elf` walker).
    trace_fn_ids = _intern_trace_fn_ids(module)
    emitters: list[_FnEmitter] = []
    for fn in module.functions.values():
        if fn.attrs.get("is_extern"):
            continue
        if fn.attrs.get("kernel"):
            raise LLVMEmitError(
                f"module: function {fn.name!r} is a @kernel (GPU) "
                f"function — the LLVM backend emits host CPU IR only, "
                f"it does not lower GPU kernels")
        emitters.append(_FnEmitter(fn, trace_fn_ids=trace_fn_ids))
    bodies = [emitter.emit() for emitter in emitters]
    # Collect the module-scope `declare`s and string constants every
    # function accumulated during emission. A symbol declared with two
    # different signatures, or one that collides with a defined
    # function name, is malformed — fail closed on both rather than
    # emit a module `llvm-as` rejects.
    # `defined` is the set of names that get a `define` — the NON-
    # extern functions only. An FFI `declare` for an extern symbol that
    # is also an `is_extern` entry in `module.functions` is the SAME
    # extern, not a `declare`/`define` clash.
    defined = {name for name, fn in module.functions.items()
               if not fn.attrs.get("is_extern")}
    # Collect the union of internal helper functions every emitter
    # depends on; one helper used by N functions still emits exactly
    # one `define internal ...` block. A helper name reserves the
    # `__helix_` prefix; collision with a user-defined function name is
    # malformed and fails closed below.
    helpers_used: set[str] = set()
    module_globals_used: set[str] = set()
    for emitter in emitters:
        helpers_used |= emitter.helper_functions
        module_globals_used |= emitter.module_globals
    helper_collision = helpers_used & defined
    if helper_collision:
        raise LLVMEmitError(
            f"module: user-defined function name(s) "
            f"{sorted(helper_collision)} collide with reserved "
            f"`__helix_` internal helper name(s); rename the user "
            f"function (the `__helix_` prefix is reserved for the "
            f"LLVM backend's internal runtime helpers)")
    # Module-global collision check: a user-defined function whose
    # name shadows a `__helix_*` module-global would emit two
    # different globals with the same `@<name>`. Same fail-closed
    # discipline as the helper-collision gate above.
    module_global_collision = module_globals_used & defined
    if module_global_collision:
        raise LLVMEmitError(
            f"module: user-defined function name(s) "
            f"{sorted(module_global_collision)} collide with reserved "
            f"`__helix_` module-global name(s); rename the user "
            f"function (the `__helix_` prefix is reserved for the "
            f"LLVM backend's internal runtime state)")
    declares: dict[str, str] = {}
    strings: dict[str, str] = {}
    for emitter in emitters:
        for symbol, decl in emitter.ffi_declares.items():
            if symbol in defined:
                raise LLVMEmitError(
                    f"module: FFI symbol {symbol!r} is also a defined "
                    f"function — an extern `declare` cannot share a "
                    f"name with a `define`")
            existing = declares.get(symbol)
            if existing is not None and existing != decl:
                raise LLVMEmitError(
                    f"module: FFI symbol {symbol!r} is declared with "
                    f"two different signatures — {existing!r} vs "
                    f"{decl!r}")
            declares[symbol] = decl
        # String constants are content-addressed — the same text maps
        # to the same global name from every function, so a plain
        # `update` deduplicates them.
        strings.update(emitter.string_globals)
    # Second helper-collision gate: a user's `is_extern` declaration
    # (e.g. an `is_extern` Helix function named `__helix_print_int` and
    # an FFI_CALL targeting it) leaves the symbol out of `defined` but
    # IN `declares`. Emitting BOTH the user's `declare` and the
    # helper's `define internal` would yield malformed IR (`llvm-as`:
    # "redefinition of @symbol") which `mock_validate_ll` does NOT
    # detect. Fail closed at module assembly.
    helper_ffi_collision = helpers_used & declares.keys()
    if helper_ffi_collision:
        raise LLVMEmitError(
            f"module: extern FFI declare(s) {sorted(helper_ffi_collision)} "
            f"collide with reserved `__helix_` internal helper name(s); "
            f"a `declare` cannot share a name with the helper's "
            f"`define internal` (the `__helix_` prefix is reserved for "
            f"the LLVM backend's internal runtime helpers)")
    # Same gate, but for module-globals: a user's extern declare of a
    # `__helix_*` symbol would collide with the global emitted below.
    module_global_ffi_collision = module_globals_used & declares.keys()
    if module_global_ffi_collision:
        raise LLVMEmitError(
            f"module: extern FFI declare(s) "
            f"{sorted(module_global_ffi_collision)} collide with "
            f"reserved `__helix_` module-global name(s); a `declare` "
            f"cannot share a name with a module-global definition")
    lines: list[str] = [
        "; helixc LLVM IR backend — v3.0 Phase D",
        f'target triple = "{LLVM_TARGET_TRIPLE}"',
        "",
    ]
    lines.extend(strings.values())
    lines.extend(declares.values())
    # Module-scope globals (e.g. the arena buffer) — emitted in
    # sorted-by-name order for determinism. Emitted BEFORE function
    # bodies so a `define` can reference any global by symbol name
    # (LLVM permits forward references, but emitting top-down keeps
    # `mock_validate_ll`'s simple line-scan reading the module the
    # way a human would).
    for name in sorted(module_globals_used):
        lines.append(_MODULE_GLOBALS[name].definition)
    if strings or declares or module_globals_used:
        lines.append("")
    for body in bodies:
        lines.append(body)
        lines.append("")
    # Emit internal helper functions in deterministic (sorted-by-name)
    # order so the output is byte-stable across runs — emitter set
    # iteration order is not. Each helper's text already terminates
    # with `}`; add a trailing blank line for readability.
    for name in sorted(helpers_used):
        lines.append(_HELPER_FUNCTIONS[name].definition)
        lines.append("")
    return "\n".join(lines)


# Line-leading tokens of an LLVM basic-block terminator instruction.
_LL_TERMINATOR_PREFIXES: tuple[str, ...] = (
    "ret ", "ret\t", "br ", "switch ", "indirectbr ",
    "unreachable", "resume ", "callbr ",
)


def mock_validate_ll(ll_text: str) -> list[str]:
    """Toolchain-free shape check on emitted LLVM IR — returns a list of
    problem strings (empty == OK). Mirrors `gpu_ci.py`'s mock-validation
    path: a structural sanity check on this emitter's `.ll` output that
    runs in CI on a machine with no LLVM installed. It is NOT a full
    LLVM parser — real `llvm-as`/`opt`/`llc` validation is Stage 201.

    Checks (line-leading tokens matched after stripping indentation, so
    an indented `.ll` is handled too): a `target triple` line is
    present; at least one `define`; braces balance (quoted spans
    masked); and each `define` body ENDS with a basic-block terminator
    (`ret` / `br` / ...). A multi-block function need not contain a
    `ret` at all — an infinite loop ends every block with `br` — so the
    check is "the body's last instruction is a terminator", not "the
    body contains a ret"."""
    problems: list[str] = []
    stripped = [ln.strip() for ln in ll_text.splitlines()]
    if not any(s.startswith("target triple =") for s in stripped):
        problems.append("missing `target triple` line")
    if not any(s.startswith("define ") for s in stripped):
        problems.append("no `define` line — module declares no functions")
    # Brace balance — counted on text with quoted spans masked out, so
    # a brace legally inside a quoted identifier (e.g. a quoted function
    # name `@"a}b"`) is not miscounted as a structural brace.
    brace_text = _QUOTED_SPAN.sub("", ll_text)
    opens = brace_text.count("{")
    closes = brace_text.count("}")
    if opens != closes:
        problems.append(
            f"unbalanced braces: {opens} '{{' vs {closes} '}}'")
    # Each `define ... {` ... `}` body must END with a terminator
    # instruction — the emitter guarantees every block is terminated;
    # this catches a grossly-broken emit (a body whose last line is a
    # non-terminator instruction, or an empty body).
    in_body = False
    last_instr = ""
    cur_fn = ""

    def _check_body_end() -> None:
        if not last_instr.startswith(_LL_TERMINATOR_PREFIXES):
            problems.append(
                f"function body does not end with a terminator "
                f"(last line {last_instr!r}): {cur_fn}")

    for s in stripped:
        if s.startswith("define "):
            if in_body:
                _check_body_end()
            in_body = True
            last_instr = ""
            cur_fn = s
        elif in_body and s == "}":
            _check_body_end()
            in_body = False
        elif in_body and s and not s.endswith(":"):
            # A non-blank, non-label body line.
            last_instr = s
    if in_body:
        _check_body_end()
    return problems
