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
from typing import Optional

from ..ir import tir


# The host target. x86_64.py emits a Linux x86-64 ELF; the LLVM path
# targets the same triple so the Stage 207 parity harness compares like
# for like.
LLVM_TARGET_TRIPLE = "x86_64-unknown-linux-gnu"


class LLVMEmitError(Exception):
    """The host IR contains a construct the Stage 200 scalar core does
    not yet emit. Raised loudly so an unsupported op can never be
    silently dropped or mis-emitted — the v3.0 "additive, parity-gated"
    discipline (docs/V3_PLAN.md): a partial backend fails closed, it
    never produces wrong IR. Stages 202-206 widen the supported set."""


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
    tuples, unit) — all out of Stage 200 scope."""
    if not isinstance(ty, tir.TIRScalar):
        raise LLVMEmitError(
            f"{ctx}: Stage 200 scalar core emits only scalar integer "
            f"types, got {type(ty).__name__}"
        )
    llvm = _LLVM_INT_TYPES.get(ty.name)
    if llvm is None:
        raise LLVMEmitError(
            f"{ctx}: Stage 200 scalar core does not emit dtype "
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

    def __init__(self, fn: tir.FnIR):
        self.fn = fn
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
        # extern symbol -> its module-scope `declare` line, filled as
        # FFI_CALLs (and TRAP's write/exit) are emitted; `emit_module`
        # collects + dedups these.
        self.ffi_declares: dict[str, str] = {}
        # content-addressed global name -> its `... = constant ...`
        # line, filled as TRAP panic messages are emitted; collected +
        # deduped by `emit_module`.
        self.string_globals: dict[str, str] = {}

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

    def _register_string(self, data: bytes) -> tuple[str, int]:
        """Register a read-only string constant (a TRAP panic
        message). Returns `(global_name, byte_length)`. The global is
        content-addressed — its name is a hash of the bytes — so two
        identical strings dedup to one module global and the name is
        stable across functions (`emit_module` collects + dedups the
        per-function `string_globals`)."""
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
        raise LLVMEmitError(
            f"function {self.fn.name!r}: the LLVM backend does not yet "
            f"emit op {kind.value} (supported: CONST_INT; the integer "
            f"arithmetic ADD/SUB/MUL/DIV/MOD; the bitwise AND/OR/XOR/NOT "
            f"and shifts SHL/SHR; the six comparisons; SELECT; NEG; "
            f"the mutable locals ALLOC_VAR/LOAD_VAR/STORE_VAR; the stack "
            f"arrays ALLOC_ARRAY/LOAD_ELEM/STORE_ELEM; direct + FFI "
            f"calls; the Result intrinsics RESULT_PACK/TAG/PAYLOAD; "
            f"TRAP; RETURN; BR; COND_BR — floats and structs are later "
            f"stages)"
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


def emit_module(module: tir.Module) -> str:
    """Emit a complete textual LLVM IR module from a host `tir.Module`.

    Additive v3.0 Phase-D backend — consumes the same IR that
    `x86_64.py::compile_module_to_elf` consumes and emits LLVM IR the
    toolchain accepts (real `opt`/`llc` dispatch is `llvm_toolchain`).
    Functions are emitted in `module.functions` insertion order so the
    output is deterministic (a Stage 207 parity prerequisite). A
    module-scope `declare` is emitted for each extern symbol an
    FFI_CALL targets."""
    emitters = [_FnEmitter(fn) for fn in module.functions.values()]
    bodies = [emitter.emit() for emitter in emitters]
    # Collect the module-scope `declare`s and string constants every
    # function accumulated during emission. A symbol declared with two
    # different signatures, or one that collides with a defined
    # function name, is malformed — fail closed on both rather than
    # emit a module `llvm-as` rejects.
    defined = set(module.functions)
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
    lines: list[str] = [
        "; helixc LLVM IR backend — v3.0 Phase D",
        f'target triple = "{LLVM_TARGET_TRIPLE}"',
        "",
    ]
    lines.extend(strings.values())
    lines.extend(declares.values())
    if strings or declares:
        lines.append("")
    for body in bodies:
        lines.append(body)
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
