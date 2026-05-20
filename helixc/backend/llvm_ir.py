"""
helixc/backend/llvm_ir.py — textual LLVM IR backend (v3.0 Phase D).

v3.0 replaces the hand-rolled x86_64 ELF emitter with a backend that
emits textual LLVM IR for the LLVM toolchain (`opt` + `llc`) to consume.
Per the v3.0 migration strategy (docs/V3_PLAN.md) this is ADDITIVE: it
consumes the same host IR — a `tir.Module` — that
`helixc/backend/x86_64.py::compile_module_to_elf` consumes, and
`x86_64.py` is left completely untouched until the Stage 221 cutover.

Supported so far (Stages 200, 202, 203):
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

Anything outside that set — memory, calls, floats, division, bitwise
ops, the wider op surface — is REJECTED with a loud `LLVMEmitError`,
never emitted wrong. Those land in later stages. A mock-validation path
(`mock_validate_ll`) checks the emitted `.ll` text shape without needing
an LLVM toolchain, mirroring `gpu_ci.py`'s mock path; real
`llvm-as`/`opt`/`llc` dispatch (`llvm_toolchain.py`) is Stage 201.

License: Apache 2.0
"""

from __future__ import annotations

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

# tir integer binary OpKind -> LLVM instruction mnemonic.
_LLVM_SCALAR_BINOPS: dict[tir.OpKind, str] = {
    tir.OpKind.ADD: "add",
    tir.OpKind.SUB: "sub",
    tir.OpKind.MUL: "mul",
}

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


def _is_unsigned_int(ty: tir.TIRType) -> bool:
    """True for a Helix unsigned-integer scalar dtype. Drives the
    signed-vs-unsigned LLVM `icmp` predicate choice (slt vs ult)."""
    return (isinstance(ty, tir.TIRScalar)
            and ty.name in _UNSIGNED_INT_DTYPES)


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
    back-edge (LLVM textual IR permits the forward reference)."""

    def __init__(self, fn: tir.FnIR):
        self.fn = fn
        # tir.Value.id -> LLVM operand text
        self.operand: dict[int, str] = {}

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
                else:
                    for r in op.results:
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
        for block in fn.blocks:
            lines.extend(self._emit_block(block, preds))
        lines.append("}")
        return "\n".join(lines)

    def _emit_block(
            self, block: tir.Block,
            preds: "dict[int, list[tuple[tir.Block, tir.Op]]]"
            ) -> list[str]:
        lines = [f"{self._block_label(block.id)}:"]
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
                lines.append(f"  {text}")
            if op.kind in (tir.OpKind.RETURN, tir.OpKind.BR,
                           tir.OpKind.COND_BR):
                saw_terminator = True
        if not saw_terminator:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: block bb{block.id} has no "
                f"terminator — every LLVM basic block must end with "
                f"RETURN, BR, or COND_BR")
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

    def _emit_op(self, op: tir.Op) -> Optional[str]:
        """Emit one op's instruction text, or None when it materializes
        no instruction (a CONST_INT — recorded as an inline literal by
        `_prepass`)."""
        kind = op.kind
        if kind == tir.OpKind.CONST_INT:
            return None  # registered as an inline literal by _prepass
        if kind in _LLVM_SCALAR_BINOPS:
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
            # i64 register).
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
            # NOTE (Stage 207 parity): plain wrapping `add`/`sub`/`mul`,
            # no `nsw`/`nuw` — whether that matches x86_64.py's
            # overflow behaviour is a Stage 207 parity decision.
            return (f"%v{result.id} = {_LLVM_SCALAR_BINOPS[kind]} "
                    f"{ty} {lhs}, {rhs}")
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
        if kind == tir.OpKind.NEG:
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: NEG must have exactly "
                    f"one result, got {len(op.results)}")
            if len(op.operands) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: NEG expects one "
                    f"operand, got {len(op.operands)}")
            result = op.results[0]
            ctx = f"function {self.fn.name!r} NEG"
            ty = _llvm_int_type(result.ty, ctx=ctx)
            operand = op.operands[0]
            operand_ty = _llvm_int_type(operand.ty, ctx=f"{ctx} operand")
            if operand_ty != ty:
                raise LLVMEmitError(
                    f"{ctx}: operand has LLVM type {operand_ty} but the "
                    f"result is {ty} — they must share one type")
            # LLVM has no integer-negate instruction; `sub <ty> 0, x`
            # is the canonical form (two's-complement, wrapping).
            return f"%v{result.id} = sub {ty} 0, {self._ref(operand)}"
        raise LLVMEmitError(
            f"function {self.fn.name!r}: the LLVM backend does not yet "
            f"emit op {kind.value} (supported: CONST_INT, ADD, SUB, MUL, "
            f"the six comparisons, SELECT, NEG, RETURN, BR, COND_BR; "
            f"division, bitwise ops, memory, and calls are later stages)"
        )


def emit_function(fn: tir.FnIR) -> str:
    """Emit the textual LLVM IR `define` for one host-IR function.
    Raises `LLVMEmitError` for any construct outside the supported set
    (Stages 200 + 202 — the scalar core plus control flow)."""
    return _FnEmitter(fn).emit()


def emit_module(module: tir.Module) -> str:
    """Emit a complete textual LLVM IR module from a host `tir.Module`.

    Additive Stage 200 substrate — consumes the same IR that
    `x86_64.py::compile_module_to_elf` consumes and emits LLVM IR the
    toolchain accepts (real `opt`/`llc` dispatch is Stage 201).
    Functions are emitted in `module.functions` insertion order so the
    output is deterministic (a Stage 207 parity prerequisite)."""
    lines: list[str] = [
        "; helixc LLVM IR backend — v3.0 Phase D (Stages 200-203)",
        f'target triple = "{LLVM_TARGET_TRIPLE}"',
        "",
    ]
    for fn in module.functions.values():
        lines.append(emit_function(fn))
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
