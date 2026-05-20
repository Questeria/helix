"""
helixc/backend/llvm_ir.py — textual LLVM IR backend (v3.0 Phase D, Stage 200).

v3.0 replaces the hand-rolled x86_64 ELF emitter with a backend that
emits textual LLVM IR for the LLVM toolchain (`opt` + `llc`) to consume.
Per the v3.0 migration strategy (docs/V3_PLAN.md) this is ADDITIVE: it
consumes the same host IR — a `tir.Module` — that
`helixc/backend/x86_64.py::compile_module_to_elf` consumes, and
`x86_64.py` is left completely untouched until the Stage 221 cutover.

Stage 200 scope — the SCALAR CORE substrate only:
  - module header + target triple
  - a `define` for each function (integer params + integer/void return)
  - integer constants (CONST_INT, materialized as inline literals)
  - integer add / sub / mul
  - `ret`

Anything outside that set — control flow, memory, calls, floats, the
wider op surface — is REJECTED with a loud `LLVMEmitError`, never
emitted wrong. Those land in Stages 202-206. A mock-validation path
(`mock_validate_ll`) checks the emitted `.ll` text shape without needing
an LLVM toolchain, mirroring `gpu_ci.py`'s mock path; real
`llvm-as`/`opt`/`llc` dispatch behind tool-detection is Stage 201.

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


# tir scalar-integer dtype name -> LLVM integer type.
_LLVM_INT_TYPES: dict[str, str] = {
    "bool": "i1",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    # `char` is also a TIRScalar integer dtype, but its bit width is not
    # yet pinned for the LLVM path — consciously deferred. A char-typed
    # function loudly raises LLVMEmitError here until a later stage
    # fixes the width (fail-closed, not a silent miss).
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


class _FnEmitter:
    """Emits the LLVM IR for one `tir.FnIR`. Holds the per-function map
    from a TIR SSA value id to its LLVM operand text — either an `%vN`
    register or, for a CONST_INT, an inline integer literal."""

    def __init__(self, fn: tir.FnIR):
        self.fn = fn
        # tir.Value.id -> LLVM operand text
        self.operand: dict[int, str] = {}

    def _ref(self, v: tir.Value) -> str:
        """The LLVM operand text for an already-defined TIR value (a
        parameter, or the result of an earlier op in this block)."""
        text = self.operand.get(v.id)
        if text is None:
            raise LLVMEmitError(
                f"function {self.fn.name!r}: value v{v.id} is used "
                f"before it is defined — Stage 200 emits straight-line "
                f"scalar code only (block parameters / phi nodes are "
                f"Stage 202)"
            )
        return text

    def emit(self) -> str:
        fn = self.fn
        if len(fn.blocks) != 1:
            raise LLVMEmitError(
                f"function {fn.name!r}: Stage 200 scalar core emits "
                f"single-block functions only (got {len(fn.blocks)} "
                f"blocks — control flow is Stage 202)"
            )
        ret_ty = _llvm_return_type(
            fn.return_ty, ctx=f"function {fn.name!r} return type")
        # Parameters become %v<id> registers.
        param_decls: list[str] = []
        for p in fn.params:
            p_ty = _llvm_int_type(
                p.ty, ctx=f"function {fn.name!r} parameter")
            reg = f"%v{p.id}"
            self.operand[p.id] = reg
            param_decls.append(f"{p_ty} {reg}")
        lines: list[str] = [
            f"define {ret_ty} {_llvm_global_name(fn.name)}"
            f"({', '.join(param_decls)}) {{"
        ]
        block = fn.blocks[0]
        if block.params:
            raise LLVMEmitError(
                f"function {fn.name!r}: entry block carries block "
                f"parameters — Stage 200 emits single-block scalar "
                f"functions with no phis (Stage 202)"
            )
        saw_terminator = False
        for op in block.ops:
            if saw_terminator:
                raise LLVMEmitError(
                    f"function {fn.name!r}: op {op.kind.value} follows "
                    f"the RETURN terminator — unreachable code"
                )
            text = self._emit_op(op)
            if text is not None:
                lines.append(f"  {text}")
            if op.kind == tir.OpKind.RETURN:
                saw_terminator = True
        if not saw_terminator:
            raise LLVMEmitError(
                f"function {fn.name!r}: no RETURN op — every LLVM basic "
                f"block must end with a terminator"
            )
        lines.append("}")
        return "\n".join(lines)

    def _emit_op(self, op: tir.Op) -> Optional[str]:
        """Emit one op. Returns the instruction text, or None when the
        op materializes no instruction (a CONST_INT is recorded as an
        inline literal and used directly at every use site)."""
        kind = op.kind
        if kind == tir.OpKind.CONST_INT:
            if len(op.results) != 1:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: CONST_INT must have "
                    f"exactly one result, got {len(op.results)}"
                )
            result = op.results[0]
            value = op.attrs.get("value")
            # `type(value) is int`, NOT isinstance — a Python `bool` is
            # an int subclass; a bool would `str()` to "True"/"False"
            # and emit malformed IR (`ret i32 True`). A real boolean
            # constant is a CONST_BOOL op, not CONST_INT.
            if type(value) is not int:
                raise LLVMEmitError(
                    f"function {self.fn.name!r}: CONST_INT op needs an "
                    f"integer 'value' attr (got {value!r}: "
                    f"{type(value).__name__})"
                )
            # Validate the result type is emittable; the literal is
            # then used inline at every use site.
            _llvm_int_type(
                result.ty, ctx=f"function {self.fn.name!r} CONST_INT")
            self.operand[result.id] = str(value)
            return None
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
            # guarantee that (IRBuilder.add copies the lhs type;
            # IRBuilder.emit accepts an arbitrary result_ty), so verify
            # it here — a mismatch would otherwise silently emit
            # malformed LLVM IR (an `add i32` referencing an i64
            # register).
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
            reg = f"%v{result.id}"
            self.operand[result.id] = reg
            # NOTE (Stage 207 parity): emitted as a plain wrapping LLVM
            # `add`/`sub`/`mul` — no `nsw`/`nuw`. Whether that matches
            # x86_64.py's integer-overflow behaviour is a parity-gate
            # (Stage 207) decision, not a Stage 200 one.
            return f"{reg} = {_LLVM_SCALAR_BINOPS[kind]} {ty} {lhs}, {rhs}"
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
        raise LLVMEmitError(
            f"function {self.fn.name!r}: Stage 200 scalar core does not "
            f"emit op {kind.value} (supported: CONST_INT, ADD, SUB, MUL, "
            f"RETURN; the wider op set is Stages 202-206)"
        )


def emit_function(fn: tir.FnIR) -> str:
    """Emit the textual LLVM IR `define` for one host-IR function.
    Raises `LLVMEmitError` for any construct outside the Stage 200
    scalar core."""
    return _FnEmitter(fn).emit()


def emit_module(module: tir.Module) -> str:
    """Emit a complete textual LLVM IR module from a host `tir.Module`.

    Additive Stage 200 substrate — consumes the same IR that
    `x86_64.py::compile_module_to_elf` consumes and emits LLVM IR the
    toolchain accepts (real `opt`/`llc` dispatch is Stage 201).
    Functions are emitted in `module.functions` insertion order so the
    output is deterministic (a Stage 207 parity prerequisite)."""
    lines: list[str] = [
        "; helixc LLVM IR backend — v3.0 Phase D Stage 200 (scalar core)",
        f'target triple = "{LLVM_TARGET_TRIPLE}"',
        "",
    ]
    for fn in module.functions.values():
        lines.append(emit_function(fn))
        lines.append("")
    return "\n".join(lines)


def mock_validate_ll(ll_text: str) -> list[str]:
    """Toolchain-free shape check on emitted LLVM IR — returns a list of
    problem strings (empty == OK). Mirrors `gpu_ci.py`'s mock-validation
    path: a structural sanity check on this emitter's `.ll` output that
    runs in CI on a machine with no LLVM installed. It is NOT a full
    LLVM parser — real `llvm-as`/`opt`/`llc` validation lands in Stage
    201 behind tool-detection.

    Checks (line-leading tokens are matched after stripping indentation,
    so an indented `.ll` is handled too): a `target triple` line is
    present; at least one `define`; braces balance; every `define` body
    contains a `ret` terminator (Stage 200 functions are single-block,
    so a per-define `ret` scan is sufficient)."""
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
    # Each `define ... {` ... `}` body must contain a `ret` terminator.
    in_body = False
    body_has_ret = False
    cur_fn = ""
    for s in stripped:
        if s.startswith("define "):
            if in_body and not body_has_ret:
                problems.append(
                    f"function body has no `ret` terminator: {cur_fn}")
            in_body = True
            body_has_ret = False
            cur_fn = s
        elif in_body and s == "}":
            if not body_has_ret:
                problems.append(
                    f"function body has no `ret` terminator: {cur_fn}")
            in_body = False
        elif in_body and s.startswith("ret "):
            body_has_ret = True
    if in_body and not body_has_ret:
        problems.append(
            f"function body has no `ret` terminator: {cur_fn}")
    return problems
