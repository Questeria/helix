"""
helixc/frontend/trace_pass.py — Stage 25: @trace attribute support.

A fn declared `@trace fn f(...) { ... }` becomes a traced function:
the codegen emits a trace-log call into the prologue (op_kind =
ENTRY) and epilogue (op_kind = EXIT) so a runtime trace buffer
captures each invocation's args + return value.

Phase-0 scope:
  * Parser accepts `@trace` as a fn attribute (existing
    _parse_attributes path handles it — no parser change needed).
  * `is_traced(fn)` helper exposed for the typechecker / lowering.
  * Trap 25001 reserved for: trace buffer overflow at runtime.
  * `TraceBuffer` Python-side simulation (used in tests / repl).
  * `trace_equiv(t1, t2)` predicate over trace buffers (returns True
    iff the two traces record identical ops in identical order with
    identical operand-hashes).

Runtime trace buffer wiring (entry/exit emission into the binary
prologue/epilogue) is bootstrap-side; this module exists so the
Python typechecker + a Python-side simulator can validate the design
before kovc.hx implements it.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import ast_nodes as A


# Trap-id reservation. Phase-0 doesn't emit; documented here so the
# bootstrap-side trace runtime knows the namespace is taken.
TRAP_TRACE_OVERFLOW = 25001
TRAP_TRACE_EQUIV_SHAPE_MISMATCH = 25002


# Default capacity for a Phase-0 trace buffer (entries before overflow
# trap fires).
DEFAULT_TRACE_CAP = 4096


@dataclass(frozen=True)
class TraceEvent:
    """One recorded trace event. op_kind is one of:
      * "entry"  — function entry; operands = arg-hashes
      * "exit"   — function exit; operands = (return-hash,)
      * "op"     — recorded primop (binop, call, etc.); op-specific
    """
    op_kind: str
    fn_name: str
    operands: tuple
    result: Optional[int] = None


@dataclass
class TraceBuffer:
    """Ring buffer for trace events. Phase-0 fixed-cap; overflow trap."""
    cap: int = DEFAULT_TRACE_CAP
    events: list[TraceEvent] = field(default_factory=list)

    def push(self, ev: TraceEvent) -> None:
        if len(self.events) >= self.cap:
            raise OverflowError(
                f"trace buffer overflow (cap={self.cap}, trap 25001)"
            )
        self.events.append(ev)

    def clear(self) -> None:
        self.events.clear()

    def __len__(self) -> int:
        return len(self.events)


def is_traced(fn: A.FnDecl) -> bool:
    """True iff this fn carries the @trace attribute."""
    return "trace" in fn.attrs


def trace_equiv(a: TraceBuffer, b: TraceBuffer) -> bool:
    """Two traces are equivalent iff they record the same sequence of
    (op_kind, fn_name, operands, result) tuples. Trap 25002 reserved
    for differently-shaped traces — Phase-0 just returns False rather
    than raising."""
    if len(a) != len(b):
        return False
    for x, y in zip(a.events, b.events):
        if x != y:
            return False
    return True


def traced_fn_names(prog: A.Program) -> list[str]:
    """List of fn names that carry `@trace` anywhere in the program.

    Stage 28.9 cycle 60 audit-R C59-1: iter_fn_decls recurses through
    ImplBlock.methods and ModBlock.items so mod-nested @trace fns
    are listed."""
    from .ast_walker import iter_fn_decls
    return [fn.name for fn in iter_fn_decls(prog) if is_traced(fn)]


def validate_trace_attrs(prog: A.Program) -> list[str]:
    """Sanity checks for @trace usage. Returns a list of diagnostic
    strings (empty on clean).

    Phase-0 rules:
      * @trace on extern "C" fn decl is rejected — extern fns have no
        body to instrument.
      * @trace on @pure fn is allowed (tracing observes but doesn't
        side-effect — the trace buffer is part of the runtime, not the
        program semantics that purity reasons about).
    """
    diags: list[str] = []
    from .ast_walker import iter_fn_decls
    for fn in iter_fn_decls(prog):
        if not is_traced(fn):
            continue
        if fn.is_extern:
            diags.append(
                f"@trace on extern \"C\" fn {fn.name!r}: not supported "
                f"(extern fns have no body to instrument)"
            )
    return diags
