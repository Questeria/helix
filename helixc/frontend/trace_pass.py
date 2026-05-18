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


def trace_filter_by_fn(buf: TraceBuffer, fn_name: str) -> TraceBuffer:
    """Stage 59 follow-on / Tier 3 #11 polish — convenience shortcut for
    `trace_filter(buf, lambda e: e.fn_name == fn_name)`. Common case
    when verifying a specific function's behavior in isolation."""
    return trace_filter(buf, lambda e: e.fn_name == fn_name)


def trace_filter_by_op(buf: TraceBuffer, op_kind: str) -> TraceBuffer:
    """Stage 59 follow-on / Tier 3 #11 polish — convenience shortcut for
    `trace_filter(buf, lambda e: e.op_kind == op_kind)`. Common when
    filtering to entry/exit pairs only, or to side-effecting ops."""
    return trace_filter(buf, lambda e: e.op_kind == op_kind)


def trace_filter(buf: TraceBuffer, predicate) -> TraceBuffer:
    """Stage 59 follow-on / Tier 3 #11 polish — filter trace events
    by a caller-supplied predicate. Returns a NEW TraceBuffer
    containing only events where `predicate(event)` is True.

    Use cases:
    - Filter to entry/exit pairs of a specific fn: lambda e:
        e.fn_name == "loss" and e.op_kind in ("entry", "exit")
    - Discard pure-observer ops, keep side-effecting ones
    - Slice to a subset before trace_diff for narrower verification

    Preserves the buffer's capacity (a filtered subset still fits).
    """
    out = TraceBuffer(cap=buf.cap)
    for ev in buf.events:
        if predicate(ev):
            out.events.append(ev)
    return out


def trace_summary(buf: TraceBuffer, max_events: int = 16) -> str:
    """Stage 59 follow-on / Tier 3 #11 polish — human-readable digest
    of a TraceBuffer. Lists first `max_events` events as one-line
    summaries plus a `(... N more)` truncation footer when needed.

    Use case: AGI verifier diagnostic reports — embed `trace_summary`
    output in the witness-of-mismatch message so the user sees the
    actual divergent trace structure, not just an opaque "trace
    mismatch" error.

    Format per event:
        "[i] {op_kind} {fn_name}({operands}) → {result}"
    where result is omitted if None.
    """
    lines: list[str] = []
    n = len(buf.events)
    show = min(n, max_events)
    for i in range(show):
        ev = buf.events[i]
        ops = ", ".join(repr(o) for o in ev.operands)
        if ev.result is None:
            lines.append(f"[{i}] {ev.op_kind} {ev.fn_name}({ops})")
        else:
            lines.append(
                f"[{i}] {ev.op_kind} {ev.fn_name}({ops}) → {ev.result}"
            )
    if n > show:
        lines.append(f"(... {n - show} more events)")
    if not lines:
        return "<empty trace>"
    return "\n".join(lines)


def trace_diff(a: TraceBuffer, b: TraceBuffer) -> Optional[tuple]:
    """Stage 59 follow-on / Tier 3 #11 polish — find the first divergent
    event between two traces. Useful debug helper when trace_equiv
    returns False and the caller wants to know WHERE.

    Returns:
      None        — traces are equivalent (matches trace_equiv == True)
      (idx, a_event, b_event) — first index where events differ
        a_event = None means a was shorter than b at this index
        b_event = None means b was shorter than a at this index

    Use case: AGI verifier comparing reference-trace vs candidate-
    trace can pinpoint the first behavioral divergence for a
    minimal-witness-of-mismatch report.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a.events[i] != b.events[i]:
            return (i, a.events[i], b.events[i])
    if len(a) < len(b):
        return (len(a), None, b.events[len(a)])
    if len(b) < len(a):
        return (len(b), a.events[len(b)], None)
    return None  # equal


def trace_size(buf: TraceBuffer) -> int:
    """Stage 59 follow-on / Tier 3 #11 polish — number of events in
    the trace buffer. Equivalent to len(buf) but spelled clearly for
    parity with the pytree.tree_size helper."""
    return len(buf.events)


def trace_count(buf: TraceBuffer, predicate) -> int:
    """Stage 59 follow-on / Tier 3 #11 polish — count events matching
    a predicate. Equivalent to len(trace_filter(buf, predicate))
    but doesn't allocate a new TraceBuffer.

    Use cases:
    - How many calls did fn "loss" make?
        trace_count(buf, lambda e: e.fn_name == "loss")
    - How many entry events (= number of function invocations)?
        trace_count(buf, lambda e: e.op_kind == "entry")
    - How many ops returned a non-None result?
        trace_count(buf, lambda e: e.result is not None)
    """
    return sum(1 for ev in buf.events if predicate(ev))


def trace_op_counts(buf: TraceBuffer) -> dict:
    """Stage 59 follow-on / Tier 3 #11 polish — histogram of op_kind
    counts across the trace.

    Returns a dict {op_kind: count}. Sorted by op_kind name
    implicitly via Python 3.7+ dict-insertion-order if iterated
    after sorted(). Useful summary for trace_summary's header line.

    Example:
        {"entry": 3, "exit": 3, "op": 17}

    Use case: quick sanity check that entry/exit are balanced (each
    entry has a matching exit), or to spot unexpectedly many ops in
    a candidate trace vs reference.
    """
    counts: dict = {}
    for ev in buf.events:
        counts[ev.op_kind] = counts.get(ev.op_kind, 0) + 1
    return counts


def trace_fn_counts(buf: TraceBuffer) -> dict:
    """Stage 59 follow-on / Tier 3 #11 polish — histogram of fn_name
    counts across the trace.

    Returns a dict {fn_name: count}. Companion to trace_op_counts.

    Example:
        {"loss": 3, "grad_pass": 1, "step": 100}

    Use case: profile which functions dominate the trace, spot
    unexpected callees (e.g., should grad_pass have been called once
    but was actually called 5 times?).
    """
    counts: dict = {}
    for ev in buf.events:
        counts[ev.fn_name] = counts.get(ev.fn_name, 0) + 1
    return counts


def trace_is_balanced(buf: TraceBuffer) -> bool:
    """Stage 59 follow-on / Tier 3 #11 polish — quick predicate for
    'every entry has a matching exit' invariant.

    Returns True iff the count of "entry" events equals the count of
    "exit" events. This is necessary-but-not-sufficient — it doesn't
    verify fn_name pairing or LIFO order, but it catches the most
    common defect (a function entered but never exited).

    For a complete LIFO-pairing check, use trace_filter to extract
    entry/exit pairs and verify nesting.
    """
    counts = trace_op_counts(buf)
    return counts.get("entry", 0) == counts.get("exit", 0)


def trace_hash(buf: TraceBuffer) -> str:
    """Stage 59 follow-on / Tier 3 #11 polish — content-addressable
    hash of a trace buffer.

    Computes SHA-256 over the canonicalized event sequence:
      op_kind|fn_name|operands|result; ...
    Two traces with identical event sequences hash identically;
    any divergence (different op, different operand, different
    order) produces a different hash.

    Pairs with tree_hash for end-to-end content addressing:
    `(tree_hash(params), trace_hash(buf))` uniquely identifies
    a (param-snapshot, execution-trace) pair. Used by:
    - Verifier cache: skip re-checking a (params, trace) pair we've
      seen before.
    - Reproducibility: log trace_hash per epoch to audit-trail the
      exact execution path.
    - trace_equiv fast path: if trace_hash differs, trace_equiv is
      necessarily False (cheap pre-check before full O(n) compare).

    Returns the hex SHA-256 (64 chars). Empty trace produces a valid
    hash too (sha256 of empty bytes — useful as a sentinel).
    """
    import hashlib
    h = hashlib.sha256()
    for ev in buf.events:
        h.update(ev.op_kind.encode("utf-8"))
        h.update(b"|")
        h.update(ev.fn_name.encode("utf-8"))
        h.update(b"|")
        h.update(repr(ev.operands).encode("utf-8"))
        h.update(b"|")
        h.update(repr(ev.result).encode("utf-8"))
        h.update(b";")
    return h.hexdigest()


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
