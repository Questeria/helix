// helixc/stdlib/provenance.hx — debug/observation helpers for the
// Stage 36 provenance machinery (Inc 5 `register_derivation` +
// `parent_*_at`, Inc 9 B2 `derive` arena auto-push). Inc 13 (Stage 36)
// deliverable.
//
// Phase-0 reminder: `Logic<T> = T` at runtime; the source tag passed
// to `prove()` is discarded at IR lowering. The only runtime-observable
// provenance is the arena side-table populated by `register_derivation`
// (Inc 5) and the Inc 9 B2 arena auto-push from `derive`. These helpers
// wrap those primitives with readability aliases + a printable
// evidence-trail line, so user code doesn't have to thread
// `parent_left_at` / `parent_right_at` + `print_int` boilerplate
// through every debug callsite.
//
// Naming convention: "evidence_*" reads better than "parent_*_at"
// when the call appears inside an audit / debug context (where the
// reader is thinking "is the evidence still there?", not "what's at
// arena[2*h]?").
//
// License: Apache 2.0

// Returns 1 iff `handle` is a valid 1-based derivation handle
// (Inc 9 A2 fix: handle 0 is the reserved null sentinel) and
// `parent_left_at(handle)` resolves to a non-(-1) value (Inc 9 A1
// bounds-check sentinel). The conservative reading is "evidence
// recoverable" — a handle that satisfies both predicates can be
// safely passed to `evidence_left` / `evidence_right` for use in
// downstream logic. Pure: only reads arena state.
@pure
fn has_evidence(handle: i32) -> i32 {
    if handle <= 0 { 0 }
    else if parent_left_at(handle) == 0 - 1 { 0 }
    else { 1 }
}

// Readability alias for `parent_left_at`. Pure: arena read.
@pure
fn evidence_left(handle: i32) -> i32 {
    parent_left_at(handle)
}

// Readability alias for `parent_right_at`. Pure: arena read.
@pure
fn evidence_right(handle: i32) -> i32 {
    parent_right_at(handle)
}

// Prints "h=<handle> L=<left> R=<right>\n" to stdout for diagnostic
// observation of a derivation's evidence trail. Returns 1 iff the
// handle was valid (caller may want to branch on this for "evidence
// missing" follow-ups). Side-effecting (calls print_*), so NOT @pure.
fn trace_evidence(handle: i32) -> i32 {
    print_str("h=");
    print_int(handle);
    print_str(" L=");
    print_int(parent_left_at(handle));
    print_str(" R=");
    print_int(parent_right_at(handle));
    print_str("\n");
    has_evidence(handle)
}
