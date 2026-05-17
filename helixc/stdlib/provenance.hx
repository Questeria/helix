// helixc/stdlib/provenance.hx — debug/observation helpers for the
// Stage 36 provenance machinery (Inc 5 `register_derivation` +
// `parent_*_at`, Inc 9 B2 `derive` arena auto-push, Inc 14
// `register_derivation3` + `parent_at`). Inc 13 (Stage 36) deliverable;
// extended in Inc 15 with three-parent observation aliases.
//
// Phase-0 reminder: `Logic<T> = T` at runtime; the source tag passed
// to `prove()` is discarded at IR lowering. The only runtime-observable
// provenance is the arena side-table populated by `register_derivation`
// (Inc 5), the Inc 9 B2 arena auto-push from `derive`, and the Inc 14
// `register_derivation3` triple-push. These helpers wrap those
// primitives with readability aliases + printable evidence-trail
// lines, so user code doesn't have to thread `parent_left_at` /
// `parent_right_at` / `parent_at` + `print_int` boilerplate through
// every debug callsite.
//
// Slot semantics (Phase-0): arity is NOT tracked in the handle, so
// slot indices are positional. For a 2-parent handle (from
// `register_derivation`): slot[0] = left, slot[1] = right. For a
// 3-parent handle (from `register_derivation3`): slot[0] = left,
// slot[1] = middle, slot[2] = right. The "evidence_right" alias below
// returns slot[1], which is the right value only for 2-parent
// handles — for 3-parent handles slot[1] is the MIDDLE value (use
// `evidence_middle` or `parent_at(h, 2)` to read the right value).
// This is a documented Phase-0 sharp edge tracked for closure by
// the stage36-inc16-arity-in-handle TODO.
//
// License: Apache 2.0

// Returns 1 iff `handle` is a 1-based derivation handle whose slot[0]
// lookup is in the recorded arena range (Inc 9 A1 sentinel: a
// -1 result means the slot is out of bounds / handle is the null
// sentinel 0). This is a NECESSARY-BUT-NOT-SUFFICIENT predicate for
// the handle to refer to a real `register_derivation*` call — the
// Phase-0 arena has no per-handle tag, so a slot whose value happens
// to be non-(-1) for any reason will pass this check. For "is this a
// concrete handle from a register call", you also need the contextual
// knowledge that nothing else has written to that arena slot. Pure:
// only reads arena state.
//
// Stage 37 post-closure correction (Stage 36 closure gate-3 M1):
// SECOND failure
// mode (false-negative): if the caller legitimately stores -1 as a
// source ID (e.g., a sentinel for "no upstream"), slot[0] collides
// with the Inc 9 A1 OOB sentinel and `has_evidence` returns 0 even
// for a fully valid handle. Until the deferred Inc 16 per-record
// arity word lands, callers should avoid -1 as a source ID or use
// direct parent_at/parent_*_at with their own validity tag. See
// docs/audit-stage36-closure-gate3-type-design.md#M1.
@pure
fn has_evidence(handle: i32) -> i32 {
    if handle <= 0 { 0 }
    else if parent_left_at(handle) == 0 - 1 { 0 }
    else { 1 }
}

// Readability alias for `parent_left_at` — returns slot[0]. For both
// 2-parent and 3-parent handles, slot[0] is the LEFT value. Pure.
@pure
fn evidence_left(handle: i32) -> i32 {
    parent_left_at(handle)
}

// Readability alias for `parent_right_at` — returns slot[1]. Honest
// only for 2-parent handles (where slot[1] is right). For 3-parent
// handles slot[1] is the MIDDLE value; use `evidence_middle` (alias)
// or `parent_at(handle, 2)` to read the right value of a 3-parent
// handle. Pure.
@pure
fn evidence_right(handle: i32) -> i32 {
    parent_right_at(handle)
}

// Inc 15 (silent-failure M1 closure): returns slot[1] explicitly under
// the "middle" name. For a 3-parent handle from register_derivation3,
// slot[1] is the middle value (and `evidence_right` returns the SAME
// slot[1], which is honest for 2-parent but confusing for 3-parent —
// `evidence_middle` exists for callers who know they have a 3-parent
// handle and want unambiguous naming). Pure.
@pure
fn evidence_middle(handle: i32) -> i32 {
    parent_at(handle, 1)
}

// Inc 15 (silent-failure M1 closure): returns slot[2] — the right
// value of a 3-parent handle from register_derivation3. For a
// 2-parent handle this reads into whatever happens to occupy the slot
// after the right value (typically the next derivation's slot[0] or
// the OOB sentinel from the Inc 9 A1 bounds check). The caller is
// responsible for knowing the handle is 3-parent before calling.
// (TODO stage36-inc16-arity-in-handle: a per-record arity word will
// let this function return -1 deterministically for non-3-parent
// handles.) Pure.
@pure
fn evidence_third(handle: i32) -> i32 {
    parent_at(handle, 2)
}

// Prints "h=<handle> slot0=<l> slot1=<r>\n" to stdout for diagnostic
// observation of a 2-parent derivation's evidence trail. The "slot0/
// slot1" labelling is intentional: pre-Inc-15 this printed "L= R="
// which silently lied for 3-parent handles (slot[1] is the middle
// value, not the right value). Returns 1 iff the handle was valid
// (caller may want to branch on this for "evidence missing" follow-
// ups). Side-effecting (calls print_*), so NOT @pure.
fn trace_evidence(handle: i32) -> i32 {
    print_str("h=");
    print_int(handle);
    print_str(" slot0=");
    print_int(parent_left_at(handle));
    print_str(" slot1=");
    print_int(parent_right_at(handle));
    print_str("\n");
    has_evidence(handle)
}

// Inc 15 (silent-failure M1 closure): three-parent variant of
// trace_evidence. Prints "h=<handle> slot0=<l> slot1=<m> slot2=<r>\n"
// for a 3-parent handle from register_derivation3. Caller is
// responsible for knowing the handle is 3-parent — for a 2-parent
// handle, slot2 will be whatever happens to live at arena[handle+1],
// which may be a sibling derivation's slot[0] or the OOB sentinel.
// Returns has_evidence(handle). Side-effecting, NOT @pure.
fn trace_evidence3(handle: i32) -> i32 {
    print_str("h=");
    print_int(handle);
    print_str(" slot0=");
    print_int(parent_left_at(handle));
    print_str(" slot1=");
    print_int(parent_at(handle, 1));
    print_str(" slot2=");
    print_int(parent_at(handle, 2));
    print_str("\n");
    has_evidence(handle)
}
