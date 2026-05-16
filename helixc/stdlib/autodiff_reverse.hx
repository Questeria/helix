// helixc/stdlib/autodiff_reverse.hx — reverse-mode autodiff via tape.
//
// Phase 2.1 step 2: reverse-mode AD. Records operations during the
// forward pass into a tape; backward pass walks the tape in reverse,
// propagating gradients. Unlike forward mode (O(N) per parameter),
// reverse mode is O(1) backward pass per OUTPUT — exactly what NN
// backprop needs (millions of params, single scalar loss).
//
// Tape format (each op = 4 arena slots):
//   slot 0: op_kind   (0=leaf, 1=add, 2=sub, 3=mul, 4=neg)
//   slot 1: in1_idx   (tape position of left operand, or -1 for leaf)
//   slot 2: in2_idx   (tape position of right operand, or -1 for unary)
//   slot 3: value     (forward i32 value at this tape position)
//
// Tape header (before the slots):
//   slot 0: count     (number of ops on tape)
//   slot 1: cap       (max ops)
//   slot 2: adj_start (arena index of adjoint array, allocated by rev_alloc_adjoints)
//   slot -1: magic    (guards against forged t1d buffers)
//   slot 3 + cap*4: footer guard derived from cap
// Adjoint metadata:
//   adj_start - 4: owner tape start
//   adj_start - 3: cap
//   adj_start - 2: logical count snapshotted at allocation
//   adj_start - 1: guard = rev_adj_guard(owner, cap, cnt, adj_start)
//   adj_start + cap: footer guard = same guard
// Adjoint arrays must be allocated immediately after their tape. That
// layout invariant prevents a forged arena slice from masquerading as a
// tape-owned adjoint buffer by copying the public metadata fields.
//
// API:
//   rev_tape_new(cap)               -> i32   allocate tape, return start
//   rev_leaf(tape, value)           -> i32   record an input; return tape index
//   rev_add(tape, ai, bi)           -> i32   record a+b
//   rev_sub(tape, ai, bi)           -> i32   record a-b
//   rev_mul(tape, ai, bi)           -> i32   record a*b
//   rev_neg(tape, ai)               -> i32   record -a
//   rev_value_at(tape, idx)         -> i32   read forward value
//   rev_alloc_adjoints(tape)        -> i32   allocate adjoint array; return start
//   rev_seed(adj_start, idx, seed)  -> i32   set adj[idx] = seed (typically 1
//                                            on the output)
//   rev_backward(tape, adj_start)   -> i32   walk tape in reverse,
//                                            accumulating partials
//   rev_grad(adj_start, idx)        -> i32   read d_output / d_input[idx]
//   rev_kind_at(tape, idx)          -> i32   op_kind at tape position
//   rev_in1_at(tape, idx)           -> i32   in1 operand index at tape position
//   rev_in2_at(tape, idx)           -> i32   in2 operand index at tape position
//   rev_is_empty(tape)              -> i32   1 if count == 0 else 0
//   rev_remaining(tape)             -> i32   cap - count (slots available)
//
// License: Apache 2.0

@pure fn rev_kind_leaf() -> i32 { 0 }
@pure fn rev_kind_add() -> i32 { 1 }
@pure fn rev_kind_sub() -> i32 { 2 }
@pure fn rev_kind_mul() -> i32 { 3 }
@pure fn rev_kind_neg() -> i32 { 4 }
@pure fn rev_tape_magic() -> i32 { 3003001 }
@pure fn rev_tape_footer(cap: i32) -> i32 { 0 - rev_tape_magic() - cap - 1 }
@pure fn rev_tape_footer_with_adj(cap: i32, adj_start: i32) -> i32 {
    rev_tape_footer(cap) - adj_start - 17
}
@pure fn rev_adj_guard(owner: i32, cap: i32, cnt: i32, adj_start: i32) -> i32 {
    rev_tape_footer_with_adj(cap, adj_start) - owner - cnt - 31
}
@pure fn rev_snapshot_footer(owner: i32, cap: i32, cnt: i32, adj_start: i32) -> i32 {
    rev_adj_guard(owner, cap, cnt, adj_start) - 73
}

fn rev_tape_new(cap: i32) -> i32 {
    __arena_push(rev_tape_magic());
    let start = __arena_len();
    let safe_cap = if cap < 0 { 0 } else { cap };
    __arena_push(0);     // count
    __arena_push(safe_cap);   // cap
    __arena_push(0 - 1); // adj_start (set later)
    let mut i: i32 = 0;
    while i < safe_cap {
        __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
        i = i + 1;
    }
    __arena_push(rev_tape_footer(safe_cap));
    start
}

@pure fn rev_count(tape: i32) -> i32 {
    if rev_tape_valid(tape) == 0 { 0 } else { __arena_get(tape) }
}
@pure fn rev_cap(tape: i32) -> i32 {
    if rev_tape_valid(tape) == 0 { 0 - 1 } else { __arena_get(tape + 1) }
}

@pure
fn rev_tape_valid(tape: i32) -> i32 {
    if tape <= 0 { 0 }
    else { if __arena_get(tape - 1) != rev_tape_magic() { 0 }
    else {
        let cnt = __arena_get(tape);
        let cap = __arena_get(tape + 1);
        if cnt < 0 { 0 }
        else { if cap < 0 { 0 }
        else { if cnt > cap { 0 }
        else { if cap > (2147483647 - tape - 3) / 4 { 0 }
        else {
            let footer = tape + 3 + cap * 4;
            if footer >= __arena_len() { 0 }
            else {
                let adj_start = __arena_get(tape + 2);
                if adj_start == (0 - 1) {
                    if __arena_get(footer) != rev_tape_footer(cap) { 0 }
                    else { if cap > (2147483647 - 5) / 4 { 0 }
                    else { if tape > 2147483647 - (5 + cap * 4) { 0 }
                    else { if arena_span_in_tensor_payload(tape - 1, 5 + cap * 4) != 0 { 0 } else { 1 } } } }
                } else { if adj_start < 0 { 0 }
                else {
                    let expected_adj = tape + 3 + cap * 4 + 5;
                    if adj_start != expected_adj { 0 }
                    else { if __arena_get(footer) != rev_tape_footer_with_adj(cap, adj_start) { 0 }
                    else { if cap > (2147483647 - 5) / 4 { 0 }
                    else { if tape > 2147483647 - (5 + cap * 4) { 0 }
                    else { if arena_span_in_tensor_payload(tape - 1, 5 + cap * 4) != 0 { 0 } else { 1 } } } } }
                }}
            }
        }}}}
    }}
}

@pure
fn rev_valid_index(tape: i32, idx: i32) -> i32 {
    if rev_tape_valid(tape) == 0 { 0 }
    else {
        let cnt = __arena_get(tape);
        let cap = __arena_get(tape + 1);
        if idx < 0 { 0 }
        else { if idx >= cnt { 0 }
        else { if idx >= cap { 0 } else { 1 } } }
    }
}

@pure
fn rev_kind_at(tape: i32, idx: i32) -> i32 {
    if rev_valid_index(tape, idx) == 0 { 0 - 1 }
    else { __arena_get(tape + 3 + idx * 4) }
}

@pure
fn rev_in1_at(tape: i32, idx: i32) -> i32 {
    if rev_valid_index(tape, idx) == 0 { 0 - 1 }
    else { __arena_get(tape + 3 + idx * 4 + 1) }
}

@pure
fn rev_in2_at(tape: i32, idx: i32) -> i32 {
    if rev_valid_index(tape, idx) == 0 { 0 - 1 }
    else { __arena_get(tape + 3 + idx * 4 + 2) }
}

@pure
fn rev_is_empty(tape: i32) -> i32 {
    if rev_tape_valid(tape) == 0 { 0 }
    else { if __arena_get(tape) == 0 { 1 } else { 0 } }
}

@pure
fn rev_remaining(tape: i32) -> i32 {
    if rev_tape_valid(tape) == 0 { 0 - 1 }
    else { __arena_get(tape + 1) - __arena_get(tape) }
}

fn rev_push(tape: i32, kind: i32, in1: i32, in2: i32, value: i32) -> i32 {
    if rev_tape_valid(tape) == 0 { 0 - 1 }
    else {
        let cnt = __arena_get(tape);
        let cap = __arena_get(tape + 1);
        let adj_slot = __arena_get(tape + 2);
        if adj_slot != (0 - 1) {
            __arena_set(tape + 3 + cap * 4, 0);
            0 - 1
        }
        else { if cnt < 0 { 0 - 1 }
        else { if cnt >= cap { 0 - 1 }
        else {
            let off = tape + 3 + cnt * 4;
            __arena_set(off, kind);
            __arena_set(off + 1, in1);
            __arena_set(off + 2, in2);
            __arena_set(off + 3, value);
            __arena_set(tape, cnt + 1);
            cnt
        }}}
    }
}

fn rev_leaf(tape: i32, value: i32) -> i32 {
    rev_push(tape, rev_kind_leaf(), 0 - 1, 0 - 1, value)
}

fn rev_add(tape: i32, ai: i32, bi: i32) -> i32 {
    if rev_valid_index(tape, ai) == 0 { 0 - 1 }
    else { if rev_valid_index(tape, bi) == 0 { 0 - 1 }
    else {
        let av = rev_value_at(tape, ai);
        let bv = rev_value_at(tape, bi);
        rev_push(tape, rev_kind_add(), ai, bi, av + bv)
    }}
}

fn rev_sub(tape: i32, ai: i32, bi: i32) -> i32 {
    if rev_valid_index(tape, ai) == 0 { 0 - 1 }
    else { if rev_valid_index(tape, bi) == 0 { 0 - 1 }
    else {
        let av = rev_value_at(tape, ai);
        let bv = rev_value_at(tape, bi);
        rev_push(tape, rev_kind_sub(), ai, bi, av - bv)
    }}
}

fn rev_mul(tape: i32, ai: i32, bi: i32) -> i32 {
    if rev_valid_index(tape, ai) == 0 { 0 - 1 }
    else { if rev_valid_index(tape, bi) == 0 { 0 - 1 }
    else {
        let av = rev_value_at(tape, ai);
        let bv = rev_value_at(tape, bi);
        rev_push(tape, rev_kind_mul(), ai, bi, av * bv)
    }}
}

fn rev_neg(tape: i32, ai: i32) -> i32 {
    if rev_valid_index(tape, ai) == 0 { 0 - 1 }
    else {
        let av = rev_value_at(tape, ai);
        rev_push(tape, rev_kind_neg(), ai, 0 - 1, 0 - av)
    }
}

@pure
fn rev_value_at(tape: i32, idx: i32) -> i32 {
    if rev_valid_index(tape, idx) == 0 { 0 }
    else { __arena_get(tape + 3 + idx * 4 + 3) }
}

@pure
fn rev_expected_adj_start(tape: i32) -> i32 {
    if rev_tape_valid(tape) == 0 { 0 - 1 }
    else {
        let cap = __arena_get(tape + 1);
        if cap > (2147483647 - tape - 8) / 4 { 0 - 1 }
        else { tape + 3 + cap * 4 + 5 }
    }
}

fn rev_alloc_adjoints(tape: i32) -> i32 {
    if rev_tape_valid(tape) == 0 { 0 - 1 }
    else {
        let cap = __arena_get(tape + 1);
        let cnt = __arena_get(tape);
        let expected = rev_expected_adj_start(tape);
        if expected < 0 { 0 - 1 }
        else { if __arena_len() != expected - 4 { 0 - 1 }
        else {
        __arena_push(tape);
        __arena_push(cap);
        __arena_push(cnt);
        __arena_push(rev_adj_guard(tape, cap, cnt, expected));
        let start = __arena_len();
        __arena_set(tape + 2, start);
        let mut i: i32 = 0;
        while i < cap {
            __arena_push(0);
            i = i + 1;
        }
        __arena_push(rev_adj_guard(tape, cap, cnt, start));
        let mut snap_i: i32 = 0;
        let snap_total = cnt * 4;
        while snap_i < snap_total {
            __arena_push(__arena_get(tape + 3 + snap_i));
            snap_i = snap_i + 1;
        }
        __arena_push(rev_snapshot_footer(tape, cap, cnt, start));
        __arena_set(tape + 3 + cap * 4, rev_tape_footer_with_adj(cap, start));
        start
        }}
    }
}

@pure
fn rev_adj_cap(adj_start: i32) -> i32 {
    let mut result: i32 = 0 - 1;
    if adj_start >= 4 {
        let owner = __arena_get(adj_start - 4);
        let cap = __arena_get(adj_start - 3);
        let cnt = __arena_get(adj_start - 2);
        let guard = __arena_get(adj_start - 1);
        if rev_tape_valid(owner) != 0 {
        if rev_expected_adj_start(owner) == adj_start {
        if __arena_get(owner + 2) == adj_start {
        if cap >= 0 {
        if cnt >= 0 {
        if cnt <= cap {
        if cap == __arena_get(owner + 1) {
        if cap <= 2147483647 - adj_start {
            let adj_footer = adj_start + cap;
            if adj_footer < __arena_len() {
                let footer = __arena_get(adj_footer);
                let expected_guard = rev_adj_guard(owner, cap, cnt, adj_start);
                if guard == expected_guard {
                if footer == expected_guard {
                    let snapshot_start = adj_footer + 1;
                    let snapshot_total = cnt * 4;
                    if cnt <= (2147483647 - snapshot_start - 1) / 4 {
                        let snapshot_footer = snapshot_start + snapshot_total;
                        if snapshot_footer < __arena_len() {
                        if __arena_get(snapshot_footer) == rev_snapshot_footer(owner, cap, cnt, adj_start) {
                            let adj_span_len = 6 + cap + snapshot_total;
                            if adj_span_len > 0 {
                            if adj_start >= 4 {
                            if adj_start - 4 <= 2147483647 - adj_span_len {
                            if arena_span_in_tensor_payload(adj_start - 4, adj_span_len) == 0 {
                                let mut snap_i: i32 = 0;
                                let mut snap_ok: i32 = 1;
                                while snap_i < snapshot_total {
                                    if __arena_get(snapshot_start + snap_i) != __arena_get(owner + 3 + snap_i) {
                                        snap_ok = 0;
                                    }
                                    snap_i = snap_i + 1;
                                }
                                if snap_ok != 0 { result = cap; }
                            }}}}
                        }}
                    }
                }}
            }
        }}}}}}}}
    }
    result
}

@pure
fn rev_adj_owner(adj_start: i32) -> i32 {
    let cap = rev_adj_cap(adj_start);
    if cap < 0 { 0 - 1 } else { __arena_get(adj_start - 4) }
}

@pure
fn rev_adj_count(adj_start: i32) -> i32 {
    let cap = rev_adj_cap(adj_start);
    if cap < 0 { 0 - 1 } else { __arena_get(adj_start - 2) }
}

fn rev_seed(adj_start: i32, idx: i32, seed: i32) -> i32 {
    let cnt = rev_adj_count(adj_start);
    if cnt < 0 { 0 - 1 }
    else { if idx < 0 { 0 - 1 }
    else { if idx >= cnt { 0 - 1 }
    else {
        __arena_set(adj_start + idx, seed);
        0
    }}}
}

@pure
fn rev_grad(adj_start: i32, idx: i32) -> i32 {
    let cnt = rev_adj_count(adj_start);
    if cnt < 0 { 0 }
    else { if idx < 0 { 0 }
    else { if idx >= cnt { 0 }
    else { __arena_get(adj_start + idx) }}}
}

// Walk tape in reverse, propagating adjoints.
// For each tape entry of kind K with inputs (a, b) and adjoint adj[i]:
//   K = leaf:  no propagation.
//   K = add:   adj[a] += adj[i]; adj[b] += adj[i]
//   K = sub:   adj[a] += adj[i]; adj[b] -= adj[i]
//   K = mul:   adj[a] += adj[i] * value(b); adj[b] += adj[i] * value(a)
//   K = neg:   adj[a] -= adj[i]
fn rev_backward(tape: i32, adj_start: i32) -> i32 {
    if rev_tape_valid(tape) == 0 { 0 - 1 }
    else {
    let cnt = __arena_get(tape);
    let cap = __arena_get(tape + 1);
    let adj_cap = rev_adj_cap(adj_start);
    let adj_cnt = rev_adj_count(adj_start);
    let adj_owner = rev_adj_owner(adj_start);
    if cnt < 0 { 0 - 1 }
    else { if cnt > cap { 0 - 1 }
    else { if adj_cap < 0 { 0 - 1 }
    else { if adj_owner != tape { 0 - 1 }
    else { if adj_cnt != cnt { 0 - 1 }
    else { if cnt > adj_cap { 0 - 1 }
    else { if __arena_get(tape + 2) != adj_start { 0 - 1 }
    else {
    let mut check_i: i32 = cnt - 1;
    let mut status: i32 = 0;
    while check_i >= 0 {
        if status == 0 {
            let check_off = tape + 3 + check_i * 4;
            let check_kind = __arena_get(check_off);
            let check_in1 = __arena_get(check_off + 1);
            let check_in2 = __arena_get(check_off + 2);
            if check_kind == 0 {
                if check_in1 != (0 - 1) { status = 0 - 1; }
                else { if check_in2 != (0 - 1) { status = 0 - 1; } }
            } else { if check_kind == 1 {
                if rev_valid_index(tape, check_in1) == 0 { status = 0 - 1; }
                else { if rev_valid_index(tape, check_in2) == 0 { status = 0 - 1; }
                else { if check_in1 >= check_i { status = 0 - 1; }
                else { if check_in2 >= check_i { status = 0 - 1; } } } }
            } else { if check_kind == 2 {
                if rev_valid_index(tape, check_in1) == 0 { status = 0 - 1; }
                else { if rev_valid_index(tape, check_in2) == 0 { status = 0 - 1; }
                else { if check_in1 >= check_i { status = 0 - 1; }
                else { if check_in2 >= check_i { status = 0 - 1; } } } }
            } else { if check_kind == 3 {
                if rev_valid_index(tape, check_in1) == 0 { status = 0 - 1; }
                else { if rev_valid_index(tape, check_in2) == 0 { status = 0 - 1; }
                else { if check_in1 >= check_i { status = 0 - 1; }
                else { if check_in2 >= check_i { status = 0 - 1; } } } }
            } else { if check_kind == 4 {
                if rev_valid_index(tape, check_in1) == 0 { status = 0 - 1; }
                else { if check_in1 >= check_i { status = 0 - 1; } }
            } else {
                status = 0 - 1;
            }}}}};
        };
        check_i = check_i - 1;
    }
    let mut i: i32 = cnt - 1;
    while i >= 0 {
        if status == 0 {
            let off = tape + 3 + i * 4;
            let kind = __arena_get(off);
            let in1 = __arena_get(off + 1);
            let in2 = __arena_get(off + 2);
            let adj_i = __arena_get(adj_start + i);
            if kind == 0 {
                status = 0;
            } else { if kind == 1 {
                if rev_valid_index(tape, in1) == 0 { status = 0 - 1; }
                else { if rev_valid_index(tape, in2) == 0 { status = 0 - 1; }
                else {
                    __arena_set(adj_start + in1, __arena_get(adj_start + in1) + adj_i);
                    __arena_set(adj_start + in2, __arena_get(adj_start + in2) + adj_i);
                }}
            } else { if kind == 2 {
                if rev_valid_index(tape, in1) == 0 { status = 0 - 1; }
                else { if rev_valid_index(tape, in2) == 0 { status = 0 - 1; }
                else {
                    __arena_set(adj_start + in1, __arena_get(adj_start + in1) + adj_i);
                    __arena_set(adj_start + in2, __arena_get(adj_start + in2) - adj_i);
                }}
            } else { if kind == 3 {
                if rev_valid_index(tape, in1) == 0 { status = 0 - 1; }
                else { if rev_valid_index(tape, in2) == 0 { status = 0 - 1; }
                else {
                    let v_a = rev_value_at(tape, in1);
                    let v_b = rev_value_at(tape, in2);
                    __arena_set(adj_start + in1, __arena_get(adj_start + in1) + adj_i * v_b);
                    __arena_set(adj_start + in2, __arena_get(adj_start + in2) + adj_i * v_a);
                }}
            } else { if kind == 4 {
                if rev_valid_index(tape, in1) == 0 { status = 0 - 1; }
                else {
                    __arena_set(adj_start + in1, __arena_get(adj_start + in1) - adj_i);
                }
            } else {
                status = 0 - 1;
            }}}}};
        };
        i = i - 1;
    }
    status
    }}}}}}}}
}
