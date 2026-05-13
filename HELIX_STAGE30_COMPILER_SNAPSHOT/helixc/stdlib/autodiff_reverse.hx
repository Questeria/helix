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

fn rev_tape_new(cap: i32) -> i32 {
    let start = __arena_len();
    __arena_push(0);     // count
    __arena_push(cap);   // cap
    __arena_push(0 - 1); // adj_start (set later)
    let mut i: i32 = 0;
    while i < cap {
        __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
        i = i + 1;
    }
    start
}

@pure fn rev_count(tape: i32) -> i32 { __arena_get(tape) }
@pure fn rev_cap(tape: i32) -> i32 { __arena_get(tape + 1) }

@pure
fn rev_kind_at(tape: i32, idx: i32) -> i32 {
    __arena_get(tape + 3 + idx * 4)
}

@pure
fn rev_in1_at(tape: i32, idx: i32) -> i32 {
    __arena_get(tape + 3 + idx * 4 + 1)
}

@pure
fn rev_in2_at(tape: i32, idx: i32) -> i32 {
    __arena_get(tape + 3 + idx * 4 + 2)
}

@pure
fn rev_is_empty(tape: i32) -> i32 {
    if __arena_get(tape) == 0 { 1 } else { 0 }
}

@pure
fn rev_remaining(tape: i32) -> i32 {
    __arena_get(tape + 1) - __arena_get(tape)
}

fn rev_push(tape: i32, kind: i32, in1: i32, in2: i32, value: i32) -> i32 {
    let cnt = __arena_get(tape);
    let off = tape + 3 + cnt * 4;
    __arena_set(off, kind);
    __arena_set(off + 1, in1);
    __arena_set(off + 2, in2);
    __arena_set(off + 3, value);
    __arena_set(tape, cnt + 1);
    cnt
}

fn rev_leaf(tape: i32, value: i32) -> i32 {
    rev_push(tape, rev_kind_leaf(), 0 - 1, 0 - 1, value)
}

fn rev_add(tape: i32, ai: i32, bi: i32) -> i32 {
    let av = rev_value_at(tape, ai);
    let bv = rev_value_at(tape, bi);
    rev_push(tape, rev_kind_add(), ai, bi, av + bv)
}

fn rev_sub(tape: i32, ai: i32, bi: i32) -> i32 {
    let av = rev_value_at(tape, ai);
    let bv = rev_value_at(tape, bi);
    rev_push(tape, rev_kind_sub(), ai, bi, av - bv)
}

fn rev_mul(tape: i32, ai: i32, bi: i32) -> i32 {
    let av = rev_value_at(tape, ai);
    let bv = rev_value_at(tape, bi);
    rev_push(tape, rev_kind_mul(), ai, bi, av * bv)
}

fn rev_neg(tape: i32, ai: i32) -> i32 {
    let av = rev_value_at(tape, ai);
    rev_push(tape, rev_kind_neg(), ai, 0 - 1, 0 - av)
}

@pure
fn rev_value_at(tape: i32, idx: i32) -> i32 {
    __arena_get(tape + 3 + idx * 4 + 3)
}

fn rev_alloc_adjoints(tape: i32) -> i32 {
    let cap = __arena_get(tape + 1);
    let start = __arena_len();
    __arena_set(tape + 2, start);
    let mut i: i32 = 0;
    while i < cap {
        __arena_push(0);
        i = i + 1;
    }
    start
}

fn rev_seed(adj_start: i32, idx: i32, seed: i32) -> i32 {
    __arena_set(adj_start + idx, seed);
    0
}

@pure
fn rev_grad(adj_start: i32, idx: i32) -> i32 {
    __arena_get(adj_start + idx)
}

// Walk tape in reverse, propagating adjoints.
// For each tape entry of kind K with inputs (a, b) and adjoint adj[i]:
//   K = leaf:  no propagation.
//   K = add:   adj[a] += adj[i]; adj[b] += adj[i]
//   K = sub:   adj[a] += adj[i]; adj[b] -= adj[i]
//   K = mul:   adj[a] += adj[i] * value(b); adj[b] += adj[i] * value(a)
//   K = neg:   adj[a] -= adj[i]
fn rev_backward(tape: i32, adj_start: i32) -> i32 {
    let cnt = __arena_get(tape);
    let mut i: i32 = cnt - 1;
    while i >= 0 {
        let off = tape + 3 + i * 4;
        let kind = __arena_get(off);
        let in1 = __arena_get(off + 1);
        let in2 = __arena_get(off + 2);
        let adj_i = __arena_get(adj_start + i);
        if kind == 1 {
            // add
            __arena_set(adj_start + in1, __arena_get(adj_start + in1) + adj_i);
            __arena_set(adj_start + in2, __arena_get(adj_start + in2) + adj_i);
        } else { if kind == 2 {
            // sub
            __arena_set(adj_start + in1, __arena_get(adj_start + in1) + adj_i);
            __arena_set(adj_start + in2, __arena_get(adj_start + in2) - adj_i);
        } else { if kind == 3 {
            // mul: adj[a] += adj[i]*v(b); adj[b] += adj[i]*v(a)
            let v_a = rev_value_at(tape, in1);
            let v_b = rev_value_at(tape, in2);
            __arena_set(adj_start + in1, __arena_get(adj_start + in1) + adj_i * v_b);
            __arena_set(adj_start + in2, __arena_get(adj_start + in2) + adj_i * v_a);
        } else { if kind == 4 {
            // neg: adj[a] -= adj[i]
            __arena_set(adj_start + in1, __arena_get(adj_start + in1) - adj_i);
        }}}};
        i = i - 1;
    }
    0
}
