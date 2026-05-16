// helixc/stdlib/agi_match.hx — pattern/similarity primitives.
//
// Phase 4 step 4: tree-shaped pattern matching for the AGI. Trees are
// arena-stored as flat (tag, p1, p2, p3) tuples; nodes reference other
// nodes by arena offset. This mirrors how the helixc parser builds
// AST nodes and gives the AGI a uniform substrate for symbolic reasoning.
//
// API:
//   tree_node_new(arena, tag, p1, p2, p3) -> i32
//        push a 4-slot node, return its arena offset
//   tree_node_tag(off)        -> i32   read tag of node at offset
//   tree_node_p1/p2/p3(off)   -> i32   read payload slots
//
//   tree_eq(a_off, b_off)     -> i32   structural equality (1/0)
//   tree_size(off)            -> i32   total node count under the subtree
//   tree_hash(off)            -> i32   stable structural hash
//
// API for bag-of-features similarity (for fast nearest-neighbor in WM):
//   bag_similarity(a_start, a_n, b_start, b_n) -> i32
//        intersection size (number of shared elements)
//
// License: Apache 2.0

@pure fn tree_node_magic() -> i32 { 7007001 }
@pure fn tree_node_footer() -> i32 { 0 - tree_node_magic() - 4 }

fn tree_node_new(tag: i32, p1: i32, p2: i32, p3: i32) -> i32 {
    __arena_push(tree_node_magic());
    let off = __arena_len();
    __arena_push(tag);
    __arena_push(p1);
    __arena_push(p2);
    __arena_push(p3);
    __arena_push(tree_node_footer());
    off
}

@pure
fn tree_node_ok(off: i32) -> i32 {
    if off <= 0 { 0 }
    else { if off > 2147483647 - 4 { 0 }
    else { if off + 4 >= __arena_len() { 0 }
    else { if __arena_get(off - 1) != tree_node_magic() { 0 }
    else { if __arena_get(off + 4) != tree_node_footer() { 0 }
    else { if arena_span_in_tensor_payload(off - 1, 6) != 0 { 0 } else { 1 } } } } } }
}

@pure fn tree_invalid_value() -> i32 { (0 - 2147483647) - 1 }
@pure fn tree_node_tag(off: i32) -> i32 {
    if tree_node_ok(off) == 0 { tree_invalid_value() } else { __arena_get(off) }
}
@pure fn tree_node_p1(off: i32) -> i32 {
    if tree_node_ok(off) == 0 { tree_invalid_value() } else { __arena_get(off + 1) }
}
@pure fn tree_node_p2(off: i32) -> i32 {
    if tree_node_ok(off) == 0 { tree_invalid_value() } else { __arena_get(off + 2) }
}
@pure fn tree_node_p3(off: i32) -> i32 {
    if tree_node_ok(off) == 0 { tree_invalid_value() } else { __arena_get(off + 3) }
}

// Structural equality: compare tag + p1 + p2 + p3 at the top level
// only. For deep equality, the caller recurses on child nodes.
@pure
fn tree_eq_shallow(a: i32, b: i32) -> i32 {
    if tree_node_ok(a) == 0 { 0 }
    else { if tree_node_ok(b) == 0 { 0 }
    else {
    if __arena_get(a) == __arena_get(b) {
        if __arena_get(a + 1) == __arena_get(b + 1) {
            if __arena_get(a + 2) == __arena_get(b + 2) {
                if __arena_get(a + 3) == __arena_get(b + 3) {
                    1
                } else { 0 }
            } else { 0 }
        } else { 0 }
    } else { 0 }
    }}
}

// Stable hash: combines tag/p1/p2/p3 into a single i32 via shifts.
// (Pseudo-random-mixing without bitwise ops; uses arithmetic only.
// For Phase 4 step 4 a deterministic-but-not-cryptographic hash is
// sufficient for use as WM keys / dedup probes.)
@pure
fn tree_hash_shallow(off: i32) -> i32 {
    if tree_node_ok(off) == 0 { 0 }
    else {
    let tag = __arena_get(off);
    let p1 = __arena_get(off + 1);
    let p2 = __arena_get(off + 2);
    let p3 = __arena_get(off + 3);
    // 4-byte-rotated linear-combination.
    tag * 31 * 31 * 31 + p1 * 31 * 31 + p2 * 31 + p3
    }
}

// Bag (multiset) similarity by intersection size. Both arrays are
// arena-stored (start + n). Counts elements of a that also appear in b.
// O(n*m) but fine for Phase-4 working-memory-sized comparisons.
@pure
fn bag_similarity(a_start: i32, a_n: i32, b_start: i32, b_n: i32) -> i32 {
    if t1d_slice_ok(a_start, a_n) == 0 { 0 }
    else { if t1d_slice_ok(b_start, b_n) == 0 { 0 }
    else {
    let mut shared: i32 = 0;
    let mut i: i32 = 0;
    while i < a_n {
        let ai = __arena_get(a_start + i);
        let mut j: i32 = 0;
        let mut hit: i32 = 0;
        while j < b_n {
            if __arena_get(b_start + j) == ai {
                if hit == 0 { hit = 1; }
            }
            j = j + 1;
        }
        shared = shared + hit;
        i = i + 1;
    }
    shared
    }}
}

// Asymmetric bag difference: count of a-positions whose value does NOT
// appear in b. Mirror of bag_similarity (which counts a-positions IN b);
// invariant: bag_similarity(a, b) + bag_difference(a, b) == a_n.
@pure
fn bag_difference(a_start: i32, a_n: i32, b_start: i32, b_n: i32) -> i32 {
    if t1d_slice_ok(a_start, a_n) == 0 { 0 }
    else { if t1d_slice_ok(b_start, b_n) == 0 { 0 }
    else {
    let mut diff: i32 = 0;
    let mut i: i32 = 0;
    while i < a_n {
        let ai = __arena_get(a_start + i);
        let mut j: i32 = 0;
        let mut hit: i32 = 0;
        while j < b_n {
            if __arena_get(b_start + j) == ai {
                if hit == 0 { hit = 1; }
            }
            j = j + 1;
        }
        if hit == 0 { diff = diff + 1; }
        i = i + 1;
    }
    diff
    }}
}

// Count of distinct values in a bag (multiset). [1,2,2,3,1] -> 3.
// Quadratic dedup-by-scan: a value contributes 1 iff its earlier
// occurrences are all distinct from it (i.e. it's the first time we see
// it in the array). Empty array -> 0.
@pure
fn bag_count_unique(a_start: i32, a_n: i32) -> i32 {
    if t1d_slice_ok(a_start, a_n) == 0 { 0 }
    else {
    let mut uniq: i32 = 0;
    let mut i: i32 = 0;
    while i < a_n {
        let ai = __arena_get(a_start + i);
        let mut j: i32 = 0;
        let mut seen: i32 = 0;
        while j < i {
            if __arena_get(a_start + j) == ai {
                if seen == 0 { seen = 1; }
            }
            j = j + 1;
        }
        if seen == 0 { uniq = uniq + 1; }
        i = i + 1;
    }
    uniq
    }
}

// Levenshtein-like sequence similarity for AGI string-of-tokens matching.
// Returns the count of equal-position elements (Hamming distance complement).
// Both sequences must be the same length n.
@pure
fn sequence_match(a_start: i32, b_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(a_start, n) == 0 { 0 }
    else { if t1d_slice_ok(b_start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        if __arena_get(a_start + i) == __arena_get(b_start + i) {
            total = total + 1;
        }
        i = i + 1;
    }
    total
    }}}
}

// =========================================================================
// Phase 4 perfection: unification with variables.
// =========================================================================
//
// A pattern is a tree where some leaves are "variables" — placeholder slots
// that match anything. Unification: try to make pattern = term by binding
// each variable to a concrete sub-term. Used for symbolic AGI (rule
// matching, equation solving, planning by analogy).
//
// Encoding:
//   tree_node_new(VAR_TAG, var_id, 0, 0)    — a pattern variable
//   tree_node_new(otherTag, p1, p2, p3)     — a concrete node
//
// Bindings: an array of (var_id -> arena_offset). Capacity bounded.

@pure fn unify_var_tag() -> i32 { 0 - 1 }   // -1 = "this node is a variable"

// Predicate: 1 if the node at off is a unification variable (tag ==
// unify_var_tag()), else 0. Saves callers from importing the var-tag
// constant at the use site when they only want to ask "is this a
// placeholder?". Pairs with bindings_get when walking partially-
// instantiated patterns.
@pure
fn tree_node_is_var(off: i32) -> i32 {
    if tree_node_ok(off) == 0 { 0 }
    else {
    if __arena_get(off) == unify_var_tag() { 1 } else { 0 }
    }
}

@pure fn bindings_magic() -> i32 { 7008001 }
@pure fn bindings_footer() -> i32 { 0 - bindings_magic() - 65 }

fn bindings_new() -> i32 {
    __arena_push(bindings_magic());
    let start = __arena_len();
    __arena_push(0);   // count
    let mut i: i32 = 0;
    while i < 32 {
        __arena_push(0 - 1);   // var_id
        __arena_push(0);       // bound arena offset
        i = i + 1;
    }
    __arena_push(bindings_footer());
    start
}

@pure
fn bindings_storage_ok(b: i32) -> i32 {
    if b <= 0 { 0 }
    else { if b > 2147483647 - 65 { 0 }
    else { if b + 65 >= __arena_len() { 0 }
    else { if __arena_get(b - 1) != bindings_magic() { 0 }
    else { if __arena_get(b + 65) != bindings_footer() { 0 }
    else { if arena_span_in_tensor_payload(b - 1, 67) != 0 { 0 } else { 1 } } } } } }
}

@pure
fn bindings_get(b: i32, var_id: i32) -> i32 {
    if bindings_storage_ok(b) == 0 { 0 - 1 }
    else {
    let cnt = __arena_get(b);
    if cnt < 0 { 0 - 1 }
    else { if cnt > 32 { 0 - 1 }
    else {
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < cnt {
        if __arena_get(b + 1 + i * 2) == var_id {
            if found < 0 { found = __arena_get(b + 1 + i * 2 + 1); }
        }
        i = i + 1;
    }
    found
    }}
    }
}

fn bindings_set(b: i32, var_id: i32, term: i32) -> i32 {
    if bindings_storage_ok(b) == 0 { 0 - 1 }
    else {
    let cnt = __arena_get(b);
    if cnt < 0 { 0 - 1 }
    else { if cnt >= 32 { 0 - 1 }
    else {
        __arena_set(b + 1 + cnt * 2, var_id);
        __arena_set(b + 1 + cnt * 2 + 1, term);
        __arena_set(b, cnt + 1);
        0
    }}
    }
}

fn bindings_rewind(b: i32, count: i32) -> i32 {
    if bindings_storage_ok(b) == 0 { 0 - 1 }
    else { if count < 0 { 0 - 1 }
    else { if count > 32 { 0 - 1 }
    else {
        __arena_set(b, count);
        0
    } } }
}

// Single-level unify: if pat is a var, bind it; else compare tags + payload.
// Returns 1 on success, 0 on failure. Sub-tree unification is the caller's
// responsibility (recurse on p1, p2, p3 if they're tree refs).
fn unify_shallow(pat_off: i32, term_off: i32, b: i32) -> i32 {
    if tree_node_ok(pat_off) == 0 { 0 }
    else { if tree_node_ok(term_off) == 0 { 0 }
    else { if bindings_storage_ok(b) == 0 { 0 }
    else {
    let pat_tag = __arena_get(pat_off);
    if pat_tag == unify_var_tag() {
        let var_id = __arena_get(pat_off + 1);
        let existing = bindings_get(b, var_id);
        if existing < 0 {
            if bindings_set(b, var_id, term_off) == 0 { 1 } else { 0 }
        } else {
            // Already bound: must match the existing binding.
            tree_eq_shallow(existing, term_off)
        }
    } else {
        tree_eq_shallow(pat_off, term_off)
    }
    }}}
}

// Deep unify: tags must match; recursively unify each child slot
// interpreted as an arena offset. Children-as-offsets convention:
// p1, p2, p3 are EITHER scalar values (unrelated to the tree) OR
// arena offsets to other tree nodes. The caller signals which by the
// `child_mask`: bit i (1<<i) set means slot p_i is a sub-tree offset.
//
// Example: a binary-op node tagged 1 with operands on p1, p2 uses
// child_mask = 3 (binary 011 — both slots are sub-trees).
//
// LIMITATION (audit fix #2): this fn applies the SAME child_mask to
// every recursive level. It is correct ONLY for HOMOGENEOUS trees
// where every node has the same shape (all binary, or all unary, or
// all leaves). For mixed-shape trees use `unify_deep_table` (Phase 4
// perfection step 4) which looks up the child_mask per-tag from a
// caller-provided table.
//
// Returns 1 on success, 0 on failure.
@partial
fn unify_deep(pat_off: i32, term_off: i32, child_mask: i32, b: i32) -> i32 {
    if tree_node_ok(pat_off) == 0 { 0 }
    else { if tree_node_ok(term_off) == 0 { 0 }
    else { if bindings_storage_ok(b) == 0 { 0 }
    else {
    let start_count = __arena_get(b);
    let pat_tag = __arena_get(pat_off);
    if pat_tag == unify_var_tag() {
        let var_id = __arena_get(pat_off + 1);
        let existing = bindings_get(b, var_id);
        if existing < 0 {
            if bindings_set(b, var_id, term_off) == 0 { 1 } else { 0 }
        } else {
            // Already-bound var: existing must structurally match term.
            unify_deep(existing, term_off, child_mask, b)
        }
    } else {
        // Concrete node: tags must match.
        let term_tag = __arena_get(term_off);
        if pat_tag == term_tag {
            // Recurse into each child marked by mask.
            let mut ok: i32 = 1;
            // child slot 0 is p1, slot 1 is p2, slot 2 is p3.
            if child_mask % 2 == 1 {
                let pc = __arena_get(pat_off + 1);
                let tc = __arena_get(term_off + 1);
                if unify_deep(pc, tc, child_mask, b) == 0 { ok = 0; }
            } else {
                if __arena_get(pat_off + 1) != __arena_get(term_off + 1) { ok = 0; }
            }
            let m1 = (child_mask / 2) % 2;
            if m1 == 1 {
                let pc = __arena_get(pat_off + 2);
                let tc = __arena_get(term_off + 2);
                if unify_deep(pc, tc, child_mask, b) == 0 { ok = 0; }
            } else {
                if __arena_get(pat_off + 2) != __arena_get(term_off + 2) { ok = 0; }
            }
            let m2 = (child_mask / 4) % 2;
            if m2 == 1 {
                let pc = __arena_get(pat_off + 3);
                let tc = __arena_get(term_off + 3);
                if unify_deep(pc, tc, child_mask, b) == 0 { ok = 0; }
            } else {
                if __arena_get(pat_off + 3) != __arena_get(term_off + 3) { ok = 0; }
            }
            if ok == 0 { bindings_rewind(b, start_count); }
            ok
        } else {
            0
        }
    }
    }}}
}

// =========================================================================
// Phase 4 perfection: per-tag child_mask via lookup table.
// =========================================================================
//
// `unify_deep` above passes ONE child_mask down through every level of
// recursion — all nodes in the tree must share the same shape. That's
// fine for homogeneous trees (all binary, or all unary) but breaks for
// mixed-shape trees: a binary at the root with leaves underneath needs
// child_mask=3 at the root but child_mask=0 at the leaves.
//
// `unify_deep_table` looks up child_mask via tag from a caller-provided
// arena array indexed by tag. Each node level uses its OWN mask, so
// mixed-shape trees compose cleanly. Tags outside the table use mask 0
// (treat all slots as scalars).
//
// Usage:
//   let mask_table = __arena_len();
//   __arena_push(0);     // mask[0] = 0 (leaf)
//   __arena_push(1);     // mask[1] = 1 (unary: only p1 is sub-tree)
//   __arena_push(3);     // mask[2] = 3 (binary: p1 + p2 are sub-trees)
//   ...
//   unify_deep_table(pat, term, mask_table, 3, b)
//
// Returns 1 on success, 0 on failure.
@partial
fn unify_deep_table(pat_off: i32, term_off: i32, mask_table: i32,
                    mask_table_len: i32, b: i32) -> i32 {
    if tree_node_ok(pat_off) == 0 { 0 }
    else { if tree_node_ok(term_off) == 0 { 0 }
    else { if bindings_storage_ok(b) == 0 { 0 }
    else {
    let start_count = __arena_get(b);
    let pat_tag = __arena_get(pat_off);
    if pat_tag == unify_var_tag() {
        let var_id = __arena_get(pat_off + 1);
        let existing = bindings_get(b, var_id);
        if existing < 0 {
            if bindings_set(b, var_id, term_off) == 0 { 1 } else { 0 }
        } else {
            // Already-bound var: existing must structurally match term
            // (recurse with the same table — the existing tree may have
            // its own per-tag masks).
            unify_deep_table(existing, term_off, mask_table, mask_table_len, b)
        }
    } else {
        let term_tag = __arena_get(term_off);
        if pat_tag == term_tag {
            // Look up THIS tag's child_mask from the table.
            let mut my_mask: i32 = 0;
            if pat_tag >= 0 {
                if pat_tag < mask_table_len {
                    my_mask = __arena_get(mask_table + pat_tag);
                }
            }
            let mut ok: i32 = 1;
            // Slot 0 (p1)
            if my_mask % 2 == 1 {
                let pc = __arena_get(pat_off + 1);
                let tc = __arena_get(term_off + 1);
                if unify_deep_table(pc, tc, mask_table, mask_table_len, b) == 0 {
                    ok = 0;
                }
            } else {
                if __arena_get(pat_off + 1) != __arena_get(term_off + 1) { ok = 0; }
            }
            // Slot 1 (p2)
            let m1 = (my_mask / 2) % 2;
            if m1 == 1 {
                let pc = __arena_get(pat_off + 2);
                let tc = __arena_get(term_off + 2);
                if unify_deep_table(pc, tc, mask_table, mask_table_len, b) == 0 {
                    ok = 0;
                }
            } else {
                if __arena_get(pat_off + 2) != __arena_get(term_off + 2) { ok = 0; }
            }
            // Slot 2 (p3)
            let m2 = (my_mask / 4) % 2;
            if m2 == 1 {
                let pc = __arena_get(pat_off + 3);
                let tc = __arena_get(term_off + 3);
                if unify_deep_table(pc, tc, mask_table, mask_table_len, b) == 0 {
                    ok = 0;
                }
            } else {
                if __arena_get(pat_off + 3) != __arena_get(term_off + 3) { ok = 0; }
            }
            if ok == 0 { bindings_rewind(b, start_count); }
            ok
        } else {
            0
        }
    }
    }}}
}

// =========================================================================
// Phase 4 perfection: hierarchical planning.
// =========================================================================
//
// Split a goal into sub-goals; track which sub-goals are achieved. Returns
// the count of completed sub-goals. The actual sub-goal achievement check
// is the caller's predicate (passed as a scoring table indexed by sub-goal id).

@pure
fn hier_count_achieved(subgoal_ids_start: i32, n: i32, achieved_table: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(subgoal_ids_start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut done: i32 = 0;
    while i < n {
        let sg = __arena_get(subgoal_ids_start + i);
        if sg >= 0 {
        if sg < 2147483647 {
        if t1d_slice_ok(achieved_table, sg + 1) == 1 {
        if __arena_get(achieved_table + sg) == 1 {
            done = done + 1;
        };
        };
        };
        };
        i = i + 1;
    }
    done
    }}
}

// =========================================================================
// Phase 4 perfection: ensemble world model.
// =========================================================================
//
// Average predictions from N models. Used to quantify uncertainty:
// agreement => high confidence; disagreement => low confidence.

@pure
fn ensemble_mean(predictions_start: i32, n: i32) -> i32 {
    if n == 0 { 0 }
    else { if t1d_slice_ok(predictions_start, n) == 0 { 0 }
    else {
        let mut i: i32 = 0;
        let mut total: i64 = 0_i64;
        while i < n {
            total = total + (__arena_get(predictions_start + i) as i64);
            i = i + 1;
        }
        let mean: i64 = total / (n as i64);
        if mean > 2147483647_i64 { 2147483647 }
        else { if mean < ((0_i64 - 2147483647_i64) - 1_i64) { (0 - 2147483647) - 1 }
        else { mean as i32 } }
    }}
}

// Range = max - min: simple uncertainty estimate.
@pure
fn ensemble_uncertainty(predictions_start: i32, n: i32) -> i32 {
    if n == 0 { 0 }
    else { if t1d_slice_ok(predictions_start, n) == 0 { 0 }
    else {
        let mut i: i32 = 1;
        let mut lo = __arena_get(predictions_start);
        let mut hi = lo;
        while i < n {
            let v = __arena_get(predictions_start + i);
            if v < lo { lo = v; }
            if v > hi { hi = v; }
            i = i + 1;
        }
        let range: i64 = (hi as i64) - (lo as i64);
        if range > 2147483647_i64 { 2147483647 } else { range as i32 }
    }}
}

// Index of the strictly-largest prediction; returns -1 on empty.
// Ties broken by lowest index (first occurrence wins).  Useful for
// model-selection ("which ensemble member predicted the highest value")
// or for argmax-over-action-values when the ensemble is per-action.
@pure
fn ensemble_argmax(predictions_start: i32, n: i32) -> i32 {
    if n == 0 { 0 - 1 }
    else { if t1d_slice_ok(predictions_start, n) == 0 { 0 - 1 }
    else {
        let mut i: i32 = 1;
        let mut best_v = __arena_get(predictions_start);
        let mut best_i: i32 = 0;
        while i < n {
            let v = __arena_get(predictions_start + i);
            if v > best_v {
                best_v = v;
                best_i = i;
            }
            i = i + 1;
        }
        best_i
    }}
}
