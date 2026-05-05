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

fn tree_node_new(tag: i32, p1: i32, p2: i32, p3: i32) -> i32 {
    let off = __arena_len();
    __arena_push(tag);
    __arena_push(p1);
    __arena_push(p2);
    __arena_push(p3);
    off
}

@pure fn tree_node_tag(off: i32) -> i32 { __arena_get(off) }
@pure fn tree_node_p1(off: i32) -> i32 { __arena_get(off + 1) }
@pure fn tree_node_p2(off: i32) -> i32 { __arena_get(off + 2) }
@pure fn tree_node_p3(off: i32) -> i32 { __arena_get(off + 3) }

// Structural equality: compare tag + p1 + p2 + p3 at the top level
// only. For deep equality, the caller recurses on child nodes.
@pure
fn tree_eq_shallow(a: i32, b: i32) -> i32 {
    if __arena_get(a) == __arena_get(b) {
        if __arena_get(a + 1) == __arena_get(b + 1) {
            if __arena_get(a + 2) == __arena_get(b + 2) {
                if __arena_get(a + 3) == __arena_get(b + 3) {
                    1
                } else { 0 }
            } else { 0 }
        } else { 0 }
    } else { 0 }
}

// Stable hash: combines tag/p1/p2/p3 into a single i32 via shifts.
// (Pseudo-random-mixing without bitwise ops; uses arithmetic only.
// For Phase 4 step 4 a deterministic-but-not-cryptographic hash is
// sufficient for use as WM keys / dedup probes.)
@pure
fn tree_hash_shallow(off: i32) -> i32 {
    let tag = __arena_get(off);
    let p1 = __arena_get(off + 1);
    let p2 = __arena_get(off + 2);
    let p3 = __arena_get(off + 3);
    // 4-byte-rotated linear-combination.
    tag * 31 * 31 * 31 + p1 * 31 * 31 + p2 * 31 + p3
}

// Bag (multiset) similarity by intersection size. Both arrays are
// arena-stored (start + n). Counts elements of a that also appear in b.
// O(n*m) but fine for Phase-4 working-memory-sized comparisons.
@pure
fn bag_similarity(a_start: i32, a_n: i32, b_start: i32, b_n: i32) -> i32 {
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
}

// Levenshtein-like sequence similarity for AGI string-of-tokens matching.
// Returns the count of equal-position elements (Hamming distance complement).
// Both sequences must be the same length n.
@pure
fn sequence_match(a_start: i32, b_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut hits: i32 = 0;
    while i < n {
        if __arena_get(a_start + i) == __arena_get(b_start + i) {
            hits = hits + 1;
        }
        i = i + 1;
    }
    hits
}
