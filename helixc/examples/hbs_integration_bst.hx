// hbs_integration_bst.hx
//
// Comprehensive integration test #2: binary search tree with insert,
// search, and in-order traversal accumulating to a vector.
//
// Exercises:
//   - Recursive enum (Tree { Empty, Node(value, left_tree, right_tree) })
//   - Pattern match on payload-bearing variants with multi-arg payload
//   - Recursive functions returning recursive enum values (via arena)
//   - Inline enum constructor as fn arg AND fn return value
//   - Vec-over-arena pattern (start, count) for traversal accumulator
//   - Conditional dispatch on enum tag

enum Tree {
    Empty,
    Node(i32, Tree, Tree),    // (value, left, right)
}

@partial
fn tree_insert(t: Tree, v: i32) -> Tree {
    match t {
        Tree::Empty => Tree::Node(v, Tree::Empty, Tree::Empty),
        Tree::Node(cur, l, r) => {
            if v < cur {
                Tree::Node(cur, tree_insert(l, v), r)
            } else {
                Tree::Node(cur, l, tree_insert(r, v))
            }
        },
    }
}

@partial
fn tree_contains(t: Tree, v: i32) -> i32 {
    match t {
        Tree::Empty => 0,
        Tree::Node(cur, l, r) => {
            if v == cur { 1 }
            else {
                if v < cur {
                    tree_contains(l, v)
                } else {
                    tree_contains(r, v)
                }
            }
        },
    }
}

// In-order traversal: pushes each value into the arena starting at
// `out_start`. Returns the new count.
@partial
fn tree_inorder_into(t: Tree, count: i32) -> i32 {
    match t {
        Tree::Empty => count,
        Tree::Node(cur, l, r) => {
            let after_left = tree_inorder_into(l, count);
            let _ = __arena_push(cur);
            let after_self = after_left + 1;
            tree_inorder_into(r, after_self)
        },
    }
}

@partial
fn tree_size(t: Tree) -> i32 {
    match t {
        Tree::Empty => 0,
        Tree::Node(_, l, r) => 1 + tree_size(l) + tree_size(r),
    }
}

@partial
fn tree_max_depth(t: Tree) -> i32 {
    match t {
        Tree::Empty => 0,
        Tree::Node(_, l, r) => {
            let dl = tree_max_depth(l);
            let dr = tree_max_depth(r);
            if dl > dr { 1 + dl } else { 1 + dr }
        },
    }
}

// Sum of values via the in-order traversal output.
@partial
fn arena_sum(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < count {
        total = total + __arena_get(start + i);
        i = i + 1;
    }
    total
}

fn main() -> i32 {
    // Build a BST by inserting [10, 5, 15, 3, 7, 12, 20].
    let t0 = Tree::Empty;
    let t1 = tree_insert(t0, 10);
    let t2 = tree_insert(t1, 5);
    let t3 = tree_insert(t2, 15);
    let t4 = tree_insert(t3, 3);
    let t5 = tree_insert(t4, 7);
    let t6 = tree_insert(t5, 12);
    let t = tree_insert(t6, 20);

    // Verify size + depth.
    let n_nodes = tree_size(t);              // 7
    let depth = tree_max_depth(t);           // 3 (a balanced-ish tree)

    // Search for present and absent values.
    let has_7 = tree_contains(t, 7);         // 1
    let has_99 = tree_contains(t, 99);       // 0

    // In-order traversal: should produce [3, 5, 7, 10, 12, 15, 20] in arena.
    let trav_start = __arena_len();
    let trav_count = tree_inorder_into(t, 0);   // 7
    let trav_sum = arena_sum(trav_start, trav_count); // 72

    // Verify in-order property: arena[start+0] should be 3 (smallest).
    let smallest = __arena_get(trav_start);
    let largest = __arena_get(trav_start + trav_count - 1);

    // Compose: 72 - 30 - smallest(3) + has_7(1) + size(7) - depth(3) -
    //          has_99(0) - largest(20) + 18 = 72-30-3+1+7-3-0-20+18 = 42
    trav_sum - 30 - smallest + has_7 + n_nodes - depth - has_99 - largest + 18
}
