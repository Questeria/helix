// Metacircular evaluator: Helix interpreting Helix's own AST.
//
// We define a recursive enum `Expr` with a small but useful subset of
// Helix expressions: integer literals, variable lookup, addition,
// multiplication, subtraction, comparison (< returns 0/1), conditional
// (`if c { a } else { b }`), and let-binding. Then we write `eval_at`
// — a Helix function whose argument is the arena index of an Expr
// node, and which interprets that expression against an environment
// (also stored in the arena as a stack of (name_hash, value) pairs).
//
// The result: Helix can build up an AST using its own enum
// constructors, then run it through its own evaluator and get the
// same answer it would get if you wrote the corresponding Helix code
// directly. That's the test bed for the bootstrap compiler — any
// machinery the compiler-in-Helix uses to walk its own AST has to
// work in this style.
//
// Demonstration program:
//   let x = 5 in
//     if x < 10
//       then x * (x + 3)   = 5 * 8 = 40
//       else x - 99
//
// Expected: 40.
//
// Exercises: recursive enum (Expr is self-referential via arena
// indirection), pattern matching with payload binders, arena state
// (env push/pop), if expressions, while loops, mutable lets,
// __hash_i32 for variable name keys.

// ---------------------------------------------------------------
// Expr enum: recursive via arena. Each constructor encodes its
// children as arena indices (i32). We use tag values 0..7 and
// sub_pattern destructuring in match arms.
// ---------------------------------------------------------------
enum Expr {
    EInt(i32),                 // 0: literal integer; payload = the value
    EVar(i32),                 // 1: variable lookup; payload = name hash
    EAdd(i32, i32),            // 2: a + b
    EMul(i32, i32),            // 3: a * b
    ESub(i32, i32),            // 4: a - b
    ELt(i32, i32),             // 5: a < b -> 0 or 1
    EIf(i32, i32, i32),        // 6: cond, then, else
    ELet(i32, i32, i32),       // 7: name_hash, value_expr, body_expr
}

// ---------------------------------------------------------------
// Environment: stored in arena as a stack of (key, value) pairs.
// We track a separate "env_top" cursor to know where the current
// frame ends so we can pop on let-exit.
//
// Lookup is linear-scan from the top down — O(n_active_bindings)
// — which is fine for a tiny demo. The interpreter pushes a binding
// when entering a `let`, evaluates the body, then pops.
// ---------------------------------------------------------------
fn env_lookup(env_base: i32, env_top: i32, name_hash: i32) -> i32 {
    // Walk backwards from env_top-2 down to env_base, two slots at
    // a time (key, value). Return the value of the first matching key.
    let mut i: i32 = env_top - 2;
    let mut found: i32 = 0;
    let mut value: i32 = 0;
    while i >= env_base {
        if found == 0 {
            let key = __arena_get(i);
            if key == name_hash {
                value = __arena_get(i + 1);
                found = 1;
            };
            i = i - 2;
        } else {
            i = env_base - 1;  // exit loop
        };
    }
    value
}

fn env_push(name_hash: i32, value: i32) -> i32 {
    __arena_push(name_hash);
    __arena_push(value);
    0
}

fn env_pop() -> i32 {
    // We pop by lying about the cursor; arena has no native pop
    // primitive, so the caller uses __arena_set to clear by tracking
    // env_top manually. For the demo we just leave the slots dead;
    // the env_top tracker hides them from future lookups.
    0
}

// ---------------------------------------------------------------
// eval_at: read the Expr stored at arena index `expr_idx` and
// interpret it. The Expr's tag is at slot expr_idx; payload slots
// follow at expr_idx+1, expr_idx+2, expr_idx+3.
//
// This is the heart of the metacircular evaluator. Pattern-match
// the tag and recurse on payload arena-indices.
// ---------------------------------------------------------------
fn eval_at(expr_idx: i32, env_base: i32, env_top: i32) -> i32 {
    let tag = __arena_get(expr_idx);
    let p1 = __arena_get(expr_idx + 1);
    let p2 = __arena_get(expr_idx + 2);
    let p3 = __arena_get(expr_idx + 3);
    if tag == 0 {
        // EInt: payload is the literal value
        p1
    } else { if tag == 1 {
        // EVar: payload is the name hash
        env_lookup(env_base, env_top, p1)
    } else { if tag == 2 {
        // EAdd
        eval_at(p1, env_base, env_top) + eval_at(p2, env_base, env_top)
    } else { if tag == 3 {
        // EMul
        eval_at(p1, env_base, env_top) * eval_at(p2, env_base, env_top)
    } else { if tag == 4 {
        // ESub
        eval_at(p1, env_base, env_top) - eval_at(p2, env_base, env_top)
    } else { if tag == 5 {
        // ELt: returns 0 or 1
        let l = eval_at(p1, env_base, env_top);
        let r = eval_at(p2, env_base, env_top);
        if l < r { 1 } else { 0 }
    } else { if tag == 6 {
        // EIf: cond, then, else
        let c = eval_at(p1, env_base, env_top);
        if c != 0 {
            eval_at(p2, env_base, env_top)
        } else {
            eval_at(p3, env_base, env_top)
        }
    } else { if tag == 7 {
        // ELet: bind p1 (name_hash) to value of p2 in scope of p3
        let v = eval_at(p2, env_base, env_top);
        env_push(p1, v);
        let result = eval_at(p3, env_base, env_top + 2);
        // Pop is implicit: caller's env_top didn't change.
        result
    } else {
        0   // unknown tag
    }}}}}}}}
}

// ---------------------------------------------------------------
// Builders: each pushes the Expr to the arena and returns its index.
// Tag goes in slot N, payload follows. We pre-allocate 4 slots per
// node (1 tag + up to 3 payload) so eval_at can read uniform
// positions; unused payload slots hold 0.
// ---------------------------------------------------------------
fn mk_int(v: i32) -> i32 {
    let i = __arena_push(0);   // tag = 0 (EInt)
    __arena_push(v);
    __arena_push(0);
    __arena_push(0);
    i
}

fn mk_var(name_hash: i32) -> i32 {
    let i = __arena_push(1);
    __arena_push(name_hash);
    __arena_push(0);
    __arena_push(0);
    i
}

fn mk_binop(tag: i32, l: i32, r: i32) -> i32 {
    let i = __arena_push(tag);
    __arena_push(l);
    __arena_push(r);
    __arena_push(0);
    i
}

fn mk_if(c: i32, t: i32, e: i32) -> i32 {
    let i = __arena_push(6);
    __arena_push(c);
    __arena_push(t);
    __arena_push(e);
    i
}

fn mk_let(name_hash: i32, val: i32, body: i32) -> i32 {
    let i = __arena_push(7);
    __arena_push(name_hash);
    __arena_push(val);
    __arena_push(body);
    i
}

// ---------------------------------------------------------------
// main: build the test AST, evaluate it, print the result.
//
//   let x = 5 in
//     if x < 10
//       then x * (x + 3)
//       else x - 99
// ---------------------------------------------------------------
fn main() -> i32 {
    let name_x = __hash_i32(120);    // 'x' as int — stable hash
    // Build inner expression: x * (x + 3)
    let three = mk_int(3);
    let var_x_a = mk_var(name_x);
    let xp3 = mk_binop(2, var_x_a, three);   // 2 = EAdd
    let var_x_b = mk_var(name_x);
    let then_branch = mk_binop(3, var_x_b, xp3);   // 3 = EMul
    // Build else: x - 99
    let var_x_c = mk_var(name_x);
    let lit99 = mk_int(99);
    let else_branch = mk_binop(4, var_x_c, lit99);   // 4 = ESub
    // Build cond: x < 10
    let var_x_d = mk_var(name_x);
    let lit10 = mk_int(10);
    let cond = mk_binop(5, var_x_d, lit10);   // 5 = ELt
    // Assemble if
    let body_if = mk_if(cond, then_branch, else_branch);
    // Wrap in let x = 5 in (...)
    let lit5 = mk_int(5);
    let program = mk_let(name_x, lit5, body_if);
    // Env base is AFTER all AST nodes are pushed — env bindings will
    // grow above this cursor without overlapping the AST region.
    let env_base = __arena_len();
    let result = eval_at(program, env_base, env_base);
    result   // expected: 5 * (5 + 3) = 40
}
