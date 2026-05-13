// DPLL SAT solver — Davis-Putnam-Logemann-Loveland.
//
// We solve a Boolean satisfiability problem in CNF (conjunctive
// normal form): a conjunction of clauses, each clause a disjunction
// of literals. A literal is a variable or its negation; we encode
// literal as a signed i32 where the absolute value is the variable
// index (1..N) and the sign carries polarity:
//
//   positive  =  variable is true
//   negative  =  variable is false
//
// A formula is satisfiable iff there exists a variable assignment
// that makes every clause have at least one true literal.
//
// DPLL is recursive backtracking with two strong heuristics:
//   1. Unit propagation: any clause with exactly one unassigned
//      literal forces that literal's value (deterministic).
//   2. Pure literal elimination: any variable that only appears
//      with one polarity can be assigned that polarity for free.
//
// We implement the recursion + unit propagation; pure literal is
// omitted for brevity (DPLL still works without it on small inputs).
//
// Demonstration formula (3-SAT, 4 vars, 7 clauses):
//   (x1 OR x2 OR -x3) AND
//   (-x1 OR x2 OR x4) AND
//   (-x1 OR -x2 OR x3) AND
//   (x1 OR -x2 OR -x4) AND
//   (-x2 OR x3 OR x4) AND
//   (x2 OR -x3 OR -x4) AND
//   (x1 OR x3 OR x4)
//
// One satisfying assignment: x1=T, x2=T, x3=T, x4=T. Sanity check:
//   c1: x1=T → T
//   c2: x2=T → T
//   c3: x3=T → T
//   c4: x1=T → T
//   c5: x3=T → T
//   c6: x2=T → T
//   c7: x1=T → T   ✓ all clauses satisfied.
//
// The solver returns 1 if satisfiable, 0 if unsatisfiable. Expected: 1.
//
// Exercises: arena-stored CNF, recursion with backtracking, pattern
// matching on assignment state, unit propagation loop.

// ---------------------------------------------------------------
// Arena layout:
//
//   slot range          purpose
//   ------------------  ----------------------------------------
//   clause_base..       packed clauses as run-length: [len, lit, lit, ..., len, lit, ...]
//   assign_base..       variable assignments: i-th slot holds the value
//                       for variable i+1 (we use 1-indexed vars).
//                       0 = unassigned, 1 = true, -1 = false.
//
// We pass clause_base, num_clauses, num_vars, assign_base around
// rather than using global state.
// ---------------------------------------------------------------

// Get the assignment state of variable `var_idx` (1-indexed).
fn get_assign(assign_base: i32, var_idx: i32) -> i32 {
    __arena_get(assign_base + var_idx - 1)
}

fn set_assign(assign_base: i32, var_idx: i32, val: i32) -> i32 {
    __arena_set(assign_base + var_idx - 1, val);
    0
}

// Evaluate a single literal under the current assignment.
// Returns: 1 if true, -1 if false (definitely violated), 0 if unassigned.
fn eval_lit(lit: i32, assign_base: i32) -> i32 {
    let var_idx = if lit < 0 { 0 - lit } else { lit };
    let a = get_assign(assign_base, var_idx);
    if a == 0 { 0 }
    else { if lit > 0 {
        a   // positive lit: same as assignment
    } else {
        0 - a   // negative lit: flipped
    }}
}

// Evaluate a clause: walks `count` literals starting at `clause_lit_base`.
// Returns: 1 if satisfied, -1 if all literals false (clause violated),
// 0 if some literals still unassigned (clause undetermined).
// Also returns count of unassigned via side-effect: writes to slot
// `unassigned_count_slot`. We return the LAST unassigned literal
// in slot `last_unassigned_slot` for unit-propagation use.
fn eval_clause(clause_lit_base: i32, count: i32, assign_base: i32,
               unassigned_count_slot: i32, last_unassigned_slot: i32) -> i32 {
    let mut i: i32 = 0;
    let mut any_true: i32 = 0;
    let mut unassigned_n: i32 = 0;
    let mut last_un: i32 = 0;
    while i < count {
        let lit = __arena_get(clause_lit_base + i);
        let v = eval_lit(lit, assign_base);
        if v == 1 { any_true = 1; };
        if v == 0 {
            unassigned_n = unassigned_n + 1;
            last_un = lit;
        };
        i = i + 1;
    }
    __arena_set(unassigned_count_slot, unassigned_n);
    __arena_set(last_unassigned_slot, last_un);
    if any_true == 1 { 1 }
    else { if unassigned_n == 0 { 0 - 1 } else { 0 }}
}

// Walk the whole formula. Returns one of:
//   1  -> all clauses satisfied
//   -1 -> at least one clause violated (formula false)
//   0  -> formula undetermined
// Also: detect a unit clause (exactly one unassigned literal in an
// otherwise unassigned clause) and write that literal to
// unit_lit_slot, or 0 if no unit clause found.
fn eval_formula(clause_base: i32, num_clauses: i32, assign_base: i32,
                unit_lit_slot: i32) -> i32 {
    // Reusable scratch slots for eval_clause's two output values.
    let scratch_unc = __arena_push(0);
    let scratch_lu = __arena_push(0);
    __arena_set(unit_lit_slot, 0);
    let mut idx = clause_base;
    let mut clauses_left = num_clauses;
    let mut all_sat = 1;
    let mut violated = 0;
    let mut unit_found = 0;
    while clauses_left > 0 {
        if violated == 0 {
            let len = __arena_get(idx);
            let lits_start = idx + 1;
            let cstatus = eval_clause(lits_start, len, assign_base, scratch_unc, scratch_lu);
            let unc = __arena_get(scratch_unc);
            let lu = __arena_get(scratch_lu);
            if cstatus == 0 - 1 {
                violated = 1;
                all_sat = 0;
            } else { if cstatus == 0 {
                all_sat = 0;
                if unc == 1 {
                    if unit_found == 0 {
                        __arena_set(unit_lit_slot, lu);
                        unit_found = 1;
                    };
                };
            } else {} };
            idx = idx + 1 + len;
            clauses_left = clauses_left - 1;
        } else {
            clauses_left = 0;
        };
    }
    if violated == 1 { 0 - 1 }
    else { if all_sat == 1 { 1 } else { 0 }}
}

// Find the next unassigned variable (1-indexed). Returns 0 if all
// variables are assigned.
fn pick_branch(assign_base: i32, num_vars: i32) -> i32 {
    let mut v: i32 = 1;
    let mut chosen: i32 = 0;
    while v <= num_vars {
        if chosen == 0 {
            if get_assign(assign_base, v) == 0 {
                chosen = v;
            };
        };
        v = v + 1;
    }
    chosen
}

// Recursive DPLL solver. Returns 1 if SAT, 0 if UNSAT under current
// assignment. Mutates assign_base in place (and undoes on backtrack).
fn dpll(clause_base: i32, num_clauses: i32, num_vars: i32,
        assign_base: i32) -> i32 {
    // Unit-propagation loop: keep applying any unit clauses we find.
    let unit_slot = __arena_push(0);
    let mut keep_going: i32 = 1;
    let mut bail: i32 = 0;
    let mut sat: i32 = 0;
    while keep_going == 1 {
        let st = eval_formula(clause_base, num_clauses, assign_base, unit_slot);
        if st == 0 - 1 {
            bail = 1;
            keep_going = 0;
        } else { if st == 1 {
            sat = 1;
            keep_going = 0;
        } else {
            let unit = __arena_get(unit_slot);
            if unit == 0 {
                keep_going = 0;   // exit propagation, time to branch
            } else {
                let v = if unit < 0 { 0 - unit } else { unit };
                let val = if unit > 0 { 1 } else { 0 - 1 };
                set_assign(assign_base, v, val);
            };
        }};
    }
    if bail == 1 { 0 }
    else { if sat == 1 { 1 } else {
        // Branch: pick an unassigned variable, try true then false.
        let v = pick_branch(assign_base, num_vars);
        if v == 0 {
            // All assigned but formula not yet SAT — shouldn't happen
            // if eval_formula is correct, but safe-default.
            0
        } else {
            // Snapshot assignments for backtrack: save the current
            // values so we can restore. We just save into a fresh
            // arena region.
            let snap_base = __arena_len();
            let mut i: i32 = 1;
            while i <= num_vars {
                __arena_push(get_assign(assign_base, i));
                i = i + 1;
            }
            // Try v = true.
            set_assign(assign_base, v, 1);
            let r1 = dpll(clause_base, num_clauses, num_vars, assign_base);
            if r1 == 1 { 1 } else {
                // Restore from snapshot.
                let mut j: i32 = 1;
                while j <= num_vars {
                    set_assign(assign_base, j, __arena_get(snap_base + j - 1));
                    j = j + 1;
                }
                // Try v = false.
                set_assign(assign_base, v, 0 - 1);
                let r2 = dpll(clause_base, num_clauses, num_vars, assign_base);
                if r2 == 1 { 1 } else {
                    // Restore again before returning.
                    let mut k: i32 = 1;
                    while k <= num_vars {
                        set_assign(assign_base, k, __arena_get(snap_base + k - 1));
                        k = k + 1;
                    }
                    0
                }
            }
        }
    }}
}

// Push a clause: header is the literal count, followed by `count` literals.
fn push_clause3(a: i32, b: i32, c: i32) -> i32 {
    __arena_push(3);
    __arena_push(a);
    __arena_push(b);
    __arena_push(c);
    0
}

fn main() -> i32 {
    let num_vars: i32 = 4;
    let num_clauses: i32 = 7;
    // Initialize the assignment region: num_vars slots, all 0.
    let assign_base = __arena_len();
    let mut i: i32 = 0;
    while i < num_vars {
        __arena_push(0);
        i = i + 1;
    }
    // Push the CNF.
    let clause_base = __arena_len();
    push_clause3(1, 2, 0 - 3);          // x1 OR x2 OR -x3
    push_clause3(0 - 1, 2, 4);          // -x1 OR x2 OR x4
    push_clause3(0 - 1, 0 - 2, 3);      // -x1 OR -x2 OR x3
    push_clause3(1, 0 - 2, 0 - 4);      // x1 OR -x2 OR -x4
    push_clause3(0 - 2, 3, 4);          // -x2 OR x3 OR x4
    push_clause3(2, 0 - 3, 0 - 4);      // x2 OR -x3 OR -x4
    push_clause3(1, 3, 4);              // x1 OR x3 OR x4
    // Solve.
    dpll(clause_base, num_clauses, num_vars, assign_base)
}
