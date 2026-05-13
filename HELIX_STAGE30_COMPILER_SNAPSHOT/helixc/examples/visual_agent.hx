// helixc/examples/visual_agent.hx
//
// Visual tutorial: live agent navigating a 6x6 grid, printing the grid
// state after each step. Demonstrates the same cognitive primitives as
// tutorial_agent.hx but PRINTS each step so you can see it happen.
//
// Output format per step:
//   Step N: action=A, agent at (r,c)
//   . . . . . .
//   . A . . . .
//   . . . . . .
//   . . . . . .
//   . . . . # .
//   . . . . . G
//
// where:
//   .  empty cell
//   A  agent's current position
//   G  goal
//   #  blocked cell
//
// Run: python -m helixc.backend.x86_64 helixc/examples/visual_agent.hx _vis.bin
//      wsl -- bash -c "chmod +x /mnt/c/Projects/Kovostov-Native/_vis.bin && /mnt/c/Projects/Kovostov-Native/_vis.bin"
//
// License: Apache 2.0

@pure fn grid_size() -> i32 { 6 }
@pure fn goal_state() -> i32 { 35 }   // bottom-right of 6x6: 5*6+5 = 35

// Block cells: 14 (row 2, col 2), 28 (row 4, col 4)
@pure
fn is_blocked(s: i32) -> i32 {
    if s == 14 { 1 }
    else { if s == 28 { 1 } else { 0 } }
}

// Build a 6x6 grid world with two blocked cells. action 0=up, 1=down, 2=left, 3=right.
fn build_world() -> i32 {
    let n = grid_size();
    let wmt = wmt_new(n * n, 4);
    let mut s: i32 = 0;
    while s < n * n {
        let row = s / n;
        let col = s % n;
        let nu = if row > 0 { (row - 1) * n + col } else { s };
        let nd = if row < n - 1 { (row + 1) * n + col } else { s };
        let nl = if col > 0 { row * n + (col - 1) } else { s };
        let nr = if col < n - 1 { row * n + (col + 1) } else { s };
        // Blocked cells reflect: stepping into one keeps you in place.
        let nu2 = if is_blocked(nu) == 1 { s } else { nu };
        let nd2 = if is_blocked(nd) == 1 { s } else { nd };
        let nl2 = if is_blocked(nl) == 1 { s } else { nl };
        let nr2 = if is_blocked(nr) == 1 { s } else { nr };
        wmt_set(wmt, s, 0, nu2);
        wmt_set(wmt, s, 1, nd2);
        wmt_set(wmt, s, 2, nl2);
        wmt_set(wmt, s, 3, nr2);
        s = s + 1;
    }
    wmt
}

// Manhattan-distance score table: scores[s] = 30 - manhattan(s, goal).
fn build_score_table() -> i32 {
    let n = grid_size();
    let goal = goal_state();
    let goal_row = goal / n;
    let goal_col = goal % n;
    let scores = t1d_new(n * n);
    let mut s: i32 = 0;
    while s < n * n {
        let row = s / n;
        let col = s % n;
        let dr = if row > goal_row { row - goal_row } else { goal_row - row };
        let dc = if col > goal_col { col - goal_col } else { goal_col - col };
        // Penalize blocked cells heavily.
        let base = 30 - dr - dc;
        let pen = if is_blocked(s) == 1 { 0 - 100 } else { 0 };
        ti1d_set(scores, s, base + pen);
        s = s + 1;
    }
    scores
}

// Print the grid with the agent's current position.
fn print_grid(pos: i32) -> i32 {
    let n = grid_size();
    let goal = goal_state();
    let mut row: i32 = 0;
    while row < n {
        let mut col: i32 = 0;
        while col < n {
            let s = row * n + col;
            if s == pos { print_str("A "); }
            else { if s == goal { print_str("G "); }
            else { if is_blocked(s) == 1 { print_str("# "); }
            else { print_str(". "); }}};
            col = col + 1;
        }
        print_str("\n");
        row = row + 1;
    }
    print_str("\n");
    0
}

// Pick action by hill-climbing on score table. Returns chosen action.
fn pick_action(wmt: i32, scores: i32, cur: i32) -> i32 {
    let neighbors = t1d_new(4);
    let mut a: i32 = 0;
    while a < 4 {
        ti1d_set(neighbors, a, wmt_predict(wmt, cur, a));
        a = a + 1;
    }
    let best_state = hillclimb_step(neighbors, 4, scores);
    // Find which action yields best_state.
    let mut chosen: i32 = 0;
    let mut a2: i32 = 0;
    let mut found: i32 = 0;
    while a2 < 4 {
        if found == 0 {
            if wmt_predict(wmt, cur, a2) == best_state {
                chosen = a2;
                found = 1;
            }
        }
        a2 = a2 + 1;
    }
    chosen
}

// Print "Step N: action=A, agent at sN" header line.
fn print_step_header(step: i32, action: i32, pos: i32) -> i32 {
    print_str("Step ");
    print_int(step);
    print_str(": action=");
    print_int(action);
    print_str(", agent_state=");
    print_int(pos);
    print_str("\n");
    0
}

fn main() -> i32 {
    print_str("=== Helix Visual Agent Tutorial ===\n");
    print_str("6x6 grid; agent at top-left, goal at bottom-right.\n");
    print_str("Two blocked cells (#). Agent uses hillclimb on Manhattan distance.\n\n");
    let wmt = build_world();
    let scores = build_score_table();
    let goal = goal_state();
    let mut pos: i32 = 0;
    print_str("Initial state:\n");
    print_grid(pos);
    let mut step: i32 = 1;
    let mut keep: i32 = 1;
    while keep == 1 {
        if pos == goal { keep = 0; }
        else { if step > 20 {
            print_str("(stopping after 20 steps)\n");
            keep = 0;
        }
        else {
            let action = pick_action(wmt, scores, pos);
            let next_state = wmt_predict(wmt, pos, action);
            pos = next_state;
            print_step_header(step, action, pos);
            print_grid(pos);
            step = step + 1;
        }};
    }
    if pos == goal {
        print_str("=== GOAL REACHED ===\n");
        42
    } else {
        print_str("=== STUCK (max steps) ===\n");
        99
    }
}
