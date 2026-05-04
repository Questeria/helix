// hbs_sample_symbol_table.hx
//
// HBS dogfood: assoc-list symbol table using the new arena builtins.
// Per research recommendation, symbol tables are "two parallel arenas
// (names, decls)" — for HBS-scale (hundreds of symbols), linear scan
// beats hashing on real wall-clock time.
//
// In v0.1, "names" are pre-hashed i32 keys (since runtime strings are
// not yet a thing). A real self-host pass would hash byte slices.
//
// API demonstrated:
//   sym_insert(key_hash, decl_idx) -> ()
//   sym_lookup(key_hash) -> i32  (returns decl_idx, or -1 if not found)
//   sym_count() -> i32

@total
fn sym_insert(key: i32, decl: i32) -> i32 {
    // Push a (key, decl) pair into the arena. Returns the slot index of
    // the key (decl is at key_idx + 1).
    let k = __arena_push(key);
    __arena_push(decl);
    k
}

@total
fn sym_lookup(arena_start: i32, sym_count: i32, key: i32) -> i32 {
    // Walk N pairs starting at arena_start. Each pair occupies 2 slots:
    //   slot[i*2]   = key
    //   slot[i*2+1] = decl
    // Returns the decl for the FIRST matching key (newest first if the
    // caller pushes in reverse order), or 0 - 1 (= -1 modulo i32) if
    // not found.
    let mut i: i32 = 0;
    let mut result: i32 = 0 - 1;
    while i < sym_count {
        let pair_idx = arena_start + i * 2;
        let k = __arena_get(pair_idx);
        if k == key {
            result = __arena_get(pair_idx + 1);
            // Continue scanning so the LAST inserted with this key wins
            // (lexical scoping convention). For a from-newest-first
            // iteration, walk from sym_count-1 down to 0 instead.
        }
        i = i + 1;
    }
    result
}

fn main() -> i32 {
    // Insert three (key, decl) pairs into the arena.
    let s0 = sym_insert(100, 10);   // start slot (always 0 since arena was empty)
    sym_insert(200, 20);
    sym_insert(300, 42);
    // Look up key=300 — should return 42.
    let count: i32 = 3;
    sym_lookup(s0, count, 300)
}
