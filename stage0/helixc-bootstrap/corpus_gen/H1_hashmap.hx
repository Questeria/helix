// H-1 corpus (charter §1.6): i32->i32 open-addressing HashMap -- new /
// insert / get / contains, INCLUDING a forced hash collision resolved by
// linear probing. Inlined from stdlib/collections.hx (standalone corpus).
//
// cap=8 -> mask=7, bucket = key & 7. COLLISION SET: keys 3, 11, 19 all hash
// to bucket 3 (3&7 = 11&7 = 19&7 = 3); inserting them forces the linear
// probe to place them in buckets 3,4,5. We then read all three back by key
// (probe must walk the chain to find each), plus a non-colliding key, plus
// an overwrite, plus contains() on present + absent keys.
fn hm_missing() -> i32 { 0 - 1 }
fn hm_new(cap: i32) -> i32 {
    let m = __arena_push(0);
    __arena_push(0);
    __arena_push(0);
    let slots = __arena_len();
    let mut i = 0;
    while i < cap * 3 { __arena_push(0); i = i + 1; };
    __arena_set(m + 1, cap);
    __arena_set(m + 2, slots);
    m
}
fn hm_insert(m: i32, k: i32, v: i32) -> i32 {
    let cap = __arena_get(m + 1);
    let slots = __arena_get(m + 2);
    let mask = cap - 1;
    let mut b = k & mask;
    let mut done = 0;
    while done == 0 {
        let off = slots + b * 3;
        let state = __arena_get(off);
        if state == 0 {
            __arena_set(off, 1);
            __arena_set(off + 1, k);
            __arena_set(off + 2, v);
            __arena_set(m, __arena_get(m) + 1);
            done = 1;
        } else {
            if __arena_get(off + 1) == k {
                __arena_set(off + 2, v);
                done = 1;
            } else {
                b = (b + 1) & mask;
            }
        }
    };
    0
}
fn hm_get(m: i32, k: i32) -> i32 {
    let cap = __arena_get(m + 1);
    let slots = __arena_get(m + 2);
    let mask = cap - 1;
    let mut b = k & mask;
    let mut res = hm_missing();
    let mut done = 0;
    while done == 0 {
        let off = slots + b * 3;
        let state = __arena_get(off);
        if state == 0 {
            done = 1;
        } else {
            if __arena_get(off + 1) == k {
                res = __arena_get(off + 2);
                done = 1;
            } else {
                b = (b + 1) & mask;
            }
        }
    };
    res
}
fn hm_contains(m: i32, k: i32) -> i32 {
    let cap = __arena_get(m + 1);
    let slots = __arena_get(m + 2);
    let mask = cap - 1;
    let mut b = k & mask;
    let mut res = 0;
    let mut done = 0;
    while done == 0 {
        let off = slots + b * 3;
        let state = __arena_get(off);
        if state == 0 {
            done = 1;
        } else {
            if __arena_get(off + 1) == k {
                res = 1;
                done = 1;
            } else {
                b = (b + 1) & mask;
            }
        }
    };
    res
}
fn hm_len(m: i32) -> i32 { __arena_get(m) }

fn main() -> i32 {
    let m = hm_new(8);
    // COLLISION: 3, 11, 19 all -> bucket 3. probing places them at 3,4,5.
    hm_insert(m, 3, 10);
    hm_insert(m, 11, 20);
    hm_insert(m, 19, 5);
    // a non-colliding key (6 -> bucket 6) for good measure.
    hm_insert(m, 6, 100);
    // overwrite an existing key (must NOT add a new slot, must replace val).
    hm_insert(m, 6, 7);
    // four distinct keys -> count must be 4 (the overwrite did not grow it).
    if hm_len(m) != 4 { return 80; };
    // read every collided key back by key (probe walks the chain):
    if hm_get(m, 3)  != 10 { return 81; };
    if hm_get(m, 11) != 20 { return 82; };
    if hm_get(m, 19) != 5  { return 83; };
    if hm_get(m, 6)  != 7  { return 84; };   // sees the overwritten value
    // a key that was never inserted must miss.
    if hm_get(m, 99) != hm_missing() { return 85; };
    // contains() on present (incl. a probed/collided key) and absent keys.
    if hm_contains(m, 19) != 1 { return 86; };   // present, found via probe
    if hm_contains(m, 99) != 0 { return 87; };   // absent
    if hm_contains(m, 27) != 0 { return 88; };   // 27&7=3 (collides) but absent
    // value sum of the four live keys: 10 + 20 + 5 + 7 = 42.
    hm_get(m, 3) + hm_get(m, 11) + hm_get(m, 19) + hm_get(m, 6)
}
