// H-1 corpus (charter §1.6): generic Vec<T> -- new / push / get / set / len /
// pop with GROWTH on push. Inlined from stdlib/collections.hx (the gate
// compiles each corpus program standalone; no external-module loader yet).
//
// Test: create with cap 2, push 1..=8 (forces TWO relocations 2->4->8 so the
// growth + copy path runs), assert len==8 and cap==8 (growth happened), sum
// all 8 back via get() (=36), then mutate via set() and shrink via pop(),
// and land on a hand-checked exit. Exercises every Vec op + growth.
struct Vec[T] { h: i32 }

fn vec_with_cap(cap0: i32) -> i32 {
    let h = __arena_push(0);
    __arena_push(0);
    __arena_push(0);
    let data = __arena_len();
    let mut i = 0;
    while i < cap0 { __arena_push(0); i = i + 1; };
    __arena_set(h + 1, cap0);
    __arena_set(h + 2, data);
    h
}
fn vec_grow(h: i32) -> i32 {
    let len = __arena_get(h);
    let cap = __arena_get(h + 1);
    let old = __arena_get(h + 2);
    let ncap = cap * 2;
    let ndata = __arena_len();
    let mut i = 0;
    while i < ncap { __arena_push(0); i = i + 1; };
    let mut j = 0;
    while j < len { __arena_set(ndata + j, __arena_get(old + j)); j = j + 1; };
    __arena_set(h + 1, ncap);
    __arena_set(h + 2, ndata);
    0
}
fn vec_push_raw(h: i32, x: i32) -> i32 {
    let len = __arena_get(h);
    let cap = __arena_get(h + 1);
    if len >= cap { vec_grow(h); };
    let data = __arena_get(h + 2);
    __arena_set(data + len, x);
    __arena_set(h, len + 1);
    0
}
impl<T> Vec<T> {
    fn len(self) -> i32 { __arena_get(self.h) }
    fn cap(self) -> i32 { __arena_get(self.h + 1) }
    fn get(self, i: i32) -> T { __arena_get(__arena_get(self.h + 2) + i) }
    fn set(self, i: i32, x: i32) -> i32 { __arena_set(__arena_get(self.h + 2) + i, x); 0 }
    fn push(self, x: i32) -> i32 { vec_push_raw(self.h, x) }
    fn pop(self) -> T {
        let n = __arena_get(self.h) - 1;
        __arena_set(self.h, n);
        __arena_get(__arena_get(self.h + 2) + n)
    }
}

fn main() -> i32 {
    let v = Vec::<i32>{ h: vec_with_cap(2) };
    let mut i = 1;
    while i <= 8 {
        v.push(i);            // 8 pushes from cap 2 -> grows 2->4->8
        i = i + 1;
    };
    // growth happened: 8 elements held in a region that began at cap 2.
    if v.len() != 8 { return 90; };
    if v.cap() != 8 { return 91; };       // proves two relocations occurred
    // read every element back: 1+2+...+8 = 36
    let mut sum = 0;
    let mut k = 0;
    while k < v.len() { sum = sum + v.get(k); k = k + 1; };
    if sum != 36 { return 92; };
    // mutate in place: set element 0 (1 -> 7), sum is now 36-1+7 = 42
    v.set(0, 7);
    // pop the last element (8); len -> 7
    let last = v.pop();
    if last != 8 { return 93; };
    if v.len() != 7 { return 94; };
    // recompute the live sum: 7 + 2 + 3 + 4 + 5 + 6 + 7 = 34 (index 0 now 7,
    // index 7's old value 8 popped off)
    let mut sum2 = 0;
    let mut j = 0;
    while j < v.len() { sum2 = sum2 + v.get(j); j = j + 1; };
    // 34 (live sum) + 8 (popped value) = 42
    sum2 + last
}
