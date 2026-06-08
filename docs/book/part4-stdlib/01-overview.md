# Standard library overview

*What this chapter covers: what the Helix "standard library" actually is, how it is organized
around a single arena plus a handful of compiler builtins, a one-line tour of all 21 modules
under [`helixc/stdlib/`](../../../helixc/stdlib/), the real I/O and arena builtins
(`print_str`, `print_int`, `read_file_to_arena`, `write_file_to_arena`), and how a program
actually uses stdlib code. Everything here is grounded in
[`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md), the as-built language spec, and the
module files themselves. The two chapters after this one go deep on math/activations and
tensors/collections/I/O.*

---

## Two things called "the standard library"

When you say "the Helix standard library," you can mean one of two distinct things, and this part
of the book is careful to keep them apart.

1. **The builtins.** A small set of operations that the compiler `kovc` lowers *directly* to
   machine code (x86-64) or, in a `@kernel`, to PTX. These are not written in Helix; they are
   primitives the compiler knows by name — arena access, file and console I/O, float math, the
   GPU intrinsics, `grad`. [`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) opens by
   making exactly this point: *"Helix has no separate stdlib library — the standard library is the
   set of compiler builtins that `kovc` lowers directly to machine code … The runtime is a single
   mmap'd arena (one i32 slot per element) plus syscalls."*

2. **The `.hx` modules under [`helixc/stdlib/`](../../../helixc/stdlib/).** A set of 21 ordinary
   Helix source files — `vec.hx`, `option.hx`, `tensor.hx`, and so on — written *in Helix on top
   of those builtins*. They are not compiler magic; they are library code you (or the compiler's
   own dogfood programs) can read, copy, and call. The compiler itself is written in this style:
   growable buffers and AST nodes are arena-backed exactly the way `vec.hx` is.

So the honest one-sentence summary is: **the builtins are the irreducible runtime; the 21
`stdlib/` modules are a convenience library written in Helix over that runtime.** This chapter
covers the shape of both; the next two chapters quote real signatures and walk specific modules.

> **For AI agents:** do not treat `helixc/stdlib/*.hx` as a linked library with a package
> resolver. There is **no `use`/`import`/module-loader path** that pulls a stdlib module into a
> program at build time (no example under [`helixc/examples/`](../../../helixc/examples/) contains
> a `use` or `import` line). The two real ways stdlib code reaches a program are (a) the source is
> **parsed alongside** your program — the dogfood demo
> [`dogfood_23_property_proofs.hx`](../../../helixc/examples/dogfood_23_property_proofs.hx) states
> in its header that "safety.hx loads cleanly **with** the stdlib" — and (b) the relevant
> functions are **copied inline** into the program. See *"How a program uses stdlib code,"* below.

---

## The arena: one runtime heap, two slot conventions

Almost every data structure in Helix — every `Vec`, `String`, tensor, hashmap, AST node, and
working-memory cell — lives in **the arena**: a single contiguous region of `i32` slots,
BSS-zeroed at load and grown by pushing. There is no general `malloc`/`free`; you append to the
arena and address everything by integer slot index.
[`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) §(a) describes it as *"the heap: a
contiguous region of i32 slots, BSS-zeroed at load, grown by `__arena_push`."*

The four core arena builtins are `[corpus-proven]`
([`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) §(a);
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5):

| Builtin | Signature | Meaning |
|---|---|---|
| `__arena_len()` | `() -> i32` | current cursor (number of slots in use) |
| `__arena_get(i)` | `(i: i32) -> i32` | read slot `i` |
| `__arena_set(i, v)` | `(i: i32, v: i32) -> i32` | write slot `i` |
| `__arena_push(v)` | `(v: i32) -> i32` | append `v`, return its index |

A subtlety worth pinning early, because it trips people up: **the slot width depends on what you
store.** General data structures use **one `i32` per element** — a `Vec<i32>` slot holds one
32-bit int. But the **file I/O builtins use one *byte* per slot**: reading a file lays each byte
into its own arena slot. That asymmetry is deliberate (it makes byte indexing trivial), and it is
why the arena-backed `String` in [`helixc/stdlib/string.hx`](../../../helixc/stdlib/string.hx)
also keeps "one ASCII byte (lower 8 bits)" per slot — its header says exactly that — so a string
loaded from a file and a string built in memory share the same layout.

> **For AI agents:** the slot-width rule is **i32-per-element for in-memory structures, but
> 1-byte-per-slot for file bytes and for the arena `String`/`&str` representation.** When you read
> a file with `read_file_to_arena`, index the result as *bytes* (slot `start + k` is byte `k`),
> not as packed i32 words. Mixing the two views silently corrupts data. This is stated in
> [`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) §(b) and the `string.hx` header.

A direct consequence of the global arena is the **"build one container at a time"** convention you
will see repeated in the module headers. Because pushes always go to the single shared arena tip,
interleaving pushes from two half-built `Vec`s would mix their slots. The `vec.hx` header puts it
plainly: *"build one Vec at a time; interleaved pushes mix slots because the arena is global. For
AGI work this is fine since most list-building is sequential."* Containers are therefore passed
around as a **"carry-pair"** — a `start` index plus a `count` (or `cap`) that the caller threads
through every call — rather than as a heap object with an internal pointer.

---

## The builtins you will actually call

Beyond the four arena primitives, the builtins that matter for everyday programs are I/O. These
are real names the compiler recognizes and lowers itself — they are **not** defined in any
`stdlib/*.hx` file. Their honest status tiers come straight from
[`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) §(b) and
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5:

| Builtin | Signature | Meaning | Status |
|---|---|---|---|
| `print_str(msg)` | `(msg: &str) -> i32` | write a string literal to stdout | `[impl]` |
| `print_int(n)` | `(n: i32) -> i32` | write a decimal integer to stdout | `[impl]` (compiler builtin) |
| `read_file_to_arena(path)` | `(path: &str) -> i32` | read a file into the arena, **one byte per slot**; returns the byte count | `[proven]` (self-host driver) |
| `write_file_to_arena(path, start, count)` | `(path: &str, start: i32, count: i32) -> i32` | write `count` arena bytes (starting at slot `start`) to a file | `[proven]` (self-host driver) |

`print_int` is a genuine compiler builtin, not a library helper: `kovc` recognizes the name and
emits the integer-to-decimal conversion as inline assembly. You can see the dispatch in
[`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) — `is_print_int_name` and
`emit_print_int_body` (the codegen comment notes it "emit[s] the inline asm for `print_int(n)`,"
[`helixc/bootstrap/kovc.hx:3995`](../../../helixc/bootstrap/kovc.hx)), and it is exercised by many
real example programs (the dashboard agents, `fog_of_war.hx`, `hbs_reference_500loc.hx`, and
others list it in [Appendix E](../appendices/E-example-index.md)).

The clearest real demonstration of console + file output is the canonical "first visible output"
program, which calls `print_str` four times and then writes a file.

**Fragment** (excerpt of [`helixc/examples/hello_world.hx`](../../../helixc/examples/hello_world.hx)
— its `print_str` / file-write core; not the whole `fn main`):

```helix
fn main() -> i32 {
    print_str("Hello from Helix!\n");
    print_str("This program emits a real Linux ELF, prints to stdout via\n");
    print_str("the write(1, ...) syscall, and writes a file via\n");
    print_str("open + write + close.\n");

    let r = write_file("/tmp/helix_hello.txt", "wrote from helix\n");
    // ... returns 42 on success, 1 on failure
}
```

> **Note — `read_file_to_arena` / `write_file_to_arena` are the proven file builtins.** The spec
> lists them by those exact names as `[proven]` (one byte per slot) via the self-host driver
> ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5,
> [`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) §(b)). The `write_file(...)`
> convenience seen in the `hello_world.hx` excerpt above is that file's own spelling; for
> arena-precise control use the `*_to_arena` forms with an explicit `(start, count)`. The
> checkpoint module ([`helixc/stdlib/checkpoint.hx`](../../../helixc/stdlib/checkpoint.hx)) builds
> on the dynamic `(path_start, path_len)` variants of these I/O builtins.

The remaining builtins — f32/f64 SSE math, the GPU intrinsics emitted inside a `@kernel`, the
`grad` forward-mode derivative, and the assorted `__hash_i32` / `__strlen` helpers — are the
subject of [the math chapter](02-math-transcendentals-activations.md) and [Part VII (the GPU
codegen part)](../part7-gpu/01-ptx-backend.md). The capstone-proven ML/tensor op set in particular is documented in
[`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) §(d); for the *language*-level taste
of `grad`, see [the autodiff section of the language tour](../part3-language/05-autodiff-agi-features.md).

---

## The 21 modules at a glance

All 21 files below ship under [`helixc/stdlib/`](../../../helixc/stdlib/) at tag `v1.3-release`
(the list is a verbatim directory listing). Each one-liner is taken from the module's own header
comment. They group naturally into five areas.

### Core containers and control types

| Module | One-line summary |
|---|---|
| [`vec.hx`](../../../helixc/stdlib/vec.hx) | arena-backed `Vec<i32>` (carry-pair `start`+`count`); both unchecked legacy ops and bounds-checked `*_checked` ops with a magic/footer header |
| [`hashmap.hx`](../../../helixc/stdlib/hashmap.hx) | fixed-capacity, linear-probing `i32→i32` hash map; 3 arena slots per bucket (occupied flag, key, value) |
| [`string.hx`](../../../helixc/stdlib/string.hx) | arena-backed byte string, one ASCII byte per slot; carry-pair `start`+`len` |
| [`option.hx`](../../../helixc/stdlib/option.hx) | `Option { Some(i32), None }` sum type plus combinators (`option_unwrap_or`, `option_max`, …) |
| [`result.hx`](../../../helixc/stdlib/result.hx) | `Result { Ok(i32), Err(i32) }` for fallible ops plus combinators (`result_unwrap_or`, `result_is_ok`, …) |
| [`iterators.hx`](../../../helixc/stdlib/iterators.hx) | iterator-style ops over an arena `Vec<i32>` (`range_to_vec`, `vec_min`, `vec_count_eq/lt/gt`); no closures — ops are op-tag or scalar specialized |

### Numerics, autodiff, and ML

| Module | One-line summary |
|---|---|
| [`tensor.hx`](../../../helixc/stdlib/tensor.hx) | 1-D and 2-D f32 tensor primitives over the arena (`t1d_*`, row-major 2-D access) |
| [`autodiff.hx`](../../../helixc/stdlib/autodiff.hx) | forward-mode autodiff in dual-number style (value + derivative as paired f64s; `<op>_v` / `<op>_dx` per op) |
| [`autodiff_reverse.hx`](../../../helixc/stdlib/autodiff_reverse.hx) | reverse-mode autodiff via a 4-slot-per-op tape walked backward (O(1) backward per output — what backprop needs) |
| [`nn.hx`](../../../helixc/stdlib/nn.hx) | tiny NN primitives on the integer tensors (`dense_layer` `W@x+b`, `relu_layer`, `softmax_argmax`, `mse_loss`) |
| [`transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx) | Taylor-series `exp`/`log`/`sin`/`cos` + Newton `sqrt`, accurate only for small \|x\| (no range reduction in v0.1) |
| [`ieee754.hx`](../../../helixc/stdlib/ieee754.hx) | integer-only construction of the IEEE-754 f32 bit pattern from `(integer_part, frac_value, frac_digits)` |
| [`mnist.hx`](../../../helixc/stdlib/mnist.hx) | IDX-format (LeCun MNIST) header parser + body bounds-check over already-loaded arena bytes |
| [`checkpoint.hx`](../../../helixc/stdlib/checkpoint.hx) | save/load of training state, built on the dynamic file-I/O builtins (`*_dyn` path-as-`(start,len)`) |

### Reasoning and safety

| Module | One-line summary |
|---|---|
| [`provenance.hx`](../../../helixc/stdlib/provenance.hx) | readability + printable-evidence helpers over the `register_derivation` / `parent_*_at` provenance side-table (note: `Logic<T> = T` at runtime; the source tag is erased) |
| [`safety.hx`](../../../helixc/stdlib/safety.hx) | construction shorthands for the Tier-S/A compile-time wrapper types plus `@property` invariant functions; wrappers are identity-erased at codegen (zero runtime cost) |
| [`csv.hx`](../../../helixc/stdlib/csv.hx) | line/field iteration over an arena-loaded CSV blob via a threaded `offset` cursor (no closures) |

### The AGI substrate (Phase-4 primitives)

| Module | One-line summary |
|---|---|
| [`agi_memory.hx`](../../../helixc/stdlib/agi_memory.hx) | bounded LRU working memory; items as `(key, value, recency)` triples (default 16 slots) |
| [`agi_search.hx`](../../../helixc/stdlib/agi_search.hx) | planning search primitives — a BFS FIFO queue and a hill-climbing step over arena adjacency |
| [`agi_world.hx`](../../../helixc/stdlib/agi_world.hx) | world model predicting `next_state` from `(state, action)` — a table form and a linear `w*state + b*action + c` form |
| [`agi_match.hx`](../../../helixc/stdlib/agi_match.hx) | tree-shaped pattern/similarity primitives; trees as flat `(tag, p1, p2, p3)` arena nodes referenced by offset |

A unifying observation across all four areas: because Phase-0 Helix has **no closures / first-class
functions** (the `csv.hx` and `iterators.hx` headers both say so), higher-order behavior is encoded
either by an **op-tag / scalar argument** (iterators) or by **threading an explicit cursor**
(`offset` in `csv.hx`, the carry-pair `start`/`count` everywhere else). And because there is one
global arena, the "build one container at a time" rule applies module-wide. If you internalize the
arena and the carry-pair, every module's API reads the same way.

> **Note — capturing closures *do* exist in the language as of v1.3.** The "no closures" remarks in
> these module headers are period notes from when the modules were written. The shipped `kovc`
> does support closures (`|x| …`), including capturing ones with i32-only captures
> ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2.2, proven by the
> `V3_*` corpus). The stdlib modules simply predate that and are written in the closure-free style;
> they still compile and behave exactly as their headers describe. Closures are covered in
> [Generics, traits & closures](../part3-language/04-generics-traits-closures.md).

---

## Real signatures: `option.hx`, `result.hx`, `vec.hx`

To make the carry-pair-and-sum-type style concrete, here are real excerpts. These are
**Fragments** — they are not standalone programs (they have no `fn main`); they are quoted verbatim
from the cited module files.

`Option` and `Result` are ordinary `@pure` enums with combinators, exactly mirroring the
"`Option`/`Result` are just user enums" point made in
[the language tour](../part3-language/01-language-tour.md). Each is **i32-specialized** in this
phase (the headers explain why: generic enum payloads need type-tagged-payload codegen, a later
item).

**Fragment** (the type + two combinators from
[`helixc/stdlib/option.hx`](../../../helixc/stdlib/option.hx); not a complete program):

```helix
@pure
enum Option {
    Some(i32),
    None,
}

@pure
fn option_unwrap_or(o: Option, default_v: i32) -> i32 {
    match o {
        Option::Some(x) => x,
        Option::None => default_v,
    }
}

@pure
fn option_is_some(o: Option) -> i32 {
    match o {
        Option::Some(_) => 1,
        Option::None => 0,
    }
}
```

**Fragment** (the type + two combinators from
[`helixc/stdlib/result.hx`](../../../helixc/stdlib/result.hx); not a complete program):

```helix
@pure
enum Result {
    Ok(i32),
    Err(i32),
}

@pure
fn result_unwrap_or(r: Result, default_v: i32) -> i32 {
    match r {
        Result::Ok(x) => x,
        Result::Err(_) => default_v,
    }
}

@pure
fn result_err_code_or(r: Result, default_v: i32) -> i32 {
    match r {
        Result::Ok(_) => default_v,
        Result::Err(c) => c,
    }
}
```

`vec.hx` is the canonical carry-pair container. `vec_new()` just records the current arena tip as
the vector's `start`; `vec_push` appends to the arena and returns the new `count`; `vec_get` reads
`start + i`. The caller owns `start` and `count`.

**Fragment** (the carry-pair core from [`helixc/stdlib/vec.hx`](../../../helixc/stdlib/vec.hx);
not a complete program):

```helix
@pure
fn vec_new() -> i32 {
    __arena_len()
}

fn vec_push(start: i32, count: i32, x: i32) -> i32 {
    __arena_push(x);
    count + 1
}

@pure
fn vec_get(start: i32, i: i32) -> i32 {
    __arena_get(start + i)
}
```

Two honesty notes the `vec.hx` source makes about itself, which you should carry forward:

- **The unchecked legacy ops do no bounds-checking.** `vec_get` / `vec_set` index `start + i`
  blindly; the header marks them deprecated for new code and points at the `*_checked` variants
  (`vec_new_checked`, `vec_get_checked`, `vec_set_checked`) that validate a magic/cap header and a
  footer. The module's own header is blunt that the checked infrastructure *"defends NO running
  code today"* — no Tier-S consumer has migrated — so it is "infrastructure-only" closure, not an
  eliminated threat surface. Prefer the checked ops in new safety-critical code.
- **`vec_push` ignores `start`.** It pushes to the arena tip regardless of which vector `start`
  came from — which is exactly why the "build one Vec at a time" rule exists, and why the header
  warns you must **not** mix `vec_push` with a `vec_new_checked` vector (the push would overrun the
  footer). Reductions like `vec_sum` / `vec_product` use an i64 accumulator with INT32 saturation
  so a fold over large values saturates cleanly instead of silently wrapping.

---

## How a program uses stdlib code

Because there is no module loader (no `use`/`import` resolves a `stdlib/*.hx` file at build time),
a Helix program brings in stdlib code in one of two concrete ways:

1. **Co-parse the module source with your program.** The stdlib `.hx` source is parsed *alongside*
   the user program, so its `fn`s and `enum`s are in scope. This is how the dogfood demo
   [`dogfood_23_property_proofs.hx`](../../../helixc/examples/dogfood_23_property_proofs.hx) uses
   [`helixc/stdlib/safety.hx`](../../../helixc/stdlib/safety.hx): its header states it
   *"validates end-to-end that … safety.hx loads cleanly with the stdlib"* and that each
   `@property` fn *"typechecks + lowers + runs."* It feeds five `@property` invariants a fixed
   five-value f32 test set (`-100, -1, 0, 1, 100`) and returns `42` if every invariant holds.

2. **Inline the functions you need.** For self-contained programs (and for the gate's own
   fixtures), the handful of stdlib functions a program needs are simply copied into the file. This
   is the dominant pattern in the examples directory: the arena-backed `Vec`, `HashMap`, and
   `String` *shapes* appear inline rather than via an import.

This second pattern is also exactly how the **gate** proves these container shapes. The gate
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`, the feature corpus) compiles
and runs corpus programs that **inline** a growable `Vec<T>`, a linear-probing `HashMap`, and a
growable arena `String`, each exercised end-to-end:

- **`H1_vec`** — a generic `Vec<T>` with `new/push/get/set/len/pop` and **real growth on push**
  (cap 2 → push 1..8 forces two relocations 2→4→8), asserting `len==8`, a sum-back of 36, a
  `set(0,7)`, a `pop()==8`, ending at exit **42** ([`scripts/gate_kovc.sh:454`](../../../scripts/gate_kovc.sh)).
- **`H1_hashmap`** — an `i32→i32` open-addressing map; inserts three keys that **all collide** to
  one bucket (resolved by linear probing), overwrites, reads each back, misses on an absent key,
  ending at exit **42** ([`scripts/gate_kovc.sh:454`](../../../scripts/gate_kovc.sh)).
- **`H2_string`** — an arena-backed `String` with `str_new/str_push_byte/str_concat/str_eq`;
  builds `"Hel"+"lix"` byte-by-byte (forcing a grow 4→8), indexes bytes back out, tests equality
  four ways, ending at exit **42** ([`scripts/gate_kovc.sh:469`](../../../scripts/gate_kovc.sh)).

So the *shapes* embodied by `vec.hx`, `hashmap.hx`, and `string.hx` (growable vector, probing map,
growable string) are part of the standing compile-and-run proof — the gate prints `GATE_PASS`,
corpus 109/0, reproduced at v1.3 ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)).

> **For AI agents:** the gate's `H1_vec` / `H1_hashmap` / `H2_string` programs live under
> `stage0/helixc-bootstrap/corpus_gen/` (referenced as `$GENC/...`), **not** under
> `helixc/stdlib/`, and they **inline** the container code — the gate comments at
> [`scripts/gate_kovc.sh:443`](../../../scripts/gate_kovc.sh) and `:457` even reference a
> `stdlib/collections.hx` / `stdlib/string.hx` as the *library shape* they embed. There is no
> `collections.hx` file inside `helixc/stdlib/`; the shipped modules are the **separate**
> [`vec.hx`](../../../helixc/stdlib/vec.hx) and [`hashmap.hx`](../../../helixc/stdlib/hashmap.hx).
> What the gate proves is the **shape/idiom**, exercised end-to-end on the freshly self-hosted
> `K2.bin`. Do not assert that a given `helixc/stdlib/*.hx` file is itself compiled by the gate as
> a standalone unit; assert only what the corpus row actually runs.

---

## Honest status: what is gate-exercised vs simply committed

The standing compile-proof has a precise edge, and the book holds to it.

- **Builtins.** The arena builtins (`__arena_*`) are `[corpus-proven]`; the file builtins
  (`read_file_to_arena` / `write_file_to_arena`) are `[proven]` via the self-host driver;
  `print_str` / `print_int` are `[impl]` (`print_int`'s codegen is a real `kovc` builtin path).
  Tiers per [`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) §(a)/(b) and
  [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5.
- **Container *shapes*** (growable `Vec`, probing `HashMap`, growable `String`) are gate-exercised
  end-to-end via the `H1_vec` / `H1_hashmap` / `H2_string` corpus rows (inlined, not loaded).
- **`safety.hx`** is exercised at runtime by
  [`dogfood_23_property_proofs.hx`](../../../helixc/examples/dogfood_23_property_proofs.hx) (the
  five `@property` invariants over five inputs), and the broader Tier-S/A wrapper stack appears in
  [`dogfood_21_typed_security_stack.hx`](../../../helixc/examples/dogfood_21_typed_security_stack.hx)
  and [`dogfood_22_full_wrapper_stack.hx`](../../../helixc/examples/dogfood_22_full_wrapper_stack.hx)
  (see [Appendix E §E.5](../appendices/E-example-index.md)).
- **The tensor / autodiff / nn surface** is the load-bearing ML stdlib and is `[capstone-proven]`
  through the end-to-end transformer capstone
  ([`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) §(d)); the `dogfood_01`–`dogfood_05`
  training programs and `nn_forward.hx` in [Appendix E](../appendices/E-example-index.md) exercise
  these idioms in-tree.
- **Everything else** — the AGI substrate modules (`agi_memory`, `agi_search`, `agi_world`,
  `agi_match`), `csv.hx`, `mnist.hx`, `ieee754.hx`, `iterators.hx`, `provenance.hx`,
  `checkpoint.hx`, `transcendentals.hx` — is **real, committed Helix source** that the dogfood/agent
  demos in [Appendix E §E.5](../appendices/E-example-index.md) compose, but the *module files
  themselves* are not each independently exit-code-asserted as standalone units by the gate. Treat
  them as readable, demonstrable library code; where you need a pinned exit code, reach for a
  gate-asserted program ([Appendix E §E.2](../appendices/E-example-index.md)) or compile-and-run the
  module's caller yourself.
- **`transcendentals.hx`** in particular is Taylor-series approximation accurate only for small
  \|x\| (its header is explicit: roughly \|x\| < 1.5 for sin/cos, x near 1 for log, \|x\| < 4 for
  exp; "production code would do range reduction; v0.1 keeps the surface simple"). Do not present it
  as a full-range libm.

> **Residual:** the ML/tensor stdlib's GPU proof is **complete to PTX, not SASS** — below PTX it
> trusts NVIDIA's closed `ptxas`, the CUDA driver, the GPU hardware, and the C host launcher; the
> reference target is a single GPU (`sm_86`), kernel performance is a **fraction of cuBLAS**
> (~50–67.5% on that box), and the end-to-end capstone speedup is **7.0–8.7×** (Amdahl-bound), not
> ≥10×. Loss parity (the hard gate) holds at ~0%. Every residual is enumerated in
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R; the full GPU story is
> [Part VII](../part7-gpu/01-ptx-backend.md).

For a categorized, navigational index of every real program that *uses* these modules — including
which ones carry a gate-asserted exit code — see
[Appendix E — Example index](../appendices/E-example-index.md). For the language surface these
modules are written in (enums, `match`, `@pure`, the arena), see
[Part III — The Helix language](../part3-language/01-language-tour.md).

---

**Next:** [Math, transcendentals & activations](02-math-transcendentals-activations.md) — the f32/f64
SSE math builtins, the Taylor-series transcendentals in
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx), and the activation
functions (and their backward passes) that feed the autodiff and NN modules, each grounded against
the spec and a real demonstrating program.
