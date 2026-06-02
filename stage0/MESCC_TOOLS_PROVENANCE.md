# stage0/mescc-tools — provenance

The mescc-tools assembler/linker family is vendored as **auxiliary verification
tools** for the M2-Planet rung (rung 7). They are *not* a new rung of the trust
ladder — they exist to give M2-Planet a self-host path it cannot get through our
own stage0-posix-derived M0/hex2.

**Why these three tools (the H6 motivation):** our `M0` (the reduced stage0-posix
amd64 assembler) corrupts one instruction when it assembles M2-Planet's own
~2.2 MB self-output, so M2 cannot self-host *through our* positional M0/hex2 (see
`M2-Planet/PROVENANCE.md`, "Self-host fixpoint"). Upstream M2-Planet self-hosts
via a **different** toolchain family — the mescc-tools `M1` macro-assembler +
`blood-elf` (debug-symbol footer) + the **flag-driven** `hex2` linker. That
family is what M2-Planet's own `test/run-pass.sh` invokes. Vendoring its source
lets the next build step assemble M2's large self-output with the assembler that
does *not* have M0's large-input bug, and link it with a linker that takes the
`--base-address` / `--architecture` / `--little-endian` flags our positional
`hex2` (rung 3) lacks.

Per project policy (`stage0/README.md`; user decision 2026-05-30), vendored trees
are **pulled from canonical, community-audited sources at pinned commits** and
**built only by prior rungs** — no pre-built binary is ever trusted. These GPL-3.0
trees are kept **statically separable** from the Apache-2.0 helix top. Building
them is a *later* step; this manifest covers the **source vendoring only**.

## Pinned commit

### oriansj/mescc-tools @ `5adfbf3364261a77109878a56b100aeeb6ef9ac4` (tag `Release_1.7.0`, GPL-3.0-or-later)

**Rationale — why this commit is compatible with M2-Planet `761c2af5` / M2libc `b8bb2a01`:**

1. **Flag syntax matches M2-Planet's own test recipe.** M2-Planet `761c2af5`'s
   `test/run-pass.sh` + `test/env.inc.sh` drive these tools as:
   - `blood-elf --64 -f <unit>.M1 --little-endian --entry _start -o <footer>.M1`
   - `M1 -f M2libc/amd64/amd64_defs.M1 -f M2libc/amd64/libc-core.M1 -f <unit>.M1 -f <footer>.M1 --little-endian --architecture amd64 -o <unit>.hex2`
   - `hex2 -f M2libc/amd64/ELF-amd64-debug.hex2 -f <unit>.hex2 --little-endian --architecture amd64 --base-address 0x00600000 -o <binary>`

   Every one of those flags is present in the `main()` argument parsers at this
   exact tag (verified against the raw sources):
   - `M1-macro.c` matches `-A`/`--architecture`, `--little-endian`, `--big-endian`, `-f`/`--file`, `-o`/`--output`.
   - `hex2.c` matches `-B`/`--base-address`, `-A`/`--architecture`, `--little-endian`, `--big-endian`, `-f`/`--file`, `-o`/`--output`.
   - `blood-elf.c` matches `--64`, `-f`/`--file`, `--little-endian`, `--big-endian`, `--entry`, `-o`/`--output`.

2. **Same era / closest tagged release.** `Release_1.7.0` is the **latest tagged
   release** of mescc-tools (tagger date 2025-07-26). M2-Planet `761c2af5` pins
   its `M2libc` submodule at `b8bb2a01`, which by `git compare` is **326 commits
   ahead** of `5a7c12a` — the M2libc commit that mescc-tools master / 1.6.0 / 1.7.0
   themselves submodule. In other words M2-Planet `761c2af5` is from at-or-after
   the 1.7.0 era, so 1.7.0 is the nearest community-audited *tagged* mescc-tools.
   A tag is preferred over a moving `master` for reproducibility, consistent with
   how the other rungs pin (`stage0-posix-amd64 @ 15535f88`).

3. **File layout matches what M2-Planet expects.** The tool set at this tag —
   `M1-macro.c`, `blood-elf.c`, and the `hex2.{c,h}` / `hex2_globals.h` /
   `hex2_linker.c` / `hex2_word.c` family — is exactly the layout M2-Planet's
   recipe calls (`M1`, `blood-elf`, `hex2`). No API drift.

> **Naming note.** mescc-tools names the macro-assembler source `M1-macro.c`
> (there is **no** file literally named `M1.c` at *any* mescc-tools commit,
> including current master). This manifest vendors upstream `M1-macro.c` to the
> local path `M1/M1.c` — content byte-identical, filename localized to the
> task's requested path. Likewise the flag-driven linker's `main()` lives in
> upstream `hex2.c`, with `hex2_linker.c` holding the link/label core; the whole
> hex2 family is vendored together under `hex2-linker/` so the linker binary
> (`hex2`) can be built standalone.

### oriansj/M2libc @ `b8bb2a0159a7376716a396ec6f6bc29dd27857b5` (GPL-3.0-or-later / public-domain-equivalent per repo)

The same M2libc pin M2-Planet `761c2af5` already submodules (matches
`M2-Planet/PROVENANCE.md`). Used here for (a) the shared
`bootstrappable.{c,h}` that all three tools `#include`, and (b) the debug ELF
header `ELF-amd64-debug.hex2` the link step consumes.

## Vendored files

GPL-3.0-or-later headers are present verbatim at the top of each `.c`/`.h`
(mescc-tools / M2-Planet copyright, Jeremiah Orians et al.). Each tool directory
carries its **full compile closure** (its own local `M2libc/bootstrappable.{c,h}`,
and `stringify.c` where the tool needs it) so it can be built standalone.

### `M1/` — the macro-assembler (upstream `M1-macro.c`)

| File | bytes | SHA-256 | upstream |
|---|---|---|---|
| `M1/M1.c` | 19912 | `d0528b3fb6f54961c1b455f2a72e06f213ab9b8d8bd04b92b31ae753c30b8700` | mescc-tools `5adfbf33` `M1-macro.c` |
| `M1/stringify.c` | 2517 | `2cc09fd299c2bdff1a5e32d1baf84bd92a01f4447c475a671a80fa6549bf1302` | mescc-tools `5adfbf33` `stringify.c` |
| `M1/M2libc/bootstrappable.c` | 3815 | `efe16699e165d1ebad2d8f942383aea11c23dcc783f291c3dbf7b6d199fdc831` | M2libc `b8bb2a01` `bootstrappable.c` |
| `M1/M2libc/bootstrappable.h` | 1096 | `81b0e0e9047a90ba766786367060f033b4ca2b8448c8f8e12820f999c6d1ec77` | M2libc `b8bb2a01` `bootstrappable.h` |

### `blood-elf/` — the debug-symbol footer generator

| File | bytes | SHA-256 | upstream |
|---|---|---|---|
| `blood-elf/blood-elf.c` | 13904 | `f052b106acc267990b8b4de14a5684d3d96eb468a2eb71bbbea96d7ec98adfaf` | mescc-tools `5adfbf33` `blood-elf.c` |
| `blood-elf/stringify.c` | 2517 | `2cc09fd299c2bdff1a5e32d1baf84bd92a01f4447c475a671a80fa6549bf1302` | mescc-tools `5adfbf33` `stringify.c` |
| `blood-elf/M2libc/bootstrappable.c` | 3815 | `efe16699e165d1ebad2d8f942383aea11c23dcc783f291c3dbf7b6d199fdc831` | M2libc `b8bb2a01` `bootstrappable.c` |
| `blood-elf/M2libc/bootstrappable.h` | 1096 | `81b0e0e9047a90ba766786367060f033b4ca2b8448c8f8e12820f999c6d1ec77` | M2libc `b8bb2a01` `bootstrappable.h` |

### `hex2-linker/` — the flag-driven linker (upstream `hex2.c` + family)

| File | bytes | SHA-256 | upstream |
|---|---|---|---|
| `hex2-linker/hex2_linker.c` | 13410 | `91d0b45264fde254758b278e437317b0ddcd0899c1e612b2cc7775c403bde56d` | mescc-tools `5adfbf33` `hex2_linker.c` |
| `hex2-linker/hex2.c` | 6489 | `6d010a9e518b50525b9aed9dbae6a55c8e5fc4771ba0125551da962f92d61f62` | mescc-tools `5adfbf33` `hex2.c` |
| `hex2-linker/hex2.h` | 1494 | `de9ba5397afa35d73452c7ac5e48da700d2612ec34b548c629837255dcfa687a` | mescc-tools `5adfbf33` `hex2.h` |
| `hex2-linker/hex2_globals.h` | 1684 | `6826a9adb174fa73ad57d5b83885b989adfc4bca95a9ae064d0685eb5879a5f2` | mescc-tools `5adfbf33` `hex2_globals.h` |
| `hex2-linker/hex2_word.c` | 9160 | `9843574faa7597d21bca30df318e09fed0e771538724ed086f6744142969fc46` | mescc-tools `5adfbf33` `hex2_word.c` |
| `hex2-linker/M2libc/bootstrappable.c` | 3815 | `efe16699e165d1ebad2d8f942383aea11c23dcc783f291c3dbf7b6d199fdc831` | M2libc `b8bb2a01` `bootstrappable.c` |
| `hex2-linker/M2libc/bootstrappable.h` | 1096 | `81b0e0e9047a90ba766786367060f033b4ca2b8448c8f8e12820f999c6d1ec77` | M2libc `b8bb2a01` `bootstrappable.h` |

### into the existing M2-Planet M2libc tree — the debug ELF header

| File | bytes | SHA-256 | upstream |
|---|---|---|---|
| `M2-Planet/M2libc/amd64/ELF-amd64-debug.hex2` | 3284 | `ea6b8b26b42ca548f55cce99efcbe8fe933259b6f2bdfb9b07d3323080a9ebd9` | M2libc `b8bb2a01` `amd64/ELF-amd64-debug.hex2` |

> SHA cross-check: `bootstrappable.c` here (`efe16699…`) is **byte-identical** to
> the copy already recorded in `M2-Planet/PROVENANCE.md` (also `efe16699…`),
> confirming the M2libc `b8bb2a01` pin is consistent across both trees.

## Why each tool is vendored

- **`M1` (M1-macro.c)** — the macro assembler that does **not** have our M0's
  large-input bug. M0 corrupts one instruction in M2's 2.2 MB self-output; M1 is
  the assembler M2-Planet self-hosts through, so it is the one that can correctly
  assemble M2's own emitted `.M1`.
- **`blood-elf`** — generates the ELF `.symtab` / `.strtab` **debug-symbol
  footer** (`--debug`-style `_start`-anchored symbol table). M2-Planet's recipe
  runs it before `M1` so the resulting binary carries debug symbols; it is a
  required stage of the upstream self-host pipeline.
- **`hex2` (flag-driven linker: hex2.c + hex2_linker.c + hex2_word.c)** — the
  linker our **positional** `hex2` (rung 3, stage0-posix family) is *not*: it
  takes `--base-address`, `--architecture`, `--little-endian`, and multiple
  `-f` inputs, which is exactly the interface M2-Planet's recipe relies on to
  place the load address and stitch the ELF header + body + footer.

## #include dependency closure (verified)

Across all three tools the **only** non-system (local, double-quote) include is
`M2libc/bootstrappable.h`; the hex2 family additionally chains
`hex2_globals.h → hex2.h`. Full map:

```
M1.c          -> <stdlib.h> <stdio.h> <string.h>            + "M2libc/bootstrappable.h"
              -> calls stringify(), LittleEndian()  ==> NEEDS stringify.c (vendored)
blood-elf.c   -> <stdio.h> <stdlib.h> <string.h> <unistd.h> <sys/stat.h> + "M2libc/bootstrappable.h"
              -> calls stringify(), LittleEndian()  ==> NEEDS stringify.c (vendored)
hex2.c        -> "hex2_globals.h"                  (main() + flag parsing live here)
hex2_linker.c -> "hex2_globals.h"                  (consume/storeLabel/link core)
hex2_word.c   -> "hex2_globals.h"                  (word/shift-register output mode)
hex2_globals.h-> "hex2.h"
hex2.h        -> <stdio.h> <stdlib.h> <string.h> <unistd.h> <sys/stat.h> + "M2libc/bootstrappable.h"
stringify.c   -> <stdio.h>   (self-contained: defines stringify + LittleEndian; no header)
bootstrappable.h -> #ifdef __M2__ : #include <bootstrappable.c> ; #else : prototypes only
bootstrappable.c -> <stdio.h> <stdlib.h>   (require/match/in_set/strtoint/int2str)
```

`get_machine.c` and `catm.c` exist upstream but are **not** included by any of
these three tools and are deliberately **not** vendored.

## Integration risk for the NEXT (build) step — read before compiling

These are source-only notes; **nothing here has been built** (no
gcc/cc_amd64/M2 was invoked, per task constraint).

1. **Build them with `M2.bin` (+ M2libc), NOT with `cc_amd64`.** All three tools
   (and the hex2 family) use `FILE*`, `fopen`, `fgetc`, `fputc`, `fputs`,
   `stderr`, `EOF` heavily for file I/O. M2-Planet's *own* `cc.c` uses exactly
   these same stdio APIs and M2 already links them via **M2libc**, so under M2
   they resolve. `cc_amd64` (rung 6, the weak C compiler) ships only the minimal
   `libc-core.M1` and is unlikely to provide `FILE`/`fopen`/`fgetc` — so the
   integration path is **M2.bin → M2libc**, almost certainly with
   `M2libc/amd64/libc-full.M1` (the full libc the self-host phases use), not the
   reduced `cc_amd64` libc.

2. **The source is explicitly M2-Planet-friendly — by design.**
   `M2libc/bootstrappable.h` has an `#ifdef __M2__` branch that, when M2 compiles
   it, `#include`s `bootstrappable.c` directly (M2's concatenation idiom) instead
   of relying on a separate link unit. `bootstrappable.c` even carries comments
   like *"M2-Planet/cc_* can't handle large signed numbers in literals"* and codes
   around it. So `__M2__` **must be defined** when building under M2 (the recipe
   passes it), and the tools' source is already written within the M2 C subset.

3. **No M2-hostile constructs found.** A scan for `switch`, `union`, variadic
   `...`, `va_list`, `long long`, and bare-`unsigned`-without-type turned up
   nothing that the M2 subset rejects. The tools use only: `enum`, `struct` with
   self-referential pointers (e.g. `struct input_files { struct input_files*
   next; ... }`), `do/while`, `for`, function forward-declarations, and the
   `#ifdef __M2__` guard — all within M2-Planet's documented capabilities
   (structs + function pointers OK).

4. **`stringify.c` is a separate compile/link unit (no header).** Both `M1.c`
   and `blood-elf.c` forward-declare `int stringify(...)` and `void
   LittleEndian(...)` inline and call them, but the bodies live in
   `stringify.c`. The build must compile/concatenate `stringify.c` alongside each
   of those two tools (it is vendored into both `M1/` and `blood-elf/`). `hex2`
   does **not** need it (it does its own little-endian byte emission inline). If
   the build uses M2's catm-concatenation model, append `stringify.c` to the M1
   and blood-elf translation units; if it uses an `#ifdef __M2__ : #include`
   trick like bootstrappable, none is present in upstream for stringify, so
   concatenation/explicit-link is required.

5. **`bootstrappable.c` may be double-defined if both `__M2__`-include and a
   separate link unit are used.** Because `bootstrappable.h` source-includes
   `bootstrappable.c` under `__M2__`, do **not** *also* pass `bootstrappable.c`
   as its own translation unit in the M2 build — that would define
   `require`/`match`/… twice. (Under a non-M2 host compiler the `#else` branch
   gives prototypes only, and `bootstrappable.c` *is* compiled separately — the
   two models are mutually exclusive.)

## License & separability

mescc-tools and the M2-Planet-origin `bootstrappable.{c,h}` / `ELF-amd64-debug.hex2`
are **GPL-3.0-or-later** (verbatim headers retained). They live entirely under
`stage0/M1/`, `stage0/blood-elf/`, `stage0/hex2-linker/`, and the existing
`stage0/M2-Planet/M2libc/` GPL tree — **statically separable** from the Apache-2.0
`helixc-bootstrap` seed and the helix top. No GPL source is linked into, or copied
into, any Apache-2.0 artifact; these tools are standalone auxiliary executables
used only to *verify* M2-Planet, not part of the shipped compiler.

## Provenance of this manifest

All files fetched via `raw.githubusercontent.com` at the pins above; SHA-256
computed under WSL (`sha256sum`). Exact source URLs:

- `https://raw.githubusercontent.com/oriansj/mescc-tools/5adfbf3364261a77109878a56b100aeeb6ef9ac4/M1-macro.c`
- `https://raw.githubusercontent.com/oriansj/mescc-tools/5adfbf3364261a77109878a56b100aeeb6ef9ac4/stringify.c`
- `https://raw.githubusercontent.com/oriansj/mescc-tools/5adfbf3364261a77109878a56b100aeeb6ef9ac4/blood-elf.c`
- `https://raw.githubusercontent.com/oriansj/mescc-tools/5adfbf3364261a77109878a56b100aeeb6ef9ac4/hex2.c`
- `https://raw.githubusercontent.com/oriansj/mescc-tools/5adfbf3364261a77109878a56b100aeeb6ef9ac4/hex2.h`
- `https://raw.githubusercontent.com/oriansj/mescc-tools/5adfbf3364261a77109878a56b100aeeb6ef9ac4/hex2_globals.h`
- `https://raw.githubusercontent.com/oriansj/mescc-tools/5adfbf3364261a77109878a56b100aeeb6ef9ac4/hex2_linker.c`
- `https://raw.githubusercontent.com/oriansj/mescc-tools/5adfbf3364261a77109878a56b100aeeb6ef9ac4/hex2_word.c`
- `https://raw.githubusercontent.com/oriansj/M2libc/b8bb2a0159a7376716a396ec6f6bc29dd27857b5/bootstrappable.c`
- `https://raw.githubusercontent.com/oriansj/M2libc/b8bb2a0159a7376716a396ec6f6bc29dd27857b5/bootstrappable.h`
- `https://raw.githubusercontent.com/oriansj/M2libc/b8bb2a0159a7376716a396ec6f6bc29dd27857b5/amd64/ELF-amd64-debug.hex2`
