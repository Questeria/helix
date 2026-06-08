# The x86-64 ELF back end

*What this chapter covers:* how `kovc` turns a parsed AST into a self-contained Linux executable
*with no external assembler and no external linker* — the `emit_elf_for_ast_to_path` path, the
byte-stream primitives that append machine code one slot at a time, the hand-built ELF and program
headers, the two-pass size patch, and the inline syscall sequences for file I/O. It closes the loop
on a quirk you have already met in Part II and Part IX: a successful `kovc` compile exits **non-zero**
because the compiler returns its **output byte count** as the process exit status. This is the **CPU**
back end; the GPU/PTX path is a separate emitter (Part VII, *planned*) and is cross-referenced, not
duplicated, here.

This chapter is for contributors and advanced operators. It assumes you have read
[part2-setup-build/03-using-kovc.md](../part2-setup-build/03-using-kovc.md) (how you *invoke* `kovc`
and stage `/tmp` paths) and, if you script the compiler, Trap 1 in
[part9-for-ai-agents/03-traps.md](../part9-for-ai-agents/03-traps.md). Everything below is grounded in
the real back-end source, [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx).

---

## The shape of the back end

`kovc` is the Helix compiler written in Helix
([`helixc/bootstrap/{lexer,parser,kovc}.hx`](../../../helixc/bootstrap/kovc.hx)). The front end
(lexer + parser) hands the back end an AST stored in the runtime **arena** — a flat array of `i32`
slots addressed by integer index, the only heap the bootstrap has. The back end's job is to walk that
AST and **append the bytes of a complete Linux ELF executable** into the same arena, then flush those
bytes to a file. There is no separate object format, no relocation table written to disk, no `as`, no
`ld`, no libc: the bytes the back end appends *are* the program.

The header comment at the top of [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) states
the contract directly:

**Fragment** (the back end's stated job; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 1–6):

```helix
// Stage-4 codegen for the Helix bootstrap compiler — kovc.
//
// Walks an AST (produced by stage-2 parser) and emits an x86-64
// Linux ELF executable byte stream into a separate arena region,
// then `write_file_to_arena` flushes it to disk.
```

The language spec records the same target precisely — a **static, syscall-only x86-64 Linux ELF** with
a single `PT_LOAD` segment and `.text` at `0x401000`
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §6):

**Fragment** (the codegen-target spec; from [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) line 152):

```text
- **CPU**: a **static, syscall-only x86-64 Linux ELF** — single `PT_LOAD`, `.text` at `0x401000`,
  no dynamic linker, System-V AMD64 ABI (6 int args in registers), a big-stack `_start` (mmaps 512 MiB
  then switches `rsp`, so deep self-compiles need no `ulimit`). Syscalls used:
  exit/read/write/mmap/mprotect/fork/execve/wait4/chmod. No register allocator or inliner beyond the ABI.
```

> **For AI agents:** the produced binary is a **single static ELF** — no interpreter, no `.so`
> dependency, no `ld`. You run it directly (after `chmod +x`); there is nothing to link. Do not look
> for a linker step in the build — there isn't one, by design.

---

## Byte emission: the arena *is* the assembler

Every byte of machine code and every byte of the ELF wrapper is produced by the same primitive:
append one `i32` slot to the arena whose low byte is the byte you want. Everything else —
little-endian integers, 64-bit fields, zero padding — is built on top of that.

**Fragment** (the byte-stream primitives; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 27–75):

```helix
fn emit_byte(b: i32) -> i32 {
    __arena_push(b);
    0
}

fn emit_u16_le(v: i32) -> i32 {
    __arena_push(v % 256);
    __arena_push((v / 256) % 256);
    0
}
```

`emit_u32_le` is more careful than it looks, and the comment explains why: the bootstrap dialect has
no bitwise operators in the codegen path and Helix's `/` is C-style truncated division, so a naïve
decomposition would emit the wrong bytes for *negative* values (a rel32 displacement, say). The fix is
to subtract each emitted byte before dividing:

**Fragment** (negative-safe 32-bit little-endian emit; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 38–57):

```helix
fn emit_u32_le(v: i32) -> i32 {
    // Phase-0 doesn't have bitwise ops, AND Helix's `/` is C-style
    // truncated division — so `-8 / 256 = 0`, not -1. A naive
    // decomposition writes `F8 00 00 00` for -8 instead of the
    // correct `F8 FF FF FF`. Workaround: subtract each emitted
    // byte before dividing, so the next round sees a value whose
    // exact division by 256 yields -1 for negative inputs.
    let b0 = (v % 256 + 256) % 256;
    let v1 = (v - b0) / 256;
    let b1 = (v1 % 256 + 256) % 256;
    let v2 = (v1 - b1) / 256;
    let b2 = (v2 % 256 + 256) % 256;
    let v3 = (v2 - b2) / 256;
    let b3 = (v3 % 256 + 256) % 256;
    __arena_push(b0);
    __arena_push(b1);
    __arena_push(b2);
    __arena_push(b3);
    0
}
```

64-bit fields are emitted as two 32-bit halves (`emit_u64_le_split(lo, hi)`), because the bootstrap
has no `i64` in scalar position. Zero padding is a loop of `__arena_push(0)` (`emit_zeros`). That is
the entire "assembler": no instruction encoder library, no relocation engine — just these few
functions, plus a few hundred `emit_byte(0x...)` calls scattered through the per-AST-tag code emitters
that spell out x86-64 opcodes literally.

You can see the literal-opcode style in the instruction helpers. The function prologue and the
process-exit stub are typical:

**Fragment** (a function prologue, byte for byte; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 1000–1006):

```helix
fn emit_prologue() -> i32 {
    emit_byte(0x55);
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xE5);
    emit_byte(0x48); emit_byte(0x81); emit_byte(0xEC);
    emit_u32_le(4096);
    11
}
```

That is `push rbp` / `mov rbp, rsp` / `sub rsp, 4096` — a 4096-byte-deep local frame — emitted as raw
bytes, returning the instruction length (`11`). The exit stub that turns a function result in `eax`
into a Linux `exit` syscall is just as literal:

**Fragment** (the `exit(eax)` stub; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 2234–2239):

```helix
fn emit_exit_with_eax() -> i32 {
    emit_byte(0x89); emit_byte(0xC7);
    emit_byte(0xB8); emit_byte(0x3C); emit_byte(0); emit_byte(0); emit_byte(0);
    emit_byte(0x0F); emit_byte(0x05);
    9
}
```

`mov edi, eax` / `mov eax, 60` (`sys_exit`) / `syscall` — this is where your `fn main() -> i32`'s
return value becomes the process exit status. (Its 8-bit truncation by the OS is covered in
[part2-setup-build/03-using-kovc.md](../part2-setup-build/03-using-kovc.md); that is the *program's*
exit code, distinct from `kovc`'s own, discussed at the end of this chapter.)

> **For AI agents:** each `emit_*` instruction helper returns the **number of bytes it emitted**, and
> the back end *sums* those returns to know where it is. When you read this code, the integer literal
> at the end of an `emit_*` function (e.g. `11`, `9`, `53`) is the encoded length, not a status — it
> must match the actual byte count pushed, or call-site offset math breaks.

---

## Building the ELF header by hand

The ELF wrapper is built from named constants the back end shares with the historical Python reference
emitter, documented in the source right above the header function:

**Fragment** (the ELF layout constants; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 78–91):

```helix
// ELF emission. Mirrors helixc/backend/x86_64.py::emit_elf with
// the same constants:
//   ELF_BASE = 0x400000
//   ENTRY_OFFSET = 0x1000
//   CODE_OFFSET = 0x1000
//   p_flags = 7 (R|W|X)
//   page_size = 0x1000
//
// Layout:
//   0x00   ELF header (64 bytes)
//   0x40   Program header (56 bytes)
//   0x78   Zero padding to 0x1000
//   0x1000 Code bytes
```

> **Note:** the cited `helixc/backend/x86_64.py` is the *historical Python-hosted* reference emitter —
> not in the shipped compile/run path (see the terminology note in the
> [Style Guide](../STYLE_GUIDE.md)). `kovc.hx` is the back end that actually ships and self-hosts; it
> reproduces the same byte layout in Helix.

The 64-byte ELF header is emitted field by field, each value a literal:

**Fragment** (the ELF64 header emitter; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 92–116):

```helix
fn emit_elf_header(code_size: i32) -> i32 {
    // EI_MAG: 0x7F 'E' 'L' 'F'
    emit_byte(127); emit_byte(69); emit_byte(76); emit_byte(70);
    emit_byte(2);    // EI_CLASS = ELFCLASS64
    emit_byte(1);    // EI_DATA = LSB
    emit_byte(1);    // EI_VERSION
    emit_byte(0);    // EI_OSABI = SysV
    emit_byte(0);    // EI_ABIVERSION
    emit_zeros(7);   // EI_PAD
    emit_u16_le(2);     // e_type = ET_EXEC
    emit_u16_le(62);    // e_machine = EM_X86_64 (0x3E)
    emit_u32_le(1);     // e_version
    // e_entry = ELF_BASE + ENTRY_OFFSET = 0x400000 + 0x1000 = 0x401000
    emit_u64_le_split(0x401000, 0);
    emit_u64_le_split(64, 0);   // e_phoff
    emit_u64_le_split(0, 0);    // e_shoff
    emit_u32_le(0);             // e_flags
    emit_u16_le(64);            // e_ehsize
    emit_u16_le(56);            // e_phentsize
    emit_u16_le(1);             // e_phnum
    emit_u16_le(0);             // e_shentsize
    emit_u16_le(0);             // e_shnum
    emit_u16_le(0);             // e_shstrndx
    0
}
```

Read against the ELF64 spec, every field is conventional: a `0x7F 'E' 'L' 'F'` magic, `ELFCLASS64`
(2), little-endian (`EI_DATA = 1`), `ET_EXEC` (a non-PIE static executable), `EM_X86_64` (62 / `0x3E`),
a fixed entry virtual address of `0x401000`, the program-header table at file offset 64 (`e_phoff`),
**no section-header table at all** (`e_shoff = 0`, `e_shnum = 0`), and exactly **one** program-header
entry (`e_phnum = 1`). A loader needs no sections to *run* a program — only the program header — so the
back end omits them entirely. `code_size` is accepted as a parameter but the header does not depend on
it (it is patched later, in the program header).

The single program header is the segment the kernel actually maps:

**Fragment** (the one `PT_LOAD` program header; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 118–130):

```helix
fn emit_program_header(code_size: i32) -> i32 {
    // total_filesz = CODE_OFFSET + code_size = 0x1000 + code_size
    let total_filesz = 4096 + code_size;
    emit_u32_le(1);                       // p_type = PT_LOAD
    emit_u32_le(7);                       // p_flags = R|W|X
    emit_u64_le_split(0, 0);              // p_offset = 0
    emit_u64_le_split(0x400000, 0);       // p_vaddr
    emit_u64_le_split(0x400000, 0);       // p_paddr
    emit_u64_le_split(total_filesz, 0);   // p_filesz
    emit_u64_le_split(total_filesz, 0);   // p_memsz
    emit_u64_le_split(4096, 0);           // p_align
    0
}
```

A few choices are worth calling out for a contributor:

- **One segment, mapped from file offset 0.** `p_offset = 0` and `p_vaddr = 0x400000` mean the whole
  file — ELF header included — is mapped at `0x400000`; code therefore lives at `0x400000 + 0x1000 =
  0x401000`, which is exactly `e_entry`.
- **`p_flags = 7` is R|W|X.** The single segment is readable, writable, *and* executable. That is
  deliberately permissive — it lets the produced program use the same region for code and for its
  mutable arena without a second segment. It is also a thing a hardening pass would later split; the
  bootstrap keeps it as one segment to stay minimal and auditable.
- **`p_filesz` / `p_memsz` are placeholders here.** They are written with the `code_size` passed in,
  but the real call passes `0` and the true sizes are patched in once codegen is done — see the next
  section. (`p_memsz` ends up *larger* than `p_filesz`, to give the program a BSS-style zero-filled
  arena beyond the bytes actually on disk.)

The padding between the end of the program header (file offset `0x78` = 120) and the code at `0x1000`
is a fixed run of zeros:

**Fragment** (header-to-code padding; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 132–137):

```helix
// Emit zero padding from end-of-phdr (file offset 0x78 = 120) to
// CODE_OFFSET (0x1000 = 4096). 4096 - 120 = 3976 bytes.
fn emit_padding_to_code() -> i32 {
    emit_zeros(3976);
    0
}
```

---

## Two passes by patching: write placeholder, fix it up later

Because sizes (and call targets) are not known until the code is laid out, the back end uses a simple
**backpatch** discipline: emit a placeholder now, overwrite it in place once the value is known. The
in-place writer is the mirror image of `emit_u32_le`, using `__arena_set(idx, byte)` instead of
`__arena_push`, and carrying the *same* negative-value workaround:

**Fragment** (in-place 32-bit patch; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 144–165):

```helix
fn patch_u32_le(idx: i32, v: i32) -> i32 {
    // Same negative-value workaround as emit_u32_le: must
    // produce `FF FF FF FF` for -1, not `FF 00 00 00`.
    let b0 = (v % 256 + 256) % 256;
    let v1 = (v - b0) / 256;
    let b1 = (v1 % 256 + 256) % 256;
    let v2 = (v1 - b1) / 256;
    let b2 = (v2 % 256 + 256) % 256;
    let v3 = (v2 - b2) / 256;
    let b3 = (v3 % 256 + 256) % 256;
    __arena_set(idx, b0);
    __arena_set(idx + 1, b1);
    __arena_set(idx + 2, b2);
    __arena_set(idx + 3, b3);
    0
}

fn patch_u64_le_split(idx: i32, lo: i32, hi: i32) -> i32 {
    patch_u32_le(idx, lo);
    patch_u32_le(idx + 4, hi);
    0
}
```

The same placeholder-then-patch mechanism resolves inter-function `call rel32` displacements (a
`patch_table` records each pending displacement slot and the callee name; after every function's code
offset is known, the table is walked and the rel32s are filled in). The header sizes are the simplest
case of it.

---

## The driver: `emit_elf_for_ast_to_path`

`emit_elf_for_ast_to_path(ast_root)` is the single entry point the compiler driver calls to produce
the whole ELF. It is large because it also runs the validation passes and lays out every function, but
its **ELF spine** is short and worth reading end to end.

First, it allocates all compile-time bookkeeping *before* the ELF region, so those slots do not land in
the middle of the contiguous code byte stream, then it writes the header, program header, and padding,
and records where the code starts:

**Fragment** (the ELF region is opened *after* compile-state allocation; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 10298–10302):

```helix
    let elf_start = __arena_len();
    emit_elf_header(0);
    emit_program_header(0);
    emit_padding_to_code();
    let code_start = __arena_len();
```

Note `emit_elf_header(0)` and `emit_program_header(0)`: the sizes are written as zero placeholders here,
to be patched at the end. `elf_start` is the arena index of the first ELF byte; `code_start` is the
first code byte (file offset `0x1000`).

Between `code_start` and the end, the driver emits the program body. For a normal multi-function
program (the parser yields an `AST_FN_LIST`, tag `15`) it emits a `_start` stub followed by every
function's code, backpatching the `call main` and inter-function calls. The `_start` stub is itself a
sequence of literal-byte emits — it switches to a 512 MiB `mmap`'d stack (so the deeply-recursive
self-compile does not overflow the kernel's default 8 MB stack), then calls `main`, then performs the
`exit` syscall:

**Fragment** (the `_start` stub layout; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 10308–10325):

```helix
        // _start stub:
        //   E8 ?? ?? ?? ??       call <main>            (5 bytes; backpatched)
        //   89 C7                mov edi, eax           (2)
        //   B8 3C 00 00 00       mov eax, 60 (sys_exit) (5)
        //   0F 05                syscall                (2)
        // task 16 / bug #1: switch to a 512 MiB mmap'd stack FIRST (before
        // `call main`), so the deeply-recursive self-compile no longer
        // overflows the 8 MB kernel stack and no external `ulimit -s
        // unlimited` is needed. Fail-safe: on mmap failure it keeps the
        // kernel stack. Adds 53 bytes at the entry; `call main` below is
        // still rel32-patched by NAME, so the entry-point shift is absorbed.
        emit_start_bigstack();
        let main_call_disp = emit_call_rel32_placeholder();
        patch_table_add(patch_state, main_call_disp, main_name_s, 4);
        emit_byte(0x89); emit_byte(0xC7);
        emit_byte(0xB8); emit_byte(0x3C); emit_byte(0); emit_byte(0); emit_byte(0);
        emit_byte(0x0F); emit_byte(0x05);
```

(The big-stack switch — `emit_start_bigstack` — is the 53-byte `mmap`/`test`/`js`/`lea rsp` sequence
documented at [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 2241–2283; on
`mmap` failure it falls through and keeps the kernel stack, so nothing regresses.)

Once the body is fully emitted, the driver measures the code, patches the two header size fields, and
**returns the total byte count of the ELF**:

**Fragment** (close-out: measure, patch sizes, return the byte count; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 10690–10700):

```helix
    let code_end = __arena_len();
    let total_filesz = 4096 + (code_end - code_start);
    // p_memsz extends past p_filesz to give the produced binary's
    // arena ~8 MB of BSS-allocated zero memory (4 bytes cursor +
    // 2097152 * 4 bytes data). Without this gap, an arena_push past
    // file bounds would SIGSEGV. Sized to match HELIX_ARENA_CAP in
    // helixc/backend/x86_64.py (the host compiler's bound).
    let total_memsz = total_filesz + 4 + helix_arena_data_bytes();
    patch_u64_le_split(elf_start + 64 + 32, total_filesz, 0);
    patch_u64_le_split(elf_start + 64 + 40, total_memsz, 0);
    total_filesz
```

Three things to take from this close-out:

1. **`total_filesz = 4096 + code bytes`** — the `0x1000` of headers+padding plus everything emitted
   from `code_start` onward. That is the exact number of bytes the file will contain.
2. **`p_filesz` and `p_memsz` are patched at the same fixed offsets.** `elf_start + 64` is the start of
   the program header (64-byte ELF header); `+ 32` and `+ 40` are the byte offsets of `p_filesz` and
   `p_memsz` inside that 56-byte program header. `p_memsz` is deliberately *larger* — it adds a 4-byte
   arena cursor plus `helix_arena_data_bytes()` of zero-filled BSS so the produced program has a heap
   to `__arena_push` into without running off the end of the file mapping and taking a `SIGSEGV`. (The
   arena cap is `helix_arena_cap() = 6291456` slots after a documented 2026-05-28 rescale;
   [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 4302–4313. The in-code comment
   above quoting `2097152` is a *period* note describing the pre-rescale size.)
3. **The function returns `total_filesz`.** This return value propagates all the way out to the
   driver's `main` — and that is the origin of the exit-status convention this chapter ends on.

---

## Writing the file: a syscall loop, not a libc call

The driver that becomes `K2` calls `emit_elf_for_ast_to_path`, then flushes the emitted region with
`write_file_to_arena`. The full driver is quoted in
[part2-setup-build/03-using-kovc.md](../part2-setup-build/03-using-kovc.md); the relevant two lines are:

**Fragment** (driver tail; from [`stage0/helixc-bootstrap/drivers/driver_k1input.hx`](../../../stage0/helixc-bootstrap/drivers/driver_k1input.hx)):

```helix
        let total = emit_elf_for_ast_to_path(ast_root);
        let elf_start = __arena_len() - total;
        write_file_to_arena("/tmp/k2_out.bin", elf_start, total)
```

`write_file_to_arena(path, start, count)` is **not** a Helix library function — it is a back-end
*intrinsic*: when `kovc` sees a call to it, the back end emits an inline syscall sequence directly into
the program (the spec lists it under *Builtins & intrinsics*,
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) line 139). The emitter is
`emit_write_file_to_arena_body`, and it is pure syscalls — `sys_open` with
`O_WRONLY|O_CREAT|O_TRUNC` (`0x241`) and mode `0644` (`0x1A4`), then a `sys_write` per byte:

**Fragment** (the inline `write_file_to_arena` open; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 4489–4496):

```helix
    // sys_open(path, O_WRONLY|O_CREAT|O_TRUNC=0x241, mode=0644=0x1A4)
    // mov esi, 0x241 (5 bytes)
    emit_byte(0xBE); emit_byte(0x41); emit_byte(0x02); emit_byte(0x00); emit_byte(0x00);
    // mov edx, 0x1A4 (5 bytes)
    emit_byte(0xBA); emit_byte(0xA4); emit_byte(0x01); emit_byte(0x00); emit_byte(0x00);
    // mov eax, 2; syscall
    emit_byte(0xB8); emit_byte(0x02); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
    emit_byte(0x0F); emit_byte(0x05);
```

The body then loops, copying one arena byte to a 1-byte stack buffer and issuing a 1-byte `sys_write`
each iteration ([`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 4512–4546). Its
read counterpart, `emit_read_file_to_arena_body`, is the symmetric `sys_open` + per-byte `sys_read`
sequence used by the *compiler itself* to slurp its source (`read_file_to_arena("/tmp/k2_in.hx")`).

This byte-at-a-time syscall I/O is exactly why builds are pathologically slow on a Windows `/mnt/c`
checkout: every one of those millions of single-byte syscalls pays a 9p round-trip across the WSL↔Windows
boundary. That performance trap — and the fix (build on WSL-native ext4) — is Trap 2 in
[part9-for-ai-agents/03-traps.md](../part9-for-ai-agents/03-traps.md); it is a wall-clock issue only,
with no effect on the bytes produced.

> **For AI agents:** `read_file_to_arena` / `write_file_to_arena` are **compiler intrinsics**, not
> stdlib calls — `kovc` inlines a raw `sys_open`/`sys_read`/`sys_write` sequence for them. There is no
> libc in the produced binary to intercept or `LD_PRELOAD`. The first argument to each **must be a
> string literal** (spec line 139, "first-arg-must-be-strlit"); you cannot pass a computed path.

---

## The byte-count exit convention (cross-reference: Part IX, Trap 1)

Here is the consequence that surprises everyone scripting `kovc`, and it falls straight out of the
back-end code above. `emit_elf_for_ast_to_path` *returns* `total_filesz` — the size of the ELF it
wrote. The `write_file_to_arena(...)` call that ends the driver also yields a count. The driver's
`main` returns that value, and a Helix program's `main` return value becomes the process exit status
(via the `mov edi, eax` / `mov eax, 60` / `syscall` stub shown earlier). So:

**A *successful* `kovc` compile exits *non-zero* — its exit status is the output byte count, modulo
256.**

This is documented behavior, not a bug. The self-compiled `kovc` is **698392 bytes**, and `698392 mod
256 = 24`, so a clean self-compile exits `24`. Trap 1 in
[part9-for-ai-agents/03-traps.md](../part9-for-ai-agents/03-traps.md) explains it as **symptom → why →
fix** and quotes the gate's own comment block; the load-bearing rule for anyone driving the compiler
is:

- **Never test a `kovc` compile with `[ $? -eq 0 ]`.** A good compile exits non-zero.
- **Test for the output file instead:** it exists and is non-empty (`[ -s /tmp/k2_out.bin ]`), and —
  where you have one — matches the pinned SHA-256. The `check()` / `chk` helpers in
  [part2-setup-build/03-using-kovc.md](../part2-setup-build/03-using-kovc.md) do exactly this, and
  always `rm -f` the output path *before* the compile so a stale ELF can't masquerade as a fresh one.

The one place an exit code *is* meaningful is a **C-compiled** leg (the `seed → K1` step, where `seed`
is a C binary that exits `0` on success). That distinction is drawn carefully in Trap 1; reserve
exit-code assertions for C-compiled binaries, never for a `kovc`/K-binary self-compile.

> **For AI agents:** a non-zero status from a `kovc` compile is the **expected success signal** (the
> output byte count `mod 256`), not an error. Validate with *output-exists + non-empty + (where
> pinned) SHA-256*, exactly as the gate does. This rule is the single most common way an operator
> wrongly marks a passing build as broken — see Trap 1.

---

## What this back end is — and is not

To keep the scope honest:

- **It is the CPU back end, all the way down.** From the raw-binary root up, the CPU path is built and
  reproduced with no trusted pre-built compiler; the ELF this chapter describes is emitted by `kovc`,
  which is itself reproduced byte-identically by the self-host fixpoint and cross-checked by the
  `gcc` diverse-double-compile. The closed state and every residual are recorded in
  [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md).
- **It is deliberately minimal.** No register allocator and no inliner beyond the System-V ABI
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §6); one R|W|X
  `PT_LOAD` segment; no section headers; fixed `/tmp` I/O paths. Every one of those choices trades
  sophistication for auditability — the whole emitter is a few thousand lines of literal-byte `emit_*`
  calls you can read top to bottom.
- **It is not the GPU path.** `kovc` has a *second*, separate emitter for `@kernel` functions that
  produces **NVIDIA PTX text** (`emit_ptx_*`, beginning at
  [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) line 10751) rather than ELF binary.
  That path is covered in **Part VII — GPU Codegen** (*planned*). The honest boundary there is sharp
  and must not be overstated: the chain is **complete to PTX, not to GPU machine code** — below PTX it
  trusts NVIDIA's closed `ptxas`, the CUDA driver, and the GPU hardware; GPU performance is a
  *fraction* of cuBLAS (~50–67.5% on the reference sm_86), and the end-to-end capstone speedup is
  **7.0–8.7×**, not ≥10× (see [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)). The
  CPU path in *this* chapter has no such residual: it is binary all the way from the hand-typed root.

---

**Next:** Part VI — *The From-Raw Bootstrap Ladder* (*planned*): how the binary this back end emits is
itself reachable from 299 hand-authored hex bytes — `hex0 → … → seed → kovc` — and how the self-host
fixpoint proves `kovc` reproduces itself byte-for-byte. (Until that part ships, see the build narrative
in [part2-setup-build/03-using-kovc.md](../part2-setup-build/03-using-kovc.md) and the trust record in
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md).)
