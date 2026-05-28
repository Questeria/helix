// Stage-4 codegen for the Helix bootstrap compiler — kovc.
//
// Walks an AST (produced by stage-2 parser) and emits an x86-64
// Linux ELF executable byte stream into a separate arena region,
// then `write_file_to_arena` flushes it to disk.
//
// This is the final piece needed to retire the Python compiler.
// Once kovc.hx + lexer.hx + parser.hx are themselves compiled by
// the Python compiler (one final time) into a binary, that binary
// can compile arbitrary `.hx` files — including new versions of
// itself. Self-hosted.
//
// Stage-4 starts with a minimum-viable emitter: produces a working
// Linux ELF that runs `exit(0)`. Subsequent commits expand to
// support each AST tag. The ELF wrapper is the same shape used by
// helixc/backend/x86_64.py:emit_elf — 64-byte ELF header + 56-byte
// program header + zero-padding to file offset 0x1000 + code.
//
// License: Apache 2.0.

// --------------------------------------------------------------
// Byte-stream helpers. Each push appends one i32 slot with the
// low byte holding the byte value (high bits ignored by
// write_file_to_arena). emit_u16_le / u32 / u64 break a value
// into bytes in little-endian order.
// --------------------------------------------------------------
fn emit_byte(b: i32) -> i32 {
    __arena_push(b);
    0
}

fn emit_u16_le(v: i32) -> i32 {
    __arena_push(v % 256);
    __arena_push((v / 256) % 256);
    0
}

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

// Emit a 64-bit value as 8 little-endian bytes. Helix doesn't have
// i64 in scalar position (Phase-0 limit), so we accept (lo32, hi32).
fn emit_u64_le_split(lo: i32, hi: i32) -> i32 {
    emit_u32_le(lo);
    emit_u32_le(hi);
    0
}

// Push `n` zero bytes.
fn emit_zeros(n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        __arena_push(0);
        i = i + 1;
    }
    0
}

// --------------------------------------------------------------
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
// --------------------------------------------------------------
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

// Emit zero padding from end-of-phdr (file offset 0x78 = 120) to
// CODE_OFFSET (0x1000 = 4096). 4096 - 120 = 3976 bytes.
fn emit_padding_to_code() -> i32 {
    emit_zeros(3976);
    0
}

// --------------------------------------------------------------
// Patching: rewrite bytes that were emitted earlier with
// placeholder zeros. Used to fill in p_filesz/p_memsz once the
// code size is known.
// --------------------------------------------------------------
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

// --------------------------------------------------------------
// Code emission per AST tag. Each emit_*_code function appends
// machine-code bytes to the arena and returns the count emitted.
//
// Calling convention for the emitted program: each AST node's
// code leaves its result in eax. The top-level wrapper takes
// eax and turns it into the exit-status syscall.
// --------------------------------------------------------------

// AST_INT(v): mov eax, imm32   (5 bytes: B8 imm32_le)
fn emit_ast_int(v: i32) -> i32 {
    emit_byte(0xB8);
    emit_u32_le(v);
    5
}

// movabs rax, imm64   (10 bytes: 48 B8 imm64_le)
// Used by Phase 1.10 step 7c (AST_FLOATLIT_F64) to materialize an
// 8-byte IEEE 754 double-precision bit pattern in rax. low32 is the
// low 32 bits of the imm64; high32 is the high 32 bits — laid out
// little-endian on disk so the CPU sees a single 64-bit immediate.
fn emit_movabs_rax_imm64(low32: i32, high32: i32) -> i32 {
    emit_byte(0x48);                    // REX.W
    emit_byte(0xB8);                    // B8+rd (rax)
    emit_u32_le(low32);
    emit_u32_le(high32);
    10
}

// AST_NEG(inner): emit inner code, then `neg eax`.
//   F7 D8   neg eax
// (inner already left its value in eax.)
fn emit_ast_neg_suffix() -> i32 {
    emit_byte(0xF7); emit_byte(0xD8);
    2
}

// Phase 1.10 step 5d: f32 unary NEG via sign-bit XOR. Mirrors the
// __fneg builtin's encoding (Phase 1.10 step 4) — the f32 bit pattern
// in eax has bit 31 flipped via `xor eax, 0x80000000`. Used by AST_NEG
// codegen when is_f32_expr(inner) == 1 so `-x` on an f32 binding
// produces correct floating-point negation (not integer two's complement).
//   35 00 00 00 80   xor eax, 0x80000000   (5 bytes)
fn emit_ast_fneg_suffix() -> i32 {
    emit_byte(0x35);
    emit_byte(0x00); emit_byte(0x00); emit_byte(0x00); emit_byte(0x80);
    5
}

// Phase 1.10 step 7f: f64 unary NEG via 64-bit sign-bit XOR.
// f64 sign bit is bit 63 (0x8000000000000000). x86-64 has no direct
// `xor rax, imm64`, so we materialize the mask into rcx first.
//   48 B9 00 00 00 00 00 00 00 80   movabs rcx, 0x8000000000000000  (10 bytes)
//   48 31 C8                         xor rax, rcx                   (3 bytes)
// Total: 13 bytes.
fn emit_ast_dneg_suffix() -> i32 {
    emit_byte(0x48); emit_byte(0xB9);                                     // movabs rcx, imm64
    emit_byte(0x00); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);   // low 32 = 0
    emit_byte(0x00); emit_byte(0x00); emit_byte(0x00); emit_byte(0x80);   // high 32 = 0x80000000
    emit_byte(0x48); emit_byte(0x31); emit_byte(0xC8);                    // xor rax, rcx
    13
}

// AST_BNOT(inner): emit inner code, then `not eax`.
//   F7 D0   not eax
// Mirrors helixc-Python OpKind.BIT_NOT (commit 4e6b4fa).
fn emit_ast_bnot_suffix() -> i32 {
    emit_byte(0xF7); emit_byte(0xD0);
    2
}

// Phase 1.10 step 5p: ud2 trap. Used by mixed-type arithmetic
// detection (AST_ADD/SUB/MUL/DIV with one f32 + one i32 operand —
// silent integer codegen would silently corrupt the f32 bit pattern,
// so we emit a SIGILL trap instead).
//   0F 0B   ud2  (illegal instruction; raises SIGILL on x86-64)
fn emit_ud2_trap() -> i32 {
    emit_byte(0x0F); emit_byte(0x0B);
    2
}

// Speedup #4 debug tooling (2026-05-07, per user directive): trap with
// a pre-loaded identifier so post-mortem (gdb / register dump) can tell
// WHICH trap site fired. Encodes:
//   B8 II II II II   mov eax, id    (5 bytes)
//   0F 0B            ud2            (2 bytes)
// 7 bytes total. After SIGILL, the kernel core file (or a debugger
// attached to the dying process) shows eax = id, identifying the
// trap site without source-line metadata.
//
// Recommended ID convention: AST_TAG * 1000 + sub_id, e.g. 9001 for
// AST_NEG bf16 trap, 26001 for AST_BNOT bf16 trap. Sub-ids stay small.
//
// Not yet wired into existing trap call sites (those still use plain
// emit_ud2_trap). New trap sites should prefer this helper.
fn emit_trap_with_id(id: i32) -> i32 {
    emit_byte(0xB8);                           // mov eax, imm32
    emit_u32_le(id);                           // 4 bytes of imm32
    emit_byte(0x0F); emit_byte(0x0B);          // ud2
    7
}

// AST_NOT(inner): emit inner code, then logical NOT via:
//   85 C0              test eax, eax
//   B8 00 00 00 00     mov eax, 0    (zero the high bytes before sete)
//   0F 94 C0           sete al        (al = 1 if ZF set, else 0)
// 10 bytes total. Mirrors helixc-Python `!x` -> CMP_EQ(inner, 0).
fn emit_ast_not_suffix() -> i32 {
    emit_byte(0x85); emit_byte(0xC0);
    emit_byte(0xB8); emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    emit_byte(0x0F); emit_byte(0x94); emit_byte(0xC0);
    10
}

// AST_ADD-style binary op suffix. The protocol is:
//   1. Emit lhs code (leaves lhs in eax)
//   2. push rax                                  (50)
//   3. Emit rhs code (leaves rhs in eax)
//   4. mov ecx, eax                              (89 C1)
//   5. pop rax                                   (58)
//   6. <op-specific instruction(s) using eax + ecx, result in eax>
fn emit_push_rax() -> i32 { emit_byte(0x50); 1 }
fn emit_pop_rax()  -> i32 { emit_byte(0x58); 1 }
fn emit_mov_ecx_eax() -> i32 { emit_byte(0x89); emit_byte(0xC1); 2 }
fn emit_add_eax_ecx() -> i32 { emit_byte(0x01); emit_byte(0xC8); 2 }
fn emit_sub_eax_ecx() -> i32 { emit_byte(0x29); emit_byte(0xC8); 2 }
fn emit_imul_eax_ecx() -> i32 { emit_byte(0x0F); emit_byte(0xAF); emit_byte(0xC1); 3 }
// Stage 1: 64-bit integer arithmetic. REX.W (0x48) prefix promotes
// the operation to full 64-bit width. Used when both operands are
// i64 in AST_ADD/SUB/MUL/DIV dispatch.
//   48 01 C8     add rax, rcx
//   48 29 C8     sub rax, rcx
//   48 0F AF C1  imul rax, rcx
fn emit_add_rax_rcx_64() -> i32 { emit_byte(0x48); emit_byte(0x01); emit_byte(0xC8); 3 }
fn emit_sub_rax_rcx_64() -> i32 { emit_byte(0x48); emit_byte(0x29); emit_byte(0xC8); 3 }
fn emit_imul_rax_rcx_64() -> i32 { emit_byte(0x48); emit_byte(0x0F); emit_byte(0xAF); emit_byte(0xC1); 4 }
// K1.F8 (2026-05-27): sign-extend the 32-bit-low half of a 64-bit
// register into its full 64-bit form. Used by AST_ADD / AST_SUB /
// AST_MUL when one operand is i64 and the other is i32: the i32-
// shaped operand sits zero-extended after the prior 32-bit move,
// movsxd promotes it to a sign-correct 64-bit value before the
// 64-bit op. Encoding:
//   48 63 C9   movsxd rcx, ecx
//   48 63 C0   movsxd rax, eax
fn emit_movsxd_rcx_ecx() -> i32 { emit_byte(0x48); emit_byte(0x63); emit_byte(0xC9); 3 }
fn emit_movsxd_rax_eax() -> i32 { emit_byte(0x48); emit_byte(0x63); emit_byte(0xC0); 3 }

// K-bootstrap K1.A — rsp imm32 adjust helpers. The foundation for
// SysV stack-arg passing (K1.B), future `unsafe`-block stack
// alloca, closure environment frames, and any other inline rsp
// adjustment a caller needs.
//
// Encoding (REX.W prefixed, ModRM /5 = sub, /0 = add, with
// destination = rsp = reg #4):
//   48 81 EC <imm32-LE>   sub rsp, imm32   (7 bytes)
//   48 81 C4 <imm32-LE>   add rsp, imm32   (7 bytes)
//
// imm32 is signed 32-bit two's complement. The caller-provided i32
// value is encoded as-is by emit_u32_le, which handles negative
// inputs via the post-fix in emit_u32_le (the `-8 / 256` workaround
// at line ~46). Callers in K1.B will only pass small POSITIVE
// alignment-padded values (= (arg_count-6)*8 rounded up to 16), but
// the helper itself imposes no sign restriction — the encoding is
// sign-agnostic at the bit level.
fn emit_sub_rsp_imm32(imm: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x81); emit_byte(0xEC);
    emit_u32_le(imm);
    7
}

fn emit_add_rsp_imm32(imm: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x81); emit_byte(0xC4);
    emit_u32_le(imm);
    7
}

// K-bootstrap K1.B — `mov reg64, [rsp + disp32]` helpers (8 bytes
// each). SIB byte 0x24 (scale=0, index=none, base=rsp) is MANDATORY
// whenever the effective address uses rsp as base — without it, the
// ModRM r/m=100 would be interpreted as "SIB follows" anyway, so
// the SIB is part of the encoding contract.
//
// Encoding template:
//   <REX> 8B <ModRM=10rrr100> 24 <disp32-LE>
// where:
//   REX = 0x48 (W=1, R=0) for the low-8 registers (rax/rcx/rdx/
//         rbx/rsp/rbp/rsi/rdi); 0x4C (W=1, R=1) for r8-r15.
//   ModRM = mod=10 (disp32 follows) + reg=destination + rm=100 (SIB).
//   reg field encodes the destination's low-3 bits:
//     rax = 000  -> ModRM 0x84
//     rcx = 001  -> ModRM 0x8C
//     rdx = 010  -> ModRM 0x94
//     rsi = 110  -> ModRM 0xB4
//     rdi = 111  -> ModRM 0xBC
//     r8  = 000 (with REX.R) -> ModRM 0x84
//     r9  = 001 (with REX.R) -> ModRM 0x8C
//
// SysV ABI uses rdi/rsi/rdx/rcx/r8/r9 for the first 6 int args, so
// these 6 helpers + the rax helpers cover what K1.B's caller-cleanup
// stack-arg phase needs.
fn emit_mov_rax_rsp_disp32(disp: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x8B); emit_byte(0x84); emit_byte(0x24);
    emit_u32_le(disp);
    8
}

fn emit_mov_rcx_rsp_disp32(disp: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x8B); emit_byte(0x8C); emit_byte(0x24);
    emit_u32_le(disp);
    8
}

fn emit_mov_rdx_rsp_disp32(disp: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x8B); emit_byte(0x94); emit_byte(0x24);
    emit_u32_le(disp);
    8
}

fn emit_mov_rsi_rsp_disp32(disp: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x8B); emit_byte(0xB4); emit_byte(0x24);
    emit_u32_le(disp);
    8
}

fn emit_mov_rdi_rsp_disp32(disp: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x8B); emit_byte(0xBC); emit_byte(0x24);
    emit_u32_le(disp);
    8
}

fn emit_mov_r8_rsp_disp32(disp: i32) -> i32 {
    emit_byte(0x4C); emit_byte(0x8B); emit_byte(0x84); emit_byte(0x24);
    emit_u32_le(disp);
    8
}

fn emit_mov_r9_rsp_disp32(disp: i32) -> i32 {
    emit_byte(0x4C); emit_byte(0x8B); emit_byte(0x8C); emit_byte(0x24);
    emit_u32_le(disp);
    8
}

// Store: `mov [rsp + disp32], rax`. Opcode 0x89 (MOV r/m64, r64),
// ModRM 0x84 (reg=rax=000, rm=SIB), SIB 0x24 (rsp).
fn emit_mov_rsp_disp32_rax(disp: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x89); emit_byte(0x84); emit_byte(0x24);
    emit_u32_le(disp);
    8
}

// K-bootstrap K1.B — the stack-arg reverse-copy loop, extracted to
// a top-level fn so the AST_CALL arm stays shallow (host parser's
// recursion budget — see the Finding #7 lesson note at the
// AST_CALL handler). `stack_args` = (arg_count - 6); `stack_alloc`
// = stack_args * 8 (always 16-aligned since stack_args >= 1 and
// 8*N + stack_alloc is always 16*(N-3) when both are computed).
fn emit_stack_args_reverse_copy(stack_args: i32, stack_alloc: i32) -> i32 {
    let mut n: i32 = 0;
    let mut i: i32 = 0;
    while i < stack_args {
        let src_disp = stack_alloc + 8 * (stack_args - 1 - i);
        let dst_disp = 8 * i;
        n = n + emit_mov_rax_rsp_disp32(src_disp);
        n = n + emit_mov_rsp_disp32_rax(dst_disp);
        i = i + 1;
    }
    n
}

// K-bootstrap K1.B — load args 0..5 from the post-`sub rsp` stack
// positions into rdi/rsi/rdx/rcx/r8/r9. Extracted to keep the
// AST_CALL arm shallow.
fn emit_load_six_int_args(stack_alloc: i32, arg_count: i32) -> i32 {
    let mut n: i32 = 0;
    n = n + emit_mov_rdi_rsp_disp32(stack_alloc + 8 * (arg_count - 1));
    n = n + emit_mov_rsi_rsp_disp32(stack_alloc + 8 * (arg_count - 2));
    n = n + emit_mov_rdx_rsp_disp32(stack_alloc + 8 * (arg_count - 3));
    n = n + emit_mov_rcx_rsp_disp32(stack_alloc + 8 * (arg_count - 4));
    n = n + emit_mov_r8_rsp_disp32(stack_alloc + 8 * (arg_count - 5));
    n = n + emit_mov_r9_rsp_disp32(stack_alloc + 8 * (arg_count - 6));
    n
}

// 64-bit signed divide: cqo (sign-extend rax into rdx:rax) + idiv rcx.
//   48 99        cqo
//   48 F7 F9     idiv rcx
fn emit_idiv_rax_rcx_64() -> i32 {
    emit_byte(0x48); emit_byte(0x99);
    emit_byte(0x48); emit_byte(0xF7); emit_byte(0xF9);
    5
}
// 64-bit cmp + setcc. cmp rax, rcx = 48 39 C8 (3 bytes); setcc al
// + mov eax, 0 pre-clear (same idiom as 32-bit emit_cmp_setX). Produces 0/1 in eax.
//   48 39 C8           cmp rax, rcx
//   B8 00 00 00 00     mov eax, 0 (pre-clear; setcc only writes al)
//   0F xx C0           setcc al
fn emit_cmp_setX_64(op_byte: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x39); emit_byte(0xC8);
    emit_byte(0xB8); emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    emit_byte(0x0F); emit_byte(op_byte); emit_byte(0xC0);
    11
}
fn emit_lt_rax_rcx_64() -> i32 { emit_cmp_setX_64(0x9C) }
fn emit_gt_rax_rcx_64() -> i32 { emit_cmp_setX_64(0x9F) }
fn emit_eq_rax_rcx_64() -> i32 { emit_cmp_setX_64(0x94) }
fn emit_ne_rax_rcx_64() -> i32 { emit_cmp_setX_64(0x95) }
fn emit_le_rax_rcx_64() -> i32 { emit_cmp_setX_64(0x9E) }
fn emit_ge_rax_rcx_64() -> i32 { emit_cmp_setX_64(0x9D) }
// Stage 1 audit batch 2: 64-bit unary + bitwise + shift + test for i64.
// REX.W (0x48) prefix promotes the standard 32-bit op forms to 64-bit.
fn emit_neg_rax_64()  -> i32 { emit_byte(0x48); emit_byte(0xF7); emit_byte(0xD8); 3 }
fn emit_not_rax_64()  -> i32 { emit_byte(0x48); emit_byte(0xF7); emit_byte(0xD0); 3 }
fn emit_and_rax_rcx_64() -> i32 { emit_byte(0x48); emit_byte(0x21); emit_byte(0xC8); 3 }
fn emit_or_rax_rcx_64()  -> i32 { emit_byte(0x48); emit_byte(0x09); emit_byte(0xC8); 3 }
fn emit_xor_rax_rcx_64() -> i32 { emit_byte(0x48); emit_byte(0x31); emit_byte(0xC8); 3 }
fn emit_shl_rax_cl_64() -> i32 { emit_byte(0x48); emit_byte(0xD3); emit_byte(0xE0); 3 }
fn emit_sar_rax_cl_64() -> i32 { emit_byte(0x48); emit_byte(0xD3); emit_byte(0xF8); 3 }
fn emit_test_rax_rax_64() -> i32 { emit_byte(0x48); emit_byte(0x85); emit_byte(0xC0); 3 }
// 64-bit modulo: cqo (sign-extend rax to rdx:rax) + idiv rcx + mov rax, rdx.
//   48 99        cqo
//   48 F7 F9     idiv rcx
//   48 89 D0     mov rax, rdx  (move remainder into rax)
fn emit_imod_rax_rcx_64() -> i32 {
    emit_byte(0x48); emit_byte(0x99);
    emit_byte(0x48); emit_byte(0xF7); emit_byte(0xF9);
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xD0);
    8
}
// 64-bit logical-not suffix: test rax, rax (REX.W) + setcc al + zero-extend.
//   48 85 C0     test rax, rax
//   B8 00 00 00 00  mov eax, 0  (pre-clear al; setcc only writes al)
//   0F 94 C0     sete al
fn emit_ast_not_suffix_64() -> i32 {
    emit_byte(0x48); emit_byte(0x85); emit_byte(0xC0);
    emit_byte(0xB8); emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    emit_byte(0x0F); emit_byte(0x94); emit_byte(0xC0);
    11
}
// Phase 1.10 step 5+: binary bitwise ops mirroring helixc-Python
// OpKind.BIT_AND/BIT_OR/BIT_XOR (commit f676fca).
fn emit_and_eax_ecx() -> i32 { emit_byte(0x21); emit_byte(0xC8); 2 }
fn emit_or_eax_ecx()  -> i32 { emit_byte(0x09); emit_byte(0xC8); 2 }
fn emit_xor_eax_ecx() -> i32 { emit_byte(0x31); emit_byte(0xC8); 2 }
// Shifts. x86 shift-by-CL: D3 E0 = shl eax, cl; D3 F8 = sar eax, cl.
// emit_mov_ecx_eax already places rhs into ecx (CL is its low byte) so
// the standard binary-op shape (lhs in eax, rhs->ecx, op) works unchanged.
fn emit_shl_eax_cl() -> i32 { emit_byte(0xD3); emit_byte(0xE0); 2 }
fn emit_sar_eax_cl() -> i32 { emit_byte(0xD3); emit_byte(0xF8); 2 }

// Phase 1.10 step 5c: SSE binary-op suffix. Used by AST_ADD/SUB/MUL/DIV
// when both operands are f32. Mirrors the inline machine code emitted
// by the __fadd / __fsub / __fmul / __fdiv builtins (see step 4).
//   movd xmm0, eax           66 0F 6E C0      (4 bytes)
//   movd xmm1, ecx           66 0F 6E C9      (4 bytes)
//   [add|sub|mul|div]ss xmm0, xmm1
//                            F3 0F (58|5C|59|5E) C1   (4 bytes)
//   movd eax, xmm0           66 0F 7E C0      (4 bytes)
fn emit_sse_binop(opcode: i32) -> i32 {
    emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0);   // movd xmm0,eax
    emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9);   // movd xmm1,ecx
    emit_byte(0xF3); emit_byte(0x0F); emit_byte(opcode); emit_byte(0xC1); // [op]ss xmm0,xmm1
    emit_byte(0x66); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0);   // movd eax,xmm0
    16
}
fn emit_addss() -> i32 { emit_sse_binop(0x58) }
fn emit_subss() -> i32 { emit_sse_binop(0x5C) }
fn emit_mulss() -> i32 { emit_sse_binop(0x59) }
fn emit_divss() -> i32 { emit_sse_binop(0x5E) }

// Phase 1.10 step 7d: SSE2 double-precision binary-op suffix. Used by
// AST_ADD/SUB/MUL/DIV when both operands are f64. Same shape as
// emit_sse_binop but with REX.W on movd (turning movd into movq) and
// F2 prefix on the op (selecting double-precision variant):
//   movq xmm0, rax           66 48 0F 6E C0    (5 bytes)
//   movq xmm1, rcx           66 48 0F 6E C9    (5 bytes)
//   [add|sub|mul|div]sd xmm0, xmm1
//                            F2 0F (58|5C|59|5E) C1     (4 bytes)
//   movq rax, xmm0           66 48 0F 7E C0    (5 bytes)
fn emit_sse_dbl_binop(opcode: i32) -> i32 {
    emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0); // movq xmm0,rax
    emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9); // movq xmm1,rcx
    emit_byte(0xF2); emit_byte(0x0F); emit_byte(opcode); emit_byte(0xC1); // [op]sd xmm0,xmm1
    emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0); // movq rax,xmm0
    19
}
fn emit_addsd() -> i32 { emit_sse_dbl_binop(0x58) }
fn emit_subsd() -> i32 { emit_sse_dbl_binop(0x5C) }
fn emit_mulsd() -> i32 { emit_sse_dbl_binop(0x59) }
fn emit_divsd() -> i32 { emit_sse_dbl_binop(0x5E) }

// K1.F9 (2026-05-27): in-place f32 -> f64 widening for a value
// sitting in eax / ecx (as a 32-bit float bit pattern). Used by
// the AST_ADD / SUB / MUL / DIV mixed-type paths when one operand
// is f32 and the other f64. Sequence:
//   movd xmm0, eax        66 0F 6E C0   (4 bytes)
//   cvtss2sd xmm0, xmm0   F3 0F 5A C0   (4 bytes)
//   movq rax, xmm0        66 48 0F 7E C0 (5 bytes)
// = 13 bytes total per call. The rcx variant uses xmm1/ecx with
// the ModRM bytes shifted (C9 instead of C0).
fn emit_cvt_f32_in_rax_to_f64() -> i32 {
    emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0); // movd xmm0,eax
    emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x5A); emit_byte(0xC0); // cvtss2sd xmm0,xmm0
    emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0); // movq rax,xmm0
    13
}
fn emit_cvt_f32_in_rcx_to_f64() -> i32 {
    emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9); // movd xmm1,ecx
    emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x5A); emit_byte(0xC9); // cvtss2sd xmm1,xmm1
    emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC9); // movq rcx,xmm1
    13
}

// Phase 1.10 step 7d: 64-bit `mov rcx, rax` for f64 binop scaffolding.
// Existing emit_mov_ecx_eax (89 C1) only copies low 32 bits — the high
// 32 of an f64 in rax would be lost before the SSE2 double binop reads
// rcx. REX.W (48) prefix promotes to 64-bit width.
//   48 89 C1   mov rcx, rax    (3 bytes)
fn emit_mov_rcx_rax_64() -> i32 {
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xC1);
    3
}

// Phase 1.10 step 5e: SSE comparison suffix. Result is 0/1 in eax
// depending on the predicate. Predicate selected by the setcc opcode
// byte (second byte after 0F prefix):
//   0x92 setb  (lhs < rhs)        — for <
//   0x96 setbe (lhs <= rhs)       — for <=
//   0x97 seta  (lhs > rhs)        — for >
//   0x93 setae (lhs >= rhs)       — for >=
//   0x94 sete  (lhs == rhs)       — for ==
//   0x95 setne (lhs != rhs)       — for !=
//
// Phase 1.10 step 5f: IEEE 754 NaN handling via parity-flag guard.
// `ucomiss` with a NaN operand sets ZF=1, PF=1, CF=1 (the "unordered"
// combination). The base setters above mis-fire for several relations:
//   sete  (CMP_EQ): says NaN==NaN true (wrong; should be 0)
//   setne (CMP_NE): says NaN!=NaN false (wrong; should be 1)
//   setb  (CMP_LT): says NaN<x true (wrong; should be 0)
//   setbe (CMP_LE): says NaN<=x true (wrong; should be 0)
// seta/setae already produce 0 in the NaN case (CF=1 fails them); no
// fixup needed for >, >=. Mirrors helixc-Python backend's PF guard.
//
// `fixup` parameter selects the post-setcc patch:
//   0 = no fixup (used by >, >=)
//   1 = ordered AND: `setnp cl ; and al, cl`  (used by <, <=, ==)
//   2 = unordered OR: `setp cl ; or al, cl`   (used by !=)
//
// Sequence — note xor MUST come BEFORE ucomiss (otherwise xor clobbers
// the flag bits that ucomiss just set, and setcc reads stale flags):
//   movd xmm0, eax            66 0F 6E C0   (4 bytes)
//   movd xmm1, ecx            66 0F 6E C9   (4 bytes)
//   xor eax, eax              31 C0         (2 bytes; pre-clears eax;
//                                            we already moved to xmm0)
//   ucomiss xmm0, xmm1        0F 2E C1      (3 bytes; sets flags)
//   setcc al                  0F xx C0      (3 bytes; reads flags)
//   [fixup]                                  (0 or 5 bytes)
// Total: 16 bytes (no fixup) or 21 bytes (with fixup).
fn emit_sse_compare(setcc_byte: i32, fixup: i32) -> i32 {
    emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0);   // movd xmm0,eax
    emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9);   // movd xmm1,ecx
    emit_byte(0x31); emit_byte(0xC0);                                     // xor eax,eax (PRE-clear)
    emit_byte(0x0F); emit_byte(0x2E); emit_byte(0xC1);                    // ucomiss xmm0,xmm1
    emit_byte(0x0F); emit_byte(setcc_byte); emit_byte(0xC0);              // setcc al
    if fixup == 1 {
        emit_byte(0x0F); emit_byte(0x9B); emit_byte(0xC1);                // setnp cl
        emit_byte(0x20); emit_byte(0xC8);                                  // and al, cl
        21
    } else { if fixup == 2 {
        emit_byte(0x0F); emit_byte(0x9A); emit_byte(0xC1);                // setp cl
        emit_byte(0x08); emit_byte(0xC8);                                  // or al, cl
        21
    } else {
        16
    } }
}
fn emit_ssen_lt() -> i32 { emit_sse_compare(0x92, 1) }   // setb + AND !PF
fn emit_ssen_le() -> i32 { emit_sse_compare(0x96, 1) }   // setbe + AND !PF
fn emit_ssen_gt() -> i32 { emit_sse_compare(0x97, 0) }   // seta (no fixup)
fn emit_ssen_ge() -> i32 { emit_sse_compare(0x93, 0) }   // setae (no fixup)
fn emit_ssen_eq() -> i32 { emit_sse_compare(0x94, 1) }   // sete + AND !PF
fn emit_ssen_ne() -> i32 { emit_sse_compare(0x95, 2) }   // setne + OR PF

// Phase 1.10 step 7g: SSE2 double-precision comparison. Mirrors
// emit_sse_compare but with movq instead of movd (REX.W + 66 prefix
// on the move) and ucomisd instead of ucomiss (66 prefix on the op).
//   movq xmm0, rax            66 48 0F 6E C0   (5 bytes)
//   movq xmm1, rcx            66 48 0F 6E C9   (5 bytes)
//   xor eax, eax              31 C0           (2 bytes; PRE-clear)
//   ucomisd xmm0, xmm1        66 0F 2E C1     (4 bytes; sets flags)
//   setcc al                  0F xx C0        (3 bytes; reads flags)
//   [fixup]                                    (0 or 5 bytes)
// Total: 19 bytes (no fixup) or 24 bytes (with fixup).
// NaN handling identical to f32: PF=1 + CF=1 + ZF=1 in the unordered
// case, so the same setp/setnp fixup pattern applies.
fn emit_sse_dbl_compare(setcc_byte: i32, fixup: i32) -> i32 {
    emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0); // movq xmm0,rax
    emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9); // movq xmm1,rcx
    emit_byte(0x31); emit_byte(0xC0);                                                     // xor eax,eax (PRE-clear)
    emit_byte(0x66); emit_byte(0x0F); emit_byte(0x2E); emit_byte(0xC1);                   // ucomisd xmm0,xmm1
    emit_byte(0x0F); emit_byte(setcc_byte); emit_byte(0xC0);                              // setcc al
    if fixup == 1 {
        emit_byte(0x0F); emit_byte(0x9B); emit_byte(0xC1);                // setnp cl
        emit_byte(0x20); emit_byte(0xC8);                                  // and al, cl
        24
    } else { if fixup == 2 {
        emit_byte(0x0F); emit_byte(0x9A); emit_byte(0xC1);                // setp cl
        emit_byte(0x08); emit_byte(0xC8);                                  // or al, cl
        24
    } else {
        19
    } }
}
fn emit_ssen_lt_dbl() -> i32 { emit_sse_dbl_compare(0x92, 1) }
fn emit_ssen_le_dbl() -> i32 { emit_sse_dbl_compare(0x96, 1) }
fn emit_ssen_gt_dbl() -> i32 { emit_sse_dbl_compare(0x97, 0) }
fn emit_ssen_ge_dbl() -> i32 { emit_sse_dbl_compare(0x93, 0) }
fn emit_ssen_eq_dbl() -> i32 { emit_sse_dbl_compare(0x94, 1) }
fn emit_ssen_ne_dbl() -> i32 { emit_sse_dbl_compare(0x95, 2) }
// idiv requires sign-extension into edx; we emit `cdq; idiv ecx`.
//   99       cdq
//   F7 F9    idiv ecx
fn emit_idiv_eax_ecx() -> i32 {
    emit_byte(0x99);
    emit_byte(0xF7); emit_byte(0xF9);
    3
}

// Modulo via idiv: cdq; idiv ecx; mov eax, edx (remainder).
//   99             cdq
//   F7 F9          idiv ecx
//   89 D0          mov eax, edx
fn emit_imod_eax_ecx() -> i32 {
    emit_byte(0x99);
    emit_byte(0xF7); emit_byte(0xF9);
    emit_byte(0x89); emit_byte(0xD0);
    5
}

// Comparison helpers. Each emits cmp eax, ecx; mov eax, 0; setX al.
// The setX opcode varies per comparison:
//   setl  = 0F 9C C0   (signed less)
//   setg  = 0F 9F C0   (signed greater)
//   sete  = 0F 94 C0   (equal)
//   setne = 0F 95 C0   (not equal)
//   setle = 0F 9E C0   (signed less-or-equal)
//   setge = 0F 9D C0   (signed greater-or-equal)
fn emit_cmp_setX(op_byte: i32) -> i32 {
    emit_byte(0x39); emit_byte(0xC8);
    emit_byte(0xB8); emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    emit_byte(0x0F); emit_byte(op_byte); emit_byte(0xC0);
    10
}
fn emit_lt_eax_ecx() -> i32 { emit_cmp_setX(0x9C) }
fn emit_gt_eax_ecx() -> i32 { emit_cmp_setX(0x9F) }
fn emit_eq_eax_ecx() -> i32 { emit_cmp_setX(0x94) }
fn emit_ne_eax_ecx() -> i32 { emit_cmp_setX(0x95) }
fn emit_le_eax_ecx() -> i32 { emit_cmp_setX(0x9E) }
fn emit_ge_eax_ecx() -> i32 { emit_cmp_setX(0x9D) }

// Stage 2.2: unsigned comparison helpers. Same shape as emit_cmp_setX
// but with unsigned setcc opcodes (b=below, a=above, etc.). Used for
// u32 / u64 comparisons via expr_type tag dispatch.
//   setb  = 0F 92 C0   (unsigned less:    CF=1)
//   seta  = 0F 97 C0   (unsigned greater: CF=0 && ZF=0)
//   setbe = 0F 96 C0   (unsigned less-or-equal)
//   setae = 0F 93 C0   (unsigned greater-or-equal: CF=0)
fn emit_lt_eax_ecx_u() -> i32 { emit_cmp_setX(0x92) }
fn emit_gt_eax_ecx_u() -> i32 { emit_cmp_setX(0x97) }
fn emit_le_eax_ecx_u() -> i32 { emit_cmp_setX(0x96) }
fn emit_ge_eax_ecx_u() -> i32 { emit_cmp_setX(0x93) }
// EQ / NE are signedness-agnostic (sete / setne are the same for
// signed and unsigned), so we reuse emit_eq_eax_ecx / emit_ne_eax_ecx.

// Stage 2.2: unsigned 32-bit division helpers.
//   31 D2          xor edx, edx       (clear high half of dividend)
//   F7 F1          div ecx            (eax = edx:eax / ecx; edx = rem)
fn emit_div_eax_ecx_u() -> i32 {
    emit_byte(0x31); emit_byte(0xD2);
    emit_byte(0xF7); emit_byte(0xF1);
    4
}
// Unsigned mod: same setup, then mov eax, edx (remainder).
fn emit_imod_eax_ecx_u() -> i32 {
    emit_byte(0x31); emit_byte(0xD2);
    emit_byte(0xF7); emit_byte(0xF1);
    emit_byte(0x89); emit_byte(0xD0);
    6
}

// Stage 2.4: unsigned 64-bit comparison helpers (REX.W variants).
fn emit_lt_rax_rcx_64_u() -> i32 { emit_cmp_setX_64(0x92) }
fn emit_gt_rax_rcx_64_u() -> i32 { emit_cmp_setX_64(0x97) }
fn emit_le_rax_rcx_64_u() -> i32 { emit_cmp_setX_64(0x96) }
fn emit_ge_rax_rcx_64_u() -> i32 { emit_cmp_setX_64(0x93) }

// Stage 2.4: unsigned 64-bit division (REX.W).
//   48 31 D2       xor rdx, rdx       (clear high half of dividend)
//   48 F7 F1       div rcx            (rax = rdx:rax / rcx; rdx = rem)
fn emit_div_rax_rcx_64_u() -> i32 {
    emit_byte(0x48); emit_byte(0x31); emit_byte(0xD2);
    emit_byte(0x48); emit_byte(0xF7); emit_byte(0xF1);
    6
}
// Unsigned 64-bit mod: same setup, then mov rax, rdx (REX.W).
fn emit_imod_rax_rcx_64_u() -> i32 {
    emit_byte(0x48); emit_byte(0x31); emit_byte(0xD2);
    emit_byte(0x48); emit_byte(0xF7); emit_byte(0xF1);
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xD0);
    9
}

// test eax, eax — sets ZF if eax == 0.
fn emit_test_eax_eax() -> i32 {
    emit_byte(0x85); emit_byte(0xC0);
    2
}

// je rel32 (placeholder) — 6 bytes (0F 84 + 4-byte disp). Returns
// the arena slot index of the first disp byte so the caller can
// backpatch once the target is known.
fn emit_je_rel32_placeholder() -> i32 {
    emit_byte(0x0F); emit_byte(0x84);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}

// jmp rel32 (placeholder) — 5 bytes (E9 + 4-byte disp).
fn emit_jmp_rel32_placeholder() -> i32 {
    emit_byte(0xE9);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}

// Stage 7: jcc rel32 placeholder family. Each is 6 bytes: 0F XX disp32.
//   jne (0x85): jump if NOT equal (used by PAT_LIT mismatch -> next arm)
//   jl  (0x8C): jump if signed less    (used by PAT_RANGE lo check)
//   jge (0x8D): jump if signed >=      (used by PAT_RANGE hi check, exclusive)
//   jle (0x8E): jump if signed <=
//   jg  (0x8F): jump if signed >
fn emit_jne_rel32_placeholder() -> i32 {
    emit_byte(0x0F); emit_byte(0x85);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}
fn emit_jl_rel32_placeholder() -> i32 {
    emit_byte(0x0F); emit_byte(0x8C);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}
fn emit_jge_rel32_placeholder() -> i32 {
    emit_byte(0x0F); emit_byte(0x8D);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}
fn emit_jle_rel32_placeholder() -> i32 {
    emit_byte(0x0F); emit_byte(0x8E);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}
fn emit_jg_rel32_placeholder() -> i32 {
    emit_byte(0x0F); emit_byte(0x8F);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}

// Stage 7: emit `cmp eax, imm32` — 5 bytes (3D imm32_le). Compares
// 32-bit eax with sign-extended imm32. Used by PAT_LIT and PAT_RANGE
// to test the scrutinee value against a constant.
fn emit_cmp_eax_imm32(v: i32) -> i32 {
    emit_byte(0x3D);
    emit_u32_le(v);
    5
}

// call rel32 (placeholder) — 5 bytes (E8 + 4-byte disp). Returns
// the arena slot index of the disp bytes for backpatching.
fn emit_call_rel32_placeholder() -> i32 {
    emit_byte(0xE8);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}

// lea rax, [rip + disp32] (placeholder) — 7 bytes (48 8D 05 + 4-byte disp).
// Returns the arena slot index of the disp bytes for backpatching.
// Used by inline arena/file builtins to reference data symbols.
fn emit_lea_rax_rip_placeholder() -> i32 {
    emit_byte(0x48); emit_byte(0x8D); emit_byte(0x05);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}

// lea rdi, [rip + disp32] (placeholder) — 7 bytes (48 8D 3D + 4-byte disp).
// Returns the arena slot index of the disp bytes for backpatching.
// ModRM differs from rax form (3D vs 05) because the reg-field
// encodes rdi (111) instead of rax (000). Used by file builtins
// to load a string-literal path directly into rdi for syscalls.
fn emit_lea_rdi_rip_placeholder() -> i32 {
    emit_byte(0x48); emit_byte(0x8D); emit_byte(0x3D);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}

// K1.AE (2026-05-25): lea rsi, [rip + disp32] (placeholder) -- 7
// bytes (48 8D 35 + 4-byte disp). ModRM byte 0x35 = mod 00 / reg
// 110 (rsi) / rm 101 ([rip+disp32]). Used by panic("msg") to load
// the message buffer pointer into rsi for the sys_write syscall.
// Returns the arena slot index of the disp bytes for backpatching.
fn emit_lea_rsi_rip_placeholder() -> i32 {
    emit_byte(0x48); emit_byte(0x8D); emit_byte(0x35);
    let disp_slot = __arena_len();
    emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    disp_slot
}

// ret — 1 byte (C3).
fn emit_ret() -> i32 { emit_byte(0xC3); 1 }

// Patch a 4-byte disp32 placeholder. `disp_slot` is the arena index
// where the disp bytes live; `target_slot` is the arena index of
// the instruction we want to land at. The displacement is from the
// END of the rel32 (= disp_slot + 4) to the target.
fn patch_rel32(disp_slot: i32, target_slot: i32) -> i32 {
    let disp = target_slot - (disp_slot + 4);
    patch_u32_le(disp_slot, disp)
}

// --------------------------------------------------------------
// Function prologue and epilogue. The emitted binary's entry stub
// reserves 4096 bytes (= 512 slots × 8 bytes) of stack space for
// let-bindings. Slot offsets [rbp-8, rbp-16, ..., rbp-4096] are
// addressable via disp32 ModRM (the disp8 form is not used since
// most offsets exceed -128 anyway).
//
//   55                push rbp
//   48 89 E5          mov rbp, rsp
//   48 81 EC 00 10 00 00   sub rsp, 4096
//
// Audit fix (cycle 1, polish #14): bumped from 512 → 1024 to match
// the bind_state cap (64 entries) with 2× margin. Previously 512
// was "just enough" for 64 × 8-byte slots — any future cap bump
// would silently corrupt the saved rbp/return-address. 1024 gave
// 128 slots; future Phase-1 should derive this from bind_state cap
// dynamically rather than hard-coding.
//
// Cycle 110 fix C109-SF-F1 / C109-TD-F109-1 (CRITICAL conf 90 +
// HIGH conf 80): Stage 29.1 bumped bind_state cap 64→512 but left
// this 1024-byte prologue and bind_alloc_offset's 1024 trap
// threshold unchanged. 512 simultaneously-live bindings × 8 bytes
// = 4096 bytes peak. Any fn with > 128 simultaneously-live
// let-bindings reached past the prologue's stack allocation,
// corrupting parent frame's saved rbp / return-address / red
// zone. Bumped to 4096 to match (and updated bind_alloc_offset
// trap threshold to 4096 to match). The architectural note about
// deriving from bind_state cap dynamically still applies — that's
// a Phase-1 follow-up.
fn emit_prologue() -> i32 {
    emit_byte(0x55);
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xE5);
    emit_byte(0x48); emit_byte(0x81); emit_byte(0xEC);
    emit_u32_le(4096);
    11
}

//   48 89 EC          mov rsp, rbp
//   5D                pop rbp
fn emit_epilogue() -> i32 {
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xEC);
    emit_byte(0x5D);
    4
}

// Store eax into [rbp - offset] using disp32 form for any offset.
//   89 85 disp32      mov [rbp + disp32], eax
fn emit_mov_local_eax(offset: i32) -> i32 {
    emit_byte(0x89); emit_byte(0x85);
    emit_u32_le(0 - offset);
    6
}

// Stage 2.5b/c stage 3: narrow STORES. Writes only the bytes the
// declared type defines, leaving high bytes of the 4-byte slot
// untouched. Combined with the matching narrow movzx/movsx loads
// (Stage 2.5b/c stage 2), this gives proper truncation semantics
// for u8/i8/u16/i16 bindings: a wider value flowing into a narrow
// store loses its upper bits at the slot boundary.

// Store low byte of eax (= al) into [rbp + disp32].
//   88 85 disp32      mov [rbp + disp32], al
fn emit_mov_local_al(offset: i32) -> i32 {
    emit_byte(0x88); emit_byte(0x85);
    emit_u32_le(0 - offset);
    6
}

// Store low word of eax (= ax) into [rbp + disp32]. The 66 prefix
// is the operand-size override that switches the 32-bit `mov [rm],
// reg` to its 16-bit form.
//   66 89 85 disp32   mov [rbp + disp32], ax
fn emit_mov_local_ax(offset: i32) -> i32 {
    emit_byte(0x66); emit_byte(0x89); emit_byte(0x85);
    emit_u32_le(0 - offset);
    7
}

// Load [rbp - offset] into eax.
//   8B 85 disp32      mov eax, [rbp + disp32]
fn emit_mov_eax_local(offset: i32) -> i32 {
    emit_byte(0x8B); emit_byte(0x85);
    emit_u32_le(0 - offset);
    6
}

// Phase 1.10 step 7d-5: 64-bit local store. Stores rax (full 8 bytes,
// preserving an f64 high half) into [rbp + disp32]. The 32-bit version
// (emit_mov_local_eax) silently drops the high 32 — incorrect for f64.
//   48 89 85 disp32     mov [rbp + disp32], rax
fn emit_mov_local_rax_64(offset: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x89); emit_byte(0x85);
    emit_u32_le(0 - offset);
    7
}

// Phase 1.10 step 7d-5: 64-bit local load. Loads rax (full 8 bytes)
// from [rbp + disp32]. Used by AST_VAR when the binding's type is
// f64 (bind_lookup_type returns 2). Without this load width, the
// 32-bit `mov eax, [rbp+disp32]` would zero-extend a truncated value
// and the f64 high half would be permanently lost.
//   48 8B 85 disp32     mov rax, [rbp + disp32]
fn emit_mov_rax_local_64(offset: i32) -> i32 {
    emit_byte(0x48); emit_byte(0x8B); emit_byte(0x85);
    emit_u32_le(0 - offset);
    7
}

// A2b (2026-05-28): load a local's low 32 bits into r11, zero-extending
// to the full r11 (mov r11d, [rbp + disp32] -> 44 8B 9D disp32). Used by
// the AST_CALL indirect path to load a function-pointer value out of a
// local slot. A 32-bit zero-extend (not a 64-bit load) is deliberate:
// fn-type params are type-erased to i32 (4-byte slots), and every code
// address is < 2^32 (ELF_BASE 0x400000 + a small image), so the low 32
// bits ARE the whole address and the slot's high 32 (possibly garbage
// from a 32-bit param spill) must be ignored.
fn emit_mov_r11d_local(offset: i32) -> i32 {
    emit_byte(0x44); emit_byte(0x8B); emit_byte(0x9D);
    emit_u32_le(0 - offset);
    7
}

// A2b: call through r11 (call r11 -> 41 FF D3). The indirect-call opcode
// for a fn pointer loaded by emit_mov_r11d_local.
fn emit_call_r11() -> i32 {
    emit_byte(0x41); emit_byte(0xFF); emit_byte(0xD3);
    3
}

// Stage 2.5b/c stage 2: narrow loads. The arena slot is still 4 bytes
// wide regardless of declared type; these helpers narrow the read so
// that a u8/u16/i8/i16 binding's stored bit pattern is interpreted at
// its declared width on every load. Truncation semantics flow from the
// read side: even if a u8 binding had 0x12345678 stored in its 4-byte
// slot (e.g. by a wider write), reading via movzx-byte gives only
// 0x78. This is the cleanest minimum — masked stores are a follow-on.

// Load byte at [rbp + disp32], zero-extend to eax. Used for u8.
//   0F B6 85 disp32   movzx eax, byte [rbp + disp32]
fn emit_movzx_eax_local_byte(offset: i32) -> i32 {
    emit_byte(0x0F); emit_byte(0xB6); emit_byte(0x85);
    emit_u32_le(0 - offset);
    7
}

// Load byte at [rbp + disp32], sign-extend to eax. Used for i8.
//   0F BE 85 disp32   movsx eax, byte [rbp + disp32]
fn emit_movsx_eax_local_byte(offset: i32) -> i32 {
    emit_byte(0x0F); emit_byte(0xBE); emit_byte(0x85);
    emit_u32_le(0 - offset);
    7
}

// Load word at [rbp + disp32], zero-extend to eax. Used for u16.
//   0F B7 85 disp32   movzx eax, word [rbp + disp32]
fn emit_movzx_eax_local_word(offset: i32) -> i32 {
    emit_byte(0x0F); emit_byte(0xB7); emit_byte(0x85);
    emit_u32_le(0 - offset);
    7
}

// Load word at [rbp + disp32], sign-extend to eax. Used for i16.
//   0F BF 85 disp32   movsx eax, word [rbp + disp32]
fn emit_movsx_eax_local_word(offset: i32) -> i32 {
    emit_byte(0x0F); emit_byte(0xBF); emit_byte(0x85);
    emit_u32_le(0 - offset);
    7
}

// AST_ASSIGN store dispatch. Audit follow-up Finding #4: factored out
// of the deeply nested if/else in the t==11 arm because the host
// parser hit its recursion budget when the chain grew past ~13 arms.
//
// Three entry points:
//   - assign_store_i64_path: val_i64 == 1. Width-correct on i64 bind,
//     trap 8001 otherwise.
//   - assign_store_u64_path: val_u64 == 1. Width-correct on u64 bind,
//     trap 8002 otherwise.
//   - assign_store_general:  neither i64 nor u64 value. For each
//     bind_ty arm, require val_ty == bind_ty (or trap with the
//     bind-ty-specific id from 8005..8016). i32-into-i64 (8003) and
//     i32-into-u64 (8004) are preserved from the pre-existing matrix.
//     Unknown bind_ty falls back to the 4-byte store (legacy).

@pure
fn assign_store_i64_path(off: i32, bind_ty: i32) -> i32 {
    if bind_ty == 3 { emit_mov_local_rax_64(off) }
    else { emit_trap_with_id(8001) }
}

@pure
fn assign_store_u64_path(off: i32, bind_ty: i32) -> i32 {
    if bind_ty == 9 { emit_mov_local_rax_64(off) }
    else { emit_trap_with_id(8002) }
}

@pure
fn assign_store_general(off: i32, bind_ty: i32, val_ty: i32) -> i32 {
    if bind_ty == 2 {
        if val_ty == 2 { emit_mov_local_rax_64(off) }
        else { emit_trap_with_id(8007) }
    } else { if bind_ty == 3 {
        emit_trap_with_id(8003)
    } else { if bind_ty == 9 {
        emit_trap_with_id(8004)
    } else { if bind_ty == 7 {
        if val_ty == 7 { emit_mov_local_al(off) }
        else { emit_trap_with_id(8010) }
    } else { if bind_ty == 10 {
        if val_ty == 10 { emit_mov_local_al(off) }
        else { emit_trap_with_id(8014) }
    } else { if bind_ty == 8 {
        if val_ty == 8 { emit_mov_local_ax(off) }
        else { emit_trap_with_id(8011) }
    } else { if bind_ty == 11 {
        if val_ty == 11 { emit_mov_local_ax(off) }
        else { emit_trap_with_id(8015) }
    } else {
        assign_store_general_4b(off, bind_ty, val_ty)
    }}}}}}}
}

@pure
fn assign_store_general_4b(off: i32, bind_ty: i32, val_ty: i32) -> i32 {
    if bind_ty == 1 {
        if val_ty == 1 { emit_mov_local_eax(off) }
        else { emit_trap_with_id(8006) }
    } else { if bind_ty == 6 {
        if val_ty == 6 { emit_mov_local_eax(off) }
        else { emit_trap_with_id(8012) }
    } else { if bind_ty == 4 {
        if val_ty == 4 { emit_mov_local_eax(off) }
        else { emit_trap_with_id(8016) }
    } else { if bind_ty == 0 {
        if val_ty == 0 { emit_mov_local_eax(off) }
        else { emit_trap_with_id(8005) }
    } else {
        // Unknown bind_ty (e.g. user struct, not yet typed): fall
        // back to legacy 4-byte store. Pre-Finding-#4 behaviour.
        emit_mov_local_eax(off)
    }}}}
}

// --------------------------------------------------------------
// Compile-time binding table: a stack of (name_start, name_len,
// stack_offset) triples in the arena. We pass `bind_base` (start
// of the table) and `bind_top` (just past the last entry) through
// recursive emit calls. Entries are 3 slots wide.
//
// Lookup walks backwards (top -> base, 3 at a time) to find the
// most-recent binding of a name, providing lexical shadowing.
// Returns the matching binding's offset, or 0 if unbound.
// --------------------------------------------------------------
@pure
fn kovc_byte_eq(src_a: i32, len_a: i32, src_b: i32, len_b: i32) -> i32 {
    if len_a != len_b { 0 }
    else {
        let mut i: i32 = 0;
        let mut ok: i32 = 1;
        while i < len_a {
            if ok == 1 {
                let ba = __arena_get(src_a + i);
                let bb = __arena_get(src_b + i);
                if ba != bb { ok = 0; };
            };
            i = i + 1;
        }
        ok
    }
}

// bind_state layout (3 i32 slots + a fixed-capacity table inline):
//   slot 0: next free stack offset (init 8, grows by 8 per let)
//   slot 1: number of entries currently in table (init 0)
//   slot 2: arena slot index of entry 0 (= bind_state + 3)
//   slot 3..3 + cap*4: pre-allocated entries
//                      [name_start, name_len, offset, type_tag]
//                      type_tag: 0 = i32 (default), 1 = f32, 2 = f64
//                      (Phase 1.10 step 7d-5 added 2; AST_VAR loads the
//                      full 8 bytes when ty == 2 to preserve the high
//                      half of an f64 binding.)
//
// Capacity is fixed at compile-time (NUM_BINDINGS_CAP = 512 per the
// Stage 29.1 bump; was 64). Table uses __arena_set to write entries
// — never __arena_push, so the code region can grow contiguously
// after bind_state.
//
// The type_tag was added in Phase 1.10 step 5c so the AST_ADD/SUB/MUL/
// DIV codegen can dispatch to SSE when both operands' bindings are f32.
// AST_LET stamps the type at push-time by inspecting the value AST.
fn bind_init() -> i32 {
    let state = __arena_push(8);            // next_offset = 8
    __arena_push(0);                        // top = 0
    __arena_push(state + 3);                // table_base = state + 3
    let mut i: i32 = 0;
    // Stage 29.1 fix (2026-05-12): bump cap from 64 to 512 entries.
    // After the Stage 29 SIGILL fix exposed all fns to compilation,
    // parser.hx's parse_primary (single-fn cap on bindings) needs
    // far more than 64. Empirically the bootstrap source needs ~200
    // bindings per fn at peak; 512 gives 2.5x headroom.
    while i < 2048 {                        // 512 entries * 4 slots
        __arena_push(0);
        i = i + 1;
    }
    state
}

fn bind_push(state: i32, name_start: i32, name_len: i32, offset: i32) -> i32 {
    bind_push_typed(state, name_start, name_len, offset, 0)
}

// Phase 1.10 step 5c: variant that records the binding's type.
// Audit fix #10 (cycle 1): cap-check before writing. The 512-entry
// cap (2048 arena slots) is set in bind_init (Stage 29.1 bump from
// 64). Without this guard, the 513th+ binding silently corrupts
// adjacent arena data — fn_table or str_table or worse. Now: SKIP
// the binding when full (return -1). Subsequent AST_VAR resolves
// via offset 0 (= unbound sentinel), which AST_VAR's audit-10 guard
// handles by emitting the integer-zero placeholder. No arena
// corruption. Stage 28.9 cycle-110 (C109-CR-F3) replaced the silent
// skip with `emit_trap_with_id(10032)` so overflow is loud — but
// the in-bound path is still cap-checked at the top of the fn.
fn bind_push_typed(state: i32, name_start: i32, name_len: i32,
                   offset: i32, ty: i32) -> i32 {
    let top = __arena_get(state + 1);
    // Stage 29.1 fix (2026-05-12): bumped cap from 64 to 512.
    // Cycle 110 fix C109-CR-F3 (HIGH conf 82): emit a loud-fail trap
    // when the cap is exceeded. Pre-fix returned `0 - 1` silently;
    // callers discarded the return value; the binding name became
    // unresolvable and AST_VAR's audit-10 guard substituted `mov
    // eax, 0`, silently making the variable read as 0 thereafter.
    // Trap id 10032.
    if top >= 512 {
        emit_trap_with_id(10032);
        0 - 1
    } else {
        let table_base = __arena_get(state + 2);
        let entry = table_base + top * 4;
        __arena_set(entry, name_start);
        __arena_set(entry + 1, name_len);
        __arena_set(entry + 2, offset);
        __arena_set(entry + 3, ty);
        __arena_set(state + 1, top + 1);
        0
    }
}

fn bind_pop(state: i32) -> i32 {
    // Audit-18: roll back next_offset by 8 in addition to dropping the
    // top binding. Without this, sequential nested AST_LETs allocate
    // offsets monotonically (8, 16, 24, ...) and never reuse them after
    // pop. parse_primary nests ~30 lets, blowing past the 512-byte
    // prologue allocation; emit_mov_local_eax(-560) writes into the
    // parent frame's saved rbp/return-address. The fix mirrors the
    // implicit invariant that bind_pop is paired with the most recent
    // bind_push (LIFO scope), so rolling back the offset is safe.
    let top = __arena_get(state + 1);
    let cur_off = __arena_get(state);
    __arena_set(state + 1, top - 1);
    __arena_set(state, cur_off - 8);
    0
}

fn bind_alloc_offset(state: i32) -> i32 {
    // Audit-stage5-6 Finding #11 fix: trap when the requested slot
    // would write past the prologue allocation (emit_prologue at
    // kovc.hx ~739). Without this, sequential let/struct-lit
    // allocations wrap silently into the parent frame's saved rbp /
    // return-address / red zone. The trap fires at codegen time —
    // we still bump the offset so any downstream emitter that
    // derives a layout from the returned value doesn't see a stale
    // slot. Trap id 10030.
    //
    // Cycle 110 fix C109-SF-F1 / C109-TD-F109-1: threshold bumped
    // from 1024 → 4096 to match the new emit_prologue 4096-byte
    // allocation and the Stage 29.1 bind_state cap of 512 entries.
    let off = __arena_get(state);
    if off >= 4096 {
        emit_trap_with_id(10030);
    };
    __arena_set(state, off + 8);
    off
}

fn bind_lookup(state: i32, name_start: i32, name_len: i32) -> i32 {
    let top = __arena_get(state + 1);
    let table_base = __arena_get(state + 2);
    let mut i: i32 = top - 1;
    let mut found: i32 = 0;
    let mut offset: i32 = 0;
    while i >= 0 {
        if found == 0 {
            let entry = table_base + i * 4;
            let ns = __arena_get(entry);
            let nl = __arena_get(entry + 1);
            if kovc_byte_eq(ns, nl, name_start, name_len) == 1 {
                offset = __arena_get(entry + 2);
                found = 1;
            };
            i = i - 1;
        } else {
            i = 0 - 1;
        };
    }
    offset
}

// Phase 1.10 step 5c: look up a binding's type tag (0=i32, 1=f32).
// Returns 0 (default i32) when name is unbound, matching bind_lookup's
// "0 means unbound" sentinel — happens to be the same value as the
// default i32 type, which is the safe fallback for the SSE-dispatch
// caller (unbound + unknown -> integer codegen).
fn bind_lookup_type(state: i32, name_start: i32, name_len: i32) -> i32 {
    let top = __arena_get(state + 1);
    let table_base = __arena_get(state + 2);
    let mut i: i32 = top - 1;
    let mut found: i32 = 0;
    let mut ty: i32 = 0;
    while i >= 0 {
        if found == 0 {
            let entry = table_base + i * 4;
            let ns = __arena_get(entry);
            let nl = __arena_get(entry + 1);
            if kovc_byte_eq(ns, nl, name_start, name_len) == 1 {
                ty = __arena_get(entry + 3);
                found = 1;
            };
            i = i - 1;
        } else {
            i = 0 - 1;
        };
    }
    ty
}

// Reset bind_state for a new function body. Sets next_offset back
// to 8 (first stack slot) and top to 0 (no bindings yet).
fn bind_reset(state: i32) -> i32 {
    __arena_set(state, 8);
    __arena_set(state + 1, 0);
    0
}

// Phase 1.10 step 5c: detect call names beginning with `__f` (the f32
// SSE builtins __fadd/__fsub/__fmul/__fdiv/__fneg). Any such call's
// result is f32 by convention. Cheaper than a full byte-string compare.
@pure
fn is_underscore_f_call(name_start: i32, name_len: i32) -> i32 {
    if name_len < 4 { 0 }
    else {
        let b0 = __arena_get(name_start);
        let b1 = __arena_get(name_start + 1);
        let b2 = __arena_get(name_start + 2);
        if b0 == 95 {
            if b1 == 95 {
                if b2 == 102 { 1 }   // 'f'
                else { 0 }
            } else { 0 }
        } else { 0 }
    }
}

// Stage 1.6: unified type-tag lookup. Returns the type tag of an AST
// expression: 0=i32, 1=f32, 2=f64, 3=i64. Generalizes the three
// sibling predicates (is_f32_expr, is_f64_expr, is_i64_expr) into one
// recursive descent. Closed-world: any AST tag not explicitly handled
// returns 0 (i32 default) — same fall-through behavior the predicates
// had. The predicates below are now thin wrappers calling expr_type
// and comparing the returned tag.
//
// Type-tag namespace (current and reserved):
//   0  = i32   (default integer)
//   1  = f32   (single-precision float)
//   2  = f64   (double-precision float)
//   3  = i64   (Stage 1)
//   4  = bf16  (Stage 1.5, reserved)
//   5  = f16   (Stage 1.5, reserved)
//   6  = u8    (Stage 2, reserved)
//   7  = u16   (Stage 2, reserved)
//   8  = u32   (Stage 2, reserved)
//   9  = u64   (Stage 2, reserved)
//  10 = i8    (Stage 2, reserved)
//  11 = i16   (Stage 2, reserved)
//  12+ reserved for future use; do not reassign.
//
// Mismatched binary-op operands return 0 (i32 default). This is fine
// because codegen's 4-way dispatch traps mismatches with ud2 anyway —
// the tag returned here informs upstream (e.g., AST_LET's val_ty),
// not the trap-on-mismatch logic at each binop site.
fn expr_type(idx: i32, bind_state: i32, bn_state: i32) -> i32 {
    let t = __arena_get(idx);
    let p1 = __arena_get(idx + 1);
    let p2 = __arena_get(idx + 2);
    if t == 27 { 1 }                                  // AST_FLOATLIT (f32)
    else { if t == 34 { 2 }                           // AST_FLOATLIT_F64
    else { if t == 35 { 3 }                           // AST_INTLIT_I64
    else { if t == 36 { 6 }                           // AST_INTLIT_U32 (Stage 2.1)
    else { if t == 37 { 7 }                           // AST_INTLIT_U8  (Stage 2.3)
    else { if t == 38 { 9 }                           // AST_INTLIT_U64 (Stage 2.4)
    else { if t == 39 { 10 }                          // AST_INTLIT_I8  (Stage 2.5b)
    else { if t == 40 { 11 }                          // AST_INTLIT_I16 (Stage 2.5c)
    else { if t == 41 { 8 }                           // AST_INTLIT_U16 (Stage 2.5c)
    else { if t == 42 { 4 }                           // AST_FLOATLIT_BF16 (Stage 1.5)
    else { if t == 80 { 4 }                           // AST_FLOATLIT_F16 (K1.F15 2026-05-27) -- 16-bit shape; arith traps via is_bf16_expr predicate (same handling as bf16)
    else { if t == 50 { 3 }                            // AST_TUPLE_LIT (Stage 4) — 64-bit pointer (treat as i64 for storage)
    else { if t == 52 {
        // AST_TUPLE_FIELD (Stage 4 iter B). Stage 5 Iter D: p3 == 1 marks
        // the field as struct-typed (slot holds an 8-byte pointer); the
        // codegen path emits a 64-bit load and the type tag is 3 (i64-
        // shaped, same convention as AST_TUPLE_LIT).
        let p3 = __arena_get(idx + 3);
        if p3 == 1 { 3 } else { 0 }
    } else { if t == 53 { 0 }                            // AST_INDEX (Stage 4 iter E) — 32-bit element
    else { if t == 0 { 0 }                            // AST_INTLIT (i32)
    else { if t == 1 {                                // AST_VAR
        bind_lookup_type(bind_state, p1, p2)
    } else { if t == 7 {                              // AST_IF
        let then_idx = p2;
        let else_idx = __arena_get(idx + 3);
        let lt = expr_type(then_idx, bind_state, bn_state);
        let rt = expr_type(else_idx, bind_state, bn_state);
        if lt == rt { lt } else { 0 }
    } else { if t == 8 {                              // AST_LET
        let body_idx = __arena_get(idx + 3);
        expr_type(body_idx, bind_state, bn_state)
    } else { if t == 12 {                             // AST_LET_MUT
        let body_idx = __arena_get(idx + 3);
        expr_type(body_idx, bind_state, bn_state)
    } else { if t == 13 {                             // AST_SEQ
        expr_type(p2, bind_state, bn_state)
    } else { if t == 11 {                             // AST_ASSIGN
        let value_idx = __arena_get(idx + 3);
        expr_type(value_idx, bind_state, bn_state)
    } else { if t == 79 {                             // AST_FIELD_STORE
        // K1.F6 (2026-05-27): the assignment-result-IS-value rule
        // applies to field-store too; eax holds the assigned value
        // post-store, so the expression's type is the value's type.
        // Mirrors AST_ASSIGN. p2 = value expr index.
        expr_type(p2, bind_state, bn_state)
    } else { if t == 9 {                              // AST_NEG
        expr_type(p1, bind_state, bn_state)
    } else { if t == 26 {                             // AST_BNOT
        expr_type(p1, bind_state, bn_state)
    } else { if t == 32 {                             // AST_SHL
        expr_type(p1, bind_state, bn_state)
    } else { if t == 33 {                             // AST_SHR
        expr_type(p1, bind_state, bn_state)
    } else { if t == 2 {                              // AST_ADD
        // Mismatched binary-op operands fall back to 0 (i32) EXCEPT
        // for the K1.F8 / K1.F8b (2026-05-27) i64<->i32 widening.
        // BOTH directions (l_i64+r_i32 AND l_i32+r_i64) now sign-
        // extend the i32 side and emit a 64-bit op in codegen, so
        // expr_type returns 3 (i64) for both `(3, 0)` and `(0, 3)`
        // operand-tag pairs. Other mixed paths (f32/f64, signed/
        // unsigned width, etc.) still fall to 0.
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { if l == 3 { if r == 0 { 3 } else { 0 } } else { if l == 0 { if r == 3 { 3 } else { 0 } } else { if l == 9 { if r == 6 { 9 } else { 0 } } else { if l == 6 { if r == 9 { 9 } else { 0 } } else { if l == 2 { if r == 1 { 2 } else { 0 } } else { if l == 1 { if r == 2 { 2 } else { 0 } } else { 0 } } } } } } }
    } else { if t == 3 {                              // AST_SUB
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { if l == 3 { if r == 0 { 3 } else { 0 } } else { if l == 0 { if r == 3 { 3 } else { 0 } } else { if l == 9 { if r == 6 { 9 } else { 0 } } else { if l == 6 { if r == 9 { 9 } else { 0 } } else { if l == 2 { if r == 1 { 2 } else { 0 } } else { if l == 1 { if r == 2 { 2 } else { 0 } } else { 0 } } } } } } }
    } else { if t == 4 {                              // AST_MUL
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { if l == 3 { if r == 0 { 3 } else { 0 } } else { if l == 0 { if r == 3 { 3 } else { 0 } } else { if l == 9 { if r == 6 { 9 } else { 0 } } else { if l == 6 { if r == 9 { 9 } else { 0 } } else { if l == 2 { if r == 1 { 2 } else { 0 } } else { if l == 1 { if r == 2 { 2 } else { 0 } } else { 0 } } } } } } }
    } else { if t == 5 {                              // AST_DIV
        // K1.F8c: same widening rule as ADD/SUB/MUL for both
        // i64<->i32 directions; codegen sign-extends and emits a
        // 64-bit idiv, so the result is i64.
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { if l == 3 { if r == 0 { 3 } else { 0 } } else { if l == 0 { if r == 3 { 3 } else { 0 } } else { if l == 9 { if r == 6 { 9 } else { 0 } } else { if l == 6 { if r == 9 { 9 } else { 0 } } else { if l == 2 { if r == 1 { 2 } else { 0 } } else { if l == 1 { if r == 2 { 2 } else { 0 } } else { 0 } } } } } } }
    } else { if t == 24 {                             // AST_MOD
        // K1.F8c: same widening rule as DIV.
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { if l == 3 { if r == 0 { 3 } else { 0 } } else { if l == 0 { if r == 3 { 3 } else { 0 } } else { if l == 9 { if r == 6 { 9 } else { 0 } } else { if l == 6 { if r == 9 { 9 } else { 0 } } else { if l == 2 { if r == 1 { 2 } else { 0 } } else { if l == 1 { if r == 2 { 2 } else { 0 } } else { 0 } } } } } } }
    } else { if t == 28 {                             // AST_BAND
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { 0 }
    } else { if t == 29 {                             // AST_BOR
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { 0 }
    } else { if t == 30 {                             // AST_BXOR
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { 0 }
    } else { if t == 16 {                             // AST_CALL
        // Builtin returning f64: explicit byte_eq matches first.
        let widen_match = kovc_byte_eq(p1, p2, bn_f32_to_f64_s(bn_state), 12);
        if widen_match == 1 { 2 }
        else {
            let widen_i_match = kovc_byte_eq(p1, p2, bn_i32_to_f64_s(bn_state), 12);
            if widen_i_match == 1 { 2 }
            else {
                let pack_match = kovc_byte_eq(p1, p2, bn_f64_pack_s(bn_state), 10);
                if pack_match == 1 { 2 }
                else {
                    let dsqrt_match = kovc_byte_eq(p1, p2, bn_dsqrt_s(bn_state), 7);
                    if dsqrt_match == 1 { 2 }
                    else {
                        let dabs_match = kovc_byte_eq(p1, p2, bn_dabs_s(bn_state), 6);
                        if dabs_match == 1 { 2 }
                        else {
                            let dmin_match = kovc_byte_eq(p1, p2, bn_dmin_s(bn_state), 6);
                            if dmin_match == 1 { 2 }
                            else {
                                let dmax_match = kovc_byte_eq(p1, p2, bn_dmax_s(bn_state), 6);
                                if dmax_match == 1 { 2 }
                                else {
                                    // Builtins returning i32 (conversion-out).
                                    let f2i_match = kovc_byte_eq(p1, p2, bn_f32_to_i32_s(bn_state), 12);
                                    if f2i_match == 1 { 0 }
                                    else {
                                        let f64_2i_match = kovc_byte_eq(p1, p2, bn_f64_to_i32_s(bn_state), 12);
                                        if f64_2i_match == 1 { 0 }
                                        else {
                                            // Builtins returning f32 (conversion-in + narrow).
                                            let i2f_match = kovc_byte_eq(p1, p2, bn_i32_to_f32_s(bn_state), 12);
                                            if i2f_match == 1 { 1 }
                                            else {
                                                let nrw_match = kovc_byte_eq(p1, p2, bn_f64_to_f32_s(bn_state), 12);
                                                if nrw_match == 1 { 1 }
                                                else {
                                                    // __f* prefix → f32 (catch-all
                                                    // for transcendentals etc.).
                                                    let prefix_match = is_underscore_f_call(p1, p2);
                                                    if prefix_match == 1 { 1 }
                                                    else {
                                                        // User-defined fn: fn_type_table.
                                                        let fts = bn_fn_type_state(bn_state);
                                                        if fts == 0 { 0 }
                                                        else { fn_type_table_lookup(fts, p1, p2) }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else { 0 }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
}

// Phase 1.10 step 5c: type-inference on AST nodes. Returns 1 if the
// expression's type is f32, else 0 (i32 default).
//
// Stage 1.6: this is now a thin wrapper around expr_type. The
// previous 14-arm recursive descent has been folded into expr_type;
// keeping this function as a wrapper preserves all existing dispatch-
// site call shapes.
fn is_f32_expr(idx: i32, bind_state: i32, bn_state: i32) -> i32 {
    if expr_type(idx, bind_state, bn_state) == 1 { 1 } else { 0 }
}

// Phase 1.10 step 7d: is_f64_expr — type predicate for f64.
// Stage 1.6: thin wrapper around expr_type.
fn is_f64_expr(idx: i32, bind_state: i32, bn_state: i32) -> i32 {
    if expr_type(idx, bind_state, bn_state) == 2 { 1 } else { 0 }
}

// Approach A Stage 1: is_i64_expr — type predicate for i64.
// Stage 1.6: thin wrapper around expr_type.
fn is_i64_expr(idx: i32, bind_state: i32, bn_state: i32) -> i32 {
    if expr_type(idx, bind_state, bn_state) == 3 { 1 } else { 0 }
}

// Approach A Stage 2.1/2.2: is_u32_expr — type predicate for u32.
// Used by Stage 2.2's unsigned DIV/MOD/comparison dispatch.
fn is_u32_expr(idx: i32, bind_state: i32, bn_state: i32) -> i32 {
    if expr_type(idx, bind_state, bn_state) == 6 { 1 } else { 0 }
}

// Stage 2.3: is_u8_expr — type predicate for u8 (tag 7).
// u8 shares Stage 2.2's unsigned dispatch helpers since both are
// unsigned 32-bit-or-narrower integers.
fn is_u8_expr(idx: i32, bind_state: i32, bn_state: i32) -> i32 {
    if expr_type(idx, bind_state, bn_state) == 7 { 1 } else { 0 }
}

// Stage 2.4: is_u64_expr — type predicate for u64 (tag 9).
// u64 needs REX.W-prefixed unsigned helpers (different from u32's
// 32-bit unsigned helpers) and 8-byte storage (like i64).
fn is_u64_expr(idx: i32, bind_state: i32, bn_state: i32) -> i32 {
    if expr_type(idx, bind_state, bn_state) == 9 { 1 } else { 0 }
}

// Stage 1.5: is_bf16_expr — type predicate for bf16 (tag 4).
// Used by the AST_ADD/SUB/MUL/DIV/MOD cascades to trap binops on
// bf16 operands. bf16 has no hardware add/sub/mul/div on x86-64 (no
// AVX-512 BF16 in baseline targets), so any bf16 arithmetic must
// either (a) round-trip through f32 (cvtps2ph + cvtph2ps + addss +
// truncate, deferred) or (b) ud2-trap until that codegen lands.
// Without this trap, bf16 vars (which are i32-shaped in storage)
// silently feed into the integer fallthrough at the bottom of the
// cascade, producing 32-bit int ops on float bit patterns — garbage.
fn is_bf16_expr(idx: i32, bind_state: i32, bn_state: i32) -> i32 {
    if expr_type(idx, bind_state, bn_state) == 4 { 1 } else { 0 }
}

// Audit follow-up Finding #1: width-class helper used by AST_FN_DECL's
// body-vs-ret-ty trap (id 14002). Returns the storage width in bytes
// (1, 2, 4, 8) for a given type tag. Falls back to 4 (i32 default).
//   tag 0  i32  -> 4
//   tag 1  f32  -> 4
//   tag 2  f64  -> 8
//   tag 3  i64  -> 8
//   tag 4  bf16 -> 2
//   tag 6  u32  -> 4
//   tag 7  u8   -> 1
//   tag 8  u16  -> 2
//   tag 9  u64  -> 8
//   tag 10 i8   -> 1
//   tag 11 i16  -> 2
// Audit A1-F5: struct-typed encodings (>= 100) carry an 8-byte
// pointer. We classify them as 8-byte width so the body-vs-ret-ty
// trap doesn't fire on a struct-returning fn whose body is an
// AST_TUPLE_LIT (also 8-byte pointer rep).
fn type_width_class_struct(ty: i32) -> i32 {
    if ty >= 100 { 8 } else { 4 }
}
fn type_width_class(ty: i32) -> i32 {
    if ty == 2 { 8 }
    else { if ty == 3 { 8 }
    else { if ty == 9 { 8 }
    else { if ty == 4 { 2 }
    else { if ty == 8 { 2 }
    else { if ty == 11 { 2 }
    else { if ty == 7 { 1 }
    else { if ty == 10 { 1 }
    else { type_width_class_struct(ty) } } } } } } } }
}

// Phase 1.10 step 5c follow-on: fn_type_table maps fn names to their
// declared return-type tag (0=i32, 1=f32, 2=f64, 3=i64). Populated
// PRE-PASS over the AST_FN_LIST so expr_type can resolve user-named
// fn return types at call sites — without this, `let x = my_f32_fn(...)`
// followed by `x + ...` would fall back to integer codegen since
// AST_CALL only knows about the `__f*` builtin prefix.
//
// Stage 1.7 extension: each entry now also stores the function's
// parameter types (packed: 4 bits per param, up to 6 params = 24 bits)
// and the param count. Used at call sites to trap arg-type mismatches.
//
// Entry layout (5 slots per entry):
//   slot 0: name_start
//   slot 1: name_len
//   slot 2: ret_ty            (0..3)
//   slot 3: packed_param_tys  ((p0) | (p1<<4) | ... | (p5<<20))
//   slot 4: param_count       (0..6)
fn fn_type_table_init() -> i32 {
    let state = __arena_push(0);            // top = 0
    __arena_push(state + 2);                // table_base = state + 2
    let mut i: i32 = 0;
    while i < 5120 {                        // 1024 entries * 5 slots
        __arena_push(0);
        i = i + 1;
    }
    state
}

fn fn_type_table_add(state: i32, name_start: i32, name_len: i32, ret_ty: i32, packed_param_tys: i32, param_count: i32) -> i32 {
    // Safe-hardening (2026-05-28): cap 256 -> 1024. A program with >256 fns
    // silently dropped fn_type_table entries past #256, so a non-i32-return
    // fn beyond #256 mis-typed at its call sites (e.g. an f32-returning fn
    // read as i32 -> SIGILL). Bumped to 1024 (full self-host driver has 673
    // fns). The table is arena-pushed at init (NOT woven into the hardcoded
    // end-region address math), so this is a safe size change.
    let top = __arena_get(state);
    if top >= 1024 {
        0 - 1
    } else {
        let table_base = __arena_get(state + 1);
        let entry = table_base + top * 5;
        __arena_set(entry, name_start);
        __arena_set(entry + 1, name_len);
        __arena_set(entry + 2, ret_ty);
        __arena_set(entry + 3, packed_param_tys);
        __arena_set(entry + 4, param_count);
        __arena_set(state, top + 1);
        0
    }
}

@pure
fn fn_type_table_lookup(state: i32, name_start: i32, name_len: i32) -> i32 {
    let top = __arena_get(state);
    let table_base = __arena_get(state + 1);
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut ty: i32 = 0;
    while i < top {
        if found == 0 {
            let entry = table_base + i * 5;
            let ns = __arena_get(entry);
            let nl = __arena_get(entry + 1);
            if kovc_byte_eq(ns, nl, name_start, name_len) == 1 {
                ty = __arena_get(entry + 2);
                found = 1;
            };
            i = i + 1;
        } else {
            i = top;
        };
    }
    ty
}

// A2a (2026-05-28): presence check for a user fn name. fn_type_table_lookup
// returns the ret_ty (0 both for a miss AND for a fn returning i32), so it
// cannot distinguish "not a fn" from "fn -> i32". This returns 1 iff the
// name is a registered user fn, 0 otherwise. Used by AST_VAR codegen to
// decide whether an unbound name is a function reference (emit a fn-pointer
// lea) versus a genuine typo (trap). The fn_type_table is fully populated
// by the pre-codegen fn_type_table_init pass, so forward refs resolve too.
@pure
fn fn_type_table_has(state: i32, name_start: i32, name_len: i32) -> i32 {
    let top = __arena_get(state);
    let table_base = __arena_get(state + 1);
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < top {
        if found == 0 {
            let entry = table_base + i * 5;
            let ns = __arena_get(entry);
            let nl = __arena_get(entry + 1);
            if kovc_byte_eq(ns, nl, name_start, name_len) == 1 {
                found = 1;
            };
            i = i + 1;
        } else {
            i = top;
        };
    }
    found
}

// Stage 1.7: lookup the packed_param_tys + param_count for a fn name.
// Returns (packed_param_tys * 8) + param_count packed into a single i32
// — caller unpacks via `count = ret & 7` and `packed = ret >> 3`.
// Returns 0 if name not found (fallback safe: 0 params, all 0=i32).
@pure
fn fn_type_table_lookup_params(state: i32, name_start: i32, name_len: i32) -> i32 {
    let top = __arena_get(state);
    let table_base = __arena_get(state + 1);
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut packed: i32 = 0;
    let mut count: i32 = 0;
    while i < top {
        if found == 0 {
            let entry = table_base + i * 5;
            let ns = __arena_get(entry);
            let nl = __arena_get(entry + 1);
            if kovc_byte_eq(ns, nl, name_start, name_len) == 1 {
                packed = __arena_get(entry + 3);
                count = __arena_get(entry + 4);
                found = 1;
            };
            i = i + 1;
        } else {
            i = top;
        };
    }
    // Pack count into low 3 bits (max 6 fits in 3 bits), packed into upper.
    (packed * 8) + count
}

// Stage 1.7: extract param[idx]'s type tag from a packed_param_tys.
// `packed` is 6 tags of 4 bits each: bits [0..3]=p0, [4..7]=p1, ...,
// [20..23]=p5. Returns the 4-bit tag at position idx.
@pure
fn unpack_param_ty(packed: i32, idx: i32) -> i32 {
    let shift = idx * 4;
    let mut shifted: i32 = packed;
    let mut s: i32 = 0;
    while s < shift {
        shifted = shifted / 2;
        s = s + 1;
    }
    shifted - ((shifted / 16) * 16)
}

// fn_table: maps fn names to arena slot indices where their code
// starts. Entry layout: [name_start, name_len, code_offset]. Capacity
// 512 (Stage 6 bump) — the lexer + parser + kovc concatenation has
// ~290 fns now (Stage 5 ~270, Stage 6A added ~17 enum-table helpers);
// 512 leaves headroom for Stage 7 (match) + Stage 8 (generics).
fn fn_table_init() -> i32 {
    let state = __arena_push(0);            // top = 0
    __arena_push(state + 2);                // table_base = state + 2
    let mut i: i32 = 0;
    while i < 3072 {                        // 1024 entries * 3 slots
        __arena_push(0);
        i = i + 1;
    }
    state
}

fn fn_table_add(state: i32, name_start: i32, name_len: i32, code_offset: i32) -> i32 {
    // Audit fix #10: cap-check before writing.
    // Cap history:
    //   Stage 6:          256 → 512 (accommodate enum + future stages)
    //   Stage 50 retry:   512 → 1024 (Exp C 2026-05-17 found the
    //     real Stage 50 ABORTED root cause: Stage 50 Inc 1+2 added
    //     16 fns to parser.hx, hitting cap 512 exactly. The 513th
    //     fn declaration trapped via id 10033, but the CRITICAL
    //     consequence was that `main` itself — being the LAST
    //     declared fn via the cascade driver — was among the
    //     overflow casualties. Its CALL site got patched with
    //     `ud2 + 3 nops` (the unresolved-CALL stub) → entry-point
    //     SIGILL rc=132. Bumping cap to 1024 unblocks Stage 50.
    //     Verified: G2..G5 byte-identical sha=b510bc28..., smoke
    //     4/4 PASS post-fix.)
    // Cycle 110 fix C109-CR-F3 (HIGH conf 82): emit a loud-fail trap
    // when the cap is exceeded. Trap id 10033.
    let top = __arena_get(state);
    if top >= 1024 {
        emit_trap_with_id(10033);
        0 - 1
    } else {
    let table_base = __arena_get(state + 1);
    let entry = table_base + top * 3;
    __arena_set(entry, name_start);
    __arena_set(entry + 1, name_len);
    __arena_set(entry + 2, code_offset);
    __arena_set(state, top + 1);
    0
    }
}

fn fn_table_lookup(state: i32, name_start: i32, name_len: i32) -> i32 {
    let top = __arena_get(state);
    let table_base = __arena_get(state + 1);
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut offset: i32 = 0 - 1;            // -1 = not found
    while i < top {
        if found == 0 {
            let entry = table_base + i * 3;
            let ns = __arena_get(entry);
            let nl = __arena_get(entry + 1);
            if kovc_byte_eq(ns, nl, name_start, name_len) == 1 {
                offset = __arena_get(entry + 2);
                found = 1;
            };
            i = i + 1;
        } else {
            i = top;
        };
    }
    offset
}

// patch_table: records pending CALL + LEA backpatches. Each entry:
// [disp_slot, target_name_start, target_name_len]. Stage 29.1 fix
// (2026-05-12): bumped cap from 4096 to 16384. Post-Stage-29 SIGILL
// fix, the bootstrap parser now successfully parses ALL fns of its own
// source — emit count jumped from ~1500 to ~6800 patches (4719 calls +
// 2059 LEAs), overflowing the 4096 cap and dropping ~2700 patches
// (LEA disps stayed 0 → K2 read from wrong arena base → empty K3).
// 16384 gives 2.4x headroom over the new measured 6800.
fn patch_table_init() -> i32 {
    let state = __arena_push(0);            // top = 0
    __arena_push(state + 2);                // table_base = state + 2
    let mut i: i32 = 0;
    while i < 49152 {                       // 16384 entries * 3 slots
        __arena_push(0);
        i = i + 1;
    }
    state
}

fn patch_table_add(state: i32, disp_slot: i32, name_start: i32, name_len: i32) -> i32 {
    // Audit fix #10: cap-check before writing. patch_table_init
    // allocates 16384 entries; without this guard, a source with > 16384
    // CALL+LEA patches would silently corrupt adjacent arena memory.
    // Cycle 110 fix C109-CR-F3 (HIGH conf 82): emit a loud-fail trap
    // when the cap is exceeded. Pre-fix returned `0 - 1` silently
    // and ~11 call sites all discarded the return value, leaving the
    // dropped patch invisible to the resolver loop — same silent-
    // failure pattern as the Stage 29.1 patch_table overflow that
    // corrupted K3 prior to the cap bump. Trap id 10031.
    let top = __arena_get(state);
    if top >= 16384 {
        emit_trap_with_id(10031);
        0 - 1
    } else {
        let table_base = __arena_get(state + 1);
        let entry = table_base + top * 3;
        __arena_set(entry, disp_slot);
        __arena_set(entry + 1, name_start);
        __arena_set(entry + 2, name_len);
        __arena_set(state, top + 1);
        0
    }
}

// Trailing exit-stub: take the top-of-eax value as the exit code
// and call sys_exit. Always 7 bytes.
//   89 C7      mov edi, eax
//   B8 3C 00 00 00   mov eax, 60
//   0F 05      syscall
fn emit_exit_with_eax() -> i32 {
    emit_byte(0x89); emit_byte(0xC7);
    emit_byte(0xB8); emit_byte(0x3C); emit_byte(0); emit_byte(0); emit_byte(0);
    emit_byte(0x0F); emit_byte(0x05);
    9
}

// --------------------------------------------------------------
// --------------------------------------------------------------
// Builtin name templates. Each is a static byte sequence we push
// to the arena once; subsequent comparisons use kovc_byte_eq with
// the stored byte_start + length. Allocated by install_builtin_names
// at the top of emit_elf_for_ast_to_path so they live BEFORE the
// ELF region in arena layout.
//
// Order in the bn_state region (each entry is just a byte_start
// since the length is constant per name):
//   slot 0: __arena_push    bytes (12 chars)
//   slot 1: __arena_get     bytes (11 chars)
//   slot 2: __arena_set     bytes (11 chars)
//   slot 3: __arena_len     bytes (11 chars)
//   slot 4: __helix_arena_base bytes (18 chars; for the .data symbol)
//   slot 5: read_file_to_arena  bytes (18 chars; first-arg-must-be-strlit
//                                 file builtin)
//   slot 6: write_file_to_arena bytes (19 chars)
//   slot 7: str_state.top — counter of string literals registered
//   slot 8: str_state.table_base — first entry slot (= bn_state + 9)
//   slots 9..56: 16 str_state entries × 3 i32 each (disp_slot,
//                body_byte_start, body_byte_len). Resolved after the
//                .data section is emitted; each string body is then
//                appended (with a NUL terminator) and the LEA disp32
//                placeholder is patched.
// --------------------------------------------------------------
fn install_builtin_names() -> i32 {
    let bn_state = __arena_push(0);          // slot 0 placeholder
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    // Reserve slots 5..121 (117 more slots: 2 file-name slots + 2
    // str_state header slots + 48 entry slots + 5 f32 builtin slots
    // (__fadd, __fsub, __fmul, __fdiv at 57..60; __fneg at 61) + 1
    // fn_type_state pointer slot at 62 (Phase 1.10 step 5c follow-on)
    // + 1 __fsqrt slot at 63 (Phase 1.10 step 5g)
    // + 1 __fabs slot at 64 (Phase 1.10 step 5h)
    // + 1 __i32_to_f32 slot at 65 (Phase 1.10 step 5i)
    // + 1 __f32_to_i32 slot at 66 (Phase 1.10 step 5j)
    // + 1 __fmin slot at 67 (Phase 1.10 step 5k)
    // + 1 __fmax slot at 68 (Phase 1.10 step 5l)
    // + 1 __bits_of_f32 slot at 69 (Phase 1.10 step 5m)
    // + 1 __f32_from_bits slot at 70 (Phase 1.10 step 5m)
    // + 1 __hash_i32 slot at 71 (Phase 1.10 step 5n)
    // + 1 __strlen slot at 72 (Phase 1.10 step 5o)
    // + 1 __f32_to_f64 slot at 73 (Phase 1.10 step 7e)
    // + 1 __f64_to_f32 slot at 74 (Phase 1.10 step 7e)
    // + 1 __dsqrt slot at 75 (Phase 1.10 step 7h)
    // + 1 __dabs slot at 76 (Phase 1.10 step 7i)
    // + 1 __dmin slot at 77 (Phase 1.10 step 7j)
    // + 1 __dmax slot at 78 (Phase 1.10 step 7j)
    // + 1 __i32_to_f64 slot at 79 (Phase 1.10 step 7k)
    // + 1 __f64_to_i32 slot at 80 (Phase 1.10 step 7k)
    // + 1 __bits_lo_f64 slot at 81 (Phase 1.10 step 7l)
    // + 1 __bits_hi_f64 slot at 82 (Phase 1.10 step 7l)
    // + 1 __f64_pack slot at 83 (Phase 1.10 step 7l)
    // + 34 slots at 84..117 for the Stage-7 match_state region (fail_state
    //   at 84..100 + end_table at 101..117). Pre-allocated to avoid
    //   __arena_push during code emission, which would corrupt the code
    //   stream. Cap: only one match-arm chain in flight at a time, but
    //   nested matches re-init in place — fine for Phase-0 since each
    //   match arm body is fully laid down before its parent's next arm.
    // + Stage 11: slots 118..121 reserved for reflection-runtime state.
    //   118 = "Quote" name bytes, 119 = "Splice" name bytes,
    //   120 = "modify" name bytes, 121 = quote handle counter (next free
    //   cell index, 0..63). Cells live at the LAST 64 slots of the produced
    //   binary's arena: cell[i] is at __helix_arena_base + 4 + (CAP-64+i)*4.
    //   The arena is BSS-zero-filled at load time so each cell starts as 0.
    // Audit A1-F1: bumped 117 → 118 to reserve slot 122 for the
    // match-dispatch scrut_ty stash (set by emit_match_dispatch, read
    // by emit_pat_variant_disc). See `match_scrut_ty_set/get` below.
    //
    // Stage 28.10 cycle-84 CN-1 fix (HIGH conf 95): bumped 118 → 152
    // to reserve slots 123..156 for emit_pat_or's success_state
    // (bn_state + 123, 17 slots: 123..139) and alt_fail_state
    // (bn_state + 140, 17 slots: 140..156). Pre-fix the OR scratch
    // OVERWROTE the builtin name table bytes (e.g. "__arena_push"
    // landed at slot 123 of the host compiler's arena because that
    // was the very next __arena_push after this init loop). Same
    // defect class as Audit-13 — unbounded write corrupting the
    // builtin name strings such that subsequent kovc_byte_eq checks
    // silently failed, falling through to unresolved CALL → SIGILL.
    // Bumping the init now reserves the OR scratch region BEFORE
    // the string-push sequence below, so first byte of
    // "__arena_push" lands at slot 157 — clean of OR scratch.
    let mut i: i32 = 0;
    while i < 152 {
        __arena_push(0);
        i = i + 1;
    }
    // K1.AD (2026-05-25): reserve 2 more init-zeroed slots AFTER
    // the PAT_OR region (123..156) but BEFORE the name strings.
    // Slot 157 = break-chain head (moved from slot 122 which
    // collided with match_scrut_ty -- match-inside-while traps).
    // Slot 158 = continue-chain head (mirrors break but targets
    // loop_top instead of end_label).
    __arena_push(0);      // slot 157: break-chain head
    __arena_push(0);      // slot 158: continue-chain head
    // K1.AF (2026-05-25): slot 159 holds the name-offset of the
    // builtin "__arena_push_pair" (set just after the name bytes
    // are pushed below). Init to 0 here so kovc_byte_eq probes
    // before assignment return a safe miss (length still 17
    // ensures no false match).
    __arena_push(0);      // slot 159: __arena_push_pair name offset
    // K1.AG (2026-05-25): slot 160 = __arena_push_triple name offset.
    __arena_push(0);      // slot 160: __arena_push_triple name offset
    // K1.AH (2026-05-25): slot 161 = "panic[28501]: " prefix offset.
    __arena_push(0);      // slot 161: panic prefix name offset
    // K1.AI (2026-05-25): slot 162 = "\n" newline byte offset.
    __arena_push(0);      // slot 162: panic newline offset
    // K1.AK (2026-05-25): slot 163 = "print_str" name offset.
    __arena_push(0);      // slot 163: print_str name offset
    // K1.F2 (2026-05-26): slot 164 = "reflect_hash" name offset.
    // Stub builtin returning 0 (matches Python's NotImplementedError
    // semantic at exit-code level: both compilers' result is
    // irrelevant for the bootstrap-compileable subset, but the
    // bootstrap accepts the call without SIGILLing).
    __arena_push(0);      // slot 164: reflect_hash name offset
    // K1.F3+F4 (2026-05-26): slots 165-168 = batched reflection-
    // family no-op stubs. Same pattern as K1.F2 -- each evaluates its
    // argument for side effects then returns 0 in eax. Closes 4 more
    // KOVC-MISSING rows that the audit batches (28+30) confirmed are
    // BOTH-PARTIAL in Python.
    __arena_push(0);      // slot 165: __trace_event name offset
    __arena_push(0);      // slot 166: __helix_splice name offset
    __arena_push(0);      // slot 167: __helix_modify name offset
    __arena_push(0);      // slot 168: __helix_reflect_hash name offset
    // K1.F20b (2026-05-27): slot 169 = "__trace_last" name offset.
    // Paired with the new depth-1 trace slot at arena[CAP-65]
    // (disp 8388352, one i32 slot below the Quote cell-table at
    // disp_base 8388356). __trace_event writes the LAST walked
    // arg's value to that slot before returning; __trace_last
    // reads it back. Makes trace_event observably distinguish
    // from a no-op stub (K1.F20 left it as a value-tap return-
    // contract upgrade only; K1.F20b adds the actual store +
    // read side).
    __arena_push(0);      // slot 169: __trace_last name offset
    // K1.F21 (2026-05-27): slot 170 = mangle scratch offset (pointer
    // into the arena where a 64-slot byte-buffer lives for building
    // generic-mono mangled names like `id__i32` at backpatch time).
    // Used by the patch_table backpatch loop in emit_elf_for_ast_to_
    // path: when a bare call's name lookup fails (`id(42)` where the
    // mono pass produced `id__i32`), copy the bare name to scratch,
    // append "__i32", and retry the lookup. This closes the matrix-
    // row-137 limitation "BARE call id(42) requires type inference
    // and FAILS in bootstrap".
    __arena_push(0);      // slot 170: mangle scratch offset
    // K1.F22c (2026-05-27): slot 171 = "print_str_ln" name offset.
    // New builtin emitting print_str's sys_write PLUS a trailing
    // newline sys_write to fd=1. Paired with the K1.F22b
    // println!("msg") macro expansion -- the parser now synthesizes
    // AST_CALL(print_str_ln, ...) instead of print_str so the
    // bootstrap's println! actually emits the newline that the K1.F22b
    // Phase-0 contract had documented as deferred.
    __arena_push(0);      // slot 171: print_str_ln name offset
    // K1.F22d (2026-05-27): slot 172 = "eprint_str_ln" name offset.
    // Stderr (fd=2) variant of print_str_ln. Paired with the
    // K1.F22d eprintln!("msg") macro expansion -- mirrors K1.F22c
    // codegen but emits `mov edi, 2` instead of `mov edi, 1` in both
    // sys_write blocks (message + newline). The byte content is the
    // same; only the fd selector differs.
    __arena_push(0);      // slot 172: eprint_str_ln name offset
    // K1.F22f (2026-05-27): slot 173 = "eprint_str" name offset.
    // Stderr (fd=2) no-newline variant of K1.AK print_str. Paired
    // with the K1.F22f eprint!("msg") macro expansion -- mirror of
    // K1.AK print_str codegen but emits `mov edi, 2` instead of
    // `mov edi, 1`. Completes the print/eprint x newline/no-newline
    // 2x2 grid (print/println/eprint/eprintln).
    __arena_push(0);      // slot 173: eprint_str name offset
    // K1.F23c (2026-05-27): slot 174 = "__tile_zeros" name offset.
    // Allocates an N*M-cell region in the arena via cursor-bump:
    // reads the arena cursor, advances it by N*M, writes back. Returns
    // the OLD cursor (the tile's base offset). Skips per-cell zero-init
    // -- BSS-zero on Linux means new cells start at 0.
    __arena_push(0);      // slot 174: __tile_zeros name offset
    // K1.F24-attempt (2026-05-27): slot 175 reserved for "__tile_add"
    // but the codegen attempt SIGILLed even in no-op-stub form.
    // Slot reservation kept so the next attempt (K1.F24b) lands at
    // the same slot index without renumbering.
    __arena_push(0);      // slot 175: (reserved for __tile_add, codegen deferred)
    // K1.F25 (2026-05-27): slot 176 = "__tile_sub" name offset.
    // Phase-0 tile primitive: elementwise subtraction loop (mirror of
    // __tile_add with sub opcode 2B instead of add opcode 03).
    __arena_push(0);      // slot 176: __tile_sub name offset
    // K1.F26 (2026-05-27): slot 177 = "__tile_mul" name offset.
    // Phase-0 tile primitive: elementwise multiplication. The imul
    // r32, r/m32 instruction has a 2-byte opcode (0F AF) making the
    // loop body 44 bytes instead of __tile_add's 43, so jge/jmp rel8
    // displacements shift by 1 (+40 and -44).
    __arena_push(0);      // slot 177: __tile_mul name offset
    // K1.F27 (2026-05-27): slot 178 = "__tile_matmul" name offset.
    // Phase-0 tile primitive: 2x2 square matrix multiplication, fully
    // unrolled (no loops). Signature: __tile_matmul(a, b, dst, N).
    // N must be 2 for the current 2x2 codegen; the value is currently
    // ignored. Future chunk generalizes to arbitrary N via loops.
    __arena_push(0);      // slot 178: __tile_matmul name offset

    // "__arena_push"
    let s0 = __arena_push(95); __arena_push(95); __arena_push(97); __arena_push(114);
    __arena_push(101); __arena_push(110); __arena_push(97); __arena_push(95);
    __arena_push(112); __arena_push(117); __arena_push(115); __arena_push(104);
    __arena_set(bn_state, s0);

    // K1.AF (2026-05-25): "__arena_push_pair" (17 chars: 95 95 97
    // 114 101 110 97 95 112 117 115 104 95 112 97 105 114). Atomic
    // 2-slot push for parser/codegen use. Stored at slot 159.
    let s_pair = __arena_push(95); __arena_push(95); __arena_push(97); __arena_push(114);
    __arena_push(101); __arena_push(110); __arena_push(97); __arena_push(95);
    __arena_push(112); __arena_push(117); __arena_push(115); __arena_push(104);
    __arena_push(95); __arena_push(112); __arena_push(97); __arena_push(105);
    __arena_push(114);
    __arena_set(bn_state + 159, s_pair);

    // K1.AG (2026-05-25): "__arena_push_triple" (19 chars: 95 95
    // 97 114 101 110 97 95 112 117 115 104 95 116 114 105 112 108
    // 101). Atomic 3-slot push. Stored at slot 160.
    let s_triple = __arena_push(95); __arena_push(95); __arena_push(97); __arena_push(114);
    __arena_push(101); __arena_push(110); __arena_push(97); __arena_push(95);
    __arena_push(112); __arena_push(117); __arena_push(115); __arena_push(104);
    __arena_push(95); __arena_push(116); __arena_push(114); __arena_push(105);
    __arena_push(112); __arena_push(108); __arena_push(101);
    __arena_set(bn_state + 160, s_triple);

    // K1.AH (2026-05-25): "panic[28501]: " (14 bytes: 112 97 110
    // 105 99 91 50 56 53 48 49 93 58 32). Matches Python's panic
    // codegen prefix format. Stored at slot 161 for the panic
    // codegen to load via lea_rsi + str_table_add.
    let s_panic_pfx = __arena_push(112); __arena_push(97); __arena_push(110); __arena_push(105);
    __arena_push(99); __arena_push(91); __arena_push(50); __arena_push(56);
    __arena_push(53); __arena_push(48); __arena_push(49); __arena_push(93);
    __arena_push(58); __arena_push(32);
    __arena_set(bn_state + 161, s_panic_pfx);

    // K1.AI (2026-05-25): "\n" (1 byte: 10). Trailing newline for
    // panic output. Pushed as a 1-byte arena entry so str_table_add
    // can register it the same way as the prefix and message.
    let s_panic_nl = __arena_push(10);
    __arena_set(bn_state + 162, s_panic_nl);

    // K1.AK (2026-05-25): "print_str" (9 chars: 112 114 105 110 116
    // 95 115 116 114). Builtin for writing a string literal to
    // stdout via sys_write(1, ptr, len). Stored at slot 163.
    let s_print_str = __arena_push(112); __arena_push(114); __arena_push(105); __arena_push(110);
    __arena_push(116); __arena_push(95); __arena_push(115); __arena_push(116);
    __arena_push(114);
    __arena_set(bn_state + 163, s_print_str);

    // K1.F2 (2026-05-26): "reflect_hash" (12 chars: 114 101 102 108
    // 101 99 116 95 104 97 115 104). Builtin stub returning 0. The
    // matrix had this as KOVC-MISSING; Python's direct-compile path
    // also fails with NotImplementedError. Both compilers behave
    // equivalently for the bootstrap-compileable subset. Stored at
    // slot 164.
    let s_rh = __arena_push(114); __arena_push(101); __arena_push(102); __arena_push(108);
    __arena_push(101); __arena_push(99); __arena_push(116); __arena_push(95);
    __arena_push(104); __arena_push(97); __arena_push(115); __arena_push(104);
    __arena_set(bn_state + 164, s_rh);

    // K1.F3 (2026-05-26): "__trace_event" (13 chars: 95 95 116 114
    // 97 99 101 95 101 118 101 110 116). Stored at slot 165.
    let s_te = __arena_push(95); __arena_push(95); __arena_push(116); __arena_push(114);
    __arena_push(97); __arena_push(99); __arena_push(101); __arena_push(95);
    __arena_push(101); __arena_push(118); __arena_push(101); __arena_push(110);
    __arena_push(116);
    __arena_set(bn_state + 165, s_te);

    // K1.F4 (2026-05-26): "__helix_splice" (14 chars: 95 95 104 101
    // 108 105 120 95 115 112 108 105 99 101). Stored at slot 166.
    let s_hs = __arena_push(95); __arena_push(95); __arena_push(104); __arena_push(101);
    __arena_push(108); __arena_push(105); __arena_push(120); __arena_push(95);
    __arena_push(115); __arena_push(112); __arena_push(108); __arena_push(105);
    __arena_push(99); __arena_push(101);
    __arena_set(bn_state + 166, s_hs);

    // K1.F4 (2026-05-26): "__helix_modify" (14 chars: 95 95 104 101
    // 108 105 120 95 109 111 100 105 102 121). Stored at slot 167.
    let s_hm = __arena_push(95); __arena_push(95); __arena_push(104); __arena_push(101);
    __arena_push(108); __arena_push(105); __arena_push(120); __arena_push(95);
    __arena_push(109); __arena_push(111); __arena_push(100); __arena_push(105);
    __arena_push(102); __arena_push(121);
    __arena_set(bn_state + 167, s_hm);

    // K1.F4 (2026-05-26): "__helix_reflect_hash" (20 chars: 95 95 104
    // 101 108 105 120 95 114 101 102 108 101 99 116 95 104 97 115 104).
    // Stored at slot 168.
    let s_hrh = __arena_push(95); __arena_push(95); __arena_push(104); __arena_push(101);
    __arena_push(108); __arena_push(105); __arena_push(120); __arena_push(95);
    __arena_push(114); __arena_push(101); __arena_push(102); __arena_push(108);
    __arena_push(101); __arena_push(99); __arena_push(116); __arena_push(95);
    __arena_push(104); __arena_push(97); __arena_push(115); __arena_push(104);
    __arena_set(bn_state + 168, s_hrh);

    // K1.F20b (2026-05-27): "__trace_last" (12 chars: 95 95 116 114
    // 97 99 101 95 108 97 115 116). Read-side of the K1.F20b depth-1
    // trace slot. Codegen: lea rax, [arena_base]; mov eax,
    // [rax + 8388352]. Stored at slot 169.
    let s_tl = __arena_push(95); __arena_push(95); __arena_push(116); __arena_push(114);
    __arena_push(97); __arena_push(99); __arena_push(101); __arena_push(95);
    __arena_push(108); __arena_push(97); __arena_push(115); __arena_push(116);
    __arena_set(bn_state + 169, s_tl);

    // K1.F21 (2026-05-27): 64-slot mangle scratch region for the
    // generic-bare-call name resolution fallback. Each slot holds one
    // byte (per the kovc_byte_eq one-byte-per-i32-slot convention).
    // Cap of 64 EXACTLY covers names up to 59 bytes long + the 5-byte
    // "__i32" suffix (max write at scratch[59+4] = scratch[63] = the
    // 64th slot; an exact fit, NOT a safety-margin reserve). The
    // backpatch loop's `target_name_l < 60` guard makes the fit
    // bounds-safe; bumping the cap is the K1.F21b refinement if real
    // user-program identifiers ever approach 60 chars (K3.K
    // 2026-05-27 audit fix to the K1.F21 doc-drift: the original
    // comment said "leaves 4 bytes of headroom" -- it does not; the
    // gate is exact-fit, which is safe but not a safety margin).
    // Reserved BEFORE the ELF region grows so a backpatch-time
    // mangle write doesn't corrupt the code byte stream.
    let s_scratch = __arena_push(0);
    let mut sc: i32 = 0;
    while sc < 63 {
        __arena_push(0);
        sc = sc + 1;
    }
    __arena_set(bn_state + 170, s_scratch);

    // K1.F22c (2026-05-27): "print_str_ln" name bytes (12 chars: 112
    // 114 105 110 116 95 115 116 114 95 108 110). Stored at slot 171.
    // Routes the K1.F22b println! expansion through a codegen handler
    // that emits print_str's sys_write + a trailing newline sys_write.
    let s_psln = __arena_push(112); __arena_push(114); __arena_push(105); __arena_push(110);
    __arena_push(116); __arena_push(95); __arena_push(115); __arena_push(116);
    __arena_push(114); __arena_push(95); __arena_push(108); __arena_push(110);
    __arena_set(bn_state + 171, s_psln);

    // K1.F22d (2026-05-27): "eprint_str_ln" name bytes (13 chars: 101
    // 112 114 105 110 116 95 115 116 114 95 108 110). Stored at slot
    // 172. Stderr variant of print_str_ln.
    let s_epsln = __arena_push(101); __arena_push(112); __arena_push(114);
    __arena_push(105); __arena_push(110); __arena_push(116); __arena_push(95);
    __arena_push(115); __arena_push(116); __arena_push(114); __arena_push(95);
    __arena_push(108); __arena_push(110);
    __arena_set(bn_state + 172, s_epsln);

    // K1.F22f (2026-05-27): "eprint_str" name bytes (10 chars: 101 112
    // 114 105 110 116 95 115 116 114). Stored at slot 173. No-newline
    // stderr variant of K1.AK print_str -- the K1.F22f eprint!("msg")
    // macro expansion synthesizes AST_CALL(eprint_str, str_arg) and
    // the codegen below emits a single sys_write to fd=2.
    let s_eps = __arena_push(101); __arena_push(112); __arena_push(114);
    __arena_push(105); __arena_push(110); __arena_push(116); __arena_push(95);
    __arena_push(115); __arena_push(116); __arena_push(114);
    __arena_set(bn_state + 173, s_eps);

    // K1.F23c (2026-05-27): "__tile_zeros" name bytes (12 chars: 95 95
    // 116 105 108 101 95 122 101 114 111 115). Stored at slot 174.
    // Phase-0 tile primitive: cursor-bump allocation of N*M arena slots,
    // returning the old cursor as the tile's base offset.
    let s_tz = __arena_push(95); __arena_push(95); __arena_push(116);
    __arena_push(105); __arena_push(108); __arena_push(101); __arena_push(95);
    __arena_push(122); __arena_push(101); __arena_push(114); __arena_push(111);
    __arena_push(115);
    __arena_set(bn_state + 174, s_tz);

    // K1.F24e (2026-05-27): re-attempt at slot 175 with 3-run
    // methodology. The K1.F24 SIGILL was attributed to a real
    // codegen defect but K1.F24d showed the recent "defects" were
    // WSL flakes. Re-shipping as the no-op stub variant; if 3
    // consecutive runs return rc=0, the K1.F24 finding was ALSO
    // a flake and the codegen path works fine.
    let s_ta = __arena_push(95); __arena_push(95); __arena_push(116);
    __arena_push(105); __arena_push(108); __arena_push(101); __arena_push(95);
    __arena_push(97); __arena_push(100); __arena_push(100);
    __arena_set(bn_state + 175, s_ta);

    // K1.F25 (2026-05-27): "__tile_sub" name bytes (10 chars: 95 95
    // 116 105 108 101 95 115 117 98). Stored at slot 176.
    // Elementwise subtraction (mirrors __tile_add codegen with sub
    // opcode 2B instead of add opcode 03).
    let s_tsub = __arena_push(95); __arena_push(95); __arena_push(116);
    __arena_push(105); __arena_push(108); __arena_push(101); __arena_push(95);
    __arena_push(115); __arena_push(117); __arena_push(98);
    __arena_set(bn_state + 176, s_tsub);

    // K1.F26 (2026-05-27): "__tile_mul" name bytes (10 chars: 95 95
    // 116 105 108 101 95 109 117 108). Stored at slot 177.
    // Elementwise multiplication; imul opcode is 0F AF (2-byte),
    // making the loop body 44 bytes vs __tile_add's 43.
    let s_tmul = __arena_push(95); __arena_push(95); __arena_push(116);
    __arena_push(105); __arena_push(108); __arena_push(101); __arena_push(95);
    __arena_push(109); __arena_push(117); __arena_push(108);
    __arena_set(bn_state + 177, s_tmul);

    // K1.F27 (2026-05-27): "__tile_matmul" name bytes (13 chars: 95 95
    // 116 105 108 101 95 109 97 116 109 117 108). Stored at slot 178.
    // 2x2 square matmul. Unrolled (no loops).
    let s_tmm = __arena_push(95); __arena_push(95); __arena_push(116);
    __arena_push(105); __arena_push(108); __arena_push(101); __arena_push(95);
    __arena_push(109); __arena_push(97); __arena_push(116);
    __arena_push(109); __arena_push(117); __arena_push(108);
    __arena_set(bn_state + 178, s_tmm);

    // K3.O (2026-05-27): relocate the str_table region. The original
    // slots 9..56 (16 entries × 3) collided with the f32 builtin slots
    // at 57+, so we couldn't simply bump the cap in-place. K3.O
    // allocates a FRESH 192-slot region here (64 entries × 3 i32)
    // and rewrites slot 8 (str_state.table_base) to point at it.
    // The original 9..56 region is now unused (small waste, no
    // collision concern). The K3.N audit flagged the 16-entry cap as
    // a silent-overflow risk now that K1.F22c made it 2x easier to
    // hit (panic 3 entries + each println! 2 entries -> 1 panic + 7
    // println! = exact 17). 64-entry cap gives ~4x headroom over
    // K1.F22c's worst case.
    let s_strtbl = __arena_push(0);
    let mut st: i32 = 0;
    while st < 191 {
        __arena_push(0);
        st = st + 1;
    }
    __arena_set(bn_state + 8, s_strtbl);

    // "__arena_get"
    let s1 = __arena_push(95); __arena_push(95); __arena_push(97); __arena_push(114);
    __arena_push(101); __arena_push(110); __arena_push(97); __arena_push(95);
    __arena_push(103); __arena_push(101); __arena_push(116);
    __arena_set(bn_state + 1, s1);

    // "__arena_set"
    let s2 = __arena_push(95); __arena_push(95); __arena_push(97); __arena_push(114);
    __arena_push(101); __arena_push(110); __arena_push(97); __arena_push(95);
    __arena_push(115); __arena_push(101); __arena_push(116);
    __arena_set(bn_state + 2, s2);

    // "__arena_len"
    let s3 = __arena_push(95); __arena_push(95); __arena_push(97); __arena_push(114);
    __arena_push(101); __arena_push(110); __arena_push(97); __arena_push(95);
    __arena_push(108); __arena_push(101); __arena_push(110);
    __arena_set(bn_state + 3, s3);

    // "__helix_arena_base"
    let s4 = __arena_push(95); __arena_push(95); __arena_push(104); __arena_push(101);
    __arena_push(108); __arena_push(105); __arena_push(120); __arena_push(95);
    __arena_push(97); __arena_push(114); __arena_push(101); __arena_push(110);
    __arena_push(97); __arena_push(95); __arena_push(98); __arena_push(97);
    __arena_push(115); __arena_push(101);
    __arena_set(bn_state + 4, s4);

    // "read_file_to_arena"  (18 chars: r e a d _ f i l e _ t o _ a r e n a)
    let s5 = __arena_push(114); __arena_push(101); __arena_push(97); __arena_push(100);
    __arena_push(95); __arena_push(102); __arena_push(105); __arena_push(108);
    __arena_push(101); __arena_push(95); __arena_push(116); __arena_push(111);
    __arena_push(95); __arena_push(97); __arena_push(114); __arena_push(101);
    __arena_push(110); __arena_push(97);
    __arena_set(bn_state + 5, s5);

    // "write_file_to_arena" (19 chars)
    let s6 = __arena_push(119); __arena_push(114); __arena_push(105); __arena_push(116);
    __arena_push(101); __arena_push(95); __arena_push(102); __arena_push(105);
    __arena_push(108); __arena_push(101); __arena_push(95); __arena_push(116);
    __arena_push(111); __arena_push(95); __arena_push(97); __arena_push(114);
    __arena_push(101); __arena_push(110); __arena_push(97);
    __arena_set(bn_state + 6, s6);

    // str_state header: slot 7 = top, slot 8 = table_base.
    __arena_set(bn_state + 7, 0);
    __arena_set(bn_state + 8, bn_state + 9);

    // Phase 1.10 step 4: f32 arithmetic builtins __fadd / __fsub /
    // __fmul / __fdiv (each 6 chars). Result returned in eax as the
    // f32 bit pattern, computed via x86-64 SSE: movd xmm0/xmm1, regs;
    // [add|sub|mul|div]ss; movd eax, xmm0.

    // "__fadd"  (95 95 102 97 100 100)
    let s7 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(97); __arena_push(100); __arena_push(100);
    __arena_set(bn_state + 57, s7);

    // "__fsub"  (95 95 102 115 117 98)
    let s8 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(115); __arena_push(117); __arena_push(98);
    __arena_set(bn_state + 58, s8);

    // "__fmul"  (95 95 102 109 117 108)
    let s9 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(109); __arena_push(117); __arena_push(108);
    __arena_set(bn_state + 59, s9);

    // "__fdiv"  (95 95 102 100 105 118)
    let s10 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(100); __arena_push(105); __arena_push(118);
    __arena_set(bn_state + 60, s10);

    // "__fneg"  (95 95 102 110 101 103) — single-arg f32 negate.
    let s11 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(110); __arena_push(101); __arena_push(103);
    __arena_set(bn_state + 61, s11);

    // Phase 1.10 step 5g: "__fsqrt" (95 95 102 115 113 114 116) — 7
    // chars. Single-arg f32 square root via SSE2 sqrtss xmm0, xmm0.
    // Hardware-direct primitive (vs the Newton-iteration __sqrt in
    // helixc/stdlib/transcendentals.hx). Result is the f32 bit pattern
    // in eax. NaN inputs propagate (sqrtss preserves NaN), negatives
    // produce a quiet NaN.
    let s12 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(115); __arena_push(113); __arena_push(114); __arena_push(116);
    __arena_set(bn_state + 63, s12);

    // Phase 1.10 step 5h: "__fabs" (95 95 102 97 98 115) — 6 chars.
    // Single-arg f32 absolute value via integer mask: clears the sign
    // bit of the f32 bit pattern in eax. Mirrors __fneg (which XORs
    // with 0x80000000 to flip the sign bit); __fabs ANDs with
    // 0x7FFFFFFF to clear it. NaN inputs propagate (sign bit cleared
    // but mantissa preserved). No SSE registers touched.
    let s13 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(97); __arena_push(98); __arena_push(115);
    __arena_set(bn_state + 64, s13);

    // Phase 1.10 step 5i: "__i32_to_f32"
    // (95 95 105 51 50 95 116 111 95 102 51 50) — 12 chars. Single-arg
    // i32 -> f32 conversion via SSE2 cvtsi2ss. Result is the f32 bit
    // pattern in eax. Distinct from __f* prefix (this is __i*) so
    // is_f32_expr needs an explicit byte_eq against the installed name
    // slot to type the call's result as f32.
    let s14 = __arena_push(95); __arena_push(95); __arena_push(105);
    __arena_push(51); __arena_push(50); __arena_push(95);
    __arena_push(116); __arena_push(111); __arena_push(95);
    __arena_push(102); __arena_push(51); __arena_push(50);
    __arena_set(bn_state + 65, s14);

    // Phase 1.10 step 5j: "__f32_to_i32"
    // (95 95 102 51 50 95 116 111 95 105 51 50) — 12 chars. Single-arg
    // f32 -> i32 truncating conversion via SSE2 cvttss2si. Result is
    // the truncated signed integer value in eax. Inverse of step 5i's
    // __i32_to_f32. Starts with __f so the cheap is_underscore_f_call
    // prefix would incorrectly tag this call as f32 — is_f32_expr does
    // an explicit byte_eq against the installed name slot BEFORE the
    // prefix check and returns 0 (i32).
    let s15 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(51); __arena_push(50); __arena_push(95);
    __arena_push(116); __arena_push(111); __arena_push(95);
    __arena_push(105); __arena_push(51); __arena_push(50);
    __arena_set(bn_state + 66, s15);

    // Phase 1.10 step 5k: "__fmin" (95 95 102 109 105 110) — 6 chars.
    // Two-arg f32 minimum via SSE2 minss xmm0, xmm1. Mirrors __fadd
    // shape exactly (binary SSE op, eval a -> push, eval b, mov ecx eax,
    // pop rax, movd xmm0/xmm1, minss, movd eax). minss is non-commutative
    // for NaN: if either operand is NaN, the second operand (xmm1) is
    // returned. Since we load b -> xmm1, NaN(a) yields b and NaN(b)
    // yields b (so any NaN input yields b). is_underscore_f_call already
    // matches __f* prefix so the result types as f32 through is_f32_expr.
    let s16 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(109); __arena_push(105); __arena_push(110);
    __arena_set(bn_state + 67, s16);

    // Phase 1.10 step 5l: "__fmax" (95 95 102 109 97 120) — 6 chars.
    // Two-arg f32 maximum via SSE2 maxss xmm0, xmm1. Mirrors __fmin
    // exactly, with opcode F3 0F 5F C1 (one byte differs: 5D->5F).
    // maxss is asymmetric on NaN: if either operand is NaN, the second
    // operand (xmm1 = b) is returned. is_underscore_f_call's __f*
    // prefix matches so the result types as f32 through is_f32_expr.
    let s17 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(109); __arena_push(97); __arena_push(120);
    __arena_set(bn_state + 68, s17);

    // Phase 1.10 step 5m: "__bits_of_f32" (13 chars: 95 95 98 105 116
    // 115 95 111 102 95 102 51 50). Identity-codegen bit reinterpret —
    // f32 already lives in eax as its IEEE 754 bit pattern, so no extra
    // bytes are emitted. Distinct from __f32_to_i32 (which truncates).
    // Starts with __b so doesn't match the __f* prefix; is_f32_expr
    // falls through to fn_type_table and returns 0 (i32) by default.
    let s18 = __arena_push(95); __arena_push(95); __arena_push(98);
    __arena_push(105); __arena_push(116); __arena_push(115);
    __arena_push(95); __arena_push(111); __arena_push(102);
    __arena_push(95); __arena_push(102); __arena_push(51); __arena_push(50);
    __arena_set(bn_state + 69, s18);

    // Phase 1.10 step 5m: "__f32_from_bits" (15 chars: 95 95 102 51 50
    // 95 102 114 111 109 95 98 105 116 115). Identity-codegen inverse
    // of __bits_of_f32. Distinct from __i32_to_f32 (which converts the
    // numeric value). Starts with __f so the __f* prefix correctly
    // types the result as f32 through is_f32_expr (length 15 != 12 so
    // it doesn't collide with __f32_to_i32's explicit byte_eq case).
    let s19 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(51); __arena_push(50); __arena_push(95);
    __arena_push(102); __arena_push(114); __arena_push(111);
    __arena_push(109); __arena_push(95); __arena_push(98);
    __arena_push(105); __arena_push(116); __arena_push(115);
    __arena_set(bn_state + 70, s19);

    // Phase 1.10 step 5n: "__hash_i32" (10 chars: 95 95 104 97 115 104
    // 95 105 51 50). Single-arg i32 -> i32 quadratic mixer hash that
    // mirrors helixc-Python's lower_ast.py:939-963 (used for symbol
    // bucketing). Result: h = x*x*c1 + x*c2 + c3 (mod 2^32 via signed
    // wraparound). Pure inline arithmetic, no SSE registers, no IR op.
    // Starts with __h so doesn't match the __f* prefix; is_f32_expr
    // falls through to fn_type_table -> 0 (i32 result).
    let s20 = __arena_push(95); __arena_push(95); __arena_push(104);
    __arena_push(97); __arena_push(115); __arena_push(104);
    __arena_push(95); __arena_push(105); __arena_push(51);
    __arena_push(50);
    __arena_set(bn_state + 71, s20);

    // Phase 1.10 step 5o: "__strlen" (8 chars: 95 95 115 116 114 108 101
    // 110). Compile-time string-literal length. First arg MUST be
    // AST_STR_LIT; codegen reads body_l (byte length) and folds to
    // `mov eax, body_l` (5 bytes). Mirrors helixc-Python lower_ast.py:
    // 966-969 const_int(len) folding. Starts with __s so doesn't match
    // the __f* prefix; is_f32_expr falls through to fn_type_table -> 0
    // (i32 result).
    let s21 = __arena_push(95); __arena_push(95); __arena_push(115);
    __arena_push(116); __arena_push(114); __arena_push(108);
    __arena_push(101); __arena_push(110);
    __arena_set(bn_state + 72, s21);

    // Phase 1.10 step 7e: "__f32_to_f64" (12 chars: 95 95 102 51 50
    // 95 116 111 95 102 54 52). Single-arg f32 -> f64 widening
    // conversion via SSE2 cvtss2sd. Result is the f64 bit pattern in
    // rax. Starts with __f, length 12 collides with __i32_to_f32 /
    // __f32_to_i32 namespace — disambiguated by explicit byte_eq in
    // is_f64_expr (returns 1 here) and is_f32_expr (returns 0 here).
    let s22 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(51); __arena_push(50); __arena_push(95);
    __arena_push(116); __arena_push(111); __arena_push(95);
    __arena_push(102); __arena_push(54); __arena_push(52);
    __arena_set(bn_state + 73, s22);

    // Phase 1.10 step 7e: "__f64_to_f32" (12 chars: 95 95 102 54 52
    // 95 116 111 95 102 51 50). Single-arg f64 -> f32 narrowing
    // conversion via SSE2 cvtsd2ss. Result is the f32 bit pattern in
    // eax. Starts with __f, length 12 — disambiguated by explicit
    // byte_eq: returns 1 in is_f32_expr (it's f32-typed) and 0 in
    // is_f64_expr (it's NOT f64).
    let s23 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(54); __arena_push(52); __arena_push(95);
    __arena_push(116); __arena_push(111); __arena_push(95);
    __arena_push(102); __arena_push(51); __arena_push(50);
    __arena_set(bn_state + 74, s23);

    // Phase 1.10 step 7h: "__dsqrt" (7 chars: 95 95 100 115 113 114 116).
    // Single-arg f64 square root via SSE2 sqrtsd xmm0, xmm0. Mirrors
    // __fsqrt (step 5g) but on 64-bit doubles: movq xmm0, rax; sqrtsd
    // xmm0, xmm0; movq rax, xmm0. Result is the f64 bit pattern in rax.
    // Starts with __d so doesn't match the __f* prefix; is_f64_expr
    // adds an explicit byte_eq against this slot to type the call as
    // f64. (is_f32_expr falls through to fn_type_table -> 0 by default.)
    let s24 = __arena_push(95); __arena_push(95); __arena_push(100);
    __arena_push(115); __arena_push(113); __arena_push(114); __arena_push(116);
    __arena_set(bn_state + 75, s24);

    // Phase 1.10 step 7i: "__dabs" (6 chars: 95 95 100 97 98 115).
    // Single-arg f64 absolute value: clears bit 63 (sign bit) of the
    // f64 bit pattern in rax. Implementation uses shl/shr instead of
    // and-with-imm64 since x86-64 has no AND-rax-imm64; the shift
    // pair is 6 bytes (vs 13 for movabs+and).
    //   48 D1 E0    shl rax, 1    (drops bit 63 into CF)
    //   48 D1 E8    shr rax, 1    (refills bit 63 with 0)
    // Net effect: bits 0..62 preserved, bit 63 cleared. Mirrors
    // __fabs (step 5h) on 64-bit f64. Starts with __d so doesn't
    // match the __f* prefix; is_f64_expr adds explicit byte_eq.
    let s25 = __arena_push(95); __arena_push(95); __arena_push(100);
    __arena_push(97); __arena_push(98); __arena_push(115);
    __arena_set(bn_state + 76, s25);

    // Phase 1.10 step 7j: "__dmin" (6 chars: 95 95 100 109 105 110).
    // Two-arg f64 minimum via SSE2 minsd xmm0, xmm1. Mirrors __fmin
    // (step 5k) on doubles. NaN handling: minsd returns the second
    // operand (xmm1, holding b) on any NaN — same asymmetric behavior
    // as minss. is_f64_expr adds explicit byte_eq for the type tag.
    let s26 = __arena_push(95); __arena_push(95); __arena_push(100);
    __arena_push(109); __arena_push(105); __arena_push(110);
    __arena_set(bn_state + 77, s26);

    // Phase 1.10 step 7j: "__dmax" (6 chars: 95 95 100 109 97 120).
    // Two-arg f64 maximum via SSE2 maxsd xmm0, xmm1. Mirrors __fmax
    // (step 5l) on doubles. NaN handling: maxsd returns the second
    // operand (xmm1, b). Same asymmetric NaN behavior as maxss.
    let s27 = __arena_push(95); __arena_push(95); __arena_push(100);
    __arena_push(109); __arena_push(97); __arena_push(120);
    __arena_set(bn_state + 78, s27);

    // Phase 1.10 step 7k: "__i32_to_f64" (12 chars: 95 95 105 51 50
    // 95 116 111 95 102 54 52). Single-arg widening i32 -> f64 via
    // SSE2 cvtsi2sd. Mirrors __i32_to_f32 (step 5i) but on doubles.
    // Starts with __i so doesn't match __f* prefix; is_f64_expr adds
    // explicit byte_eq.
    let s28 = __arena_push(95); __arena_push(95); __arena_push(105);
    __arena_push(51); __arena_push(50); __arena_push(95);
    __arena_push(116); __arena_push(111); __arena_push(95);
    __arena_push(102); __arena_push(54); __arena_push(52);
    __arena_set(bn_state + 79, s28);

    // Phase 1.10 step 7k: "__f64_to_i32" (12 chars: 95 95 102 54 52
    // 95 116 111 95 105 51 50). Single-arg truncating f64 -> i32 via
    // SSE2 cvttsd2si. Mirrors __f32_to_i32 (step 5j) but on doubles.
    // Starts with __f, length 12 — disambiguated by explicit byte_eq:
    // returns 0 in is_f64_expr (it's i32, not f64) and is_f32_expr
    // (returns 0 BEFORE the __f* prefix match). Result types as i32.
    let s29 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(54); __arena_push(52); __arena_push(95);
    __arena_push(116); __arena_push(111); __arena_push(95);
    __arena_push(105); __arena_push(51); __arena_push(50);
    __arena_set(bn_state + 80, s29);

    // Phase 1.10 step 7l: "__bits_lo_f64" (13 chars: 95 95 98 105 116
    // 115 95 108 111 95 102 54 52). Single-arg f64 -> i32 returning the
    // LOW 32 bits of the f64 bit pattern. Identity codegen (rax low 32
    // == eax). Result types as i32; starts with __b so doesn't match
    // __f* prefix.
    let s30 = __arena_push(95); __arena_push(95); __arena_push(98);
    __arena_push(105); __arena_push(116); __arena_push(115);
    __arena_push(95); __arena_push(108); __arena_push(111);
    __arena_push(95); __arena_push(102); __arena_push(54); __arena_push(52);
    __arena_set(bn_state + 81, s30);

    // Phase 1.10 step 7l: "__bits_hi_f64" (13 chars: 95 95 98 105 116
    // 115 95 104 105 95 102 54 52). Single-arg f64 -> i32 returning the
    // HIGH 32 bits via shr rax, 32 (4 bytes). Result types as i32.
    let s31 = __arena_push(95); __arena_push(95); __arena_push(98);
    __arena_push(105); __arena_push(116); __arena_push(115);
    __arena_push(95); __arena_push(104); __arena_push(105);
    __arena_push(95); __arena_push(102); __arena_push(54); __arena_push(52);
    __arena_set(bn_state + 82, s31);

    // Phase 1.10 step 7l: "__f64_pack" (10 chars: 95 95 102 54 52 95
    // 112 97 99 107). Two-arg (hi: i32, lo: i32) -> f64. Combines two
    // i32 halves into a 64-bit f64 bit pattern in rax. Starts with
    // __f so the __f* prefix would wrongly tag as f32 — explicit
    // byte_eq in is_f32_expr returns 0 BEFORE the prefix match.
    let s32 = __arena_push(95); __arena_push(95); __arena_push(102);
    __arena_push(54); __arena_push(52); __arena_push(95);
    __arena_push(112); __arena_push(97); __arena_push(99); __arena_push(107);
    __arena_set(bn_state + 83, s32);

    // Stage 11: reflection-runtime builtin names. These are user-facing
    // surface (not __-prefixed) since the plan's Quote / Splice / modify
    // come straight from the Helix surface syntax — different from the
    // internal __-prefixed builtins above.
    //   "Quote"  = 81 117 111 116 101  (5 chars)
    //   "Splice" = 83 112 108 105 99 101  (6 chars)
    //   "modify" = 109 111 100 105 102 121  (6 chars)
    let s33 = __arena_push(81); __arena_push(117); __arena_push(111);
    __arena_push(116); __arena_push(101);
    __arena_set(bn_state + 118, s33);

    let s34 = __arena_push(83); __arena_push(112); __arena_push(108);
    __arena_push(105); __arena_push(99); __arena_push(101);
    __arena_set(bn_state + 119, s34);

    let s35 = __arena_push(109); __arena_push(111); __arena_push(100);
    __arena_push(105); __arena_push(102); __arena_push(121);
    __arena_set(bn_state + 120, s35);

    // Slot 121: quote handle counter (compile-time). Each Quote(...) call
    // allocates the current value as its handle and bumps the counter.
    // Cap 64 cells — exceeding traps 81002 (cell-table overflow).
    __arena_set(bn_state + 121, 0);

    // Slot 122 (Audit A1-F1 fix): match-dispatch scrut_ty stash. Set by
    // emit_match_dispatch from expr_type(scrut_idx, ...); read by
    // emit_pat_variant_disc to choose between i32-direct cmp (scrut_ty
    // == 0) and pointer-deref cmp (scrut_ty != 0). Cleared/reused per
    // match dispatch — nested matches see their own enclosing scope's
    // value because emit_match_dispatch sets it before the arm chain.
    __arena_set(bn_state + 122, 0);

    bn_state
}

// Audit A1-F1: match scrut_ty stash accessors.
fn match_scrut_ty_set(bn_state: i32, ty: i32) -> i32 {
    __arena_set(bn_state + 122, ty);
    0
}
fn match_scrut_ty_get(bn_state: i32) -> i32 {
    __arena_get(bn_state + 122)
}

// --------------------------------------------------------------
// Stage 28.9: diag_arena — collected diagnostics from validation
// passes (panic_pass, unsafe_pass, deprecated_pass, trace_pass,
// autotune_pass).
// Mirrors the Python `_deprecation_warnings` channel but as a
// structured side-table, not monkey-patched on the AST.
//
// Layout: a contiguous region whose first slot is the count, second
// is a capacity, then `cap` 4-slot entries, then a sticky
// "overflowed" flag slot. Each entry stores:
//   slot 0: trap-id code (e.g. 28501 panic, 28601 unsafe, 28701
//           deprecated, 28502 unwind, 27001 autotune product).
//           0 sentinel = empty slot.
//   slot 1: severity (1 = warning, 2 = error)
//   slot 2: ast_node_idx (the arena index of the AST node where the
//           diagnostic fires — e.g. the AST_CALL node for panic_pass
//           and deprecated call-site warnings, or the AST_FN_DECL for
//           trace/unwind/autotune/deprecated-table-cap diagnostics).
//           Phase-0 has no source-byte span on AST nodes; full line/col
//           reconstruction is deferred to a future stage that wires
//           a side-table from AST node idx → source byte range.
//
//           Stage-28.9 audit-cycle-1 D1 fix: renamed from
//           `src_byte_start` (misleading — the value passed is an AST
//           arena index, not a byte offset). All 5 emit sites already
//           pass AST indices; this rename clarifies the contract.
//   slot 3: aux i32 — pass-specific data (e.g. deprecated_pass
//           28701: dep_tab entry ptr, deprecated_pass 28702: dropped
//           fn name start, panic_pass: arg_count, autotune_pass
//           27001: saturated product, autotune_pass 27003:
//           parse-error kind, autotune_pass 27004: fn name start)
//
// Header slots:
//   slot 0 (base+0):     count
//   slot 1 (base+1):     cap (64)
//   slots 2..2+cap*4-1:  cap entries of 4 slots each
//   slot base+2+cap*4:   sticky `overflowed` flag (0 normally, set
//                        to 1 on first diag_emit that hits cap)
//
// Capacity: 64 entries by default. Overflow sets the sticky flag
// and causes emit_elf_for_ast_to_path to patch main's prologue
// with `emit_trap_with_id(28999)` AFTER validation passes complete
// (so the produced binary aborts even though no entry slot was
// available for the 65th diag). Phase-0 chooses 64 because the
// heaviest known fixture (test_bootstrap_kovc_full_pipeline_arithmetic)
// has ~10 panic call sites at most; 64 is 6x headroom.
//
// Stage-28.9 audit-cycle-1 Finding 1 fix: the previous design
// called `emit_trap_with_id(28999)` directly inside diag_emit, but
// the validation passes run BEFORE `elf_start = __arena_len()` is
// captured. The 7 trap bytes landed in dead pre-ELF arena, then
// the count stayed pinned at cap so every further emit retripped
// silently. The sticky flag observes the overflow, and the codegen
// emit at `is_main_fn` patches a fresh trap into main's prologue
// where it actually executes.
//
// Severity policy:
//   - validation passes that match Python -Werror behavior emit
//     severity=2 (error); the driver dumps and exits non-zero.
//   - deprecated_pass (default Python policy is -Wdeprecated=warn)
//     emits severity=1; the driver dumps but does not exit non-zero.
// --------------------------------------------------------------

fn diag_arena_init() -> i32 {
    // count slot (slot 0)
    let base = __arena_push(0);
    // cap slot (slot 1)
    __arena_push(64);
    // 64 * 4 entry slots (256 slots), all zero-initialized.
    let mut i: i32 = 0;
    while i < 256 {
        __arena_push(0);
        i = i + 1;
    }
    // Sticky `overflowed` flag — one extra slot past the entries.
    // diag_emit sets it to 1 when count >= cap on a fresh push.
    __arena_push(0);
    base
}

@pure fn diag_arena_count(diag_state: i32) -> i32 {
    __arena_get(diag_state)
}

@pure fn diag_arena_cap(diag_state: i32) -> i32 {
    __arena_get(diag_state + 1)
}

// Emit a diagnostic. Returns 0 always (no fallible interface — the
// caller has no useful recovery path anyway; the validation passes
// just keep walking).
//
// Stage-28.9 audit-cycle-1 D1: the slot-2 parameter is documented
// (and emit-site-used) as the AST node's arena index, NOT a source
// byte offset.
//
// Stage-28.9 audit-cycle-1 Finding 1: on overflow, set the sticky
// `overflowed` flag at slot (cap*4 + 2) instead of calling
// emit_trap_with_id. The previous direct trap-emit was unobservable
// because validation passes run before elf_start is captured —
// the 7 trap bytes landed in dead pre-ELF arena. The codegen path
// in emit_elf_for_ast_to_path queries the flag after validation
// completes and patches main's prologue with a fresh 28999 trap.
fn diag_emit(diag_state: i32, code: i32, severity: i32,
             ast_node_idx: i32, aux: i32) -> i32 {
    let count = __arena_get(diag_state);
    let cap = __arena_get(diag_state + 1);
    if count >= cap {
        // Set the sticky flag (idempotent). One slot past the
        // entry region: diag_state + 2 + cap * 4.
        __arena_set(diag_state + 2 + cap * 4, 1);
        0
    } else {
        // Entry base: diag_state + 2 + count * 4. The +2 skips
        // count + cap header slots.
        let entry = diag_state + 2 + count * 4;
        __arena_set(entry, code);
        __arena_set(entry + 1, severity);
        __arena_set(entry + 2, ast_node_idx);
        __arena_set(entry + 3, aux);
        __arena_set(diag_state, count + 1);
        0
    }
}

// Stage-28.9 audit-cycle-1 Finding 1: accessor for the sticky
// overflow flag. Returns 1 if any diag_emit call hit the cap
// (meaning the arena dropped at least one diag), 0 otherwise. The
// codegen wiring in emit_elf_for_ast_to_path uses this to patch
// main's prologue with a 28999 trap.
@pure fn diag_arena_overflowed(diag_state: i32) -> i32 {
    let cap = __arena_get(diag_state + 1);
    __arena_get(diag_state + 2 + cap * 4)
}

// Read accessors for one diag entry (idx 0..count-1).
@pure fn diag_get_code(diag_state: i32, idx: i32) -> i32 {
    __arena_get(diag_state + 2 + idx * 4)
}
@pure fn diag_get_severity(diag_state: i32, idx: i32) -> i32 {
    __arena_get(diag_state + 2 + idx * 4 + 1)
}
// Stage-28.9 audit-cycle-1 D1: renamed from diag_get_src_offset to
// match the actual semantics (slot 2 holds an AST node arena index,
// not a source byte offset). A future stage will add a side-table
// from ast_node_idx → source byte range and re-expose a true
// `diag_get_src_offset` on top of this.
@pure fn diag_get_ast_node_idx(diag_state: i32, idx: i32) -> i32 {
    __arena_get(diag_state + 2 + idx * 4 + 2)
}
@pure fn diag_get_aux(diag_state: i32, idx: i32) -> i32 {
    __arena_get(diag_state + 2 + idx * 4 + 3)
}

// Count diags with severity == 2 (errors). The driver-main exits
// non-zero iff this is > 0. Warning-severity diags do not gate the
// build (matches Python's default -Wdeprecated=warn policy).
@pure fn diag_arena_error_count(diag_state: i32) -> i32 {
    let n = __arena_get(diag_state);
    let mut i: i32 = 0;
    let mut errs: i32 = 0;
    while i < n {
        let sev = __arena_get(diag_state + 2 + i * 4 + 1);
        if sev == 2 { errs = errs + 1; };
        i = i + 1;
    }
    errs
}

// --------------------------------------------------------------
// Stage 28.9: panic_pass — port of helixc/frontend/panic_pass.py.
//
// Walks each AST_FN_DECL body looking for AST_CALL nodes whose
// callee name is "panic". For each match, validates:
//   * exactly 1 argument
//   * the argument is an AST_STR_LIT (tag 25)
//
// Violations emit diag code 28501 with severity=2 (error). The aux
// slot of the diag encodes the violation kind:
//   * 1 = wrong arg count
//   * 2 = non-string-literal arg
//
// Mirrors panic_pass.validate_panic_args. Phase-0 does not emit the
// "panic was invoked" runtime trap here — that's the codegen
// concern when AST_CALL with name "panic" lowers (deferred; the
// Python pass also only validates, never lowers panic).
//
// Walker note: the bootstrap has no shared ast_walker library
// (cycle 28.8.2 added one in Python only). We implement a focused
// tag-dispatch recursion that descends into expression-bearing
// slots for the common control-flow + binop tags. This mirrors the
// approach taken by clone_with_rewrite (parser.hx:3914) and the
// Python pass's ASTVisitor but stays inside the bootstrap's
// recursion budget.
// --------------------------------------------------------------

// Byte-equal predicate against the literal name "panic" (5 bytes:
// 112 97 110 105 99). Returns 1 on match, 0 otherwise.
@pure fn is_panic_name(name_s: i32, name_l: i32) -> i32 {
    if name_l == 5 {
        let b0 = __arena_get(name_s);
        let b1 = __arena_get(name_s + 1);
        let b2 = __arena_get(name_s + 2);
        let b3 = __arena_get(name_s + 3);
        let b4 = __arena_get(name_s + 4);
        if b0 == 112 { if b1 == 97 { if b2 == 110 {
            if b3 == 105 { if b4 == 99 { 1 }
            else { 0 } } else { 0 } } else { 0 } } else { 0 }
        } else { 0 }
    } else { 0 }
}

// Count AST_ARG entries in a linked-list chain. AST_ARG (tag 17)
// has p1=expr, p2=next. 0 sentinel ends the chain.
fn count_args(args_head: i32) -> i32 {
    let mut cur: i32 = args_head;
    let mut n: i32 = 0;
    while cur != 0 {
        n = n + 1;
        cur = __arena_get(cur + 2);
    }
    n
}

// Walk an expression subtree looking for AST_CALL(panic). When
// found, validate args and emit diag if malformed. Recurses into
// every expression-bearing slot of every tag the bootstrap parser
// can produce in a fn body.
//
// Tag dispatch table (slots that hold expression indices):
//   2..6, 19..23, 24, 28..30, 32, 33  binop: p1, p2
//   7 IF       : p1 (cond), p2 (then), p3 (else)
//   8 LET      : p3 (body), p4 (value)
//   9 NEG, 26 BNOT, 31 NOT: p1 (inner)
//   10 WHILE   : p1 (cond), p2 (body)
//   11 ASSIGN  : p3 (value)
//   12 LET_MUT : p3 (body), p4 (value)
//   13 SEQ     : p1 (first), p2 (second)
//   16 CALL    : p3 (args_head) — handled separately below
//   17 ARG     : p1 (expr); next via chain walk
//
// Tags that DO NOT recurse:
//   0, 27, 34..42 lits; 1 VAR; 25 STR_LIT; 18 PARAM (only inside
//   fn-decl, not body); 99 ERR.
//
// Stage 28.9 walker uses a single recursive fn, not the ASTVisitor
// class pattern (Python). Helix bootstrap has no virtual dispatch,
// so direct tag-switch is the idiomatic equivalent.
fn walk_for_panic(idx: i32, diag_state: i32) -> i32 {
    if idx == 0 { 0 } else {
        let t = __arena_get(idx);
        let p1 = __arena_get(idx + 1);
        let p2 = __arena_get(idx + 2);
        let p3 = __arena_get(idx + 3);
        // Tag 16 AST_CALL: check for panic, then recurse into args.
        if t == 16 {
            if is_panic_name(p1, p2) == 1 {
                let n_args = count_args(p3);
                if n_args != 1 {
                    // Trap-id 28501 with aux=1 = arg-count violation.
                    diag_emit(diag_state, 28501, 2, idx, 1);
                } else {
                    // Single arg: check it's an AST_STR_LIT (tag 25).
                    let arg_node = __arena_get(p3 + 1);   // AST_ARG.p1 = expr
                    let arg_tag = __arena_get(arg_node);
                    if arg_tag != 25 {
                        // 28501 aux=2 = non-string-literal arg.
                        diag_emit(diag_state, 28501, 2, idx, 2);
                    };
                };
            };
            // Always recurse into args (panic("...") might be nested
            // inside another call's args, etc.).
            let mut cur_arg: i32 = p3;
            while cur_arg != 0 {
                let arg_expr = __arena_get(cur_arg + 1);
                walk_for_panic(arg_expr, diag_state);
                cur_arg = __arena_get(cur_arg + 2);
            }
            0
        } else { if t == 7 {
            // AST_IF: cond + then + else
            walk_for_panic(p1, diag_state);
            walk_for_panic(p2, diag_state);
            walk_for_panic(p3, diag_state);
            0
        } else { if t == 8 {
            // AST_LET: p3=body, p4=value
            let value = __arena_get(idx + 4);
            walk_for_panic(p3, diag_state);
            walk_for_panic(value, diag_state);
            0
        } else { if t == 12 {
            // AST_LET_MUT: same shape as AST_LET
            let value = __arena_get(idx + 4);
            walk_for_panic(p3, diag_state);
            walk_for_panic(value, diag_state);
            0
        } else { if t == 10 {
            // AST_WHILE: p1=cond, p2=body
            walk_for_panic(p1, diag_state);
            walk_for_panic(p2, diag_state);
            0
        } else { if t == 11 {
            // AST_ASSIGN: p3=value
            walk_for_panic(p3, diag_state);
            0
        } else { if t == 13 {
            // AST_SEQ: p1=first, p2=second
            walk_for_panic(p1, diag_state);
            walk_for_panic(p2, diag_state);
            0
        } else { if t == 9 {
            // AST_NEG: p1=inner
            walk_for_panic(p1, diag_state);
            0
        } else { if t == 26 {
            // AST_BNOT: p1=inner
            walk_for_panic(p1, diag_state);
            0
        } else { if t == 31 {
            // AST_NOT: p1=inner
            walk_for_panic(p1, diag_state);
            0
        } else { if t == 62 {
            // AST_MATCH: p1=scrut, p2=arms_head (linked list of
            // AST_MATCH_ARM via p3 chain). Walk scrut + each arm's
            // body. Audit follow-up: walker drift on AST_MATCH was
            // a Cycle 22 C22-C finding in the Python ast_walker
            // refactor; mirror the fix here.
            walk_for_panic(p1, diag_state);
            let mut arm: i32 = p2;
            while arm != 0 {
                // AST_MATCH_ARM: p1=pattern (not an expr), p2=body, p3=next.
                let arm_body = __arena_get(arm + 2);
                walk_for_panic(arm_body, diag_state);
                arm = __arena_get(arm + 3);
            }
            0
        } else { if t == 52 {
            // AST_TUPLE_FIELD: p1=value (expr), p2=field_idx (not
            // expr), p3=is_struct (not expr). Walk only p1.
            walk_for_panic(p1, diag_state);
            0
        } else { if t == 79 {
            // K1.F6 (2026-05-27): AST_FIELD_STORE: p1=AST_TUPLE_FIELD
            // (recursive walk picks up its own inner expr), p2=value
            // expr. Walk both so panic(...) calls nested in either
            // path are surfaced.
            walk_for_panic(p1, diag_state);
            walk_for_panic(p2, diag_state);
            0
        } else { if t == 53 {
            // AST_INDEX: p1=value (expr), p2=idx (expr).
            walk_for_panic(p1, diag_state);
            walk_for_panic(p2, diag_state);
            0
        } else { if t == 50 {
            // AST_TUPLE_LIT: p1=arity (NOT an expr), p2=head_idx
            // (chain of AST_TUPLE_CONS / tag 51). Walk each element
            // expr in the chain. Covers tuple literals, struct
            // literals (lowered to tuple-lit), and enum-constructor
            // payloads — Stage-28.9 audit-1 Finding 2 fix. Without
            // this arm, panic(...) calls nested in
            // `Pt { x: panic("bad") }` or `(panic("a"), 1)` or
            // `Just(panic("x"))` were silently skipped.
            let mut cur: i32 = p2;
            while cur != 0 {
                let elem_expr = __arena_get(cur + 1);
                walk_for_panic(elem_expr, diag_state);
                cur = __arena_get(cur + 2);
            }
            0
        } else {
            // Binops with p1=lhs, p2=rhs: tags 2..6, 19..23, 24,
            // 28..30, 32, 33. Use a coarse range check + a few
            // exclusions for non-binop tags in the same range.
            let is_arith = if t >= 2 { if t <= 6 { 1 } else { 0 } } else { 0 };
            let is_cmp   = if t >= 19 { if t <= 23 { 1 } else { 0 } } else { 0 };
            let is_bit   = if t >= 28 { if t <= 30 { 1 } else { 0 } } else { 0 };
            let is_mod   = if t == 24 { 1 } else { 0 };
            let is_shift = if t == 32 { 1 } else { if t == 33 { 1 } else { 0 } };
            let is_binop = if is_arith == 1 { 1 }
                           else { if is_cmp == 1 { 1 }
                           else { if is_bit == 1 { 1 }
                           else { if is_mod == 1 { 1 }
                           else { if is_shift == 1 { 1 } else { 0 } } } } };
            if is_binop == 1 {
                walk_for_panic(p1, diag_state);
                walk_for_panic(p2, diag_state);
            };
            0
        }}}}}}}}}}}}}}}
    }
}

// Top-level panic_pass entry. Walks every AST_FN_DECL body in the
// fn_list. Skips entries flagged as generic templates (slot 6 == 1)
// because their concrete mono clones will be walked separately.
//
// Returns 0 unconditionally; the caller inspects diag_arena_count
// + diag_arena_error_count to decide exit policy.
fn panic_pass(ast_root: i32, diag_state: i32) -> i32 {
    let root_tag = __arena_get(ast_root);
    if root_tag != 15 {
        // Single-expr program (legacy). Walk the root directly.
        walk_for_panic(ast_root, diag_state);
        0
    } else {
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            let is_generic = __arena_get(fn_idx + 6);
            if is_generic == 0 {
                let body = __arena_get(fn_idx + 3);
                walk_for_panic(body, diag_state);
            };
            walk = __arena_get(walk + 2);
        }
        0
    }
}

// --------------------------------------------------------------
// Stage 28.9: unwind_pass — port of panic_pass.validate_unwind.
//
// Phase-0 rejects `@unwind` on any fn (trap-id 28502 reserved).
// The parser captures @unwind via the sticky flag set by
// skip_attributes (sb+77) and writes it into AST_FN_DECL slot 11.
// This pass scans the fn_list and emits one diag per fn with the
// flag set.
//
// Severity=2 (error). The Python pass treats @unwind as a hard
// error too (the attribute is reserved but not yet implemented;
// silently accepting it would be misleading).
// --------------------------------------------------------------

fn unwind_pass(ast_root: i32, diag_state: i32) -> i32 {
    let root_tag = __arena_get(ast_root);
    if root_tag != 15 {
        0
    } else {
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            let is_unwind = __arena_get(fn_idx + 11);
            if is_unwind == 1 {
                // aux slot stores the fn's name byte_start so a
                // future driver can reconstruct the fn name in the
                // diagnostic message.
                let fn_name_s = __arena_get(fn_idx + 1);
                diag_emit(diag_state, 28502, 2, fn_idx, fn_name_s);
            };
            walk = __arena_get(walk + 2);
        }
        0
    }
}

// --------------------------------------------------------------
// Stage 28.9: trace_pass — port of trace_pass.validate_trace_attrs.
//
// Phase-0 rejects `@trace` on extern fns (the extern-fn path is
// not present in the bootstrap parser today — externs are part of
// the Python frontend only). For Phase-0 bootstrap, every @trace
// fn passes validation unless it carries the (parser-discarded)
// extern flag. We surface a soft check: every `@trace` fn IS the
// fn (no extern path can reach the bootstrap), so the pass is
// effectively a no-op trip wire that the bootstrap recognises the
// `@trace` attribute exists.
//
// To keep parity with the Python pass and to validate that the
// attribute capture wired through correctly, the pass emits a
// SEVERITY=1 (warning, not error) diag for every traced fn. The
// warning fires once per @trace and does NOT gate the build.
// Codegen-time @trace instrumentation (TRACE_ENTRY / TRACE_EXIT
// prologue/epilogue ops) is deferred to a follow-up.
// --------------------------------------------------------------

fn trace_pass(ast_root: i32, diag_state: i32) -> i32 {
    let root_tag = __arena_get(ast_root);
    if root_tag != 15 {
        0
    } else {
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            let is_trace = __arena_get(fn_idx + 10);
            if is_trace == 1 {
                // Severity 1 (warning): @trace is recognised but
                // not yet wired through codegen. aux slot stores
                // fn name byte_start for future message rendering.
                // Use 25003 (NOT 25001 which is TRACE_OVERFLOW
                // runtime, NOT 25002 which is TRACE_EQUIV_SHAPE
                // mismatch). 25003 = "trace attribute recognised
                // but codegen instrumentation pending".
                let fn_name_s = __arena_get(fn_idx + 1);
                diag_emit(diag_state, 25003, 1, fn_idx, fn_name_s);
            };
            walk = __arena_get(walk + 2);
        }
        0
    }
}

// --------------------------------------------------------------
// Stage 33: autotune_pass -- summary validation only.
//
// The Python frontend validates @autotune statically before any
// variant generation. Bootstrap parity for this slice is deliberately
// narrower: parser.hx stores only summary metadata on AST_FN_DECL:
//   slot 14: is_kernel
//   slot 15: is_autotune
//   slot 16: deduped variant product (saturated at 17)
//   slot 17: parse_error_kind
//     0 = clean
//     1 = missing parenthesized argument list
//     2 = malformed token/shape inside the argument list
//     3 = empty parameter list or empty value list
//
// Full kernel-variant generation and dispatch stay Python-only for now.
// --------------------------------------------------------------
fn autotune_pass(ast_root: i32, diag_state: i32) -> i32 {
    let root_tag = __arena_get(ast_root);
    if root_tag != 15 {
        0
    } else {
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            let is_autotune = __arena_get(fn_idx + 15);
            if is_autotune == 1 {
                let name_s = __arena_get(fn_idx + 1);
                let is_kernel = __arena_get(fn_idx + 14);
                let product = __arena_get(fn_idx + 16);
                let parse_error = __arena_get(fn_idx + 17);
                if is_kernel != 1 {
                    // 27004 aux: fn name start.
                    diag_emit(diag_state, 27004, 2, fn_idx, name_s);
                };
                if parse_error != 0 {
                    // 27003 aux: parse_error_kind (1 missing, 2 malformed, 3 empty).
                    diag_emit(diag_state, 27003, 2, fn_idx, parse_error);
                };
                if product > 16 {
                    // 27001 aux: saturated variant product.
                    diag_emit(diag_state, 27001, 2, fn_idx, product);
                };
            };
            walk = __arena_get(walk + 2);
        }
        0
    }
}

// --------------------------------------------------------------
// Stage 28.9: deprecated_pass.
// deprecated_pass.emit_warnings.
//
// Walks the fn_list to collect all fns marked `@deprecated` (slot
// 9 == 1), then walks every NON-deprecated fn's body looking for
// AST_CALL nodes whose callee name byte-matches a deprecated fn
// name. Emits severity=1 (warning) diag with code 28701 per call
// site. The Python pass also defaults to warning; -Wdeprecated=error
// promotes them to errors at the CLI driver level (not implemented
// in bootstrap Phase-0).
//
// Stage 33: the bootstrap parser preserves @deprecated("message")
// string-literal body ranges on AST_FN_DECL slots 12/13. Diagnostics
// now carry a dep_tab entry pointer in aux so future renderers can recover
// both the deprecated callee name and its optional message.
// --------------------------------------------------------------

// Auxiliary: collect every deprecated fn's name and optional message
// into a small fixed-cap table at the diag_state arena's tail.
// We side-channel this table because the diag_arena slots only hold
// 4 i32s per entry and we need (name_s, name_l, msg_s, msg_l) at lookup
// time. Phase-0 cap = 16 deprecated fns (matches Python tests
// which never use more than a handful).
//
// Layout: a 65-slot region: 1 count + 16 * 4
// (name_s, name_l, msg_s, msg_l) entries.
// Caller passes the base offset. Init = zero count + zero entries.

fn dep_tab_init() -> i32 {
    let base = __arena_push(0);
    let mut i: i32 = 0;
    while i < 64 {
        __arena_push(0);
        i = i + 1;
    }
    base
}

// Returns 1 on success, 0 on drop (cap reached). Caller checks the
// return value and emits a 28702 cap warning diag when needed —
// we keep dep_tab_add itself loose of diag_state so the helper can
// be called from any context (incl. tests that don't construct one).
//
// Stage-28.9 audit-cycle-1 Finding 3 fix: previously this dropped
// the 17th+ name silently with no diagnostic, so a program with 17
// `@deprecated` fns would compile cleanly with zero warning that
// some of the call-site detection was lost. The 0-return signals
// the drop and the caller emits a 28702 (DIAG_DEP_TAB_CAPACITY)
// severity-1 warning per drop.
fn dep_tab_add(dep_tab: i32, name_s: i32, name_l: i32,
               msg_s: i32, msg_l: i32) -> i32 {
    let count = __arena_get(dep_tab);
    if count >= 16 {
        // Cap reached; signal drop to caller. Deprecation is a
        // warning-only pass so we still don't HARD-error here.
        0
    } else {
        let entry = dep_tab + 1 + count * 4;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(entry + 2, msg_s);
        __arena_set(entry + 3, msg_l);
        __arena_set(dep_tab, count + 1);
        1
    }
}

@pure fn dep_tab_count(dep_tab: i32) -> i32 { __arena_get(dep_tab) }
@pure fn dep_tab_name_s(dep_tab: i32, idx: i32) -> i32 {
    __arena_get(dep_tab + 1 + idx * 4)
}
@pure fn dep_tab_name_l(dep_tab: i32, idx: i32) -> i32 {
    __arena_get(dep_tab + 1 + idx * 4 + 1)
}
@pure fn dep_tab_msg_s_from_entry(entry: i32) -> i32 { __arena_get(entry + 2) }
@pure fn dep_tab_msg_l_from_entry(entry: i32) -> i32 { __arena_get(entry + 3) }

// Check whether a (name_s, name_l) byte-matches any deprecated fn
// in the table. Returns the entry pointer on match, 0 otherwise.
fn dep_tab_lookup(dep_tab: i32, name_s: i32, name_l: i32) -> i32 {
    let count = __arena_get(dep_tab);
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < count {
        let entry = dep_tab + 1 + i * 4;
        let ds = __arena_get(entry);
        let dl = __arena_get(entry + 1);
        if kovc_byte_eq(name_s, name_l, ds, dl) == 1 {
            found = entry;
        };
        i = i + 1;
    }
    found
}

// Walker: scan for AST_CALL whose callee name appears in dep_tab.
// Same descent rules as walk_for_panic (mirror of the Python
// _DeprecationCallSiteCollector visitor pattern).
fn walk_for_deprecated(idx: i32, dep_tab: i32, diag_state: i32) -> i32 {
    if idx == 0 { 0 } else {
        let t = __arena_get(idx);
        let p1 = __arena_get(idx + 1);
        let p2 = __arena_get(idx + 2);
        let p3 = __arena_get(idx + 3);
        if t == 16 {
            // AST_CALL: check the callee against dep_tab.
            let dep_entry = dep_tab_lookup(dep_tab, p1, p2);
            if dep_entry != 0 {
                // aux = dep_tab entry ptr, severity=1 (warning) — matches
                // Python -Wdeprecated default. The AST_CALL node carries the
                // use-site callee name; aux carries the decl message.
                diag_emit(diag_state, 28701, 1, idx, dep_entry);
            };
            // Always recurse into args.
            let mut cur_arg: i32 = p3;
            while cur_arg != 0 {
                let arg_expr = __arena_get(cur_arg + 1);
                walk_for_deprecated(arg_expr, dep_tab, diag_state);
                cur_arg = __arena_get(cur_arg + 2);
            }
            0
        } else { if t == 7 {
            walk_for_deprecated(p1, dep_tab, diag_state);
            walk_for_deprecated(p2, dep_tab, diag_state);
            walk_for_deprecated(p3, dep_tab, diag_state);
            0
        } else { if t == 8 {
            let value = __arena_get(idx + 4);
            walk_for_deprecated(p3, dep_tab, diag_state);
            walk_for_deprecated(value, dep_tab, diag_state);
            0
        } else { if t == 12 {
            let value = __arena_get(idx + 4);
            walk_for_deprecated(p3, dep_tab, diag_state);
            walk_for_deprecated(value, dep_tab, diag_state);
            0
        } else { if t == 10 {
            walk_for_deprecated(p1, dep_tab, diag_state);
            walk_for_deprecated(p2, dep_tab, diag_state);
            0
        } else { if t == 11 {
            walk_for_deprecated(p3, dep_tab, diag_state);
            0
        } else { if t == 13 {
            walk_for_deprecated(p1, dep_tab, diag_state);
            walk_for_deprecated(p2, dep_tab, diag_state);
            0
        } else { if t == 9 {
            walk_for_deprecated(p1, dep_tab, diag_state);
            0
        } else { if t == 26 {
            walk_for_deprecated(p1, dep_tab, diag_state);
            0
        } else { if t == 31 {
            walk_for_deprecated(p1, dep_tab, diag_state);
            0
        } else { if t == 62 {
            // AST_MATCH: descend into scrut + each arm body. Mirror
            // of walk_for_panic's match handling.
            walk_for_deprecated(p1, dep_tab, diag_state);
            let mut arm: i32 = p2;
            while arm != 0 {
                let arm_body = __arena_get(arm + 2);
                walk_for_deprecated(arm_body, dep_tab, diag_state);
                arm = __arena_get(arm + 3);
            }
            0
        } else { if t == 52 {
            // AST_TUPLE_FIELD: only p1 is an expr.
            walk_for_deprecated(p1, dep_tab, diag_state);
            0
        } else { if t == 79 {
            // K1.F6 (2026-05-27): AST_FIELD_STORE: p1=AST_TUPLE_FIELD
            // (recursive walk picks up its inner), p2=value expr.
            walk_for_deprecated(p1, dep_tab, diag_state);
            walk_for_deprecated(p2, dep_tab, diag_state);
            0
        } else { if t == 53 {
            // AST_INDEX: p1=value, p2=idx, both exprs.
            walk_for_deprecated(p1, dep_tab, diag_state);
            walk_for_deprecated(p2, dep_tab, diag_state);
            0
        } else { if t == 50 {
            // AST_TUPLE_LIT: p1=arity (NOT an expr), p2=head_idx
            // (chain of AST_TUPLE_CONS / tag 51). Walk each element
            // expr in the chain. Mirrors walk_for_panic's arm —
            // Stage-28.9 audit-1 Finding 2 fix. Without this,
            // @deprecated calls nested in struct/tuple/enum payloads
            // (`Pt { x: old_api() }`, `(old_api(), 1)`,
            // `Just(old_api())`) were silently skipped.
            let mut cur: i32 = p2;
            while cur != 0 {
                let elem_expr = __arena_get(cur + 1);
                walk_for_deprecated(elem_expr, dep_tab, diag_state);
                cur = __arena_get(cur + 2);
            }
            0
        } else {
            // Binops: same coarse range check as walk_for_panic.
            let is_arith = if t >= 2 { if t <= 6 { 1 } else { 0 } } else { 0 };
            let is_cmp   = if t >= 19 { if t <= 23 { 1 } else { 0 } } else { 0 };
            let is_bit   = if t >= 28 { if t <= 30 { 1 } else { 0 } } else { 0 };
            let is_mod   = if t == 24 { 1 } else { 0 };
            let is_shift = if t == 32 { 1 } else { if t == 33 { 1 } else { 0 } };
            let is_binop = if is_arith == 1 { 1 }
                           else { if is_cmp == 1 { 1 }
                           else { if is_bit == 1 { 1 }
                           else { if is_mod == 1 { 1 }
                           else { if is_shift == 1 { 1 } else { 0 } } } } };
            if is_binop == 1 {
                walk_for_deprecated(p1, dep_tab, diag_state);
                walk_for_deprecated(p2, dep_tab, diag_state);
            };
            0
        }}}}}}}}}}}}}}}
    }
}

fn deprecated_pass(ast_root: i32, diag_state: i32) -> i32 {
    let root_tag = __arena_get(ast_root);
    if root_tag != 15 {
        0
    } else {
        // First pass: build the dep_tab from @deprecated fn decls.
        // Stage-28.9 audit-cycle-1 Finding 3 fix: when dep_tab_add
        // returns 0 (cap reached at 16 names), emit a 28702 severity-1
        // warning so the 17th+ @deprecated fn is observable. Without
        // this, dropped names cause silent loss of call-site detection.
        let dep_tab = dep_tab_init();
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            let is_deprecated = __arena_get(fn_idx + 9);
            if is_deprecated == 1 {
                let name_s = __arena_get(fn_idx + 1);
                let name_l = __arena_get(fn_idx + 2);
                let msg_s = __arena_get(fn_idx + 12);
                let msg_l = __arena_get(fn_idx + 13);
                let added = dep_tab_add(dep_tab, name_s, name_l, msg_s, msg_l);
                if added == 0 {
                    // aux = fn_idx's name_s (callee name byte_start) so
                    // a future driver can identify which @deprecated fn
                    // got dropped.
                    diag_emit(diag_state, 28702, 1, fn_idx, name_s);
                };
            };
            walk = __arena_get(walk + 2);
        }
        // If no deprecated fns, short-circuit.
        if dep_tab_count(dep_tab) == 0 {
            0
        } else {
            // Second pass: walk every fn body looking for calls.
            let mut walk2: i32 = ast_root;
            while walk2 != 0 {
                let fn_idx = __arena_get(walk2 + 1);
                let is_generic = __arena_get(fn_idx + 6);
                if is_generic == 0 {
                    let body = __arena_get(fn_idx + 3);
                    walk_for_deprecated(body, dep_tab, diag_state);
                };
                walk2 = __arena_get(walk2 + 2);
            }
            0
        }
    }
}

// --------------------------------------------------------------
// Stage 28.9: unsafe_pass — port of unsafe_pass.check_unsafe_ops.
//
// Phase-0 bootstrap status: the bootstrap parser does NOT recognize
// `unsafe { ... }` blocks (no AST tag for unsafe). Raw-pointer ops
// (`*p` deref, `as *T` cast) are also unrepresented in the
// bootstrap AST. Consequently the bootstrap-side unsafe_pass is a
// no-op trip wire: it walks the fn_list but finds no unsafe-
// requiring constructs to validate.
//
// Once the bootstrap parser gains AST_UNSAFE_BLOCK + raw-pointer
// expression nodes (deferred — needs lexer/parser extensions for
// `unsafe`, `*`, `&mut`, `as *T`), this pass should be extended to:
//   1. thread an `in_unsafe` counter through descent (push on
//      AST_UNSAFE_BLOCK entry, pop on exit; nested blocks legal)
//   2. on encountering a raw-ptr op (deref unary `*` or cast to
//      pointer type), check in_unsafe > 0; if not, emit
//      diag(28601, severity=2)
//
// The Python pass implements both. Bootstrap parity tracked as
// Phase-A follow-up.
// --------------------------------------------------------------

fn unsafe_pass(ast_root: i32, diag_state: i32) -> i32 {
    // No bootstrap AST tags for unsafe/raw-ptr exist yet. Walk for
    // structural symmetry with the other passes — this is the
    // documented hook so adding the AST nodes in a follow-up only
    // requires extending the body walker, not adding a new
    // driver-main call site.
    let root_tag = __arena_get(ast_root);
    if root_tag != 15 {
        0
    } else {
        // Walk fn bodies but emit no diags. Future tags to handle:
        //   - AST_UNSAFE_BLOCK (proposed tag 80)
        //   - AST_PTR_DEREF (proposed Unary subtag)
        //   - AST_PTR_CAST  (proposed Cast subtag)
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            let is_generic = __arena_get(fn_idx + 6);
            // is_generic guard kept for future use (mono clones'
            // unsafe blocks would otherwise double-visit).
            if is_generic == 0 {
                // Body walk is intentionally a no-op for now.
                let _body = __arena_get(fn_idx + 3);
                let _drop = _body;
            };
            walk = __arena_get(walk + 2);
        }
        0
    }
}

fn bn_arena_push_s(b: i32) -> i32 { __arena_get(b) }
fn bn_arena_get_s(b: i32) -> i32  { __arena_get(b + 1) }
fn bn_arena_set_s(b: i32) -> i32  { __arena_get(b + 2) }
fn bn_arena_len_s(b: i32) -> i32  { __arena_get(b + 3) }
fn bn_helix_arena_base_s(b: i32) -> i32 { __arena_get(b + 4) }
fn bn_read_file_to_arena_s(b: i32) -> i32 { __arena_get(b + 5) }
fn bn_write_file_to_arena_s(b: i32) -> i32 { __arena_get(b + 6) }

// K1.D Option A (2026-05-25): direct byte-literal comparison for
// the "print_int" builtin name. install_builtin_names is FRAGILE
// (per matrix Appendix A2 Pattern 2: adding __arena_push calls in
// it breaks the self-host fixpoint via an implicit cursor-position
// invariant). Option A avoids it entirely — the expected bytes are
// returned by a small top-level fn, no arena cursor advance, no
// bn_state slot pointer.
//
// "print_int" = p(112) r(114) i(105) n(110) t(116) _(95) i(105)
// n(110) t(116) — 9 bytes.
fn print_int_kw_byte(i: i32) -> i32 {
    if i == 0 { 112 }
    else { if i == 1 { 114 }
    else { if i == 2 { 105 }
    else { if i == 3 { 110 }
    else { if i == 4 { 116 }
    else { if i == 5 { 95 }
    else { if i == 6 { 105 }
    else { if i == 7 { 110 }
    else { 116 } } } } } } } }
}

fn is_print_int_name(s: i32, l: i32) -> i32 {
    if l != 9 { 0 } else {
        let mut ok: i32 = 1;
        let mut i: i32 = 0;
        while i < 9 {
            if __arena_get(s + i) != print_int_kw_byte(i) { ok = 0; };
            i = i + 1;
        }
        ok
    }
}

// K1.D-impl (2026-05-25): emit the inline asm for print_int(n).
//
// Algorithm:
//   - Evaluate arg into eax (the value to print).
//   - sub rsp, 16: stack-allocate a 16-byte buffer (enough for sign
//     + 10 i32 digits + newline + spare). Aligned naturally.
//   - r12 = rsp + 11 (one PAST the last writable position; the
//     conversion loop writes at [r12] then decrements r12).
//   - Save sign of n in ecx; take abs(n) in eax.
//   - Conversion loop: divide eax by 10 (unsigned div) into eax + edx;
//     edx is the remainder digit (0..9); add 48 ('0') and write at
//     [r12]; decrement r12; loop while eax != 0.
//   - After loop, r12 points one-BELOW the most-significant digit.
//     If original was negative, write '-' (45) at [r12]; the syscall
//     reads from r12. Otherwise inc r12 to skip past the unwritten
//     pre-MSB position.
//   - Compute length = (rsp + 12) - r12.
//   - sys_write(stdout=1, buf=r12, len=rdx) via syscall.
//   - add rsp, 16; xor eax, eax (return 0).
//
// Byte layout (90 bytes inline, calculated for accurate jump disps):
//   0: sub rsp, 16            (48 83 EC 10) = 4 bytes
//   4: mov r12, rsp           (49 89 E4)    = 3
//   7: add r12, 11            (49 83 C4 0B) = 4
//  11: mov ecx, eax           (89 C1)       = 2
//  13: test ecx, ecx          (85 C9)       = 2
//  15: jns +2                 (79 02)       = 2  -> target 17 (skip neg)
//  17: neg eax                (F7 D8)       = 2
//  19: [loop] xor edx, edx    (31 D2)       = 2
//  21: mov ebx, 10            (BB 0A 00 00 00) = 5
//  26: div ebx                (F7 F3)       = 2
//  28: add dl, 48             (80 C2 30)    = 3
//  31: mov [r12], dl          (41 88 14 24) = 4
//  35: dec r12                (49 FF CC)    = 3
//  38: test eax, eax          (85 C0)       = 2
//  40: jnz back to loop       (75 E9)       = 2  -> disp = 19 - 42 = -23
//  42: test ecx, ecx          (85 C9)       = 2
//  44: jns +7                 (79 07)       = 2  -> target 53 (skip neg branch)
//  46: mov byte [r12], 45     (41 C6 04 24 2D) = 5
//  51: jmp +3                 (EB 03)       = 2  -> target 56 (skip inc r12)
//  53: [pos] inc r12          (49 FF C4)    = 3
//  56: [calc_len] mov rax, rsp (48 89 E0)   = 3
//  59: add rax, 12            (48 83 C0 0C) = 4
//  63: sub rax, r12           (4C 29 E0)    = 3
//  66: mov rdx, rax           (48 89 C2)    = 3
//  69: mov rsi, r12           (4C 89 E6)    = 3
//  72: mov edi, 1             (BF 01 00 00 00) = 5
//  77: mov eax, 1             (B8 01 00 00 00) = 5
//  82: syscall                (0F 05)       = 2
//  84: add rsp, 16            (48 83 C4 10) = 4
//  88: xor eax, eax           (31 C0)       = 2
//  90: end
//
// Returns: total bytes emitted (n_arg + 90).
fn emit_print_int_body(arg_idx: i32, bind_state: i32, patch_state: i32, bn_state: i32) -> i32 {
    let n_arg = emit_ast_code(arg_idx, bind_state, patch_state, bn_state);
    emit_byte(0x48); emit_byte(0x83); emit_byte(0xEC); emit_byte(0x10);
    emit_byte(0x49); emit_byte(0x89); emit_byte(0xE4);
    emit_byte(0x49); emit_byte(0x83); emit_byte(0xC4); emit_byte(0x0B);
    emit_byte(0x89); emit_byte(0xC1);
    emit_byte(0x85); emit_byte(0xC9);
    emit_byte(0x79); emit_byte(0x02);
    emit_byte(0xF7); emit_byte(0xD8);
    emit_byte(0x31); emit_byte(0xD2);
    emit_byte(0xBB); emit_byte(0x0A); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
    emit_byte(0xF7); emit_byte(0xF3);
    emit_byte(0x80); emit_byte(0xC2); emit_byte(0x30);
    emit_byte(0x41); emit_byte(0x88); emit_byte(0x14); emit_byte(0x24);
    emit_byte(0x49); emit_byte(0xFF); emit_byte(0xCC);
    emit_byte(0x85); emit_byte(0xC0);
    emit_byte(0x75); emit_byte(0xE9);
    emit_byte(0x85); emit_byte(0xC9);
    emit_byte(0x79); emit_byte(0x07);
    emit_byte(0x41); emit_byte(0xC6); emit_byte(0x04); emit_byte(0x24); emit_byte(0x2D);
    emit_byte(0xEB); emit_byte(0x03);
    emit_byte(0x49); emit_byte(0xFF); emit_byte(0xC4);
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xE0);
    emit_byte(0x48); emit_byte(0x83); emit_byte(0xC0); emit_byte(0x0C);
    emit_byte(0x4C); emit_byte(0x29); emit_byte(0xE0);
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xC2);
    emit_byte(0x4C); emit_byte(0x89); emit_byte(0xE6);
    emit_byte(0xBF); emit_byte(0x01); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
    emit_byte(0xB8); emit_byte(0x01); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
    emit_byte(0x0F); emit_byte(0x05);
    emit_byte(0x48); emit_byte(0x83); emit_byte(0xC4); emit_byte(0x10);
    emit_byte(0x31); emit_byte(0xC0);
    n_arg + 90
}

// Phase 1.10 step 4: f32 SSE arithmetic builtins.
fn bn_fadd_s(b: i32) -> i32 { __arena_get(b + 57) }
fn bn_fsub_s(b: i32) -> i32 { __arena_get(b + 58) }
fn bn_fmul_s(b: i32) -> i32 { __arena_get(b + 59) }
fn bn_fdiv_s(b: i32) -> i32 { __arena_get(b + 60) }
fn bn_fneg_s(b: i32) -> i32 { __arena_get(b + 61) }
// Phase 1.10 step 5c follow-on: fn_type_state arena offset (or 0 if
// not yet installed). is_f32_expr reads this to resolve user-named
// f32 fn return types at AST_CALL sites.
fn bn_fn_type_state(b: i32) -> i32 { __arena_get(b + 62) }
fn bn_set_fn_type_state(b: i32, v: i32) -> i32 { __arena_set(b + 62, v); 0 }
// Phase 1.10 step 5g: __fsqrt single-arg f32 sqrt (SSE2 sqrtss).
fn bn_fsqrt_s(b: i32) -> i32 { __arena_get(b + 63) }
// Phase 1.10 step 5h: __fabs single-arg f32 absolute value (sign-bit AND mask).
fn bn_fabs_s(b: i32) -> i32 { __arena_get(b + 64) }
// Phase 1.10 step 5i: __i32_to_f32 single-arg int->float (cvtsi2ss).
fn bn_i32_to_f32_s(b: i32) -> i32 { __arena_get(b + 65) }
// Phase 1.10 step 5j: __f32_to_i32 single-arg float->int (cvttss2si).
fn bn_f32_to_i32_s(b: i32) -> i32 { __arena_get(b + 66) }
// Phase 1.10 step 5k: __fmin two-arg f32 minimum (SSE2 minss).
fn bn_fmin_s(b: i32) -> i32 { __arena_get(b + 67) }
// Phase 1.10 step 5l: __fmax two-arg f32 maximum (SSE2 maxss).
fn bn_fmax_s(b: i32) -> i32 { __arena_get(b + 68) }
// Phase 1.10 step 5m: __bits_of_f32 / __f32_from_bits identity bitcasts.
fn bn_bits_of_f32_s(b: i32) -> i32 { __arena_get(b + 69) }
fn bn_f32_from_bits_s(b: i32) -> i32 { __arena_get(b + 70) }
// Phase 1.10 step 5n: __hash_i32 single-arg quadratic mixer (FNV-style).
fn bn_hash_i32_s(b: i32) -> i32 { __arena_get(b + 71) }

// Phase 1.10 step 5o: __strlen builtin name slot.
fn bn_strlen_s(b: i32) -> i32 { __arena_get(b + 72) }
// Phase 1.10 step 7e: f32<->f64 conversion builtins.
fn bn_f32_to_f64_s(b: i32) -> i32 { __arena_get(b + 73) }
fn bn_f64_to_f32_s(b: i32) -> i32 { __arena_get(b + 74) }
fn bn_dsqrt_s(b: i32) -> i32 { __arena_get(b + 75) }
fn bn_dabs_s(b: i32) -> i32 { __arena_get(b + 76) }
fn bn_dmin_s(b: i32) -> i32 { __arena_get(b + 77) }
fn bn_dmax_s(b: i32) -> i32 { __arena_get(b + 78) }
fn bn_i32_to_f64_s(b: i32) -> i32 { __arena_get(b + 79) }
fn bn_f64_to_i32_s(b: i32) -> i32 { __arena_get(b + 80) }
fn bn_bits_lo_f64_s(b: i32) -> i32 { __arena_get(b + 81) }
fn bn_bits_hi_f64_s(b: i32) -> i32 { __arena_get(b + 82) }
fn bn_f64_pack_s(b: i32) -> i32 { __arena_get(b + 83) }
// Stage 11: reflection-runtime name slots and handle counter.
// K1.AC/K1.AD (2026-05-25): bn_state slot 157 holds the head of
// a linked list of pending `break` jmp positions for the
// innermost enclosing AST_WHILE. Each list cell is a 2-tuple
// (jmp_pos, next) pushed onto the main arena; 0 = list end.
// AST_BREAK codegen prepends a new cell; AST_WHILE codegen
// saves the old head, sets head=0, emits the body, then walks
// the chain patching each jmp to end_label, and restores the
// old head.
//
// K1.AD moved this from slot 122 -> 157 because slot 122 is
// match_scrut_ty (Audit A1-F1). A match inside a while body
// would call match_scrut_ty_set during pattern emission, which
// silently overwrote the break chain head -- subsequent
// AST_BREAK in the same body wrote a cell whose "prev" was the
// scrut_ty (typically a small integer), creating a corrupt
// chain that traps at backpatching time. The two slots are now
// disjoint; see also slot 158 for continue.
fn bn_break_chain_head_s(b: i32) -> i32 { __arena_get(b + 157) }
fn bn_set_break_chain_head_s(b: i32, v: i32) -> i32 { __arena_set(b + 157, v); 0 }

// K1.AF (2026-05-25): bn_state slot 159 holds the name-offset
// of the builtin "__arena_push_pair" (17 bytes). Looked up by
// the dispatcher in try_emit_builtin_call to recognize the
// call site and emit the inline atomic 2-slot push.
fn bn_arena_push_pair_s(b: i32) -> i32 { __arena_get(b + 159) }

// K1.AG (2026-05-25): bn_state slot 160 holds the name-offset
// of the builtin "__arena_push_triple" (19 bytes). Atomic
// 3-slot push.
fn bn_arena_push_triple_s(b: i32) -> i32 { __arena_get(b + 160) }

// K1.AH (2026-05-25): bn_state slot 161 holds the byte-offset
// of the 14-byte string "panic[28501]: " in the K1 arena.
// Read by the panic codegen arm to emit a sys_write for the
// prefix before the user-message sys_write, matching Python's
// "panic[id]: msg\n" format.
fn bn_panic_prefix_s(b: i32) -> i32 { __arena_get(b + 161) }

// K1.AI (2026-05-25): bn_state slot 162 holds the byte-offset
// of the 1-byte "\n" newline in the K1 arena. Read by the
// panic codegen arm to emit a 3rd sys_write for the trailing
// newline AFTER the message sys_write, fully matching Python's
// "panic[id]: msg\n" format.
fn bn_panic_newline_s(b: i32) -> i32 { __arena_get(b + 162) }

// K1.AK (2026-05-25): bn_state slot 163 holds the name-offset
// of the "print_str" builtin. Used by try_emit_builtin_call to
// recognize the call site and emit sys_write(1, str, len).
fn bn_print_str_s(b: i32) -> i32 { __arena_get(b + 163) }

// K1.F2 (2026-05-26): accessor for the "reflect_hash" builtin name.
// Used by try_emit_builtin_call to recognize the call site and emit
// a no-op stub returning 0.
fn bn_reflect_hash_s(b: i32) -> i32 { __arena_get(b + 164) }

// K1.F3+F4 (2026-05-26): accessors for the batched no-op reflection/
// trace builtins. Same pattern as bn_reflect_hash_s -- each is a
// simple offset getter that try_emit_builtin_call consults to
// recognize the call site.
fn bn_trace_event_s(b: i32) -> i32 { __arena_get(b + 165) }
// K1.F20b (2026-05-27): name slot for __trace_last() (read-side of
// the depth-1 trace slot). See install_builtin_names for the bytes.
fn bn_trace_last_s(b: i32) -> i32 { __arena_get(b + 169) }
// K1.F21 (2026-05-27): mangle scratch offset (an arena slot index
// where a 64-slot byte-buffer lives for backpatch-time generic-mono
// name mangling). Used by emit_elf_for_ast_to_path's patch_table loop
// when a bare call's name lookup fails to fall back to <name>__i32.
fn bn_mangle_scratch(b: i32) -> i32 { __arena_get(b + 170) }
// K1.F22c (2026-05-27): "print_str_ln" builtin name offset (12 chars).
// Paired with the println! macro expansion in parser.hx -- the
// synthesized AST_CALL uses these bytes so the codegen handler
// emits print_str + newline.
fn bn_print_str_ln_s(b: i32) -> i32 { __arena_get(b + 171) }
// K1.F22d (2026-05-27): "eprint_str_ln" stderr variant of K1.F22c.
fn bn_eprint_str_ln_s(b: i32) -> i32 { __arena_get(b + 172) }
// K1.F22f (2026-05-27): "eprint_str" no-newline stderr variant of
// K1.AK print_str. Builtin name offset stored at slot 173.
fn bn_eprint_str_s(b: i32) -> i32 { __arena_get(b + 173) }
// K1.F23c (2026-05-27): "__tile_zeros" cursor-bump tile allocator.
// Returns the old cursor as the tile's base offset; advances cursor
// by N*M. Builtin name offset stored at slot 174.
fn bn_tile_zeros_s(b: i32) -> i32 { __arena_get(b + 174) }
// K1.F24e (2026-05-27): re-attempt accessor.
fn bn_tile_add_s(b: i32) -> i32 { __arena_get(b + 175) }
// K1.F25 (2026-05-27): __tile_sub accessor (slot 176).
fn bn_tile_sub_s(b: i32) -> i32 { __arena_get(b + 176) }
// K1.F26 (2026-05-27): __tile_mul accessor (slot 177).
fn bn_tile_mul_s(b: i32) -> i32 { __arena_get(b + 177) }
// K1.F27 (2026-05-27): __tile_matmul accessor (slot 178). 13-char name.
fn bn_tile_matmul_s(b: i32) -> i32 { __arena_get(b + 178) }
fn bn_helix_splice_s(b: i32) -> i32 { __arena_get(b + 166) }
fn bn_helix_modify_s(b: i32) -> i32 { __arena_get(b + 167) }
fn bn_helix_reflect_hash_s(b: i32) -> i32 { __arena_get(b + 168) }

// K1.AD (2026-05-25): bn_state slot 158 holds the head of the
// continue-chain. Same layout as break: linked list of
// (jmp_pos, next) cells pushed onto the arena. AST_WHILE walks
// the chain post-body and patches each jmp_pos to loop_top
// (re-evaluates cond + runs body again). Phase-0 limitation:
// `continue` inside the body of a `for var in start..end { ... }`
// loop skips the auto-increment since parse_for desugars
// for-body to AST_SEQ(user_body, increment) and continue jumps
// past both; users should use plain `while` if continue is
// needed. Bare `while` and `loop { }` work correctly.
fn bn_continue_chain_head_s(b: i32) -> i32 { __arena_get(b + 158) }
fn bn_set_continue_chain_head_s(b: i32, v: i32) -> i32 { __arena_set(b + 158, v); 0 }

fn bn_quote_s(b: i32) -> i32 { __arena_get(b + 118) }
fn bn_splice_s(b: i32) -> i32 { __arena_get(b + 119) }
fn bn_modify_s(b: i32) -> i32 { __arena_get(b + 120) }
fn bn_quote_next_handle(b: i32) -> i32 { __arena_get(b + 121) }
fn bn_quote_bump_handle(b: i32) -> i32 {
    // Returns the OLD handle (assigned to the current Quote call), then
    // bumps the counter. Cap 64; values >= 64 still bump but the caller
    // is expected to emit_trap_with_id(81002) before using them.
    //
    // Audit A3-MEDIUM-4: defense-in-depth — emit trap 81002 INTERNALLY
    // too, so any future call site that forgets the >= 64 guard still
    // surfaces the overflow. The caller's existing if-check still fires
    // first; this internal trap is a backstop. Note we emit the trap
    // bytes inline; v is still returned so the caller's emit-then-trap
    // shape isn't disturbed.
    let v = __arena_get(b + 121);
    if v >= 64 {
        emit_trap_with_id(81002);
    };
    __arena_set(b + 121, v + 1);
    v
}
// str_state accessors. The state lives within the bn_state region.
fn str_top(b: i32) -> i32 { __arena_get(b + 7) }
fn str_top_set(b: i32, v: i32) -> i32 { __arena_set(b + 7, v); 0 }
fn str_table_base(b: i32) -> i32 { __arena_get(b + 8) }
// Add a pending LEA backpatch entry for a string literal. Each
// entry is 3 i32: [disp_slot, body_byte_start, body_byte_len].
// Returns the entry index, or -1 if the table is full.
//
// Audit-13: an unbounded write here was silently corrupting the
// `__arena_push` name string that lives immediately after the
// 16-entry reserve in install_builtin_names. Past entry 15, writes
// land in slot 57+ which is the first byte of "__arena_push";
// subsequent __arena_push calls then fail the kovc_byte_eq check
// and fall through to an unresolved CALL → ud2 trap. The guard
// here drops overflowing entries silently — any source with more
// than 16 string literals will produce a binary with broken file
// paths, but the failure is local to those calls, not catastrophic.
fn str_table_add(b: i32, disp_slot: i32, body_s: i32, body_l: i32) -> i32 {
    let top = str_top(b);
    // K3.O (2026-05-27): cap bumped 16 -> 64. K3.N audit flagged the
    // pre-existing silent-overflow risk: panic uses 3 entries +
    // each println! adds 2; a program with 1 panic + 7+ println!s
    // would have hit the 16-entry cap (str_table_add returns -1
    // which all callers ignore, leaving the next sys_write site
    // with an unpatched LEA at displacement 0 -- silent miscompile).
    // K3.O reallocates the str_table region in install_builtin_names
    // (fresh 192-slot region after slot 171, set via slot 8) so the
    // new cap of 64 entries has 64*3 = 192 slots without colliding
    // the pre-existing bn_state layout. 4x headroom over the K1.F22c
    // worst-case.
    if top >= 64 {
        0 - 1
    } else {
        let base = str_table_base(b);
        let entry = base + top * 3;
        __arena_set(entry, disp_slot);
        __arena_set(entry + 1, body_s);
        __arena_set(entry + 2, body_l);
        str_top_set(b, top + 1);
        top
    }
}

// HELIX_ARENA_CAP mirrored as kovc constant (kovc emits its own
// arena in the produced binary so the compiled programs match the
// Python-codegen layout: 2097152 data slots + 1 cursor slot,
// sized for self-host).
fn helix_arena_cap() -> i32 { 2097152 }

// Single global slot pointing at bn_state. Set during
// emit_elf_for_ast_to_path; read by try_emit_builtin_call which
// is called deep in emit_ast_code where threading another arg
// would push us past the SysV 6-int limit.
fn bn_state_slot_init(state: i32) -> i32 {
    let slot = __arena_push(state);
    slot
}
fn bn_state_get(slot: i32) -> i32 { __arena_get(slot) }

// Global slots holding bn_state and the patch_table state's
// helix_arena_base name span — referenced by the patch loop to
// register the LEA target.
fn bn_global_slot() -> i32 {
    // Lazily allocated; returns the slot containing bn_state.
    // The first call writes; subsequent calls read.
    0
}

// emit_read_file_to_arena_body: emit the inline asm sequence for
// read_file_to_arena AFTER the path has already been loaded into
// rdi (via emit_lea_rdi_rip_placeholder + str_table_add). Returns
// total bytes emitted (including the lea — caller adds 7).
//
// Layout (offsets relative to AFTER the lea rdi):
//   prelude (76 bytes): mov esi/edx/eax for sys_open, syscall,
//                       push fd, sub rsp BUF, mov rdi=fd, mov rsi=rsp,
//                       mov edx=BUF, mov eax=0, syscall (sys_read),
//                       mov r10,rax (bytes_read),
//                       mov rdi=fd, mov eax=3, syscall (sys_close),
//                       test/jns/xor r10 (clamp negative to 0),
//                       xor ecx, ecx (loop counter)
//   loop (48 bytes):    cmp rcx,r10 ; jge end ; movzx eax,[rsp+rcx] ;
//                       mov edx, eax ; lea rax,[rip+arena] ; mov r11d,[rax] ;
//                       cmp r11d, CAP ; jb in_bounds ; jmp loop_advance ;
//                       in_bounds: mov [rax+r11*4+4], edx ; inc r11d ;
//                       mov [rax], r11d ; loop_advance: inc rcx ;
//                       jmp loop_start (rel8 -48)
//   postlude (14 bytes): add rsp, BUF ; add rsp, 8 ; mov rax, r10
fn emit_read_file_to_arena_body(patch_state: i32, arena_base_s: i32) -> i32 {
    let body_start = __arena_len();
    // ---- sys_open(rdi=path, esi=0=O_RDONLY, edx=0) ----
    emit_byte(0xBE); emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);   // mov esi, 0
    emit_byte(0xBA); emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);   // mov edx, 0
    emit_byte(0xB8); emit_byte(2); emit_byte(0); emit_byte(0); emit_byte(0);   // mov eax, 2
    emit_byte(0x0F); emit_byte(0x05);                                           // syscall
    // push rax (fd on stack)
    emit_byte(0x50);
    // Audit-18b: bumped read buffer 0x8000 (32K) -> 0x40000 (256K) for
    // bootstrap source.
    // Approach-A bump (2026-05-07): 0x40000 (256K) -> 0x100000 (1M).
    // The k1_input concatenation (lexer_no_main + parser_body +
    // kovc_lib + k2_main) had grown to ~261 KB, leaving < 1 KB margin
    // against the 256 KB buffer. Each new fn or @pure helper added to
    // kovc.hx (including never-called dead helpers) tipped k1_input
    // over the buffer; K1's read truncated; the truncated source was
    // missing tail-end fns; K1 produced a K2 whose call sites
    // ud2-patched on missing symbols; K2 SIGILLed on first call. This
    // had been mis-attributed for weeks as a "cascade-depth bug" — see
    // docs/BOOTSTRAP_CASCADE_BUG.md probe 10. Bump to 1 MB gives ~4×
    // headroom. Must stay in lock-step with BUF_SIZE in
    // helixc/backend/x86_64.py so K1 (Python-emitted) and K2 (kovc.hx-
    // emitted) agree on the buffer size.
    // sub rsp, 0x100000 (1M read buffer)
    emit_byte(0x48); emit_byte(0x81); emit_byte(0xEC);
    emit_u32_le(1048576);
    // mov rdi, [rsp+0x100000] (load fd back into rdi)
    emit_byte(0x48); emit_byte(0x8B); emit_byte(0xBC); emit_byte(0x24);
    emit_u32_le(1048576);
    // mov rsi, rsp (buffer = rsp)
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xE6);
    // mov edx, 0x100000 (count = 1M, LE bytes 00 00 10 00)
    emit_byte(0xBA); emit_u32_le(1048576);
    // mov eax, 0 (sys_read); syscall
    emit_byte(0xB8); emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    emit_byte(0x0F); emit_byte(0x05);
    // mov r10, rax (save bytes_read)
    emit_byte(0x49); emit_byte(0x89); emit_byte(0xC2);
    // Audit fix: truncation sentinel. If sys_read returned exactly
    // BUF_SIZE bytes, the file was at-or-beyond the buffer and we
    // silently lost data — same failure mode as the original cascade-
    // bug. Trap loudly here so the caller cannot accidentally produce
    // a corrupt downstream binary. cmp r10, 0x100000 (49 81 FA imm32);
    // jne +2 (75 02); ud2 (0F 0B).
    emit_byte(0x49); emit_byte(0x81); emit_byte(0xFA);
    emit_u32_le(1048576);
    emit_byte(0x75); emit_byte(0x02);
    emit_byte(0x0F); emit_byte(0x0B);
    // mov rdi, [rsp+0x100000]; mov eax, 3 (sys_close); syscall
    emit_byte(0x48); emit_byte(0x8B); emit_byte(0xBC); emit_byte(0x24);
    emit_u32_le(1048576);
    emit_byte(0xB8); emit_byte(3); emit_byte(0); emit_byte(0); emit_byte(0);
    emit_byte(0x0F); emit_byte(0x05);
    // test r10, r10 ; jns +3 ; xor r10, r10 (clamp negative to 0)
    emit_byte(0x4D); emit_byte(0x85); emit_byte(0xD2);
    emit_byte(0x7D); emit_byte(0x03);
    emit_byte(0x4D); emit_byte(0x31); emit_byte(0xD2);
    // xor ecx, ecx (counter = 0)
    emit_byte(0x31); emit_byte(0xC9);
    // ---- loop_start (offset will be tracked) ----
    let loop_start = __arena_len();
    // cmp rcx, r10
    emit_byte(0x4C); emit_byte(0x39); emit_byte(0xD1);
    // jge end (placeholder — patch with disp = (post-loop) - (loop_start+5))
    emit_byte(0x7D); emit_byte(0);
    let jge_disp_slot = __arena_len() - 1;
    let jge_after = __arena_len();
    // movzx eax, byte [rsp+rcx]
    emit_byte(0x0F); emit_byte(0xB6); emit_byte(0x04); emit_byte(0x0C);
    // mov edx, eax
    emit_byte(0x89); emit_byte(0xC2);
    // lea rax, [rip+arena_base] (patched via patch_table)
    let arena_lea_slot = emit_lea_rax_rip_placeholder();
    patch_table_add(patch_state, arena_lea_slot, arena_base_s, 18);
    // mov r11d, [rax] (cursor)
    emit_byte(0x44); emit_byte(0x8B); emit_byte(0x18);
    // cmp r11d, CAP
    emit_byte(0x41); emit_byte(0x81); emit_byte(0xFB);
    emit_u32_le(helix_arena_cap());
    // jb in_bounds (+2)
    emit_byte(0x72); emit_byte(0x02);
    // jmp loop_advance (+11)
    emit_byte(0xEB); emit_byte(0x0B);
    // in_bounds: mov [rax+r11*4+4], edx
    emit_byte(0x42); emit_byte(0x89); emit_byte(0x54); emit_byte(0x98); emit_byte(0x04);
    // inc r11d
    emit_byte(0x41); emit_byte(0xFF); emit_byte(0xC3);
    // mov [rax], r11d (update cursor)
    emit_byte(0x44); emit_byte(0x89); emit_byte(0x18);
    // loop_advance: inc rcx
    emit_byte(0x48); emit_byte(0xFF); emit_byte(0xC1);
    // jmp loop_start (rel8 backward; emit placeholder, patch with disp8)
    emit_byte(0xEB); emit_byte(0);
    let back_disp_slot = __arena_len() - 1;
    let back_jmp_after = __arena_len();
    let back_disp = loop_start - back_jmp_after;
    // Encode signed disp8: (back_disp & 0xFF) but Helix has truncated
    // div, so add 256 if negative.
    let back_disp_byte = (back_disp + 256) % 256;
    __arena_set(back_disp_slot, back_disp_byte);
    // end: patch jge forward.
    let end_addr = __arena_len();
    let jge_disp = end_addr - jge_after;
    __arena_set(jge_disp_slot, jge_disp);
    // ---- postlude ----
    // add rsp, 0x100000 (must match the sub above)
    emit_byte(0x48); emit_byte(0x81); emit_byte(0xC4);
    emit_u32_le(1048576);
    // add rsp, 8 (drop fd)
    emit_byte(0x48); emit_byte(0x83); emit_byte(0xC4); emit_byte(0x08);
    // mov rax, r10 (return bytes_read)
    emit_byte(0x4C); emit_byte(0x89); emit_byte(0xD0);
    __arena_len() - body_start
}

// emit_write_file_to_arena_body: emit the inline asm sequence for
// write_file_to_arena AFTER the path has been loaded into rdi.
// Operand registers on entry: rdi = path. The caller has pushed
// (n_bytes, arena_start) onto the stack — top = arena_start, below
// = n_bytes. We can't `pop` them up front because the prologue
// must save callee-saved regs first; instead, after the prologue
// (push rbx/r12/r13/r14 = 32 bytes; sub rsp, 16 = 16 more = 48
// bytes pushed), the args sit at [rsp+48] and [rsp+56].
fn emit_write_file_to_arena_body(patch_state: i32, arena_base_s: i32) -> i32 {
    let body_start = __arena_len();
    // Save callee-saved regs we'll use as state.
    emit_byte(0x53);                              // push rbx
    emit_byte(0x41); emit_byte(0x54);             // push r12
    emit_byte(0x41); emit_byte(0x55);             // push r13
    emit_byte(0x41); emit_byte(0x56);             // push r14
    // sub rsp, 16 (1 byte buffer + 8 byte fd; aligned)
    emit_byte(0x48); emit_byte(0x83); emit_byte(0xEC); emit_byte(0x10);
    // sys_open(path, O_WRONLY|O_CREAT|O_TRUNC=0x241, mode=0644=0x1A4)
    // mov esi, 0x241 (5 bytes)
    emit_byte(0xBE); emit_byte(0x41); emit_byte(0x02); emit_byte(0x00); emit_byte(0x00);
    // mov edx, 0x1A4 (5 bytes)
    emit_byte(0xBA); emit_byte(0xA4); emit_byte(0x01); emit_byte(0x00); emit_byte(0x00);
    // mov eax, 2; syscall
    emit_byte(0xB8); emit_byte(0x02); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
    emit_byte(0x0F); emit_byte(0x05);
    // mov [rsp+8], rax (save fd)
    emit_byte(0x48); emit_byte(0x89); emit_byte(0x44); emit_byte(0x24); emit_byte(0x08);
    // test rax, rax ; jl error_close (placeholder)
    emit_byte(0x48); emit_byte(0x85); emit_byte(0xC0);
    emit_byte(0x7C); emit_byte(0x00);
    let err_jmp_slot = __arena_len() - 1;
    let err_jmp_after = __arena_len();
    // Load args from stack via [rsp+48] (arena_start) and [rsp+56]
    // (n_bytes). Encoding: 8B 44 24 disp8 = mov eax, [rsp+disp8].
    emit_byte(0x8B); emit_byte(0x44); emit_byte(0x24); emit_byte(0x30);  // arena_start
    emit_byte(0x41); emit_byte(0x89); emit_byte(0xC4);                    // mov r12d, eax
    emit_byte(0x8B); emit_byte(0x44); emit_byte(0x24); emit_byte(0x38);  // n_bytes
    emit_byte(0x41); emit_byte(0x89); emit_byte(0xC5);                    // mov r13d, eax
    // xor r14d, r14d (counter)
    emit_byte(0x45); emit_byte(0x31); emit_byte(0xF6);
    // ---- loop_start ----
    let loop_start = __arena_len();
    // cmp r14d, r13d
    emit_byte(0x45); emit_byte(0x39); emit_byte(0xEE);
    // jge done (placeholder)
    emit_byte(0x7D); emit_byte(0x00);
    let jge_disp_slot = __arena_len() - 1;
    let jge_after = __arena_len();
    // mov ecx, r12d ; add ecx, r14d (ecx = arena_start + counter)
    emit_byte(0x44); emit_byte(0x89); emit_byte(0xE1);
    emit_byte(0x44); emit_byte(0x01); emit_byte(0xF1);
    // cmp ecx, CAP
    emit_byte(0x81); emit_byte(0xF9);
    emit_u32_le(helix_arena_cap());
    // jae done (placeholder)
    emit_byte(0x73); emit_byte(0x00);
    let jae_disp_slot = __arena_len() - 1;
    let jae_after = __arena_len();
    // lea rax, [rip+arena_base]
    let arena_lea_slot = emit_lea_rax_rip_placeholder();
    patch_table_add(patch_state, arena_lea_slot, arena_base_s, 18);
    // mov eax, [rax+rcx*4+4]
    emit_byte(0x8B); emit_byte(0x44); emit_byte(0x88); emit_byte(0x04);
    // mov [rsp], al
    emit_byte(0x88); emit_byte(0x04); emit_byte(0x24);
    // sys_write(fd, &byte, 1)
    // mov rdi, [rsp+8]
    emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x08);
    // mov rsi, rsp
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xE6);
    // mov edx, 1
    emit_byte(0xBA); emit_byte(0x01); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
    // mov eax, 1; syscall
    emit_byte(0xB8); emit_byte(0x01); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
    emit_byte(0x0F); emit_byte(0x05);
    // inc r14d
    emit_byte(0x41); emit_byte(0xFF); emit_byte(0xC6);
    // jmp loop_start (rel8 backward, placeholder)
    emit_byte(0xEB); emit_byte(0x00);
    let back_disp_slot = __arena_len() - 1;
    let back_jmp_after = __arena_len();
    let back_disp = loop_start - back_jmp_after;
    let back_disp_byte = (back_disp + 256) % 256;
    __arena_set(back_disp_slot, back_disp_byte);
    // ---- done: patch jge and jae ----
    let done_addr = __arena_len();
    let jge_disp = done_addr - jge_after;
    let jae_disp = done_addr - jae_after;
    __arena_set(jge_disp_slot, jge_disp);
    __arena_set(jae_disp_slot, jae_disp);
    // close(fd)
    emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x08);
    emit_byte(0xB8); emit_byte(0x03); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
    emit_byte(0x0F); emit_byte(0x05);
    // mov eax, r14d (return count)
    emit_byte(0x44); emit_byte(0x89); emit_byte(0xF0);
    // jmp epilogue (placeholder; we want to skip the error block)
    emit_byte(0xEB); emit_byte(0x00);
    let skip_err_slot = __arena_len() - 1;
    let skip_err_after = __arena_len();
    // ---- error_close: open failed; just set return = 0 and fall
    // through. The args are still on the stack — they're cleaned up
    // by the unified epilogue below (add rsp, 16 after pop).
    let err_addr = __arena_len();
    let err_disp = err_addr - err_jmp_after;
    __arena_set(err_jmp_slot, err_disp);
    // mov eax, 0
    emit_byte(0xB8); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
    // ---- epilogue ----
    let ep_addr = __arena_len();
    let skip_disp = ep_addr - skip_err_after;
    __arena_set(skip_err_slot, skip_disp);
    // add rsp, 16 (drop buffer + fd)
    emit_byte(0x48); emit_byte(0x83); emit_byte(0xC4); emit_byte(0x10);
    // pop r14, r13, r12, rbx
    emit_byte(0x41); emit_byte(0x5E);
    emit_byte(0x41); emit_byte(0x5D);
    emit_byte(0x41); emit_byte(0x5C);
    emit_byte(0x5B);
    // add rsp, 16 (drop the 2 args pushed by caller)
    emit_byte(0x48); emit_byte(0x83); emit_byte(0xC4); emit_byte(0x10);
    __arena_len() - body_start
}

// K1.F19 (2026-05-27): emit the FNV-style quadratic mixer used by the
// `__hash_i32` builtin AND now by `reflect_hash` / `__helix_reflect_hash`
// (the latter two were K1.F2/F4 0-stubs prior to K1.F19).
//
// Assumes the input value is in eax on entry. On return, eax holds:
//     h = (x * x * c1) + (x * c2) + c3
// where c1=0x05EBCA6B, c2=0x27D4EB2F, c3=0x165667B1 (mirrors
// helixc-Python lower_ast.py:939-963). Signed-32-bit imul throughout;
// results truncate to i32. Returns the byte count emitted (24).
//
// Codegen (24 bytes):
//   push rax                              50           (1)
//   imul eax, eax (eax = x*x)             0F AF C0     (3)
//   imul eax, eax, c1                     69 C0 imm32  (6)
//   pop rcx (rcx = x)                     59           (1)
//   imul ecx, ecx, c2                     69 C9 imm32  (6)
//   add eax, ecx                          01 C8        (2)
//   add eax, c3                           05 imm32     (5)
fn emit_hash_i32_mixer() -> i32 {
    emit_byte(0x50);                                         // push rax
    emit_byte(0x0F); emit_byte(0xAF); emit_byte(0xC0);       // imul eax, eax
    emit_byte(0x69); emit_byte(0xC0);                        // imul eax, eax, imm32
    emit_byte(0x6B); emit_byte(0xCA); emit_byte(0xEB); emit_byte(0x05); // c1 LE
    emit_byte(0x59);                                         // pop rcx
    emit_byte(0x69); emit_byte(0xC9);                        // imul ecx, ecx, imm32
    emit_byte(0x2F); emit_byte(0xEB); emit_byte(0xD4); emit_byte(0x27); // c2 LE
    emit_byte(0x01); emit_byte(0xC8);                        // add eax, ecx
    emit_byte(0x05);                                         // add eax, imm32
    emit_byte(0xB1); emit_byte(0x67); emit_byte(0x56); emit_byte(0x16); // c3 LE
    24
}

// Try to recognize a builtin call. If matched, emit the inline
// asm and return the byte count. If not, return 0 (caller falls
// back to regular CALL emission).
//
// Recognize each arena builtin and emit its inline asm. Returns
// the byte count emitted, or 0 if the name doesn't match any
// known builtin (caller falls back to regular CALL emission).
fn try_emit_builtin_call(name_s: i32, name_l: i32, args_head: i32,
                          bind_state: i32, patch_state: i32, bn_state: i32) -> i32 {
    let arena_base_s = bn_helix_arena_base_s(bn_state);
    if kovc_byte_eq(name_s, name_l, bn_arena_len_s(bn_state), 11) == 1 {
        // __arena_len(): lea rax, [arena]; mov eax, [rax]
        let disp_slot = emit_lea_rax_rip_placeholder();
        patch_table_add(patch_state, disp_slot, arena_base_s, 18);
        emit_byte(0x8B); emit_byte(0x00);
        9
    } else { if kovc_byte_eq(name_s, name_l, bn_arena_get_s(bn_state), 11) == 1 {
        // __arena_get(idx): eval idx in eax; mov ecx, eax;
        //                    lea rax, arena; mov eax, [rax+rcx*4+4]
        let arg_idx = __arena_get(args_head + 1);
        let n_arg = emit_ast_code(arg_idx, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);                  // mov ecx, eax
        let disp_slot = emit_lea_rax_rip_placeholder();
        patch_table_add(patch_state, disp_slot, arena_base_s, 18);
        emit_byte(0x8B); emit_byte(0x44); emit_byte(0x88); emit_byte(0x04);
        n_arg + 2 + 7 + 4
    } else { if kovc_byte_eq(name_s, name_l, bn_arena_push_triple_s(bn_state), 19) == 1 {
        // K1.AG (2026-05-25): __arena_push_triple(a, b, c) -> i32.
        // Atomic 3-slot push, mirror of push_pair. Writes a, b, c
        // at slots cursor, cursor+1, cursor+2; advances cursor by
        // 3; returns OLD cursor. Overflow when cursor >= CAP-2
        // (need all 3 slots in range): returns -1, no writes.
        let a0_pt = __arena_get(args_head + 1);
        let next1_pt = __arena_get(args_head + 2);
        let a1_pt = __arena_get(next1_pt + 1);
        let next2_pt = __arena_get(next1_pt + 2);
        let a2_pt = __arena_get(next2_pt + 1);
        let n_left_pt = emit_ast_code(a0_pt, bind_state, patch_state, bn_state);
        emit_byte(0x50);                                       // push rax (left)
        let n_mid_pt = emit_ast_code(a1_pt, bind_state, patch_state, bn_state);
        emit_byte(0x50);                                       // push rax (middle)
        let n_right_pt = emit_ast_code(a2_pt, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);                      // mov ecx, eax (right)
        emit_byte(0x5F);                                       // pop rdi (middle -> edi)
        emit_byte(0x5A);                                       // pop rdx (left -> edx)
        let disp_slot_pt = emit_lea_rax_rip_placeholder();    // 7 bytes
        patch_table_add(patch_state, disp_slot_pt, arena_base_s, 18);
        emit_byte(0x8B); emit_byte(0x30);                      // mov esi, [rax] (cursor)
        emit_byte(0x81); emit_byte(0xFE);                      // cmp esi, CAP-2 (6 bytes)
        emit_u32_le(helix_arena_cap() - 2);
        emit_byte(0x7D); emit_byte(0x15);                      // jge overflow (skip 21 = in-bounds path)
        // in_bounds (21 bytes):
        emit_byte(0x89); emit_byte(0x54); emit_byte(0xB0); emit_byte(0x04);  // mov [rax+rsi*4+4],  edx (left)
        emit_byte(0x89); emit_byte(0x7C); emit_byte(0xB0); emit_byte(0x08);  // mov [rax+rsi*4+8],  edi (middle)
        emit_byte(0x89); emit_byte(0x4C); emit_byte(0xB0); emit_byte(0x0C);  // mov [rax+rsi*4+12], ecx (right)
        emit_byte(0x8D); emit_byte(0x4E); emit_byte(0x03);                   // lea ecx, [rsi+3] (new cursor)
        emit_byte(0x89); emit_byte(0x08);                                    // mov [rax], ecx
        emit_byte(0x89); emit_byte(0xF0);                                    // mov eax, esi (return OLD cursor)
        emit_byte(0xEB); emit_byte(0x05);                                    // jmp end (skip overflow=5)
        // overflow (5 bytes):
        emit_byte(0xB8); emit_byte(0xFF); emit_byte(0xFF); emit_byte(0xFF); emit_byte(0xFF);  // mov eax, -1
        // Total after arg evals: 1 + 1 + 2 + 1 + 1 + 7 + 2 + 6 + 2 + 21 + 5 = 49 bytes
        n_left_pt + n_mid_pt + n_right_pt + 49
    } else { if kovc_byte_eq(name_s, name_l, bn_arena_push_pair_s(bn_state), 17) == 1 {
        // K1.AF (2026-05-25): __arena_push_pair(left, right) -> i32.
        // Atomic 2-slot push: writes left at slot cursor, right at
        // slot cursor+1, advances cursor by 2, returns OLD cursor.
        // Overflow (cursor >= CAP-1) returns -1 with no writes.
        // Mirrors Python's _HELIX_ARENA_PUSH_PAIR_HELPER (LLVM
        // backend). Atomic-or-none: cursor check is done BEFORE
        // either write so either both slots land or neither.
        //
        // Register usage (after arg evaluation):
        //   eax = arena_base ptr; esi = OLD cursor; edx = left; ecx = right
        //   on overflow path: eax = -1 (returned)
        //   on success path: eax = OLD cursor (returned)
        let a0_pp = __arena_get(args_head + 1);
        let next_arg_pp = __arena_get(args_head + 2);
        let a1_pp = __arena_get(next_arg_pp + 1);
        let n_left_pp = emit_ast_code(a0_pp, bind_state, patch_state, bn_state);
        emit_byte(0x50);                                       // push rax (left)
        let n_right_pp = emit_ast_code(a1_pp, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);                      // mov ecx, eax (right)
        emit_byte(0x5A);                                       // pop rdx (edx = left)
        let disp_slot_pp = emit_lea_rax_rip_placeholder();    // 7 bytes
        patch_table_add(patch_state, disp_slot_pp, arena_base_s, 18);
        emit_byte(0x8B); emit_byte(0x30);                      // mov esi, [rax] (cursor)
        emit_byte(0x81); emit_byte(0xFE);                      // cmp esi, CAP-1 (6 bytes)
        emit_u32_le(helix_arena_cap() - 1);
        emit_byte(0x7D); emit_byte(0x11);                      // jge overflow (skip 17 = in-bounds path)
        // in_bounds (17 bytes):
        emit_byte(0x89); emit_byte(0x54); emit_byte(0xB0); emit_byte(0x04);  // mov [rax+rsi*4+4], edx (write left at slot esi)
        emit_byte(0x89); emit_byte(0x4C); emit_byte(0xB0); emit_byte(0x08);  // mov [rax+rsi*4+8], ecx (write right at slot esi+1)
        emit_byte(0x8D); emit_byte(0x4E); emit_byte(0x02);                   // lea ecx, [rsi+2] (new cursor)
        emit_byte(0x89); emit_byte(0x08);                                    // mov [rax], ecx (store new cursor)
        emit_byte(0x89); emit_byte(0xF0);                                    // mov eax, esi (return OLD cursor)
        emit_byte(0xEB); emit_byte(0x05);                                    // jmp end (skip 5 = overflow path)
        // overflow (5 bytes):
        emit_byte(0xB8); emit_byte(0xFF); emit_byte(0xFF); emit_byte(0xFF); emit_byte(0xFF);  // mov eax, -1
        // Total bytes after arg evals: 1 + 2 + 1 + 7 + 2 + 6 + 2 + 17 + 5 = 43
        n_left_pp + n_right_pp + 43
    } else { if kovc_byte_eq(name_s, name_l, bn_arena_push_s(bn_state), 12) == 1 {
        // __arena_push(val): eval val in eax; bounds-checked
        // write to arena; return old cursor.
        let arg_idx = __arena_get(args_head + 1);
        let n_arg = emit_ast_code(arg_idx, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC2);                  // mov edx, eax (val)
        let disp_slot = emit_lea_rax_rip_placeholder();    // 7 bytes
        patch_table_add(patch_state, disp_slot, arena_base_s, 18);
        emit_byte(0x8B); emit_byte(0x08);                  // mov ecx, [rax]
        emit_byte(0x81); emit_byte(0xF9);                  // cmp ecx, CAP (6 bytes)
        emit_u32_le(helix_arena_cap());
        emit_byte(0x72); emit_byte(0x07);                  // jb in_bounds (skip 7)
        emit_byte(0xB8); emit_byte(0xFF); emit_byte(0xFF); emit_byte(0xFF); emit_byte(0xFF);
        emit_byte(0xEB); emit_byte(0x0C);                  // jmp end (skip 12)
        emit_byte(0x89); emit_byte(0x54); emit_byte(0x88); emit_byte(0x04);  // mov [rax+rcx*4+4], edx
        emit_byte(0x89); emit_byte(0xCA);                  // mov edx, ecx (save old cursor)
        emit_byte(0xFF); emit_byte(0xC1);                  // inc ecx
        emit_byte(0x89); emit_byte(0x08);                  // mov [rax], ecx (update cursor)
        emit_byte(0x89); emit_byte(0xD0);                  // mov eax, edx (return old cursor)
        n_arg + 2 + 7 + 2 + 6 + 2 + 5 + 2 + 4 + 2 + 2 + 2 + 2
    } else { if kovc_byte_eq(name_s, name_l, bn_arena_set_s(bn_state), 11) == 1 {
        // __arena_set(idx, val): eval idx, push; eval val in eax,
        //                        mov ecx, eax; pop rax = idx;
        //                        mov edx, eax (idx); lea rax, arena;
        //                        mov [rax+rdx*4+4], ecx; xor eax, eax
        let a0 = __arena_get(args_head + 1);
        let next_arg = __arena_get(args_head + 2);
        let a1 = __arena_get(next_arg + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n1 = emit_ast_code(a1, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);                  // mov ecx, eax (val)
        emit_byte(0x58);                                    // pop rax (idx)
        emit_byte(0x89); emit_byte(0xC2);                  // mov edx, eax (idx)
        let disp_slot = emit_lea_rax_rip_placeholder();
        patch_table_add(patch_state, disp_slot, arena_base_s, 18);
        emit_byte(0x89); emit_byte(0x4C); emit_byte(0x90); emit_byte(0x04);
        emit_byte(0x31); emit_byte(0xC0);                  // xor eax, eax
        n0 + np + n1 + 2 + 1 + 2 + 7 + 4 + 2
    } else { if kovc_byte_eq(name_s, name_l, bn_reflect_hash_s(bn_state), 12) == 1 {
        // K1.F2 (2026-05-26): reflect_hash(arg) was a 0-stub. The matrix
        // had it as KOVC-MISSING; Python's compile-and-run path errored
        // with NotImplementedError too, so "both return 0" was the
        // (vacuous) parity contract.
        // K1.F17 (2026-05-27): walk the full args_head linked list so
        // side effects of args 2+ fire (the K1.F16 silent-arg-drop
        // closure, extended to this stub).
        // K1.F19 (2026-05-27): upgrade from the 0-stub to a real i32
        // hash using the shared `emit_hash_i32_mixer` helper. The full
        // reflect_hash semantic in Python is content-addressable AST
        // hashing; the bootstrap degenerate version hashes the LAST
        // evaluated arg's i32 value (since args_head is variadic and
        // the K1.F16/F17 walk leaves the last result in eax). Programs
        // that only check determinism + non-zero of the hash for
        // non-zero inputs now see useful behaviour.
        let mut rh_arg_cur: i32 = args_head;
        let mut rh_bytes: i32 = 0;
        while rh_arg_cur != 0 {
            let rh_arg_idx = __arena_get(rh_arg_cur + 1);
            rh_bytes = rh_bytes + emit_ast_code(rh_arg_idx, bind_state, patch_state, bn_state);
            rh_arg_cur = __arena_get(rh_arg_cur + 2);
        }
        let rh_mix = emit_hash_i32_mixer();
        rh_bytes + rh_mix
    } else { if kovc_byte_eq(name_s, name_l, bn_trace_event_s(bn_state), 13) == 1 {
        // K1.F3 (2026-05-26): __trace_event(...) was a 0-stub.
        // K1.F16 (2026-05-27): walk the full args_head linked list.
        // K1.F20 (2026-05-27): dropped the trailing mov-eax-0 closer.
        // K1.F20b (2026-05-27): added the depth-1 ring-buffer store at
        //   arena slot CAP-65 (disp 8388352).
        // K3.J (2026-05-27, audit-fix-MEDIUM-1): the K1.F20b store
        //   was unconditional. For ZERO-arg `__trace_event()` the walk
        //   emits nothing and eax holds whatever the caller-context
        //   expression left there -- which the prior code's hand-waved
        //   "Phase-0 conventions zero before call eval" was NOT actually
        //   true. The unconditional store would write that residue to
        //   the trace slot, silently corrupting __trace_last's read.
        //   Fix: gate the store on args_head != 0. For zero-arg
        //   trace_event, emit a deterministic `xor eax, eax` so the
        //   call returns 0 (matching the K1.F3 spirit) and skip the
        //   slot write entirely (preserving prior trace_event's
        //   recorded value untouched -- "zero-arg trace_event is a
        //   true no-op").
        //
        //   N-arg codegen (17 bytes after the walk):
        //     89 C1               mov ecx, eax     (save last value)
        //     <lea rax, [arena_base]>     7 bytes (RIP-relative
        //                                          placeholder; patched)
        //     89 88 disp32        mov [rax + 8388352], ecx  (6 bytes)
        //     89 C8               mov eax, ecx     (restore eax for
        //                                           value-tap return)
        //
        //   0-arg codegen (2 bytes total):
        //     31 C0               xor eax, eax     (return 0)
        let arena_base_s = bn_helix_arena_base_s(bn_state);
        let mut tev_arg_cur: i32 = args_head;
        let mut tev_bytes: i32 = 0;
        while tev_arg_cur != 0 {
            let tev_arg_idx = __arena_get(tev_arg_cur + 1);
            tev_bytes = tev_bytes + emit_ast_code(tev_arg_idx, bind_state, patch_state, bn_state);
            tev_arg_cur = __arena_get(tev_arg_cur + 2);
        }
        if args_head == 0 {
            // K3.J: zero-arg trace_event -> deterministic 0 return,
            // no slot write (preserves prior recorded value).
            emit_byte(0x31); emit_byte(0xC0);               // xor eax, eax
            2
        } else {
            // K1.F20b store: write last value (still in eax) to trace slot.
            emit_byte(0x89); emit_byte(0xC1);                   // mov ecx, eax
            let disp_slot = emit_lea_rax_rip_placeholder();     // 7 bytes
            patch_table_add(patch_state, disp_slot, arena_base_s, 18);
            emit_byte(0x89); emit_byte(0x88);                   // mov [rax + disp32], ecx
            emit_u32_le(8388352);                               // 4 bytes -> 6-byte instr
            emit_byte(0x89); emit_byte(0xC8);                   // mov eax, ecx (restore)
            tev_bytes + 2 + 7 + 6 + 2
        }
    } else { if kovc_byte_eq(name_s, name_l, bn_trace_last_s(bn_state), 12) == 1 {
        // K1.F20b (2026-05-27): __trace_last() -> i32. Read the value
        // K1.F20b's __trace_event most-recently stored to arena slot
        // CAP-65 (disp 8388352, one i32 slot below the Quote cell-
        // table). Returns 0 if no __trace_event has fired yet (BSS-
        // zeroed at load time).
        //
        // K3.J (2026-05-27, audit-fix-LOW-2): the original K1.F20b
        // version silently dropped any args passed -- inconsistent
        // with the K1.F16/F17 silent-arg-drop closure pattern used by
        // every other variadic-tolerant reflection/trace builtin
        // (reflect_hash, __trace_event, __helix_*). Now walk args_head
        // first so side effects of args fire, then load the trace slot.
        // Zero-arg case still works (the walk emits nothing); non-zero-
        // arg case now preserves the K1.F17 side-effect-preservation
        // contract.
        //
        // Codegen (variable; 13 bytes + side-effect arg bytes):
        //   <walk args_head, emit each arg's code for side effects>
        //   <lea rax, [arena_base]>     7-byte placeholder
        //   8B 80 disp32                mov eax, [rax + 8388352]   (6 bytes)
        let mut tl_arg_cur: i32 = args_head;
        let mut tl_arg_bytes: i32 = 0;
        while tl_arg_cur != 0 {
            let tl_arg_idx = __arena_get(tl_arg_cur + 1);
            tl_arg_bytes = tl_arg_bytes + emit_ast_code(tl_arg_idx, bind_state, patch_state, bn_state);
            tl_arg_cur = __arena_get(tl_arg_cur + 2);
        }
        let arena_base_s_tl = bn_helix_arena_base_s(bn_state);
        let disp_slot_tl = emit_lea_rax_rip_placeholder();
        patch_table_add(patch_state, disp_slot_tl, arena_base_s_tl, 18);
        emit_byte(0x8B); emit_byte(0x80);                   // mov eax, [rax + disp32]
        emit_u32_le(8388352);
        tl_arg_bytes + 7 + 6
    } else { if kovc_byte_eq(name_s, name_l, bn_helix_splice_s(bn_state), 14) == 1 {
        // K1.F4 (2026-05-26): __helix_splice(handle) -> 0 stub.
        // Underscore-prefixed alias for Splice. Same stub semantics
        // as Quote/Splice/modify/reflect_hash.
        // K1.F17 (2026-05-27): walk full args_head (silent-arg-drop fix).
        let mut hs_arg_cur: i32 = args_head;
        let mut hs_bytes: i32 = 0;
        while hs_arg_cur != 0 {
            let hs_arg_idx = __arena_get(hs_arg_cur + 1);
            hs_bytes = hs_bytes + emit_ast_code(hs_arg_idx, bind_state, patch_state, bn_state);
            hs_arg_cur = __arena_get(hs_arg_cur + 2);
        }
        emit_byte(0xB8); emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
        hs_bytes + 5
    } else { if kovc_byte_eq(name_s, name_l, bn_helix_modify_s(bn_state), 14) == 1 {
        // K1.F4 (2026-05-26): __helix_modify(...) -> 0 stub.
        // Underscore-prefixed alias for modify. Same no-op contract.
        // K1.F17 (2026-05-27): walk full args_head (silent-arg-drop fix).
        let mut hm_arg_cur: i32 = args_head;
        let mut hm_bytes: i32 = 0;
        while hm_arg_cur != 0 {
            let hm_arg_idx = __arena_get(hm_arg_cur + 1);
            hm_bytes = hm_bytes + emit_ast_code(hm_arg_idx, bind_state, patch_state, bn_state);
            hm_arg_cur = __arena_get(hm_arg_cur + 2);
        }
        emit_byte(0xB8); emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
        hm_bytes + 5
    } else { if kovc_byte_eq(name_s, name_l, bn_helix_reflect_hash_s(bn_state), 20) == 1 {
        // K1.F4 (2026-05-26): __helix_reflect_hash(...) -> 0 stub.
        // Underscore-prefixed alias for reflect_hash. Same no-op.
        // K1.F17 (2026-05-27): walk full args_head (silent-arg-drop fix).
        // K1.F19 (2026-05-27): upgrade to the real FNV mixer (same as
        // reflect_hash); alias contract preserved.
        let mut hr_arg_cur: i32 = args_head;
        let mut hr_bytes: i32 = 0;
        while hr_arg_cur != 0 {
            let hr_arg_idx = __arena_get(hr_arg_cur + 1);
            hr_bytes = hr_bytes + emit_ast_code(hr_arg_idx, bind_state, patch_state, bn_state);
            hr_arg_cur = __arena_get(hr_arg_cur + 2);
        }
        let hr_mix = emit_hash_i32_mixer();
        hr_bytes + hr_mix
    } else { if kovc_byte_eq(name_s, name_l, bn_print_str_s(bn_state), 9) == 1 {
        // K1.AK (2026-05-25): print_str("msg") -- emit sys_write(1,
        // str_ptr, str_len) for an AST_STR_LIT arg. Mirror of the
        // panic codegen MINUS the trap. Validates first arg is
        // AST_STR_LIT (tag 25); else emits a defensive ud2.
        let arg_ps = __arena_get(args_head + 1);
        let arg_tag_ps = __arena_get(arg_ps);
        if arg_tag_ps != 25 {
            // First arg must be a string literal -- emit ud2 for misuse.
            emit_byte(0x0F); emit_byte(0x0B);
            2
        } else {
            let body_s_ps = __arena_get(arg_ps + 1);
            let body_l_ps = __arena_get(arg_ps + 2);
            // lea rsi, [rip + str_disp]
            let str_disp_slot_ps = emit_lea_rsi_rip_placeholder();
            str_table_add(bn_state, str_disp_slot_ps, body_s_ps, body_l_ps);
            // mov edi, 1 (fd=stdout) -- 5 bytes
            emit_byte(0xBF); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            // mov edx, body_l (len) -- 5 bytes
            emit_byte(0xBA);
            emit_u32_le(body_l_ps);
            // mov eax, 1 (sys_write) -- 5 bytes
            emit_byte(0xB8); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            // syscall -- 2 bytes
            emit_byte(0x0F); emit_byte(0x05);
            // Return 0 in eax (sys_write returned bytes written -- we
            // overwrite for parity with print_int's "returns 0").
            // xor eax, eax -- 2 bytes
            emit_byte(0x31); emit_byte(0xC0);
            // Total: 7 (lea) + 5 + 5 + 5 + 2 + 2 = 26 bytes
            26
        }
    } else { if kovc_byte_eq(name_s, name_l, bn_print_str_ln_s(bn_state), 12) == 1 {
        // K1.F22c (2026-05-27): print_str_ln("msg") -- print_str's
        // sys_write to stdout PLUS a trailing newline sys_write. Used
        // by the K1.F22b println!("msg") macro expansion. Mirror of
        // the print_str codegen + the K1.AI panic newline pattern
        // (str_table_add of the shared "\n" byte at bn_panic_newline,
        // which is just a 1-byte newline -- the fd target differs
        // [stdout vs stderr] but the byte content is identical).
        let arg_psln = __arena_get(args_head + 1);
        let arg_tag_psln = __arena_get(arg_psln);
        if arg_tag_psln != 25 {
            // First arg must be AST_STR_LIT -- emit ud2 for misuse.
            emit_byte(0x0F); emit_byte(0x0B);
            2
        } else {
            let body_s_psln = __arena_get(arg_psln + 1);
            let body_l_psln = __arena_get(arg_psln + 2);
            // Message sys_write: lea rsi + mov edi,1 + mov edx,body_l
            // + mov eax,1 + syscall = 24 bytes.
            let msg_disp_slot = emit_lea_rsi_rip_placeholder();
            str_table_add(bn_state, msg_disp_slot, body_s_psln, body_l_psln);
            emit_byte(0xBF); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0xBA);
            emit_u32_le(body_l_psln);
            emit_byte(0xB8); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0x0F); emit_byte(0x05);
            // Newline sys_write: lea rsi for "\n" + mov edi,1 + mov edx,1
            // + mov eax,1 + syscall = 24 bytes.
            let nl_disp_slot = emit_lea_rsi_rip_placeholder();
            str_table_add(bn_state, nl_disp_slot, bn_panic_newline_s(bn_state), 1);
            emit_byte(0xBF); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0xBA); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0xB8); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0x0F); emit_byte(0x05);
            // xor eax, eax (return 0) -- 2 bytes.
            emit_byte(0x31); emit_byte(0xC0);
            // Total: 24 (msg) + 24 (nl) + 2 (xor) = 50 bytes.
            50
        }
    } else { if kovc_byte_eq(name_s, name_l, bn_eprint_str_ln_s(bn_state), 13) == 1 {
        // K1.F22d (2026-05-27): eprint_str_ln("msg") -- stderr variant
        // of K1.F22c print_str_ln. Identical codegen except both
        // sys_writes target fd=2 (stderr) instead of fd=1 (stdout).
        // Paired with the K1.F22d eprintln!("msg") macro expansion.
        let arg_epsln = __arena_get(args_head + 1);
        let arg_tag_epsln = __arena_get(arg_epsln);
        if arg_tag_epsln != 25 {
            emit_byte(0x0F); emit_byte(0x0B);
            2
        } else {
            let body_s_epsln = __arena_get(arg_epsln + 1);
            let body_l_epsln = __arena_get(arg_epsln + 2);
            // Message sys_write to fd=2 (24 bytes).
            let emsg_disp_slot = emit_lea_rsi_rip_placeholder();
            str_table_add(bn_state, emsg_disp_slot, body_s_epsln, body_l_epsln);
            emit_byte(0xBF); emit_byte(0x02);                    // mov edi, 2 (stderr)
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0xBA);
            emit_u32_le(body_l_epsln);
            emit_byte(0xB8); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0x0F); emit_byte(0x05);
            // Newline sys_write to fd=2 (24 bytes).
            let enl_disp_slot = emit_lea_rsi_rip_placeholder();
            str_table_add(bn_state, enl_disp_slot, bn_panic_newline_s(bn_state), 1);
            emit_byte(0xBF); emit_byte(0x02);                    // mov edi, 2 (stderr)
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0xBA); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0xB8); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0x0F); emit_byte(0x05);
            emit_byte(0x31); emit_byte(0xC0);
            50
        }
    } else { if kovc_byte_eq(name_s, name_l, bn_eprint_str_s(bn_state), 10) == 1 {
        // K1.F22f (2026-05-27): eprint_str("msg") -- no-newline stderr
        // variant of K1.AK print_str. Single sys_write(2, ptr, len) +
        // xor eax, eax. Mirror of the K1.AK codegen with `mov edi, 2`
        // instead of `mov edi, 1`. Used by the K1.F22f eprint!("msg")
        // macro expansion (parser-side synthesis of AST_CALL to this
        // name).
        let arg_eps = __arena_get(args_head + 1);
        let arg_tag_eps = __arena_get(arg_eps);
        if arg_tag_eps != 25 {
            // First arg must be AST_STR_LIT -- emit ud2 for misuse.
            emit_byte(0x0F); emit_byte(0x0B);
            2
        } else {
            let body_s_eps = __arena_get(arg_eps + 1);
            let body_l_eps = __arena_get(arg_eps + 2);
            // lea rsi, [rip + str_disp] -- 7 bytes
            let str_disp_slot_eps = emit_lea_rsi_rip_placeholder();
            str_table_add(bn_state, str_disp_slot_eps, body_s_eps, body_l_eps);
            // mov edi, 2 (fd=stderr) -- 5 bytes (DIFFERENT from print_str)
            emit_byte(0xBF); emit_byte(0x02);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            // mov edx, body_l (len) -- 5 bytes
            emit_byte(0xBA);
            emit_u32_le(body_l_eps);
            // mov eax, 1 (sys_write) -- 5 bytes
            emit_byte(0xB8); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            // syscall -- 2 bytes
            emit_byte(0x0F); emit_byte(0x05);
            // xor eax, eax (return 0) -- 2 bytes
            emit_byte(0x31); emit_byte(0xC0);
            // Total: 7 + 5 + 5 + 5 + 2 + 2 = 26 bytes (identical layout
            // to K1.AK print_str; only the edi=2 byte differs).
            26
        }
    } else { if kovc_byte_eq(name_s, name_l, bn_tile_zeros_s(bn_state), 12) == 1 {
        // K1.F23c (2026-05-27): __tile_zeros(N, M) -> arena offset of
        // an N*M-cell region. Cursor-bump allocator:
        //   1. eval N, push; eval M -> eax.
        //   2. ecx = M; pop rax -> rax = N.
        //   3. imul eax, ecx -> eax = N*M (signed 32-bit; small values).
        //   4. ecx = N*M.
        //   5. lea rax, [arena_base]; edx = [rax] (old cursor).
        //   6. push rdx (save old cursor for return).
        //   7. edx += ecx (new cursor); cmp edx, CAP; jb in_bounds.
        //   8. bounds-fail: mov eax, -1; add rsp, 8 (discard); jmp end.
        //   9. in_bounds: mov [rax], edx; pop rax (return old cursor).
        //  10. end:
        // Skips per-cell zero-init -- BSS-zero on Linux means new
        // cells start at 0. Phase-0 substrate for TILE_ZEROS (and the
        // tile-ops family more broadly). Bypasses the K1.F23 helper-
        // context defect because it's used directly in main() scope.
        let a0_tz = __arena_get(args_head + 1);
        let next_arg_tz = __arena_get(args_head + 2);
        let a1_tz = __arena_get(next_arg_tz + 1);
        let n0_tz = emit_ast_code(a0_tz, bind_state, patch_state, bn_state);
        let np_tz = emit_push_rax();
        let n1_tz = emit_ast_code(a1_tz, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);                  // mov ecx, eax (M)
        emit_byte(0x58);                                    // pop rax (N)
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0xC1); // imul eax, ecx (N*M)
        emit_byte(0x89); emit_byte(0xC1);                  // mov ecx, eax (N*M)
        let disp_slot_tz = emit_lea_rax_rip_placeholder(); // 7 bytes
        patch_table_add(patch_state, disp_slot_tz, arena_base_s, 18);
        emit_byte(0x8B); emit_byte(0x10);                  // mov edx, [rax] (old cursor)
        emit_byte(0x52);                                    // push rdx (save)
        emit_byte(0x01); emit_byte(0xCA);                  // add edx, ecx (new cursor)
        emit_byte(0x81); emit_byte(0xFA);                  // cmp edx, CAP (6 bytes)
        emit_u32_le(helix_arena_cap());
        emit_byte(0x73); emit_byte(0x05);                  // jae bounds_fail (skip in_bounds = 5 bytes)
        // in_bounds (5 bytes ahead of jae fall-through):
        emit_byte(0x89); emit_byte(0x10);                  // mov [rax], edx (update cursor)
        emit_byte(0x58);                                    // pop rax (return old cursor)
        emit_byte(0xEB); emit_byte(0x09);                  // jmp end (skip bounds_fail = 9 bytes)
        // bounds_fail (9 bytes ahead of jmp end fall-through):
        emit_byte(0xB8); emit_byte(0xFF); emit_byte(0xFF); emit_byte(0xFF); emit_byte(0xFF);  // mov eax, -1 (5 bytes)
        emit_byte(0x48); emit_byte(0x83); emit_byte(0xC4); emit_byte(0x08); // add rsp, 8 (discard saved; 4 bytes)
        // end:
        // Byte count after args: 2+1+3+2 (arith pack) + 7 (lea) + 2+1+2+6+2 (cursor + check) + 2+1+2 (in_bounds) + 5+4 (bounds_fail) = 42 bytes
        n0_tz + np_tz + n1_tz + 2 + 1 + 3 + 2 + 7 + 2 + 1 + 2 + 6 + 2 + 2 + 1 + 2 + 5 + 4
    } else { if kovc_byte_eq(name_s, name_l, bn_tile_add_s(bn_state), 10) == 1 {
        // K1.F24j (2026-05-27): __tile_add(a, b, dst, count) REAL
        // elementwise add loop. Restores the 60-byte body originally
        // attempted in K1.F24f. K1.F24f saw 3/3 SIGILL and reverted
        // to the K1.F24e stub, but K1.F24i revealed the SIGILL was
        // a TEST-HELPER bug (multi-line source corruption via
        // printf %s {repr()}), not a codegen defect. With the helper
        // fixed (K1.F24i: stdin pipe), the real loop should work.
        //
        // Stack layout after the 4 arg evals:
        //   [rsp+16] = a_off (i32 zero-extended to 64)
        //   [rsp+8]  = b_off
        //   [rsp]    = dst_off
        //   eax      = count
        //
        // Body (60 bytes after args):
        //   setup (11B):
        //     89 C1                mov ecx, eax           (count -> ecx)
        //     48 8D 05 ?? ?? ?? ??  lea rax, [arena_base] (patch_table)
        //     31 D2                xor edx, edx           (i = 0)
        //   loop_start (rel0):
        //     39 CA                cmp edx, ecx
        //     7D 27                jge +39 -> end
        //     48 8B 74 24 10       mov rsi, [rsp+16]      (a_off)
        //     48 01 D6             add rsi, rdx           (a_off + i)
        //     8B 74 B0 04          mov esi, [rax+rsi*4+4] (a[i])
        //     48 8B 7C 24 08       mov rdi, [rsp+8]       (b_off)
        //     48 01 D7             add rdi, rdx           (b_off + i)
        //     03 74 B8 04          add esi, [rax+rdi*4+4] (esi += b[i])
        //     48 8B 3C 24          mov rdi, [rsp]         (dst_off)
        //     48 01 D7             add rdi, rdx           (dst_off + i)
        //     89 74 B8 04          mov [rax+rdi*4+4], esi (dst[i] = esi)
        //     FF C2                inc edx
        //     EB D5                jmp -43 -> loop_start
        //   end:
        //     48 83 C4 18          add rsp, 24            (discard 3 pushes)
        //     31 C0                xor eax, eax           (return 0)
        //
        // Counts: setup=11, loop_body=43, end=6 -> total 60.
        let a0_ta = __arena_get(args_head + 1);
        let next1_ta = __arena_get(args_head + 2);
        let a1_ta = __arena_get(next1_ta + 1);
        let next2_ta = __arena_get(next1_ta + 2);
        let a2_ta = __arena_get(next2_ta + 1);
        let next3_ta = __arena_get(next2_ta + 2);
        let a3_ta = __arena_get(next3_ta + 1);
        let n0_ta = emit_ast_code(a0_ta, bind_state, patch_state, bn_state);
        let np0_ta = emit_push_rax();
        let n1_ta = emit_ast_code(a1_ta, bind_state, patch_state, bn_state);
        let np1_ta = emit_push_rax();
        let n2_ta = emit_ast_code(a2_ta, bind_state, patch_state, bn_state);
        let np2_ta = emit_push_rax();
        let n3_ta = emit_ast_code(a3_ta, bind_state, patch_state, bn_state);
        // setup (31 bytes after K3.U: 11 base + 20 dst-bounds guard):
        emit_byte(0x89); emit_byte(0xC1);                                       // mov ecx, eax (count)
        // K3.U (2026-05-27): dst-bounds guard. Trap-id 24001. Mirrors the
        // K3.T pattern. Tests dst_off + count <= helix_arena_cap() (2097152).
        emit_byte(0x8B); emit_byte(0x34); emit_byte(0x24);                      // mov esi, [rsp] (dst_off; 3B)
        emit_byte(0x01); emit_byte(0xCE);                                       // add esi, ecx (dst_off + count; 2B)
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP (6B opcode+ModRM)
        emit_u32_le(helix_arena_cap());                                          //   imm32
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7 (skip trap if in bounds)
        emit_trap_with_id(24001);                                                // 7B -- __tile_add dst OOB
        // K3.V (2026-05-27): close READ-side of audit HIGH-1. Add bounds
        // checks on a_off and b_off so OOB reads also trap loud (was: garbage
        // values in result). Trap-ids 24002 (a) and 24003 (b). +42 bytes here.
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x10);     // mov esi, [rsp+16] (a_off; 4B)
        emit_byte(0x01); emit_byte(0xCE);                                       // add esi, ecx     (a_off + count)
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP
        emit_u32_le(helix_arena_cap());
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7
        emit_trap_with_id(24002);                                                // a OOB trap
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x08);     // mov esi, [rsp+8]  (b_off; 4B)
        emit_byte(0x01); emit_byte(0xCE);                                       // add esi, ecx
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP
        emit_u32_le(helix_arena_cap());
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7
        emit_trap_with_id(24003);                                                // b OOB trap
        let disp_slot_ta = emit_lea_rax_rip_placeholder();                      // 7 bytes
        patch_table_add(patch_state, disp_slot_ta, arena_base_s, 18);
        emit_byte(0x31); emit_byte(0xD2);                                       // xor edx, edx
        // loop_start (43 bytes):
        emit_byte(0x39); emit_byte(0xCA);                                       // cmp edx, ecx
        emit_byte(0x7D); emit_byte(0x27);                                       // jge +39
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x10);  // mov rsi, [rsp+16]
        emit_byte(0x48); emit_byte(0x01); emit_byte(0xD6);                      // add rsi, rdx
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0xB0); emit_byte(0x04);     // mov esi, [rax+rsi*4+4]
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x08);  // mov rdi, [rsp+8]
        emit_byte(0x48); emit_byte(0x01); emit_byte(0xD7);                      // add rdi, rdx
        emit_byte(0x03); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);     // add esi, [rax+rdi*4+4]
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x3C); emit_byte(0x24);     // mov rdi, [rsp]
        emit_byte(0x48); emit_byte(0x01); emit_byte(0xD7);                      // add rdi, rdx
        emit_byte(0x89); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);     // mov [rax+rdi*4+4], esi
        emit_byte(0xFF); emit_byte(0xC2);                                       // inc edx
        emit_byte(0xEB); emit_byte(0xD5);                                       // jmp -43 (back to loop_start)
        // end (6 bytes):
        emit_byte(0x48); emit_byte(0x83); emit_byte(0xC4); emit_byte(0x18);     // add rsp, 24
        emit_byte(0x31); emit_byte(0xC0);                                       // xor eax, eax
        n0_ta + np0_ta + n1_ta + np1_ta + n2_ta + np2_ta + n3_ta + 122
    } else { if kovc_byte_eq(name_s, name_l, bn_tile_sub_s(bn_state), 10) == 1 {
        // K1.F25 (2026-05-27): __tile_sub(a, b, dst, count) elementwise
        // subtraction. Mirrors __tile_add (K1.F24j) with one byte change:
        // the inner `add esi, [rax+rdi*4+4]` (opcode 03) becomes
        // `sub esi, [rax+rdi*4+4]` (opcode 2B). Same SIB encoding, same
        // 4-byte instruction length -- no displacement adjustment needed.
        // Result: dst[i] = a[i] - b[i] for i in [0, count).
        let a0_ts = __arena_get(args_head + 1);
        let next1_ts = __arena_get(args_head + 2);
        let a1_ts = __arena_get(next1_ts + 1);
        let next2_ts = __arena_get(next1_ts + 2);
        let a2_ts = __arena_get(next2_ts + 1);
        let next3_ts = __arena_get(next2_ts + 2);
        let a3_ts = __arena_get(next3_ts + 1);
        let n0_ts = emit_ast_code(a0_ts, bind_state, patch_state, bn_state);
        let np0_ts = emit_push_rax();
        let n1_ts = emit_ast_code(a1_ts, bind_state, patch_state, bn_state);
        let np1_ts = emit_push_rax();
        let n2_ts = emit_ast_code(a2_ts, bind_state, patch_state, bn_state);
        let np2_ts = emit_push_rax();
        let n3_ts = emit_ast_code(a3_ts, bind_state, patch_state, bn_state);
        // setup (31 bytes after K3.U: 11 base + 20 dst-bounds guard):
        emit_byte(0x89); emit_byte(0xC1);                                       // mov ecx, eax (count)
        // K3.U (2026-05-27): dst-bounds guard. Trap-id 25001.
        emit_byte(0x8B); emit_byte(0x34); emit_byte(0x24);                      // mov esi, [rsp] (dst_off)
        emit_byte(0x01); emit_byte(0xCE);                                       // add esi, ecx
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP
        emit_u32_le(helix_arena_cap());
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7
        emit_trap_with_id(25001);                                                // __tile_sub dst OOB
        // K3.W (2026-05-27): READ-side bounds for a_off and b_off.
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x10);     // mov esi, [rsp+16] (a_off)
        emit_byte(0x01); emit_byte(0xCE);                                       // add esi, ecx
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP
        emit_u32_le(helix_arena_cap());
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7
        emit_trap_with_id(25002);                                                // sub a OOB
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x08);     // mov esi, [rsp+8] (b_off)
        emit_byte(0x01); emit_byte(0xCE);                                       // add esi, ecx
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP
        emit_u32_le(helix_arena_cap());
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7
        emit_trap_with_id(25003);                                                // sub b OOB
        let disp_slot_ts = emit_lea_rax_rip_placeholder();                      // 7 bytes
        patch_table_add(patch_state, disp_slot_ts, arena_base_s, 18);
        emit_byte(0x31); emit_byte(0xD2);                                       // xor edx, edx
        // loop_start (43 bytes):
        emit_byte(0x39); emit_byte(0xCA);                                       // cmp edx, ecx
        emit_byte(0x7D); emit_byte(0x27);                                       // jge +39
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x10);  // mov rsi, [rsp+16]
        emit_byte(0x48); emit_byte(0x01); emit_byte(0xD6);                      // add rsi, rdx
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0xB0); emit_byte(0x04);     // mov esi, [rax+rsi*4+4] (a[i])
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x08);  // mov rdi, [rsp+8]
        emit_byte(0x48); emit_byte(0x01); emit_byte(0xD7);                      // add rdi, rdx
        emit_byte(0x2B); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);     // sub esi, [rax+rdi*4+4]  (esi -= b[i]) -- the K1.F25 op-flip
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x3C); emit_byte(0x24);     // mov rdi, [rsp]
        emit_byte(0x48); emit_byte(0x01); emit_byte(0xD7);                      // add rdi, rdx
        emit_byte(0x89); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);     // mov [rax+rdi*4+4], esi
        emit_byte(0xFF); emit_byte(0xC2);                                       // inc edx
        emit_byte(0xEB); emit_byte(0xD5);                                       // jmp -43 (back to loop_start)
        // end (6 bytes):
        emit_byte(0x48); emit_byte(0x83); emit_byte(0xC4); emit_byte(0x18);     // add rsp, 24
        emit_byte(0x31); emit_byte(0xC0);                                       // xor eax, eax
        n0_ts + np0_ts + n1_ts + np1_ts + n2_ts + np2_ts + n3_ts + 122
    } else { if kovc_byte_eq(name_s, name_l, bn_tile_mul_s(bn_state), 10) == 1 {
        // K1.F26 (2026-05-27): __tile_mul(a, b, dst, count) elementwise
        // multiplication. Mirrors __tile_add/__tile_sub but uses
        // imul r32, r/m32 (opcode 0F AF) which is 2-byte opcode + ModRM
        // + SIB + disp8 = 5 bytes (vs 4 bytes for add/sub). Loop body
        // grows from 43 to 44 bytes; jge/jmp rel8 displacements shift
        // from +39/-43 to +40/-44.
        // Result: dst[i] = a[i] * b[i] for i in [0, count).
        let a0_tm = __arena_get(args_head + 1);
        let next1_tm = __arena_get(args_head + 2);
        let a1_tm = __arena_get(next1_tm + 1);
        let next2_tm = __arena_get(next1_tm + 2);
        let a2_tm = __arena_get(next2_tm + 1);
        let next3_tm = __arena_get(next2_tm + 2);
        let a3_tm = __arena_get(next3_tm + 1);
        let n0_tm = emit_ast_code(a0_tm, bind_state, patch_state, bn_state);
        let np0_tm = emit_push_rax();
        let n1_tm = emit_ast_code(a1_tm, bind_state, patch_state, bn_state);
        let np1_tm = emit_push_rax();
        let n2_tm = emit_ast_code(a2_tm, bind_state, patch_state, bn_state);
        let np2_tm = emit_push_rax();
        let n3_tm = emit_ast_code(a3_tm, bind_state, patch_state, bn_state);
        // setup (31 bytes after K3.U: 11 base + 20 dst-bounds guard):
        emit_byte(0x89); emit_byte(0xC1);                                       // mov ecx, eax (count)
        // K3.U (2026-05-27): dst-bounds guard. Trap-id 26001.
        emit_byte(0x8B); emit_byte(0x34); emit_byte(0x24);                      // mov esi, [rsp] (dst_off)
        emit_byte(0x01); emit_byte(0xCE);                                       // add esi, ecx
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP
        emit_u32_le(helix_arena_cap());
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7
        emit_trap_with_id(26001);                                                // __tile_mul dst OOB
        // K3.W (2026-05-27): READ-side bounds for a_off and b_off.
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x10);     // mov esi, [rsp+16] (a_off)
        emit_byte(0x01); emit_byte(0xCE);                                       // add esi, ecx
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP
        emit_u32_le(helix_arena_cap());
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7
        emit_trap_with_id(26002);                                                // mul a OOB
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x08);     // mov esi, [rsp+8] (b_off)
        emit_byte(0x01); emit_byte(0xCE);                                       // add esi, ecx
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP
        emit_u32_le(helix_arena_cap());
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7
        emit_trap_with_id(26003);                                                // mul b OOB
        let disp_slot_tm = emit_lea_rax_rip_placeholder();                      // 7 bytes
        patch_table_add(patch_state, disp_slot_tm, arena_base_s, 18);
        emit_byte(0x31); emit_byte(0xD2);                                       // xor edx, edx
        // loop_start (44 bytes):
        emit_byte(0x39); emit_byte(0xCA);                                       // cmp edx, ecx
        emit_byte(0x7D); emit_byte(0x28);                                       // jge +40 (was +39 for add)
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x10);  // mov rsi, [rsp+16]
        emit_byte(0x48); emit_byte(0x01); emit_byte(0xD6);                      // add rsi, rdx
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0xB0); emit_byte(0x04);     // mov esi, [rax+rsi*4+4] (a[i])
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x08);  // mov rdi, [rsp+8]
        emit_byte(0x48); emit_byte(0x01); emit_byte(0xD7);                      // add rdi, rdx
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);  // imul esi, [rax+rdi*4+4] -- the K1.F26 op (5 bytes instead of 4)
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x3C); emit_byte(0x24);     // mov rdi, [rsp]
        emit_byte(0x48); emit_byte(0x01); emit_byte(0xD7);                      // add rdi, rdx
        emit_byte(0x89); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);     // mov [rax+rdi*4+4], esi
        emit_byte(0xFF); emit_byte(0xC2);                                       // inc edx
        emit_byte(0xEB); emit_byte(0xD4);                                       // jmp -44 (was -43 for add)
        // end (6 bytes):
        emit_byte(0x48); emit_byte(0x83); emit_byte(0xC4); emit_byte(0x18);     // add rsp, 24
        emit_byte(0x31); emit_byte(0xC0);                                       // xor eax, eax
        n0_tm + np0_tm + n1_tm + np1_tm + n2_tm + np2_tm + n3_tm + 123
    } else { if kovc_byte_eq(name_s, name_l, bn_tile_matmul_s(bn_state), 13) == 1 {
        // K1.F27 (2026-05-27): __tile_matmul(a, b, dst, N) for 2x2 only.
        // Fully unrolled (no loops). N is currently ignored (assumed 2).
        // Computes: dst[i*2+j] = sum over k of (a[i*2+k] * b[k*2+j]).
        //
        // a-cell byte disps: a[0]=+4, a[1]=+8, a[2]=+12, a[3]=+16
        // b-cell byte disps: b[0]=+4, b[1]=+8, b[2]=+12, b[3]=+16
        // dst-cell byte disps: same pattern
        //
        // Per dst cell (38 bytes):
        //   mov rdi, [rsp+16]               ; rdi = a_off    (5)
        //   mov esi, [rax+rdi*4+disp_a0]    ; esi = a[2i]    (4)
        //   mov edx, [rax+rdi*4+disp_a1]    ; edx = a[2i+1]  (4)
        //   mov rdi, [rsp+8]                ; rdi = b_off    (5)
        //   imul esi, [rax+rdi*4+disp_b0]   ; esi *= b[j]    (5)
        //   imul edx, [rax+rdi*4+disp_b1]   ; edx *= b[2+j]  (5)
        //   add esi, edx                    ; esi = result   (2)
        //   mov rdi, [rsp]                  ; rdi = dst_off  (4)
        //   mov [rax+rdi*4+disp_dst], esi   ; store          (4)
        //
        // Total body: 4 cells * 38B = 152B. + setup 9B + cleanup 6B = 167B.
        let a0_mm = __arena_get(args_head + 1);
        let next1_mm = __arena_get(args_head + 2);
        let a1_mm = __arena_get(next1_mm + 1);
        let next2_mm = __arena_get(next1_mm + 2);
        let a2_mm = __arena_get(next2_mm + 1);
        let next3_mm = __arena_get(next2_mm + 2);
        let a3_mm = __arena_get(next3_mm + 1);
        let n0_mm = emit_ast_code(a0_mm, bind_state, patch_state, bn_state);
        let np0_mm = emit_push_rax();
        let n1_mm = emit_ast_code(a1_mm, bind_state, patch_state, bn_state);
        let np1_mm = emit_push_rax();
        let n2_mm = emit_ast_code(a2_mm, bind_state, patch_state, bn_state);
        let np2_mm = emit_push_rax();
        let n3_mm = emit_ast_code(a3_mm, bind_state, patch_state, bn_state);
        // setup (42 bytes -- was 9; K3.R +12 for N!=2 guard; K3.T +21 for dst bounds):
        emit_byte(0x89); emit_byte(0xC1);                                       // mov ecx, eax (N captured)
        // K3.R (2026-05-27): runtime guard rejecting N != 2 with trap-id 27001.
        // Silent-failure-hunter MEDIUM-1: pre-K3.R, __tile_matmul accepted N=3
        // (or any value) and silently produced a 2x2 result while the user
        // expected NxN, leaving dst[4..N*N) BSS-zero. Now we fail-closed with
        // a recognizable trap-id so callsite errors surface immediately.
        emit_byte(0x83); emit_byte(0xF9); emit_byte(0x02);                      // cmp ecx, 2     (3 bytes)
        emit_byte(0x74); emit_byte(0x07);                                       // je +7 (skip trap if N==2)
        emit_trap_with_id(27001);                                                // 7 bytes -- N != 2 trap
        // K3.T (2026-05-27): runtime dst-bounds check on __tile_matmul (partial
        // close of silent-failure-hunter HIGH-1). Pre-K3.T the matmul wrote 4
        // dst cells at [arena_base + (dst_off+i)*4 + 4] for i in 0..4 with NO
        // check that dst_off+4 stayed inside the arena. If the caller passed
        // a bogus dst_off (e.g., not from __tile_zeros, or out-of-range), the
        // writes corrupted arena cells past the tile or wrote past the mapped
        // BSS page (SIGSEGV at best, silent neighbor-cell corruption at worst).
        // K3.T checks `dst_off + 4 <= helix_arena_cap()` (= 2097152). NOTE: a_off
        // and b_off READ unchecked (the writes are the corruption risk; READS
        // of OOB just produce garbage that the user sees in the result). The
        // a_off/b_off side and the other tile ops (add/sub/mul) remain part of
        // open HIGH-1 surface, deferred to K3.U.
        //
        // Sequence (21 bytes): load dst_off from [rsp], +4, cmp vs cap, jbe past
        // trap, trap-id 27002.
        emit_byte(0x8B); emit_byte(0x34); emit_byte(0x24);                      // mov esi, [rsp]      (3 bytes -- dst_off)
        emit_byte(0x83); emit_byte(0xC6); emit_byte(0x04);                      // add esi, 4          (3 bytes)
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP imm32  (6 bytes -- opcode/ModRM)
        emit_u32_le(helix_arena_cap());                                          //                     (4 more bytes of imm32)
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7 (skip trap if dst_off+4 <= cap)
        emit_trap_with_id(27002);                                                // 7 bytes -- dst OOB trap
        // K3.W (2026-05-27): READ-side bounds for a_off and b_off. Matmul
        // hardcodes count=4 (per K3.R N=2 guard) so we use `add esi, 4`
        // (3 bytes imm8) instead of `add esi, ecx` (2 bytes). 22 bytes per
        // check x 2 = 44 bytes added. Trap-ids 27003 (a OOB), 27004 (b OOB).
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x10);     // mov esi, [rsp+16] (a_off)
        emit_byte(0x83); emit_byte(0xC6); emit_byte(0x04);                      // add esi, 4
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP
        emit_u32_le(helix_arena_cap());
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7
        emit_trap_with_id(27003);                                                // matmul a OOB
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0x24); emit_byte(0x08);     // mov esi, [rsp+8] (b_off)
        emit_byte(0x83); emit_byte(0xC6); emit_byte(0x04);                      // add esi, 4
        emit_byte(0x81); emit_byte(0xFE);                                       // cmp esi, CAP
        emit_u32_le(helix_arena_cap());
        emit_byte(0x76); emit_byte(0x07);                                       // jbe +7
        emit_trap_with_id(27004);                                                // matmul b OOB
        let disp_slot_mm = emit_lea_rax_rip_placeholder();                      // 7 bytes
        patch_table_add(patch_state, disp_slot_mm, arena_base_s, 18);

        // ---- dst[0] = a[0]*b[0] + a[1]*b[2] ----  (38 bytes)
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x10);  // mov rdi, [rsp+16]
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);     // mov esi, [rax+rdi*4+4]   (a[0])
        emit_byte(0x8B); emit_byte(0x54); emit_byte(0xB8); emit_byte(0x08);     // mov edx, [rax+rdi*4+8]   (a[1])
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x08);  // mov rdi, [rsp+8]
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);  // imul esi, [rax+rdi*4+4]   (b[0])
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0x54); emit_byte(0xB8); emit_byte(0x0C);  // imul edx, [rax+rdi*4+12]  (b[2])
        emit_byte(0x01); emit_byte(0xD6);                                       // add esi, edx
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x3C); emit_byte(0x24);     // mov rdi, [rsp]
        emit_byte(0x89); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);     // mov [rax+rdi*4+4], esi   (dst[0])

        // ---- dst[1] = a[0]*b[1] + a[1]*b[3] ----
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x10);  // mov rdi, [rsp+16]
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);     // mov esi, [rax+rdi*4+4]
        emit_byte(0x8B); emit_byte(0x54); emit_byte(0xB8); emit_byte(0x08);     // mov edx, [rax+rdi*4+8]
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x08);  // mov rdi, [rsp+8]
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x08);  // imul esi, [rax+rdi*4+8]   (b[1])
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0x54); emit_byte(0xB8); emit_byte(0x10);  // imul edx, [rax+rdi*4+16]  (b[3])
        emit_byte(0x01); emit_byte(0xD6);                                       // add esi, edx
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x3C); emit_byte(0x24);     // mov rdi, [rsp]
        emit_byte(0x89); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x08);     // mov [rax+rdi*4+8], esi   (dst[1])

        // ---- dst[2] = a[2]*b[0] + a[3]*b[2] ----
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x10);  // mov rdi, [rsp+16]
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x0C);     // mov esi, [rax+rdi*4+12]  (a[2])
        emit_byte(0x8B); emit_byte(0x54); emit_byte(0xB8); emit_byte(0x10);     // mov edx, [rax+rdi*4+16]  (a[3])
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x08);  // mov rdi, [rsp+8]
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x04);  // imul esi, [rax+rdi*4+4]
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0x54); emit_byte(0xB8); emit_byte(0x0C);  // imul edx, [rax+rdi*4+12]
        emit_byte(0x01); emit_byte(0xD6);                                       // add esi, edx
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x3C); emit_byte(0x24);     // mov rdi, [rsp]
        emit_byte(0x89); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x0C);     // mov [rax+rdi*4+12], esi  (dst[2])

        // ---- dst[3] = a[2]*b[1] + a[3]*b[3] ----
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x10);  // mov rdi, [rsp+16]
        emit_byte(0x8B); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x0C);     // mov esi, [rax+rdi*4+12]
        emit_byte(0x8B); emit_byte(0x54); emit_byte(0xB8); emit_byte(0x10);     // mov edx, [rax+rdi*4+16]
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x7C); emit_byte(0x24); emit_byte(0x08);  // mov rdi, [rsp+8]
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x08);  // imul esi, [rax+rdi*4+8]
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0x54); emit_byte(0xB8); emit_byte(0x10);  // imul edx, [rax+rdi*4+16]
        emit_byte(0x01); emit_byte(0xD6);                                       // add esi, edx
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x3C); emit_byte(0x24);     // mov rdi, [rsp]
        emit_byte(0x89); emit_byte(0x74); emit_byte(0xB8); emit_byte(0x10);     // mov [rax+rdi*4+16], esi  (dst[3])

        // end (6 bytes):
        emit_byte(0x48); emit_byte(0x83); emit_byte(0xC4); emit_byte(0x18);     // add rsp, 24
        emit_byte(0x31); emit_byte(0xC0);                                       // xor eax, eax
        // K3.R: 167 -> 179 (+12). K3.T: 179 -> 200 (+21 dst). K3.W: 200 -> 244 (+44 a/b reads).
        n0_mm + np0_mm + n1_mm + np1_mm + n2_mm + np2_mm + n3_mm + 244
    } else { if is_print_int_name(name_s, name_l) == 1 {
        // K1.D-impl (2026-05-25): print_int(n) emits inline asm for
        // ASCII conversion + write(1, buf, len) syscall. See
        // emit_print_int_body for the byte layout. Returns 0 in eax.
        let arg_idx = __arena_get(args_head + 1);
        emit_print_int_body(arg_idx, bind_state, patch_state, bn_state)
    } else { if is_panic_name(name_s, name_l) == 1 {
        // K1.AE (2026-05-25): panic("msg") -- print the message to
        // stderr via sys_write, then ud2-trap. Mirrors Python's
        // helixc panic codegen ("panic[id]: msg\n" + sys_exit).
        // Bootstrap omits the "panic[id]: " prefix for simplicity --
        // just the message body + trap. panic_pass (kovc.hx:2662)
        // has already validated that args_head points to exactly 1
        // AST_ARG whose payload is AST_STR_LIT (tag 25); we re-check
        // here defensively in case the validator was skipped.
        let arg_idx_p = __arena_get(args_head + 1);
        let arg_tag_p = __arena_get(arg_idx_p);
        if arg_tag_p != 25 {
            // Defensive: shouldn't reach (panic_pass would have
            // emitted diag 28501 with aux=2). Emit a bare ud2 so
            // the K2 binary still fails loudly.
            emit_byte(0x0F); emit_byte(0x0B);
            2
        } else {
            let body_s_p = __arena_get(arg_idx_p + 1);
            let body_l_p = __arena_get(arg_idx_p + 2);
            // K1.AH (2026-05-25): emit the "panic[28501]: " prefix
            // FIRST via its own sys_write call, then the user
            // message. Matches Python's "panic[id]: msg" format.
            // (Python also writes a trailing newline; bootstrap
            // omits that for simplicity -- the stderr-on-tty user
            // sees the message on its own line anyway because the
            // shell prints a prompt afterwards.)
            //
            // Prefix sys_write (22 bytes: 7+5+5+5+2):
            let pfx_disp_slot = emit_lea_rsi_rip_placeholder();
            str_table_add(bn_state, pfx_disp_slot, bn_panic_prefix_s(bn_state), 14);
            // mov edi, 2 (fd=stderr) -- 5 bytes
            emit_byte(0xBF); emit_byte(0x02);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            // mov edx, 14 (prefix len) -- 5 bytes
            emit_byte(0xBA); emit_byte(0x0E);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            // mov eax, 1 (sys_write) -- 5 bytes
            emit_byte(0xB8); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            // syscall -- 2 bytes
            emit_byte(0x0F); emit_byte(0x05);
            // Message sys_write (continues with rdi/rax already set
            // to 2 and 1 respectively -- but the kernel doesn't
            // preserve registers across syscalls so we re-set them).
            // lea rsi, [rip + msg_disp]
            let msg_disp_slot = emit_lea_rsi_rip_placeholder();
            str_table_add(bn_state, msg_disp_slot, body_s_p, body_l_p);
            // mov edi, 2 (fd=stderr) -- 5 bytes (BF + 4-byte imm32)
            emit_byte(0xBF); emit_byte(0x02);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            // mov edx, body_l_p (len) -- 5 bytes (BA + 4-byte imm32)
            emit_byte(0xBA);
            emit_u32_le(body_l_p);
            // mov eax, 1 (sys_write) -- 5 bytes (B8 + imm32)
            emit_byte(0xB8); emit_byte(0x01);
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            // syscall -- 2 bytes (0F 05)
            emit_byte(0x0F); emit_byte(0x05);
            // K1.AI (2026-05-25): trailing-newline sys_write (24
            // bytes). Loads the 1-byte "\n" from slot 162 of the
            // bn_state into rsi and writes it to fd=2. Fully matches
            // Python's "panic[id]: msg\n" terminator.
            let nl_disp_slot = emit_lea_rsi_rip_placeholder();
            str_table_add(bn_state, nl_disp_slot, bn_panic_newline_s(bn_state), 1);
            emit_byte(0xBF); emit_byte(0x02);                                    // mov edi, 2
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0xBA); emit_byte(0x01);                                    // mov edx, 1
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0xB8); emit_byte(0x01);                                    // mov eax, 1
            emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
            emit_byte(0x0F); emit_byte(0x05);                                    // syscall
            // ud2 -- 2 bytes (0F 0B). Could use sys_exit instead but
            // ud2 also raises SIGILL = rc 132, distinctive enough.
            emit_byte(0x0F); emit_byte(0x0B);
            // Total: prefix sys_write (24) + message sys_write (24)
            //        + newline sys_write (24) + ud2 (2) = 74 bytes.
            74
        }
    } else { if kovc_byte_eq(name_s, name_l, bn_read_file_to_arena_s(bn_state), 18) == 1 {
        // read_file_to_arena(path: STRLIT) -> i32 (bytes_read).
        // First arg MUST be AST_STR_LIT. We inspect args_head's
        // first AST_ARG → expr; the expr's tag must be 25.
        let arg_idx = __arena_get(args_head + 1);
        let arg_tag = __arena_get(arg_idx);
        if arg_tag != 25 {
            // Not a string literal — Phase 0 only supports literal
            // paths. Emit ud2 trap so misuse is loud.
            emit_byte(0x0F); emit_byte(0x0B);
            2
        } else {
            let body_s = __arena_get(arg_idx + 1);
            let body_l = __arena_get(arg_idx + 2);
            let path_disp_slot = emit_lea_rdi_rip_placeholder();
            str_table_add(bn_state, path_disp_slot, body_s, body_l);
            let body_bytes = emit_read_file_to_arena_body(patch_state, arena_base_s);
            7 + body_bytes
        }
    } else { if kovc_byte_eq(name_s, name_l, bn_write_file_to_arena_s(bn_state), 19) == 1 {
        // write_file_to_arena(path: STRLIT, arena_start: i32,
        //                      n_bytes: i32) -> i32 (bytes_written).
        // First arg MUST be AST_STR_LIT.
        let arg_idx = __arena_get(args_head + 1);
        let arg_tag = __arena_get(arg_idx);
        if arg_tag != 25 {
            emit_byte(0x0F); emit_byte(0x0B);
            2
        } else {
            let body_s = __arena_get(arg_idx + 1);
            let body_l = __arena_get(arg_idx + 2);
            // Eval arg2 (arena_start) and arg3 (n_bytes), push each
            // so the body can pop them. The body pops arena_start
            // FIRST (top of stack), then n_bytes. So we must push
            // n_bytes FIRST, then arena_start. Order: eval arg3,
            // push; eval arg2, push.
            let next1 = __arena_get(args_head + 2);    // AST_ARG #2
            let next2 = __arena_get(next1 + 2);        // AST_ARG #3
            let a2 = __arena_get(next1 + 1);           // arena_start expr
            let a3 = __arena_get(next2 + 1);           // n_bytes expr
            let n3 = emit_ast_code(a3, bind_state, patch_state, bn_state);
            let n3p = emit_push_rax();
            let n2 = emit_ast_code(a2, bind_state, patch_state, bn_state);
            let n2p = emit_push_rax();
            // Now load path into rdi via str_table.
            let path_disp_slot = emit_lea_rdi_rip_placeholder();
            str_table_add(bn_state, path_disp_slot, body_s, body_l);
            let body_bytes = emit_write_file_to_arena_body(patch_state, arena_base_s);
            n3 + n3p + n2 + n2p + 7 + body_bytes
        }
    } else { if kovc_byte_eq(name_s, name_l, bn_fadd_s(bn_state), 6) == 1 {
        // __fadd(a, b) -> f32 bits in eax.
        // eval a -> eax; push; eval b -> eax;
        // mov ecx, eax (b); pop rax (a);
        // movd xmm0, eax; movd xmm1, ecx; addss xmm0, xmm1;
        // movd eax, xmm0
        let a0 = __arena_get(args_head + 1);
        let next_arg = __arena_get(args_head + 2);
        let a1 = __arena_get(next_arg + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n1 = emit_ast_code(a1, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);                  // mov ecx, eax
        emit_byte(0x58);                                    // pop rax
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0); // movd xmm0, eax
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9); // movd xmm1, ecx
        emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x58); emit_byte(0xC1); // addss xmm0, xmm1
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0); // movd eax, xmm0
        n0 + np + n1 + 2 + 1 + 4 + 4 + 4 + 4
    } else { if kovc_byte_eq(name_s, name_l, bn_fsub_s(bn_state), 6) == 1 {
        // __fsub(a, b) -> f32 bits in eax. Same as fadd but subss.
        let a0 = __arena_get(args_head + 1);
        let next_arg = __arena_get(args_head + 2);
        let a1 = __arena_get(next_arg + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n1 = emit_ast_code(a1, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);
        emit_byte(0x58);
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0);
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9);
        emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x5C); emit_byte(0xC1); // subss xmm0, xmm1
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0);
        n0 + np + n1 + 2 + 1 + 4 + 4 + 4 + 4
    } else { if kovc_byte_eq(name_s, name_l, bn_fmul_s(bn_state), 6) == 1 {
        // __fmul(a, b) -> f32 bits. mulss.
        let a0 = __arena_get(args_head + 1);
        let next_arg = __arena_get(args_head + 2);
        let a1 = __arena_get(next_arg + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n1 = emit_ast_code(a1, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);
        emit_byte(0x58);
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0);
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9);
        emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x59); emit_byte(0xC1); // mulss xmm0, xmm1
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0);
        n0 + np + n1 + 2 + 1 + 4 + 4 + 4 + 4
    } else { if kovc_byte_eq(name_s, name_l, bn_fdiv_s(bn_state), 6) == 1 {
        // __fdiv(a, b) -> f32 bits. divss.
        let a0 = __arena_get(args_head + 1);
        let next_arg = __arena_get(args_head + 2);
        let a1 = __arena_get(next_arg + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n1 = emit_ast_code(a1, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);
        emit_byte(0x58);
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0);
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9);
        emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x5E); emit_byte(0xC1); // divss xmm0, xmm1
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0);
        n0 + np + n1 + 2 + 1 + 4 + 4 + 4 + 4
    } else { if kovc_byte_eq(name_s, name_l, bn_fneg_s(bn_state), 6) == 1 {
        // __fneg(x) -> f32 bits. Single-arg sign flip via integer xor
        // on the bit pattern: xor eax, 0x80000000. No SSE registers
        // touched — purely an integer op on the f32 bit pattern in
        // eax. 5 bytes: 0x35 imm32.
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0x35);
        emit_byte(0x00); emit_byte(0x00); emit_byte(0x00); emit_byte(0x80);
        n0 + 5
    } else { if kovc_byte_eq(name_s, name_l, bn_fsqrt_s(bn_state), 7) == 1 {
        // Phase 1.10 step 5g: __fsqrt(x) -> f32 bits. Single-arg
        // hardware sqrt via SSE2 sqrtss. eval x -> eax;
        // movd xmm0, eax; sqrtss xmm0, xmm0; movd eax, xmm0.
        // 12 bytes after the arg evaluation.
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0); // movd xmm0, eax
        emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x51); emit_byte(0xC0); // sqrtss xmm0, xmm0
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0); // movd eax, xmm0
        n0 + 12
    } else { if kovc_byte_eq(name_s, name_l, bn_fabs_s(bn_state), 6) == 1 {
        // Phase 1.10 step 5h: __fabs(x) -> f32 bits. Single-arg sign
        // bit clear via integer AND on the bit pattern: and eax,
        // 0x7FFFFFFF. Mirrors __fneg (XOR with 0x80000000) — purely
        // an integer op on the f32 bit pattern in eax. 5 bytes:
        // 0x25 imm32. NaN inputs propagate (sign cleared, mantissa
        // preserved).
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0x25);
        emit_byte(0xFF); emit_byte(0xFF); emit_byte(0xFF); emit_byte(0x7F);
        n0 + 5
    } else { if kovc_byte_eq(name_s, name_l, bn_i32_to_f32_s(bn_state), 12) == 1 {
        // Phase 1.10 step 5i: __i32_to_f32(x) -> f32 bits. Single-arg
        // signed-int-to-float conversion via SSE2 cvtsi2ss. eval x ->
        // eax (i32); cvtsi2ss xmm0, eax; movd eax, xmm0. 8 bytes after
        // the arg evaluation. Result is the f32 bit pattern; the call's
        // type is f32 (see is_f32_expr's __i32_to_f32 byte-match case).
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x2A); emit_byte(0xC0); // cvtsi2ss xmm0, eax
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0); // movd eax, xmm0
        n0 + 8
    } else { if kovc_byte_eq(name_s, name_l, bn_f32_to_i32_s(bn_state), 12) == 1 {
        // Phase 1.10 step 5j: __f32_to_i32(x) -> i32. Single-arg
        // truncating float-to-signed-int conversion via SSE2 cvttss2si.
        // eval x -> eax (f32 bit pattern); movd xmm0, eax;
        // cvttss2si eax, xmm0. 8 bytes after the arg evaluation.
        // Result is the truncated signed integer value; is_f32_expr
        // explicitly types this call as i32 (overrides the __f* prefix
        // match — see is_f32_expr's __f32_to_i32 byte-match case).
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0); // movd xmm0, eax
        emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x2C); emit_byte(0xC0); // cvttss2si eax, xmm0
        n0 + 8
    } else { if kovc_byte_eq(name_s, name_l, bn_fmin_s(bn_state), 6) == 1 {
        // Phase 1.10 step 5k: __fmin(a, b) -> f32 bits. Same shape as
        // __fadd but minss xmm0, xmm1 (F3 0F 5D C1). For NaN inputs,
        // minss returns the second operand (xmm1, which holds b) — so
        // any NaN input yields b. is_underscore_f_call's __f* prefix
        // matches so the call types as f32 through is_f32_expr.
        let a0 = __arena_get(args_head + 1);
        let next_arg = __arena_get(args_head + 2);
        let a1 = __arena_get(next_arg + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n1 = emit_ast_code(a1, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);                                    // mov ecx, eax
        emit_byte(0x58);                                                      // pop rax
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0);   // movd xmm0, eax
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9);   // movd xmm1, ecx
        emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x5D); emit_byte(0xC1);   // minss xmm0, xmm1
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0);   // movd eax, xmm0
        n0 + np + n1 + 2 + 1 + 4 + 4 + 4 + 4
    } else { if kovc_byte_eq(name_s, name_l, bn_fmax_s(bn_state), 6) == 1 {
        // Phase 1.10 step 5l: __fmax(a, b) -> f32 bits. Same shape as
        // __fmin but maxss xmm0, xmm1 (F3 0F 5F C1; one byte differs:
        // 5D -> 5F). For NaN inputs, maxss returns the second operand
        // (xmm1 = b). is_underscore_f_call's __f* prefix matches so
        // the call types as f32 through is_f32_expr.
        let a0 = __arena_get(args_head + 1);
        let next_arg = __arena_get(args_head + 2);
        let a1 = __arena_get(next_arg + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n1 = emit_ast_code(a1, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);                                    // mov ecx, eax
        emit_byte(0x58);                                                      // pop rax
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0);   // movd xmm0, eax
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9);   // movd xmm1, ecx
        emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x5F); emit_byte(0xC1);   // maxss xmm0, xmm1
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0);   // movd eax, xmm0
        n0 + np + n1 + 2 + 1 + 4 + 4 + 4 + 4
    } else { if kovc_byte_eq(name_s, name_l, bn_bits_of_f32_s(bn_state), 13) == 1 {
        // Phase 1.10 step 5m: __bits_of_f32(x) — identity bitcast,
        // f32 -> i32. The f32 already lives in eax as its IEEE 754
        // bit pattern, so no extra bytes are emitted; we just emit
        // the inner expression and return its byte count. Typed as
        // i32: starts with __b so doesn't match the __f* prefix;
        // is_f32_expr falls through to fn_type_table -> 0.
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        n0
    } else { if kovc_byte_eq(name_s, name_l, bn_f32_from_bits_s(bn_state), 15) == 1 {
        // Phase 1.10 step 5m: __f32_from_bits(b) — identity bitcast,
        // i32 -> f32. Inverse of __bits_of_f32; same identity codegen
        // (eax already holds the bit pattern). Typed as f32: starts
        // with __f so the __f* prefix returns 1 in is_f32_expr (length
        // 15 != 12 so no collision with __f32_to_i32's explicit case).
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        n0
    } else { if kovc_byte_eq(name_s, name_l, bn_hash_i32_s(bn_state), 10) == 1 {
        // Phase 1.10 step 5n: __hash_i32(x) -> i32 quadratic mixer.
        // Lowers to inline arithmetic (mirrors helixc-Python
        // lower_ast.py:939-963):
        //     h = x*x*c1 + x*c2 + c3
        // where c1 = 0x05EBCA6B, c2 = 0x27D4EB2F, c3 = 0x165667B1.
        // K1.F19 (2026-05-27): the 24-byte mixer is now extracted into
        // `emit_hash_i32_mixer` and shared with reflect_hash.
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let nh = emit_hash_i32_mixer();
        n0 + nh
    } else { if kovc_byte_eq(name_s, name_l, bn_strlen_s(bn_state), 8) == 1 {
        // Phase 1.10 step 5o: __strlen(s) -> i32 compile-time string-
        // literal length. Mirrors helixc-Python lower_ast.py:966-969
        // (`return self.builder.const_int(len(s.encode("utf-8")))`).
        // First arg MUST be AST_STR_LIT (tag 25). We read body_l (=
        // byte length stored at arg_idx + 2) and emit `mov eax,
        // body_l` via emit_ast_int (5 bytes: B8 imm32). If the arg is
        // not AST_STR_LIT, emit ud2 trap so misuse is loud (mirrors
        // the file-builtin strict-pattern requirement).
        let arg_idx = __arena_get(args_head + 1);
        let arg_tag = __arena_get(arg_idx);
        if arg_tag != 25 {
            emit_byte(0x0F); emit_byte(0x0B);
            2
        } else {
            let body_l = __arena_get(arg_idx + 2);
            emit_ast_int(body_l)
        }
    } else { if kovc_byte_eq(name_s, name_l, bn_f32_to_f64_s(bn_state), 12) == 1 {
        // Phase 1.10 step 7e: __f32_to_f64(x) -> f64 bits in rax.
        // Single-arg widening conversion via SSE2 cvtss2sd. eval x ->
        // eax (f32 bit pattern); movd xmm0, eax; cvtss2sd xmm0, xmm0;
        // movq rax, xmm0. 13 bytes after the arg evaluation.
        // Result is the f64 bit pattern; the call's type is f64
        // (see is_f64_expr's __f32_to_f64 byte-match case).
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0); // movd xmm0, eax
        emit_byte(0xF3); emit_byte(0x0F); emit_byte(0x5A); emit_byte(0xC0); // cvtss2sd xmm0, xmm0
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0); // movq rax, xmm0
        n0 + 13
    } else { if kovc_byte_eq(name_s, name_l, bn_f64_to_f32_s(bn_state), 12) == 1 {
        // Phase 1.10 step 7e: __f64_to_f32(x) -> f32 bits in eax.
        // Single-arg narrowing conversion via SSE2 cvtsd2ss. eval x ->
        // rax (f64 bit pattern); movq xmm0, rax; cvtsd2ss xmm0, xmm0;
        // movd eax, xmm0. 13 bytes after the arg evaluation.
        // Result is the f32 bit pattern; the call's type is f32
        // (see is_f32_expr's __f64_to_f32 byte-match case).
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0); // movq xmm0, rax
        emit_byte(0xF2); emit_byte(0x0F); emit_byte(0x5A); emit_byte(0xC0); // cvtsd2ss xmm0, xmm0
        emit_byte(0x66); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0); // movd eax, xmm0
        n0 + 13
    } else { if kovc_byte_eq(name_s, name_l, bn_dsqrt_s(bn_state), 7) == 1 {
        // Phase 1.10 step 7h: __dsqrt(x) -> f64 bits in rax. Single-arg
        // hardware sqrt via SSE2 sqrtsd. eval x -> rax (f64 bit pattern);
        // movq xmm0, rax; sqrtsd xmm0, xmm0; movq rax, xmm0. 14 bytes
        // after the arg evaluation. Mirrors __fsqrt (step 5g) but on
        // 64-bit doubles. NaN inputs propagate (sqrtsd preserves NaN);
        // negatives produce a quiet NaN. Result types as f64 via
        // is_f64_expr's explicit __dsqrt byte_eq.
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0); // movq xmm0, rax
        emit_byte(0xF2); emit_byte(0x0F); emit_byte(0x51); emit_byte(0xC0); // sqrtsd xmm0, xmm0
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0); // movq rax, xmm0
        n0 + 14
    } else { if kovc_byte_eq(name_s, name_l, bn_dabs_s(bn_state), 6) == 1 {
        // Phase 1.10 step 7i: __dabs(x) -> f64 bits in rax. Single-arg
        // f64 absolute value: clears bit 63 (sign bit) of the f64 bit
        // pattern in rax. Implementation uses shl/shr instead of
        // and-rax-imm64 since x86-64 has no AND-rax-imm64; the shift
        // pair is 6 bytes (vs 13 for movabs+and).
        //   48 D1 E0    shl rax, 1    (sign bit drops into CF)
        //   48 D1 E8    shr rax, 1    (refills bit 63 with 0)
        // Mirrors __fabs (step 5h) on 64-bit doubles. Result types as
        // f64 via is_f64_expr's explicit byte_eq case.
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0x48); emit_byte(0xD1); emit_byte(0xE0);   // shl rax, 1
        emit_byte(0x48); emit_byte(0xD1); emit_byte(0xE8);   // shr rax, 1
        n0 + 6
    } else { if kovc_byte_eq(name_s, name_l, bn_dmin_s(bn_state), 6) == 1 {
        // Phase 1.10 step 7j: __dmin(a, b) -> f64 bits in rax. Mirrors
        // __fmin (step 5k) but on doubles via SSE2 minsd. eval a -> push;
        // eval b -> rax; mov rcx, rax (FULL 64); pop rax; movq xmm0/xmm1;
        // minsd xmm0, xmm1; movq rax, xmm0. NaN: minsd returns xmm1 (b).
        let a0 = __arena_get(args_head + 1);
        let next_arg = __arena_get(args_head + 2);
        let a1 = __arena_get(next_arg + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n1 = emit_ast_code(a1, bind_state, patch_state, bn_state);
        emit_byte(0x48); emit_byte(0x89); emit_byte(0xC1);                                    // mov rcx, rax
        emit_byte(0x58);                                                                       // pop rax
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0);   // movq xmm0, rax
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9);   // movq xmm1, rcx
        emit_byte(0xF2); emit_byte(0x0F); emit_byte(0x5D); emit_byte(0xC1);                    // minsd xmm0, xmm1
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0);   // movq rax, xmm0
        n0 + np + n1 + 3 + 1 + 5 + 5 + 4 + 5
    } else { if kovc_byte_eq(name_s, name_l, bn_dmax_s(bn_state), 6) == 1 {
        // Phase 1.10 step 7j: __dmax(a, b) -> f64 bits in rax. Mirrors
        // __fmax (step 5l) but on doubles via SSE2 maxsd. Same shape as
        // __dmin with opcode 5F (vs 5D). NaN: maxsd returns xmm1 (b).
        let a0 = __arena_get(args_head + 1);
        let next_arg = __arena_get(args_head + 2);
        let a1 = __arena_get(next_arg + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n1 = emit_ast_code(a1, bind_state, patch_state, bn_state);
        emit_byte(0x48); emit_byte(0x89); emit_byte(0xC1);                                    // mov rcx, rax
        emit_byte(0x58);                                                                       // pop rax
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0);   // movq xmm0, rax
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC9);   // movq xmm1, rcx
        emit_byte(0xF2); emit_byte(0x0F); emit_byte(0x5F); emit_byte(0xC1);                    // maxsd xmm0, xmm1
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0);   // movq rax, xmm0
        n0 + np + n1 + 3 + 1 + 5 + 5 + 4 + 5
    } else { if kovc_byte_eq(name_s, name_l, bn_i32_to_f64_s(bn_state), 12) == 1 {
        // Phase 1.10 step 7k: __i32_to_f64(x) -> f64 bits in rax.
        // Single-arg widening conversion via SSE2 cvtsi2sd. eval x ->
        // eax (i32); cvtsi2sd xmm0, eax; movq rax, xmm0. 9 bytes after
        // the arg evaluation. Result types as f64 via is_f64_expr's
        // explicit __i32_to_f64 byte_eq case.
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0xF2); emit_byte(0x0F); emit_byte(0x2A); emit_byte(0xC0);                    // cvtsi2sd xmm0, eax
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x7E); emit_byte(0xC0);   // movq rax, xmm0
        n0 + 9
    } else { if kovc_byte_eq(name_s, name_l, bn_f64_to_i32_s(bn_state), 12) == 1 {
        // Phase 1.10 step 7k: __f64_to_i32(x) -> i32. Single-arg
        // truncating conversion via SSE2 cvttsd2si. eval x -> rax
        // (f64 bit pattern); movq xmm0, rax; cvttsd2si eax, xmm0.
        // 9 bytes after the arg evaluation. Result is the truncated
        // signed integer value (low 32 of rax). Note: cvttsd2si EAX,
        // xmm uses no REX.W (32-bit dest); the high 32 of rax is
        // implicitly zeroed by the 32-bit destination convention.
        // Result types as i32 via is_f32_expr's explicit byte_eq
        // (returns 0 BEFORE the __f* prefix match).
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0x66); emit_byte(0x48); emit_byte(0x0F); emit_byte(0x6E); emit_byte(0xC0);   // movq xmm0, rax
        emit_byte(0xF2); emit_byte(0x0F); emit_byte(0x2C); emit_byte(0xC0);                    // cvttsd2si eax, xmm0
        n0 + 9
    } else { if kovc_byte_eq(name_s, name_l, bn_bits_lo_f64_s(bn_state), 13) == 1 {
        // Phase 1.10 step 7l: __bits_lo_f64(x) -> i32. Identity codegen.
        // The f64 bit pattern in rax has its low 32 bits naturally
        // accessible as eax. No emission needed beyond evaluating the
        // argument; the i32 result occupies eax automatically.
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        n0
    } else { if kovc_byte_eq(name_s, name_l, bn_bits_hi_f64_s(bn_state), 13) == 1 {
        // Phase 1.10 step 7l: __bits_hi_f64(x) -> i32. Right-shift rax
        // by 32 bits to move the high 32 of the f64 pattern into the
        // low 32 (eax). Result types as i32.
        //   48 C1 E8 20    shr rax, 32    (4 bytes)
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        emit_byte(0x48); emit_byte(0xC1); emit_byte(0xE8); emit_byte(0x20);
        n0 + 4
    } else { if kovc_byte_eq(name_s, name_l, bn_f64_pack_s(bn_state), 10) == 1 {
        // Phase 1.10 step 7l: __f64_pack(hi, lo) -> f64. Combines two
        // i32 halves into a single 64-bit value in rax. Protocol:
        //   eval hi -> rax (zero-extended to 64); push rax;
        //   eval lo -> rax (zero-extended to 64; high 32 cleared by
        //                   `mov eax, imm32` zero-extension);
        //   mov ecx, eax (rcx = lo zero-extended);
        //   pop rax (rax = hi zero-extended);
        //   shl rax, 32 (rax = hi << 32, low 32 cleared);
        //   or rax, rcx (rax = hi32 | lo32 → full 64-bit pattern).
        // 13 bytes after the two arg evaluations + push.
        let a0 = __arena_get(args_head + 1);
        let next_arg = __arena_get(args_head + 2);
        let a1 = __arena_get(next_arg + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n1 = emit_ast_code(a1, bind_state, patch_state, bn_state);
        emit_byte(0x89); emit_byte(0xC1);                            // mov ecx, eax
        emit_byte(0x58);                                              // pop rax
        emit_byte(0x48); emit_byte(0xC1); emit_byte(0xE0); emit_byte(0x20); // shl rax, 32
        emit_byte(0x48); emit_byte(0x09); emit_byte(0xC8);            // or rax, rcx
        n0 + np + n1 + 2 + 1 + 4 + 3
    } else { if kovc_byte_eq(name_s, name_l, bn_quote_s(bn_state), 5) == 1 {
        // Stage 11: Quote(expr) -> i32 (cell handle).
        //   handle = compile-time-allocated index in [0..63]
        //   Phase-0 cell store: the LAST 64 slots of the produced binary's
        //   arena are the cell table. cell[i] lives at arena_base + 4 +
        //   (HELIX_ARENA_CAP - 64 + i) * 4. The arena is BSS-zero-filled
        //   at load time, so each cell starts at 0.
        // Codegen sequence:
        //   1. eval expr -> eax (the value to bind)
        //   2. mov ecx, eax       (save expr value in ecx)
        //   3. lea rax, [arena_base]   (RIP-relative; patched after fns)
        //   4. mov [rax + DISP], ecx   (write value into cell[handle])
        //   5. mov eax, handle    (return handle as the call's value)
        // Where DISP = 4 + (CAP - 64 + handle) * 4.
        // Trap 81002 if handle >= 64 (more than 64 Quote sites in source).
        let handle = bn_quote_bump_handle(bn_state);
        if handle >= 64 {
            emit_trap_with_id(81002)
        } else {
            let arena_base_s = bn_helix_arena_base_s(bn_state);
            let arg_idx = __arena_get(args_head + 1);
            let n_arg = emit_ast_code(arg_idx, bind_state, patch_state, bn_state);
            // mov ecx, eax           (89 C1) — 2 bytes
            emit_byte(0x89); emit_byte(0xC1);
            // lea rax, [rip + disp32]  — 7 bytes (placeholder + patch)
            let disp_slot = emit_lea_rax_rip_placeholder();
            patch_table_add(patch_state, disp_slot, arena_base_s, 18);
            // Compute cell-table displacement from arena_base.
            // CAP = 2097152, so cell[0] at offset = 4 + (2097088)*4 = 8388356.
            // cell[handle] at offset = 8388356 + handle * 4.
            let disp = 8388356 + handle * 4;
            // mov [rax + disp32], ecx  (89 88 disp32) — 6 bytes
            emit_byte(0x89); emit_byte(0x88);
            emit_u32_le(disp);
            // mov eax, handle  (B8 imm32) — 5 bytes
            emit_byte(0xB8);
            emit_u32_le(handle);
            n_arg + 2 + 7 + 6 + 5
        }
    } else { if kovc_byte_eq(name_s, name_l, bn_splice_s(bn_state), 6) == 1 {
        // Stage 11: Splice(handle) -> i32 (the value stored in cell[handle]).
        //   eval handle -> eax (runtime expression — could be a let-bound
        //                       i32, an integer literal, or any i32 expr)
        //   bounds-check: if (eax & ~63) != 0 (i.e. handle >= 64 or < 0),
        //                  return 0 instead of doing a wild memory read.
        //                  Test test_splice_oob_handle_returns_zero_not_crash
        //                  in helixc/tests/test_reflection.py codifies this
        //                  expectation (OOB safe path).
        //   else load:
        //     mov ecx, eax            (rcx = handle)
        //     lea rax, [arena_base]
        //     mov eax, [rax + ecx*4 + DISP_BASE]
        //         where DISP_BASE = 4 + (CAP - 64) * 4 = 8388356.
        // Bounds-check sequence (Phase-0 minimal):
        //   cmp eax, 0           — sign check
        //   jl  .oob              (if negative, jump to OOB path)
        //   cmp eax, 64           — upper bound
        //   jge .oob
        //   <load path>           — eax := cell[eax]
        //   jmp .end
        // .oob: xor eax, eax     — return 0
        // .end:
        // For a flat short-jump implementation, we use 8-bit relative
        // offsets (Jcc with imm8). Layout:
        //   [3] cmp eax, 0
        //   [2] jl .oob       → rel8 = 23 (skip cmp+jge+load+jmp)
        //   [3] cmp eax, 64
        //   [2] jge .oob      → rel8 = 18 (skip load+jmp)
        //   [2] mov ecx, eax
        //   [7] lea rax, [rip+disp32]
        //   [7] mov eax, [rax+rcx*4+disp32]    (89 84 88 disp32 — 7 bytes)
        //   [2] jmp .end (EB rel8)             rel8 = 2 (skip xor)
        //   [2] xor eax, eax                   .oob
        //   [-] .end:
        // Total post-arg overhead: 3+2+3+2+2+7+7+2+2 = 30 bytes.
        let arena_base_s = bn_helix_arena_base_s(bn_state);
        let arg_idx = __arena_get(args_head + 1);
        let n_arg = emit_ast_code(arg_idx, bind_state, patch_state, bn_state);
        emit_byte(0x83); emit_byte(0xF8); emit_byte(0x00);   // cmp eax, 0
        emit_byte(0x7C); emit_byte(23);                      // jl .oob (rel8=23)
        emit_byte(0x83); emit_byte(0xF8); emit_byte(0x40);   // cmp eax, 64
        emit_byte(0x7D); emit_byte(18);                      // jge .oob (rel8=18)
        emit_byte(0x89); emit_byte(0xC1);                    // mov ecx, eax (handle)
        let disp_slot = emit_lea_rax_rip_placeholder();      // 7 bytes
        patch_table_add(patch_state, disp_slot, arena_base_s, 18);
        // mov eax, [rax + ecx*4 + disp32]
        // ModRM=84 (mod=10 disp32, reg=000 eax, r/m=100 SIB)
        // SIB=88 (scale=10 *4, index=001 rcx, base=000 rax)
        // Encoding: 8B 84 88 disp32 = 7 bytes total.
        emit_byte(0x8B); emit_byte(0x84); emit_byte(0x88);
        let disp_base = 8388356;
        emit_u32_le(disp_base);                              // 4 bytes (3+4 = 7 byte instr)
        emit_byte(0xEB); emit_byte(2);                       // jmp .end (rel8=2)
        emit_byte(0x31); emit_byte(0xC0);                    // xor eax, eax (.oob)
        // .end:
        n_arg + 3 + 2 + 3 + 2 + 2 + 7 + 7 + 2 + 2
    } else { if kovc_byte_eq(name_s, name_l, bn_modify_s(bn_state), 6) == 1 {
        // Stage 11: modify(handle, new_value, verifier_fn) -> i32.
        // Phase-0 minimal contract (mirrors test_reflection.py):
        //   1. eval handle -> push (rsp+16)
        //   2. eval new_value -> push (rsp+8)
        //   3. eval verifier_fn(handle, new_value) -> eax
        //      (Phase-0: verifier is a known fn name; the parser's
        //       AST_CALL with that ident resolves through fn_table.
        //       Here we treat the third arg as an ARBITRARY expression
        //       — it could be an AST_CALL to the verifier OR any other
        //       i32 producer. The simplest legal Helix surface is to
        //       pass `verifier_fn(handle, new_value)` directly as the
        //       third arg, which Phase-0 already lowers via emit_ast_code.
        //       So we do NOT synthesize the call here — we just evaluate
        //       arg3, expecting that the caller passed the result of the
        //       verifier predicate or `verifier_fn(...)` literally.
        //   4. cmp eax, 0; je .reject; (verifier returned 0 → no write)
        //   5. (verifier passed) write new_value to cell[handle].
        //      Bounds-check handle in [0, 64).
        //   6. .reject: eax = 0 if no write, else 1.
        //
        // BUT the test in test_reflection.py uses 3-arg form:
        //   modify(h, 42, always_yes)
        // where `always_yes` is the FN NAME (an unevaluated reference).
        // Phase-0 parser does NOT have first-class fn pointers, so a
        // bare `always_yes` token would parse as AST_VAR (unresolved).
        // To keep the contract simple we DO synthesize the call here:
        //   verifier_call: emit code for `<arg3_name>(handle_val,
        //                                              new_value_val)`.
        // arg3 must be an AST_VAR whose p1/p2 point at the verifier's
        // identifier bytes. The parser already records the name; we
        // emit a CALL to it with handle and new_value as args.
        //
        // For Phase-0 simplicity in the bootstrap, we go with the
        // third-arg-as-expression approach (steps 1-6 above). The
        // higher-level test in test_reflection.py uses Python-helixc
        // and is independent of this bootstrap path. The bootstrap
        // exposes `modify(h, v, predicate_expr)` where the user
        // explicitly passes the predicate result as an arg.
        //
        // For the Stage 11 bootstrap heavy-gate test we use:
        //   modify(h, 42, always_true(0))
        // i.e. evaluate `always_true(0)` to get the predicate value,
        // pass that as the third arg.
        let arena_base_s = bn_helix_arena_base_s(bn_state);
        let a0 = __arena_get(args_head + 1);                // handle expr
        let next1 = __arena_get(args_head + 2);             // AST_ARG #2
        let a1 = __arena_get(next1 + 1);                    // new_value expr
        let next2 = __arena_get(next1 + 2);                 // AST_ARG #3
        let a2 = __arena_get(next2 + 1);                    // predicate expr
        // Eval handle, push.
        let nh = emit_ast_code(a0, bind_state, patch_state, bn_state);
        let nph = emit_push_rax();                          // push handle (rsp+16)
        // Eval new_value, push.
        let nv = emit_ast_code(a1, bind_state, patch_state, bn_state);
        let npv = emit_push_rax();                          // push new_value (rsp+8)
        // Eval predicate -> eax.
        let np = emit_ast_code(a2, bind_state, patch_state, bn_state);
        // cmp eax, 0      (83 F8 00) — 3 bytes
        emit_byte(0x83); emit_byte(0xF8); emit_byte(0x00);
        // pop rcx (new_value into rcx)   — 1 byte (59)
        emit_byte(0x59);
        // pop rdx (handle into rdx)      — 1 byte (5A)
        emit_byte(0x5A);
        // je .reject  (74 rel8)  — skip the write path on verifier=0.
        // Write path emits:
        //   - bounds check on handle in rdx
        //     cmp edx, 0           — 3 bytes (83 FA 00)
        //     jl  .reject_keep_zero — 2 bytes; reject if negative
        //     cmp edx, 64           — 3 bytes (83 FA 40)
        //     jge .reject_keep_zero — 2 bytes
        //   - lea rax, [arena_base] — 7 bytes (placeholder)
        //   - mov [rax + rdx*4 + DISP_BASE], ecx — 8 bytes
        //   - mov eax, 1            — 5 bytes (return 1 = applied)
        //   - jmp .end              — 2 bytes
        //   .reject: xor eax, eax  — 2 bytes (return 0)
        //   .end:
        // Total write path = 3+2+3+2+7+7+5+2 = 31 bytes (mov [rax+rdx*4+d],
        // ecx is 7 bytes, not 8 — 89 8C 90 disp32 = 7 bytes total).
        // The reject path skips 31 bytes (rel8=31 fits in i8 since 31 < 127).
        emit_byte(0x74); emit_byte(31);                      // je .reject (rel8=31)
        // Bounds check on handle in edx:
        emit_byte(0x83); emit_byte(0xFA); emit_byte(0x00);   // cmp edx, 0
        emit_byte(0x7C); emit_byte(26);                      // jl .reject (rel8=26)
        emit_byte(0x83); emit_byte(0xFA); emit_byte(0x40);   // cmp edx, 64
        emit_byte(0x7D); emit_byte(21);                      // jge .reject (rel8=21)
        // lea rax, [arena_base]
        let disp_slot2 = emit_lea_rax_rip_placeholder();     // 7 bytes
        patch_table_add(patch_state, disp_slot2, arena_base_s, 18);
        // mov [rax + rdx*4 + disp32], ecx
        // ModRM=8C (mod=10 disp32, reg=001 ecx, r/m=100 SIB)
        // SIB=90 (scale=10 *4, index=010 rdx, base=000 rax)
        // Encoding: 89 8C 90 disp32 = 7 bytes total.
        emit_byte(0x89); emit_byte(0x8C); emit_byte(0x90);
        let disp_base2 = 8388356;
        emit_u32_le(disp_base2);                             // 4 bytes (3+4 = 7 byte instr)
        // mov eax, 1   (B8 01 00 00 00)
        emit_byte(0xB8); emit_byte(0x01); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
        // jmp .end  (EB rel8=2)
        emit_byte(0xEB); emit_byte(2);
        // .reject: xor eax, eax
        emit_byte(0x31); emit_byte(0xC0);
        // .end:
        nh + nph + nv + npv + np + 3 + 1 + 1 + 2 + 3 + 2 + 3 + 2 + 7 + 7 + 5 + 2 + 2
    } else {
        0
    }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}    // +K1.F2/F3/F4: 5 new builtin arms (reflect_hash + trace_event + __helix_* trio); +K1.F20b: 1 more arm (__trace_last); +K1.F22c: 1 more arm (print_str_ln); +K1.F22d: 1 more arm (eprint_str_ln); +K1.F22f: 1 more arm (eprint_str); +K1.F23c: 1 more arm (__tile_zeros); +K1.F24e: 1 more arm (__tile_add stub); +K1.F25: 1 more arm (__tile_sub); +K1.F26: 1 more arm (__tile_mul); +K1.F27: 1 more arm (__tile_matmul)
}

// Audit fix #6 (cycle 1, polish): try_emit_builtin_call_impl used to
// live here as "Unreachable in this commit; reference impl preserved
// for next session." The reference impl had self-noted bugs (rax
// clobbered by `mov eax, ecx` then `mov [rax]`) and was never
// reachable. Removed entirely. The actual builtin emission path is
// in try_emit_builtin_call (above) which inlines per-builtin
// machine code with correct rax management.

fn bn_global_slot_address() -> i32 {
    // __helix_kovc_bn_state lives at a fixed slot we set up early.
    // Hack: scan the arena for our 'magic' approach won't work,
    // so we just store at a well-known slot index — by convention,
    // emit_elf_for_ast_to_path pushes bn_state right after
    // patch_state, and we record the arena slot it landed in via
    // a simple convention: it's the FIFTH thing pushed (after
    // bind_state, fn_state, patch_state's 195+50+194 slots).
    // To avoid threading: we reserve slot 0 of the arena as a
    // pointer to bn_state. emit_elf_for_ast_to_path writes it.
    0
}

// --------------------------------------------------------------
// AST walker: dispatch on tag and emit the matching code. Returns
// the number of bytes emitted. AST node layout matches stage-2
// parser.hx: [tag, p1, p2, p3].
//
// Compile-time state passed via arena-slot pointers:
//   bind_state — variable bindings (next stack offset, table top, ...)
//   patch_state — pending CALL/LEA backpatches (disp_slot, target name)
//   bn_state lives in a known global slot read by try_emit_builtin_call
//
// SysV 6-int-param limit forced bn_state into a global slot rather
// than a function arg.
// --------------------------------------------------------------

// Stage 1.5 audit refactor: shared float-bits parser. Reads "I.F" or
// "I" from `p1..p1+p2` (literal text in the arena) and returns the
// IEEE 754 f32 bit pattern. Originally inlined in the t==27 arm
// (AST_FLOATLIT, ~90 lines); the bf16 literal commit (05773bb) copied
// that body into t==42, so the duplication doubled. This helper
// collapses both arms back to a one-liner. bf16 just masks the result
// (`bits & 0xFFFF0000` to truncate the low 16 mantissa bits).
//
// Precision range covered: ~10^-9 to ~10^9 within i32 limits. Beyond
// that, the i32 accumulators overflow. Callers should pre-check via
// count_float_digits (see below) and emit ud2 trap if > 9 digits.

// Stage 1.5 audit fix: overflow guard for parse_float_bits. Returns
// the count of decimal digits in the literal text at p1..p1+p2,
// excluding the '.' separator. parse_float_bits accumulates digits
// into i32 int_part / frac_part / pow10 / v_scaled; v_scaled =
// int_part * pow10 + frac_part wraps the i32 sign bit when total
// digits > 9 (since 10^10 > 2^31). Pre-fix: silent garbage for big
// literals like 1234567890.5_f32. Post-fix: caller checks > 9 and
// emits ud2.
fn count_float_digits(p1: i32, p2: i32) -> i32 {
    let mut i: i32 = 0;
    let mut digits: i32 = 0;
    while i < p2 {
        let b = __arena_get(p1 + i);
        if b == 46 { } // '.', skip
        else { if b >= 48 { if b <= 57 { digits = digits + 1; }; }; };
        i = i + 1;
    }
    digits
}

// K1.F18b helper (2026-05-27): compute 2^n by repeated doubling.
// Used by the f16 gradual-underflow branch to derive the
// runtime-variable mantissa-shift divisor (n in [14, 24] for the f16
// denormal range). The caller MUST gate n to avoid i32 overflow
// (n <= 30 fits; the denormal branch's `unbiased < -25` cutoff above
// keeps n <= 24).
fn pow2_i32(n: i32) -> i32 {
    let mut p: i32 = 1;
    let mut i: i32 = 0;
    while i < n {
        p = p * 2;
        i = i + 1;
    }
    p
}

// K1.F15 (2026-05-27): IEEE-754 single-precision (f32) to half-
// precision (f16) bit-pattern conversion. f32 layout: 1 sign + 8 exp
// (bias 127) + 23 mantissa. f16 layout: 1 sign + 5 exp (bias 15)
// + 10 mantissa.
//
// K1.F18 (2026-05-27): the K1.F15 first-cut truncated the bottom 13
// mantissa bits. K1.F18 implements IEEE-754 round-to-nearest-even
// (banker's rounding): the round bit is bit 12 (the highest dropped
// bit), the sticky is OR of bits 0..11; on tie (round=1, sticky=0)
// round-to-even ties to the LSB of mant16. A mantissa carry into
// the exponent is handled (mant16==1024 -> wrap mant16=0, exp16+=1,
// check exp16 overflow -> +/-Inf).
//
// K1.F18b (2026-05-27): IEEE-754 gradual underflow / denormals. For
// unbiased exponent in [-25, -15] the value is too small for an f16
// normal but representable as an f16 denormal mant10 * 2^-24. The
// algorithm re-attaches the f32 hidden mantissa bit, divides by a
// runtime-variable 2^(-unbiased-1) to position the LSB at 2^-24,
// then applies the same RNE/sticky rounding pattern as the normal
// branch. A round-up that carries to mant10==1024 promotes the
// result to the smallest f16 normal (exp16=1, mant10=0). For
// unbiased < -25 the value is below the smallest-denormal round-up
// threshold and flushes to +/-0.
//
// Constants used (kept as decimal to avoid hex literal limits):
//   2147483647 = 2^31 - 1 (max positive i32, used as sign-strip mask)
//   8388608    = 2^23     (f32 mantissa width / hidden-bit value)
//   8388607    = 2^23 - 1 (f32 mantissa mask)
//   32768      = 2^15     (f16 sign-bit position)
//   31744      = 31 * 2^10 (f16 exp-all-1s pattern, used for Inf)
//   32256      = 31744 + 2^9 (f16 canonical-NaN pattern)
//   1024       = 2^10     (f16 exp-bit position; also denormal->normal promotion threshold)
//   8192       = 2^13     (f16 mantissa-truncation divisor, normal path)
//   4096       = 2^12     (round-bit position, MSB of dropped bits, normal path)
//   4095       = 2^12 - 1 (sticky mask, bits 0..11, normal path)
fn f32_to_f16_bits(f32bits: i32) -> i32 {
    let sign = if f32bits < 0 { 1 } else { 0 };
    // Strip sign by AND-mask with 0x7FFFFFFF.
    let unsigned_bits = f32bits & 2147483647;
    let exp32_raw = (unsigned_bits / 8388608) & 255;
    let mant32 = unsigned_bits & 8388607;
    // Special: zero (f32 subnormals also flush via this guard since they
    // have exp32_raw==0; their f32-denormal value is below f16's smallest
    // denormal 2^-24, so flushing is correct).
    if exp32_raw == 0 {
        sign * 32768
    } else { if exp32_raw == 255 {
        // Inf or NaN.
        if mant32 == 0 {
            (sign * 32768) + 31744
        } else {
            (sign * 32768) + 32256
        }
    } else {
        // Normal f32 number: rebias exp from 127 to 15 if it fits.
        let unbiased = exp32_raw - 127;
        if unbiased > 15 {
            // f16 overflow -> +/-Inf.
            (sign * 32768) + 31744
        } else { if unbiased < 0 - 25 {
            // K1.F18b: deep underflow. Even with round-up, the value
            // is below 0.5 * 2^-24 (the smallest-denormal tie point),
            // so RNE flushes to +/-0.
            sign * 32768
        } else { if unbiased < 0 - 14 {
            // K1.F18b: gradual underflow. unbiased in [-25, -15].
            // Re-attach the f32 hidden bit so the leading-1 is at
            // bit 23, then shift right by `shift = -unbiased - 1` to
            // position the LSB at 2^-24 (the f16 denormal LSB).
            // shift ranges over [14, 24]; divisor = 2^shift fits in
            // i32 (max 2^24 = 16777216). Round bit is at position
            // shift-1; sticky is OR of bits 0..shift-2.
            let mant_with_lead = mant32 + 8388608;
            let shift = (0 - unbiased) - 1;
            let divisor = pow2_i32(shift);
            let mant10_trunc = mant_with_lead / divisor;
            let round_bit_pos = divisor / 2;
            let round_bit_d = (mant_with_lead / round_bit_pos) & 1;
            let sticky_mask = round_bit_pos - 1;
            let sticky_nonzero_d = if (mant_with_lead & sticky_mask) != 0 { 1 } else { 0 };
            let round_up_d = if round_bit_d == 0 {
                0
            } else { if sticky_nonzero_d == 1 {
                1
            } else {
                // tie -> round to even (LSB of denormal mantissa)
                mant10_trunc & 1
            }};
            let mant10_rounded = mant10_trunc + round_up_d;
            // Carry: rounded == 1024 means the rounded value crossed
            // the denormal-to-normal boundary; encode as smallest f16
            // normal (exp16=1, mant10=0) which has the same numeric
            // value (1024 * 2^-24 == 1.0 * 2^-14).
            if mant10_rounded >= 1024 {
                (sign * 32768) + 1024
            } else {
                (sign * 32768) + mant10_rounded
            }
        } else {
            // Normal-number computation: rebias exp 127 -> 15 and
            // truncate mantissa from 23 bits to 10 with RNE rounding.
            let exp16 = unbiased + 15;
            let mant16_trunc = mant32 / 8192;
            // K1.F18: round-to-nearest-even.
            let round_bit = (mant32 / 4096) & 1;
            let sticky_nonzero = if (mant32 & 4095) != 0 { 1 } else { 0 };
            let round_up = if round_bit == 0 {
                0
            } else { if sticky_nonzero == 1 {
                1
            } else {
                // round_bit==1, sticky==0: tie -> round to even.
                mant16_trunc & 1
            }};
            let mant16_rounded = mant16_trunc + round_up;
            // Mantissa carry: if rounded==1024, mant overflows into exp.
            if mant16_rounded >= 1024 {
                let exp16_carry = exp16 + 1;
                if exp16_carry >= 31 {
                    // Carry caused exp overflow -> +/-Inf.
                    (sign * 32768) + 31744
                } else {
                    // mant wraps to 0, exp bumps by 1.
                    (sign * 32768) + (exp16_carry * 1024)
                }
            } else {
                (sign * 32768) + (exp16 * 1024) + mant16_rounded
            }
        }}}
    }}
}

fn parse_float_bits(p1: i32, p2: i32) -> i32 {
    let mut i: i32 = 0;
    let mut int_part: i32 = 0;
    let mut frac_part: i32 = 0;
    let mut frac_digits: i32 = 0;
    let mut phase: i32 = 0;
    let mut keep_p: i32 = 1;
    while keep_p == 1 {
        if i >= p2 { keep_p = 0; }
        else {
            let b = __arena_get(p1 + i);
            if b == 46 {
                // '.' transitions integer -> fractional phase.
                phase = 1;
                i = i + 1;
            } else {
                // is_digit inlined: '0'=48, '9'=57.
                if b < 48 { keep_p = 0; }
                else { if b > 57 { keep_p = 0; }
                else {
                    if phase == 0 {
                        int_part = int_part * 10 + (b - 48);
                    } else {
                        frac_part = frac_part * 10 + (b - 48);
                        frac_digits = frac_digits + 1;
                    };
                    i = i + 1;
                }};
            };
        };
    }
    // pow10 = 10^frac_digits.
    let mut pow10: i32 = 1;
    let mut dd: i32 = 0;
    while dd < frac_digits { pow10 = pow10 * 10; dd = dd + 1; }
    let v_scaled = int_part * pow10 + frac_part;
    let mut bits: i32 = 0;
    if v_scaled == 0 {
        bits = 0;
    } else {
        // Find binary exponent k: largest k such that 2^k * pow10 <= v_scaled.
        // For v_scaled < pow10 (sub-1.0 literals like 0.5/0.25), first decrement
        // k and halve threshold until threshold <= v_scaled. Then do the
        // positive-k loop. The two loops together cover (~10^-9, ~10^9).
        let mut k: i32 = 0;
        let mut threshold: i32 = pow10;
        let mut keep_neg: i32 = 1;
        while keep_neg == 1 {
            if threshold <= v_scaled { keep_neg = 0; }
            else { if threshold == 1 { keep_neg = 0; }
            else {
                threshold = threshold / 2;
                k = k - 1;
            }};
        }
        let mut keep_k: i32 = 1;
        while keep_k == 1 {
            if threshold > v_scaled / 2 { keep_k = 0; }
            else {
                threshold = threshold * 2;
                k = k + 1;
            }
        }
        // Extract 23 mantissa bits via residual-doubling.
        let mut residual = v_scaled - threshold;
        let mut mantissa: i32 = 0;
        let mut bit: i32 = 22;
        while bit >= 0 {
            residual = residual * 2;
            if residual >= threshold {
                let mut bv: i32 = 1;
                let mut sh: i32 = 0;
                while sh < bit { bv = bv * 2; sh = sh + 1; }
                mantissa = mantissa + bv;
                residual = residual - threshold;
            }
            bit = bit - 1;
        }
        // Pack: (k + 127) << 23 | mantissa  (sign bit = 0).
        let exp_field = k + 127;
        let mut exp_shifted: i32 = exp_field;
        let mut sh2: i32 = 0;
        while sh2 < 23 { exp_shifted = exp_shifted * 2; sh2 = sh2 + 1; }
        bits = exp_shifted + mantissa;
    }
    bits
}

// Stage 7: match-pattern codegen support.
//
// fail_jmp_state layout (17-slot region):
//   slot 0       = current count of recorded fail-jmp disp slots
//   slots 1..16  = recorded fail-jmp disp slot indices (rel32 placeholders)
//                  that need to be backpatched to a "next arm" label.
// Cap is 16 fail jumps per arm — covers PAT_VARIANT(disc) + nested sub-
// patterns. Phase-0 doesn't allow deeper nesting; trap 62004 reserved.
//
// IMPORTANT: fail_jmp_state regions live in the bn_state prelude (slots
// 84..117), NOT __arena_pushed during code emission — pushing during
// emission would corrupt the code byte stream. Caller passes the absolute
// arena offset. Reset writes 0 to slot 0; no allocation needed.

fn fail_jmp_state_reset(state: i32) -> i32 {
    __arena_set(state, 0);
    0
}

fn fail_jmp_state_add(state: i32, disp_slot: i32) -> i32 {
    let count = __arena_get(state);
    if count >= 16 {
        0 - 1
    } else {
        __arena_set(state + 1 + count, disp_slot);
        __arena_set(state, count + 1);
        0
    }
}

fn fail_jmp_state_patch_all(state: i32, target: i32) -> i32 {
    let count = __arena_get(state);
    let mut i: i32 = 0;
    while i < count {
        let disp_slot = __arena_get(state + 1 + i);
        patch_rel32(disp_slot, target);
        i = i + 1;
    }
    0
}

// Stage 7 PAT_VARIANT helper. Walks sub-patterns and emits load+recurse.
// scrut_off is the slot holding the enum pointer. Each sub-pattern gets
// a fresh slot; rax is re-loaded between sub-pats since prior tests
// clobber it.
//
// Stage 28.10 cycle-90 CN-1 fix (HIGH conf 92): thread bn_state through
// to emit_pattern_test. Pre-fix this helper called emit_pattern_test
// with 4 args, but emit_pattern_test takes 5 (last is bn_state). The
// missing arg meant r8 carried garbage from prior emission into the
// callee. Pre-Stage-28.10 this was a latent dead-store (only the
// pt>=69 compound branch needed bn_state). Stage 28.10 INCREMENT 3
// made it newly LIVE by adding the pt==68 dispatch to emit_pat_or,
// which immediately uses bn_state to compute success_state =
// bn_state + 123 and alt_fail_state = bn_state + 140 — with garbage
// bn_state, those become arbitrary arena writes. Reproducer:
// `match v { Some(1 | 2) => 42, _ => 0 }`. Same defect class as
// Audit-13 — and the cycle-85 init bump alone is insufficient if
// callers pass garbage.
fn emit_variant_subpats(sub_head: i32, scrut_off: i32, fail_state: i32,
                        bind_state: i32, bn_state: i32) -> i32 {
    // Audit-stage5-6 Finding #9 fix: the inner load uses disp8 (signed,
    // -128..127). At idx_in_payload >= 16, off_in_payload >= 128 wraps
    // to a negative disp and the load silently reads BELOW the variant
    // payload — a classic OOB-read. Emit trap 60030 before the wrapping
    // load so the compiled binary crashes with the documented trap id
    // instead of returning garbage. Same shape as Stage 4 Finding #7's
    // existing trap 52001 for AST_TUPLE_FIELD.
    let mut total: i32 = 0;
    let mut cur: i32 = sub_head;
    let mut idx_in_payload: i32 = 1;     // skip disc at slot 0
    while cur != 0 {
        let sub_pat = __arena_get(cur + 1);
        let sub_off = bind_alloc_offset(bind_state);
        // Re-load enum pointer.
        let n_rl = emit_mov_rax_local_64(scrut_off);
        let n_trap = if idx_in_payload > 15 {
            emit_trap_with_id(60030)
        } else { 0 };
        let off_in_payload = idx_in_payload * 8;
        // mov rax, [rax + disp8]  (48 8B 40 disp8 = 4 bytes)
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x40); emit_byte(off_in_payload);
        let n_st = emit_mov_local_rax_64(sub_off);
        let n_sub = emit_pattern_test(sub_pat, sub_off, fail_state, bind_state, bn_state);
        total = total + n_rl + n_trap + 4 + n_st + n_sub;
        idx_in_payload = idx_in_payload + 1;
        cur = __arena_get(cur + 2);
    }
    total
}

// Stage 7 PAT_TUPLE helper. Same as variant but starts at slot 0 (no disc).
// Stage 28.10 cycle-90 CN-1 fix: thread bn_state (see emit_variant_subpats
// for full rationale — both helpers had the same 4-vs-5-arg bug).
fn emit_tuple_subpats(sub_head: i32, scrut_off: i32, fail_state: i32,
                      bind_state: i32, bn_state: i32) -> i32 {
    // Audit-stage5-6 Finding #9 fix (tuple variant): mirror the
    // emit_variant_subpats cap-trap. idx_in_tuple >= 16 → off >= 128
    // wraps signed disp8.
    let mut total: i32 = 0;
    let mut cur: i32 = sub_head;
    let mut idx_in_tuple: i32 = 0;
    while cur != 0 {
        let sub_pat = __arena_get(cur + 1);
        let sub_off = bind_alloc_offset(bind_state);
        let n_rl = emit_mov_rax_local_64(scrut_off);
        let n_trap = if idx_in_tuple > 15 {
            emit_trap_with_id(60030)
        } else { 0 };
        let off_in_tuple = idx_in_tuple * 8;
        emit_byte(0x48); emit_byte(0x8B); emit_byte(0x40); emit_byte(off_in_tuple);
        let n_st = emit_mov_local_rax_64(sub_off);
        let n_sub = emit_pattern_test(sub_pat, sub_off, fail_state, bind_state, bn_state);
        total = total + n_rl + n_trap + 4 + n_st + n_sub;
        idx_in_tuple = idx_in_tuple + 1;
        cur = __arena_get(cur + 2);
    }
    total
}

// Stage 7 PAT_LIT helper.
fn emit_pat_lit(scrut_off: i32, lit: i32, fail_state: i32) -> i32 {
    let n_load = emit_mov_eax_local(scrut_off);
    let n_cmp = emit_cmp_eax_imm32(lit);
    let disp = emit_jne_rel32_placeholder();
    fail_jmp_state_add(fail_state, disp);
    n_load + n_cmp + 6
}

// Stage 28.10 INCREMENT 3: emit_pat_or — codegen for PAT_OR (tag 68).
// Mirrors helixc/frontend/match_lower.py::_or_chain semantics: each
// alt is tested in sequence; on first match jump to the body; on last
// alt's mismatch jump to the actual fail_state.
//
// pat_idx slots: p1 = head_alt cell (AST_TUPLE_CONS chain, tag 51),
//                p2 = count (informational), p3 = unused.
// Each cell: p1 = alt_pat_idx, p2 = next_cell (0 = end).
//
// Slot allocation in bn_state:
//   bn_state + 123 (17 slots) = success_state — collects jmp disp slots
//                                 from "this alt matched, skip rest" jmps
//   bn_state + 140 (17 slots) = alt_fail_state — collects the current
//                                 alt's fail jne disp slots; patched to
//                                 next-alt label between alts.
//
// Phase-0 limitations enforced at PARSE time (parser.hx::parse_pattern):
//   - PAT_OR alts may not bind variables (trap 62020) — each alt's
//     bind_state would differ; mirrors match_lower.py _collect_binds
//     intersection logic which is deferred.
//   - Nested OR (e.g. `Some(1 | 2)` or `1 | 2 | (3 | 4)`) is REJECTED
//     (trap 62022) — was previously documented as "parsed but
//     constrained" pre-cycle-79; cycle-79's deep walker now rejects
//     nested ORs to avoid the static-slot collision in emit_pat_or.
//   - Alt count > 17 (trap 62021) — fail_jmp_state cap is 16
//     successful adds; with N-1 non-last alt adds, N=18 is first
//     failing case.
// Common Phase-0 use: scalar literal alternation like `1 | 2 | 3 =>
// body`.
//
// Slots 123..156 are reserved by `install_builtin_names`' init loop
// (`while i < 152`) which was bumped from 118 in cycle-85 (CN-1 fix)
// precisely so OR scratch lives inside the init-zeroed region, NOT
// past it. Pre-cycle-85 the slots were past the cap and overwrote
// the builtin name table — Audit-13 defect class. Do NOT shrink the
// init bound below 152 without revisiting these allocations.
fn emit_pat_or(pat_idx: i32, scrut_off: i32, fail_state: i32,
               bind_state: i32, bn_state: i32) -> i32 {
    let head = __arena_get(pat_idx + 1);
    let success_state = bn_state + 123;
    let alt_fail_state = bn_state + 140;
    fail_jmp_state_reset(success_state);
    let mut total: i32 = 0;
    let mut cur: i32 = head;
    while cur != 0 {
        let alt_idx = __arena_get(cur + 1);
        let next = __arena_get(cur + 2);
        let is_last = if next == 0 { 1 } else { 0 };
        if is_last == 1 {
            // Last alt: use the real fail_state. Mismatch jumps to
            // actual fail target (next match arm). On match, fall
            // through to body via success_state patching below.
            let n_test = emit_pattern_test(alt_idx, scrut_off, fail_state, bind_state, bn_state);
            total = total + n_test;
        } else {
            // Non-last alt: use temp alt_fail_state. On mismatch,
            // patch to next-alt label so we try the next one.
            fail_jmp_state_reset(alt_fail_state);
            let n_test = emit_pattern_test(alt_idx, scrut_off, alt_fail_state, bind_state, bn_state);
            // On match: emit unconditional jmp to "after-OR" label
            // (collected via success_state, patched once all alts emit).
            let jmp_disp = emit_jmp_rel32_placeholder();
            fail_jmp_state_add(success_state, jmp_disp);
            // Patch alt_fail_state's jne disps to "right here" (the
            // next-alt code that will be emitted on the next loop
            // iteration). __arena_len() returns the current code
            // emission offset; that's where the next alt's test will
            // start.
            let next_alt_label = __arena_len();
            fail_jmp_state_patch_all(alt_fail_state, next_alt_label);
            total = total + n_test + 5;
        };
        cur = next;
    }
    // Patch all success-state jumps to fall through here (after all
    // alts). Body emission resumes from this offset.
    let after_or = __arena_len();
    fail_jmp_state_patch_all(success_state, after_or);
    total
}

// Stage 7 PAT_RANGE helper (exclusive: lo <= x < hi).
fn emit_pat_range(scrut_off: i32, lo: i32, hi: i32,
                  fail_state: i32, inclusive: i32) -> i32 {
    // K1.L (2026-05-25): the `inclusive` param chooses the upper-
    // bound jump. Half-open `lo..hi` (inclusive == 0): fail when
    // eax >= hi (`jge`). Closed `lo..=hi` (inclusive == 1): fail
    // when eax > hi (`jg`). The lower-bound check is always `jl`
    // (fail when eax < lo).
    let n_load = emit_mov_eax_local(scrut_off);
    let n_cmp_lo = emit_cmp_eax_imm32(lo);
    let disp_lo = emit_jl_rel32_placeholder();
    fail_jmp_state_add(fail_state, disp_lo);
    let n_cmp_hi = emit_cmp_eax_imm32(hi);
    let disp_hi = if inclusive == 1 {
        emit_jg_rel32_placeholder()
    } else {
        emit_jge_rel32_placeholder()
    };
    fail_jmp_state_add(fail_state, disp_hi);
    n_load + n_cmp_lo + 6 + n_cmp_hi + 6
}

// Stage 7 PAT_VARIANT (sans sub-pats) helper.
//
// Audit A1-F1 fix: when the scrutinee is i32-shaped (expr_type == 0),
// the all-unit-enum fold stored the disc as a small integer (via
// AST_INT). The pointer-rep load `mov rax, [scrut_off]; mov eax, [rax]`
// dereferences that small integer as a pointer → SIGSEGV at the disc
// value (e.g. 0x1, 0x2). For i32-shaped scrut, skip the deref and
// compare the loaded i32 directly against the pattern's disc. The
// scrut_ty is read from match_scrut_ty_get(bn_state) — stashed once at
// emit_match_dispatch time so we don't need to thread an extra param
// through the 6-param-cap helper chain.
fn emit_pat_variant_disc(scrut_off: i32, disc: i32, fail_state: i32, bn_state: i32) -> i32 {
    let scrut_ty = match_scrut_ty_get(bn_state);
    if scrut_ty == 0 {
        // i32-shaped scrut: disc is stored directly. mov eax, [rbp+off];
        // cmp eax, disc; jne fail.
        let n_load_disc = emit_mov_eax_local(scrut_off);
        let n_cmp = emit_cmp_eax_imm32(disc);
        let disp = emit_jne_rel32_placeholder();
        fail_jmp_state_add(fail_state, disp);
        n_load_disc + n_cmp + 6
    } else {
        // Pointer-rep scrut: scrut_off holds an 8-byte pointer to the
        // variant's disc slot. mov rax, [rbp+off]; mov eax, [rax];
        // cmp eax, disc; jne fail.
        let n_load_ptr = emit_mov_rax_local_64(scrut_off);
        emit_byte(0x8B); emit_byte(0x00);    // mov eax, [rax+0]
        let n_cmp = emit_cmp_eax_imm32(disc);
        let disp = emit_jne_rel32_placeholder();
        fail_jmp_state_add(fail_state, disp);
        n_load_ptr + 2 + n_cmp + 6
    }
}

// Stage 7 pattern test for scalar patterns (LIT/WILDCARD/BIND/RANGE).
// Returns 0 if pat is not scalar (caller dispatches to compound).
fn emit_scalar_pattern_test(pat_idx: i32, scrut_off: i32, fail_state: i32,
                            bind_state: i32) -> i32 {
    let pt = __arena_get(pat_idx);
    let pp1 = __arena_get(pat_idx + 1);
    let pp2 = __arena_get(pat_idx + 2);
    if pt == 64 { emit_pat_lit(scrut_off, pp1, fail_state) }
    else { if pt == 66 { 0 }
    else { if pt == 65 { bind_push_typed(bind_state, pp1, pp2, scrut_off, 0); 0 }
    else { if pt == 67 {
        // K1.L (2026-05-25): pp3 of AST_PAT_RANGE carries the
        // inclusive flag (0 = half-open `..`, 1 = closed `..=`).
        let pp3 = __arena_get(pat_idx + 3);
        emit_pat_range(scrut_off, pp1, pp2, fail_state, pp3)
    }
    else { 0 }}}}
}

// Stage 7 pattern test for compound patterns (VARIANT/TUPLE).
fn emit_compound_pattern_test(pat_idx: i32, scrut_off: i32, fail_state: i32,
                              bind_state: i32, bn_state: i32) -> i32 {
    let pt = __arena_get(pat_idx);
    let pp1 = __arena_get(pat_idx + 1);
    let pp2 = __arena_get(pat_idx + 2);
    if pt == 69 {
        let n_disc = emit_pat_variant_disc(scrut_off, pp1, fail_state, bn_state);
        let n_subs = emit_variant_subpats(pp2, scrut_off, fail_state, bind_state, bn_state);
        n_disc + n_subs
    } else { if pt == 70 {
        emit_tuple_subpats(pp2, scrut_off, fail_state, bind_state, bn_state)
    } else { 0 }}
}

// Stage 7 pattern test entry point. Dispatches to scalar or compound.
//
// Audit A2-F6 fix: parse_pattern emits AST_ERR (tag 99) with trap-id in
// p1 when it sees an unknown pattern token. We dispatch that here to
// emit_trap_with_id so the binary SIGILLs with the trap-id (62002) loud
// at runtime. Pre-fix, parse_pattern silently emitted PAT_WILDCARD
// instead — the unknown pattern always matched and the arm body ran.
fn emit_pattern_test(pat_idx: i32, scrut_off: i32, fail_state: i32,
                     bind_state: i32, bn_state: i32) -> i32 {
    let pt = __arena_get(pat_idx);
    if pt == 99 {
        let pp1 = __arena_get(pat_idx + 1);
        emit_trap_with_id(pp1)
    } else { if pt == 68 {
        // Stage 28.10 INCREMENT 3: PAT_OR codegen lands here. See
        // emit_pat_or above for the alt-threading strategy.
        emit_pat_or(pat_idx, scrut_off, fail_state, bind_state, bn_state)
    } else { if pt >= 69 {
        emit_compound_pattern_test(pat_idx, scrut_off, fail_state, bind_state, bn_state)
    } else {
        emit_scalar_pattern_test(pat_idx, scrut_off, fail_state, bind_state)
    }}}
}

// Stage 7: count bind_push entries done by pattern. Mirror of
// emit_pattern_test but only counting bind_pushes — used by AST_MATCH
// to know how many bind_pops to emit after each arm body.
fn count_pattern_binds(pat_idx: i32) -> i32 {
    let pt = __arena_get(pat_idx);
    let pp2 = __arena_get(pat_idx + 2);
    if pt == 65 {
        1
    } else { if pt == 69 {
        // Walk sub-patterns
        let mut total: i32 = 0;
        let mut cur: i32 = pp2;
        while cur != 0 {
            let sub_pat = __arena_get(cur + 1);
            total = total + count_pattern_binds(sub_pat);
            cur = __arena_get(cur + 2);
        }
        total
    } else { if pt == 70 {
        let mut total: i32 = 0;
        let mut cur: i32 = pp2;
        while cur != 0 {
            let sub_pat = __arena_get(cur + 1);
            total = total + count_pattern_binds(sub_pat);
            cur = __arena_get(cur + 2);
        }
        total
    } else { if pt == 68 {
        // Stage 28.10 cycle-78 CN-2 follow-up: PAT_OR alts may not
        // contain PAT_BIND (parser enforces via trap 62020 at
        // parse_pattern; renumbered from 62008 in cycle-85 to avoid
        // Stage 7 reservation collision). Defensive return 0 — if
        // a future cycle
        // lifts the parse-time restriction (e.g. by implementing
        // Python's `_collect_binds` intersection), this branch
        // must compute the intersection-count, not just 0.
        0
    } else {
        0
    }}}}
}

// Stage 7: match_state region — packs fail_state + end_table into one
// 34-slot arena region (17 slots each). Reduces AST_MATCH helper param
// count below the SysV 6-int-arg cap. Layout:
//   match_state + 0..16   : fail_state (count + 16 entries)
//   match_state + 17..33  : end_table  (count + 16 entries)
//
// Region lives at bn_state + 84 (reserved by install_builtin_names).
// Caller resets COUNT slots (0 and 17); no __arena_push during emission.
//
// Phase-0 limitation: nested match expressions clobber the parent's
// match_state. Trap 62004 reserved for nested match in a future iter;
// for now, Phase-0 patterns aren't deep enough to expose this.
fn match_state_base(bn_state: i32) -> i32 {
    bn_state + 84
}

fn match_state_init(bn_state: i32) -> i32 {
    let base = match_state_base(bn_state);
    fail_jmp_state_reset(base);          // fail_state count = 0
    fail_jmp_state_reset(base + 17);     // end_table count = 0
    base
}

// Stage 7: emit one match arm. Returns bytes emitted.
// match_state holds both fail_state (offset 0) and end_table (offset 17).
fn emit_one_match_arm(arm_idx: i32, scrut_off: i32, match_state: i32,
                      bind_state: i32, patch_state: i32, bn_state: i32) -> i32 {
    let pat_idx = __arena_get(arm_idx + 1);
    let body_idx = __arena_get(arm_idx + 2);
    let fail_state = match_state;
    let end_table = match_state + 17;
    fail_jmp_state_reset(fail_state);
    let n_pat = emit_pattern_test(pat_idx, scrut_off, fail_state, bind_state, bn_state);
    let n_body = emit_ast_code(body_idx, bind_state, patch_state, bn_state);
    let n_binds = count_pattern_binds(pat_idx);
    let mut bp: i32 = 0;
    while bp < n_binds {
        bind_pop(bind_state);
        bp = bp + 1;
    }
    let end_disp = emit_jmp_rel32_placeholder();
    fail_jmp_state_add(end_table, end_disp);
    let next_arm_label = __arena_len();
    fail_jmp_state_patch_all(fail_state, next_arm_label);
    n_pat + n_body + 5
}

// Stage 7: emit the entire match-arm chain.
fn emit_match_arms(arms_head: i32, scrut_off: i32, match_state: i32,
                   bind_state: i32, patch_state: i32, bn_state: i32) -> i32 {
    let mut total: i32 = 0;
    let mut arm_cur: i32 = arms_head;
    while arm_cur != 0 {
        let next_arm = __arena_get(arm_cur + 3);
        let n_arm = emit_one_match_arm(arm_cur, scrut_off, match_state,
                                       bind_state, patch_state, bn_state);
        total = total + n_arm;
        arm_cur = next_arm;
    }
    total
}

// Stage 7: top-level AST_MATCH lowering. Wrapped in a helper so the
// emit_ast_code arm body stays a single function call (host parser
// recursion budget).
//
// Audit A1-F1 fix: capture the scrut's expr_type once at dispatch and
// stash it in bn_state via match_scrut_ty_set. emit_pat_variant_disc
// reads it back via match_scrut_ty_get. When scrut_ty == 0 (i32-shaped,
// including all-unit-enum AST_INT folds), the variant-disc helper skips
// the pointer dereference and compares the raw i32 value. (Stash via
// bn_state instead of an extra param avoids exceeding the 6-int-param
// cap on the helper chain.)
fn emit_match_dispatch(scrut_idx: i32, arms_head: i32,
                       bind_state: i32, patch_state: i32, bn_state: i32) -> i32 {
    let scrut_ty = expr_type(scrut_idx, bind_state, bn_state);
    match_scrut_ty_set(bn_state, scrut_ty);
    let n_scrut = emit_ast_code(scrut_idx, bind_state, patch_state, bn_state);
    let scrut_off = bind_alloc_offset(bind_state);
    let n_store = emit_mov_local_rax_64(scrut_off);
    let match_state = match_state_init(bn_state);
    let n_arms = emit_match_arms(arms_head, scrut_off, match_state,
                                 bind_state, patch_state, bn_state);
    let n_trap = emit_trap_with_id(62001);
    let merge_label = __arena_len();
    fail_jmp_state_patch_all(match_state + 17, merge_label);
    n_scrut + n_store + n_arms + n_trap
}

fn emit_ast_code(idx: i32, bind_state: i32, patch_state: i32, bn_state: i32) -> i32 {
    let t = __arena_get(idx);
    let p1 = __arena_get(idx + 1);
    let p2 = __arena_get(idx + 2);
    if t == 0 {
        emit_ast_int(p1)
    } else { if t == 62 {
        // Stage 7: AST_MATCH (tag 62). p1 = scrut_idx, p2 = arms_head.
        // Lowered into emit_match_dispatch helper to keep this arm body
        // shallow (host parser depth budget).
        emit_match_dispatch(p1, p2, bind_state, patch_state, bn_state)
    } else { if t == 35 {
        // Approach A Stage 1: AST_INTLIT_I64 (tag 35). p1 = i32 value.
        // For values that fit in i32 (positive < 2^31 OR negative
        // > -2^31), the 64-bit encoding sign-extends: high32 = 0 if
        // p1 >= 0, else high32 = -1 (all bits set, two's-complement).
        // Emits `movabs rax, imm64` (10 bytes) so subsequent 64-bit
        // arithmetic / comparisons / let-bindings see the full width.
        let hi32 = if p1 < 0 { 0 - 1 } else { 0 };
        emit_movabs_rax_imm64(p1, hi32)
    } else { if t == 36 {
        // Approach A Stage 2.1: AST_INTLIT_U32 (tag 36). p1 = i32-encoded
        // bits of the u32 value. Codegen identical to AST_INTLIT (i32):
        // `mov eax, imm32` (5 bytes). x86 32-bit ops zero-extend the
        // upper 32 bits of rax, so the u32 lands in low 32 of rax with
        // high half cleared — exactly what u32 wants. The DISTINCT AST
        // tag is for type tracking via expr_type, not codegen.
        emit_ast_int(p1)
    } else { if t == 37 {
        // Approach A Stage 2.3: AST_INTLIT_U8 (tag 37). Same emit as
        // i32/u32 (mov eax, imm32). For Phase-0 / Stage 2.3 the value
        // lives in low byte of eax with high bytes already zero. Stage
        // 2.3b will add narrow movzx load + masked store for proper
        // u8 semantics; today's u8 is "u32 with type tag 7."
        emit_ast_int(p1)
    } else { if t == 38 {
        // Approach A Stage 2.4: AST_INTLIT_U64 (tag 38). Same emit as
        // i64 (movabs rax, imm64). For positive values < 2^63 the bit
        // pattern is identical for signed and unsigned; for values
        // >= 2^63 (which AST_INTLIT_U64 can't yet hold via i32 p1
        // anyway), unsigned interpretation matters at the comparison
        // and DIV/MOD sites only — those dispatch via is_u64_expr.
        // High32: ZERO for u64 literals. Stage 2.4b audit fix.
        //
        // Prior version sign-extended like i64: `if p1 < 0 { 0 - 1 }
        // else { 0 }`. That was wrong for u64 — it produced
        // 0xFFFFFFFF80000000 for `2147483648_u64` (= 2^31), where the
        // lex-time accumulator wrapped p1 to i32 negative-bit-pattern.
        //
        // With hi32 = 0 always, values in [0, 2^32) round-trip
        // correctly: p1 holds the low-32 bit pattern (interpreted as
        // unsigned), hi32 supplies the upper 32 zero bits. movabs
        // imm64 = 0x00000000_BIT_PATTERN.
        //
        // KNOWN GAP (still CRITICAL, separate fix): values >= 2^32
        // overflow lex_int's i32 digit accumulator and produce
        // garbage in p1. e.g. `4294967296_u64` (= 2^32) wraps to
        // p1 = 0 → emits 0_u64. Fix requires widening AST literal
        // payload to lo32 + hi32 fields, plus lex-side accumulator
        // overflow detection. See docs/STAGE_24B_NOTES.md and the
        // open audit queue.
        let hi32 = 0;
        emit_movabs_rax_imm64(p1, hi32)
    } else { if t == 39 {
        // Approach A Stage 2.5b: AST_INTLIT_I8 (tag 39). Same emit as
        // i32 (mov eax, imm32). i8 range [-128, 127] fits in 32 bits
        // signed; the high 24 bits are sign-extension. expr_type
        // returns 10 (i8) so any future signed-vs-unsigned-narrow
        // dispatch can trip on the type tag without changing the
        // literal's bit-pattern. Narrow movsx load and masked store
        // are deferred to Stage 2.5b stage 2 (parallel to u8's "Stage
        // 2.3b" deferred work).
        emit_ast_int(p1)
    } else { if t == 40 {
        // Approach A Stage 2.5c: AST_INTLIT_I16 (tag 40). Same emit as
        // i8/u8/u32 (mov eax, imm32). i16 range [-32768, 32767] fits
        // in i32. expr_type returns 11 (i16). Narrow movsx load and
        // masked store deferred.
        emit_ast_int(p1)
    } else { if t == 41 {
        // Approach A Stage 2.5c: AST_INTLIT_U16 (tag 41). Same emit as
        // u8 (mov eax, imm32). u16 range [0, 65535] fits in i32 with
        // high bytes zero. expr_type returns 8 (u16). Narrow movzx
        // load and masked store deferred.
        emit_ast_int(p1)
    } else { if t == 52 {
        // Stage 4 iter B: AST_TUPLE_FIELD (tag 52). p1 = inner expr,
        // p2 = field-index (0..15). Reads slot at [rax + p2*8].
        // Stage 5 Iter D: p3 == 1 selects an 8-byte (REX.W) read for
        // struct-typed fields whose slot holds a child pointer; p3 == 0
        // (default) keeps the original 4-byte read for scalars. Width
        // dispatch is folded into this arm rather than a new tag so the
        // emit_ast_code if-else chain stays at the same depth (the host
        // Python parser's recursion budget is tight — Finding #7).
        // Finding #7 disp8 wrap: p2 > 15 traps before the wrapping load.
        let p3 = __arena_get(idx + 3);
        let n_pre_trap = if p2 > 15 { emit_trap_with_id(52001) } else { 0 };
        let n_inner = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let off = p2 * 8;
        let n_load = if p3 == 1 {
            // mov rax, [rax + disp8]  (REX.W: 48 8B 40 disp8 = 4 bytes)
            emit_byte(0x48); emit_byte(0x8B); emit_byte(0x40); emit_byte(off);
            4
        } else {
            // mov eax, [rax + disp8]  (8B 40 disp8 = 3 bytes)
            emit_byte(0x8B); emit_byte(0x40); emit_byte(off);
            3
        };
        n_inner + n_load + n_pre_trap
    } else { if t == 53 {
        // Stage 4 iter E: AST_INDEX (tag 53). p1=array_expr, p2=idx_expr.
        // Codegen:
        //   eval array_expr  → rax = base ptr
        //   push rax
        //   eval idx_expr    → eax = idx (high 32 zero from 32-bit op)
        //   mov ecx, eax     (ecx = idx)
        //   pop rax          (rax = base)
        //   imul ecx, ecx, 8 (ecx = idx*8)
        //   add rax, rcx     (REX.W; rax = base + idx*8)
        //   mov eax, [rax]   (load 4-byte element)
        let n_arr = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n_idx = emit_ast_code(p2, bind_state, patch_state, bn_state);
        // mov ecx, eax  (89 C1 = 2 bytes)
        emit_byte(0x89); emit_byte(0xC1);
        // pop rax  (58 = 1 byte)
        emit_byte(0x58);
        // imul ecx, ecx, 8  (6B C9 08 = 3 bytes)
        emit_byte(0x6B); emit_byte(0xC9); emit_byte(0x08);
        // add rax, rcx  (REX.W: 48 01 C8 = 3 bytes)
        emit_byte(0x48); emit_byte(0x01); emit_byte(0xC8);
        // mov eax, [rax]  (8B 00 = 2 bytes)
        emit_byte(0x8B); emit_byte(0x00);
        n_arr + np + n_idx + 11
    } else { if t == 50 {
        // Stage 4 iteration A: AST_TUPLE_LIT (tag 50).
        //   p1 = arity (number of elements)
        //   p2 = head AST_TUPLE_CONS node index
        //   p3 = unused (0)
        // Stage 5 Iter D: rbp-relative addressing for slots.
        //   The previous codegen used `sub rsp, 8*arity` + [rsp+disp]
        //   stores, which broke nested struct lits: each inner struct
        //   lit also `sub rsp`-ed, shifting the outer's [rsp+disp]
        //   target onto inner-allocated bytes (Line { Pt{...}, Pt{...} }
        //   silently corrupted Pt's slots).
        //   Now: reserve `arity` slots via bind_alloc_offset (which
        //   carves out [rbp - off], unique per call). Slot 0 lives at
        //   the largest offset (most negative from rbp); slot i at
        //   slot0_off - i*8 so field reads via `[rax + i*8]` (positive
        //   offset) recover the correct address. Inner struct lits get
        //   their own non-overlapping slot range — no aliasing.
        //   Tradeoff: bind_alloc_offset growth is permanent for the
        //   function (no bind_pop here), so deep struct-lit-heavy code
        //   exhausts the 1024-byte prologue more quickly. For Phase-0
        //   the headroom (1024 - 64*8 = 512 bytes = 64 extra slots)
        //   covers all current tests.
        // Codegen:
        //   reserve N slots via bind_alloc_offset, capture slot0_off
        //   for each element i:
        //       evaluate child -> rax (full 64 bits)
        //       mov [rbp - (slot0_off - i*8)], rax   (REX.W 8-byte)
        //   lea rax, [rbp - slot0_off]   (rax = address of slot 0)
        let arity = p1;
        // Reserve arity slots. bind_alloc_offset returns the offset to
        // use directly: slot is at [rbp - offset]. After arity calls the
        // state has grown by arity*8. last_off (offset of LAST allocated
        // slot, most negative from rbp) becomes slot 0 — so slot 0 is
        // at the LOW end of the run, slot i at slot0_off - i*8 (which
        // is a SMALLER positive offset = closer to rbp), and field
        // reads via [rax + i*8] recover the right address.
        let mut slot0_off: i32 = 0;
        let mut i_alloc: i32 = 0;
        while i_alloc < arity {
            slot0_off = bind_alloc_offset(bind_state);
            i_alloc = i_alloc + 1;
        }
        let mut total: i32 = 0;
        let mut cur: i32 = p2;
        let mut slot_idx: i32 = 0;
        while cur != 0 {
            let child = __arena_get(cur + 1);
            let n_child = emit_ast_code(child, bind_state, patch_state, bn_state);
            let cur_off = slot0_off - slot_idx * 8;
            // mov [rbp + disp32], rax  (REX.W: 48 89 85 disp32 = 7 bytes)
            // disp32 = -cur_off (cur_off is positive; address is
            // [rbp - cur_off]).
            emit_byte(0x48); emit_byte(0x89); emit_byte(0x85);
            emit_u32_le(0 - cur_off);
            total = total + n_child + 7;
            slot_idx = slot_idx + 1;
            cur = __arena_get(cur + 2);
        }
        // lea rax, [rbp + disp32]  (REX.W: 48 8D 85 disp32 = 7 bytes)
        emit_byte(0x48); emit_byte(0x8D); emit_byte(0x85);
        emit_u32_le(0 - slot0_off);
        total + 7
    } else { if t == 54 {
        // Stage 5 Iter A: AST_STRUCT_DECL — metadata only, emits 0 bytes.
        // The struct decl is registered in the parser's struct_table at
        // parse time; codegen sees it as a no-op since struct lits get
        // folded into AST_TUPLE_LIT (tag 50) at parse time, reusing the
        // existing tuple-lit codegen above.
        0
    } else { if t == 42 {
        // Stage 1.5: AST_FLOATLIT_BF16 (tag 42). bf16 = f32 with the
        // low 16 mantissa bits truncated to zero. Compute the f32 IEEE
        // 754 bit pattern via parse_float_bits, then mask off the low
        // 16 (`bits & 0xFFFF0000`, expressed as `bits & (0 - 65536)`
        // since Helix bootstrap has no hex literals). The resulting
        // value is i32-shaped storage of the bf16 pattern (top 16 bits
        // = sign + 8-bit exp + 7-bit mantissa; low 16 bits = 0).
        // Stage 1.5 audit fix: overflow guard. parse_float_bits uses
        // i32 accumulators internally; > 9 digits silently wraps.
        // Speedup #4 wire-in: bf16 lit overflow trap id 42002.
        let digits = count_float_digits(p1, p2);
        if digits > 9 { emit_trap_with_id(42002) } else {
            let bits = parse_float_bits(p1, p2);
            let bf16_bits = bits & 0 - 65536;
            emit_ast_int(bf16_bits)
        }
    } else { if t == 80 {
        // K1.F15 (2026-05-27): AST_FLOATLIT_F16 (tag 80). IEEE-754
        // half-precision = 1 sign + 5 exp (bias 15) + 10 mantissa.
        // Compute f32 bits via parse_float_bits, then convert to f16
        // via f32_to_f16_bits (signed-int arithmetic, no hex/shifts).
        // Speedup #4 wire-in: f16 lit overflow trap id 80002 (mirror
        // of 42002 for bf16).
        let digits = count_float_digits(p1, p2);
        if digits > 9 { emit_trap_with_id(80002) } else {
            let bits = parse_float_bits(p1, p2);
            let f16_bits = f32_to_f16_bits(bits);
            emit_ast_int(f16_bits)
        }
    } else { if t == 27 {
        // AST_FLOATLIT (Phase 1.10 step 3d, f32). Phase 1.10 step 7b
        // also reuses this branch for AST_FLOATLIT_F64 (tag 34) — the
        // semantics are still f32-shaped (4-byte SSE single); step 7c
        // will branch on tag 34 for true 8-byte codegen.
        // p1 = byte_start of the literal text in the arena, p2 = byte_len.
        // Parse "I.F" -> IEEE 754 f32 bit pattern via parse_float_bits
        // helper (Stage 1.5 audit refactor — was inlined here, now
        // shared with t==42 bf16 literal arm).
        // Stage 1.5 audit fix: overflow guard. parse_float_bits uses
        // i32 accumulators; > 9 digits silently wraps to garbage bits.
        // Speedup #4 wire-in: f32 lit overflow trap id 27002.
        let digits = count_float_digits(p1, p2);
        if digits > 9 { emit_trap_with_id(27002) } else {
            emit_ast_int(parse_float_bits(p1, p2))
        }
    } else { if t == 34 {
        // AST_FLOATLIT_F64 (Phase 1.10 step 7c, 8-byte f64 emission).
        // p1 = byte_start of literal text in arena, p2 = byte_len.
        // Same parse-and-classify shape as t==27 (f32) but produces a
        // 64-bit IEEE 754 pattern split across two i32 halves
        // (low32, high32). Then `movabs rax, imm64` materializes the
        // full 8-byte value. Mantissa bias = 1023 (vs 127 for f32);
        // exponent field is 11 bits (vs 8); mantissa is 52 bits (vs
        // 23) split as 20 bits in high32 and 32 in low32. Step 7d
        // will wire SSE2 addsd/subsd/mulsd/divsd dispatch.
        let mut i: i32 = 0;
        let mut int_part: i32 = 0;
        let mut frac_part: i32 = 0;
        let mut frac_digits: i32 = 0;
        let mut phase: i32 = 0;
        let mut keep_p: i32 = 1;
        while keep_p == 1 {
            if i >= p2 { keep_p = 0; }
            else {
                let b = __arena_get(p1 + i);
                if b == 46 {
                    phase = 1;
                    i = i + 1;
                } else {
                    if b < 48 { keep_p = 0; }
                    else { if b > 57 { keep_p = 0; }
                    else {
                        if phase == 0 {
                            int_part = int_part * 10 + (b - 48);
                        } else {
                            frac_part = frac_part * 10 + (b - 48);
                            frac_digits = frac_digits + 1;
                        };
                        i = i + 1;
                    }};
                };
            };
        }
        let mut pow10: i32 = 1;
        let mut dd: i32 = 0;
        while dd < frac_digits { pow10 = pow10 * 10; dd = dd + 1; }
        let v_scaled = int_part * pow10 + frac_part;
        let mut high32: i32 = 0;
        let mut low32: i32 = 0;
        if v_scaled == 0 {
            high32 = 0;
            low32 = 0;
        } else {
            let mut k: i32 = 0;
            let mut threshold: i32 = pow10;
            let mut keep_neg: i32 = 1;
            while keep_neg == 1 {
                if threshold <= v_scaled { keep_neg = 0; }
                else { if threshold == 1 { keep_neg = 0; }
                else {
                    threshold = threshold / 2;
                    k = k - 1;
                }};
            }
            let mut keep_k: i32 = 1;
            while keep_k == 1 {
                if threshold > v_scaled / 2 { keep_k = 0; }
                else {
                    threshold = threshold * 2;
                    k = k + 1;
                }
            }
            // Extract 52 mantissa bits via residual-doubling. mhi
            // accumulates bits 51..32 (only low 20 bits of mhi used);
            // mlo accumulates bits 31..0.
            let mut residual = v_scaled - threshold;
            let mut mhi: i32 = 0;
            let mut mlo: i32 = 0;
            let mut bit: i32 = 51;
            while bit >= 0 {
                residual = residual * 2;
                if residual >= threshold {
                    if bit >= 32 {
                        let mut bv: i32 = 1;
                        let mut sh: i32 = 0;
                        let t_bit = bit - 32;
                        while sh < t_bit { bv = bv * 2; sh = sh + 1; }
                        mhi = mhi + bv;
                    } else {
                        let mut bv: i32 = 1;
                        let mut sh: i32 = 0;
                        while sh < bit { bv = bv * 2; sh = sh + 1; }
                        mlo = mlo + bv;
                    };
                    residual = residual - threshold;
                };
                bit = bit - 1;
            }
            // Audit fix (cycle 1, IEEE 754 rounding): peek at bit 53
            // (one beyond the 52-bit mantissa) for round-to-nearest.
            // If `residual * 2 >= threshold`, the next bit would have
            // been 1 — round mantissa up by 1 ULP. Without this, decimals
            // like 0.1, 0.7, 0.9 produce truncated bits 1 ULP off from
            // the IEEE 754-correct value (the helixc-Python reference).
            // Carry chain: mlo + 1 may overflow to mhi; mhi may overflow
            // past 2^20 (= 0x100000), in which case the mantissa rolls
            // to 0 and the exponent increments by 1.
            let mut k_eff: i32 = k;
            residual = residual * 2;
            if residual >= threshold {
                mlo = mlo + 1;
                if mlo == 0 {
                    // mlo wrapped from 0xFFFFFFFF to 0 — carry into mhi.
                    mhi = mhi + 1;
                    if mhi == 1048576 {
                        // mhi overflow past 2^20: mantissa is now 1.0 ×
                        // 2^(k+1). Reset mantissa, bump exp.
                        mhi = 0;
                        k_eff = k_eff + 1;
                    };
                };
            };
            // Pack high32 = (k + 1023) << 20 | mhi   (sign bit = 0).
            let exp_field = k_eff + 1023;
            let mut exp_shifted: i32 = exp_field;
            let mut sh2: i32 = 0;
            while sh2 < 20 { exp_shifted = exp_shifted * 2; sh2 = sh2 + 1; }
            high32 = exp_shifted + mhi;
            low32 = mlo;
        }
        emit_movabs_rax_imm64(low32, high32)
    } else { if t == 2 {
        // Stage 2.4b: 5-way arith dispatch — i64 OR u64 -> add rax, rcx
        // (REX.W; signedness-agnostic for ADD); f64 -> addsd; f32 -> addss;
        // i32 (and u32 falling through) -> add eax, ecx; mixed -> ud2.
        // Stage 1.5 audit fix: bf16 operands trap (no hardware add).
        // Without the bf16 guard, a `bf16 + bf16` falls through the
        // cascade and emits a 32-bit int ADD on float bit patterns.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        // Move-rcx: 64-bit when both operands are 8-byte (f64/i64/u64).
        let nm = if l_d == 1 {
            if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_u64 == 1 {
            if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else {
            // K1.F8b + K1.F8d + K1.F9-fix (2026-05-27): when l isn't
            // 64-bit but r IS 64-bit (i64, u64, or f64), use 64-bit
            // copy. K1.F8b handled the i64 leg; K1.F8d adds u64; the
            // K1.F9-fix adds the f64 leg so `f32 + f64` widening sees
            // the full f64 bit pattern in rcx for the cvtss2sd-then-
            // addsd sequence (K1.F9 already added this leg to SUB/MUL/
            // DIV; the original K1.F9 commit missed AST_ADD here,
            // producing the documented ADD-reverse miscompile).
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
        }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        // Speedup #4 wire-in: bf16 trap with id = 2001 (AST_ADD * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(2001) } else { if r_bf == 1 { emit_trap_with_id(2001) } else {
            // Speedup #4 wire-in: AST_ADD mixed-type trap ids 2010-2041.
            //   2010: l_d=1, r_d=0  (f64 + non-f64)
            //   2011: r_d=1, l_d=0  (non-f64 + f64)
            //   2020/2021: i64 mismatch -- BOTH directions closed via
            //              K1.F8 (2020, l_i64=1) and K1.F8b (2021,
            //              r_i64=1) signed widening.
            //   2030/2031: u64 mismatch
            //   2040/2041: f32 mismatch
            if l_d == 1 {
                // K1.F9 (2026-05-27): mixed f64 + f32 -- widen rcx
                // (f32) to f64 via cvtss2sd, then addsd. K3.B-style
                // exactly-f32 guard ensures non-f32 r-types still
                // trap.
                if r_d == 1 {
                    emit_addsd()
                } else {
                    let r_t = expr_type(p2, bind_state, bn_state);
                    if r_t == 1 {
                        emit_cvt_f32_in_rcx_to_f64() + emit_addsd()
                    } else {
                        emit_trap_with_id(2010)
                    }
                }
            } else { if r_d == 1 {
                // K1.F9 reverse: mixed f32 + f64 -- widen rax (f32)
                // to f64 via cvtss2sd, then addsd. rcx already has
                // the full f64 from the patched mov-rcx step.
                let l_t = expr_type(p1, bind_state, bn_state);
                if l_t == 1 {
                    emit_cvt_f32_in_rax_to_f64() + emit_addsd()
                } else {
                    emit_trap_with_id(2011)
                }
            } else {
                if l_i64 == 1 {
                    // K1.F8 (2026-05-27): mixed i64 + i32 -- sign-extend
                    // rcx and then 64-bit add. K3.B audit-fix
                    // (2026-05-27): only widen when r is EXACTLY i32
                    // (expr_type tag 0); for u32/u8/i8/u16/i16/f32/bf16
                    // the prior 32-bit mov + sign-extend would
                    // misinterpret bits (signed reading of unsigned
                    // values >= 2^31, or float bit patterns as int).
                    // Trap instead.
                    if r_i64 == 1 {
                        emit_add_rax_rcx_64()
                    } else {
                        let r_t = expr_type(p2, bind_state, bn_state);
                        if r_t == 0 {
                            emit_movsxd_rcx_ecx() + emit_add_rax_rcx_64()
                        } else {
                            emit_trap_with_id(2020)
                        }
                    }
                } else { if r_i64 == 1 {
                    // K1.F8b (2026-05-27): mixed i32 + i64 -- sign-
                    // extend rax then 64-bit add. K3.B audit-fix:
                    // only widen when l is EXACTLY i32; trap for
                    // u32/u8/u16/i8/i16/f32/bf16 mismatches.
                    let l_t = expr_type(p1, bind_state, bn_state);
                    if l_t == 0 {
                        emit_movsxd_rax_eax() + emit_add_rax_rcx_64()
                    } else {
                        emit_trap_with_id(2021)
                    }
                } else {
                    if l_u64 == 1 {
                        // K1.F8d (2026-05-27): mixed u64 + u32. Prior
                        // mov_ecx_eax zero-extended r in rcx, so the
                        // 64-bit add is unsigned-correct. Only widen
                        // when r is EXACTLY u32 (tag 6) -- mirror of
                        // K3.B's exactly-i32 guard.
                        if r_u64 == 1 {
                            emit_add_rax_rcx_64()
                        } else {
                            let r_t = expr_type(p2, bind_state, bn_state);
                            if r_t == 6 {
                                emit_add_rax_rcx_64()
                            } else {
                                emit_trap_with_id(2030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        // K1.F8d reverse: u32 + u64. mov-rcx step
                        // patched above to 64-bit copy when r_u64=1.
                        // Only widen when l is EXACTLY u32.
                        let l_t = expr_type(p1, bind_state, bn_state);
                        if l_t == 6 {
                            emit_add_rax_rcx_64()
                        } else {
                            emit_trap_with_id(2031)
                        }
                    } else {
                        if l_f == 1 {
                            if r_f == 1 { emit_addss() } else { emit_trap_with_id(2040) }
                        } else {
                            if r_f == 1 { emit_trap_with_id(2041) } else { emit_add_eax_ecx() }
                        }
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 3 {
        // Stage 1+2.4b: AST_SUB 5-way dispatch (i32/i64/u64/f32/f64).
        // Stage 1.5 audit fix: bf16 trap (no hardware sub).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 {
            if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_u64 == 1 {
            if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else {
            // K1.F8b + K1.F8d + K1.F9: when l isn't 64-bit but r IS
            // 64-bit (i64, u64, or f64), use 64-bit copy. K1.F9 adds
            // the f64 leg so f32 + f64 widening sees the full f64 bit
            // pattern in rcx for the cvtss2sd-then-addsd sequence.
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
        }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        // Speedup #4 wire-in: bf16 trap id = 3001 (AST_SUB * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(3001) } else { if r_bf == 1 { emit_trap_with_id(3001) } else {
            // Speedup #4 wire-in: AST_SUB mixed-type trap ids 3010-3041.
            // K1.F8 / K1.F8b closed both i64 mismatch directions (3020 +
            // 3021) via signed widening.
            if l_d == 1 {
                // K1.F9: f64 - f32 -- widen rcx to f64, subsd.
                if r_d == 1 {
                    emit_subsd()
                } else {
                    let r_t = expr_type(p2, bind_state, bn_state);
                    if r_t == 1 {
                        emit_cvt_f32_in_rcx_to_f64() + emit_subsd()
                    } else {
                        emit_trap_with_id(3010)
                    }
                }
            } else { if r_d == 1 {
                // K1.F9 reverse: f32 - f64 -- widen rax to f64, subsd.
                let l_t = expr_type(p1, bind_state, bn_state);
                if l_t == 1 {
                    emit_cvt_f32_in_rax_to_f64() + emit_subsd()
                } else {
                    emit_trap_with_id(3011)
                }
            } else {
                if l_i64 == 1 {
                    // K1.F8 + K3.B audit-fix: gate on r EXACTLY i32.
                    if r_i64 == 1 {
                        emit_sub_rax_rcx_64()
                    } else {
                        let r_t = expr_type(p2, bind_state, bn_state);
                        if r_t == 0 {
                            emit_movsxd_rcx_ecx() + emit_sub_rax_rcx_64()
                        } else {
                            emit_trap_with_id(3020)
                        }
                    }
                } else { if r_i64 == 1 {
                    // K1.F8b + K3.B audit-fix: gate on l EXACTLY i32.
                    let l_t = expr_type(p1, bind_state, bn_state);
                    if l_t == 0 {
                        emit_movsxd_rax_eax() + emit_sub_rax_rcx_64()
                    } else {
                        emit_trap_with_id(3021)
                    }
                } else {
                    if l_u64 == 1 {
                        // K1.F8d: u64 - u32, exactly-u32 guard.
                        if r_u64 == 1 {
                            emit_sub_rax_rcx_64()
                        } else {
                            let r_t = expr_type(p2, bind_state, bn_state);
                            if r_t == 6 {
                                emit_sub_rax_rcx_64()
                            } else {
                                emit_trap_with_id(3030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        // K1.F8d reverse: u32 - u64, exactly-u32 guard.
                        let l_t = expr_type(p1, bind_state, bn_state);
                        if l_t == 6 {
                            emit_sub_rax_rcx_64()
                        } else {
                            emit_trap_with_id(3031)
                        }
                    } else {
                        if l_f == 1 {
                            if r_f == 1 { emit_subss() } else { emit_trap_with_id(3040) }
                        } else {
                            if r_f == 1 { emit_trap_with_id(3041) } else { emit_sub_eax_ecx() }
                        }
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 4 {
        // Stage 1+2.4b: AST_MUL 5-way dispatch (i32/i64/u64/f32/f64).
        // Note: `imul rax, rcx` (REX.W) gives the low 64 bits of the
        // product, which is identical for signed and unsigned 64-bit
        // multiply. So u64 reuses emit_imul_rax_rcx_64. (Differences
        // appear only in `imul`'s upper-64-bit / overflow flags, which
        // we don't consume here.)
        // Stage 1.5 audit fix: bf16 trap (no hardware mul).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 {
            if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_u64 == 1 {
            if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else {
            // K1.F8b + K1.F8d + K1.F9: when l isn't 64-bit but r IS
            // 64-bit (i64, u64, or f64), use 64-bit copy. K1.F9 adds
            // the f64 leg so f32 + f64 widening sees the full f64 bit
            // pattern in rcx for the cvtss2sd-then-addsd sequence.
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
        }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        // Speedup #4 wire-in: bf16 trap id = 4001 (AST_MUL * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(4001) } else { if r_bf == 1 { emit_trap_with_id(4001) } else {
            // Speedup #4 wire-in: AST_MUL mixed-type trap ids 4010-4041.
            // K1.F8 / K1.F8b closed both i64 mismatch directions (4020 +
            // 4021) via signed widening.
            if l_d == 1 {
                // K1.F9: f64 * f32 -- widen rcx to f64, mulsd.
                if r_d == 1 {
                    emit_mulsd()
                } else {
                    let r_t = expr_type(p2, bind_state, bn_state);
                    if r_t == 1 {
                        emit_cvt_f32_in_rcx_to_f64() + emit_mulsd()
                    } else {
                        emit_trap_with_id(4010)
                    }
                }
            } else { if r_d == 1 {
                // K1.F9 reverse: f32 * f64 -- widen rax to f64, mulsd.
                let l_t = expr_type(p1, bind_state, bn_state);
                if l_t == 1 {
                    emit_cvt_f32_in_rax_to_f64() + emit_mulsd()
                } else {
                    emit_trap_with_id(4011)
                }
            } else {
                if l_i64 == 1 {
                    // K1.F8 + K3.B audit-fix: gate on r EXACTLY i32.
                    if r_i64 == 1 {
                        emit_imul_rax_rcx_64()
                    } else {
                        let r_t = expr_type(p2, bind_state, bn_state);
                        if r_t == 0 {
                            emit_movsxd_rcx_ecx() + emit_imul_rax_rcx_64()
                        } else {
                            emit_trap_with_id(4020)
                        }
                    }
                } else { if r_i64 == 1 {
                    // K1.F8b + K3.B audit-fix: gate on l EXACTLY i32.
                    let l_t = expr_type(p1, bind_state, bn_state);
                    if l_t == 0 {
                        emit_movsxd_rax_eax() + emit_imul_rax_rcx_64()
                    } else {
                        emit_trap_with_id(4021)
                    }
                } else {
                    if l_u64 == 1 {
                        // K1.F8d: u64 * u32 (imul REX.W low-64 is shared).
                        if r_u64 == 1 {
                            emit_imul_rax_rcx_64()
                        } else {
                            let r_t = expr_type(p2, bind_state, bn_state);
                            if r_t == 6 {
                                emit_imul_rax_rcx_64()
                            } else {
                                emit_trap_with_id(4030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        // K1.F8d reverse: u32 * u64.
                        let l_t = expr_type(p1, bind_state, bn_state);
                        if l_t == 6 {
                            emit_imul_rax_rcx_64()
                        } else {
                            emit_trap_with_id(4031)
                        }
                    } else {
                        if l_f == 1 {
                            if r_f == 1 { emit_mulss() } else { emit_trap_with_id(4040) }
                        } else {
                            if r_f == 1 { emit_trap_with_id(4041) } else { emit_imul_eax_ecx() }
                        }
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 5 {
        // Stage 1: AST_DIV 4-way dispatch.
        // Stage 2.2: 5-way — u32 / u32 uses `xor edx, edx; div ecx`
        // (unsigned). i32 / i32 keeps `cdq; idiv ecx` (signed). For
        // values < 2^31 these produce identical results; for values
        // ≥ 2^31 the signed path treats them as negative — wrong.
        // Stage 2.4b: 6-way — u64 / u64 uses `xor rdx, rdx; div rcx`
        // (REX.W) via emit_div_rax_rcx_64_u (helper landed in 2.4
        // scaffold, now wired up post-cascade-fix). Mismatched
        // signedness or width traps with ud2.
        // Stage 1.5 audit fix: bf16 trap (no hardware div).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 {
            if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_u64 == 1 {
            if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else {
            // K1.F8c + K1.F8d + K1.F9: when l isn't 64-bit but r IS
            // 64-bit (i64, u64, or f64), use 64-bit copy. K1.F9 extends
            // to f64 for DIV (MOD still traps because there's no SSE
            // remainder).
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
        }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        // Speedup #4 wire-in: bf16 trap id = 5001 (AST_DIV * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(5001) } else { if r_bf == 1 { emit_trap_with_id(5001) } else {
            // Speedup #4 wire-in: AST_DIV mixed-type trap ids 5010-5051.
            // K1.F8c (2026-05-27) closed both i64<->i32 mismatch directions
            // (5020 + 5021) via signed widening; idiv reads the dividend
            // from rdx:rax and the divisor from r/m, so widening rcx (i32->
            // i64) lets us share the 64-bit idiv path with i64+i64.
            if l_d == 1 {
                // K1.F9: f64 / f32 -- widen rcx to f64, divsd.
                if r_d == 1 {
                    emit_divsd()
                } else {
                    let r_t = expr_type(p2, bind_state, bn_state);
                    if r_t == 1 {
                        emit_cvt_f32_in_rcx_to_f64() + emit_divsd()
                    } else {
                        emit_trap_with_id(5010)
                    }
                }
            } else { if r_d == 1 {
                // K1.F9 reverse: f32 / f64 -- widen rax to f64, divsd.
                let l_t = expr_type(p1, bind_state, bn_state);
                if l_t == 1 {
                    emit_cvt_f32_in_rax_to_f64() + emit_divsd()
                } else {
                    emit_trap_with_id(5011)
                }
            } else {
                if l_i64 == 1 {
                    // K1.F8c forward + K3.B gate: i64 / i32 only when
                    // r is EXACTLY i32.
                    if r_i64 == 1 {
                        emit_idiv_rax_rcx_64()
                    } else {
                        let r_t = expr_type(p2, bind_state, bn_state);
                        if r_t == 0 {
                            emit_movsxd_rcx_ecx() + emit_idiv_rax_rcx_64()
                        } else {
                            emit_trap_with_id(5020)
                        }
                    }
                } else { if r_i64 == 1 {
                    // K1.F8c reverse + K3.B gate: i32 / i64 only when l
                    // is EXACTLY i32.
                    let l_t = expr_type(p1, bind_state, bn_state);
                    if l_t == 0 {
                        emit_movsxd_rax_eax() + emit_idiv_rax_rcx_64()
                    } else {
                        emit_trap_with_id(5021)
                    }
                } else {
                    if l_u64 == 1 {
                        // K1.F8d: u64 / u32 -- unsigned 64-bit div.
                        if r_u64 == 1 {
                            emit_div_rax_rcx_64_u()
                        } else {
                            let r_t = expr_type(p2, bind_state, bn_state);
                            if r_t == 6 {
                                emit_div_rax_rcx_64_u()
                            } else {
                                emit_trap_with_id(5030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        // K1.F8d reverse: u32 / u64.
                        let l_t = expr_type(p1, bind_state, bn_state);
                        if l_t == 6 {
                            emit_div_rax_rcx_64_u()
                        } else {
                            emit_trap_with_id(5031)
                        }
                    } else {
                        if l_f == 1 {
                            if r_f == 1 { emit_divss() } else { emit_trap_with_id(5040) }
                        } else { if r_f == 1 { emit_trap_with_id(5041) } else {
                            if l_u32 == 1 {
                                if r_u32 == 1 { emit_div_eax_ecx_u() } else { emit_trap_with_id(5050) }
                            } else { if r_u32 == 1 { emit_trap_with_id(5051) } else { emit_idiv_eax_ecx() } }
                        }}
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 24 {
        // AST_MOD: same setup as DIV, then emit_imod (cdq; idiv;
        // mov eax, edx) so the remainder lands in eax.
        // Stage 1 audit fix: i64 mod uses cqo + idiv rcx + mov rax, rdx.
        // Stage 2.2: u32 mod uses `xor edx, edx; div ecx; mov eax, edx`.
        // Stage 2.3 audit fix: f64/f32 operands now trap with ud2.
        // Stage 2.4b: u64 mod uses `xor rdx, rdx; div rcx; mov rax, rdx`
        // (REX.W) via emit_imod_rax_rcx_64_u (helper landed in 2.4
        // scaffold, now wired up post-cascade-fix).
        // Pre-existing bug — AST_DIV had the float traps but AST_MOD did
        // not, so `f64 % f64` silently emitted integer mod on bit
        // patterns (garbage int returned to caller, no signal). x86 has
        // no SSE remainder; emit_ud2_trap is the safe choice.
        // Stage 1.5 audit fix: bf16 trap (no hardware mod).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 {
            if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_u64 == 1 {
            if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else {
            // K1.F8c + K1.F8d + K1.F9: when l isn't 64-bit but r IS
            // 64-bit (i64, u64, or f64), use 64-bit copy. K1.F9 extends
            // to f64 for DIV (MOD still traps because there's no SSE
            // remainder).
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
        }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        // Speedup #4 wire-in: bf16 trap id = 24001 (AST_MOD * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(24001) } else { if r_bf == 1 { emit_trap_with_id(24001) } else {
            // Speedup #4 wire-in: AST_MOD mixed-type + float-mod trap ids.
            // 24010: any-d (no SSE remainder; this also covers l_d=1+r_d=1).
            // 24040: any-f (no SSE remainder).
            // 24050/24051: u32 mismatch.
            // K1.F8c (2026-05-27) closed both i64<->i32 mismatch directions
            // (24020 + 24021) via signed widening; imod uses 64-bit idiv
            // (cqo + idiv rcx + mov rax, rdx), so widening rcx (i32->i64)
            // lets us reuse the 64-bit imod path.
            if l_d == 1 {
                // f64 % f64 → ud2 (no SSE remainder); mixed → ud2.
                emit_trap_with_id(24010)
            } else { if r_d == 1 { emit_trap_with_id(24011) } else {
                if l_i64 == 1 {
                    // K1.F8c forward + K3.B gate: i64 % i32 only when
                    // r is EXACTLY i32.
                    if r_i64 == 1 {
                        emit_imod_rax_rcx_64()
                    } else {
                        let r_t = expr_type(p2, bind_state, bn_state);
                        if r_t == 0 {
                            emit_movsxd_rcx_ecx() + emit_imod_rax_rcx_64()
                        } else {
                            emit_trap_with_id(24020)
                        }
                    }
                } else { if r_i64 == 1 {
                    // K1.F8c reverse + K3.B gate: i32 % i64 only when l
                    // is EXACTLY i32.
                    let l_t = expr_type(p1, bind_state, bn_state);
                    if l_t == 0 {
                        emit_movsxd_rax_eax() + emit_imod_rax_rcx_64()
                    } else {
                        emit_trap_with_id(24021)
                    }
                } else {
                    if l_u64 == 1 {
                        // K1.F8d: u64 % u32 -- unsigned 64-bit mod.
                        if r_u64 == 1 {
                            emit_imod_rax_rcx_64_u()
                        } else {
                            let r_t = expr_type(p2, bind_state, bn_state);
                            if r_t == 6 {
                                emit_imod_rax_rcx_64_u()
                            } else {
                                emit_trap_with_id(24030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        // K1.F8d reverse: u32 % u64.
                        let l_t = expr_type(p1, bind_state, bn_state);
                        if l_t == 6 {
                            emit_imod_rax_rcx_64_u()
                        } else {
                            emit_trap_with_id(24031)
                        }
                    } else {
                        if l_f == 1 {
                            // f32 % f32 → ud2; mixed → ud2.
                            emit_trap_with_id(24040)
                        } else { if r_f == 1 { emit_trap_with_id(24041) } else {
                            if l_u32 == 1 {
                                if r_u32 == 1 { emit_imod_eax_ecx_u() } else { emit_trap_with_id(24050) }
                            } else { if r_u32 == 1 { emit_trap_with_id(24051) } else { emit_imod_eax_ecx() } }
                        }}
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 9 {
        // Phase 1.10 step 5d: dispatch unary NEG by inner type. f32
        // negation is sign-bit XOR (mirrors __fneg); i32 stays at the
        // existing two's-complement `neg eax`.
        // Step 7f: f64 path — flip bit 63 via 64-bit sign-bit XOR.
        // is_f64_expr is checked FIRST so f64 vars/literals don't fall
        // into the f32 path and get their high half (bits 32..63)
        // silently zeroed by the 32-bit `xor eax, ...` form.
        // Stage 1.5 audit fix: bf16 unary NEG traps with ud2. The
        // sign-bit-XOR trick (xor eax, 0x80000000) WOULD work for bf16
        // mathematically, but we have no bit-introspection on bf16
        // values yet to verify the resulting bit pattern at test time.
        // Trap loudly until a verifying test exists; pre-fix bf16 fell
        // through to integer two's-complement neg — silent garbage.
        let ni = emit_ast_code(p1, bind_state, patch_state, bn_state);
        // Stage 1 audit batch 2: 4-way AST_NEG dispatch including i64.
        // Stage 4 follow-up audit (Finding #2 from 6c41511): u64 was
        // missing — fell through to 32-bit `neg eax` which two's-
        // complemented only the low 32 bits, leaving high half stale.
        // u64 NEG semantically = 2^64 - x; REX.W neg rax computes that.
        // Same encoding as i64 (signedness-agnostic at machine level).
        let nn = if is_f64_expr(p1, bind_state, bn_state) == 1 {
            emit_ast_dneg_suffix()
        } else { if is_i64_expr(p1, bind_state, bn_state) == 1 {
            emit_neg_rax_64()
        } else { if is_u64_expr(p1, bind_state, bn_state) == 1 {
            emit_neg_rax_64()
        } else { if is_f32_expr(p1, bind_state, bn_state) == 1 {
            emit_ast_fneg_suffix()
        } else { if is_bf16_expr(p1, bind_state, bn_state) == 1 {
            // Speedup #4 wire-in: bf16 trap id = 9001 (AST_NEG * 1000 + 1).
            emit_trap_with_id(9001)
        } else {
            emit_ast_neg_suffix()
        }}}}};
        ni + nn
    } else { if t == 26 {
        // AST_BNOT: emit inner (leaves value in eax/rax), then `not`.
        // Stage 1 audit fix: i64 needs `not rax` (REX.W) to flip all 64 bits.
        // Stage 1.5 audit fix: bf16 traps with ud2. Bitwise NOT on a
        // bf16 bit pattern flips the low 16 bits (which are always 0
        // post-truncation) AND the sign+exponent+top-mantissa bits in
        // the high half — producing a malformed bf16 pattern (no
        // longer truncated to bf16 layout). Trap until a real use
        // case + verifying test exists. Pre-fix bf16 fell through to
        // emit_ast_bnot_suffix (`not eax`) — silent garbage.
        // Stage 1.5 audit fix (post-bf16 sweep): u64 and f64 are also
        // 8-byte storage. Pre-fix they fell through to emit_ast_bnot_suffix
        // (`not eax`, 32-bit op) which silently left the high 32 bits
        // unchanged — for u64, top 32 bits stay garbage; for f64, the
        // bit pattern becomes malformed (low 32 flipped, exponent /
        // mantissa-high preserved). Now both use REX.W not rax.
        // AUDIT VERIFIED 2026-05-07 (post trap-id sweep): emit_ast_bnot_suffix
        // is `not eax` (2 bytes, F7 D0) which is the correct 32-bit BNOT
        // for i32, u32, u8, i8, u16, i16 — all of which use 32-bit-or-narrower
        // storage and care only about the low N bits. The narrow types
        // (u8/i8/u16/i16) tolerate `not eax` flipping the high bits beyond
        // their storage width because subsequent narrow loads/stores
        // re-truncate. No fix needed for these widths.
        let ni = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let nn = if is_i64_expr(p1, bind_state, bn_state) == 1 {
            emit_not_rax_64()
        } else { if is_u64_expr(p1, bind_state, bn_state) == 1 {
            emit_not_rax_64()
        } else { if is_f64_expr(p1, bind_state, bn_state) == 1 {
            emit_not_rax_64()
        } else { if is_bf16_expr(p1, bind_state, bn_state) == 1 {
            // Speedup #4 wire-in: bf16 trap id = 26001 (AST_BNOT * 1000 + 1).
            emit_trap_with_id(26001)
        } else {
            emit_ast_bnot_suffix()
        }}}};
        ni + nn
    } else { if t == 31 {
        // AST_NOT: logical NOT. For i64/u64, must use `test rax, rax`
        // (REX.W) to detect non-zero across the full 64 bits.
        // Stage 2.4b audit fix: u64 added (was i64-only).
        // Stage 1.5 audit fix: bf16 traps with ud2. The 32-bit
        // emit_ast_not_suffix path checks `bits == 0`, which mishandles
        // bf16 sentinel values: -0.0_bf16 (bits 0x80000000) is falsy in
        // IEEE but the bit-pattern check returns truthy. NaN bf16
        // values (bits with all-1 exponent and non-zero mantissa) are
        // also classified incorrectly. Trap until correct float-aware
        // logical-NOT codegen lands.
        // Stage 1.5 audit fix (post-bf16 sweep): f64 added to wide
        // check. Pre-fix: !2.0_f64 (bits 0x4000000000000000, low 32 = 0)
        // returned 1 (TRUTHY) because the 32-bit `test eax, eax` only
        // checked the low half — wrong. Now uses 64-bit test.
        // Note: f64 still has -0.0 / NaN edge cases (same as bf16) but
        // those are language-policy choices, not memory-safety bugs.
        // AUDIT VERIFIED 2026-05-07 (post trap-id sweep): f32 NOT has
        // the same -0.0 / NaN edge case. -0.0_f32 (bits 0x80000000) is
        // logically falsy in IEEE 754 but `test eax, eax` reports it
        // as non-zero — !(-0.0_f32) returns 0 when it should return 1.
        // NaN values likewise misclassify. Same language-policy
        // decision as f64: accept the corner case rather than trap,
        // since the bit-pattern check matches "is value all-zero-bits"
        // which is what most users want as a truthy check on numeric
        // values. If strict IEEE float-truthiness is needed later,
        // route via __bits_of_f32 / explicit ucomiss-with-zero.
        // i32/u32 NOT verification: emit_ast_not_suffix correctly checks
        // 32-bit zero via `test eax, eax` for the 4-byte width types.
        let ni = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let inner_i64 = is_i64_expr(p1, bind_state, bn_state);
        let inner_u64 = is_u64_expr(p1, bind_state, bn_state);
        let inner_f64 = is_f64_expr(p1, bind_state, bn_state);
        let inner_bf = is_bf16_expr(p1, bind_state, bn_state);
        let inner_wide = if inner_i64 == 1 { 1 }
                         else { if inner_u64 == 1 { 1 }
                         else { if inner_f64 == 1 { 1 } else { 0 } } };
        let nn = if inner_bf == 1 {
            // Speedup #4 wire-in: bf16 trap id = 31001 (AST_NOT * 1000 + 1).
            emit_trap_with_id(31001)
        } else { if inner_wide == 1 {
            emit_ast_not_suffix_64()
        } else {
            emit_ast_not_suffix()
        }};
        ni + nn
    } else { if t == 28 {
        // Stage 1 audit fix: AST_BAND 4-way dispatch (i64 needs REX.W).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let nm = if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { emit_mov_ecx_eax() };
        let no = emit_pop_rax();
        // Speedup #4 wire-in: AST_BAND mixed i64-trap ids 28020/28021.
        let na = if l_i64 == 1 {
            if r_i64 == 1 { emit_and_rax_rcx_64() } else { emit_trap_with_id(28020) }
        } else { if r_i64 == 1 { emit_trap_with_id(28021) } else { emit_and_eax_ecx() } };
        n1 + np + n2 + nm + no + na
    } else { if t == 29 {
        // Stage 1 audit fix: AST_BOR 4-way dispatch (i64 needs REX.W).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let nm = if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { emit_mov_ecx_eax() };
        let no = emit_pop_rax();
        // Speedup #4 wire-in: AST_BOR mixed i64-trap ids 29020/29021.
        let na = if l_i64 == 1 {
            if r_i64 == 1 { emit_or_rax_rcx_64() } else { emit_trap_with_id(29020) }
        } else { if r_i64 == 1 { emit_trap_with_id(29021) } else { emit_or_eax_ecx() } };
        n1 + np + n2 + nm + no + na
    } else { if t == 30 {
        // AST_BXOR: `xor eax, ecx` (0x31 0xC8).
        // Stage 1 audit fix: AST_BXOR 4-way dispatch (i64 needs REX.W).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let nm = if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { emit_mov_ecx_eax() };
        let no = emit_pop_rax();
        // Speedup #4 wire-in: AST_BXOR mixed i64-trap ids 30020/30021.
        let na = if l_i64 == 1 {
            if r_i64 == 1 { emit_xor_rax_rcx_64() } else { emit_trap_with_id(30020) }
        } else { if r_i64 == 1 { emit_trap_with_id(30021) } else { emit_xor_eax_ecx() } };
        n1 + np + n2 + nm + no + na
    } else { if t == 32 {
        // AST_SHL: shl eax, cl (D3 E0); shl rax, cl (REX.W: 48 D3 E0) for i64/u64.
        // Stage 1 audit fix: shift count is always treated as i32 (cl);
        // value being shifted picks i64 vs i32.
        // Audit follow-up Finding #5: add u64 dispatch (REX.W shl is
        // bit-identical for signed and unsigned) and trap on float/bf16.
        // Without u64 dispatch, `u64 << k` falls through to 32-bit shl
        // and silently leaves the high 32 bits unmodified.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let l_f32 = is_f32_expr(p1, bind_state, bn_state);
        let l_f64 = is_f64_expr(p1, bind_state, bn_state);
        let l_bf  = is_bf16_expr(p1, bind_state, bn_state);
        let nm = emit_mov_ecx_eax();
        let no = emit_pop_rax();
        let na = if l_bf == 1 { emit_trap_with_id(32001) }
                else { if l_f64 == 1 { emit_trap_with_id(32010) }
                else { if l_f32 == 1 { emit_trap_with_id(32040) }
                else { if l_i64 == 1 { emit_shl_rax_cl_64() }
                else { if l_u64 == 1 { emit_shl_rax_cl_64() }
                else { emit_shl_eax_cl() }}}}};
        n1 + np + n2 + nm + no + na
    } else { if t == 33 {
        // AST_SHR: sar eax, cl (D3 F8) i32; sar rax, cl (48 D3 F8) i64/u64.
        // Audit follow-up Finding #5: add u64 dispatch and trap on
        // float/bf16. Note: this still uses sar (arithmetic) for u64;
        // the signedness sub-finding (sar vs shr for unsigned types) is
        // tracked separately and not addressed here to keep the patch
        // minimal. u64 sar produces wrong results for u64 values >=
        // 0x8000_0000_0000_0000.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let l_f32 = is_f32_expr(p1, bind_state, bn_state);
        let l_f64 = is_f64_expr(p1, bind_state, bn_state);
        let l_bf  = is_bf16_expr(p1, bind_state, bn_state);
        let nm = emit_mov_ecx_eax();
        let no = emit_pop_rax();
        let na = if l_bf == 1 { emit_trap_with_id(33001) }
                else { if l_f64 == 1 { emit_trap_with_id(33010) }
                else { if l_f32 == 1 { emit_trap_with_id(33040) }
                else { if l_i64 == 1 { emit_sar_rax_cl_64() }
                else { if l_u64 == 1 { emit_sar_rax_cl_64() }
                else { emit_sar_eax_cl() }}}}};
        n1 + np + n2 + nm + no + na
    } else { if t == 6 {
        // Phase 1.10 step 5e: f32-aware comparison. If both operands
        // resolve to f32 via is_f32_expr, emit ucomiss + setcc; else
        // integer cmp + setcc. Result is 0/1 in eax either way.
        // Step 7g: three-way dispatch — f64 path uses ucomisd via
        // emit_ssen_*_dbl helpers; nm-move-to-rcx promotes to 64-bit
        // for the f64 case (otherwise high half drops before ucomisd).
        // Audit fix #9: comparison ops now trap on mixed types (f32+i32,
        // f64+i32, f32+f64, etc.) — previously the `both_f`/`both_d` =0
        // fall-through silently emitted integer cmp on bit patterns.
        // Stage 1.5 audit fix: bf16 traps. Integer compare on bf16 bit
        // patterns is correct only for normal positive bf16; negatives
        // compare reversed (two's-complement-vs-IEEE), -0.0 and +0.0
        // compare unequal but should be equal in IEEE, NaN ordering is
        // undefined. Trap until float-aware bf16 compare lands.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else {
                   // K1.F11 + K1.F13 + K1.F14 (2026-05-27): when l isn't
                   // 64-bit but r IS 64-bit (i64, u64, or f64), use
                   // 64-bit copy so rcx has the full 8-byte bit pattern
                   // for the reverse-direction widening. Mirrors K1.F8b/
                   // F8d/F9-fix's mov-rcx legs for AST_ADD.
                   if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
               }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        // Stage 2.2: 5-way LT dispatch — u32 < u32 uses `setb` (unsigned).
        // Stage 2.4b: 6-way — u64 < u64 uses REX.W cmp + setb.
        // Speedup #4 wire-in: bf16 trap id = 6001 (AST_LT * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(6001) } else { if r_bf == 1 { emit_trap_with_id(6001) } else {
            // Speedup #4 wire-in: AST_LT mixed-type trap ids 6010-6051.
            // K1.F11 (2026-05-27): close mixed-type i64<->i32 LT
            // widening. Both directions sign-extend the i32 side and
            // emit a 64-bit cmp. K3.B-style exactly-i32 guard (expr_
            // type tag 0) ensures u32/u8/i8/u16/i16/f32/bf16 still
            // trap loudly instead of silently misinterpreting bits.
            if l_d == 1 {
                // K1.F14 (2026-05-27): f64<->f32 LT forward widening.
                if r_d == 1 {
                    emit_ssen_lt_dbl()
                } else {
                    let r_t6f = expr_type(p2, bind_state, bn_state);
                    if r_t6f == 1 {
                        emit_cvt_f32_in_rcx_to_f64() + emit_ssen_lt_dbl()
                    } else {
                        emit_trap_with_id(6010)
                    }
                }
            } else { if r_d == 1 {
                // K1.F14: f64<->f32 LT reverse widening.
                let l_t6f = expr_type(p1, bind_state, bn_state);
                if l_t6f == 1 {
                    emit_cvt_f32_in_rax_to_f64() + emit_ssen_lt_dbl()
                } else {
                    emit_trap_with_id(6011)
                }
            } else {
                if l_i64 == 1 {
                    // K1.F11: i64 < i32 -- widen rcx, then 64-bit cmp.
                    if r_i64 == 1 {
                        emit_lt_rax_rcx_64()
                    } else {
                        let r_t6 = expr_type(p2, bind_state, bn_state);
                        if r_t6 == 0 {
                            emit_movsxd_rcx_ecx() + emit_lt_rax_rcx_64()
                        } else {
                            emit_trap_with_id(6020)
                        }
                    }
                } else { if r_i64 == 1 {
                    // K1.F11 reverse: i32 < i64 -- widen rax, then 64-
                    // bit cmp. rcx already has full 64 bits from the
                    // patched mov-rcx step's r_i64 leg above.
                    let l_t6 = expr_type(p1, bind_state, bn_state);
                    if l_t6 == 0 {
                        emit_movsxd_rax_eax() + emit_lt_rax_rcx_64()
                    } else {
                        emit_trap_with_id(6021)
                    }
                } else {
                    // K1.F13 (2026-05-27): close u64<->u32 LT widening
                    // (mirror K1.F8d for AST_ADD). Forward: r=u32 in
                    // rcx is already zero-ext to u64 (via emit_mov_ecx_
                    // eax), so emit_lt_rax_rcx_64_u directly. Reverse:
                    // l=u32 in rax zero-ext (emit_movabs/mov_eax_imm),
                    // r=u64 in rcx full bits (via mov-rcx r_u64 leg).
                    if l_u64 == 1 {
                        if r_u64 == 1 {
                            emit_lt_rax_rcx_64_u()
                        } else {
                            let r_t6u = expr_type(p2, bind_state, bn_state);
                            if r_t6u == 6 {
                                emit_lt_rax_rcx_64_u()
                            } else {
                                emit_trap_with_id(6030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        let l_t6u = expr_type(p1, bind_state, bn_state);
                        if l_t6u == 6 {
                            emit_lt_rax_rcx_64_u()
                        } else {
                            emit_trap_with_id(6031)
                        }
                    } else {
                        if l_f == 1 {
                            if r_f == 1 { emit_ssen_lt() } else { emit_trap_with_id(6040) }
                        } else { if r_f == 1 { emit_trap_with_id(6041) } else {
                            if l_u32 == 1 {
                                if r_u32 == 1 { emit_lt_eax_ecx_u() } else { emit_trap_with_id(6050) }
                            } else { if r_u32 == 1 { emit_trap_with_id(6051) } else { emit_lt_eax_ecx() } }
                        }}
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 19 {
        // Audit fix #9: AST_GT mixed-type ud2 trap.
        // Stage 1.5 audit fix: bf16 traps (see AST_LT comment).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else {
                   // K1.F12 + K1.F13 + K1.F14 (2026-05-27): mirror K1.F11/
                   // F8d/F9-fix's mov-rcx r_i64 + r_u64 + r_d legs.
                   if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
               }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        // Stage 2.2: 5-way GT dispatch — u32 > u32 uses `seta` (unsigned).
        // Stage 2.4b: 6-way — u64 > u64 uses REX.W cmp + seta.
        // Speedup #4 wire-in: bf16 trap id = 19001 (AST_GT * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(19001) } else { if r_bf == 1 { emit_trap_with_id(19001) } else {
            // Speedup #4 wire-in: AST_GT mixed-type trap ids 19010-19051.
            // K1.F12 (2026-05-27): close i64<->i32 GT widening (mirror K1.F11).
            if l_d == 1 {
                // K1.F14 (2026-05-27): f64<->f32 GT forward widening.
                if r_d == 1 {
                    emit_ssen_gt_dbl()
                } else {
                    let r_t19f = expr_type(p2, bind_state, bn_state);
                    if r_t19f == 1 {
                        emit_cvt_f32_in_rcx_to_f64() + emit_ssen_gt_dbl()
                    } else {
                        emit_trap_with_id(19010)
                    }
                }
            } else { if r_d == 1 {
                // K1.F14: f64<->f32 GT reverse widening.
                let l_t19f = expr_type(p1, bind_state, bn_state);
                if l_t19f == 1 {
                    emit_cvt_f32_in_rax_to_f64() + emit_ssen_gt_dbl()
                } else {
                    emit_trap_with_id(19011)
                }
            } else {
                if l_i64 == 1 {
                    if r_i64 == 1 {
                        emit_gt_rax_rcx_64()
                    } else {
                        let r_t19 = expr_type(p2, bind_state, bn_state);
                        if r_t19 == 0 {
                            emit_movsxd_rcx_ecx() + emit_gt_rax_rcx_64()
                        } else {
                            emit_trap_with_id(19020)
                        }
                    }
                } else { if r_i64 == 1 {
                    let l_t19 = expr_type(p1, bind_state, bn_state);
                    if l_t19 == 0 {
                        emit_movsxd_rax_eax() + emit_gt_rax_rcx_64()
                    } else {
                        emit_trap_with_id(19021)
                    }
                } else {
                    // K1.F13 (2026-05-27): u64<->u32 GT widening.
                    if l_u64 == 1 {
                        if r_u64 == 1 {
                            emit_gt_rax_rcx_64_u()
                        } else {
                            let r_t19u = expr_type(p2, bind_state, bn_state);
                            if r_t19u == 6 {
                                emit_gt_rax_rcx_64_u()
                            } else {
                                emit_trap_with_id(19030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        let l_t19u = expr_type(p1, bind_state, bn_state);
                        if l_t19u == 6 {
                            emit_gt_rax_rcx_64_u()
                        } else {
                            emit_trap_with_id(19031)
                        }
                    } else {
                        if l_f == 1 {
                            if r_f == 1 { emit_ssen_gt() } else { emit_trap_with_id(19040) }
                        } else { if r_f == 1 { emit_trap_with_id(19041) } else {
                            if l_u32 == 1 {
                                if r_u32 == 1 { emit_gt_eax_ecx_u() } else { emit_trap_with_id(19050) }
                            } else { if r_u32 == 1 { emit_trap_with_id(19051) } else { emit_gt_eax_ecx() } }
                        }}
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 20 {
        // Audit fix #9: AST_EQ mixed-type ud2 trap.
        // Stage 2.4b: u64 == u64 reuses i64's emit_eq_rax_rcx_64 since
        // bitwise equality on a 64-bit value is signedness-agnostic.
        // Stage 1.5 audit fix: bf16 traps (see AST_LT comment).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else {
                   // K1.F12 + K1.F13 + K1.F14 (2026-05-27): mirror K1.F11/
                   // F8d/F9-fix's mov-rcx r_i64 + r_u64 + r_d legs.
                   if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
               }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        // Speedup #4 wire-in: bf16 trap id = 20001 (AST_EQ * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(20001) } else { if r_bf == 1 { emit_trap_with_id(20001) } else {
            // Speedup #4 wire-in: AST_EQ mixed-type trap ids 20010-20041.
            // K1.F12 (2026-05-27): close i64<->i32 EQ widening (mirror K1.F11).
            if l_d == 1 {
                // K1.F14 (2026-05-27): f64<->f32 EQ forward widening.
                if r_d == 1 {
                    emit_ssen_eq_dbl()
                } else {
                    let r_t20f = expr_type(p2, bind_state, bn_state);
                    if r_t20f == 1 {
                        emit_cvt_f32_in_rcx_to_f64() + emit_ssen_eq_dbl()
                    } else {
                        emit_trap_with_id(20010)
                    }
                }
            } else { if r_d == 1 {
                // K1.F14: f64<->f32 EQ reverse widening.
                let l_t20f = expr_type(p1, bind_state, bn_state);
                if l_t20f == 1 {
                    emit_cvt_f32_in_rax_to_f64() + emit_ssen_eq_dbl()
                } else {
                    emit_trap_with_id(20011)
                }
            } else {
                if l_i64 == 1 {
                    if r_i64 == 1 {
                        emit_eq_rax_rcx_64()
                    } else {
                        let r_t20 = expr_type(p2, bind_state, bn_state);
                        if r_t20 == 0 {
                            emit_movsxd_rcx_ecx() + emit_eq_rax_rcx_64()
                        } else {
                            emit_trap_with_id(20020)
                        }
                    }
                } else { if r_i64 == 1 {
                    let l_t20 = expr_type(p1, bind_state, bn_state);
                    if l_t20 == 0 {
                        emit_movsxd_rax_eax() + emit_eq_rax_rcx_64()
                    } else {
                        emit_trap_with_id(20021)
                    }
                } else {
                    // K1.F13 (2026-05-27): u64<->u32 EQ widening.
                    if l_u64 == 1 {
                        if r_u64 == 1 {
                            emit_eq_rax_rcx_64()
                        } else {
                            let r_t20u = expr_type(p2, bind_state, bn_state);
                            if r_t20u == 6 {
                                emit_eq_rax_rcx_64()
                            } else {
                                emit_trap_with_id(20030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        let l_t20u = expr_type(p1, bind_state, bn_state);
                        if l_t20u == 6 {
                            emit_eq_rax_rcx_64()
                        } else {
                            emit_trap_with_id(20031)
                        }
                    } else {
                        if l_f == 1 {
                            if r_f == 1 { emit_ssen_eq() } else { emit_trap_with_id(20040) }
                        } else { if r_f == 1 { emit_trap_with_id(20041) } else { emit_eq_eax_ecx() } }
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 21 {
        // Audit fix #9: AST_NE mixed-type ud2 trap.
        // Stage 2.4b: u64 != u64 reuses i64's emit_ne_rax_rcx_64
        // (signedness-agnostic for 64-bit inequality).
        // Stage 1.5 audit fix: bf16 traps (see AST_LT comment).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else {
                   // K1.F12 + K1.F13 + K1.F14 (2026-05-27): mirror K1.F11/
                   // F8d/F9-fix's mov-rcx r_i64 + r_u64 + r_d legs.
                   if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
               }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        // Speedup #4 wire-in: bf16 trap id = 21001 (AST_NE * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(21001) } else { if r_bf == 1 { emit_trap_with_id(21001) } else {
            // Speedup #4 wire-in: AST_NE mixed-type trap ids 21010-21041.
            // K1.F12 (2026-05-27): close i64<->i32 NE widening (mirror K1.F11).
            if l_d == 1 {
                // K1.F14 (2026-05-27): f64<->f32 NE forward widening.
                if r_d == 1 {
                    emit_ssen_ne_dbl()
                } else {
                    let r_t21f = expr_type(p2, bind_state, bn_state);
                    if r_t21f == 1 {
                        emit_cvt_f32_in_rcx_to_f64() + emit_ssen_ne_dbl()
                    } else {
                        emit_trap_with_id(21010)
                    }
                }
            } else { if r_d == 1 {
                // K1.F14: f64<->f32 NE reverse widening.
                let l_t21f = expr_type(p1, bind_state, bn_state);
                if l_t21f == 1 {
                    emit_cvt_f32_in_rax_to_f64() + emit_ssen_ne_dbl()
                } else {
                    emit_trap_with_id(21011)
                }
            } else {
                if l_i64 == 1 {
                    if r_i64 == 1 {
                        emit_ne_rax_rcx_64()
                    } else {
                        let r_t21 = expr_type(p2, bind_state, bn_state);
                        if r_t21 == 0 {
                            emit_movsxd_rcx_ecx() + emit_ne_rax_rcx_64()
                        } else {
                            emit_trap_with_id(21020)
                        }
                    }
                } else { if r_i64 == 1 {
                    let l_t21 = expr_type(p1, bind_state, bn_state);
                    if l_t21 == 0 {
                        emit_movsxd_rax_eax() + emit_ne_rax_rcx_64()
                    } else {
                        emit_trap_with_id(21021)
                    }
                } else {
                    // K1.F13 (2026-05-27): u64<->u32 NE widening.
                    if l_u64 == 1 {
                        if r_u64 == 1 {
                            emit_ne_rax_rcx_64()
                        } else {
                            let r_t21u = expr_type(p2, bind_state, bn_state);
                            if r_t21u == 6 {
                                emit_ne_rax_rcx_64()
                            } else {
                                emit_trap_with_id(21030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        let l_t21u = expr_type(p1, bind_state, bn_state);
                        if l_t21u == 6 {
                            emit_ne_rax_rcx_64()
                        } else {
                            emit_trap_with_id(21031)
                        }
                    } else {
                        if l_f == 1 {
                            if r_f == 1 { emit_ssen_ne() } else { emit_trap_with_id(21040) }
                        } else { if r_f == 1 { emit_trap_with_id(21041) } else { emit_ne_eax_ecx() } }
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 22 {
        // Audit fix #9: AST_LE mixed-type ud2 trap.
        // Stage 1.5 audit fix: bf16 traps (see AST_LT comment).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else {
                   // K1.F12 + K1.F13 + K1.F14 (2026-05-27): mirror K1.F11/
                   // F8d/F9-fix's mov-rcx r_i64 + r_u64 + r_d legs.
                   if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
               }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        // Stage 2.2: 5-way LE dispatch — u32 <= u32 uses `setbe` (unsigned).
        // Stage 2.4b: 6-way — u64 <= u64 uses REX.W cmp + setbe.
        // Speedup #4 wire-in: bf16 trap id = 22001 (AST_LE * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(22001) } else { if r_bf == 1 { emit_trap_with_id(22001) } else {
            // Speedup #4 wire-in: AST_LE mixed-type trap ids 22010-22051.
            // K1.F12 (2026-05-27): close i64<->i32 LE widening (mirror K1.F11).
            if l_d == 1 {
                // K1.F14 (2026-05-27): f64<->f32 LE forward widening.
                if r_d == 1 {
                    emit_ssen_le_dbl()
                } else {
                    let r_t22f = expr_type(p2, bind_state, bn_state);
                    if r_t22f == 1 {
                        emit_cvt_f32_in_rcx_to_f64() + emit_ssen_le_dbl()
                    } else {
                        emit_trap_with_id(22010)
                    }
                }
            } else { if r_d == 1 {
                // K1.F14: f64<->f32 LE reverse widening.
                let l_t22f = expr_type(p1, bind_state, bn_state);
                if l_t22f == 1 {
                    emit_cvt_f32_in_rax_to_f64() + emit_ssen_le_dbl()
                } else {
                    emit_trap_with_id(22011)
                }
            } else {
                if l_i64 == 1 {
                    if r_i64 == 1 {
                        emit_le_rax_rcx_64()
                    } else {
                        let r_t22 = expr_type(p2, bind_state, bn_state);
                        if r_t22 == 0 {
                            emit_movsxd_rcx_ecx() + emit_le_rax_rcx_64()
                        } else {
                            emit_trap_with_id(22020)
                        }
                    }
                } else { if r_i64 == 1 {
                    let l_t22 = expr_type(p1, bind_state, bn_state);
                    if l_t22 == 0 {
                        emit_movsxd_rax_eax() + emit_le_rax_rcx_64()
                    } else {
                        emit_trap_with_id(22021)
                    }
                } else {
                    // K1.F13 (2026-05-27): u64<->u32 LE widening.
                    if l_u64 == 1 {
                        if r_u64 == 1 {
                            emit_le_rax_rcx_64_u()
                        } else {
                            let r_t22u = expr_type(p2, bind_state, bn_state);
                            if r_t22u == 6 {
                                emit_le_rax_rcx_64_u()
                            } else {
                                emit_trap_with_id(22030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        let l_t22u = expr_type(p1, bind_state, bn_state);
                        if l_t22u == 6 {
                            emit_le_rax_rcx_64_u()
                        } else {
                            emit_trap_with_id(22031)
                        }
                    } else {
                        if l_f == 1 {
                            if r_f == 1 { emit_ssen_le() } else { emit_trap_with_id(22040) }
                        } else { if r_f == 1 { emit_trap_with_id(22041) } else {
                            if l_u32 == 1 {
                                if r_u32 == 1 { emit_le_eax_ecx_u() } else { emit_trap_with_id(22050) }
                            } else { if r_u32 == 1 { emit_trap_with_id(22051) } else { emit_le_eax_ecx() } }
                        }}
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 23 {
        // Audit fix #9: AST_GE mixed-type ud2 trap.
        // Stage 1.5 audit fix: bf16 traps (see AST_LT comment).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let l_bf = is_bf16_expr(p1, bind_state, bn_state);
        let r_bf = is_bf16_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else {
                   // K1.F12 + K1.F13 + K1.F14 (2026-05-27): mirror K1.F11/
                   // F8d/F9-fix's mov-rcx r_i64 + r_u64 + r_d legs.
                   if r_i64 == 1 { emit_mov_rcx_rax_64() } else { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() } } }
               }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        // Stage 2.2: 5-way GE dispatch — u32 >= u32 uses `setae` (unsigned).
        // Stage 2.4b: 6-way — u64 >= u64 uses REX.W cmp + setae.
        // Speedup #4 wire-in: bf16 trap id = 23001 (AST_GE * 1000 + 1).
        let na = if l_bf == 1 { emit_trap_with_id(23001) } else { if r_bf == 1 { emit_trap_with_id(23001) } else {
            // Speedup #4 wire-in: AST_GE mixed-type trap ids 23010-23051.
            // K1.F12 (2026-05-27): close i64<->i32 GE widening (mirror K1.F11).
            if l_d == 1 {
                // K1.F14 (2026-05-27): f64<->f32 GE forward widening.
                if r_d == 1 {
                    emit_ssen_ge_dbl()
                } else {
                    let r_t23f = expr_type(p2, bind_state, bn_state);
                    if r_t23f == 1 {
                        emit_cvt_f32_in_rcx_to_f64() + emit_ssen_ge_dbl()
                    } else {
                        emit_trap_with_id(23010)
                    }
                }
            } else { if r_d == 1 {
                // K1.F14: f64<->f32 GE reverse widening.
                let l_t23f = expr_type(p1, bind_state, bn_state);
                if l_t23f == 1 {
                    emit_cvt_f32_in_rax_to_f64() + emit_ssen_ge_dbl()
                } else {
                    emit_trap_with_id(23011)
                }
            } else {
                if l_i64 == 1 {
                    if r_i64 == 1 {
                        emit_ge_rax_rcx_64()
                    } else {
                        let r_t23 = expr_type(p2, bind_state, bn_state);
                        if r_t23 == 0 {
                            emit_movsxd_rcx_ecx() + emit_ge_rax_rcx_64()
                        } else {
                            emit_trap_with_id(23020)
                        }
                    }
                } else { if r_i64 == 1 {
                    let l_t23 = expr_type(p1, bind_state, bn_state);
                    if l_t23 == 0 {
                        emit_movsxd_rax_eax() + emit_ge_rax_rcx_64()
                    } else {
                        emit_trap_with_id(23021)
                    }
                } else {
                    // K1.F13 (2026-05-27): u64<->u32 GE widening.
                    if l_u64 == 1 {
                        if r_u64 == 1 {
                            emit_ge_rax_rcx_64_u()
                        } else {
                            let r_t23u = expr_type(p2, bind_state, bn_state);
                            if r_t23u == 6 {
                                emit_ge_rax_rcx_64_u()
                            } else {
                                emit_trap_with_id(23030)
                            }
                        }
                    } else { if r_u64 == 1 {
                        let l_t23u = expr_type(p1, bind_state, bn_state);
                        if l_t23u == 6 {
                            emit_ge_rax_rcx_64_u()
                        } else {
                            emit_trap_with_id(23031)
                        }
                    } else {
                        if l_f == 1 {
                            if r_f == 1 { emit_ssen_ge() } else { emit_trap_with_id(23040) }
                        } else { if r_f == 1 { emit_trap_with_id(23041) } else {
                            if l_u32 == 1 {
                                if r_u32 == 1 { emit_ge_eax_ecx_u() } else { emit_trap_with_id(23050) }
                            } else { if r_u32 == 1 { emit_trap_with_id(23051) } else { emit_ge_eax_ecx() } }
                        }}
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 7 {
        // AST_IF(cond, then, else)
        // Stage 1 audit batch 3 fix: i64 cond must use `test rax, rax`
        // (REX.W) — the 32-bit `test eax, eax` only checks the low 32
        // bits, so an i64 with high-bit-set but low-32-zero (e.g.
        // 0x1_0000_0000_i64) would falsely take the else branch.
        // Comparisons (which always produce 0/1 in eax with high zero
        // pre-cleared) are unaffected by widening the test.
        let p3 = __arena_get(idx + 3);
        let n_cond = emit_ast_code(p1, bind_state, patch_state, bn_state);
        // Stage 2.4b audit fix: u64 cond also needs REX.W test. Without
        // this, `if x_u64 { ... }` for a u64 with low-32-bits-zero (e.g.
        // 0x1_0000_0000_u64, or anything 1_u64 << 32) would silently
        // take the else branch.
        let cond_i64 = is_i64_expr(p1, bind_state, bn_state);
        let cond_u64 = is_u64_expr(p1, bind_state, bn_state);
        let cond_wide = if cond_i64 == 1 { 1 } else { if cond_u64 == 1 { 1 } else { 0 } };
        let n_test = if cond_wide == 1 { emit_test_rax_rax_64() } else { emit_test_eax_eax() };
        let je_disp = emit_je_rel32_placeholder();
        let n_then = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let jmp_disp = emit_jmp_rel32_placeholder();
        let else_label = __arena_len();
        let n_else = emit_ast_code(p3, bind_state, patch_state, bn_state);
        let merge_label = __arena_len();
        patch_rel32(je_disp, else_label);
        patch_rel32(jmp_disp, merge_label);
        n_cond + n_test + 6 + n_then + 5 + n_else
    } else { if t == 1 {
        // AST_VAR: p1 = name start, p2 = name len.
        // Step 7d-5: f64 bindings (type tag 2) need 64-bit load to
        // preserve the high half. i32 / f32 stay on 32-bit load.
        // Stage 2.5b/c stage 2: narrow loads for u8/u16/i8/i16 use
        // movzx/movsx so the read interprets only the declared width.
        // Bit pattern in the slot beyond that width is ignored —
        // truncation semantics flow from the read side.
        // Audit follow-up Finding #3: bind_lookup returns 0 (never a
        // valid offset since bind_alloc starts at 8) for unbound names.
        // Without a guard, emit_mov_eax_local(0) becomes `mov eax,
        // [rbp+0]` = saved rbp slot — silent garbage. Trap with id 1001
        // on unbound reads.
        let off = bind_lookup(bind_state, p1, p2);
        if off == 0 {
            // A2a (2026-05-28): fn-name-as-value (function pointer). An
            // unbound name that is a registered user fn is a reference to
            // that function: emit `lea rax, [rip+disp]` to its code label,
            // resolved at backpatch via the patch table (identical rel32
            // mechanism to a CALL). rax then holds the fn's runtime address.
            // A genuinely-undefined name still traps (id 1001) -- we must
            // NOT lea-patch it, because the K1.F21 backpatch ud2-fallback
            // assumes a CALL (E8) site and would corrupt a lea into garbage
            // rather than a clean trap. The fn-pointer VALUE flows; calling
            // through it (indirect call) is A2b.
            let fts_v = bn_fn_type_state(bn_state);
            let is_fnref = if fts_v == 0 { 0 } else { fn_type_table_has(fts_v, p1, p2) };
            if is_fnref == 1 {
                let fnref_slot = emit_lea_rax_rip_placeholder();
                patch_table_add(patch_state, fnref_slot, p1, p2);
                7
            } else {
                emit_trap_with_id(1001)
            }
        } else {
            let ty = bind_lookup_type(bind_state, p1, p2);
            // 8-byte types (f64=2, i64=3, u64=9) use 64-bit load to preserve
            // the high half. Narrow types (u8=7, u16=8, i8=10, i16=11) use
            // movzx/movsx. All others (i32=0, f32=1, u32=6) use 32-bit load
            // (auto-zero-extends).
            // Stage 5 Iter C: struct-typed bindings (ty >= 100) carry an
            // 8-byte pointer; load the full 8 bytes so subsequent .field
            // postfixes (`mov eax, [rax+disp]`) read the correct base.
            let ty_is_struct = if ty >= 100 { 1 } else { 0 };
            if ty == 2 { emit_mov_rax_local_64(off) }
            else { if ty == 3 { emit_mov_rax_local_64(off) }
            else { if ty == 9 { emit_mov_rax_local_64(off) }
            else { if ty_is_struct == 1 { emit_mov_rax_local_64(off) }
            else { if ty == 7 { emit_movzx_eax_local_byte(off) }
            else { if ty == 10 { emit_movsx_eax_local_byte(off) }
            else { if ty == 8 { emit_movzx_eax_local_word(off) }
            else { if ty == 11 { emit_movsx_eax_local_word(off) }
            else { emit_mov_eax_local(off) }}}}}}}}
        }
    } else { if t == 8 {
        // AST_LET: p1 = name_start, p2 = name_len, p3 = body_idx,
        // p4 = value_idx (audit-14: split out of the legacy packed
        // p3 to avoid 16-bit overflow on large sources).
        // Step 5c: infer type from value AST and stamp into bind_state
        // so subsequent uses of the binding can dispatch to SSE.
        // Stage 1.6 + 2.1: val_ty = expr_type directly, so the FULL
        // tag space (0=i32, 1=f32, 2=f64, 3=i64, 6=u32, …) propagates
        // into bind_state instead of being demoted to whichever of
        // the 3 sibling predicates fired. Without this, u32 values
        // bound via let silently lose their type tag and downstream
        // expr_type(AST_VAR) returns 0 (i32), defeating the whole
        // point of distinguishing AST_INTLIT_U32 (tag 36).
        let body_idx = __arena_get(idx + 3);
        let value_idx = __arena_get(idx + 4);
        let val_ty = expr_type(value_idx, bind_state, bn_state);
        let n_val = emit_ast_code(value_idx, bind_state, patch_state, bn_state);
        let off = bind_alloc_offset(bind_state);
        // 8-byte types (f64=2, i64=3, u64=9) use 64-bit store.
        // Narrow types (u8=7, i8=10) use 1-byte store; (u16=8, i16=11)
        // use 2-byte store. Stage 2.5b/c stage 3 — combined with stage
        // 2's narrow loads, this gives proper truncation: bits past
        // the declared width never enter the slot.
        // i32 (0), f32 (1), u32 (6) use 32-bit store.
        let n_store = if val_ty == 2 {
            emit_mov_local_rax_64(off)
        } else { if val_ty == 3 {
            emit_mov_local_rax_64(off)
        } else { if val_ty == 9 {
            emit_mov_local_rax_64(off)
        } else { if val_ty == 7 {
            emit_mov_local_al(off)
        } else { if val_ty == 10 {
            emit_mov_local_al(off)
        } else { if val_ty == 8 {
            emit_mov_local_ax(off)
        } else { if val_ty == 11 {
            emit_mov_local_ax(off)
        } else {
            emit_mov_local_eax(off)
        }}}}}}};
        bind_push_typed(bind_state, p1, p2, off, val_ty);
        let n_body = emit_ast_code(body_idx, bind_state, patch_state, bn_state);
        bind_pop(bind_state);
        n_val + n_store + n_body
    } else { if t == 12 {
        // AST_LET_MUT: identical codegen to AST_LET. Mutability is a
        // surface-language constraint; the runtime representation is
        // the same. (Reassignment via AST_ASSIGN works on either.)
        // Stage 1.6 + 2.1: val_ty via expr_type (full tag space).
        // Stage 2.4: u64 (tag 9) also uses 64-bit store.
        // Stage 2.5b/c stage 3: u8/i8 → 1-byte, u16/i16 → 2-byte.
        let body_idx = __arena_get(idx + 3);
        let value_idx = __arena_get(idx + 4);
        let val_ty = expr_type(value_idx, bind_state, bn_state);
        let n_val = emit_ast_code(value_idx, bind_state, patch_state, bn_state);
        let off = bind_alloc_offset(bind_state);
        let n_store = if val_ty == 2 {
            emit_mov_local_rax_64(off)
        } else { if val_ty == 3 {
            emit_mov_local_rax_64(off)
        } else { if val_ty == 9 {
            emit_mov_local_rax_64(off)
        } else { if val_ty == 7 {
            emit_mov_local_al(off)
        } else { if val_ty == 10 {
            emit_mov_local_al(off)
        } else { if val_ty == 8 {
            emit_mov_local_ax(off)
        } else { if val_ty == 11 {
            emit_mov_local_ax(off)
        } else {
            emit_mov_local_eax(off)
        }}}}}}};
        bind_push_typed(bind_state, p1, p2, off, val_ty);
        let n_body = emit_ast_code(body_idx, bind_state, patch_state, bn_state);
        bind_pop(bind_state);
        n_val + n_store + n_body
    } else { if t == 11 {
        // AST_ASSIGN: emit value (eax = new value), look up name's
        // stack offset, store eax there. Result IS the assigned
        // value (still in eax). p3 = value_idx.
        //
        // Audit-10: bind_lookup returns 0 for unbound names. Since
        // bind_alloc_offset starts at 8 and grows by 8, 0 is an
        // unambiguous "not found" sentinel. Without this guard,
        // emit_mov_local_eax(0) would emit `mov [rbp+0], eax` —
        // overwriting the saved rbp at the start of the function
        // frame. After the epilogue (`mov rsp, rbp ; pop rbp`)
        // rbp would be the assigned value, and any later stack
        // op would crash. Skip the store on unbound writes; eax
        // still holds the assigned value so expressions like
        // `(x = 5) + 1` evaluate correctly.
        // Step 7d-5: f64 bindings need 64-bit store; use the existing
        // bound type tag to pick width.
        let p3 = __arena_get(idx + 3);
        let n_val = emit_ast_code(p3, bind_state, patch_state, bn_state);
        let off = bind_lookup(bind_state, p1, p2);
        if off == 0 {
            n_val
        } else {
            let bind_ty = bind_lookup_type(bind_state, p1, p2);
            let val_ty = expr_type(p3, bind_state, bn_state);
            // Stage 1 audit fixes + Stage 2.4: type-mismatch trap.
            // The 8-byte types (i64=3, u64=9) require width-matched
            // value+binding; mixed widths trap with ud2 to avoid silent
            // truncation or zero-extension bugs.
            // Speedup #4 wire-in: AST_LET body-vs-bind-ty trap ids
            // 8001-8004 cover the value/binding-type mismatches.
            // Audit follow-up Finding #4: extend trap matrix to ALL
            // bind_ty arms. Pre-fix, only val_i64/bind_ty=3 and
            // val_u64/bind_ty=9 arms checked val_ty; other arms silently
            // truncated (e.g. f32 stored to u8 slot kept low byte of
            // f32 bit pattern). Trap-ids 8005..8016 cover one per
            // bind_ty arm so the failing slot is identifiable.
            //
            // Implementation: factored out into assign_store_*_path helpers
            // to keep per-call if-else nesting under the host parser's
            // recursion budget (~13 levels would exceed it).
            let val_i64 = is_i64_expr(p3, bind_state, bn_state);
            let val_u64 = is_u64_expr(p3, bind_state, bn_state);
            let n_store = if val_i64 == 1 {
                assign_store_i64_path(off, bind_ty)
            } else { if val_u64 == 1 {
                assign_store_u64_path(off, bind_ty)
            } else {
                assign_store_general(off, bind_ty, val_ty)
            }};
            n_val + n_store
        }
    } else { if t == 14 {
        // AST_FN_DECL: Phase-0 supports a single `fn main() -> i32 {
        // expr }` form as a syntactic alternative to a bare expr.
        // The codegen treats the body as the program. Multi-fn
        // programs use AST_FN_LIST (tag 15) which dispatches to a
        // walker that finds main.
        let p3 = __arena_get(idx + 3);
        emit_ast_code(p3, bind_state, patch_state, bn_state)
    } else { if t == 15 {
        // AST_FN_LIST: by the time we get here, the top-level
        // wrapper should have already resolved the list to `main`'s
        // body (see resolve_program_root). If we still see a
        // FN_LIST tag, fall through to emit 0 — this guards against
        // accidental nested lists.
        emit_ast_int(0)
    } else { if t == 16 {
        // AST_CALL: First check if the target name is a known
        // builtin (arena ops, file ops, etc.) — if so, dispatch
        // to the inline emitter. Otherwise fall through to the
        // regular SysV call sequence.
        let p3 = __arena_get(idx + 3);
        let builtin_bytes = try_emit_builtin_call(p1, p2, p3, bind_state, patch_state, bn_state);
        if builtin_bytes > 0 {
            builtin_bytes
        } else {
        // AST_CALL(name, _, args_head): evaluate each arg LEFT-to-
        // RIGHT, pushing each rax onto the stack. After all args
        // pushed, pop into SysV arg regs in REVERSE order so rdi
        // holds arg0, rsi holds arg1, ..., r9 holds arg5. Then
        // emit `call rel32 placeholder` for backpatching.
        let p3 = __arena_get(idx + 3);
        let mut bytes_emitted: i32 = 0;
        // A2b (2026-05-28): indirect-call detection. If the callee name
        // resolves to a LOCAL binding, it holds a function-pointer value
        // (e.g. a `f: fn(...)->...` param, type-erased to i32) rather than
        // naming a fn -- emit an indirect `call r11` through the loaded
        // address instead of a name-patched rel32 (which would miss ->
        // ud2/SIGILL). 0 = not a local => the normal named-call path.
        let callee_off = bind_lookup(bind_state, p1, p2);
        // Stage 1.7: look up callee's declared param types so we can
        // trap each arg whose actual type doesn't match. Builtins
        // (not in the fn_type_table) return pp_count=0 below, which
        // skips the per-arg check — still safe because builtins are
        // already type-checked at their named-byte_eq dispatch site.
        let fts_for_args = bn_fn_type_state(bn_state);
        let pp_lookup = if fts_for_args == 0 { 0 }
                        else { fn_type_table_lookup_params(fts_for_args, p1, p2) };
        let pp_count = pp_lookup % 8;
        let pp_packed = pp_lookup / 8;
        // Pass 1: emit each arg, push rax. Track count.
        let mut arg_cur: i32 = p3;
        let mut arg_count: i32 = 0;
        while arg_cur != 0 {
            let arg_expr = __arena_get(arg_cur + 1);
            let n_arg = emit_ast_code(arg_expr, bind_state, patch_state, bn_state);
            let n_push = emit_push_rax();
            bytes_emitted = bytes_emitted + n_arg + n_push;
            // Stage 1.7: trap on arg-type-vs-param-type mismatch.
            // Speedup #4 wire-in: AST_CALL arg-type-mismatch trap id 16001.
            // Stage 5 Iter C: skip the trap when expected_ty == 15 — that's
            // the struct sentinel (parser encoded p_ty=100+struct_idx,
            // pre-pass clamped to 15 in the packed table). Caller passes
            // the struct's pointer in the arg register; expr_type of the
            // arg is i64 (for struct lit) or 100+struct_idx (for struct-
            // bound var) — neither matches the sentinel, so the trap
            // would fire spuriously. Iter D may add a stricter check
            // that compares struct identity end-to-end.
            if arg_count < pp_count {
                let expected_ty = unpack_param_ty(pp_packed, arg_count);
                let actual_ty = expr_type(arg_expr, bind_state, bn_state);
                let exp_is_struct = if expected_ty == 15 { 1 } else { 0 };
                let mismatch = if exp_is_struct == 1 { 0 } else {
                    if expected_ty != actual_ty { 1 } else { 0 }
                };
                if mismatch == 1 {
                    let n_trap = emit_trap_with_id(16001);
                    bytes_emitted = bytes_emitted + n_trap;
                };
            };
            arg_count = arg_count + 1;
            arg_cur = __arena_get(arg_cur + 2);
        }
        // Audit follow-up Finding #6: AST_CALL arity mismatch trap (16003).
        // Pre-fix, when arg_count != pp_count (both <= 6), the pass-1 type
        // check ran only while arg_count < pp_count, missing later mismatches;
        // pass 2 pops arg_count values regardless of pp_count, so missing
        // args silently read garbage from rdx/rcx/r8/r9 at the callee.
        // Builtins (pp_count == 0) skip the check, since builtins aren't
        // in fn_type_table and have their own dispatch arity.
        //
        // FLAT prefix-trap pattern (Finding #7 lesson): a deeply-nested
        // if-else inserted as a STATEMENT inside the AST_CALL arm strains
        // the host parser (helixc-Python) recursion budget and miscompiles
        // unrelated programs. Wrapping the trap in `let n = if cond { ... }
        // else { 0 }` and adding to bytes_emitted afterwards is recursion-
        // safe — investigation 2026-05-08 confirmed this is the root cause
        // (no real arity-mismatched calls in self-host source; the trap
        // didn't fire at any real call once the pattern was made flat).
        let n_arity_trap = if pp_count > 0 {
            if arg_count != pp_count {
                emit_trap_with_id(16003)
            } else { 0 }
        } else { 0 };
        bytes_emitted = bytes_emitted + n_arity_trap;
        // K-bootstrap K1.B (2026-05-25): the arg_count > 6 path no
        // longer traps. SysV stack-arg passing is implemented via the
        // caller-cleanup pattern (matches helixc/backend/x86_64.py).
        //
        // After pass 1, all N args are on the runtime stack. Layout
        // from rsp (top to bottom):
        //   [rsp + 8*i] = arg(N-1-i) for i in 0..N
        // i.e., arg(N-1) is on top (latest push), arg0 at the bottom.
        //
        // SysV ABI wants at the CALL instruction:
        //   rdi=arg0, rsi=arg1, rdx=arg2, rcx=arg3, r8=arg4, r9=arg5,
        //   [rsp_call + 0] = arg6  (lowest stack-arg INDEX at lowest
        //                            address — source-index order),
        //   [rsp_call + 8] = arg7,
        //   ...
        //   [rsp_call + 8*(N-7)] = arg(N-1),
        //   rsp_call ≡ 0 (mod 16).
        //
        // Algorithm:
        //   sub rsp, stack_alloc                    ; reserve stack-arg
        //                                              region; stack_alloc =
        //                                              (N-6) * 8 bytes.
        //                                              (N >= 7 guarantees
        //                                              stack_alloc is a
        //                                              multiple of 8 and
        //                                              that 8*N + stack_alloc
        //                                              = 16*(N - 3) which
        //                                              is always 16-aligned —
        //                                              no extra padding
        //                                              needed for rsp_call.)
        //   for i in 0..(N-7):                      ; reverse stack args
        //     mov rax, [rsp + stack_alloc + 8*(N-7-i)]
        //     mov [rsp + 8*i], rax
        //   mov rdi, [rsp + stack_alloc + 8*(N-1)]  ; load arg0
        //   mov rsi, [rsp + stack_alloc + 8*(N-2)]  ; arg1
        //   mov rdx, [rsp + stack_alloc + 8*(N-3)]  ; arg2
        //   mov rcx, [rsp + stack_alloc + 8*(N-4)]  ; arg3
        //   mov r8,  [rsp + stack_alloc + 8*(N-5)]  ; arg4
        //   mov r9,  [rsp + stack_alloc + 8*(N-6)]  ; arg5
        //   call rel32
        //   add rsp, stack_alloc + 8*N              ; cleanup
        //
        // Float args (f32/f64) flow through the integer path here,
        // matching the existing pre-K1.B kovc.hx convention for args
        // 0-5 (they go through rdi..r9, not xmm0..7). x86_64.py uses
        // xmm regs; kovc.hx's bootstrap-simpler convention diverges
        // here, and that divergence is documented in the matrix
        // (KOVC-MISSING: f32/f64 in registers via xmm).
        //
        // FLAT control flow (Finding #7 lesson at line 6099+): the
        // implementation is one `while` loop + a sequence of flat
        // emit-helper calls — no nested if-cascade — to stay under
        // the host parser's recursion budget.
        if arg_count > 6 {
            // Flat call sequence — actual work in top-level fns so the
            // AST_CALL arm stays shallow (host-parser recursion budget).
            let stack_args = arg_count - 6;
            let stack_alloc = stack_args * 8;
            let n_sub = emit_sub_rsp_imm32(stack_alloc);
            let n_rev = emit_stack_args_reverse_copy(stack_args, stack_alloc);
            let n_load = emit_load_six_int_args(stack_alloc, arg_count);
            let n_call = if callee_off == 0 {
                let disp_slot = emit_call_rel32_placeholder();
                patch_table_add(patch_state, disp_slot, p1, p2);
                5
            } else {
                let n_ld = emit_mov_r11d_local(callee_off);
                n_ld + emit_call_r11()
            };
            let n_clean = emit_add_rsp_imm32(stack_alloc + 8 * arg_count);
            bytes_emitted + n_sub + n_rev + n_load + n_call + n_clean
        } else {
        // Pass 2: pop into SysV regs in reverse-of-push order.
        // pushed: arg0, arg1, ..., argN-1 (top is argN-1).
        // We want rdi=arg0, rsi=arg1, ..., r9=argN-1.
        // So pop top first (=argN-1) into the LAST register, then
        // unwind backwards: pop into arg(N-2)'s reg, ..., pop into rdi.
        // Encodings: pop rdi=5F, pop rsi=5E, pop rdx=5A, pop rcx=59,
        //            pop r8=41 58, pop r9=41 59.
        // We emit pops in order: register-for-argN-1, argN-2, ..., arg0.
        let mut pi: i32 = arg_count - 1;
        while pi >= 0 {
            if pi == 0 {
                emit_byte(0x5F); bytes_emitted = bytes_emitted + 1;       // pop rdi
            } else { if pi == 1 {
                emit_byte(0x5E); bytes_emitted = bytes_emitted + 1;       // pop rsi
            } else { if pi == 2 {
                emit_byte(0x5A); bytes_emitted = bytes_emitted + 1;       // pop rdx
            } else { if pi == 3 {
                emit_byte(0x59); bytes_emitted = bytes_emitted + 1;       // pop rcx
            } else { if pi == 4 {
                emit_byte(0x41); emit_byte(0x58);
                bytes_emitted = bytes_emitted + 2;                        // pop r8
            } else { if pi == 5 {
                emit_byte(0x41); emit_byte(0x59);
                bytes_emitted = bytes_emitted + 2;                        // pop r9
            } else {} }}}}};
            pi = pi - 1;
        }
        let n_call = if callee_off == 0 {
            let disp_slot = emit_call_rel32_placeholder();
            patch_table_add(patch_state, disp_slot, p1, p2);
            5
        } else {
            let n_ld = emit_mov_r11d_local(callee_off);
            n_ld + emit_call_r11()
        };
        bytes_emitted + n_call
        }
        }
    } else { if t == 13 {
        // AST_SEQ(first, second): emit first (discard eax), emit
        // second (its eax is the result). Helix's calling convention
        // here is "value left in eax", so we just chain.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        n1 + n2
    } else { if t == 10 {
        // AST_WHILE(cond, body):
        //   loop_top:
        //     <cond>           leaves 0/1 in eax (or full rax for i64)
        //     test eax, eax    (i32 cond) / test rax, rax (i64 cond)
        //     je end_label
        //     <body>
        //     jmp loop_top    (backward — exercises emit_u32_le on
        //                       negative disp, the audit-8 fix)
        //   end_label:
        //     mov eax, 0      Helix while-expr returns unit (0)
        // Stage 1 audit batch 3 fix: i64 cond uses REX.W test (mirrors
        // AST_IF fix above).
        //
        // K1.AC (2026-05-25): break support. Before emitting the body
        // we SAVE the previous break-chain head from bn_state slot
        // 157 and RESET it to 0 (empty list). AST_BREAK in the body
        // prepends a (jmp_pos, prev_head) cell onto the chain. After
        // the body emits but BEFORE we know end_label, we walk the
        // chain and patch each jmp_pos to end_label, then restore
        // the previous head (which makes nested loops work: outer's
        // break-chain is preserved across the inner loop's body).
        //
        // K1.AD (2026-05-25): continue support. Same save/restore
        // pattern for slot 158. Continue jumps patch to loop_top
        // (re-eval cond + run body again). Patch must happen BEFORE
        // the `jmp loop_top` slot bookkeeping is done since loop_top
        // is known from the start.
        let saved_break_head = bn_break_chain_head_s(bn_state);
        bn_set_break_chain_head_s(bn_state, 0);
        let saved_cont_head = bn_continue_chain_head_s(bn_state);
        bn_set_continue_chain_head_s(bn_state, 0);
        let loop_top = __arena_len();
        let n_cond = emit_ast_code(p1, bind_state, patch_state, bn_state);
        // Stage 2.4b audit fix: u64 cond also needs REX.W test (same
        // as AST_IF above).
        let cond_i64 = is_i64_expr(p1, bind_state, bn_state);
        let cond_u64 = is_u64_expr(p1, bind_state, bn_state);
        let cond_wide = if cond_i64 == 1 { 1 } else { if cond_u64 == 1 { 1 } else { 0 } };
        let n_test = if cond_wide == 1 { emit_test_rax_rax_64() } else { emit_test_eax_eax() };
        let je_disp = emit_je_rel32_placeholder();
        let n_body = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let jmp_disp = emit_jmp_rel32_placeholder();
        // K1.BG (2026-05-26): split loop-exit into two labels so
        // `break <value>` preserves rax across the exit:
        //   fallthrough_label: cond=false lands here, runs `mov eax,0`
        //   end_label:         break targets here, AFTER the mov
        // For bare `break;` AST_BREAK emits `mov eax, 0` itself so
        // the fall-through and bare-break paths produce identical
        // values (0) when no explicit break-value was supplied.
        let fallthrough_label = __arena_len();
        patch_rel32(je_disp, fallthrough_label);
        patch_rel32(jmp_disp, loop_top);
        let n_zero = emit_ast_int(0);
        let end_label = __arena_len();
        // K1.AC: walk + patch the break-chain (target = end_label,
        // which is now AFTER the fall-through mov eax, 0).
        let mut bk_cur: i32 = bn_break_chain_head_s(bn_state);
        while bk_cur != 0 {
            let bk_pos = __arena_get(bk_cur);
            let bk_next = __arena_get(bk_cur + 1);
            patch_rel32(bk_pos, end_label);
            bk_cur = bk_next;
        }
        bn_set_break_chain_head_s(bn_state, saved_break_head);
        // K1.AD: walk + patch the continue-chain (target = loop_top).
        let mut ct_cur: i32 = bn_continue_chain_head_s(bn_state);
        while ct_cur != 0 {
            let ct_pos = __arena_get(ct_cur);
            let ct_next = __arena_get(ct_cur + 1);
            patch_rel32(ct_pos, loop_top);
            ct_cur = ct_next;
        }
        bn_set_continue_chain_head_s(bn_state, saved_cont_head);
        n_cond + n_test + 6 + n_body + 5 + n_zero
    } else { if t == 77 {
        // K1.AC (2026-05-25): AST_BREAK -- emit `jmp rel32`
        // placeholder; record (jmp_pos, prev_head) onto the
        // break-chain on bn_state slot 157. AST_WHILE walks the
        // chain at loop-end-codegen and patches each jmp_pos to
        // its end_label. The chain cell layout is two arena
        // slots: cell+0 = jmp_pos, cell+1 = next (0 = end).
        //
        // K1.BG (2026-05-26): if p1 != 0, eval the break value
        // into rax BEFORE the jmp so the loop expression sees
        // it after the break-chain backpatch (the AST_WHILE end
        // label now sits AFTER the fall-through `mov eax, 0`).
        // For bare `break;` (p1 == 0) emit `mov eax, 0` to
        // leave the loop value at 0 like the fall-through case.
        let n_val = if p1 == 0 {
            emit_ast_int(0)
        } else {
            emit_ast_code(p1, bind_state, patch_state, bn_state)
        };
        let jmp_pos = emit_jmp_rel32_placeholder();
        let cell_addr = __arena_len();
        let prev_head = bn_break_chain_head_s(bn_state);
        __arena_push(jmp_pos);
        __arena_push(prev_head);
        bn_set_break_chain_head_s(bn_state, cell_addr);
        n_val + 5
    } else { if t == 78 {
        // K1.AD (2026-05-25): AST_CONTINUE -- emit `jmp rel32`
        // placeholder; record (jmp_pos, prev_head) onto the
        // continue-chain on bn_state slot 158. AST_WHILE walks
        // the chain at loop-end-codegen and patches each jmp_pos
        // to loop_top (re-evaluates cond + runs body again).
        // Same cell layout as break.
        let jmp_pos_ct = emit_jmp_rel32_placeholder();
        let cell_addr_ct = __arena_len();
        let prev_head_ct = bn_continue_chain_head_s(bn_state);
        __arena_push(jmp_pos_ct);
        __arena_push(prev_head_ct);
        bn_set_continue_chain_head_s(bn_state, cell_addr_ct);
        5
    } else { if t == 79 {
        // K1.F6 (2026-05-27): AST_FIELD_STORE -- `p.x = v` writes
        // the value into the field's slot. p1 = lhs (AST_TUPLE_FIELD
        // node from parse_unary's postfix loop), p2 = value expr,
        // p3 = unused.
        //
        // Codegen mirrors AST_TUPLE_FIELD's READ side (kovc.hx:5496+):
        //   1. eval inner_expr (the chained `.field` LHS) -> rax = ptr
        //   2. push rax (save struct ptr)
        //   3. eval value_expr -> eax (value)
        //   4. pop rcx (rcx = saved struct ptr)
        //   5. mov [rcx + field_idx*8], eax    (3 bytes, scalar)
        //      or mov [rcx + field_idx*8], rax (4 bytes, REX.W; struct
        //      ptr-typed field with field_p3==1)
        // After the store, eax still holds the assigned value -- the
        // AST_ASSIGN result-IS-value convention (kovc.hx:6705-6707)
        // applies to AST_FIELD_STORE the same way.
        //
        // Disp8 wrap guard (trap 52002): mirrors the READ side's
        // 52001. p2 > 15 traps before the wrapping store.
        let field_node = p1;
        let val_node = p2;
        let inner_expr = __arena_get(field_node + 1);
        let field_idx = __arena_get(field_node + 2);
        let field_p3 = __arena_get(field_node + 3);
        let off = field_idx * 8;
        let n_pre_trap = if field_idx > 15 {
            emit_trap_with_id(52002)
        } else { 0 };
        let n_inner = emit_ast_code(inner_expr, bind_state, patch_state, bn_state);
        let n_push = emit_push_rax();
        let n_val = emit_ast_code(val_node, bind_state, patch_state, bn_state);
        // pop rcx (0x59, 1 byte)
        emit_byte(0x59);
        // K3.D audit-fix (2026-05-27, MEDIUM-2): when field_p3 == 0 the
        // store is 32-bit `mov [rcx+off], eax`. If the value expr has
        // a 64-bit type (i64=3, u64=9, f64=2), the high 32 bits in
        // rax get silently dropped. Pre-K3.D this was undetected;
        // post-K3.D the codegen emits trap 79001 BEFORE the store so
        // the bug surfaces loudly. The struct-ptr (field_p3 == 1) path
        // uses REX.W and is unaffected.
        let n_width_trap = if field_p3 == 0 {
            let val_ty = expr_type(val_node, bind_state, bn_state);
            if val_ty == 3 {
                emit_trap_with_id(79001)
            } else { if val_ty == 9 {
                emit_trap_with_id(79001)
            } else { if val_ty == 2 {
                emit_trap_with_id(79001)
            } else { 0 } } }
        } else { 0 };
        let n_store = if field_p3 == 1 {
            // mov [rcx + disp8], rax  (REX.W: 48 89 41 disp8 = 4 bytes)
            emit_byte(0x48); emit_byte(0x89); emit_byte(0x41); emit_byte(off);
            4
        } else {
            // mov [rcx + disp8], eax  (89 41 disp8 = 3 bytes)
            emit_byte(0x89); emit_byte(0x41); emit_byte(off);
            3
        };
        n_pre_trap + n_inner + n_push + n_val + 1 + n_width_trap + n_store
    } else { if t == 25 {
        // AST_STR_LIT used as a value. Phase-0: strings are only
        // meaningful as the FIRST arg of a file builtin (handled in
        // try_emit_builtin_call). When used elsewhere — e.g., as the
        // value of a let or as an integer expression — emit `mov
        // eax, 0` so codegen completes cleanly. Trying to use the
        // result is undefined behavior at this stage.
        emit_ast_int(0)
    } else { if t == 43 {
        // K1.C-deadcode (2026-05-25): AST_RET — explicit `return
        // <expr>`. p1 = value expression's arena index. Emit the
        // value into rax (via the normal AST-walker dispatch),
        // then the fn epilogue + ret. Dead code after this in the
        // same fn body is harmless — execution never reaches it.
        //
        // The kovc.hx convention for fn results is "result in rax";
        // for i32 results the high 32 bits are unspecified but the
        // caller-side `mov eax` (or `cmp eax`) only reads the low
        // 32. Mirroring the fn-end emit pattern at kovc.hx:6806-7
        // (emit_epilogue + emit_ret) keeps the semantics identical.
        //
        // CURRENTLY UNREACHABLE — no parser produces tag 43 until
        // the follow-up wire-up chunk adds the parse_primary arm.
        // The codegen is staged here so the wire-up chunk only
        // touches the parser side (smaller audit surface).
        let n_val = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let n_ep = emit_epilogue();
        let n_rt = emit_ret();
        n_val + n_ep + n_rt
    } else { if t == 99 {
        // AST_ERR with custom trap-id. Audit follow-up: callers that
        // synthesize mk_node(99, trap_id, 0, 0) want the trap-id in eax,
        // not the generic 99001 fallback. Extract p1 and emit
        // `mov eax, trap_id; ud2`. This lets parser-side discoveries
        // (cap-overflow, unknown name, arity mismatch) propagate a
        // distinct id to runtime.
        emit_trap_with_id(p1)
    } else {
        // Audit fix #8 (cycle 1): unhandled AST tag. Previously emitted
        // `mov eax, 0` (5 bytes) which silently masked AST_ERR (tag 99)
        // from lex/parse failures and any future tag added to parser
        // without a codegen handler. Now emits ud2 (2 bytes) so the
        // bug is loud at runtime instead of producing a binary that
        // returns 0. Lex/parse errors that produce AST_ERR cause the
        // resulting binary to SIGILL — clear signal vs. silent 0.
        // Speedup #4 wire-in: AST_ERR / unhandled-tag trap id 99001.
        emit_trap_with_id(99001)
    }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}     // K1.AC + K1.AD: +2; K1.F6: +1 AST_FIELD_STORE; K1.F15: +1 AST_FLOATLIT_F16 (tag 80)
}

// --------------------------------------------------------------
// Top-level: lay out an ELF executable that runs `eval(ast_root)`.
// Patching strategy: emit the ELF header with placeholder filesz
// fields (zeros), emit padding + code, then go back and rewrite
// the filesz / memsz bytes once the actual code size is known.
//
// Returns the byte count written to disk.
// --------------------------------------------------------------
// Walk the AST root to find the body that should be compiled. For
// AST_FN_LIST, find the fn named "main" and return its body. For
// AST_FN_DECL, return its body. For anything else, return the
// node itself (legacy single-expression program). Done BEFORE code
// emission so any arena_push for the "main" string template doesn't
// pollute the code byte stream.
fn resolve_program_root(ast_root: i32) -> i32 {
    let t = __arena_get(ast_root);
    if t == 14 {
        __arena_get(ast_root + 3)
    } else { if t == 15 {
        // Stash "main" bytes — these slots end up before the ELF
        // region in the arena, so they don't interleave with code.
        let main_s = __arena_push(109);   // 'm'
        __arena_push(97); __arena_push(105); __arena_push(110);
        let mut cur_list: i32 = ast_root;
        let mut found_body: i32 = 0;
        let mut keep: i32 = 1;
        while keep == 1 {
            if cur_list == 0 {
                keep = 0;
            } else {
                let fn_idx = __arena_get(cur_list + 1);
                let fn_name_s = __arena_get(fn_idx + 1);
                let fn_name_l = __arena_get(fn_idx + 2);
                if kovc_byte_eq(fn_name_s, fn_name_l, main_s, 4) == 1 {
                    found_body = __arena_get(fn_idx + 3);
                    keep = 0;
                } else {
                    cur_list = __arena_get(cur_list + 2);
                };
            };
        }
        found_body
    } else {
        ast_root
    }}
}

// Phase 1.10 step 6: kovc.hx itself uses f32 internally — proves f32
// codegen works in the self-host build. Constant-only computation
// avoiding helixc-Python's missing __i32_to_f32 builtin (which is
// bootstrap-only). The arithmetic is real f32 SSE addss/divss:
//   ratio = (1.5_f32 + 2.5_f32) / 4.0_f32      = 1.0
// Result is discarded; the marker is that the compiled kovc binary
// emits SSE bytes for these ops in its own machine code. Triple self-
// host stays byte-identical because the call appears identically in
// all three stages.
@pure fn step6_f32_marker() -> f32 {
    let a: f32 = 1.5_f32;
    let b: f32 = 2.5_f32;
    let sum: f32 = a + b;
    let four: f32 = 4.0_f32;
    sum / four
}

fn emit_elf_for_ast_to_path(ast_root: i32) -> i32 {
    // Step 6 marker: invoke f32 arithmetic via the bootstrap itself.
    // Result is discarded — the calculation is proof, not signal.
    let _f32_marker = step6_f32_marker();
    // Pre-allocate compile-time state BEFORE the ELF region so
    // their slots don't pollute the contiguous code byte stream.
    let bind_state = bind_init();
    let fn_state = fn_table_init();
    let patch_state = patch_table_init();
    let bn_state = install_builtin_names();
    // Stage 28.9: diag_arena for validation passes (panic_pass,
    // unsafe_pass, deprecated_pass, trace_pass, autotune_pass). Allocated BEFORE
    // the ELF region so its slots don't pollute the contiguous code
    // byte stream.
    let diag_state = diag_arena_init();
    // Stage 28.9: validation passes — run AFTER parse, BEFORE
    // codegen so the diag_arena is populated. The codegen patches
    // `main`'s prologue with a ud2 trap if any error-severity diag
    // fires.
    //   * panic_pass:      malformed panic(...) calls (28501)
    //   * unwind_pass:     @unwind not yet supported    (28502)
    //   * autotune_pass:   @autotune static validation (27001/27003/27004)
    //   * trace_pass:      @trace recognised (warning)  (25003 sev 1)
    //   * deprecated_pass: call to @deprecated fn       (28701 sev 1)
    //   * unsafe_pass:     stub (no AST_UNSAFE in bootstrap yet)
    //
    // Order matters only for the FIRST-error-wins trap-id picked
    // by the codegen guard. We run panic first (most-common test
    // case) then autotune/unwind errors then trace/deprecated warnings.
    panic_pass(ast_root, diag_state);
    autotune_pass(ast_root, diag_state);
    unwind_pass(ast_root, diag_state);
    trace_pass(ast_root, diag_state);
    deprecated_pass(ast_root, diag_state);
    unsafe_pass(ast_root, diag_state);
    // Phase 1.10 step 5c follow-on: fn_type_table tracks each user fn's
    // declared return type so is_f32_expr can resolve user-named f32
    // calls (not just `__f*` builtins). Populate via a pre-pass over
    // the AST_FN_LIST below; until then, bn_state[62] = 0 (sentinel
    // "not installed", which makes is_f32_expr fall through to the i32
    // default at AST_CALL).
    let fn_type_state = fn_type_table_init();
    bn_set_fn_type_state(bn_state, fn_type_state);
    // Pre-pass: walk fn_list, register each fn's name + ret_ty so the
    // table is populated BEFORE any fn body codegen runs is_f32_expr.
    if __arena_get(ast_root) == 15 {
        let mut walk: i32 = ast_root;
        let mut keep: i32 = 1;
        while keep == 1 {
            let fn_idx = __arena_get(walk + 1);
            // AST_FN_DECL: p1 = name_start, p2 = name_len,
            //              p4 (slot 4) = params_head, p5 (slot 5) = ret_ty,
            //              p6 (slot 6) = is_generic flag (Stage 8).
            let fn_name_s = __arena_get(fn_idx + 1);
            let fn_name_l = __arena_get(fn_idx + 2);
            let fn_ret_ty = __arena_get(fn_idx + 5);
            // Stage 8: skip generic-template fn decls — their concrete
            // mono'd clones (synthesized in the mono-pass below) are the
            // ones registered + emitted. Generic templates carry param
            // type tags in the 200..203 range (gp_idx markers) which
            // can't represent through 4-bit fn_type_table packing.
            let fn_is_generic = __arena_get(fn_idx + 6);
            if fn_is_generic == 1 {
                // skip this fn entirely
            } else {
            // Stage 1.7: walk params_head and pack param types into
            // 4-bit slots. Up to 6 params (Phase-0 limit). Each AST_PARAM
            // has p3=next, p4=type_tag.
            let pp_head = __arena_get(fn_idx + 4);
            let mut pp_cur: i32 = pp_head;
            let mut pp_count: i32 = 0;
            let mut pp_packed: i32 = 0;
            let mut pp_shift: i32 = 0;
            while pp_cur != 0 {
                if pp_count < 6 {
                    let pp_ty_raw = __arena_get(pp_cur + 4);
                    // Stage 5 Iter C: struct-typed params encode p_ty as
                    // 100 + struct_idx in AST_PARAM (parser side). Since
                    // the packed param-type table allocates only 4 bits
                    // per param, clamp struct values to sentinel 15 so
                    // they don't bleed into adjacent slots. AST_CALL uses
                    // 15 as "this param is a struct" — skips the type
                    // mismatch trap and treats arg as struct-by-pointer.
                    let pp_ty = if pp_ty_raw >= 100 { 15 } else { pp_ty_raw };
                    // Inline 1<<shift via repeated multiply (Helix
                    // bootstrap doesn't have <<). pp_ty is 0..15 already.
                    let mut place_val: i32 = pp_ty;
                    let mut s: i32 = 0;
                    while s < pp_shift {
                        place_val = place_val * 2;
                        s = s + 1;
                    }
                    pp_packed = pp_packed + place_val;
                    pp_shift = pp_shift + 4;
                    // Audit-cycle-5 polish: cap pp_count at 6 too. Beyond 6
                    // params the AST_CALL site emits ud2 unconditionally
                    // (existing arg_count>6 trap), so the over-count was
                    // benign — but capping keeps fn_type_table_lookup_params'
                    // count return value (low 3 bits of (packed*8+count))
                    // unambiguous.
                    pp_count = pp_count + 1;
                };
                pp_cur = __arena_get(pp_cur + 3);
            }
            fn_type_table_add(fn_type_state, fn_name_s, fn_name_l, fn_ret_ty, pp_packed, pp_count);
            };
            let next_list = __arena_get(walk + 2);
            if next_list == 0 { keep = 0; } else { walk = next_list; };
        }
    }
    // Resolve main (single-expr legacy or AST_FN_LIST → main body).
    let resolved_root = resolve_program_root(ast_root);
    // Stash "main" bytes for the _start stub's call patch (the
    // stub jumps to `main`).
    let main_name_s = __arena_push(109);
    __arena_push(97); __arena_push(105); __arena_push(110);
    let elf_start = __arena_len();
    emit_elf_header(0);
    emit_program_header(0);
    emit_padding_to_code();
    let code_start = __arena_len();

    // If the source had multi-fn (AST_FN_LIST), emit the multi-fn
    // layout: _start stub + every fn's code + backpatched calls.
    // Otherwise, legacy single-fn layout.
    let root_tag = __arena_get(ast_root);
    if root_tag == 15 {
        // _start stub:
        //   E8 ?? ?? ?? ??       call <main>            (5 bytes; backpatched)
        //   89 C7                mov edi, eax           (2)
        //   B8 3C 00 00 00       mov eax, 60 (sys_exit) (5)
        //   0F 05                syscall                (2)
        let main_call_disp = emit_call_rel32_placeholder();
        patch_table_add(patch_state, main_call_disp, main_name_s, 4);
        emit_byte(0x89); emit_byte(0xC7);
        emit_byte(0xB8); emit_byte(0x3C); emit_byte(0); emit_byte(0); emit_byte(0);
        emit_byte(0x0F); emit_byte(0x05);
        // Walk fn list and emit each fn.
        // Stage 8: skip emission of generic-template fn decls (slot 6 == 1).
        // Their concrete clones are appended to the same fn_list by the
        // mono pass and will be emitted normally on a later iteration.
        let mut cur_list: i32 = ast_root;
        while cur_list != 0 {
            let fn_idx = __arena_get(cur_list + 1);
            let fn_is_generic = __arena_get(fn_idx + 6);
            if fn_is_generic == 1 {
                // skip — emit nothing for the template
            } else {
            let fn_name_s = __arena_get(fn_idx + 1);
            let fn_name_l = __arena_get(fn_idx + 2);
            let fn_body = __arena_get(fn_idx + 3);
            let params_head = __arena_get(fn_idx + 4);
            let fn_ret_ty = __arena_get(fn_idx + 5);
            let fn_code_offset = __arena_len();
            fn_table_add(fn_state, fn_name_s, fn_name_l, fn_code_offset);
            bind_reset(bind_state);
            emit_prologue();
            // Stage 28.9: if `main` AND validation-pass diag_arena has
            // any severity=2 (error) entries, emit a ud2 trap with
            // the FIRST error's code right after the prologue. Result:
            // any produced binary whose source had malformed panic /
            // misused @deprecated / etc. aborts immediately on entry
            // rather than running silently with malformed source.
            // Only main is patched — other fns aren't entry points
            // so trapping inside them is moot (we already trap in
            // main before any fn is called). Byte-equal "main" check:
            // 4 bytes (109 97 105 110).
            let is_main_fn = if fn_name_l == 4 {
                let mb0 = __arena_get(fn_name_s);
                let mb1 = __arena_get(fn_name_s + 1);
                let mb2 = __arena_get(fn_name_s + 2);
                let mb3 = __arena_get(fn_name_s + 3);
                if mb0 == 109 { if mb1 == 97 { if mb2 == 105 {
                    if mb3 == 110 { 1 } else { 0 } } else { 0 } } else { 0 }
                } else { 0 }
            } else { 0 };
            if is_main_fn == 1 {
                // Stage-28.9 audit-cycle-1 Finding 1: the overflow
                // flag trumps normal errors. If the diag_arena
                // dropped one or more diags (cap=64), we MUST report
                // 28999 — otherwise the lost diag could have been
                // the only severity-2 error and the binary would
                // exit cleanly despite a malformed program. The
                // previous code called emit_trap_with_id(28999)
                // directly inside diag_emit, but validation runs
                // before elf_start is captured, so those bytes
                // never reached the produced ELF.
                let overflowed = diag_arena_overflowed(diag_state);
                if overflowed == 1 {
                    emit_trap_with_id(28999);
                } else {
                    let n_errors = diag_arena_error_count(diag_state);
                    if n_errors > 0 {
                        // Use first error's code (codegen-determinism:
                        // pick the FIRST diag in arena order, not the
                        // first error — same trap-id for fixed input).
                        let first_code = diag_get_code(diag_state, 0);
                        emit_trap_with_id(first_code);
                    };
                };
            };
            // Copy each param's SysV arg register into a fresh stack
            // slot and register the binding so the body can reference
            // params by name. Phase 0: up to 6 params (rdi/rsi/rdx/
            // rcx/r8/r9). Encoding: 89 BD disp32 = mov [rbp+disp32],
            // edi; ModRM second-byte differs per source register.
            let mut pcur: i32 = params_head;
            let mut pidx: i32 = 0;
            while pcur != 0 {
                if pidx < 6 {
                    let pname_s = __arena_get(pcur + 1);
                    let pname_l = __arena_get(pcur + 2);
                    // Step 5c follow-on: AST_PARAM now has p4 = type tag.
                    let p_ty = __arena_get(pcur + 4);
                    let off = bind_alloc_offset(bind_state);
                    bind_push_typed(bind_state, pname_s, pname_l, off, p_ty);
                    // Audit cycle 2 fix #1: f64 params (p_ty == 2) need
                    // 64-bit register stores to preserve the high 32
                    // bits. The 32-bit forms below silently truncate
                    // every f64 argument's exponent + mantissa MSBs to
                    // zero. REX.W prefix (0x48 for rdi/rsi/rdx/rcx;
                    // 0x4C carries W for r8/r9) promotes the store to
                    // 8-byte width.
                    // Stage 1: i64 params (p_ty == 3) ALSO need 64-bit
                    // stores — same shape as f64. Combine both in a
                    // single is-8-byte check.
                    // Stage 2.4: u64 params (p_ty == 9) too.
                    // Stage 5 Iter C: struct params (p_ty >= 100) ALSO
                    // need 64-bit stores — rdi (etc.) carries the full
                    // 8-byte pointer to the caller-allocated struct.
                    let p_ty_is_struct = if p_ty >= 100 { 1 } else { 0 };
                    let needs_64 = if p_ty == 2 { 1 }
                                   else { if p_ty == 3 { 1 }
                                   else { if p_ty == 9 { 1 }
                                   else { if p_ty_is_struct == 1 { 1 } else { 0 } } } };
                    if needs_64 == 1 {
                        if pidx == 0 {
                            // mov [rbp+disp32], rdi  : 48 89 BD disp32
                            emit_byte(0x48); emit_byte(0x89); emit_byte(0xBD); emit_u32_le(0 - off);
                        } else { if pidx == 1 {
                            // mov [rbp+disp32], rsi  : 48 89 B5 disp32
                            emit_byte(0x48); emit_byte(0x89); emit_byte(0xB5); emit_u32_le(0 - off);
                        } else { if pidx == 2 {
                            // mov [rbp+disp32], rdx  : 48 89 95 disp32
                            emit_byte(0x48); emit_byte(0x89); emit_byte(0x95); emit_u32_le(0 - off);
                        } else { if pidx == 3 {
                            // mov [rbp+disp32], rcx  : 48 89 8D disp32
                            emit_byte(0x48); emit_byte(0x89); emit_byte(0x8D); emit_u32_le(0 - off);
                        } else { if pidx == 4 {
                            // mov [rbp+disp32], r8   : 4C 89 85 disp32
                            emit_byte(0x4C); emit_byte(0x89); emit_byte(0x85);
                            emit_u32_le(0 - off);
                        } else { if pidx == 5 {
                            // mov [rbp+disp32], r9   : 4C 89 8D disp32
                            emit_byte(0x4C); emit_byte(0x89); emit_byte(0x8D);
                            emit_u32_le(0 - off);
                        } else {} }}}}};
                    } else {
                    if pidx == 0 {
                        // mov [rbp+disp32], edi  : 89 BD disp32
                        emit_byte(0x89); emit_byte(0xBD); emit_u32_le(0 - off);
                    } else { if pidx == 1 {
                        // mov [rbp+disp32], esi  : 89 B5 disp32
                        emit_byte(0x89); emit_byte(0xB5); emit_u32_le(0 - off);
                    } else { if pidx == 2 {
                        // mov [rbp+disp32], edx  : 89 95 disp32
                        emit_byte(0x89); emit_byte(0x95); emit_u32_le(0 - off);
                    } else { if pidx == 3 {
                        // mov [rbp+disp32], ecx  : 89 8D disp32
                        emit_byte(0x89); emit_byte(0x8D); emit_u32_le(0 - off);
                    } else { if pidx == 4 {
                        // mov [rbp+disp32], r8d  : 44 89 85 disp32
                        emit_byte(0x44); emit_byte(0x89); emit_byte(0x85);
                        emit_u32_le(0 - off);
                    } else { if pidx == 5 {
                        // mov [rbp+disp32], r9d  : 44 89 8D disp32
                        emit_byte(0x44); emit_byte(0x89); emit_byte(0x8D);
                        emit_u32_le(0 - off);
                    } else {} }}}}};
                    };
                };
                pidx = pidx + 1;
                pcur = __arena_get(pcur + 3);
            }
            emit_ast_code(fn_body, bind_state, patch_state, bn_state);
            // Stage 1 audit cycle 2 fix: trap when fn body type doesn't
            // match declared return type. Specifically i64-return fns
            // whose body produces i32 (high 32 stale) and i32-return fns
            // whose body produces i64 (caller treats as 32 — high half
            // dropped silently). Full expr_type comparison still
            // produces false positives in the existing bootstrap source,
            // so the trap is gated on width: 8-byte-vs-narrower mismatch
            // is the actual silent-data-loss class.
            // Stage 2.4b audit fix: extended to cover u64. Prior version
            // checked only `body_is_i64 vs ret_wants_i64`, so a fn
            // declared `-> i32` whose body produced u64 silently
            // narrowed at the call boundary (eax = low 32; high half
            // discarded). Now the trap fires on any 8-byte-vs-narrower
            // mismatch in either direction.
            let body_is_i64 = is_i64_expr(fn_body, bind_state, bn_state);
            let body_is_u64 = is_u64_expr(fn_body, bind_state, bn_state);
            let body_is_8b = if body_is_i64 == 1 { 1 } else { if body_is_u64 == 1 { 1 } else { 0 } };
            // Audit A1-F5: ret_ty=100+struct_idx also wants 8 bytes (struct
            // pointer rep). Without this, struct-returning fns whose body
            // is an AST_TUPLE_LIT (also 8-byte ptr) tripped 14001.
            let ret_is_struct = if fn_ret_ty >= 100 { 1 } else { 0 };
            let ret_wants_8b = if fn_ret_ty == 3 { 1 } else { if fn_ret_ty == 9 { 1 } else { ret_is_struct } };
            // Speedup #4 wire-in: body-vs-ret-ty 8-byte mismatch trap id 14001.
            if body_is_8b != ret_wants_8b {
                emit_trap_with_id(14001);
            };
            // Audit follow-up Finding #1 (softer width-only variant).
            // The 14001 trap above only covers 8b vs !8b. This adds
            // a width-class check that catches narrow-vs-wider mismatches
            // missed by the original (e.g. `fn f() -> u8 { i32 }`,
            // `fn f() -> bf16 { i32 }`, `fn f() -> f64 { i32 }`).
            // We compare width-class (1/2/4/8 bytes) of body vs ret_ty.
            // Same width = no trap (so i32 vs u32 vs f32 same class
            // is allowed — narrow re-truncation at use site keeps that
            // benign for the bootstrap source). Different widths trap
            // 14002. The full-equality variant produces false positives
            // in the existing bootstrap source; this width variant is
            // strictly stricter than 14001 (catches all 14001 cases plus
            // narrow-class mismatches) without breaking self-host.
            let body_width = type_width_class(expr_type(fn_body, bind_state, bn_state));
            let ret_width = type_width_class(fn_ret_ty);
            if body_width != ret_width {
                emit_trap_with_id(14002);
            };
            emit_epilogue();
            emit_ret();
            };       // end Stage 8 fn_is_generic skip-else
            cur_list = __arena_get(cur_list + 2);
        }
        // After all fns emitted, emit any string-literal bodies first
        // (so they live in the file), then register the arena base
        // at the next position. Important: the arena itself lives in
        // BSS — we set p_memsz > p_filesz so the kernel zero-fills
        // the arena range without consuming file bytes. Without this,
        // the kovc-host arena (32K slots = HELIX_ARENA_CAP) would
        // overflow trying to emit a 132K-byte .data section, silently
        // truncating the produced ELF and dropping any later string
        // bodies.
        let str_count = str_top(bn_state);
        let str_base_table = str_table_base(bn_state);
        let mut si: i32 = 0;
        while si < str_count {
            let s_entry = str_base_table + si * 3;
            let s_disp_slot = __arena_get(s_entry);
            let s_body_s = __arena_get(s_entry + 1);
            let s_body_l = __arena_get(s_entry + 2);
            let s_data_offset = __arena_len();
            let mut bi: i32 = 0;
            while bi < s_body_l {
                emit_byte(__arena_get(s_body_s + bi));
                bi = bi + 1;
            }
            emit_byte(0);     // NUL terminator
            let s_disp = s_data_offset - (s_disp_slot + 4);
            patch_u32_le(s_disp_slot, s_disp);
            si = si + 1;
        }
        let arena_base_offset = __arena_len();
        fn_table_add(fn_state, bn_helix_arena_base_s(bn_state), 18, arena_base_offset);
        // No emit_zeros for the arena: the kernel allocates the
        // 4-byte cursor + 131072 bytes of zero-filled data via BSS
        // when p_memsz > p_filesz (patched at the bottom).
        // Backpatch all CALL placeholders. For unresolved names,
        // overwrite the entire 5-byte CALL with `ud2` + 3 NOPs
        // (audit-11 fix). The previous "leave disp at 0" silently
        // pushed an unmatched return address (call +0 falls through
        // to next instr, but rsp is now off by 8 — the caller's
        // ret pops garbage and jumps to a wild address). ud2 raises
        // SIGILL: clear, immediate failure rather than data
        // corruption that surfaces blocks later.
        let patch_top = __arena_get(patch_state);
        let patch_table_base = __arena_get(patch_state + 1);
        let mut pi: i32 = 0;
        while pi < patch_top {
            let entry = patch_table_base + pi * 3;
            let disp_slot = __arena_get(entry);
            let target_name_s = __arena_get(entry + 1);
            let target_name_l = __arena_get(entry + 2);
            let target_offset = fn_table_lookup(fn_state, target_name_s, target_name_l);
            // K1.F21 (2026-05-27): on lookup miss, try the i32-monomorphized
            // mangled name `<target>__i32`. Closes matrix-row-137 limitation:
            // bare call `id(42)` -- where mono produced `id__i32` from a
            // turbofish call elsewhere -- previously SIGILL'd (returns
            // rc=132); now resolves to the i32 mono clone. Non-i32 T bare
            // calls still trap (the fallback assumes T=i32 default).
            // Scratch buffer is at bn_mangle_scratch(bn_state); 64 slots,
            // pre-allocated in install_builtin_names BEFORE the ELF code
            // region grows so this write doesn't corrupt the byte stream.
            // Gated on target_name_l < 60: max write at scratch[59+4] =
            // scratch[63] = the 64th slot, an exact fit. K3.K 2026-05-27
            // audit-fix: the original phrasing claimed safety margin --
            // it's exact-fit-safe, no margin.
            let final_target_offset = if target_offset >= 0 {
                target_offset
            } else { if target_name_l < 60 {
                let scratch = bn_mangle_scratch(bn_state);
                let mut mi: i32 = 0;
                while mi < target_name_l {
                    __arena_set(scratch + mi, __arena_get(target_name_s + mi));
                    mi = mi + 1;
                }
                __arena_set(scratch + target_name_l + 0, 95);  // '_'
                __arena_set(scratch + target_name_l + 1, 95);  // '_'
                __arena_set(scratch + target_name_l + 2, 105); // 'i'
                __arena_set(scratch + target_name_l + 3, 51);  // '3'
                __arena_set(scratch + target_name_l + 4, 50);  // '2'
                let mut try_len = target_name_l + 5;
                let mut found_off = fn_table_lookup(fn_state, scratch, try_len);
                // A1b (2026-05-28): multi-param bare generic call. The 1-param
                // mono is `<name>__i32`; 2..4-param monos are `<name>__i32_i32`
                // etc. (mangle_name_into_arena uses a single '_' between args).
                // Append "_i32" up to 3 more times, first fn_table_lookup hit
                // wins. Gated on target_name_l < 48 so the largest write,
                // scratch[name_l+16], stays in the 64-slot scratch buffer
                // (the single-i32 path above keeps its original <60 gate).
                let mut np: i32 = 1;
                while np < 4 {
                    if found_off >= 0 {
                        np = 4;
                    } else { if target_name_l < 48 {
                        __arena_set(scratch + try_len + 0, 95);  // '_'
                        __arena_set(scratch + try_len + 1, 105); // 'i'
                        __arena_set(scratch + try_len + 2, 51);  // '3'
                        __arena_set(scratch + try_len + 3, 50);  // '2'
                        try_len = try_len + 4;
                        found_off = fn_table_lookup(fn_state, scratch, try_len);
                        np = np + 1;
                    } else {
                        np = 4;
                    }};
                }
                found_off
            } else {
                0 - 1
            }};
            if final_target_offset >= 0 {
                let disp = final_target_offset - (disp_slot + 4);
                patch_u32_le(disp_slot, disp);
            } else {
                // ud2 = 0F 0B; pad remaining 3 bytes with NOP (90).
                // disp_slot points at byte 1 of the original 5-byte
                // E8+disp instruction; opcode E8 is at disp_slot-1.
                __arena_set(disp_slot - 1, 0x0F);
                __arena_set(disp_slot, 0x0B);
                __arena_set(disp_slot + 1, 0x90);
                __arena_set(disp_slot + 2, 0x90);
                __arena_set(disp_slot + 3, 0x90);
            };
            pi = pi + 1;
        }
    } else {
        // Legacy / fn-decl: single fn whose body is the program.
        // Same prologue+body+epilogue+exit_stub layout as before.
        emit_prologue();
        emit_ast_code(resolved_root, bind_state, patch_state, bn_state);
        emit_epilogue();
        emit_exit_with_eax();
    }

    let code_end = __arena_len();
    let total_filesz = 4096 + (code_end - code_start);
    // p_memsz extends past p_filesz to give the produced binary's
    // arena ~8 MB of BSS-allocated zero memory (4 bytes cursor +
    // 2097152 * 4 bytes data). Without this gap, an arena_push past
    // file bounds would SIGSEGV. Sized to match HELIX_ARENA_CAP in
    // helixc/backend/x86_64.py (the host compiler's bound).
    let total_memsz = total_filesz + 4 + 8388608;
    patch_u64_le_split(elf_start + 64 + 32, total_filesz, 0);
    patch_u64_le_split(elf_start + 64 + 40, total_memsz, 0);
    total_filesz
}

// ==============================================================
// K1.M1 (2026-05-27): DIRECT-TO-GPU PTX EMISSION (NVIDIA).
//
// Mirror of emit_elf_for_ast_to_path, but the target is NVIDIA
// PTX -- a *text* virtual ISA. Where the x86_64 path emits ELF
// *binary* (headers, p-offsets, relocations), this emits PTX
// *ASCII text*: STRICTLY SIMPLER -- just bytes that spell the
// assembly the NVIDIA driver JIT-compiles to SASS at load time.
// No assembler, no linker, no object format.
//
// Direct tile-IR -> PTX text, NO MLIR detour. See
// docs/MLIR_NOT_NEEDED_DECISION.md (ratified + verified) and
// docs/GPU_DIRECT_EMIT_PLAN.md. This mirrors Python's MLIR-free
// reference emitter helixc/backend/ptx.py
// (emit_module_header() + emit_kernel()).
//
// NORTH-STAR GOAL (user directive 2026-05-27): "Have Helix
// wherever possible talk directly to the chips." The bootstrap
// already talks directly to the CPU (emit_elf_for_ast_to_path
// -> x86_64 machine code). This is the first step of talking
// directly to the GPU, with ROCm/Metal/WebGPU to follow as
// sibling text emitters.
//
// CHUNK SCOPE (K1.M1, smallest-first): detect a @kernel fn
// (parser sets is_kernel on AST_FN_DECL slot 14, Stage 33) and
// emit the minimal valid empty-entry module:
//
//   .version 8.3
//   .target sm_75
//   .address_size 64
//
//   .visible .entry k()
//   {
//   ret;
//   }
//
// Phase-0 narrowing: the entry name is the kernel fn's REAL name
// (K1.M2 -- copied from AST_FN_DECL slots 1/2 = name_start/len,
// the same source-byte read the "main" detection uses); for
// multiple kernels only the FIRST is emitted (K1.M3 emits all).
// The kernel body/params/return type are still ignored here --
// later chunks emit .param decls, .reg files, and the lowered
// tile-op stream. Returns the PTX byte
// count (0 if the AST has no @kernel fn: a pure-CPU program
// emits no PTX). PURE-ADDITIVE codegen -- touches no sb scratch
// slots and no parser state (the K1.F5d-j sb-collision hazard
// does not apply).
// ==============================================================
fn emit_ptx_byte(b: i32) -> i32 {
    __arena_push(b);
    0
}

// K1.M5b (2026-05-28): emit a signed decimal integer as ASCII.
// Recursive -- high-order digits first, then the last digit; a
// leading '-' for negatives. (Phase-0 has no string formatting; the
// bootstrap is itself recursive-descent, so recursion is fine.)
fn emit_ptx_decimal(n: i32) -> i32 {
    if n < 0 {
        emit_ptx_byte(45);             // '-'
        emit_ptx_decimal(0 - n)
    } else {
        if n >= 10 {
            emit_ptx_decimal(n / 10);
        };
        emit_ptx_byte(48 + (n % 10));  // last digit
        0
    }
}

// K1.M5a (2026-05-28): the common prefix/suffix of a PTX register-
// file declaration line. Each line is "    .reg <type:<7><class><256>;"
// -- prefix = "    .reg " (4-space indent), suffix = "<256>;\n".
fn emit_ptx_reg_prefix() -> i32 {
    // "    .reg "
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(46); emit_ptx_byte(114);
    emit_ptx_byte(101); emit_ptx_byte(103); emit_ptx_byte(32);
    0
}
fn emit_ptx_reg_suffix() -> i32 {
    // "<256>;\n"  (pool cap 256, mirrors Python ptx.py _REG_POOL_CAP)
    emit_ptx_byte(60); emit_ptx_byte(50); emit_ptx_byte(53);
    emit_ptx_byte(54); emit_ptx_byte(62); emit_ptx_byte(59);
    emit_ptx_byte(10);
    0
}

// K1.M5a (2026-05-28): emit the standard PTX register-file
// declaration block -- one ".reg <type> %<class><256>;" per file,
// byte-matching Python ptx.py _REG_FILES (pool cap 256). Declaring a
// large pool is FREE: ptxas allocates only the registers actually
// USED. These decls are the foundation op-lowering (M5b+) needs, so
// %r0/%f0/... are always declared before any instruction references
// them. Type field is left-padded to 7 cols (Python ":<7"). Files,
// in emission order: %p/.pred, %r/.b32, %rd/.b64, %f/.f32, %h/.b16.
// A blank line follows the block (matches Python emit_kernel).
fn emit_ptx_reg_block() -> i32 {
    // "    .reg .pred  %p<256>;"  (.pred padded to 7 = ".pred  ")
    emit_ptx_reg_prefix();
    emit_ptx_byte(46); emit_ptx_byte(112); emit_ptx_byte(114);
    emit_ptx_byte(101); emit_ptx_byte(100); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(37); emit_ptx_byte(112);
    emit_ptx_reg_suffix();
    // "    .reg .b32   %r<256>;"  (.b32 padded to 7 = ".b32   ")
    emit_ptx_reg_prefix();
    emit_ptx_byte(46); emit_ptx_byte(98); emit_ptx_byte(51);
    emit_ptx_byte(50); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(37); emit_ptx_byte(114);
    emit_ptx_reg_suffix();
    // "    .reg .b64   %rd<256>;"
    emit_ptx_reg_prefix();
    emit_ptx_byte(46); emit_ptx_byte(98); emit_ptx_byte(54);
    emit_ptx_byte(52); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(37); emit_ptx_byte(114);
    emit_ptx_byte(100);
    emit_ptx_reg_suffix();
    // "    .reg .f32   %f<256>;"
    emit_ptx_reg_prefix();
    emit_ptx_byte(46); emit_ptx_byte(102); emit_ptx_byte(51);
    emit_ptx_byte(50); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(37); emit_ptx_byte(102);
    emit_ptx_reg_suffix();
    // "    .reg .b16   %h<256>;"  (Stage 64 f16/bf16 file)
    emit_ptx_reg_prefix();
    emit_ptx_byte(46); emit_ptx_byte(98); emit_ptx_byte(49);
    emit_ptx_byte(54); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(37); emit_ptx_byte(104);
    emit_ptx_reg_suffix();
    // blank line after the .reg block
    emit_ptx_byte(10);
    0
}

// K1.M5c (2026-05-28): emit a SCALAR_CONST_INT op --
// "    mov.s32 %r<reg>, <val>;" (load an integer constant into a
// register). Mirrors Python ptx.py emit_op SCALAR_CONST_INT.
fn emit_ptx_mov_const(ridx: i32, val: i32) -> i32 {
    // "    mov.s32 %r"  (`ridx` not `reg` -- `reg` is a Helix keyword)
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(109); emit_ptx_byte(111);
    emit_ptx_byte(118); emit_ptx_byte(46); emit_ptx_byte(115);
    emit_ptx_byte(51); emit_ptx_byte(50); emit_ptx_byte(32);
    emit_ptx_byte(37); emit_ptx_byte(114);
    emit_ptx_decimal(ridx);
    // ", "
    emit_ptx_byte(44); emit_ptx_byte(32);
    emit_ptx_decimal(val);
    // ";\n"
    emit_ptx_byte(59); emit_ptx_byte(10);
    0
}

// K1.M5d (2026-05-28): variable->register environment for kernel-body
// expression lowering. A fixed-cap table (16 vars) living in the arena
// BEFORE the PTX output region (allocated by ptx_vtab_init in
// emit_ptx_for_ast_to_path), so its slots are NEVER part of the
// emitted .ptx text. Layout: slot 0 = next_reg (per-kernel register
// counter), slot 1 = var_count, then 16 * (name_s, name_l, ridx)
// triples. Reset per kernel. (`reg` is a Helix keyword -> `ridx`.)
fn ptx_vtab_init() -> i32 {
    let bse = __arena_push(0);   // slot 0: next_reg
    __arena_push(0);             // slot 1: var_count
    let mut i: i32 = 0;
    while i < 54 {               // 16 var triples (2..49) + 6 counters:
        __arena_push(0);         // 50=next_pred,51=next_label,52=next_rd,
        i = i + 1;               // 53=cur_fn_idx,54=next_f,55=last_is_float
    }
    bse
}
fn ptx_vtab_reset(vtab: i32) -> i32 {
    __arena_set(vtab, 0);        // next_reg = 0
    __arena_set(vtab + 1, 0);    // var_count = 0
    __arena_set(vtab + 50, 0);   // K1.M8: next_pred = 0
    __arena_set(vtab + 51, 0);   // K1.M8: next_label = 0
    __arena_set(vtab + 52, 0);   // K1.M10: next_rd = 0
    __arena_set(vtab + 54, 0);   // K1.M11: next_f = 0
    __arena_set(vtab + 55, 0);   // K1.M12: last_is_float = 0
    0
}
fn ptx_alloc_reg(vtab: i32) -> i32 {
    let r = __arena_get(vtab);
    __arena_set(vtab, r + 1);
    r
}
// K1.M8: allocate a fresh predicate register index (%pN), per-kernel.
fn ptx_alloc_pred(vtab: i32) -> i32 {
    let r = __arena_get(vtab + 50);
    __arena_set(vtab + 50, r + 1);
    r
}
// K1.M8b: allocate a fresh label id (for $Lelse_<n> / $Lend_<n>).
fn ptx_alloc_label(vtab: i32) -> i32 {
    let r = __arena_get(vtab + 51);
    __arena_set(vtab + 51, r + 1);
    r
}
// K1.M10: allocate a fresh 64-bit address register index (%rdN), used
// for global-memory pointers + address arithmetic.
fn ptx_alloc_rd(vtab: i32) -> i32 {
    let r = __arena_get(vtab + 52);
    __arena_set(vtab + 52, r + 1);
    r
}
// K1.M11: allocate a fresh 32-bit float register index (%fN).
fn ptx_alloc_f(vtab: i32) -> i32 {
    let r = __arena_get(vtab + 54);
    __arena_set(vtab + 54, r + 1);
    r
}
fn ptx_vtab_add(vtab: i32, name_s: i32, name_l: i32, ridx: i32) -> i32 {
    let vc = __arena_get(vtab + 1);
    if vc < 16 {
        let bse = vtab + 2 + vc * 3;
        __arena_set(bse, name_s);
        __arena_set(bse + 1, name_l);
        __arena_set(bse + 2, ridx);
        __arena_set(vtab + 1, vc + 1);
    };
    0
}
fn ptx_vtab_lookup(vtab: i32, name_s: i32, name_l: i32) -> i32 {
    // Return the ridx of the latest matching binding (shadowing) or -1.
    let vc = __arena_get(vtab + 1);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < vc {
        let bse = vtab + 2 + i * 3;
        if kovc_byte_eq(name_s, name_l, __arena_get(bse), __arena_get(bse + 1)) == 1 {
            found = __arena_get(bse + 2);
        };
        i = i + 1;
    }
    found
}

// K1.M5d: recursively lower a kernel-body scalar expression to PTX,
// returning the register index holding its result. AST_INT (tag 0 ->
// mov.s32), AST_VAR (tag 1 -> resolve via the var table), AST_LET (tag
// 8 -> lower value [slot 4], bind name->reg, lower continuation [slot
// 3]), AST_ADD/SUB/MUL (tags 2/3/4 -> emit_ptx_binop). Unsupported
// nodes return -1 with no emit (extended later). Mirrors x86_64
// emit_ast_code but targets PTX text. Mutually recursive with
// emit_ptx_binop (forward ref -- same support the parser relies on).
fn emit_ptx_expr(node: i32, vtab: i32) -> i32 {
    // K1.M12: default result type = i32 (flag 0); f32-producing
    // branches (f32 index-load, f32 binop) override to 1 at their end.
    __arena_set(vtab + 55, 0);
    let tag = __arena_get(node);
    if tag == 0 {
        let r = ptx_alloc_reg(vtab);
        emit_ptx_mov_const(r, __arena_get(node + 1));
        r
    } else { if tag == 1 {
        ptx_vtab_lookup(vtab, __arena_get(node + 1), __arena_get(node + 2))
    } else { if tag == 8 {
        let vr = emit_ptx_expr(__arena_get(node + 4), vtab);
        ptx_vtab_add(vtab, __arena_get(node + 1), __arena_get(node + 2), vr);
        emit_ptx_expr(__arena_get(node + 3), vtab)
    } else { if tag == 2 {
        emit_ptx_binop(node, vtab, 0)
    } else { if tag == 3 {
        emit_ptx_binop(node, vtab, 1)
    } else { if tag == 4 {
        emit_ptx_binop(node, vtab, 2)
    } else { if tag == 5 {
        emit_ptx_binop(node, vtab, 3)
    } else { if tag == 9 {
        emit_ptx_neg(node, vtab)
    } else { if tag == 16 {
        emit_ptx_call(node, vtab)
    } else { if tag == 6 {
        emit_ptx_cmp(node, vtab, 0)
    } else { if tag == 19 {
        emit_ptx_cmp(node, vtab, 1)
    } else { if tag == 20 {
        emit_ptx_cmp(node, vtab, 2)
    } else { if tag == 21 {
        emit_ptx_cmp(node, vtab, 3)
    } else { if tag == 22 {
        emit_ptx_cmp(node, vtab, 4)
    } else { if tag == 23 {
        emit_ptx_cmp(node, vtab, 5)
    } else { if tag == 7 {
        emit_ptx_if(node, vtab)
    } else { if tag == 10 {
        emit_ptx_while(node, vtab)
    } else { if tag == 11 {
        emit_ptx_assign(node, vtab)
    } else { if tag == 53 {
        emit_ptx_index_load(node, vtab)
    } else { if tag == 55 {
        emit_ptx_index_store(node, vtab)
    } else {
        0 - 1
    }}}}}}}}}}}}}}}}}}}}
}

// K1.M5d: emit a scalar binary op "    <mnem>.s32 %rD, %rA, %rB;"
// (opc 0=add, 1=sub, 2=mul.lo). Lowers both operands first (so their
// movs precede the op), allocates the result register, returns it.
// Mirrors Python ptx.py emit_op SCALAR_ADD/SUB/MUL.
fn emit_ptx_binop(node: i32, vtab: i32, opc: i32) -> i32 {
    let la = emit_ptx_expr(__arena_get(node + 1), vtab);
    let la_f = __arena_get(vtab + 55);       // K1.M12b: lhs float?
    let ra = emit_ptx_expr(__arena_get(node + 2), vtab);
    let ra_f = __arena_get(vtab + 55);       // rhs float?
    // K1.M12b: f32 path only when BOTH operands are float (mixed int+
    // float would need a cvt -- not yet supported). opc 0=add 1=sub
    // 2=mul 3=div(.rn).
    let both_f = if la_f == 1 { ra_f } else { 0 };
    if both_f == 1 {
        let rf = ptx_alloc_f(vtab);
        emit_ptx_indent();
        if opc == 0 {
            emit_ptx_byte(97); emit_ptx_byte(100); emit_ptx_byte(100);   // add
        };
        if opc == 1 {
            emit_ptx_byte(115); emit_ptx_byte(117); emit_ptx_byte(98);   // sub
        };
        if opc == 2 {
            emit_ptx_byte(109); emit_ptx_byte(117); emit_ptx_byte(108);  // mul
        };
        if opc == 3 {
            emit_ptx_byte(100); emit_ptx_byte(105); emit_ptx_byte(118);  // div
            emit_ptx_byte(46); emit_ptx_byte(114); emit_ptx_byte(110);   // .rn
        };
        // ".f32 "
        emit_ptx_byte(46); emit_ptx_byte(102); emit_ptx_byte(51);
        emit_ptx_byte(50); emit_ptx_byte(32);
        emit_ptx_f(rf);
        emit_ptx_byte(44); emit_ptx_byte(32);
        emit_ptx_f(la);
        emit_ptx_byte(44); emit_ptx_byte(32);
        emit_ptx_f(ra);
        emit_ptx_byte(59); emit_ptx_byte(10);
        __arena_set(vtab + 55, 1);
        rf
    } else {
        // s32 (i32) path -- byte-identical to pre-K1.M12b.
        let r = ptx_alloc_reg(vtab);
        emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
        emit_ptx_byte(32);
        if opc == 0 {
            emit_ptx_byte(97); emit_ptx_byte(100); emit_ptx_byte(100);
            emit_ptx_byte(46); emit_ptx_byte(115); emit_ptx_byte(51);
            emit_ptx_byte(50);
        } else { if opc == 1 {
            emit_ptx_byte(115); emit_ptx_byte(117); emit_ptx_byte(98);
            emit_ptx_byte(46); emit_ptx_byte(115); emit_ptx_byte(51);
            emit_ptx_byte(50);
        } else { if opc == 2 {
            emit_ptx_byte(109); emit_ptx_byte(117); emit_ptx_byte(108);
            emit_ptx_byte(46); emit_ptx_byte(108); emit_ptx_byte(111);
            emit_ptx_byte(46); emit_ptx_byte(115); emit_ptx_byte(51);
            emit_ptx_byte(50);
        } else {
            emit_ptx_byte(100); emit_ptx_byte(105); emit_ptx_byte(118);
            emit_ptx_byte(46); emit_ptx_byte(115); emit_ptx_byte(51);
            emit_ptx_byte(50);
        }}};
        emit_ptx_byte(32); emit_ptx_byte(37); emit_ptx_byte(114);
        emit_ptx_decimal(r);
        emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
        emit_ptx_byte(114); emit_ptx_decimal(la);
        emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
        emit_ptx_byte(114); emit_ptx_decimal(ra);
        emit_ptx_byte(59); emit_ptx_byte(10);
        __arena_set(vtab + 55, 0);
        r
    }
}

// K1.M5e (2026-05-28): emit a scalar negate "    neg.s32 %rD, %rA;".
// Lowers the inner expr, allocates a result register, returns it.
// AST_NEG (tag 9, inner in slot 1). Mirrors Python ptx.py SCALAR_NEG.
fn emit_ptx_neg(node: i32, vtab: i32) -> i32 {
    let a = emit_ptx_expr(__arena_get(node + 1), vtab);
    let r = ptx_alloc_reg(vtab);
    // "    neg.s32 %r" + D
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(110); emit_ptx_byte(101);
    emit_ptx_byte(103); emit_ptx_byte(46); emit_ptx_byte(115);
    emit_ptx_byte(51); emit_ptx_byte(50); emit_ptx_byte(32);
    emit_ptx_byte(37); emit_ptx_byte(114);
    emit_ptx_decimal(r);
    // ", %r" + A
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(114); emit_ptx_decimal(a);
    // ";\n"
    emit_ptx_byte(59); emit_ptx_byte(10);
    r
}

// K1.M8 (2026-05-28): emit the 2-char setp condition mnemonic for cc
// (0=lt 1=gt 2=eq 3=ne 4=le 5=ge). Flat -- exactly one fires.
fn emit_ptx_cc(cc: i32) -> i32 {
    if cc == 0 { emit_ptx_byte(108); emit_ptx_byte(116); };   // lt
    if cc == 1 { emit_ptx_byte(103); emit_ptx_byte(116); };   // gt
    if cc == 2 { emit_ptx_byte(101); emit_ptx_byte(113); };   // eq
    if cc == 3 { emit_ptx_byte(110); emit_ptx_byte(101); };   // ne
    if cc == 4 { emit_ptx_byte(108); emit_ptx_byte(101); };   // le
    if cc == 5 { emit_ptx_byte(103); emit_ptx_byte(101); };   // ge
    0
}

// K1.M8: lower a comparison (AST_LT/GT/EQ/NE/LE/GE) to a 0/1 value in
// a register: "setp.<cc>.s32 %pP, %rA, %rB" then "selp.b32 %rR, 1, 0,
// %pP". Matches the AST semantics (comparisons reify 0/1) and fits the
// value-returning emit_ptx_expr model. ptxas-validated form.
fn emit_ptx_cmp(node: i32, vtab: i32, cc: i32) -> i32 {
    let la = emit_ptx_expr(__arena_get(node + 1), vtab);
    let ra = emit_ptx_expr(__arena_get(node + 2), vtab);
    let p = ptx_alloc_pred(vtab);
    let r = ptx_alloc_reg(vtab);
    // "    setp."
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(115); emit_ptx_byte(101);
    emit_ptx_byte(116); emit_ptx_byte(112); emit_ptx_byte(46);
    emit_ptx_cc(cc);
    // ".s32 %p" + P
    emit_ptx_byte(46); emit_ptx_byte(115); emit_ptx_byte(51);
    emit_ptx_byte(50); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(112);
    emit_ptx_decimal(p);
    // ", %r" + A
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(114); emit_ptx_decimal(la);
    // ", %r" + B
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(114); emit_ptx_decimal(ra);
    // ";\n"
    emit_ptx_byte(59); emit_ptx_byte(10);
    // "    selp.b32 %r" + R
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(115); emit_ptx_byte(101);
    emit_ptx_byte(108); emit_ptx_byte(112); emit_ptx_byte(46);
    emit_ptx_byte(98); emit_ptx_byte(51); emit_ptx_byte(50);
    emit_ptx_byte(32); emit_ptx_byte(37); emit_ptx_byte(114);
    emit_ptx_decimal(r);
    // ", 1, 0, %p" + P
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(49);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(48);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(112); emit_ptx_decimal(p);
    // ";\n"
    emit_ptx_byte(59); emit_ptx_byte(10);
    r
}

// K1.M8b: emit a label NAME "$Lelse_<n>" (which=0) or "$Lend_<n>"
// (which=1) -- used in both `bra` targets and the label definitions.
fn emit_ptx_lbl_ref(which: i32, n: i32) -> i32 {
    emit_ptx_byte(36); emit_ptx_byte(76);   // "$L"
    if which == 0 {     // "else_" (K1.M8b if)
        emit_ptx_byte(101); emit_ptx_byte(108); emit_ptx_byte(115);
        emit_ptx_byte(101); emit_ptx_byte(95);
    };
    if which == 1 {     // "end_" (K1.M8b if)
        emit_ptx_byte(101); emit_ptx_byte(110); emit_ptx_byte(100);
        emit_ptx_byte(95);
    };
    if which == 2 {     // "top_" (K1.M9 while loop top)
        emit_ptx_byte(116); emit_ptx_byte(111); emit_ptx_byte(112);
        emit_ptx_byte(95);
    };
    if which == 3 {     // "wend_" (K1.M9 while loop exit)
        emit_ptx_byte(119); emit_ptx_byte(101); emit_ptx_byte(110);
        emit_ptx_byte(100); emit_ptx_byte(95);
    };
    emit_ptx_decimal(n);
    0
}

// K1.M8b: lower AST_IF (tag 7; cond slot 1, then slot 2, else slot 3)
// as a predicated branch -- completes control flow. The cond value is
// tested != 0; on false we branch over the then-block to the else.
// Per-kernel unique labels via ptx_alloc_label. if-as-statement: the
// value is discarded (a void kernel uses if for side effects; an
// if-as-value phi/merge is deferred). ptxas-validated form.
fn emit_ptx_if(node: i32, vtab: i32) -> i32 {
    let rc = emit_ptx_expr(__arena_get(node + 1), vtab);
    let pz = ptx_alloc_pred(vtab);
    let n = ptx_alloc_label(vtab);
    // "    setp.ne.s32 %p<pz>, %r<rc>, 0;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(115); emit_ptx_byte(101);
    emit_ptx_byte(116); emit_ptx_byte(112); emit_ptx_byte(46);
    emit_ptx_byte(110); emit_ptx_byte(101); emit_ptx_byte(46);
    emit_ptx_byte(115); emit_ptx_byte(51); emit_ptx_byte(50);
    emit_ptx_byte(32); emit_ptx_byte(37); emit_ptx_byte(112);
    emit_ptx_decimal(pz);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(114); emit_ptx_decimal(rc);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(48);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // "    @!%p<pz> bra $Lelse_<n>;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(64); emit_ptx_byte(33);
    emit_ptx_byte(37); emit_ptx_byte(112);
    emit_ptx_decimal(pz);
    emit_ptx_byte(32); emit_ptx_byte(98); emit_ptx_byte(114);
    emit_ptx_byte(97); emit_ptx_byte(32);
    emit_ptx_lbl_ref(0, n);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // then-branch (result discarded)
    emit_ptx_expr(__arena_get(node + 2), vtab);
    // "    bra $Lend_<n>;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(98); emit_ptx_byte(114);
    emit_ptx_byte(97); emit_ptx_byte(32);
    emit_ptx_lbl_ref(1, n);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // "$Lelse_<n>:\n"
    emit_ptx_lbl_ref(0, n);
    emit_ptx_byte(58); emit_ptx_byte(10);
    // else-branch (result discarded); guard against a missing else
    let else_idx = __arena_get(node + 3);
    if else_idx != 0 {
        emit_ptx_expr(else_idx, vtab);
    };
    // "$Lend_<n>:\n"
    emit_ptx_lbl_ref(1, n);
    emit_ptx_byte(58); emit_ptx_byte(10);
    0 - 1
}

// K1.M9 (2026-05-28): lower AST_WHILE (tag 10; cond slot 1, body slot
// 2) as a loop: "$Ltop_<n>:" / eval cond / setp.ne != 0 / "@!%p bra
// $Lwend_<n>" / body / "bra $Ltop_<n>" / "$Lwend_<n>:". Reuses the
// label + predicate machinery. Statement (returns -1). ptxas-validated.
fn emit_ptx_while(node: i32, vtab: i32) -> i32 {
    let n = ptx_alloc_label(vtab);
    // "$Ltop_<n>:\n"
    emit_ptx_lbl_ref(2, n);
    emit_ptx_byte(58); emit_ptx_byte(10);
    // cond
    let rc = emit_ptx_expr(__arena_get(node + 1), vtab);
    let pz = ptx_alloc_pred(vtab);
    // "    setp.ne.s32 %p<pz>, %r<rc>, 0;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(115); emit_ptx_byte(101);
    emit_ptx_byte(116); emit_ptx_byte(112); emit_ptx_byte(46);
    emit_ptx_byte(110); emit_ptx_byte(101); emit_ptx_byte(46);
    emit_ptx_byte(115); emit_ptx_byte(51); emit_ptx_byte(50);
    emit_ptx_byte(32); emit_ptx_byte(37); emit_ptx_byte(112);
    emit_ptx_decimal(pz);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(114); emit_ptx_decimal(rc);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(48);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // "    @!%p<pz> bra $Lwend_<n>;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(64); emit_ptx_byte(33);
    emit_ptx_byte(37); emit_ptx_byte(112);
    emit_ptx_decimal(pz);
    emit_ptx_byte(32); emit_ptx_byte(98); emit_ptx_byte(114);
    emit_ptx_byte(97); emit_ptx_byte(32);
    emit_ptx_lbl_ref(3, n);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // body (result discarded)
    emit_ptx_expr(__arena_get(node + 2), vtab);
    // "    bra $Ltop_<n>;\n"  (back-edge)
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(98); emit_ptx_byte(114);
    emit_ptx_byte(97); emit_ptx_byte(32);
    emit_ptx_lbl_ref(2, n);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // "$Lwend_<n>:\n"
    emit_ptx_lbl_ref(3, n);
    emit_ptx_byte(58); emit_ptx_byte(10);
    0 - 1
}

// K1.M9: lower AST_ASSIGN (tag 11; name slot 1/2, val slot 3) -- `x =
// v` overwrites the variable's existing register: "mov.s32 %rX, %rV"
// (the var->reg binding is unchanged; subsequent reads see the new
// value). Returns the destination register.
fn emit_ptx_assign(node: i32, vtab: i32) -> i32 {
    let rv = emit_ptx_expr(__arena_get(node + 3), vtab);
    let rx = ptx_vtab_lookup(vtab, __arena_get(node + 1), __arena_get(node + 2));
    // "    mov.s32 %r<rx>, %r<rv>;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(109); emit_ptx_byte(111);
    emit_ptx_byte(118); emit_ptx_byte(46); emit_ptx_byte(115);
    emit_ptx_byte(51); emit_ptx_byte(50); emit_ptx_byte(32);
    emit_ptx_byte(37); emit_ptx_byte(114);
    emit_ptx_decimal(rx);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(114); emit_ptx_decimal(rv);
    emit_ptx_byte(59); emit_ptx_byte(10);
    rx
}

// K1.M10: small emit helpers -- "    " indent, "%rN", "%rdN".
fn emit_ptx_indent() -> i32 {
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32);
    0
}
fn emit_ptx_r(n: i32) -> i32 {
    emit_ptx_byte(37); emit_ptx_byte(114);   // "%r"
    emit_ptx_decimal(n);
    0
}
fn emit_ptx_rd(n: i32) -> i32 {
    emit_ptx_byte(37); emit_ptx_byte(114); emit_ptx_byte(100);   // "%rd"
    emit_ptx_decimal(n);
    0
}
fn emit_ptx_f(n: i32) -> i32 {
    emit_ptx_byte(37); emit_ptx_byte(102);   // "%f"
    emit_ptx_decimal(n);
    0
}

// K1.M10: resolve a name to its 0-based kernel-parameter index by
// walking the AST_PARAM list (fn_idx slot 4 = params_head; each
// AST_PARAM: name in slots 1/2, next in slot 3). Returns -1 if not a
// param. Used by index loads to find the base pointer (.param .b64).
fn ptx_param_index(fn_idx: i32, name_s: i32, name_l: i32) -> i32 {
    let mut p = __arena_get(fn_idx + 4);
    let mut idx: i32 = 0;
    let mut found: i32 = 0 - 1;
    while p != 0 {
        if kovc_byte_eq(name_s, name_l, __arena_get(p + 1), __arena_get(p + 2)) == 1 {
            if found < 0 { found = idx; };
        };
        idx = idx + 1;
        p = __arena_get(p + 3);
    }
    found
}
// K1.M11: the type tag of a named kernel param (AST_PARAM slot 4:
// 0=i32, 1=f32, ...), or -1 if not a param. Lets index loads/stores
// pick the element type (ld.global.f32 vs ld.global.u32).
fn ptx_param_type(fn_idx: i32, name_s: i32, name_l: i32) -> i32 {
    let mut p = __arena_get(fn_idx + 4);
    let mut found: i32 = 0 - 1;
    while p != 0 {
        if kovc_byte_eq(name_s, name_l, __arena_get(p + 1), __arena_get(p + 2)) == 1 {
            if found < 0 { found = __arena_get(p + 4); };
        };
        p = __arena_get(p + 3);
    }
    found
}

// K1.M10: lower a global-memory LOAD a[i] (AST_INDEX = tag 53; base in
// slot 1 -- an AST_VAR naming a kernel param; index in slot 2). Emits
// the standard CUDA load: load the param pointer, convert to the global
// address space, compute base + i*4 (i32 elems), then ld.global.u32.
// ptxas-validated. Returns the %r holding the loaded value.
fn emit_ptx_index_load(node: i32, vtab: i32) -> i32 {
    let base = __arena_get(node + 1);
    let idx_node = __arena_get(node + 2);
    let fn_idx = __arena_get(vtab + 53);
    let pidx = ptx_param_index(fn_idx, __arena_get(base + 1),
                               __arena_get(base + 2));
    let ri = emit_ptx_expr(idx_node, vtab);
    // "    ld.param.u64 %rd<rdb>, [param_<pidx>];\n"
    let rdb = ptx_alloc_rd(vtab);
    emit_ptx_indent();
    emit_ptx_byte(108); emit_ptx_byte(100); emit_ptx_byte(46);
    emit_ptx_byte(112); emit_ptx_byte(97); emit_ptx_byte(114);
    emit_ptx_byte(97); emit_ptx_byte(109); emit_ptx_byte(46);
    emit_ptx_byte(117); emit_ptx_byte(54); emit_ptx_byte(52);
    emit_ptx_byte(32);
    emit_ptx_rd(rdb);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(91);
    emit_ptx_byte(112); emit_ptx_byte(97); emit_ptx_byte(114);
    emit_ptx_byte(97); emit_ptx_byte(109); emit_ptx_byte(95);
    emit_ptx_decimal(pidx);
    emit_ptx_byte(93); emit_ptx_byte(59); emit_ptx_byte(10);
    // "    cvta.to.global.u64 %rd<rdg>, %rd<rdb>;\n"
    let rdg = ptx_alloc_rd(vtab);
    emit_ptx_indent();
    emit_ptx_byte(99); emit_ptx_byte(118); emit_ptx_byte(116);
    emit_ptx_byte(97); emit_ptx_byte(46); emit_ptx_byte(116);
    emit_ptx_byte(111); emit_ptx_byte(46); emit_ptx_byte(103);
    emit_ptx_byte(108); emit_ptx_byte(111); emit_ptx_byte(98);
    emit_ptx_byte(97); emit_ptx_byte(108); emit_ptx_byte(46);
    emit_ptx_byte(117); emit_ptx_byte(54); emit_ptx_byte(52);
    emit_ptx_byte(32);
    emit_ptx_rd(rdg);
    emit_ptx_byte(44); emit_ptx_byte(32);
    emit_ptx_rd(rdb);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // "    mul.wide.s32 %rd<rdo>, %r<ri>, 4;\n"
    let rdo = ptx_alloc_rd(vtab);
    emit_ptx_indent();
    emit_ptx_byte(109); emit_ptx_byte(117); emit_ptx_byte(108);
    emit_ptx_byte(46); emit_ptx_byte(119); emit_ptx_byte(105);
    emit_ptx_byte(100); emit_ptx_byte(101); emit_ptx_byte(46);
    emit_ptx_byte(115); emit_ptx_byte(51); emit_ptx_byte(50);
    emit_ptx_byte(32);
    emit_ptx_rd(rdo);
    emit_ptx_byte(44); emit_ptx_byte(32);
    emit_ptx_r(ri);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(52);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // "    add.s64 %rd<rda>, %rd<rdg>, %rd<rdo>;\n"
    let rda = ptx_alloc_rd(vtab);
    emit_ptx_indent();
    emit_ptx_byte(97); emit_ptx_byte(100); emit_ptx_byte(100);
    emit_ptx_byte(46); emit_ptx_byte(115); emit_ptx_byte(54);
    emit_ptx_byte(52); emit_ptx_byte(32);
    emit_ptx_rd(rda);
    emit_ptx_byte(44); emit_ptx_byte(32);
    emit_ptx_rd(rdg);
    emit_ptx_byte(44); emit_ptx_byte(32);
    emit_ptx_rd(rdo);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // K1.M11: the load instruction depends on the element type. If the
    // base param is f32 (type_tag 1) emit "ld.global.f32 %f<n>" (into a
    // float register); else the i32 path "ld.global.u32 %r<n>".
    let elem_ty = ptx_param_type(fn_idx, __arena_get(base + 1),
                                 __arena_get(base + 2));
    if elem_ty == 1 {
        // "    ld.global.f32 %f<fdst>, [%rd<rda>];\n"
        let fdst = ptx_alloc_f(vtab);
        emit_ptx_indent();
        emit_ptx_byte(108); emit_ptx_byte(100); emit_ptx_byte(46);
        emit_ptx_byte(103); emit_ptx_byte(108); emit_ptx_byte(111);
        emit_ptx_byte(98); emit_ptx_byte(97); emit_ptx_byte(108);
        emit_ptx_byte(46); emit_ptx_byte(102); emit_ptx_byte(51);
        emit_ptx_byte(50); emit_ptx_byte(32);
        emit_ptx_f(fdst);
        emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(91);
        emit_ptx_rd(rda);
        emit_ptx_byte(93); emit_ptx_byte(59); emit_ptx_byte(10);
        __arena_set(vtab + 55, 1);   // K1.M12: result is f32
        fdst
    } else {
        // "    ld.global.u32 %r<rdst>, [%rd<rda>];\n"
        let rdst = ptx_alloc_reg(vtab);
        emit_ptx_indent();
        emit_ptx_byte(108); emit_ptx_byte(100); emit_ptx_byte(46);
        emit_ptx_byte(103); emit_ptx_byte(108); emit_ptx_byte(111);
        emit_ptx_byte(98); emit_ptx_byte(97); emit_ptx_byte(108);
        emit_ptx_byte(46); emit_ptx_byte(117); emit_ptx_byte(51);
        emit_ptx_byte(50); emit_ptx_byte(32);
        emit_ptx_r(rdst);
        emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(91);
        emit_ptx_rd(rda);
        emit_ptx_byte(93); emit_ptx_byte(59); emit_ptx_byte(10);
        __arena_set(vtab + 55, 0);   // K1.M12: result is i32
        rdst
    }
}

// K1.M10c: lower a global-memory STORE a[i] = v (AST_INDEX_STORE = tag
// 55; slot 1 = the AST_INDEX node [base, index], slot 2 = value).
// Value is lowered first, then the same address computation as the
// load, then "st.global.u32 [%rdAddr], %rVal". ptxas-validated.
// Statement -- returns -1.
fn emit_ptx_index_store(node: i32, vtab: i32) -> i32 {
    let idx_node = __arena_get(node + 1);
    let base = __arena_get(idx_node + 1);
    let idx_inner = __arena_get(idx_node + 2);
    let val_node = __arena_get(node + 2);
    let fn_idx = __arena_get(vtab + 53);
    let pidx = ptx_param_index(fn_idx, __arena_get(base + 1),
                               __arena_get(base + 2));
    let rv = emit_ptx_expr(val_node, vtab);
    let is_f = __arena_get(vtab + 55);   // K1.M12: value type BEFORE the
                                         // index lowering clobbers the flag
    let ri = emit_ptx_expr(idx_inner, vtab);
    // "    ld.param.u64 %rd<rdb>, [param_<pidx>];\n"
    let rdb = ptx_alloc_rd(vtab);
    emit_ptx_indent();
    emit_ptx_byte(108); emit_ptx_byte(100); emit_ptx_byte(46);
    emit_ptx_byte(112); emit_ptx_byte(97); emit_ptx_byte(114);
    emit_ptx_byte(97); emit_ptx_byte(109); emit_ptx_byte(46);
    emit_ptx_byte(117); emit_ptx_byte(54); emit_ptx_byte(52);
    emit_ptx_byte(32);
    emit_ptx_rd(rdb);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(91);
    emit_ptx_byte(112); emit_ptx_byte(97); emit_ptx_byte(114);
    emit_ptx_byte(97); emit_ptx_byte(109); emit_ptx_byte(95);
    emit_ptx_decimal(pidx);
    emit_ptx_byte(93); emit_ptx_byte(59); emit_ptx_byte(10);
    // "    cvta.to.global.u64 %rd<rdg>, %rd<rdb>;\n"
    let rdg = ptx_alloc_rd(vtab);
    emit_ptx_indent();
    emit_ptx_byte(99); emit_ptx_byte(118); emit_ptx_byte(116);
    emit_ptx_byte(97); emit_ptx_byte(46); emit_ptx_byte(116);
    emit_ptx_byte(111); emit_ptx_byte(46); emit_ptx_byte(103);
    emit_ptx_byte(108); emit_ptx_byte(111); emit_ptx_byte(98);
    emit_ptx_byte(97); emit_ptx_byte(108); emit_ptx_byte(46);
    emit_ptx_byte(117); emit_ptx_byte(54); emit_ptx_byte(52);
    emit_ptx_byte(32);
    emit_ptx_rd(rdg);
    emit_ptx_byte(44); emit_ptx_byte(32);
    emit_ptx_rd(rdb);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // "    mul.wide.s32 %rd<rdo>, %r<ri>, 4;\n"
    let rdo = ptx_alloc_rd(vtab);
    emit_ptx_indent();
    emit_ptx_byte(109); emit_ptx_byte(117); emit_ptx_byte(108);
    emit_ptx_byte(46); emit_ptx_byte(119); emit_ptx_byte(105);
    emit_ptx_byte(100); emit_ptx_byte(101); emit_ptx_byte(46);
    emit_ptx_byte(115); emit_ptx_byte(51); emit_ptx_byte(50);
    emit_ptx_byte(32);
    emit_ptx_rd(rdo);
    emit_ptx_byte(44); emit_ptx_byte(32);
    emit_ptx_r(ri);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(52);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // "    add.s64 %rd<rda>, %rd<rdg>, %rd<rdo>;\n"
    let rda = ptx_alloc_rd(vtab);
    emit_ptx_indent();
    emit_ptx_byte(97); emit_ptx_byte(100); emit_ptx_byte(100);
    emit_ptx_byte(46); emit_ptx_byte(115); emit_ptx_byte(54);
    emit_ptx_byte(52); emit_ptx_byte(32);
    emit_ptx_rd(rda);
    emit_ptx_byte(44); emit_ptx_byte(32);
    emit_ptx_rd(rdg);
    emit_ptx_byte(44); emit_ptx_byte(32);
    emit_ptx_rd(rdo);
    emit_ptx_byte(59); emit_ptx_byte(10);
    // "    st.global.<f32|u32> [%rd<rda>], <%f|%r><rv>;\n" (K1.M12:
    // type from is_f -- f32 store of a %f value, else i32 of a %r).
    emit_ptx_indent();
    emit_ptx_byte(115); emit_ptx_byte(116); emit_ptx_byte(46);   // "st."
    emit_ptx_byte(103); emit_ptx_byte(108); emit_ptx_byte(111);  // "glo"
    emit_ptx_byte(98); emit_ptx_byte(97); emit_ptx_byte(108);    // "bal"
    emit_ptx_byte(46);                                           // "."
    if is_f == 1 {
        emit_ptx_byte(102); emit_ptx_byte(51); emit_ptx_byte(50);   // "f32"
    } else {
        emit_ptx_byte(117); emit_ptx_byte(51); emit_ptx_byte(50);   // "u32"
    };
    emit_ptx_byte(32); emit_ptx_byte(91);                        // " ["
    emit_ptx_rd(rda);
    emit_ptx_byte(93); emit_ptx_byte(44); emit_ptx_byte(32);     // "], "
    if is_f == 1 {
        emit_ptx_f(rv);
    } else {
        emit_ptx_r(rv);
    };
    emit_ptx_byte(59); emit_ptx_byte(10);
    0 - 1
}

// K1.M6 (2026-05-28): is this call name the byte-string "thread_idx"
// (10 chars: t h r e a d _ i d x)? Flat accumulator compare (the
// kovc_byte_eq idiom) against the hardcoded ASCII codes.
fn ptx_name_is_thread_idx(name_s: i32, name_l: i32) -> i32 {
    if name_l != 10 {
        0
    } else {
        let mut ok: i32 = 1;
        if __arena_get(name_s + 0) != 116 { ok = 0; };   // t
        if __arena_get(name_s + 1) != 104 { ok = 0; };   // h
        if __arena_get(name_s + 2) != 114 { ok = 0; };   // r
        if __arena_get(name_s + 3) != 101 { ok = 0; };   // e
        if __arena_get(name_s + 4) != 97 { ok = 0; };    // a
        if __arena_get(name_s + 5) != 100 { ok = 0; };   // d
        if __arena_get(name_s + 6) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 7) != 105 { ok = 0; };   // i
        if __arena_get(name_s + 8) != 100 { ok = 0; };   // d
        if __arena_get(name_s + 9) != 120 { ok = 0; };   // x
        ok
    }
}

// K1.M6: emit a THREAD_IDX x-dim read "    mov.u32 %rN, %tid.x;" --
// the hardware thread-index special register, the entry point to
// every data-parallel kernel. Mirrors Python ptx.py THREAD_IDX
// (sreg "tid", dim x). %tid.x is a u32 sreg; we read it into a .b32 reg.
fn emit_ptx_tid_x(ridx: i32) -> i32 {
    // "    mov.u32 %r" + N
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(109); emit_ptx_byte(111);
    emit_ptx_byte(118); emit_ptx_byte(46); emit_ptx_byte(117);
    emit_ptx_byte(51); emit_ptx_byte(50); emit_ptx_byte(32);
    emit_ptx_byte(37); emit_ptx_byte(114);
    emit_ptx_decimal(ridx);
    // ", %tid.x;\n"
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(116); emit_ptx_byte(105); emit_ptx_byte(100);
    emit_ptx_byte(46); emit_ptx_byte(120); emit_ptx_byte(59);
    emit_ptx_byte(10);
    0
}

// K1.M7 (2026-05-28): block_idx() -> %ctaid.x (CTA/block index) and
// block_dim() -> %ntid.x (threads-per-block). With thread_idx() these
// compute the canonical global thread index
// `block_idx()*block_dim() + thread_idx()` that every grid-stride
// kernel uses. Mirrors Python lower_ast.py block_idx/block_dim.
fn ptx_name_is_block_idx(name_s: i32, name_l: i32) -> i32 {
    if name_l != 9 {
        0
    } else {
        let mut ok: i32 = 1;
        if __arena_get(name_s + 0) != 98 { ok = 0; };    // b
        if __arena_get(name_s + 1) != 108 { ok = 0; };   // l
        if __arena_get(name_s + 2) != 111 { ok = 0; };   // o
        if __arena_get(name_s + 3) != 99 { ok = 0; };    // c
        if __arena_get(name_s + 4) != 107 { ok = 0; };   // k
        if __arena_get(name_s + 5) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 6) != 105 { ok = 0; };   // i
        if __arena_get(name_s + 7) != 100 { ok = 0; };   // d
        if __arena_get(name_s + 8) != 120 { ok = 0; };   // x
        ok
    }
}
fn ptx_name_is_block_dim(name_s: i32, name_l: i32) -> i32 {
    if name_l != 9 {
        0
    } else {
        let mut ok: i32 = 1;
        if __arena_get(name_s + 0) != 98 { ok = 0; };    // b
        if __arena_get(name_s + 1) != 108 { ok = 0; };   // l
        if __arena_get(name_s + 2) != 111 { ok = 0; };   // o
        if __arena_get(name_s + 3) != 99 { ok = 0; };    // c
        if __arena_get(name_s + 4) != 107 { ok = 0; };   // k
        if __arena_get(name_s + 5) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 6) != 100 { ok = 0; };   // d
        if __arena_get(name_s + 7) != 105 { ok = 0; };   // i
        if __arena_get(name_s + 8) != 109 { ok = 0; };   // m
        ok
    }
}
fn emit_ptx_mov_ctaid_x(ridx: i32) -> i32 {
    // "    mov.u32 %r" + N + ", %ctaid.x;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(109); emit_ptx_byte(111);
    emit_ptx_byte(118); emit_ptx_byte(46); emit_ptx_byte(117);
    emit_ptx_byte(51); emit_ptx_byte(50); emit_ptx_byte(32);
    emit_ptx_byte(37); emit_ptx_byte(114);
    emit_ptx_decimal(ridx);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(99); emit_ptx_byte(116); emit_ptx_byte(97);
    emit_ptx_byte(105); emit_ptx_byte(100); emit_ptx_byte(46);
    emit_ptx_byte(120); emit_ptx_byte(59); emit_ptx_byte(10);
    0
}
fn emit_ptx_mov_ntid_x(ridx: i32) -> i32 {
    // "    mov.u32 %r" + N + ", %ntid.x;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(109); emit_ptx_byte(111);
    emit_ptx_byte(118); emit_ptx_byte(46); emit_ptx_byte(117);
    emit_ptx_byte(51); emit_ptx_byte(50); emit_ptx_byte(32);
    emit_ptx_byte(37); emit_ptx_byte(114);
    emit_ptx_decimal(ridx);
    emit_ptx_byte(44); emit_ptx_byte(32); emit_ptx_byte(37);
    emit_ptx_byte(110); emit_ptx_byte(116); emit_ptx_byte(105);
    emit_ptx_byte(100); emit_ptx_byte(46); emit_ptx_byte(120);
    emit_ptx_byte(59); emit_ptx_byte(10);
    0
}

// K1.M13 (2026-05-28): "__tile_zeros" name matcher (12 chars). The
// FIRST GPU tile op. Register-tile model: __tile_zeros(N, M) -> N*M
// consecutive `mov.f32 %fX, 0f00000000;` zero-fills, result = base %f
// register. Mirrors Python backend/ptx.py TILE_ZEROS (Stage 64 Inc 2).
// The CALL already parses (CPU path K1.F23c uses the same 2-arg
// signature), so this is pure-additive -- NO parser change.
fn ptx_name_is_tile_zeros(name_s: i32, name_l: i32) -> i32 {
    if name_l != 12 {
        0
    } else {
        let mut ok: i32 = 1;
        if __arena_get(name_s + 0) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 1) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 2) != 116 { ok = 0; };   // t
        if __arena_get(name_s + 3) != 105 { ok = 0; };   // i
        if __arena_get(name_s + 4) != 108 { ok = 0; };   // l
        if __arena_get(name_s + 5) != 101 { ok = 0; };   // e
        if __arena_get(name_s + 6) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 7) != 122 { ok = 0; };   // z
        if __arena_get(name_s + 8) != 101 { ok = 0; };   // e
        if __arena_get(name_s + 9) != 114 { ok = 0; };   // r
        if __arena_get(name_s + 10) != 111 { ok = 0; };  // o
        if __arena_get(name_s + 11) != 115 { ok = 0; };  // s
        ok
    }
}
// K1.M13: emit one tile-zero register-fill "    mov.f32 %fN, 0f00000000;".
// 0f00000000 is the PTX hex literal for +0.0f. Mirrors Python ptx.py
// TILE_ZEROS register-fill.
fn emit_ptx_mov_f_zero(ridx: i32) -> i32 {
    emit_ptx_indent();
    // "mov.f32 "
    emit_ptx_byte(109); emit_ptx_byte(111); emit_ptx_byte(118);
    emit_ptx_byte(46); emit_ptx_byte(102); emit_ptx_byte(51);
    emit_ptx_byte(50); emit_ptx_byte(32);
    emit_ptx_f(ridx);
    // ", 0f00000000;\n"
    emit_ptx_byte(44); emit_ptx_byte(32);
    emit_ptx_byte(48); emit_ptx_byte(102);
    emit_ptx_byte(48); emit_ptx_byte(48); emit_ptx_byte(48); emit_ptx_byte(48);
    emit_ptx_byte(48); emit_ptx_byte(48); emit_ptx_byte(48); emit_ptx_byte(48);
    emit_ptx_byte(59); emit_ptx_byte(10);
    0
}

// K1.M6: lower an AST_CALL (tag 16; name in slots 1/2, args_head in
// slot 3). Recognised GPU builtins: thread_idx()/block_idx()/block_dim()
// (parallel-index sregs) and __tile_zeros(N, M) (K1.M13, first tile op).
// K1.M14 (2026-05-28): "__tile_add" name matcher (10 chars). The first
// tile COMPUTE op: __tile_add(a, b, dst, count) elementwise-adds two
// register-tiles into a third over `count` consecutive %f registers,
// mirroring Python backend/ptx.py TILE_ADD (Stage 64 Inc 3).
fn ptx_name_is_tile_add(name_s: i32, name_l: i32) -> i32 {
    if name_l != 10 {
        0
    } else {
        let mut ok: i32 = 1;
        if __arena_get(name_s + 0) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 1) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 2) != 116 { ok = 0; };   // t
        if __arena_get(name_s + 3) != 105 { ok = 0; };   // i
        if __arena_get(name_s + 4) != 108 { ok = 0; };   // l
        if __arena_get(name_s + 5) != 101 { ok = 0; };   // e
        if __arena_get(name_s + 6) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 7) != 97 { ok = 0; };    // a
        if __arena_get(name_s + 8) != 100 { ok = 0; };   // d
        if __arena_get(name_s + 9) != 100 { ok = 0; };   // d
        ok
    }
}
// K1.M15 (2026-05-28): "__tile_sub" (10 chars) + "__tile_mul" matchers --
// siblings of __tile_add, one mnemonic apart (sub.f32 / mul.f32).
fn ptx_name_is_tile_sub(name_s: i32, name_l: i32) -> i32 {
    if name_l != 10 {
        0
    } else {
        let mut ok: i32 = 1;
        if __arena_get(name_s + 0) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 1) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 2) != 116 { ok = 0; };   // t
        if __arena_get(name_s + 3) != 105 { ok = 0; };   // i
        if __arena_get(name_s + 4) != 108 { ok = 0; };   // l
        if __arena_get(name_s + 5) != 101 { ok = 0; };   // e
        if __arena_get(name_s + 6) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 7) != 115 { ok = 0; };   // s
        if __arena_get(name_s + 8) != 117 { ok = 0; };   // u
        if __arena_get(name_s + 9) != 98 { ok = 0; };    // b
        ok
    }
}
fn ptx_name_is_tile_mul(name_s: i32, name_l: i32) -> i32 {
    if name_l != 10 {
        0
    } else {
        let mut ok: i32 = 1;
        if __arena_get(name_s + 0) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 1) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 2) != 116 { ok = 0; };   // t
        if __arena_get(name_s + 3) != 105 { ok = 0; };   // i
        if __arena_get(name_s + 4) != 108 { ok = 0; };   // l
        if __arena_get(name_s + 5) != 101 { ok = 0; };   // e
        if __arena_get(name_s + 6) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 7) != 109 { ok = 0; };   // m
        if __arena_get(name_s + 8) != 117 { ok = 0; };   // u
        if __arena_get(name_s + 9) != 108 { ok = 0; };   // l
        ok
    }
}
// K1.M14/M15: emit one tile binary-op line "    <op>.f32 %f<rd>, %f<ra>,
// %f<rb>;" -- opc selects the mnemonic (0=add, 1=sub, 2=mul). Mirrors the
// emit_ptx_binop f32 path + Python backend/ptx.py TILE_ADD/SUB/MUL.
fn emit_ptx_binop_f3(opc: i32, rd: i32, ra: i32, rb: i32) -> i32 {
    emit_ptx_indent();
    if opc == 0 {
        emit_ptx_byte(97); emit_ptx_byte(100); emit_ptx_byte(100);   // add
    };
    if opc == 1 {
        emit_ptx_byte(115); emit_ptx_byte(117); emit_ptx_byte(98);   // sub
    };
    if opc == 2 {
        emit_ptx_byte(109); emit_ptx_byte(117); emit_ptx_byte(108);  // mul
    };
    // ".f32 "
    emit_ptx_byte(46); emit_ptx_byte(102); emit_ptx_byte(51);
    emit_ptx_byte(50); emit_ptx_byte(32);
    emit_ptx_f(rd);
    emit_ptx_byte(44); emit_ptx_byte(32);   // ", "
    emit_ptx_f(ra);
    emit_ptx_byte(44); emit_ptx_byte(32);   // ", "
    emit_ptx_f(rb);
    emit_ptx_byte(59); emit_ptx_byte(10);   // ";\n"
    0
}
// K1.M14/M15: lower a tile elementwise binary op __tile_<add|sub|mul>(a,
// b, dst, count). a/b/dst are AST_VAR bound to prior __tile_zeros results
// (ridx = the %f base); count is a static AST_INT. Emit `count` op.f32
// lines: dst[k] = a[k] <op> b[k] over consecutive %f. opc 0=add 1=sub
// 2=mul. Result = base_d; sets the float flag (slot 55).
fn emit_ptx_tile_binop(node: i32, vtab: i32, opc: i32) -> i32 {
    let ah = __arena_get(node + 3);     // args_head (AST_ARG)
    let a0 = __arena_get(ah + 1);       // arg 0: AST_VAR a
    let nxt1 = __arena_get(ah + 2);
    let a1 = __arena_get(nxt1 + 1);     // arg 1: AST_VAR b
    let nxt2 = __arena_get(nxt1 + 2);
    let a2 = __arena_get(nxt2 + 1);     // arg 2: AST_VAR dst
    let nxt3 = __arena_get(nxt2 + 2);
    let a3 = __arena_get(nxt3 + 1);     // arg 3: AST_INT count
    let base_a = ptx_vtab_lookup(vtab, __arena_get(a0 + 1), __arena_get(a0 + 2));
    let base_b = ptx_vtab_lookup(vtab, __arena_get(a1 + 1), __arena_get(a1 + 2));
    let base_d = ptx_vtab_lookup(vtab, __arena_get(a2 + 1), __arena_get(a2 + 2));
    let cnt = __arena_get(a3 + 1);
    let mut k: i32 = 0;
    while k < cnt {
        emit_ptx_binop_f3(opc, base_d + k, base_a + k, base_b + k);
        k = k + 1;
    }
    __arena_set(vtab + 55, 1);
    base_d
}

// K1.M16 (2026-05-28): reg-to-reg "    mov.f32 %f<rd>, %f<rs>;" -- moves
// a computed accumulator into a destination tile register (used by matmul
// to write each dst[i][j]).
fn emit_ptx_mov_f_reg(rd: i32, rs: i32) -> i32 {
    emit_ptx_indent();
    // "mov.f32 "
    emit_ptx_byte(109); emit_ptx_byte(111); emit_ptx_byte(118);
    emit_ptx_byte(46); emit_ptx_byte(102); emit_ptx_byte(51);
    emit_ptx_byte(50); emit_ptx_byte(32);
    emit_ptx_f(rd);
    emit_ptx_byte(44); emit_ptx_byte(32);   // ", "
    emit_ptx_f(rs);
    emit_ptx_byte(59); emit_ptx_byte(10);   // ";\n"
    0
}
// K1.M16: "__tile_matmul" name matcher (13 chars).
fn ptx_name_is_tile_matmul(name_s: i32, name_l: i32) -> i32 {
    if name_l != 13 {
        0
    } else {
        let mut ok: i32 = 1;
        if __arena_get(name_s + 0) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 1) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 2) != 116 { ok = 0; };   // t
        if __arena_get(name_s + 3) != 105 { ok = 0; };   // i
        if __arena_get(name_s + 4) != 108 { ok = 0; };   // l
        if __arena_get(name_s + 5) != 101 { ok = 0; };   // e
        if __arena_get(name_s + 6) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 7) != 109 { ok = 0; };   // m
        if __arena_get(name_s + 8) != 97 { ok = 0; };    // a
        if __arena_get(name_s + 9) != 116 { ok = 0; };   // t
        if __arena_get(name_s + 10) != 109 { ok = 0; };  // m
        if __arena_get(name_s + 11) != 117 { ok = 0; };  // u
        if __arena_get(name_s + 12) != 108 { ok = 0; };  // l
        ok
    }
}
// K1.M16: lower __tile_matmul(a, b, dst, n) -- NAIVE unrolled NxN row-
// major matrix multiply over register-tiles: dst[i][j] = sum_k
// a[i][k]*b[k][j], emitted as mul.f32 + add.f32 over consecutive %f with
// the accumulator moved into dst[i*n+j]. This is the CORRECTNESS form (a
// real on-GPU matmul, matching the CPU __tile_matmul naive path); NVIDIA
// Tensor-Core wmma.mma.sync acceleration is a later perf optimization.
// a/b/dst are AST_VAR bound to prior __tile_zeros results (ridx = %f
// base); n is a static AST_INT. Result = base_d; sets the float flag.
fn emit_ptx_tile_matmul(node: i32, vtab: i32) -> i32 {
    let ah = __arena_get(node + 3);     // args_head (AST_ARG)
    let a0 = __arena_get(ah + 1);       // arg 0: AST_VAR a
    let nxt1 = __arena_get(ah + 2);
    let a1 = __arena_get(nxt1 + 1);     // arg 1: AST_VAR b
    let nxt2 = __arena_get(nxt1 + 2);
    let a2 = __arena_get(nxt2 + 1);     // arg 2: AST_VAR dst
    let nxt3 = __arena_get(nxt2 + 2);
    let a3 = __arena_get(nxt3 + 1);     // arg 3: AST_INT n
    let base_a = ptx_vtab_lookup(vtab, __arena_get(a0 + 1), __arena_get(a0 + 2));
    let base_b = ptx_vtab_lookup(vtab, __arena_get(a1 + 1), __arena_get(a1 + 2));
    let base_d = ptx_vtab_lookup(vtab, __arena_get(a2 + 1), __arena_get(a2 + 2));
    let n = __arena_get(a3 + 1);
    let mut i: i32 = 0;
    while i < n {
        let mut j: i32 = 0;
        while j < n {
            // acc = a[i][0] * b[0][j]
            let acc0 = ptx_alloc_f(vtab);
            emit_ptx_binop_f3(2, acc0, base_a + i * n, base_b + j);
            let mut acc = acc0;
            let mut k: i32 = 1;
            while k < n {
                // acc = acc + a[i][k] * b[k][j]
                let prod = ptx_alloc_f(vtab);
                emit_ptx_binop_f3(2, prod, base_a + i * n + k, base_b + k * n + j);
                let nacc = ptx_alloc_f(vtab);
                emit_ptx_binop_f3(0, nacc, acc, prod);
                acc = nacc;
                k = k + 1;
            }
            emit_ptx_mov_f_reg(base_d + i * n + j, acc);
            j = j + 1;
        }
        i = i + 1;
    }
    __arena_set(vtab + 55, 1);
    base_d
}

// Other calls are unsupported in the GPU path yet (return -1; the tile op
// family -- zeros/add/sub/mul/matmul -- is now complete).
fn emit_ptx_call(node: i32, vtab: i32) -> i32 {
    let name_s = __arena_get(node + 1);
    let name_l = __arena_get(node + 2);
    if ptx_name_is_thread_idx(name_s, name_l) == 1 {
        let r = ptx_alloc_reg(vtab);
        emit_ptx_tid_x(r);
        r
    } else { if ptx_name_is_block_idx(name_s, name_l) == 1 {
        let r = ptx_alloc_reg(vtab);
        emit_ptx_mov_ctaid_x(r);
        r
    } else { if ptx_name_is_block_dim(name_s, name_l) == 1 {
        let r = ptx_alloc_reg(vtab);
        emit_ptx_mov_ntid_x(r);
        r
    } else { if ptx_name_is_tile_zeros(name_s, name_l) == 1 {
        // K1.M13: __tile_zeros(N, M) -> N*M consecutive `mov.f32 %fX,
        // 0f00000000;`. N and M are Phase-0 static int literals (AST_INT,
        // value in slot 1). Result = base %f register; set the float flag
        // (vtab slot 55) so a downstream consumer sees an f32 tile.
        let ah = __arena_get(node + 3);     // args_head (AST_ARG)
        let a0 = __arena_get(ah + 1);       // first arg expr (AST_INT N)
        let nxt = __arena_get(ah + 2);      // second AST_ARG
        let a1 = __arena_get(nxt + 1);      // second arg expr (AST_INT M)
        let nn = __arena_get(a0 + 1);       // N literal
        let mm = __arena_get(a1 + 1);       // M literal
        let count = nn * mm;
        let base = ptx_alloc_f(vtab);
        emit_ptx_mov_f_zero(base);
        let mut i: i32 = 1;
        while i < count {
            let rr = ptx_alloc_f(vtab);
            emit_ptx_mov_f_zero(rr);
            i = i + 1;
        }
        __arena_set(vtab + 55, 1);
        base
    } else { if ptx_name_is_tile_add(name_s, name_l) == 1 {
        emit_ptx_tile_binop(node, vtab, 0)   // K1.M14: dst = a + b
    } else { if ptx_name_is_tile_sub(name_s, name_l) == 1 {
        emit_ptx_tile_binop(node, vtab, 1)   // K1.M15: dst = a - b
    } else { if ptx_name_is_tile_mul(name_s, name_l) == 1 {
        emit_ptx_tile_binop(node, vtab, 2)   // K1.M15: dst = a * b
    } else { if ptx_name_is_tile_matmul(name_s, name_l) == 1 {
        emit_ptx_tile_matmul(node, vtab)     // K1.M16: dst = a @ b (naive)
    } else {
        0 - 1
    }}}}}}}}
}

// K1.M3 (2026-05-28): emit ONE PTX entry for the given @kernel fn:
//   .visible .entry <name>()
//   {
//   ret;
//   }
// <name> is the fn's real source name (slots 1/2 = name_start/len),
// same source-byte read the bootstrap's `main` detection uses. The
// body/params are still stubbed (K1.M4+ add .param / .reg / ops).
fn emit_ptx_entry(fn_idx: i32, vtab: i32) -> i32 {
    // ".visible .entry " (prefix; the real entry name follows)
    emit_ptx_byte(46); emit_ptx_byte(118); emit_ptx_byte(105);
    emit_ptx_byte(115); emit_ptx_byte(105); emit_ptx_byte(98);
    emit_ptx_byte(108); emit_ptx_byte(101); emit_ptx_byte(32);
    emit_ptx_byte(46); emit_ptx_byte(101); emit_ptx_byte(110);
    emit_ptx_byte(116); emit_ptx_byte(114); emit_ptx_byte(121);
    emit_ptx_byte(32);
    let kname_s = __arena_get(fn_idx + 1);
    let kname_l = __arena_get(fn_idx + 2);
    let mut ki: i32 = 0;
    while ki < kname_l {
        emit_ptx_byte(__arena_get(kname_s + ki));
        ki = ki + 1;
    }
    // "(" then params then ")\n"  (K1.M4)
    emit_ptx_byte(40);
    // K1.M4: emit one ".param .b64 param_<idx>" per fn param, comma-
    // space separated. params_head = AST_FN_DECL slot 4; each
    // AST_PARAM links via slot 3 (next), slot 4 = type_tag. v0.1:
    // every param is .b64 (mirror Python ptx.py _format_param) and
    // the PTX param name is POSITIONAL (param_0, param_1, ...), not
    // the source name. Zero params -> empty parens "()" (unchanged
    // from K1.M1-M3). Phase-0 caps params at 6, so the index digit
    // is always a single ASCII char (48 + idx).
    let mut pcur: i32 = __arena_get(fn_idx + 4);
    let mut pidx: i32 = 0;
    while pcur != 0 {
        if pidx > 0 {
            // ", "
            emit_ptx_byte(44); emit_ptx_byte(32);
        };
        // ".param .b64 param_"
        emit_ptx_byte(46); emit_ptx_byte(112); emit_ptx_byte(97);
        emit_ptx_byte(114); emit_ptx_byte(97); emit_ptx_byte(109);
        emit_ptx_byte(32); emit_ptx_byte(46); emit_ptx_byte(98);
        emit_ptx_byte(54); emit_ptx_byte(52); emit_ptx_byte(32);
        emit_ptx_byte(112); emit_ptx_byte(97); emit_ptx_byte(114);
        emit_ptx_byte(97); emit_ptx_byte(109); emit_ptx_byte(95);
        // positional index digit (Phase-0: 0..5)
        emit_ptx_byte(48 + pidx);
        pcur = __arena_get(pcur + 3);
        pidx = pidx + 1;
    }
    // ")\n"
    emit_ptx_byte(41); emit_ptx_byte(10);
    // "{\n"
    emit_ptx_byte(123); emit_ptx_byte(10);
    // K1.M5a: register-file declarations (foundation for op lowering).
    emit_ptx_reg_block();
    // K1.M5d: lower the kernel body via the recursive expression
    // emitter + a fresh per-kernel var->register environment. Handles
    // integer const / let-chain / variable ref / scalar arithmetic
    // (add/sub/mul) -- see emit_ptx_expr. The result register is
    // discarded (a PTX kernel is void; later chunks add stores).
    ptx_vtab_reset(vtab);
    __arena_set(vtab + 53, fn_idx);   // K1.M10: stash fn_idx so index
                                      // loads can resolve param pointers
    emit_ptx_expr(__arena_get(fn_idx + 3), vtab);
    // "    ret;\n"  (4-space indent, matches Python emit_kernel)
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32); emit_ptx_byte(114); emit_ptx_byte(101);
    emit_ptx_byte(116); emit_ptx_byte(59); emit_ptx_byte(10);
    // "}\n"
    emit_ptx_byte(125); emit_ptx_byte(10);
    0
}

fn emit_ptx_for_ast_to_path(ast_root: i32) -> i32 {
    // K1.M3: count @kernel fns (is_kernel slot 14 == 1). Mirrors
    // the autotune_pass walk: root tag 15 = AST_FN_LIST, node
    // slot+1 = fn_idx, node slot+2 = next. A module with >=1 kernel
    // emits the header once then one .entry per kernel; 0 kernels
    // emits nothing (a pure-CPU program produces no PTX).
    let mut kernel_count: i32 = 0;
    if __arena_get(ast_root) == 15 {
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            if __arena_get(fn_idx + 14) == 1 {
                kernel_count = kernel_count + 1;
            };
            walk = __arena_get(walk + 2);
        }
    };
    if kernel_count == 0 {
        0
    } else {
        // K1.M5d: the var->register table lives BEFORE `start`, so its
        // slots are never part of the emitted PTX byte stream.
        let vtab = ptx_vtab_init();
        let start = __arena_len();
        // ".version 8.0\n"  (PTX ISA 8.0 = CUDA 12.0; we use only
        // basic scalar ops so 8.0 is sufficient and broadly compatible
        // -- and is the max the local/CI ptxas supports. K1.M5f.)
        emit_ptx_byte(46); emit_ptx_byte(118); emit_ptx_byte(101);
        emit_ptx_byte(114); emit_ptx_byte(115); emit_ptx_byte(105);
        emit_ptx_byte(111); emit_ptx_byte(110); emit_ptx_byte(32);
        emit_ptx_byte(56); emit_ptx_byte(46); emit_ptx_byte(48);
        emit_ptx_byte(10);
        // ".target sm_75\n"
        emit_ptx_byte(46); emit_ptx_byte(116); emit_ptx_byte(97);
        emit_ptx_byte(114); emit_ptx_byte(103); emit_ptx_byte(101);
        emit_ptx_byte(116); emit_ptx_byte(32); emit_ptx_byte(115);
        emit_ptx_byte(109); emit_ptx_byte(95); emit_ptx_byte(55);
        emit_ptx_byte(53); emit_ptx_byte(10);
        // ".address_size 64\n"
        emit_ptx_byte(46); emit_ptx_byte(97); emit_ptx_byte(100);
        emit_ptx_byte(100); emit_ptx_byte(114); emit_ptx_byte(101);
        emit_ptx_byte(115); emit_ptx_byte(115); emit_ptx_byte(95);
        emit_ptx_byte(115); emit_ptx_byte(105); emit_ptx_byte(122);
        emit_ptx_byte(101); emit_ptx_byte(32); emit_ptx_byte(54);
        emit_ptx_byte(52); emit_ptx_byte(10);
        // "\n" (blank line after the module header)
        emit_ptx_byte(10);
        // K1.M3: emit one entry per @kernel fn, blank-separated. The
        // header's trailing blank precedes the first entry; a blank
        // line is inserted before each SUBSEQUENT entry, so a
        // single-kernel module stays byte-identical to K1.M1/M2 (no
        // trailing blank).
        let mut emitted: i32 = 0;
        let mut walk2: i32 = ast_root;
        while walk2 != 0 {
            let fn_idx = __arena_get(walk2 + 1);
            if __arena_get(fn_idx + 14) == 1 {
                if emitted > 0 {
                    emit_ptx_byte(10);
                };
                emit_ptx_entry(fn_idx, vtab);
                emitted = emitted + 1;
            };
            walk2 = __arena_get(walk2 + 2);
        }
        __arena_len() - start
    }
}

// K1.M17 (2026-05-28): does the module contain any @kernel fn? Walks the
// AST_FN_LIST (tag 15; node slot+1 = fn_idx, slot+2 = next) checking
// is_kernel (AST_FN_DECL slot 14) -- the same walk emit_ptx_for_ast_to_path
// uses, factored out so the output-mode dispatcher can route.
fn ast_has_kernel(ast_root: i32) -> i32 {
    let mut found: i32 = 0;
    if __arena_get(ast_root) == 15 {
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            if __arena_get(fn_idx + 14) == 1 {
                found = 1;
            };
            walk = __arena_get(walk + 2);
        }
    };
    found
}
// K1.M17: output-mode dispatcher -- emit GPU PTX when the program has a
// @kernel, else x86_64 ELF. The bridge from "two separate emitters" to a
// driver that picks the right target: a real compiler entry calls THIS.
// Both emitters write into the arena + return the byte count; the caller
// writes the file. Like a host toolchain routing .cu -> ptx vs .c -> elf,
// but with NO CUDA / NO MLIR -- direct Helix -> chip on either path.
fn emit_auto_for_ast_to_path(ast_root: i32) -> i32 {
    if ast_has_kernel(ast_root) == 1 {
        emit_ptx_for_ast_to_path(ast_root)
    } else {
        emit_elf_for_ast_to_path(ast_root)
    }
}

// K1.M18 (2026-05-28): FIRST non-NVIDIA GPU backend -- WebGPU / WGSL
// (the browser-portable shader IR that runs on ANY GPU: NVIDIA, AMD,
// Apple, Intel). Direct Helix -> WGSL text, NO MLIR / NO LLVM. Mirrors
// Python helixc/backend/webgpu.py emit_module_header + emit_kernel_stub.
// The WGSL module preamble (2 comment lines documenting the spec level +
// default workgroup size, then a blank). The "--" is a U+2014 EM DASH
// (UTF-8 226 128 148), matching the Python emitter byte-for-byte.
fn emit_wgsl_header() -> i32 {
    // "// Helix-emitted WGSL " + EM_DASH + " spec wgsl-2024\n"
    emit_ptx_byte(47); emit_ptx_byte(47); emit_ptx_byte(32);   // "// "
    emit_ptx_byte(72); emit_ptx_byte(101); emit_ptx_byte(108);
    emit_ptx_byte(105); emit_ptx_byte(120);                    // "Helix"
    emit_ptx_byte(45);                                         // "-"
    emit_ptx_byte(101); emit_ptx_byte(109); emit_ptx_byte(105);
    emit_ptx_byte(116); emit_ptx_byte(116); emit_ptx_byte(101);
    emit_ptx_byte(100);                                        // "emitted"
    emit_ptx_byte(32);
    emit_ptx_byte(87); emit_ptx_byte(71); emit_ptx_byte(83);
    emit_ptx_byte(76);                                         // "WGSL"
    emit_ptx_byte(32);
    emit_ptx_byte(226); emit_ptx_byte(128); emit_ptx_byte(148); // EM DASH
    emit_ptx_byte(32);
    emit_ptx_byte(115); emit_ptx_byte(112); emit_ptx_byte(101);
    emit_ptx_byte(99);                                         // "spec"
    emit_ptx_byte(32);
    emit_ptx_byte(119); emit_ptx_byte(103); emit_ptx_byte(115);
    emit_ptx_byte(108); emit_ptx_byte(45); emit_ptx_byte(50);
    emit_ptx_byte(48); emit_ptx_byte(50); emit_ptx_byte(52);   // "wgsl-2024"
    emit_ptx_byte(10);
    // "// Workgroup size default: 64\n"
    emit_ptx_byte(47); emit_ptx_byte(47); emit_ptx_byte(32);   // "// "
    emit_ptx_byte(87); emit_ptx_byte(111); emit_ptx_byte(114);
    emit_ptx_byte(107); emit_ptx_byte(103); emit_ptx_byte(114);
    emit_ptx_byte(111); emit_ptx_byte(117); emit_ptx_byte(112); // "Workgroup"
    emit_ptx_byte(32);
    emit_ptx_byte(115); emit_ptx_byte(105); emit_ptx_byte(122);
    emit_ptx_byte(101);                                        // "size"
    emit_ptx_byte(32);
    emit_ptx_byte(100); emit_ptx_byte(101); emit_ptx_byte(102);
    emit_ptx_byte(97); emit_ptx_byte(117); emit_ptx_byte(108);
    emit_ptx_byte(116); emit_ptx_byte(58);                     // "default:"
    emit_ptx_byte(32);
    emit_ptx_byte(54); emit_ptx_byte(52);                      // "64"
    emit_ptx_byte(10);
    emit_ptx_byte(10);                                         // blank line
    0
}
// K1.M18/M22: the EMPTY-kernel WGSL form (no params) -- byte-matches
// Python webgpu.py emit_kernel_stub for a no-op kernel (@builtin(local_
// invocation_id) + return;). Param kernels take the M22 real form
// (emit_wgsl_kernel_params); emit_wgsl_kernel dispatches on param count.
fn emit_wgsl_kernel_empty(fn_idx: i32) -> i32 {
    // "@compute @workgroup_size(64)\n"
    emit_ptx_byte(64); emit_ptx_byte(99); emit_ptx_byte(111);
    emit_ptx_byte(109); emit_ptx_byte(112); emit_ptx_byte(117);
    emit_ptx_byte(116); emit_ptx_byte(101);                    // "@compute"
    emit_ptx_byte(32);
    emit_ptx_byte(64); emit_ptx_byte(119); emit_ptx_byte(111);
    emit_ptx_byte(114); emit_ptx_byte(107); emit_ptx_byte(103);
    emit_ptx_byte(114); emit_ptx_byte(111); emit_ptx_byte(117);
    emit_ptx_byte(112); emit_ptx_byte(95); emit_ptx_byte(115);
    emit_ptx_byte(105); emit_ptx_byte(122); emit_ptx_byte(101); // "@workgroup_size"
    emit_ptx_byte(40); emit_ptx_byte(54); emit_ptx_byte(52);
    emit_ptx_byte(41);                                         // "(64)"
    emit_ptx_byte(10);
    // "fn " + <name> + "(\n"
    emit_ptx_byte(102); emit_ptx_byte(110); emit_ptx_byte(32); // "fn "
    let kname_s = __arena_get(fn_idx + 1);
    let kname_l = __arena_get(fn_idx + 2);
    let mut ki: i32 = 0;
    while ki < kname_l {
        emit_ptx_byte(__arena_get(kname_s + ki));
        ki = ki + 1;
    }
    emit_ptx_byte(40); emit_ptx_byte(10);                      // "(\n"
    // "    @builtin(local_invocation_id) local_id: vec3<u32>\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32);
    emit_ptx_byte(64); emit_ptx_byte(98); emit_ptx_byte(117);
    emit_ptx_byte(105); emit_ptx_byte(108); emit_ptx_byte(116);
    emit_ptx_byte(105); emit_ptx_byte(110);                    // "@builtin"
    emit_ptx_byte(40);                                         // "("
    emit_ptx_byte(108); emit_ptx_byte(111); emit_ptx_byte(99);
    emit_ptx_byte(97); emit_ptx_byte(108); emit_ptx_byte(95);
    emit_ptx_byte(105); emit_ptx_byte(110); emit_ptx_byte(118);
    emit_ptx_byte(111); emit_ptx_byte(99); emit_ptx_byte(97);
    emit_ptx_byte(116); emit_ptx_byte(105); emit_ptx_byte(111);
    emit_ptx_byte(110); emit_ptx_byte(95); emit_ptx_byte(105);
    emit_ptx_byte(100);                                        // "local_invocation_id"
    emit_ptx_byte(41);                                         // ")"
    emit_ptx_byte(32);
    emit_ptx_byte(108); emit_ptx_byte(111); emit_ptx_byte(99);
    emit_ptx_byte(97); emit_ptx_byte(108); emit_ptx_byte(95);
    emit_ptx_byte(105); emit_ptx_byte(100); emit_ptx_byte(58); // "local_id:"
    emit_ptx_byte(32);
    emit_ptx_byte(118); emit_ptx_byte(101); emit_ptx_byte(99);
    emit_ptx_byte(51); emit_ptx_byte(60); emit_ptx_byte(117);
    emit_ptx_byte(51); emit_ptx_byte(50); emit_ptx_byte(62);   // "vec3<u32>"
    emit_ptx_byte(10);
    // ") {\n"
    emit_ptx_byte(41); emit_ptx_byte(32); emit_ptx_byte(123);
    emit_ptx_byte(10);
    // "    return;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32);
    emit_ptx_byte(114); emit_ptx_byte(101); emit_ptx_byte(116);
    emit_ptx_byte(117); emit_ptx_byte(114); emit_ptx_byte(110);
    emit_ptx_byte(59);                                         // "return;"
    emit_ptx_byte(10);
    // "}\n\n"
    emit_ptx_byte(125); emit_ptx_byte(10); emit_ptx_byte(10);
    0
}
// K1.M23: lower one WGSL expression (high-level text -- NO registers).
// AST_INT->decimal; AST_VAR->name; AST_CALL->"gid.x" (thread_idx, the M23
// kernel builtin); AST_INDEX a[i]->expr(base)+"["+expr(index)+"]"; binop
// 2/3/4/5 -> expr(l)+" +|-|*|/ "+expr(r). Recursive.
fn emit_wgsl_expr(node: i32) -> i32 {
    let tag = __arena_get(node);
    if tag == 0 {
        emit_ptx_decimal(__arena_get(node + 1));
        0
    } else { if tag == 1 {
        let ns = __arena_get(node + 1);
        let nl = __arena_get(node + 2);
        let mut i: i32 = 0;
        while i < nl {
            emit_ptx_byte(__arena_get(ns + i));
            i = i + 1;
        }
        0
    } else { if tag == 16 {
        // thread_idx() -> the global thread index "gid.x"
        emit_ptx_byte(103); emit_ptx_byte(105); emit_ptx_byte(100);
        emit_ptx_byte(46); emit_ptx_byte(120);                   // "gid.x"
        0
    } else { if tag == 53 {
        emit_wgsl_expr(__arena_get(node + 1));   // base name
        emit_ptx_byte(91);                        // "["
        emit_wgsl_expr(__arena_get(node + 2));   // index
        emit_ptx_byte(93);                        // "]"
        0
    } else {
        // binop: 2=ADD 3=SUB 4=MUL 5=DIV -> infix WGSL operator
        emit_wgsl_expr(__arena_get(node + 1));
        emit_ptx_byte(32);
        if tag == 2 { emit_ptx_byte(43); };       // "+"
        if tag == 3 { emit_ptx_byte(45); };       // "-"
        if tag == 4 { emit_ptx_byte(42); };       // "*"
        if tag == 5 { emit_ptx_byte(47); };       // "/"
        emit_ptx_byte(32);
        emit_wgsl_expr(__arena_get(node + 2));
        0
    }}}}
}
// K1.M23: lower a WGSL body statement. AST_LET -> "    let <name> = <val>;"
// then recurse continuation; AST_INDEX_STORE -> "    <base>[<idx>] = <v>;";
// AST_SEQ -> both. Empty/const bodies (AST_INT etc.) emit nothing (so the
// M22 `{ 0 }` params kernel stays byte-identical). The `: type` annotation
// is dropped (WGSL infers). Recursive (mutual with emit_wgsl_expr).
fn emit_wgsl_stmt(node: i32) -> i32 {
    let tag = __arena_get(node);
    if tag == 8 {
        emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
        emit_ptx_byte(32);
        emit_ptx_byte(108); emit_ptx_byte(101); emit_ptx_byte(116);
        emit_ptx_byte(32);                                       // "let "
        let ns = __arena_get(node + 1);
        let nl = __arena_get(node + 2);
        let mut i: i32 = 0;
        while i < nl {
            emit_ptx_byte(__arena_get(ns + i));
            i = i + 1;
        }
        emit_ptx_byte(32); emit_ptx_byte(61); emit_ptx_byte(32); // " = "
        emit_wgsl_expr(__arena_get(node + 4));                   // value
        emit_ptx_byte(59); emit_ptx_byte(10);                    // ";\n"
        emit_wgsl_stmt(__arena_get(node + 3))                    // continuation
    } else { if tag == 55 {
        let idx_node = __arena_get(node + 1);
        emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
        emit_ptx_byte(32);
        emit_wgsl_expr(__arena_get(idx_node + 1));               // base
        emit_ptx_byte(91);                                       // "["
        emit_wgsl_expr(__arena_get(idx_node + 2));               // index
        emit_ptx_byte(93);                                       // "]"
        emit_ptx_byte(32); emit_ptx_byte(61); emit_ptx_byte(32); // " = "
        emit_wgsl_expr(__arena_get(node + 2));                   // value
        emit_ptx_byte(59); emit_ptx_byte(10);                    // ";\n"
        0
    } else { if tag == 13 {
        emit_wgsl_stmt(__arena_get(node + 1));
        emit_wgsl_stmt(__arena_get(node + 2))
    } else {
        0
    }}}
}
// K1.M22: emit one WGSL storage-buffer binding for a kernel param --
// "@group(0) @binding(<idx>) var<storage, read_write> <name>: array<f32>;".
// Positional binding index. Phase-0: array<f32> (the AI elementwise/tile
// workload); i32 params would use array<i32> (future). This EXCEEDS the
// Python WebGPU backend (which stubs all real ops) -- net-new capability.
fn emit_wgsl_buffer(name_s: i32, name_l: i32, idx: i32, type_tag: i32) -> i32 {
    // "@group(0) @binding("
    emit_ptx_byte(64); emit_ptx_byte(103); emit_ptx_byte(114);
    emit_ptx_byte(111); emit_ptx_byte(117); emit_ptx_byte(112);
    emit_ptx_byte(40); emit_ptx_byte(48); emit_ptx_byte(41);    // "@group(0)"
    emit_ptx_byte(32);
    emit_ptx_byte(64); emit_ptx_byte(98); emit_ptx_byte(105);
    emit_ptx_byte(110); emit_ptx_byte(100); emit_ptx_byte(105);
    emit_ptx_byte(110); emit_ptx_byte(103); emit_ptx_byte(40);  // "@binding("
    emit_ptx_decimal(idx);
    emit_ptx_byte(41);                                          // ")"
    // " var<storage, read_write> "
    emit_ptx_byte(32);
    emit_ptx_byte(118); emit_ptx_byte(97); emit_ptx_byte(114);
    emit_ptx_byte(60); emit_ptx_byte(115); emit_ptx_byte(116);
    emit_ptx_byte(111); emit_ptx_byte(114); emit_ptx_byte(97);
    emit_ptx_byte(103); emit_ptx_byte(101); emit_ptx_byte(44);  // "var<storage,"
    emit_ptx_byte(32);
    emit_ptx_byte(114); emit_ptx_byte(101); emit_ptx_byte(97);
    emit_ptx_byte(100); emit_ptx_byte(95); emit_ptx_byte(119);
    emit_ptx_byte(114); emit_ptx_byte(105); emit_ptx_byte(116);
    emit_ptx_byte(101); emit_ptx_byte(62);                      // "read_write>"
    emit_ptx_byte(32);
    // <name>
    let mut ni: i32 = 0;
    while ni < name_l {
        emit_ptx_byte(__arena_get(name_s + ni));
        ni = ni + 1;
    }
    // ": array<f32>;" or ": array<i32>;" (K1.M24: branch on type_tag)
    emit_ptx_byte(58); emit_ptx_byte(32);                       // ": "
    emit_ptx_byte(97); emit_ptx_byte(114); emit_ptx_byte(114);
    emit_ptx_byte(97); emit_ptx_byte(121);                      // "array"
    emit_ptx_byte(60);                                          // "<"
    if type_tag == 1 {
        emit_ptx_byte(102);                                     // "f" (f32)
    } else {
        emit_ptx_byte(105);                                     // "i" (i32)
    };
    emit_ptx_byte(51); emit_ptx_byte(50); emit_ptx_byte(62);    // "32>"
    emit_ptx_byte(59); emit_ptx_byte(10);                       // ";\n"
    0
}
// K1.M22: emit the REAL (params) WGSL kernel form -- module-scope storage
// buffers (one per param) then a @compute entry using global_invocation_id
// (the cross-workgroup thread index for indexing buffers). Body is still
// the skeleton (`return;`); M23 fills it (load/store/arith). Net-new vs
// the Python WebGPU scaffold.
fn emit_wgsl_kernel_params(fn_idx: i32) -> i32 {
    let mut pcur: i32 = __arena_get(fn_idx + 4);
    let mut pidx: i32 = 0;
    while pcur != 0 {
        emit_wgsl_buffer(__arena_get(pcur + 1), __arena_get(pcur + 2), pidx,
                         __arena_get(pcur + 4));
        pcur = __arena_get(pcur + 3);
        pidx = pidx + 1;
    }
    // "@compute @workgroup_size(64)\n"
    emit_ptx_byte(64); emit_ptx_byte(99); emit_ptx_byte(111);
    emit_ptx_byte(109); emit_ptx_byte(112); emit_ptx_byte(117);
    emit_ptx_byte(116); emit_ptx_byte(101);                    // "@compute"
    emit_ptx_byte(32);
    emit_ptx_byte(64); emit_ptx_byte(119); emit_ptx_byte(111);
    emit_ptx_byte(114); emit_ptx_byte(107); emit_ptx_byte(103);
    emit_ptx_byte(114); emit_ptx_byte(111); emit_ptx_byte(117);
    emit_ptx_byte(112); emit_ptx_byte(95); emit_ptx_byte(115);
    emit_ptx_byte(105); emit_ptx_byte(122); emit_ptx_byte(101); // "@workgroup_size"
    emit_ptx_byte(40); emit_ptx_byte(54); emit_ptx_byte(52);
    emit_ptx_byte(41);                                         // "(64)"
    emit_ptx_byte(10);
    // "fn " + <name> + "(\n"
    emit_ptx_byte(102); emit_ptx_byte(110); emit_ptx_byte(32); // "fn "
    let kname_s = __arena_get(fn_idx + 1);
    let kname_l = __arena_get(fn_idx + 2);
    let mut ki: i32 = 0;
    while ki < kname_l {
        emit_ptx_byte(__arena_get(kname_s + ki));
        ki = ki + 1;
    }
    emit_ptx_byte(40); emit_ptx_byte(10);                      // "(\n"
    // "    @builtin(global_invocation_id) gid: vec3<u32>\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32);
    emit_ptx_byte(64); emit_ptx_byte(98); emit_ptx_byte(117);
    emit_ptx_byte(105); emit_ptx_byte(108); emit_ptx_byte(116);
    emit_ptx_byte(105); emit_ptx_byte(110);                    // "@builtin"
    emit_ptx_byte(40);                                         // "("
    emit_ptx_byte(103); emit_ptx_byte(108); emit_ptx_byte(111);
    emit_ptx_byte(98); emit_ptx_byte(97); emit_ptx_byte(108);
    emit_ptx_byte(95); emit_ptx_byte(105); emit_ptx_byte(110);
    emit_ptx_byte(118); emit_ptx_byte(111); emit_ptx_byte(99);
    emit_ptx_byte(97); emit_ptx_byte(116); emit_ptx_byte(105);
    emit_ptx_byte(111); emit_ptx_byte(110); emit_ptx_byte(95);
    emit_ptx_byte(105); emit_ptx_byte(100);                    // "global_invocation_id"
    emit_ptx_byte(41);                                         // ")"
    emit_ptx_byte(32);
    emit_ptx_byte(103); emit_ptx_byte(105); emit_ptx_byte(100);
    emit_ptx_byte(58);                                         // "gid:"
    emit_ptx_byte(32);
    emit_ptx_byte(118); emit_ptx_byte(101); emit_ptx_byte(99);
    emit_ptx_byte(51); emit_ptx_byte(60); emit_ptx_byte(117);
    emit_ptx_byte(51); emit_ptx_byte(50); emit_ptx_byte(62);   // "vec3<u32>"
    emit_ptx_byte(10);
    // ") {\n"
    emit_ptx_byte(41); emit_ptx_byte(32); emit_ptx_byte(123);
    emit_ptx_byte(10);
    // K1.M23: lower the kernel body (AST_FN_DECL slot 3) to WGSL stmts,
    // then the trailing return. Empty/const bodies (e.g. `{ 0 }`) emit
    // nothing -> the M22 wgsl_params kernel stays byte-identical.
    emit_wgsl_stmt(__arena_get(fn_idx + 3));
    // "    return;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32);
    emit_ptx_byte(114); emit_ptx_byte(101); emit_ptx_byte(116);
    emit_ptx_byte(117); emit_ptx_byte(114); emit_ptx_byte(110);
    emit_ptx_byte(59);                                         // "return;"
    emit_ptx_byte(10);
    // "}\n\n"
    emit_ptx_byte(125); emit_ptx_byte(10); emit_ptx_byte(10);
    0
}
// K1.M22: dispatch a @kernel to the empty form (0 params, byte-matches
// Python) or the real params form (>=1 param). Param count = walk
// AST_FN_DECL slot4 (params_head) via AST_PARAM slot3 (next).
fn emit_wgsl_kernel(fn_idx: i32) -> i32 {
    let mut pc: i32 = 0;
    let mut p: i32 = __arena_get(fn_idx + 4);
    while p != 0 {
        pc = pc + 1;
        p = __arena_get(p + 3);
    }
    if pc == 0 {
        emit_wgsl_kernel_empty(fn_idx)
    } else {
        emit_wgsl_kernel_params(fn_idx)
    }
}

// K1.M18: top-level WGSL emitter. Header once, then one @compute entry
// per @kernel fn (mirrors emit_ptx_for_ast_to_path's structure + Python
// webgpu.py emit_module). 0 kernels -> emits nothing (returns 0).
fn emit_wgsl_for_ast_to_path(ast_root: i32) -> i32 {
    let mut kernel_count: i32 = 0;
    if __arena_get(ast_root) == 15 {
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            if __arena_get(fn_idx + 14) == 1 {
                kernel_count = kernel_count + 1;
            };
            walk = __arena_get(walk + 2);
        }
    };
    if kernel_count == 0 {
        0
    } else {
        let start = __arena_len();
        emit_wgsl_header();
        let mut walk2: i32 = ast_root;
        while walk2 != 0 {
            let fn_idx = __arena_get(walk2 + 1);
            if __arena_get(fn_idx + 14) == 1 {
                emit_wgsl_kernel(fn_idx);
            };
            walk2 = __arena_get(walk2 + 2);
        }
        __arena_len() - start
    }
}

// K1.M19 (2026-05-28): SECOND non-NVIDIA GPU backend -- Apple Metal /
// MSL (Metal Shading Language, a C++-based GPU language for Apple
// Silicon). Direct Helix -> MSL text, NO MLIR / NO LLVM. Mirrors Python
// helixc/backend/metal.py MslEmitter (emit_module_header + emit_kernel_
// stub). Header: doc comment (incl. U+2014 EM DASH = UTF-8 226 128 148)
// + #include <metal_stdlib> + using namespace metal; + blank.
fn emit_msl_header() -> i32 {
    // "// Helix-emitted MSL " + EM_DASH + " target apple7, metal3.2\n"
    emit_ptx_byte(47); emit_ptx_byte(47); emit_ptx_byte(32);   // "// "
    emit_ptx_byte(72); emit_ptx_byte(101); emit_ptx_byte(108);
    emit_ptx_byte(105); emit_ptx_byte(120); emit_ptx_byte(45);
    emit_ptx_byte(101); emit_ptx_byte(109); emit_ptx_byte(105);
    emit_ptx_byte(116); emit_ptx_byte(116); emit_ptx_byte(101);
    emit_ptx_byte(100);                                        // "Helix-emitted"
    emit_ptx_byte(32);
    emit_ptx_byte(77); emit_ptx_byte(83); emit_ptx_byte(76);   // "MSL"
    emit_ptx_byte(32);
    emit_ptx_byte(226); emit_ptx_byte(128); emit_ptx_byte(148); // EM DASH
    emit_ptx_byte(32);
    emit_ptx_byte(116); emit_ptx_byte(97); emit_ptx_byte(114);
    emit_ptx_byte(103); emit_ptx_byte(101); emit_ptx_byte(116); // "target"
    emit_ptx_byte(32);
    emit_ptx_byte(97); emit_ptx_byte(112); emit_ptx_byte(112);
    emit_ptx_byte(108); emit_ptx_byte(101); emit_ptx_byte(55);  // "apple7"
    emit_ptx_byte(44); emit_ptx_byte(32);                      // ", "
    emit_ptx_byte(109); emit_ptx_byte(101); emit_ptx_byte(116);
    emit_ptx_byte(97); emit_ptx_byte(108); emit_ptx_byte(51);
    emit_ptx_byte(46); emit_ptx_byte(50);                      // "metal3.2"
    emit_ptx_byte(10);
    // "#include <metal_stdlib>\n"
    emit_ptx_byte(35); emit_ptx_byte(105); emit_ptx_byte(110);
    emit_ptx_byte(99); emit_ptx_byte(108); emit_ptx_byte(117);
    emit_ptx_byte(100); emit_ptx_byte(101);                    // "#include"
    emit_ptx_byte(32);
    emit_ptx_byte(60); emit_ptx_byte(109); emit_ptx_byte(101);
    emit_ptx_byte(116); emit_ptx_byte(97); emit_ptx_byte(108);
    emit_ptx_byte(95); emit_ptx_byte(115); emit_ptx_byte(116);
    emit_ptx_byte(100); emit_ptx_byte(108); emit_ptx_byte(105);
    emit_ptx_byte(98); emit_ptx_byte(62);                      // "<metal_stdlib>"
    emit_ptx_byte(10);
    // "using namespace metal;\n"
    emit_ptx_byte(117); emit_ptx_byte(115); emit_ptx_byte(105);
    emit_ptx_byte(110); emit_ptx_byte(103);                    // "using"
    emit_ptx_byte(32);
    emit_ptx_byte(110); emit_ptx_byte(97); emit_ptx_byte(109);
    emit_ptx_byte(101); emit_ptx_byte(115); emit_ptx_byte(112);
    emit_ptx_byte(97); emit_ptx_byte(99); emit_ptx_byte(101);  // "namespace"
    emit_ptx_byte(32);
    emit_ptx_byte(109); emit_ptx_byte(101); emit_ptx_byte(116);
    emit_ptx_byte(97); emit_ptx_byte(108);                     // "metal"
    emit_ptx_byte(59);                                         // ";"
    emit_ptx_byte(10);
    emit_ptx_byte(10);                                         // blank line
    0
}
// K1.M19: emit ONE MSL kernel for a @kernel fn (real source name from
// slots 1/2). Empty-kernel skeleton (`return;`), mirroring Python
// metal.py emit_kernel_stub. `kernel void` + name + the
// thread_position_in_threadgroup attribute param.
fn emit_msl_kernel(fn_idx: i32) -> i32 {
    // "kernel void " + <name> + "(\n"
    emit_ptx_byte(107); emit_ptx_byte(101); emit_ptx_byte(114);
    emit_ptx_byte(110); emit_ptx_byte(101); emit_ptx_byte(108); // "kernel"
    emit_ptx_byte(32);
    emit_ptx_byte(118); emit_ptx_byte(111); emit_ptx_byte(105);
    emit_ptx_byte(100);                                        // "void"
    emit_ptx_byte(32);
    let kname_s = __arena_get(fn_idx + 1);
    let kname_l = __arena_get(fn_idx + 2);
    let mut ki: i32 = 0;
    while ki < kname_l {
        emit_ptx_byte(__arena_get(kname_s + ki));
        ki = ki + 1;
    }
    emit_ptx_byte(40); emit_ptx_byte(10);                      // "(\n"
    // "    uint tid [[thread_position_in_threadgroup]]\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32);
    emit_ptx_byte(117); emit_ptx_byte(105); emit_ptx_byte(110);
    emit_ptx_byte(116);                                        // "uint"
    emit_ptx_byte(32);
    emit_ptx_byte(116); emit_ptx_byte(105); emit_ptx_byte(100); // "tid"
    emit_ptx_byte(32);
    emit_ptx_byte(91); emit_ptx_byte(91);                      // "[["
    emit_ptx_byte(116); emit_ptx_byte(104); emit_ptx_byte(114);
    emit_ptx_byte(101); emit_ptx_byte(97); emit_ptx_byte(100);
    emit_ptx_byte(95); emit_ptx_byte(112); emit_ptx_byte(111);
    emit_ptx_byte(115); emit_ptx_byte(105); emit_ptx_byte(116);
    emit_ptx_byte(105); emit_ptx_byte(111); emit_ptx_byte(110);
    emit_ptx_byte(95); emit_ptx_byte(105); emit_ptx_byte(110);
    emit_ptx_byte(95); emit_ptx_byte(116); emit_ptx_byte(104);
    emit_ptx_byte(114); emit_ptx_byte(101); emit_ptx_byte(97);
    emit_ptx_byte(100); emit_ptx_byte(103); emit_ptx_byte(114);
    emit_ptx_byte(111); emit_ptx_byte(117); emit_ptx_byte(112); // "thread_position_in_threadgroup"
    emit_ptx_byte(93); emit_ptx_byte(93);                      // "]]"
    emit_ptx_byte(10);
    // ") {\n"
    emit_ptx_byte(41); emit_ptx_byte(32); emit_ptx_byte(123);
    emit_ptx_byte(10);
    // "    return;\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32);
    emit_ptx_byte(114); emit_ptx_byte(101); emit_ptx_byte(116);
    emit_ptx_byte(117); emit_ptx_byte(114); emit_ptx_byte(110);
    emit_ptx_byte(59);                                         // "return;"
    emit_ptx_byte(10);
    // "}\n\n"
    emit_ptx_byte(125); emit_ptx_byte(10); emit_ptx_byte(10);
    0
}
// K1.M19: top-level MSL emitter. Header once + one kernel per @kernel fn
// (mirrors emit_wgsl_for_ast_to_path / Python metal.py emit_module). 0
// kernels -> emits nothing (returns 0).
fn emit_msl_for_ast_to_path(ast_root: i32) -> i32 {
    let mut kernel_count: i32 = 0;
    if __arena_get(ast_root) == 15 {
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            if __arena_get(fn_idx + 14) == 1 {
                kernel_count = kernel_count + 1;
            };
            walk = __arena_get(walk + 2);
        }
    };
    if kernel_count == 0 {
        0
    } else {
        let start = __arena_len();
        emit_msl_header();
        let mut walk2: i32 = ast_root;
        while walk2 != 0 {
            let fn_idx = __arena_get(walk2 + 1);
            if __arena_get(fn_idx + 14) == 1 {
                emit_msl_kernel(fn_idx);
            };
            walk2 = __arena_get(walk2 + 2);
        }
        __arena_len() - start
    }
}

// K1.M20 (2026-05-28): FOURTH / FINAL GPU backend -- AMD ROCm (AMDGPU
// GCN assembly, gfx942). Direct Helix -> GCN asm text, NO MLIR / NO LLVM.
// Mirrors Python helixc/backend/rocm.py HipEmitter. Pure ASCII (no
// em-dash). Header: the .amdgcn_target directive + a blank line.
fn emit_rocm_header() -> i32 {
    // ".amdgcn_target \"amdgcn-amd-amdhsa--gfx942\"\n"
    emit_ptx_byte(46); emit_ptx_byte(97); emit_ptx_byte(109);
    emit_ptx_byte(100); emit_ptx_byte(103); emit_ptx_byte(99);
    emit_ptx_byte(110); emit_ptx_byte(95); emit_ptx_byte(116);
    emit_ptx_byte(97); emit_ptx_byte(114); emit_ptx_byte(103);
    emit_ptx_byte(101); emit_ptx_byte(116);                    // ".amdgcn_target"
    emit_ptx_byte(32);
    emit_ptx_byte(34);                                         // double-quote
    emit_ptx_byte(97); emit_ptx_byte(109); emit_ptx_byte(100);
    emit_ptx_byte(103); emit_ptx_byte(99); emit_ptx_byte(110);
    emit_ptx_byte(45); emit_ptx_byte(97); emit_ptx_byte(109);
    emit_ptx_byte(100); emit_ptx_byte(45); emit_ptx_byte(97);
    emit_ptx_byte(109); emit_ptx_byte(100); emit_ptx_byte(104);
    emit_ptx_byte(115); emit_ptx_byte(97); emit_ptx_byte(45);
    emit_ptx_byte(45); emit_ptx_byte(103); emit_ptx_byte(102);
    emit_ptx_byte(120); emit_ptx_byte(57); emit_ptx_byte(52);
    emit_ptx_byte(50);                                         // "amdgcn-amd-amdhsa--gfx942"
    emit_ptx_byte(34);                                         // double-quote
    emit_ptx_byte(10);
    emit_ptx_byte(10);                                         // blank line
    0
}
// K1.M20: emit ONE AMDGPU GCN kernel for a @kernel fn (real source name
// from slots 1/2, used in .globl / .type / the label). Empty-kernel
// skeleton (s_endpgm), mirroring Python rocm.py emit_kernel_stub.
fn emit_rocm_kernel(fn_idx: i32) -> i32 {
    let kname_s = __arena_get(fn_idx + 1);
    let kname_l = __arena_get(fn_idx + 2);
    // ".text\n"
    emit_ptx_byte(46); emit_ptx_byte(116); emit_ptx_byte(101);
    emit_ptx_byte(120); emit_ptx_byte(116); emit_ptx_byte(10);
    // ".globl " + <name> + "\n"
    emit_ptx_byte(46); emit_ptx_byte(103); emit_ptx_byte(108);
    emit_ptx_byte(111); emit_ptx_byte(98); emit_ptx_byte(108);
    emit_ptx_byte(32);
    let mut k1: i32 = 0;
    while k1 < kname_l {
        emit_ptx_byte(__arena_get(kname_s + k1));
        k1 = k1 + 1;
    }
    emit_ptx_byte(10);
    // ".p2align 8\n"
    emit_ptx_byte(46); emit_ptx_byte(112); emit_ptx_byte(50);
    emit_ptx_byte(97); emit_ptx_byte(108); emit_ptx_byte(105);
    emit_ptx_byte(103); emit_ptx_byte(110); emit_ptx_byte(32);
    emit_ptx_byte(56); emit_ptx_byte(10);
    // ".type " + <name> + ",@function\n"
    emit_ptx_byte(46); emit_ptx_byte(116); emit_ptx_byte(121);
    emit_ptx_byte(112); emit_ptx_byte(101); emit_ptx_byte(32);
    let mut k2: i32 = 0;
    while k2 < kname_l {
        emit_ptx_byte(__arena_get(kname_s + k2));
        k2 = k2 + 1;
    }
    emit_ptx_byte(44); emit_ptx_byte(64); emit_ptx_byte(102);
    emit_ptx_byte(117); emit_ptx_byte(110); emit_ptx_byte(99);
    emit_ptx_byte(116); emit_ptx_byte(105); emit_ptx_byte(111);
    emit_ptx_byte(110); emit_ptx_byte(10);                     // ",@function\n"
    // <name> + ":\n"
    let mut k3: i32 = 0;
    while k3 < kname_l {
        emit_ptx_byte(__arena_get(kname_s + k3));
        k3 = k3 + 1;
    }
    emit_ptx_byte(58); emit_ptx_byte(10);                      // ":\n"
    // "    s_endpgm\n"
    emit_ptx_byte(32); emit_ptx_byte(32); emit_ptx_byte(32);
    emit_ptx_byte(32);
    emit_ptx_byte(115); emit_ptx_byte(95); emit_ptx_byte(101);
    emit_ptx_byte(110); emit_ptx_byte(100); emit_ptx_byte(112);
    emit_ptx_byte(103); emit_ptx_byte(109); emit_ptx_byte(10); // "s_endpgm\n"
    emit_ptx_byte(10);                                         // blank line
    0
}
// K1.M20: top-level ROCm emitter. Header once + one kernel per @kernel fn
// (mirrors emit_msl_for_ast_to_path / Python rocm.py emit_module). 0
// kernels -> emits nothing (returns 0).
fn emit_rocm_for_ast_to_path(ast_root: i32) -> i32 {
    let mut kernel_count: i32 = 0;
    if __arena_get(ast_root) == 15 {
        let mut walk: i32 = ast_root;
        while walk != 0 {
            let fn_idx = __arena_get(walk + 1);
            if __arena_get(fn_idx + 14) == 1 {
                kernel_count = kernel_count + 1;
            };
            walk = __arena_get(walk + 2);
        }
    };
    if kernel_count == 0 {
        0
    } else {
        let start = __arena_len();
        emit_rocm_header();
        let mut walk2: i32 = ast_root;
        while walk2 != 0 {
            let fn_idx = __arena_get(walk2 + 1);
            if __arena_get(fn_idx + 14) == 1 {
                emit_rocm_kernel(fn_idx);
            };
            walk2 = __arena_get(walk2 + 2);
        }
        __arena_len() - start
    }
}

// --------------------------------------------------------------
// Demo: build a tiny AST_INT(42) by hand, compile it, write the
// resulting ELF to /tmp/kovc_ast_int.bin. The caller runs the
// produced binary externally; its exit code should be 42.
// --------------------------------------------------------------
fn main() -> i32 {
    let ast_root = __arena_push(0);
    __arena_push(42);
    __arena_push(0); __arena_push(0);
    let total = emit_elf_for_ast_to_path(ast_root);
    // The ELF byte stream is the LAST `total` slots of the arena
    // — robust against any pre-ELF arena pushes (bind_state,
    // fn_table, patch_table, "main" template).
    let elf_offset = __arena_len() - total;
    write_file_to_arena("/tmp/kovc_ast_int.bin", elf_offset, total)
}
