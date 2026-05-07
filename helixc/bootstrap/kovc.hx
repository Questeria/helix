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
// reserves 64 slots (= 512 bytes) of stack space for let-bindings.
// Slot offsets [rbp-8, rbp-16, ..., rbp-512] are addressable via
// disp8 ModRM (since 8..512 fits in signed-disp8 only for offsets
// up to 128; we use disp32 for safety on every access).
//
//   55                push rbp
//   48 89 E5          mov rbp, rsp
//   48 81 EC 00 04 00 00   sub rsp, 1024
//
// Audit fix (cycle 1, polish #14): bumped from 512 → 1024 to match the
// bind_state cap (64 entries) with 2× margin. Previously 512 was
// "just enough" for 64 × 8-byte slots — any future cap bump would
// silently corrupt the saved rbp/return-address. 1024 gives 128
// slots; future Phase-1 should derive this from bind_state cap
// dynamically rather than hard-coding.
fn emit_prologue() -> i32 {
    emit_byte(0x55);
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xE5);
    emit_byte(0x48); emit_byte(0x81); emit_byte(0xEC);
    emit_u32_le(1024);
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
// Capacity is fixed at compile-time (NUM_BINDINGS_CAP = 64). Table
// uses __arena_set to write entries — never __arena_push, so the
// code region can grow contiguously after bind_state.
//
// The type_tag was added in Phase 1.10 step 5c so the AST_ADD/SUB/MUL/
// DIV codegen can dispatch to SSE when both operands' bindings are f32.
// AST_LET stamps the type at push-time by inspecting the value AST.
fn bind_init() -> i32 {
    let state = __arena_push(8);            // next_offset = 8
    __arena_push(0);                        // top = 0
    __arena_push(state + 3);                // table_base = state + 3
    let mut i: i32 = 0;
    while i < 256 {                         // 64 entries * 4 slots
        __arena_push(0);
        i = i + 1;
    }
    state
}

fn bind_push(state: i32, name_start: i32, name_len: i32, offset: i32) -> i32 {
    bind_push_typed(state, name_start, name_len, offset, 0)
}

// Phase 1.10 step 5c: variant that records the binding's type.
// Audit fix #10 (cycle 1): cap-check before writing. The 64-entry
// cap (256 arena slots) is set in bind_init; without this guard,
// the 65th+ binding silently corrupts adjacent arena data — fn_table
// or str_table or worse. Now: silently SKIP the binding when full
// (return -1). Subsequent AST_VAR resolves via offset 0 (= unbound
// sentinel), which AST_VAR's audit-10 guard handles by emitting the
// integer-zero placeholder. No arena corruption. Sources hitting
// this cap are pathological; future Phase-1 should bump cap or
// implement spill.
fn bind_push_typed(state: i32, name_start: i32, name_len: i32,
                   offset: i32, ty: i32) -> i32 {
    let top = __arena_get(state + 1);
    if top >= 64 {
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
    let off = __arena_get(state);
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
    } else { if t == 9 {                              // AST_NEG
        expr_type(p1, bind_state, bn_state)
    } else { if t == 26 {                             // AST_BNOT
        expr_type(p1, bind_state, bn_state)
    } else { if t == 32 {                             // AST_SHL
        expr_type(p1, bind_state, bn_state)
    } else { if t == 33 {                             // AST_SHR
        expr_type(p1, bind_state, bn_state)
    } else { if t == 2 {                              // AST_ADD
        // Mismatched binary-op operands fall back to 0 (i32). This is
        // safe because the per-op codegen dispatch in emit_ast_code
        // emits ud2 on every (i64 op i32), (f32 op i32), etc. mixed
        // case — the expr_type return value is consumed by upstream
        // consumers (val_ty stamping, AST_RET trap) which are
        // unaffected since the wrapping context already trapped.
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { 0 }
    } else { if t == 3 {                              // AST_SUB
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { 0 }
    } else { if t == 4 {                              // AST_MUL
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { 0 }
    } else { if t == 5 {                              // AST_DIV
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { 0 }
    } else { if t == 24 {                             // AST_MOD
        let l = expr_type(p1, bind_state, bn_state);
        let r = expr_type(p2, bind_state, bn_state);
        if l == r { l } else { 0 }
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
    } else { 0 }}}}}}}}}}}}}}}}}}}}}}}}}}}}}
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
    while i < 1280 {                        // 256 entries * 5 slots
        __arena_push(0);
        i = i + 1;
    }
    state
}

fn fn_type_table_add(state: i32, name_start: i32, name_len: i32, ret_ty: i32, packed_param_tys: i32, param_count: i32) -> i32 {
    let top = __arena_get(state);
    if top >= 256 {
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
// 256 — generous so the lexer + parser + kovc concatenation (60-100
// fn declarations) fits comfortably with room for the
// __helix_arena_base symbol entry.
fn fn_table_init() -> i32 {
    let state = __arena_push(0);            // top = 0
    __arena_push(state + 2);                // table_base = state + 2
    let mut i: i32 = 0;
    while i < 768 {                         // 256 entries * 3 slots
        __arena_push(0);
        i = i + 1;
    }
    state
}

fn fn_table_add(state: i32, name_start: i32, name_len: i32, code_offset: i32) -> i32 {
    // Audit fix #10: cap-check before writing. Cap = 256 entries
    // (set in fn_table_init: 256 * 3 slots = 768).
    let top = __arena_get(state);
    if top >= 256 {
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
// [disp_slot, target_name_start, target_name_len]. Capacity 4096 —
// kovc.hx self-compiling has ~1500 patch entries (1159 fn calls +
// 339 inline-builtin LEAs across lexer + parser + kovc). 4096
// gives 2.7x headroom.
fn patch_table_init() -> i32 {
    let state = __arena_push(0);            // top = 0
    __arena_push(state + 2);                // table_base = state + 2
    let mut i: i32 = 0;
    while i < 12288 {                       // 4096 entries * 3 slots
        __arena_push(0);
        i = i + 1;
    }
    state
}

fn patch_table_add(state: i32, disp_slot: i32, name_start: i32, name_len: i32) -> i32 {
    // Audit fix #10: cap-check before writing. patch_table_init
    // allocates 4096 entries; without this guard, a source with > 4096
    // CALL+LEA patches would silently corrupt adjacent arena memory.
    let top = __arena_get(state);
    if top >= 4096 {
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
    // Reserve slots 5..83 (79 more slots: 2 file-name slots + 2
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
    // + 1 __f64_pack slot at 83 (Phase 1.10 step 7l).
    let mut i: i32 = 0;
    while i < 79 {
        __arena_push(0);
        i = i + 1;
    }

    // "__arena_push"
    let s0 = __arena_push(95); __arena_push(95); __arena_push(97); __arena_push(114);
    __arena_push(101); __arena_push(110); __arena_push(97); __arena_push(95);
    __arena_push(112); __arena_push(117); __arena_push(115); __arena_push(104);
    __arena_set(bn_state, s0);

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

    bn_state
}

fn bn_arena_push_s(b: i32) -> i32 { __arena_get(b) }
fn bn_arena_get_s(b: i32) -> i32  { __arena_get(b + 1) }
fn bn_arena_set_s(b: i32) -> i32  { __arena_get(b + 2) }
fn bn_arena_len_s(b: i32) -> i32  { __arena_get(b + 3) }
fn bn_helix_arena_base_s(b: i32) -> i32 { __arena_get(b + 4) }
fn bn_read_file_to_arena_s(b: i32) -> i32 { __arena_get(b + 5) }
fn bn_write_file_to_arena_s(b: i32) -> i32 { __arena_get(b + 6) }
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
    if top >= 16 {
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
        // Codegen layout (24 bytes after arg eval):
        //   eval x -> eax
        //   push rax                              50           (1)
        //   imul eax, eax (eax = x*x)             0F AF C0     (3)
        //   imul eax, eax, c1 (eax = x*x*c1)      69 C0 imm32  (6)
        //   pop rcx (rcx = x)                     59           (1)
        //   imul ecx, ecx, c2 (ecx = x*c2)        69 C9 imm32  (6)
        //   add eax, ecx                          01 C8        (2)
        //   add eax, c3                           05 imm32     (5)
        let a0 = __arena_get(args_head + 1);
        let n0 = emit_ast_code(a0, bind_state, patch_state, bn_state);
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
        n0 + 1 + 3 + 6 + 1 + 6 + 2 + 5
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
    } else {
        0
    }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
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
fn emit_ast_code(idx: i32, bind_state: i32, patch_state: i32, bn_state: i32) -> i32 {
    let t = __arena_get(idx);
    let p1 = __arena_get(idx + 1);
    let p2 = __arena_get(idx + 2);
    if t == 0 {
        emit_ast_int(p1)
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
        // High32: sign-extended from the i32-encoded `p1` (same shape as
        // AST_INTLIT_I64). For p1 >= 0, hi32 = 0 (correct for u64). For
        // p1 < 0, hi32 = 0xFFFFFFFF (all bits set), which preserves bit-
        // equality when u64 values are round-tripped through i64
        // conversions. Today the lexer encodes ALL literals as i32-signed
        // p1, so u64 values >= 2^31 cannot yet be expressed at lex time:
        // the high half is always 0 for any literal the parser actually
        // produces today.
        // TODO(main-thread / CRITICAL audit finding): lex-time literal-
        // overflow gap. The parser stores u64 literals in a single i32
        // p1 field, so values in [2^31, 2^64-1] silently truncate or mis-
        // encode at parse time. e.g. `2147483648_u64` (= 2^31) wraps to
        // i32 -2147483648 here, then hi32 = 0xFFFFFFFF, and the resulting
        // movabs emits 0xFFFFFFFF80000000_u64 instead of 0x80000000.
        // Fix requires widening the AST literal payload to two i32 fields
        // (lo32 + hi32) before this codegen site can be corrected. Do NOT
        // change the `let hi32` line below until the parser is fixed.
        let hi32 = if p1 < 0 { 0 - 1 } else { 0 };
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
    } else { if t == 27 {
        // AST_FLOATLIT (Phase 1.10 step 3d, f32). Phase 1.10 step 7b
        // also reuses this branch for AST_FLOATLIT_F64 (tag 34) — the
        // semantics are still f32-shaped (4-byte SSE single); step 7c
        // will branch on tag 34 for true 8-byte codegen.
        // p1 = byte_start of the literal text in the arena, p2 = byte_len.
        // Parse "I.F" -> int_part, frac_part, frac_digits; compute the
        // IEEE 754 f32 bit pattern via integer-only arithmetic; emit
        // `mov eax, BITS`. The downstream code can then store BITS as
        // i32 or movd into xmm0 for f32 arithmetic.
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
            // Step 3e: for v_scaled < pow10 (sub-1.0 literals like 0.5/0.25),
            // first decrement k and halve threshold until threshold <= v_scaled.
            // Step 3d: then do the existing positive-k loop. The two loops
            // together cover values in (~10^-9, ~10^9) within i32 limits.
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
            // Extract 23 mantissa bits via residual-doubling. residual stays
            // bounded by 2*threshold across iterations (no i32 overflow for
            // common literals like 1.5, 3.14, 100.25).
            let mut residual = v_scaled - threshold;
            let mut mantissa: i32 = 0;
            let mut bit: i32 = 22;
            while bit >= 0 {
                residual = residual * 2;
                if residual >= threshold {
                    // bit_val = 2^bit, computed inline.
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
        emit_ast_int(bits)
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
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        // Move-rcx: 64-bit when both operands are 8-byte (f64/i64/u64).
        let nm = if l_d == 1 {
            if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_u64 == 1 {
            if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let na = if l_d == 1 {
            if r_d == 1 { emit_addsd() } else { emit_ud2_trap() }
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_add_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_add_rax_rcx_64() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        if r_f == 1 { emit_addss() } else { emit_ud2_trap() }
                    } else {
                        if r_f == 1 { emit_ud2_trap() } else { emit_add_eax_ecx() }
                    }
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 3 {
        // Stage 1+2.4b: AST_SUB 5-way dispatch (i32/i64/u64/f32/f64).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 {
            if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_u64 == 1 {
            if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let na = if l_d == 1 {
            if r_d == 1 { emit_subsd() } else { emit_ud2_trap() }
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_sub_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_sub_rax_rcx_64() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        if r_f == 1 { emit_subss() } else { emit_ud2_trap() }
                    } else {
                        if r_f == 1 { emit_ud2_trap() } else { emit_sub_eax_ecx() }
                    }
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
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 {
            if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_u64 == 1 {
            if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let na = if l_d == 1 {
            if r_d == 1 { emit_mulsd() } else { emit_ud2_trap() }
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_imul_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_imul_rax_rcx_64() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        if r_f == 1 { emit_mulss() } else { emit_ud2_trap() }
                    } else {
                        if r_f == 1 { emit_ud2_trap() } else { emit_imul_eax_ecx() }
                    }
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
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 {
            if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_u64 == 1 {
            if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        let na = if l_d == 1 {
            if r_d == 1 { emit_divsd() } else { emit_ud2_trap() }
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_idiv_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_div_rax_rcx_64_u() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        if r_f == 1 { emit_divss() } else { emit_ud2_trap() }
                    } else { if r_f == 1 { emit_ud2_trap() } else {
                        if l_u32 == 1 {
                            if r_u32 == 1 { emit_div_eax_ecx_u() } else { emit_ud2_trap() }
                        } else { if r_u32 == 1 { emit_ud2_trap() } else { emit_idiv_eax_ecx() } }
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
        let nm = if l_d == 1 {
            if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_i64 == 1 {
            if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { if l_u64 == 1 {
            if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
        } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let na = if l_d == 1 {
            // f64 % f64 → ud2 (no SSE remainder); mixed → ud2.
            emit_ud2_trap()
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_imod_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_imod_rax_rcx_64_u() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        // f32 % f32 → ud2; mixed → ud2.
                        emit_ud2_trap()
                    } else { if r_f == 1 { emit_ud2_trap() } else {
                        if l_u32 == 1 {
                            if r_u32 == 1 { emit_imod_eax_ecx_u() } else { emit_ud2_trap() }
                        } else { if r_u32 == 1 { emit_ud2_trap() } else { emit_imod_eax_ecx() } }
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
        let ni = emit_ast_code(p1, bind_state, patch_state, bn_state);
        // Stage 1 audit batch 2: 4-way AST_NEG dispatch including i64.
        let nn = if is_f64_expr(p1, bind_state, bn_state) == 1 {
            emit_ast_dneg_suffix()
        } else { if is_i64_expr(p1, bind_state, bn_state) == 1 {
            emit_neg_rax_64()
        } else { if is_f32_expr(p1, bind_state, bn_state) == 1 {
            emit_ast_fneg_suffix()
        } else {
            emit_ast_neg_suffix()
        }}};
        ni + nn
    } else { if t == 26 {
        // AST_BNOT: emit inner (leaves value in eax/rax), then `not`.
        // Stage 1 audit fix: i64 needs `not rax` (REX.W) to flip all 64 bits.
        let ni = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let nn = if is_i64_expr(p1, bind_state, bn_state) == 1 {
            emit_not_rax_64()
        } else {
            emit_ast_bnot_suffix()
        };
        ni + nn
    } else { if t == 31 {
        // AST_NOT: logical NOT. For i64/u64, must use `test rax, rax`
        // (REX.W) to detect non-zero across the full 64 bits.
        // Stage 2.4b audit fix: u64 added (was i64-only).
        let ni = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let inner_i64 = is_i64_expr(p1, bind_state, bn_state);
        let inner_u64 = is_u64_expr(p1, bind_state, bn_state);
        let inner_wide = if inner_i64 == 1 { 1 } else { if inner_u64 == 1 { 1 } else { 0 } };
        let nn = if inner_wide == 1 {
            emit_ast_not_suffix_64()
        } else {
            emit_ast_not_suffix()
        };
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
        let na = if l_i64 == 1 {
            if r_i64 == 1 { emit_and_rax_rcx_64() } else { emit_ud2_trap() }
        } else { if r_i64 == 1 { emit_ud2_trap() } else { emit_and_eax_ecx() } };
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
        let na = if l_i64 == 1 {
            if r_i64 == 1 { emit_or_rax_rcx_64() } else { emit_ud2_trap() }
        } else { if r_i64 == 1 { emit_ud2_trap() } else { emit_or_eax_ecx() } };
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
        let na = if l_i64 == 1 {
            if r_i64 == 1 { emit_xor_rax_rcx_64() } else { emit_ud2_trap() }
        } else { if r_i64 == 1 { emit_ud2_trap() } else { emit_xor_eax_ecx() } };
        n1 + np + n2 + nm + no + na
    } else { if t == 32 {
        // AST_SHL: shl eax, cl (D3 E0); shl rax, cl (REX.W: 48 D3 E0) for i64.
        // Stage 1 audit fix: shift count is always treated as i32 (cl);
        // value being shifted picks i64 vs i32.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let nm = emit_mov_ecx_eax();
        let no = emit_pop_rax();
        let na = if l_i64 == 1 { emit_shl_rax_cl_64() } else { emit_shl_eax_cl() };
        n1 + np + n2 + nm + no + na
    } else { if t == 33 {
        // AST_SHR: sar eax, cl (D3 F8) i32; sar rax, cl (48 D3 F8) i64.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let nm = emit_mov_ecx_eax();
        let no = emit_pop_rax();
        let na = if l_i64 == 1 { emit_sar_rax_cl_64() } else { emit_sar_eax_cl() };
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
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        // Stage 2.2: 5-way LT dispatch — u32 < u32 uses `setb` (unsigned).
        // Stage 2.4b: 6-way — u64 < u64 uses REX.W cmp + setb.
        let na = if l_d == 1 {
            if r_d == 1 { emit_ssen_lt_dbl() } else { emit_ud2_trap() }
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_lt_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_lt_rax_rcx_64_u() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        if r_f == 1 { emit_ssen_lt() } else { emit_ud2_trap() }
                    } else { if r_f == 1 { emit_ud2_trap() } else {
                        if l_u32 == 1 {
                            if r_u32 == 1 { emit_lt_eax_ecx_u() } else { emit_ud2_trap() }
                        } else { if r_u32 == 1 { emit_ud2_trap() } else { emit_lt_eax_ecx() } }
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 19 {
        // Audit fix #9: AST_GT mixed-type ud2 trap.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        // Stage 2.2: 5-way GT dispatch — u32 > u32 uses `seta` (unsigned).
        // Stage 2.4b: 6-way — u64 > u64 uses REX.W cmp + seta.
        let na = if l_d == 1 {
            if r_d == 1 { emit_ssen_gt_dbl() } else { emit_ud2_trap() }
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_gt_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_gt_rax_rcx_64_u() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        if r_f == 1 { emit_ssen_gt() } else { emit_ud2_trap() }
                    } else { if r_f == 1 { emit_ud2_trap() } else {
                        if l_u32 == 1 {
                            if r_u32 == 1 { emit_gt_eax_ecx_u() } else { emit_ud2_trap() }
                        } else { if r_u32 == 1 { emit_ud2_trap() } else { emit_gt_eax_ecx() } }
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 20 {
        // Audit fix #9: AST_EQ mixed-type ud2 trap.
        // Stage 2.4b: u64 == u64 reuses i64's emit_eq_rax_rcx_64 since
        // bitwise equality on a 64-bit value is signedness-agnostic.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let na = if l_d == 1 {
            if r_d == 1 { emit_ssen_eq_dbl() } else { emit_ud2_trap() }
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_eq_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_eq_rax_rcx_64() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        if r_f == 1 { emit_ssen_eq() } else { emit_ud2_trap() }
                    } else { if r_f == 1 { emit_ud2_trap() } else { emit_eq_eax_ecx() } }
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 21 {
        // Audit fix #9: AST_NE mixed-type ud2 trap.
        // Stage 2.4b: u64 != u64 reuses i64's emit_ne_rax_rcx_64
        // (signedness-agnostic for 64-bit inequality).
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let na = if l_d == 1 {
            if r_d == 1 { emit_ssen_ne_dbl() } else { emit_ud2_trap() }
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_ne_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_ne_rax_rcx_64() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        if r_f == 1 { emit_ssen_ne() } else { emit_ud2_trap() }
                    } else { if r_f == 1 { emit_ud2_trap() } else { emit_ne_eax_ecx() } }
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 22 {
        // Audit fix #9: AST_LE mixed-type ud2 trap.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        // Stage 2.2: 5-way LE dispatch — u32 <= u32 uses `setbe` (unsigned).
        // Stage 2.4b: 6-way — u64 <= u64 uses REX.W cmp + setbe.
        let na = if l_d == 1 {
            if r_d == 1 { emit_ssen_le_dbl() } else { emit_ud2_trap() }
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_le_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_le_rax_rcx_64_u() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        if r_f == 1 { emit_ssen_le() } else { emit_ud2_trap() }
                    } else { if r_f == 1 { emit_ud2_trap() } else {
                        if l_u32 == 1 {
                            if r_u32 == 1 { emit_le_eax_ecx_u() } else { emit_ud2_trap() }
                        } else { if r_u32 == 1 { emit_ud2_trap() } else { emit_le_eax_ecx() } }
                    }}
                }}
            }}
        }};
        n1 + np + n2 + nm + no + na
    } else { if t == 23 {
        // Audit fix #9: AST_GE mixed-type ud2 trap.
        let n1 = emit_ast_code(p1, bind_state, patch_state, bn_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state, patch_state, bn_state);
        let l_d = is_f64_expr(p1, bind_state, bn_state);
        let r_d = is_f64_expr(p2, bind_state, bn_state);
        let l_i64 = is_i64_expr(p1, bind_state, bn_state);
        let r_i64 = is_i64_expr(p2, bind_state, bn_state);
        let l_u64 = is_u64_expr(p1, bind_state, bn_state);
        let r_u64 = is_u64_expr(p2, bind_state, bn_state);
        let nm = if l_d == 1 { if r_d == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_i64 == 1 { if r_i64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { if l_u64 == 1 { if r_u64 == 1 { emit_mov_rcx_rax_64() } else { emit_mov_ecx_eax() }
               } else { emit_mov_ecx_eax() }}};
        let no = emit_pop_rax();
        let l_f = is_f32_expr(p1, bind_state, bn_state);
        let r_f = is_f32_expr(p2, bind_state, bn_state);
        let l_u32 = is_u32_expr(p1, bind_state, bn_state);
        let r_u32 = is_u32_expr(p2, bind_state, bn_state);
        // Stage 2.2: 5-way GE dispatch — u32 >= u32 uses `setae` (unsigned).
        // Stage 2.4b: 6-way — u64 >= u64 uses REX.W cmp + setae.
        let na = if l_d == 1 {
            if r_d == 1 { emit_ssen_ge_dbl() } else { emit_ud2_trap() }
        } else { if r_d == 1 { emit_ud2_trap() } else {
            if l_i64 == 1 {
                if r_i64 == 1 { emit_ge_rax_rcx_64() } else { emit_ud2_trap() }
            } else { if r_i64 == 1 { emit_ud2_trap() } else {
                if l_u64 == 1 {
                    if r_u64 == 1 { emit_ge_rax_rcx_64_u() } else { emit_ud2_trap() }
                } else { if r_u64 == 1 { emit_ud2_trap() } else {
                    if l_f == 1 {
                        if r_f == 1 { emit_ssen_ge() } else { emit_ud2_trap() }
                    } else { if r_f == 1 { emit_ud2_trap() } else {
                        if l_u32 == 1 {
                            if r_u32 == 1 { emit_ge_eax_ecx_u() } else { emit_ud2_trap() }
                        } else { if r_u32 == 1 { emit_ud2_trap() } else { emit_ge_eax_ecx() } }
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
        let off = bind_lookup(bind_state, p1, p2);
        let ty = bind_lookup_type(bind_state, p1, p2);
        // 8-byte types (f64=2, i64=3, u64=9) use 64-bit load to preserve
        // the high half. All others use 32-bit load (auto-zero-extends).
        if ty == 2 { emit_mov_rax_local_64(off) }
        else { if ty == 3 { emit_mov_rax_local_64(off) }
        else { if ty == 9 { emit_mov_rax_local_64(off) }
        else { emit_mov_eax_local(off) }}}
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
        // 8-byte types (f64=2, i64=3, u64=9) use 64-bit store. All
        // other types (i32=0, f32=1, u32=6, u8=7, future i8/i16/u16)
        // use 32-bit store; their 8-byte slot has the high 32 cleared
        // by x86's auto-zero-extension on 32-bit ops.
        let n_store = if val_ty == 2 {
            emit_mov_local_rax_64(off)
        } else { if val_ty == 3 {
            emit_mov_local_rax_64(off)
        } else { if val_ty == 9 {
            emit_mov_local_rax_64(off)
        } else {
            emit_mov_local_eax(off)
        }}};
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
        } else {
            emit_mov_local_eax(off)
        }}};
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
            // Stage 1 audit fixes + Stage 2.4: type-mismatch trap.
            // The 8-byte types (i64=3, u64=9) require width-matched
            // value+binding; mixed widths trap with ud2 to avoid silent
            // truncation or zero-extension bugs.
            let val_i64 = is_i64_expr(p3, bind_state, bn_state);
            let val_u64 = is_u64_expr(p3, bind_state, bn_state);
            let n_store = if val_i64 == 1 {
                if bind_ty == 3 { emit_mov_local_rax_64(off) }
                else { emit_ud2_trap() }
            } else { if val_u64 == 1 {
                if bind_ty == 9 { emit_mov_local_rax_64(off) }
                else { emit_ud2_trap() }
            } else { if bind_ty == 2 {
                emit_mov_local_rax_64(off)
            } else { if bind_ty == 3 {
                // Stage 1 audit batch 5 fix: i32 value into i64 binding
                // is NOT a safe widening — `mov eax, X` zero-extends to
                // rax, so for negative i32 values the i64 slot ends up
                // holding (val + 2^32) instead of val. Trap instead of
                // silently producing wrong results. Mirrors batch 3's
                // i64-into-i32 trap.
                emit_ud2_trap()
            } else { if bind_ty == 9 {
                // Stage 2.4 mirror: i32-into-u64 also traps.
                emit_ud2_trap()
            } else {
                emit_mov_local_eax(off)
            }}}}};
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
            if arg_count < pp_count {
                let expected_ty = unpack_param_ty(pp_packed, arg_count);
                let actual_ty = expr_type(arg_expr, bind_state, bn_state);
                if expected_ty != actual_ty {
                    let n_trap = emit_ud2_trap();
                    bytes_emitted = bytes_emitted + n_trap;
                };
            };
            arg_count = arg_count + 1;
            arg_cur = __arena_get(arg_cur + 2);
        }
        // Audit fix #7 (cycle 1): if arg_count > 6, emit ud2 trap.
        // Phase 0 doesn't yet implement SysV stack args (args 6+ are
        // supposed to be passed on the stack at [rsp+0], [rsp+8], ...
        // with the caller adding `add rsp, N` after the call). Without
        // that, the args 6+ remain on the stack and the call still
        // happens — corrupting both the stack and any subsequent
        // pop/cmp operations. Loud SIGILL is much better than silent
        // corruption. Implement stack-args properly when needed.
        if arg_count > 6 {
            let n_trap = emit_ud2_trap();
            bytes_emitted + n_trap
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
        let disp_slot = emit_call_rel32_placeholder();
        patch_table_add(patch_state, disp_slot, p1, p2);
        bytes_emitted + 5
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
        let end_label = __arena_len();
        patch_rel32(je_disp, end_label);
        patch_rel32(jmp_disp, loop_top);
        let n_zero = emit_ast_int(0);
        n_cond + n_test + 6 + n_body + 5 + n_zero
    } else { if t == 25 {
        // AST_STR_LIT used as a value. Phase-0: strings are only
        // meaningful as the FIRST arg of a file builtin (handled in
        // try_emit_builtin_call). When used elsewhere — e.g., as the
        // value of a let or as an integer expression — emit `mov
        // eax, 0` so codegen completes cleanly. Trying to use the
        // result is undefined behavior at this stage.
        emit_ast_int(0)
    } else {
        // Audit fix #8 (cycle 1): unhandled AST tag. Previously emitted
        // `mov eax, 0` (5 bytes) which silently masked AST_ERR (tag 99)
        // from lex/parse failures and any future tag added to parser
        // without a codegen handler. Now emits ud2 (2 bytes) so the
        // bug is loud at runtime instead of producing a binary that
        // returns 0. Lex/parse errors that produce AST_ERR cause the
        // resulting binary to SIGILL — clear signal vs. silent 0.
        emit_ud2_trap()
    }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
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
            //              p4 (slot 4) = params_head, p5 (slot 5) = ret_ty.
            let fn_name_s = __arena_get(fn_idx + 1);
            let fn_name_l = __arena_get(fn_idx + 2);
            let fn_ret_ty = __arena_get(fn_idx + 5);
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
                    let pp_ty = __arena_get(pp_cur + 4);
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
        let mut cur_list: i32 = ast_root;
        while cur_list != 0 {
            let fn_idx = __arena_get(cur_list + 1);
            let fn_name_s = __arena_get(fn_idx + 1);
            let fn_name_l = __arena_get(fn_idx + 2);
            let fn_body = __arena_get(fn_idx + 3);
            let params_head = __arena_get(fn_idx + 4);
            let fn_ret_ty = __arena_get(fn_idx + 5);
            let fn_code_offset = __arena_len();
            fn_table_add(fn_state, fn_name_s, fn_name_l, fn_code_offset);
            bind_reset(bind_state);
            emit_prologue();
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
                    let needs_64 = if p_ty == 2 { 1 }
                                   else { if p_ty == 3 { 1 }
                                   else { if p_ty == 9 { 1 } else { 0 } } };
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
            let ret_wants_8b = if fn_ret_ty == 3 { 1 } else { if fn_ret_ty == 9 { 1 } else { 0 } };
            if body_is_8b != ret_wants_8b {
                emit_ud2_trap();
            };
            emit_epilogue();
            emit_ret();
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
            if target_offset >= 0 {
                let disp = target_offset - (disp_slot + 4);
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
