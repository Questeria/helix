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

// AST_NEG(inner): emit inner code, then `neg eax`.
//   F7 D8   neg eax
// (inner already left its value in eax.)
fn emit_ast_neg_suffix() -> i32 {
    emit_byte(0xF7); emit_byte(0xD8);
    2
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
// idiv requires sign-extension into edx; we emit `cdq; idiv ecx`.
//   99       cdq
//   F7 F9    idiv ecx
fn emit_idiv_eax_ecx() -> i32 {
    emit_byte(0x99);
    emit_byte(0xF7); emit_byte(0xF9);
    3
}

// AST_LT: cmp eax, ecx; mov eax, 0; setl al — leaves 0 or 1 in eax.
//   39 C8         cmp eax, ecx
//   B8 00 00 00 00   mov eax, 0
//   0F 9C C0      setl al
fn emit_lt_eax_ecx() -> i32 {
    emit_byte(0x39); emit_byte(0xC8);
    emit_byte(0xB8); emit_byte(0); emit_byte(0); emit_byte(0); emit_byte(0);
    emit_byte(0x0F); emit_byte(0x9C); emit_byte(0xC0);
    10
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
//   48 81 EC 00 02 00 00   sub rsp, 512
fn emit_prologue() -> i32 {
    emit_byte(0x55);
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xE5);
    emit_byte(0x48); emit_byte(0x81); emit_byte(0xEC);
    emit_u32_le(512);
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
//   slot 3..3 + cap*3: pre-allocated entries [name_start, name_len, offset]
//
// Capacity is fixed at compile-time (NUM_BINDINGS_CAP = 64). Table
// uses __arena_set to write entries — never __arena_push, so the
// code region can grow contiguously after bind_state.
fn bind_init() -> i32 {
    let state = __arena_push(8);            // next_offset = 8
    __arena_push(0);                        // top = 0
    __arena_push(state + 3);                // table_base = state + 3
    let mut i: i32 = 0;
    while i < 192 {                         // 64 entries * 3 slots
        __arena_push(0);
        i = i + 1;
    }
    state
}

fn bind_push(state: i32, name_start: i32, name_len: i32, offset: i32) -> i32 {
    let top = __arena_get(state + 1);
    let table_base = __arena_get(state + 2);
    let entry = table_base + top * 3;
    __arena_set(entry, name_start);
    __arena_set(entry + 1, name_len);
    __arena_set(entry + 2, offset);
    __arena_set(state + 1, top + 1);
    0
}

fn bind_pop(state: i32) -> i32 {
    let top = __arena_get(state + 1);
    __arena_set(state + 1, top - 1);
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
            let entry = table_base + i * 3;
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
// AST walker: dispatch on tag and emit the matching code. Returns
// the number of bytes emitted. AST node layout matches stage-2
// parser.hx: [tag, p1, p2, p3].
//
// Compile-time state:
//   bind_state holds (next_offset_slot, bind_table_top_slot) — both
//   arena slot indices that the recursive emit calls read/update.
//   Slot[bind_state]   = next free stack offset (starts at 8)
//   Slot[bind_state+1] = bind table top (= __arena_len() when empty)
// --------------------------------------------------------------
fn emit_ast_code(idx: i32, bind_state: i32) -> i32 {
    let t = __arena_get(idx);
    let p1 = __arena_get(idx + 1);
    let p2 = __arena_get(idx + 2);
    if t == 0 {
        emit_ast_int(p1)
    } else { if t == 2 {
        let n1 = emit_ast_code(p1, bind_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state);
        let nm = emit_mov_ecx_eax();
        let no = emit_pop_rax();
        let na = emit_add_eax_ecx();
        n1 + np + n2 + nm + no + na
    } else { if t == 3 {
        let n1 = emit_ast_code(p1, bind_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state);
        let nm = emit_mov_ecx_eax();
        let no = emit_pop_rax();
        let na = emit_sub_eax_ecx();
        n1 + np + n2 + nm + no + na
    } else { if t == 4 {
        let n1 = emit_ast_code(p1, bind_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state);
        let nm = emit_mov_ecx_eax();
        let no = emit_pop_rax();
        let na = emit_imul_eax_ecx();
        n1 + np + n2 + nm + no + na
    } else { if t == 5 {
        let n1 = emit_ast_code(p1, bind_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state);
        let nm = emit_mov_ecx_eax();
        let no = emit_pop_rax();
        let na = emit_idiv_eax_ecx();
        n1 + np + n2 + nm + no + na
    } else { if t == 9 {
        let ni = emit_ast_code(p1, bind_state);
        let nn = emit_ast_neg_suffix();
        ni + nn
    } else { if t == 6 {
        let n1 = emit_ast_code(p1, bind_state);
        let np = emit_push_rax();
        let n2 = emit_ast_code(p2, bind_state);
        let nm = emit_mov_ecx_eax();
        let no = emit_pop_rax();
        let na = emit_lt_eax_ecx();
        n1 + np + n2 + nm + no + na
    } else { if t == 7 {
        // AST_IF(cond, then, else)
        let p3 = __arena_get(idx + 3);
        let n_cond = emit_ast_code(p1, bind_state);
        let n_test = emit_test_eax_eax();
        let je_disp = emit_je_rel32_placeholder();
        let n_then = emit_ast_code(p2, bind_state);
        let jmp_disp = emit_jmp_rel32_placeholder();
        let else_label = __arena_len();
        let n_else = emit_ast_code(p3, bind_state);
        let merge_label = __arena_len();
        patch_rel32(je_disp, else_label);
        patch_rel32(jmp_disp, merge_label);
        n_cond + n_test + 6 + n_then + 5 + n_else
    } else { if t == 1 {
        // AST_VAR: p1 = name start, p2 = name len.
        let off = bind_lookup(bind_state, p1, p2);
        emit_mov_eax_local(off)
    } else { if t == 8 {
        // AST_LET: p1 = name start, p2 = name len, p3 = packed
        // (value_idx * 65536 + body_idx).
        let p3 = __arena_get(idx + 3);
        let value_idx = p3 / 65536;
        let body_idx = p3 - value_idx * 65536;
        let n_val = emit_ast_code(value_idx, bind_state);
        let off = bind_alloc_offset(bind_state);
        let n_store = emit_mov_local_eax(off);
        bind_push(bind_state, p1, p2, off);
        let n_body = emit_ast_code(body_idx, bind_state);
        bind_pop(bind_state);
        // Slot is leaked (we don't reset next_offset) — fine since
        // the function frame is pre-sized at 512 bytes (64 slots).
        n_val + n_store + n_body
    } else { if t == 10 {
        // AST_WHILE(cond, body):
        //   loop_top:
        //     <cond>           leaves 0/1 in eax
        //     test eax, eax
        //     je end_label
        //     <body>
        //     jmp loop_top    (backward — exercises emit_u32_le on
        //                       negative disp, the audit-8 fix)
        //   end_label:
        //     mov eax, 0      Helix while-expr returns unit (0)
        let loop_top = __arena_len();
        let n_cond = emit_ast_code(p1, bind_state);
        let n_test = emit_test_eax_eax();
        let je_disp = emit_je_rel32_placeholder();
        let n_body = emit_ast_code(p2, bind_state);
        let jmp_disp = emit_jmp_rel32_placeholder();
        let end_label = __arena_len();
        patch_rel32(je_disp, end_label);
        patch_rel32(jmp_disp, loop_top);
        let n_zero = emit_ast_int(0);
        n_cond + n_test + 6 + n_body + 5 + n_zero
    } else {
        emit_ast_int(0)
    }}}}}}}}}}}
}

// --------------------------------------------------------------
// Top-level: lay out an ELF executable that runs `eval(ast_root)`.
// Patching strategy: emit the ELF header with placeholder filesz
// fields (zeros), emit padding + code, then go back and rewrite
// the filesz / memsz bytes once the actual code size is known.
//
// Returns the byte count written to disk.
// --------------------------------------------------------------
fn emit_elf_for_ast_to_path(ast_root: i32) -> i32 {
    // Pre-allocate bind_state BEFORE the ELF region so it doesn't
    // pollute the contiguous code byte stream. bind_init pushes
    // 3 + 64*3 = 195 slots and writes them via __arena_set during
    // codegen — never __arena_push, which would interleave with
    // the code bytes.
    let bind_state = bind_init();
    let elf_start = __arena_len();
    emit_elf_header(0);
    emit_program_header(0);
    emit_padding_to_code();
    let code_start = __arena_len();
    emit_prologue();
    emit_ast_code(ast_root, bind_state);
    emit_epilogue();
    emit_exit_with_eax();
    let code_end = __arena_len();
    let total_filesz = 4096 + (code_end - code_start);
    patch_u64_le_split(elf_start + 64 + 32, total_filesz, 0);
    patch_u64_le_split(elf_start + 64 + 40, total_filesz, 0);
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
    // emit_elf_for_ast_to_path allocates bind_state internally
    // (195 slots) BEFORE the ELF, so the ELF byte range begins at
    // ast_root + 4 + 195 = ast_root + 199.
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_offset = ast_root + 4 + 195;
    write_file_to_arena("/tmp/kovc_ast_int.bin", elf_offset, total)
}
