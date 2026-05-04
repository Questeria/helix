; ============================================================================
; hex0.s — Kovostov-Native seed monitor (x86_64 Linux)
; ============================================================================
;
; Reads hex characters from stdin, writes decoded bytes to stdout.
; - Whitespace (space, tab, \n, \r) skipped.
; - Comments: ';' or '#' to end-of-line.
; - Hex digits: 0-9, A-F, a-f. Pairs combine high-nibble-first.
; - Other characters silently skipped.
; - EOF -> exit 0. Read error -> exit 1.
;
; This is the human-readable form. The canonical artifact is hex0.bin,
; whose bytes are hand-encoded and annotated in hex0.bytes.md.
; nasm is used only as cross-check, never for shipping.
;
; Calling convention reminder (Linux x86_64 syscall):
;   rax = syscall number   (0=read, 1=write, 60=exit)
;   rdi = arg1   rsi = arg2   rdx = arg3
;   syscall instruction = 0F 05
;   rax returns (or -errno); rcx, r11 clobbered
;
; Register usage:
;   rbp = state. low bit (bpl & 1): "high nibble buffered"
;                bits 8..11 of ebp: the buffered high nibble (already shifted)
;                Actually we use bl to store the shifted high nibble.
;   ebx = bl is high-nibble-shifted-left-by-4. Set when ebp's low bit is 1.
;   al  = scratch (currently-read character, computed nibble)
;
; ============================================================================

BITS 64
DEFAULT REL

; ----------------------------------------------------------------------------
; ELF64 header  (64 bytes)
; ----------------------------------------------------------------------------
ELF_BASE        equ 0x600000
ENTRY_OFFSET    equ 0x78          ; ELF header (64) + program header (56) = 120 = 0x78

ehdr:
    db 0x7F, "ELF"                ; e_ident[EI_MAG]  (4 bytes)
    db 2                          ; EI_CLASS = ELFCLASS64
    db 1                          ; EI_DATA = ELFDATA2LSB (little-endian)
    db 1                          ; EI_VERSION = EV_CURRENT
    db 0                          ; EI_OSABI = System V
    db 0                          ; EI_ABIVERSION
    times 7 db 0                  ; EI_PAD
    dw 2                          ; e_type = ET_EXEC
    dw 0x3E                       ; e_machine = EM_X86_64
    dd 1                          ; e_version
    dq ELF_BASE + ENTRY_OFFSET    ; e_entry
    dq 64                         ; e_phoff (program header table at offset 64)
    dq 0                          ; e_shoff (no section headers)
    dd 0                          ; e_flags
    dw 64                         ; e_ehsize
    dw 56                         ; e_phentsize
    dw 1                          ; e_phnum
    dw 0                          ; e_shentsize
    dw 0                          ; e_shnum
    dw 0                          ; e_shstrndx

; ----------------------------------------------------------------------------
; Program header (56 bytes) — single PT_LOAD segment, R+X
; ----------------------------------------------------------------------------
phdr:
    dd 1                          ; p_type = PT_LOAD
    dd 5                          ; p_flags = PF_R | PF_X
    dq 0                          ; p_offset = 0 (segment includes ELF headers)
    dq ELF_BASE                   ; p_vaddr
    dq ELF_BASE                   ; p_paddr
    dq filesz                     ; p_filesz (computed at assembly)
    dq filesz                     ; p_memsz (same; no .bss)
    dq 0x1000                     ; p_align (page)

; ============================================================================
; Code starts at offset 0x78 (entry = ELF_BASE + 0x78)
; ============================================================================
_start:
    xor    ebp, ebp               ; ebp = 0, no high nibble buffered

; ----------------------------------------------------------------------------
; Main read loop: read 1 byte, dispatch
; ----------------------------------------------------------------------------
read_loop:
    push   rax                    ; reserve 8 bytes on stack as read buffer
                                  ; (rax's value irrelevant; we only use [rsp])
    xor    eax, eax               ; sys_read = 0
    xor    edi, edi               ; fd = stdin = 0
    mov    rsi, rsp               ; buf = stack
    mov    edx, 1                 ; count = 1
    syscall

    test   rax, rax
    jle    do_exit                ; rax <= 0: EOF (=0) or error (<0) -> exit
                                  ; (status set inside do_exit)

    movzx  eax, byte [rsp]        ; load the byte we just read into al (zero-extended)
    pop    rcx                    ; deallocate stack slot (value discarded)

; ----------------------------------------------------------------------------
; Dispatch on the character in al
; ----------------------------------------------------------------------------
    cmp    al, '#'
    je     skip_comment
    cmp    al, ';'
    je     skip_comment

    ; whitespace: skip
    cmp    al, ' '
    je     read_loop
    cmp    al, 9                  ; \t
    je     read_loop
    cmp    al, 10                 ; \n
    je     read_loop
    cmp    al, 13                 ; \r
    je     read_loop

    ; hex digit ranges
    cmp    al, '0'
    jl     read_loop              ; below '0': invalid, skip
    cmp    al, '9'
    jle    digit_0_9
    cmp    al, 'A'
    jl     read_loop
    cmp    al, 'F'
    jle    digit_A_F
    cmp    al, 'a'
    jl     read_loop
    cmp    al, 'f'
    jle    digit_a_f
    jmp    read_loop              ; above 'f': invalid, skip

digit_0_9:
    sub    al, '0'                ; al -= 0x30, now 0..9
    jmp    got_nibble

digit_A_F:
    sub    al, 'A' - 10           ; al -= 0x37, now 10..15
    jmp    got_nibble

digit_a_f:
    sub    al, 'a' - 10           ; al -= 0x57, now 10..15

; ----------------------------------------------------------------------------
; got_nibble: al = 0..15 (the nibble value)
;             ebp & 1 = "high nibble already buffered" flag
;             bl     = high nibble shifted left by 4 (when flag set)
; ----------------------------------------------------------------------------
got_nibble:
    test   bpl, 1
    jnz    combine

    ; First nibble: shift, store, set flag
    shl    al, 4
    mov    bl, al
    mov    ebp, 1
    jmp    read_loop

combine:
    or     al, bl                 ; al = (high << 4) | low
    push   rax                    ; reserve stack slot, put al at [rsp]
    mov    [rsp], al
    xor    eax, eax
    inc    eax                    ; sys_write = 1   (smaller than mov eax, 1)
    mov    edi, eax               ; fd = stdout = 1 (eax is already 1)
    mov    rsi, rsp               ; buf
    mov    edx, eax               ; count = 1 (eax is 1)
    syscall                       ; we ignore short writes for this seed
    pop    rax                    ; deallocate
    xor    ebp, ebp               ; clear high-nibble flag
    jmp    read_loop

; ----------------------------------------------------------------------------
; skip_comment: read until '\n' or EOF
; ----------------------------------------------------------------------------
skip_comment:
    push   rax
    xor    eax, eax               ; sys_read
    xor    edi, edi               ; stdin
    mov    rsi, rsp
    mov    edx, 1
    syscall
    test   rax, rax
    jle    do_exit_after_pop
    movzx  eax, byte [rsp]
    pop    rcx
    cmp    al, 10                 ; \n
    je     read_loop
    jmp    skip_comment

do_exit_after_pop:
    pop    rcx                    ; clean stack
    ; fall through to do_exit

; ----------------------------------------------------------------------------
; Exit path
; ----------------------------------------------------------------------------
do_exit:
    xor    edi, edi               ; status = 0
    mov    eax, 60                ; sys_exit
    syscall
    ; (unreachable)

end:
filesz equ end - ehdr
