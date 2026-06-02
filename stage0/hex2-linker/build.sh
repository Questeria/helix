#!/usr/bin/env bash
# build.sh -- build stage0/hex2-linker/hex2.bin: the mescc-tools 1.7.0
# FLAG-DRIVEN hex2 linker (accepts --base-address / --architecture /
# --little-endian, unlike stage0's OLD positional hex2 which hung on flags).
#
# Built by M2.bin (../M2-Planet), NOT cc_amd64 -- hex2 uses
# FILE*/fopen/fgetc/fputc/stderr/EOF/chmod. hex2 is multi-file; main() lives in
# hex2.c, the globals are DEFINED in hex2_linker.c, helpers in hex2_word.c.
#
# M2 ignores #include (--expand-includes off), so we hand M2 the full
# translation unit as ordered -f files. Order is load-bearing:
#   1. the 3 M2libc bootstrap C units   -- syscalls + FILE*/fgetc/.../malloc
#   2. M2libc/bootstrappable.c          -- require/match/in_set/strtoint/...
#   3. (generated) hex2_defs.m2.h       -- hex2.h's constants re-expressed as an
#                                          enum + its two struct defs (see below)
#   4. hex2_globals.h                   -- function prototypes + extern globals
#                                          (extern decls are harmless forward
#                                          decls inside one combined TU)
#   5. hex2_linker.c                    -- DEFINES the globals + core passes
#   6. hex2_word.c                      -- word-oriented passes (riscv etc.)
#   7. hex2.c                           -- main() + CLI flag parsing
# (Every #include inside those files is a no-op under M2; the symbols come from
# the units we pass explicitly.)
#
# WHY THE GENERATED HEADER (not hex2.h directly): our M2.bin is a *bootstrap*
# build (cc_amd64 -> M2), so it MUST run in --bootstrap-mode (that's the only
# mode where FILE is a built-in type; the full non-bootstrap M2libc stdio is not
# vendored). But --bootstrap-mode emulates the weaker cc_* compilers, which do
# NOT expand object-like #define value-macros in expression context. Verified:
# `#define MS 4096; return (MS>3);` compiles WITHOUT --bootstrap-mode but fails
# WITH it ("MS is not a defined symbol"). hex2.h uses object-like #defines
# (max_string, KNIGHT, X86, AMD64, ...) as values, so under bootstrap-mode they
# would not resolve. M1.c and blood-elf.c avoid this by using `enum` for the
# same constants -- which IS the canonical M2-Planet idiom. So we mechanically
# re-express hex2.h's #define block as an enum (semantically identical) and keep
# the vendored hex2.h byte-for-byte untouched on disk for provenance.
#
# M0 is fine here (small input). Run under WSL. GPL-3.0 (mescc-tools 1.7.0).
set -euo pipefail
cd "$(dirname "$0")"

M2=../M2-Planet/M2.bin
CATM=../catm/catm.bin
M0=../M0/M0.bin
HEX2=../hex2/hex2.bin
MLIB=../M2-Planet/M2libc
MDEFS=$MLIB/amd64/amd64_defs.M1
MLIBC=$MLIB/amd64/libc-core.M1
MELF=$MLIB/amd64/ELF-amd64.hex2
OUT=hex2.bin

for t in "$M2" "$CATM" "$M0" "$HEX2"; do [ -x "$t" ] || chmod +x "$t"; done

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# 0. Generate the bootstrap-mode constants header: hex2.h's #define block as an
#    enum + hex2.h's two struct defs verbatim. (TRUE/FALSE come from the M2libc
#    bootstrap.c enum, so we omit them here to avoid a redefinition.)
cat > "$T/hex2_defs.m2.h" <<'EOF'
/* GENERATED for bootstrap-mode M2 by hex2-linker/build.sh -- NOT vendored.
 * Re-expresses hex2.h's object-like #define constants as an enum (the M2-Planet
 * canonical idiom) because our bootstrap-built M2.bin does not expand
 * object-like #define value-macros under --bootstrap-mode. Values copied
 * verbatim from the vendored hex2.h; the two struct defs are copied verbatim. */
enum
{
	max_string = 4096,
	KNIGHT = 0,
	X86 = 0x03,
	AMD64 = 0x3E,
	ARMV7L = 0x28,
	AARM64 = 0xB7,
	PPC64LE = 0x15,
	RISCV32 = 0xF3,
	RISCV64 = 0x100F3,
	HEX = 16,
	OCTAL = 8,
	BINARY = 2,
};

struct input_files
{
	struct input_files* next;
	char* filename;
};

struct entry
{
	struct entry* next;
	unsigned target;
	char* name;
};
EOF

# 0b. Generate the two libc functions hex2 needs that the vendored 3-file
#     M2libc bootstrap subset does NOT provide (M2-Planet's own cc.c needs
#     neither, so they were never added):
#       * fflush -- this M2libc does DIRECT unbuffered write() syscalls
#                   (see fputc/fwrite in M2libc/bootstrap.c), so fflush is a
#                   semantic no-op. Return 0.
#       * chmod  -- amd64/linux syscall 90 (0x5A). hex2 uses it to mark its
#                   output executable. 2-arg asm idiom mirrors open()/close()
#                   in M2libc/amd64/linux/bootstrap.c: with 2 args, arg1 is at
#                   [rsp+DWORD] %16 (-> rdi), arg2 at [rsp+DWORD] %8 (-> rsi).
cat > "$T/hex2_libc_extra.m2.c" <<'EOF'
int fflush(FILE* f)
{
	return 0;
}

int chmod(char* pathname, int mode)
{
	asm("lea_rdi,[rsp+DWORD] %16"
	    "mov_rdi,[rdi]"
	    "lea_rsi,[rsp+DWORD] %8"
	    "mov_rsi,[rsi]"
	    "mov_rax, %90"
	    "syscall");
}
EOF

# 1. M2 compiles the whole hex2 family + bootstrap C units -> M1.
"$M2" --architecture amd64 \
    -f "$MLIB/amd64/linux/bootstrap.c" \
    -f "$MLIB/bootstrap.c" \
    -f "$MLIB/bootstrappable.c" \
    -f "$T/hex2_libc_extra.m2.c" \
    -f "$T/hex2_defs.m2.h" \
    -f hex2_globals.h \
    -f hex2_linker.c \
    -f hex2_word.c \
    -f hex2.c \
    --bootstrap-mode -o "$T/hex2.M1"
echo "M2 -> hex2.M1: $(stat -c%s "$T/hex2.M1") bytes"

# 2. Assemble with M2libc amd64 defs + libc-core.
"$CATM" "$T/hex2-0.M1" "$MDEFS" "$MLIBC" "$T/hex2.M1"
"$M0"   "$T/hex2-0.M1" "$T/hex2.hex2"
echo "M0 -> hex2.hex2: $(stat -c%s "$T/hex2.hex2") bytes"

# 3. Link.
"$CATM" "$T/hex2-0.hex2" "$MELF" "$T/hex2.hex2"
"$HEX2" "$T/hex2-0.hex2" "$OUT"
chmod +x "$OUT"
echo "Built $OUT: $(stat -c%s "$OUT") bytes"

# 4. ELF sanity.
if ! file "$OUT" | grep -q "ELF 64-bit LSB executable"; then
    echo "ERROR: $OUT is not a valid x86-64 ELF" >&2; file "$OUT"; exit 1
fi
echo "ELF: $(file -b "$OUT")"

# 5. Capability: the NEW hex2 must ACCEPT flags the old positional one rejected
#    (--base-address / --architecture / --little-endian) AND complete a link.
#    We link the real M2libc debug ELF header + a minimal text body. That header
#    references :ELF_end (its comment: "add :ELF_end to the end of your hex2
#    files"), so the body supplies it. A nonempty executable ELF out proves
#    flag parsing + both link passes ran.
#    The debug header references &_start (e_entry) and %ELF_end (sizes), so the
#    body defines :_start (a real exit(0) stub: mov edi,0; mov eax,60; syscall)
#    and :ELF_end. The linked ELF is then actually RUN -> must exit 0.
printf '%s\n' ':_start' 'BF 00 00 00 00' 'B8 3C 00 00 00' '0F 05' ':ELF_end' > "$T/body.hex2"
"./$OUT" -f "$MELF" -f "$T/body.hex2" --little-endian --architecture amd64 --base-address 0x00600000 -o "$T/out.bin"; rc=$?
echo "hex2.bin link rc=$rc; out.bin = $([ -s "$T/out.bin" ] && stat -c%s "$T/out.bin" || echo 0) bytes"
if [ "$rc" != "0" ] || [ ! -s "$T/out.bin" ]; then
    echo "ERROR: hex2.bin did not link with flag-driven CLI (rc=$rc)" >&2; exit 1
fi
if ! file "$T/out.bin" | grep -q "ELF 64-bit"; then
    echo "ERROR: hex2.bin output is not an ELF" >&2; file "$T/out.bin"; exit 1
fi
chmod +x "$T/out.bin"; "$T/out.bin"; erc=$?
echo "linked ELF exit=$erc (want 0)"
if [ "$erc" != "0" ]; then echo "ERROR: hex2-linked ELF did not run cleanly (exit $erc)" >&2; exit 1; fi
echo "OK: hex2.bin runs, accepts --base-address/--architecture/--little-endian/-f/-o, links a RUNNABLE ELF."
