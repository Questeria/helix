#!/usr/bin/env bash
# selfhost_v2.sh -- H6 (trust criterion) self-host via the UPSTREAM mescc-tools
# assemble path: M2.bin --debug --bootstrap-mode  ->  blood-elf footer  ->
# M1 (macro assembler)  ->  hex2 (flag-driven linker, --base-address).
#
# WHY a v2: our stage0 M0 (positional) corrupts exactly one lea_rax,[rip+DWORD]
# on M2's 2.2MB self-output (-> illegal 0xEA -> SIGILL). Upstream NEVER uses our
# M0/positional-hex2 for self-host; it uses M1 + blood-elf + flag-driven hex2.
# We vendored mescc-tools 1.7.0 and built M1.bin/blood-elf.bin/hex2.bin with
# M2.bin (see ../M1/build.sh, ../blood-elf/build.sh, ../hex2-linker/build.sh).
# These tools are AUXILIARY verifiers built by M2; the main hex0..M2 ladder
# stays pure. A self-host fixpoint with this fixed assembler is a valid H6 proof.
#
# In --debug mode M2 emits :ELF_data and does NOT emit :ELF_end (cc.c:382/388);
# blood-elf supplies the :ELF_end label + the debug symbol table, and hex2 links
# against the DEBUG ELF header (ELF-amd64-debug.hex2, which carries the section
# headers the symtab needs).
#
# GATES:
#   G1: gen2-M2.bin compiles `int main(){return 42;}` (same M1+blood-elf+hex2
#       assemble) -> run -> exit 42.
#   G2: gen2-M2.bin self-compiles the 12 sources -> gen3.M1 nonempty.
#   G3: sha256(gen2.M1) == sha256(gen3.M1)          [source-level fixpoint]
#   G4: sha256(gen2-M2.bin) == sha256(gen3-M2.bin)  [binary-level fixpoint]
#
# Run under WSL. Does NOT touch any frozen file. Sources GPL-3.0.
set -u
cd "$(dirname "$0")"

M2=./M2.bin
BLOODELF=../blood-elf/blood-elf.bin
M1=../M1/M1.bin
HEX2L=../hex2-linker/hex2.bin
MDEFS=M2libc/amd64/amd64_defs.M1
MLIBC=M2libc/amd64/libc-core.M1
MELFDBG=M2libc/amd64/ELF-amd64-debug.hex2
BASE=0x00600000

for t in "$M2" "$BLOODELF" "$M1" "$HEX2L"; do
    [ -x "$t" ] || chmod +x "$t" 2>/dev/null
    [ -s "$t" ] || { echo "FATAL: missing tool $t -- run its build.sh first"; exit 3; }
done

# The exact 12 M2-Planet sources, build.sh catm order, as -f units.
SRCS="-f M2libc/amd64/linux/bootstrap.c -f M2libc/bootstrap.c -f M2-Planet/cc.h -f M2libc/bootstrappable.c -f M2-Planet/cc_globals.c -f M2-Planet/cc_reader.c -f M2-Planet/cc_strings.c -f M2-Planet/cc_types.c -f M2-Planet/cc_emit.c -f M2-Planet/cc_core.c -f M2-Planet/cc_macro.c -f M2-Planet/cc.c"

T=$(mktemp -d)
echo "scratch: $T"

# --- Supplementary M1 macro definition (NOT a vendored-file edit) ---------
# M2's self-output emits `lea_rax,[rdi+DWORD]` (3x), but the vendored
# M2libc/amd64/amd64_defs.M1 defines every other reg-indexed lea EXCEPT this
# one (it has [rbp]/[rsp]/[rip]/[r13..r15]/[rdi for other dest regs] but no
# lea_rax,[rdi+DWORD]). M0 in the OLD pipeline never hit it the same way; the
# new M1 assembler correctly rejects the undefined macro
# ("Received invalid other; lea_rax,[rdi+DWORD]"). Rather than touch the
# provenance-pinned amd64_defs.M1 (sha 6357f709...), we supply the single
# missing DEFINE as an extra -f unit. M1 accumulates DEFINEs across all -f
# files (it tokenizes them all, then line_macro() resolves once), so this is
# equivalent to having the line in amd64_defs.M1.
#   Encoding: lea rax,[rdi+disp32], REX.W=48, opcode 8D,
#   ModRM = mod(10) reg(000=rax) rm(111=rdi) = 0x87  ->  48 8D 87 = 488D87
#   (cross-check: lea_rax,[r15+DWORD]=498D87 -- same ModRM 87, REX.B set for r15;
#    rdi is a low reg so no REX.B, hence 48 not 49.)
DEFEXT="$T/amd64_defs_ext.M1"
printf 'DEFINE lea_rax,[rdi+DWORD] 488D87\n' > "$DEFEXT"

# assemble_upstream <compiler.bin> <out.M1-target> <final.bin> <SRCS...>
#   Runs <compiler.bin> --debug --bootstrap-mode over the given sources to make
#   the .M1, then blood-elf footer -> M1 link -> hex2 link -> final ELF.
#   Writes the produced .M1 to <out.M1-target> for fixpoint comparison.
assemble_upstream() {
    local cc="$1"; local m1out="$2"; local finalbin="$3"; shift 3
    local tag; tag=$(basename "$m1out" .M1)
    # 1. compile to M1 (debug + bootstrap)
    "$cc" --architecture amd64 "$@" --debug --bootstrap-mode -o "$m1out" 2>"$T/$tag.cc.err"; local crc=$?
    if [ ! -s "$m1out" ]; then
        echo "  [$tag] compile FAILED rc=$crc:"; head -4 "$T/$tag.cc.err"; return 21
    fi
    echo "  [$tag] compile rc=$crc -> $(stat -c%s "$m1out") bytes M1"
    # 2. blood-elf footer (provides :ELF_end + debug symtab)
    "$BLOODELF" --64 -f "$m1out" --little-endian --entry _start -o "$T/$tag-footer.M1" 2>"$T/$tag.be.err"; local berc=$?
    if [ ! -s "$T/$tag-footer.M1" ]; then
        echo "  [$tag] blood-elf FAILED rc=$berc:"; head -4 "$T/$tag.be.err"; return 22
    fi
    echo "  [$tag] blood-elf rc=$berc -> $(stat -c%s "$T/$tag-footer.M1") bytes footer"
    # 3. M1 assemble: defs (+ the one supplementary DEFINE) + libc-core +
    #    program + footer -> hex2
    "$M1" -f "$MDEFS" -f "$DEFEXT" -f "$MLIBC" -f "$m1out" -f "$T/$tag-footer.M1" --little-endian --architecture amd64 -o "$T/$tag.hex2" 2>"$T/$tag.m1.err"; local mrc=$?
    if [ ! -s "$T/$tag.hex2" ]; then
        echo "  [$tag] M1 assemble FAILED rc=$mrc:"; head -4 "$T/$tag.m1.err"; return 23
    fi
    echo "  [$tag] M1 assemble rc=$mrc -> $(stat -c%s "$T/$tag.hex2") bytes hex2"
    # 4. hex2 link: debug ELF header + program hex2 -> final ELF
    "$HEX2L" -f "$MELFDBG" -f "$T/$tag.hex2" --little-endian --architecture amd64 --base-address "$BASE" -o "$finalbin" 2>"$T/$tag.hx.err"; local hrc=$?
    if [ ! -s "$finalbin" ]; then
        echo "  [$tag] hex2 link FAILED rc=$hrc:"; head -4 "$T/$tag.hx.err"; return 24
    fi
    chmod +x "$finalbin"
    echo "  [$tag] hex2 link rc=$hrc -> $(stat -c%s "$finalbin") bytes ELF ($(file -b "$finalbin" 2>/dev/null | cut -c1-28))"
    return 0
}

echo ""
echo "=== STEP 1: gen2 = M2.bin self-compiles the 12 sources (upstream assemble) ==="
assemble_upstream "$M2" "$T/gen2.M1" "$T/gen2-M2.bin" $SRCS || { echo "VERDICT: H6_BLOCKED (gen2 build)"; rm -rf "$T"; exit 1; }

echo ""
echo "=== GATE G1: gen2-M2.bin compiles 'int main(){return 42;}' -> run -> 42 ==="
printf 'int main() { return 42; }\n' > "$T/r42.c"
G1SRCS="-f M2libc/amd64/linux/bootstrap.c -f M2libc/bootstrap.c -f M2libc/bootstrappable.c -f $T/r42.c"
assemble_upstream "$T/gen2-M2.bin" "$T/r42.M1" "$T/r42.bin" $G1SRCS; arc=$?
if [ "$arc" != "0" ]; then echo "G1=FAIL (r42 assemble rc=$arc)"; echo "VERDICT: H6_BLOCKED (G1 assemble)"; rm -rf "$T"; exit 1; fi
"$T/r42.bin"; g1rc=$?
echo "  G1: r42.bin exit=$g1rc (want 42)"
if [ "$g1rc" = "42" ]; then G1=PASS; else G1=FAIL; fi
echo "  >>> G1=$G1"

echo ""
echo "=== GATE G2: gen2-M2.bin self-compiles the 12 sources -> gen3.M1 nonempty ==="
assemble_upstream "$T/gen2-M2.bin" "$T/gen3.M1" "$T/gen3-M2.bin" $SRCS; arc=$?
if [ "$arc" != "0" ]; then
    # G2 only requires gen3.M1 nonempty; the .bin steps may matter for G4 only.
    if [ -s "$T/gen3.M1" ]; then echo "  (note: gen3 full assemble rc=$arc but gen3.M1 produced)"; else echo "G2=FAIL (gen3.M1 empty)"; echo "VERDICT: H6_BLOCKED (G2)"; rm -rf "$T"; exit 1; fi
fi
if [ -s "$T/gen3.M1" ]; then G2=PASS; else G2=FAIL; fi
echo "  gen3.M1 = $([ -s "$T/gen3.M1" ] && stat -c%s "$T/gen3.M1" || echo 0) bytes"
echo "  >>> G2=$G2"

echo ""
echo "=== GATE G3: sha256(gen2.M1) == sha256(gen3.M1) [source fixpoint] ==="
s2=$(sha256sum "$T/gen2.M1" | cut -d' ' -f1)
s3=$(sha256sum "$T/gen3.M1" 2>/dev/null | cut -d' ' -f1)
echo "  gen2.M1 sha=$s2"
echo "  gen3.M1 sha=$s3"
if [ -n "$s3" ] && [ "$s2" = "$s3" ]; then G3=PASS; else G3=FAIL; fi
echo "  >>> G3=$G3"

echo ""
echo "=== GATE G4: sha256(gen2-M2.bin) == sha256(gen3-M2.bin) [binary fixpoint] ==="
b2=$(sha256sum "$T/gen2-M2.bin" 2>/dev/null | cut -d' ' -f1)
b3=$(sha256sum "$T/gen3-M2.bin" 2>/dev/null | cut -d' ' -f1)
echo "  gen2-M2.bin sha=$b2"
echo "  gen3-M2.bin sha=$b3"
if [ -n "$b2" ] && [ -n "$b3" ] && [ "$b2" = "$b3" ]; then G4=PASS; else G4=FAIL; fi
echo "  >>> G4=$G4"

echo ""
echo "================ H6 SELF-HOST v2 SUMMARY ================"
echo "  G1 (gen2 compiles+runs return 42): $G1"
echo "  G2 (gen2 self-compiles -> gen3.M1): $G2"
echo "  G3 (gen2.M1 == gen3.M1):            $G3"
echo "  G4 (gen2-M2.bin == gen3-M2.bin):    $G4"
if [ "$G1" = "PASS" ] && [ "$G2" = "PASS" ] && [ "$G3" = "PASS" ] && [ "$G4" = "PASS" ]; then
    echo "  VERDICT: H6_GREEN"
else
    echo "  VERDICT: H6_BLOCKED (see failing gate above)"
fi
echo "========================================================"

# Leave scratch for diagnosis if any gate failed; else clean.
if [ "$G1" = "PASS" ] && [ "$G2" = "PASS" ] && [ "$G3" = "PASS" ] && [ "$G4" = "PASS" ]; then
    rm -rf "$T"
else
    echo "(scratch kept at $T for diagnosis)"
fi
