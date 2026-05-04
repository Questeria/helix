#!/usr/bin/env python3
"""
hex0_reference.py — Python reference implementation of hex0.

NOT a shipped artifact. Used as a behavioral oracle: hex0.bin (the hand-encoded
binary) and hex0.nasm.bin (the nasm cross-check) must produce identical output
to this script for every input in test/.

Same lenient parsing as hex0.s:
- Whitespace (space, tab, \\n, \\r) skipped.
- ';' or '#' starts a comment to end-of-line.
- Hex digits 0-9, A-F, a-f. Pairs combine high-nibble-first.
- Other characters silently skipped.
"""

import sys


def hex0(input_bytes: bytes) -> bytes:
    output = bytearray()
    high = None  # buffered high nibble (0..15) or None
    in_comment = False

    for b in input_bytes:
        c = chr(b)

        if in_comment:
            if c == "\n":
                in_comment = False
            continue

        if c == ";" or c == "#":
            in_comment = True
            continue

        if c in (" ", "\t", "\n", "\r"):
            continue

        if "0" <= c <= "9":
            nibble = b - 0x30
        elif "A" <= c <= "F":
            nibble = b - 0x37  # 0x41 -> 10, ..., 0x46 -> 15
        elif "a" <= c <= "f":
            nibble = b - 0x57  # 0x61 -> 10, ..., 0x66 -> 15
        else:
            continue  # invalid: silently skip

        if high is None:
            high = nibble
        else:
            output.append((high << 4) | nibble)
            high = None

    return bytes(output)


def main() -> int:
    data = sys.stdin.buffer.read()
    sys.stdout.buffer.write(hex0(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
