"""
helixc/ir/mlir/validate.py — MLIR-text validation
(v3.0 Phase E, Stages 211 + 213).

`mock_validate_mlir` is the mock-path MLIR validator: a toolchain-free
STRUCTURAL shape check on MLIR textual IR, the MLIR analogue of
`helixc.backend.llvm_ir.mock_validate_ll`. It runs in CI on a machine
with no MLIR toolchain — it never `import mlir`, never shells out to
`mlir-opt`.

It is NOT a real verifier. Real MLIR verification — verifier traits,
type-correctness, SSA dominance — needs `mlir-opt` (or the in-process
bindings) and is built in Stage 212. So `mock_validate_mlir` returns a
frozen tri-state `MLIRValidation`:

- FAILED — a definite STRUCTURAL defect (non-str / empty input, no
  top-level structure, an unterminated string literal, unbalanced
  braces / parentheses). A malformed shape is malformed regardless of
  any toolchain, so the mock check FAILS with confidence.
- DEFERRED — the shape check found no defect, but that is NOT a
  certification of real MLIR validity; a real check is needed and was
  not run. This is the honest outcome for well-formed text from a
  toolchain-free checker — never a false PASSED.
- PASSED — reserved for the Stage-212 REAL validator (a successful
  `mlir-opt` verification). `mock_validate_mlir`, being toolchain-free,
  NEVER returns PASSED — it can only confidently FAIL or honestly
  DEFER. PASSED is in the tri-state so the one `MLIRValidation` type
  serves both the mock and the future real validator.

Stage 213 chunk B adds `validate_mlir_with_toolchain`, the first real
validation dispatch seam: it still runs the mock shape check first,
then invokes `mlir-opt` when that tool is available. A tool-less
machine continues to return DEFERRED — never a false PASS — so CI on a
binding-less runner stays green and the home-grown tile-IR path stays
the reversible fallback.

License: Apache 2.0
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .toolchain import MLIRSupport, detect_mlir_support


class MLIRValidationVerdict(Enum):
    """The tri-state outcome of an MLIR-text validation — the Stage 210
    decision's mock-path tri-state (section 3).

    `mock_validate_mlir` produces only FAILED (a definite structural
    defect) and DEFERRED (no defect found, real validity unverified);
    PASSED is reserved for the Stage-212 real `mlir-opt` validator."""
    PASSED = "passed"
    FAILED = "failed"
    DEFERRED = "deferred"


def _check_validation_verdicts() -> None:
    """Module-load guard: `MLIRValidationVerdict` is exactly the
    tri-state {PASSED, FAILED, DEFERRED} the Stage 210 decision's
    mock-path discipline (section 3) defines — no more, no less. A
    fourth verdict added without updating `MLIRValidation.__post_init__`
    (which branches on each verdict) and `mock_validate_mlir`
    would silently widen the contract. Mirrors the module-load drift
    guards of `toolchain.py` / `mapping.py`."""
    names = {v.name for v in MLIRValidationVerdict}
    if names != {"PASSED", "FAILED", "DEFERRED"}:
        raise AssertionError(
            f"helixc.ir.mlir.validate: MLIRValidationVerdict must be "
            f"exactly PASSED / FAILED / DEFERRED — got {sorted(names)}")


_check_validation_verdicts()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class MLIRValidation:
    """The result of validating a piece of MLIR textual IR — a frozen
    tri-state verdict plus the findings that explain it.

    Frozen + `__post_init__`-guarded, the house discipline of
    `toolchain.MLIRSupport`:
    - a FAILED or DEFERRED result MUST carry at least one finding — it
      is never silent about why (the mock-path rule);
    - a PASSED result MUST carry NO findings — a clean pass has
      nothing to report; `findings` describes a defect or a deferral
      reason, and a PASSED has neither, so a PASSED with findings is
      an incoherent result and is rejected;
    - every finding carries text.

    PASSED status is additionally bound to the real-validator runner's
    identity registry for normal public construction, copy, and pickle
    paths. This is an integrity check inside Helix's own code, not a
    Python security boundary against adversarial same-process
    introspection.
    """
    verdict: MLIRValidationVerdict
    findings: tuple[str, ...]
    provenance: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        raise TypeError(
            "MLIRValidation is final; subclassing could bypass "
            "the validator result invariants")

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, MLIRValidationVerdict):
            raise ValueError(
                f"MLIRValidation: verdict must be a "
                f"MLIRValidationVerdict — got {self.verdict!r}")
        if not isinstance(self.findings, tuple):
            raise ValueError(
                "MLIRValidation: findings must be a tuple, got "
                f"{type(self.findings).__name__}")
        for entry in self.findings:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    f"MLIRValidation: findings has a blank or non-str "
                    f"entry ({entry!r}) — every finding carries text")
        if not isinstance(self.provenance, tuple):
            raise ValueError(
                "MLIRValidation: provenance must be a tuple, got "
                f"{type(self.provenance).__name__}")
        for entry in self.provenance:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    "MLIRValidation: provenance has a blank or non-str "
                    f"entry ({entry!r})")
        if (self.verdict in (MLIRValidationVerdict.FAILED,
                             MLIRValidationVerdict.DEFERRED)
                and not self.findings):
            raise ValueError(
                f"MLIRValidation: a {self.verdict.name} result must "
                f"carry at least one finding explaining why — it must "
                f"never be silent about a defect or a deferral")
        if self.verdict is MLIRValidationVerdict.PASSED and self.findings:
            raise ValueError(
                f"MLIRValidation: a PASSED result must carry NO "
                f"findings ({len(self.findings)} given) — a clean pass "
                f"has nothing to report; `findings` describes a defect "
                f"or a deferral reason, and a PASSED has neither, so a "
                f"PASSED carrying a finding is an incoherent result")
        if self.verdict is MLIRValidationVerdict.PASSED:
            if not self.provenance:
                raise ValueError(
                    "MLIRValidation: a PASSED result must carry "
                    "toolchain provenance")
            if not _has_real_validation_pass_shape(self):
                raise ValueError(
                    "MLIRValidation: a PASSED result must carry a "
                    "coherent real-validator registry entry")
            raise ValueError(
                "MLIRValidation: a PASSED result must be created by the "
                "real validator after successful toolchain validation")
        elif self.provenance:
            raise ValueError(
                "MLIRValidation: only PASSED results may carry "
                "toolchain provenance")

    def passed(self) -> bool:
        """True iff a real validator confirmed the IR is valid."""
        return _has_real_validation_pass_shape(self)

    def failed(self) -> bool:
        """True iff a definite structural defect was found."""
        return self.verdict is MLIRValidationVerdict.FAILED

    def deferred(self) -> bool:
        """True iff no defect was found but real validity is unverified
        — the honest mock-path outcome for well-formed text."""
        return self.verdict is MLIRValidationVerdict.DEFERRED

    def is_positive_assertion(self) -> bool:
        """True iff this is a CHECKED-pass result — real `mlir-opt`
        confirmed the IR is valid. Stage 216 close-audit MEDIUM-1:
        release-gate callers MUST use this instead of `not failed()`
        — the latter is silently DEFERRED-permissive on toolchain-
        less CI machines, which would ship unverified IR. Same
        discipline as `ParityResult.is_positive_assertion()` and
        `MLIRBackendResult.is_positive_assertion()`."""
        return self.passed()

    def __copy__(self) -> "MLIRValidation":
        return _copy_mlir_validation(self)

    def __deepcopy__(self, memo: dict[int, object]) -> "MLIRValidation":
        return _copy_mlir_validation(self, memo)

    def __reduce_ex__(self, protocol: int) -> object:
        if _has_real_validation_pass_shape(self):
            raise TypeError(
                "PASSED MLIRValidation is a runner-registry entry "
                "and cannot be pickled")
        return _reduce_mlir_validation(self)


_GENERIC_OP_SOURCE_TOKEN = "_generic_op"
_GENERIC_OP_SENTINEL = "\x00helix_generic_op\x00"
_TOP_LEVEL_STRUCTURE_TOKENS: tuple[str, ...] = (
    "builtin.module", "func.func", "module", _GENERIC_OP_SENTINEL,
)

# Wall-clock cap on the real `mlir-opt` verifier dispatch. Small MLIR
# text should validate quickly; the cap exists only to avoid hanging on
# a broken tool.
_MLIR_VALIDATE_TIMEOUT_S = 30
_PASS_PROVENANCE_DIGEST_PREFIXES = (
    "input_sha256=", "output_sha256=",
)
_TOOL_DIAGNOSTIC_PREFIXES = (
    "error:", "fatal:", "failed:", "note:", "traceback",
    "remark:", "warning:",
)


def _captured_tool_diagnostic(stdout: str | None,
                              stderr: str | None) -> str:
    for stream_name, stream in (("stderr", stderr), ("stdout", stdout)):
        if not stream:
            continue
        for line in stream.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if any(lowered.startswith(prefix)
                   or f": {prefix}" in lowered
                   for prefix in _TOOL_DIAGNOSTIC_PREFIXES):
                return f"{stream_name}: {stripped[:500]}"
    return ""


def _mlir_invalid_smoke_specs(nonce: str) -> tuple[tuple[str, str], ...]:
    names = tuple(
        "_" + hashlib.sha256(f"{nonce}:{index}".encode(
            "ascii")).hexdigest()[:16]
        for index in range(4)
    )
    return (
        (
            "invalid-IR smoke",
            f"module {{ func.func @{names[0]}() {{ "
            f"return %{names[1]} : i32 }} }}\n",
        ),
        (
            "invalid-type smoke",
            f"module {{ func.func @{names[2]}() -> i32 {{ return }} }}\n",
        ),
        (
            "fresh invalid-type smoke",
            f"module {{ func.func @{names[3]}() -> i32 {{ return }} }}\n",
        ),
    )


def _mlir_text_is_invalid_smoke_probe(mlir_text: str) -> bool:
    compact = " ".join(mlir_text.split())
    return (
        "func.func @" in compact
        and (
            ("() { return %" in compact and " : i32 } }" in compact)
            or "() -> i32 { return }" in compact
        )
    )


def _mlir_opt_path_matches_fresh_probe(mlir_opt: str) -> bool:
    try:
        support = detect_mlir_support()
    except Exception:
        return False
    return isinstance(support, MLIRSupport) and support.mlir_opt == mlir_opt


def _structural_text(mlir_text: str) -> str:
    """`mlir_text` with string literals and comments removed,
    so brace / parenthesis validation sees only structural punctuation.

    The scanner treats comment markers inside strings as string content
    and `"` inside comments as comment content. Quoted generic op names such as
    `"builtin.module"()` are preserved as tokens so valid generic MLIR
    can reach real validation. Other strings are masked. A dangling
    string or raw newline inside a string leaves a sentinel quote so the
    caller can report the malformed literal before trusting delimiter
    checks."""

    def _next_nonspace(start: int) -> str:
        j = start
        while j < len(mlir_text) and mlir_text[j].isspace():
            j += 1
        return mlir_text[j] if j < len(mlir_text) else ""

    out: list[str] = []
    in_line_comment = False
    in_block_comment = False
    i = 0
    while i < len(mlir_text):
        char = mlir_text[i]
        nxt = mlir_text[i + 1] if i + 1 < len(mlir_text) else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                out.append(char)
            i += 1
            continue
        if in_block_comment:
            if char == "\n":
                out.append(char)
                i += 1
                continue
            if char == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if char == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if char == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if char == '"':
            i += 1
            quoted: list[str] = []
            escaping = False
            had_escape = False
            closed = False
            invalid_raw_newline = False
            while i < len(mlir_text):
                qchar = mlir_text[i]
                if escaping:
                    if qchar in "\n\r\v\f":
                        invalid_raw_newline = True
                        break
                    quoted.append(qchar)
                    escaping = False
                    i += 1
                    continue
                if qchar == "\\":
                    had_escape = True
                    escaping = True
                    i += 1
                    continue
                if qchar == '"':
                    closed = True
                    i += 1
                    break
                if qchar in "\n\r\v\f":
                    invalid_raw_newline = True
                    break
                quoted.append(qchar)
                i += 1
            if invalid_raw_newline or not closed:
                out.append('"')
                break
            quoted_text = "".join(quoted)
            if out and out[-1] == "@":
                normalized = "" if had_escape else _normalize_symbol_ref(
                    '@"' + quoted_text + '"')
                out.append(
                    normalized[1:]
                    if normalized.startswith("@")
                    and not normalized.startswith('@"')
                    else "_quoted_symbol")
            elif not had_escape and _next_nonspace(i) == "(":
                out.append(_GENERIC_OP_SENTINEL)
                out.append(" ")
            else:
                out.append(" string_lit ")
            continue
        out.append(char)
        i += 1

    if in_block_comment:
        out.append(" unterminated_block_comment ")
    return "".join(out)


def _skip_spaces(text: str, start: int) -> int:
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _read_bare_word(text: str, start: int) -> tuple[str, int]:
    i = start
    if i >= len(text) or not (text[i].isalpha() or text[i] == "_"):
        return "", start
    i += 1
    while i < len(text):
        char = text[i]
        if not (char.isalnum() or char in "_$."):
            break
        i += 1
    return text[start:i], i


def _block_label_line_is_plausible(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("^"):
        return False
    i = 1
    if i >= len(stripped) or not (stripped[i].isalnum()
                                  or stripped[i] == "_"):
        return False
    i += 1
    while i < len(stripped) and (
            stripped[i].isalnum() or stripped[i] in "_$.-"):
        i += 1
    i = _skip_spaces(stripped, i)
    if i < len(stripped) and stripped[i] == "(":
        args_end = _matching_closer_index(stripped, i, "(", ")")
        if args_end is None:
            return False
        args = stripped[i + 1:args_end].strip()
        if args:
            parts = _split_depth_zero_commas(args)
            if parts is None:
                return False
            for part in parts:
                colon = _depth_zero_colon_index(part, 0)
                if colon is None:
                    return False
                if not _ssa_result_list_is_plausible(part[:colon]):
                    return False
                if not _bare_dialect_type_tail_is_plausible(
                        part[colon + 1:]):
                    return False
        i = _skip_spaces(stripped, args_end + 1)
    if i >= len(stripped) or stripped[i] != ":":
        return False
    return not stripped[i + 1:].strip()


def _is_lowercase_sha256(text: str) -> bool:
    return (len(text) == 64
            and all(char in "0123456789abcdef" for char in text))


def _validation_pass_provenance_is_coherent(
        result: MLIRValidation,
        expected_provenance: tuple[str, ...] | None = None) -> bool:
    if type(result) is not MLIRValidation:
        return False
    if result.verdict is not MLIRValidationVerdict.PASSED:
        return False
    if result.findings:
        return False
    provenance = getattr(result, "provenance", None)
    if not isinstance(provenance, tuple):
        return False
    if expected_provenance is not None and provenance != expected_provenance:
        return False
    if not any(isinstance(entry, str) and entry.startswith("mlir-opt=")
               and entry != "mlir-opt="
               for entry in provenance):
        return False
    if not any(
            isinstance(entry, str) and entry.startswith("artifact_name=")
            and entry != "artifact_name="
            for entry in provenance):
        return False
    for prefix in _PASS_PROVENANCE_DIGEST_PREFIXES:
        matches = [
            entry[len(prefix):]
            for entry in provenance
            if isinstance(entry, str) and entry.startswith(prefix)
        ]
        if len(matches) != 1 or not _is_lowercase_sha256(matches[0]):
            return False
    return True


def _validation_pass_payload(result: MLIRValidation) -> tuple[str, ...]:
    return (
        "verdict=" + result.verdict.value,
        "findings=" + repr(result.findings),
        *result.provenance,
    )


def _make_mlir_validation_pass_registry():
    passes: dict[
        int,
        tuple[weakref.ReferenceType[MLIRValidation], tuple[str, ...]],
    ] = {}

    def _brand(result: MLIRValidation) -> MLIRValidation:
        passes[id(result)] = (
            weakref.ref(result), _validation_pass_payload(result))
        return result

    def _has(result: MLIRValidation) -> bool:
        if type(result) is not MLIRValidation:
            return False
        entry = passes.get(id(result))
        if entry is None:
            return False
        ref, payload = entry
        if ref() is not result:
            passes.pop(id(result), None)
            return False
        return payload == _validation_pass_payload(result)

    return _brand, _has


def _reduce_mlir_validation(result: MLIRValidation) -> object:
    return (
        MLIRValidation,
        (result.verdict, result.findings, result.provenance),
    )


def _copy_mlir_validation(
        result: MLIRValidation, memo: dict[int, object] | None = None,
        ) -> MLIRValidation:
    if _has_real_validation_pass_shape(result):
        if memo is not None:
            memo[id(result)] = result
        return result
    return MLIRValidation(result.verdict, result.findings,
                          result.provenance)


def _read_bare_op_name(text: str, start: int) -> tuple[str, int]:
    i = start
    if i >= len(text) or not (text[i].isalpha() or text[i] == "_"):
        return "", start
    i += 1
    while i < len(text):
        char = text[i]
        if not (char.isalnum() or char in "_$."):
            break
        i += 1
    name = text[start:i]
    if name.endswith("."):
        return "", start
    return (name, i) if "." in name else ("", start)


def _is_word_char(char: str) -> bool:
    return bool(char) and (char.isalnum() or char in "_$.")


def _bare_word_at(text: str, start: int, word: str) -> bool:
    end = start + len(word)
    if not text.startswith(word, start):
        return False
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not _is_word_char(before) and not _is_word_char(after)


def _matching_closer_index(text: str, start: int,
                          opener: str, closer: str) -> int | None:
    depth = 0
    i = start
    while i < len(text):
        char = text[i]
        if char == '"':
            quoted_end = _quoted_span_end(text, i)
            if quoted_end <= i + 1 or text[quoted_end - 1] != '"':
                return None
            i = quoted_end
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _line_end(text: str, start: int) -> int:
    end = text.find("\n", start)
    return len(text) if end == -1 else end


def _read_symbol_ref_after_at(text: str, start: int) -> int | None:
    """Return the index after a structural symbol reference."""
    if start >= len(text) or text[start] != "@":
        return None
    i = start + 1
    line_end = _line_end(text, start)
    if i >= line_end:
        return None
    if text[i].isspace():
        return None
    if not (text[i].isalnum() or text[i] in "_$"):
        return None
    while i < line_end and (
            text[i].isalnum() or text[i] in "_$.-"):
        i += 1
    return i if i > start + 1 else None


def _return_type_text_is_plausible(text: str) -> bool:
    """Reject obvious same-line junk after a function return type."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("("):
        close = _matching_closer_index(stripped, 0, "(", ")")
        if close != len(stripped) - 1:
            return False
        inner = stripped[1:close].strip()
        if not inner:
            return True
        parts = _split_depth_zero_commas(inner)
        if parts is None:
            return False
        return all(_result_type_item_is_plausible(part) for part in parts)
    return _result_type_item_is_plausible(stripped)


def _parenthesized_type_list_is_plausible(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("(") or not stripped.endswith(")"):
        return False
    close = _matching_closer_index(stripped, 0, "(", ")")
    if close != len(stripped) - 1:
        return False
    inner = stripped[1:close].strip()
    if not inner:
        return True
    parts = _split_depth_zero_commas(inner)
    if parts is None:
        return False
    return all(_result_type_item_is_plausible(part) for part in parts)


def _result_type_item_is_plausible(text: str) -> bool:
    stripped = text.strip()
    if _single_type_text_is_plausible(stripped):
        return True
    attr_start = _depth_zero_brace_index(stripped)
    if attr_start is None:
        return False
    ty_text = stripped[:attr_start].strip()
    if not _single_type_text_is_plausible(ty_text):
        return False
    attr_end = _matching_closer_index(stripped, attr_start, "{", "}")
    if attr_end is None or attr_end != len(stripped) - 1:
        return False
    return bool(stripped[attr_start + 1:attr_end].strip())


def _single_type_text_is_plausible(text: str) -> bool:
    """True for one MLIR type token, allowing spaces only in nested
    angle/paren/bracket contexts."""
    stripped = text.strip()
    if not stripped:
        return False
    pairs = {"(": ")", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    saw_type_char = False
    i = 0
    while i < len(stripped):
        char = stripped[i]
        if char == "-" and i + 1 < len(stripped) and stripped[i + 1] == ">":
            if not stack:
                return False
            saw_type_char = True
            i += 2
            continue
        if char.isspace() and not stack:
            return False
        if char == "," and not stack:
            return False
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers:
            if not stack or stack.pop() != char:
                return False
        elif not char.isspace():
            saw_type_char = True
        i += 1
    return saw_type_char and not stack


def _bare_dialect_type_tail_is_plausible(text: str) -> bool:
    """Reject obvious same-line junk after a bare dialect op type tail."""
    if _return_type_text_is_plausible(text):
        return True
    if _function_type_tail_is_plausible(text):
        return True
    if _type_list_text_is_plausible(text):
        return True
    parts = text.strip().split()
    if len(parts) != 3 or parts[1] != "to":
        return False
    return (_single_type_text_is_plausible(parts[0])
            and _single_type_text_is_plausible(parts[2]))


def _split_depth_zero_commas(text: str) -> list[str] | None:
    pairs = {"(": ")", "{": "}", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    parts: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == '"':
            quoted_end = _quoted_span_end(text, i)
            if quoted_end <= i + 1 or text[quoted_end - 1] != '"':
                return None
            i = quoted_end
            continue
        if char == "-" and i + 1 < len(text) and text[i + 1] == ">":
            i += 2
            continue
        if char == "<" and i + 1 < len(text) and text[i + 1] == "=":
            i += 2
            continue
        if (char == ">" and
                ((i > 0 and text[i - 1] in "-=")
                 or (i + 1 < len(text) and text[i + 1] == "="))):
            i += 1
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers:
            if not stack or stack.pop() != char:
                return None
        elif char == "," and not stack:
            parts.append(text[start:i].strip())
            start = i + 1
        i += 1
    if stack:
        return None
    parts.append(text[start:].strip())
    return parts


def _depth_zero_brace_index(text: str) -> int | None:
    pairs = {"(": ")", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "-" and i + 1 < len(text) and text[i + 1] == ">":
            i += 2
            continue
        if char == "<" and i + 1 < len(text) and text[i + 1] == "=":
            i += 2
            continue
        if (char == ">" and
                ((i > 0 and text[i - 1] in "-=")
                 or (i + 1 < len(text) and text[i + 1] == "="))):
            i += 1
            continue
        if char == "{" and not stack:
            return i
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers:
            if not stack or stack.pop() != char:
                return None
        i += 1
    return None


def _type_list_text_is_plausible(text: str) -> bool:
    parts = _split_depth_zero_commas(text.strip())
    if parts is None or len(parts) <= 1:
        return False
    return all(_single_type_text_is_plausible(part) for part in parts)


def _function_type_tail_is_plausible(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("("):
        return False
    args_end = _matching_closer_index(stripped, 0, "(", ")")
    if args_end is None:
        return False
    j = _skip_spaces(stripped, args_end + 1)
    if not stripped.startswith("->", j):
        return False
    result_start = _skip_spaces(stripped, j + 2)
    return _return_type_text_is_plausible(stripped[result_start:])


def _next_func_section(text: str, start: int,
                       limit: int) -> tuple[str, int]:
    pairs = {"(": ")", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    i = start
    while i < limit:
        char = text[i]
        if char == '"':
            quoted_end = _quoted_span_end(text, i)
            if quoted_end <= i + 1 or text[quoted_end - 1] != '"':
                return limit
            i = quoted_end
            continue
        if char == "-" and i + 1 < limit and text[i + 1] == ">":
            i += 2
            continue
        if not stack:
            if i > start and (
                    _bare_word_at(text, i, "func.func")
                    or text.startswith(_GENERIC_OP_SENTINEL, i)):
                return "line_end", i
            if _bare_word_at(text, i, "attributes"):
                return "attributes", i
            if _bare_word_at(text, i, "loc"):
                return "loc", i
            if char == "{":
                return "body", i
            if char == "}":
                return "parent_close", i
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers and stack and stack[-1] == char:
            stack.pop()
        i += 1
    return "line_end", limit


def _skip_attributes(text: str, start: int) -> int | None:
    j = _skip_spaces(text, start)
    word, end = _read_bare_word(text, j)
    if word != "attributes":
        return j
    j = _skip_spaces(text, end)
    if j >= len(text) or text[j] != "{":
        return None
    attr_end = _matching_closer_index(text, j, "{", "}")
    if attr_end is None:
        return None
    return _skip_spaces(text, attr_end + 1)


def _module_body_start_after_metadata(structural: str,
                                      start: int) -> int | None:
    j = _skip_spaces(structural, start)
    if j < len(structural) and structural[j] == "@":
        after_symbol = _read_symbol_ref_after_at(structural, j)
        if after_symbol is None:
            return None
        j = _skip_spaces(structural, after_symbol)
    while True:
        after_attrs = _skip_attributes(structural, j)
        if after_attrs is None:
            return None
        if after_attrs == j:
            break
        j = after_attrs
    return j if j < len(structural) and structural[j] == "{" else None


def _has_brace_after_metadata(structural: str, start: int) -> bool:
    return _module_body_start_after_metadata(structural, start) is not None


def _next_op_boundary(text: str, start: int, limit: int) -> int:
    pairs = {"(": ")", "{": "}", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    at_depth_zero_line_start = False
    i = start
    while i < limit:
        char = text[i]
        if char == '"':
            quoted_end = _quoted_span_end(text, i)
            if quoted_end <= i + 1 or text[quoted_end - 1] != '"':
                return limit
            i = quoted_end
            continue
        if char == "-" and i + 1 < limit and text[i + 1] == ">":
            i += 2
            continue
        if char == "<" and i + 1 < limit and text[i + 1] == "=":
            i += 2
            continue
        if (char == ">" and
                ((i > 0 and text[i - 1] in "-=")
                 or (i + 1 < limit and text[i + 1] == "="))):
            i += 1
            continue
        if not stack:
            if char == "\n":
                at_depth_zero_line_start = True
                i += 1
                continue
            if at_depth_zero_line_start and char.isspace():
                i += 1
                continue
            if at_depth_zero_line_start:
                if char in "#!":
                    return i
                for token in _TOP_LEVEL_STRUCTURE_TOKENS:
                    if token == _GENERIC_OP_SENTINEL:
                        if text.startswith(token, i):
                            return i
                    elif _bare_word_at(text, i, token):
                        return i
                op_name, _name_end = _read_bare_op_name(text, i)
                if op_name:
                    return i
                at_depth_zero_line_start = False
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers and stack and stack[-1] == char:
            stack.pop()
        i += 1
    return limit


def _func_op_end(structural: str, start: int) -> int | None:
    j = _skip_spaces(structural, start)
    line_end = _next_op_boundary(structural, start, len(structural))
    if j >= len(structural):
        return None
    if structural[j] == "(":
        return None
    if structural[j] != "@":
        word, end = _read_bare_word(structural, j)
        if word not in ("private", "public", "nested"):
            return None
        j = _skip_spaces(structural, end)
    after_symbol = _read_symbol_ref_after_at(structural, j)
    if after_symbol is None:
        return None
    j = _skip_spaces(structural, after_symbol)
    if j >= line_end or structural[j] != "(":
        return None
    sig_close = _matching_closer_index(structural, j, "(", ")")
    if sig_close is None or sig_close > line_end:
        return None
    arg_text = structural[j + 1:sig_close]
    if not _func_arg_list_is_plausible(arg_text, allow_unnamed=True):
        return None
    j = _skip_spaces(structural, sig_close + 1)
    if structural.startswith("->", j):
        return_start = _skip_spaces(structural, j + 2)
        section, section_index = _next_func_section(
            structural, return_start, line_end)
        if not _return_type_text_is_plausible(
                structural[return_start:section_index]):
            return None
        if section == "attributes":
            after_attrs = _skip_attributes(structural, section_index)
            if after_attrs is None:
                return None
            section, section_index = _next_func_section(
                structural, after_attrs, line_end)
            if section == "line_end":
                if structural[after_attrs:line_end].strip():
                    return None
                j = line_end
            else:
                if structural[after_attrs:section_index].strip():
                    return None
                j = _skip_spaces(structural, section_index)
        elif section == "body":
            j = _skip_spaces(structural, section_index)
        elif section == "loc":
            loc_end = _skip_loc_suffix(structural, section_index)
            if loc_end == section_index:
                return None
            return loc_end
        elif section == "parent_close":
            j = section_index
        else:
            j = line_end
    elif structural.startswith("attributes", j):
        after_attrs = _skip_attributes(structural, j)
        if after_attrs is None:
            return None
        section, section_index = _next_func_section(
            structural, after_attrs, line_end)
        if section == "line_end":
            if structural[after_attrs:line_end].strip():
                return None
        else:
            if structural[after_attrs:section_index].strip():
                return None
        if section == "body":
            j = _skip_spaces(structural, section_index)
        elif section == "loc":
            loc_end = _skip_loc_suffix(structural, section_index)
            if loc_end == section_index:
                return None
            return loc_end
        elif section == "parent_close":
            j = section_index
        elif section == "line_end":
            j = line_end
        else:
            return None
    else:
        section, section_index = _next_func_section(structural, j, line_end)
        if structural[j:section_index].strip():
            return None
        if section == "body":
            j = _skip_spaces(structural, section_index)
        elif section == "loc":
            loc_end = _skip_loc_suffix(structural, section_index)
            if loc_end == section_index:
                return None
            return loc_end
        elif section == "parent_close":
            j = section_index
        elif section == "line_end":
            j = line_end
        else:
            return None
    if j < line_end and structural[j] == "}":
        return j
    after_attrs = _skip_attributes(structural, j)
    if after_attrs is None:
        return None
    j = after_attrs
    if j < line_end and structural[j] != "{":
        loc_end = _skip_loc_suffix(structural, j)
        if loc_end != j and not structural[loc_end:_line_end(
                structural, loc_end)].strip():
            return loc_end
        return None
    if j < len(structural) and structural[j] == "{":
        if not _func_arg_list_is_plausible(arg_text, allow_unnamed=False):
            return None
        body_end = _matching_closer_index(structural, j, "{", "}")
        if body_end is None:
            return None
        if _func_body_findings(structural, j, body_end):
            return None
        if _func_body_terminator_finding(structural, j, body_end):
            return None
        return _skip_loc_suffix(structural, body_end + 1)
    return _skip_loc_suffix(structural, line_end)


def _func_arg_list_is_plausible(text: str, *, allow_unnamed: bool) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    parts = _split_depth_zero_commas(stripped)
    if parts is None:
        return False
    for part in parts:
        colon = _depth_zero_colon_index(part, 0)
        if colon is None:
            if not allow_unnamed or "%" in part:
                return False
            if not _result_type_item_is_plausible(part):
                return False
            continue
        if not _ssa_result_list_is_plausible(part[:colon]):
            return False
        if not _result_type_item_is_plausible(part[colon + 1:]):
            return False
    return True


def _module_op_end(structural: str, start: int) -> int | None:
    j = _skip_spaces(structural, start)
    if j < len(structural) and structural[j] == "(":
        return None
    body_start = _module_body_start_after_metadata(structural, start)
    if body_start is None:
        return None
    body_end = _matching_closer_index(structural, body_start, "{", "}")
    if body_end is None:
        return None
    if _module_body_findings(structural, body_start, body_end):
        return None
    return _skip_loc_suffix(structural, body_end + 1)


def _alias_line_end(structural: str, start: int) -> int | None:
    boundary = _next_op_boundary(structural, start, len(structural))
    line = structural[start:boundary]
    eq_index = line.find("=")
    if eq_index <= 1:
        return None
    if not line[:eq_index].strip() or not line[eq_index + 1:].strip():
        return None
    return boundary - 1 if boundary < len(structural) else boundary


def _skip_loc_suffix(structural: str, start: int) -> int:
    j = _skip_spaces(structural, start)
    if not structural.startswith("loc", j):
        return start
    after = j + len("loc")
    after = _skip_spaces(structural, after)
    if after >= len(structural) or structural[after] != "(":
        return start
    loc_end = _matching_closer_index(structural, after, "(", ")")
    if loc_end is None:
        return start
    return loc_end + 1


def _generic_type_suffix_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != ":":
        return None
    i = _skip_spaces(text, start + 1)
    first_end = _matching_closer_index(text, i, "(", ")")
    if first_end is None:
        return None
    if not _parenthesized_type_list_is_plausible(text[i:first_end + 1]):
        return None
    i = _skip_spaces(text, first_end + 1)
    if not text.startswith("->", i):
        return None
    i = _skip_spaces(text, i + 2)
    if i >= len(text):
        return None
    if text[i] == "(":
        result_end = _matching_closer_index(text, i, "(", ")")
        if result_end is None:
            return None
        if not _parenthesized_type_list_is_plausible(text[i:result_end + 1]):
            return None
        return result_end + 1
    pairs = {"(": ")", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    end = i
    while end < len(text):
        char = text[end]
        if char == "-" and end + 1 < len(text) and text[end + 1] == ">":
            end += 2
            continue
        if char == "<" and end + 1 < len(text) and text[end + 1] == "=":
            end += 2
            continue
        if (char == ">" and
                ((end > i and text[end - 1] in "-=")
                 or (end + 1 < len(text) and text[end + 1] == "="))):
            end += 1
            continue
        if not stack and (char.isspace() or char in "}]),"):
            break
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers:
            if not stack or stack.pop() != char:
                return None
        end += 1
    if end <= i or stack:
        return None
    return end if _single_type_text_is_plausible(text[i:end]) else None


def _depth_zero_colon_index(
        text: str, start: int, end: int | None = None) -> int | None:
    if end is None:
        end = len(text)
    pairs = {"(": ")", "{": "}", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    i = start
    while i < end:
        char = text[i]
        if char == "-" and i + 1 < end and text[i + 1] == ">":
            i += 2
            continue
        if char == "<" and i + 1 < end and text[i + 1] == "=":
            i += 2
            continue
        if (char == ">" and
                ((i > 0 and text[i - 1] in "-=")
                 or (i + 1 < end and text[i + 1] == "="))):
            i += 1
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers:
            if stack and stack[-1] == char:
                stack.pop()
            elif not stack:
                return None
        elif char == ":" and not stack:
            return i
        i += 1
    return None


def _depth_zero_equals_index(
        text: str, start: int, end: int | None = None) -> int | None:
    if end is None:
        end = len(text)
    pairs = {"(": ")", "{": "}", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    i = start
    while i < end:
        char = text[i]
        if char == "-" and i + 1 < end and text[i + 1] == ">":
            i += 2
            continue
        if char == "<" and i + 1 < end and text[i + 1] == "=":
            i += 2
            continue
        if (char == ">" and
                ((i > 0 and text[i - 1] in "-=")
                 or (i + 1 < end and text[i + 1] == "="))):
            i += 1
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers:
            if stack and stack[-1] == char:
                stack.pop()
            elif not stack:
                return None
        elif char == "=" and not stack:
            before = text[i - 1] if i > 0 else ""
            after = text[i + 1] if i + 1 < end else ""
            if ((not before or before not in "<>!=")
                    and (not after or after not in "=>")):
                return i
        i += 1
    return None


def _generic_top_level_op_end(structural: str, start: int) -> int | None:
    start = _skip_spaces(structural, start)
    if start >= len(structural) or structural[start] != "(":
        return None
    operand_end = _matching_closer_index(structural, start, "(", ")")
    if operand_end is None:
        return None
    suffix_start = _depth_zero_colon_index(structural, operand_end + 1)
    if suffix_start is None:
        return None
    if _has_depth_zero_generic_junk(structural, operand_end + 1,
                                    suffix_start):
        return None
    if _generic_region_findings(structural, operand_end + 1, suffix_start):
        return None
    end = _generic_type_suffix_end(structural, suffix_start)
    if end is None:
        return None
    end = _skip_loc_suffix(structural, end)
    return end


def _has_depth_zero_generic_junk(text: str, start: int, end: int) -> bool:
    stack: list[str] = []
    pairs = {"(": ")", "{": "}", "[": "]", "<": ">"}
    closers = set(pairs.values())
    i = start
    while i < end:
        char = text[i]
        if char == "<" and i + 1 < end and text[i + 1] == "{":
            prop_end = _matching_closer_index(text, i + 1, "{", "}")
            if prop_end is None or prop_end >= end:
                return True
            after_prop = _skip_spaces(text, prop_end + 1)
            if after_prop >= end or text[after_prop] != ">":
                return True
            if _generic_property_dict_has_junk(text[i + 2:prop_end]):
                return True
            i = after_prop + 1
            continue
        if char == "-" and i + 1 < end and text[i + 1] == ">":
            i += 2
            continue
        if char == "<" and i + 1 < end and text[i + 1] == "=":
            i += 2
            continue
        if char.isspace() or char in ",:":
            i += 1
            continue
        if stack == [">"] and (char.isalnum() or char in "_$.-"):
            return True
        if char in pairs:
            stack.append(pairs[char])
            i += 1
            continue
        if char in closers:
            if (char == ">" and
                    ((i > start and text[i - 1] in "-=")
                     or (i + 1 < end and text[i + 1] == "="))):
                i += 1
                continue
            if not stack or stack.pop() != char:
                return True
            i += 1
            continue
        if not stack:
            return True
        i += 1
    return bool(stack)


def _generic_property_dict_has_junk(text: str) -> bool:
    parts = _split_depth_zero_commas(text)
    if parts is None:
        return True
    for part in parts:
        if not part:
            continue
        if not _property_assignment_is_plausible(part):
            return True
    return False


def _property_assignment_is_plausible(text: str) -> bool:
    eq_index = _depth_zero_equals_index(text, 0)
    if eq_index is None:
        return False
    if not text[:eq_index].strip():
        return False
    value_tail = text[eq_index + 1:].strip()
    if not value_tail:
        return False
    type_index = _depth_zero_colon_index(text, eq_index + 1)
    if type_index is None:
        return True
    if not text[eq_index + 1:type_index].strip():
        return False
    return _bare_dialect_type_tail_is_plausible(
        text[type_index + 1:].strip())


def _generic_region_findings(structural: str, start: int,
                             end: int) -> tuple[str, ...]:
    findings: list[str] = []
    i = start
    while i < end:
        if structural[i] != "{":
            i += 1
            continue
        prev = i - 1
        while prev >= start and structural[prev].isspace():
            prev -= 1
        if prev < start or structural[prev] not in "(,":
            i += 1
            continue
        body_end = _matching_closer_index(structural, i, "{", "}")
        if body_end is None or body_end > end:
            findings.append("malformed generic operation region")
            break
        if structural[i + 1:body_end].strip():
            body_findings = _module_body_findings(structural, i, body_end)
            if body_findings:
                findings.extend(body_findings)
                break
        i = body_end + 1
    return tuple(findings)


def _func_body_findings(structural: str, body_start: int,
                        body_end: int) -> tuple[str, ...]:
    findings: list[str] = []
    arg_finding = _function_arg_finding_before_body(structural, body_start)
    if arg_finding is not None:
        return (arg_finding,)
    function_ssa_types = _function_arg_types_before_body(
        structural, body_start)
    function_ssa_types.update(_region_arg_types_before_body(
        structural, body_start))
    ssa_types = dict(function_ssa_types)
    block_labels = _function_block_label_arities(structural, body_start,
                                                 body_end)
    call_signatures = _function_call_signature_table(structural)
    label_finding = _function_block_label_finding(
        structural, body_start, body_end)
    if label_finding is not None:
        return (label_finding,)
    tokens = ("builtin.module", "func.func", "module",
              _GENERIC_OP_SENTINEL)
    i = body_start + 1
    entry_ssa_types: dict[str, str | None] | None = None
    while True:
        i = _skip_spaces(structural, i)
        if i >= body_end:
            break
        line_end = min(_line_end(structural, i), body_end)
        if structural[i] == "^":
            if not _block_label_line_is_plausible(
                    structural[i:line_end]):
                findings.append(
                    "malformed block label in function body")
                break
            block_arg_finding = _block_label_arg_finding(
                structural[i:line_end])
            if block_arg_finding is not None:
                findings.append(block_arg_finding)
                break
            if entry_ssa_types is None:
                entry_ssa_types = dict(ssa_types)
            ssa_types = dict(entry_ssa_types)
            ssa_types.update(_block_label_arg_types(structural[i:line_end]))
            i = line_end
            continue
        if structural[i] in "{}":
            findings.append(
                "unexpected standalone region in function body")
            break
        result_names: tuple[str, ...] = ()
        if structural[i] == "%":
            eq_index = structural.find("=", i, line_end)
            if eq_index == -1:
                findings.append(
                    "unexpected SSA text in function body")
                break
            if not _ssa_result_list_is_plausible(structural[i:eq_index]):
                findings.append(
                    "malformed SSA result list in function body")
                break
            result_names = _ssa_result_names(structural[i:eq_index])
            duplicate = next(
                (name for name in result_names if name in ssa_types),
                None,
            )
            if duplicate is not None:
                findings.append(
                    f"duplicate SSA definition in function body: "
                    f"{duplicate}")
                break
            i = _skip_spaces(structural, eq_index + 1)
            if i >= line_end:
                findings.append(
                    "empty SSA assignment in function body")
                break
        matched = False
        for token in tokens:
            if not structural.startswith(token, i):
                continue
            before = structural[i - 1] if i > 0 else ""
            after_i = i + len(token)
            after = structural[after_i] if after_i < len(structural) else ""
            if _is_word_char(before) or before in ".%@^#!-":
                continue
            if after and not (after.isspace() or after in "{(@"):
                continue
            if token == _GENERIC_OP_SENTINEL:
                end = _generic_top_level_op_end(structural, after_i)
                if end is None or end > body_end:
                    findings.append(
                        "malformed nested generic operation in "
                        "function body")
                    return tuple(findings)
                ssa_finding = _op_undefined_ssa_finding(
                    structural[i:end], ssa_types, call_signatures)
                if ssa_finding is not None:
                    findings.append(ssa_finding)
                    break
                arity_finding = _op_result_arity_finding(
                    structural[i:end], result_names, ssa_types)
                if arity_finding is not None:
                    findings.append(arity_finding)
                    break
                if result_names:
                    result_type = _mlir_simple_result_type_from_op(
                        structural, i, end)
                    for name in result_names:
                        ssa_types[name] = result_type
                    result_names = ()
                i = end
                matched = True
                break
            findings.append(
                f"unsupported nested `{token}` operation in function body")
            return tuple(findings)
        if not matched:
            bare_end = _bare_dialect_op_end(
                structural, i, body_end, consume_body=False)
            bare_region_end: int | None = None
            bare_check_end = bare_end
            if bare_check_end is None:
                bare_region_end = _bare_dialect_op_end(
                    structural, i, body_end, consume_body=True)
                if bare_region_end is not None:
                    region_start = structural.find("{", i, bare_region_end)
                    if region_start != -1:
                        bare_check_end = region_start
            if bare_end is not None:
                op_name, _op_name_end = _read_bare_op_name(structural, i)
            elif bare_check_end is not None:
                op_name, _op_name_end = _read_bare_op_name(structural, i)
                bare_end = bare_region_end
            if bare_check_end is not None and bare_end is not None:
                branch_finding = _branch_target_finding(
                    structural[i:bare_check_end], block_labels)
                if branch_finding is not None:
                    findings.append(branch_finding)
                    break
                ssa_finding = _op_undefined_ssa_finding(
                    structural[i:bare_check_end], ssa_types, call_signatures)
                if ssa_finding is not None:
                    findings.append(ssa_finding)
                    break
                arity_finding = _op_result_arity_finding(
                    structural[i:bare_check_end], result_names, ssa_types)
                if arity_finding is not None:
                    findings.append(arity_finding)
                    break
                if result_names:
                    result_type = _mlir_simple_result_type_from_op(
                        structural, i, bare_check_end)
                    for name in result_names:
                        ssa_types[name] = result_type
                    result_names = ()
                i = bare_end
                continue
            word, _end = _read_bare_word(structural, i)
            if word == "return":
                if result_names:
                    findings.append("return cannot bind SSA results")
                    break
                tail = structural[_end:line_end].strip()
                if tail:
                    loc_end = _skip_loc_suffix(structural, _end)
                    if loc_end != _end and not structural[
                            loc_end:line_end].strip():
                        i = line_end
                        continue
                    type_index = _depth_zero_colon_index(
                        structural, _end)
                    if type_index is None or type_index >= line_end:
                        findings.append(
                            "malformed return in function body")
                        break
                    if not structural[_end:type_index].strip():
                        findings.append(
                            "malformed return in function body")
                        break
                    value_text = structural[_end:type_index].strip()
                    if not _return_values_are_ssa(value_text):
                        findings.append(
                            "malformed return in function body: non-SSA "
                            "return operands are not statically "
                            "translatable")
                        break
                    return_type = _mlir_simple_type_text(
                        structural, type_index + 1, line_end)
                    ssa_finding = _return_ssa_type_finding(
                        value_text, return_type, ssa_types)
                    if ssa_finding is not None:
                        findings.append(ssa_finding)
                        break
                    if not _bare_dialect_type_tail_with_optional_loc_is_plausible(
                            structural, type_index + 1, line_end):
                        findings.append(
                            "malformed return in function body")
                        break
                i = line_end
                continue
            findings.append(
            "unexpected text in function body")
            break
    return tuple(findings)


def _function_arg_types_before_body(
        structural: str, body_start: int) -> dict[str, str | None]:
    func_start = structural.rfind("func.func", 0, body_start)
    if func_start == -1:
        return {}
    open_index = structural.find("(", func_start, body_start)
    if open_index == -1:
        return {}
    close_index = _matching_closer_index(structural, open_index, "(", ")")
    if close_index is None or close_index > body_start:
        return {}
    args = _split_depth_zero_commas(
        structural[open_index + 1:close_index].strip())
    if not args:
        return {}
    types: dict[str, str | None] = {}
    for arg in args:
        arg = arg.strip()
        if not arg.startswith("%"):
            continue
        colon = _depth_zero_colon_index(arg, 0)
        if colon is None:
            continue
        name = arg[:colon].strip()
        type_text = _mlir_function_signature_item_type(arg[colon + 1:])
        if name:
            types[name] = type_text or None
    return types


def _region_arg_types_before_body(
        structural: str, body_start: int) -> dict[str, str | None]:
    types: dict[str, str | None] = {}
    scf_for_header = _region_header_before_body(
        structural, body_start, "scf.for")
    if scf_for_header:
        types.update(_scf_for_region_arg_types(scf_for_header))
    scf_while_header = _region_header_before_body(
        structural, body_start, "scf.while")
    if scf_while_header:
        types.update(_region_assignment_arg_types(scf_while_header))
    return types


def _region_header_before_body(
        structural: str, body_start: int, op_name: str) -> str:
    op_index = _last_bare_op_index_before(structural, body_start, op_name)
    if op_index == -1:
        return ""
    boundary = max(
        structural.rfind("{", 0, body_start),
        structural.rfind("}", 0, body_start),
    )
    if boundary > op_index:
        return ""
    return structural[op_index:body_start]


def _last_bare_op_index_before(
        structural: str, end: int, op_name: str) -> int:
    op_index = structural.rfind(op_name, 0, end)
    while op_index != -1:
        name, _name_end = _read_bare_op_name(structural, op_index)
        if name == op_name:
            return op_index
        op_index = structural.rfind(op_name, 0, op_index)
    return -1


def _scf_for_region_arg_types(header: str) -> dict[str, str | None]:
    types: dict[str, str | None] = {}
    _name, name_end = _read_bare_op_name(header, 0)
    eq_index = _depth_zero_equals_index(header, name_end)
    if eq_index is not None:
        lhs = header[name_end:eq_index].strip()
        for name in _ssa_result_names(lhs):
            if name.startswith("%"):
                types[name] = None
    iter_arg_text = _scf_iter_args_text(header)
    if iter_arg_text:
        types.update(_region_assignment_arg_types(iter_arg_text))
    return types


def _region_assignment_arg_types(header: str) -> dict[str, str | None]:
    open_index = header.find("(")
    if open_index == -1:
        return {}
    close_index = _matching_closer_index(header, open_index, "(", ")")
    if close_index is None:
        return {}
    args = _split_depth_zero_commas(header[open_index + 1:close_index])
    if args is None:
        return {}
    types: dict[str, str | None] = {}
    for arg in args:
        eq_index = _depth_zero_equals_index(arg, 0)
        if eq_index is None:
            continue
        lhs = arg[:eq_index if eq_index is not None else len(arg)].strip()
        for name in _ssa_result_names(lhs):
            if name.startswith("%"):
                types[name] = None
    return types


def _function_arg_finding_before_body(
        structural: str, body_start: int) -> str | None:
    func_start = structural.rfind("func.func", 0, body_start)
    if func_start == -1:
        return None
    open_index = structural.find("(", func_start, body_start)
    if open_index == -1:
        return None
    close_index = _matching_closer_index(structural, open_index, "(", ")")
    if close_index is None or close_index > body_start:
        return None
    args = _split_depth_zero_commas(
        structural[open_index + 1:close_index].strip())
    if not args:
        return None
    seen: set[str] = set()
    for arg in args:
        colon = _depth_zero_colon_index(arg, 0)
        if colon is None or not arg.strip().startswith("%"):
            continue
        for name in _ssa_result_names(arg[:colon]):
            if name in seen:
                return f"duplicate function argument: {name}"
            seen.add(name)
    return None


def _function_block_label_arities(
        structural: str, body_start: int,
        body_end: int) -> dict[str, int]:
    labels: dict[str, int] = {}
    i = body_start + 1
    while i < body_end:
        i = _skip_spaces(structural, i)
        if i >= body_end:
            break
        line_end = min(_line_end(structural, i), body_end)
        if structural[i] == "^" and _block_label_line_is_plausible(
                structural[i:line_end]):
            label = _block_label_name_and_arity(structural[i:line_end])
            if label is not None:
                labels[label[0]] = label[1]
        i = line_end + 1
    return labels


def _function_block_label_finding(
        structural: str, body_start: int,
        body_end: int) -> str | None:
    seen: set[str] = set()
    i = body_start + 1
    while i < body_end:
        i = _skip_spaces(structural, i)
        if i >= body_end:
            break
        line_end = min(_line_end(structural, i), body_end)
        if structural[i] == "^" and _block_label_line_is_plausible(
                structural[i:line_end]):
            label = _block_label_name_and_arity(structural[i:line_end])
            if label is not None:
                if label[0] in seen:
                    return f"duplicate block label: {label[0]}"
                seen.add(label[0])
        i = line_end + 1
    return None


def _block_label_name_and_arity(text: str) -> tuple[str, int] | None:
    stripped = text.strip()
    if not stripped.startswith("^"):
        return None
    end = 1
    while end < len(stripped) and (
            stripped[end].isalnum() or stripped[end] in "_$.-"):
        end += 1
    if end <= 1:
        return None
    label = stripped[:end]
    i = _skip_spaces(stripped, end)
    if i >= len(stripped) or stripped[i] != "(":
        return label, 0
    close = _matching_closer_index(stripped, i, "(", ")")
    if close is None:
        return None
    args = _split_depth_zero_commas(stripped[i + 1:close].strip())
    return label, len(args or ())


def _block_label_arg_finding(text: str) -> str | None:
    stripped = text.strip()
    open_index = stripped.find("(")
    if open_index == -1:
        return None
    close_index = _matching_closer_index(stripped, open_index, "(", ")")
    if close_index is None:
        return None
    args = _split_depth_zero_commas(
        stripped[open_index + 1:close_index].strip())
    if not args:
        return None
    seen: set[str] = set()
    for arg in args:
        colon = _depth_zero_colon_index(arg, 0)
        if colon is None:
            continue
        for name in _ssa_result_names(arg[:colon]):
            if name in seen:
                return f"duplicate block argument: {name}"
            seen.add(name)
    return None


def _block_label_arg_types(text: str) -> dict[str, str | None]:
    stripped = text.strip()
    open_index = stripped.find("(")
    if open_index == -1:
        return {}
    close_index = _matching_closer_index(stripped, open_index, "(", ")")
    if close_index is None:
        return {}
    args = _split_depth_zero_commas(
        stripped[open_index + 1:close_index].strip())
    if not args:
        return {}
    types: dict[str, str | None] = {}
    for arg in args:
        colon = _depth_zero_colon_index(arg, 0)
        if colon is None:
            continue
        names = _ssa_result_names(arg[:colon])
        type_text = _normalized_mlir_fragment(arg[colon + 1:])
        for name in names:
            types[name] = type_text or None
    return types


def _branch_target_finding(
        op_text: str, block_labels: dict[str, int]) -> str | None:
    op_name, _op_end = _read_bare_op_name(op_text, 0)
    if op_name not in _BRANCH_WITH_BLOCK_TARGET_OPS:
        return None
    for label, argc in _branch_target_refs(op_text):
        if label not in block_labels:
            return f"undefined block target: {label}"
        expected = block_labels[label]
        if expected != argc:
            return (
                f"block target argument mismatch for {label}: "
                f"expected {expected}, got {argc}")
    return None


def _branch_target_refs(op_text: str) -> tuple[tuple[str, int], ...]:
    refs: list[tuple[str, int]] = []
    i = 0
    while i < len(op_text):
        if op_text[i] != "^":
            i += 1
            continue
        end = i + 1
        while end < len(op_text) and (
                op_text[end].isalnum() or op_text[end] in "_$.-"):
            end += 1
        if end <= i + 1:
            i += 1
            continue
        label = op_text[i:end]
        j = _skip_spaces(op_text, end)
        argc = 0
        if j < len(op_text) and op_text[j] == "(":
            close = _matching_closer_index(op_text, j, "(", ")")
            if close is not None:
                args = _split_depth_zero_commas(
                    op_text[j + 1:close].strip())
                argc = len(args or ())
                end = close + 1
        refs.append((label, argc))
        i = end
    return tuple(refs)


def _ssa_result_names(text: str) -> tuple[str, ...]:
    names: list[str] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        count_index = _depth_zero_colon_index(item, 0)
        if count_index is not None:
            base = item[:count_index].strip()
            count_text = item[count_index + 1:].strip()
            if base.startswith("%") and count_text.isdigit():
                count = int(count_text)
                names.extend(f"{base}#{index}" for index in range(count))
                continue
            item = base
        names.append(item)
    return tuple(names)


def _mlir_simple_result_type_from_op(
        structural: str, op_start: int, op_end: int) -> str | None:
    op_name, _name_end = _read_bare_op_name(structural, op_start)
    type_index = _depth_zero_colon_index(structural, op_start, op_end)
    if type_index is None:
        return None
    tail = _mlir_type_tail_without_loc(structural, type_index + 1, op_end)
    if op_name in ("arith.cmpf", "arith.cmpi"):
        return _mlir_bool_result_type_for_compare(tail)
    arrow = tail.rfind("->")
    if arrow != -1:
        tail = tail[arrow + 2:].strip()
        if tail.startswith("(") and tail.endswith(")"):
            inner = tail[1:-1].strip()
            parts = _split_depth_zero_commas(inner)
            if parts is None or len(parts) != 1:
                return None
            tail = parts[0].strip()
    elif op_name in (
            "arith.index_cast", "vector.multi_reduction",
            "vector.shape_cast"):
        result = _mlir_result_type_after_depth_zero_to(tail)
        if result is not None:
            return result
        return None
    elif op_name in ("memref.load", "vector.transfer_read"):
        parts = _split_depth_zero_commas(tail)
        if parts is not None and len(parts) > 1:
            return _normalized_mlir_fragment(parts[-1]) or None
        return None
    return _normalized_mlir_fragment(tail) if tail and tail != "()" else None


def _mlir_bool_result_type_for_compare(operand_type: str) -> str | None:
    stripped = _normalized_mlir_fragment(operand_type)
    if not stripped:
        return None
    if not stripped.startswith(("vector<", "tensor<")):
        return "i1"
    open_index = stripped.find("<")
    close_index = _matching_closer_index(stripped, open_index, "<", ">")
    if close_index != len(stripped) - 1:
        return None
    inner = stripped[open_index + 1:close_index]
    layout_parts = _split_depth_zero_commas(inner)
    if layout_parts is None or not layout_parts:
        return None
    shape_parts = list(_split_depth_zero_x(layout_parts[0]))
    if len(shape_parts) == 1:
        shape_parts = ["i1"]
    elif stripped.startswith("vector<") and not all(
            _mlir_vector_dim_is_plausible(part)
            for part in shape_parts[:-1]):
        return None
    else:
        shape_parts[-1] = "i1"
    element_shape = "x".join(part.strip() for part in shape_parts)
    if not element_shape:
        return None
    rebuilt = [element_shape]
    rebuilt.extend(part.strip() for part in layout_parts[1:])
    return f"{stripped[:open_index]}<{', '.join(rebuilt)}>"


def _mlir_result_type_after_depth_zero_to(tail: str) -> str | None:
    pairs = {"(": ")", "{": "}", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    i = 0
    while i < len(tail):
        char = tail[i]
        if char == "-" and i + 1 < len(tail) and tail[i + 1] == ">":
            i += 2
            continue
        if char in pairs:
            stack.append(pairs[char])
            i += 1
            continue
        if char in closers:
            if stack and stack[-1] == char:
                stack.pop()
            i += 1
            continue
        if not stack and _bare_word_at(tail, i, "to"):
            result = tail[i + 2:].strip()
            return _normalized_mlir_fragment(result) or None
        i += 1
    return None


def _mlir_simple_type_text(
        structural: str, start: int, end: int) -> str | None:
    tail = _mlir_type_tail_without_loc(structural, start, end)
    return _normalized_mlir_fragment(tail) if tail else None


def _mlir_type_tail_without_loc(structural: str, start: int, end: int) -> str:
    loc_index = _depth_zero_keyword_index(structural, start, end, "loc")
    type_end = loc_index if loc_index is not None else end
    return structural[start:type_end].strip()


def _op_undefined_ssa_finding(
        op_text: str, ssa_types: dict[str, str | None],
        call_signatures: dict[str, tuple[str, str]] | None = None,
        ) -> str | None:
    op_name, _op_name_end = _read_bare_op_name(op_text, 0)
    if op_name not in _SSA_OPERAND_VALIDATED_OPS:
        return _op_static_type_finding(op_text, op_name, ssa_types)
    non_ssa_finding = _op_non_ssa_operand_finding(op_text, op_name)
    if non_ssa_finding is not None:
        return non_ssa_finding
    for value in _op_ssa_operands_to_validate(op_text, op_name):
        if value not in ssa_types:
            return f"undefined SSA value in operation: {value}"
    type_finding = _op_static_type_finding(op_text, op_name, ssa_types)
    if type_finding is not None:
        return type_finding
    if op_name in _TYPED_SSA_OPERAND_OPS:
        typed_finding = _typed_ssa_operand_type_finding(op_text, ssa_types)
        if typed_finding is not None:
            return typed_finding
    if op_name == "func.call" and call_signatures is not None:
        call_finding = _func_call_signature_finding(
            op_text, ssa_types, call_signatures)
        if call_finding is not None:
            return call_finding
    return None


def _op_result_arity_finding(
        op_text: str, result_names: tuple[str, ...],
        ssa_types: dict[str, str | None]) -> str | None:
    op_name, _op_end = _read_bare_op_name(op_text, 0)
    if not op_name:
        return None
    expected_arity = _op_static_result_arity(op_text, op_name, ssa_types)
    if expected_arity is None:
        return None
    if len(result_names) != expected_arity:
        return (
            f"{op_name} result arity mismatch: {len(result_names)} "
            f"SSA result(s) for {expected_arity} produced value(s)")
    return None


def _op_static_result_arity(
        op_text: str, op_name: str,
        ssa_types: dict[str, str | None]) -> int | None:
    if op_name in _VOID_RESULT_OPS:
        return 0
    if op_name == "func.call":
        parts = _func_call_signature_parts(op_text)
        if parts is None:
            return None
        _symbol, _arg_values, _call_args, call_result = parts
        return len(_mlir_return_type_items(call_result))
    if op_name in _KNOWN_SINGLE_RESULT_OPS:
        return 1
    if op_name == "memref.load":
        return 1
    return None


_VOID_RESULT_OPS = frozenset((
    "cf.assert", "cf.br", "cf.cond_br", "func.return", "memref.store",
    "scf.condition", "scf.yield",
))

_KNOWN_SINGLE_RESULT_OPS = frozenset((
    "arith.addf", "arith.addi", "arith.andi", "arith.constant",
    "arith.divf", "arith.divsi", "arith.divui", "arith.cmpf",
    "arith.cmpi", "arith.index_cast", "arith.maxf", "arith.maximumf",
    "arith.maxsi", "arith.maxui", "arith.minf", "arith.minimumf",
    "arith.minsi", "arith.minui", "arith.mulf", "arith.muli",
    "arith.ori", "arith.remf", "arith.remsi", "arith.remui",
    "arith.shli", "arith.shrsi", "arith.shrui", "arith.subf",
    "arith.subi", "arith.xori", "memref.load", "vector.multi_reduction",
    "vector.shape_cast", "vector.transfer_read",
))


_SSA_OPERAND_VALIDATED_OPS = frozenset((
    "arith.addf", "arith.addi", "arith.andi", "arith.divf",
    "arith.cmpf", "arith.cmpi", "arith.divsi", "arith.divui",
    "arith.index_cast", "arith.maxf", "arith.maximumf",
    "arith.maxsi", "arith.maxui", "arith.minf", "arith.minimumf",
    "arith.minsi", "arith.minui", "arith.mulf", "arith.muli",
    "arith.ori", "arith.remf", "arith.remsi", "arith.remui",
    "arith.shli", "arith.shrsi", "arith.shrui", "arith.subf",
    "arith.subi", "arith.xori", "cf.br", "cf.cond_br", "func.call",
    "cf.assert", "func.return", "gpu.launch", "memref.load",
    "memref.store", "scf.condition", "scf.for", "scf.if",
    "scf.while", "scf.yield", "vector.multi_reduction",
    "vector.transfer_read",
))


_REGION_HEADER_SSA_VALIDATED_OPS = frozenset((
    "gpu.launch", "scf.if",
))


_TYPED_SSA_OPERAND_OPS = frozenset((
    "func.return", "scf.yield",
))
_FUNC_BLOCK_TERMINATOR_OPS = frozenset((
    "cf.br", "cf.cond_br", "func.return", "return",
))


_ARITH_SSA_OPERAND_OPS = frozenset((
    "arith.addf", "arith.addi", "arith.andi", "arith.divf",
    "arith.cmpf", "arith.cmpi", "arith.divsi", "arith.divui",
    "arith.index_cast", "arith.maxf", "arith.maximumf",
    "arith.maxsi", "arith.maxui", "arith.minf", "arith.minimumf",
    "arith.minsi", "arith.minui", "arith.mulf", "arith.muli",
    "arith.ori", "arith.remf", "arith.remsi", "arith.remui",
    "arith.shli", "arith.shrsi", "arith.shrui", "arith.subf",
    "arith.subi", "arith.xori",
))


_ARITH_COMPARE_OPS = frozenset(("arith.cmpf", "arith.cmpi"))


# Control ops whose first SSA operand is an `i1` predicate. `mlir-opt`'s
# real verifier rejects a non-`i1` predicate, but a smoke-aware echo
# tool would not — the static preflight has to reject it here, before
# tool dispatch, or a fake `mlir-opt` could mint `MLIRValidation.PASSED`
# for a malformed program. The check looks up the SSA operand's type
# in the function-local `ssa_types` map and rejects anything but `i1`.
_CONTROL_PREDICATE_OPS = frozenset(("scf.if", "cf.cond_br", "cf.assert"))


_ARITH_FLOAT_TYPED_OPS = frozenset((
    "arith.addf", "arith.cmpf", "arith.divf", "arith.maxf",
    "arith.maximumf", "arith.minf", "arith.minimumf", "arith.mulf",
    "arith.remf", "arith.subf",
))
_ARITH_INTEGER_TYPED_OPS = frozenset((
    "arith.addi", "arith.andi", "arith.cmpi", "arith.divsi",
    "arith.divui", "arith.maxsi", "arith.maxui", "arith.minsi",
    "arith.minui", "arith.muli", "arith.ori", "arith.remsi",
    "arith.remui", "arith.shli", "arith.shrsi", "arith.shrui",
    "arith.subi", "arith.xori",
))
_MLIR_FLOAT_SCALAR_TYPES = frozenset((
    "bf16", "f16", "f32", "f64", "f80", "f128", "tf32",
))


def _control_predicate_type_finding(
        op_text: str, op_name: str,
        ssa_types: dict[str, str | None]) -> str | None:
    """Verify the first SSA operand of an `scf.if` / `cf.cond_br` /
    `cf.assert` is `i1`. A non-`i1` predicate (e.g. `%c: i32`) is
    invalid MLIR a smoke-aware echo `mlir-opt` would accept silently —
    the static preflight rejects it here. Returns `None` when the
    predicate type cannot be resolved (the SSA-definedness pass catches
    the unresolved case separately)."""
    ssa_values = _op_ssa_operands_to_validate(op_text, op_name)
    if not ssa_values:
        return None
    predicate = ssa_values[0]
    actual_type = ssa_types.get(predicate)
    if actual_type is None:
        return None
    if _normalized_mlir_fragment(actual_type) != "i1":
        return (
            f"{op_name} requires an i1 predicate, got {predicate} of "
            f"type {actual_type}")
    return None


def _memref_rank_from_type(memref_type: str) -> int | None:
    """Return the rank of a `memref<...>` type, or None if unparseable.

    `memref<10x20xi32>` -> 2; `memref<?xf32>` -> 1; `memref<f32>` -> 0.
    A trailing memory-space (`, 3>`) is stripped. The element-type
    tail (`xi32`) is dropped — for the common scalar-element case the
    "split on x then drop the last" rule is correct; a vector/tensor
    element type may itself contain `x`, in which case parsing returns
    `None` and the caller defers to other checks."""
    stripped = memref_type.strip()
    if not stripped.startswith("memref<") or not stripped.endswith(">"):
        return None
    inner = stripped[len("memref<"):-1]
    parts = _split_depth_zero_commas(inner)
    shape_and_type = (parts[0] if parts else inner).strip()
    # If the element type would be ambiguous (any "<" in the tail
    # suggests a nested type whose x's would confuse our split), bail.
    if "<" in shape_and_type:
        return None
    tokens = shape_and_type.split("x")
    if not tokens:
        return None
    # tokens[-1] is the element type; everything before is shape.
    return max(0, len(tokens) - 1)


def _memref_access_type_finding(
        op_text: str, op_name: str,
        ssa_types: dict[str, str | None]) -> str | None:
    """Verify `memref.load` / `memref.store` index arity and index
    operand types. A real `mlir-opt` rejects an arity mismatch or a
    non-`index` index operand; a smoke-aware echo tool would not, so
    the static preflight rejects them here.

    Checks:
    - the number of bracketed index operands matches the memref rank;
    - each `%idx` operand has type `index` (when resolvable in
      `ssa_types`).
    The result-element / stored-value type checks remain a sibling
    follow-up; this catches the dominant rank / index-type holes."""
    type_text = _op_declared_type_text(op_text)
    if type_text is None or not type_text.startswith("memref<"):
        return None
    rank = _memref_rank_from_type(type_text)
    if rank is None:
        return None
    bracket_start = op_text.find("[")
    if bracket_start == -1:
        return None
    bracket_end = _matching_closer_index(
        op_text, bracket_start, "[", "]")
    if bracket_end is None:
        # An unbalanced `[...]` would otherwise silently slice to
        # end-of-string via `op_text[...:None]` and produce a bogus
        # arity reading — fail closed instead. `_matching_closer_index`
        # returns `None` on no-match (NOT -1, unlike `str.find`).
        return (
            f"malformed {op_name}: unbalanced index bracket — the "
            "translator fails closed")
    indices_text = op_text[bracket_start + 1:bracket_end].strip()
    if not indices_text:
        index_operands: list[str] = []
    else:
        parts = _split_depth_zero_commas(indices_text)
        if parts is None:
            return None
        index_operands = [p.strip() for p in parts if p.strip()]
    if len(index_operands) != rank:
        return (
            f"{op_name} index arity mismatch: {len(index_operands)} "
            f"indices for {type_text} (rank {rank})")
    for idx in index_operands:
        if not idx.startswith("%"):
            continue
        actual_type = ssa_types.get(idx)
        if actual_type is None:
            continue
        if _normalized_mlir_fragment(actual_type) != "index":
            return (
                f"{op_name} index operand {idx} has type "
                f"{actual_type}, expected index")
    return None


_MEMREF_ACCESS_OPS = frozenset(("memref.load", "memref.store"))


def _arith_constant_value_type_finding(
        op_text: str, type_text: str) -> str | None:
    """Verify the literal value of `arith.constant` matches its
    declared type. `true` / `false` require `i1`; an integer literal
    requires an integer / `index` type; a float literal (a token with
    `.` or scientific-notation `e`) requires a floating-point type. A
    `dense<...>` splat / vector constant is deferred."""
    _name, name_end = _read_bare_op_name(op_text, 0)
    type_index = _depth_zero_colon_index(op_text, name_end)
    if type_index is None:
        return None
    value_text = op_text[name_end:type_index].strip()
    if not value_text or value_text.startswith("dense<"):
        return None
    is_bool = value_text in ("true", "false")
    if is_bool:
        if _normalized_mlir_fragment(type_text) != "i1":
            return (
                f"arith.constant boolean literal {value_text} requires "
                f"i1, got {type_text}")
        return None
    # Strip an optional leading sign for the int/float discrimination.
    body = value_text[1:] if value_text[:1] in "+-" else value_text
    # Hex / octal / binary prefixes are integer literals BUT may also be
    # the bit-pattern form of a float (e.g. `0x7FC00000 : f32` for NaN).
    # The static preflight cannot disambiguate without parsing the type,
    # so defer rather than misclassify — let `mlir-opt` (or a future
    # tightening) decide.
    if body[:2].lower() in ("0x", "0o", "0b"):
        return None
    # Underscored decimal integer literals (`1_000`) are integer-shaped.
    bare = body.replace("_", "")
    is_int_literal = bare.isdigit() and bool(bare)
    is_float_literal = (
        not is_int_literal
        and any(ch in value_text for ch in ".eE")
        and any(ch.isdigit() for ch in value_text))
    if is_int_literal and _mlir_type_is_float_like(type_text):
        return (
            f"arith.constant integer literal {value_text} requires an "
            f"integer / index type, got {type_text}")
    if is_float_literal and _mlir_type_is_integer_like(type_text):
        return (
            f"arith.constant float literal {value_text} requires a "
            f"floating-point type, got {type_text}")
    return None


def _scf_for_bounds_type_finding(
        op_text: str, ssa_types: dict[str, str | None]) -> str | None:
    """Verify `scf.for` lower / upper / step bounds are all `index`.
    MLIR's `scf.for` requires this; a smoke-aware echo tool would
    not catch a non-`index` bound."""
    operand_text = _scf_for_operand_text(op_text)
    ssa_values = _mlir_raw_ssa_values(operand_text)
    for value in ssa_values[:3]:
        actual_type = ssa_types.get(value)
        if actual_type is None:
            continue
        if _normalized_mlir_fragment(actual_type) != "index":
            return (
                f"scf.for bound {value} has type {actual_type}, "
                f"expected index")
    return None


_VECTOR_MULTI_REDUCTION_KINDS = frozenset((
    "add", "mul", "and", "or", "xor",
    "maxf", "maximumf", "minf", "minimumf",
    "maxnumf", "minnumf",
    "maxsi", "maxui", "minsi", "minui",
))


def _vector_multi_reduction_kind_finding(op_text: str) -> str | None:
    """Verify `vector.multi_reduction <kind>, ...` uses a known reduction
    kind. A smoke-aware echo tool would accept `<bogus>` because the
    kind appears only as a bare identifier inside angle brackets. The
    angle-bracketed kind is REQUIRED — its absence is a hard fail."""
    _name, name_end = _read_bare_op_name(op_text, 0)
    cursor = _skip_spaces(op_text, name_end)
    if cursor >= len(op_text) or op_text[cursor] != "<":
        return "vector.multi_reduction is missing the required <kind>"
    close = _matching_closer_index(op_text, cursor, "<", ">")
    if close is None:
        return "vector.multi_reduction has malformed kind delimiter"
    kind = op_text[cursor + 1:close].strip()
    if not kind:
        return "vector.multi_reduction has empty reduction kind"
    if kind in _VECTOR_MULTI_REDUCTION_KINDS:
        return None
    return (
        f"vector.multi_reduction unsupported reduction kind: <{kind}>")


def _vector_type_parts(type_text: str) -> tuple[tuple[int, ...], str] | None:
    """Decompose a `vector<...>` type into `(dims, element_type)`. The
    parser is conservative: it returns None for dynamic / scalable /
    parametric element types (`vector<?xi32>`, `vector<[4]xi32>`,
    `vector<4x!quant.uniform<...>>`) so the caller defers rather than
    misclassifies."""
    stripped = type_text.strip()
    if not (stripped.startswith("vector<") and stripped.endswith(">")):
        return None
    inner = stripped[len("vector<"):-1]
    if "<" in inner:
        return None
    tokens = inner.split("x")
    if len(tokens) < 2:
        return None
    dims: list[int] = []
    for token in tokens[:-1]:
        dim = token.strip()
        if not dim.isdigit():
            return None
        dims.append(int(dim))
    element = tokens[-1].strip()
    if not element:
        return None
    return tuple(dims), element


def _vector_type_element_count(type_text: str) -> int | None:
    """Compute the static element count of a `vector<...>` type, e.g.
    `vector<4x3xi32>` -> 12. Returns None when the type is not fully
    resolvable (delegating to `_vector_type_parts`)."""
    parts = _vector_type_parts(type_text)
    if parts is None:
        return None
    dims, _element = parts
    count = 1
    for dim in dims:
        count *= dim
    return count


def _depth_zero_substring_index(
        text: str, needle: str, start: int = 0) -> int:
    """Find `needle` at bracket-depth zero in `text`. Returns -1 if not
    found. Tracks `()`, `{}`, `[]`, `<>` (with the `->` / `<=` / `=>`
    escape hatches `_depth_zero_colon_index` uses)."""
    pairs = {"(": ")", "{": "}", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    end = len(text)
    i = start
    n = len(needle)
    while i < end:
        char = text[i]
        if char == "-" and i + 1 < end and text[i + 1] == ">":
            i += 2
            continue
        if char == "<" and i + 1 < end and text[i + 1] == "=":
            i += 2
            continue
        if (char == ">" and
                ((i > 0 and text[i - 1] in "-=")
                 or (i + 1 < end and text[i + 1] == "="))):
            i += 1
            continue
        if not stack and text.startswith(needle, i):
            return i
        if char in pairs:
            stack.append(pairs[char])
            i += 1
            continue
        if char in closers and stack and stack[-1] == char:
            stack.pop()
            i += 1
            continue
        i += 1
    return -1


def _strip_trailing_loc_or_attrs(text: str) -> str:
    """Trim a trailing ` loc(...)` or ` {attr = ...}` suffix from a type
    fragment. Returns the input unchanged if neither pattern applies."""
    stripped = text.rstrip()
    if stripped.endswith(")"):
        loc_start = stripped.rfind("loc(")
        if loc_start != -1:
            opener = stripped.rfind("(", 0, loc_start + len("loc("))
            if opener == loc_start + len("loc(") - 1:
                pre = stripped[:loc_start].rstrip()
                return pre
    if stripped.endswith("}"):
        opener = stripped.rfind("{")
        if opener != -1:
            return stripped[:opener].rstrip()
    return text


def _vector_shape_cast_finding(op_text: str) -> str | None:
    """Verify `vector.shape_cast %src : vector<A> to vector<B>` has
    matching total element counts AND matching element types. A
    smoke-aware echo tool would accept `vector<4xi32> to vector<3xi32>`
    (count drift) or `vector<4xi32> to vector<4xf32>` (element-type
    drift) because both sides are well-formed types in isolation."""
    type_index = _depth_zero_colon_index(op_text, 0)
    if type_index is None:
        return None
    tail = op_text[type_index + 1:]
    to_index = _depth_zero_substring_index(tail, " to ")
    if to_index == -1:
        return None
    src_raw = tail[:to_index]
    dst_raw = _strip_trailing_loc_or_attrs(tail[to_index + len(" to "):])
    src_type = _normalized_mlir_fragment(src_raw)
    dst_type = _normalized_mlir_fragment(dst_raw)
    if not src_type or not dst_type:
        return None
    src_parts = _vector_type_parts(src_type)
    dst_parts = _vector_type_parts(dst_type)
    if src_parts is None or dst_parts is None:
        return None
    src_dims, src_element = src_parts
    dst_dims, dst_element = dst_parts
    if src_element != dst_element:
        return (
            f"vector.shape_cast element-type mismatch: "
            f"{src_type} has element {src_element}, "
            f"{dst_type} has element {dst_element}")
    src_count = 1
    for dim in src_dims:
        src_count *= dim
    dst_count = 1
    for dim in dst_dims:
        dst_count *= dim
    if src_count != dst_count:
        return (
            f"vector.shape_cast element-count mismatch: "
            f"{src_type} has {src_count}, {dst_type} has {dst_count}")
    return None


def _vector_transfer_read_index_finding(
        op_text: str, ssa_types: dict[str, str | None]) -> str | None:
    """Verify `vector.transfer_read %src[%i, %j], %pad : ...` index
    operands are all `index`-typed when their types are resolvable. The
    real `mlir-opt` rejects non-`index` indices; a smoke-aware echo
    would not.

    The bracket is located by walking past the op name + the first SSA
    source operand, so we don't accidentally pick up a `[...]` token
    inside the type tail (e.g. `strided<[1], offset: 0>`, `[in_bounds]`
    attribute lists)."""
    _name, name_end = _read_bare_op_name(op_text, 0)
    cursor = _skip_spaces(op_text, name_end)
    if cursor >= len(op_text) or op_text[cursor] != "%":
        return None
    while cursor < len(op_text) and not op_text[cursor].isspace() \
            and op_text[cursor] != "[":
        cursor += 1
    cursor = _skip_spaces(op_text, cursor)
    if cursor >= len(op_text) or op_text[cursor] != "[":
        return None
    bracket_start = cursor
    bracket_end = _matching_closer_index(op_text, bracket_start, "[", "]")
    if bracket_end is None:
        return (
            "malformed vector.transfer_read: unbalanced index bracket "
            "— the translator fails closed")
    indices_text = op_text[bracket_start + 1:bracket_end].strip()
    if not indices_text:
        return None
    parts = _split_depth_zero_commas(indices_text)
    if parts is None:
        return None
    for raw in parts:
        idx = raw.strip()
        if not idx.startswith("%"):
            continue
        actual_type = ssa_types.get(idx)
        if actual_type is None:
            continue
        if _normalized_mlir_fragment(actual_type) != "index":
            return (
                f"vector.transfer_read index operand {idx} has type "
                f"{actual_type}, expected index")
    return None


def _op_static_type_finding(
        op_text: str, op_name: str,
        ssa_types: dict[str, str | None]) -> str | None:
    if op_name in _CONTROL_PREDICATE_OPS:
        return _control_predicate_type_finding(op_text, op_name, ssa_types)
    if op_name in _MEMREF_ACCESS_OPS:
        return _memref_access_type_finding(op_text, op_name, ssa_types)
    if op_name == "scf.for":
        return _scf_for_bounds_type_finding(op_text, ssa_types)
    if op_name == "vector.multi_reduction":
        return _vector_multi_reduction_kind_finding(op_text)
    if op_name == "vector.shape_cast":
        return _vector_shape_cast_finding(op_text)
    if op_name == "vector.transfer_read":
        return _vector_transfer_read_index_finding(op_text, ssa_types)
    if op_name == "arith.constant":
        type_text = _op_declared_type_text(op_text)
        if type_text is None:
            return None
        if not _mlir_type_is_known_arith_value_type(type_text):
            return (
                "unsupported arith.constant result type for static "
                f"preflight: {type_text}")
        value_finding = _arith_constant_value_type_finding(
            op_text, type_text)
        if value_finding is not None:
            return value_finding
        return None
    if op_name in _ARITH_FLOAT_TYPED_OPS:
        type_text = _op_declared_type_text(op_text)
        if type_text is None:
            return None
        if not _mlir_type_is_float_like(type_text):
            return f"{op_name} requires a floating-point type, got {type_text}"
        return _arith_operand_type_mismatch_finding(
            op_text, op_name, type_text, ssa_types)
    if op_name in _ARITH_INTEGER_TYPED_OPS:
        type_text = _op_declared_type_text(op_text)
        if type_text is None:
            return None
        if not _mlir_type_is_integer_like(type_text):
            return f"{op_name} requires an integer/index type, got {type_text}"
        return _arith_operand_type_mismatch_finding(
            op_text, op_name, type_text, ssa_types)
    return None


def _op_declared_type_text(op_text: str) -> str | None:
    _op_name, name_end = _read_bare_op_name(op_text, 0)
    type_index = _depth_zero_colon_index(op_text, name_end)
    if type_index is None:
        return None
    type_text = _mlir_simple_type_text(op_text, type_index + 1, len(op_text))
    return _normalized_mlir_fragment(type_text) if type_text else None


def _arith_operand_type_mismatch_finding(
        op_text: str, op_name: str, declared_type: str,
        ssa_types: dict[str, str | None]) -> str | None:
    for value in _op_ssa_operands_to_validate(op_text, op_name):
        actual_type = ssa_types.get(value)
        if actual_type is not None \
                and _normalized_mlir_fragment(actual_type) != declared_type:
            return (
                f"{op_name} operand type mismatch for {value}: defined as "
                f"{actual_type}, operation declares {declared_type}")
    return None


def _mlir_type_is_known_arith_value_type(type_text: str) -> bool:
    return (_mlir_type_is_integer_like(type_text)
            or _mlir_type_is_float_like(type_text))


def _mlir_type_is_supported_static_type(type_text: str) -> bool:
    stripped = _normalized_mlir_fragment(type_text)
    if not stripped:
        return False
    if stripped.startswith("!"):
        return _single_type_text_is_plausible(stripped)
    if _mlir_type_is_known_arith_value_type(stripped):
        return True
    if stripped.startswith("memref<"):
        element_type = _mlir_shaped_element_type(stripped)
        return element_type is not None \
            and _mlir_type_is_supported_static_type(element_type)
    if stripped.startswith("tuple<"):
        inner = _mlir_enclosed_type_inner(stripped, "tuple")
        if inner is None:
            return False
        parts = _split_depth_zero_commas(inner)
        return parts is not None and all(
            _mlir_type_is_supported_static_type(part) for part in parts)
    if stripped.startswith("complex<"):
        inner = _mlir_enclosed_type_inner(stripped, "complex")
        return inner is not None and _mlir_type_is_float_like(inner)
    return False


def _mlir_type_is_integer_like(type_text: str) -> bool:
    stripped = _normalized_mlir_fragment(type_text)
    element_type = _mlir_vector_tensor_element_type(stripped)
    if element_type is not None:
        return _mlir_type_is_integer_like(element_type)
    if stripped == "index":
        return True
    if len(stripped) > 1 and stripped[0] == "i":
        return stripped[1:].isdigit() and int(stripped[1:]) > 0
    if len(stripped) > 2 and stripped[:2] in ("si", "ui"):
        return stripped[2:].isdigit() and int(stripped[2:]) > 0
    return False


def _mlir_type_is_float_like(type_text: str) -> bool:
    stripped = _normalized_mlir_fragment(type_text)
    element_type = _mlir_vector_tensor_element_type(stripped)
    if element_type is not None:
        return _mlir_type_is_float_like(element_type)
    return stripped in _MLIR_FLOAT_SCALAR_TYPES


def _mlir_shaped_element_type(type_text: str) -> str | None:
    stripped = type_text.strip()
    if not stripped.startswith(("memref<", "vector<", "tensor<")):
        return None
    if stripped.startswith("vector<"):
        return _mlir_vector_element_type(stripped)
    open_index = stripped.find("<")
    close_index = _matching_closer_index(stripped, open_index, "<", ">")
    if close_index != len(stripped) - 1:
        return None
    inner = stripped[open_index + 1:close_index]
    layout_parts = _split_depth_zero_commas(inner)
    if layout_parts is None or not layout_parts:
        return None
    parts = _split_depth_zero_x(layout_parts[0])
    if len(parts) < 2:
        return None
    element_type = parts[-1].strip()
    return element_type or None


def _mlir_vector_element_type(type_text: str) -> str | None:
    open_index = type_text.find("<")
    close_index = _matching_closer_index(type_text, open_index, "<", ">")
    if close_index != len(type_text) - 1:
        return None
    inner = type_text[open_index + 1:close_index]
    layout_parts = _split_depth_zero_commas(inner)
    if layout_parts is None or len(layout_parts) != 1:
        return None
    parts = _split_depth_zero_x(layout_parts[0])
    if not parts:
        return None
    if len(parts) > 1 and not all(
            _mlir_vector_dim_is_plausible(part) for part in parts[:-1]):
        return None
    element_type = parts[-1].strip()
    return element_type or None


def _mlir_vector_dim_is_plausible(dim: str) -> bool:
    stripped = dim.strip()
    if not stripped:
        return False
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        return inner.isdigit() and int(inner) > 0
    return stripped.isdigit() and int(stripped) > 0


def _mlir_vector_tensor_element_type(type_text: str) -> str | None:
    stripped = type_text.strip()
    if not stripped.startswith(("vector<", "tensor<")):
        return None
    return _mlir_shaped_element_type(stripped)


def _mlir_enclosed_type_inner(type_text: str, prefix: str) -> str | None:
    stripped = type_text.strip()
    head = prefix + "<"
    if not stripped.startswith(head):
        return None
    open_index = len(prefix)
    close_index = _matching_closer_index(stripped, open_index, "<", ">")
    if close_index != len(stripped) - 1:
        return None
    return stripped[open_index + 1:close_index].strip()


def _split_depth_zero_x(text: str) -> tuple[str, ...]:
    pairs = {"(": ")", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    parts: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers:
            if stack and stack[-1] == char:
                stack.pop()
        elif char == "x" and not stack \
                and (i == 0 or not text[i - 1].isalpha()):
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    return tuple(parts)


def _op_ssa_operands_to_validate(
        op_text: str, op_name: str) -> tuple[str, ...]:
    if op_name == "scf.for":
        return _mlir_raw_ssa_values(_scf_for_operand_text(op_text))
    if op_name == "scf.while":
        return _mlir_raw_ssa_values(_scf_while_operand_text(op_text))
    return _mlir_raw_ssa_values(_op_ssa_validation_text(op_text, op_name))


def _op_ssa_validation_text(op_text: str, op_name: str) -> str:
    if op_name not in _REGION_HEADER_SSA_VALIDATED_OPS:
        return op_text
    body_start = op_text.find("{")
    return op_text if body_start == -1 else op_text[:body_start]


def _op_non_ssa_operand_finding(
        op_text: str, op_name: str) -> str | None:
    if op_name not in _TYPED_SSA_OPERAND_OPS:
        if op_name == "scf.if":
            return _single_header_operand_ssa_finding(op_text, op_name)
        if op_name == "cf.assert":
            return _first_comma_operand_ssa_finding(op_text, op_name)
        if op_name == "cf.br":
            return _branch_non_ssa_operand_finding(op_text, op_name)
        if op_name == "cf.cond_br":
            finding = _first_comma_operand_ssa_finding(op_text, op_name)
            return finding or _branch_non_ssa_operand_finding(
                op_text, op_name)
        if op_name == "scf.condition":
            return _scf_condition_non_ssa_operand_finding(op_text)
        if op_name == "scf.for":
            assignment_finding = _scf_assignment_list_finding(
                _scf_iter_args_text(op_text), op_name)
            return assignment_finding or _value_list_non_ssa_finding(
                _scf_for_operand_text(op_text), op_name)
        if op_name == "scf.while":
            assignment_finding = _scf_assignment_list_finding(
                op_text, op_name)
            return assignment_finding or _value_list_non_ssa_finding(
                _scf_while_operand_text(op_text), op_name)
        if op_name == "func.call":
            return _func_call_non_ssa_arg_finding(op_text)
        if op_name == "memref.load":
            return _memref_access_non_ssa_operand_finding(
                op_text, op_name)
        if op_name == "memref.store":
            return _memref_access_non_ssa_operand_finding(
                op_text, op_name)
        if op_name in _ARITH_SSA_OPERAND_OPS:
            return _arith_non_ssa_operand_finding(op_text, op_name)
        return None
    type_index = _depth_zero_colon_index(op_text, 0)
    if type_index is None:
        return None
    _name, name_end = _read_bare_op_name(op_text, 0)
    operand_text = op_text[name_end:type_index].strip()
    if not operand_text:
        return None
    operands = _split_depth_zero_commas(operand_text) or [operand_text]
    if not operands:
        return None
    if not all(operand.strip().startswith("%") for operand in operands):
        return (
            f"malformed {op_name}: non-SSA operands are not "
            "statically translatable")
    return None


def _single_header_operand_ssa_finding(
        op_text: str, op_name: str) -> str | None:
    _name, name_end = _read_bare_op_name(op_text, 0)
    body_start = op_text.find("{")
    limit = len(op_text) if body_start == -1 else body_start
    operand = op_text[name_end:limit].strip().split(None, 1)[0:1]
    if operand and not operand[0].startswith("%"):
        return (
            f"malformed {op_name}: non-SSA operands are not "
            "statically translatable")
    return None


def _first_comma_operand_ssa_finding(
        op_text: str, op_name: str) -> str | None:
    _name, name_end = _read_bare_op_name(op_text, 0)
    body_start = op_text.find("{")
    limit = len(op_text) if body_start == -1 else body_start
    segment = op_text[name_end:limit].strip()
    parts = _split_depth_zero_commas(segment)
    if not parts:
        return None
    first = parts[0].strip()
    if first and not first.startswith("%"):
        return (
            f"malformed {op_name}: non-SSA operands are not "
            "statically translatable")
    return None


def _branch_non_ssa_operand_finding(
        op_text: str, op_name: str) -> str | None:
    for value in _branch_target_arg_values(op_text):
        if not value.startswith("%"):
            return (
                f"malformed {op_name}: non-SSA branch operands are not "
                "statically translatable")
    return None


def _branch_target_arg_values(op_text: str) -> tuple[str, ...]:
    values: list[str] = []
    i = 0
    while i < len(op_text):
        if op_text[i] != "^":
            i += 1
            continue
        end = i + 1
        while end < len(op_text) and (
                op_text[end].isalnum() or op_text[end] in "_$.-"):
            end += 1
        j = _skip_spaces(op_text, end)
        if j >= len(op_text) or op_text[j] != "(":
            i = max(end, i + 1)
            continue
        close = _matching_closer_index(op_text, j, "(", ")")
        if close is None:
            i = max(end, i + 1)
            continue
        for part in _split_depth_zero_commas(
                op_text[j + 1:close].strip()) or ():
            colon = _depth_zero_colon_index(part, 0)
            values.append(part[:colon if colon is not None
                               else len(part)].strip())
        i = close + 1
    return tuple(values)


def _func_call_non_ssa_arg_finding(op_text: str) -> str | None:
    _name, name_end = _read_bare_op_name(op_text, 0)
    at_index = op_text.find("@", name_end)
    if at_index == -1:
        return None
    open_index = op_text.find("(", at_index)
    if open_index == -1:
        return None
    close_index = _matching_closer_index(op_text, open_index, "(", ")")
    if close_index is None:
        return None
    return _value_list_non_ssa_finding(
        op_text[open_index + 1:close_index], "func.call")


def _typed_ssa_operand_type_finding(
        op_text: str, ssa_types: dict[str, str | None]) -> str | None:
    op_name, name_end = _read_bare_op_name(op_text, 0)
    type_index = _depth_zero_colon_index(op_text, name_end)
    if type_index is None:
        return None
    value_text = op_text[name_end:type_index].strip()
    type_text = _mlir_simple_type_text(op_text, type_index + 1, len(op_text))
    return _return_ssa_type_finding(value_text, type_text, ssa_types)


def _function_call_signature_table(
        structural: str) -> dict[str, tuple[str, str]]:
    signatures: dict[str, tuple[str, str]] = {}
    for interface in _mlir_func_interfaces(structural):
        fields = _mlir_func_interface_fields(interface)
        if fields is None:
            continue
        symbol, _visibility, args, result, _body_kind = fields
        signatures[symbol] = args, result
    return signatures


def _func_call_signature_finding(
        op_text: str,
        ssa_types: dict[str, str | None],
        call_signatures: dict[str, tuple[str, str]]) -> str | None:
    parts = _func_call_signature_parts(op_text)
    if parts is None:
        return None
    symbol, arg_values, call_args, call_result = parts
    callee_signature = call_signatures.get(symbol)
    if callee_signature is None:
        return None
    callee_args, callee_result = callee_signature
    if call_args != callee_args or call_result != callee_result:
        return (
            f"func.call signature mismatch for {symbol}: call site "
            f"{call_args} -> {call_result or '()'} does not match callee "
            f"{callee_args} -> {callee_result or '()'}")
    call_result_arity = len(_mlir_return_type_items(call_result))
    if call_result_arity > 0 and all(
            typ is None for typ in _mlir_return_type_items(call_result)):
        return f"func.call has malformed result type for {symbol}"
    expected_arg_types = _mlir_parenthesized_type_items(callee_args)
    if len(arg_values) != len(expected_arg_types):
        return (
            f"func.call arity mismatch for {symbol}: "
            f"{len(arg_values)} argument(s) for "
            f"{len(expected_arg_types)} parameter(s)")
    for value, expected_type in zip(arg_values, expected_arg_types):
        actual_type = ssa_types.get(value)
        if actual_type is not None and actual_type != expected_type:
            return (
                f"func.call argument type mismatch for {symbol}: {value} "
                f"defined as {actual_type}, passed as {expected_type}")
    return None


def _func_call_signature_parts(
        op_text: str) -> tuple[str, tuple[str, ...], str, str] | None:
    _name, name_end = _read_bare_op_name(op_text, 0)
    at_index = op_text.find("@", name_end)
    if at_index == -1:
        return None
    symbol_ref = _mlir_symbol_ref_at(op_text, at_index, len(op_text))
    if symbol_ref is None:
        return None
    symbol, symbol_end = symbol_ref
    symbol = _normalize_symbol_ref(symbol)
    open_index = _skip_spaces(op_text, symbol_end)
    if open_index >= len(op_text) or op_text[open_index] != "(":
        return None
    close_index = _matching_closer_index(op_text, open_index, "(", ")")
    if close_index is None:
        return None
    arg_values = tuple(
        part.strip() for part in _split_depth_zero_commas(
            op_text[open_index + 1:close_index].strip()) or ()
        if part.strip())
    type_index = _depth_zero_colon_index(op_text, close_index + 1)
    if type_index is None:
        return None
    signature = op_text[type_index + 1:].strip()
    arrow = _depth_zero_arrow_index(signature, 0, len(signature))
    if arrow is None:
        return None
    args = _mlir_func_signature_arg_types(signature[:arrow])
    result = _mlir_func_signature_result_types(signature[arrow + 2:])
    return symbol, arg_values, args, result


def _mlir_parenthesized_type_items(text: str) -> tuple[str | None, ...]:
    stripped = text.strip()
    if stripped == "()":
        return ()
    if not stripped.startswith("(") or not stripped.endswith(")"):
        return (_normalized_mlir_fragment(stripped),)
    close = _matching_closer_index(stripped, 0, "(", ")")
    if close != len(stripped) - 1:
        return (_normalized_mlir_fragment(stripped),)
    inner = stripped[1:close].strip()
    if not inner:
        return ()
    parts = _split_depth_zero_commas(inner)
    if parts is None:
        return (_normalized_mlir_fragment(stripped),)
    return tuple(_normalized_mlir_fragment(part.strip()) or None
                 for part in parts)


def _strict_static_mlir_findings(mlir_text: str) -> tuple[str, ...]:
    structural = _comment_stripped_text(mlir_text)
    signatures = _function_call_signature_table(structural)
    findings: list[str] = []
    findings.extend(_strict_static_duplicate_func_symbol_findings(mlir_text))
    findings.extend(_strict_static_func_signature_type_findings(mlir_text))
    findings.extend(_strict_static_empty_return_findings(structural))
    findings.extend(_strict_static_func_terminator_findings(structural))
    i = 0
    while i < len(structural):
        if structural[i] == '"':
            i = _quoted_span_end(structural, i)
            continue
        if not _bare_word_at(structural, i, "func.call"):
            i += 1
            continue
        op_end = _bare_dialect_op_end(
            structural, i, len(structural), consume_body=False)
        if op_end is None:
            i += len("func.call")
            continue
        parts = _func_call_signature_parts(structural[i:op_end])
        if parts is not None:
            symbol, _arg_values, _call_args, _call_result = parts
            if symbol not in signatures:
                findings.append(f"func.call references undefined callee: {symbol}")
        i = op_end
    return tuple(findings)


def _strict_static_func_terminator_findings(
        structural: str) -> tuple[str, ...]:
    findings: list[str] = []
    i = 0
    while i < len(structural):
        if structural[i] == '"':
            i = _quoted_span_end(structural, i)
            continue
        if not _bare_word_at(structural, i, "func.func"):
            i += 1
            continue
        interface = _mlir_func_interface_at(structural, i)
        fields = _mlir_func_interface_fields(interface) \
            if interface is not None else None
        if fields is None:
            i += len("func.func")
            continue
        symbol, _visibility, _args, _result, body_kind = fields
        if body_kind != "body":
            i += len("func.func")
            continue
        body = _func_body_span_at(structural, i)
        if body is None:
            i += len("func.func")
            continue
        body_start, body_end = body
        finding = _func_body_terminator_finding(
            structural, body_start, body_end)
        if finding is not None:
            findings.append(f"function {symbol} {finding}")
        i = body_end + 1
    for symbol, body_start, body_end in _mlir_generic_func_body_spans(
            structural):
        finding = _func_body_terminator_finding(
            structural, body_start, body_end)
        if finding is not None:
            findings.append(f"function {symbol} (generic form) {finding}")
    return tuple(findings)


def _func_body_terminator_finding(
        structural: str, body_start: int, body_end: int) -> str | None:
    i = body_start + 1
    block_started = False
    block_terminated = False
    while True:
        i = _skip_spaces(structural, i)
        if i >= body_end:
            break
        line_end = min(_line_end(structural, i), body_end)
        if structural.startswith(_GENERIC_OP_SENTINEL, i):
            end = _generic_top_level_op_end(
                structural, i + len(_GENERIC_OP_SENTINEL))
            if end is None or end > body_end:
                return "has malformed generic operation in function body"
            if block_terminated:
                return "has operation after block terminator"
            block_started = True
            i = end
            continue
        if structural[i] == "^":
            if block_started and not block_terminated:
                return "has block missing terminator"
            block_started = True
            block_terminated = False
            i = line_end
            continue
        op_start = i
        if structural[i] == "%":
            eq_index = structural.find("=", i, line_end)
            if eq_index == -1:
                i = line_end
                continue
            op_start = _skip_spaces(structural, eq_index + 1)
        if structural.startswith(_GENERIC_OP_SENTINEL, op_start):
            end = _generic_top_level_op_end(
                structural, op_start + len(_GENERIC_OP_SENTINEL))
            if end is None or end > body_end:
                return "has malformed generic operation in function body"
            if block_terminated:
                return "has operation after block terminator"
            block_started = True
            i = end
            continue
        op_name, name_end = _read_bare_op_name(structural, op_start)
        if not op_name:
            word, word_end = _read_bare_word(structural, op_start)
            op_name = word
            name_end = word_end
        if not op_name:
            i = line_end
            continue
        if block_terminated:
            return "has operation after block terminator"
        block_started = True
        if op_name in _FUNC_BLOCK_TERMINATOR_OPS:
            block_terminated = True
        if op_name == "return":
            i = line_end
            continue
        end = _bare_dialect_op_end(
            structural, op_start, body_end, consume_body=False)
        if end is None:
            end = _bare_dialect_op_end(
                structural, op_start, body_end, consume_body=True)
        i = end if end is not None and end > op_start else line_end
    if not block_started or not block_terminated:
        return "has block missing terminator"
    return None


def _strict_static_duplicate_func_symbol_findings(
        mlir_text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    findings: list[str] = []
    for symbol in _mlir_func_symbol_occurrences(mlir_text):
        if symbol in seen:
            findings.append(f"duplicate func.func symbol: {symbol}")
        else:
            seen.add(symbol)
    return tuple(findings)


def _mlir_func_symbol_occurrences(mlir_text: str) -> tuple[str, ...]:
    structural = _comment_stripped_text(mlir_text)
    symbols: list[str] = []
    i = 0
    while i < len(structural):
        if structural[i] == '"':
            i = _quoted_span_end(structural, i)
            continue
        if not _bare_word_at(structural, i, "func.func"):
            i += 1
            continue
        interface = _mlir_func_interface_at(structural, i)
        fields = _mlir_func_interface_fields(interface) \
            if interface is not None else None
        if fields is not None:
            symbols.append(fields[0])
        i += len("func.func")
    symbols.extend(_mlir_generic_func_symbol_occurrences(mlir_text))
    return tuple(symbols)


def _mlir_generic_func_symbol_occurrences(mlir_text: str) -> tuple[str, ...]:
    structural = _comment_stripped_text(mlir_text)
    symbols: list[str] = []
    i = 0
    while i < len(structural):
        found = structural.find('"func.func"', i)
        if found == -1:
            break
        if not _mlir_op_start_context_allows(structural, found):
            i = found + 1
            continue
        quoted_end = found + len('"func.func"')
        j = _skip_spaces(structural, quoted_end)
        if j >= len(structural) or structural[j] != "(":
            i = quoted_end
            continue
        operands_end = _matching_closer_index(structural, j, "(", ")")
        if operands_end is None:
            i = quoted_end
            continue
        props = _generic_property_dict_after(structural, operands_end + 1)
        if props is None:
            i = operands_end + 1
            continue
        props_text, props_end = props
        assignments = _generic_property_assignments(props_text)
        sym_name = assignments.get("sym_name")
        if sym_name is not None:
            symbol = _symbol_from_generic_string_property(sym_name)
            if symbol is not None:
                symbols.append(symbol)
        i = props_end
    return tuple(symbols)


def _strict_static_func_signature_type_findings(
        mlir_text: str) -> tuple[str, ...]:
    findings: list[str] = []
    for interface in _mlir_func_interfaces(mlir_text):
        fields = _mlir_func_interface_fields(interface)
        if fields is None:
            continue
        symbol, _visibility, args, result, _body_kind = fields
        for type_text in _mlir_parenthesized_type_items(args):
            if type_text is not None \
                    and not _mlir_type_is_supported_static_type(type_text):
                findings.append(
                    f"unsupported function argument type for {symbol}: "
                    f"{type_text}")
        for type_text in _mlir_return_type_items(result):
            if type_text is not None \
                    and not _mlir_type_is_supported_static_type(type_text):
                findings.append(
                    f"unsupported function result type for {symbol}: "
                    f"{type_text}")
    return tuple(findings)


def _strict_static_empty_return_findings(
        structural: str) -> tuple[str, ...]:
    findings: list[str] = []
    i = 0
    while i < len(structural):
        if structural[i] == '"':
            i = _quoted_span_end(structural, i)
            continue
        if not _bare_word_at(structural, i, "func.func"):
            i += 1
            continue
        interface = _mlir_func_interface_at(structural, i)
        fields = _mlir_func_interface_fields(interface) \
            if interface is not None else None
        if fields is None:
            i += len("func.func")
            continue
        symbol, _visibility, _args, result, body_kind = fields
        if body_kind != "body" or not _mlir_return_type_items(result):
            i += len("func.func")
            continue
        body = _func_body_span_at(structural, i)
        if body is None:
            i += len("func.func")
            continue
        body_start, body_end = body
        if _func_body_has_empty_return(
                structural[body_start + 1:body_end]):
            findings.append(
                f"function {symbol} declares result type {result} "
                "but has an empty return")
        i = body_end + 1
    for symbol, body_start, body_end, result in (
            _mlir_generic_func_body_spans_with_result(structural)):
        if not _mlir_return_type_items(result):
            continue
        if _func_body_has_empty_return(
                structural[body_start + 1:body_end]):
            findings.append(
                f"function {symbol} (generic form) declares result type "
                f"{result} but has an empty return")
    return tuple(findings)


def _func_body_span_at(
        structural: str, token_start: int) -> tuple[int, int] | None:
    end = _func_op_end(structural, token_start + len("func.func"))
    if end is None:
        return None
    cursor = token_start
    while cursor < end:
        body_start = structural.find("{", cursor, end)
        if body_start == -1:
            return None
        body_end = _matching_closer_index(structural, body_start, "{", "}")
        if body_end is None or body_end > end:
            return None
        content = structural[body_start + 1:body_end]
        if _braced_content_looks_like_property_dict(content):
            cursor = body_end + 1
            continue
        return body_start, body_end
    return None


def _func_body_has_empty_return(body: str) -> bool:
    i = 0
    while i < len(body):
        if body[i] == '"':
            i = _quoted_span_end(body, i)
            continue
        for token in ("func.return", "return"):
            if not _bare_word_at(body, i, token):
                continue
            line_end = _line_end(body, i)
            tail = body[i + len(token):line_end].strip()
            loc_end = _skip_loc_suffix(body, i + len(token))
            if not tail or (
                    loc_end != i + len(token)
                    and not body[loc_end:line_end].strip()):
                return True
        i += 1
    return False


def _memref_access_non_ssa_operand_finding(
        op_text: str, op_name: str) -> str | None:
    segment = _op_operand_segment_before_type(op_text)
    parts = _split_depth_zero_commas(segment) or [segment]
    for part in parts:
        if _memref_operand_non_ssa(part):
            return (
                f"malformed {op_name}: non-SSA operands are not "
                "statically translatable")
    return None


def _memref_operand_non_ssa(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    bracket = value.find("[")
    head = value[:bracket if bracket != -1 else len(value)].strip()
    if not head.startswith("%"):
        return True
    if bracket == -1:
        return False
    close = _matching_closer_index(value, bracket, "[", "]")
    if close is None:
        return False
    indices = _split_depth_zero_commas(value[bracket + 1:close].strip())
    for index in indices or ():
        stripped = index.strip()
        if stripped and not stripped.startswith("%"):
            return True
    return False


def _arith_non_ssa_operand_finding(
        op_text: str, op_name: str) -> str | None:
    segment = _op_operand_segment_before_type(op_text)
    parts = list(_split_depth_zero_commas(segment) or [segment])
    if op_name in _ARITH_COMPARE_OPS and parts:
        parts = parts[1:]
    return _value_list_non_ssa_finding(", ".join(parts), op_name)


def _op_operand_segment_before_type(op_text: str) -> str:
    _name, name_end = _read_bare_op_name(op_text, 0)
    body_start = op_text.find("{")
    limit = len(op_text) if body_start == -1 else body_start
    type_index = _depth_zero_colon_index(op_text, name_end)
    if type_index is not None and type_index < limit:
        limit = type_index
    return op_text[name_end:limit].strip()


def _scf_condition_non_ssa_operand_finding(op_text: str) -> str | None:
    _name, name_end = _read_bare_op_name(op_text, 0)
    i = _skip_spaces(op_text, name_end)
    if i < len(op_text) and op_text[i] == "(":
        close = _matching_closer_index(op_text, i, "(", ")")
        if close is not None:
            if _value_list_non_ssa_finding(
                    op_text[i + 1:close], "scf.condition"):
                return (
                    "malformed scf.condition: non-SSA operands are not "
                    "statically translatable")
            value_tail = op_text[close + 1:]
            type_index = _depth_zero_colon_index(value_tail, 0)
            if type_index is not None:
                return _value_list_non_ssa_finding(
                    value_tail[:type_index], "scf.condition")
    return None


def _scf_for_operand_text(op_text: str) -> str:
    _name, name_end = _read_bare_op_name(op_text, 0)
    body_start = op_text.find("{")
    header = op_text[name_end:len(op_text) if body_start == -1
                     else body_start]
    eq_index = _depth_zero_equals_index(header, 0)
    if eq_index is None:
        return ""
    values = _scf_for_header_operand_values(header[eq_index + 1:])
    return ", ".join(values)


def _scf_for_header_operand_values(tail: str) -> tuple[str, ...]:
    to_index = _depth_zero_keyword_index(tail, 0, len(tail), "to")
    if to_index is None:
        return (tail.strip(),) if tail.strip() else ()
    step_index = _depth_zero_keyword_index(
        tail, to_index + len("to"), len(tail), "step")
    if step_index is None:
        return (tail[:to_index].strip(),
                tail[to_index + len("to"):].strip())
    iter_args_index = _depth_zero_keyword_index(
        tail, step_index + len("step"), len(tail), "iter_args")
    loc_index = _depth_zero_keyword_index(
        tail, step_index + len("step"), len(tail), "loc")
    step_end_candidates = [
        index for index in (iter_args_index, loc_index) if index is not None
    ]
    step_end = min(step_end_candidates) if step_end_candidates else len(tail)
    values = [
        tail[:to_index].strip(),
        tail[to_index + len("to"):step_index].strip(),
        tail[step_index + len("step"):step_end].strip(),
    ]
    if iter_args_index is not None:
        values.extend(_scf_assignment_rhs_values(tail[iter_args_index:]))
    return tuple(value for value in values if value)


def _scf_iter_args_text(header: str) -> str:
    iter_args_index = _depth_zero_keyword_index(
        header, 0, len(header), "iter_args")
    if iter_args_index is None:
        return ""
    return header[iter_args_index:]


def _scf_assignment_rhs_values(text: str) -> tuple[str, ...]:
    open_index = text.find("(")
    if open_index == -1:
        return ()
    close_index = _matching_closer_index(text, open_index, "(", ")")
    if close_index is None:
        return ()
    values: list[str] = []
    for part in _split_depth_zero_commas(
            text[open_index + 1:close_index].strip()) or ():
        eq_index = _depth_zero_equals_index(part, 0)
        if eq_index is not None:
            values.append(part[eq_index + 1:].strip())
    return tuple(value for value in values if value)


def _scf_assignment_list_finding(text: str, op_name: str) -> str | None:
    if not text.strip():
        return None
    open_index = text.find("(")
    if open_index == -1:
        return None
    close_index = _matching_closer_index(text, open_index, "(", ")")
    if close_index is None:
        return f"malformed {op_name}: malformed region argument list"
    parts = _split_depth_zero_commas(text[open_index + 1:close_index].strip())
    if parts is None:
        return f"malformed {op_name}: malformed region argument list"
    for part in parts:
        eq_index = _depth_zero_equals_index(part, 0)
        if eq_index is None:
            return (
                f"malformed {op_name}: region arguments must be "
                "initialized with SSA operands")
        lhs = part[:eq_index].strip()
        rhs = part[eq_index + 1:].strip()
        if not _ssa_result_names(lhs):
            return f"malformed {op_name}: malformed region argument list"
        if not rhs.startswith("%"):
            return (
                f"malformed {op_name}: non-SSA operands are not "
                "statically translatable")
    return None


def _scf_while_operand_text(op_text: str) -> str:
    _name, name_end = _read_bare_op_name(op_text, 0)
    body_start = op_text.find("{")
    header = op_text[name_end:len(op_text) if body_start == -1
                     else body_start]
    open_index = header.find("(")
    if open_index == -1:
        return ""
    close_index = _matching_closer_index(header, open_index, "(", ")")
    if close_index is None:
        return ""
    rhs_values: list[str] = []
    for part in _split_depth_zero_commas(
            header[open_index + 1:close_index].strip()) or ():
        eq_index = _depth_zero_equals_index(part, 0)
        if eq_index is not None:
            rhs_values.append(part[eq_index + 1:].strip())
    return ", ".join(rhs_values)


def _value_list_non_ssa_finding(
        value_text: str, op_name: str) -> str | None:
    values = [part.strip() for part in _split_depth_zero_commas(
        value_text.strip()) or ()]
    for value in values:
        if not value:
            continue
        if not value.startswith("%"):
            return (
                f"malformed {op_name}: non-SSA operands are not "
                "statically translatable")
    return None


def _return_ssa_type_finding(
        value_text: str, return_type: str | None,
        ssa_types: dict[str, str | None]) -> str | None:
    values = [part.strip() for part in _split_depth_zero_commas(value_text)
              or [value_text]]
    return_types = _mlir_return_type_items(return_type)
    if len(values) != len(return_types):
        return (
            "return arity mismatch: "
            f"{len(values)} value(s) for {len(return_types)} type(s)")
    for value_text_item, expected_type in zip(values, return_types):
        if not value_text_item.startswith("%"):
            continue
        value = value_text_item.split()[0]
        if value not in ssa_types:
            return f"undefined SSA value in return: {value}"
        value_type = ssa_types[value]
        if value_type is not None and expected_type is not None \
                and value_type != expected_type:
            return (
                f"return type mismatch for {value}: defined as "
                f"{value_type}, returned as {expected_type}")
    return None


def _mlir_return_type_items(return_type: str | None) -> tuple[str | None, ...]:
    if return_type is None:
        return ()
    text = return_type.strip()
    if not text or text == "()":
        return ()
    if text.startswith("("):
        close = _matching_closer_index(text, 0, "(", ")")
        if close == len(text) - 1:
            inner = text[1:close].strip()
            if not inner:
                return ()
            parts = _split_depth_zero_commas(inner)
            if parts is not None:
                return tuple(_normalized_mlir_fragment(part) or None
                             for part in parts)
    parts = _split_depth_zero_commas(text)
    if parts is not None and len(parts) > 1:
        return tuple(_normalized_mlir_fragment(part) or None
                     for part in parts)
    return (_normalized_mlir_fragment(text),)


def _return_values_are_ssa(value_text: str) -> bool:
    values = [part.strip() for part in _split_depth_zero_commas(value_text)
              or [value_text]]
    return bool(values) and all(value.startswith("%") for value in values)


def _module_body_findings(structural: str, body_start: int,
                          body_end: int) -> tuple[str, ...]:
    """Scan a module operation list for known/generic op boundaries."""
    findings: list[str] = []
    i = body_start + 1
    while True:
        i = _skip_spaces(structural, i)
        if i >= body_end:
            break
        if structural[i] == "^":
            line_end = min(_line_end(structural, i), body_end)
            if not _block_label_line_is_plausible(
                    structural[i:line_end]):
                findings.append(
                    "malformed block label in module operation list")
                break
            i = line_end
            continue
        if structural[i] in "#!":
            end = _alias_line_end(structural, i)
            if end is None or end > body_end:
                findings.append(
                    "malformed alias in module operation list")
                break
            i = end + 1
            continue
        matched = False
        for token in _TOP_LEVEL_STRUCTURE_TOKENS:
            if not structural.startswith(token, i):
                continue
            after_i = i + len(token)
            after = structural[after_i] if after_i < len(structural) else ""
            if after and not (after.isspace() or after in "{(@"):
                continue
            if token == _GENERIC_OP_SENTINEL:
                end = _generic_top_level_op_end(structural, after_i)
            elif token in ("module", "builtin.module"):
                end = _module_op_end(structural, after_i)
            else:
                end = _func_op_end(structural, after_i)
            if end is None or end > body_end:
                if token == "func.func":
                    detail_findings = _func_op_body_findings_for_diagnostic(
                        structural, i, body_end)
                    if detail_findings:
                        findings.extend(detail_findings)
                        return tuple(findings)
                findings.append(
                    f"malformed nested `{token}` operation in module "
                    "operation list")
                return tuple(findings)
            i = end
            matched = True
            break
        if matched:
            continue
        bare_end = _bare_dialect_op_end(structural, i, body_end)
        if bare_end is not None:
            i = bare_end
            continue
        findings.append(
            "unexpected text in module operation list")
        break
    return tuple(findings)


def _top_level_op_detail_findings(structural: str, token: str,
                                  start: int,
                                  after_token: int) -> tuple[str, ...]:
    if token in ("module", "builtin.module"):
        body_start = _module_body_start_after_metadata(structural, after_token)
        if body_start is None:
            return ()
        body_end = _matching_closer_index(structural, body_start, "{", "}")
        if body_end is None:
            return ()
        return _module_body_findings(structural, body_start, body_end)
    if token == "func.func":
        return _func_op_body_findings_for_diagnostic(
            structural, start, len(structural))
    return ()


def _func_op_body_findings_for_diagnostic(
        structural: str, start: int, limit: int) -> tuple[str, ...]:
    cursor = start
    while cursor < limit:
        body_start = structural.find("{", cursor, limit)
        if body_start == -1:
            return ()
        body_end = _matching_closer_index(structural, body_start, "{", "}")
        if body_end is None or body_end > limit:
            return ()
        content = structural[body_start + 1:body_end]
        if _braced_content_looks_like_property_dict(content):
            cursor = body_end + 1
            continue
        findings = _func_body_findings(structural, body_start, body_end)
        if findings:
            return findings
        terminator_finding = _func_body_terminator_finding(
            structural, body_start, body_end)
        return (terminator_finding,) if terminator_finding else ()
    return ()


def _bare_dialect_op_end(structural: str, start: int,
                         limit: int, *, consume_body: bool = True,
                         ) -> int | None:
    op_name, name_end = _read_bare_op_name(structural, start)
    if not op_name:
        return None
    if (op_name in _TOP_LEVEL_STRUCTURE_TOKENS
            or op_name == _GENERIC_OP_SOURCE_TOKEN):
        return None
    j = _skip_spaces(structural, name_end)
    line_end = min(_line_end(structural, start), limit)
    op_boundary = _next_op_boundary(structural, start, limit)
    if not consume_body:
        type_index = _depth_zero_colon_index(
            structural, start, line_end)
        if type_index is None:
            continuation = _skip_spaces(structural, line_end)
            if continuation < limit and structural[continuation] == ":":
                type_index = continuation
                line_end = min(_line_end(structural, type_index), limit)
        next_op_index = (
            _next_body_op_start_after_type_tail(
                structural, type_index + 1, line_end)
            if type_index is not None and type_index < line_end
            else None)
        region_search_limit = (
            next_op_index if next_op_index is not None else op_boundary)
        region = _find_bare_dialect_region_body(
            structural, j, region_search_limit, _func_body_findings)
        if region is _MALFORMED_BARE_DIALECT_REGION:
            return None
        if region is not None and (
                (type_index is None or type_index < region[0])
                and (next_op_index is None or region[0] < next_op_index)):
            body_index, body_end = region
            if not _bare_dialect_region_header_is_plausible(
                    structural[j:body_index]):
                return None
            after_body = _skip_spaces(structural, body_end + 1)
            word, word_end = _read_bare_word(structural, after_body)
            if word in ("do", "else"):
                next_body = _skip_spaces(structural, word_end)
                if next_body >= limit or structural[next_body] != "{":
                    return None
                next_end = _matching_closer_index(
                    structural, next_body, "{", "}")
                if next_end is None or next_end > limit:
                    return None
                if (structural[next_body + 1:next_end].strip()
                        and _func_body_findings(
                            structural, next_body, next_end)):
                    return None
                return _skip_spaces(structural, next_end + 1)
            return after_body
        if type_index is None and op_name in ("func.return", "scf.yield"):
            tail_start = _skip_spaces(structural, name_end)
            if tail_start < line_end:
                loc_end = _skip_loc_suffix(structural, tail_start)
                if loc_end == tail_start or structural[loc_end:line_end].strip():
                    return None
        if type_index is not None and type_index < line_end:
            if op_name in ("func.return", "scf.yield") \
                    and not structural[name_end:type_index].strip():
                return None
            type_end = line_end if next_op_index is None else next_op_index
            if not _bare_dialect_type_tail_with_optional_loc_is_plausible(
                    structural, type_index + 1, type_end):
                return None
            if next_op_index is not None:
                return next_op_index
        elif op_name not in ("func.return", "scf.yield"):
            next_op_index = _next_body_op_start_after_no_type_tail(
                structural, op_name, name_end, line_end)
            tail_end = line_end if next_op_index is None else next_op_index
            tail = structural[name_end:tail_end].strip()
            if not _bare_dialect_no_type_tail_is_plausible(op_name, tail):
                return None
            if next_op_index is not None:
                return next_op_index
        return line_end
    validator = _func_body_findings if op_name.endswith(
        ".func") else _module_body_findings
    region = _find_bare_dialect_region_body(
        structural, j, op_boundary, validator)
    if region is _MALFORMED_BARE_DIALECT_REGION:
        return None
    if region is not None:
        body_index, body_end = region
        if not _bare_dialect_region_header_is_plausible(
                structural[j:body_index]):
            return None
        return _skip_loc_suffix(structural, body_end + 1)
    type_index = _depth_zero_colon_index(structural, start, line_end)
    if type_index is None:
        tail = structural[name_end:line_end].strip()
        if not _bare_dialect_no_type_tail_is_plausible(op_name, tail):
            return None
        return line_end
    if not _bare_dialect_type_tail_with_optional_loc_is_plausible(
            structural, type_index + 1, line_end):
        return None
    return line_end


def _next_body_op_start_after_type_tail(
        structural: str, start: int, line_end: int) -> int | None:
    i = start
    while i < line_end:
        if structural[i].isalpha() or structural[i] == "_":
            word, word_end = _read_bare_word(structural, i)
            if (word == "return" or "." in word) \
                    and _bare_dialect_type_tail_with_optional_loc_is_plausible(
                        structural, start, i):
                return i
            i = max(word_end, i + 1)
            continue
        i += 1
    return None


def _next_body_op_start_after_no_type_tail(
        structural: str, op_name: str, start: int,
        line_end: int) -> int | None:
    i = start
    while i < line_end:
        if structural[i].isalpha() or structural[i] == "_":
            word, word_end = _read_bare_word(structural, i)
            if (word == "return" or "." in word) \
                    and _bare_dialect_no_type_tail_is_plausible(
                        op_name, structural[start:i].strip()):
                return i
            i = max(word_end, i + 1)
            continue
        i += 1
    return None


def _bare_dialect_no_type_tail_is_plausible(op_name: str,
                                            tail: str) -> bool:
    if op_name in _GPU_DIMENSION_ID_OPS:
        return tail in ("x", "y", "z")
    if op_name in _BARE_DIALECT_TERMINATOR_OPS:
        return not tail
    if op_name == "gpu.barrier":
        return not tail
    if op_name == "cf.assert":
        return (bool(tail.strip())
                and "string_lit" in tail
                and all(char not in "{}:" for char in tail))
    if op_name in _BRANCH_WITH_BLOCK_TARGET_OPS:
        return "^" in tail and all(char not in "{}" for char in tail)
    if op_name.startswith("linalg."):
        return _linalg_no_type_tail_is_plausible(tail)
    return False


def _linalg_no_type_tail_is_plausible(tail: str) -> bool:
    stripped = tail.strip()
    if not stripped or "{" in stripped or "}" in stripped:
        return False
    if not ("ins(" in stripped or "outs(" in stripped):
        return False
    pairs = {"(": ")", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    i = 0
    while i < len(stripped):
        char = stripped[i]
        if char == "-" and i + 1 < len(stripped) \
                and stripped[i + 1] == ">":
            i += 2
            continue
        if char == "<" and i + 1 < len(stripped) \
                and stripped[i + 1] == "=":
            i += 2
            continue
        if (char == ">" and
                ((i > 0 and stripped[i - 1] in "-=")
                 or (i + 1 < len(stripped)
                     and stripped[i + 1] == "="))):
            i += 1
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers:
            if not stack or stack.pop() != char:
                return False
        i += 1
    return not stack


_GPU_DIMENSION_ID_OPS = frozenset((
    "gpu.block_dim", "gpu.block_id", "gpu.grid_dim", "gpu.thread_id",
))
_BARE_DIALECT_TERMINATOR_OPS = frozenset((
    "gpu.return", "gpu.terminator",
))
_BRANCH_WITH_BLOCK_TARGET_OPS = frozenset((
    "cf.br", "cf.cond_br",
))


_MALFORMED_BARE_DIALECT_REGION = object()


def _find_bare_dialect_region_body(
        structural: str,
        search_start: int,
        search_limit: int,
        body_validator,
        ) -> tuple[int, int] | object | None:
    cursor = search_start
    while cursor < search_limit:
        body_index = structural.find("{", cursor, search_limit)
        if body_index == -1:
            return None
        body_end = _matching_closer_index(structural, body_index, "{", "}")
        if body_end is None or body_end > search_limit:
            return None
        content = structural[body_index + 1:body_end]
        if not content.strip() or not body_validator(
                structural, body_index, body_end):
            return body_index, body_end
        if (not _braced_content_looks_like_property_dict(content)
                or not _text_before_brace_can_have_property_dict(
                    structural[search_start:body_index])):
            return _MALFORMED_BARE_DIALECT_REGION
        cursor = body_end + 1
    return None


def _braced_content_looks_like_property_dict(text: str) -> bool:
    parts = _split_depth_zero_commas(text.strip())
    if parts is None or not parts:
        return False
    for part in parts:
        stripped = part.strip()
        if not stripped:
            return False
        if stripped.startswith(("%", "^", _GENERIC_OP_SENTINEL)):
            return False
        if not _property_assignment_is_plausible(part):
            return False
    return True


def _text_before_brace_can_have_property_dict(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith(("%", "@", "#")):
        return False
    if stripped.endswith(("<", "=", ",")):
        return True
    return not any(char.isspace() for char in stripped)


_BARE_DIALECT_REGION_HEADER_WORDS = frozenset((
    "attributes", "bf16", "f16", "f32", "f64", "i1", "i8", "i16",
    "blocks", "clusters", "dynamic_shared_memory_size", "function",
    "i32", "i64", "in", "index", "ins", "iter_args", "kernel",
    "memref", "module", "nested", "none", "outs", "private", "public",
    "step", "tensor", "threads", "to", "vector", "workgroup",
))


def _bare_dialect_region_header_is_plausible(text: str) -> bool:
    """Reject obvious junk between a bare op name and its region body."""
    i = 0
    stripped = text.strip()
    while i < len(stripped):
        i = _skip_spaces(stripped, i)
        if i >= len(stripped):
            return True
        char = stripped[i]
        if char == "@":
            symbol_end = _read_symbol_ref_after_at(stripped, i)
            if symbol_end is None:
                return False
            i = symbol_end
            continue
        if char == "%":
            i += 1
            if i >= len(stripped) or not (
                    stripped[i].isalnum() or stripped[i] in "_$"):
                return False
            while i < len(stripped) and (
                    stripped[i].isalnum() or stripped[i] in "_$.-"):
                i += 1
            continue
        if char == "#":
            i += 1
            if i >= len(stripped) or not (
                    stripped[i].isalnum() or stripped[i] in "_$."):
                return False
            while i < len(stripped) and (
                    stripped[i].isalnum() or stripped[i] in "_$.-"):
                i += 1
            continue
        if char == "{":
            close = _matching_closer_index(stripped, i, "{", "}")
            if close is None or not _braced_content_looks_like_property_dict(
                    stripped[i + 1:close]):
                return False
            i = close + 1
            continue
        if char in "(<[":
            close = _matching_closer_index(
                stripped, i, char, {"(": ")", "<": ">", "[": "]"}[char])
            if close is None:
                return False
            i = close + 1
            continue
        if char in "=,:":
            i += 1
            continue
        if char == "-" and i + 1 < len(stripped) and stripped[i + 1] == ">":
            i += 2
            continue
        if char in "+-" or char.isdigit():
            i += 1
            while i < len(stripped) and (
                    stripped[i].isalnum() or stripped[i] in "._+-"):
                i += 1
            continue
        word, end = _read_bare_word(stripped, i)
        if not word or word not in _BARE_DIALECT_REGION_HEADER_WORDS:
            return False
        i = end
    return True


def _ssa_result_list_is_plausible(text: str) -> bool:
    parts = text.split(",")
    if not parts:
        return False
    for part in parts:
        item = part.strip()
        if not item.startswith("%") or item == "%":
            return False
        if any(char.isspace() for char in item):
            return False
    return True


def _body_keyword_index(text: str, start: int,
                        limit: int, keyword: str) -> int | None:
    i = start
    while i < limit:
        i = text.find(keyword, i, limit)
        if i == -1:
            return None
        if _bare_word_at(text, i, keyword):
            return i
        i += len(keyword)
    return None


def _depth_zero_keyword_index(text: str, start: int,
                              limit: int, keyword: str) -> int | None:
    pairs = {"(": ")", "{": "}", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    i = start
    while i < limit:
        char = text[i]
        if char == "-" and i + 1 < limit and text[i + 1] == ">":
            i += 2
            continue
        if char == "<" and i + 1 < limit and text[i + 1] == "=":
            i += 2
            continue
        if (char == ">" and
                ((i > 0 and text[i - 1] in "-=")
                 or (i + 1 < limit and text[i + 1] == "="))):
            i += 1
            continue
        if not stack and _bare_word_at(text, i, keyword):
            return i
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers:
            if stack and stack[-1] == char:
                stack.pop()
            elif not stack:
                return None
        i += 1
    return None


def _bare_dialect_type_tail_with_optional_loc_is_plausible(
        structural: str, start: int, end: int) -> bool:
    loc_index = _depth_zero_keyword_index(structural, start, end, "loc")
    if loc_index is None:
        return _bare_dialect_type_tail_is_plausible(structural[start:end])
    if not _bare_dialect_type_tail_is_plausible(
            structural[start:loc_index]):
        return False
    loc_end = _skip_loc_suffix(structural, loc_index)
    return loc_end != loc_index and not structural[loc_end:end].strip()


def _top_level_stream(
        structural: str, *, allow_bare_top_level: bool = True,
        ) -> tuple[bool, tuple[str, ...]]:
    """Scan top-level text for allowed aliases and full top-level ops."""
    findings: list[str] = []
    found_structure = False
    i = 0
    while True:
        i = _skip_spaces(structural, i)
        if i >= len(structural):
            break
        if structural[i] in "#!":
            end = _alias_line_end(structural, i)
            if end is None:
                findings.append(
                    "unexpected top-level text before the first MLIR "
                    "operation")
                break
            i = end + 1
            continue
        matched = False
        for token in _TOP_LEVEL_STRUCTURE_TOKENS:
            if not structural.startswith(token, i):
                continue
            after_i = i + len(token)
            after = structural[after_i] if after_i < len(structural) else ""
            if after and not (after.isspace() or after in "{(@"):
                continue
            if token == _GENERIC_OP_SENTINEL:
                end = _generic_top_level_op_end(structural, after_i)
            elif token in ("module", "builtin.module"):
                end = _module_op_end(structural, after_i)
            else:
                end = _func_op_end(structural, after_i)
            if end is None:
                detail_findings = _top_level_op_detail_findings(
                    structural, token, i, after_i)
                if detail_findings:
                    found_structure = True
                    findings.extend(detail_findings)
                else:
                    findings.append(
                        "unexpected top-level text before the first MLIR "
                        "operation")
                break
            found_structure = True
            i = end
            matched = True
            break
        if findings:
            break
        if matched:
            continue
        if allow_bare_top_level:
            bare_end = _bare_dialect_op_end(structural, i, len(structural))
            if bare_end is not None:
                found_structure = True
                i = bare_end
                continue
        findings.append(
            "unexpected top-level text before the first MLIR operation")
        break
    return found_structure, tuple(findings)


def _mlir_opt_output_artifact_findings(output_text: str) -> tuple[str, ...]:
    structural = _structural_text(output_text)
    has_structure, findings = _top_level_stream(
        structural, allow_bare_top_level=False)
    if findings:
        return findings
    if not has_structure:
        return ("no canonical top-level MLIR operation in output artifact",)
    return ()


def _mlir_symbol_refs(mlir_text: str) -> frozenset[str]:
    refs: set[str] = set()
    i = 0
    in_line_comment = False
    in_block_comment = False
    while i < len(mlir_text):
        char = mlir_text[i]
        nxt = mlir_text[i + 1] if i + 1 < len(mlir_text) else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if char == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if char == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if char == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if char == '"':
            i = _quoted_span_end(mlir_text, i)
            continue
        if char != "@":
            i += 1
            continue
        if nxt == '"':
            quoted_end = _quoted_span_end(mlir_text, i + 1)
            if quoted_end <= i + 2:
                i += 1
                continue
            refs.add(_normalize_symbol_ref(mlir_text[i:quoted_end]))
            i = quoted_end
            continue
        end = _read_symbol_ref_after_at(mlir_text, i)
        if end is None:
            i += 1
            continue
        refs.add(_normalize_symbol_ref(mlir_text[i:end]))
        i = end
    return frozenset(refs)


def _mlir_generic_sym_names(mlir_text: str) -> frozenset[str]:
    names: set[str] = set()
    structural = _comment_stripped_text(mlir_text)
    i = 0
    while True:
        found = structural.find("sym_name", i)
        if found == -1:
            break
        before = structural[found - 1] if found > 0 else ""
        after_i = found + len("sym_name")
        after = structural[after_i] if after_i < len(structural) else ""
        if _is_word_char(before) or _is_word_char(after):
            i = found + 1
            continue
        eq_index = _skip_spaces(structural, after_i)
        if eq_index >= len(structural) or structural[eq_index] != "=":
            i = after_i
            continue
        value_index = _skip_spaces(structural, eq_index + 1)
        if value_index < len(structural) and structural[value_index] == '"':
            quoted_end = _quoted_span_end(structural, value_index)
            if quoted_end > value_index + 2:
                names.add(_normalize_symbol_ref(
                    '@' + structural[value_index:quoted_end]))
            i = quoted_end
            continue
        i = after_i
    return frozenset(names)


def _mlir_symbol_fingerprint(mlir_text: str) -> tuple[str, ...]:
    refs: list[tuple[int, str]] = []
    structural = _comment_stripped_text(mlir_text)
    i = 0
    while i < len(structural):
        if structural[i] == '"':
            i = _quoted_span_end(structural, i)
            continue
        if structural[i] == "@":
            if i + 1 < len(structural) and structural[i + 1] == '"':
                quoted_end = _quoted_span_end(structural, i + 1)
                if quoted_end > i + 2:
                    refs.append((
                        i, _normalize_symbol_ref(
                            structural[i:quoted_end])))
                    i = quoted_end
                    continue
            end = _read_symbol_ref_after_at(structural, i)
            if end is not None:
                refs.append((i, _normalize_symbol_ref(structural[i:end])))
                i = end
                continue
        if _bare_word_at(structural, i, "sym_name"):
            eq_index = _skip_spaces(structural, i + len("sym_name"))
            if eq_index < len(structural) and structural[eq_index] == "=":
                value_index = _skip_spaces(structural, eq_index + 1)
                if value_index < len(structural) \
                        and structural[value_index] == '"':
                    quoted_end = _quoted_span_end(structural, value_index)
                    if quoted_end > value_index + 2:
                        refs.append((
                            i, _normalize_symbol_ref(
                                "@" + structural[value_index:quoted_end])))
                        i = quoted_end
                        continue
        i += 1
    return tuple(symbol for _index, symbol in refs)


def _comment_stripped_text(mlir_text: str) -> str:
    out: list[str] = []
    i = 0
    in_line_comment = False
    in_block_comment = False
    quoted = False
    escaped = False
    while i < len(mlir_text):
        char = mlir_text[i]
        nxt = mlir_text[i + 1] if i + 1 < len(mlir_text) else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                out.append(char)
            i += 1
            continue
        if in_block_comment:
            if char == "\n":
                out.append(char)
                i += 1
                continue
            if char == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if quoted:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            i += 1
            continue
        if char == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if char == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if char == '"':
            quoted = True
        out.append(char)
        i += 1
    return "".join(out)


def _normalize_symbol_ref(symbol: str) -> str:
    if not (symbol.startswith('@"') and symbol.endswith('"')):
        return symbol
    bare = _decode_quoted_symbol_body(symbol[2:-1])
    if not bare or not (bare[0].isalnum() or bare[0] in "_$"):
        return '@"' + _encode_quoted_symbol_body(bare) + '"'
    if any(not (char.isalnum() or char in "_$.-") for char in bare):
        return '@"' + _encode_quoted_symbol_body(bare) + '"'
    return "@" + bare


def _decode_quoted_symbol_body(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 2 < len(text) \
                and all(c in "0123456789abcdefABCDEF"
                        for c in text[i + 1:i + 3]):
            out.append(chr(int(text[i + 1:i + 3], 16)))
            i += 3
            continue
        if char == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _encode_quoted_symbol_body(text: str) -> str:
    out: list[str] = []
    for char in text:
        code = ord(char)
        if char in {'"', "\\"} or code < 0x20 or code == 0x7F:
            out.append(f"\\{code:02X}")
        else:
            out.append(char)
    return "".join(out)


def _quoted_span_end(text: str, start: int) -> int:
    i = start + 1
    escaping = False
    while i < len(text):
        char = text[i]
        if escaping:
            escaping = False
            i += 1
            continue
        if char == "\\":
            escaping = True
            i += 1
            continue
        if char == '"':
            return i + 1
        i += 1
    return len(text)


def _mlir_opt_output_correspondence_findings(
        mlir_text: str, output_text: str) -> tuple[str, ...]:
    input_symbols = _mlir_symbol_refs(mlir_text) | _mlir_generic_sym_names(
        mlir_text)
    output_symbols = _mlir_symbol_refs(output_text) | _mlir_generic_sym_names(
        output_text)
    missing = tuple(sorted(input_symbols - output_symbols))
    if missing:
        return (
            "mlir-opt output artifact is missing input symbol references: "
            + ", ".join(missing[:5]),
        )
    input_symbol_fingerprint = _mlir_symbol_fingerprint(mlir_text)
    output_symbol_fingerprint = _mlir_symbol_fingerprint(output_text)
    if input_symbol_fingerprint != output_symbol_fingerprint:
        return (
            "mlir-opt output artifact does not preserve the input "
            "symbol reference structure: "
            f"{input_symbol_fingerprint} -> {output_symbol_fingerprint}",
        )
    input_interfaces = _mlir_func_interfaces(mlir_text)
    if input_interfaces:
        output_interfaces = _mlir_func_interfaces(output_text)
        missing_interfaces = tuple(
            sorted(input_interfaces - output_interfaces))
        if missing_interfaces:
            return (
                "mlir-opt output artifact is missing input function "
                "interfaces: " + ", ".join(missing_interfaces[:5]),
            )
    input_fingerprint = _mlir_structural_op_fingerprint(mlir_text)
    output_fingerprint = _mlir_structural_op_fingerprint(output_text)
    if input_fingerprint != output_fingerprint:
        return (
            "mlir-opt output artifact does not preserve the input "
            "operation structure: "
            f"{input_fingerprint} -> {output_fingerprint}",
        )
    input_props = _mlir_generic_property_fingerprint(mlir_text)
    output_props = _mlir_generic_property_fingerprint(output_text)
    if input_props != output_props \
            and not _generic_property_mismatch_is_canonicalization(
                input_props, output_props):
        return (
            "mlir-opt output artifact does not preserve the input "
            "generic operation properties: "
            f"{input_props} -> {output_props}",
        )
    input_attr_dicts = _mlir_normal_attr_dict_fingerprint(mlir_text)
    output_attr_dicts = _mlir_normal_attr_dict_fingerprint(output_text)
    if input_attr_dicts != output_attr_dicts:
        return (
            "mlir-opt output artifact does not preserve the input "
            "attribute dictionaries: "
            f"{input_attr_dicts} -> {output_attr_dicts}",
        )
    input_values = _mlir_literal_value_fingerprint(mlir_text)
    output_values = _mlir_literal_value_fingerprint(output_text)
    if input_values != output_values:
        return (
            "mlir-opt output artifact does not preserve the input "
            "literal/attribute values: "
            f"{input_values} -> {output_values}",
        )
    input_ssa = _mlir_ssa_value_fingerprint(mlir_text)
    output_ssa = _mlir_ssa_value_fingerprint(output_text)
    if input_ssa != output_ssa:
        return (
            "mlir-opt output artifact does not preserve the input "
            "SSA value references: "
            f"{input_ssa} -> {output_ssa}",
        )
    input_blocks = _mlir_block_label_fingerprint(mlir_text)
    output_blocks = _mlir_block_label_fingerprint(output_text)
    if input_blocks != output_blocks:
        return (
            "mlir-opt output artifact does not preserve the input "
            "block label references: "
            f"{input_blocks} -> {output_blocks}",
        )
    input_types = _mlir_type_atom_fingerprint(mlir_text)
    output_types = _mlir_type_atom_fingerprint(output_text)
    if input_types != output_types:
        return (
            "mlir-opt output artifact does not preserve the input "
            "type annotations: "
            f"{input_types} -> {output_types}",
        )
    input_attrs = _mlir_dialect_attribute_fingerprint(mlir_text)
    output_attrs = _mlir_dialect_attribute_fingerprint(output_text)
    if input_attrs != output_attrs:
        return (
            "mlir-opt output artifact does not preserve the input "
            "attribute/property payloads: "
            f"{input_attrs} -> {output_attrs}",
        )
    return ()


def _mlir_structural_op_fingerprint(mlir_text: str) -> tuple[str, ...]:
    structural = _comment_stripped_text(mlir_text)
    names: list[str] = []
    i = 0
    while i < len(structural):
        if structural[i] == '"':
            quoted_end = _quoted_span_end(structural, i)
            if quoted_end <= i + 1:
                break
            quoted_text = structural[i + 1:quoted_end - 1]
            j = _skip_spaces(structural, quoted_end)
            if (j < len(structural) and structural[j] == "("
                    and _mlir_op_start_context_allows(structural, i)):
                if "\\" in quoted_text:
                    names.append(_GENERIC_OP_SOURCE_TOKEN)
                elif quoted_text in ("builtin.module", "module"):
                    names.append("module")
                elif quoted_text == "func.func":
                    names.append("func.func")
                elif "." in quoted_text:
                    names.append(quoted_text)
                else:
                    names.append(_GENERIC_OP_SOURCE_TOKEN)
            i = quoted_end
            continue
        if (_bare_word_at(structural, i, "func.func")
                and _mlir_op_start_context_allows(structural, i)
                and _mlir_op_token_tail_allows(structural, i + len(
                    "func.func"))):
            names.append("func.func")
            i += len("func.func")
            continue
        if (_bare_word_at(structural, i, "builtin.module")
                and _mlir_op_start_context_allows(structural, i)
                and _mlir_op_token_tail_allows(structural, i + len(
                    "builtin.module"))):
            names.append("module")
            i += len("builtin.module")
            continue
        if (_bare_word_at(structural, i, "module")
                and _mlir_op_start_context_allows(structural, i)
                and _mlir_op_token_tail_allows(structural, i + len(
                    "module"))):
            names.append("module")
            i += len("module")
            continue
        op_name, op_end = _read_bare_op_name(structural, i)
        if op_name and _mlir_op_start_context_allows(
                structural, i) and _mlir_op_token_tail_allows(
                    structural, op_end):
            if op_name in ("builtin.module", "module"):
                names.append("module")
            elif op_name == "func.func":
                names.append("func.func")
            else:
                names.append(op_name)
            i = op_end
            continue
        i += 1
    return tuple(names)


def _mlir_literal_value_fingerprint(mlir_text: str) -> tuple[str, ...]:
    text = _mlir_text_without_locations(_comment_stripped_text(mlir_text))
    replacements = _mlir_literal_payload_replacements(text)
    replacement_index = 0
    values: list[str] = []
    i = 0
    while i < len(text):
        while replacement_index < len(replacements) \
                and replacements[replacement_index][1] <= i:
            replacement_index += 1
        if replacement_index < len(replacements):
            start, end, replacement_values = replacements[replacement_index]
            if start <= i < end:
                values.extend(replacement_values)
                i = end
                continue
        char = text[i]
        if char == '"':
            quoted_end = _quoted_span_end(text, i)
            if quoted_end <= i + 1:
                break
            j = _skip_spaces(text, quoted_end)
            if not (j < len(text) and text[j] == "("
                    and _mlir_op_start_context_allows(text, i)) \
                    and not _mlir_quoted_span_is_symbol_metadata(
                        text, i, quoted_end):
                values.append("str:" + text[i:quoted_end])
            i = quoted_end
            continue
        number_end = _mlir_numeric_literal_end(text, i)
        if number_end is not None:
            values.append("num:" + text[i:number_end].lower())
            i = number_end
            continue
        word, word_end = _read_bare_word(text, i)
        if word in ("false", "true"):
            values.append("bool:" + word)
            i = word_end
            continue
        i += 1
    return tuple(values)


def _mlir_literal_payload_replacements(
        text: str) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    replacements: list[tuple[int, int, tuple[str, ...]]] = []
    i = 0
    while i < len(text):
        if text.startswith("<{", i):
            end = _matching_closer_index(text, i + 1, "{", "}")
            if end is None:
                i += 2
                continue
            after = _skip_spaces(text, end + 1)
            if after < len(text) and text[after] == ">":
                replacements.append((
                    i + 2, end,
                    _mlir_literal_values_in_dictionary_content(
                        text[i + 2:end])))
                i = after + 1
                continue
        if text[i] != "{":
            i += 1
            continue
        end = _matching_closer_index(text, i, "{", "}")
        if end is None:
            i += 1
            continue
        content = text[i + 1:end].strip()
        if (_normal_attr_dict_content_is_plausible(content)
                and _normal_attr_dict_context_allows(text, i, end)):
            replacements.append((
                i + 1, end,
                _mlir_literal_values_in_dictionary_content(content)))
            i = end + 1
            continue
        i += 1
    return tuple(replacements)


def _mlir_literal_values_in_dictionary_content(content: str) -> tuple[str, ...]:
    parts = _split_depth_zero_commas(content)
    if parts is None:
        return ()
    values: list[str] = []
    for part in parts:
        eq_index = _depth_zero_equals_index(part, 0)
        key = part[:eq_index].strip() if eq_index is not None else ""
        if key in {"sym_name", "sym_visibility"}:
            continue
        payload = part[eq_index + 1:] if eq_index is not None else part
        values.extend(_mlir_literal_values_in_text(payload))
    return tuple(sorted(values))


def _mlir_literal_values_in_text(text: str) -> tuple[str, ...]:
    values: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == '"':
            quoted_end = _quoted_span_end(text, i)
            if quoted_end <= i + 1:
                break
            values.append("str:" + text[i:quoted_end])
            i = quoted_end
            continue
        number_end = _mlir_numeric_literal_end(text, i)
        if number_end is not None:
            values.append("num:" + text[i:number_end].lower())
            i = number_end
            continue
        word, word_end = _read_bare_word(text, i)
        if word in ("false", "true"):
            values.append("bool:" + word)
            i = word_end
            continue
        i += 1
    return tuple(values)


def _mlir_ssa_value_fingerprint(mlir_text: str) -> tuple[str, ...]:
    return _normalized_reference_fingerprint(_mlir_raw_ssa_values(mlir_text),
                                             "%v")


def _mlir_raw_ssa_values(mlir_text: str) -> tuple[str, ...]:
    text = _comment_stripped_text(mlir_text)
    values: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == '"':
            i = _quoted_span_end(text, i)
            continue
        if text[i] != "%":
            i += 1
            continue
        end = i + 1
        while end < len(text) and (
                text[end].isalnum() or text[end] in "_$.#"):
            end += 1
        if end > i + 1:
            values.append(text[i:end])
        i = max(end, i + 1)
    return tuple(values)


def _mlir_block_label_fingerprint(mlir_text: str) -> tuple[str, ...]:
    return _normalized_reference_fingerprint(
        _mlir_raw_block_labels(mlir_text), "^b")


def _mlir_raw_block_labels(mlir_text: str) -> tuple[str, ...]:
    text = _comment_stripped_text(mlir_text)
    labels: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == '"':
            i = _quoted_span_end(text, i)
            continue
        if text[i] != "^":
            i += 1
            continue
        end = i + 1
        while end < len(text) and (
                text[end].isalnum() or text[end] in "_$.-"):
            end += 1
        if end > i + 1:
            labels.append(text[i:end])
        i = max(end, i + 1)
    return tuple(labels)


def _normalized_reference_fingerprint(
        refs: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    mapping: dict[str, str] = {}
    normalized: list[str] = []
    for ref in refs:
        mapped = mapping.get(ref)
        if mapped is None:
            mapped = f"{prefix}{len(mapping)}"
            mapping[ref] = mapped
        normalized.append(mapped)
    return tuple(normalized)


def _mlir_type_atom_fingerprint(mlir_text: str) -> tuple[str, ...]:
    text = _mlir_text_without_locations(_comment_stripped_text(mlir_text))
    atoms: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != ":" or _mlir_colon_is_inside_generic_props(text, i):
            i += 1
            continue
        end = _next_op_boundary(text, i + 1, len(text))
        atoms.extend(_mlir_type_atoms_in_text(text[i + 1:end]))
        i = end
    return tuple(atoms)


def _mlir_type_atoms_in_text(text: str) -> tuple[str, ...]:
    atoms: list[str] = []
    i = 0
    while i < len(text):
        if not (text[i].isalpha() or text[i] in "!_"):
            i += 1
            continue
        start = i
        while i < len(text) and (
                text[i].isalnum() or text[i] in "_$.!"):
            i += 1
        atom = text[start:i]
        if _mlir_type_atom_is_plausible(atom):
            atoms.append(atom)
    return tuple(atoms)


def _mlir_colon_is_inside_generic_props(text: str, index: int) -> bool:
    prop_depth = 0
    i = 0
    while i < index:
        if text[i] == '"':
            quoted_end = _quoted_span_end(text, i)
            if quoted_end <= i + 1 or quoted_end > index:
                return False
            i = quoted_end
            continue
        if text.startswith("<{", i):
            prop_depth += 1
            i += 2
            continue
        if prop_depth and text.startswith("}>", i):
            prop_depth -= 1
            i += 2
            continue
        i += 1
    return prop_depth > 0


def _mlir_type_atom_is_plausible(atom: str) -> bool:
    if atom in {
            "bf16", "f16", "f32", "f64", "i1", "index", "none",
            "opaque", "tensor", "tuple"}:
        return True
    if atom.startswith("i") and atom[1:].isdigit():
        return True
    if atom.startswith(("!llvm", "!spirv", "!gpu", "!memref")):
        return True
    if atom in {"memref", "tensor", "vector"}:
        return True
    return False


def _mlir_dialect_attribute_fingerprint(mlir_text: str) -> tuple[str, ...]:
    text = _mlir_text_without_locations(_comment_stripped_text(mlir_text))
    payloads: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("<{", i):
            end = _matching_closer_index(text, i + 1, "{", "}")
            if end is not None:
                after = _skip_spaces(text, end + 1)
                if after < len(text) and text[after] == ">":
                    i = after + 1
                    continue
        word, word_end = _read_bare_word(text, i)
        if word and word_end < len(text) and text[word_end] == "<":
            close = _matching_angle_closer_index(text, word_end)
            if close is not None:
                payloads.append(
                    "attr:" + _normalized_mlir_fragment(text[i:close + 1]))
                i = close + 1
                continue
        if text[i] == "#":
            end = _mlir_attribute_alias_end(text, i)
            if end is not None:
                payloads.append(
                    "attr:" + _normalized_mlir_fragment(text[i:end]))
                i = end
                continue
        i += 1
    return tuple(payloads)


def _mlir_generic_property_fingerprint(mlir_text: str) -> tuple[str, ...]:
    text = _mlir_text_without_locations(_comment_stripped_text(mlir_text))
    payloads: list[str] = []
    i = 0
    while i < len(text):
        if not text.startswith("<{", i):
            i += 1
            continue
        end = _matching_closer_index(text, i + 1, "{", "}")
        if end is None:
            i += 2
            continue
        after = _skip_spaces(text, end + 1)
        if after >= len(text) or text[after] != ">":
            i += 2
            continue
        op_name = _generic_op_name_before_property_dict(text, i)
        parts = _split_depth_zero_commas(text[i + 2:end])
        if parts is not None:
            dict_payloads: list[str] = []
            for part in parts:
                eq_index = _depth_zero_equals_index(part, 0)
                key = part[:eq_index].strip() if eq_index is not None else ""
                if key and key != "sym_name":
                    payload = _normalized_mlir_fragment(part)
                    if key in {"function_type", "sym_visibility", "value"} \
                            and op_name:
                        payload = f"{op_name}:{payload}"
                    dict_payloads.append("prop:" + payload)
            payloads.extend(sorted(dict_payloads))
        i = after + 1
    return tuple(payloads)


def _generic_op_name_before_property_dict(
        text: str, props_start: int) -> str:
    j = props_start - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    if j < 0 or text[j] != ")":
        return ""
    open_index = _matching_opener_index(text, j, "(", ")")
    if open_index is None:
        return ""
    quote_end = open_index - 1
    while quote_end >= 0 and text[quote_end].isspace():
        quote_end -= 1
    if quote_end < 0 or text[quote_end] != '"':
        return ""
    quote_start = quote_end
    while quote_start >= 0:
        if text[quote_start] == '"' \
                and _quoted_span_end(text, quote_start) == quote_end + 1:
            break
        quote_start -= 1
    if quote_start < 0:
        return ""
    if not _mlir_op_start_context_allows(text, quote_start):
        return ""
    return text[quote_start + 1:quote_end]


def _matching_opener_index(text: str, close_index: int,
                           opener: str, closer: str) -> int | None:
    depth = 0
    i = close_index
    while i >= 0:
        if text[i] == '"':
            quote_start = i
            while quote_start >= 0:
                if text[quote_start] == '"' \
                        and _quoted_span_end(text, quote_start) == i + 1:
                    i = quote_start - 1
                    break
                quote_start -= 1
            else:
                return None
            continue
        char = text[i]
        if char == closer:
            depth += 1
        elif char == opener:
            depth -= 1
            if depth == 0:
                return i
        i -= 1
    return None


def _generic_property_mismatch_is_canonicalization(
        input_props: tuple[str, ...],
        output_props: tuple[str, ...]) -> bool:
    if not input_props or output_props:
        return False
    return all(_generic_property_is_canonicalizable(prop)
               for prop in input_props)


def _generic_property_is_canonicalizable(prop: str) -> bool:
    if not prop.startswith("prop:"):
        return False
    payload = prop[5:]
    if payload.startswith("func.func:"):
        key = payload[len("func.func:"):].split("=", 1)[0].strip()
        return key in {"function_type", "sym_visibility"}
    if payload.startswith("arith.constant:"):
        raw = payload[len("arith.constant:"):]
        key = raw.split("=", 1)[0].strip()
        return key == "value" and _generic_constant_value_is_canonicalizable(
            raw)
    return False


def _generic_constant_value_is_canonicalizable(prop_payload: str) -> bool:
    eq_index = _depth_zero_equals_index(prop_payload, 0)
    if eq_index is None:
        return False
    value = prop_payload[eq_index + 1:].strip()
    if value.startswith(("true", "false")):
        end = len("true") if value.startswith("true") else len("false")
        return end == len(value) or value[end].isspace() \
            or value[end] in ",:}>)]"
    if value.startswith('"'):
        return _quoted_span_end(value, 0) > 1
    return _mlir_numeric_literal_end(value, 0) is not None


def _mlir_normal_attr_dict_fingerprint(mlir_text: str) -> tuple[str, ...]:
    text = _mlir_text_without_locations(_comment_stripped_text(mlir_text))
    payloads: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("<{", i):
            end = _matching_closer_index(text, i + 1, "{", "}")
            if end is None:
                i += 2
                continue
            after = _skip_spaces(text, end + 1)
            if after < len(text) and text[after] == ">":
                i = after + 1
                continue
        if text[i] != "{":
            i += 1
            continue
        end = _matching_closer_index(text, i, "{", "}")
        if end is None:
            i += 1
            continue
        content = text[i + 1:end].strip()
        if (_normal_attr_dict_content_is_plausible(content)
                and _normal_attr_dict_context_allows(text, i, end)):
            payloads.append(
                "attrs:" + _normalized_mlir_dictionary_content(content))
            i = end + 1
            continue
        i += 1
    return tuple(payloads)


def _normal_attr_dict_content_is_plausible(content: str) -> bool:
    if not content:
        return False
    parts = _split_depth_zero_commas(content)
    if parts is None or not parts:
        return False
    return all(_normal_attr_item_is_plausible(part.strip())
               for part in parts)


def _normalized_mlir_dictionary_content(content: str) -> str:
    parts = _split_depth_zero_commas(content)
    if parts is None:
        return _normalized_mlir_fragment(content)
    return ", ".join(sorted(
        _normalized_mlir_fragment(part.strip()) for part in parts))


def _normal_attr_dict_context_allows(
        text: str, start: int, end: int) -> bool:
    before = start - 1
    while before >= 0 and text[before].isspace():
        before -= 1
    if _normal_attr_dict_preceded_by_attributes_keyword(text, before):
        return True
    after = _skip_spaces(text, end + 1)
    if after >= len(text) or text[after] != ":":
        return False
    return before >= 0 and (
        text[before] in ")]}>"
        or _normal_attr_dict_preceded_by_operation_text(text, start))


def _normal_attr_dict_preceded_by_attributes_keyword(
        text: str, before: int) -> bool:
    if before < 0:
        return False
    word_end = before + 1
    word_start = before
    while word_start >= 0 and (
            text[word_start].isalnum() or text[word_start] in "_."):
        word_start -= 1
    return text[word_start + 1:word_end] == "attributes"


def _normal_attr_dict_preceded_by_operation_text(
        text: str, start: int) -> bool:
    segment_start = max(
        text.rfind("\n", 0, start),
        text.rfind("{", 0, start),
        text.rfind("}", 0, start),
    ) + 1
    segment = text[segment_start:start]
    i = 0
    while i < len(segment):
        if segment[i] == '"':
            quoted_end = _quoted_span_end(segment, i)
            if quoted_end <= i + 1:
                break
            j = _skip_spaces(segment, quoted_end)
            if j < len(segment) and segment[j] == "(":
                return True
            i = quoted_end
            continue
        word, word_end = _read_bare_op_name(segment, i)
        if word and "." in word:
            return True
        i = max(word_end, i + 1)
    return False


def _normal_attr_item_is_plausible(item: str) -> bool:
    if not item:
        return False
    eq_index = _depth_zero_equals_index(item, 0)
    if eq_index is not None:
        return _property_assignment_is_plausible(item)
    if any(char.isspace() for char in item):
        return False
    if not (item[0].isalpha() or item[0] == "_"):
        return False
    return all(char.isalnum() or char in "_$.-" for char in item)


def _mlir_attribute_alias_end(text: str, start: int) -> int | None:
    i = start + 1
    if i >= len(text) or not (text[i].isalpha() or text[i] == "_"):
        return None
    i += 1
    while i < len(text) and (
            text[i].isalnum() or text[i] in "_$."):
        i += 1
    if i < len(text) and text[i] == "<":
        close = _matching_angle_closer_index(text, i)
        if close is None:
            return None
        return close + 1
    return i


def _matching_angle_closer_index(text: str, start: int) -> int | None:
    depth = 0
    i = start
    while i < len(text):
        char = text[i]
        if char == '"':
            i = _quoted_span_end(text, i)
            continue
        if char == "<" and i + 1 < len(text) and text[i + 1] == "=":
            i += 2
            continue
        if (char == ">" and
                ((i > 0 and text[i - 1] in "-=")
                 or (i + 1 < len(text) and text[i + 1] == "="))):
            i += 1
            continue
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _mlir_quoted_span_is_symbol_metadata(
        text: str, start: int, quoted_end: int) -> bool:
    prev = start - 1
    while prev >= 0 and text[prev].isspace():
        prev -= 1
    if prev >= 0 and text[prev] == "@":
        return True
    if prev < 0 or text[prev] != "=":
        return False
    key_end = prev - 1
    while key_end >= 0 and text[key_end].isspace():
        key_end -= 1
    key_start = key_end
    while key_start >= 0 and (
            text[key_start].isalnum() or text[key_start] in "_."):
        key_start -= 1
    key = text[key_start + 1:key_end + 1]
    if key not in {"sym_name", "sym_visibility"}:
        return False
    after = _skip_spaces(text, quoted_end)
    return after >= len(text) or text[after] in ",}>)]"


def _mlir_text_without_locations(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == '"':
            quoted_end = _quoted_span_end(text, i)
            if quoted_end <= i + 1:
                out.append(text[i])
                i += 1
                continue
            out.append(text[i:quoted_end])
            i = quoted_end
            continue
        if _bare_word_at(text, i, "loc"):
            after = _skip_spaces(text, i + len("loc"))
            if after < len(text) and text[after] == "(":
                close = _matching_closer_index(text, after, "(", ")")
                if close is not None:
                    i = close + 1
                    continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _mlir_numeric_literal_end(text: str, start: int) -> int | None:
    char = text[start]
    if char in "+-":
        if start + 1 >= len(text) or not text[start + 1].isdigit():
            return None
        if start > 0 and (
                _is_word_char(text[start - 1])
                or text[start - 1] in "%^@"):
            return None
        i = start + 1
    elif char.isdigit():
        if start > 0 and (
                _is_word_char(text[start - 1])
                or text[start - 1] in "%^@"):
            return None
        i = start
    else:
        return None
    while i < len(text) and (
            text[i].isalnum() or text[i] in "._+-"):
        i += 1
    if i < len(text) and _is_word_char(text[i]):
        return None
    token = text[start:i]
    if not any(ch.isdigit() for ch in token):
        return None
    return i


def _mlir_op_start_context_allows(text: str, start: int) -> bool:
    prev = start - 1
    while prev >= 0 and text[prev].isspace():
        prev -= 1
    if prev < 0 or text[prev] in "{)}":
        return True
    if text[prev] == "=":
        return _mlir_preceding_equals_binds_ssa(text, prev)
    return False


def _mlir_preceding_equals_binds_ssa(text: str, eq_index: int) -> bool:
    segment_start = max(
        text.rfind("\n", 0, eq_index),
        text.rfind("{", 0, eq_index),
        text.rfind("}", 0, eq_index),
    ) + 1
    lhs = text[segment_start:eq_index].strip()
    return _ssa_result_list_is_plausible(lhs)


def _mlir_op_token_tail_allows(text: str, token_end: int) -> bool:
    j = _skip_spaces(text, token_end)
    return j >= len(text) or text[j] != "="


def _mlir_func_interfaces(mlir_text: str) -> frozenset[str]:
    structural = _comment_stripped_text(mlir_text)
    interfaces: set[str] = set()
    i = 0
    while i < len(structural):
        if structural[i] == '"':
            i = _quoted_span_end(structural, i)
            continue
        if not _bare_word_at(structural, i, "func.func"):
            i += 1
            continue
        interface = _mlir_func_interface_at(structural, i)
        if interface is not None:
            interfaces.add(interface)
        i += len("func.func")
    interfaces.update(_mlir_generic_func_interfaces(mlir_text))
    return frozenset(interfaces)


def _mlir_generic_func_interfaces(mlir_text: str) -> frozenset[str]:
    structural = _comment_stripped_text(mlir_text)
    interfaces: set[str] = set()
    i = 0
    while i < len(structural):
        found = structural.find('"func.func"', i)
        if found == -1:
            break
        if not _mlir_op_start_context_allows(structural, found):
            i = found + 1
            continue
        quoted_end = found + len('"func.func"')
        j = _skip_spaces(structural, quoted_end)
        if j >= len(structural) or structural[j] != "(":
            i = quoted_end
            continue
        operands_end = _matching_closer_index(structural, j, "(", ")")
        if operands_end is None:
            i = quoted_end
            continue
        props = _generic_property_dict_after(structural, operands_end + 1)
        if props is None:
            i = operands_end + 1
            continue
        props_text, props_end = props
        interface = _mlir_generic_func_interface_from_props(
            structural, props_text, props_end)
        if interface is not None:
            interfaces.add(interface)
        i = props_end
    return frozenset(interfaces)


def _mlir_generic_func_body_spans(
        structural: str) -> tuple[tuple[str, int, int], ...]:
    """Yield `(symbol, body_start, body_end)` triples for each
    generic-form `"func.func"` op with a non-empty region body. Used by
    the strict-static terminator check so generic-form bypasses the
    bare-form walker can't sneak past."""
    spans: list[tuple[str, int, int]] = []
    i = 0
    while i < len(structural):
        found = structural.find('"func.func"', i)
        if found == -1:
            break
        if not _mlir_op_start_context_allows(structural, found):
            i = found + 1
            continue
        quoted_end = found + len('"func.func"')
        j = _skip_spaces(structural, quoted_end)
        if j >= len(structural) or structural[j] != "(":
            i = quoted_end
            continue
        operands_end = _matching_closer_index(structural, j, "(", ")")
        if operands_end is None:
            i = quoted_end
            continue
        props = _generic_property_dict_after(structural, operands_end + 1)
        if props is None:
            i = operands_end + 1
            continue
        props_text, props_end = props
        props_map = _generic_property_assignments(props_text)
        sym_name = props_map.get("sym_name")
        if sym_name is None:
            i = props_end
            continue
        symbol = _symbol_from_generic_string_property(sym_name)
        if symbol is None:
            i = props_end
            continue
        span = _generic_func_body_span(structural, props_end)
        if span is not None:
            spans.append((symbol, span[0], span[1]))
        i = props_end
    return tuple(spans)


def _mlir_generic_func_body_spans_with_result(
        structural: str) -> tuple[tuple[str, int, int, str], ...]:
    """Same as `_mlir_generic_func_body_spans` but also returns the
    declared result-type text for the empty-return strict check."""
    out: list[tuple[str, int, int, str]] = []
    i = 0
    while i < len(structural):
        found = structural.find('"func.func"', i)
        if found == -1:
            break
        if not _mlir_op_start_context_allows(structural, found):
            i = found + 1
            continue
        quoted_end = found + len('"func.func"')
        j = _skip_spaces(structural, quoted_end)
        if j >= len(structural) or structural[j] != "(":
            i = quoted_end
            continue
        operands_end = _matching_closer_index(structural, j, "(", ")")
        if operands_end is None:
            i = quoted_end
            continue
        props = _generic_property_dict_after(structural, operands_end + 1)
        if props is None:
            i = operands_end + 1
            continue
        props_text, props_end = props
        props_map = _generic_property_assignments(props_text)
        sym_name = props_map.get("sym_name")
        function_type = props_map.get("function_type")
        if sym_name is None or function_type is None:
            i = props_end
            continue
        symbol = _symbol_from_generic_string_property(sym_name)
        if symbol is None:
            i = props_end
            continue
        signature = _generic_function_type_signature(function_type)
        if signature is None:
            i = props_end
            continue
        _args, result = signature
        span = _generic_func_body_span(structural, props_end)
        if span is not None:
            out.append((symbol, span[0], span[1], result))
        i = props_end
    return tuple(out)


def _generic_func_body_span(
        structural: str, props_end: int) -> tuple[int, int] | None:
    """Find the `({ body })` region operand of a generic `"func.func"`,
    returning `(body_start, body_end)` (the indices of the inner braces)
    or None if the region is empty / absent."""
    cursor = _skip_spaces(structural, props_end)
    if cursor >= len(structural) or structural[cursor] != "(":
        return None
    region_end = _matching_closer_index(structural, cursor, "(", ")")
    if region_end is None:
        return None
    inner_start = _skip_spaces(structural, cursor + 1)
    if inner_start >= region_end or structural[inner_start] != "{":
        return None
    inner_end = _matching_closer_index(structural, inner_start, "{", "}")
    if inner_end is None or inner_end > region_end:
        return None
    if not structural[inner_start + 1:inner_end].strip():
        return None
    return inner_start, inner_end


def _generic_property_dict_after(
        structural: str, start: int) -> tuple[str, int] | None:
    i = _skip_spaces(structural, start)
    if not structural.startswith("<{", i):
        return None
    close = _matching_closer_index(structural, i + 1, "{", "}")
    if close is None:
        return None
    after = _skip_spaces(structural, close + 1)
    if after >= len(structural) or structural[after] != ">":
        return None
    return structural[i + 2:close], after + 1


def _mlir_generic_func_interface_from_props(
        structural: str, props_text: str, props_end: int) -> str | None:
    props = _generic_property_assignments(props_text)
    sym_name = props.get("sym_name")
    function_type = props.get("function_type")
    if sym_name is None or function_type is None:
        return None
    symbol = _symbol_from_generic_string_property(sym_name)
    if symbol is None:
        return None
    signature = _generic_function_type_signature(function_type)
    if signature is None:
        return None
    visibility = ""
    if "sym_visibility" in props:
        visibility_value = _unquoted_string_property(props["sym_visibility"])
        if visibility_value in {"private", "nested"}:
            visibility = visibility_value
    has_body = _generic_func_op_has_region(structural, props_end)
    return _mlir_func_interface_record(
        symbol, visibility, signature[0], signature[1],
        "body" if has_body else "decl")


def _generic_property_assignments(props_text: str) -> dict[str, str]:
    props: dict[str, str] = {}
    parts = _split_depth_zero_commas(props_text)
    if parts is None:
        return props
    for part in parts:
        eq_index = _depth_zero_equals_index(part, 0)
        if eq_index is None:
            continue
        key = part[:eq_index].strip()
        value = part[eq_index + 1:].strip()
        if key:
            props[key] = value
    return props


def _symbol_from_generic_string_property(value: str) -> str | None:
    unquoted = _unquoted_string_property(value)
    if unquoted is None:
        return None
    return _normalize_symbol_ref('@"' + unquoted + '"')


def _unquoted_string_property(value: str) -> str | None:
    stripped = value.strip()
    if not stripped.startswith('"') or not stripped.endswith('"'):
        return None
    quoted_end = _quoted_span_end(stripped, 0)
    if quoted_end != len(stripped):
        return None
    return stripped[1:-1]


def _generic_function_type_signature(
        function_type: str) -> tuple[str, str] | None:
    arrow = _depth_zero_arrow_index(function_type, 0, len(function_type))
    if arrow is None:
        return None
    args = _mlir_func_signature_arg_types(function_type[:arrow])
    result = _mlir_func_signature_result_types(function_type[arrow + 2:])
    return args, result


def _depth_zero_arrow_index(
        text: str, start: int, end: int) -> int | None:
    pairs = {"(": ")", "{": "}", "[": "]", "<": ">"}
    closers = set(pairs.values())
    stack: list[str] = []
    i = start
    while i + 1 < end:
        char = text[i]
        if char in pairs:
            stack.append(pairs[char])
            i += 1
            continue
        if char in closers:
            if stack and stack[-1] == char:
                stack.pop()
            elif not stack:
                return None
            i += 1
            continue
        if not stack and char == "-" and text[i + 1] == ">":
            return i
        i += 1
    return None


def _generic_func_op_has_region(structural: str, props_end: int) -> bool:
    i = _skip_spaces(structural, props_end)
    if i >= len(structural) or structural[i] != "(":
        return False
    close = _matching_closer_index(structural, i, "(", ")")
    if close is None:
        return False
    return bool(structural[i + 1:close].strip())


def _mlir_symbol_ref_at(
        structural: str, start: int,
        line_end: int) -> tuple[str, int] | None:
    if start >= line_end or structural[start] != "@":
        return None
    if start + 1 < line_end and structural[start + 1] == '"':
        quoted_end = _quoted_span_end(structural, start + 1)
        if quoted_end <= start + 2 or quoted_end > line_end:
            return None
        return structural[start:quoted_end], quoted_end
    symbol_end = _read_symbol_ref_after_at(structural, start)
    if symbol_end is None or symbol_end > line_end:
        return None
    return structural[start:symbol_end], symbol_end


def _mlir_func_interface_at(structural: str, token_start: int) -> str | None:
    token_end = token_start + len("func.func")
    line_end = _next_op_boundary(structural, token_start, len(structural))
    j = _skip_spaces(structural, token_end)
    visibility = ""
    word, word_end = _read_bare_word(structural, j)
    if word in ("private", "public", "nested"):
        visibility = word
        j = _skip_spaces(structural, word_end)
    symbol_ref = _mlir_symbol_ref_at(structural, j, line_end)
    if symbol_ref is None:
        return None
    symbol, symbol_end = symbol_ref
    symbol = _normalize_symbol_ref(symbol)
    if not symbol.startswith("@"):
        return None
    j = _skip_spaces(structural, symbol_end)
    if j >= line_end or structural[j] != "(":
        return None
    args_end = _matching_closer_index(structural, j, "(", ")")
    if args_end is None or args_end > line_end:
        return None
    args = _mlir_func_signature_arg_types(structural[j:args_end + 1])
    j = _skip_spaces(structural, args_end + 1)
    result = ""
    if structural.startswith("->", j):
        result_start = _skip_spaces(structural, j + 2)
        section, section_index = _next_func_section(
            structural, result_start, line_end)
        if section == "line_end" and section_index < line_end:
            line_end = section_index
        result = _mlir_func_signature_result_types(
            structural[result_start:section_index])
        j = section_index
    has_body = _func_header_has_body(structural, j, line_end)
    return _mlir_func_interface_record(
        symbol, visibility, args, result, "body" if has_body else "decl")


def _mlir_func_interface_record(
        symbol: str, visibility: str, args: str,
        result: str, body_kind: str) -> str:
    visibility = "" if visibility == "public" else visibility
    return "|".join(
        _mlir_func_interface_escape(field)
        for field in (symbol, visibility, args, result, body_kind))


def _mlir_func_interface_fields(
        interface: str) -> tuple[str, str, str, str, str] | None:
    fields: list[str] = []
    current: list[str] = []
    escaping = False
    for char in interface:
        if escaping:
            current.append(char)
            escaping = False
            continue
        if char == "\\":
            escaping = True
            continue
        if char == "|":
            fields.append("".join(current))
            current = []
            continue
        current.append(char)
    if escaping:
        return None
    fields.append("".join(current))
    if len(fields) != 5:
        return None
    return fields[0], fields[1], fields[2], fields[3], fields[4]


def _mlir_func_interface_escape(field: str) -> str:
    return field.replace("\\", "\\\\").replace("|", "\\|")


def _mlir_func_signature_arg_types(args_text: str) -> str:
    stripped = args_text.strip()
    if not stripped.startswith("(") or not stripped.endswith(")"):
        return _normalized_mlir_fragment(args_text)
    close = _matching_closer_index(stripped, 0, "(", ")")
    if close != len(stripped) - 1:
        return _normalized_mlir_fragment(args_text)
    inner = stripped[1:close].strip()
    if not inner:
        return "()"
    parts = _split_depth_zero_commas(inner)
    if parts is None:
        return _normalized_mlir_fragment(args_text)
    normalized_parts: list[str] = []
    for part in parts:
        item = part.strip()
        colon = _depth_zero_colon_index(item, 0)
        if item.startswith("%") and colon is not None:
            item = item[colon + 1:].strip()
        normalized_parts.append(_mlir_function_signature_item_type(item))
    return "(" + ",".join(normalized_parts) + ")"


def _mlir_func_signature_result_types(result_text: str) -> str:
    stripped = _mlir_type_tail_without_loc(
        result_text, 0, len(result_text)).strip()
    if not stripped:
        return ""
    if not stripped.startswith("("):
        return _mlir_function_signature_item_type(stripped)
    close = _matching_closer_index(stripped, 0, "(", ")")
    if close != len(stripped) - 1:
        return _mlir_function_signature_item_type(stripped)
    inner = stripped[1:close].strip()
    if not inner:
        return ""
    parts = _split_depth_zero_commas(inner)
    if parts is None:
        return _mlir_function_signature_item_type(stripped)
    normalized_parts = tuple(
        _mlir_function_signature_item_type(part.strip()) for part in parts)
    if len(normalized_parts) == 1:
        return normalized_parts[0]
    return "(" + ",".join(normalized_parts) + ")"


def _mlir_function_signature_item_type(text: str) -> str:
    stripped = text.strip()
    attr_start = _mlir_trailing_attribute_dict_start(stripped)
    if attr_start is not None:
        stripped = stripped[:attr_start].strip()
    return _normalized_mlir_fragment(stripped)


def _mlir_trailing_attribute_dict_start(text: str) -> int | None:
    stripped = text.rstrip()
    if not stripped.endswith("}"):
        return None
    opener = _matching_opener_index(stripped, len(stripped) - 1, "{", "}")
    if opener is None:
        return None
    before = stripped[:opener].rstrip()
    if not before or before.endswith((">", ")", "]", "}")):
        return None
    return opener


def _func_header_has_body(structural: str, start: int, line_end: int) -> bool:
    j = _skip_spaces(structural, start)
    while j < line_end:
        if structural[j] == "{":
            return True
        if _bare_word_at(structural, j, "attributes"):
            after_attrs = _skip_attributes(structural, j)
            if after_attrs is None or after_attrs <= j:
                return False
            j = _skip_spaces(structural, after_attrs)
            continue
        if _bare_word_at(structural, j, "loc"):
            loc_end = _skip_loc_suffix(structural, j)
            return loc_end != j and _func_header_has_body(
                structural, loc_end, line_end)
        section, section_index = _next_func_section(
            structural, j, line_end)
        if section == "body":
            return True
        if section in ("attributes", "loc") and section_index > j:
            j = section_index
            continue
        return False
    return False


def _normalized_mlir_fragment(text: str) -> str:
    return " ".join(text.split())


def _delimiter_findings(structural: str) -> tuple[str, ...]:
    """Find definite brace / parenthesis count or ordering defects.

    This is deliberately lighter than a real MLIR parser, but it must
    still fail closed on impossible delimiter order (`}{`, `)(`, or
    mixed nesting such as `{ ( } )`).
    """
    findings: list[str] = []
    angle_context_prefixes = frozenset("abcdefghijklmnopqrstuvwxyz"
                                       "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                       "0123456789_#!")
    closers = {
        "}": ("{", "brace", "braces"),
        ")": ("(", "parenthesis", "parentheses"),
        "]": ("[", "square bracket", "square brackets"),
        ">": ("<", "angle bracket", "angle brackets"),
    }
    opener_names = {
        "{": "brace",
        "(": "parenthesis",
        "[": "square bracket",
        "<": "angle bracket",
    }
    stack: list[tuple[str, int]] = []
    for offset, char in enumerate(structural):
        if char == "-" and offset + 1 < len(structural) \
                and structural[offset + 1] == ">":
            continue
        if char in opener_names:
            if char == "<" and offset + 1 < len(structural) \
                    and structural[offset + 1] == "{":
                stack.append((char, offset))
                continue
            if char == "<" and offset + 1 < len(structural) \
                    and structural[offset + 1] in "#!":
                stack.append((char, offset))
                continue
            if char == "<" and offset + 1 < len(structural) \
                    and structural[offset + 1] == "=":
                continue
            if (char == "<"
                    and (offset == 0
                         or structural[offset - 1]
                         not in angle_context_prefixes)):
                if (offset + 1 < len(structural)
                        and (structural[offset + 1].isalpha()
                             or structural[offset + 1] == "_")):
                    stack.append((char, offset))
                    continue
                continue
            stack.append((char, offset))
            continue
        if char not in closers:
            continue
        if (char == ">" and
                ((offset > 0 and structural[offset - 1] in "-=")
                 or (offset + 1 < len(structural)
                     and structural[offset + 1] == "="))):
            continue
        expected_opener, name, plural = closers[char]
        if not stack:
            findings.append(
                f"misordered {plural}: {char!r} at offset {offset} "
                f"appears before a matching {expected_opener!r}")
            continue
        actual_opener, actual_offset = stack[-1]
        if actual_opener != expected_opener:
            if not any(opener == expected_opener
                       for opener, _ in stack):
                findings.append(
                    f"misordered {plural}: {char!r} at offset {offset} "
                    f"appears before a matching {expected_opener!r}")
                continue
            actual_name = opener_names[actual_opener]
            findings.append(
                f"misnested delimiters: {char!r} at offset {offset} "
                f"closes a {name}, but the innermost opener is "
                f"{actual_opener!r} ({actual_name}) at offset "
                f"{actual_offset}")
            continue
        stack.pop()

    for opener, closer, plural in (("{", "}", "braces"),
                                   ("(", ")", "parentheses"),
                                   ("[", "]", "square brackets")):
        opens = structural.count(opener)
        closes = structural.count(closer)
        if opens != closes:
            findings.append(
                f"unbalanced {plural}: {opens} {opener!r} vs "
                f"{closes} {closer!r}")
    angle_opens = 0
    angle_closes = 0
    angle_depth = 0
    for offset, char in enumerate(structural):
        if (char == "<"
                and offset > 0
                and (structural[offset - 1] in angle_context_prefixes
                     or (offset + 1 < len(structural)
                         and structural[offset + 1] in "#!")
                     or (offset + 1 < len(structural)
                         and structural[offset + 1] == "{"))
                and not (offset + 1 < len(structural)
                         and structural[offset + 1] == "=")):
            angle_opens += 1
            angle_depth += 1
        elif (char == ">" and
              ((offset > 0 and structural[offset - 1] in "-=")
               or (offset + 1 < len(structural)
                   and structural[offset + 1] == "="))):
            continue
        elif char == ">" and angle_depth > 0:
            angle_closes += 1
            angle_depth -= 1
    if angle_opens != angle_closes:
        findings.append(
            f"unbalanced angle brackets: {angle_opens} '<' vs "
            f"{angle_closes} '>'")
    return tuple(findings)


def mock_validate_mlir(mlir_text: str) -> MLIRValidation:
    """Toolchain-free STRUCTURAL shape check on MLIR textual IR.

    Returns a frozen `MLIRValidation`: FAILED on a definite structural
    defect, otherwise DEFERRED — never PASSED, because a toolchain-free
    check cannot certify real MLIR validity (see the module docstring).

    Checks — deliberately conservative, so a clean shape never yields a
    false FAILED (the real `mlir-opt` verifier at Stage 212 catches
    what this cannot):
    - the argument is a `str` — a non-str input is itself a FAILED, not
      an exception;
    - the text is non-empty;
    - it has a top-level structure — a `module` or a `func.func`;
    - string literals are terminated;
    - braces `{}` and parentheses `()` balance and do not close before
      a matching opener.
    The structure / brace / parenthesis checks run on the text with
    string literals and `//` comments removed, so punctuation inside
    them is never miscounted. When a string literal is unterminated the
    balance checks are SKIPPED — the dangling run makes the counts
    meaningless — and the unterminated literal is reported instead, so
    the finding names the real defect, not a spurious imbalance.

    Never raises — a defect (including a non-str argument) is reported
    as a FAILED finding, the discipline of `mock_validate_ll`."""
    if not isinstance(mlir_text, str):
        return MLIRValidation(
            MLIRValidationVerdict.FAILED,
            (f"not MLIR text — expected a str, got "
             f"{type(mlir_text).__name__}",))
    problems: list[str] = []
    if not mlir_text.strip():
        problems.append("empty — no MLIR text to validate")
    elif "\x00" in mlir_text:
        problems.append("NUL byte in MLIR text")
    else:
        try:
            mlir_text.encode("utf-8")
        except UnicodeError as exc:
            problems.append(
                f"MLIR text is not encodable as UTF-8 "
                f"({type(exc).__name__}: {exc})")
            structural = ""
        else:
            structural = _structural_text(mlir_text)
        if '"' in structural:
            # `_structural_text` leaves a sentinel quote for a dangling
            # or raw-newline string literal. Its run could hold any
            # punctuation, so the delimiter checks would be unreliable:
            # report the real defect and skip those checks.
            problems.append(
                "unterminated string literal — a dangling double-quote; "
                "brace / parenthesis balance not checked")
        else:
            has_structure, top_level_findings = _top_level_stream(structural)
            problems.extend(top_level_findings)
            if not has_structure:
                problems.append(
                    "no top-level structure — neither a `module` nor a "
                    "`func.func` is present")
            problems.extend(_delimiter_findings(structural))

    if problems:
        return MLIRValidation(MLIRValidationVerdict.FAILED,
                              tuple(problems))
    return MLIRValidation(
        MLIRValidationVerdict.DEFERRED,
        ("the toolchain-free shape check found no structural defect, "
         "but real MLIR validity — verifier traits, type-correctness, "
         "SSA dominance — needs `mlir-opt`; validation is DEFERRED to "
         "`validate_mlir_with_toolchain`",))


def _run_mlir_opt_validate(
        mlir_text: str,
        mlir_opt: str,
        *,
        timeout_s: int = _MLIR_VALIDATE_TIMEOUT_S) -> MLIRValidation:
    """Run `mlir-opt` as the real MLIR verifier.

    A zero exit is necessary but not sufficient: the output artifact
    must exist, be non-empty on disk, and contain non-blank text,
    mirroring the LLVM dispatch hygiene. Tool errors are captured as
    FAILED findings, never uncaught tracebacks.
    """
    if not isinstance(mlir_opt, str) or not mlir_opt.strip():
        return MLIRValidation(
            MLIRValidationVerdict.FAILED,
            ("mlir-opt validation requested with a blank or non-str "
             f"tool path ({mlir_opt!r})",))
    mlir_opt = mlir_opt.strip()
    if not _mlir_opt_path_matches_fresh_probe(mlir_opt):
        return MLIRValidation(
            MLIRValidationVerdict.FAILED,
            ("mlir-opt validation requested with a tool path that does "
             "not match a fresh detect_mlir_support() probe; refusing "
             "to mint a real validation pass",))
    shape = mock_validate_mlir(mlir_text)
    if shape.failed():
        return shape
    strict_static_findings = _strict_static_mlir_findings(mlir_text)
    if strict_static_findings:
        return MLIRValidation(
            MLIRValidationVerdict.FAILED,
            strict_static_findings)

    with tempfile.TemporaryDirectory(prefix="helix_mlir_validate_") as tmpdir:
        nonce = os.urandom(8).hex()
        smoke_specs = _mlir_invalid_smoke_specs(nonce)
        smoke_paths = tuple(
            (label,
             os.path.join(
                 tmpdir,
                 hashlib.sha256(f"{nonce}:in:{index}".encode(
                     "ascii")).hexdigest()[:24] + ".mlir"),
             os.path.join(
                 tmpdir,
                 hashlib.sha256(f"{nonce}:out:{index}".encode(
                     "ascii")).hexdigest()[:24] + ".mlir"),
             text)
            for index, (label, text) in enumerate(smoke_specs)
        )
        mlir_path = os.path.join(
            tmpdir,
            hashlib.sha256(f"{nonce}:real:in".encode(
                "ascii")).hexdigest()[:24] + ".mlir")
        out_path = os.path.join(
            tmpdir,
            hashlib.sha256(f"{nonce}:real:out".encode(
                "ascii")).hexdigest()[:24] + ".mlir")
        try:
            for _label, smoke_path, _smoke_out_path, smoke_text in smoke_paths:
                with open(smoke_path, "w", encoding="utf-8") as f:
                    f.write(smoke_text)
            with open(mlir_path, "w", encoding="utf-8") as f:
                f.write(mlir_text)
        except (OSError, UnicodeError) as exc:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"could not write temp MLIR input {mlir_path!r} "
                 f"({type(exc).__name__}: {exc})",))

        for smoke_label, smoke_path, smoke_out_path, _smoke_text in smoke_paths:
            try:
                smoke_proc = subprocess.run(
                [mlir_opt, smoke_path, "-o", smoke_out_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return MLIRValidation(
                    MLIRValidationVerdict.FAILED,
                    (f"mlir-opt {smoke_label} check timed out after "
                     f"{timeout_s}s",))
            except (OSError, UnicodeError, ValueError) as exc:
                return MLIRValidation(
                    MLIRValidationVerdict.FAILED,
                    (f"mlir-opt {smoke_label} check: tool unusable at "
                     f"invocation ({type(exc).__name__}: {exc})",))
            if smoke_proc.returncode == 0:
                return MLIRValidation(
                    MLIRValidationVerdict.FAILED,
                    (f"mlir-opt {smoke_label} check accepted verifier-"
                     "invalid MLIR; refusing to trust this toolchain",))

        try:
            proc = subprocess.run(
                [mlir_opt, mlir_path, "-o", out_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"mlir-opt validation timed out after {timeout_s}s",))
        except (OSError, UnicodeError, ValueError) as exc:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"mlir-opt validation: tool unusable at invocation "
                 f"({type(exc).__name__}: {exc})",))

        if proc.returncode != 0:
            diag = (proc.stderr or "").strip() \
                or (proc.stdout or "").strip()
            if not diag:
                diag = "no diagnostic emitted"
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"mlir-opt exit {proc.returncode}: {diag[:500]}",))
        zero_exit_diag = _captured_tool_diagnostic(proc.stdout, proc.stderr)
        if zero_exit_diag:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                ("mlir-opt exited 0 but emitted a diagnostic: "
                 + zero_exit_diag,),)

        try:
            size = os.path.getsize(out_path)
        except OSError:
            size = -1
        if size <= 0:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"mlir-opt exited 0 but produced no output artifact at "
                 f"{out_path!r} - a 0 exit with no artifact is not a "
                 "validation pass",))
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                output_text = f.read()
        except (OSError, UnicodeError) as exc:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"could not read mlir-opt validation artifact "
                 f"{out_path!r} ({type(exc).__name__}: {exc})",))
        if not output_text.strip():
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                ("mlir-opt exited 0 but produced only blank output; "
                 "a blank artifact is not a validation pass",))
        output_shape = mock_validate_mlir(output_text)
        if output_shape.failed():
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                ("mlir-opt exited 0 but produced an artifact that is "
                 "not structurally valid MLIR: "
                 + "; ".join(output_shape.findings[:3]),))
        artifact_findings = _mlir_opt_output_artifact_findings(output_text)
        if artifact_findings:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                ("mlir-opt exited 0 but produced an artifact without a "
                 "canonical top-level MLIR container operation: "
                 + "; ".join(artifact_findings[:3]),))
        correspondence_findings = _mlir_opt_output_correspondence_findings(
            mlir_text, output_text)
        if correspondence_findings:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                ("mlir-opt exited 0 but produced an artifact that does "
                 "not preserve the input module/function identity: "
                 + "; ".join(correspondence_findings[:3]),))

    input_digest = hashlib.sha256(mlir_text.encode("utf-8")).hexdigest()
    output_digest = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    result = object.__new__(MLIRValidation)
    object.__setattr__(result, "verdict", MLIRValidationVerdict.PASSED)
    object.__setattr__(result, "findings", ())
    object.__setattr__(
        result,
        "provenance",
        (
            f"mlir-opt={mlir_opt}",
            f"artifact_name={os.path.basename(out_path)}",
            f"input_sha256={input_digest}",
            f"output_sha256={output_digest}",
        ),
    )
    return result


def _make_mlir_validation_runner(raw_runner):
    brand_pass, has_pass = _make_mlir_validation_pass_registry()

    class _MLIRValidationRunner:
        __slots__ = ("__raw_runner",)

        def __init__(self, runner) -> None:
            self.__raw_runner = runner

        def __call__(
                self, mlir_text: str, mlir_opt: str, *,
                timeout_s: int = _MLIR_VALIDATE_TIMEOUT_S
                ) -> MLIRValidation:
            result = self.__raw_runner(
                mlir_text, mlir_opt, timeout_s=timeout_s)
            if _validation_pass_provenance_is_coherent(result):
                return brand_pass(result)
            return result

        def has_real_pass(self, result: MLIRValidation) -> bool:
            return (_validation_pass_provenance_is_coherent(result)
                    and has_pass(result))

    runner = _MLIRValidationRunner(raw_runner)
    return runner, runner.has_real_pass


_validation_runner, _has_real_validation_pass_shape = (
    _make_mlir_validation_runner(_run_mlir_opt_validate))
_run_mlir_opt_validate = _validation_runner


def validate_mlir_with_toolchain(
        mlir_text: str,
        *,
        support: Optional[MLIRSupport] = None) -> MLIRValidation:
    """Validate MLIR text with the strongest available verifier.

    Always run `mock_validate_mlir` first. If it finds a structural
    defect, return that FAILED result and do not probe or invoke tools.
    If the mock shape is clean and `mlir-opt` is available, dispatch to
    it for real verification. If `mlir-opt` is absent, return an honest
    DEFERRED with the support details; the in-process bindings are only
    a capability surface here, not a verifier runner yet.
    """
    mock = mock_validate_mlir(mlir_text)
    if mock.failed():
        return mock

    detected_support = detect_mlir_support()
    support_was_injected = support is not None
    if support is None:
        support = detected_support
    if not isinstance(support, MLIRSupport):
        raise ValueError(
            "validate_mlir_with_toolchain: support must be an "
            f"MLIRSupport or None, got {support!r}")

    trusted_mlir_opt = (
        support.mlir_opt
        if (not support_was_injected
            or support.mlir_opt == detected_support.mlir_opt)
        else None
    )

    if trusted_mlir_opt is None:
        details = tuple(f"MLIR support probe: {line}"
                        for line in support.detail)
        if support_was_injected and support.mlir_opt is not None:
            details = details + (
                "caller-supplied MLIRSupport cannot mint a real "
                "validation PASS unless its mlir_opt path matches a "
                "fresh detect_mlir_support() probe",
            )
        return MLIRValidation(
            MLIRValidationVerdict.DEFERRED,
            (mock.findings
             + details
             + ("real MLIR validation is DEFERRED because `mlir-opt` "
                "is not available; in-process binding validation is "
                "not wired in Stage 213 chunk B",)))

    return _run_mlir_opt_validate(mlir_text, trusted_mlir_opt)
