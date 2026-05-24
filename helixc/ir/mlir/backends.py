"""
helixc/ir/mlir/backends.py - MLIR backend-lowering scaffold
(v3.0 Phase E, Stage 213 chunk A).

Stage 213 starts the "MLIR -> backends" seam. This module deliberately
does NOT claim that any backend consumes MLIR yet: it defines the five
targets, records the dialect contract each target will need, validates
MLIR text through the existing mock path, and returns a frozen
PASSED/FAILED/DEFERRED result.

Until Stage 214 supplies the target pass pipelines, valid MLIR returns
DEFERRED with an explicit reason. That is the Stage 210 mock-path rule:
no MLIR toolchain on this machine must never become a false pass, and
the legacy Tile-IR backend path remains the fallback.

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
from types import MappingProxyType
from typing import Callable, Optional

from helixc.backend.gpu_ci import BackendKind as GPUBackendKind

from .toolchain import MLIRSupport, detect_mlir_support
from .validate import (
    MLIRValidation, MLIRValidationVerdict, mock_validate_mlir,
    validate_mlir_with_toolchain, _has_real_validation_pass_shape,
    _decode_quoted_symbol_body, _mlir_func_interfaces,
    _mlir_func_interface_fields,
    _bare_word_at, _comment_stripped_text, _mlir_symbol_ref_at,
    _next_op_boundary, _normalize_symbol_ref, _read_bare_word,
    _matching_closer_index, _quoted_span_end, _skip_spaces,
    _generic_property_assignments, _generic_property_dict_after,
    _mlir_op_start_context_allows, _symbol_from_generic_string_property,
)


# Pin the public surface of this module so `from ... import *` and any
# documentation generator see only the wrappers, not the runner /
# branding internals. Underscore-prefixed names like
# `_run_mlir_opt_pipeline`, `_run_mlir_translate_step`,
# `_BackendOutputValidationBrandingRunner`,
# `_BackendPipelineRunner`, and the AUTHORITY mappings are convention-
# private — `__all__` makes that boundary explicit.
__all__ = (
    "MLIRBackendTarget",
    "MLIRBackendStatus",
    "MLIRBackendResult",
    "MLIRBackendOutputValidation",
    "MLIRBackendOutputValidator",
    "MLIR_BACKEND_TARGETS",
    "MLIR_BACKEND_REQUIRED_DIALECTS",
    "MLIR_BACKEND_LOWERING_PIPELINES",
    "MLIR_BACKEND_OUTPUT_VALIDATORS",
    "MLIR_BACKEND_TRANSLATORS",
    "GPU_BACKEND_TO_MLIR_TARGET",
    "backend_required_dialects",
    "backend_lowering_pipeline",
    "backend_translator",
    "mlir_target_for_gpu_backend",
    "lower_mlir_to_backend",
)


class MLIRBackendTarget(Enum):
    """The backend targets Stage 213 must eventually lower MLIR into."""
    LLVM_IR = "llvm_ir"
    PTX = "ptx"
    ROCM_HIP = "rocm_hip"
    METAL_MSL = "metal_msl"
    WEBGPU_WGSL = "webgpu_wgsl"


class MLIRBackendStatus(Enum):
    """Tri-state result for MLIR backend lowering."""
    PASSED = "passed"
    FAILED = "failed"
    DEFERRED = "deferred"


MLIR_BACKEND_TARGETS: tuple[MLIRBackendTarget, ...] = (
    MLIRBackendTarget.LLVM_IR,
    MLIRBackendTarget.PTX,
    MLIRBackendTarget.ROCM_HIP,
    MLIRBackendTarget.METAL_MSL,
    MLIRBackendTarget.WEBGPU_WGSL,
)


_GPU_BACKEND_TO_MLIR_TARGET_AUTHORITY = MappingProxyType({
    GPUBackendKind.PTX: MLIRBackendTarget.PTX,
    GPUBackendKind.ROCM_HIP: MLIRBackendTarget.ROCM_HIP,
    GPUBackendKind.METAL_MSL: MLIRBackendTarget.METAL_MSL,
    GPUBackendKind.WEBGPU_WGSL: MLIRBackendTarget.WEBGPU_WGSL,
})
GPU_BACKEND_TO_MLIR_TARGET = _GPU_BACKEND_TO_MLIR_TARGET_AUTHORITY


_MLIR_BACKEND_REQUIRED_DIALECTS_AUTHORITY = MappingProxyType({
    MLIRBackendTarget.LLVM_IR: (
        "func", "arith", "cf", "scf", "memref", "linalg", "vector",
        "llvm",
    ),
    MLIRBackendTarget.PTX: (
        "func", "arith", "memref", "gpu", "nvgpu", "nvvm",
    ),
    MLIRBackendTarget.ROCM_HIP: (
        "func", "arith", "memref", "gpu", "rocdl",
    ),
    MLIRBackendTarget.METAL_MSL: (
        "func", "arith", "memref", "gpu", "spirv",
    ),
    MLIRBackendTarget.WEBGPU_WGSL: (
        "func", "arith", "memref", "gpu", "spirv",
    ),
})
MLIR_BACKEND_REQUIRED_DIALECTS = _MLIR_BACKEND_REQUIRED_DIALECTS_AUTHORITY


# Stage 213 chunk A records the table and leaves every target empty on
# purpose. A future chunk must fill this table and teach
# `lower_mlir_to_backend` how to execute those passes before any target
# can return PASSED. Stage 213 chunk C defines the runner contract:
# each entry is a complete `mlir-opt` pass argument (e.g.
# "--canonicalize" or "--pass-pipeline=..."), not a shell fragment.
# Stage 214 chunk E wires LLVM_IR's canonical mlir-opt lowering pipeline:
# scf -> cf, then arith/cf/func/index/memref/vector into the llvm dialect,
# then clean up any unrealized casts the conversions leave behind. The
# output is LLVM-dialect MLIR; chunk D's translator step runs
# `mlir-translate --mlir-to-llvmir` after this to produce raw LLVM IR.
#
# `--convert-index-to-llvm` is required before `--finalize-memref-to-llvm-
# conversion`: `scf`/`memref` materialise `index` SSA values (loop IVs,
# memref offsets); without the index conversion, the memref finalize pass
# leaves unrealized `index` casts the reconcile pass cannot legalize and
# the whole pipeline aborts with `failed to legalize operation
# 'builtin.unrealized_conversion_cast'`.
_LLVM_IR_LOWERING_PIPELINE: tuple[str, ...] = (
    "--convert-scf-to-cf",
    "--convert-cf-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--convert-vector-to-llvm",
    "--convert-index-to-llvm",
    "--finalize-memref-to-llvm-conversion",
    "--reconcile-unrealized-casts",
)


# Stage 214 chunk G wires PTX's mlir-opt lowering pipeline: outline the
# GPU kernel into its own module-level op, lower the gpu / arith / cf /
# index / memref dialects toward NVVM + LLVM. Translator stage then
# runs `mlir-translate --mlir-to-llvmir` to get raw LLVM IR; chunk-F
# chained tool `llc -mtriple=nvptx64 -mcpu=sm_80` produces PTX text.
_PTX_LOWERING_PIPELINE: tuple[str, ...] = (
    "--gpu-kernel-outlining",
    "--convert-scf-to-cf",
    "--convert-cf-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--convert-vector-to-llvm",
    "--convert-index-to-llvm",
    "--finalize-memref-to-llvm-conversion",
    "--convert-gpu-to-nvvm",
    "--reconcile-unrealized-casts",
)

_MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY = MappingProxyType({
    MLIRBackendTarget.LLVM_IR: _LLVM_IR_LOWERING_PIPELINE,
    MLIRBackendTarget.PTX: _PTX_LOWERING_PIPELINE,
    MLIRBackendTarget.ROCM_HIP: (),
    MLIRBackendTarget.METAL_MSL: (),
    MLIRBackendTarget.WEBGPU_WGSL: (),
})
MLIR_BACKEND_LOWERING_PIPELINES = _MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY


# Stage 214 chunk A — the translator-step table. `mlir-opt` only lowers
# between MLIR dialects; its output is still MLIR text. The
# transformation from "MLIR in target dialect" to "raw target artifact
# the downstream consumer reads" requires a separate target-specific
# tool (`mlir-translate --mlir-to-llvmir` for LLVM IR; that plus
# `llc -mtriple=nvptx64` for PTX; SPIR-V serialization + `spirv-cross`
# for Metal MSL; SPIR-V serialization + `tint` for WGSL).
#
# Stage 214 chunk A declares the table type and a None-everywhere
# baseline so the drift-guard can enforce totality. Subsequent chunks
# wire one target at a time. The dev machine has none of these tools,
# so production results stay DEFERRED with informative findings until
# a future toolchain becomes available — the same fail-closed
# discipline Stage 213's pipeline + validator tables use.
#
# Entry shape: a tuple of `(tool_name, mlir_translate_flag, follow_up_args)`
# where `tool_name` is the executable (`"mlir-translate"`),
# `mlir_translate_flag` is the leading flag (`"--mlir-to-llvmir"`), and
# `follow_up_args` is a tuple of further argv tokens for the subsequent
# tool when the artifact requires a chained transformation (e.g.
# `("llc", "-mtriple=nvptx64", "-mcpu=sm_80")` for PTX). None means
# "translator step is not wired for this target; lowering stays
# DEFERRED."
# Stage 214 chunk E wires LLVM_IR's translator entry. LLVM_IR is the
# simplest target — one `mlir-translate --mlir-to-llvmir` invocation
# converts the dialect-MLIR output of the lowering pipeline directly to
# raw LLVM IR text. No chained follow-up tool needed (the other targets
# wait on the chunk-E+ chained-tool runner).
_LLVM_IR_TRANSLATOR: tuple[str, str, tuple[str, ...]] = (
    "mlir-translate",
    "--mlir-to-llvmir",
    (),
)

# Stage 214 chunk G wires PTX's translator. `mlir-translate
# --mlir-to-llvmir` produces raw LLVM IR (the LLVM dialect already has
# NVVM intrinsics from `--convert-gpu-to-nvvm`); then chunk-F's chained
# tool `llc -mtriple=nvptx64 -mcpu=sm_80` produces PTX text. The
# `-O2` flag is a reasonable default for the toolchain-aware case (a
# future chunk could lift it into a per-target config).
_PTX_TRANSLATOR: tuple[str, str, tuple[str, ...]] = (
    "mlir-translate",
    "--mlir-to-llvmir",
    ("llc", "-mtriple=nvptx64", "-mcpu=sm_80", "-O2"),
)

_MLIR_BACKEND_TRANSLATORS_AUTHORITY = MappingProxyType({
    MLIRBackendTarget.LLVM_IR: _LLVM_IR_TRANSLATOR,
    MLIRBackendTarget.PTX: _PTX_TRANSLATOR,
    MLIRBackendTarget.ROCM_HIP: None,
    MLIRBackendTarget.METAL_MSL: None,
    MLIRBackendTarget.WEBGPU_WGSL: None,
})
MLIR_BACKEND_TRANSLATORS = _MLIR_BACKEND_TRANSLATORS_AUTHORITY

_BACKEND_OUTPUT_FAILURE_PREFIX = "target output contract for "
_ARTIFACT_DIAGNOSTIC_PREFIXES = (
    "error:", "fatal:", "failed:", "note:", "traceback",
    "remark:", "warning:",
)


def _is_lowercase_sha256(text: str) -> bool:
    return (len(text) == 64
            and all(char in "0123456789abcdef" for char in text))


def _artifact_line_is_diagnostic(line: str) -> bool:
    lowered = line.strip().lower()
    return any(lowered.startswith(prefix) or f": {prefix}" in lowered
               for prefix in _ARTIFACT_DIAGNOSTIC_PREFIXES)


def _captured_tool_diagnostic(stdout: str | None,
                              stderr: str | None) -> str:
    for stream_name, stream in (("stderr", stderr), ("stdout", stdout)):
        if not stream:
            continue
        for line in stream.splitlines():
            stripped = line.strip()
            if _artifact_line_is_diagnostic(stripped):
                return f"{stream_name}: {stripped[:500]}"
    return ""


def _ptx_code_line(line: str) -> str:
    return line.split("//", 1)[0].strip()


def _ptx_next_code_line(lines: tuple[str, ...], start: int) -> str:
    for line in lines[start:]:
        code = _ptx_code_line(line)
        if code:
            return code
    return ""


def _ptx_next_code_line_index(
        lines: tuple[str, ...], start: int) -> int | None:
    for index, line in enumerate(lines[start:], start=start):
        if _ptx_code_line(line):
            return index
    return None


def _ptx_balanced_body_end(lines: tuple[str, ...],
                           first_index: int,
                           first_text: str) -> int | None:
    index = first_index
    code = first_text.strip()
    open_pos = code.find("{")
    if open_pos == -1 or code[:open_pos].strip():
        return None
    tail = code[open_pos + 1:].strip()
    if tail:
        if tail == "}":
            return index + 1
        if tail.endswith("}"):
            body_text = tail[:-1].strip()
            if body_text and not _ptx_entry_body_fragment_is_plausible(
                    body_text):
                return None
            return index + 1
        if not _ptx_entry_body_fragment_is_plausible(tail):
            return None
    index += 1
    while index < len(lines):
        code = _ptx_code_line(lines[index])
        if not code:
            index += 1
            continue
        if code == "}":
            return index + 1
        if code.startswith("}"):
            return None
        if not _ptx_entry_body_fragment_is_plausible(code):
            return None
        index += 1
    return None


def _ptx_entry_name_is_plausible(name: str) -> bool:
    return _ptx_param_identifier_is_plausible(name.strip())


def _ptx_entry_body_follows(
        after_close: str, lines: tuple[str, ...],
        close_line_index: int, next_index: int) -> int | None:
    after_close = after_close.strip()
    if after_close:
        if after_close.startswith("{"):
            return _ptx_balanced_body_end(
                lines, close_line_index, after_close)
        if not _ptx_entry_scope_directive_is_plausible(after_close):
            return None
    index = next_index
    while index < len(lines):
        next_code = _ptx_code_line(lines[index])
        if not next_code:
            index += 1
            continue
        if next_code.startswith("{"):
            return _ptx_balanced_body_end(lines, index, next_code)
        if not _ptx_entry_scope_directive_is_plausible(next_code):
            return None
        index += 1
    return None


def _ptx_func_body_follows(
        after_close: str, lines: tuple[str, ...],
        close_line_index: int, next_index: int) -> int | None:
    after_close = after_close.strip()
    if after_close:
        if after_close.startswith("{"):
            return _ptx_balanced_body_end(
                lines, close_line_index, after_close)
        directive_tail = _ptx_func_scope_directive_tail(after_close)
        if directive_tail is None:
            return None
        if directive_tail:
            if directive_tail.startswith("{"):
                return _ptx_balanced_body_end(
                    lines, close_line_index, directive_tail)
            return None
    index = next_index
    while index < len(lines):
        next_code = _ptx_code_line(lines[index])
        if not next_code:
            index += 1
            continue
        if next_code.startswith("{"):
            return _ptx_balanced_body_end(lines, index, next_code)
        directive_tail = _ptx_func_scope_directive_tail(next_code)
        if directive_tail is None:
            return None
        if directive_tail:
            if directive_tail.startswith("{"):
                return _ptx_balanced_body_end(lines, index, directive_tail)
            return None
        index += 1
    return None


def _ptx_entry_scope_directive_is_plausible(line: str) -> bool:
    directive = line.split(None, 1)[0].rstrip(";")
    if directive not in _PTX_ENTRY_SCOPE_DIRECTIVES:
        return False
    return _ptx_line_has_balanced_braces(line)


def _ptx_func_scope_directive_is_plausible(line: str) -> bool:
    return _ptx_func_scope_directive_tail(line) == ""


def _ptx_func_scope_directive_tail(line: str) -> str | None:
    directive = line.split(None, 1)[0].rstrip(";")
    if directive not in _PTX_FUNC_SCOPE_DIRECTIVES:
        return None
    tail = line[len(line.split(None, 1)[0]):].strip()
    if tail and not tail.startswith("{"):
        return None
    return tail


def _ptx_entry_tail(line: str) -> str | None:
    if line.startswith(".entry "):
        return line[len(".entry "):].strip()
    if line.startswith(".visible .entry "):
        return line[len(".visible .entry "):].strip()
    return None


def _ptx_parse_entry_end(
        lines: tuple[str, ...], index: int) -> int | None:
    line = _ptx_code_line(lines[index])
    entry_tail = _ptx_entry_tail(line)
    if entry_tail is None:
        return None
    open_index = entry_tail.find("(")
    if open_index == -1:
        return None
    if not _ptx_entry_name_is_plausible(entry_tail[:open_index]):
        return None
    close_index = entry_tail.find(")", open_index + 1)
    if close_index != -1:
        if not _ptx_param_list_is_plausible(
                entry_tail[open_index + 1:close_index]):
            return None
        return _ptx_entry_body_follows(
            entry_tail[close_index + 1:], lines, index, index + 1)
    param_lines = [entry_tail[open_index + 1:]]
    for cont_index, continuation in enumerate(
            lines[index + 1:], start=index + 1):
        continuation = _ptx_code_line(continuation)
        if not continuation:
            continue
        if continuation.startswith(".entry ") \
                or continuation.startswith(".visible .entry ") \
                or continuation.startswith(".version ") \
                or continuation.startswith(".target "):
            return None
        close_index = continuation.find(")")
        if close_index != -1:
            param_lines.append(continuation[:close_index])
            if not _ptx_param_list_is_plausible("\n".join(param_lines)):
                return None
            return _ptx_entry_body_follows(
                continuation[close_index + 1:],
                lines,
                cont_index,
                cont_index + 1,
            )
        param_lines.append(continuation)
    return None


def _ptx_param_list_is_plausible(
        text: str, *, allow_register_params: bool = False) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    parts: list[str] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(",")
        if any(not field.strip() for field in fields[:-1]):
            return False
        if fields[-1].strip():
            parts.append(fields[-1].strip())
        elif len(fields) == 1:
            return False
        parts.extend(field.strip() for field in fields[:-1])
    if not parts:
        return False
    for part in parts:
        if not part.startswith(".param ") and not (
                allow_register_params and part.startswith(".reg ")):
            return False
        tokens = part.split()
        if not _ptx_param_tokens_are_plausible(
                tokens, allow_register_params=allow_register_params):
            return False
        name = tokens[-1]
        if any(char.isspace() or char in "{}();" for char in name):
            return False
    return True


def _ptx_param_tokens_are_plausible(
        tokens: list[str], *, allow_register_params: bool = False) -> bool:
    storage_tokens = (".param", ".reg") if allow_register_params else (
        ".param",)
    if len(tokens) == 3:
        return (tokens[0] in storage_tokens
                and tokens[1] in _PTX_PARAM_TYPES
                and _ptx_param_declarator_is_plausible(tokens[2]))
    if len(tokens) == 5:
        return (tokens[0] == ".param"
                and tokens[1] == ".align"
                and tokens[2].isdigit()
                and int(tokens[2]) > 0
                and tokens[3] in _PTX_PARAM_TYPES
                and _ptx_param_declarator_is_plausible(tokens[4]))
    return False


def _ptx_param_declarator_is_plausible(name: str) -> bool:
    base, bracket, tail = name.partition("[")
    if not _ptx_param_identifier_is_plausible(base):
        return False
    if not bracket:
        return True
    return tail.endswith("]") and tail[:-1].isdigit() and int(tail[:-1]) > 0


def _ptx_param_identifier_is_plausible(name: str) -> bool:
    return (bool(name)
            and (name[0].isalpha() or name[0] in "_$")
            and all(char.isalnum() or char in "_$." for char in name))


def _ptx_line_has_balanced_braces(line: str) -> bool:
    depth = 0
    for char in line:
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return False
            depth -= 1
    return depth == 0


def _ptx_entry_body_fragment_is_plausible(fragment: str) -> bool:
    fragment = fragment.strip()
    if not fragment:
        return True
    if fragment.startswith((".version", ".target")):
        return False
    if fragment.startswith("."):
        directive = fragment.split(None, 1)[0].rstrip(";")
        return _ptx_body_directive_is_plausible(directive, fragment)
    if fragment.endswith(":"):
        return _ptx_label_is_plausible(fragment[:-1])
    if ";" not in fragment:
        return False
    if not _ptx_line_has_balanced_braces(fragment):
        return False
    statements = fragment.split(";")
    if statements[-1].strip():
        return False
    return all(
        _ptx_instruction_statement_is_plausible(statement)
        for statement in statements[:-1]
        if statement.strip()
    )


def _ptx_instruction_statement_is_plausible(statement: str) -> bool:
    text = statement.strip()
    if not text:
        return False
    parts = text.split(None, 1)
    if parts[0].startswith("@"):
        if len(parts) == 1 or not _ptx_predicate_guard_is_plausible(
                parts[0]):
            return False
        text = parts[1].strip()
        parts = text.split(None, 1)
    opcode = parts[0].rstrip(";")
    operand_text = parts[1].strip() if len(parts) > 1 else ""
    base_opcode = opcode.split(".", 1)[0]
    if base_opcode in _PTX_NO_OPERAND_OPCODES:
        return not operand_text
    if not operand_text:
        return False
    operands = _split_ptx_operands(operand_text)
    if operands is None:
        return False
    if not all(_ptx_operand_is_plausible(operand) for operand in operands):
        return False
    if base_opcode in _PTX_EXACT_OPERAND_COUNTS:
        return len(operands) == _PTX_EXACT_OPERAND_COUNTS[base_opcode]
    return base_opcode in _PTX_BODY_OPCODES


def _ptx_predicate_guard_is_plausible(token: str) -> bool:
    if token.startswith("@!"):
        predicate = token[2:]
    elif token.startswith("@"):
        predicate = token[1:]
    else:
        return False
    return _llvm_ir_identifier_token_is_plausible(predicate, "%")


def _ptx_operand_is_plausible(operand: str) -> bool:
    operand = operand.strip()
    if not operand:
        return False
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_$.%+-[](){}<>,")
    return all(char in allowed or char.isspace() for char in operand)


def _ptx_label_is_plausible(label: str) -> bool:
    label = label.strip()
    return bool(label) and all(
        char.isalnum() or char in "_$."
        for char in label
    )


def _split_ptx_operands(text: str) -> tuple[str, ...] | None:
    pairs = {"{": "}", "[": "]", "(": ")"}
    closes = {"}": "{", "]": "[", ")": "("}
    stack: list[str] = []
    operands: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char in pairs:
            stack.append(char)
            continue
        if char in closes:
            if not stack or stack[-1] != closes[char]:
                return None
            stack.pop()
            continue
        if char == "," and not stack:
            operand = text[start:index].strip()
            if not operand:
                return None
            operands.append(operand)
            start = index + 1
    if stack:
        return None
    operand = text[start:].strip()
    if not operand:
        return None
    operands.append(operand)
    if len(operands) > 1:
        return tuple(operands)
    return tuple(operands) if "," not in text else None


def _ptx_body_directive_is_plausible(directive: str,
                                     fragment: str) -> bool:
    if directive not in _PTX_BODY_DIRECTIVES:
        return False
    if not _ptx_line_has_balanced_braces(fragment):
        return False
    tokens = fragment.rstrip(";").split()
    if directive == ".reg":
        return (len(tokens) >= 3
                and tokens[1].startswith(".")
                and all(_ptx_identifier_is_plausible(token)
                        for token in tokens[2:]))
    return True


def _ptx_identifier_is_plausible(token: str) -> bool:
    token = token.strip().rstrip(",")
    return bool(token) and not any(char in "{}();" or char.isspace()
                                  for char in token)


_PTX_BODY_OPCODES = frozenset((
    "abs", "activemask", "add", "and", "atom", "bar", "bra", "brkpt",
    "call", "copysign", "cos", "cp", "createpolicy", "cvt", "cvta",
    "discard", "div", "elect", "ex2", "exit", "fence", "fma", "isspacep",
    "ld", "lg2", "mad", "match", "max", "membar", "min", "mov", "mul",
    "nanosleep", "neg", "not", "or", "prefetch", "rcp", "red", "rem",
    "ret", "rsqrt", "selp", "set", "setp", "shfl", "sin", "slct", "sqrt",
    "st", "sub", "suld", "suq", "sured", "sust", "tex", "trap", "vote",
    "wmma", "xor",
))
_PTX_NO_OPERAND_OPCODES = frozenset((
    "brkpt", "discard", "exit", "ret", "trap",
))
_PTX_EXACT_OPERAND_COUNTS = MappingProxyType({
    "add": 3,
    "ld": 2,
    "mov": 2,
    "mul": 3,
    "st": 2,
    "sub": 3,
})
_PTX_PARAM_TYPES = frozenset((
    ".b8", ".b16", ".b32", ".b64", ".b128",
    ".u8", ".u16", ".u32", ".u64",
    ".s8", ".s16", ".s32", ".s64",
    ".f16", ".f16x2", ".f32", ".f64",
    ".pred",
))
_PTX_BODY_DIRECTIVES = frozenset((
    ".align", ".file", ".loc", ".local", ".param", ".pragma", ".reg",
    ".shared",
))
_PTX_TOP_LEVEL_DIRECTIVES = frozenset((
    ".address_size", ".align", ".const", ".extern", ".file", ".global",
    ".loc", ".pragma", ".shared", ".visible",
))
_PTX_ENTRY_SCOPE_DIRECTIVES = frozenset((
    ".maxnreg", ".maxntid", ".minnctapersm", ".reqntid",
))
_PTX_FUNC_SCOPE_DIRECTIVES = frozenset((
    ".noreturn",
))


def _ptx_artifact_is_plausible(lines: tuple[str, ...]) -> bool:
    saw_entry = False
    saw_version = False
    saw_target = False
    index = 0
    while index < len(lines):
        line = _ptx_code_line(lines[index])
        if not line:
            index += 1
            continue
        if line.startswith(".version "):
            if saw_entry:
                return False
            saw_version = True
            index += 1
            continue
        if line.startswith(".target "):
            if saw_entry or not saw_version:
                return False
            saw_target = True
            index += 1
            continue
        if _ptx_entry_tail(line) is not None:
            if not (saw_version and saw_target):
                return False
            next_index = _ptx_parse_entry_end(lines, index)
            if next_index is None:
                return False
            saw_entry = True
            index = next_index
            continue
        if _ptx_func_tail(line) is not None:
            if not (saw_version and saw_target):
                return False
            parsed_func = _ptx_parse_func(lines, index)
            if parsed_func is None:
                return False
            index = parsed_func[0]
            continue
        if _artifact_line_is_diagnostic(line) or not line.startswith("."):
            return False
        directive = line.split(None, 1)[0].rstrip(";")
        if directive not in _PTX_TOP_LEVEL_DIRECTIVES:
            return False
        if not _ptx_line_has_balanced_braces(line):
            return False
        index += 1
    return saw_entry and saw_version and saw_target


def _looks_like_backend_output(target: MLIRBackendTarget,
                               output_text: str) -> bool:
    text = output_text.strip()
    lines = tuple(line.strip() for line in text.splitlines()
                  if line.strip())
    if target is MLIRBackendTarget.LLVM_IR:
        return _llvm_ir_artifact_is_plausible(lines)
    if target is MLIRBackendTarget.PTX:
        return _ptx_artifact_is_plausible(lines)
    if target is MLIRBackendTarget.ROCM_HIP:
        return _rocm_hip_artifact_is_plausible(lines)
    if target is MLIRBackendTarget.METAL_MSL:
        return _metal_msl_artifact_is_plausible(lines)
    if target is MLIRBackendTarget.WEBGPU_WGSL:
        return _webgpu_wgsl_artifact_is_plausible(lines)
    return False


def _llvm_ir_code_line(line: str) -> str:
    quoted = False
    escaped = False
    for index, char in enumerate(line):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
            continue
        if char == ";":
            return line[:index].strip()
    return line.strip()


def _llvm_ir_body_delta_from_open(code_line: str,
                                  expected_return_type: str = "") -> int | None:
    tail_start = _llvm_ir_function_header_tail_start(code_line)
    open_index = _llvm_ir_function_body_open_index(code_line)
    if open_index == -1:
        if "}" in code_line[tail_start:]:
            return None
        return 0
    if "}" in code_line[tail_start:open_index]:
        return None
    close_index = _llvm_ir_last_unquoted_char_index(
        code_line, "}", open_index + 1)
    if close_index is None:
        body_fragment = code_line[open_index + 1:]
        depth = 1
    else:
        if code_line[close_index + 1:].strip():
            return None
        body_fragment = code_line[open_index + 1:close_index]
        depth = 0
    if not _llvm_ir_body_fragment_is_plausible(
            body_fragment.strip(), expected_return_type):
        return None
    return depth


def _llvm_ir_last_unquoted_char_index(
        text: str, needle: str, start: int) -> int | None:
    found: int | None = None
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
            continue
        if char == needle:
            found = index
    return found


def _llvm_ir_body_fragment_is_plausible(
        fragment: str, expected_return_type: str = "") -> bool:
    if not fragment:
        return True
    return _llvm_ir_instruction_line_is_plausible(
        fragment, expected_return_type)


def _llvm_ir_body_fragment_has_terminator(fragment: str) -> bool:
    if not fragment:
        return False
    return _llvm_ir_instruction_line_is_terminator(fragment)


def _llvm_ir_body_fragment_from_open(code_line: str) -> str:
    open_index = _llvm_ir_function_body_open_index(code_line)
    if open_index == -1:
        return ""
    close_index = _llvm_ir_last_unquoted_char_index(
        code_line, "}", open_index + 1)
    end = close_index if close_index is not None else len(code_line)
    return code_line[open_index + 1:end].strip()


def _llvm_ir_instruction_line_is_plausible(
        line: str, expected_return_type: str = "") -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.endswith(":"):
        return _llvm_ir_label_is_plausible(stripped[:-1])
    if stripped == "unreachable":
        return True
    tokens = stripped.split()
    if not tokens:
        return False
    if tokens[0] == "ret":
        if len(tokens) == 2 and tokens[1] == "void":
            return expected_return_type in ("", "void")
        actual_return_type = " ".join(tokens[1:-1])
        return (len(tokens) >= 3
                and _llvm_ir_type_text_is_plausible(
                    actual_return_type)
                and (not expected_return_type
                     or actual_return_type == expected_return_type)
                and _llvm_ir_value_token_matches_type(
                    tokens[-1], actual_return_type))
    if tokens[0] == "br":
        return _llvm_ir_branch_line_is_plausible(stripped)
    if tokens[0] == "store":
        return _llvm_ir_store_line_is_plausible(stripped)
    if tokens[0] == "call":
        return _llvm_ir_call_line_is_plausible(stripped)
    if stripped.startswith("%"):
        eq_index = stripped.find(" = ")
        if eq_index == -1:
            return False
        rhs = stripped[eq_index + 3:].strip()
        return _llvm_ir_assignment_rhs_is_plausible(rhs)
    return False


def _llvm_ir_instruction_line_is_terminator(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.endswith(":"):
        return False
    tokens = stripped.split()
    return bool(tokens) and tokens[0] in ("ret", "br", "unreachable")


def _llvm_ir_value_token_is_plausible(token: str) -> bool:
    token = token.strip()
    return (_llvm_ir_identifier_token_is_plausible(token, "%")
            or _llvm_ir_identifier_token_is_plausible(token, "@")
            or token.lstrip("-").isdigit()
            or _llvm_ir_float_literal_is_plausible(token)
            or token in (
                "true", "false", "null", "poison", "undef",
                "zeroinitializer"))


def _llvm_ir_value_token_matches_type(token: str, type_text: str) -> bool:
    token = token.strip()
    type_text = type_text.strip()
    if not _llvm_ir_value_token_is_plausible(token) \
            or not _llvm_ir_type_text_is_plausible(type_text):
        return False
    if (_llvm_ir_identifier_token_is_plausible(token, "%")
            or _llvm_ir_identifier_token_is_plausible(token, "@")
            or token in ("poison", "undef")):
        return True
    if token == "zeroinitializer":
        return type_text not in ("void", "label", "metadata", "token")
    if _llvm_ir_type_is_pointer_like(type_text):
        return token == "null"
    if token == "null":
        return False
    if _llvm_ir_type_is_integer_like(type_text):
        if token in ("true", "false"):
            return type_text == "i1"
        if not token.lstrip("-").isdigit():
            return False
        if type_text == "i1":
            return token in ("0", "1")
        return True
    if token in ("true", "false"):
        return False
    if _llvm_ir_type_is_float_like(type_text):
        return _llvm_ir_float_literal_is_plausible(token)
    # Aggregate / vector / array types accept named (`%x`, `@g`), explicit
    # `zeroinitializer`, `undef`, `poison`, or multi-token literal forms
    # (e.g. `<i32 0, ...>`) — never a bare scalar literal. The
    # multi-token forms never reach this single-token-matcher in the
    # first place; the single-token forms are already handled above. A
    # bare `0` here is a smoke-aware-echo bug shape (e.g.
    # `ret { i32 } 0` or `ret <4 x i32> 0`), so fail closed.
    return False


def _llvm_ir_type_is_pointer_like(type_text: str) -> bool:
    stripped = type_text.strip()
    return stripped == "ptr" or stripped.startswith("ptr addrspace(")


def _llvm_ir_type_is_integer_like(type_text: str) -> bool:
    stripped = type_text.strip()
    return len(stripped) > 1 and stripped[0] == "i" \
        and stripped[1:].isdigit() and int(stripped[1:]) > 0


def _llvm_ir_type_is_float_like(type_text: str) -> bool:
    return type_text.strip() in (
        "half", "bfloat", "float", "double", "fp128", "x86_fp80",
        "ppc_fp128")


def _llvm_ir_identifier_token_is_plausible(token: str, prefix: str) -> bool:
    if not token.startswith(prefix):
        return False
    name = token[len(prefix):]
    if not name:
        return False
    if name[0] == '"':
        quoted_end = _llvm_ir_quoted_identifier_end(name, 0)
        return quoted_end == len(name)
    return all(char.isalnum() or char in "_$.-" for char in name)


def _llvm_ir_branch_line_is_plausible(line: str) -> bool:
    if line.startswith("br label "):
        target = line[len("br label "):].strip()
        return _llvm_ir_identifier_token_is_plausible(target, "%")
    if not line.startswith("br i1 "):
        return False
    rest = line[len("br i1 "):].strip()
    parts = _split_top_level_commas(rest)
    if len(parts) != 3:
        return False
    return (_llvm_ir_value_token_matches_type(parts[0], "i1")
            and all(_llvm_ir_branch_label_operand_is_plausible(part)
                    for part in parts[1:]))


def _llvm_ir_branch_label_operand_is_plausible(text: str) -> bool:
    tokens = text.strip().split()
    return (len(tokens) == 2 and tokens[0] == "label"
            and _llvm_ir_identifier_token_is_plausible(tokens[1], "%"))


def _llvm_ir_float_literal_is_plausible(token: str) -> bool:
    lowered = token.lower()
    if lowered.startswith(("0x", "-0x", "+0x")):
        digits = lowered[2:] if lowered.startswith("0x") else lowered[3:]
        return bool(digits) and all(
            char in "0123456789abcdef" for char in digits)
    if not any(char in lowered for char in ".e"):
        return False
    try:
        float(lowered)
    except ValueError:
        return False
    return all(char in "0123456789abcdefx+-." for char in lowered) \
        or "e" in lowered


def _llvm_ir_store_line_is_plausible(line: str) -> bool:
    if not line.startswith("store "):
        return False
    text = line[len("store "):].strip()
    parts = _split_top_level_commas(text)
    if len(parts) < 2:
        return False
    value_tokens = parts[0].split()
    ptr_tokens = parts[1].split()
    return (len(value_tokens) == 2
            and _llvm_ir_type_text_is_plausible(value_tokens[0])
            and _llvm_ir_value_token_matches_type(
                value_tokens[-1], value_tokens[0])
            and len(ptr_tokens) == 2
            and _llvm_ir_type_text_is_plausible(ptr_tokens[0])
            and _llvm_ir_value_token_matches_type(
                ptr_tokens[-1], ptr_tokens[0])
            and _llvm_ir_memory_access_attrs_are_plausible(parts[2:]))


def _llvm_ir_memory_access_attrs_are_plausible(
        attrs: tuple[str, ...]) -> bool:
    for attr in attrs:
        tokens = attr.split()
        if not tokens:
            return False
        if tokens[0] == "align":
            if len(tokens) != 2 or not _llvm_ir_alignment_value_is_plausible(
                    tokens[1]):
                return False
            continue
        return False
    return True


def _llvm_ir_call_line_is_plausible(line: str) -> bool:
    if not line.startswith("call "):
        return False
    tokens = line.split()
    if len(tokens) < 3:
        return False
    for index, token in enumerate(tokens[1:], start=1):
        if (token.startswith(("@", "%")) and "(" in token
                and ")" in " ".join(tokens[index:])):
            callee = token[:token.find("(")]
            tail = " ".join(tokens[index:])
            close = _balanced_paren_close(tail, tail.find("("))
            if close == -1 or tail[close + 1:].strip():
                return False
            args = _split_top_level_commas(
                tail[tail.find("(") + 1:close].strip())
            return (_llvm_ir_type_text_is_plausible(" ".join(tokens[1:index]))
                    and (_llvm_ir_identifier_token_is_plausible(callee, "@")
                         or _llvm_ir_identifier_token_is_plausible(
                             callee, "%"))
                    and all(_llvm_ir_typed_value_part_is_plausible(arg)
                            for arg in args))
    return False


def _llvm_ir_assignment_rhs_is_plausible(rhs: str) -> bool:
    tokens = rhs.split()
    if not tokens:
        return False
    opcode = tokens[0].split(".", 1)[0]
    if opcode not in _LLVM_IR_ASSIGNMENT_OPCODES:
        return False
    if opcode in _LLVM_IR_BINARY_OPCODES:
        operand_text = rhs[len(tokens[0]):].strip()
        while operand_text:
            flag, sep, rest = operand_text.partition(" ")
            if not sep or flag not in _LLVM_IR_BINARY_FLAGS:
                break
            operand_text = rest.strip()
        type_end = _llvm_ir_type_prefix_end(operand_text)
        if type_end is None:
            return False
        type_text = operand_text[:type_end].strip()
        if not _llvm_ir_type_text_is_plausible(type_text):
            return False
        operands = _split_top_level_commas(operand_text[type_end:].strip())
        return (len(operands) == 2
                and all(_llvm_ir_value_token_matches_type(
                    operand.strip(), type_text)
                        for operand in operands))
    if opcode == "load":
        parts = _split_top_level_commas(rhs[len(tokens[0]):].strip())
        if len(parts) < 2:
            return False
        ptr_tokens = parts[1].split()
        return (_llvm_ir_type_text_is_plausible(parts[0])
                and len(ptr_tokens) == 2
                and _llvm_ir_type_text_is_plausible(ptr_tokens[0])
                and _llvm_ir_value_token_matches_type(
                    ptr_tokens[-1], ptr_tokens[0])
                and _llvm_ir_memory_access_attrs_are_plausible(parts[2:]))
    if opcode == "alloca":
        return _llvm_ir_alloca_rhs_is_plausible(rhs)
    if opcode == "call":
        return _llvm_ir_call_line_is_plausible(rhs)
    if opcode in ("icmp", "fcmp"):
        return _llvm_ir_compare_rhs_is_plausible(rhs)
    if opcode == "phi":
        return _llvm_ir_phi_rhs_is_plausible(rhs)
    if opcode == "select":
        return _llvm_ir_select_rhs_is_plausible(rhs)
    if opcode == "getelementptr":
        return _llvm_ir_getelementptr_rhs_is_plausible(rhs)
    if opcode in _LLVM_IR_CAST_OPCODES:
        return _llvm_ir_cast_rhs_is_plausible(rhs)
    if opcode in ("extractelement", "extractvalue"):
        return _llvm_ir_extract_rhs_is_plausible(rhs)
    if opcode in ("insertelement", "insertvalue"):
        return _llvm_ir_insert_rhs_is_plausible(rhs)
    return False


def _llvm_ir_alloca_rhs_is_plausible(rhs: str) -> bool:
    rest = rhs[len("alloca"):].strip()
    parts = _split_top_level_commas(rest)
    if not parts or not parts[0].strip():
        return False
    if not _llvm_ir_type_text_is_plausible(parts[0]):
        return False
    index = 1
    if index < len(parts):
        count_tokens = parts[index].split()
        if (len(count_tokens) == 2
                and _llvm_ir_type_text_is_plausible(count_tokens[0])
                and _llvm_ir_value_token_matches_type(
                    count_tokens[1], count_tokens[0])):
            index += 1
    while index < len(parts):
        if not _llvm_ir_alloca_attr_is_plausible(parts[index]):
            return False
        index += 1
    return True


def _llvm_ir_alloca_attr_is_plausible(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("align "):
        tokens = stripped.split()
        return len(tokens) == 2 and _llvm_ir_alignment_value_is_plausible(
            tokens[1])
    if stripped.startswith("addrspace(") and stripped.endswith(")"):
        inner = stripped[len("addrspace("):-1].strip()
        return inner.isdigit()
    return False


def _llvm_ir_compare_rhs_is_plausible(rhs: str) -> bool:
    tokens = rhs.split()
    if len(tokens) < 5:
        return False
    type_index = 2
    while (type_index < len(tokens)
           and tokens[type_index] in _LLVM_IR_BINARY_FLAGS):
        type_index += 1
    if (type_index >= len(tokens)
            or not _llvm_ir_type_text_is_plausible(tokens[type_index])):
        return False
    parts = _split_top_level_commas(" ".join(tokens[type_index + 1:]))
    type_text = tokens[type_index]
    return (len(parts) == 2
            and all(_llvm_ir_value_token_matches_type(
                part.strip(), type_text) for part in parts))


def _llvm_ir_phi_rhs_is_plausible(rhs: str) -> bool:
    after_opcode = rhs.split(None, 1)
    if len(after_opcode) != 2:
        return False
    rest = after_opcode[1].strip()
    if not rest:
        return False
    tokens = rest.split(None, 1)
    if len(tokens) != 2 or not _llvm_ir_type_text_is_plausible(tokens[0]):
        return False
    entries = _split_top_level_commas(tokens[1])
    if not entries:
        return False
    for entry in entries:
        entry = entry.strip()
        if not (entry.startswith("[") and entry.endswith("]")):
            return False
        parts = _split_top_level_commas(entry[1:-1])
        if len(parts) != 2:
            return False
        if (not _llvm_ir_value_token_matches_type(
                    parts[0].strip(), tokens[0])
                or not _llvm_ir_identifier_token_is_plausible(
                    parts[1].strip(), "%")):
            return False
    return True


def _llvm_ir_select_rhs_is_plausible(rhs: str) -> bool:
    parts = _split_top_level_commas(rhs[len("select "):].strip())
    if len(parts) != 3:
        return False
    first = parts[0].split()
    if not (len(first) == 2 and first[0] == "i1"
            and _llvm_ir_value_token_matches_type(first[-1], "i1")):
        return False
    value_parts = [part.split() for part in parts[1:]]
    if not all(len(part) == 2 and _llvm_ir_type_text_is_plausible(part[0])
               and _llvm_ir_value_token_matches_type(part[-1], part[0])
               for part in value_parts):
        return False
    return value_parts[0][0] == value_parts[1][0]


def _llvm_ir_getelementptr_rhs_is_plausible(rhs: str) -> bool:
    rest = rhs[len("getelementptr "):].strip()
    if rest.startswith("inbounds "):
        rest = rest[len("inbounds "):].strip()
    parts = _split_top_level_commas(rest)
    if len(parts) < 2:
        return False
    pointer = parts[1].split()
    return (_llvm_ir_type_text_is_plausible(parts[0])
            and len(pointer) == 2
            and _llvm_ir_type_text_is_plausible(pointer[0])
            and _llvm_ir_value_token_matches_type(pointer[-1], pointer[0])
            and all(_llvm_ir_typed_value_part_is_plausible(part)
                    for part in parts[2:]))


def _llvm_ir_typed_value_part_is_plausible(text: str) -> bool:
    tokens = text.split()
    type_text = " ".join(tokens[:-1])
    return (len(tokens) >= 2
            and _llvm_ir_type_text_is_plausible(type_text)
            and _llvm_ir_value_token_matches_type(tokens[-1], type_text))


def _llvm_ir_cast_rhs_is_plausible(rhs: str) -> bool:
    _opcode, rest = rhs.split(None, 1)
    source_type_end = _llvm_ir_type_prefix_end(rest)
    if source_type_end is None:
        return False
    source_type = rest[:source_type_end].strip()
    after_type = rest[source_type_end:].strip()
    value_text, sep, dest_type = after_type.partition(" to ")
    if not sep:
        return False
    return (_llvm_ir_type_text_is_plausible(source_type)
            and _llvm_ir_value_token_matches_type(value_text, source_type)
            and _llvm_ir_type_text_is_plausible(dest_type.strip()))


def _llvm_ir_extract_rhs_is_plausible(rhs: str) -> bool:
    _opcode, rest = rhs.split(None, 1)
    parts = _split_top_level_commas(rest)
    if len(parts) < 2:
        return False
    first_tokens = parts[0].split()
    return (len(first_tokens) >= 2
            and _llvm_ir_type_text_is_plausible(" ".join(first_tokens[:-1]))
            and _llvm_ir_value_token_matches_type(
                first_tokens[-1], " ".join(first_tokens[:-1]))
            and all(_llvm_ir_typed_value_part_is_plausible(part)
                    for part in parts[1:]))


def _llvm_ir_insert_rhs_is_plausible(rhs: str) -> bool:
    _opcode, rest = rhs.split(None, 1)
    parts = _split_top_level_commas(rest)
    if len(parts) < 3:
        return False
    for part in parts[:2]:
        tokens = part.split()
        if (len(tokens) < 2
                or not _llvm_ir_type_text_is_plausible(
                    " ".join(tokens[:-1]))
                or not _llvm_ir_value_token_matches_type(
                    tokens[-1], " ".join(tokens[:-1]))):
            return False
    return True


def _llvm_ir_type_prefix_end(text: str) -> int | None:
    stripped = text.lstrip()
    offset = len(text) - len(stripped)
    if not stripped:
        return None
    if stripped[0] in "<{[":
        close = {"<": ">", "{": "}", "[": "]"}[stripped[0]]
        close_index = _balanced_delimiter_close(stripped, 0, stripped[0], close)
        return None if close_index == -1 else offset + close_index + 1
    tokens = stripped.split()
    if not tokens:
        return None
    if (len(tokens) >= 2 and tokens[0] == "ptr"
            and tokens[1].startswith("addrspace(")):
        return offset + len(tokens[0]) + 1 + len(tokens[1])
    return offset + len(tokens[0])


def _balanced_delimiter_close(
        text: str, open_index: int, opener: str, closer: str) -> int:
    if open_index == -1 or open_index >= len(text) \
            or text[open_index] != opener:
        return -1
    depth = 0
    index = open_index
    while index < len(text):
        char = text[index]
        if char == '"':
            quoted_end = _llvm_ir_quoted_identifier_end(text, index)
            if quoted_end is None:
                return -1
            index = quoted_end
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _llvm_ir_function_header_is_plausible(line: str) -> bool:
    keyword = "define " if line.startswith("define ") else "declare "
    after_keyword = line[len(keyword):]
    at_index = after_keyword.find("@")
    if at_index <= 0:
        return False
    if not _llvm_ir_return_type_prefix_is_plausible(
            after_keyword[:at_index]):
        return False
    symbol_span = _llvm_ir_function_symbol_span(after_keyword, at_index)
    if symbol_span is None:
        return False
    symbol, open_index = symbol_span
    close_index = _balanced_paren_close(after_keyword, open_index)
    if close_index == -1:
        return False
    if not _llvm_ir_function_symbol_is_plausible(symbol):
        return False
    if not _llvm_ir_param_list_is_plausible(
            after_keyword[open_index + 1:close_index]):
        return False
    tail = after_keyword[close_index + 1:]
    body_index = tail.find("{")
    if body_index != -1:
        tail = tail[:body_index]
    return _llvm_ir_attribute_tail_is_plausible(tail)


def _llvm_ir_function_symbol_is_plausible(symbol: str) -> bool:
    if not symbol:
        return False
    if symbol.startswith('"') and symbol.endswith('"'):
        return len(symbol) > 2
    return all(char.isalnum() or char in "_$.-" for char in symbol)


def _llvm_ir_function_return_type(line: str) -> str:
    keyword = "define " if line.startswith("define ") else "declare "
    after_keyword = line[len(keyword):]
    at_index = after_keyword.find("@")
    if at_index <= 0:
        return ""
    prefix = after_keyword[:at_index].strip()
    return _llvm_ir_return_type_from_prefix(prefix)


def _llvm_ir_function_argument_values(line: str) -> set[str]:
    tail_start = _llvm_ir_function_header_tail_start(line)
    if tail_start <= 0:
        return set()
    before_tail = line[:tail_start]
    open_index = before_tail.find("(")
    close_index = before_tail.rfind(")")
    if open_index == -1 or close_index <= open_index:
        return set()
    return set(_llvm_ir_percent_tokens(before_tail[open_index + 1:close_index]))


def _llvm_ir_function_header_tail_start(line: str) -> int:
    if not line.startswith(("define ", "declare ")):
        return 0
    keyword = "define " if line.startswith("define ") else "declare "
    after_keyword = line[len(keyword):]
    at_index = after_keyword.find("@")
    if at_index <= 0:
        return 0
    symbol_span = _llvm_ir_function_symbol_span(after_keyword, at_index)
    if symbol_span is None:
        return 0
    _symbol, open_index = symbol_span
    close_index = _balanced_paren_close(after_keyword, open_index)
    if close_index == -1:
        return 0
    return len(keyword) + close_index + 1


def _llvm_ir_function_symbol_span(
        after_keyword: str, at_index: int) -> tuple[str, int] | None:
    symbol_start = at_index + 1
    if symbol_start >= len(after_keyword):
        return None
    if after_keyword[symbol_start] == '"':
        quoted_end = _llvm_quoted_symbol_end(after_keyword, symbol_start)
        if quoted_end is None:
            return None
        open_index = _skip_ascii_spaces(after_keyword, quoted_end)
        if open_index >= len(after_keyword) or after_keyword[open_index] != "(":
            return None
        return after_keyword[symbol_start:quoted_end], open_index
    open_index = after_keyword.find("(", symbol_start)
    if open_index == -1:
        return None
    symbol = after_keyword[symbol_start:open_index].strip()
    return symbol, open_index


def _skip_ascii_spaces(text: str, start: int) -> int:
    i = start
    while i < len(text) and text[i] in " \t":
        i += 1
    return i


def _llvm_ir_function_body_open_index(line: str) -> int:
    tail_start = _llvm_ir_function_header_tail_start(line)
    if tail_start == 0:
        return line.find("{")
    body_index = line.find("{", tail_start)
    return body_index


def _llvm_ir_assignment_result(line: str) -> str:
    stripped = line.strip()
    if not stripped.startswith("%"):
        return ""
    eq_index = stripped.find(" = ")
    if eq_index == -1:
        return ""
    token = stripped[:eq_index].strip()
    return token if _llvm_ir_identifier_token_is_plausible(token, "%") else ""


def _llvm_ir_percent_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "%":
            index += 1
            continue
        end = index + 1
        if end < len(text) and text[end] == '"':
            quoted_end = _llvm_ir_quoted_identifier_end(text, end)
            if quoted_end is None:
                index += 1
                continue
            token = text[index:quoted_end]
            end = quoted_end
        else:
            while end < len(text) and (
                    text[end].isalnum() or text[end] in "_$.-"):
                end += 1
            token = text[index:end]
        if _llvm_ir_identifier_token_is_plausible(token, "%"):
            tokens.append(token)
        index = max(end, index + 1)
    return tuple(tokens)


def _llvm_ir_instruction_uses_undefined_values(
        line: str, defined_values: set[str]) -> bool:
    stripped = line.strip()
    if not stripped or stripped.endswith(":"):
        return False
    if stripped.startswith("ret "):
        tokens = stripped.split()
        return bool(tokens and tokens[-1].startswith("%")
                    and tokens[-1] not in defined_values)
    if stripped.startswith("br i1 "):
        cond = stripped[len("br i1 "):].split(",", 1)[0].strip()
        return cond.startswith("%") and cond not in defined_values
    if stripped.startswith("%"):
        eq_index = stripped.find(" = ")
        if eq_index == -1:
            return False
        return any(token not in defined_values
                   for token in _llvm_ir_percent_tokens(
                       stripped[eq_index + 3:]))
    return False


def _llvm_ir_inline_body_uses_undefined_values(
        line: str, defined_values: set[str]) -> bool:
    open_index = _llvm_ir_function_body_open_index(line)
    if open_index == -1:
        return False
    close_index = _llvm_ir_last_unquoted_char_index(
        line, "}", open_index + 1)
    body = line[open_index + 1:close_index].strip() \
        if close_index is not None else line[open_index + 1:].strip()
    if not body:
        return False
    return _llvm_ir_instruction_uses_undefined_values(body, defined_values)


def _llvm_ir_return_type_from_prefix(prefix: str) -> str:
    stripped, _trailing_attrs = _llvm_ir_strip_trailing_return_attrs(
        prefix.strip())
    if not stripped:
        return ""
    if stripped.endswith("}") and "{" in stripped:
        return stripped[stripped.rfind("{"):]
    if stripped.endswith(">") and "<" in stripped:
        return stripped[stripped.rfind("<"):]
    if stripped.endswith("]") and "[" in stripped:
        return stripped[stripped.rfind("["):]
    tokens = stripped.split()
    if len(tokens) >= 2 and tokens[-2] == "ptr" \
            and tokens[-1].startswith("addrspace("):
        return " ".join(tokens[-2:])
    return tokens[-1] if tokens else ""


def _llvm_ir_return_type_prefix_is_plausible(text: str) -> bool:
    prefix, trailing_attrs = _llvm_ir_strip_trailing_return_attrs(text)
    return_type = _llvm_ir_return_type_from_prefix(prefix)
    if not _llvm_ir_type_text_is_plausible(return_type):
        return False
    if not prefix.endswith(return_type):
        return False
    attrs = prefix[:len(prefix) - len(return_type)].strip()
    attrs_ok = (not attrs or all(
        token in _LLVM_IR_FUNCTION_PREFIX_ATTRIBUTES
        or _llvm_ir_return_attribute_token_is_plausible(token, return_type)
        for token in attrs.split())) and all(
            _llvm_ir_return_attribute_token_is_plausible(token, return_type)
            for token in trailing_attrs)
    return attrs_ok


def _llvm_ir_strip_trailing_return_attrs(
        text: str) -> tuple[str, tuple[str, ...]]:
    stripped = text.strip()
    attrs: list[str] = []
    while stripped:
        before, sep, token = stripped.rpartition(" ")
        if not sep or not _llvm_ir_return_attribute_spelling_is_plausible(
                token):
            break
        attrs.append(token)
        stripped = before.rstrip()
    attrs.reverse()
    return stripped, tuple(attrs)


def _llvm_ir_return_attribute_token_is_plausible(
        token: str, return_type: str) -> bool:
    if not _llvm_ir_return_attribute_spelling_is_plausible(token):
        return False
    if token in _LLVM_IR_POINTER_RETURN_ATTRIBUTES:
        return _llvm_ir_param_type_accepts_pointer_attrs(return_type)
    if _llvm_ir_numeric_param_attribute_prefix_end(token) is not None:
        return _llvm_ir_param_type_accepts_pointer_attrs(return_type)
    return True


def _llvm_ir_return_attribute_spelling_is_plausible(token: str) -> bool:
    if token in _LLVM_IR_RETURN_ATTRIBUTES:
        return True
    attr_end = _llvm_ir_numeric_param_attribute_prefix_end(token)
    return attr_end is not None and attr_end == len(token)


def _llvm_ir_type_text_is_plausible(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("<") and stripped.endswith(">"):
        if stripped.count("<") != stripped.count(">"):
            return False
        inner = stripped[1:-1].strip()
        if inner.startswith("{") and inner.endswith("}"):
            return _llvm_ir_type_text_is_plausible(inner)
        return _llvm_ir_sized_sequence_type_is_plausible(inner)
    if stripped.startswith("{") and stripped.endswith("}"):
        if stripped.count("{") != stripped.count("}"):
            return False
        inner = stripped[1:-1].strip()
        if not inner:
            return True
        parts = _split_top_level_commas(inner)
        return bool(parts) and all(
            _llvm_ir_type_text_is_plausible(part) for part in parts)
    if stripped.startswith("[") and stripped.endswith("]"):
        if stripped.count("[") != stripped.count("]"):
            return False
        return _llvm_ir_sized_sequence_type_is_plausible(
            stripped[1:-1].strip())
    if stripped.startswith("ptr addrspace(") and stripped.endswith(")"):
        inner = stripped[len("ptr addrspace("):-1].strip()
        return inner.isdigit()
    return _llvm_ir_type_token_is_plausible(stripped)


def _llvm_ir_sized_sequence_type_is_plausible(text: str) -> bool:
    if " x " not in text:
        return False
    size, element_type = text.split(" x ", 1)
    size = size.strip()
    if not size.isdigit():
        return False
    return _llvm_ir_type_text_is_plausible(element_type.strip())


def _llvm_ir_type_token_is_plausible(token: str) -> bool:
    if token in _LLVM_IR_SCALAR_TYPES:
        return True
    if token.startswith("i") and token[1:].isdigit() \
            and int(token[1:]) > 0:
        return True
    if token.endswith("*"):
        return _llvm_ir_type_token_is_plausible(token[:-1])
    if token.startswith("%") and len(token) > 1:
        return _llvm_ir_identifier_token_is_plausible(token, "%")
    return False


def _balanced_paren_close(text: str, open_index: int) -> int:
    if open_index == -1 or open_index >= len(text) or text[open_index] != "(":
        return -1
    depth = 0
    index = open_index
    while index < len(text):
        char = text[index]
        if char == '"':
            quoted_end = _llvm_ir_quoted_identifier_end(text, index)
            if quoted_end is None:
                return -1
            index = quoted_end
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _split_top_level_commas(text: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    pairs = {"(": ")", "<": ">", "{": "}", "[": "]"}
    closes = {")": "(", ">": "<", "}": "{", "]": "["}
    stack: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            quoted_end = _llvm_ir_quoted_identifier_end(text, index)
            if quoted_end is None:
                return ()
            index = quoted_end
            continue
        if char in pairs:
            stack.append(char)
            depth += 1
        elif char in closes:
            if not stack or stack[-1] != closes[char]:
                return ()
            stack.pop()
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
        index += 1
    if stack:
        return ()
    parts.append(text[start:].strip())
    return tuple(parts)


def _llvm_ir_quoted_identifier_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != '"':
        return None
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
    return None


def _llvm_ir_param_list_is_plausible(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    parts = _split_top_level_commas(stripped)
    if not parts:
        return False
    for part in parts:
        param = part.strip()
        if not param:
            return False
        if param == "...":
            continue
        if not _llvm_ir_param_type_is_plausible(param):
            return False
    return True


def _llvm_ir_param_type_is_plausible(param: str) -> bool:
    tokens = param.split()
    if not tokens:
        return False
    while tokens and tokens[0] in _LLVM_IR_BARE_PARAM_ATTRIBUTES:
        tokens = tokens[1:]
    if not tokens:
        return False
    param = " ".join(tokens)
    if param[0] in "<{[":
        close = {"<": ">", "{": "}", "[": "]"}[param[0]]
        close_index = _balanced_delimiter_close(param, 0, param[0], close)
        if close_index == -1:
            return False
        tail = param[close_index + 1:].strip()
        return _llvm_ir_type_text_is_plausible(param[:close_index + 1]) \
            and _llvm_ir_optional_param_name_is_plausible(
                tail, param[:close_index + 1])
    if len(tokens) >= 2 and tokens[0] == "ptr" \
            and tokens[1].startswith("addrspace("):
        tail = " ".join(tokens[2:]).strip()
        if not _llvm_ir_type_text_is_plausible(" ".join(tokens[:2])):
            return False
        return _llvm_ir_optional_param_name_is_plausible(
            tail, " ".join(tokens[:2]))
    tail = " ".join(tokens[1:]).strip()
    return _llvm_ir_type_token_is_plausible(tokens[0]) \
        and _llvm_ir_optional_param_name_is_plausible(tail, tokens[0])


def _llvm_ir_optional_param_name_is_plausible(
        tail: str, param_type: str) -> bool:
    stripped = tail.strip()
    while stripped and not stripped.startswith("%"):
        attr = _llvm_ir_param_attribute_prefix_end(stripped)
        if attr is None:
            return False
        attr_end, requires_pointer = attr
        if requires_pointer and not _llvm_ir_param_type_accepts_pointer_attrs(
                param_type):
            return False
        stripped = stripped[attr_end:].strip()
    if not stripped:
        return True
    return _llvm_ir_identifier_token_is_plausible(stripped, "%")


def _llvm_ir_param_type_accepts_pointer_attrs(param_type: str) -> bool:
    stripped = param_type.strip()
    return (stripped == "ptr"
            or (stripped.startswith("ptr addrspace(")
                and _llvm_ir_type_text_is_plausible(stripped))
            or (stripped.startswith("<") and " x ptr" in stripped))


def _llvm_ir_param_attribute_prefix_end(
        text: str) -> tuple[int, bool] | None:
    stripped = text.lstrip()
    offset = len(text) - len(stripped)
    if not stripped:
        return None
    if stripped.startswith("align"):
        align_end = _llvm_ir_param_align_attribute_end(stripped)
        if align_end is not None:
            return offset + align_end, True
    for attr in _LLVM_IR_TYPED_PARAM_ATTRIBUTES:
        prefix = attr + "("
        if not stripped.startswith(prefix):
            continue
        close = _balanced_paren_close(stripped, len(attr))
        if close == -1:
            return None
        inner = stripped[len(prefix):close]
        if _llvm_ir_type_text_is_plausible(inner):
            return offset + close + 1, True
        return None
    for attr in _LLVM_IR_NUMERIC_PARAM_ATTRIBUTES:
        attr_end = _llvm_ir_numeric_param_attribute_prefix_end(
            stripped, attr)
        if attr_end is None:
            continue
        return offset + attr_end, True
    for attr in sorted(_LLVM_IR_BARE_PARAM_ATTRIBUTES, key=len, reverse=True):
        if stripped == attr or (
                stripped.startswith(attr)
                and stripped[len(attr)].isspace()):
            return offset + len(attr), attr in _LLVM_IR_POINTER_PARAM_ATTRIBUTES
    return None


def _llvm_ir_param_align_attribute_end(text: str) -> int | None:
    if text.startswith("align("):
        close = _balanced_paren_close(text, len("align"))
        if close == -1:
            return None
        inner = text[len("align("):close].strip()
        return close + 1 if _llvm_ir_alignment_value_is_plausible(
            inner) else None
    if text == "align":
        return None
    if not text.startswith("align") or not text[len("align")].isspace():
        return None
    i = _skip_ascii_spaces(text, len("align"))
    start = i
    while i < len(text) and text[i].isdigit():
        i += 1
    if i == start:
        return None
    if i != len(text) and not text[i].isspace():
        return None
    return i if _llvm_ir_alignment_value_is_plausible(
        text[start:i]) else None


def _llvm_ir_numeric_param_attribute_prefix_end(
        text: str, attr: str | None = None) -> int | None:
    attrs = (attr,) if attr is not None else _LLVM_IR_NUMERIC_PARAM_ATTRIBUTES
    for name in attrs:
        prefix = name + "("
        if not text.startswith(prefix):
            continue
        close = _balanced_paren_close(text, len(name))
        if close == -1:
            return None
        inner = text[len(prefix):close].strip()
        if inner.isdigit() and int(inner) > 0:
            return close + 1
        return None
    return None


def _llvm_ir_alignment_value_is_plausible(value: str) -> bool:
    if not value.isdigit():
        return False
    alignment = int(value)
    return alignment > 0 and alignment & (alignment - 1) == 0


def _llvm_ir_attribute_tail_is_plausible(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return True
    for token in tokens:
        if token.startswith("#") and token[1:].isdigit():
            continue
        if _llvm_ir_function_attribute_token_is_plausible(token):
            continue
        return False
    return True


def _llvm_ir_function_attribute_token_is_plausible(token: str) -> bool:
    if token in _LLVM_IR_FUNCTION_ATTRIBUTES:
        return True
    if token.startswith("alignstack(") and token.endswith(")"):
        inner = token[len("alignstack("):-1]
        return _llvm_ir_alignment_value_is_plausible(inner)
    if token.startswith('"'):
        quoted_end = _llvm_ir_quoted_identifier_end(token, 0)
        if quoted_end == len(token):
            return True
        if quoted_end is not None and quoted_end + 1 < len(token) \
                and token[quoted_end] == "=" \
                and token[quoted_end + 1] == '"':
            value_end = _llvm_ir_quoted_identifier_end(token, quoted_end + 1)
            return value_end == len(token)
    return False


def _llvm_ir_artifact_is_plausible(lines: tuple[str, ...]) -> bool:
    saw_marker = False
    saw_body = False
    saw_declaration = False
    pending_body = False
    pending_return_type = ""
    pending_value_defs: set[str] = set()
    active_return_type = ""
    active_value_defs: set[str] = set()
    active_body_has_terminator = False
    active_block_has_content = False
    active_block_terminated = False
    body_depth = 0
    for raw_line in lines:
        line = _llvm_ir_code_line(raw_line)
        if _artifact_line_is_diagnostic(line):
            return False
        if not line:
            continue
        if line in ("{", "}"):
            if line == "{":
                if pending_body:
                    saw_body = True
                    pending_body = False
                    active_return_type = pending_return_type
                    active_value_defs = set(pending_value_defs)
                    active_body_has_terminator = False
                    active_block_has_content = False
                    active_block_terminated = False
                    pending_return_type = ""
                    pending_value_defs = set()
                body_depth += 1
            else:
                if body_depth == 0:
                    return False
                if body_depth == 1 and (
                        not active_body_has_terminator
                        or not active_block_terminated):
                    return False
                body_depth -= 1
                if body_depth == 0:
                    active_return_type = ""
                    active_value_defs = set()
                    active_body_has_terminator = False
                    active_block_has_content = False
                    active_block_terminated = False
            continue
        if line.startswith(";") or line in ("{", "}"):
            continue
        if line.startswith(("source_filename = ", "target triple = ",
                            "target datalayout = ")):
            if " = " not in line:
                return False
            saw_marker = True
            continue
        if line.startswith(("define ", "declare ")):
            if not _llvm_ir_function_header_is_plausible(line):
                return False
            saw_marker = True
            if line.startswith("declare "):
                if "{" in line or "}" in line:
                    return False
                saw_declaration = True
            if line.startswith("define "):
                return_type = _llvm_ir_function_return_type(line)
                value_defs = _llvm_ir_function_argument_values(line)
                body_open_index = _llvm_ir_function_body_open_index(line)
                delta = _llvm_ir_body_delta_from_open(line, return_type)
                if delta is None:
                    return False
                if _llvm_ir_inline_body_uses_undefined_values(
                        line, value_defs):
                    return False
                if body_open_index != -1:
                    body_fragment = _llvm_ir_body_fragment_from_open(line)
                    body_fragment_has_terminator = (
                        _llvm_ir_body_fragment_has_terminator(body_fragment))
                    saw_body = True
                    body_depth += delta
                    if delta == 0:
                        if not body_fragment_has_terminator:
                            return False
                    else:
                        active_return_type = return_type
                        active_value_defs = set(value_defs)
                        active_body_has_terminator = (
                            body_fragment_has_terminator)
                        active_block_has_content = bool(body_fragment)
                        active_block_terminated = (
                            body_fragment_has_terminator)
                else:
                    pending_body = True
                    pending_return_type = return_type
                    pending_value_defs = set(value_defs)
            continue
        if line.startswith("@"):
            if " global " not in line and " constant " not in line:
                return False
            saw_marker = True
            saw_body = True
            continue
        if line.startswith("%") and " = type " in line:
            if not _llvm_ir_type_definition_is_plausible(line):
                return False
            saw_marker = True
            continue
        if line.startswith(("!", "attributes #")):
            continue
        if (body_depth > 0 and line.endswith(":")
                and _llvm_ir_label_is_plausible(line[:-1])):
            if active_block_has_content and not active_block_terminated:
                return False
            active_block_has_content = False
            active_block_terminated = False
            continue
        if body_depth > 0:
            if active_block_terminated:
                return False
            if not _llvm_ir_instruction_line_is_plausible(
                    line, active_return_type):
                return False
            if _llvm_ir_instruction_uses_undefined_values(
                    line, active_value_defs):
                return False
            active_block_has_content = True
            if _llvm_ir_instruction_line_is_terminator(line):
                active_body_has_terminator = True
                active_block_terminated = True
            result = _llvm_ir_assignment_result(line)
            if result:
                active_value_defs.add(result)
            continue
        return False
    return (saw_marker and (saw_body or saw_declaration)
            and not pending_body and body_depth == 0)


def _llvm_ir_type_definition_is_plausible(line: str) -> bool:
    name, sep, type_text = line.partition(" = type ")
    if not sep:
        return False
    return (_llvm_ir_identifier_token_is_plausible(name.strip(), "%")
            and _llvm_ir_type_text_is_plausible(type_text.strip()))


_LLVM_IR_BODY_PREFIXES = (
    "%", "add ", "alloca ", "and ", "ashr ", "br ", "call ", "extract",
    "addrspacecast ", "bitcast ", "fadd ", "fcmp ", "fdiv ", "fmul ",
    "fpext ", "fptosi ", "fptoui ", "fptrunc ", "frem ", "fsub ",
    "getelementptr ", "icmp ", "indirectbr ", "insertelement ",
    "insertvalue ", "inttoptr ", "invoke ", "landingpad ", "load ",
    "lshr ", "mul ", "or ", "phi ", "ptrtoint ", "ret ", "resume ",
    "sdiv ", "select ", "sext ", "shl ", "sitofp ", "srem ", "store ",
    "sub ", "switch ", "trunc ", "udiv ", "uitofp ", "unreachable",
    "urem ", "xor ", "zext ",
)
_LLVM_IR_ASSIGNMENT_OPCODES = frozenset((
    "add", "addrspacecast", "alloca", "and", "ashr", "bitcast", "call",
    "extractelement", "extractvalue", "fadd", "fcmp", "fdiv", "fmul",
    "fpext", "fptosi", "fptoui", "fptrunc", "frem", "fsub",
    "getelementptr", "icmp", "insertelement", "insertvalue", "inttoptr",
    "load", "lshr", "mul", "or", "phi", "ptrtoint", "sdiv", "select",
    "sext", "shl", "sitofp", "srem", "sub", "trunc", "udiv", "uitofp",
    "urem", "xor", "zext",
))
_LLVM_IR_BINARY_OPCODES = frozenset((
    "add", "and", "ashr", "fadd", "fdiv", "fmul", "frem", "fsub", "lshr",
    "mul", "or", "sdiv", "shl", "srem", "sub", "udiv", "urem", "xor",
))
_LLVM_IR_BINARY_FLAGS = frozenset((
    "afn", "arcp", "contract", "exact", "fast", "ninf", "nnan", "nsw",
    "nsz", "nuw", "reassoc",
))
_LLVM_IR_CAST_OPCODES = frozenset((
    "addrspacecast", "bitcast", "fpext", "fptosi", "fptoui", "fptrunc",
    "inttoptr", "ptrtoint", "sext", "sitofp", "trunc", "uitofp", "zext",
))
_LLVM_IR_STANDALONE_OPCODES = frozenset((
    "br", "call", "indirectbr", "invoke", "landingpad", "resume", "ret",
    "store", "switch", "unreachable",
))
_LLVM_IR_SCALAR_TYPES = frozenset((
    "void", "half", "bfloat", "float", "double", "fp128", "x86_fp80",
    "ppc_fp128", "ptr", "label", "metadata", "token",
))
_LLVM_IR_FUNCTION_ATTRIBUTES = frozenset((
    "alwaysinline", "cold", "convergent", "dso_local", "local_unnamed_addr",
    "mustprogress", "noinline", "noreturn", "nounwind", "optnone", "optsize",
    "readnone", "readonly", "ssp", "sspreq", "uwtable", "willreturn",
))
_LLVM_IR_FUNCTION_PREFIX_ATTRIBUTES = frozenset((
    "amdgpu_kernel", "appending", "available_externally", "ccc", "coldcc",
    "common", "default", "dllimport", "dllexport", "dso_local",
    "dso_preemptable", "external", "extern_weak", "fastcc", "hidden",
    "internal", "linkonce", "linkonce_odr", "local_unnamed_addr",
    "private", "protected", "unnamed_addr", "weak", "weak_odr",
))
_LLVM_IR_BARE_PARAM_ATTRIBUTES = frozenset((
    "immarg", "inreg", "nest", "noalias", "nocapture", "nofree", "nonnull",
    "noundef", "readnone", "readonly", "signext", "swiftasync",
    "swifterror", "swiftself", "writeonly", "zeroext",
))
_LLVM_IR_POINTER_PARAM_ATTRIBUTES = frozenset((
    "noalias", "nocapture", "nofree", "nonnull", "readnone", "readonly",
    "writeonly",
))
_LLVM_IR_TYPED_PARAM_ATTRIBUTES = frozenset((
    "byref", "byval", "elementtype", "inalloca", "preallocated", "sret",
))
_LLVM_IR_NUMERIC_PARAM_ATTRIBUTES = frozenset((
    "align", "dereferenceable", "dereferenceable_or_null",
))
_LLVM_IR_RETURN_ATTRIBUTES = frozenset((
    "noalias", "nonnull", "noundef", "readnone", "readonly", "signext",
    "zeroext",
))
_LLVM_IR_POINTER_RETURN_ATTRIBUTES = frozenset((
    "noalias", "nonnull", "readnone", "readonly",
))


def _rocm_hip_artifact_is_plausible(lines: tuple[str, ...]) -> bool:
    has_amdgcn_triple = any(
        line.startswith("target triple = ") and "amdgcn" in line.lower()
        for line in lines)
    has_kernel_definition = _llvm_ir_lines_have_amdgpu_kernel_definition(lines)
    if has_amdgcn_triple or has_kernel_definition:
        return (_llvm_ir_artifact_is_plausible(lines)
                and has_amdgcn_triple and has_kernel_definition)
    saw_marker = False
    saw_kernel = False
    body_depth = 0
    in_block_comment = False
    for index, line in enumerate(lines):
        lowered = line.lower()
        if _artifact_line_is_diagnostic(line):
            return False
        code_line, in_block_comment = _c_like_code_line(
            line, in_block_comment=in_block_comment)
        previous_depth = body_depth
        body_depth = _c_like_body_depth_after_line(body_depth, code_line)
        if body_depth < 0:
            return False
        if line.startswith(("#include <hip/hip_runtime.h>",
                            "#include <hip/hip_runtime_api.h>")):
            saw_marker = True
            continue
        if line.startswith("extern ") and "__global__" in line and "(" in line:
            saw_marker = True
            saw_kernel = _line_has_or_next_body(lines, index, line)
            if not saw_kernel:
                return False
            if not _hip_kernel_signature_is_plausible(
                    _c_like_signature_text(lines, index, line)):
                return False
            if not _c_like_body_fragments_are_plausible(code_line):
                return False
            continue
        if line.startswith("__global__") and "(" in line:
            saw_marker = True
            saw_kernel = _line_has_or_next_body(lines, index, line)
            if not saw_kernel:
                return False
            if not _hip_kernel_signature_is_plausible(
                    _c_like_signature_text(lines, index, line)):
                return False
            if not _c_like_body_fragments_are_plausible(code_line):
                return False
            continue
        if "hiplaunchkernelggl(" in lowered:
            saw_marker = True
            continue
        if line.startswith("target triple = ") and "amdgcn" in lowered:
            saw_marker = True
            continue
        if line.startswith("define ") and "amdgpu_kernel" in line:
            saw_marker = True
            saw_kernel = _line_has_or_next_body(lines, index, line)
            if not saw_kernel:
                return False
            if not _c_like_body_fragments_are_plausible(code_line):
                return False
            continue
        if previous_depth > 0 or body_depth > 0:
            if not _c_like_body_line_is_plausible(code_line):
                return False
            continue
        if _c_like_artifact_line_is_plausible(line):
            continue
        return False
    return (saw_marker and saw_kernel and body_depth == 0
            and not in_block_comment)


def _metal_msl_artifact_is_plausible(lines: tuple[str, ...]) -> bool:
    saw_marker = False
    saw_kernel = False
    body_depth = 0
    in_block_comment = False
    for index, line in enumerate(lines):
        if _artifact_line_is_diagnostic(line):
            return False
        code_line, in_block_comment = _c_like_code_line(
            line, in_block_comment=in_block_comment)
        previous_depth = body_depth
        body_depth = _c_like_body_depth_after_line(body_depth, code_line)
        if body_depth < 0:
            return False
        if line.startswith("#include <metal_stdlib>"):
            saw_marker = True
            continue
        if line == "using namespace metal;":
            saw_marker = True
            continue
        if line.startswith("kernel ") and "(" in line:
            saw_marker = True
            saw_kernel = _line_has_or_next_body(lines, index, line)
            if not saw_kernel:
                return False
            if not _metal_kernel_signature_is_plausible(
                    _c_like_signature_text(lines, index, line)):
                return False
            if not _c_like_body_fragments_are_plausible(code_line):
                return False
            continue
        if line.startswith("[[kernel]]") and "(" in line:
            saw_marker = True
            saw_kernel = _line_has_or_next_body(lines, index, line)
            if not saw_kernel:
                return False
            if not _metal_kernel_signature_is_plausible(
                    _c_like_signature_text(lines, index, line)):
                return False
            if not _c_like_body_fragments_are_plausible(code_line):
                return False
            continue
        if previous_depth > 0 or body_depth > 0:
            if not _c_like_body_line_is_plausible(code_line):
                return False
            continue
        if _c_like_artifact_line_is_plausible(line):
            continue
        return False
    return (saw_marker and saw_kernel and body_depth == 0
            and not in_block_comment)


def _webgpu_wgsl_artifact_is_plausible(lines: tuple[str, ...]) -> bool:
    saw_compute = False
    saw_workgroup_size = False
    saw_fn = False
    saw_body = False
    pending_compute = False
    pending_workgroup_size = False
    body_depth = 0
    in_block_comment = False
    for index, line in enumerate(lines):
        if _artifact_line_is_diagnostic(line):
            return False
        code_line, in_block_comment = _c_like_code_line(
            line, in_block_comment=in_block_comment)
        previous_depth = body_depth
        body_depth = _c_like_body_depth_after_line(body_depth, code_line)
        if body_depth < 0:
            return False
        stripped = code_line.strip()
        if not stripped:
            continue
        attr_prefix = _wgsl_compute_attribute_prefix(stripped)
        if pending_compute or pending_workgroup_size:
            if not (attr_prefix is not None or stripped.startswith("fn ")
                    or stripped.startswith("//")):
                return False
        if attr_prefix is not None:
            attrs, after_attrs = attr_prefix
            if "@compute" in attrs and pending_compute:
                return False
            if "@workgroup_size" in attrs and pending_workgroup_size:
                return False
            pending_compute = (
                pending_compute
                or "@compute" in attrs)
            pending_workgroup_size = (
                pending_workgroup_size
                or "@workgroup_size" in attrs)
            if not after_attrs:
                continue
            if not after_attrs.startswith("fn "):
                return False
            line = after_attrs
            code_line = after_attrs
            stripped = after_attrs
        if line.startswith("fn ") and "(" in line:
            if not (pending_compute and pending_workgroup_size):
                return False
            saw_compute = True
            saw_workgroup_size = True
            saw_fn = True
            saw_body = _line_has_or_next_body(lines, index, line)
            if not saw_body:
                return False
            if not _wgsl_signature_is_plausible(
                    _c_like_signature_text(lines, index, line)):
                return False
            if not _c_like_body_fragments_are_plausible(code_line):
                return False
            pending_compute = False
            pending_workgroup_size = False
            continue
        if previous_depth > 0 or body_depth > 0:
            if not _c_like_body_line_is_plausible(code_line):
                return False
            continue
        if _wgsl_artifact_line_is_plausible(line):
            continue
        return False
    return (saw_compute and saw_workgroup_size and saw_fn and saw_body
            and not (pending_compute or pending_workgroup_size)
            and body_depth == 0 and not in_block_comment)


def _c_like_code_line(
        line: str, *, in_block_comment: bool = False) -> tuple[str, bool]:
    code: list[str] = []
    quoted = ""
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        nxt = line[index + 1] if index + 1 < len(line) else ""
        if in_block_comment:
            if char == "*" and nxt == "/":
                in_block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quoted:
            code.append(" ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quoted:
                quoted = ""
            index += 1
            continue
        if char in ("'", '"'):
            quoted = char
            code.append(" ")
            index += 1
            continue
        if char == "/" and nxt == "/":
            break
        if char == "/" and nxt == "*":
            in_block_comment = True
            index += 2
            continue
        code.append(char)
        index += 1
    return "".join(code).strip(), in_block_comment


def _c_like_body_depth_after_line(depth: int, line: str) -> int:
    for char in line:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return -1
    return depth


def _c_like_body_fragments_are_plausible(code_line: str) -> bool:
    open_index = code_line.find("{")
    if open_index == -1:
        return True
    depth = 0
    fragment: list[str] = []
    saw_body_close = False
    for char in code_line[open_index:]:
        if saw_body_close:
            if char.isspace():
                continue
            return False
        if char == "{":
            depth += 1
            continue
        if char == "}":
            if depth > 0 and fragment:
                if not _c_like_body_line_is_plausible(
                        "".join(fragment).strip()):
                    return False
                fragment = []
            if depth > 0:
                depth -= 1
            if depth == 0:
                saw_body_close = True
            continue
        if depth > 0:
            fragment.append(char)
    if fragment and not _c_like_body_line_is_plausible(
            "".join(fragment).strip()):
        return False
    return True


def _c_like_body_line_is_plausible(code_line: str) -> bool:
    stripped = code_line.strip()
    if not stripped:
        return True
    if stripped in ("{", "}", "};"):
        return True
    if stripped.endswith("{") and stripped[:-1].strip() in ("", ")"):
        return True
    if not _c_like_statement_delimiters_are_balanced(stripped):
        return False
    if not _c_like_control_header_is_plausible(stripped):
        return False
    if stripped.endswith(":"):
        return True
    if _c_like_inline_block_is_plausible(stripped):
        return True
    if stripped.endswith(";"):
        statement = stripped[:-1].strip()
        if (not statement
                or not _c_like_statement_chars_are_plausible(statement)
                or _c_like_rvalue_is_only_operator(statement)
                or _c_like_rvalue_has_bad_operator_tail(statement)):
            return False
        if "=" in statement:
            if not _c_like_assignment_statement_is_plausible(statement):
                return False
            rhs = statement.split("=", 1)[1].strip()
            if (not rhs
                    or _c_like_rvalue_is_only_operator(rhs)
                    or _c_like_rvalue_has_bad_operator_tail(rhs)
                    or _c_like_rvalue_has_missing_operator(rhs)):
                return False
        if not _c_like_declaration_statement_is_plausible(statement):
            return False
        if not any(char.isalnum() or char == "_" for char in statement):
            return False
        if statement in ("break", "continue", "discard", "return"):
            return True
        if not _c_like_call_groups_are_plausible(statement):
            return False
        if statement and statement.replace("_", "").isalnum() and (
                statement[0].isalpha() or statement[0] == "_"):
            return False
        tokens = statement.split()
        if (len(tokens) > 1
                and all(token.replace("_", "").isalnum()
                        for token in tokens)
                and tokens[0] not in _C_LIKE_TYPE_WORDS):
            return False
    return _c_like_artifact_line_is_plausible(
        stripped, allow_statement=True)


def _c_like_call_groups_are_plausible(statement: str) -> bool:
    index = 0
    while index < len(statement):
        if statement[index] != "(":
            index += 1
            continue
        close_index = _balanced_paren_close(statement, index)
        if close_index == -1:
            return False
        before = statement[:index].rstrip()
        before_char = before[-1] if before else ""
        if before_char.isalnum() or before_char in "_)]":
            inner = statement[index + 1:close_index].strip()
            if inner:
                parts = _split_top_level_commas(inner)
                if not parts or any(not part.strip() for part in parts):
                    return False
        index = close_index + 1
    return True


def _c_like_statement_chars_are_plausible(statement: str) -> bool:
    return not any(char in statement for char in "@`#")


def _c_like_assignment_statement_is_plausible(statement: str) -> bool:
    lhs = statement.split("=", 1)[0].strip()
    if not lhs:
        return False
    first = lhs.split(None, 1)[0].split("<", 1)[0]
    if first in _C_LIKE_TYPE_WORDS:
        return True
    return any(marker in lhs for marker in ("[", ".", "->", "*"))


def _c_like_inline_block_is_plausible(statement: str) -> bool:
    open_index = statement.find("{")
    if open_index == -1 or not statement.endswith("}"):
        return False
    header = statement[:open_index].strip()
    body = statement[open_index + 1:-1].strip()
    if not _c_like_control_header_is_plausible(header):
        return False
    if not header.startswith(("if ", "for ", "while ", "switch ")):
        return False
    if not body:
        return True
    return _c_like_body_line_is_plausible(body)


def _c_like_signature_params_are_plausible(
        code_line: str, *, wgsl: bool = False) -> bool:
    open_index = code_line.find("(")
    if open_index == -1:
        return False
    close_index = _balanced_paren_close(code_line, open_index)
    if close_index == -1:
        return False
    params = code_line[open_index + 1:close_index].strip()
    if not params:
        return True
    parts = _split_top_level_commas(params)
    if not parts:
        return False
    return all(
        _wgsl_param_is_plausible(part)
        if wgsl else _c_like_param_is_plausible(part)
        for part in parts
    )


def _hip_kernel_signature_is_plausible(code_line: str) -> bool:
    if not _c_like_signature_params_are_plausible(code_line):
        return False
    if not _c_like_signature_tail_is_plausible(code_line):
        return False
    prefix = code_line[:code_line.find("(")].strip()
    tokens = prefix.replace("*", " * ").replace("&", " & ").split()
    if tokens and tokens[0] == "extern":
        tokens = tokens[1:]
    if "__global__" not in tokens:
        return False
    global_index = tokens.index("__global__")
    trailing = [token for token in tokens[global_index + 1:]
                if token not in ("*", "&")]
    return (len(trailing) >= 2
            and trailing[0] in _C_LIKE_TYPE_WORDS
            and _c_like_identifier_is_plausible(trailing[-1]))


def _metal_kernel_signature_is_plausible(code_line: str) -> bool:
    if not _c_like_signature_params_are_plausible(code_line):
        return False
    if not _c_like_signature_tail_is_plausible(code_line):
        return False
    prefix = code_line[:code_line.find("(")].strip()
    tokens = prefix.replace("*", " * ").replace("&", " & ").split()
    if tokens and tokens[0] == "kernel":
        tokens = tokens[1:]
    if tokens and tokens[0] == "[[kernel]]":
        tokens = tokens[1:]
    trailing = [token for token in tokens if token not in ("*", "&")]
    return (len(trailing) >= 2
            and trailing[0] in _C_LIKE_TYPE_WORDS
            and _c_like_identifier_is_plausible(trailing[-1]))


def _wgsl_signature_is_plausible(code_line: str) -> bool:
    if not _c_like_signature_params_are_plausible(code_line, wgsl=True):
        return False
    if not _c_like_signature_tail_is_plausible(code_line, wgsl=True):
        return False
    prefix = code_line[:code_line.find("(")].strip()
    tokens = prefix.split()
    return (len(tokens) == 2 and tokens[0] == "fn"
            and _c_like_identifier_is_plausible(tokens[1]))


def _wgsl_compute_attribute_prefix(
        line: str) -> tuple[tuple[str, ...], str] | None:
    attrs: list[str] = []
    index = 0
    while index < len(line):
        index = _skip_spaces(line, index)
        if index >= len(line):
            break
        if line.startswith("fn ", index):
            return (tuple(attrs), line[index:]) if attrs else None
        if not line.startswith("@", index):
            return None
        if line.startswith("@compute", index):
            end = index + len("@compute")
            if end < len(line) and _wgsl_identifier_char(line[end]):
                return None
            if end < len(line) and line[end] == "(":
                return None
            if "@compute" in attrs:
                return None
            attrs.append("@compute")
            index = end
            if index < len(line) and not line[index].isspace():
                return None
            continue
        if line.startswith("@workgroup_size(", index):
            open_index = index + len("@workgroup_size")
            close_index = _balanced_paren_close(line, open_index)
            if close_index == -1:
                return None
            args = line[open_index + 1:close_index]
            if not _wgsl_workgroup_size_args_are_plausible(args):
                return None
            if "@workgroup_size" in attrs:
                return None
            attrs.append("@workgroup_size")
            index = close_index + 1
            if index < len(line) and not line[index].isspace():
                return None
            continue
        return None
    return (tuple(attrs), "") if attrs else None


def _wgsl_identifier_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _wgsl_workgroup_size_args_are_plausible(args: str) -> bool:
    parts = _split_top_level_commas(args)
    if not (1 <= len(parts) <= 3
            and all(_wgsl_workgroup_size_arg_is_plausible(part)
                    for part in parts)):
        return False
    literal_kinds = {
        kind for kind in (
            _wgsl_workgroup_size_arg_literal_kind(part) for part in parts)
        if kind
    }
    return len(literal_kinds) <= 1


def _wgsl_workgroup_size_arg_is_plausible(arg: str) -> bool:
    stripped = arg.strip()
    if not stripped:
        return False
    if any(char in stripped for char in ";{}@"):
        return False
    if not _c_like_statement_delimiters_are_balanced(stripped):
        return False
    literal = stripped[:-1] if stripped.endswith("u") else stripped
    if all(char.isdigit() or char == "u" for char in stripped):
        if not literal.isdigit():
            return False
        return int(literal) > 0
    if _wgsl_identifier_is_plausible(stripped):
        return True
    if stripped[0] in _C_LIKE_BAD_RVALUE_TAIL_CHARS \
            or stripped[-1] in _C_LIKE_BAD_RVALUE_TAIL_CHARS:
        return False
    if (not any(char in "+*/%" for char in stripped)
            or _c_like_rvalue_has_missing_operator(stripped)
            or _c_like_rvalue_has_bad_operator_tail(stripped)
            or _c_like_rvalue_is_only_operator(stripped)):
        return False
    if any(char not in _WGSL_WORKGROUP_SIZE_EXPR_CHARS for char in stripped):
        return False
    return any(_wgsl_identifier_is_plausible(token) or token.isdigit()
               for token in _wgsl_workgroup_size_expr_tokens(stripped))


def _wgsl_workgroup_size_arg_literal_kind(arg: str) -> str:
    stripped = arg.strip()
    literal = stripped[:-1] if stripped.endswith("u") else stripped
    if literal.isdigit():
        return "u" if stripped.endswith("u") else "i"
    return ""


def _wgsl_identifier_is_plausible(text: str) -> bool:
    return (text != "_"
            and bool(text)
            and (text[0].isalpha() or text[0] == "_")
            and all(_wgsl_identifier_char(char) for char in text))


def _wgsl_workgroup_size_expr_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if _wgsl_identifier_char(char):
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


_WGSL_WORKGROUP_SIZE_EXPR_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ "
    "\t()+*/%")
_WGSL_TYPE_TEXT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ "
    "\t<>,()")
_WGSL_PARAM_ATTRIBUTE_ARG_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ "
    "\t<>,()+-*/%")
_WGSL_INTEGER_SCALAR_TYPES = frozenset(("i32", "u32"))
_WGSL_FLOAT_SCALAR_TYPES = frozenset(("f16", "f32"))
_WGSL_NUMERIC_SCALAR_TYPES = (
    _WGSL_INTEGER_SCALAR_TYPES | _WGSL_FLOAT_SCALAR_TYPES)
_WGSL_SCALAR_TYPES = frozenset(("bool",)) | _WGSL_NUMERIC_SCALAR_TYPES
_WGSL_ADDRESS_SPACES = frozenset((
    "function", "private", "storage", "uniform", "workgroup",
))
_WGSL_ACCESS_MODES = frozenset(("read", "read_write", "write"))


def _c_like_signature_tail_is_plausible(
        code_line: str, *, wgsl: bool = False) -> bool:
    open_index = code_line.find("(")
    close_index = _balanced_paren_close(code_line, open_index)
    if close_index == -1:
        return False
    tail = code_line[close_index + 1:].strip()
    body_index = tail.find("{")
    before_body = tail[:body_index].strip() if body_index != -1 else tail
    if not before_body:
        return True
    if wgsl:
        if not before_body.startswith("->"):
            return False
        return _wgsl_type_text_is_plausible(before_body[2:].strip())
    while before_body.startswith("[["):
        attr_end = before_body.find("]]")
        if attr_end == -1:
            return False
        before_body = before_body[attr_end + 2:].strip()
    return not before_body


def _c_like_identifier_is_plausible(name: str) -> bool:
    return (bool(name)
            and (name[0].isalpha() or name[0] == "_")
            and all(char.isalnum() or char == "_" for char in name))


def _c_like_signature_text(
        lines: tuple[str, ...], index: int, line: str) -> str:
    code_line, in_block_comment = _c_like_code_line(line)
    collected = [code_line]
    open_index = code_line.find("(")
    if open_index == -1:
        return code_line
    if _balanced_paren_close(code_line, open_index) != -1:
        return code_line
    for candidate in lines[index + 1:]:
        stripped, in_block_comment = _c_like_code_line(
            candidate, in_block_comment=in_block_comment)
        if not stripped:
            continue
        collected.append(stripped)
        joined = " ".join(collected)
        if _balanced_paren_close(joined, open_index) != -1:
            return joined
        if "{" in stripped or ";" in stripped:
            return joined
    return " ".join(collected)


def _c_like_param_is_plausible(param: str) -> bool:
    stripped = param.strip()
    if not stripped:
        return False
    stripped = _c_like_strip_attributes(stripped)
    if not stripped or any(char in stripped for char in "?!;{}@="):
        return False
    if not _c_like_statement_delimiters_are_balanced(stripped):
        return False
    tokens = stripped.replace("*", " * ").replace("&", " & ").split()
    if len(tokens) < 2:
        return False
    name = ""
    for token in reversed(tokens):
        if token not in ("*", "&"):
            name = token
            break
    if not _c_like_identifier_is_plausible(name):
        return False
    type_tokens = [token for token in tokens[:tokens.index(name)]
                   if token not in ("*", "&")]
    if not type_tokens:
        return False
    return any(token.split("<", 1)[0] in _C_LIKE_TYPE_WORDS
               for token in type_tokens)


def _c_like_strip_attributes(text: str) -> str:
    result = text
    while "[[" in result:
        start = result.find("[[")
        end = result.find("]]", start + 2)
        if end == -1:
            return ""
        result = (result[:start] + " " + result[end + 2:]).strip()
    return result


def _wgsl_param_is_plausible(param: str) -> bool:
    stripped = param.strip()
    colon_index = stripped.find(":")
    if colon_index == -1:
        return False
    name = stripped[:colon_index].strip()
    type_text = stripped[colon_index + 1:].strip()
    return (_wgsl_param_name_is_plausible(name)
            and _wgsl_type_text_is_plausible(type_text))


def _wgsl_param_name_is_plausible(name_text: str) -> bool:
    stripped = name_text.strip()
    if not stripped:
        return False
    index = 0
    while index < len(stripped) and stripped[index] == "@":
        attr_start = index + 1
        index = attr_start
        while index < len(stripped) and _wgsl_identifier_char(
                stripped[index]):
            index += 1
        attr_name = stripped[attr_start:index]
        if not _wgsl_identifier_is_plausible(attr_name):
            return False
        if index < len(stripped) and stripped[index] == "(":
            close_index = _balanced_paren_close(stripped, index)
            if close_index == -1:
                return False
            args = stripped[index + 1:close_index].strip()
            if not _wgsl_attribute_args_are_plausible(args):
                return False
            index = close_index + 1
        if index < len(stripped) and not stripped[index].isspace():
            return False
        index = _skip_spaces(stripped, index)
    return _wgsl_identifier_is_plausible(stripped[index:].strip())


def _wgsl_attribute_args_are_plausible(args: str) -> bool:
    if not args:
        return False
    if any(char in args for char in "?!;{}@"):
        return False
    if not _c_like_statement_delimiters_are_balanced(args):
        return False
    if any(char not in _WGSL_PARAM_ATTRIBUTE_ARG_CHARS for char in args):
        return False
    tokens = _wgsl_workgroup_size_expr_tokens(args)
    return bool(tokens) and all(
        token.isdigit() or _wgsl_identifier_is_plausible(token)
        for token in tokens)


def _wgsl_type_text_is_plausible(type_text: str) -> bool:
    stripped = type_text.strip()
    if not stripped:
        return False
    if any(char in stripped for char in "?!;{}@"):
        return False
    if not _c_like_statement_delimiters_are_balanced(stripped):
        return False
    if stripped[0] in ",:<>" or stripped[-1] in ",:<.":
        return False
    if any(char not in _WGSL_TYPE_TEXT_CHARS for char in stripped):
        return False
    tokens = _wgsl_workgroup_size_expr_tokens(stripped)
    return (bool(tokens) and all(
        token.isdigit() or _wgsl_identifier_is_plausible(token)
        for token in tokens)
        and _wgsl_type_semantics_are_plausible(stripped))


def _wgsl_type_semantics_are_plausible(type_text: str) -> bool:
    stripped = type_text.strip()
    if stripped in _WGSL_SCALAR_TYPES:
        return True
    for prefix in ("vec2", "vec3", "vec4"):
        inner = _wgsl_template_type_arg(stripped, prefix)
        if inner is not None:
            return inner in _WGSL_NUMERIC_SCALAR_TYPES
    for prefix in ("mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3",
                   "mat3x4", "mat4x2", "mat4x3", "mat4x4"):
        inner = _wgsl_template_type_arg(stripped, prefix)
        if inner is not None:
            return inner in _WGSL_FLOAT_SCALAR_TYPES
    inner = _wgsl_template_type_arg(stripped, "atomic")
    if inner is not None:
        return inner in _WGSL_INTEGER_SCALAR_TYPES
    inner = _wgsl_template_type_arg(stripped, "array")
    if inner is not None:
        parts = _split_top_level_commas(inner)
        return (1 <= len(parts) <= 2
                and _wgsl_type_semantics_are_plausible(parts[0].strip())
                and (len(parts) == 1
                     or _wgsl_workgroup_size_arg_is_plausible(parts[1])))
    inner = _wgsl_template_type_arg(stripped, "ptr")
    if inner is not None:
        parts = _split_top_level_commas(inner)
        return (2 <= len(parts) <= 3
                and parts[0].strip() in _WGSL_ADDRESS_SPACES
                and _wgsl_type_semantics_are_plausible(parts[1].strip())
                and (len(parts) == 2
                     or parts[2].strip() in _WGSL_ACCESS_MODES))
    return False


def _wgsl_template_type_arg(type_text: str, prefix: str) -> str | None:
    head = prefix + "<"
    if not type_text.startswith(head) or not type_text.endswith(">"):
        return None
    open_index = len(prefix)
    close_index = _balanced_angle_close(type_text, open_index)
    if close_index != len(type_text) - 1:
        return None
    return type_text[open_index + 1:close_index].strip()


def _balanced_angle_close(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _c_like_rvalue_is_only_operator(text: str) -> bool:
    compact = "".join(text.split())
    return bool(compact) and all(
        char in _C_LIKE_OPERATOR_ONLY_RVALUE_CHARS for char in compact)


def _c_like_rvalue_has_bad_operator_tail(text: str) -> bool:
    compact = "".join(text.split())
    return bool(compact) and compact[-1] in _C_LIKE_BAD_RVALUE_TAIL_CHARS


def _c_like_rvalue_has_missing_operator(text: str) -> bool:
    tokens = text.split()
    return len(tokens) > 1 and all(
        token.replace("_", "").isalnum() for token in tokens)


def _c_like_statement_delimiters_are_balanced(text: str) -> bool:
    pairs = {"(": ")", "[": "]"}
    closes = {")": "(", "]": "["}
    stack: list[str] = []
    for char in text:
        if char in pairs:
            stack.append(char)
            continue
        if char in closes:
            if not stack or stack[-1] != closes[char]:
                return False
            stack.pop()
    return not stack


def _c_like_control_header_is_plausible(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith(("if ", "for ", "while ", "switch ")):
        return True
    open_index = stripped.find("(")
    close_index = _balanced_paren_close(stripped, open_index)
    if open_index == -1 or close_index == -1:
        return False
    return bool(stripped[open_index + 1:close_index].strip())


def _c_like_declaration_statement_is_plausible(statement: str) -> bool:
    tokens = statement.split()
    if len(tokens) < 2:
        return True
    first = tokens[0]
    first_word = first.split("<", 1)[0]
    if first_word not in _C_LIKE_TYPE_WORDS:
        return True
    declarator = statement[len(first):].strip()
    if not declarator:
        return False
    if ":" in declarator:
        name_part, type_part = declarator.split(":", 1)
        type_part = type_part.split("=", 1)[0].strip()
        if not name_part.strip() or not type_part:
            return False
    lhs = declarator.split("=", 1)[0].split(":", 1)[0].strip()
    if any(char in lhs for char in ",[]()"):
        return True
    lhs_tokens = lhs.replace("*", " ").replace("&", " ").split()
    if len(lhs_tokens) > 1 and all(
            token.replace("_", "").isalnum() for token in lhs_tokens):
        return False
    # The trailing token of a declaration must be a valid identifier:
    # it starts with a letter or `_`, not a digit. This catches bug
    # shapes like `float * 123;` that the earlier alnum-checks accept
    # by virtue of `123` being alnum-only.
    if lhs_tokens:
        last = lhs_tokens[-1]
        if last and not (last[0].isalpha() or last[0] == "_"):
            return False
    return True


_C_LIKE_TYPE_WORDS = frozenset((
    "bool", "char", "const", "constant", "device", "double", "float", "half",
    "int", "int16_t", "int32_t", "int64_t", "int8_t", "let", "long", "ptr",
    "short", "signed", "size_t", "thread", "uint", "uint16_t", "uint32_t",
    "uint64_t", "uint8_t", "unsigned", "var", "void",
))
_C_LIKE_OPERATOR_ONLY_RVALUE_CHARS = frozenset(
    "+-*/%&|^!~=<>?:,.()[]{}")
_C_LIKE_BAD_RVALUE_TAIL_CHARS = frozenset("+-*/%&|^!~=<>?:,.")


def _line_has_or_next_body(lines: tuple[str, ...],
                           index: int, line: str) -> bool:
    code_line, in_block_comment = _c_like_code_line(line)
    open_index = code_line.find("{")
    semicolon_index = code_line.find(";")
    if open_index != -1 and (
            semicolon_index == -1 or open_index < semicolon_index):
        return True
    signature_closed = ")" in code_line and (
        code_line.rfind(")") > code_line.rfind("("))
    for candidate in lines[index + 1:]:
        stripped, in_block_comment = _c_like_code_line(
            candidate, in_block_comment=in_block_comment)
        if not stripped:
            continue
        if stripped == "{" and signature_closed:
            return True
        open_index = stripped.find("{")
        semicolon_index = stripped.find(";")
        if open_index != -1:
            before_open = stripped[:open_index].strip()
            if ((signature_closed or before_open == ")")
                    and (not before_open or before_open == ")")
                    and (semicolon_index == -1
                         or open_index < semicolon_index)):
                return True
            return False
        if semicolon_index != -1:
            return False
        if stripped.endswith(")"):
            signature_closed = True
            continue
        if (not signature_closed
                and _c_like_signature_continuation_is_plausible(stripped)):
            continue
        if stripped.startswith((
                "#include", "@compute", "extern ", "__global__", "kernel ",
                "[[kernel]]", "fn ", "target triple = ", "define ")):
            return False
    return False


def _c_like_signature_continuation_is_plausible(line: str) -> bool:
    stripped = line.strip().rstrip(",")
    if not stripped:
        return True
    if stripped.startswith("@"):
        return True
    if "[[" in stripped and "]]" in stripped:
        return True
    if ":" in stripped and "<" in stripped and ">" in stripped:
        return True
    return "*" in stripped or "&" in stripped


def _llvm_ir_artifact_line_is_body_like(line: str) -> bool:
    return (line.startswith((";", "!", "@", "attributes #"))
            or line in ("{", "}")
            or (line.endswith(":")
                and _llvm_ir_label_is_plausible(line[:-1]))
            or _llvm_ir_instruction_line_is_plausible(line))


def _llvm_ir_label_is_plausible(label: str) -> bool:
    label = label.strip()
    return bool(label) and all(
        char.isalnum() or char in "_$.-"
        for char in label
    )


def _c_like_artifact_line_is_plausible(
        line: str, *, allow_statement: bool = False) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith(("#error", "#warning")):
        return False
    if stripped.startswith("#"):
        return stripped.startswith(_C_LIKE_ALLOWED_PREPROCESSOR_PREFIXES)
    if stripped.startswith(("//", "/*", "*")):
        return True
    if stripped in ("{", "}", "};"):
        return True
    if stripped.endswith("{") or stripped.endswith((",", ")")):
        return True
    if allow_statement and stripped.endswith(";"):
        return True
    if "[[" in stripped and "]]" in stripped:
        return True
    if "*" in stripped and "(" not in stripped:
        return True
    return False


def _wgsl_artifact_line_is_plausible(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("//"):
        return True
    if stripped in ("{", "}", "};", ")"):
        return True
    if stripped.endswith(("{", ",", ")")):
        return True
    if stripped.startswith("alias "):
        return _wgsl_alias_line_is_plausible(stripped)
    if stripped.startswith("type "):
        return _wgsl_type_alias_line_is_plausible(stripped)
    if stripped.startswith(("var ", "var<")):
        return _wgsl_var_line_is_plausible(stripped)
    if stripped.startswith(("const ", "override ")):
        return _wgsl_const_line_is_plausible(stripped)
    if stripped.startswith("enable "):
        return _wgsl_enable_line_is_plausible(stripped)
    return stripped.startswith(("@", "fn ", "struct "))


def _wgsl_alias_line_is_plausible(line: str) -> bool:
    if not line.endswith(";"):
        return False
    body = line[len("alias "):-1].strip()
    name, sep, type_text = body.partition("=")
    return (bool(sep)
            and _wgsl_identifier_is_plausible(name.strip())
            and _wgsl_type_text_is_plausible(type_text.strip()))


def _wgsl_type_alias_line_is_plausible(line: str) -> bool:
    if not line.endswith(";"):
        return False
    body = line[len("type "):-1].strip()
    name, sep, type_text = body.partition("=")
    return (bool(sep)
            and _wgsl_identifier_is_plausible(name.strip())
            and _wgsl_type_text_is_plausible(type_text.strip()))


def _wgsl_var_line_is_plausible(line: str) -> bool:
    if not line.endswith(";"):
        return False
    body = line[:-1].strip()
    if body.startswith("var<"):
        close = _balanced_angle_close(body, len("var"))
        if close == -1:
            return False
        attrs = _split_top_level_commas(body[len("var<"):close])
        if not attrs or len(attrs) > 2:
            return False
        if attrs[0].strip() not in _WGSL_ADDRESS_SPACES:
            return False
        if len(attrs) == 2 and attrs[1].strip() not in _WGSL_ACCESS_MODES:
            return False
        tail = body[close + 1:].strip()
    elif body.startswith("var "):
        tail = body[len("var "):].strip()
    else:
        return False
    declaration = tail.split("=", 1)[0].strip()
    name, sep, type_text = declaration.partition(":")
    return (bool(sep)
            and _wgsl_identifier_is_plausible(name.strip())
            and _wgsl_type_text_is_plausible(type_text.strip()))


def _wgsl_const_line_is_plausible(line: str) -> bool:
    if not line.endswith(";"):
        return False
    keyword, _sep, rest = line[:-1].partition(" ")
    if keyword not in ("const", "override") or not rest.strip():
        return False
    declaration = rest.split("=", 1)[0].strip()
    name, sep, type_text = declaration.partition(":")
    return (_wgsl_identifier_is_plausible(name.strip())
            and (not sep or _wgsl_type_text_is_plausible(
                type_text.strip())))


def _wgsl_enable_line_is_plausible(line: str) -> bool:
    if not line.endswith(";"):
        return False
    body = line[len("enable "):-1].strip()
    parts = _split_top_level_commas(body)
    return bool(parts) and all(
        _wgsl_identifier_is_plausible(part.strip()) for part in parts)


_C_LIKE_ALLOWED_PREPROCESSOR_PREFIXES = (
    "#define", "#elif", "#else", "#endif", "#if", "#ifdef", "#ifndef",
    "#include", "#line", "#pragma", "#undef",
)


_BackendProofSnapshot = tuple[
    str, ...]


def _backend_result_proof_snapshot(
        result: "MLIRBackendResult") -> _BackendProofSnapshot | None:
    if not isinstance(result.target, MLIRBackendTarget):
        return None
    if not isinstance(result.validation, MLIRValidation):
        return None
    if not isinstance(result.lowering_tool, str) \
            or not result.lowering_tool.strip():
        return None
    if result.output_text is None:
        return None
    if not isinstance(result.output_provenance, tuple):
        return None
    return (
        "target=" + result.target.value,
        "validation_id=" + str(id(result.validation)),
        "lowering_attempted=" + repr(result.lowering_attempted),
        "lowering_passed=" + repr(result.lowering_passed),
        *(f"validation_provenance={entry}"
          for entry in result.validation.provenance),
        "lowering_tool=" + result.lowering_tool,
        *(f"output_provenance={entry}"
          for entry in result.output_provenance),
        "output_sha256="
        + hashlib.sha256(result.output_text.encode("utf-8")).hexdigest(),
    )


def _backend_result_pass_shape_is_coherent(
        result: "MLIRBackendResult",
        expected_snapshot: _BackendProofSnapshot | None = None) -> bool:
    if not _validation_is_real_passed(result.validation):
        return False
    if result.lowering_attempted is not True:
        return False
    if result.lowering_passed is not True:
        return False
    if result.lowering_findings:
        return False
    if result.output_text is None:
        return False
    if not isinstance(result.output_provenance, tuple):
        return False
    snapshot = _backend_result_proof_snapshot(result)
    if snapshot is None:
        return False
    if expected_snapshot is not None and snapshot != expected_snapshot:
        return False
    output_digest = snapshot[-1].removeprefix("output_sha256=")
    return (
        f"output_sha256={output_digest}" in result.output_provenance
        and any(entry.startswith("target_validation=validator=")
                for entry in result.output_provenance)
        and any(entry.startswith("target_validation=predicate=")
                for entry in result.output_provenance)
    )


def _make_backend_result_pass_registry():
    passes: dict[
        int,
        tuple[
            weakref.ReferenceType["MLIRBackendResult"],
            _BackendProofSnapshot,
        ],
    ] = {}

    def _brand(result: "MLIRBackendResult") -> "MLIRBackendResult":
        snapshot = _backend_result_proof_snapshot(result)
        if snapshot is not None:
            passes[id(result)] = (weakref.ref(result), snapshot)
        return result

    def _has(result: "MLIRBackendResult") -> bool:
        if type(result) is not MLIRBackendResult:
            return False
        entry = passes.get(id(result))
        if entry is None:
            return False
        ref, snapshot = entry
        if ref() is not result:
            passes.pop(id(result), None)
            return False
        return snapshot == _backend_result_proof_snapshot(result)

    return _brand, _has


def _validation_is_failed(validation: MLIRValidation) -> bool:
    return (type(validation) is MLIRValidation
            and validation.verdict is MLIRValidationVerdict.FAILED)


def _validation_is_real_passed(validation: MLIRValidation) -> bool:
    return (type(validation) is MLIRValidation
            and _has_real_validation_pass_shape(validation))


def _make_backend_output_validation_registry():
    passes: dict[
        int,
        tuple[
            weakref.ReferenceType["MLIRBackendOutputValidation"],
            tuple[str, ...],
        ],
    ] = {}

    def _payload(
            result: "MLIRBackendOutputValidation") -> tuple[str, ...]:
        return (
            "target=" + result.target.value,
            "output_sha256=" + result.output_sha256,
            "findings=" + repr(result.findings),
            *result.evidence,
        )

    def _brand(
            result: "MLIRBackendOutputValidation"
            ) -> "MLIRBackendOutputValidation":
        passes[id(result)] = (weakref.ref(result), _payload(result))
        return result

    def _has(result: "MLIRBackendOutputValidation") -> bool:
        if type(result) is not MLIRBackendOutputValidation:
            return False
        entry = passes.get(id(result))
        if entry is None:
            return False
        ref, payload = entry
        if ref() is not result:
            passes.pop(id(result), None)
            return False
        return payload == _payload(result)

    return _brand, _has


_backend_output_validation_brand, _has_backend_output_validation_pass_shape = (
    _make_backend_output_validation_registry())


@dataclass(frozen=True, slots=True, weakref_slot=True)
class MLIRBackendOutputValidation:
    """Target-validator result for one backend artifact.

    A clean validator result must carry explicit evidence. Empty
    findings alone are not proof: Stage 213 must not let a placeholder
    validator accidentally bless arbitrary target-looking text.
    """
    target: MLIRBackendTarget
    output_sha256: str
    findings: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        raise TypeError(
            "MLIRBackendOutputValidation is final; subclassing could "
            "bypass target-validator invariants")

    def __post_init__(self) -> None:
        _require_backend_target(self.target)
        if not isinstance(self.output_sha256, str) \
                or not _is_lowercase_sha256(self.output_sha256):
            raise ValueError(
                "MLIRBackendOutputValidation: output_sha256 must be a "
                "lowercase sha256 digest")
        if not isinstance(self.findings, tuple):
            raise ValueError(
                "MLIRBackendOutputValidation: findings must be a tuple")
        if not isinstance(self.evidence, tuple):
            raise ValueError(
                "MLIRBackendOutputValidation: evidence must be a tuple")
        for entry in self.findings:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    "MLIRBackendOutputValidation: findings has a blank "
                    f"or non-str entry {entry!r}")
        for entry in self.evidence:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    "MLIRBackendOutputValidation: evidence has a blank "
                    f"or non-str entry {entry!r}")
            key, sep, value = entry.partition("=")
            if not sep or not key or not value.strip():
                raise ValueError(
                    "MLIRBackendOutputValidation: evidence entries must "
                    f"be key=value proof facts, got {entry!r}")
        if not self.findings and not self.evidence:
            raise ValueError(
                "MLIRBackendOutputValidation: a clean output validation "
                "must carry explicit evidence")
        if not self.findings:
            keys = {entry.partition("=")[0] for entry in self.evidence}
            if not {"validator", "predicate"}.issubset(keys):
                raise ValueError(
                    "MLIRBackendOutputValidation: a clean output "
                    "validation must carry validator=... and "
                    "predicate=... evidence")

    def passed(self) -> bool:
        return (not self.findings
                and _has_backend_output_validation_pass_shape(self))

    def failed(self) -> bool:
        return bool(self.findings)

    def candidate(self) -> bool:
        """True for an evidence-carrying clean validator candidate.

        The backend runner must brand a candidate before it becomes a
        pass. This explicit third state prevents callers from treating
        `not failed()` as acceptance.
        """
        return not self.findings and not self.passed()

    def __copy__(self) -> "MLIRBackendOutputValidation":
        if _has_backend_output_validation_pass_shape(self):
            return self
        return MLIRBackendOutputValidation(
            self.target, self.output_sha256, self.findings, self.evidence)

    def __deepcopy__(
            self,
            memo: dict[int, object] | None = None,
            ) -> "MLIRBackendOutputValidation":
        if _has_backend_output_validation_pass_shape(self):
            if memo is not None:
                memo[id(self)] = self
            return self
        return MLIRBackendOutputValidation(
            self.target, self.output_sha256, self.findings, self.evidence)

    def __reduce__(self) -> object:
        if _has_backend_output_validation_pass_shape(self):
            raise TypeError(
                "passed MLIRBackendOutputValidation cannot be pickled; "
                "the runner-registry target validator mark is not portable")
        return (
            MLIRBackendOutputValidation,
            (self.target, self.output_sha256, self.findings, self.evidence),
        )


# A target pipeline alone is not enough to claim a backend pass: after
# `mlir-opt` runs, a target-specific validator must prove the output is
# the backend-consumable artifact Stage 214+ promised. Stage 213 leaves
# every validator unwired, so production lowering remains DEFERRED even
# if a test or future branch experiments with a non-empty pipeline.
MLIRBackendOutputValidator = Callable[
    [MLIRBackendTarget, str], MLIRBackendOutputValidation]


def _llvm_ir_output_validator(
        target: MLIRBackendTarget,
        output_text: str) -> MLIRBackendOutputValidation:
    """Stage 214 chunk E — the LLVM_IR target output validator.

    Confirms the post-chain artifact is raw LLVM IR via the existing
    `_llvm_ir_artifact_is_plausible` predicate (which has been hardened
    through the Stage-213 audit batches against malformed input).
    Returns a candidate result whose evidence names the validator and
    the predicate so the runner-registry brand can promote it to
    PASSED. A fail returns the named finding, never a silent skip.
    """
    if target is not MLIRBackendTarget.LLVM_IR:
        raise ValueError(
            f"_llvm_ir_output_validator: target must be LLVM_IR, got "
            f"{target.value}")
    output_digest = hashlib.sha256(
        output_text.encode("utf-8")).hexdigest()
    if not _looks_like_backend_output(target, output_text):
        return MLIRBackendOutputValidation(
            target=target,
            output_sha256=output_digest,
            findings=(
                "LLVM_IR target output validator: artifact does not "
                "parse as raw LLVM IR (failed "
                "`_llvm_ir_artifact_is_plausible` shape probe)",),
            evidence=(),
        )
    return MLIRBackendOutputValidation(
        target=target,
        output_sha256=output_digest,
        findings=(),
        evidence=(
            "validator=_llvm_ir_output_validator",
            "predicate=_llvm_ir_artifact_is_plausible",
            f"target={target.value}",
        ),
    )


def _ptx_output_validator(
        target: MLIRBackendTarget,
        output_text: str) -> MLIRBackendOutputValidation:
    """Stage 214 chunk G — the PTX target output validator.

    Confirms the post-chain artifact is PTX text via the existing
    `_ptx_artifact_is_plausible` predicate (which has been hardened
    through the Stage-213 audit batches against malformed PTX
    structures including `.func`/`.entry` forms, predicate guards,
    `.reg`/`.noreturn` directives, and byte-array params). Returns a
    candidate result whose evidence names the validator and the
    predicate so the runner-registry brand can promote it to PASSED.
    A fail returns the named finding, never a silent skip.
    """
    if target is not MLIRBackendTarget.PTX:
        raise ValueError(
            f"_ptx_output_validator: target must be PTX, got "
            f"{target.value}")
    output_digest = hashlib.sha256(
        output_text.encode("utf-8")).hexdigest()
    if not _looks_like_backend_output(target, output_text):
        return MLIRBackendOutputValidation(
            target=target,
            output_sha256=output_digest,
            findings=(
                "PTX target output validator: artifact does not parse "
                "as PTX text (failed `_ptx_artifact_is_plausible` "
                "shape probe)",),
            evidence=(),
        )
    return MLIRBackendOutputValidation(
        target=target,
        output_sha256=output_digest,
        findings=(),
        evidence=(
            "validator=_ptx_output_validator",
            "predicate=_ptx_artifact_is_plausible",
            f"target={target.value}",
        ),
    )


_MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY = MappingProxyType({
        MLIRBackendTarget.LLVM_IR: _llvm_ir_output_validator,
        MLIRBackendTarget.PTX: _ptx_output_validator,
        MLIRBackendTarget.ROCM_HIP: None,
        MLIRBackendTarget.METAL_MSL: None,
        MLIRBackendTarget.WEBGPU_WGSL: None,
})
MLIR_BACKEND_OUTPUT_VALIDATORS = _MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY

# Wall-clock cap on a target pass-pipeline dispatch. Real production
# pipelines should be short for the small modules exercised here; the
# cap is only a dead-tool guard.
_MLIR_BACKEND_PIPELINE_TIMEOUT_S = 60


def _check_mlir_backend_tables() -> None:
    """Module-load drift guard for the Stage 213 backend tables."""
    status_names = {status.name for status in MLIRBackendStatus}
    if status_names != {"PASSED", "FAILED", "DEFERRED"}:
        raise AssertionError(
            "helixc.ir.mlir.backends: MLIRBackendStatus must be "
            "exactly PASSED / FAILED / DEFERRED, got "
            f"{sorted(status_names)}")

    expected = set(MLIRBackendTarget)
    if set(MLIR_BACKEND_TARGETS) != expected:
        raise AssertionError(
            "helixc.ir.mlir.backends: MLIR_BACKEND_TARGETS must cover "
            f"exactly {expected}, got {set(MLIR_BACKEND_TARGETS)}")

    expected_values = {
        "llvm_ir", "ptx", "rocm_hip", "metal_msl", "webgpu_wgsl",
    }
    if {target.value for target in MLIRBackendTarget} != expected_values:
        raise AssertionError(
            "helixc.ir.mlir.backends: MLIRBackendTarget values drifted "
            f"from {expected_values}")

    gpu_expected = set(GPUBackendKind)
    if set(_GPU_BACKEND_TO_MLIR_TARGET_AUTHORITY) != gpu_expected:
        raise AssertionError(
            "helixc.ir.mlir.backends: GPU_BACKEND_TO_MLIR_TARGET keys "
            f"{set(_GPU_BACKEND_TO_MLIR_TARGET_AUTHORITY)} != "
            f"{gpu_expected}")
    gpu_targets = expected - {MLIRBackendTarget.LLVM_IR}
    if set(_GPU_BACKEND_TO_MLIR_TARGET_AUTHORITY.values()) != gpu_targets:
        raise AssertionError(
            "helixc.ir.mlir.backends: GPU_BACKEND_TO_MLIR_TARGET values "
            f"{set(_GPU_BACKEND_TO_MLIR_TARGET_AUTHORITY.values())} != "
            f"{gpu_targets}")

    for table_name, table in (
        ("MLIR_BACKEND_REQUIRED_DIALECTS",
         _MLIR_BACKEND_REQUIRED_DIALECTS_AUTHORITY),
        ("MLIR_BACKEND_LOWERING_PIPELINES",
         _MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY),
        ("MLIR_BACKEND_OUTPUT_VALIDATORS",
         _MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY),
    ):
        if set(table) != expected:
            raise AssertionError(
                f"helixc.ir.mlir.backends: {table_name} keys "
                f"{set(table)} != MLIRBackendTarget members {expected}")

    for target, dialects in _MLIR_BACKEND_REQUIRED_DIALECTS_AUTHORITY.items():
        if not dialects:
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} has no "
                "required dialects")
        if len(dialects) != len(set(dialects)):
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} has duplicate "
                f"dialect entries {dialects}")
        for dialect in dialects:
            if not isinstance(dialect, str) or not dialect.isidentifier():
                raise AssertionError(
                    f"helixc.ir.mlir.backends: {target.name} has a "
                    f"blank / non-identifier dialect {dialect!r}")

    for target, pipeline in _MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY.items():
        if not isinstance(pipeline, tuple):
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} pipeline must "
                f"be a tuple, got {type(pipeline).__name__}")
        for pass_arg in pipeline:
            if not isinstance(pass_arg, str) or not pass_arg.strip():
                raise AssertionError(
                    f"helixc.ir.mlir.backends: {target.name} has a "
                    f"blank / non-str pass argument {pass_arg!r}")
            if pass_arg != pass_arg.strip():
                raise AssertionError(
                    f"helixc.ir.mlir.backends: {target.name} pass "
                    f"argument {pass_arg!r} has leading/trailing "
                    "whitespace")
            if not pass_arg.startswith("--"):
                raise AssertionError(
                    f"helixc.ir.mlir.backends: {target.name} pass "
                    f"argument {pass_arg!r} must start with '--' so it "
                    "is a complete argv token, not an implicit shell "
                    "fragment")

    for target, validator in _MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY.items():
        if validator is not None and not callable(validator):
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} output "
                f"validator must be callable or None, got "
                f"{type(validator).__name__}")

    if set(_MLIR_BACKEND_TRANSLATORS_AUTHORITY) != expected:
        raise AssertionError(
            "helixc.ir.mlir.backends: MLIR_BACKEND_TRANSLATORS keys "
            f"{set(_MLIR_BACKEND_TRANSLATORS_AUTHORITY)} != "
            f"MLIRBackendTarget members {expected}")
    for target, translator in _MLIR_BACKEND_TRANSLATORS_AUTHORITY.items():
        if translator is None:
            continue
        if not isinstance(translator, tuple) or len(translator) != 3:
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} translator "
                "must be a (tool_name, flag, follow_up_args) tuple "
                f"or None, got {translator!r}")
        tool_name, flag, follow_up = translator
        if not isinstance(tool_name, str) or not tool_name.strip() \
                or tool_name != tool_name.strip():
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} translator "
                f"tool_name {tool_name!r} must be a non-blank "
                "whitespace-stripped string")
        if not isinstance(flag, str) or not flag.strip() \
                or flag != flag.strip() or not flag.startswith("--"):
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} translator "
                f"flag {flag!r} must be a non-blank, whitespace-stripped "
                "argv token starting with '--'")
        if not isinstance(follow_up, tuple):
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} translator "
                f"follow_up_args must be a tuple, got "
                f"{type(follow_up).__name__}")
        for arg in follow_up:
            if not isinstance(arg, str) or not arg.strip() \
                    or arg != arg.strip():
                raise AssertionError(
                    f"helixc.ir.mlir.backends: {target.name} "
                    f"translator follow-up arg {arg!r} must be a "
                    "non-blank, whitespace-stripped string")


_check_mlir_backend_tables()


def _require_backend_target(target: MLIRBackendTarget) -> MLIRBackendTarget:
    if not isinstance(target, MLIRBackendTarget):
        raise ValueError(
            f"unknown MLIR backend target {target!r}; expected one of "
            f"{list(MLIRBackendTarget)}")
    return target


def backend_required_dialects(
        target: MLIRBackendTarget) -> tuple[str, ...]:
    """Return the MLIR dialect contract for a backend target."""
    target = _require_backend_target(target)
    return _MLIR_BACKEND_REQUIRED_DIALECTS_AUTHORITY[target]


def backend_translator(
        target: MLIRBackendTarget,
) -> tuple[str, str, tuple[str, ...]] | None:
    """Return the Stage 214 translator-step descriptor for a backend
    target, or None when not yet wired.

    The translator step runs AFTER the `mlir-opt` lowering pipeline.
    Its output is the raw target artifact the downstream consumer
    reads (LLVM IR for `LLVM_IR`, PTX text for `PTX`, etc.). A None
    means the target's lowering still ends at `mlir-opt`'s LLVM /
    NVVM / ROCDL / SPIR-V dialect output, which the Stage-213 runner
    refuses to mint as PASSED.

    Returns a tuple `(tool_name, flag, follow_up_args)`:
    `tool_name` is the executable (e.g. `"mlir-translate"`); `flag` is
    the leading argv flag (must start with `--`); `follow_up_args` is
    a tuple of any further argv tokens for a chained tool (e.g.
    `("llc", "-mtriple=nvptx64", "-mcpu=sm_80")`)."""
    target = _require_backend_target(target)
    return _MLIR_BACKEND_TRANSLATORS_AUTHORITY[target]


def backend_lowering_pipeline(
        target: MLIRBackendTarget) -> tuple[str, ...]:
    """Return the Stage 214 pass pipeline for a backend target.

    Stage 213 chunk A intentionally returns an empty tuple for every
    target. Empty means "not wired", not "no passes needed".
    """
    target = _require_backend_target(target)
    return _MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY[target]


def mlir_target_for_gpu_backend(
        backend: GPUBackendKind) -> MLIRBackendTarget:
    """Map an existing GPU backend enum to the Stage 213 MLIR target."""
    if not isinstance(backend, GPUBackendKind):
        raise ValueError(
            f"unknown GPU backend {backend!r}; expected one of "
            f"{list(GPUBackendKind)}")
    return _GPU_BACKEND_TO_MLIR_TARGET_AUTHORITY[backend]


def _run_mlir_translate_step(
        dialect_mlir_text: str,
        *,
        mlir_translate: str,
        flag: str,
        timeout_s: int = _MLIR_BACKEND_PIPELINE_TIMEOUT_S,
) -> tuple[str | None, tuple[str, ...]]:
    """Run `mlir-translate` to convert dialect-MLIR text into the raw
    target artifact AS TEXT (e.g. raw LLVM IR via `--mlir-to-llvmir`).

    Text-only by design — the artifact is read with `encoding="utf-8"`
    at the end, so flags that produce binary output (e.g. SPIR-V
    serialization, cubin) are not supported by this helper. A binary
    chain step needs a separate sibling helper that reads `"rb"`.

    Returns `(output_text, findings)`:
    - on success: `(output_text, ())` with output_text a non-blank
      raw target artifact;
    - on failure: `(None, (one-or-more finding strings,))`.

    Stage 214 chunk C — the foundational helper. Same subprocess
    hygiene `_run_mlir_opt_validate` / `_run_mlir_opt_pipeline` use
    applies: argv-list dispatch, explicit timeout, captured Timeout /
    OSError / nonzero diagnostics, and a non-empty output artifact
    requirement.
    """
    if not isinstance(dialect_mlir_text, str) or not dialect_mlir_text.strip():
        return None, (
            "_run_mlir_translate_step: dialect_mlir_text must be "
            "non-empty text",)
    if not isinstance(mlir_translate, str) or not mlir_translate.strip() \
            or mlir_translate != mlir_translate.strip():
        return None, (
            "_run_mlir_translate_step: mlir_translate must be a "
            f"non-blank, whitespace-stripped path, got {mlir_translate!r}",)
    if not isinstance(flag, str) or not flag.strip() \
            or flag != flag.strip() or not flag.startswith("--"):
        return None, (
            "_run_mlir_translate_step: flag must be a non-blank, "
            "whitespace-stripped argv token starting with '--', got "
            f"{flag!r}",)
    if ((not isinstance(timeout_s, (int, float)))
            or isinstance(timeout_s, bool) or timeout_s <= 0):
        return None, (
            "_run_mlir_translate_step: timeout_s must be a positive "
            f"number, got {timeout_s!r}",)

    with tempfile.TemporaryDirectory(prefix="helix_mlir_translate_") as tmpdir:
        in_path = os.path.join(tmpdir, "dialect.mlir")
        out_path = os.path.join(tmpdir, "artifact.out")
        try:
            with open(in_path, "w", encoding="utf-8") as f:
                f.write(dialect_mlir_text)
        except (OSError, UnicodeError, ValueError) as exc:
            return None, (
                f"_run_mlir_translate_step: could not write temp "
                f"input {in_path!r} ({type(exc).__name__}: {exc})",)
        try:
            proc = subprocess.run(
                [mlir_translate, flag, in_path, "-o", out_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, (
                f"_run_mlir_translate_step: mlir-translate timed out "
                f"after {timeout_s}s",)
        except (OSError, UnicodeError, ValueError) as exc:
            return None, (
                f"_run_mlir_translate_step: tool unusable at invocation "
                f"({type(exc).__name__}: {exc})",)
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip().splitlines()[-3:]
            stderr_snippet = " | ".join(stderr_tail) if stderr_tail else ""
            return None, (
                f"_run_mlir_translate_step: mlir-translate exited "
                f"{proc.returncode}"
                + (f" — stderr: {stderr_snippet}" if stderr_snippet else ""),)
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                output_text = f.read()
        except (OSError, UnicodeError) as exc:
            return None, (
                f"_run_mlir_translate_step: could not read output "
                f"artifact {out_path!r} "
                f"({type(exc).__name__}: {exc})",)
        if not output_text.strip():
            return None, (
                "_run_mlir_translate_step: mlir-translate exited 0 but "
                "produced blank output",)
        return output_text, ()


def _run_chained_tool_step(
        input_text: str,
        *,
        tool_path: str,
        args: tuple[str, ...],
        timeout_s: int = _MLIR_BACKEND_PIPELINE_TIMEOUT_S,
) -> tuple[str | None, tuple[str, ...]]:
    """Run a chained third-stage tool (typically `llc`) on the raw
    text output of `mlir-translate`. Same subprocess hygiene as
    `_run_mlir_translate_step`. Returns `(output_text, findings)` —
    on success: `(output_text, ())`; on failure: `(None, (finding,...))`.

    Stage 214 chunk F — the chained-tool helper. Invocation shape is
    `[tool_path, *args, "-o", out_path, in_path]`, which works for
    `llc` and most modern target assemblers (the canonical `llc`
    usage `llc -mtriple=nvptx64 -mcpu=sm_80 -o out.ptx in.ll`).
    """
    if not isinstance(input_text, str) or not input_text.strip():
        return None, (
            "_run_chained_tool_step: input_text must be non-empty text",)
    if not isinstance(tool_path, str) or not tool_path.strip() \
            or tool_path != tool_path.strip():
        return None, (
            "_run_chained_tool_step: tool_path must be a non-blank, "
            f"whitespace-stripped path, got {tool_path!r}",)
    if not isinstance(args, tuple):
        return None, (
            "_run_chained_tool_step: args must be a tuple of argv "
            f"tokens, got {type(args).__name__}",)
    for arg in args:
        if not isinstance(arg, str) or not arg.strip() \
                or arg != arg.strip():
            return None, (
                "_run_chained_tool_step: each arg must be a non-blank, "
                f"whitespace-stripped string, got {arg!r}",)
    if ((not isinstance(timeout_s, (int, float)))
            or isinstance(timeout_s, bool) or timeout_s <= 0):
        return None, (
            "_run_chained_tool_step: timeout_s must be a positive "
            f"number, got {timeout_s!r}",)

    with tempfile.TemporaryDirectory(prefix="helix_mlir_chained_") as tmpdir:
        in_path = os.path.join(tmpdir, "input.txt")
        out_path = os.path.join(tmpdir, "artifact.out")
        try:
            with open(in_path, "w", encoding="utf-8") as f:
                f.write(input_text)
        except (OSError, UnicodeError, ValueError) as exc:
            return None, (
                f"_run_chained_tool_step: could not write temp input "
                f"{in_path!r} ({type(exc).__name__}: {exc})",)
        try:
            proc = subprocess.run(
                [tool_path, *args, "-o", out_path, in_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, (
                f"_run_chained_tool_step: chained tool timed out after "
                f"{timeout_s}s",)
        except (OSError, UnicodeError, ValueError) as exc:
            return None, (
                f"_run_chained_tool_step: tool unusable at invocation "
                f"({type(exc).__name__}: {exc})",)
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip().splitlines()[-3:]
            stderr_snippet = " | ".join(stderr_tail) if stderr_tail else ""
            return None, (
                f"_run_chained_tool_step: chained tool exited "
                f"{proc.returncode}"
                + (f" — stderr: {stderr_snippet}"
                   if stderr_snippet else ""),)
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                output_text = f.read()
        except (OSError, UnicodeError) as exc:
            return None, (
                f"_run_chained_tool_step: could not read output "
                f"artifact {out_path!r} "
                f"({type(exc).__name__}: {exc})",)
        if not output_text.strip():
            return None, (
                "_run_chained_tool_step: chained tool exited 0 but "
                "produced blank output",)
        return output_text, ()


def _run_mlir_opt_pipeline(
        mlir_text: str,
        *,
        target: MLIRBackendTarget,
        validation: MLIRValidation,
        mlir_opt: str,
        pipeline: tuple[str, ...],
        output_validator: MLIRBackendOutputValidator,
        timeout_s: int = _MLIR_BACKEND_PIPELINE_TIMEOUT_S,
        mlir_translate: str | None = None,
        chained_tool: str | None = None,
        _brand_output_validation: Callable[
            ["MLIRBackendOutputValidation"], "MLIRBackendOutputValidation"
        ] | None = None,
) -> "MLIRBackendResult":
    """Run a declared `mlir-opt` target lowering pipeline.

    This is the Stage 213 chunk-C runner contract. It only runs after
    real validation has PASSED. A 0 exit is not enough: the output
    artifact must exist, be non-empty, and read back as text before the
    backend result can be PASSED.

    Stage 214 chunk D — translator-step chaining. When the target's
    `_MLIR_BACKEND_TRANSLATORS_AUTHORITY` entry is populated, the
    `mlir-opt` output is dialect-MLIR (the EXPECTED shape, not the
    Stage-213 reject case). The runner then invokes
    `_run_mlir_translate_step` to convert dialect-MLIR into the raw
    target artifact, and downstream validation operates on the
    translated text. `mlir_translate` is the tool path (typically
    `support.mlir_translate`); it must be present when the translator
    entry is populated and absent / unused when it is None.
    """
    target = _require_backend_target(target)
    if not isinstance(mlir_text, str) or not mlir_text.strip():
        raise ValueError(
            "_run_mlir_opt_pipeline: mlir_text must be non-empty text")
    if type(validation) is not MLIRValidation:
        raise ValueError(
            "_run_mlir_opt_pipeline: validation must be an "
            f"MLIRValidation result, got {validation!r}")
    if not _validation_is_real_passed(validation):
        raise ValueError(
            "_run_mlir_opt_pipeline: validation must be PASSED before "
            "a backend lowering pipeline can run")
    if not isinstance(mlir_opt, str) or not mlir_opt.strip():
        raise ValueError(
            "_run_mlir_opt_pipeline: mlir_opt must be a non-empty "
            f"string, got {mlir_opt!r}")
    mlir_opt = mlir_opt.strip()
    try:
        input_digest = hashlib.sha256(mlir_text.encode("utf-8")).hexdigest()
    except UnicodeError as exc:
        return MLIRBackendResult(
            target=target,
            validation=validation,
            lowering_attempted=True,
            lowering_passed=False,
            lowering_tool=mlir_opt,
            lowering_findings=(
                f"could not encode MLIR input "
                f"({type(exc).__name__}: {exc})",),
            output_text=None,
        )
    if f"input_sha256={input_digest}" not in validation.provenance:
        raise ValueError(
            "_run_mlir_opt_pipeline: validation provenance does not "
            "match mlir_text")
    if f"mlir-opt={mlir_opt}" not in validation.provenance:
        raise ValueError(
            "_run_mlir_opt_pipeline: validation provenance does not "
            "match the lowering tool path")
    if not isinstance(pipeline, tuple) or not pipeline:
        raise ValueError(
            "_run_mlir_opt_pipeline: pipeline must be a non-empty tuple")
    for pass_arg in pipeline:
        if not isinstance(pass_arg, str) or not pass_arg.strip():
            raise ValueError(
                "_run_mlir_opt_pipeline: pipeline has a blank or "
                f"non-str pass argument {pass_arg!r}")
        if pass_arg != pass_arg.strip():
            raise ValueError(
                "_run_mlir_opt_pipeline: pipeline pass argument "
                f"{pass_arg!r} must not have leading/trailing whitespace")
        if not pass_arg.startswith("--"):
            raise ValueError(
                "_run_mlir_opt_pipeline: pipeline pass argument "
                f"{pass_arg!r} must start with '--'")
    registered_pipeline = _MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY[target]
    if pipeline != registered_pipeline:
        raise ValueError(
            "_run_mlir_opt_pipeline: pipeline must match the registered "
            f"lowering pipeline for {target.value}")
    if not callable(output_validator):
        raise ValueError(
            "_run_mlir_opt_pipeline: output_validator must be callable")
    registered_validator = _MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY[target]
    if output_validator is not registered_validator:
        raise ValueError(
            "_run_mlir_opt_pipeline: output_validator must be the "
            f"registered validator for {target.value}")
    if ((not isinstance(timeout_s, (int, float)))
            or isinstance(timeout_s, bool)
            or timeout_s <= 0):
        raise ValueError(
            "_run_mlir_opt_pipeline: timeout_s must be a positive number")
    registered_translator = _MLIR_BACKEND_TRANSLATORS_AUTHORITY[target]
    if registered_translator is None and mlir_translate is not None:
        raise ValueError(
            f"_run_mlir_opt_pipeline: target {target.value} has no "
            "registered translator; mlir_translate must be None when "
            "no translator-step is declared")
    if registered_translator is None and chained_tool is not None:
        raise ValueError(
            f"_run_mlir_opt_pipeline: target {target.value} has no "
            "registered translator; chained_tool must be None when "
            "no chain is declared")
    if registered_translator is not None:
        # Defensive runtime shape-check — the module-load drift guard
        # cannot see a monkeypatched authority table. tool_name is
        # METADATA ONLY (the runner trusts `mlir_translate` from the
        # caller, not the table); flag has independent re-validation
        # downstream; follow_up_args' first element is the chained tool
        # name (validated below via the chained_tool path requirement).
        if not isinstance(registered_translator, tuple) \
                or len(registered_translator) != 3:
            raise ValueError(
                f"_run_mlir_opt_pipeline: registered translator for "
                f"{target.value} must be a 3-tuple "
                f"(tool_name, flag, follow_up_args), got "
                f"{registered_translator!r}")
        if not isinstance(mlir_translate, str) \
                or not mlir_translate.strip() \
                or mlir_translate != mlir_translate.strip():
            raise ValueError(
                f"_run_mlir_opt_pipeline: target {target.value} has a "
                "registered translator; mlir_translate must be a "
                "non-blank, whitespace-stripped path")
        _follow_up = registered_translator[2]
        if _follow_up:
            if not isinstance(chained_tool, str) \
                    or not chained_tool.strip() \
                    or chained_tool != chained_tool.strip():
                raise ValueError(
                    f"_run_mlir_opt_pipeline: target {target.value} "
                    "declares chained follow_up_args; chained_tool must "
                    "be a non-blank, whitespace-stripped path")
        elif chained_tool is not None:
            raise ValueError(
                f"_run_mlir_opt_pipeline: target {target.value} "
                "translator has empty follow_up_args; chained_tool must "
                "be None when no chained step is declared")

    with tempfile.TemporaryDirectory(prefix="helix_mlir_backend_") as tmpdir:
        mlir_path = os.path.join(tmpdir, "module.mlir")
        out_path = os.path.join(tmpdir, "lowered.mlir")
        try:
            with open(mlir_path, "w", encoding="utf-8") as f:
                f.write(mlir_text)
        except (OSError, UnicodeError, ValueError) as exc:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"could not write temp MLIR input {mlir_path!r} "
                    f"({type(exc).__name__}: {exc})",),
                output_text=None,
            )

        try:
            proc = subprocess.run(
                [mlir_opt, *pipeline, mlir_path, "-o", out_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value} "
                    f"timed out after {timeout_s}s",),
                output_text=None,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value}: "
                    f"tool unusable at invocation ({type(exc).__name__}: "
                    f"{exc})",),
                output_text=None,
            )

        if proc.returncode != 0:
            diag = (proc.stderr or "").strip() \
                or (proc.stdout or "").strip()
            if not diag:
                diag = "no diagnostic emitted"
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value} exit "
                    f"{proc.returncode}: {diag[:500]}",),
                output_text=None,
            )
        zero_exit_diag = _captured_tool_diagnostic(proc.stdout, proc.stderr)
        if zero_exit_diag:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value} "
                    "exited 0 but emitted a diagnostic: "
                    f"{zero_exit_diag}",),
                output_text=None,
            )

        try:
            size = os.path.getsize(out_path)
        except OSError:
            size = -1
        if size <= 0:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value} "
                    f"exited 0 but produced no output artifact at "
                    f"{out_path!r} - a 0 exit with no artifact is not "
                    "a backend pass",),
                output_text=None,
            )

        try:
            with open(out_path, "r", encoding="utf-8") as f:
                output_text = f.read()
        except (OSError, UnicodeError) as exc:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"could not read backend output artifact {out_path!r} "
                    f"({type(exc).__name__}: {exc})",),
                output_text=None,
            )

        if not output_text.strip():
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value} "
                    "exited 0 but produced only blank output",),
                output_text=None,
            )
        translator_tool: str | None = None
        chained_tool_used: str | None = None
        if registered_translator is not None:
            _tool_name, translator_flag, follow_up_args = registered_translator
            translated_text, translate_findings = _run_mlir_translate_step(
                output_text,
                mlir_translate=mlir_translate,  # type: ignore[arg-type]
                flag=translator_flag,
                timeout_s=timeout_s,
            )
            if translate_findings:
                return MLIRBackendResult(
                    target=target,
                    validation=validation,
                    lowering_attempted=True,
                    lowering_passed=False,
                    lowering_tool=mlir_opt,
                    lowering_findings=(
                        f"mlir-translate step for {target.value} failed: "
                        + translate_findings[0],
                    ),
                    output_text=None,
                )
            if translated_text is None:
                return MLIRBackendResult(
                    target=target,
                    validation=validation,
                    lowering_attempted=True,
                    lowering_passed=False,
                    lowering_tool=mlir_opt,
                    lowering_findings=(
                        f"mlir-translate step for {target.value} produced "
                        "no output but also no findings; refusing to mint "
                        "a backend pass",),
                    output_text=None,
                )
            output_text = translated_text
            translator_tool = mlir_translate
            if follow_up_args:
                # Stage 214 chunk F — chained third-stage tool. The first
                # entry of follow_up_args names the tool; the rest are
                # passed as argv. The path is resolved upstream (in
                # `lower_mlir_to_backend`) and passed through here.
                if chained_tool is None:
                    return MLIRBackendResult(
                        target=target,
                        validation=validation,
                        lowering_attempted=True,
                        lowering_passed=False,
                        lowering_tool=mlir_opt,
                        lowering_findings=(
                            f"chained tool step for {target.value} requires "
                            f"the {follow_up_args[0]!r} tool path but none "
                            "was provided to the runner",),
                        output_text=None,
                    )
                chained_text, chained_findings = _run_chained_tool_step(
                    output_text,
                    tool_path=chained_tool,
                    args=follow_up_args[1:],
                    timeout_s=timeout_s,
                )
                if chained_findings:
                    return MLIRBackendResult(
                        target=target,
                        validation=validation,
                        lowering_attempted=True,
                        lowering_passed=False,
                        lowering_tool=mlir_opt,
                        lowering_findings=(
                            f"chained-tool step for {target.value} "
                            f"({follow_up_args[0]}) failed: "
                            + chained_findings[0],
                        ),
                        output_text=None,
                    )
                if chained_text is None:
                    return MLIRBackendResult(
                        target=target,
                        validation=validation,
                        lowering_attempted=True,
                        lowering_passed=False,
                        lowering_tool=mlir_opt,
                        lowering_findings=(
                            f"chained-tool step for {target.value} "
                            f"({follow_up_args[0]}) produced no output "
                            "but also no findings; refusing to mint a "
                            "backend pass",),
                        output_text=None,
                    )
                output_text = chained_text
                chained_tool_used = chained_tool
        if _looks_like_mlir_pipeline_output(output_text):
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    (f"mlir-translate step for {target.value} produced "
                     "MLIR, not a raw target artifact; the translator "
                     "flag may be wrong")
                    if translator_tool is not None
                    else (f"mlir-opt backend pipeline for {target.value} "
                          "exited 0 but produced MLIR, not a target "
                          "artifact; the artifact translation step is "
                          "not wired"),),
                output_text=None,
            )
        if not _looks_like_backend_output(target, output_text):
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value} "
                    "exited 0 but produced output that does not match "
                    "the target artifact shape",),
                output_text=None,
            )
        correspondence_finding = _backend_output_symbol_finding(
            mlir_text, target, output_text)
        if correspondence_finding is not None:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(correspondence_finding,),
                output_text=None,
            )
        try:
            output_validation = output_validator(target, output_text)
        except Exception as exc:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"target output validator for {target.value} raised "
                    f"{type(exc).__name__}: {exc}",),
                output_text=None,
            )
        if type(output_validation) is not MLIRBackendOutputValidation:
            raise ValueError(
                "_run_mlir_opt_pipeline: output_validator must return an "
                "MLIRBackendOutputValidation result, got "
                f"{type(output_validation).__name__}")
        if output_validation.target is not target:
            raise ValueError(
                "_run_mlir_opt_pipeline: output_validator returned "
                f"validation for {output_validation.target.value}, "
                f"expected {target.value}")
        output_digest = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        if output_validation.output_sha256 != output_digest:
            raise ValueError(
                "_run_mlir_opt_pipeline: output_validator digest does "
                "not match output_text")
        if output_validation.failed():
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=output_validation.findings,
                output_text=None,
            )
        if _brand_output_validation is None:
            raise AssertionError(
                "_run_mlir_opt_pipeline: missing private output "
                "validation brand")
        output_validation = _brand_output_validation(output_validation)
        if not output_validation.passed():
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool=mlir_opt,
                lowering_findings=(
                    f"target output validator for {target.value} returned "
                    "an unbranded clean result",),
                output_text=None,
            )
        if translator_tool is not None:
            _tool_name, translator_flag, follow_up_args = (
                registered_translator)
            translator_provenance = (
                f"mlir-translate={translator_tool}",
                f"mlir-translate-flag={translator_flag}",
            )
            if chained_tool_used is not None:
                chained_provenance = (
                    f"chained-tool={chained_tool_used}",
                    f"chained-tool-name={follow_up_args[0]}",
                    "chained-tool-args=" + " ".join(follow_up_args[1:]),
                )
            else:
                chained_provenance = ()
        else:
            translator_provenance = ()
            chained_provenance = ()
        output_provenance = (
            f"mlir-opt={mlir_opt}",
            "pipeline=" + " ".join(pipeline),
            *translator_provenance,
            *chained_provenance,
            f"output_sha256={output_digest}",
            *(f"target_validation={entry}"
              for entry in output_validation.evidence),
        )

    result = object.__new__(MLIRBackendResult)
    object.__setattr__(result, "target", target)
    object.__setattr__(result, "validation", validation)
    object.__setattr__(result, "lowering_attempted", True)
    object.__setattr__(result, "lowering_passed", True)
    object.__setattr__(result, "lowering_tool", mlir_opt)
    object.__setattr__(result, "lowering_findings", ())
    object.__setattr__(result, "output_text", output_text)
    object.__setattr__(result, "output_provenance", output_provenance)
    return result


class _BackendOutputValidationBrandingRunner:
    __slots__ = ("__raw_runner", "__brand_output_validation")

    def __init__(self, raw_runner, brand_output_validation) -> None:
        self.__raw_runner = raw_runner
        self.__brand_output_validation = brand_output_validation

    def __call__(
            self,
            mlir_text: str,
            *,
            target: MLIRBackendTarget,
            validation: MLIRValidation,
            mlir_opt: str,
            pipeline: tuple[str, ...],
            output_validator: MLIRBackendOutputValidator,
            timeout_s: int = _MLIR_BACKEND_PIPELINE_TIMEOUT_S,
            mlir_translate: str | None = None,
            chained_tool: str | None = None,
            ) -> "MLIRBackendResult":
        return self.__raw_runner(
            mlir_text,
            target=target,
            validation=validation,
            mlir_opt=mlir_opt,
            pipeline=pipeline,
            output_validator=output_validator,
            timeout_s=timeout_s,
            mlir_translate=mlir_translate,
            chained_tool=chained_tool,
            _brand_output_validation=self.__brand_output_validation,
        )


_run_mlir_opt_pipeline = _BackendOutputValidationBrandingRunner(
    _run_mlir_opt_pipeline, _backend_output_validation_brand)
del _backend_output_validation_brand


def _looks_like_mlir_pipeline_output(output_text: str) -> bool:
    validation = mock_validate_mlir(output_text)
    return validation.deferred() or validation.passed()


def _backend_output_symbol_finding(
        mlir_text: str,
        target: MLIRBackendTarget,
        output_text: str) -> str | None:
    input_symbols, require_kernel_entries = _backend_input_symbol_binding(
        mlir_text, target)
    if not input_symbols:
        if _mlir_text_declares_body_form_function_shape(mlir_text):
            return (
                f"mlir-opt backend pipeline for {target.value} exited 0 "
                "but the input declares body-form function-shape ops the "
                "structural symbol extractor cannot bind; refusing to "
                "mint a backend pass without a verifiable symbol "
                "correspondence")
        return None
    ptx_entry_finding = _ptx_entry_symbol_binding_finding(
        input_symbols, target, output_text, require_kernel_entries)
    if ptx_entry_finding is not None:
        return ptx_entry_finding
    output_symbols = _backend_output_defined_symbols(
        target, output_text, kernel_entry_only=require_kernel_entries)
    missing = tuple(symbol for symbol in input_symbols
                    if symbol not in output_symbols)
    if missing:
        return (
            f"mlir-opt backend pipeline for {target.value} exited 0 "
            "but the target artifact is missing lowered input function "
            "definitions: " + ", ".join(missing[:5]))
    return None


_BODY_FORM_FUNCTION_SHAPE_TOKENS = (
    "func.func", "llvm.func", "gpu.func",
)
_BODY_FORM_FUNCTION_SHAPE_GENERIC_TOKENS = (
    '"func.func"', '"llvm.func"', '"gpu.func"',
)


def _mlir_text_declares_body_form_function_shape(mlir_text: str) -> bool:
    """Return True iff the input MLIR contains a body-form
    function-shape op (custom or generic) that the symbol extractor
    might not have picked up. The walker is intentionally cheap — it
    only confirms the *presence* of such an op so the symbol-binding
    gate can fail closed rather than silently bind nothing.

    Sibling-dialect coverage (`spirv.func`, `nvvm.kernel`, etc.) is
    deliberately out of scope: those targets are not in
    `MLIRBackendTarget` yet, so a declaration of them is itself a
    backend-shape mismatch that the upstream contract refuses earlier.
    """
    structural = _comment_stripped_text(mlir_text)
    i = 0
    while i < len(structural):
        if structural[i] == '"':
            quoted_end = _quoted_span_end(structural, i)
            for token in _BODY_FORM_FUNCTION_SHAPE_GENERIC_TOKENS:
                if structural.startswith(token, i):
                    if _generic_body_form_func_has_body(
                            structural, i + len(token)):
                        return True
            i = quoted_end
            continue
        matched = False
        for token in _BODY_FORM_FUNCTION_SHAPE_TOKENS:
            if not _bare_word_at(structural, i, token):
                continue
            if _custom_body_form_func_has_body(
                    structural, i + len(token)):
                return True
            i += len(token)
            matched = True
            break
        if not matched:
            i += 1
    return False


def _custom_body_form_func_has_body(
        structural: str, start: int) -> bool:
    """Custom-form function-shape op has a body iff a non-empty
    `{ ... }` follows the args parenthetical. The walker is generic
    over func.func / llvm.func / gpu.func — the latter three all
    share the bare grammar `<op> [modifiers] @symbol ( ... ) [tail] { ... }`."""
    line_end = _next_op_boundary(structural, start, len(structural))
    cursor = _skip_spaces(structural, start)
    while cursor < line_end:
        word, after = _read_bare_word(structural, cursor)
        if not word or word not in _LLVM_FUNC_MODIFIER_WORDS \
                and word not in ("private", "public", "nested"):
            break
        cursor = _skip_spaces(structural, after)
    if cursor >= line_end or structural[cursor] != "@":
        return False
    symbol = _mlir_symbol_ref_at(structural, cursor, line_end)
    if symbol is None:
        return False
    cursor = _skip_spaces(structural, symbol[1])
    if cursor >= len(structural) or structural[cursor] != "(":
        return False
    paren_end = _matching_closer_index(structural, cursor, "(", ")")
    if paren_end is None:
        return False
    return _llvm_func_custom_has_region_after(structural, paren_end + 1)


def _generic_body_form_func_has_body(
        structural: str, start: int) -> bool:
    """Generic-form body iff the trailing region operand contains a
    non-empty `{ ... }`."""
    cursor = _skip_spaces(structural, start)
    if cursor >= len(structural) or structural[cursor] != "(":
        return False
    operands_end = _matching_closer_index(structural, cursor, "(", ")")
    if operands_end is None:
        return False
    props = _generic_property_dict_after(structural, operands_end + 1)
    props_end = props[1] if props is not None else operands_end + 1
    return _generic_llvm_func_has_body(structural, props_end)


def _ptx_entry_symbol_binding_finding(
        input_symbols: tuple[str, ...],
        target: MLIRBackendTarget,
        output_text: str,
        require_kernel_entries: bool) -> str | None:
    if target is not MLIRBackendTarget.PTX or require_kernel_entries:
        return None
    entry_symbols = _ptx_entry_symbols(
        output_text, include_device_functions=False)
    required = ("main",) if "main" in input_symbols else (
        input_symbols if len(input_symbols) == 1 else ())
    missing = tuple(symbol for symbol in required
                    if symbol not in entry_symbols)
    if missing:
        return (
            "mlir-opt backend pipeline for ptx exited 0 but the target "
            "artifact is missing lowered PTX entry definitions: "
            + ", ".join(missing[:5]))
    if not required and entry_symbols.isdisjoint(input_symbols):
        return (
            "mlir-opt backend pipeline for ptx exited 0 but the target "
            "artifact has no PTX entry corresponding to input functions")
    return None


def _backend_input_function_symbols(
        mlir_text: str,
        target: MLIRBackendTarget) -> tuple[str, ...]:
    return _backend_input_symbol_binding(mlir_text, target)[0]


def _backend_input_symbol_binding(
        mlir_text: str,
        target: MLIRBackendTarget) -> tuple[tuple[str, ...], bool]:
    if target in _GPU_MLIR_BACKEND_TARGETS:
        gpu_kernel_symbols = _mlir_gpu_kernel_symbols(mlir_text)
        if gpu_kernel_symbols:
            return gpu_kernel_symbols, True
    return tuple(dict.fromkeys(_mlir_defined_function_symbols(mlir_text))), False


def _backend_output_defined_symbols(
        target: MLIRBackendTarget,
        output_text: str, *,
        kernel_entry_only: bool = False) -> frozenset[str]:
    if not _looks_like_backend_output(target, output_text):
        return frozenset()
    if target is MLIRBackendTarget.LLVM_IR:
        return _llvm_ir_defined_function_symbols(output_text)
    if target is MLIRBackendTarget.PTX:
        return _ptx_entry_symbols(
            output_text, include_device_functions=not kernel_entry_only)
    if target is MLIRBackendTarget.ROCM_HIP:
        return _rocm_hip_defined_kernel_symbols(output_text)
    if target is MLIRBackendTarget.METAL_MSL:
        return _metal_msl_defined_kernel_symbols(output_text)
    if target is MLIRBackendTarget.WEBGPU_WGSL:
        return _webgpu_wgsl_defined_compute_symbols(output_text)
    return frozenset()


_GPU_MLIR_BACKEND_TARGETS = frozenset((
    MLIRBackendTarget.PTX,
    MLIRBackendTarget.ROCM_HIP,
    MLIRBackendTarget.METAL_MSL,
    MLIRBackendTarget.WEBGPU_WGSL,
))


def _mlir_defined_function_symbols(mlir_text: str) -> tuple[str, ...]:
    symbols: list[str] = []
    for interface in sorted(_mlir_func_interfaces(mlir_text)):
        fields = _mlir_func_interface_fields(interface)
        if fields is not None and fields[-1] == "body":
            symbol = fields[0]
            if symbol.startswith("@"):
                symbols.append(_llvm_symbol_from_mlir_symbol(symbol))
    symbols.extend(_mlir_defined_llvm_func_symbols(mlir_text))
    return tuple(dict.fromkeys(symbols))


def _mlir_defined_llvm_func_symbols(mlir_text: str) -> tuple[str, ...]:
    """Extract LLVM-dialect `llvm.func` body-form symbols from both
    custom (`llvm.func @sym() { ... }`) and generic
    (`"llvm.func"() <{sym_name = "sym"}> ({ ... })`) syntactic forms.

    Without this, a malformed input that declares `llvm.func @expected`
    in generic form alongside a backend artifact that emits a wholly
    unrelated symbol would clear the symbol-binding gate because the
    input's required-symbol set comes back empty."""
    structural = _comment_stripped_text(mlir_text)
    symbols: list[str] = []
    i = 0
    while i < len(structural):
        if structural[i] == '"':
            i = _quoted_span_end(structural, i)
            continue
        if not _bare_word_at(structural, i, "llvm.func"):
            i += 1
            continue
        symbol = _llvm_func_custom_symbol_at(structural, i)
        if symbol:
            symbols.append(_llvm_symbol_from_mlir_symbol(symbol))
        i += len("llvm.func")
    symbols.extend(_mlir_generic_llvm_func_symbols(mlir_text))
    return tuple(dict.fromkeys(symbols))


_LLVM_FUNC_MODIFIER_WORDS = frozenset((
    # LLVM::Linkage enum (custom-form spellings).
    "private", "internal", "available_externally", "linkonce",
    "linkonce_odr", "weak", "weak_odr", "common", "appending",
    "extern_weak", "external",
    # Visibility, addr-space hints, dso scope (MLIR may emit any of
    # these between `llvm.func` and `@symbol`).
    "default", "hidden", "protected",
    "unnamed_addr", "local_unnamed_addr", "dso_local", "dso_preemptable",
    "public", "nested",
))


def _llvm_func_custom_symbol_at(
        structural: str, token_start: int) -> str | None:
    line_end = _next_op_boundary(structural, token_start, len(structural))
    cursor = _skip_spaces(structural, token_start + len("llvm.func"))
    while cursor < line_end:
        word, after = _read_bare_word(structural, cursor)
        if not word or word not in _LLVM_FUNC_MODIFIER_WORDS:
            break
        cursor = _skip_spaces(structural, after)
    symbol_ref = _mlir_symbol_ref_at(structural, cursor, line_end)
    if symbol_ref is None:
        return None
    after_symbol = _skip_spaces(structural, symbol_ref[1])
    if after_symbol >= len(structural) or structural[after_symbol] != "(":
        return None
    paren_end = _matching_closer_index(structural, after_symbol, "(", ")")
    if paren_end is None:
        return None
    if not _llvm_func_custom_has_region_after(structural, paren_end + 1):
        return None
    return symbol_ref[0]


def _llvm_func_custom_has_region_after(
        structural: str, start: int) -> bool:
    """Look past return type / attributes for a `{ ... }` region. The
    structural pass doesn't try to fully parse the return-type grammar;
    it just searches for the next depth-zero `{` before the next
    op-boundary and confirms its matching `}` contains non-whitespace
    content."""
    line_end = _next_op_boundary(structural, start, len(structural))
    brace_open = -1
    depth = 0
    i = start
    while i < line_end:
        char = structural[i]
        if char == "{" and depth == 0:
            brace_open = i
            break
        if char in "(<[":
            depth += 1
        elif char in ")>]" and depth > 0:
            depth -= 1
        i += 1
    if brace_open == -1:
        return False
    brace_close = _matching_closer_index(structural, brace_open, "{", "}")
    if brace_close is None:
        return False
    return bool(structural[brace_open + 1:brace_close].strip())


def _mlir_generic_llvm_func_symbols(mlir_text: str) -> tuple[str, ...]:
    """Walk generic-form `"llvm.func"() <{sym_name = "..."}>` ops and
    return the declared symbols. Body-vs-decl is detected by whether a
    region (a `({...})` operand follows the property dict)."""
    structural = _comment_stripped_text(mlir_text)
    symbols: list[str] = []
    i = 0
    while i < len(structural):
        found = structural.find('"llvm.func"', i)
        if found == -1:
            break
        if not _mlir_op_start_context_allows(structural, found):
            i = found + 1
            continue
        quoted_end = found + len('"llvm.func"')
        j = _skip_spaces(structural, quoted_end)
        if j >= len(structural) or structural[j] != "(":
            i = quoted_end
            continue
        operands_end = _matching_closer_index(structural, j, "(", ")")
        if operands_end is None:
            i = quoted_end
            continue
        props = _generic_property_dict_after(
            structural, operands_end + 1)
        if props is None:
            i = operands_end + 1
            continue
        props_text, props_end = props
        props_map = _generic_property_assignments(props_text)
        symbol_prop = props_map.get("sym_name")
        if symbol_prop is not None:
            symbol = _symbol_from_generic_string_property(symbol_prop)
            if symbol is not None and _generic_llvm_func_has_body(
                    structural, props_end):
                symbols.append(_llvm_symbol_from_mlir_symbol(symbol))
        i = props_end
    return tuple(dict.fromkeys(symbols))


def _generic_llvm_func_has_body(structural: str, start: int) -> bool:
    """A generic-form llvm.func has a body iff it carries a non-empty
    region after its property dict, written as `({ ... })`. An empty
    region `({})` is a declaration (no body)."""
    cursor = _skip_spaces(structural, start)
    if cursor >= len(structural) or structural[cursor] != "(":
        return False
    region_end = _matching_closer_index(structural, cursor, "(", ")")
    if region_end is None:
        return False
    inner = structural[cursor + 1:region_end].strip()
    if not inner.startswith("{") or not inner.endswith("}"):
        return False
    body = inner[1:-1].strip()
    return bool(body)


def _mlir_gpu_kernel_symbols(mlir_text: str) -> tuple[str, ...]:
    structural = _comment_stripped_text(mlir_text)
    symbols: list[str] = []
    i = 0
    while i < len(structural):
        if structural[i] == '"':
            i = _quoted_span_end(structural, i)
            continue
        if not _bare_word_at(structural, i, "gpu.func"):
            i += 1
            continue
        symbol = _mlir_gpu_kernel_symbol_at(structural, i)
        if symbol:
            symbols.append(symbol)
        i += len("gpu.func")
    symbols.extend(_mlir_generic_gpu_kernel_symbols(mlir_text))
    return tuple(dict.fromkeys(symbols))


def _mlir_generic_gpu_kernel_symbols(mlir_text: str) -> tuple[str, ...]:
    structural = _comment_stripped_text(mlir_text)
    symbols: list[str] = []
    i = 0
    while i < len(structural):
        found = structural.find('"gpu.func"', i)
        if found == -1:
            break
        if not _mlir_op_start_context_allows(structural, found):
            i = found + 1
            continue
        quoted_end = found + len('"gpu.func"')
        j = _skip_spaces(structural, quoted_end)
        if j >= len(structural) or structural[j] != "(":
            i = quoted_end
            continue
        operands_end = _matching_closer_index(structural, j, "(", ")")
        if operands_end is None:
            i = quoted_end
            continue
        props = _generic_gpu_kernel_properties_after(
            structural, operands_end + 1)
        if props is None:
            i = operands_end + 1
            continue
        props_text, props_end = props
        props_map = _generic_gpu_kernel_property_assignments(props_text)
        if not _generic_gpu_kernel_property_is_set(props_map.get("kernel")):
            i = props_end
            continue
        symbol_prop = props_map.get("sym_name")
        if symbol_prop is not None:
            symbol = _symbol_from_generic_string_property(symbol_prop)
            if symbol is not None:
                symbols.append(_llvm_symbol_from_mlir_symbol(symbol))
        i = props_end
    return tuple(dict.fromkeys(symbols))


def _generic_gpu_kernel_properties_after(
        structural: str, start: int) -> tuple[str, int] | None:
    i = _skip_spaces(structural, start)
    props_parts: list[str] = []
    props_end = i
    while i < len(structural):
        props = _generic_property_dict_after(structural, i)
        if props is not None:
            props_text, props_end = props
            props_parts.append(props_text)
            i = _skip_spaces(structural, props_end)
            continue
        if structural[i] == "{":
            close = _matching_closer_index(structural, i, "{", "}")
            if close is None:
                break
            props_parts.append(structural[i + 1:close])
            props_end = close + 1
            i = _skip_spaces(structural, props_end)
            continue
        if structural[i] == "(":
            close = _matching_closer_index(structural, i, "(", ")")
            if close is None:
                break
            i = _skip_spaces(structural, close + 1)
            continue
        break
    if not props_parts:
        return None
    return ", ".join(props_parts), props_end


def _generic_gpu_kernel_property_assignments(
        props_text: str) -> dict[str, str]:
    props = _generic_property_assignments(props_text)
    parts = _split_top_level_commas(props_text)
    if not parts:
        return props
    for part in parts:
        key, value = _generic_gpu_kernel_property_part(part)
        if key:
            props[key] = value
    return props


def _generic_gpu_kernel_property_part(part: str) -> tuple[str, str]:
    stripped = part.strip()
    if not stripped:
        return "", ""
    split_index = _generic_gpu_property_split_index(stripped)
    if split_index is None:
        return stripped, "unit"
    key = stripped[:split_index].strip()
    value = stripped[split_index + 1:].strip()
    return key, value


def _generic_gpu_property_split_index(text: str) -> int | None:
    stack: list[str] = []
    pairs = {"(": ")", "<": ">", "{": "}", "[": "]"}
    closes = {")": "(", ">": "<", "}": "{", "]": "["}
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            quoted_end = _llvm_ir_quoted_identifier_end(text, index)
            if quoted_end is None:
                return None
            index = quoted_end
            continue
        if char in pairs:
            stack.append(char)
        elif char in closes:
            if not stack or stack[-1] != closes[char]:
                return None
            stack.pop()
        elif char in "=:" and not stack:
            return index
        index += 1
    return None


def _generic_gpu_kernel_property_is_set(value: str | None) -> bool:
    if value is None:
        return False
    stripped = value.strip().lower()
    if not stripped:
        return False
    body = stripped.split(":", 1)[0].strip()
    return body in ("#unit", "1", "true", "unit")


def _mlir_gpu_kernel_symbol_at(
        structural: str, token_start: int) -> str | None:
    token_end = token_start + len("gpu.func")
    op_end = _next_op_boundary(structural, token_start, len(structural))
    j = _skip_spaces(structural, token_end)
    word, word_end = _read_bare_word(structural, j)
    if word in ("private", "public", "nested"):
        j = _skip_spaces(structural, word_end)
    symbol_ref = _mlir_symbol_ref_at(structural, j, op_end)
    if symbol_ref is None:
        return None
    symbol, symbol_end = symbol_ref
    symbol = _normalize_symbol_ref(symbol)
    j = _skip_spaces(structural, symbol_end)
    if j >= op_end or structural[j] != "(":
        return None
    args_end = _matching_closer_index(structural, j, "(", ")")
    if args_end is None or args_end > op_end:
        return None
    tail = structural[args_end + 1:op_end]
    if "kernel" not in tail.split():
        return None
    return _llvm_symbol_from_mlir_symbol(symbol)


def _llvm_symbol_from_mlir_symbol(symbol: str) -> str:
    if symbol.startswith('@"') and symbol.endswith('"'):
        return _decode_quoted_symbol_body(symbol[2:-1])
    return symbol[1:] if symbol.startswith("@") else symbol


def _ptx_entry_symbols(
        output_text: str, *,
        include_device_functions: bool = True) -> frozenset[str]:
    symbols: set[str] = set()
    lines = tuple(output_text.splitlines())
    for index, raw_line in enumerate(lines):
        code_line = _ptx_code_line(raw_line)
        entry_tail = _ptx_entry_tail(code_line)
        if entry_tail is not None and _ptx_parse_entry_end(lines, index):
            symbol = _ptx_callable_symbol_from_tail(entry_tail)
            if symbol:
                symbols.add(symbol)
            continue
        func_tail = _ptx_func_tail(code_line)
        if include_device_functions and func_tail is not None:
            parsed_func = _ptx_parse_func(lines, index)
            if parsed_func is not None:
                symbols.add(parsed_func[1])
    return frozenset(symbols)


def _ptx_func_tail(line: str) -> str | None:
    if line.startswith(".func "):
        return line[len(".func "):].strip()
    if line.startswith(".visible .func "):
        return line[len(".visible .func "):].strip()
    return None


def _ptx_parse_func_end(
        lines: tuple[str, ...], index: int) -> int | None:
    parsed = _ptx_parse_func(lines, index)
    return parsed[0] if parsed is not None else None


def _ptx_parse_func(
        lines: tuple[str, ...], index: int) -> tuple[int, str] | None:
    line = _ptx_code_line(lines[index])
    func_tail = _ptx_func_tail(line)
    if func_tail is None:
        return None
    call_tail, call_line_index, next_index = _ptx_func_call_tail_from_lines(
        lines, index, func_tail)
    if call_tail is None:
        return None
    open_index = call_tail.find("(")
    symbol = call_tail[:open_index].strip()
    if not _ptx_entry_name_is_plausible(symbol):
        return None
    params = _ptx_collect_paren_group(lines, call_line_index, call_tail,
                                     open_index)
    if params is None:
        return None
    param_text, after_close, close_line_index, after_line_index = params
    if not _ptx_param_list_is_plausible(
            param_text, allow_register_params=True):
        return None
    next_body_index = _ptx_func_body_follows(
        after_close, lines, close_line_index, after_line_index)
    if next_body_index is None:
        return None
    return next_body_index, symbol


def _ptx_func_call_tail_from_lines(
        lines: tuple[str, ...], index: int,
        func_tail: str) -> tuple[str | None, int, int]:
    tail = func_tail.strip()
    line_index = index
    next_index = index + 1
    if tail.startswith("("):
        ret_params = _ptx_collect_paren_group(lines, line_index, tail, 0)
        if ret_params is None:
            return None, line_index, next_index
        param_text, after_close, line_index, next_index = ret_params
        if not _ptx_param_list_is_plausible(
                param_text, allow_register_params=True):
            return None, line_index, next_index
        tail = after_close.strip()
        if not tail:
            next_code_index = _ptx_next_code_line_index(lines, next_index)
            if next_code_index is None:
                return None, line_index, next_index
            line_index = next_code_index
            next_index = line_index + 1
            tail = _ptx_code_line(lines[line_index])
    open_index = tail.find("(")
    if open_index == -1 or not _ptx_entry_name_is_plausible(
            tail[:open_index]):
        return None, line_index, next_index
    return tail, line_index, next_index


def _ptx_collect_paren_group(
        lines: tuple[str, ...], start_index: int, text: str,
        open_index: int) -> tuple[str, str, int, int] | None:
    if open_index < 0 or open_index >= len(text) or text[open_index] != "(":
        return None
    depth = 0
    collected: list[str] = []
    line_index = start_index
    current = text
    pos = open_index
    first_line = True
    while line_index < len(lines):
        if not first_line:
            current = _ptx_code_line(lines[line_index])
            pos = 0
            if not current:
                line_index += 1
                continue
            if current.startswith((".func ", ".visible .func ", ".entry ",
                                   ".visible .entry ", ".version ",
                                   ".target ")):
                return None
        while pos < len(current):
            char = current[pos]
            if char == "(":
                if depth > 0:
                    collected.append(char)
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return None
                if depth == 0:
                    return (
                        "".join(collected).strip(),
                        current[pos + 1:].strip(),
                        line_index,
                        line_index + 1,
                    )
                collected.append(char)
            elif depth > 0:
                collected.append(char)
            pos += 1
        if depth > 0:
            collected.append("\n")
        line_index += 1
        first_line = False
    return None


def _ptx_func_call_tail(func_tail: str) -> str | None:
    tail = func_tail.strip()
    if tail.startswith("("):
        close_index = _balanced_paren_close(tail, 0)
        if close_index == -1:
            return None
        if not _ptx_param_list_is_plausible(
                tail[1:close_index], allow_register_params=True):
            return None
        tail = tail[close_index + 1:].strip()
    open_index = tail.find("(")
    if open_index == -1:
        return None
    if not _ptx_entry_name_is_plausible(tail[:open_index]):
        return None
    return tail


def _ptx_callable_symbol_from_tail(tail: str) -> str:
    call_tail = _ptx_func_call_tail(tail)
    if call_tail is not None:
        tail = call_tail
    open_index = tail.find("(")
    if open_index == -1:
        return ""
    symbol = tail[:open_index].strip()
    return symbol if _ptx_entry_name_is_plausible(symbol) else ""


def _rocm_hip_defined_kernel_symbols(output_text: str) -> frozenset[str]:
    lines = tuple(line.strip() for line in output_text.splitlines()
                  if line.strip())
    if any(line.startswith("target triple = ") and "amdgcn" in line.lower()
           for line in lines) or _llvm_ir_lines_have_amdgpu_kernel_definition(
               lines):
        return _llvm_ir_defined_amdgpu_kernel_symbols(output_text)
    symbols: set[str] = set()
    in_block_comment = False
    for index, line in enumerate(lines):
        code_line, in_block_comment = _c_like_code_line(
            line, in_block_comment=in_block_comment)
        if not code_line:
            continue
        if ((code_line.startswith("extern ") and "__global__" in code_line
             and "(" in code_line)
                or (code_line.startswith("__global__")
                    and "(" in code_line)):
            symbol = _hip_kernel_symbol_from_signature(
                _c_like_signature_text(lines, index, code_line))
            if symbol:
                symbols.add(symbol)
    return frozenset(symbols)


def _hip_kernel_symbol_from_signature(code_line: str) -> str:
    if not _hip_kernel_signature_is_plausible(code_line):
        return ""
    prefix = code_line[:code_line.find("(")].strip()
    tokens = prefix.replace("*", " * ").replace("&", " & ").split()
    if tokens and tokens[0] == "extern":
        tokens = tokens[1:]
    if "__global__" not in tokens:
        return ""
    global_index = tokens.index("__global__")
    trailing = [token for token in tokens[global_index + 1:]
                if token not in ("*", "&")]
    symbol = trailing[-1] if trailing else ""
    return symbol if _c_like_identifier_is_plausible(symbol) else ""


def _metal_msl_defined_kernel_symbols(output_text: str) -> frozenset[str]:
    lines = tuple(line.strip() for line in output_text.splitlines()
                  if line.strip())
    symbols: set[str] = set()
    in_block_comment = False
    for index, line in enumerate(lines):
        code_line, in_block_comment = _c_like_code_line(
            line, in_block_comment=in_block_comment)
        if not code_line:
            continue
        if ((code_line.startswith("kernel ") and "(" in code_line)
                or (code_line.startswith("[[kernel]]")
                    and "(" in code_line)):
            symbol = _metal_kernel_symbol_from_signature(
                _c_like_signature_text(lines, index, code_line))
            if symbol:
                symbols.add(symbol)
    return frozenset(symbols)


def _metal_kernel_symbol_from_signature(code_line: str) -> str:
    if not _metal_kernel_signature_is_plausible(code_line):
        return ""
    prefix = code_line[:code_line.find("(")].strip()
    tokens = prefix.replace("*", " * ").replace("&", " & ").split()
    if tokens and tokens[0] == "kernel":
        tokens = tokens[1:]
    if tokens and tokens[0] == "[[kernel]]":
        tokens = tokens[1:]
    trailing = [token for token in tokens if token not in ("*", "&")]
    symbol = trailing[-1] if trailing else ""
    return symbol if _c_like_identifier_is_plausible(symbol) else ""


def _webgpu_wgsl_defined_compute_symbols(output_text: str) -> frozenset[str]:
    lines = tuple(line.strip() for line in output_text.splitlines()
                  if line.strip())
    symbols: set[str] = set()
    pending_compute = False
    pending_workgroup_size = False
    in_block_comment = False
    for index, line in enumerate(lines):
        code_line, in_block_comment = _c_like_code_line(
            line, in_block_comment=in_block_comment)
        stripped = code_line.strip()
        if not stripped:
            continue
        attr_prefix = _wgsl_compute_attribute_prefix(stripped)
        if attr_prefix is not None:
            attrs, after_attrs = attr_prefix
            if "@compute" in attrs and pending_compute:
                pending_compute = False
                pending_workgroup_size = False
                continue
            if "@workgroup_size" in attrs and pending_workgroup_size:
                pending_compute = False
                pending_workgroup_size = False
                continue
            pending_compute = (
                pending_compute
                or "@compute" in attrs)
            pending_workgroup_size = (
                pending_workgroup_size
                or "@workgroup_size" in attrs)
            if not after_attrs:
                continue
            if not after_attrs.startswith("fn "):
                pending_compute = False
                pending_workgroup_size = False
                continue
            stripped = after_attrs
        if stripped.startswith("fn ") and "(" in stripped:
            symbol = _wgsl_compute_symbol_from_signature(
                _c_like_signature_text(lines, index, stripped))
            if pending_compute and pending_workgroup_size and symbol:
                symbols.add(symbol)
            pending_compute = False
            pending_workgroup_size = False
    return frozenset(symbols)


def _wgsl_compute_symbol_from_signature(code_line: str) -> str:
    if not _wgsl_signature_is_plausible(code_line):
        return ""
    prefix = code_line[:code_line.find("(")].strip()
    tokens = prefix.split()
    symbol = tokens[1] if len(tokens) == 2 else ""
    return symbol if _c_like_identifier_is_plausible(symbol) else ""


def _llvm_ir_defined_function_symbols(output_text: str) -> frozenset[str]:
    symbols: set[str] = set()
    for raw_line in output_text.splitlines():
        line = _llvm_ir_code_line(raw_line)
        if not line.startswith("define "):
            continue
        symbol = _llvm_ir_function_symbol_from_header(line)
        if symbol:
            symbols.add(symbol)
    return frozenset(symbols)


def _llvm_ir_lines_have_amdgpu_kernel_definition(
        lines: tuple[str, ...]) -> bool:
    return any(_llvm_ir_function_header_has_attribute(
        _llvm_ir_code_line(line), "amdgpu_kernel") for line in lines)


def _llvm_ir_defined_amdgpu_kernel_symbols(output_text: str) -> frozenset[str]:
    symbols: set[str] = set()
    for raw_line in output_text.splitlines():
        line = _llvm_ir_code_line(raw_line)
        if not _llvm_ir_function_header_has_attribute(
                line, "amdgpu_kernel"):
            continue
        symbol = _llvm_ir_function_symbol_from_header(line)
        if symbol:
            symbols.add(symbol)
    return frozenset(symbols)


def _llvm_ir_function_header_has_attribute(line: str, attr: str) -> bool:
    if not line.startswith("define "):
        return False
    after_keyword = line[len("define "):]
    at_index = after_keyword.find("@")
    if at_index <= 0:
        return False
    return attr in after_keyword[:at_index].split()


def _llvm_ir_function_symbol_from_header(line: str) -> str:
    if not line.startswith("define "):
        return ""
    after_keyword = line[len("define "):]
    at_index = after_keyword.find("@")
    if at_index <= 0:
        return ""
    symbol_span = _llvm_ir_function_symbol_span(after_keyword, at_index)
    if symbol_span is None:
        return ""
    symbol = symbol_span[0]
    if symbol.startswith('"') and symbol.endswith('"'):
        return _decode_quoted_symbol_body(symbol[1:-1])
    return symbol


def _llvm_quoted_symbol_end(text: str, start: int) -> int | None:
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
    return None


@dataclass(frozen=True, slots=True, weakref_slot=True)
class MLIRBackendResult:
    """Outcome of trying to lower MLIR text to one backend target.

    The mock structural validator always runs first. Real target
    lowering is represented separately and must never be silent:
    - malformed MLIR fails before any lowering attempt;
    - a valid mock shape plus no target pipeline is DEFERRED with a
      finding explaining why;
    - a real lowering failure must carry at least one diagnostic;
    - a real lowering pass must carry concrete backend output text.

    PASSED status is additionally bound to the backend runner's
    identity registry for normal public construction, copy, and pickle
    paths. This is an integrity check inside Helix's own code, not a
    Python security boundary against adversarial same-process
    introspection.
    """
    target: MLIRBackendTarget
    validation: MLIRValidation
    lowering_attempted: bool
    lowering_passed: Optional[bool]
    # `lowering_tool` is the PRIMARY tool path — `mlir-opt` for any
    # Stage 213+ backend run. When a target's Stage 214 translator
    # entry is populated, the runner ALSO invokes `mlir-translate`
    # after `mlir-opt`; that second-stage path appears in
    # `output_provenance` as `mlir-translate=<path>` /
    # `mlir-translate-flag=<flag>` entries, not in `lowering_tool`.
    # Treat `lowering_tool` as the primary identifier for the
    # lowering, not the complete chain inventory.
    lowering_tool: Optional[str]
    lowering_findings: tuple[str, ...]
    output_text: Optional[str] = None
    output_provenance: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        raise TypeError(
            "MLIRBackendResult is final; subclassing could bypass "
            "the backend result invariants")

    def __post_init__(self) -> None:
        _require_backend_target(self.target)
        if type(self.validation) is not MLIRValidation:
            raise ValueError(
                "MLIRBackendResult: validation must be an MLIRValidation "
                f"result, got {self.validation!r}")
        if not isinstance(self.lowering_attempted, bool):
            raise ValueError(
                "MLIRBackendResult: lowering_attempted must be a bool, "
                f"got {self.lowering_attempted!r}")
        if not isinstance(self.lowering_findings, tuple):
            raise ValueError(
                "MLIRBackendResult: lowering_findings must be a tuple, "
                f"got {type(self.lowering_findings).__name__}")
        if self.output_text is not None:
            if not isinstance(self.output_text, str):
                raise ValueError(
                    "MLIRBackendResult: output_text must be a str or "
                    f"None, got {type(self.output_text).__name__}")
            if not self.output_text.strip():
                raise ValueError(
                    "MLIRBackendResult: output_text must carry text "
                    "when present")
        if not isinstance(self.output_provenance, tuple):
            raise ValueError(
                "MLIRBackendResult: output_provenance must be a tuple")
        for entry in self.output_provenance:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    "MLIRBackendResult: output_provenance has a blank "
                    f"or non-str entry {entry!r}")

        for entry in self.lowering_findings:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    "MLIRBackendResult: lowering_findings has a blank "
                    f"or non-str entry ({entry!r})")

        if not self.lowering_attempted:
            if self.lowering_passed is not None:
                raise ValueError(
                    "MLIRBackendResult: lowering_attempted=False but "
                    f"lowering_passed={self.lowering_passed!r}")
            if self.lowering_tool is not None:
                raise ValueError(
                    "MLIRBackendResult: lowering_attempted=False but "
                    f"lowering_tool={self.lowering_tool!r}")
            if self.output_text is not None:
                raise ValueError(
                    "MLIRBackendResult: lowering_attempted=False but "
                    "output_text is present")
            if self.output_provenance:
                raise ValueError(
                    "MLIRBackendResult: lowering_attempted=False but "
                    "output_provenance is present")
            if _validation_is_failed(self.validation) and self.lowering_findings:
                raise ValueError(
                    "MLIRBackendResult: validation FAILED before backend "
                    "lowering, so lowering_findings must be empty")
            if not _validation_is_failed(self.validation) \
                    and not self.lowering_findings:
                raise ValueError(
                    "MLIRBackendResult: mock-valid MLIR with no lowering "
                    "attempt must carry at least one finding explaining "
                    "why lowering is DEFERRED")
            return

        if _validation_is_failed(self.validation):
            raise ValueError(
                "MLIRBackendResult: cannot attempt backend lowering after "
                "MLIR validation FAILED")
        if not _validation_is_real_passed(self.validation):
            raise ValueError(
                "MLIRBackendResult: attempted backend lowering requires "
                "validation to be PASSED; mock-deferred validation cannot "
                "be used for a real backend attempt")
        if self.lowering_passed is None:
            raise ValueError(
                "MLIRBackendResult: lowering_attempted=True but "
                "lowering_passed is None")
        if not isinstance(self.lowering_passed, bool):
            raise ValueError(
                "MLIRBackendResult: lowering_attempted=True requires "
                "lowering_passed to be a bool, got "
                f"{self.lowering_passed!r}")
        if (not isinstance(self.lowering_tool, str)
                or not self.lowering_tool.strip()):
            raise ValueError(
                "MLIRBackendResult: lowering_attempted=True requires a "
                "non-empty lowering_tool")
        if self.lowering_passed is False and not self.lowering_findings:
            raise ValueError(
                "MLIRBackendResult: lowering_passed=False but "
                "lowering_findings is empty - a real lowering failure "
                "must carry a diagnostic")
        if self.lowering_passed is False and self.output_text is not None:
            raise ValueError(
                "MLIRBackendResult: lowering_passed=False but "
                "output_text is present; failed lowering must not expose "
                "backend output")
        if self.lowering_passed is False and self.output_provenance:
            raise ValueError(
                "MLIRBackendResult: lowering_passed=False but "
                "output_provenance is present")
        if self.lowering_passed is True:
            if not _validation_is_real_passed(self.validation):
                raise ValueError(
                    "MLIRBackendResult: lowering_passed=True requires "
                    "validation to be PASSED; mock-deferred validation "
                    "cannot be promoted to a backend pass")
            if self.lowering_findings:
                raise ValueError(
                    "MLIRBackendResult: lowering_passed=True but "
                    "lowering_findings is non-empty")
            if self.output_text is None:
                raise ValueError(
                    "MLIRBackendResult: lowering_passed=True requires "
                    "non-empty output_text")
            if not self.output_provenance:
                raise ValueError(
                    "MLIRBackendResult: lowering_passed=True requires "
                    "output_provenance")
            output_digest = hashlib.sha256(
                self.output_text.encode("utf-8")).hexdigest()
            if f"output_sha256={output_digest}" not in self.output_provenance:
                raise ValueError(
                    "MLIRBackendResult: output_provenance digest does "
                    "not match output_text")
            if not _has_backend_result_pass_shape(self):
                raise ValueError(
                    "MLIRBackendResult: lowering_passed=True must carry "
                    "a coherent backend-runner registry entry")
            raise ValueError(
                "MLIRBackendResult: lowering_passed=True must be "
                "created by the backend runner after target output "
                "validation")

    def status(self) -> MLIRBackendStatus:
        if _validation_is_failed(self.validation):
            return MLIRBackendStatus.FAILED
        if not self.lowering_attempted:
            return MLIRBackendStatus.DEFERRED
        if self.lowering_passed is False:
            return MLIRBackendStatus.FAILED
        if self.lowering_passed is True:
            if not _has_backend_result_pass_shape(self):
                raise AssertionError(
                    "MLIRBackendResult.status reached an unbranded "
                    "PASSED state")
            return MLIRBackendStatus.PASSED
        raise AssertionError(
            "MLIRBackendResult.status reached an illegal state: "
            f"lowering_passed={self.lowering_passed!r}")

    def passed(self) -> bool:
        return self.status() is MLIRBackendStatus.PASSED

    def failed(self) -> bool:
        return self.status() is MLIRBackendStatus.FAILED

    def deferred(self) -> bool:
        return self.status() is MLIRBackendStatus.DEFERRED

    def __copy__(self) -> "MLIRBackendResult":
        return _copy_backend_result(self)

    def __deepcopy__(
            self, memo: dict[int, object]) -> "MLIRBackendResult":
        return _copy_backend_result(self, memo)

    def __reduce_ex__(self, protocol: int) -> object:
        if _has_backend_result_pass_shape(self):
            raise TypeError(
                "PASSED MLIRBackendResult is a backend-runner "
                "registry entry and cannot be pickled")
        return (
            MLIRBackendResult,
            (
                self.target,
                self.validation,
                self.lowering_attempted,
                self.lowering_passed,
                self.lowering_tool,
                self.lowering_findings,
                self.output_text,
                self.output_provenance,
            ),
        )


def _copy_backend_result(
        result: MLIRBackendResult, memo: dict[int, object] | None = None,
        ) -> MLIRBackendResult:
    if _has_backend_result_pass_shape(result):
        if memo is not None:
            memo[id(result)] = result
        return result
    return MLIRBackendResult(
        result.target,
        result.validation,
        result.lowering_attempted,
        result.lowering_passed,
        result.lowering_tool,
        result.lowering_findings,
        result.output_text,
        result.output_provenance,
    )


def _make_backend_pipeline_runner(raw_runner):
    brand_pass, has_pass = _make_backend_result_pass_registry()

    class _BackendPipelineRunner:
        __slots__ = ("__raw_runner",)

        def __init__(self, runner) -> None:
            self.__raw_runner = runner

        def __call__(
                self, mlir_text: str, *, target: MLIRBackendTarget,
                validation: MLIRValidation, mlir_opt: str,
                pipeline: tuple[str, ...],
                output_validator: MLIRBackendOutputValidator,
                timeout_s: int = _MLIR_BACKEND_PIPELINE_TIMEOUT_S,
                mlir_translate: str | None = None,
                chained_tool: str | None = None,
                ) -> MLIRBackendResult:
            result = self.__raw_runner(
                mlir_text,
                target=target,
                validation=validation,
                mlir_opt=mlir_opt,
                pipeline=pipeline,
                output_validator=output_validator,
                timeout_s=timeout_s,
                mlir_translate=mlir_translate,
                chained_tool=chained_tool,
            )
            if _backend_result_pass_shape_is_coherent(result):
                return brand_pass(result)
            return result

        def has_backend_pass(self, result: MLIRBackendResult) -> bool:
            return (_backend_result_pass_shape_is_coherent(result)
                    and has_pass(result))

    runner = _BackendPipelineRunner(raw_runner)
    return runner, runner.has_backend_pass


_backend_pipeline_runner, _has_backend_result_pass_shape = (
    _make_backend_pipeline_runner(_run_mlir_opt_pipeline))
_run_mlir_opt_pipeline = _backend_pipeline_runner


def lower_mlir_to_backend(
        mlir_text: str,
        target: MLIRBackendTarget,
        *,
        support: Optional[MLIRSupport] = None) -> MLIRBackendResult:
    """Validate MLIR text and try to lower it to one backend target.

    Stage 213 chunk A only establishes the seam. If the text is
    structurally malformed, the result is FAILED and no target lowering
    is attempted. If the text has a clean mock shape, the result is
    DEFERRED until a real MLIR surface and a target lowering pipeline
    are both wired.
    """
    target = _require_backend_target(target)
    mock_validation = mock_validate_mlir(mlir_text)
    if _validation_is_failed(mock_validation):
        return MLIRBackendResult(
            target=target,
            validation=mock_validation,
            lowering_attempted=False,
            lowering_passed=None,
            lowering_tool=None,
            lowering_findings=(),
            output_text=None,
        )

    if support is None:
        support = detect_mlir_support()
    if not isinstance(support, MLIRSupport):
        raise ValueError(
            "lower_mlir_to_backend: support must be an MLIRSupport "
            f"or None, got {support!r}")

    validation = validate_mlir_with_toolchain(
        mlir_text, support=support)
    if type(validation) is not MLIRValidation:
        raise ValueError(
            "lower_mlir_to_backend: validate_mlir_with_toolchain must "
            f"return MLIRValidation, got {validation!r}")
    if _validation_is_failed(validation):
        return MLIRBackendResult(
            target=target,
            validation=validation,
            lowering_attempted=False,
            lowering_passed=None,
            lowering_tool=None,
            lowering_findings=(),
            output_text=None,
        )

    findings: list[str] = []
    if not support.is_available():
        findings.extend(
            f"no real MLIR surface available: {line}"
            for line in support.detail)

    pipeline = backend_lowering_pipeline(target)
    output_validator = _MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY[target]
    translator = _MLIR_BACKEND_TRANSLATORS_AUTHORITY[target]
    if not pipeline:
        findings.append(
            f"Stage 213 MLIR lowering pipeline for {target.value} is "
            "not wired yet; Stage 214 must supply target passes before "
            "this backend can consume MLIR")
    elif not _validation_is_real_passed(validation):
        findings.append(
            f"Stage 213 MLIR lowering pipeline for {target.value} is "
            "declared, but real MLIR validation is not PASSED; "
            "refusing to attempt backend lowering")
    elif support.mlir_opt is None:
        findings.append(
            f"Stage 213 MLIR lowering pipeline for {target.value} is "
            "declared, but `mlir-opt` is not available to run it")
    elif output_validator is None:
        findings.append(
            f"Stage 213 MLIR lowering pipeline for {target.value} is "
            "declared, but the target output validator is not wired; "
            "refusing to claim backend output from a pass pipeline alone")
    elif translator is not None and support.mlir_translate is None:
        findings.append(
            f"Stage 214 translator step for {target.value} is declared, "
            "but `mlir-translate` is not on PATH; refusing to attempt a "
            "chain that cannot complete")
    else:
        chained_tool_path: str | None = None
        if translator is not None and translator[2]:
            chained_name = translator[2][0]
            chained_tool_path = support.chained_tool_path(chained_name)
            if chained_tool_path is None:
                findings.append(
                    f"Stage 214 chained tool {chained_name!r} for "
                    f"{target.value} is declared, but is not on PATH "
                    "(or not recognized by MLIRSupport); refusing to "
                    "attempt a chain that cannot complete")
                return MLIRBackendResult(
                    target=target,
                    validation=validation,
                    lowering_attempted=False,
                    lowering_passed=None,
                    lowering_tool=None,
                    lowering_findings=tuple(findings),
                    output_text=None,
                )
        return _run_mlir_opt_pipeline(
            mlir_text,
            target=target,
            validation=validation,
            mlir_opt=support.mlir_opt,
            pipeline=pipeline,
            output_validator=output_validator,
            mlir_translate=(
                support.mlir_translate if translator is not None else None),
            chained_tool=chained_tool_path,
        )

    return MLIRBackendResult(
        target=target,
        validation=validation,
        lowering_attempted=False,
        lowering_passed=None,
        lowering_tool=None,
        lowering_findings=tuple(findings),
        output_text=None,
    )
