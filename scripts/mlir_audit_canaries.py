"""Fast MLIR audit canaries for the Helix v3 verifier/proof loop.

The script is intentionally small and dependency-free. It exercises
known audit bug families that are expensive to rediscover with three
fresh reviewers every restart:

- fake/smoke-aware mlir-opt proof holes;
- canonical terminator/control-op SSA preflight gaps;
- generic func.func signature correspondence gaps;
- backend artifact identity gaps.

Default mode is report-only so it is safe to run while the repo has
known-open findings. Use --strict in a clean gate; it exits nonzero if
any canary still exposes a hole.
"""
from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helixc.ir.mlir import backends, validate  # noqa: E402
from helixc.ir.mlir.backends import MLIRBackendTarget  # noqa: E402


@dataclass(frozen=True)
class CanaryResult:
    name: str
    passed: bool
    detail: str


@contextlib.contextmanager
def _patched_run(module, fake_run: Callable) -> Iterator[None]:
    old_run = module.subprocess.run
    old_detect = getattr(module, "detect_mlir_support", None)
    module.subprocess.run = fake_run
    if old_detect is not None:
        module.detect_mlir_support = lambda: module.MLIRSupport(
            bindings=False,
            dialects=False,
            mlir_opt="/fake/mlir-opt",
            detail=("audit canary fake mlir-opt",),
        )
    try:
        yield
    finally:
        module.subprocess.run = old_run
        if old_detect is not None:
            module.detect_mlir_support = old_detect


def _smoke_aware_echo_run(cmd, *, capture_output, text, timeout):
    o_index = cmd.index("-o")
    input_path = Path(cmd[o_index - 1])
    output_path = Path(cmd[o_index + 1])
    input_text = input_path.read_text(encoding="utf-8")
    if validate._mlir_text_is_invalid_smoke_probe(input_text):
        return subprocess.CompletedProcess(
            cmd, 1, "", "invalid smoke rejected")
    output_path.write_text(input_text, encoding="utf-8")
    return subprocess.CompletedProcess(cmd, 0, "", "")


def _fake_validator_must_not_pass(name: str, mlir_text: str) -> CanaryResult:
    with _patched_run(validate, _smoke_aware_echo_run):
        result = validate._run_mlir_opt_validate(
            mlir_text, "/fake/mlir-opt")
    if result.passed():
        return CanaryResult(
            name,
            False,
            "smoke-aware echo tool minted MLIRValidation.PASSED")
    return CanaryResult(
        name,
        True,
        f"blocked with verdict={result.verdict.value} findings="
        f"{result.findings[:2]}")


def _generic_func_signature_must_be_preserved() -> CanaryResult:
    generic = (
        '"builtin.module"() ({ '
        '"func.func"() <{function_type = () -> i32, sym_name = "f"}> '
        '({}) : () -> () }) : () -> ()'
    )
    rewritten = "module { func.func @f() { return } }\n"
    findings = validate._mlir_opt_output_correspondence_findings(
        generic, rewritten)
    if findings:
        return CanaryResult(
            "generic-func-signature-correspondence",
            True,
            findings[0])
    return CanaryResult(
        "generic-func-signature-correspondence",
        False,
        "generic func.func return type changed to void with no finding")


def _gpu_backend_symbol_must_be_bound() -> CanaryResult:
    helper = getattr(backends, "_backend_output_symbol_finding", None)
    if helper is None:
        return CanaryResult(
            "gpu-backend-symbol-binding",
            False,
            "backend symbol correspondence helper is missing")
    mlir_text = "module { func.func @expected() { return } }\n"
    ptx_text = (
        ".version 8.3\n"
        ".target sm_80\n"
        ".visible .entry totally_wrong() {\n"
        "  ret;\n"
        "}\n"
    )
    finding = helper(mlir_text, MLIRBackendTarget.PTX, ptx_text)
    if finding:
        return CanaryResult(
            "gpu-backend-symbol-binding",
            True,
            finding)
    return CanaryResult(
        "gpu-backend-symbol-binding",
        False,
        "PTX artifact with unrelated entry was accepted as correlated")


def _backend_shape_must_reject(
        name: str, target: MLIRBackendTarget, output_text: str,
) -> CanaryResult:
    if backends._looks_like_backend_output(target, output_text):
        return CanaryResult(
            name,
            False,
            f"{target.value} backend shape accepted malformed artifact")
    return CanaryResult(name, True, "malformed artifact rejected")


def _generic_llvm_func_symbol_must_bind() -> CanaryResult:
    helper = getattr(backends, "_mlir_defined_function_symbols", None)
    if helper is None:
        return CanaryResult(
            "generic-llvm-func-symbol-binding",
            False,
            "MLIR symbol extraction helper is missing")
    mlir = (
        'module { "llvm.func"() <{sym_name = "expected", '
        'function_type = !llvm.func<void ()>}> '
        '({ llvm.return }) : () -> () }')
    symbols = helper(mlir)
    if "expected" in symbols:
        return CanaryResult(
            "generic-llvm-func-symbol-binding",
            True,
            f"symbols={symbols!r}")
    return CanaryResult(
        "generic-llvm-func-symbol-binding",
        False,
        f"generic llvm.func sym_name not picked up: {symbols!r}")


def _llvm_typed_value_must_reject_scalar_in_aggregate() -> CanaryResult:
    helper = getattr(backends,
                     "_llvm_ir_typed_value_part_is_plausible", None)
    if helper is None:
        return CanaryResult(
            "llvm-typed-value-aggregate-scalar",
            False,
            "LLVM typed-value helper is missing")
    fails = (
        helper("{ i32 } 0"),
        helper("<4 x i32> 0"),
        helper("[4 x i32] 0"),
    )
    if any(fails):
        return CanaryResult(
            "llvm-typed-value-aggregate-scalar",
            False,
            f"aggregate/vector accepted scalar literal: {fails!r}")
    return CanaryResult(
        "llvm-typed-value-aggregate-scalar",
        True,
        "scalar literal rejected for aggregate/vector return types")


def _c_like_declaration_must_reject_digit_identifier() -> CanaryResult:
    helper = getattr(backends, "_c_like_body_line_is_plausible", None)
    if helper is None:
        return CanaryResult(
            "c-like-declaration-digit-identifier",
            False,
            "C-like body line helper is missing")
    if helper("float * 123;"):
        return CanaryResult(
            "c-like-declaration-digit-identifier",
            False,
            "declaration with digit-prefixed identifier accepted")
    return CanaryResult(
        "c-like-declaration-digit-identifier",
        True,
        "declaration with digit-prefixed identifier rejected")


def _quoted_symbol_must_not_collapse() -> CanaryResult:
    helper = getattr(backends, "_mlir_defined_function_symbols", None)
    if helper is None:
        return CanaryResult(
            "quoted-symbol-preservation",
            False,
            "MLIR symbol extraction helper is missing")
    symbols = helper('module { func.func @"foo/bar"() { return } }\n')
    if "foo/bar" in symbols:
        return CanaryResult(
            "quoted-symbol-preservation",
            True,
            f"symbols={symbols!r}")
    return CanaryResult(
        "quoted-symbol-preservation",
        False,
        f'quoted symbol @"foo/bar" collapsed or vanished: {symbols!r}')


def run_canaries() -> tuple[CanaryResult, ...]:
    return (
        _fake_validator_must_not_pass(
            "fake-validator-bad-type",
            "module { func.func @f() { "
            "%0 = arith.constant 1 : bananas return } }\n",
        ),
        _fake_validator_must_not_pass(
            "fake-validator-addf-i32",
            "module { func.func @f(%arg0: i32) { "
            "%0 = arith.addf %arg0, %arg0 : i32 return } }\n",
        ),
        _fake_validator_must_not_pass(
            "canonical-func-return-missing-ssa",
            "module { func.func @f() -> i32 { "
            "func.return %missing : i32 } }\n",
        ),
        _fake_validator_must_not_pass(
            "canonical-scf-if-missing-ssa",
            "module { func.func @f() { "
            "scf.if %missing { scf.yield } return } }\n",
        ),
        _fake_validator_must_not_pass(
            "fake-validator-missing-terminator",
            "module { func.func @f() { } }\n",
        ),
        _fake_validator_must_not_pass(
            "fake-validator-vector-bad-dim",
            "module { func.func @f(%x: vector<?xi32>) { return } }\n",
        ),
        # Control-predicate non-i1 (HIGH-1).
        _fake_validator_must_not_pass(
            "control-predicate-scf-if-non-i1",
            "module { func.func @f(%c: i32) { "
            "scf.if %c { scf.yield } func.return } }\n",
        ),
        _fake_validator_must_not_pass(
            "control-predicate-cf-assert-non-i1",
            "module { func.func @f(%c: f32) { "
            "cf.assert %c, \"bad\" func.return } }\n",
        ),
        # memref.load / memref.store arity + index type (HIGH-2).
        _fake_validator_must_not_pass(
            "memref-load-index-arity-mismatch",
            "module { func.func @f(%m: memref<10x20xi32>, %i: index) "
            "-> i32 { %r = memref.load %m[%i] : memref<10x20xi32> "
            "func.return %r : i32 } }\n",
        ),
        _fake_validator_must_not_pass(
            "memref-load-non-index-idx",
            "module { func.func @f(%m: memref<10xi32>, %i: i32) -> i32 "
            "{ %r = memref.load %m[%i] : memref<10xi32> "
            "func.return %r : i32 } }\n",
        ),
        _fake_validator_must_not_pass(
            "memref-store-index-arity-mismatch",
            "module { func.func @f(%m: memref<10x20xi32>, %i: index, "
            "%v: i32) { memref.store %v, %m[%i] : memref<10x20xi32> "
            "func.return } }\n",
        ),
        # arith.constant value-type matching (HIGH-3).
        _fake_validator_must_not_pass(
            "arith-constant-bool-non-i1",
            "module { func.func @f() -> i32 { "
            "%c = arith.constant true : i32 func.return %c : i32 } }\n",
        ),
        _fake_validator_must_not_pass(
            "arith-constant-int-float-type",
            "module { func.func @f() -> f32 { "
            "%c = arith.constant 1 : f32 func.return %c : f32 } }\n",
        ),
        # Generic-op body bypass (HIGH-4): a custom func.func body with
        # only a generic-form op and no custom terminator must FAIL.
        _fake_validator_must_not_pass(
            "generic-op-body-without-terminator",
            'module { func.func @f() { "test.op"() : () -> () } }\n',
        ),
        _fake_validator_must_not_pass(
            "generic-form-terminator-not-recognized",
            'module { func.func @f() { "func.return"() : () -> () } }\n',
        ),
        # Remaining HIGH-3 vector sub-cases.
        _fake_validator_must_not_pass(
            "vector-multi-reduction-bogus-kind",
            "module { func.func @f(%v: vector<4xi32>) -> vector<4xi32> { "
            "%0 = vector.multi_reduction <bogus>, %v, %v [0] : "
            "vector<4xi32> to vector<4xi32> func.return "
            "%0 : vector<4xi32> } }\n",
        ),
        _fake_validator_must_not_pass(
            "vector-shape-cast-element-count-mismatch",
            "module { func.func @f(%v: vector<4xi32>) -> vector<3xi32> { "
            "%0 = vector.shape_cast %v : vector<4xi32> to vector<3xi32> "
            "func.return %0 : vector<3xi32> } }\n",
        ),
        _fake_validator_must_not_pass(
            "vector-transfer-read-non-index-idx",
            "module { func.func @f(%A: memref<4xf32>, %i: i32, %c0: f32) "
            "{ %0 = vector.transfer_read %A[%i], %c0 : "
            "memref<4xf32>, vector<4xf32> func.return } }\n",
        ),
        # Audit-fix follow-ups: shape_cast element-type drift, and
        # missing kind delimiter on multi_reduction must fail closed.
        _fake_validator_must_not_pass(
            "vector-shape-cast-element-type-mismatch",
            "module { func.func @f(%v: vector<4xi32>) -> vector<4xf32> "
            "{ %0 = vector.shape_cast %v : vector<4xi32> to "
            "vector<4xf32> func.return %0 : vector<4xf32> } }\n",
        ),
        _fake_validator_must_not_pass(
            "vector-multi-reduction-missing-kind",
            "module { func.func @f(%v: vector<4xi32>) -> i32 { "
            "%0 = vector.multi_reduction %v [0] : vector<4xi32> to i32 "
            "func.return %0 : i32 } }\n",
        ),
        _generic_func_signature_must_be_preserved(),
        _gpu_backend_symbol_must_be_bound(),
        _backend_shape_must_reject(
            "llvm-typed-value-shape",
            MLIRBackendTarget.LLVM_IR,
            "define i32 @expected() { ret i32 true }\n",
        ),
        _backend_shape_must_reject(
            "hip-c-like-param-shape",
            MLIRBackendTarget.ROCM_HIP,
            "#include <hip/hip_runtime.h>\n"
            "__global__ void expected(??? * p) {}\n",
        ),
        _backend_shape_must_reject(
            "wgsl-top-level-declaration-shape",
            MLIRBackendTarget.WEBGPU_WGSL,
            "alias Lane = ???;\n"
            "@compute @workgroup_size(1)\n"
            "fn expected() {}\n",
        ),
        _quoted_symbol_must_not_collapse(),
        _generic_llvm_func_symbol_must_bind(),
        _llvm_typed_value_must_reject_scalar_in_aggregate(),
        _c_like_declaration_must_reject_digit_identifier(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fast MLIR audit canaries.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any canary fails; default is report-only")
    args = parser.parse_args(argv)

    results = run_canaries()
    failures = [result for result in results if not result.passed]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    print(
        f"summary: {len(results) - len(failures)} passed, "
        f"{len(failures)} failed")
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
