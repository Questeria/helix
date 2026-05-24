"""Tests for helixc.ir.mlir.validate — v3.0 Phase E, Stages 211 + 213:
the toolchain-free and real MLIR-text validators.

`mock_validate_mlir` is the mock-path MLIR validator — a toolchain-free
STRUCTURAL shape check on MLIR textual IR (the MLIR analogue of
`llvm_ir.mock_validate_ll`). It returns a frozen tri-state
`MLIRValidation`: FAILED on a definite structural defect, DEFERRED when
the shape is clean but real validity is unverified, and — never from
the mock checker — PASSED (reserved for the Stage-212 real validator).

These tests pin: the `MLIRValidationVerdict` tri-state and its
module-load guard; the `MLIRValidation` frozen result's `__post_init__`
rejections (a FAILED / DEFERRED is never silent, a PASSED never
carries a finding); the predicates;
`mock_validate_mlir`'s defect detection (empty, no structure,
unbalanced braces / parens) and its honest DEFERRED on clean text; that
it NEVER returns a false PASSED; that string-literal / comment
punctuation is masked from the brace count; and — the mock-path rule —
that the module never `import mlir`.
"""
from __future__ import annotations

import ast
import copy
import pickle
import subprocess
from pathlib import Path

import pytest

from helixc.ir.mlir import validate
from helixc.ir.mlir.toolchain import MLIRSupport
from helixc.ir.mlir.validate import (
    MLIRValidation, MLIRValidationVerdict, mock_validate_mlir,
    validate_mlir_with_toolchain,
)

# A structurally well-formed MLIR module — the DEFERRED baseline.
_WELL_FORMED = """\
module {
  func.func @main() -> i32 {
    %0 = arith.constant 1 : i32
    return %0 : i32
  }
}
"""

_REAL_RUN_MLIR_OPT_VALIDATE = validate._run_mlir_opt_validate


def _reject_invalid_smoke(input_text: str) -> bool:
    return validate._mlir_text_is_invalid_smoke_probe(input_text)


@pytest.fixture(autouse=True)
def _trust_fake_mlir_opt_for_direct_runner(monkeypatch):
    monkeypatch.setattr(
        validate,
        "detect_mlir_support",
        lambda: MLIRSupport(
            bindings=False,
            dialects=False,
            mlir_opt="/fake/mlir-opt",
            detail=("test fake mlir-opt",),
        ),
    )


def _real_passed_validation(
        mlir_text: str = _WELL_FORMED) -> MLIRValidation:
    old_run = validate.subprocess.run

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        assert input_text == mlir_text
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    try:
        validate.subprocess.run = _fake_run
        result = _REAL_RUN_MLIR_OPT_VALIDATE(mlir_text, "/fake/mlir-opt")
    finally:
        validate.subprocess.run = old_run
    assert result.passed()
    return result


# --------------------------------------------------------------------------
# MLIRValidationVerdict — the tri-state + its guard
# --------------------------------------------------------------------------
def test_validation_verdict_members():
    """`MLIRValidationVerdict` is exactly the tri-state PASSED / FAILED
    / DEFERRED, each with a distinct string value."""
    assert {v.name for v in MLIRValidationVerdict} == {
        "PASSED", "FAILED", "DEFERRED"}
    values = [v.value for v in MLIRValidationVerdict]
    assert len(values) == len(set(values)), "values must be unique"


def test_check_validation_verdicts_guard():
    """The module-load guard `_check_validation_verdicts` is callable
    and passes for the current tri-state enum."""
    validate._check_validation_verdicts()  # must not raise


# --------------------------------------------------------------------------
# MLIRValidation — __post_init__ rejects illegal / silent results
# --------------------------------------------------------------------------
def test_mlir_validation_rejects_silent_failed():
    """A FAILED result with no findings would be silent about the
    defect it found — rejected."""
    with pytest.raises(ValueError, match="must carry at least one"):
        MLIRValidation(MLIRValidationVerdict.FAILED, ())


def test_mlir_validation_rejects_silent_deferred():
    """A DEFERRED result with no findings would be silent about why it
    deferred — rejected (the mock-path rule: never a silent DEFERRED)."""
    with pytest.raises(ValueError, match="must carry at least one"):
        MLIRValidation(MLIRValidationVerdict.DEFERRED, ())


def test_mlir_validation_rejects_blank_finding():
    """A blank / non-str finding is a reason-shaped object with no
    reason — rejected."""
    with pytest.raises(ValueError, match="blank or non-str"):
        MLIRValidation(MLIRValidationVerdict.FAILED, ("   ",))


def test_mlir_validation_rejects_mutable_findings():
    """Frozen validation results must not retain a mutable findings
    list that can be cleared after invariant checks."""
    with pytest.raises(ValueError, match="findings must be a tuple"):
        MLIRValidation(
            MLIRValidationVerdict.FAILED,
            ["a reason"],  # type: ignore[arg-type]
        )


def test_mlir_validation_rejects_passed_with_findings():
    """A PASSED carrying any finding is incoherent — `findings`
    describes a defect or a deferral reason, and a PASSED has neither.
    The frozen result rejects it, settling the Stage-211 chunk-E
    coherence carry-over: a PASSED can never carry a defect-shaped
    finding because it carries no finding at all."""
    with pytest.raises(ValueError, match="PASSED result must carry NO"):
        MLIRValidation(MLIRValidationVerdict.PASSED, ("a defect note",))


def test_mlir_validation_passed_requires_real_validator():
    """A PASSED result needs the real-validator creation path, not just
    an empty finding tuple."""
    with pytest.raises(ValueError, match="toolchain provenance"):
        MLIRValidation(MLIRValidationVerdict.PASSED, ())
    with pytest.raises(ValueError, match="registry entry"):
        MLIRValidation(
            MLIRValidationVerdict.PASSED,
            (),
            provenance=("mlir-opt=/fake/mlir-opt",),
        )
    ok = _real_passed_validation()
    assert ok.passed() and ok.findings == ()
    assert ok.provenance
    assert any(f.startswith("input_sha256=") for f in ok.provenance)
    assert not any(f.startswith("artifact=") for f in ok.provenance)


def test_mlir_validation_is_final_for_predicate_integrity():
    with pytest.raises(TypeError, match="MLIRValidation is final"):
        class ForgedValidation(MLIRValidation):
            def passed(self) -> bool:
                return True


def test_mlir_validation_passed_predicate_requires_runner_registry():
    forged = object.__new__(MLIRValidation)
    object.__setattr__(forged, "verdict", MLIRValidationVerdict.PASSED)
    object.__setattr__(forged, "findings", ())
    object.__setattr__(
        forged,
        "provenance",
        ("input_sha256=" + "0" * 64,),
    )
    assert not forged.passed()


def test_mlir_validation_passed_predicate_checks_full_pass_shape():
    forged = object.__new__(MLIRValidation)
    object.__setattr__(forged, "verdict", MLIRValidationVerdict.PASSED)
    object.__setattr__(forged, "findings", ("this should fail",))
    object.__setattr__(
        forged,
        "provenance",
        (
            "mlir-opt=/fake/mlir-opt",
            "artifact_name=out.mlir",
            "input_sha256=" + "0" * 64,
            "output_sha256=" + "0" * 64,
        ),
    )
    assert not forged.passed()


def test_mlir_validation_passed_registry_rejects_copied_fields():
    ok = _real_passed_validation()
    forged = object.__new__(MLIRValidation)
    object.__setattr__(forged, "verdict", ok.verdict)
    object.__setattr__(forged, "findings", ok.findings)
    object.__setattr__(forged, "provenance", ok.provenance)
    assert not forged.passed()
    with pytest.raises(AttributeError):
        object.__setattr__(forged, "_helix_validation_pass_token", "0" * 64)
    assert not any("secret" in name.lower()
                   for name in dir(validate._validation_runner))


def test_mlir_validation_passed_copy_keeps_runner_registry_mark():
    ok = _real_passed_validation()
    assert copy.copy(ok) is ok
    assert copy.deepcopy(ok) is ok
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(ok)


def test_mlir_validation_runner_does_not_expose_closure_brand():
    assert not hasattr(validate._run_mlir_opt_validate, "__closure__")
    assert not hasattr(
        validate._run_mlir_opt_validate,
        "_MLIRValidationRunner__passes",
    )


def test_mlir_validation_result_has_no_public_dict():
    r = MLIRValidation(MLIRValidationVerdict.FAILED, ("x",))
    assert not hasattr(r, "__dict__")


def test_mlir_validation_predicates():
    """`passed` / `failed` / `deferred` derive from the verdict, and
    exactly one holds for any result."""
    for verdict in MLIRValidationVerdict:
        if verdict is MLIRValidationVerdict.PASSED:
            r = _real_passed_validation()
        else:
            r = MLIRValidation(verdict, ("a reason",))
        flags = [r.passed(), r.failed(), r.deferred()]
        assert sum(flags) == 1, verdict
        assert r.passed() == (verdict is MLIRValidationVerdict.PASSED)
        assert r.failed() == (verdict is MLIRValidationVerdict.FAILED)
        assert r.deferred() == (
            verdict is MLIRValidationVerdict.DEFERRED)


# --------------------------------------------------------------------------
# mock_validate_mlir — defect detection
# --------------------------------------------------------------------------
def test_mock_validate_mlir_empty_is_failed():
    """Empty or whitespace-only text is a FAILED — there is no MLIR to
    validate."""
    for text in ("", "   ", "\n\n  \t\n"):
        r = mock_validate_mlir(text)
        assert r.failed(), text
        assert any("empty" in f for f in r.findings), r.findings


def test_mock_validate_mlir_no_structure_is_failed():
    """Text with neither a `module` nor a `func.func` has no top-level
    MLIR structure — FAILED."""
    r = mock_validate_mlir("%0 = arith.constant 1 : i32")
    assert r.failed()
    assert any("no top-level structure" in f for f in r.findings)


def test_mock_validate_mlir_rejects_structure_substrings():
    """A substring such as `notmodule` is not a top-level MLIR op."""
    for text in (
        "notmodule { }",
        "foo$module { }",
        "foo-module { }",
        "foo:module { }",
        "foo/module { }",
        "@module { }",
        "%func.func { }",
        "^module { }",
        "#foo<module> { }",
        "!foo<func.func> { }",
        "#foo<bar module, baz> { }",
        "!foo<bar func.func, baz> { }",
        "module<fake> { }",
        "func.func<fake> { }",
        "mod\"ignored\"ule {}",
        "func.\"ignored\"func @f() { return }",
        "foo { module { } }",
        "foo { func.func @f() { return } }",
        "#foo<(d0) -> module> { }",
        "module world",
        "hello module world",
        "func.func world",
    ):
        r = mock_validate_mlir(text)
        assert r.failed()
        assert any("no top-level structure" in f
                   or "unexpected top-level text" in f
                   or "malformed nested `func.func`" in f
                   for f in r.findings)


def test_mock_validate_mlir_accepts_generic_quoted_top_level_ops():
    """Generic MLIR operation names are quoted; do not false-fail
    `"builtin.module"() ...` before real validation can run."""
    for text in (
        '"builtin.module"() ({}) : () -> ()',
        '"module"() ({}) : () -> ()',
        '"func.func"() : () -> ()',
    ):
        r = mock_validate_mlir(text)
        assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_multiline_generic_quoted_module():
    r = mock_validate_mlir(
        '"builtin.module"() ({\n'
        '  func.func @f() { return }\n'
        '}) : () -> ()\n')
    assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_valid_custom_top_level_forms():
    """The structural gate recognizes valid custom forms with symbols,
    attributes, or visibility and lets real MLIR validation judge them."""
    for text in (
        "module @m { }",
        "module @my-module { }",
        'module @"my-module" { }',
        "module attributes {test.attr = true} { }",
        "func.func private @abort()",
        "func.func private @scribble(i32, i64, memref<?x128xf32>) -> f64",
        "func.func @f(\n"
        "  %arg0: i32\n"
        ") {\n"
        "  return\n"
        "}\n",
        "func.func @f() -> i32 {\n"
        "  %0 = arith.constant 1 : i32\n"
        "  return %0 : i32\n"
        "}\n",
        "func.func @f() -> i32 attributes {test.attr = true} "
        "{ return }",
        "func.func @count(%x: i64) -> (i64, i64) attributes "
        '{fruit = "banana"} { return %x, %x : i64, i64 }\n',
        "func.func @f() -> tuple<i32, f32> { return }",
        "func.func @f() -> memref<4x4xf32, strided<[4, 1]>> "
        "{ return }",
        'module { } loc("m")\n',
        'func.func @f() { return } loc("f")\n',
        "func.func private @example_fn_attr() attributes "
        "{dialectName.attrName = false}",
        "module { func.func @f() { %module = arith.constant 0 : i32\n"
        "return } }",
        'module { func.func @f() { %0 = arith.constant 1 : i32 '
        'loc("c")\nreturn } }',
        'module { func.func @f() { func.return loc("r") } }',
        'module { func.func @f() { return loc("y") } }',
        "module { func.func @f(%idx: index) { %0 = arith.index_cast %idx "
        ": index to i32\nreturn } }",
        "module { func.func @g(i32) -> i32\n"
        "func.func @f(%arg0: i32) { %0 = func.call @g(%arg0) "
        ": (i32) -> i32\nreturn } }",
        "module { func.func @side_effect()\n"
        "func.func @caller() { func.call @side_effect() "
        ": () -> ()\nfunc.return } }",
        "module { func.func @f(%A: memref<4xf32>, %i: index, "
        "%c0: f32) { %0 = vector.transfer_read %A[%i], %c0 : "
        "memref<4xf32>, vector<4xf32>\nreturn } }",
        "module {\n"
        "  func.func @f(%A: memref<4xf32>, %i: index, %c0: f32) {\n"
        "    %0 = vector.transfer_read %A[%i], %c0\n"
        "      : memref<4xf32>, vector<4xf32>\n"
        "    return\n"
        "  }\n"
        "}\n",
        'module { func.func @f() { "test.op"() : () -> '
        'memref<4xf32, affine_map<(d0) -> (d0)>> } }',
        "module { func.func @f(%c: i1) {\n"
        "scf.if %c {\n"
        "scf.yield\n"
        "} else {\n"
        "scf.yield\n"
        "}\n"
        "func.return\n"
        "} }",
        '"test.region"() ({\n'
        "^bb0:\n"
        '  "test.yield"() : () -> ()\n'
        "}) : () -> ()",
        "module { func.func @f() { ^bb0:\nreturn } }",
        "module { func.func @f() { ^bb0(%arg0: i32):\nreturn } }",
        "module { func.func @func.func() { return } }",
        "module { func.func @module() { return } }",
        "module { func.func @my-module() { return } }",
        "func.func @f() attributes {module = true} { return }",
        "module { func.func private @abort()\n"
        "func.func @main() { return }\n"
        "}\n",
        "module { gpu.module @kernels { } }",
        "module { gpu.module @symbol_name2 "
        "<#gpu.select_object<1>> [#nvvm.target] { } }",
        "module { gpu.module @kernels { gpu.func @kernel() kernel { "
        "%0 = arith.constant 1 : i32\n gpu.return } } }",
        "module {\n"
        "  func.func @f(%init: i32, %cond: i1) {\n"
        "    %res = scf.while (%arg = %init) : (i32) -> i32 {\n"
        "      scf.condition(%cond) %arg : i32\n"
        "    } do {\n"
        "    ^bb0(%arg2: i32):\n"
        "      scf.yield %arg2 : i32\n"
        "    }\n"
        "    return\n"
        "  }\n"
        "}\n",
        "module {\n"
        "  func.func @copy(%A: memref<4xf32>, %B: memref<4xf32>) {\n"
        "    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>],\n"
        "                    iterator_types = []}\n"
        "      ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {\n"
        "    ^bb0(%a: f32, %b: f32):\n"
        "      linalg.yield %a : f32\n"
        "    }\n"
        "    return\n"
        "  }\n"
        "}\n",
        "module {\n"
        "  func.func @kernel(%bx: index, %by: index, %bz: index,\n"
        "                    %tx: index, %ty: index, %tz: index) {\n"
        "    gpu.launch blocks(%bx, %by, %bz) in (%bx, %by, %bz)\n"
        "      threads(%tx, %ty, %tz) in (%tx, %ty, %tz) {\n"
        "      gpu.terminator\n"
        "    }\n"
        "    return\n"
        "  }\n"
        "}\n",
        "module {\n"
        "  func.func @kernel(%bx: index, %by: index, %bz: index,\n"
        "                    %tx: index, %ty: index, %tz: index,\n"
        "                    %smem: i32) {\n"
        "    gpu.launch blocks(%bx, %by, %bz) in (%bx, %by, %bz)\n"
        "      threads(%tx, %ty, %tz) in (%tx, %ty, %tz)\n"
        "      dynamic_shared_memory_size %smem {\n"
        "      gpu.terminator\n"
        "    }\n"
        "    return\n"
        "  }\n"
        "}\n",
    ):
        r = mock_validate_mlir(text)
        assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_affine_arrow_in_angle_context():
    r = mock_validate_mlir("#map = affine_map<(d0) -> (d0)>\nmodule {}\n")
    assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_multiline_aliases():
    r = mock_validate_mlir(
        "#map = affine_map<\n"
        "  (d0) -> (d0)\n"
        ">\n"
        "module {}\n")
    assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_affine_set_comparisons():
    for text in (
        "#set = affine_set<(d0) : (d0 >= 0)>\nmodule {}\n",
        "#set = affine_set<(d0) : (d0 <= 0)>\nmodule {}\n",
    ):
        r = mock_validate_mlir(text)
        assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_property_dictionary_syntax():
    for text in (
        'module { "test.op"() <{foo = 1 : i32}> : () -> () }',
        'module { "test.op"() <{set = affine_set<(d0) : '
        '(d0 <= 0)>}> : () -> () }',
    ):
        r = mock_validate_mlir(text)
        assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_generic_function_type_properties():
    r = mock_validate_mlir(
        '"func.func"() <{function_type = () -> (), '
        'sym_name = "func"}> ({}) : () -> ()\n')
    assert r.deferred(), r.findings


def test_mock_validate_mlir_typed_bare_module_op_does_not_raise():
    r = mock_validate_mlir("module { foo.bar : () -> () }")
    assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_func_result_attributes():
    r = mock_validate_mlir(
        "func.func private @example_fn_result() -> "
        "(f64 {dialectName.attrName = 0 : i64})")
    assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_nested_generic_func_ops():
    for text in (
        'module { "func.func"() : () -> () }',
        'module { "func.func"() <{function_type = () -> (), '
        'sym_name = "func"}> ({}) : () -> () }',
        'module { func.func @f() { "test.op"() : () -> () } }',
        'module { func.func @f() -> i32 { %0 = "arith.constant"() '
        '{value = 1 : i32} : () -> i32 return %0 : i32 } }',
    ):
        r = mock_validate_mlir(text)
        assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_custom_top_level_generic_ops():
    r = mock_validate_mlir('"test.top"() : () -> ()')
    assert r.deferred(), r.findings


def test_mock_validate_mlir_rejects_return_type_mismatch():
    r = mock_validate_mlir(
        "module { func.func @main() -> i32 { "
        "%0 = arith.constant 1.0 : f32 return %0 : i32 } }")
    assert r.failed()
    assert any("return type mismatch" in f for f in r.findings), r.findings


def test_mock_validate_mlir_rejects_undefined_operation_operand():
    r = mock_validate_mlir(
        "module { func.func @main(%x: i32) -> i32 { "
        "%0 = arith.addi %x, %missing : i32 return %0 : i32 } }")
    assert r.failed()
    assert any("undefined SSA value in operation" in f
               for f in r.findings), r.findings


def test_mock_validate_mlir_allows_block_argument_operands():
    r = mock_validate_mlir(
        "module { func.func @main(%c: i1, %x: i32) -> i32 {\n"
        "cf.cond_br %c, ^bb1(%x : i32), ^bb2\n"
        "^bb1(%a: i32):\n"
        "%0 = arith.addi %a, %a : i32\n"
        "return %0 : i32\n"
        "^bb2:\n"
        "return %x : i32\n"
        "} }")
    assert r.deferred(), r.findings


def test_mock_validate_mlir_rejects_duplicate_function_argument():
    r = mock_validate_mlir(
        "module { func.func @main(%x: i32, %x: i32) -> i32 { "
        "return %x : i32 } }")
    assert r.failed()
    assert any("duplicate function argument" in f
               for f in r.findings), r.findings


def test_mock_validate_mlir_rejects_return_arity_mismatch():
    r = mock_validate_mlir(
        "module { func.func @main(%x: i32, %y: i32) -> i32 { "
        "return %x, %y : i32 } }")
    assert r.failed()
    assert any("return arity mismatch" in f for f in r.findings), r.findings


def test_mock_validate_mlir_rejects_undefined_branch_target():
    r = mock_validate_mlir(
        "module { func.func @main(%c: i1) { "
        "cf.cond_br %c, ^bb1, ^missing ^bb1: return } }")
    assert r.failed()
    assert any("undefined block target" in f for f in r.findings), r.findings


def test_mock_validate_mlir_rejects_branch_target_arg_mismatch():
    r = mock_validate_mlir(
        "module { func.func @main(%x: i32) {\n"
        "cf.br ^bb1\n"
        "^bb1(%a: i32):\n"
        "return\n"
        "} }")
    assert r.failed()
    assert any("block target argument mismatch" in f
               for f in r.findings), r.findings


def test_mock_validate_mlir_accepts_linalg_fill_and_vector_reduction():
    text = """module {
  func.func @f(%cst: f32, %out: memref<?xf32>, %v: vector<4xf32>) {
    linalg.fill ins(%cst : f32) outs(%out : memref<?xf32>)
    %r = vector.multi_reduction <add>, %v [0] : vector<4xf32> to f32
    return
  }
}
"""
    r = mock_validate_mlir(text)
    assert r.deferred(), r.findings


def test_mock_validate_mlir_accepts_custom_result_type_ops():
    for text in (
        "module { func.func @cmp(%a: i32, %b: i32) -> i1 { "
        "%0 = arith.cmpi slt, %a, %b : i32 return %0 : i1 } }",
        "module { func.func @vcmp(%a: vector<4xi32>, "
        "%b: vector<4xi32>) -> vector<4xi1> { "
        "%0 = arith.cmpi slt, %a, %b : vector<4xi32> "
        "return %0 : vector<4xi1> } }",
        "module { func.func @tcmp(%a: tensor<4xf32>, "
        "%b: tensor<4xf32>) -> tensor<4xi1> { "
        "%0 = arith.cmpf olt, %a, %b : tensor<4xf32> "
        "return %0 : tensor<4xi1> } }",
        "module { func.func @v0(%a: vector<f32>, "
        "%b: vector<f32>) -> vector<f32> { "
        "%0 = arith.addf %a, %b : vector<f32> "
        "return %0 : vector<f32> } }",
        "module { func.func @idx(%idx: index) -> i32 { "
        "%0 = arith.index_cast %idx : index to i32 "
        "return %0 : i32 } }",
        "module { func.func @f(%v: vector<4xf32>) -> f32 { "
        "%0 = vector.multi_reduction <add>, %v [0] "
        ": vector<4xf32> to f32 return %0 : f32 } }",
    ):
        r = mock_validate_mlir(text)
        assert r.deferred(), r.findings


def test_mock_validate_mlir_rejects_incomplete_top_level_headers():
    for text in (
        "module @m",
        "module attributes",
        "module attributes {foo = true}",
        "module @m attributes {foo = true}",
        "func.func @f",
        "func.func private @f",
        "func.func @f() garbage",
        "func.func @f() -> i32 garbage",
        "func.func @f() -> i32 @bad",
        "func.func @f() -> i32 garbage!",
        "func.func @f() -> i32 , garbage",
        "module { func.func @f(%a: i32 junk) { return } }",
        "module { func.func @f(%: i32) { return } }",
        "module { func.func @f(i32) { return } }",
        "module { func.func @f() garbage }",
        "module { func.func @f() -> i32 garbage }",
        "module { func.func @f() -> i32, garbage }",
        "module @ { }",
        "module @- { }",
        "module @m, { }",
        "module @bad# { }",
        "func.func @ ()",
        "func.func @-() { return }",
        "func.func @f,() { return }",
        "func.func @f() attributes {foo = true} garbage",
        "func.func @f() -> i32 attributes {foo = true} garbage",
        "module { module @ { } }",
        "module { func.func @outer() { func.func @bad } }",
        "module { func.func @outer() { func.func @inner() "
        "{ return } } }",
        "module { func.func @outer() { module {} } }",
        "module { func.func @f(%c: i1) { scf.if %c { module {} junk } "
        "func.return } }",
        "module { func.func @f(%c: i1) { scf.if %c { junk } "
        "func.return } }",
        "module { func.func @f(%c: i1) { scf.if %c { foo = bar } "
        "func.return } }",
        "module { func.func @f() { return junk } }",
        "module { func.func @f() { return : i32 } }",
        "module { func.func @f() { func.return : i32 } }",
        "module { func.func @f() { scf.yield junk } }",
        "module { func.func @f() { scf.yield : i32 } }",
        "module { func.func @f() { return } garbage }",
        "module {\n  func.func @f() {\n    else {\n"
        "      return\n    }\n  }\n}",
        "module { module {} garbage }",
        'module { "test.op"() garbage : () -> () }',
        'module { "test.op"() ({ garbage }) : () -> () }',
        'module { "test.op"() ({}, { junk }) : () -> () }',
        'module { func.func @f() { "test.op"() : () -> () junk } }',
        'module { func.func @f() { %0 junk = "test.op"() '
        ': () -> i32 return } }',
        "module { func.func @f() { % = arith.constant 1 : i32\n"
        "return } }",
        'module { func.func @f() { arith.constant 1 : i32 junk } }',
        'module { func.func @f() { foo.bar junk\nreturn } }',
        'module { func.func @f() { test.op {foo = } : () -> ()\n'
        'return } }',
        "module { gpu.module @kernels { junk } }",
        "module { gpu.module @kernels <#gpu.select_object<1>> junk { } }",
        'module { "test.op"() <{foo = 1 : i32} junk> : () -> () }',
        '"test.op"() <{foo = }> : () -> ()',
        'module { "test.op"() <{foo = 1 : i32 junk}> : () -> () }',
        'module { "test.op"() : (i32 junk) -> () }',
        'module { "test.op"() : () -> (i32 junk) }',
        "_generic_op() : () -> ()",
        "module { _generic_op() : () -> () }",
        "module { func.func @f() { _generic_op() : () -> () } }",
        "\x00helix_generic_op\x00() : () -> ()",
        "module { func.func @f() { ^\nreturn } }",
        "module { func.func @f() { ^bb0\nreturn } }",
        "module { func.func @f() { ^!!:\nreturn } }",
        "module { func.func @f() { {} return } }",
        "module { func.func @f() {\n"
        "%0 = arith.constant 1 : i32\n"
        "func.return %0\n"
        "%1 = arith.constant 2 : i32\n"
        "} }",
        'module { "test.region"() ({\n^\n}) : () -> () }',
        "module () ({}) : () -> ()",
        "func.func () : () -> ()",
        'module { func.func @f() { "test.op" garbage : () -> () } }',
        "module @ attributes {foo = true} { }",
        "module @m\nmodule {}\n",
        "#bad =\nmodule {}\n",
        "!bad =\nmodule {}\n",
        "func.func @f\nmodule {}\n",
        "func.func @f\nmodule { func.func @g() { return } }",
        "garbage\nmodule {}\n",
        "module {}\ngarbage",
        "#garbage module {}",
        "[]\nmodule {}",
        "()\nmodule {}",
        "#bad\nmodule {}",
        "!bad\nmodule {}",
        '"module"() garbage : () -> ()',
        '"module"() ({}) : () -> () garbage',
    ):
        r = mock_validate_mlir(text)
        assert r.failed()
        assert any("no top-level structure" in f
                   or "unexpected top-level text" in f
                   or "malformed nested `func.func`" in f
                   or "malformed nested `module`" in f
                   or "module operation list" in f
                   or "function body" in f
                   or "terminator" in f
                   or "block label" in f
                   or "standalone region" in f
                   or "NUL byte" in f
                   for f in r.findings)


def test_mock_validate_mlir_allows_known_no_type_branch_ops():
    text = """module {
  func.func @f(%c: i1) {
    cf.assert %c, "bad"
    cf.cond_br %c, ^bb1, ^bb2
  ^bb1:
    gpu.barrier
    return
  ^bb2:
    return
  }
}
"""
    r = mock_validate_mlir(text)
    assert r.deferred(), r.findings


def test_mock_validate_mlir_allows_compact_typed_ops_before_next_op():
    text = (
        "module { func.func @f() { "
        "%c0 = arith.constant 0 : index "
        "scf.for %i = %c0 to %c0 step %c0 { scf.yield } "
        "return } }"
    )
    r = mock_validate_mlir(text)
    assert r.deferred(), r.findings


def test_mock_validate_mlir_unbalanced_braces_is_failed():
    """An unbalanced brace is a definite structural defect — FAILED,
    with the open/close counts named."""
    r = mock_validate_mlir("module {\n  func.func @f() {\n")
    assert r.failed()
    assert any("unbalanced brace" in f for f in r.findings), r.findings


def test_mock_validate_mlir_misordered_braces_is_failed():
    """Equal delimiter counts are not enough; a close-before-open brace
    is structurally malformed and must fail closed."""
    r = mock_validate_mlir("module }{")
    assert r.failed()
    assert any("misordered brace" in f for f in r.findings), r.findings


def test_mock_validate_mlir_unbalanced_parens_is_failed():
    """An unbalanced parenthesis is a definite structural defect —
    FAILED. The test input is brace-BALANCED, so the finding isolates
    the parenthesis defect (no spurious brace finding)."""
    r = mock_validate_mlir(
        "module { func.func @f(%a: i32) ) { return } }")
    assert r.failed()
    assert any("unbalanced parenthes" in f for f in r.findings), \
        r.findings
    assert not any("brace" in f for f in r.findings), r.findings


def test_mock_validate_mlir_misordered_parens_is_failed():
    """Equal paren counts with `)(` order are structurally malformed."""
    r = mock_validate_mlir("module { func.func @f)( { return } }")
    assert r.failed()
    assert any("misordered parenthes" in f for f in r.findings), \
        r.findings


def test_mock_validate_mlir_misnested_delimiters_is_failed():
    """Cross-type nesting such as `{ ( } )` is structurally malformed."""
    r = mock_validate_mlir("module { func.func @f( } )")
    assert r.failed()
    assert any("misnested delimiters" in f for f in r.findings), \
        r.findings


def test_mock_validate_mlir_unbalanced_square_brackets_is_failed():
    r = mock_validate_mlir(
        "module { func.func @f() { %0 = foo.bar [1, 2 : i32 } }")
    assert r.failed()
    assert any("unbalanced square bracket" in f for f in r.findings), \
        r.findings


def test_mock_validate_mlir_unbalanced_angle_brackets_is_failed():
    r = mock_validate_mlir(
        "module { func.func @f(%x: memref<4xi32) { return } }")
    assert r.failed()
    assert any("unbalanced angle bracket" in f for f in r.findings), \
        r.findings


def test_mock_validate_mlir_extra_angle_closer_is_failed():
    r = mock_validate_mlir(
        "module { func.func @f(%x: memref<4xi32>>) { return } }")
    assert r.failed()
    assert any("angle bracket" in f for f in r.findings), r.findings


def test_mock_validate_mlir_rejects_non_str_without_raising():
    """A non-str argument is itself a FAILED — `mock_validate_mlir`
    NEVER raises. A caller that passes `None` (an upstream lowering
    that produced nothing) gets a named defect, not an opaque
    `AttributeError`."""
    for bad in (None, 123, b"module {}", ["module"]):
        r = mock_validate_mlir(bad)        # must not raise
        assert r.failed(), bad
        assert any("not MLIR text" in f for f in r.findings), r.findings


def test_mock_validate_mlir_unterminated_string_is_failed():
    """An unterminated string literal is FAILED — and reported AS an
    unterminated string, NOT misattributed to a brace imbalance the
    dangling quote would otherwise fake (the balance checks are
    skipped when a quote dangles)."""
    r = mock_validate_mlir(
        'module { func.func @f() { %0 = x.y "oops }')
    assert r.failed()
    assert any("unterminated string" in f for f in r.findings), \
        r.findings
    assert not any("unbalanced brace" in f for f in r.findings), \
        r.findings


# --------------------------------------------------------------------------
# mock_validate_mlir — well-formed text DEFERS, never falsely PASSES
# --------------------------------------------------------------------------
def test_mock_validate_mlir_well_formed_defers():
    """A structurally well-formed module DEFERS — the toolchain-free
    check found no defect, but cannot certify real validity, so it is
    honestly DEFERRED, not a false PASSED. The finding explains."""
    r = mock_validate_mlir(_WELL_FORMED)
    assert r.deferred()
    assert any("mlir-opt" in f for f in r.findings), r.findings


def test_mock_validate_mlir_never_returns_passed():
    """`mock_validate_mlir` — being toolchain-free — NEVER returns
    PASSED for any input: it can only confidently FAIL or honestly
    DEFER. PASSED is reserved for the Stage-212 real validator."""
    samples = (
        _WELL_FORMED, "", "   ", "module {}",
        "func.func @f() { return }",
        "module {\n  func.func @f() {\n",      # unbalanced
        "%0 = arith.constant 1 : i32",          # no structure
        'module { func.func @g() { "s}(" } }',  # quoted punctuation
    )
    for text in samples:
        assert mock_validate_mlir(text).verdict is not \
            MLIRValidationVerdict.PASSED, text


def test_mock_validate_mlir_bare_func_defers():
    """A top-level `func.func` with no enclosing `module` is still
    recognised structure — MLIR allows it — so a balanced one DEFERS."""
    r = mock_validate_mlir("func.func @f() {\n  return\n}\n")
    assert r.deferred()


# --------------------------------------------------------------------------
# validate_mlir_with_toolchain — real mlir-opt dispatch seam
# --------------------------------------------------------------------------
def test_validate_mlir_with_toolchain_mock_failure_skips_support_probe(
        monkeypatch):
    """A definite mock structural failure returns immediately. Tool
    probing must not run and mask the real input defect."""
    def _boom():
        raise AssertionError("support probe should not run")

    monkeypatch.setattr(validate, "detect_mlir_support", _boom)
    r = validate_mlir_with_toolchain("module {")
    assert r.failed()
    assert any("unbalanced brace" in f for f in r.findings)


def test_validate_mlir_with_toolchain_absent_mlir_opt_defers():
    """A clean mock shape with no `mlir-opt` remains DEFERRED, with
    the support details preserved."""
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt=None,
        detail=("`mlir-opt` is not on PATH",),
    )
    r = validate_mlir_with_toolchain(_WELL_FORMED, support=support)
    assert r.deferred()
    assert any("mlir-opt" in f for f in r.findings), r.findings
    assert any("support probe" in f for f in r.findings), r.findings


def test_validate_mlir_with_toolchain_injected_mlir_opt_defers_if_untrusted(
        monkeypatch):
    monkeypatch.setattr(
        validate,
        "detect_mlir_support",
        lambda: MLIRSupport(
            bindings=False,
            dialects=False,
            mlir_opt=None,
            detail=("`mlir-opt` is not on PATH",),
        ),
    )
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/fake/mlir-opt",
        detail=("caller supplied fake",),
    )
    r = validate_mlir_with_toolchain(_WELL_FORMED, support=support)
    assert r.deferred()
    assert any("caller-supplied MLIRSupport" in f for f in r.findings)


def test_validate_mlir_with_toolchain_rejects_bad_support():
    with pytest.raises(ValueError, match="support must be"):
        validate_mlir_with_toolchain(
            _WELL_FORMED,
            support="not support",  # type: ignore[arg-type]
        )


def test_validate_mlir_with_toolchain_mlir_opt_success(monkeypatch):
    """When `mlir-opt` succeeds, the real validation result is PASSED
    with no findings."""
    def _fake_run(mlir_text, mlir_opt):
        assert mlir_text == _WELL_FORMED
        assert mlir_opt == "/fake/mlir-opt"
        return _real_passed_validation()

    monkeypatch.setattr(validate, "_run_mlir_opt_validate", _fake_run)
    monkeypatch.setattr(
        validate,
        "detect_mlir_support",
        lambda: MLIRSupport(
            bindings=False,
            dialects=False,
            mlir_opt="/fake/mlir-opt",
            detail=("`mlir-opt` is on PATH at '/fake/mlir-opt'",),
        ),
    )
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/fake/mlir-opt",
        detail=("`mlir-opt` is on PATH at '/fake/mlir-opt'",),
    )
    r = validate_mlir_with_toolchain(_WELL_FORMED, support=support)
    assert r.passed()
    assert r.findings == ()


def test_validate_mlir_with_toolchain_mlir_opt_failure(monkeypatch):
    """A real verifier rejection stays FAILED; it is not downgraded to
    DEFERRED after the tool was selected."""
    def _fake_run(mlir_text, mlir_opt):
        return MLIRValidation(
            MLIRValidationVerdict.FAILED,
            ("mlir-opt exit 1: bad IR",),
        )

    monkeypatch.setattr(validate, "_run_mlir_opt_validate", _fake_run)
    monkeypatch.setattr(
        validate,
        "detect_mlir_support",
        lambda: MLIRSupport(
            bindings=False,
            dialects=False,
            mlir_opt="/fake/mlir-opt",
            detail=("`mlir-opt` is on PATH at '/fake/mlir-opt'",),
        ),
    )
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/fake/mlir-opt",
        detail=("`mlir-opt` is on PATH at '/fake/mlir-opt'",),
    )
    r = validate_mlir_with_toolchain(_WELL_FORMED, support=support)
    assert r.failed()
    assert any("bad IR" in f for f in r.findings)


def test_run_mlir_opt_validate_tool_not_found_is_failed():
    """A vanished or missing `mlir-opt` is a structured FAILED result,
    never an uncaught FileNotFoundError."""
    r = validate._run_mlir_opt_validate(
        _WELL_FORMED, "helix_no_such_mlir_opt_xyz123")
    assert r.failed()
    assert any("fresh detect_mlir_support" in f for f in r.findings), \
        r.findings


def test_run_mlir_opt_validate_success_writes_and_checks_artifact(
        monkeypatch):
    """The real-dispatch helper must pass `-o <artifact>` and require a
    non-empty output artifact for PASS."""
    seen: dict[str, object] = {}

    def _fake_run(cmd, *, capture_output, text, timeout):
        seen["cmd"] = cmd
        seen["capture_output"] = capture_output
        seen["text"] = text
        seen["timeout"] = timeout
        in_path = cmd[1]
        out_path = cmd[3]
        assert cmd[2] == "-o"
        input_text = Path(in_path).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        assert input_text == _WELL_FORMED
        Path(out_path).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.passed()
    assert seen["cmd"][0] == "/fake/mlir-opt"
    assert seen["capture_output"] is True
    assert seen["text"] is True


def test_run_mlir_opt_validate_rejects_fake_tool_that_accepts_smoke(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[3]
        Path(out_path).write_text("module {}\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("invalid-IR smoke check accepted" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_fake_tool_echoing_bad_ir(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if "__helix_invalid_smoke" in input_text:
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("smoke check accepted" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_fake_tool_keyed_to_fixed_smokes(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if ("__helix_invalid_smoke" in input_text
                or "__helix_invalid_type_smoke" in input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("smoke check accepted" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_fake_tool_keyed_to_helix_marker(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if "__helix" in input_text:
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("smoke check accepted" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_same_shape_constant_rewrite(
        monkeypatch):
    source = (
        "module { func.func @main() -> i32 { "
        "%0 = arith.constant 1 : i32 return %0 : i32 } }\n"
    )
    rewritten = source.replace("constant 1 : i32", "constant 999 : i32")

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("literal/attribute values" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_dropped_second_anonymous_op(
        monkeypatch):
    source = 'module { "test.foo"() : () -> () "test.bar"() : () -> () }\n'

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(
            'module { "test.foo"() : () -> () }\n',
            encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("operation structure" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_same_shape_ssa_operand_rewrite(
        monkeypatch):
    source = (
        "module { func.func @main(%x: i32, %y: i32) -> i32 { "
        "%0 = arith.addi %x, %x : i32 return %0 : i32 } }\n"
    )
    rewritten = source.replace("arith.addi %x, %x", "arith.addi %x, %y")

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("SSA value references" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_allows_consistent_ssa_renaming():
    source = (
        "module { func.func @f() -> i32 { "
        "%0 = arith.constant 1 : i32 return %0 : i32 } }\n"
    )
    renamed = (
        "module { func.func @f() -> i32 { "
        "%c1_i32 = arith.constant 1 : i32 return %c1_i32 : i32 } }\n"
    )
    assert not validate._mlir_opt_output_correspondence_findings(
        source, renamed)


def test_run_mlir_opt_validate_rejects_same_shape_block_label_rewrite(
        monkeypatch):
    source = (
        "module { func.func @main(%c: i1) {\n"
        "cf.cond_br %c, ^bb1, ^bb2\n"
        "^bb1:\n"
        "return\n"
        "^bb2:\n"
        "return\n"
        "} }\n"
    )
    rewritten = source.replace("cf.cond_br %c, ^bb1, ^bb2",
                               "cf.cond_br %c, ^bb2, ^bb1")

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("block label references" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_changed_symbol_uses(monkeypatch):
    source = (
        "module { func.func @foo() { return } "
        "func.func @main() { func.call @foo() : () -> () return } }\n"
    )
    rewritten = (
        "module { func.func @foo() { return } "
        "func.func @main() { func.call @evil() : () -> () return } }\n"
    )
    assert any("symbol reference structure" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("symbol reference structure" in f or "undefined callee" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_same_shape_type_rewrite(monkeypatch):
    source = (
        "module { func.func @main() -> i32 { "
        "%0 = arith.constant 1 : i32 return %0 : i32 } }\n"
    )
    rewritten = source.replace("constant 1 : i32", "constant 1 : i64")
    assert any("type annotations" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("type annotations" in f or "return type mismatch" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_same_shape_type_reorder(monkeypatch):
    source = (
        "module { func.func @main() { "
        '"test.pair"() : () -> (i32, f32) return } }\n'
    )
    rewritten = source.replace("(i32, f32)", "(f32, i32)")
    assert any("type annotations" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("type annotations" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_same_shape_attribute_rewrite(
        monkeypatch):
    source = (
        "module { func.func @main(%x: f32) -> f32 { "
        "%0 = arith.addf %x, %x fastmath<nnan> : f32 "
        "return %0 : f32 } }\n"
    )
    rewritten = source.replace("fastmath<nnan>", "fastmath<contract>")
    assert any("attribute/property payloads" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("attribute/property payloads" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_same_shape_generic_property_rewrite(
        monkeypatch):
    source = (
        'module { "func.func"() '
        '<{function_type = (i32) -> i32, sym_name = "f"}> '
        '({}) : () -> () }\n'
    )
    rewritten = source.replace("(i32) -> i32", "(i64) -> i64")
    assert any("function interfaces" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("function interfaces" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_dropped_generic_property(monkeypatch):
    source = 'module { "test.op"() <{enabled = true : i1}> : () -> () }\n'
    rewritten = 'module { "test.op"() : () -> () }\n'
    assert any("generic operation properties" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("generic operation properties" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_dropped_non_func_function_type():
    source = (
        'module { "test.op"() <{function_type = (i32) -> i64}> '
        ': () -> () }\n'
    )
    rewritten = 'module { "test.op"() : () -> () }\n'
    assert any("generic operation properties" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))


@pytest.mark.parametrize("value", ("#foo.bar", "unit",
                                   "affine_map<(d0) -> (d0)>"))
def test_run_mlir_opt_validate_rejects_dropped_nonliteral_generic_value(
        value):
    source = f'module {{ "test.op"() <{{value = {value}}}> : () -> () }}\n'
    rewritten = 'module { "test.op"() : () -> () }\n'
    assert any("generic operation properties" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))


def test_run_mlir_opt_validate_rejects_bare_function_type_after_func():
    source = (
        'module { "func.func"() '
        '<{function_type = () -> (), sym_name = "f"}> '
        '({}) : () -> () test.op <{function_type = () -> ()}> '
        ': () -> () }\n'
    )
    rewritten = 'module { func.func @f() { return } test.op : () -> () }\n'
    assert any("generic operation properties" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))


def test_run_mlir_opt_validate_rejects_same_shape_bool_rewrite(monkeypatch):
    source = (
        "module { func.func @main() -> i1 { "
        "%0 = arith.constant true : i1 return %0 : i1 } }\n"
    )
    rewritten = source.replace("constant true", "constant false")
    assert any("literal/attribute values" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("literal/attribute values" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_attribute_dict_rewrite(monkeypatch):
    source = 'module { "test.op"() {flag} : () -> () }\n'
    rewritten = 'module { "test.op"() {} : () -> () }\n'
    assert any("attribute dictionaries" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("attribute dictionaries" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_static_preflight_blocks_fake_echo(
        monkeypatch):
    bad = (
        "module { func.func @main() -> i32 { "
        "%0 = arith.constant 1.0 : f32 return %0 : i32 } }\n"
    )

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(bad, "/fake/mlir-opt")
    assert r.failed()
    assert any("return type mismatch" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_static_preflight_blocks_undefined_operand_echo(
        monkeypatch):
    bad = (
        "module { func.func @main(%x: i32) -> i32 { "
        "%0 = arith.addi %x, %missing : i32 return %0 : i32 } }\n"
    )

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(bad, "/fake/mlir-opt")
    assert r.failed()
    assert any("undefined SSA value in operation" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_static_preflight_blocks_memref_store_echo(
        monkeypatch):
    bad = (
        "module { func.func @main(%m: memref<4xi32>) { "
        "memref.store %missing, %m[] : memref<4xi32> return } }\n"
    )

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(bad, "/fake/mlir-opt")
    assert r.failed()
    assert any("undefined SSA value in operation" in f
               for f in r.findings), r.findings


@pytest.mark.parametrize(
    ("bad", "expected"),
    (
        (
            "module { func.func @f() -> i32 { "
            "func.return %missing : i32 } }\n",
            "undefined SSA value in operation",
        ),
        (
            "module { func.func @f() -> i32 { "
            "func.return 1 : i32 } }\n",
            "non-SSA operands",
        ),
        (
            "module { func.func @main(%x: i32) { "
            "arith.addi %x, %x : i32 return } }\n",
            "result arity mismatch",
        ),
        (
            "module { func.func @main(%x: i32) { "
            "%0, %1 = arith.addi %x, %x : i32 return } }\n",
            "result arity mismatch",
        ),
        (
            "module { func.func @main() { %0 = return } }\n",
            "malformed nested",
        ),
        (
            "module { func.func @main() { "
            "func.call @missing() : () -> () return } }\n",
            "undefined callee",
        ),
        (
            "module { func.func @callee() { return } "
            "func.func @main() { "
            "%0 = func.call @callee() : () -> () return } }\n",
            "result arity mismatch",
        ),
        (
            "module { func.func @f() -> i32 { "
            "%0 = arith.constant 1 : i32 func.return %0 : i64 } }\n",
            "return type mismatch",
        ),
        (
            "module {\n"
            "  func.func @callee(%x: i32) -> i32\n"
            "  func.func @f(%x: i32) -> i32 {\n"
            "    %0 = func.call @callee(%x) : (i64) -> i32\n"
            "    func.return %0 : i32\n"
            "  }\n"
            "}\n",
            "func.call signature mismatch",
        ),
        (
            "module { func.func @f() { "
            "scf.yield %missing : i32 } }\n",
            "undefined SSA value in operation",
        ),
        (
            "module { func.func @f() { "
            "scf.yield 1 : i32 } }\n",
            "non-SSA operands",
        ),
        (
            "module { func.func @f() { "
            "scf.if %missing { scf.yield } return } }\n",
            "undefined SSA value in operation",
        ),
        (
            "module { func.func @f(%cond: i1) { "
            "scf.condition(%cond) %missing : i32 } }\n",
            "undefined SSA value in operation",
        ),
        (
            "module { func.func @f() { "
            "cf.assert 1, \"bad\" return } }\n",
            "non-SSA operands",
        ),
        (
            "module { func.func @f() { "
            "func.call @g(1) : (i32) -> () return } }\n",
            "non-SSA operands",
        ),
        (
            "module { func.func @f(%m: memref<4xi32>) { "
            "memref.store 1, %m[] : memref<4xi32> return } }\n",
            "non-SSA operands",
        ),
        (
            "module { func.func @f(%x: i32) -> i32 { "
            "%0 = arith.addi %x, 1 : i32 return %0 : i32 } }\n",
            "non-SSA operands",
        ),
        (
            "module { func.func @f() { "
            "%0 = arith.constant 1 : bananas return } }\n",
            "unsupported arith.constant result type",
        ),
        (
            "module { func.func @f(%arg0: i32) { "
            "%0 = arith.addf %arg0, %arg0 : i32 return } }\n",
            "requires a floating-point type",
        ),
        (
            "module { func.func @f(%x: bananas) { return } }\n",
            "unsupported function argument type",
        ),
        (
            "module { func.func @f(%x: vector<?xi32>) { return } }\n",
            "unsupported function argument type",
        ),
        (
            "module { func.func @f(%x: vector<0xi32>) { return } }\n",
            "unsupported function argument type",
        ),
        (
            "module { func.func @f(%x: vector<*xi32>) { return } }\n",
            "unsupported function argument type",
        ),
        (
            "module { func.func @f(%x: vector<[0]xi32>) { return } }\n",
            "unsupported function argument type",
        ),
        (
            "module { func.func @f() -> f32 { return } }\n",
            "declares result type",
        ),
        (
            "module { func.func @f() { } }\n",
            "missing terminator",
        ),
        (
            "module { func.func @f() { ^bb0: } }\n",
            "missing terminator",
        ),
        (
            "module { func.func @f() { "
            "%0 = arith.constant 1 : i32 } }\n",
            "malformed nested",
        ),
        (
            "module { func.func @f() -> i32 { "
            "%0 = arith.constant 1 : i32 } }\n",
            "malformed nested",
        ),
        (
            "module { func.func @f(%m: memref<4xbananas>) { return } }\n",
            "unsupported function argument type",
        ),
        (
            "module { func.func @f() { return } "
            "func.func @f() { return } }\n",
            "duplicate func.func symbol",
        ),
        (
            'module { "func.func"() '
            '<{function_type = () -> (), sym_name = "f"}> '
            '({}) : () -> () func.func @f() { return } }\n',
            "duplicate func.func symbol",
        ),
    ),
)
def test_run_mlir_opt_validate_static_preflight_blocks_control_op_echo(
        monkeypatch, bad, expected):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(bad, "/fake/mlir-opt")
    assert r.failed()
    assert any(expected in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_static_preflight_allows_loop_region_args(
        monkeypatch):
    source = """\
module {
  func.func @f(%c0: index, %m: memref<4xi32>, %init: i32, %cond: i1) {
    scf.for %i = %c0 to %c0 step %c0 {
      %v = memref.load %m[%i] : memref<4xi32>
      scf.yield
    }
    scf.while (%arg = %init) : (i32) -> i32 {
      cf.assert %cond, "ok"
      scf.condition(%cond) %arg : i32
    } do {
    ^bb0(%arg: i32):
      scf.yield %arg : i32
    }
    return
  }
}
"""

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.passed(), r.findings


def test_run_mlir_opt_validate_static_preflight_allows_dominated_block_ssa(
        monkeypatch):
    source = """\
module {
  func.func @f() -> i32 {
    %0 = arith.constant 1 : i32
    cf.br ^bb1
  ^bb1:
    return %0 : i32
  }
}
"""

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.passed(), r.findings


def test_run_mlir_opt_validate_static_preflight_allows_index_vector_arith(
        monkeypatch):
    source = (
        "module { func.func @f(%arg0: vector<4xindex>) { "
        "%0 = arith.addi %arg0, %arg0 : vector<4xindex> return } }\n"
    )

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.passed(), r.findings


def test_mlir_func_interface_stops_at_same_line_sibling_decl():
    source = (
        "module { func.func private @decl() -> i32 "
        "func.func @main() { return } }\n"
    )

    assert validate._mlir_func_interfaces(source) == frozenset((
        "@decl|private|()|i32|decl",
        "@main||()||body",
    ))


def test_mock_validate_mlir_ignores_function_type_attributes():
    source = """
module {
  func.func @callee(%x: i32 {foo = true}) -> (i32 {bar = true}) {
    return %x : i32
  }
  func.func @main(%x: i32) -> i32 {
    %0 = func.call @callee(%x) : (i32) -> i32 loc("x":1:1)
    return %0 : i32
  }
}
"""

    assert mock_validate_mlir(source).deferred()


@pytest.mark.parametrize(
    ("op_text", "ssa_types", "expected"),
    (
        (
            "scf.for %i = %missing to %ub step %step {",
            {"%ub": None, "%step": None},
            "undefined SSA value in operation: %missing",
        ),
        (
            "scf.for %i = %lb to %missing step %step {",
            {"%lb": None, "%step": None},
            "undefined SSA value in operation: %missing",
        ),
        (
            "scf.for %i = %lb to %ub step %missing {",
            {"%lb": None, "%ub": None},
            "undefined SSA value in operation: %missing",
        ),
        (
            "scf.while (%arg = %missing) : (i32) -> i32 {",
            {},
            "undefined SSA value in operation: %missing",
        ),
    ),
)
def test_static_preflight_loop_header_undefined_ssa(
        op_text, ssa_types, expected):
    assert validate._op_undefined_ssa_finding(
        op_text, ssa_types) == expected


@pytest.mark.parametrize(
    "op_text",
    (
        "scf.for %i = 1 to %ub step %step {",
        "scf.for %i = %lb to 1 step %step {",
        "scf.for %i = %lb to %ub step 1 {",
        "scf.while (%arg = 1) : (i32) -> i32 {",
    ),
)
def test_static_preflight_loop_header_literal_operands(op_text):
    finding = validate._op_non_ssa_operand_finding(
        op_text, op_text.split()[0])
    assert finding is not None
    assert "non-SSA operands" in finding


@pytest.mark.parametrize(
    ("bad", "expected"),
    (
        (
            """\
module {
  func.func @f() {
    cf.br ^bb2
  ^bb1(%x: i32):
    cf.br ^bb2
  ^bb2:
    func.return %x : i32
  }
}
""",
            "undefined SSA value in operation: %x",
        ),
        (
            """\
module {
  func.func @f() {
  ^bb1:
    cf.br ^bb1
  ^bb1:
    return
  }
}
""",
            "duplicate block label: ^bb1",
        ),
        (
            """\
module {
  func.func @f(%cond: i1) {
    scf.while (%arg) : (i32) -> i32 {
      scf.condition(%cond) %arg : i32
    } do {
    ^bb0(%arg: i32):
      scf.yield %arg : i32
    }
    return
  }
}
""",
            "region arguments must be initialized",
        ),
    ),
)
def test_run_mlir_opt_validate_static_preflight_blocks_cfg_holes(
        monkeypatch, bad, expected):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(bad, "/fake/mlir-opt")
    assert r.failed()
    assert any(expected in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_allows_multi_result_ssa_group_use(
        monkeypatch):
    source = (
        'module { func.func @f() -> i32 { %0:2 = "test.two"() '
        ': () -> (i32, i32) return %0#1 : i32 } }\n'
    )

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.passed(), r.findings


def test_run_mlir_opt_validate_static_preflight_blocks_literal_return_echo(
        monkeypatch):
    bad = "module { func.func @f() -> i32 { return 1.0 : f32 } }\n"

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(bad, "/fake/mlir-opt")
    assert r.failed()
    assert any("non-SSA return operands" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_fake_tool_non_mlir_artifact(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text("not mlir, but nonblank\n",
                                encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("not structurally valid MLIR" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_weak_artifact_shape(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text("foo.bar junk\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("canonical top-level MLIR container" in f
               or "not structurally valid MLIR" in f
               for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_mismatched_artifact_identity(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text("module {}\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("does not preserve" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_mismatched_function_interface(
        monkeypatch):
    source = (
        "module { func.func @main() -> i32 { "
        "%0 = arith.constant 1 : i32 return %0 : i32 } }\n"
    )

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(
            "module { func.func @main() { return } }\n",
            encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("function interfaces" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_dropped_anonymous_generic_op(
        monkeypatch):
    source = 'module { "test.op"() : () -> () }\n'

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text("module {}\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("operation structure" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_rejects_dropped_bare_dialect_op(monkeypatch):
    source = (
        "module { func.func @f() -> i32 { "
        "%0 = arith.constant 1 : i32 return %0 : i32 } }\n"
    )

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(
            "module { func.func @f() -> i32 { return %0 : i32 } }\n",
            encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(source, "/fake/mlir-opt")
    assert r.failed()
    assert any("undefined SSA value" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_zero_exit_diagnostic_fails(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, 0, "", "error: verifier failed")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("emitted a diagnostic" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_zero_exit_file_diagnostic_fails(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, 0, "", "tmp.mlir:1:1: warning: canonicalization skipped")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("emitted a diagnostic" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_zero_exit_remark_fails(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, 0, "", "remark: canonicalization skipped")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("emitted a diagnostic" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_smoke_paths_are_nonce_opaque(monkeypatch):
    smoke_paths: list[str] = []

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            smoke_paths.append(str(cmd[1]))
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.passed()
    assert smoke_paths
    assert not any("invalid_smoke" in path for path in smoke_paths)
    assert not any("probe" in path for path in smoke_paths)


def test_run_mlir_opt_validate_preserves_quoted_symbol_identity(monkeypatch):
    quoted = 'module { func.func @"foo-bar"() { return } }\n'

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        Path(cmd[3]).write_text("module {}\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(quoted, "/fake/mlir-opt")
    assert r.failed()
    assert any("@foo-bar" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_normalizes_legal_quoted_symbol_spelling():
    assert not validate._mlir_opt_output_correspondence_findings(
        'module { func.func @"foo"() { return } }',
        'module { func.func @foo() { return } }',
    )


def test_run_mlir_opt_validate_preserves_generic_func_symbol_identity():
    generic = (
        '"builtin.module"() ({ '
        '"func.func"() <{function_type = () -> (), sym_name = "f"}> '
        '({}) : () -> () }) : () -> ()'
    )
    assert validate._mlir_opt_output_correspondence_findings(
        generic,
        "module {}\n",
    )
    assert not validate._mlir_opt_output_correspondence_findings(
        generic,
        "module { func.func @f() { return } }\n",
    )
    generic_typed = (
        '"builtin.module"() ({ '
        '"func.func"() <{function_type = (i32) -> i64, sym_name = "f"}> '
        '({}) : () -> () }) : () -> ()'
    )
    assert validate._mlir_opt_output_correspondence_findings(
        generic_typed,
        "module { func.func @f() { return } }\n",
    )
    generic_typed_decl = (
        '"builtin.module"() ({ '
        '"func.func"() <{function_type = (i32) -> i64, sym_name = "f"}> '
        ': () -> () }) : () -> ()'
    )
    assert not validate._mlir_opt_output_correspondence_findings(
        generic_typed_decl,
        "module { func.func @f(i32) -> i64 }\n",
    )
    generic_multi_arg_decl = (
        '"builtin.module"() ({ '
        '"func.func"() '
        '<{function_type = (i32, i64) -> i64, sym_name = "f"}> '
        ': () -> () }) : () -> ()'
    )
    assert not validate._mlir_opt_output_correspondence_findings(
        generic_multi_arg_decl,
        "module { func.func @f(i32, i64) -> i64 }\n",
    )
    generic_private = (
        'module { "func.func"() <{function_type = () -> (), '
        'sym_name = "f", sym_visibility = "private"}> '
        '({}) : () -> () }'
    )
    assert not validate._mlir_opt_output_correspondence_findings(
        generic_private,
        "module { func.func private @f() { return } }\n",
    )
    generic_public = (
        'module { "func.func"() <{function_type = () -> (), '
        'sym_name = "f", sym_visibility = "public"}> '
        '({}) : () -> () }'
    )
    assert not validate._mlir_opt_output_correspondence_findings(
        generic_public,
        "module { func.func @f() { return } }\n",
    )
    assert not validate._mlir_opt_output_correspondence_findings(
        "module { func.func public @f() { return } }\n",
        "module { func.func @f() { return } }\n",
    )
    assert not validate._mlir_opt_output_correspondence_findings(
        (
            'module { "func.func"() '
            '<{function_type = () -> (i64), sym_name = "f"}> '
            '({}) : () -> () }'
        ),
        "module { func.func @f() -> i64 { return } }\n",
    )
    assert not validate._mlir_opt_output_correspondence_findings(
        (
            'module { "func.func"() '
            '<{function_type = () -> (i32, i64), sym_name = "f"}> '
            '({}) : () -> () }'
        ),
        "module { func.func @f() -> (i32, i64) { return } }\n",
    )
    generic_constant = (
        'module { func.func @f() { %0 = "arith.constant"() '
        '<{value = 1 : i32}> : () -> i32 return } }'
    )
    custom_constant = (
        "module { func.func @f() { %0 = arith.constant 1 : i32 return } }"
    )
    assert not validate._mlir_opt_output_correspondence_findings(
        generic_constant, custom_constant)


def test_mlir_structural_op_fingerprint_is_operation_bound():
    assert validate._mlir_structural_op_fingerprint(
        "module { func.func @f() { "
        "%module = arith.constant 1 : i32 return } }"
    ) == ("module", "func.func", "arith.constant")
    assert validate._mlir_structural_op_fingerprint(
        "module { func.func @f attributes {module = true} "
        "() { return } }"
    ) == ("module", "func.func")


def test_mock_validate_mlir_masks_op_name_string_literals():
    for message in ("module", "builtin.module", "func.func"):
        r = mock_validate_mlir(
            f'module {{ func.func @f(%c: i1) {{ '
            f'cf.assert %c, "{message}" return }} }}')
        assert r.deferred(), (message, r.findings)


def test_mlir_symbol_correspondence_decodes_hex_quoted_symbols():
    assert not validate._mlir_opt_output_correspondence_findings(
        'module { func.func @"foo\\2Fbar"() { return } }',
        'module { func.func @"foo/bar"() { return } }',
    )


def test_mlir_type_fingerprint_ignores_quoted_generic_prop_sentinel():
    source = (
        'module { func.func @f() { %0 = "test.op"() '
        '{note = "<{"} : () -> i32 return } }\n'
    )
    rewritten = source.replace("-> i32", "-> i64")
    assert any("type annotations" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))


def test_mlir_location_stripping_preserves_quoted_payloads():
    source = (
        'module { func.func @f(%c: i1) { '
        'cf.assert %c, "loc(secret)" return } }\n'
    )
    rewritten = source.replace('"loc(secret)"', '""')
    assert any("literal/attribute values" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   source, rewritten))

    generic_source = 'module { "test.op"() <{note = "loc(secret)"}> : () -> () }\n'
    generic_rewritten = 'module { "test.op"() <{note = ""}> : () -> () }\n'
    assert any("generic operation properties" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   generic_source, generic_rewritten))


def test_mlir_function_attribute_dict_changes_are_fingerprinted():
    assert any("attribute dictionaries" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   "module { func.func @f() "
                   "attributes {foo.attr = 1 : i32} { return } }",
                   "module { func.func @f() "
                   "attributes {bar.attr = 1 : i32} { return } }"))
    assert any("attribute dictionaries" in f for f in
               validate._mlir_opt_output_correspondence_findings(
                   "module { func.func @f(%x: i32) { "
                   "test.op %x {flag = true} : i32 return } }",
                   "module { func.func @f(%x: i32) { "
                   "test.op %x {other = true} : i32 return } }"))
    assert not validate._mlir_opt_output_correspondence_findings(
        "module { func.func @f() "
        "attributes {z.attr = 1 : i32, a.attr = 2 : i32} { return } }",
        "module { func.func @f() "
        "attributes {a.attr = 2 : i32, z.attr = 1 : i32} { return } }")
    assert not validate._mlir_opt_output_correspondence_findings(
        'module { "test.op"() <{z = true, a = false}> : () -> () }',
        'module { "test.op"() <{a = false, z = true}> : () -> () }')


def test_run_mlir_opt_validate_surrogate_input_fails():
    r = validate._run_mlir_opt_validate(
        'module { "bad\ud800" }',
        "/fake/mlir-opt",
    )
    assert r.failed()
    assert any("UnicodeEncodeError" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_nonzero_captures_diagnostic(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 7, "", "bad mlir")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("mlir-opt exit 7" in f for f in r.findings), r.findings
    assert any("bad mlir" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_nonzero_uses_stdout_if_stderr_blank(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        return subprocess.CompletedProcess(
            cmd, 7, "real verifier error", " \n")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("real verifier error" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_timeout_is_failed(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("timed out" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_subprocess_unicode_error_is_failed(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("UnicodeDecodeError" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_subprocess_value_error_is_failed():
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "bad\0tool")
    assert r.failed()
    assert any("fresh detect_mlir_support" in f for f in r.findings), \
        r.findings


def test_run_mlir_opt_validate_zero_exit_without_artifact_fails(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("produced no output artifact" in f for f in r.findings), \
        r.findings


def test_run_mlir_opt_validate_blank_artifact_fails(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        out_path = cmd[3]
        Path(out_path).write_text(" \n\t", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("blank output" in f for f in r.findings), r.findings


def test_run_mlir_opt_validate_invalid_utf8_artifact_fails(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        out_path = cmd[3]
        Path(out_path).write_bytes(b"\xff\xfe\xfa")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", _fake_run)
    r = validate._run_mlir_opt_validate(_WELL_FORMED, "/fake/mlir-opt")
    assert r.failed()
    assert any("UnicodeDecodeError" in f for f in r.findings), r.findings


# --------------------------------------------------------------------------
# mock_validate_mlir — string-literal / comment punctuation is masked
# --------------------------------------------------------------------------
def test_mock_validate_mlir_masks_quoted_punctuation():
    """A brace or parenthesis inside a string literal must NOT be
    miscounted as structural — the well-formed module stays DEFERRED
    even with `{` / `(` characters inside a quoted attribute."""
    r = mock_validate_mlir(
        'module { func.func @f() { %0 = foo.bar {tag = "a}b)c{"} '
        ': i32 return } }')
    assert r.deferred(), r.findings


def test_mock_validate_mlir_masks_dotted_string_literals():
    r = mock_validate_mlir(
        'module { func.func @f() { return loc("foo.mlir":1:2) } }')
    assert r.deferred(), r.findings

    r = mock_validate_mlir(
        'module { func.func @f(%c: i1) { '
        'cf.assert %c, "bad.value" return } }')
    assert r.deferred(), r.findings


def test_mock_validate_mlir_masks_comment_punctuation():
    """A brace inside a `//` line comment must NOT be miscounted — the
    comment's stray `}` does not unbalance a well-formed module."""
    r = mock_validate_mlir(
        "module {\n  // a dangling } in a comment\n"
        "  func.func @f() { return }\n}\n")
    assert r.deferred(), r.findings


def test_mock_validate_mlir_masks_block_comment_punctuation():
    r = mock_validate_mlir(
        "module {\n  /* a dangling } in a block comment */\n"
        "  func.func @f() { return }\n}\n")
    assert r.deferred(), r.findings


def test_run_mlir_opt_validate_ignores_block_comment_symbols():
    assert not validate._mlir_opt_output_correspondence_findings(
        'module { /* @dead */ func.func @main() { return } }',
        'module { func.func @main() { return } }',
    )


def test_mock_validate_mlir_comment_quote_cannot_mask_real_text():
    """A `"` in a line comment must not pair with a later real string
    quote and hide malformed structural text from the mock validator."""
    r = mock_validate_mlir(
        '// comment starts "\n'
        'module { func.func @f)( { %0 = foo.bar {tag = "ok"} : i32 } }')
    assert r.failed()
    assert any("misordered parenthes" in f for f in r.findings), \
        r.findings


def test_mock_validate_mlir_raw_newline_inside_string_is_failed():
    r = mock_validate_mlir(
        'module { func.func @f() { %0 = foo.bar {tag = "bad\n'
        'still"} : i32 } }')
    assert r.failed()
    assert any("unterminated string literal" in f for f in r.findings), \
        r.findings


def test_mock_validate_mlir_escaped_raw_newline_inside_string_is_failed():
    r = mock_validate_mlir(
        'module { func.func @f() { %0 = foo.bar {tag = "bad\\\n'
        'still"} : i32 } }')
    assert r.failed()
    assert any("unterminated string literal" in f for f in r.findings), \
        r.findings


def test_validate_mlir_with_toolchain_rejects_non_utf8_text_without_tool():
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt=None,
        detail=("no MLIR toolchain",),
    )
    r = validate_mlir_with_toolchain('module { "bad\ud800" }',
                                     support=support)
    assert r.failed()
    assert any("UTF-8" in f for f in r.findings), r.findings


# --------------------------------------------------------------------------
# the mock-path rule — validate is toolchain-free, never `import mlir`
# --------------------------------------------------------------------------
def test_validate_module_imports_without_mlir_bindings():
    """THE MOCK-PATH RULE (Stage 210 decision, section 3): `validate` is
    safe on machines with no MLIR bindings — it NEVER `import mlir`, at
    module top level or anywhere. Parse the module's AST and confirm
    not one `import mlir` / `from mlir ...` statement — a host-
    independent structural pin."""
    tree = ast.parse(
        Path(validate.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("mlir"), a.name
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("mlir"), node.module
