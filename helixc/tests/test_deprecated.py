"""Stage 28.7: tests for @deprecated + @since version gating."""

from __future__ import annotations

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.deprecated_pass import (
    deprecation_msg,
    since_marker,
    find_deprecated_decls,
    find_deprecation_call_sites,
    emit_warnings,
)
from helixc.check import main


def test_parse_deprecated_attribute():
    src = '@deprecated("use foo_v2") fn foo() -> i32 { 0 }'
    prog = parse(src)
    fn = prog.items[0]
    assert "deprecated" in fn.attrs
    assert deprecation_msg(fn) == "use foo_v2"


def test_parse_deprecated_no_msg():
    src = '@deprecated fn foo() -> i32 { 0 }'
    prog = parse(src)
    fn = prog.items[0]
    assert "deprecated" in fn.attrs
    assert deprecation_msg(fn) == ""


def test_parse_since():
    src = '@since("v0.3") fn new_api() -> i32 { 0 }'
    prog = parse(src)
    fn = prog.items[0]
    assert "since" in fn.attrs
    assert since_marker(fn) == "v0.3"


def test_not_deprecated():
    src = 'fn ok() -> i32 { 0 }'
    prog = parse(src)
    fn = prog.items[0]
    assert deprecation_msg(fn) is None


def test_find_deprecated_decls():
    src = """
@deprecated("old") fn old1() -> i32 { 0 }
fn ok() -> i32 { 0 }
@deprecated fn old2() -> i32 { 0 }
"""
    prog = parse(src)
    decls = find_deprecated_decls(prog)
    assert set(decls.keys()) == {"old1", "old2"}
    assert decls["old1"] == "old"
    assert decls["old2"] == ""


def test_find_call_sites_to_deprecated():
    src = """
@deprecated("use new_id") fn old_id(x: i32) -> i32 { x }
fn new_caller() -> i32 { old_id(5) }
fn another() -> i32 { old_id(3) + 1 }
fn safe() -> i32 { 42 }
"""
    prog = parse(src)
    sites = find_deprecation_call_sites(prog)
    assert len(sites) == 2
    names = {n for (n, _, _) in sites}
    assert names == {"old_id"}


def test_emit_warnings_messages():
    src = """
@deprecated("renamed to id2") fn id_old(x: i32) -> i32 { x }
fn user() -> i32 { id_old(5) }
"""
    prog = parse(src)
    out = emit_warnings(prog)
    assert len(out) == 1
    assert "renamed to id2" in out[0]
    assert "id_old" in out[0]


def test_emit_warnings_no_calls():
    src = """
@deprecated("x") fn dead() -> i32 { 0 }
fn safe() -> i32 { 42 }
"""
    prog = parse(src)
    out = emit_warnings(prog)
    assert out == []


def test_emit_warnings_returns_list():
    """Audit 28.8 C1-M1: emit_warnings should return its list, NOT
    monkey-patch `_deprecation_warnings` onto A.Program. Verify the
    return is the source of truth and multiple calls are idempotent."""
    src = """
@deprecated fn d() -> i32 { 0 }
fn u() -> i32 { d() }
"""
    prog = parse(src)
    first = emit_warnings(prog)
    assert isinstance(first, list)
    assert len(first) == 1
    # Re-invocation: same result. The pass no longer mutates prog.
    second = emit_warnings(prog)
    assert second == first
    # The monkey-patched attribute should NOT exist (caller-store model).
    assert not hasattr(prog, "_deprecation_warnings"), \
        "emit_warnings must not couple AST to pass output (Audit 28.8 C1-M1)"


def test_cli_deprecated_warning_logged(capsys, tmp_path):
    src = """
@deprecated("use new_api") fn old_api() -> i32 { 0 }
fn main() -> i32 { old_api() }
"""
    p = str(tmp_path / "in.hx")
    with open(p, "w") as f:
        f.write(src)
    rc = main([p])
    out = capsys.readouterr().out
    assert "deprecated:" in out
    assert "old_api" in out
    assert rc == 0  # warning doesn't fail


def test_cli_deprecated_error_promotion(capsys, tmp_path):
    src = """
@deprecated fn old() -> i32 { 0 }
fn main() -> i32 { old() }
"""
    p = str(tmp_path / "in.hx")
    with open(p, "w") as f:
        f.write(src)
    rc = main([p, "-Wdeprecated=error"])
    captured = capsys.readouterr()
    # When any -W*=error policy is active, helixc routes diagnostics
    # to stderr (so warning-policy callers can redirect cleanly).
    # Check both streams for robustness.
    combined = captured.out + captured.err
    assert "ERROR" in combined, \
        f"expected 'ERROR' in CLI output; got stdout={captured.out!r}, " \
        f"stderr={captured.err!r}"
    assert rc == 1


def test_cli_clean_no_deprecation(capsys, tmp_path):
    src = "fn main() -> i32 { 0 }\n"
    p = str(tmp_path / "in.hx")
    with open(p, "w") as f:
        f.write(src)
    rc = main([p])
    assert rc == 0
    out = capsys.readouterr().out
    assert "deprecated:" not in out


# ----------------------------------------------------------------------
# Audit-28.8 A5 regressions: the deprecated-call walker must recurse
# into `Index.indices`, `For.iter_expr`, `Range.start/end`, etc. Without
# these arms, deprecation warnings were silently lost when the offending
# call appeared in those positions.
# ----------------------------------------------------------------------
def test_find_call_sites_inside_index():
    """A5: `arr[deprecated_fn(2)]` — deprecated call inside Index.indices."""
    src = """
@deprecated("use new_fn") fn old_fn(x: i32) -> i32 { x }
fn main() -> i32 {
    let arr: [i32; 4] = [0, 0, 0, 0];
    arr[old_fn(2)]
}
"""
    prog = parse(src)
    sites = find_deprecation_call_sites(prog)
    names = {n for (n, _, _) in sites}
    assert "old_fn" in names, (
        f"old_fn(2) inside arr[..] should be detected; got names={names}"
    )


def test_find_call_sites_inside_range_end():
    """A5: `for i in 0..old_fn(10) { ... }` — deprecated call in Range.end."""
    src = """
@deprecated("use new_fn") fn old_fn(x: i32) -> i32 { x }
fn main() -> i32 {
    for i in 0..old_fn(10) {
        let unused: i32 = i;
    }
    0
}
"""
    prog = parse(src)
    sites = find_deprecation_call_sites(prog)
    names = {n for (n, _, _) in sites}
    assert "old_fn" in names, (
        f"old_fn(10) in Range.end should be detected; got names={names}"
    )


def test_c61_cn1_no_false_positive_mod_nested_collision():
    """Cycle 61 CN-1 (HIGH conf 88, both silent-failure + code-review):
    pre-cycle-62 the C59-3 fix recursed into ModBlock.items, collecting
    `@deprecated fn foo()` from inside `mod m { ... }` into a flat
    `dict[str, str]` keyed by short name. A top-level `fn foo()` call
    in `main` was then flagged as deprecated because `Name("foo")` hit
    the dict — even though it actually resolves to the top-level un-
    deprecated `foo`.

    Post-fix (cycle-62): find_deprecated_decls iterates only top-level
    items. mod-nested decls are surfaced via flatten_modules in the
    canonical pipeline (which mangles `m::foo` to `m__foo`).
    """
    src = """
mod legacy { @deprecated("internal v0") fn helper() -> i32 { 0 } }
fn helper() -> i32 { 42 }
fn main() -> i32 { helper() }
"""
    prog = parse(src)
    deps = find_deprecated_decls(prog)
    # Top-level `helper` is NOT @deprecated; the dict should not be
    # poisoned by the mod-nested one.
    assert "helper" not in deps, (
        f"top-level helper should not be marked deprecated; got deps={deps}"
    )
    sites = find_deprecation_call_sites(prog)
    names = {n for (n, _, _) in sites}
    assert "helper" not in names, (
        f"top-level helper() call should NOT produce a deprecation site; "
        f"got sites={names}"
    )


def test_c59_3_post_flatten_mod_nested_decls_detected():
    """Cycle 59 C59-3 (MED conf 78) — restored contract: after
    `flatten_modules` runs, mod-nested `@deprecated fn foo()` becomes
    top-level `m__foo`, so a call rewritten to `m__foo()` SHOULD be
    detected. This exercises the canonical pipeline path."""
    from helixc.frontend.flatten_modules import flatten_modules
    src = """
mod inner { @deprecated("module v0") fn helper() -> i32 { 0 } }
fn caller() -> i32 { inner::helper() }
"""
    prog = parse(src)
    flatten_modules(prog)
    deps = find_deprecated_decls(prog)
    # The lifted name is `inner__helper`.
    assert "inner__helper" in deps, (
        f"post-flatten mod-nested @deprecated should be visible; got deps={deps}"
    )
    sites = find_deprecation_call_sites(prog)
    names = {n for (n, _, _) in sites}
    assert "inner__helper" in names, (
        f"post-flatten call to inner::helper() should warn; got sites={names}"
    )


def test_c63_cn1_helixc_check_runs_flatten_modules(tmp_path, capsys):
    """Cycle 63 CN-1 (HIGH conf 92/95, silent-failure + code-review):
    pre-fix the `helixc check` surface tool did NOT run
    `flatten_modules`, which meant the cycle-62 `find_deprecated_decls`
    post-flatten contract was vacuous — mod-nested `@deprecated` decls
    were silently invisible. End-to-end repro: a `mod m { @deprecated
    fn old_api() ... } fn main() { m::old_api() }` source compiled
    cleanly with zero diagnostics.

    Cycle 64 fix: `helixc/check.py` now calls flatten_modules between
    flatten_impls and the analysis passes (matching the codegen
    driver's pass order). This test asserts the warning fires through
    the canonical CLI driver path, not just direct-API."""
    src_path = tmp_path / "mod_deprecated.hx"
    src_path.write_text(
        'mod legacy {\n'
        '    @deprecated("use new_helper") fn old_api() -> i32 { 0 }\n'
        '}\n'
        'fn main() -> i32 { legacy::old_api() }\n'
    )
    rc = main([str(src_path)])
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "deprecated" in output.lower(), (
        f"helixc check should report @deprecated call from mod-nested fn; "
        f"got rc={rc}, output={output!r}"
    )
    assert "old_api" in output or "use new_helper" in output, (
        f"deprecation warning should name the symbol or its message; "
        f"got output={output!r}"
    )


def test_c59_1_panic_in_mod_nested_fn_detected():
    """Cycle 59 C59-1 (HIGH conf 88): pre-fix the panic_pass walker
    iterated only top-level FnDecl. A `panic("...")` call inside
    `mod m { fn x() { panic("oops"); 0 } }` was invisible in the
    `helixc check` surface tool because flatten_modules doesn't run
    there. iter_fn_decls now recurses through ModBlock pre-flatten.
    """
    from helixc.frontend.panic_pass import collect_panics
    src = """
mod inner {
    fn x() -> i32 { panic("oops"); 0 }
}
"""
    prog = parse(src)
    panics = collect_panics(prog)
    fn_names = {n for (n, _, _) in panics}
    assert "x" in fn_names, (
        f"panic inside mod inner::x should be detected; got panics={panics}"
    )


def test_f59_1_bare_return_through_match_lower():
    """Cycle 59 F59-1 (HIGH conf 95): pre-fix match_lower._rewrite_expr
    Return arm was gated by `expr.value is not None`. A bare `return;`
    inside a fn body whose body contains a Match would walk into the
    Return-None case via _rewrite_expr and crash the cycle-58 loud
    NotImplementedError catchall. Cycle-60 fix: mirror Break's value-
    None-as-noop pattern.

    Crafted minimally: build the AST directly (the surface grammar
    requires `return X;` since match arms are expressions, not blocks
    of stmts that can host a value-less return)."""
    from helixc.frontend.match_lower import _rewrite_expr
    from helixc.frontend import ast_nodes as A
    span = A.Span(line=1, col=1)
    # A bare `Return(value=None)` — pre-fix this hit the loud catchall.
    bare_return = A.Return(span=span, value=None)
    out = _rewrite_expr(bare_return)
    assert out is bare_return, (
        f"bare return should pass through unchanged; got {out!r}"
    )


def test_c61_o60f_flatten_modules_preserves_extern():
    """Cycle 61 type-design O60-F (conf 85): pre-fix
    `flatten_modules._flatten_one` dropped `is_extern` and `extern_abi`
    when lifting FnDecls from a module to top-level. `mod m { extern
    "C" fn foo() -> i32; }` post-flatten became a regular FnDecl with
    an empty placeholder body — and a call to `m__foo` lowered to a
    zero-byte stub instead of a GOT/PLT relocation."""
    from helixc.frontend.flatten_modules import flatten_modules
    src = """
mod m {
    extern "C" fn foo() -> i32;
}
"""
    prog = parse(src)
    flatten_modules(prog)
    foo = next(it for it in prog.items
               if isinstance(it, A.FnDecl) and it.name == "m__foo")
    assert foo.is_extern, "lifted FnDecl should retain is_extern=True"
    assert foo.extern_abi == "C", (
        f"lifted FnDecl should retain extern_abi='C'; got {foo.extern_abi!r}"
    )


def test_c65_cn1_intra_mod_calls_rewritten():
    """Cycle 65 silent-failure CN-1 (HIGH conf 95): pre-fix
    flatten_modules renamed `mod m { fn foo }` to top-level
    `m__foo` but did NOT rewrite unqualified intra-mod calls. So
    `mod m { fn foo() { foo() } }` post-flatten produced top-level
    `m__foo` whose body still called `Name("foo")` — name-based
    downstream passes (totality self-call detection, deprecated
    call-site walker, etc.) failed to match.

    Cycle 66 fix: flatten_modules._flatten_one now builds an
    intra-mod-aliases dict and rewrites call sites in the items it
    lifted from each mod. Verify by parsing + flattening and
    inspecting the rewritten AST."""
    from helixc.frontend.flatten_modules import flatten_modules
    src = """
mod inner {
    fn helper(n: i32) -> i32 { helper(n + 1) }
    fn caller() -> i32 { helper(0) }
}
fn main() -> i32 { 0 }
"""
    prog = parse(src)
    flatten_modules(prog)
    # Find inner__helper and inner__caller; assert their bodies
    # reference the mangled names.
    fns = {it.name: it for it in prog.items if isinstance(it, A.FnDecl)}
    assert "inner__helper" in fns, f"inner::helper should be lifted; got {list(fns.keys())}"
    assert "inner__caller" in fns, f"inner::caller should be lifted; got {list(fns.keys())}"

    # inner__helper's body should call inner__helper (self-recursion mangled).
    helper_body = fns["inner__helper"].body
    helper_call = helper_body.final_expr
    assert isinstance(helper_call, A.Call), f"expected Call, got {type(helper_call).__name__}"
    assert isinstance(helper_call.callee, A.Name), f"expected Name callee, got {type(helper_call.callee).__name__}"
    assert helper_call.callee.name == "inner__helper", (
        f"self-call inside lifted mod fn should be mangled; got {helper_call.callee.name!r}"
    )

    # inner__caller's body should call inner__helper (sibling-call mangled).
    caller_body = fns["inner__caller"].body
    caller_call = caller_body.final_expr
    assert isinstance(caller_call, A.Call), f"expected Call, got {type(caller_call).__name__}"
    assert caller_call.callee.name == "inner__helper", (
        f"sibling-call inside lifted mod fn should be mangled; got {caller_call.callee.name!r}"
    )


def test_c67_cn1_silent_failure_no_cross_mod_name_capture():
    """Cycle 67 silent-failure CN-1 (HIGH conf 92): pre-fix the
    cycle-66 `direct_lifts_start` slice incorrectly included items
    appended by recursive nested-mod calls, so the OUTER mod's
    intra_mod_aliases were applied to INNER mod bodies.

    Pre-fix `mod outer { mod inner { fn caller() { sibling() } }
    fn sibling() }` would rewrite inner.caller's `sibling` call to
    `outer__sibling` even though `sibling` is undefined inside the
    inner scope.

    Cycle-68 fix: per-direct-lift index list (`local_lift_indices`)
    replaces the slice-range, scoping the rewrite to THIS call's
    direct lifts only.
    """
    from helixc.frontend.flatten_modules import flatten_modules
    src = """
mod outer {
    mod inner { fn caller() -> i32 { sibling() } }
    fn sibling() -> i32 { 42 }
}
"""
    prog = parse(src)
    flatten_modules(prog)
    fns = {it.name: it for it in prog.items if isinstance(it, A.FnDecl)}
    caller = fns.get("outer__inner__caller")
    assert caller is not None, f"outer__inner__caller should be lifted; got {list(fns.keys())}"
    callee = caller.body.final_expr.callee
    assert isinstance(callee, A.Name), f"expected Name callee, got {type(callee).__name__}"
    # CRITICAL: must NOT be rewritten to `outer__sibling` (cross-mod capture).
    assert callee.name == "sibling", (
        f"cross-mod name capture: inner.caller's `sibling` call must not "
        f"be rewritten by outer's intra_mod_aliases; got {callee.name!r}"
    )


def test_c67_cn1_nested_mod_name_not_aliased():
    """Cycle 67 code-review CN-1 (HIGH conf 92): nested-mod names
    must NOT enter intra_mod_aliases. Pre-fix `mod outer { mod inner
    { ... } fn foo() { inner() } }` rewrote `inner()` to
    `outer__inner` — a non-existent symbol.

    Cycle-68 fix: only FnDecl/StructDecl/EnumDecl/ConstDecl/TypeAlias
    enter intra_mod_aliases; ModBlock, ImplBlock, UseDecl, AgentDecl,
    catch-all-else items are skipped.

    Cycle 69 code-review CN-1 (LOW conf 70): test strengthened to
    actually exercise the rule — `caller`'s body references `inner`
    (as if it were a callable). Pre-fix the rewriter would turn
    `Name("inner")` into `Name("outer__inner")` because `inner`
    (the mod's name) was wrongly registered as an alias. Post-fix
    `inner` stays unrewritten (and downstream typecheck rejects it
    as an unknown symbol — which is the correct diagnostic).
    """
    from helixc.frontend.flatten_modules import flatten_modules
    # Use a callable that aliases inner's name to test the rewrite.
    # The body references `inner` as if it were a function. Pre-fix
    # this got silently rewritten to `outer__inner`; post-fix it
    # stays as `inner` for a downstream typecheck diagnostic.
    src = """
mod outer {
    mod inner { fn helper() -> i32 { 0 } }
    fn caller() -> i32 { inner() }
}
"""
    prog = parse(src)
    flatten_modules(prog)
    fn_names = {it.name for it in prog.items if isinstance(it, A.FnDecl)}
    assert "outer__inner__helper" in fn_names
    assert "outer__caller" in fn_names
    # CRITICAL: the body's `inner()` callee must NOT be rewritten to
    # `outer__inner` (a non-existent symbol). It should remain `inner`
    # so typecheck surfaces a clean "unknown function" diagnostic.
    caller = next(it for it in prog.items
                  if isinstance(it, A.FnDecl) and it.name == "outer__caller")
    callee = caller.body.final_expr.callee
    assert isinstance(callee, A.Name), f"expected Name callee, got {type(callee).__name__}"
    assert callee.name == "inner", (
        f"nested-mod name should NOT be rewritten by intra_mod_aliases; "
        f"got {callee.name!r} (pre-fix value: 'outer__inner')"
    )


def test_c67_cn2_impl_block_target_mangled():
    """Cycle 67 code-review CN-2 (HIGH conf 85): ImplBlock nested in
    a ModBlock must have its target mangled to match the sibling
    StructDecl's post-flatten name. Pre-fix it was lifted verbatim
    with target unchanged, causing flatten_impls to lift methods
    under a stale name (`Foo__get` while struct was `m__Foo`).
    """
    from helixc.frontend.flatten_modules import flatten_modules
    src = """
mod m {
    struct Foo { x: i32 }
    fn helper(x: i32) -> i32 { x + 1 }
    impl Foo { fn get(self: Foo) -> i32 { helper(self.x) } }
}
"""
    prog = parse(src)
    flatten_modules(prog)
    impls = [it for it in prog.items if isinstance(it, A.ImplBlock)]
    assert len(impls) == 1, f"expected 1 ImplBlock; got {len(impls)}"
    impl = impls[0]
    assert impl.target == "m__Foo", (
        f"ImplBlock.target should be mangled to match struct's name; "
        f"got {impl.target!r}"
    )
    # Method body should also have its intra-mod helper() call
    # rewritten to m__helper.
    get_method = impl.methods[0]
    body_call = get_method.body.final_expr
    assert isinstance(body_call, A.Call)
    assert isinstance(body_call.callee, A.Name)
    assert body_call.callee.name == "m__helper", (
        f"intra-mod call in impl method body should be rewritten; "
        f"got {body_call.callee.name!r}"
    )


def test_c65_cn1_totality_catches_mod_nested_recursion():
    """Cycle 65 silent-failure CN-1 follow-on: totality post-cycle-66
    catches non-terminating mod-nested recursion. Pre-fix the intra-
    mod self-call wasn't rewritten so totality's name-matching
    failed silently."""
    from helixc.frontend.flatten_modules import flatten_modules
    from helixc.frontend.totality import check_totality
    src = """
mod m {
    fn foo(n: i32) -> i32 { foo(n + 1) }
}
fn main() -> i32 { 0 }
"""
    prog = parse(src)
    flatten_modules(prog)
    fails = check_totality(prog)
    fail_names = {n for (n, _) in fails}
    assert "m__foo" in fail_names, (
        f"non-terminating mod-nested fn should be caught by totality; "
        f"got fails={fails}"
    )


def test_c71_struct_lit_name_mangled_in_mod():
    """Cycle 71 code-review CN-3 (HIGH conf 88): pre-fix
    `flatten_modules._rewrite_expr`'s StructLit arm did NOT remap
    `e.name` through the aliases dict. `mod m { struct Foo {x:i32};
    fn make() { Foo { x: 1 } } }` post-flatten left the
    `StructLit(name="Foo")` inside m__make's body even though the
    StructDecl became `m__Foo`. Backend (which runs flatten before
    typecheck) compiled a stale name; `helixc check` (which runs
    typecheck before flatten) was shielded by pipeline accident.

    Cycle-71 fix: `new_name = aliases.get(e.name, e.name)` in the
    StructLit arm — same alias mapping the call-site walker uses.
    """
    from helixc.frontend.flatten_modules import flatten_modules
    src = """
mod m {
    struct Foo { x: i32 }
    fn make() -> Foo { Foo { x: 42 } }
}
"""
    prog = parse(src)
    flatten_modules(prog)
    make_fn = next(it for it in prog.items
                   if isinstance(it, A.FnDecl) and it.name == "m__make")
    struct_lit = make_fn.body.final_expr
    assert isinstance(struct_lit, A.StructLit), (
        f"expected StructLit, got {type(struct_lit).__name__}"
    )
    assert struct_lit.name == "m__Foo", (
        f"mod-nested StructLit.name must be remapped to the lifted "
        f"struct's mangled name; got {struct_lit.name!r}"
    )


def test_c71_struct_lit_top_level_unchanged():
    """Cycle 71 code-review CN-3 follow-on: verify top-level StructLits
    (not inside a mod) are NOT incorrectly rewritten. The aliases dict
    only contains mod-nested decl mappings, so top-level struct names
    pass through unchanged via the `aliases.get(name, name)` fallback.
    """
    from helixc.frontend.flatten_modules import flatten_modules
    src = """
struct Bar { y: i32 }
fn build() -> Bar { Bar { y: 7 } }
"""
    prog = parse(src)
    flatten_modules(prog)
    build_fn = next(it for it in prog.items
                    if isinstance(it, A.FnDecl) and it.name == "build")
    struct_lit = build_fn.body.final_expr
    assert isinstance(struct_lit, A.StructLit)
    assert struct_lit.name == "Bar", (
        f"top-level StructLit.name must NOT be rewritten; got "
        f"{struct_lit.name!r}"
    )


def test_c73_cn1_totality_no_double_descent():
    """Cycle 73 type-design CN-1 (HIGH conf 90): pre-fix
    `_SelfCallCollector.visit_Call` explicitly called
    `self.generic_visit(node)` AND returned None, causing
    ASTVisitor.visit's base-class generic_visit to run a SECOND time.
    Nested self-calls inside args expressions got duplicated in
    `self.calls`, inflating the `len(recursive_calls)` count in the
    diagnostic.

    Cycle-74 fix: remove the explicit generic_visit; rely on
    ASTVisitor base-class auto-descent. Verify by counting calls
    discovered for a recursive fn whose body has a nested self-call:
    `fn rec(n) { rec(rec(n - 1)) }` — should find exactly 2 self-
    calls. Pre-fix (cycle 73) produced 3 for this source — the inner
    `rec(n-1)` got recorded twice (once from the override's explicit
    `generic_visit`, once from the base-class's post-override
    `generic_visit`). Cycle 75 code-review verified the exact 3-vs-2
    discriminator empirically.
    """
    from helixc.frontend.totality import _SelfCallCollector
    src = """
fn rec(n: i32) -> i32 { rec(rec(n - 1)) }
"""
    prog = parse(src)
    fn = next(it for it in prog.items
              if isinstance(it, A.FnDecl) and it.name == "rec")
    collector = _SelfCallCollector("rec")
    collector.visit(fn.body)
    # Body is Block(stmts=[], final_expr=Call(rec, [Call(rec, [n-1])])).
    # The outer Call AND the inner Call should each be counted once.
    assert len(collector.calls) == 2, (
        f"expected exactly 2 self-calls (outer + inner), got "
        f"{len(collector.calls)} (pre-fix produced exactly 3 — "
        f"one outer + two duplicate inner records from double-descent; "
        f"cycle-75 code-review empirically verified the 3-vs-2 "
        f"discriminator)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
