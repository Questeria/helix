"""
helixc/frontend/ast.py — Helix AST node definitions.

Plain Python dataclasses. Source positions on every node for diagnostics.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


# ============================================================================
# Source positions (every node carries this)
# ============================================================================
@dataclass(frozen=True)
class Span:
    line: int
    col: int


# ============================================================================
# Type AST
# ============================================================================
@dataclass
class TyNode:
    span: Span


@dataclass
class TyName(TyNode):
    """Reference to a named type or generic parameter (e.g., 'i32', 'N')."""
    name: str


@dataclass
class TyTuple(TyNode):
    elems: list["TyNode"]


@dataclass
class TyArray(TyNode):
    """[T; N]"""
    elem: "TyNode"
    size: "Expr"  # size expression (could be a literal int, a name, or arithmetic)


@dataclass
class TyRef(TyNode):
    """&T or &mut T"""
    inner: "TyNode"
    is_mut: bool


@dataclass
class TyPtr(TyNode):
    """*const T or *mut T — raw pointer (Stage 16.5 FFI).
    Lowered to u64 in IR for Phase-0; only used at FFI call sites for now."""
    inner: "TyNode"
    is_mut: bool


@dataclass
class TyFn(TyNode):
    """fn(T1, T2) -> R"""
    params: list["TyNode"]
    ret: "TyNode"


@dataclass
class TyTensor(TyNode):
    """tensor<dtype, [d1, ..., dN], device?, layout?>"""
    dtype: "TyNode"
    shape: list["Expr"]
    device: Optional["Expr"] = None
    layout: Optional["Expr"] = None


@dataclass
class TyTile(TyNode):
    """tile<dtype, [d1, ..., dN], memspace>"""
    dtype: "TyNode"
    shape: list["Expr"]
    memspace: "Expr"


@dataclass
class TyGeneric(TyNode):
    """Foo<A, B, C> — type with generic args"""
    base: str
    args: list["TyNode"]


# ============================================================================
# Expression AST
# ============================================================================
@dataclass
class Expr:
    span: Span


@dataclass
class IntLit(Expr):
    value: int
    type_suffix: Optional[str] = None


@dataclass
class FloatLit(Expr):
    value: float
    type_suffix: Optional[str] = None


@dataclass
class StrLit(Expr):
    value: str


@dataclass
class CharLit(Expr):
    value: str  # single char


@dataclass
class BoolLit(Expr):
    value: bool


@dataclass
class Name(Expr):
    """Identifier reference, possibly with generic args via :: <T>"""
    name: str
    generics: list["TyNode"] = field(default_factory=list)


@dataclass
class Path(Expr):
    """foo::bar::baz — dotted path"""
    segments: list[str]


@dataclass
class Unary(Expr):
    op: str        # "-", "!", "~", "&", "&mut", "*"
    operand: "Expr"


@dataclass
class Binary(Expr):
    op: str        # "+", "-", "*", "==", etc.
    left: "Expr"
    right: "Expr"


@dataclass
class Call(Expr):
    callee: "Expr"
    args: list["Expr"]


@dataclass
class Index(Expr):
    """a[i, j, k]"""
    callee: "Expr"
    indices: list["Expr"]


@dataclass
class Field(Expr):
    """obj.field"""
    obj: "Expr"
    name: str


@dataclass
class TupleLit(Expr):
    elems: list["Expr"]


@dataclass
class ArrayLit(Expr):
    """[1, 2, 3]"""
    elems: list["Expr"]


@dataclass
class StructLit(Expr):
    """`Point { x: 1, y: 2 }` — struct construction by name + named fields.
    Field order in `fields` matches source order; typecheck reorders to
    the declaration's order and verifies all required fields are present."""
    name: str
    fields: list[tuple[str, "Expr"]]


@dataclass
class Block(Expr):
    """{ stmt; stmt; expr? }"""
    stmts: list["Stmt"]
    final_expr: Optional["Expr"] = None


@dataclass
class UnsafeBlock(Expr):
    """unsafe { ... } (Stage 28.6).

    The inner Block is parsed normally; codegen / effect-check treats
    expressions inside the block as having the 'unsafe' capability.
    Raw-pointer deref / arithmetic outside any UnsafeBlock context
    traps 28601 (no surface alternative).
    """
    body: "Block"


@dataclass
class If(Expr):
    cond: "Expr"
    then: "Block"
    else_: Optional[Union["Block", "If"]] = None


@dataclass
class Match(Expr):
    scrutinee: "Expr"
    arms: list["MatchArm"]


@dataclass
class MatchArm:
    span: Span
    pattern: "Pattern"
    guard: Optional["Expr"]
    body: "Expr"


@dataclass
class Pattern:
    span: Span


@dataclass
class PatLit(Pattern):
    value: "Expr"


@dataclass
class PatBind(Pattern):
    name: str
    is_mut: bool


@dataclass
class PatWildcard(Pattern):
    pass


@dataclass
class PatTuple(Pattern):
    elems: list["Pattern"]


@dataclass
class PatOr(Pattern):
    """`a | b | c` — match if any alternative matches."""
    alts: list["Pattern"]


@dataclass
class PatRange(Pattern):
    """`lo..hi` (exclusive) or `lo..=hi` (inclusive)."""
    lo: "Expr"
    hi: "Expr"
    inclusive: bool


@dataclass
class PatVariant(Pattern):
    """`EnumName::Variant(p1, p2, ...)` — match the variant tag AND
    recursively match each payload sub-pattern. Empty sub_patterns
    means tag-only (equivalent to PatLit-of-Path in legacy form, but
    explicitly typed as a variant pattern)."""
    path: "Path"
    sub_patterns: list["Pattern"]


@dataclass
class PatStruct(Pattern):
    """Stage 59 / Tier 4 #15 — struct destructuring pattern.

    `StructName { field1: pat1, field2: pat2, ... }` matches a value
    of struct type `StructName` by recursively matching each field's
    pattern. Short-form `field` (no `: pat`) is sugar for
    `field: <bind to same name>`.

    Examples:
        Point { x, y }                       // bind x and y by name
        Point { x: 0, y }                    // x must equal 0; bind y
        Layer { weight: Tensor { data }, .. }  // nested (.. = ignore rest)

    The `ignore_rest` flag is True when the pattern ends with `..`
    (Rust syntax) — fields not listed are silently allowed.
    """
    name: str
    fields: list[tuple[str, "Pattern"]]
    ignore_rest: bool = False


@dataclass
class For(Expr):
    """for x in iter { body }"""
    var_name: str
    iter_expr: "Expr"
    body: "Block"


@dataclass
class While(Expr):
    cond: "Expr"
    body: "Block"


@dataclass
class Loop(Expr):
    body: "Block"


@dataclass
class Break(Expr):
    value: Optional["Expr"] = None


@dataclass
class Continue(Expr):
    pass


@dataclass
class Return(Expr):
    value: Optional["Expr"] = None


@dataclass
class Range(Expr):
    """a .. b"""
    start: Optional["Expr"]
    end: Optional["Expr"]


@dataclass
class Assign(Expr):
    """x = expr or x += expr (target; op; value)"""
    target: "Expr"
    op: str        # "=", "+=", "-=", "*=", "/=", "%="
    value: "Expr"


@dataclass
class Cast(Expr):
    """expr as TargetType — type conversion (e.g., 3.14 as i32)"""
    value: "Expr"
    target_ty: "TyNode"


@dataclass
class TileLit(Expr):
    """tile<dtype, [N, M], memspace>::zeros() / ::ones() — Stage 15.

    A tile literal is a compile-time-shaped allocation with all elements
    initialized to the same value (0.0 for zeros, 1.0 for ones). Phase-0
    only supports REG memspace and f32 dtype, capping shape at 8x8.

    Lowered to ALLOC_ARRAY of (N*M) elements in the backend.
    """
    dtype: "TyNode"          # the tile element type (e.g. TyName("f32"))
    shape: list["Expr"]      # the shape dims (e.g. [IntLit(4), IntLit(4)])
    memspace: "Expr"         # memspace marker (e.g. Name("REG"))
    init: str                # "zeros" or "ones"


# ============================================================================
# AGI-specific expression nodes
# ============================================================================
@dataclass
class Quote(Expr):
    """quote { ... } — captures the contained expression as an AST value of
    type AstNode. Unique to Helix: programs can read their own source as data."""
    inner: "Expr"


@dataclass
class Splice(Expr):
    """splice(ast_value) — re-injects an AstNode value at the source position
    where it appears. The inverse of `quote`."""
    inner: "Expr"


@dataclass
class Modify(Expr):
    """modify(target, transformation, verifier) — verifier-gated self-modification.
    The AGI proposes a transformation; the verifier must accept it before commit."""
    target: "Expr"
    transformation: "Expr"
    verifier: "Expr"


# ============================================================================
# Statement AST
# ============================================================================
@dataclass
class Stmt:
    span: Span


@dataclass
class Let(Stmt):
    name: str
    is_mut: bool
    ty: Optional["TyNode"]
    value: Optional["Expr"]


@dataclass
class ExprStmt(Stmt):
    expr: "Expr"


@dataclass
class ConstStmt(Stmt):
    name: str
    ty: "TyNode"
    value: "Expr"


# ============================================================================
# Top-level items
# ============================================================================
@dataclass
class Item:
    span: Span


@dataclass
class GenericParam:
    """N: size, T: type, etc."""
    span: Span
    name: str
    kind: str        # "type", "size", "device", or other


@dataclass
class FnParam:
    span: Span
    name: str
    ty: "TyNode"
    is_mut: bool = False


@dataclass
class WhereClause:
    """size constraints like `N % 16 == 0` or `M >= K`"""
    span: Span
    constraint: "Expr"


@dataclass
class FnDecl(Item):
    name: str
    generics: list[GenericParam]
    params: list[FnParam]
    return_ty: Optional["TyNode"]
    where_clauses: list[WhereClause]
    body: "Block"
    attrs: list[str]            # @kernel, @pure, @inline, etc.
    is_pub: bool = False
    # Stage 16.5: True for `extern "C" fn name(...) -> ret;` declarations.
    # The `body` is an empty placeholder Block in this case (set by parser).
    # Calls to extern fns are resolved by the dynamic linker at runtime via
    # GOT/PLT relocations rather than emitted as user-fn calls.
    is_extern: bool = False
    extern_abi: Optional[str] = None  # currently always "C" when is_extern


@dataclass
class StructDecl(Item):
    name: str
    generics: list[GenericParam]
    fields: list[FnParam]       # reuse FnParam shape: name+ty
    is_pub: bool = False
    # Stage 66 Inc 4: struct-level attributes. `@copy` opts a struct
    # into Copy semantics (assignments duplicate instead of moving), so
    # the borrow checker doesn't flag re-use after assignment / pass-by-value.
    # Default empty list keeps every existing call site (and every
    # existing test that constructs StructDecl positionally) source-compatible.
    attrs: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.attrs is None:
            self.attrs = []


@dataclass
class EnumVariant:
    span: Span
    name: str
    payload_tys: list["TyNode"]


@dataclass
class EnumDecl(Item):
    name: str
    generics: list[GenericParam]
    variants: list[EnumVariant]
    is_pub: bool = False


@dataclass
class TypeAlias(Item):
    name: str
    generics: list[GenericParam]
    target: "TyNode"
    is_pub: bool = False
    where_clauses: list[WhereClause] = field(default_factory=list)


@dataclass
class UseDecl(Item):
    """use foo::bar::baz; or use foo::{a, b};"""
    path: list[str]


@dataclass
class ConstDecl(Item):
    name: str
    ty: "TyNode"
    value: "Expr"
    is_pub: bool = False


@dataclass
class AgentMethod:
    """Method signature inside an agent declaration."""
    span: Span
    name: str
    params: list[FnParam] = field(default_factory=list)
    return_ty: Optional["TyNode"] = None


@dataclass
class AgentDecl(Item):
    """agent Planner { fn propose(state: i32) -> i32; ... }
    A cognitive-architecture primitive: a typed bundle of methods
    that participate in society::dispatch."""
    name: str
    methods: list[AgentMethod] = field(default_factory=list)
    is_pub: bool = False


@dataclass
class ModuleDecl(Item):
    """module path::to::module"""
    path: list[str]


@dataclass
class ModBlock(Item):
    """`mod foo { items... }` — a block module that namespaces a group
    of items. Resolved by the flatten_modules pass: every nested item is
    lifted out and renamed to `foo__name`, while every `foo::name(...)`
    call is rewritten to `foo__name(...)`.
    """
    name: str
    items: list["Item"]
    is_pub: bool = False


@dataclass
class ImplBlock(Item):
    """`impl TypeName { fn method(self, ...) ... }` — inherent impl block.
    Resolved by the flatten_impls pass: every method is lifted to a top-
    level fn named `TypeName__method_name`, and every `x.method(args)`
    call where `method` matches an impl-block method is rewritten to
    `TypeName__method(x, args)`.
    Trait impls (`impl Trait for Type`) are accepted as the same shape;
    Phase 1.8 only does inherent dispatch (trait_name field is metadata).
    """
    target: str           # type name being impl'd, e.g. "Point"
    methods: list["FnDecl"]
    trait_name: Optional[str] = None  # for `impl Trait for Type`
    is_pub: bool = False


# ============================================================================
# Program
# ============================================================================
@dataclass
class Program:
    """A whole .hx file."""
    module: Optional[ModuleDecl]
    items: list[Item]
