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
class Block(Expr):
    """{ stmt; stmt; expr? }"""
    stmts: list["Stmt"]
    final_expr: Optional["Expr"] = None


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


@dataclass
class StructDecl(Item):
    name: str
    generics: list[GenericParam]
    fields: list[FnParam]       # reuse FnParam shape: name+ty
    is_pub: bool = False


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


# ============================================================================
# Program
# ============================================================================
@dataclass
class Program:
    """A whole .hx file."""
    module: Optional[ModuleDecl]
    items: list[Item]
