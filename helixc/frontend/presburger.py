"""
helixc/frontend/presburger.py — Linear arithmetic constraint solver for size types.

Helix's killer feature: catch tensor-shape bugs at compile time.

When you write:
    fn matmul[N: size, M: size, P: size](
        a: tensor<f32, [N, M]>,
        b: tensor<f32, [M, P]>,
    ) -> tensor<f32, [N, P]>

The compiler must prove that the inner dimension of `a` (M) equals the outer
dimension of `b` (M) at every call site. Most languages defer this check to
runtime. Helix does it at compile time using Presburger arithmetic — the
fragment of integer linear arithmetic with variables, addition, multiplication
by constants, comparisons, and divisibility.

This module implements a small, decidable solver:
- Linear expressions over integer-valued symbolic variables
- Equality and inequality constraints
- A simple Fourier-Motzkin / Omega-test style solver
- Sufficient for matmul / conv / broadcast / concat shape arithmetic

For v0.1 we implement:
- Linear-equality reasoning (Gaussian elimination over integer affine forms)
- Linear-inequality satisfiability (basic case detection)
- Modular constraints (`N % 16 == 0`) handled via divisibility predicates

What's NOT yet implemented:
- Full Omega-decision-procedure (handles any Presburger formula)
- Quantifier elimination
- Disjunctive normal-form solver

For an MVP that catches "matmul inner dim mismatch", linear-equality reasoning
is sufficient. The harder cases ship with a "could not prove" diagnostic.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


# ============================================================================
# Linear expressions over symbolic size variables
# ============================================================================
@dataclass(frozen=True)
class LinExpr:
    """Linear expression: sum of (coef * var) plus a constant.
    Represented as a frozen dict of variable name -> integer coefficient,
    plus an integer constant.

    Example: 2*N + 3*M + 5  -> coefs={"N": 2, "M": 3}, const=5
    """
    coefs: tuple[tuple[str, int], ...]   # sorted by var name; ints
    const: int

    def __add__(self, other: "LinExpr") -> "LinExpr":
        return _normalize(_merge(self.coefs, other.coefs, lambda a, b: a + b),
                          self.const + other.const)

    def __sub__(self, other: "LinExpr") -> "LinExpr":
        return _normalize(_merge(self.coefs, other.coefs, lambda a, b: a - b),
                          self.const - other.const)

    def __mul__(self, k: int) -> "LinExpr":
        return _normalize(tuple((v, c * k) for v, c in self.coefs), self.const * k)

    def __neg__(self) -> "LinExpr":
        return _normalize(tuple((v, -c) for v, c in self.coefs), -self.const)

    def is_zero(self) -> bool:
        return not self.coefs and self.const == 0

    def is_const(self) -> bool:
        return not self.coefs

    def vars(self) -> set[str]:
        return {v for v, _ in self.coefs}

    def coef_of(self, var: str) -> int:
        for v, c in self.coefs:
            if v == var:
                return c
        return 0

    def pretty(self) -> str:
        terms = []
        for v, c in self.coefs:
            if c == 1:
                terms.append(f"{v}")
            elif c == -1:
                terms.append(f"-{v}")
            else:
                terms.append(f"{c}*{v}")
        if self.const != 0 or not terms:
            terms.append(str(self.const))
        return " + ".join(terms).replace("+ -", "- ")


def _merge(a: tuple, b: tuple, op) -> tuple:
    out: dict[str, int] = {}
    for v, c in a:
        out[v] = c
    for v, c in b:
        out[v] = op(out.get(v, 0), c)
    return tuple(sorted(out.items()))


def _normalize(coefs: tuple, const: int) -> LinExpr:
    return LinExpr(tuple((v, c) for v, c in coefs if c != 0), const)


def lit(n: int) -> LinExpr:
    return LinExpr((), n)


def var(name: str) -> LinExpr:
    return LinExpr(((name, 1),), 0)


# ============================================================================
# Constraint kinds
# ============================================================================
@dataclass(frozen=True)
class Eq:
    """expr == 0"""
    expr: LinExpr

    def pretty(self) -> str:
        return f"{self.expr.pretty()} == 0"


@dataclass(frozen=True)
class Le:
    """expr <= 0"""
    expr: LinExpr

    def pretty(self) -> str:
        return f"{self.expr.pretty()} <= 0"


@dataclass(frozen=True)
class Divides:
    """expr divisible by k (i.e., expr mod k == 0)"""
    expr: LinExpr
    k: int

    def pretty(self) -> str:
        return f"{self.expr.pretty()} % {self.k} == 0"


Constraint = Union[Eq, Le, Divides]


# ============================================================================
# Solver
# ============================================================================
@dataclass
class Solver:
    """Maintains a set of constraints. Can answer:
    - Is the system satisfiable? (returns Sat / Unsat / Unknown)
    - Does it imply a target constraint? (returns Yes / No / Unknown)
    """

    constraints: list[Constraint] = field(default_factory=list)

    def add_eq(self, e: LinExpr) -> None:
        if not e.is_zero():
            self.constraints.append(Eq(e))

    def add_le(self, e: LinExpr) -> None:
        # e <= 0
        self.constraints.append(Le(e))

    def add_lt(self, e: LinExpr) -> None:
        # e < 0  =>  e + 1 <= 0
        self.add_le(e + lit(1))

    def add_eq_pair(self, a: LinExpr, b: LinExpr) -> None:
        self.add_eq(a - b)

    def add_divides(self, e: LinExpr, k: int) -> None:
        if k > 1:
            self.constraints.append(Divides(e, k))

    # ---- Decision: is this constraint already implied? ----
    def implies(self, c: Constraint) -> Optional[bool]:
        """Returns True if the constraint set provably implies `c`,
        False if provably contradicts, None if unknown."""
        if isinstance(c, Eq):
            return self._implies_eq(c.expr)
        if isinstance(c, Le):
            return self._implies_le(c.expr)
        if isinstance(c, Divides):
            return self._implies_divides(c.expr, c.k)
        return None

    def _implies_eq(self, e: LinExpr) -> Optional[bool]:
        """Does the system imply e == 0?

        Strategy: substitute all known equalities, then check if e reduces to 0.
        """
        reduced = self._reduce_via_eqs(e)
        if reduced.is_zero():
            return True
        if reduced.is_const() and reduced.const != 0:
            # Reduces to a non-zero constant — provably false
            return False
        return None

    def _implies_le(self, e: LinExpr) -> Optional[bool]:
        """Does the system imply e <= 0?

        Strategy: reduce via equalities, then if the result is a constant,
        check sign. Else unknown (we'd need full Fourier-Motzkin).
        """
        reduced = self._reduce_via_eqs(e)
        if reduced.is_const():
            return reduced.const <= 0
        # Try a few inequalities — combine with existing Les via simple bounds
        for c in self.constraints:
            if isinstance(c, Le):
                # If c.expr - e <= 0 known, and that means e >= c.expr,
                # which doesn't immediately give us e <= 0.
                pass
        return None

    def _implies_divides(self, e: LinExpr, k: int) -> Optional[bool]:
        """Does the system imply e % k == 0?

        For v0.1: only handle the simple cases.
        - If e reduces to a constant: check `const % k == 0`
        - If e is k * (anything): yes
        - Else: check if every term's coefficient is divisible by k
          (sufficient condition, but not necessary)
        """
        reduced = self._reduce_via_eqs(e)
        if reduced.is_const():
            return (reduced.const % k) == 0
        # If every coefficient and the constant are divisible by k, yes.
        if reduced.const % k == 0 and all(c % k == 0 for _, c in reduced.coefs):
            return True
        # Check existing Divides constraints
        for cn in self.constraints:
            if isinstance(cn, Divides) and cn.k == k:
                # If e == cn.expr (provably), yes
                diff = self._reduce_via_eqs(reduced - cn.expr)
                if diff.is_zero():
                    return True
                # If e == cn.expr + (k-divisible), yes
                if diff.is_const() and diff.const % k == 0:
                    return True
                if diff.const % k == 0 and all(c % k == 0 for _, c in diff.coefs):
                    return True
        return None

    # ---- Internal: reduce an expression via equalities ----
    def _reduce_via_eqs(self, e: LinExpr) -> LinExpr:
        """Substitute eq-defined variables. Iterates to fixpoint."""
        eqs = [c.expr for c in self.constraints if isinstance(c, Eq)]
        cur = e
        changed = True
        while changed:
            changed = False
            for eq_expr in eqs:
                # If eq_expr is `var - <stuff> == 0`, we can substitute.
                # Pick the variable with smallest |coefficient| (= 1 ideally)
                solvable_for = None
                for v, c in eq_expr.coefs:
                    if c == 1 or c == -1:
                        solvable_for = (v, c); break
                if solvable_for is None:
                    continue
                v, c = solvable_for
                # eq_expr = c*v + rest = 0  =>  v = -rest/c
                rest_coefs = tuple((vv, cc) for vv, cc in eq_expr.coefs if vv != v)
                rest = LinExpr(rest_coefs, eq_expr.const)
                # Restart 50 B3: simplified — was previously
                # `(-rest * (1 // c if c == 1 else -1)) if False else
                # (rest * -1 if c == 1 else rest)` which had a dead
                # `if False else` outer ternary (only the right branch
                # ever ran). Eliminating it makes the actual computed
                # value obvious: for c == 1 negate rest; for c == -1
                # keep rest (since v = -rest / c = -rest / -1 = rest).
                substitution = rest * -1 if c == 1 else rest
                # If cur has v with coefficient k, replace v with substitution
                k = cur.coef_of(v)
                if k != 0:
                    new_coefs = tuple((vv, cc) for vv, cc in cur.coefs if vv != v)
                    cur = LinExpr(new_coefs, cur.const)
                    cur = cur + (substitution * k)
                    changed = True
        return cur

    def is_sat(self) -> Optional[bool]:
        """Is the constraint system satisfiable?"""
        # Trivially: if any Eq reduces to non-zero const => unsat
        for c in self.constraints:
            if isinstance(c, Eq):
                r = self._reduce_via_eqs(c.expr)
                if r.is_const() and r.const != 0:
                    return False
        return None  # otherwise unknown for v0.1


# ============================================================================
# Helpers for building constraints from AST/IR shape exprs
# ============================================================================
def lin_from_int(n: int) -> LinExpr:
    return lit(n)


def lin_from_var(name: str) -> LinExpr:
    return var(name)


def lin_add(a: LinExpr, b: LinExpr) -> LinExpr:
    return a + b


def lin_sub(a: LinExpr, b: LinExpr) -> LinExpr:
    return a - b


def lin_mul_const(a: LinExpr, k: int) -> LinExpr:
    return a * k


# ============================================================================
# Quick self-test
# ============================================================================
if __name__ == "__main__":
    s = Solver()
    s.add_eq_pair(var("M"), var("K"))   # M = K
    s.add_divides(var("N"), 16)          # N % 16 == 0

    # Should imply M == K
    assert s.implies(Eq(var("M") - var("K"))) == True

    # Should imply N % 16 == 0 (trivially)
    assert s.implies(Divides(var("N"), 16)) == True

    # Should NOT imply N % 32 == 0 (we only know N % 16 == 0)
    assert s.implies(Divides(var("N"), 32)) is None

    # Should imply 0 == 0 (vacuous)
    assert s.implies(Eq(lit(0))) == True

    # Should refute 1 == 0
    assert s.implies(Eq(lit(1))) == False

    # Linear: M + K = M + K
    assert s.implies(Eq(var("M") + var("K") - (var("M") + var("K")))) == True

    print("Presburger MVP self-test PASS")
