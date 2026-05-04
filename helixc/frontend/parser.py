"""
helixc/frontend/parser.py — Helix recursive-descent parser.

Consumes tokens from helixc.frontend.lexer, produces AST in helixc.frontend.ast.

Operator precedence (lowest -> highest):
    1.  =, +=, -=, *=, /=, %=         (right-assoc, statement-level)
    2.  ||                            (left-assoc)
    3.  &&
    4.  ==, !=
    5.  <, <=, >, >=
    6.  |
    7.  ^
    8.  &
    9.  <<, >>
   10.  +, -
   11.  *, /, %
   12.  unary -, !, ~, &, &mut, *
   13.  postfix . [ (
   14.  primary

License: Apache 2.0
"""

from __future__ import annotations

from .lexer import Token, T, lex
from . import ast_nodes as ast


class ParseError(Exception):
    def __init__(self, msg: str, tok: Token):
        super().__init__(f"{tok.line}:{tok.col}: parse error: {msg} (got {tok.kind.name} {tok.value!r})")
        self.token = tok


class Parser:
    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.i = 0
        # When True, the relational layer does NOT treat < and > as operators
        # (they belong to surrounding generic-args like tensor<...>).
        self._no_cmp_lt_gt = 0

    # ---- token helpers ----
    def _peek(self, k: int = 0) -> Token:
        idx = self.i + k
        if idx >= len(self.toks):
            return self.toks[-1]
        return self.toks[idx]

    def _at(self, kind: T) -> bool:
        return self._peek().kind == kind

    def _at_any(self, *kinds: T) -> bool:
        return self._peek().kind in kinds

    def _eat(self, kind: T) -> Token:
        t = self._peek()
        # Special case: when expecting GT, accept SHR ('>>') as two GTs.
        # This handles nested generics like D<tensor<f32, [N, N]>>
        # where the lexer emits SHR for the closing '>>'.
        if kind == T.GT and t.kind == T.SHR:
            # Replace the SHR token with a single GT in place; consume one half.
            self.toks[self.i] = Token(
                T.GT, ">", t.line, t.col + 1,
            )
            return Token(T.GT, ">", t.line, t.col)
        if t.kind != kind:
            raise ParseError(f"expected {kind.name}", t)
        self.i += 1
        return t

    def _match(self, kind: T) -> Token | None:
        if self._at(kind):
            return self._eat(kind)
        return None

    def _span_of(self, tok: Token) -> ast.Span:
        return ast.Span(tok.line, tok.col)

    # ========================================================================
    # Top-level
    # ========================================================================
    def parse_program(self) -> ast.Program:
        module = None
        # Optional module declaration first
        if self._at(T.KW_MODULE):
            module = self._parse_module_decl()
        items: list[ast.Item] = []
        while not self._at(T.EOF):
            items.append(self._parse_item())
        return ast.Program(module=module, items=items)

    def _parse_module_decl(self) -> ast.ModuleDecl:
        tok = self._eat(T.KW_MODULE)
        path = self._parse_path_segments()
        # optional ;
        self._match(T.SEMI)
        return ast.ModuleDecl(span=self._span_of(tok), path=path)

    def _eat_name_token(self) -> Token:
        """Eat an IDENT or any keyword that's name-like (used in paths, use decls,
        module decls). Most keywords are reserved at expression-position but
        valid as path segments."""
        t = self._peek()
        if t.kind == T.IDENT:
            self.i += 1
            return t
        # Allow any keyword token whose lexeme is alphanumeric — i.e., it could
        # have been an identifier if not for being reserved.
        if t.value and t.value[0].isalpha() and all(c.isalnum() or c == "_" for c in t.value):
            self.i += 1
            return t
        raise ParseError("expected name", t)

    def _parse_path_segments(self) -> list[str]:
        first = self._eat_name_token()
        segs = [first.value]
        while self._at(T.COLONCOLON):
            self._eat(T.COLONCOLON)
            nxt = self._eat_name_token()
            segs.append(nxt.value)
        return segs

    # ---- items ----
    def _parse_item(self) -> ast.Item:
        attrs = self._parse_attributes()
        is_pub = bool(self._match(T.KW_PUB))

        t = self._peek()
        if t.kind == T.KW_FN:
            return self._parse_fn_decl(is_pub, attrs)
        if t.kind == T.KW_STRUCT:
            return self._parse_struct_decl(is_pub)
        if t.kind == T.KW_ENUM:
            return self._parse_enum_decl(is_pub)
        if t.kind == T.KW_TYPE:
            return self._parse_type_alias(is_pub)
        if t.kind == T.KW_USE:
            return self._parse_use_decl()
        if t.kind == T.KW_CONST:
            return self._parse_const_decl(is_pub)
        if t.kind == T.KW_AGENT:
            return self._parse_agent_decl(is_pub)
        raise ParseError(f"expected item (fn/struct/enum/type/use/const/agent)", t)

    def _parse_attributes(self) -> list[str]:
        attrs: list[str] = []
        while self._at(T.AT):
            self._eat(T.AT)
            # Either an ident or a keyword (kernel, pure, etc.)
            t = self._peek()
            if t.kind == T.IDENT:
                attrs.append(t.value)
                self.i += 1
            elif t.kind == T.KW_KERNEL:
                attrs.append("kernel"); self.i += 1
            elif t.kind == T.KW_GRAD:
                attrs.append("grad"); self.i += 1
            else:
                # accept any identifier-shaped keyword
                if t.value.replace("_", "").isalnum():
                    attrs.append(t.value); self.i += 1
                else:
                    raise ParseError("expected attribute name after @", t)
            # optional (args) — not used in v0.1
            if self._at(T.LPAREN):
                # skip balanced parens
                depth = 0
                while True:
                    t = self._peek()
                    if t.kind == T.EOF:
                        raise ParseError("unclosed attribute args", t)
                    if t.kind == T.LPAREN: depth += 1
                    elif t.kind == T.RPAREN:
                        depth -= 1
                        if depth == 0:
                            self.i += 1
                            break
                    self.i += 1
        return attrs

    # ---- fn ----
    def _parse_fn_decl(self, is_pub: bool, attrs: list[str]) -> ast.FnDecl:
        kw = self._eat(T.KW_FN)
        name_tok = self._eat(T.IDENT)
        generics = self._parse_generic_params()
        self._eat(T.LPAREN)
        params: list[ast.FnParam] = []
        if not self._at(T.RPAREN):
            params.append(self._parse_fn_param())
            while self._match(T.COMMA):
                if self._at(T.RPAREN):  # trailing comma
                    break
                params.append(self._parse_fn_param())
        self._eat(T.RPAREN)
        ret_ty: ast.TyNode | None = None
        if self._match(T.ARROW):
            ret_ty = self._parse_type()
        where_clauses: list[ast.WhereClause] = []
        if self._match(T.KW_WHERE):
            where_clauses = self._parse_where_clauses()
        body = self._parse_block()
        return ast.FnDecl(
            span=self._span_of(kw),
            name=name_tok.value,
            generics=generics,
            params=params,
            return_ty=ret_ty,
            where_clauses=where_clauses,
            body=body,
            attrs=attrs,
            is_pub=is_pub,
        )

    def _parse_generic_params(self) -> list[ast.GenericParam]:
        params: list[ast.GenericParam] = []
        if not self._match(T.LBRACK):
            return params
        while not self._at(T.RBRACK):
            t = self._eat(T.IDENT)
            kind = "type"  # default
            if self._match(T.COLON):
                kt = self._peek()
                if kt.kind == T.KW_SIZE:
                    kind = "size"; self.i += 1
                elif kt.kind == T.IDENT:
                    kind = kt.value; self.i += 1
                elif kt.kind == T.KW_DEVICE:
                    kind = "device"; self.i += 1
                else:
                    raise ParseError("expected kind after ':'", kt)
            params.append(ast.GenericParam(span=self._span_of(t), name=t.value, kind=kind))
            if not self._match(T.COMMA):
                break
        self._eat(T.RBRACK)
        return params

    def _parse_fn_param(self) -> ast.FnParam:
        t_start = self._peek()
        is_mut = bool(self._match(T.KW_MUT))
        name_tok = self._eat(T.IDENT)
        self._eat(T.COLON)
        ty = self._parse_type()
        return ast.FnParam(
            span=self._span_of(t_start),
            name=name_tok.value, ty=ty, is_mut=is_mut,
        )

    def _parse_where_clauses(self) -> list[ast.WhereClause]:
        out: list[ast.WhereClause] = []
        while True:
            t_start = self._peek()
            constraint = self._parse_expr()
            out.append(ast.WhereClause(span=self._span_of(t_start), constraint=constraint))
            if not self._match(T.COMMA):
                break
            # If next is `{` (start of body), stop
            if self._at(T.LBRACE):
                break
        return out

    # ---- struct / enum / type / use / const ----
    def _parse_struct_decl(self, is_pub: bool) -> ast.StructDecl:
        kw = self._eat(T.KW_STRUCT)
        name = self._eat(T.IDENT).value
        generics = self._parse_generic_params()
        self._eat(T.LBRACE)
        fields: list[ast.FnParam] = []
        while not self._at(T.RBRACE):
            t_start = self._peek()
            n = self._eat(T.IDENT).value
            self._eat(T.COLON)
            ty = self._parse_type()
            fields.append(ast.FnParam(span=self._span_of(t_start), name=n, ty=ty))
            if not self._match(T.COMMA):
                break
        self._eat(T.RBRACE)
        return ast.StructDecl(span=self._span_of(kw), name=name,
                              generics=generics, fields=fields, is_pub=is_pub)

    def _parse_enum_decl(self, is_pub: bool) -> ast.EnumDecl:
        kw = self._eat(T.KW_ENUM)
        name = self._eat(T.IDENT).value
        generics = self._parse_generic_params()
        self._eat(T.LBRACE)
        variants: list[ast.EnumVariant] = []
        while not self._at(T.RBRACE):
            t_start = self._peek()
            n = self._eat(T.IDENT).value
            payload: list[ast.TyNode] = []
            if self._match(T.LPAREN):
                if not self._at(T.RPAREN):
                    payload.append(self._parse_type())
                    while self._match(T.COMMA):
                        if self._at(T.RPAREN): break
                        payload.append(self._parse_type())
                self._eat(T.RPAREN)
            variants.append(ast.EnumVariant(span=self._span_of(t_start),
                                            name=n, payload_tys=payload))
            if not self._match(T.COMMA):
                break
        self._eat(T.RBRACE)
        return ast.EnumDecl(span=self._span_of(kw), name=name,
                            generics=generics, variants=variants, is_pub=is_pub)

    def _parse_type_alias(self, is_pub: bool) -> ast.TypeAlias:
        kw = self._eat(T.KW_TYPE)
        name = self._eat(T.IDENT).value
        generics = self._parse_generic_params()
        self._eat(T.EQ)
        target = self._parse_type()
        self._eat(T.SEMI)
        return ast.TypeAlias(span=self._span_of(kw), name=name,
                             generics=generics, target=target, is_pub=is_pub)

    def _parse_use_decl(self) -> ast.UseDecl:
        kw = self._eat(T.KW_USE)
        path = self._parse_path_segments()
        self._eat(T.SEMI)
        return ast.UseDecl(span=self._span_of(kw), path=path)

    def _parse_const_decl(self, is_pub: bool) -> ast.ConstDecl:
        kw = self._eat(T.KW_CONST)
        name = self._eat(T.IDENT).value
        self._eat(T.COLON)
        ty = self._parse_type()
        self._eat(T.EQ)
        value = self._parse_expr()
        self._eat(T.SEMI)
        return ast.ConstDecl(span=self._span_of(kw), name=name,
                             ty=ty, value=value, is_pub=is_pub)

    def _parse_agent_decl(self, is_pub: bool) -> ast.AgentDecl:
        """agent Foo { fn method_name(args) -> ReturnTy; ... }"""
        kw = self._eat(T.KW_AGENT)
        name = self._eat(T.IDENT).value
        self._eat(T.LBRACE)
        methods: list[ast.AgentMethod] = []
        while not self._at(T.RBRACE):
            mt_start = self._eat(T.KW_FN)
            mname = self._eat(T.IDENT).value
            self._eat(T.LPAREN)
            params: list[ast.FnParam] = []
            if not self._at(T.RPAREN):
                params.append(self._parse_fn_param())
                while self._match(T.COMMA):
                    if self._at(T.RPAREN): break
                    params.append(self._parse_fn_param())
            self._eat(T.RPAREN)
            ret_ty: ast.TyNode | None = None
            if self._match(T.ARROW):
                ret_ty = self._parse_type()
            self._eat(T.SEMI)
            methods.append(ast.AgentMethod(
                span=self._span_of(mt_start), name=mname,
                params=params, return_ty=ret_ty,
            ))
        self._eat(T.RBRACE)
        return ast.AgentDecl(span=self._span_of(kw), name=name,
                             methods=methods, is_pub=is_pub)

    # ========================================================================
    # Types
    # ========================================================================
    def _parse_type(self) -> ast.TyNode:
        t = self._peek()
        # tensor<...>
        if t.kind == T.KW_TENSOR:
            return self._parse_tensor_type()
        # tile<...>
        if t.kind == T.KW_TILE:
            return self._parse_tile_type()
        # tuple type (T1, T2)
        if t.kind == T.LPAREN:
            return self._parse_tuple_type()
        # array type [T; N]
        if t.kind == T.LBRACK:
            return self._parse_array_type()
        # reference type & or &mut
        if t.kind == T.AMP:
            return self._parse_ref_type()
        # function type fn(...) -> ...
        if t.kind == T.KW_FN:
            return self._parse_fn_type()
        # primitive or user type — IDENT or builtin type keyword
        return self._parse_named_type()

    def _builtin_kind_to_name(self, kind: T) -> str | None:
        mapping = {
            T.KW_I8: "i8", T.KW_I16: "i16", T.KW_I32: "i32", T.KW_I64: "i64",
            T.KW_ISIZE: "isize",
            T.KW_U8: "u8", T.KW_U16: "u16", T.KW_U32: "u32", T.KW_U64: "u64",
            T.KW_USIZE: "usize",
            T.KW_BOOL: "bool", T.KW_CHAR_TY: "char",
            T.KW_BF16: "bf16", T.KW_F16: "f16", T.KW_F32: "f32", T.KW_F64: "f64",
            T.KW_FP8: "fp8", T.KW_MXFP4: "mxfp4", T.KW_NVFP4: "nvfp4",
            T.KW_TERNARY: "ternary",
        }
        return mapping.get(kind)

    def _parse_named_type(self) -> ast.TyNode:
        t = self._peek()
        builtin = self._builtin_kind_to_name(t.kind)
        if builtin is not None:
            self.i += 1
            return ast.TyName(span=self._span_of(t), name=builtin)
        if t.kind == T.IDENT:
            self.i += 1
            # Optional generic args: Foo<A, B>
            if self._at(T.LT):
                # Distinguish generic args from comparison: in type position, <...> is generics
                args = self._parse_type_generic_args()
                return ast.TyGeneric(span=self._span_of(t), base=t.value, args=args)
            return ast.TyName(span=self._span_of(t), name=t.value)
        raise ParseError("expected type", t)

    def _parse_type_generic_args(self) -> list[ast.TyNode]:
        self._eat(T.LT)
        args: list[ast.TyNode] = []
        if not self._at(T.GT):
            args.append(self._parse_type())
            while self._match(T.COMMA):
                if self._at(T.GT): break
                args.append(self._parse_type())
        self._eat(T.GT)
        return args

    def _parse_tensor_type(self) -> ast.TyTensor:
        kw = self._eat(T.KW_TENSOR)
        self._eat(T.LT)
        self._no_cmp_lt_gt += 1
        try:
            dtype = self._parse_type()
            self._eat(T.COMMA)
            # shape: [d1, d2, ...]
            self._eat(T.LBRACK)
            shape: list[ast.Expr] = []
            if not self._at(T.RBRACK):
                shape.append(self._parse_expr())
                while self._match(T.COMMA):
                    if self._at(T.RBRACK): break
                    shape.append(self._parse_expr())
            self._eat(T.RBRACK)
            device: ast.Expr | None = None
            layout: ast.Expr | None = None
            if self._match(T.COMMA):
                # device or layout — accept first as device, second as layout
                device = self._parse_expr()
                if self._match(T.COMMA):
                    layout = self._parse_expr()
        finally:
            self._no_cmp_lt_gt -= 1
        self._eat(T.GT)
        return ast.TyTensor(span=self._span_of(kw), dtype=dtype, shape=shape,
                            device=device, layout=layout)

    def _parse_tile_type(self) -> ast.TyTile:
        kw = self._eat(T.KW_TILE)
        self._eat(T.LT)
        self._no_cmp_lt_gt += 1
        try:
            dtype = self._parse_type()
            self._eat(T.COMMA)
            self._eat(T.LBRACK)
            shape: list[ast.Expr] = []
            if not self._at(T.RBRACK):
                shape.append(self._parse_expr())
                while self._match(T.COMMA):
                    if self._at(T.RBRACK): break
                    shape.append(self._parse_expr())
            self._eat(T.RBRACK)
            self._eat(T.COMMA)
            memspace = self._parse_expr()
        finally:
            self._no_cmp_lt_gt -= 1
        self._eat(T.GT)
        return ast.TyTile(span=self._span_of(kw), dtype=dtype, shape=shape, memspace=memspace)

    def _parse_tuple_type(self) -> ast.TyNode:
        kw = self._eat(T.LPAREN)
        if self._at(T.RPAREN):
            self._eat(T.RPAREN)
            return ast.TyName(span=self._span_of(kw), name="()")
        first = self._parse_type()
        if not self._at(T.COMMA):
            self._eat(T.RPAREN)
            return first
        elems = [first]
        while self._match(T.COMMA):
            if self._at(T.RPAREN): break
            elems.append(self._parse_type())
        self._eat(T.RPAREN)
        return ast.TyTuple(span=self._span_of(kw), elems=elems)

    def _parse_array_type(self) -> ast.TyArray:
        kw = self._eat(T.LBRACK)
        elem = self._parse_type()
        self._eat(T.SEMI)
        size = self._parse_expr()
        self._eat(T.RBRACK)
        return ast.TyArray(span=self._span_of(kw), elem=elem, size=size)

    def _parse_ref_type(self) -> ast.TyRef:
        kw = self._eat(T.AMP)
        is_mut = bool(self._match(T.KW_MUT))
        inner = self._parse_type()
        return ast.TyRef(span=self._span_of(kw), inner=inner, is_mut=is_mut)

    def _parse_fn_type(self) -> ast.TyFn:
        kw = self._eat(T.KW_FN)
        self._eat(T.LPAREN)
        params: list[ast.TyNode] = []
        if not self._at(T.RPAREN):
            params.append(self._parse_type())
            while self._match(T.COMMA):
                if self._at(T.RPAREN): break
                params.append(self._parse_type())
        self._eat(T.RPAREN)
        self._eat(T.ARROW)
        ret = self._parse_type()
        return ast.TyFn(span=self._span_of(kw), params=params, ret=ret)

    # ========================================================================
    # Statements / Block
    # ========================================================================
    def _parse_block(self) -> ast.Block:
        kw = self._eat(T.LBRACE)
        stmts: list[ast.Stmt] = []
        final_expr: ast.Expr | None = None
        while not self._at(T.RBRACE):
            t = self._peek()
            if t.kind == T.KW_LET:
                stmts.append(self._parse_let_stmt())
            elif t.kind == T.KW_CONST:
                stmts.append(self._parse_const_stmt())
            else:
                # Expression statement: either ends with ; (stmt) or is the final expr
                e = self._parse_expr()
                if isinstance(e, (ast.For, ast.While, ast.Loop)):
                    # These never produce values; always treated as stmts
                    self._match(T.SEMI)  # optional
                    stmts.append(ast.ExprStmt(span=e.span, expr=e))
                elif self._match(T.SEMI):
                    stmts.append(ast.ExprStmt(span=e.span, expr=e))
                elif self._at(T.RBRACE):
                    final_expr = e
                else:
                    # Block-y expressions (if, match, block) don't require semicolons
                    if isinstance(e, (ast.If, ast.Match, ast.Block)):
                        stmts.append(ast.ExprStmt(span=e.span, expr=e))
                    else:
                        raise ParseError("expected ';' or '}' after expression", self._peek())
        self._eat(T.RBRACE)
        return ast.Block(span=self._span_of(kw), stmts=stmts, final_expr=final_expr)

    def _parse_let_stmt(self) -> ast.Let:
        kw = self._eat(T.KW_LET)
        is_mut = bool(self._match(T.KW_MUT))
        name = self._eat(T.IDENT).value
        ty: ast.TyNode | None = None
        if self._match(T.COLON):
            ty = self._parse_type()
        value: ast.Expr | None = None
        if self._match(T.EQ):
            value = self._parse_expr()
        self._eat(T.SEMI)
        return ast.Let(span=self._span_of(kw), name=name, is_mut=is_mut,
                       ty=ty, value=value)

    def _parse_const_stmt(self) -> ast.ConstStmt:
        kw = self._eat(T.KW_CONST)
        name = self._eat(T.IDENT).value
        self._eat(T.COLON)
        ty = self._parse_type()
        self._eat(T.EQ)
        value = self._parse_expr()
        self._eat(T.SEMI)
        return ast.ConstStmt(span=self._span_of(kw), name=name, ty=ty, value=value)

    # ========================================================================
    # Expressions (precedence climbing)
    # ========================================================================
    def _parse_expr(self) -> ast.Expr:
        return self._parse_assign()

    def _parse_assign(self) -> ast.Expr:
        lhs = self._parse_logical_or()
        if self._at_any(T.EQ, T.PLUSEQ, T.MINUSEQ, T.STAREQ, T.SLASHEQ, T.PERCENTEQ):
            op_tok = self._peek()
            self.i += 1
            rhs = self._parse_assign()  # right-assoc
            op = op_tok.value
            return ast.Assign(span=lhs.span, target=lhs, op=op, value=rhs)
        return lhs

    def _parse_logical_or(self) -> ast.Expr:
        left = self._parse_logical_and()
        while self._at(T.LOR):
            self.i += 1
            right = self._parse_logical_and()
            left = ast.Binary(span=left.span, op="||", left=left, right=right)
        return left

    def _parse_logical_and(self) -> ast.Expr:
        left = self._parse_equality()
        while self._at(T.LAND):
            self.i += 1
            right = self._parse_equality()
            left = ast.Binary(span=left.span, op="&&", left=left, right=right)
        return left

    def _parse_equality(self) -> ast.Expr:
        left = self._parse_relational()
        while self._at_any(T.EQEQ, T.NEQ):
            t = self._peek(); self.i += 1
            right = self._parse_relational()
            left = ast.Binary(span=left.span, op=t.value, left=left, right=right)
        return left

    def _parse_relational(self) -> ast.Expr:
        left = self._parse_bitor()
        while True:
            if self._no_cmp_lt_gt > 0:
                # Inside type generic args: only allow <= and >= (those have
                # distinct tokens), NOT < or > which close generics.
                if self._at_any(T.LEQ, T.GEQ):
                    t = self._peek(); self.i += 1
                else:
                    break
            else:
                if self._at_any(T.LT, T.LEQ, T.GT, T.GEQ):
                    t = self._peek(); self.i += 1
                else:
                    break
            right = self._parse_bitor()
            left = ast.Binary(span=left.span, op=t.value, left=left, right=right)
        return left

    def _parse_bitor(self) -> ast.Expr:
        left = self._parse_bitxor()
        while self._at(T.PIPE):
            self.i += 1
            right = self._parse_bitxor()
            left = ast.Binary(span=left.span, op="|", left=left, right=right)
        return left

    def _parse_bitxor(self) -> ast.Expr:
        left = self._parse_bitand()
        while self._at(T.CARET):
            self.i += 1
            right = self._parse_bitand()
            left = ast.Binary(span=left.span, op="^", left=left, right=right)
        return left

    def _parse_bitand(self) -> ast.Expr:
        left = self._parse_shift()
        while self._at(T.AMP):
            self.i += 1
            right = self._parse_shift()
            left = ast.Binary(span=left.span, op="&", left=left, right=right)
        return left

    def _parse_shift(self) -> ast.Expr:
        left = self._parse_additive()
        while self._at_any(T.SHL, T.SHR):
            t = self._peek(); self.i += 1
            right = self._parse_additive()
            left = ast.Binary(span=left.span, op=t.value, left=left, right=right)
        return left

    def _parse_additive(self) -> ast.Expr:
        left = self._parse_multiplicative()
        while self._at_any(T.PLUS, T.MINUS):
            t = self._peek(); self.i += 1
            right = self._parse_multiplicative()
            left = ast.Binary(span=left.span, op=t.value, left=left, right=right)
        return left

    def _parse_multiplicative(self) -> ast.Expr:
        left = self._parse_range()
        while self._at_any(T.STAR, T.SLASH, T.PERCENT):
            t = self._peek(); self.i += 1
            right = self._parse_range()
            left = ast.Binary(span=left.span, op=t.value, left=left, right=right)
        return left

    def _parse_range(self) -> ast.Expr:
        # Ranges only at this precedence: a .. b
        left = self._parse_unary()
        if self._at(T.DOTDOT):
            tok = self._peek(); self.i += 1
            # Optional end
            if self._at_any(T.RPAREN, T.RBRACK, T.RBRACE, T.SEMI, T.COMMA, T.LBRACE):
                return ast.Range(span=left.span, start=left, end=None)
            end = self._parse_unary()
            return ast.Range(span=left.span, start=left, end=end)
        return left

    def _parse_unary(self) -> ast.Expr:
        if self._at_any(T.MINUS, T.BANG, T.TILDE):
            t = self._peek(); self.i += 1
            operand = self._parse_unary()
            return ast.Unary(span=self._span_of(t), op=t.value, operand=operand)
        if self._at(T.STAR):
            t = self._peek(); self.i += 1
            operand = self._parse_unary()
            return ast.Unary(span=self._span_of(t), op="*", operand=operand)
        if self._at(T.AMP):
            t = self._peek(); self.i += 1
            is_mut = bool(self._match(T.KW_MUT))
            operand = self._parse_unary()
            return ast.Unary(span=self._span_of(t), op=("&mut" if is_mut else "&"),
                             operand=operand)
        return self._parse_postfix()

    def _parse_postfix(self) -> ast.Expr:
        expr = self._parse_primary()
        while True:
            if self._at(T.KW_AS):
                # `expr as Type` — cast
                self.i += 1
                target_ty = self._parse_type()
                expr = ast.Cast(span=expr.span, value=expr, target_ty=target_ty)
                continue
            if self._at(T.DOT):
                self.i += 1
                name = self._eat(T.IDENT).value
                expr = ast.Field(span=expr.span, obj=expr, name=name)
            elif self._at(T.LPAREN):
                self.i += 1
                args: list[ast.Expr] = []
                if not self._at(T.RPAREN):
                    args.append(self._parse_expr())
                    while self._match(T.COMMA):
                        if self._at(T.RPAREN): break
                        args.append(self._parse_expr())
                self._eat(T.RPAREN)
                expr = ast.Call(span=expr.span, callee=expr, args=args)
            elif self._at(T.LBRACK):
                self.i += 1
                indices: list[ast.Expr] = []
                if not self._at(T.RBRACK):
                    indices.append(self._parse_expr())
                    while self._match(T.COMMA):
                        if self._at(T.RBRACK): break
                        indices.append(self._parse_expr())
                self._eat(T.RBRACK)
                expr = ast.Index(span=expr.span, callee=expr, indices=indices)
            elif self._at(T.COLONCOLON):
                # path or turbofish: foo::bar OR foo::<T>(args)
                self.i += 1
                if self._at(T.LT):
                    # turbofish: foo::<T1, T2>
                    args = self._parse_type_generic_args()
                    if isinstance(expr, ast.Name):
                        expr = ast.Name(span=expr.span, name=expr.name, generics=args)
                    elif isinstance(expr, ast.Path):
                        # path::<T> — apply generics to last segment by returning a Name
                        # with combined name
                        name = "::".join(expr.segments)
                        expr = ast.Name(span=expr.span, name=name, generics=args)
                    else:
                        raise ParseError("turbofish on non-name", self._peek())
                else:
                    # next ident is path continuation
                    ident = self._eat(T.IDENT)
                    if isinstance(expr, ast.Name) and not expr.generics:
                        expr = ast.Path(span=expr.span, segments=[expr.name, ident.value])
                    elif isinstance(expr, ast.Path):
                        expr = ast.Path(span=expr.span, segments=expr.segments + [ident.value])
                    else:
                        raise ParseError("path segment on non-name", ident)
            else:
                break
        return expr

    def _parse_primary(self) -> ast.Expr:
        t = self._peek()
        # Literals
        if t.kind == T.INT:
            self.i += 1
            return ast.IntLit(span=self._span_of(t), value=t.int_value, type_suffix=t.type_suffix)
        if t.kind == T.FLOAT:
            self.i += 1
            return ast.FloatLit(span=self._span_of(t), value=t.float_value, type_suffix=t.type_suffix)
        if t.kind == T.STRING:
            self.i += 1
            return ast.StrLit(span=self._span_of(t), value=t.string_value)
        if t.kind == T.CHAR:
            self.i += 1
            return ast.CharLit(span=self._span_of(t), value=t.char_value)
        if t.kind == T.KW_TRUE:
            self.i += 1
            return ast.BoolLit(span=self._span_of(t), value=True)
        if t.kind == T.KW_FALSE:
            self.i += 1
            return ast.BoolLit(span=self._span_of(t), value=False)

        # Block, if, match, for, while, loop
        if t.kind == T.LBRACE:
            return self._parse_block()
        if t.kind == T.KW_IF:
            return self._parse_if_expr()
        if t.kind == T.KW_MATCH:
            return self._parse_match_expr()
        if t.kind == T.KW_FOR:
            return self._parse_for_expr()
        if t.kind == T.KW_WHILE:
            return self._parse_while_expr()
        if t.kind == T.KW_LOOP:
            return self._parse_loop_expr()
        if t.kind == T.KW_BREAK:
            self.i += 1
            value = None
            if not self._at_any(T.SEMI, T.COMMA, T.RBRACE, T.RPAREN):
                # crude heuristic
                if not self._at(T.EOF):
                    value = self._parse_expr()
            return ast.Break(span=self._span_of(t), value=value)
        if t.kind == T.KW_CONTINUE:
            self.i += 1
            return ast.Continue(span=self._span_of(t))
        if t.kind == T.KW_RETURN:
            self.i += 1
            value = None
            if not self._at_any(T.SEMI, T.RBRACE):
                value = self._parse_expr()
            return ast.Return(span=self._span_of(t), value=value)

        # AGI-specific primaries
        if t.kind == T.KW_QUOTE:
            # quote { expr }  or  quote(expr)
            self.i += 1
            if self._at(T.LBRACE):
                inner = self._parse_block()
            elif self._at(T.LPAREN):
                self._eat(T.LPAREN)
                inner = self._parse_expr()
                self._eat(T.RPAREN)
            else:
                raise ParseError("expected '{' or '(' after quote", self._peek())
            return ast.Quote(span=self._span_of(t), inner=inner)
        if t.kind == T.KW_SPLICE:
            self.i += 1
            self._eat(T.LPAREN)
            inner = self._parse_expr()
            self._eat(T.RPAREN)
            return ast.Splice(span=self._span_of(t), inner=inner)
        if t.kind == T.KW_MODIFY:
            self.i += 1
            self._eat(T.LPAREN)
            target = self._parse_expr()
            self._eat(T.COMMA)
            transformation = self._parse_expr()
            self._eat(T.COMMA)
            verifier = self._parse_expr()
            self._eat(T.RPAREN)
            return ast.Modify(span=self._span_of(t), target=target,
                              transformation=transformation, verifier=verifier)

        # Parenthesized / tuple
        if t.kind == T.LPAREN:
            self.i += 1
            if self._at(T.RPAREN):
                self.i += 1
                return ast.TupleLit(span=self._span_of(t), elems=[])
            first = self._parse_expr()
            if self._match(T.COMMA):
                elems = [first]
                # Guarded loop: track i to detect an iteration that fails
                # to advance — that's a parser bug producing infinite loop
                # on malformed input. Bail loudly.
                last_i = self.i - 1
                while not self._at(T.RPAREN):
                    if self.i == last_i:
                        raise ParseError(
                            "tuple literal: malformed element (parser made "
                            "no progress)", self._tok())
                    last_i = self.i
                    elems.append(self._parse_expr())
                    if not self._match(T.COMMA):
                        break
                self._eat(T.RPAREN)
                return ast.TupleLit(span=self._span_of(t), elems=elems)
            self._eat(T.RPAREN)
            return first

        # Array literal
        if t.kind == T.LBRACK:
            self.i += 1
            elems: list[ast.Expr] = []
            if not self._at(T.RBRACK):
                elems.append(self._parse_expr())
                while self._match(T.COMMA):
                    if self._at(T.RBRACK): break
                    elems.append(self._parse_expr())
            self._eat(T.RBRACK)
            return ast.ArrayLit(span=self._span_of(t), elems=elems)

        # Identifier (also accept some "type" keywords as identifiers in expr context — gpu(0), cpu, etc.)
        if t.kind == T.IDENT:
            self.i += 1
            return ast.Name(span=self._span_of(t), name=t.value)
        # Allow type-like keywords as expression names: tensor::zeros(), gpu(0),
        # cpu, smem, reg, hbm, tmem (device/memspace markers), grad/jvp/vjp/vmap (transforms)
        EXPR_NAME_KEYWORDS = (
            T.KW_TENSOR, T.KW_TILE,
            T.KW_GPU, T.KW_CPU, T.KW_HBM, T.KW_SMEM, T.KW_REG, T.KW_TMEM, T.KW_DEVICE,
            T.KW_GRAD, T.KW_JVP, T.KW_VJP, T.KW_VMAP, T.KW_KERNEL,
        )
        if t.kind in EXPR_NAME_KEYWORDS:
            self.i += 1
            return ast.Name(span=self._span_of(t), name=t.value)

        raise ParseError("expected expression", t)

    def _parse_if_expr(self) -> ast.If:
        kw = self._eat(T.KW_IF)
        cond = self._parse_expr()
        then_block = self._parse_block()
        else_branch: ast.Block | ast.If | None = None
        if self._match(T.KW_ELSE):
            if self._at(T.KW_IF):
                else_branch = self._parse_if_expr()
            else:
                else_branch = self._parse_block()
        return ast.If(span=self._span_of(kw), cond=cond, then=then_block, else_=else_branch)

    def _parse_match_expr(self) -> ast.Match:
        kw = self._eat(T.KW_MATCH)
        scrutinee = self._parse_expr()
        self._eat(T.LBRACE)
        arms: list[ast.MatchArm] = []
        while not self._at(T.RBRACE):
            arm_start = self._peek()
            pat = self._parse_pattern()
            guard = None
            if self._match(T.KW_IF):
                guard = self._parse_expr()
            self._eat(T.FATARROW)
            body = self._parse_expr()
            arms.append(ast.MatchArm(span=self._span_of(arm_start),
                                     pattern=pat, guard=guard, body=body))
            if not self._match(T.COMMA):
                # last arm may omit comma
                break
        self._eat(T.RBRACE)
        return ast.Match(span=self._span_of(kw), scrutinee=scrutinee, arms=arms)

    def _parse_pattern(self) -> ast.Pattern:
        t = self._peek()
        if t.kind == T.IDENT and t.value == "_":
            self.i += 1
            return ast.PatWildcard(span=self._span_of(t))
        if t.kind == T.IDENT:
            # variable bind (or could be enum variant in v0.2)
            self.i += 1
            return ast.PatBind(span=self._span_of(t), name=t.value, is_mut=False)
        if t.kind in (T.INT, T.FLOAT, T.STRING, T.CHAR, T.KW_TRUE, T.KW_FALSE):
            # Literal pattern: parse as a primary expr then wrap
            lit = self._parse_primary()
            return ast.PatLit(span=lit.span, value=lit)
        if t.kind == T.LPAREN:
            self.i += 1
            elems: list[ast.Pattern] = []
            if not self._at(T.RPAREN):
                elems.append(self._parse_pattern())
                while self._match(T.COMMA):
                    if self._at(T.RPAREN): break
                    elems.append(self._parse_pattern())
            self._eat(T.RPAREN)
            return ast.PatTuple(span=self._span_of(t), elems=elems)
        raise ParseError("expected pattern", t)

    def _parse_for_expr(self) -> ast.For:
        kw = self._eat(T.KW_FOR)
        var_tok = self._eat(T.IDENT)
        self._eat(T.KW_IN)
        iter_expr = self._parse_expr()
        body = self._parse_block()
        return ast.For(span=self._span_of(kw), var_name=var_tok.value,
                       iter_expr=iter_expr, body=body)

    def _parse_while_expr(self) -> ast.While:
        kw = self._eat(T.KW_WHILE)
        cond = self._parse_expr()
        body = self._parse_block()
        return ast.While(span=self._span_of(kw), cond=cond, body=body)

    def _parse_loop_expr(self) -> ast.Loop:
        kw = self._eat(T.KW_LOOP)
        body = self._parse_block()
        return ast.Loop(span=self._span_of(kw), body=body)


def parse(source: str, filename: str = "<input>",
          include_stdlib: bool = False) -> ast.Program:
    """Convenience: lex and parse a full source string.

    Pass include_stdlib=True to also append the stdlib transcendentals
    (__exp/__log/__sin/__cos/__sqrt/__sigmoid/__relu) to the program.
    The function-DCE pass drops unused stdlib functions, so this is
    cheap. The end-to-end test runners enable it by default; callers
    that operate on raw user-AST (e.g. counting specific op kinds in
    optimizer tests) leave it off.
    """
    user_prog = Parser(lex(source, filename)).parse_program()
    if include_stdlib:
        import os as _os
        stdlib_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "stdlib", "transcendentals.hx"
        )
        if _os.path.isfile(stdlib_path):
            with open(stdlib_path, encoding="utf-8") as f:
                stdlib_src = f.read()
            stdlib_prog = Parser(lex(stdlib_src, stdlib_path)).parse_program()
            # Don't redefine functions the user already defined with the same
            # name — user wins.
            user_names = {item.name for item in user_prog.items
                          if isinstance(item, ast.FnDecl)}
            for item in stdlib_prog.items:
                if isinstance(item, ast.FnDecl) and item.name not in user_names:
                    user_prog.items.append(item)
    return user_prog


# ============================================================================
# CLI for quick testing
# ============================================================================
if __name__ == "__main__":
    import sys, pprint
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            src = f.read()
    else:
        src = sys.stdin.read()
    prog = parse(src)
    pprint.pp(prog, depth=8)
