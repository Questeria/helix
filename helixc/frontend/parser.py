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

from typing import Optional

from .lexer import Token, T, lex
from . import ast_nodes as ast
from .diagnostics import render_caret


class ParseError(Exception):
    def __init__(self, msg: str, tok: Token):
        super().__init__(f"{tok.line}:{tok.col}: parse error: {msg} (got {tok.kind.name} {tok.value!r})")
        self.token = tok
        self.msg = msg

    def render(self, source: "str | None" = None,
               filename: str = "<input>",
               color: "bool | None" = None) -> str:
        """Format with source-line + caret display via the shared
        diagnostics module (Stage 22). Falls back to bare str(self) if
        source is None or the line is out of range."""
        if source is None:
            return str(self)
        return render_caret(
            filename=filename,
            line=self.token.line,
            col=self.token.col,
            msg=self.msg,
            source=source,
            level="error",
            color=color,
        )


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
        if t.kind == T.KW_EXTERN:
            # Stage 16.5: extern "C" fn name(args) -> ret;
            return self._parse_extern_decl(is_pub, attrs)
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
        if t.kind == T.KW_MOD:
            return self._parse_mod_block(is_pub)
        if t.kind == T.KW_IMPL:
            return self._parse_impl_block(is_pub)
        if t.kind == T.KW_TRAIT:
            return self._parse_trait_decl(is_pub)
        raise ParseError(f"expected item (fn/struct/enum/type/use/const/agent/mod/impl/trait)", t)

    def _parse_impl_block(self, is_pub: bool) -> ast.ImplBlock:
        kw = self._eat(T.KW_IMPL)
        # Allow `impl Trait for Type` and `impl Type` forms.
        first_name = self._eat_name_token().value
        trait_name: Optional[str] = None
        target_name = first_name
        if self._at(T.KW_FOR):
            self._eat(T.KW_FOR)
            trait_name = first_name
            target_name = self._eat_name_token().value
        self._eat(T.LBRACE)
        methods: list[ast.FnDecl] = []
        while not self._at(T.RBRACE):
            attrs = self._parse_attributes()
            is_pub_method = bool(self._match(T.KW_PUB))
            if self._at(T.KW_FN):
                methods.append(self._parse_fn_decl(is_pub_method, attrs))
            else:
                raise ParseError("expected fn inside impl block", self._peek())
        self._eat(T.RBRACE)
        return ast.ImplBlock(span=self._span_of(kw), target=target_name,
                             methods=methods, trait_name=trait_name,
                             is_pub=is_pub)

    def _parse_trait_decl(self, is_pub: bool) -> ast.ImplBlock:
        # Phase 1.8: traits are accepted but only as documentation.
        # Their method signatures are not yet checked against impls.
        # Parse `trait Name { fn sig; ... }` and discard methods (since
        # they have no bodies). Returns a stub ImplBlock just so the
        # item type is uniform — the lift-pass treats no-bodies as no-op.
        kw = self._eat(T.KW_TRAIT)
        name = self._eat_name_token().value
        # Optional generics: trait Name[T] {}
        self._parse_generic_params()
        self._eat(T.LBRACE)
        # Skip method signatures (with optional default bodies).
        while not self._at(T.RBRACE):
            self._parse_attributes()
            self._match(T.KW_PUB)
            if self._at(T.KW_FN):
                self.i += 1
                self._eat(T.IDENT)
                self._parse_generic_params()
                self._eat(T.LPAREN)
                while not self._at(T.RPAREN):
                    if self._at(T.COMMA):
                        self.i += 1
                        continue
                    self.i += 1
                self._eat(T.RPAREN)
                if self._at(T.ARROW):
                    self.i += 1
                    self._parse_type()
                if self._at(T.LBRACE):
                    # default body — skip it
                    self._parse_block()
                else:
                    self._match(T.SEMI)
            else:
                # Skip unknown tokens up to the closing brace.
                self.i += 1
        self._eat(T.RBRACE)
        return ast.ImplBlock(span=self._span_of(kw), target=name,
                             methods=[], trait_name=None, is_pub=is_pub)

    def _parse_mod_block(self, is_pub: bool) -> ast.ModBlock:
        kw = self._eat(T.KW_MOD)
        name = self._eat(T.IDENT).value
        self._eat(T.LBRACE)
        items: list[ast.Item] = []
        while not self._at(T.RBRACE):
            items.append(self._parse_item())
        self._eat(T.RBRACE)
        return ast.ModBlock(span=self._span_of(kw), name=name, items=items, is_pub=is_pub)

    def _parse_attributes(self) -> list[str]:
        attrs: list[str] = []
        while self._at(T.AT):
            self._eat(T.AT)
            # Either an ident or a keyword (kernel, pure, etc.)
            t = self._peek()
            attr_name: str
            if t.kind == T.IDENT:
                attr_name = t.value
                self.i += 1
            elif t.kind == T.KW_KERNEL:
                attr_name = "kernel"; self.i += 1
            elif t.kind == T.KW_GRAD:
                attr_name = "grad"; self.i += 1
            else:
                # accept any identifier-shaped keyword
                if t.value.replace("_", "").isalnum():
                    attr_name = t.value; self.i += 1
                else:
                    raise ParseError("expected attribute name after @", t)

            # Stage 27: special-case @autotune(KEY: [v1, v2], KEY2: [v])
            # before falling through to the generic ident-capture path.
            if attr_name == "autotune" and self._at(T.LPAREN):
                pairs = self._parse_autotune_args()
                attrs.append("autotune")
                for k, vs in pairs:
                    vs_str = ",".join(str(v) for v in vs)
                    attrs.append(f"autotune:{k}={vs_str}")
                continue

            # Stage 28.7: @deprecated("msg"), @since("v0.3") capture a
            # single string arg.
            if attr_name in ("deprecated", "since") and self._at(T.LPAREN):
                msg = self._parse_string_attr_arg()
                attrs.append(attr_name)
                if msg is not None:
                    attrs.append(f"{attr_name}:{msg}")
                continue

            # optional (args) — for @effect(io), @effect(io, rng) we
            # capture each comma-separated ident as `effect:<name>`.
            # For other attributes we record the bare name and the
            # arg-bracketed payload as `<attr>:<arg>` for each arg too,
            # so callers can introspect.
            arg_idents: list[str] = []
            if self._at(T.LPAREN):
                self._eat(T.LPAREN)
                depth = 1
                while depth > 0:
                    t = self._peek()
                    if t.kind == T.EOF:
                        raise ParseError("unclosed attribute args", t)
                    if t.kind == T.LPAREN:
                        depth += 1
                        self.i += 1
                    elif t.kind == T.RPAREN:
                        depth -= 1
                        self.i += 1
                    elif t.kind == T.COMMA:
                        self.i += 1
                    elif t.kind == T.IDENT:
                        if depth == 1:
                            arg_idents.append(t.value)
                        self.i += 1
                    else:
                        # Skip non-ident tokens inside the parens
                        self.i += 1
            if attr_name == "effect" and arg_idents:
                for arg in arg_idents:
                    attrs.append(f"effect:{arg}")
            else:
                attrs.append(attr_name)
        return attrs

    def _parse_autotune_args(self) -> list[tuple[str, list[int]]]:
        """Parse `(KEY: [v1, v2, ...], KEY2: [v3])` form. Stage 27."""
        self._eat(T.LPAREN)
        pairs: list[tuple[str, list[int]]] = []
        if not self._at(T.RPAREN):
            pairs.append(self._parse_autotune_pair())
            while self._match(T.COMMA):
                if self._at(T.RPAREN):
                    break
                pairs.append(self._parse_autotune_pair())
        self._eat(T.RPAREN)
        return pairs

    def _parse_autotune_pair(self) -> tuple[str, list[int]]:
        kt = self._peek()
        if kt.kind != T.IDENT:
            raise ParseError("expected autotune key (ident)", kt)
        key = kt.value
        self.i += 1
        self._eat(T.COLON)
        self._eat(T.LBRACK)
        vals: list[int] = []
        if not self._at(T.RBRACK):
            vals.append(self._parse_autotune_int())
            while self._match(T.COMMA):
                if self._at(T.RBRACK):
                    break
                vals.append(self._parse_autotune_int())
        self._eat(T.RBRACK)
        return (key, vals)

    def _parse_autotune_int(self) -> int:
        t = self._peek()
        if t.kind != T.INT:
            raise ParseError("expected integer in autotune list", t)
        self.i += 1
        # Stage 28.9 cycle 95 audit-R F2 fix (HIGH conf 90): pre-fix
        # used `t.value.split("_")[0]` to strip the type-suffix, but
        # `_` is ALSO the digit-separator character. So `1_000_000`
        # was split to `["1", "000", "000"]` and only `"1"` survived,
        # silently truncating `@autotune(block: [1_000_000])` to
        # `[1]`. The lexer already computed `t.int_value` with full
        # precision (line 354); use that directly.
        if t.int_value is None:
            raise ParseError(f"bad integer literal {t.value!r}", t)
        return t.int_value

    def _parse_string_attr_arg(self) -> "str | None":
        """Parse a single string literal in `(msg)` form for
        @deprecated and @since. Returns None if no string was found
        (caller's job to handle)."""
        self._eat(T.LPAREN)
        msg: "str | None" = None
        if self._at(T.STRING):
            t = self._peek()
            msg = t.string_value or ""
            self.i += 1
        # Skip any other tokens to RPAREN (lenient)
        while not self._at(T.RPAREN) and self._peek().kind != T.EOF:
            self.i += 1
        self._eat(T.RPAREN)
        return msg

    # ---- extern fn (FFI declaration, Stage 16.5) ----
    def _parse_extern_decl(self, is_pub: bool, attrs: list[str]) -> ast.FnDecl:
        # extern "C" fn name(args) -> ret;
        # No body — just a declaration. Calls go through PLT/GOT at runtime.
        ekw = self._eat(T.KW_EXTERN)
        # ABI string — required, must currently be "C".
        abi_tok = self._eat(T.STRING)
        abi = abi_tok.string_value or ""
        if abi != "C":
            raise ParseError(
                f"only extern \"C\" is supported (got {abi!r})", abi_tok)
        kw = self._eat(T.KW_FN)
        name_tok = self._eat(T.IDENT)
        # No generics on extern fns (Phase-0 simplification).
        self._eat(T.LPAREN)
        params: list[ast.FnParam] = []
        if not self._at(T.RPAREN):
            params.append(self._parse_fn_param())
            while self._match(T.COMMA):
                if self._at(T.RPAREN):
                    break
                params.append(self._parse_fn_param())
        self._eat(T.RPAREN)
        ret_ty: ast.TyNode | None = None
        if self._match(T.ARROW):
            ret_ty = self._parse_type()
        # Body is just a `;` — emit an empty Block placeholder.
        self._eat(T.SEMI)
        empty_body = ast.Block(span=self._span_of(ekw), stmts=[],
                               final_expr=None)
        return ast.FnDecl(
            span=self._span_of(ekw),
            name=name_tok.value,
            generics=[],
            params=params,
            return_ty=ret_ty,
            where_clauses=[],
            body=empty_body,
            attrs=attrs,
            is_pub=is_pub,
            is_extern=True,
            extern_abi=abi,
        )

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
        where_clauses: list[ast.WhereClause] = []
        if self._match(T.KW_WHERE):
            where_clauses = self._parse_where_clauses()
        self._eat(T.SEMI)
        return ast.TypeAlias(span=self._span_of(kw), name=name,
                             generics=generics, target=target, is_pub=is_pub,
                             where_clauses=where_clauses)

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
        # raw pointer *const T or *mut T (Stage 16.5 FFI)
        if t.kind == T.STAR:
            return self._parse_ptr_type()
        # function type fn(...) -> ...
        if t.kind == T.KW_FN:
            return self._parse_fn_type()
        # primitive or user type — IDENT or builtin type keyword
        return self._parse_named_type()

    def _parse_ptr_type(self) -> ast.TyPtr:
        # *const T  |  *mut T
        star = self._eat(T.STAR)
        is_mut = False
        t = self._peek()
        if t.kind == T.KW_CONST:
            self.i += 1
        elif t.kind == T.KW_MUT:
            self.i += 1
            is_mut = True
        else:
            raise ParseError(
                "expected `const` or `mut` after `*` in pointer type", t)
        inner = self._parse_type()
        return ast.TyPtr(span=self._span_of(star), inner=inner, is_mut=is_mut)

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
            if t.kind == T.EOF:
                raise ParseError(
                    f"unclosed block — expected '}}' to close the '{{' opened at "
                    f"{kw.line}:{kw.col}",
                    t,
                )
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
                    # Block-y expressions (if, match, block, unsafe-block)
                    # don't require semicolons
                    if isinstance(e, (ast.If, ast.Match, ast.Block,
                                       ast.UnsafeBlock)):
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
        # Ranges sit between multiplicative and unary in the call chain,
        # but the END operand is parsed at additive precedence so
        # `0 .. n * 2` and `0 .. n + 1` correctly group as
        # `0 .. (n * 2)` / `0 .. (n + 1)` (matching Rust). Without this,
        # `0..3*2` parsed as `(0..3) * 2` and broke for-loops with
        # arithmetic bounds.
        left = self._parse_unary()
        if self._at(T.DOTDOT):
            tok = self._peek(); self.i += 1
            # Optional end
            if self._at_any(T.RPAREN, T.RBRACK, T.RBRACE, T.SEMI, T.COMMA, T.LBRACE):
                return ast.Range(span=left.span, start=left, end=None)
            end = self._parse_additive()
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
                # Tuple field access: `tup.0`, `tup.1`. The "name" is a
                # stringified integer (e.g. "0") so the existing Field AST
                # is reused. Typecheck distinguishes by checking whether
                # obj_ty is a TyTuple.
                if self._at(T.INT):
                    idx_tok = self._peek()
                    self.i += 1
                    expr = ast.Field(span=expr.span, obj=expr,
                                     name=str(idx_tok.int_value))
                else:
                    name = self._eat(T.IDENT).value
                    expr = ast.Field(span=expr.span, obj=expr, name=name)
            elif self._at(T.LPAREN):
                # Reject postfix-call on statement-position control-flow
                # expressions. Without this, `while c { ... } (foo)`
                # parses as `(while ...)(foo)` — the while result
                # treated as a callable. That broke common patterns
                # like `while loop_body { } (last * 10.0) as i32` by
                # consuming the trailing parenthesized expression as
                # call args. While/For/Loop never produce callable
                # values; treat the LPAREN here as the start of a
                # new statement/expression instead.
                if isinstance(expr, (ast.While, ast.For, ast.Loop)):
                    break
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
        # Stage 28.6: unsafe { ... }
        if t.kind == T.KW_UNSAFE:
            self.i += 1
            body = self._parse_block()
            return ast.UnsafeBlock(span=self._span_of(t), body=body)
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
                # Initialize last_i to self.i (not self.i - 1) so the
                # very first iteration is also checked for progress.
                last_i = self.i
                while not self._at(T.RPAREN):
                    elems.append(self._parse_expr())
                    if self.i == last_i:
                        raise ParseError(
                            "tuple literal: malformed element (parser made "
                            "no progress)", self._peek())
                    last_i = self.i
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
            # Struct literal: `Name { f: v, ... }`. Disambiguate against the
            # ambiguous "Name followed by a block" by requiring `{` then
            # `IDENT :` (i.e., named-field syntax). If the brace contents
            # don't look like fields, fall through to treating Name as a
            # bare identifier (the surrounding parser may then consume the
            # brace as a separate block).
            if self._at(T.LBRACE) and self._peek_struct_lit_start():
                return self._parse_struct_lit_after_name(t.value, t)
            return ast.Name(span=self._span_of(t), name=t.value)
        # Stage 15: special-case `tile<dtype, [N, M], memspace>::method()` in
        # expression position. Without this, `tile<f32, ...>` parses as a
        # comparison `tile < f32`. We detect the pattern by looking ahead:
        # `tile` (KW_TILE) immediately followed by `<` is unambiguous in
        # expression position because comparisons against `tile` (a type
        # keyword) are nonsense.
        if t.kind == T.KW_TILE and self._peek_at(1, T.LT):
            return self._parse_tile_lit_or_op()

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

    def _peek_at(self, offset: int, kind: "T") -> bool:
        """Lookahead: is the token at index self.i + offset of kind `kind`?
        Returns False if past EOF."""
        idx = self.i + offset
        if idx >= len(self.toks):
            return False
        return self.toks[idx].kind == kind

    def _parse_tile_lit_or_op(self) -> ast.Expr:
        """Stage 15 — parse `tile<dtype, [N, M], memspace>::method()` as a
        primary expression. The caller has verified that the current token is
        KW_TILE followed by `<`. We parse the tile type using the existing
        `_parse_tile_type` helper (which leaves us positioned after the
        closing `>`), then expect `::IDENT` followed by `(args)` and build a
        TileLit AST node carrying the tile type info plus the init kind.

        Phase-0 only supports `::zeros()` and `::ones()` with no args. Other
        methods raise ParseError so misuse is loud.
        """
        kw = self._peek()
        # _parse_tile_type consumes `tile<...>` including the closing `>`.
        ty = self._parse_tile_type()
        # Now expect `::IDENT(args)`.
        self._eat(T.COLONCOLON)
        method_tok = self._eat(T.IDENT)
        method = method_tok.value
        if method not in ("zeros", "ones"):
            raise ParseError(
                f"tile<>::{method}() — only ::zeros() and ::ones() are "
                f"supported in Phase 0",
                method_tok,
            )
        self._eat(T.LPAREN)
        # Phase-0: zeros()/ones() take no args. Reject extras loudly.
        if not self._at(T.RPAREN):
            raise ParseError(
                f"tile<>::{method}() takes no arguments in Phase 0",
                self._peek(),
            )
        self._eat(T.RPAREN)
        return ast.TileLit(
            span=self._span_of(kw),
            dtype=ty.dtype,
            shape=ty.shape,
            memspace=ty.memspace,
            init=method,
        )

    def _peek_struct_lit_start(self) -> bool:
        """Are we positioned at `{ IDENT :` (the start of a struct literal)?
        Lookahead-only; never advances `self.i`. Also accepts an empty brace
        `{ }` (a unit-struct literal). Returns False otherwise so the brace
        can be reinterpreted as a regular block."""
        if not self._at(T.LBRACE):
            return False
        save_i = self.i
        try:
            self.i += 1  # consume '{'
            if self._at(T.RBRACE):
                # Empty `{}` could be a unit struct or an empty block. Treat
                # as block by default (less surprising); user can write
                # `Foo {}` explicitly later if we want literal-empty-struct.
                return False
            if self._at(T.IDENT):
                save2 = self.i
                self.i += 1
                ok = self._at(T.COLON)
                self.i = save2
                return ok
            return False
        finally:
            self.i = save_i

    def _parse_struct_lit_after_name(self, name: str,
                                      name_tok) -> ast.StructLit:
        """We've consumed the IDENT; now consume `{ field: val, ... }`."""
        self._eat(T.LBRACE)
        fields: list[tuple[str, ast.Expr]] = []
        while not self._at(T.RBRACE):
            fname = self._eat(T.IDENT).value
            self._eat(T.COLON)
            fval = self._parse_expr()
            fields.append((fname, fval))
            if not self._match(T.COMMA):
                break
        self._eat(T.RBRACE)
        return ast.StructLit(span=self._span_of(name_tok), name=name,
                             fields=fields)

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
        """Parse a pattern, optionally with `|` alternatives at top level."""
        first = self._parse_pattern_atom()
        if not self._at(T.PIPE):
            return first
        alts: list[ast.Pattern] = [first]
        while self._match(T.PIPE):
            alts.append(self._parse_pattern_atom())
        return ast.PatOr(span=first.span, alts=alts)

    def _parse_pattern_atom(self) -> ast.Pattern:
        t = self._peek()
        # Negative numeric literal in a pattern position (e.g. `-10`,
        # `-10..=-1`, `-100..=100`). Eat the MINUS, parse the underlying
        # primary, and wrap as Unary("-") so PatLit/PatRange downstream
        # gets a numeric expression that the IR lowerer can fold.
        if t.kind == T.MINUS:
            minus_tok = t
            self.i += 1
            inner_tok = self._peek()
            if inner_tok.kind not in (T.INT, T.FLOAT):
                raise ParseError(
                    f"expected numeric literal after '-' in pattern (got {inner_tok.kind} {inner_tok.value!r})",
                    inner_tok,
                )
            inner = self._parse_primary()
            lit_expr = ast.Unary(span=self._span_of(minus_tok), op="-",
                                 operand=inner)
            if self._at(T.DOTDOTEQ) or self._at(T.DOTDOT):
                inclusive = self._at(T.DOTDOTEQ)
                self.i += 1
                # The high bound may also be negative.
                if self._at(T.MINUS):
                    self.i += 1
                    hi_inner = self._parse_primary()
                    hi = ast.Unary(span=hi_inner.span, op="-",
                                   operand=hi_inner)
                else:
                    hi = self._parse_primary()
                return ast.PatRange(span=self._span_of(minus_tok),
                                    lo=lit_expr, hi=hi, inclusive=inclusive)
            return ast.PatLit(span=self._span_of(minus_tok), value=lit_expr)
        if t.kind == T.IDENT and t.value == "_":
            self.i += 1
            return ast.PatWildcard(span=self._span_of(t))
        if t.kind == T.IDENT:
            # IDENT followed by `::` is a path pattern (enum variant). If
            # the variant has a payload `Some(x)` we accept that too — for
            # now the payload binders aren't bound (codegen doesn't yet
            # emit per-variant payload extraction). Bare `IDENT` without
            # `::` is a variable binder.
            self.i += 1
            if self._at(T.COLONCOLON):
                segments = [t.value]
                while self._match(T.COLONCOLON):
                    seg = self._eat(T.IDENT)
                    segments.append(seg.value)
                path_expr = ast.Path(
                    span=self._span_of(t),
                    segments=segments,
                )
                # Payload pattern: `Variant(p1, p2, ...)` — parse each
                # sub-pattern and build a PatVariant. Tag-only stays as
                # PatLit-of-Path (legacy path, simpler test).
                if self._at(T.LPAREN):
                    self.i += 1
                    sub_pats: list[ast.Pattern] = []
                    if not self._at(T.RPAREN):
                        sub_pats.append(self._parse_pattern())
                        while self._match(T.COMMA):
                            if self._at(T.RPAREN): break
                            sub_pats.append(self._parse_pattern())
                    self._eat(T.RPAREN)
                    return ast.PatVariant(span=self._span_of(t),
                                          path=path_expr,
                                          sub_patterns=sub_pats)
                return ast.PatLit(span=self._span_of(t), value=path_expr)
            return ast.PatBind(span=self._span_of(t), name=t.value, is_mut=False)
        if t.kind in (T.INT, T.FLOAT, T.STRING, T.CHAR, T.KW_TRUE, T.KW_FALSE):
            # Literal pattern, or `lo..hi` / `lo..=hi` range pattern.
            lit = self._parse_primary()
            if self._at(T.DOTDOTEQ) or self._at(T.DOTDOT):
                inclusive = self._at(T.DOTDOTEQ)
                self.i += 1
                # High bound may be negative (e.g. `0..-1` or `5..=-1`).
                if self._at(T.MINUS):
                    self.i += 1
                    hi_inner = self._parse_primary()
                    hi = ast.Unary(span=hi_inner.span, op="-",
                                   operand=hi_inner)
                else:
                    hi = self._parse_primary()
                return ast.PatRange(span=lit.span, lo=lit, hi=hi, inclusive=inclusive)
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
        _merge_stdlib(user_prog)
    return user_prog


# Public list of stdlib files merged when `include_stdlib=True`. Held
# at module scope so tests can monkey-patch / override the auto-include
# set (Audit 28.8 A8 regression test does this).
STDLIB_FILES: list[str] = [
    "transcendentals.hx", "option.hx", "result.hx", "vec.hx",
    "hashmap.hx", "string.hx", "iterators.hx", "autodiff.hx",
    "autodiff_reverse.hx", "tensor.hx", "nn.hx", "agi_memory.hx",
    "agi_search.hx", "agi_match.hx", "agi_world.hx", "ieee754.hx",
    # Stage 36 Inc 13: provenance debug/observation helpers
    # (trace_evidence, has_evidence, evidence_left, evidence_right)
    # over the Inc 5/9 arena side-table. Inc 15 extended with
    # evidence_middle, evidence_third, trace_evidence3 for the
    # 3-parent register_derivation3 handles from Inc 14.
    "provenance.hx",
]


# Audit 28.8 A8: if a stdlib file in STDLIB_FILES is missing on disk
# AND this env var is set to "1" / "true", the merge raises
# FileNotFoundError instead of silently skipping. Default (env unset)
# is the lenient behaviour for backward compatibility — but a warning
# is always emitted to stderr.
STDLIB_STRICT_ENV = "HELIXC_STDLIB_STRICT"


def _merge_stdlib(user_prog: "ast.Program") -> None:
    """Merge stdlib items into `user_prog.items`.

    Audit 28.8 A8: the prior implementation only merged `FnDecl` and
    `EnumDecl` items; every other item kind (`StructDecl`, `ImplBlock`,
    `ConstDecl`, `TypeAlias`, `ModuleDecl`/`ModBlock`/`UseDecl`) was
    silently dropped. A user program importing a stdlib `struct Vec<T>`
    silently saw `TyUnknown` propagation and miscompiled. This pass now
    merges ALL named-item kinds with proper conflict handling.

    Conflict policy: user code takes precedence. If `user_prog` already
    declares an item with the same `name` as a stdlib item OF THE SAME
    KIND (fn vs fn, struct vs struct, etc.), the stdlib item is
    skipped. Cross-kind name overlap (a user fn and a stdlib struct
    with the same name) is left to downstream typecheck — we don't
    silently merge those, except for `type`/`struct`/`enum` names. Those
    share one type namespace during stdlib merge so user-defined nominal
    types override same-named stdlib aliases.

    Missing-stdlib-file handling: by default the file is silently
    skipped (legacy behaviour). With `HELIXC_STDLIB_STRICT=1`, missing
    files raise `FileNotFoundError`. Either way, a warning is emitted
    to stderr so users see partial-install symptoms.
    """
    import os as _os
    import sys as _sys
    stdlib_dir = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "stdlib",
    )
    strict = _os.environ.get(STDLIB_STRICT_ENV, "").lower() in ("1", "true", "yes")

    # Index existing user items by (kind_tag, name) for fast conflict checks.
    # Each item-kind goes in its own namespace because the parser allows
    # e.g. fn `foo` and struct `foo` to coexist (downstream typecheck
    # may flag, but the parser doesn't).
    def _kind_tag(item) -> str:
        # ImplBlock has no `name`; key by (target_name, optional_trait_name,
        # method-name-list-hash). We don't dedup ImplBlocks across
        # user/stdlib for v0.1 — only collapse exact-duplicate decls
        # within the stdlib itself if encountered.
        return type(item).__name__

    def _is_type_namespace_item(item) -> bool:
        return isinstance(item, (ast.TypeAlias, ast.StructDecl, ast.EnumDecl))

    def _mark_stdlib_item(item) -> None:
        if isinstance(item, ast.FnDecl):
            if "__stdlib" not in item.attrs:
                item.attrs.append("__stdlib")
            return
        if isinstance(item, ast.ImplBlock):
            for method in item.methods:
                _mark_stdlib_item(method)
            return
        if isinstance(item, ast.ModBlock):
            for sub in item.items:
                _mark_stdlib_item(sub)

    user_keys: set[tuple[str, str]] = set()
    user_type_names: set[str] = set()
    for it in user_prog.items:
        name = getattr(it, "name", None)
        if name is not None:
            user_keys.add((_kind_tag(it), name))
            if _is_type_namespace_item(it):
                user_type_names.add(name)

    for fname in STDLIB_FILES:
        stdlib_path = _os.path.join(stdlib_dir, fname)
        if not _os.path.isfile(stdlib_path):
            msg = f"helixc: stdlib file missing: {stdlib_path}"
            if strict:
                raise FileNotFoundError(msg)
            print(msg, file=_sys.stderr)
            continue
        with open(stdlib_path, encoding="utf-8") as f:
            stdlib_src = f.read()
        stdlib_prog = Parser(lex(stdlib_src, stdlib_path)).parse_program()
        for item in stdlib_prog.items:
            _mark_stdlib_item(item)
            name = getattr(item, "name", None)
            if name is not None:
                key = (_kind_tag(item), name)
                if key in user_keys or (
                    _is_type_namespace_item(item) and name in user_type_names
                ):
                    # User declared something of the same kind+name —
                    # type-namespace conflicts; user takes precedence.
                    continue
                user_keys.add(key)
                if _is_type_namespace_item(item):
                    user_type_names.add(name)
            # Audit 28.8 A8: merge ALL named-item kinds, not just
            # FnDecl/EnumDecl. StructDecl, TraitDecl-via-ImplBlock,
            # ImplBlock, ConstDecl, TypeAlias, ModuleDecl, ModBlock,
            # UseDecl — anything the parser produced gets propagated
            # so downstream passes see the full stdlib surface.
            user_prog.items.append(item)


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
