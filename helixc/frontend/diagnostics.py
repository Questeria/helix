"""
helixc/frontend/diagnostics.py — Stage 22: Pretty error display.

Unified diagnostic rendering for parser, typechecker, and downstream
passes. Provides:

  * `render_caret(filename, line, col, msg, source, hint=, level=)` —
    the source-with-caret format used by ParseError.render and
    TypeError_.render. Now centralized.
  * `did_you_mean(name, candidates, cutoff=0.6, n=1)` — Levenshtein-
    based suggestion helper (delegates to difflib.get_close_matches).
  * `use_color(stream=sys.stderr)` — heuristic for "should ANSI escapes
    be enabled?". Honors `NO_COLOR`, `HELIXC_COLOR=0/1`, and isatty().
  * `Diagnostic` dataclass + `DiagSink` collector for stages that want
    to aggregate multiple errors before bailing.
  * Severity levels: "error" / "warning" / "note".

Color scheme (when enabled):
  * error    -> red bold
  * warning  -> yellow bold
  * note     -> blue bold
  * code     -> default
  * carets   -> red bold (matching level)
  * hint     -> dim green

Color codes are short ANSI sequences; they are stripped when color is
off so output stays diffable in test suites and grep-friendly in logs.

License: Apache 2.0
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Optional


# ----------------------------------------------------------------------
# Color handling
# ----------------------------------------------------------------------
_ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "cyan": "\x1b[36m",
}


def use_color(stream=None) -> bool:
    """Decide whether to emit ANSI escapes. Honors:
      * `NO_COLOR` env var (any value disables, per no-color.org)
      * `HELIXC_COLOR` env var: "1" forces on, "0" forces off
      * stream.isatty() otherwise (default: sys.stderr)
    """
    if "NO_COLOR" in os.environ:
        return False
    forced = os.environ.get("HELIXC_COLOR")
    if forced == "1":
        return True
    if forced == "0":
        return False
    if stream is None:
        stream = sys.stderr
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    # Restart 61 B1: preserve loud-fail discipline. Pre-fix bare
    # `except Exception` swallowed NotImplementedError / AssertionError
    # from stream subclasses that legitimately want to propagate. Narrow
    # to (AttributeError, OSError, ValueError) which are the actual
    # isatty failure modes (closed stream, non-tty stream object,
    # missing fileno). Mirrors the restart 47 B1 narrowing pattern.
    try:
        return bool(isatty())
    except (NotImplementedError, AssertionError, KeyboardInterrupt,
            SystemExit, MemoryError):
        raise
    except (AttributeError, OSError, ValueError):
        return False


def _wrap(s: str, *styles: str, color: bool) -> str:
    if not color or not styles:
        return s
    prefix = "".join(_ANSI.get(st, "") for st in styles)
    if not prefix:
        return s
    return f"{prefix}{s}{_ANSI['reset']}"


# ----------------------------------------------------------------------
# Suggestion helper
# ----------------------------------------------------------------------
def did_you_mean(name: str, candidates, *, cutoff: float = 0.6, n: int = 1):
    """Return up to `n` close matches for `name` from `candidates`,
    sorted best-first. Returns a list (possibly empty).

    Thin wrapper around `difflib.get_close_matches` so callers don't
    have to import it; centralizes the cutoff default at 0.6 (matches
    the existing typechecker behavior)."""
    if not name:
        return []
    candidates = list(candidates)
    return list(get_close_matches(name, candidates, n=n, cutoff=cutoff))


# ----------------------------------------------------------------------
# Caret formatter
# ----------------------------------------------------------------------
def render_caret(
    *,
    filename: str,
    line: int,
    col: int,
    msg: str,
    source: Optional[str],
    hint: Optional[str] = None,
    level: str = "error",
    code: Optional[int] = None,
    color: Optional[bool] = None,
    span_len: int = 1,
) -> str:
    """Render a pretty diagnostic with source-line + caret.

    Output (color disabled):

        error[E123]: <msg>
              --> file.hx:5:12
               |
            5  | let x = ;
               |          ^^^
               = hint: <hint>

    When `source` is None or the line is out of range, falls back to a
    one-line message: `file:line:col: <level>: <msg>`.

    Args:
      filename: source filename (e.g. "loss.hx" or "<input>").
      line, col: 1-indexed.
      msg: error text (single line preferred).
      source: full source buffer (used to slice the offending line).
      hint: optional secondary line, rendered as `= hint:`.
      level: "error" | "warning" | "note".
      code: optional integer error code, e.g. 24100 (provenance).
      color: tri-state. True/False forces, None uses `use_color()`.
      span_len: number of carets to draw (default 1).
    """
    if color is None:
        color = use_color()

    level_color = {
        "error": ("red", "bold"),
        "warning": ("yellow", "bold"),
        "note": ("blue", "bold"),
    }.get(level, ("red", "bold"))

    code_str = f"[E{code}]" if code is not None else ""
    head = _wrap(f"{level}{code_str}", *level_color, color=color) + ": " + msg

    if source is None:
        return f"{filename}:{line}:{col}: {level}: {msg}"

    lines = source.splitlines()
    if not (1 <= line <= len(lines)):
        return f"{filename}:{line}:{col}: {level}: {msg}"

    src_line = lines[line - 1]
    ln_str = str(line)
    pad = " " * len(ln_str)
    caret_pad = " " * max(0, col - 1)
    caret = "^" * max(1, span_len)

    arrow = _wrap("-->", "blue", "bold", color=color)
    pipe = _wrap("|", "blue", "bold", color=color)
    caret_styled = _wrap(caret, *level_color, color=color)
    ln_styled = _wrap(ln_str, "blue", "bold", color=color)

    lines_out = [
        head,
        f"{pad} {arrow} {filename}:{line}:{col}",
        f"{pad}  {pipe}",
        f"{ln_styled} {pipe} {src_line}",
        f"{pad}  {pipe} {caret_pad}{caret_styled}",
    ]
    if hint:
        eq = _wrap("=", "blue", "bold", color=color)
        h_label = _wrap("hint", "green", "bold", color=color)
        lines_out.append(f"{pad}  {eq} {h_label}: {hint}")
    return "\n".join(lines_out)


# ----------------------------------------------------------------------
# Diagnostic dataclass + sink (optional aggregation)
# ----------------------------------------------------------------------
@dataclass
class Diagnostic:
    """A single diagnostic message: location, severity, content."""
    filename: str
    line: int
    col: int
    msg: str
    level: str = "error"        # "error" | "warning" | "note"
    code: Optional[int] = None  # e.g. 24100 for provenance traps
    hint: Optional[str] = None
    span_len: int = 1

    def render(self, source: Optional[str], color: Optional[bool] = None) -> str:
        return render_caret(
            filename=self.filename,
            line=self.line,
            col=self.col,
            msg=self.msg,
            source=source,
            hint=self.hint,
            level=self.level,
            code=self.code,
            color=color,
            span_len=self.span_len,
        )


@dataclass
class DiagSink:
    """Collector for diagnostics from a single pass over a single file.
    Lets a typechecker or linter accumulate multiple errors before
    handing the list back to the CLI."""
    diags: list[Diagnostic] = field(default_factory=list)

    def add(self, d: Diagnostic) -> None:
        self.diags.append(d)

    def error(self, filename: str, line: int, col: int, msg: str,
              **kw) -> None:
        self.add(Diagnostic(filename=filename, line=line, col=col,
                            msg=msg, level="error", **kw))

    def warning(self, filename: str, line: int, col: int, msg: str,
                **kw) -> None:
        self.add(Diagnostic(filename=filename, line=line, col=col,
                            msg=msg, level="warning", **kw))

    def __bool__(self) -> bool:
        return bool(self.diags)

    def __len__(self) -> int:
        return len(self.diags)

    def has_errors(self) -> bool:
        return any(d.level == "error" for d in self.diags)

    def render_all(self, source: Optional[str], color: Optional[bool] = None,
                   limit: Optional[int] = None) -> str:
        diags = self.diags if limit is None else self.diags[:limit]
        return "\n\n".join(d.render(source, color=color) for d in diags)
