"""
helixc/examples/dashboard_server.py

Tiny HTTP server that:
  - Serves dashboard.html at / (the JS dashboard)
  - On GET /run: compiles dashboard_agent.hx via helixc, runs the binary
    via WSL, captures the JSON-per-line stdout, and returns it as the
    response body.

The browser dashboard streams those events back at user-controlled
playback speed.

Usage:
  cd C:\\Projects\\Kovostov-Native
  python helixc\\examples\\dashboard_server.py
  # Then open http://localhost:8765/ in a browser.

License: Apache 2.0
"""

import http.server
import socketserver
import subprocess
import os
import sys
import urllib.parse

PORT = 8765
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(PROJ, "helixc", "examples")

AGENTS = {
    "hillclimb": ("dashboard_agent.hx", "_dashboard.bin"),
    "qlearn":    ("dashboard_qlearn.hx", "_qlearn.bin"),
    "nn":        ("dashboard_nn_agent.hx", "_nn.bin"),
}


def _rewrite_knobs(src, hx, kind, seed, maze, grid_size):
    """Apply the requested compile-time knobs to an agent's source.

    Each knob substitutes a `@pure fn` constant. A knob whose target
    function is absent from this agent's source is REJECTED rather than
    silently dropped (v2.x re-audit R3 RT-M1/M2): pre-fix `maze` for the
    `nn` agent (dashboard_nn_agent.hx has no use_maze()) and `size` for
    `nn` (the grid_n rewrite was gated `kind == "qlearn"` even though
    dashboard_nn_agent.hx HAS a grid_n()) no-op'd silently — the request
    was accepted and logged maze=True / size=N, but the binary compiled
    with the knob ignored.

    Returns (new_src, None) on success, or (None, error_message) when a
    requested knob is unsupported by this agent's source."""
    new_src = src
    if seed is not None:
        target = "@pure fn map_seed() -> i32 { 12345 }"
        if target not in new_src:
            return None, (f"agent kind {kind!r} does not support the "
                          f"'seed' knob (no map_seed() in {hx})")
        new_src = new_src.replace(
            target, f"@pure fn map_seed() -> i32 {{ {int(seed)} }}")
    if maze:
        target = "@pure fn use_maze() -> i32 { 0 }"
        if target not in new_src:
            return None, (f"agent kind {kind!r} does not support the "
                          f"'maze' knob (no use_maze() in {hx})")
        new_src = new_src.replace(
            target, "@pure fn use_maze() -> i32 { 1 }")
    if grid_size is not None:
        # v2.x re-audit R4 (RT HIGH-1): the 'size' knob is sound ONLY
        # for the qlearn agent — its grid_total() / goal_id() derive
        # from grid_n(). dashboard_nn_agent.hx hardcodes those
        # (grid_total -> 100, goal_id -> 99), so rewriting grid_n()
        # alone yields an INCONSISTENT binary (a resized world with a
        # stale grid_total / goal_id). The R3 fix checked only that
        # grid_n() is present — true for nn too — turning a silent
        # no-op into a silent miscompile. Restrict to qlearn; reject
        # loudly otherwise.
        if kind != "qlearn":
            return None, (f"agent kind {kind!r} does not support the "
                          f"'size' knob: only the qlearn agent derives "
                          f"its grid constants from grid_n()")
        target = "@pure fn grid_n() -> i32 { 10 }"
        if target not in new_src:
            return None, (f"agent kind {kind!r} does not support the "
                          f"'size' knob (no grid_n() in {hx})")
        new_src = new_src.replace(
            target, f"@pure fn grid_n() -> i32 {{ {int(grid_size)} }}")
    return new_src, None


def compile_helix(kind, seed=None, maze=False, grid_size=None):
    """Compile the chosen agent .hx -> ELF binary.

    seed: int substituted into map_seed() for reproducible random maps.
    maze: True flips use_maze() to 1, switching to wall-line layout.
    """
    if kind not in AGENTS:
        return None, f"unknown agent kind: {kind}"
    hx, bin_name = AGENTS[kind]
    src_path = os.path.join(EXAMPLES, hx)
    compile_path = src_path
    if seed is not None or maze or grid_size is not None:
        # v2.x re-audit R4 (RT MEDIUM-1): _rewrite_knobs is the single
        # authority — it rejects every knob unsupported by THIS agent.
        # The prior `and kind in ("qlearn","nn")` gate skipped it for
        # `hillclimb`, silently dropping a seed/maze knob there.
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        new_src, knob_err = _rewrite_knobs(
            src, hx, kind, seed, maze, grid_size)
        if knob_err is not None:
            return None, knob_err
        compile_path = os.path.join(EXAMPLES, f"_{kind}_compiled.hx")
        # Restart 47 B2: atomic-write so a Ctrl-C / OOM mid-write does not
        # leave a truncated source at compile_path for the next backend
        # invocation to consume. Mirrors examples/run.py (restart 46 B5) and
        # helixc.check._atomic_write_bytes.
        import tempfile
        directory = os.path.dirname(os.path.abspath(compile_path)) or "."
        base = os.path.basename(compile_path)
        tmp_path = ""
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{base}.",
                suffix=".tmp",
                dir=directory,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_src)
            os.replace(tmp_path, compile_path)
        except BaseException:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
    rel = os.path.relpath(compile_path, PROJ).replace("\\", "/")
    cmd = [
        sys.executable, "-m", "helixc.backend.x86_64",
        rel,
        bin_name,
    ]
    proc = subprocess.run(cmd, cwd=PROJ, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc.stderr
    return os.path.join(PROJ, bin_name), None


def run_helix(kind):
    """Run the compiled binary via WSL.

    v2.2 5-clean-gate RT MED-1 audit-fix: this is the parallel codepath
    to `examples/run.py::_build_and_run` which was fixed in item 8 to
    surface WSL stderr + propagate exit code. Pre-fix this function
    silently discarded `proc.stderr` and `proc.returncode` — a
    segfaulting binary (exit 139), WSL initialization error, or
    runtime panic returned an empty/partial NDJSON body with HTTP 200,
    rendering "no events" in the browser with no signal that the
    agent crashed.

    R1 fix: return `(stdout, stderr, returncode)`. The HTTP handler
    at `Handler.do_GET` consumes the tuple and surfaces non-zero
    returncodes via HTTP 500 with stderr in the body.

    Raises:
        subprocess.TimeoutExpired: WSL binary exceeded the 60s budget.
    """
    _, bin_name = AGENTS[kind]
    wsl_path = f"/mnt/c/Projects/Kovostov-Native/{bin_name}"
    cmd = [
        "wsl", "--", "bash", "-c",
        f"chmod +x {wsl_path} && {wsl_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    return (
        proc.stdout.decode("utf-8", errors="replace"),
        proc.stderr.decode("utf-8", errors="replace"),
        proc.returncode,
    )


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Quieter logs: print briefly to stderr.
        sys.stderr.write(f"[{self.address_string()}] {fmt % args}\n")

    def _send_400(self, msg: str) -> None:
        """v2.2 polish item 7: emit HTTP 400 with diagnostic body so
        malformed query-strings surface visibly to the client instead
        of silently coercing to defaults."""
        body = msg.encode("utf-8", errors="replace")
        self.send_response(400)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/dashboard.html":
            with open(os.path.join(EXAMPLES, "dashboard.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/run":
            qs = urllib.parse.parse_qs(parsed.query)
            # v2.2 polish item 7 (RT M1 from v2.1 5-clean-gate): the
            # prior parsing silently coerced typo'd query keys to
            # default values — `?kihd=foo` would run hillclimb, and
            # `?seed=abc` would run seed=None, with no signal to the
            # client that their request was malformed. R1 fix:
            # reject (a) unknown query keys, (b) invalid int-coercion
            # for `seed`/`size` with HTTP 400 + diagnostic body. The
            # default-when-absent semantics still hold (no key →
            # default value, present-but-malformed → 400).
            ALLOWED_KEYS = {"kind", "seed", "maze", "size"}
            unknown_keys = set(qs.keys()) - ALLOWED_KEYS
            if unknown_keys:
                self._send_400(
                    f"unknown query key(s): {sorted(unknown_keys)}. "
                    f"allowed: {sorted(ALLOWED_KEYS)}"
                )
                return
            kind = qs.get("kind", ["hillclimb"])[0]
            seed = qs.get("seed", [None])[0]
            if seed is not None and seed != "":
                try:
                    seed_i = int(seed)
                except ValueError:
                    self._send_400(
                        f"invalid `seed` value: {seed!r} (expected "
                        f"integer or absent)"
                    )
                    return
            else:
                seed_i = None
            maze = qs.get("maze", ["0"])[0] == "1"
            size_str = qs.get("size", ["10"])[0]
            try:
                grid_n = int(size_str)
            except ValueError:
                self._send_400(
                    f"invalid `size` value: {size_str!r} (expected "
                    f"integer)"
                )
                return
            grid_n = max(5, min(20, grid_n))
            sys.stderr.write(f"Compiling helix agent ({kind}, seed={seed_i}, maze={maze}, grid={grid_n})...\n")
            bin_path, err = compile_helix(kind, seed=seed_i, maze=maze, grid_size=grid_n)
            if bin_path is None:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(("compile error:\n" + (err or "")).encode())
                return
            sys.stderr.write("Running helix agent via WSL...\n")
            try:
                out, err_stream, rc = run_helix(kind)
            except subprocess.TimeoutExpired:
                self.send_response(504)
                self.end_headers()
                self.wfile.write(b"helix run timed out")
                return
            # v2.2 5-clean-gate RT MED-1 audit-fix: surface WSL failure
            # via HTTP 500 + stderr body instead of silently shipping
            # empty/partial NDJSON with HTTP 200. Mirrors examples/
            # run.py item 8 fix.
            if rc != 0:
                self.send_response(500)
                self.send_header("Content-Type",
                                 "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    f"helix exit {rc}\nstderr:\n{err_stream}".encode("utf-8")
                )
                return
            if err_stream:
                # Non-zero stderr on success path: log server-side, do
                # not block the response. (NDJSON-mode dashboards may
                # render this as a debug pane in the future.)
                sys.stderr.write(f"[helix-stderr] {err_stream}\n")
            body = out.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # Default: 404
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"not found")


def main():
    handler = Handler
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"Helix dashboard running at: http://localhost:{PORT}/")
        print("Open that URL in your browser, then click Run.")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down")


if __name__ == "__main__":
    main()
