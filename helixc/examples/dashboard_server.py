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


def compile_helix(kind, seed=None):
    """Compile the chosen agent .hx -> ELF binary.

    If seed is given (int), substitute it into the qlearn agent's
    map_seed() function so each run uses a different random map.
    """
    if kind not in AGENTS:
        return None, f"unknown agent kind: {kind}"
    hx, bin_name = AGENTS[kind]
    src_path = os.path.join(EXAMPLES, hx)
    compile_path = src_path
    if seed is not None and kind in ("qlearn", "nn"):
        # Read the qlearn source, substitute the seed, write a tmp file.
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        new_src = src.replace(
            "@pure fn map_seed() -> i32 { 12345 }",
            f"@pure fn map_seed() -> i32 {{ {int(seed)} }}",
        )
        compile_path = os.path.join(EXAMPLES, f"_{kind}_compiled.hx")
        with open(compile_path, "w", encoding="utf-8") as f:
            f.write(new_src)
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
    """Run the compiled binary via WSL, capture stdout."""
    _, bin_name = AGENTS[kind]
    wsl_path = f"/mnt/c/Projects/Kovostov-Native/{bin_name}"
    cmd = [
        "wsl", "--", "bash", "-c",
        f"chmod +x {wsl_path} && {wsl_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    return proc.stdout.decode("utf-8", errors="replace")


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Quieter logs: print briefly to stderr.
        sys.stderr.write(f"[{self.address_string()}] {fmt % args}\n")

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
            kind = qs.get("kind", ["hillclimb"])[0]
            seed = qs.get("seed", [None])[0]
            try:
                seed_i = int(seed) if seed else None
            except ValueError:
                seed_i = None
            sys.stderr.write(f"Compiling helix agent ({kind}, seed={seed_i})...\n")
            bin_path, err = compile_helix(kind, seed=seed_i)
            if bin_path is None:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(("compile error:\n" + (err or "")).encode())
                return
            sys.stderr.write("Running helix agent via WSL...\n")
            try:
                out = run_helix(kind)
            except subprocess.TimeoutExpired:
                self.send_response(504)
                self.end_headers()
                self.wfile.write(b"helix run timed out")
                return
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
