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
HX_FILE = os.path.join(EXAMPLES, "dashboard_agent.hx")
BIN_FILE = os.path.join(PROJ, "_dashboard.bin")


def compile_helix():
    """Compile dashboard_agent.hx -> _dashboard.bin."""
    cmd = [
        sys.executable, "-m", "helixc.backend.x86_64",
        "helixc/examples/dashboard_agent.hx",
        "_dashboard.bin",
    ]
    proc = subprocess.run(cmd, cwd=PROJ, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc.stderr
    return os.path.join(PROJ, "_dashboard.bin"), None


def run_helix():
    """Run the compiled binary via WSL, capture stdout."""
    # Convert Windows path to WSL path.
    wsl_path = "/mnt/c/Projects/Kovostov-Native/_dashboard.bin"
    cmd = [
        "wsl", "--", "bash", "-c",
        f"chmod +x {wsl_path} && {wsl_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
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
            sys.stderr.write("Compiling helix agent...\n")
            bin_path, err = compile_helix()
            if bin_path is None:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(("compile error:\n" + (err or "")).encode())
                return
            sys.stderr.write("Running helix agent via WSL...\n")
            try:
                out = run_helix()
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
