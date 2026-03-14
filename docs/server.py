#!/usr/bin/env python3
"""
Development server for AI System Design Guide.

Serves the docs UI and proxies markdown content from the guide directory.

Usage:
    python server.py          # runs on port 8080
    python server.py 3000     # runs on port 3000
"""

import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

WEBAPP_DIR = Path(__file__).parent.resolve()
GUIDE_DIR  = WEBAPP_DIR.parent.resolve()

MIME = {
    '.html': 'text/html; charset=utf-8',
    '.css':  'text/css; charset=utf-8',
    '.js':   'application/javascript; charset=utf-8',
    '.md':   'text/plain; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.ico':  'image/x-icon',
    '.png':  'image/png',
    '.svg':  'image/svg+xml',
}


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        status = args[1] if len(args) > 1 else '?'
        color  = '\033[32m' if str(status).startswith('2') else '\033[31m'
        reset  = '\033[0m'
        print(f"  {color}{status}{reset}  {self.command} {args[0]}")

    def send_bytes(self, data: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url  = urlparse(self.path)
        path = unquote(url.path)

        # /content/<rel-path> → serve markdown file from guide root
        if path.startswith('/content/'):
            rel       = path[len('/content/'):]
            file_path = GUIDE_DIR / rel

            # Security: prevent directory traversal
            try:
                file_path.resolve().relative_to(GUIDE_DIR)
            except ValueError:
                self.send_error(403, 'Forbidden')
                return

            if file_path.is_file():
                self.send_bytes(file_path.read_bytes(), 'text/plain; charset=utf-8')
            else:
                self.send_error(404, f'Not found: {rel}')
            return

        # Serve docs static files
        if path == '/':
            path = '/index.html'

        file_path = WEBAPP_DIR / path.lstrip('/')

        if file_path.is_file():
            ct = MIME.get(file_path.suffix, 'application/octet-stream')
            self.send_bytes(file_path.read_bytes(), ct)
        else:
            # SPA fallback: serve index.html for any unknown route
            index = WEBAPP_DIR / 'index.html'
            self.send_bytes(index.read_bytes(), 'text/html; charset=utf-8')

    def do_HEAD(self):
        self.do_GET()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(('localhost', port), Handler)

    print(f"\n  \033[1m📚 AI Engineering Guide\033[0m")
    print(f"  Local:    \033[36mhttp://localhost:{port}\033[0m")
    print(f"  Guide:    {GUIDE_DIR}")
    print(f"  Webapp:   {WEBAPP_DIR}")
    print(f"\n  Press \033[1mCtrl+C\033[0m to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n  Server stopped.\n')


if __name__ == '__main__':
    main()
