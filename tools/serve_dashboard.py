#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/serve_dashboard.py
#  Serve the single-file paper dashboard on localhost, live.
#
#  Why a server at all, when dashboard.html is self-contained? Because the page
#  is generated from the journal, and the journal is append-only and grows. The
#  static file is a snapshot; this server re-reads the journal on EVERY request
#  and exposes /api/state, which the page's inlined script polls — so the
#  equity curve, the positions table and the regime table update in place.
#
#  Endpoints
#    GET /                 the dashboard (re-rendered per request)
#    GET /api/state        the same numbers as JSON (what the poller reads)
#    GET /journal.csv      the raw journal, for anyone who wants it
#    GET /healthz          liveness + which files are being read
#
#  Binds to 127.0.0.1 by default. This is a read-only view of a paper book;
#  it is not meant to be reachable from anywhere but this machine.
#
#  Usage:
#    python tools/serve_dashboard.py
#    python tools/serve_dashboard.py --port 9000 --refresh-ms 5000 --open
# =============================================================================
import argparse
import datetime as dt
import http.server
import json
import os
import sys
import urllib.parse
import webbrowser

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import paper_dashboard as pd  # noqa: E402
import paper_report as pr  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_JOURNAL = "state/paper_journal.csv"
DEFAULT_STATUS = "state/paper_status.json"

HTML_TYPE = "text/html; charset=utf-8"
JSON_TYPE = "application/json; charset=utf-8"
CSV_TYPE = "text/csv; charset=utf-8"
TEXT_TYPE = "text/plain; charset=utf-8"


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """One handler class, parameterised by class attributes (set in serve())."""

    server_version = "OmniSeedDashboard/1.0"
    protocol_version = "HTTP/1.1"

    journal = DEFAULT_JOURNAL
    status = DEFAULT_STATUS
    refresh_ms = pd.DEFAULT_REFRESH_MS
    quiet = False

    # -- plumbing ----------------------------------------------------------
    def log_message(self, fmt, *args):
        if not self.quiet:
            sys.stderr.write("[dashboard] %s - %s\n"
                             % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _analysis(self):
        return pr.analyze(self.journal, self.status)

    # -- routes ------------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path in ("/", "/index.html", "/dashboard.html"):
                a = self._analysis()
                return self._send(200, pd.render_html(a, refresh_ms=self.refresh_ms),
                                  HTML_TYPE)

            if path in ("/api/state", "/state.json"):
                return self._send(200, pd.render_state_json(self._analysis()),
                                  JSON_TYPE)

            if path in ("/healthz", "/api/health"):
                a = self._analysis()
                return self._send(200, json.dumps({
                    "ok": True,
                    "journal": os.path.abspath(self.journal),
                    "journal_exists": os.path.exists(self.journal),
                    "status": os.path.abspath(self.status),
                    "status_exists": os.path.exists(self.status),
                    "records": a["records"],
                    "equity": a["last_equity"],
                    "refresh_ms": self.refresh_ms,
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                }, sort_keys=True), JSON_TYPE)

            if path == "/journal.csv":
                if not os.path.exists(self.journal):
                    return self._send(404, f"no journal at {self.journal}\n",
                                      TEXT_TYPE)
                with open(self.journal, "rb") as f:
                    return self._send(200, f.read(), CSV_TYPE)

            return self._send(404, "not found\n", TEXT_TYPE)
        except BrokenPipeError:
            pass
        except Exception as e:                      # noqa: BLE001
            # A dashboard that dies on a malformed journal is worse than one
            # that reports the problem.
            self._send(500, f"dashboard error: {e}\n", TEXT_TYPE)


def build_server(host=DEFAULT_HOST, port=DEFAULT_PORT,
                 journal=DEFAULT_JOURNAL, status=DEFAULT_STATUS,
                 refresh_ms=pd.DEFAULT_REFRESH_MS, quiet=False):
    """-> (httpd, handler_class). Split out so tests can drive it without
    binding a socket."""
    handler = type("BoundDashboardHandler", (DashboardHandler,), {
        "journal": journal, "status": status,
        "refresh_ms": refresh_ms, "quiet": quiet,
    })
    httpd = http.server.ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd, handler


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Serve the OmniSeed paper dashboard on localhost")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--journal", default=DEFAULT_JOURNAL)
    ap.add_argument("--status", default=DEFAULT_STATUS)
    ap.add_argument("--refresh-ms", type=int, default=pd.DEFAULT_REFRESH_MS)
    ap.add_argument("--open", action="store_true",
                    help="open the dashboard in the default browser")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[dashboard] WARNING: binding {a.host} exposes a read-only view "
              "of your paper book to the network. 127.0.0.1 is the default "
              "for a reason.", file=sys.stderr)

    try:
        httpd, _ = build_server(a.host, a.port, a.journal, a.status,
                                a.refresh_ms, a.quiet)
    except OSError as e:
        print(f"[dashboard] cannot bind {a.host}:{a.port}: {e}", file=sys.stderr)
        return 2

    url = f"http://{a.host}:{a.port}/"
    a0 = pr.analyze(a.journal, a.status)
    print(f"[dashboard] {url}")
    print(f"[dashboard] journal {os.path.abspath(a.journal)} "
          f"({a0['records']} records, equity {pr.fmt_money(a0['last_equity'])})")
    print(f"[dashboard] live poll every {a.refresh_ms} ms · Ctrl-C to stop")
    if not a0["records"]:
        print("[dashboard] NOTE: the journal is empty — run tools/paper_loop.py "
              "first, or the page will show an empty book.")

    if a.open:
        try:
            webbrowser.open(url)
        except Exception:                           # noqa: BLE001
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
