# dashboard.py -- serveur HTTP local pour WishAI
# Sert dashboard.html, library.html ET expose une API pour les telechargements
# python src/dashboard.py   (ou simplement : python go.py)

import os, sys, socket, threading, time, webbrowser, json, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(ROOT, "cache", "pycache")

from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# --- Garde en memoire les telechargements en cours ---
_downloads_en_cours = {}

# --- Etat du catalogue automatique ---
_catalogue_status = {"state": "idle", "count": 0, "ts": 0}


class WishAIHandler(SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # Silencieux

    # -- Routing GET ---------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        if path == "/api/ping":
            self._json({"ok": True, "version": "wishai-2"})
        elif path == "/api/downloads":
            self._json(_downloads_en_cours)
        elif path == "/api/catalogue/status":
            self._json(_catalogue_status)
        elif path == "/api/pwc":
            self._handle_pwc(parsed.query)
        else:
            super().do_GET()

    # -- Routing POST --------------------------------------------------------
    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/download":
            self._handle_download(payload)
        elif path == "/api/catalogue":
            self._handle_catalogue(payload)
        else:
            self._error(404, "Route inconnue")

    # -- Handler telechargement ----------------------------------------------
    def _handle_download(self, payload):
        dataset_id = payload.get("dataset_id", "").strip()
        mo         = int(payload.get("mo", 200))
        is_raw_hf  = payload.get("is_raw_hf", False)
        lang       = payload.get("lang", "multi")

        if not dataset_id:
            self._error(400, "dataset_id manquant")
            return

        if dataset_id in _downloads_en_cours and _downloads_en_cours[dataset_id] == "running":
            self._json({"status": "already_running"}, 200)
            return

        _downloads_en_cours[dataset_id] = "running"

        def run_dl():
            try:
                cmd = [
                    sys.executable,
                    os.path.join(SRC, "telecharger.py"),
                    "--download", dataset_id,
                    "--mo", str(mo),
                    "--combine",
                ]
                if is_raw_hf:
                    cmd.extend(["--is_raw_hf", "--lang", lang])

                proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
                _downloads_en_cours[dataset_id] = "done" if proc.returncode == 0 else ("error:" + str(proc.returncode))
            except Exception as e:
                _downloads_en_cours[dataset_id] = "error:" + str(e)

        threading.Thread(target=run_dl, daemon=True).start()
        self._json({"status": "started", "dataset_id": dataset_id, "mo": mo})

    # -- Proxy Papers with Code (contourne le blocage CORS) ------------------
    def _handle_pwc(self, query_string):
        import urllib.request as _ur, urllib.parse as _up
        params = dict(p.split("=", 1) for p in query_string.split("&") if "=" in p)
        q      = _up.unquote_plus(params.get("q", ""))
        page   = params.get("page_size", "24")
        url    = "https://paperswithcode.com/api/v1/datasets/?q=" + _up.quote(q) + "&page_size=" + page
        try:
            req  = _ur.Request(url, headers={"User-Agent": "wishai/1.0"})
            with _ur.urlopen(req, timeout=10) as r:
                data = r.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._json({"results": [], "error": str(e)})

    # -- Handler catalogue automatique ---------------------------------------
    def _handle_catalogue(self, payload=None):
        global _catalogue_status

        if _catalogue_status["state"] == "running":
            self._json({"status": "already_running"})
            return

        limit = None
        if payload:
            try:
                limit = int(payload.get("limit") or 0) or None
            except (ValueError, TypeError):
                limit = None

        _catalogue_status = {"state": "running", "count": 0, "ts": time.time()}

        def run_cat():
            global _catalogue_status
            try:
                cmd = [sys.executable, os.path.join(SRC, "catalogue.py")]
                if limit:
                    cmd += ["--limit", str(limit)]
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT, capture_output=True, text=True,
                    encoding="utf-8", errors="replace"
                )
                # Cherche "  OK: N nouveaux datasets ajoutes"
                count = 0
                for line in proc.stdout.splitlines():
                    if "nouveaux datasets" in line:
                        for token in line.split():
                            try:
                                count = int(token)
                                break
                            except ValueError:
                                pass
                        break
                _catalogue_status = {
                    "state": "done" if proc.returncode == 0 else "error",
                    "count": count,
                    "ts": time.time()
                }
            except Exception as e:
                _catalogue_status = {"state": "error", "count": 0, "ts": time.time(), "err": str(e)}

        threading.Thread(target=run_cat, daemon=True).start()
        self._json({"status": "started"})

    # -- Helpers reponse -----------------------------------------------------
    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg):
        self._json({"error": msg}, code)


def _port_libre():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    os.chdir(ROOT)

    port = _port_libre()
    srv = HTTPServer(("localhost", port), WishAIHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    url     = "http://localhost:" + str(port) + "/dashboard.html"
    lib_url = "http://localhost:" + str(port) + "/library.html"
    print("\n  dashboard    -> " + url)
    print("  bibliotheque -> " + lib_url)
    print("  Ctrl+C pour quitter\n")

    webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    srv.shutdown()


if __name__ == "__main__":
    main()
