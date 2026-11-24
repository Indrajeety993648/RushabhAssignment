"""Lightweight stdlib HTTP server (no framework, no Streamlit — spec item 8).

Routes:
  GET  /                     -> UI
  GET  /api/search?q=        -> company name autocomplete over the 500
  POST /api/find {company}   -> start a run, returns {run_id}
  GET  /api/run?id=          -> progress + results (poll)
"""
import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import db
import pipeline

HERE = os.path.dirname(__file__)


def _ensure_directory(con):
    if con.execute("SELECT COUNT(*) FROM companies").fetchone()[0] < 500:
        xlsx = os.path.join(HERE, "500 Companies list(1).xlsx")
        if os.path.exists(xlsx):
            db.load_directory(con, xlsx)


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        con = db.connect()
        if parsed.path == "/":
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/search":
            self._json(200, db.search_companies(con, qs.get("q", [""])[0]))
        elif parsed.path == "/api/run":
            res = pipeline.results(con, qs.get("id", [""])[0])
            self._json(200 if res else 404, res or {"error": "unknown run"})
        else:
            self._json(404, {"error": "not found"})
        con.close()

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/find":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        company = (payload.get("company") or "").strip()
        if not company:
            return self._json(400, {"error": "company required"})
        con = db.connect()
        if not db.get_company(con, company):
            con.close()
            return self._json(404, {"error": f'"{company}" is not in the 500-company directory'})
        import uuid
        run_id = uuid.uuid4().hex[:12]
        con.close()

        def _worker():
            c = db.connect()
            try:
                pipeline.run(c, company, run_id)
            except Exception as e:
                c.execute("UPDATE runs SET status='error', blocker=? WHERE id=?",
                          (f"internal error: {e}", run_id))
                c.commit()
            finally:
                c.close()
        threading.Thread(target=_worker, daemon=True).start()
        self._json(200, {"run_id": run_id})


def main():
    con = db.connect()
    _ensure_directory(con)
    con.close()
    port = int(os.environ.get("PORT", 8000))
    print(f"peer-finder on http://localhost:{port}  (LLM provider: "
          f"{'enabled' if __import__('providers').llm_available() else 'offline seed only'})")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
