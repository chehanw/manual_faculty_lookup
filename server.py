"""
Flask SSE server for manual faculty lookup.

Usage:
    python3 server.py
    # then open http://localhost:5050
"""
import json
import os
import queue
import threading

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

load_dotenv()

import faculty_search  # noqa: E402 — must come after load_dotenv

app = Flask(__name__)

_run_lock = threading.Lock()
_run_queue: queue.Queue | None = None
_pipeline_running = False


def _make_sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


_SITE_PASSWORD = os.environ.get("SITE_PASSWORD")


def _check_auth(req) -> bool:
    if not _SITE_PASSWORD:
        return True
    auth = req.authorization
    return bool(auth and auth.password == _SITE_PASSWORD)


@app.before_request
def require_auth():
    if not _check_auth(request):
        return Response(
            "Access denied.",
            401,
            {"WWW-Authenticate": 'Basic realm="Faculty Lookup"'},
        )


@app.route("/")
def index():
    with open("medfaculty.html", "r") as f:
        html = f.read()
    # Always inject MANUAL_ONLY — this deployment only does single-person lookup
    html = html.replace("<head>", "<head><script>window.MANUAL_ONLY=true;</script>", 1)
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.route("/api/run-names", methods=["POST"])
def start_names_run():
    global _run_queue, _pipeline_running

    with _run_lock:
        if _pipeline_running:
            return jsonify({"error": "Pipeline already running"}), 409
        _pipeline_running = True

    data           = request.json or {}
    names          = data.get("names") or []
    discipline     = data.get("discipline") or ""
    considerations = data.get("considerations") or ""

    records = [
        {
            "full_name":    n.get("full_name", "").strip(),
            "title":        None,
            "subspecialty": None,
            "email":        None,
            "profile_url":  None,
            "school":       n.get("institution", "").strip(),
        }
        for n in names
        if n.get("full_name", "").strip() and n.get("institution", "").strip()
    ]

    if not records:
        _pipeline_running = False
        return jsonify({"error": "No valid name + institution pairs provided"}), 400

    _run_queue = queue.Queue()

    def on_enriched(faculty_record: dict, completed: int, total: int):
        _run_queue.put({
            "type":      "faculty",
            "data":      faculty_record,
            "completed": completed,
            "total":     total,
        })

    def on_qc_complete(faculty_record: dict):
        _run_queue.put({
            "type": "qc_update",
            "data": faculty_record,
        })

    def run():
        global _pipeline_running
        try:
            faculty_search.run_pipeline_from_records(
                records,
                discipline=discipline,
                on_enriched=on_enriched,
                on_qc_complete=on_qc_complete,
                considerations=considerations,
            )
        except Exception as exc:
            _run_queue.put({"type": "error", "message": str(exc)})
        finally:
            _run_queue.put({"type": "done"})
            _pipeline_running = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/stream-names")
def stream_names():
    q = _run_queue

    def generate():
        if q is None:
            yield _make_sse({"type": "error", "message": "No pipeline running"})
            return
        while True:
            try:
                msg = q.get(timeout=30)
                yield _make_sse(msg)
                if msg["type"] in ("done", "error"):
                    break
            except queue.Empty:
                yield _make_sse({"type": "ping"})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"[Server] Starting at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
