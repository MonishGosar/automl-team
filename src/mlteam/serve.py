"""Serve the approved model over HTTP.

    mlteam serve latest --port 8000

    POST /predict   {"rows": [{"col": value, ...}, ...]}   -> predictions, in the order sent
    GET  /model     what is being served: model, version, metric, scores, features, cut-off
    GET  /health    {"status": "ok", ...}

Every request is scored exactly like a file handed to ``mlteam predict``: the
approved contract version, the approved features, the approved model with its
cut-off, rescaling and range. Each request is also recorded as a batch, with its
drift check, so live traffic shows up in ``mlteam monitor``.

The model card is read on every request, so ``mlteam promote`` takes effect
without a restart. Standard library only; it binds to 127.0.0.1 unless told
otherwise. There is no authentication: put it behind something that has some
before exposing it beyond your machine.
"""

from __future__ import annotations

import csv
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from mlteam.predict import PredictError, predict_file
from mlteam.runstore import RunPaths

MAX_BODY = 10 * 1024 * 1024
MAX_ROWS = 10_000


class ServeError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _card(run: RunPaths) -> dict[str, Any]:
    if not run.model_latest.exists():
        raise ServeError("This run has no approved model yet. Approve one first.", 503)
    return json.loads(run.model_latest.read_text(encoding="utf-8"))


def model_info(run: RunPaths) -> dict[str, Any]:
    card = _card(run)
    delivery = card.get("delivery") or {}
    return {
        "run": run.run_id,
        "model": card.get("model"),
        "version": card.get("version"),
        "family": card.get("family"),
        "task": card.get("task"),
        "classes": card.get("classes"),
        "metric": card.get("metric"),
        "test_score": card.get("test_score"),
        "cv_mean": card.get("cv_mean"),
        "baseline_test_score": card.get("baseline_test_score"),
        "features": card.get("features"),
        "cutoff": delivery.get("cutoff"),
        "range_half_width": delivery.get("half_width"),
        "approved_by": card.get("approved_by"),
        "approved_at": card.get("approved_at"),
    }


def _rows_from(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise ServeError('Send {"rows": [{...}, ...]}: a list of rows, each an object of column: value.')
    if len(rows) > MAX_ROWS:
        raise ServeError(f"At most {MAX_ROWS:,} rows per request; this one has {len(rows):,}.")
    if not all(isinstance(r, dict) for r in rows):
        raise ServeError("Every row must be an object of column: value.")
    return rows


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        raise ServeError("Values must be plain numbers, text or true/false, not nested objects.")
    return value


def predict_rows(run: RunPaths, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score rows sent as JSON and return the predictions in the order they came."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with tempfile.TemporaryDirectory(prefix="mlteam_serve_") as folder:
        path = Path(folder) / "request.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([_cell(row.get(c)) for c in columns])
        try:
            prediction = predict_file(run, path)
        except PredictError as exc:
            raise ServeError(str(exc)) from exc
    output = run.root / prediction.output
    with output.open(encoding="utf-8", newline="") as handle:
        scored = list(csv.DictReader(handle))
    scored = _in_request_order(scored, rows, columns)
    for item in scored:
        for key, value in list(item.items()):
            item[key] = _number(value)
    return {
        "model": prediction.model,
        "version": prediction.model_version,
        "rows": prediction.rows,
        "predictions": scored,
        "health": (prediction.health or {}).get("status"),
        "notes": prediction.notes,
        "batch": prediction.output,
    }


def _in_request_order(scored: list[dict[str, str]], rows: list[dict[str, Any]], columns: list[str]) -> list[dict]:
    """Scoring sorts by the row key when there is one. Put the answers back in the order sent."""
    if not scored:
        return scored
    keys = [k for k in scored[0] if k in columns]
    if not keys:
        return scored  # no key: the file's own order is kept
    position: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        position.setdefault(tuple(str(_cell(row.get(k))) for k in keys), []).append(i)
    ordered: list[tuple[int, dict]] = []
    for j, item in enumerate(scored):
        slots = position.get(tuple(str(item.get(k, "")) for k in keys))
        ordered.append((slots.pop(0) if slots else len(rows) + j, item))
    return [item for _, item in sorted(ordered, key=lambda pair: pair[0])]


def _number(value: str) -> Any:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value and "e" not in value.lower() else number


def make_handler(run: RunPaths, lock: threading.Lock) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "mlteam"

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - the http.server naming
            try:
                if self.path.rstrip("/") == "/health":
                    info = model_info(run)
                    self._send(200, {"status": "ok", "model": info["model"], "version": info["version"]})
                elif self.path.rstrip("/") == "/model":
                    self._send(200, model_info(run))
                else:
                    self._send(404, {"error": "Not found. Try GET /health, GET /model or POST /predict."})
            except ServeError as exc:
                self._send(exc.status, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/predict":
                self._send(404, {"error": "Not found. POST to /predict."})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    raise ServeError("Empty request.")
                if length > MAX_BODY:
                    raise ServeError(f"Request too large (over {MAX_BODY // (1024 * 1024)} MB).", 413)
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ServeError(f"The body is not valid JSON: {exc}") from exc
                rows = _rows_from(payload)
                with lock:  # one scoring at a time: batch numbers and the database are shared
                    result = predict_rows(run, rows)
                self._send(200, result)
            except ServeError as exc:
                self._send(exc.status, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - never drop the connection without an answer
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            if getattr(self.server, "quiet", False):
                return
            super().log_message(format, *args)

    return Handler


def make_server(run: RunPaths, host: str = "127.0.0.1", port: int = 8000, quiet: bool = False) -> ThreadingHTTPServer:
    _card(run)  # refuse to start without an approved model
    server = ThreadingHTTPServer((host, port), make_handler(run, threading.Lock()))
    server.quiet = quiet  # type: ignore[attr-defined]
    return server
