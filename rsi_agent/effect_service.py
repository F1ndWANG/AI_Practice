"""Local reference service with atomic, idempotent counter updates and queryable receipts."""

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def initialize(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER);
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY, request_sha256 TEXT, receipt TEXT);
        """)


def apply(path, operation, raw):
    body = json.loads(raw)
    if (not isinstance(body, dict) or set(body) != {"name", "amount"}
            or not isinstance(body["name"], str) or not body["name"]
            or type(body["amount"]) is not int or not -(2**53) < body["amount"] < 2**53):
        raise ValueError("Payload requires a nonempty name and an integer amount")
    digest = hashlib.sha256(raw).hexdigest()
    with closing(sqlite3.connect(path, timeout=30)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute("SELECT request_sha256,receipt FROM operations WHERE id=?",
                                     (operation,)).fetchone()
            if old:
                if old[0] != digest:
                    raise ValueError("Idempotency key already binds a different request")
                return json.loads(old[1])
            row = connection.execute("SELECT value FROM counters WHERE name=?", (body["name"],)).fetchone()
            value = (row[0] if row else 0) + body["amount"]
            if not -(2**53) < value < 2**53:
                raise ValueError("Counter value exceeds supported range")
            connection.execute(
                "INSERT INTO counters VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                (body["name"], value))
            receipt = {"idempotency_key": operation, "request_sha256": digest,
                       "status": "succeeded", "result": {"name": body["name"], "value": value}}
            connection.execute("INSERT INTO operations VALUES (?,?,?)",
                               (operation, digest, json.dumps(receipt)))
            return receipt


def handler(database):
    class Handler(BaseHTTPRequestHandler):
        def operation(self):
            operation = self.path[len("/operations/"):]
            if (not self.path.startswith("/operations/") or len(operation) != 64
                    or any(character not in "0123456789abcdef" for character in operation)):
                raise ValueError("Use an operations path with a SHA-256 operation key")
            return operation

        def reply(self, status, value):
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            try:
                operation = self.operation()
                with closing(sqlite3.connect(database)) as connection:
                    row = connection.execute("SELECT receipt FROM operations WHERE id=?",
                                             (operation,)).fetchone()
                self.reply(200 if row else 404, json.loads(row[0]) if row else {"error": "absent"})
            except ValueError as error:
                self.reply(400, {"error": str(error)})

        def do_PUT(self):
            try:
                self.connection.settimeout(5)
                operation = self.operation()
                size = int(self.headers.get("Content-Length", "-1"))
                if not 0 <= size <= 1048576:
                    raise ValueError("A request body up to 1 MiB is required")
                raw = self.rfile.read(size)
                if (len(raw) != size or self.headers.get("Idempotency-Key") != operation
                        or self.headers.get("X-Request-SHA256") != hashlib.sha256(raw).hexdigest()):
                    raise ValueError("Request key or digest does not match")
                self.reply(200, apply(database, operation, raw))
            except (ValueError, OverflowError) as error:
                self.reply(409, {"error": str(error)})

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("--port", type=int, default=8085)
    args = parser.parse_args()
    initialize(args.database)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.database))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
