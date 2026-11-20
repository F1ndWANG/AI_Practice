"""Durable HTTP writes against an explicitly idempotent service contract."""

import hashlib
import json
import sys
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from .live_verifier import NoRedirect, origin
from .execution_control import json_process


class GatewayError(RuntimeError):
    pass


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def response_json(opener, request, timeout, max_bytes):
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("External response exceeds byte budget")
        return json.loads(raw)


def execute(runtime, task_id, index, request, limits, check=None):
    url = request["url"].rstrip("/")
    if origin(url) not in {origin(value) for value in runtime.permissions["origins"]}:
        raise ValueError("External service origin is not permitted")
    body = encode(request["payload"])
    if len(body) > limits["file_bytes"]:
        raise ValueError("External request exceeds byte budget")
    digest = hashlib.sha256(body).hexdigest()
    operation = hashlib.sha256(f"{task_id}:{index}".encode()).hexdigest()
    endpoint = url + "/" + operation
    runtime.db.execute("""CREATE TABLE IF NOT EXISTS external_effects (
        task_id TEXT, step INTEGER, operation TEXT UNIQUE, request_sha256 TEXT,
        state TEXT, receipt TEXT, PRIMARY KEY(task_id,step))""")
    runtime.db.commit()
    with runtime.db:
        old = runtime.db.execute(
            "SELECT * FROM external_effects WHERE task_id=? AND step=?", (task_id, index)).fetchone()
        if old and old["request_sha256"] != digest:
            raise ValueError("External operation payload changed")
        if old and old["state"] == "confirmed":
            return json.loads(old["receipt"])
        if old is None:
            runtime.db.execute("INSERT INTO external_effects VALUES (?,?,?,?,'intent',NULL)",
                               (task_id, index, operation, digest))
            runtime.event(task_id, "external_intent", {"step": index, "operation": operation,
                                                      "request_sha256": digest})
    try:
        payload = {"url": endpoint, "body": body.decode(), "operation": operation,
                   "digest": digest, "timeout": limits["timeout_seconds"],
                   "max_bytes": limits["file_bytes"]}
        try:
            receipt = json_process("rsi_agent.external_effects", payload,
                                   limits["timeout_seconds"], check=check)
        except RuntimeError as error:
            raise GatewayError(str(error)) from error
        if (not isinstance(receipt, dict) or receipt.get("idempotency_key") != operation
                or receipt.get("request_sha256") != digest or receipt.get("status") != "succeeded"):
            raise ValueError("External operation has no matching, successful service receipt")
    except Exception:
        with runtime.db:
            runtime.db.execute("UPDATE external_effects SET state='uncertain' WHERE task_id=? AND step=?",
                               (task_id, index))
            runtime.event(task_id, "external_uncertain", {"step": index, "operation": operation})
        raise
    with runtime.db:
        runtime.db.execute(
            "UPDATE external_effects SET state='confirmed',receipt=? WHERE task_id=? AND step=?",
            (encode(receipt).decode(), task_id, index))
        runtime.event(task_id, "external_confirmed", {"step": index, "operation": operation})
    return receipt



def main():
    request = json.load(sys.stdin)
    opener = build_opener(ProxyHandler({}), NoRedirect())
    headers = {"Idempotency-Key": request["operation"], "X-Request-SHA256": request["digest"],
               "Content-Type": "application/json"}
    try:
        receipt = response_json(opener, Request(request["url"], headers=headers),
                                request["timeout"], request["max_bytes"])
    except HTTPError as error:
        if error.code != 404:
            raise
        error.close()
        receipt = response_json(
            opener, Request(request["url"], data=request["body"].encode(), headers=headers, method="PUT"),
            request["timeout"], request["max_bytes"])
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
