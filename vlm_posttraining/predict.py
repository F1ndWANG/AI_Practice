"""Run manifest inference with a resumable per-sample journal."""

import fcntl
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile

from ai_practice.artifacts import sha256

from .blind import fingerprint
from .inference import VisualModel
from .pipeline import read_jsonl


def prediction_inputs(manifest):
    seen = set()
    images = {}
    rows = []
    for item in read_jsonl(manifest):
        sample_id = item["id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Invalid or repeated prediction ID: {sample_id}")
        question = item["question"]
        evidence = item.get("evidence_observed", True)
        if not isinstance(question, str) or not question.strip() or type(evidence) is not bool:
            raise ValueError(f"Invalid prediction prompt: {sample_id}")
        path = (manifest.parent / item["image"]).resolve()
        if path not in images:
            images[path] = sha256(path)
        if "image_sha256" in item and item["image_sha256"] != images[path]:
            raise ValueError(f"Prediction image changed: {path}")
        rows.append({"id": sample_id, "image": str(path), "image_sha256": images[path],
                     "question": question, "evidence_observed": evidence})
        seen.add(sample_id)
    if not rows:
        raise ValueError("Prediction manifest is empty")
    return rows


def publish(output, rows):
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def predict(manifest, output, model, adapter=None, device="cuda", resume=False):
    if output.resolve() == manifest.resolve():
        raise ValueError("Prediction manifest and output must be distinct")
    rows = prediction_inputs(manifest)
    if not resume:
        visual = VisualModel(model, adapter, device)
        output.parent.mkdir(parents=True, exist_ok=True)
        publish(output, (predict_row(visual, row) for row in rows))
        return

    adapter_files = [adapter / name for name in ("adapter_model.safetensors", "adapter_config.json")] if adapter else []
    adapter_digests = {path.name: sha256(path) for path in adapter_files}
    spec = {"model": model, "adapter_digests": adapter_digests, "device": device,
            "input_sha256": fingerprint(rows), "schema": 1}
    checkpoint = output.with_name(f".{output.name}.prediction.sqlite")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(f".{output.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is predicting into this output") from error
        with closing(sqlite3.connect(checkpoint)) as database:
            cache_predictions(database, rows, spec, output, model, adapter, device, adapter_files)


def predict_row(visual, row):
    result = visual.ask(row["image"], row["question"], row["evidence_observed"])
    return {**result, "id": row["id"]}


def cache_predictions(database, rows, spec, output, model, adapter, device, adapter_files):
    encoded_spec = json.dumps(spec, sort_keys=True)
    with database:
        database.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        database.execute("CREATE TABLE IF NOT EXISTS predictions (sample_id TEXT PRIMARY KEY, value TEXT NOT NULL)")
        entry = database.execute("SELECT value FROM settings WHERE id=1").fetchone()
        if entry is not None and entry[0] != encoded_spec:
            raise ValueError("Prediction inputs or model settings changed; choose another output")
        if entry is None:
            database.execute("INSERT INTO settings VALUES (1, ?)", (encoded_spec,))
    def adapter_changed():
        return any(sha256(path) != spec["adapter_digests"][path.name] for path in adapter_files)

    visual = None
    for row in rows:
        if database.execute("SELECT 1 FROM predictions WHERE sample_id=?", (row["id"],)).fetchone():
            continue
        if visual is None:
            visual = VisualModel(model, adapter, device)
            if adapter_changed():
                raise ValueError("Adapter changed while loading the model")
        result = predict_row(visual, row)
        with database:
            database.execute("INSERT INTO predictions VALUES (?, ?)",
                             (row["id"], json.dumps(result, ensure_ascii=False, allow_nan=False)))
    if adapter_changed():
        with database:
            database.execute("DELETE FROM predictions")
        raise ValueError("Adapter changed during prediction; cached predictions were discarded")

    def completed():
        for row in rows:
            (encoded,) = database.execute(
                "SELECT value FROM predictions WHERE sample_id=?", (row["id"],)
            ).fetchone()
            result = json.loads(encoded)
            if result["id"] != row["id"] or not isinstance(result["completion"], str):
                raise ValueError(f"Malformed cached prediction: {row['id']}")
            yield result

    publish(output, completed())
