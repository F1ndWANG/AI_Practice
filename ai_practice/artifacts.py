"""Record and verify file contents for completed experiment stages."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def artifact_index(root, paths):
    root = root.resolve()
    return {
        (root / path).resolve().relative_to(root).as_posix(): sha256(root / path)
        for path in paths
    }


def verify_index(root, index):
    if not isinstance(index, dict) or not index:
        raise ValueError("Empty or invalid artifact index")
    root = root.resolve()
    for name, digest in index.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or changed artifact: {name}")
