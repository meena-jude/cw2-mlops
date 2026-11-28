"""Write an md5 + row-count manifest for a data folder (stdlib only).

Usage: python3 scripts/make_manifest.py ~/cw2data/aiops-v1 --version 1 --out v1.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def build_manifest(folder: Path, name: str, version: str) -> dict:
    files = {}
    for p in sorted(folder.iterdir()):
        if p.is_file():
            entry = {"md5": md5(p), "bytes": p.stat().st_size}
            if p.suffix == ".csv":
                entry["rows"] = csv_rows(p)
            files[p.name] = entry
    return {
        "name": name,
        "version": version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--name", default="aiops-model-ready")
    p.add_argument("--version", default="1")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    text = json.dumps(build_manifest(args.folder, args.name, args.version), indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
