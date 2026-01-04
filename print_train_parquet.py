#!/usr/bin/env python3
"""Print contents of a parquet file with multiple fallbacks."""
from __future__ import annotations

import argparse
import os
import struct
import sys
from typing import List, Tuple


DEFAULT_PATH = "/mnt/hpfs/xiangc/mxy/verl-agent/video_data/train.parquet"


def _diagnose_parquet(path: str) -> None:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-8, os.SEEK_END)
            footer = f.read(8)
        meta_len = struct.unpack("<i", footer[:4])[0]
        meta_start = size - 8 - meta_len
        print("[diagnose] size:", size, file=sys.stderr)
        print("[diagnose] head magic:", head, file=sys.stderr)
        print("[diagnose] footer magic:", footer[4:], file=sys.stderr)
        print("[diagnose] meta_len:", meta_len, file=sys.stderr)
        print("[diagnose] meta_start:", meta_start, file=sys.stderr)
    except Exception as e:
        print(f"[diagnose] failed: {e!r}", file=sys.stderr)


def _print_df(df) -> None:
    import pandas as pd

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        0,
        "display.max_colwidth",
        None,
    ):
        print(df)


def _read_with_pandas_pyarrow(path: str):
    import pandas as pd

    return pd.read_parquet(path, engine="pyarrow")


def _read_with_pandas_fastparquet(path: str):
    import pandas as pd

    return pd.read_parquet(path, engine="fastparquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_path", nargs="?", default=DEFAULT_PATH)
    args = ap.parse_args()

    path = args.parquet_path
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")

    readers: List[Tuple[str, callable]] = [
        ("pandas+pyarrow", _read_with_pandas_pyarrow),
        ("pandas+fastparquet", _read_with_pandas_fastparquet),
    ]

    errors = []
    for name, fn in readers:
        try:
            df = fn(path)
            _print_df(df)
            return
        except Exception as e:
            errors.append((name, e))

    print("[ERROR] All readers failed.", file=sys.stderr)
    for name, e in errors:
        print(f"  - {name}: {e!r}", file=sys.stderr)

    _diagnose_parquet(path)
    print(
        "If this file is valid Parquet, try installing fastparquet. "
        "Otherwise the file may be corrupted or mislabeled.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
