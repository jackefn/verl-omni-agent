import argparse
import json
import sys

def _jsonify(x):
    """Make Arrow/Pandas objects JSON-serializable for pretty printing."""
    try:
        import pyarrow as pa
        if isinstance(x, (pa.Scalar, pa.Array)):
            return x.as_py()
    except Exception:
        pass

    # bytes -> try decode
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return {"__bytes_len__": len(x)}
    return x


def try_read_with_dataset(path: str):
    import pyarrow.dataset as ds
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(limit=1)
    if table.num_rows == 0:
        return None
    row = table.slice(0, 1).to_pylist()[0]  # list[dict] with one row
    return row


def try_read_with_parquet(path: str):
    import pyarrow.parquet as pq
    # read_table may succeed where ParquetFile() fails on some weird metadata edge cases
    table = pq.read_table(path, use_threads=False)
    if table.num_rows == 0:
        return None
    row = table.slice(0, 1).to_pylist()[0]
    return row


def try_read_with_pandas_fastparquet(path: str):
    import pandas as pd
    df = pd.read_parquet(path, engine="fastparquet")
    if len(df) == 0:
        return None
    # Convert first row to dict
    return df.iloc[0].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_path")
    ap.add_argument("--pretty", action="store_true", help="Pretty JSON output")
    args = ap.parse_args()

    last_err = None
    readers = [
        ("pyarrow.dataset", try_read_with_dataset),
        ("pyarrow.parquet.read_table", try_read_with_parquet),
        ("pandas+fastparquet", try_read_with_pandas_fastparquet),
    ]

    for name, fn in readers:
        try:
            row = fn(args.parquet_path)
            if row is None:
                print("Parquet has 0 rows.", file=sys.stderr)
                sys.exit(1)

            # Print all fields
            if args.pretty:
                print(json.dumps(row, ensure_ascii=False, indent=2, default=_jsonify))
            else:
                # one field per line, stable ordering
                for k in sorted(row.keys()):
                    v = row[k]
                    print(f"{k}: {json.dumps(_jsonify(v), ensure_ascii=False, default=_jsonify)}")
            return

        except Exception as e:
            last_err = (name, e)

    name, e = last_err
    print(f"[ERROR] All readers failed. Last failed reader: {name}", file=sys.stderr)
    print(repr(e), file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()