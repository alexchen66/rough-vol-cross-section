"""
One-time fix: convert timestamp[ns] date columns in all parquet files
to string (YYYY-MM-DD), avoiding the pandas timestamp segfault.

Run once, then all pd.read_parquet() calls will work normally.
"""
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent


def ts_ns_to_date_str_array(arr: pa.Array) -> pa.Array:
    """Convert a timestamp[ns] pyarrow array to string 'YYYY-MM-DD' array."""
    secs = arr.cast(pa.int64()).to_pylist()
    strings = [
        datetime.utcfromtimestamp(int(s) // 1_000_000_000).strftime('%Y-%m-%d')
        if s is not None else None
        for s in secs
    ]
    return pa.array(strings, type=pa.string())


def fix_file(path: Path) -> bool:
    """Return True if the file was modified."""
    t = pq.read_table(path)
    changed = False
    for i, field in enumerate(t.schema):
        if pa.types.is_timestamp(field.type):
            t = t.set_column(i, field.name, ts_ns_to_date_str_array(t.column(field.name)))
            changed = True
    if changed:
        pq.write_table(t, path)
    return changed


def main():
    dirs = [
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
        ROOT / "data" / "features",
    ]
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.parquet")):
            changed = fix_file(f)
            status = "fixed" if changed else "ok"
            print(f"  [{status}] {f.relative_to(ROOT)}")
    print("\nAll done. pd.read_parquet should work now.")


if __name__ == "__main__":
    main()
