"""Parse BCBench Parquet files -> labels.json + code/*.java"""
import hashlib, json
from pathlib import Path
import pandas as pd

raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "bcbench"
code_dir = raw_dir / "code"
code_dir.mkdir(parents=True, exist_ok=True)

sources = {}
labels = []
hashes_seen = set()

for fname in sorted(raw_dir.glob("*.parquet")):
    print(f"  Processing {fname.name} ({fname.stat().st_size / 1024 / 1024:.0f} MB)...")
    df = pd.read_parquet(str(fname))

    for _, row in df.iterrows():
        func1 = str(row["func1"])
        func2 = str(row["func2"])
        label = int(bool(row["label"]))

        h1 = hashlib.md5(func1.encode("utf-8")).hexdigest()[:16]
        h2 = hashlib.md5(func2.encode("utf-8")).hexdigest()[:16]

        for h, code in [(h1, func1), (h2, func2)]:
            orig_h = h
            c = 1
            while h in hashes_seen:
                h = f"{orig_h}_{c}"
                c += 1
            hashes_seen.add(h)
            if orig_h not in sources:
                sources[h] = code
                fpath = code_dir / f"{h}.java"
                fpath.write_text(code, encoding="utf-8")

        labels.append({"file_a": h1, "file_b": h2, "label": label})

with open(raw_dir / "labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f)

print(f"\nBCBench: {len(sources)} unique sources, {len(labels)} pairs")
