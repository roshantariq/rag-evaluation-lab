"""SHA256 manifest of the frozen extraction, and checks against it.

data/interim/ is not committed: arXiv's default licence grants arXiv a
non-exclusive licence to distribute a paper, not third parties a right to
redistribute it, so neither the PDFs nor their extracted text can ship
with this repository. The gold set's 101 evidence spans are anchored to
exact character offsets in that text, so a clone cannot use the gold set
unless it regenerates the extraction byte for byte. This manifest is what
a clone verifies against.

One row per extracted document:

  interim_sha256   the JSON file as written by 02_extract_text.py
  text_sha256      its "text" field alone, UTF-8: what the spans anchor to
  pdf_sha256       the source PDF in data/raw/, so a mismatch can be
                   traced to arXiv serving different bytes rather than to
                   extraction

  python scripts\\30_extraction_hashes.py --write
      Writes data/eval/interim_sha256.csv from this tree's data/. Refuses
      to overwrite: the manifest describes the frozen extraction, once.

  python scripts\\30_extraction_hashes.py --check DIR [--raw DIR] [--tag NAME]
      Compares an extraction directory (and optionally its PDFs) with the
      manifest, file by file -> results/tables/extraction_check_<tag>.csv

  python scripts\\30_extraction_hashes.py --compare DIR_A DIR_B [--tag NAME]
      Compares two extraction runs with each other: determinism, which is a
      different claim from matching the freeze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from rageval.config import EVAL_DIR, INTERIM_DIR, RAW_DIR, TABLES_DIR
from rageval.provenance import record

MANIFEST = EVAL_DIR / "interim_sha256.csv"
COLUMNS = ("arxiv_id", "interim_file", "interim_bytes", "interim_sha256",
           "text_chars", "text_sha256", "pdf_file", "pdf_sha256")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def describe(path: Path, raw_dir: Path | None) -> dict:
    blob = path.read_bytes()
    doc = json.loads(blob.decode("utf-8"))
    pdf = (raw_dir / f"{path.stem}.pdf") if raw_dir else None
    return {
        "arxiv_id": doc["arxiv_id"],
        "interim_file": path.name,
        "interim_bytes": len(blob),
        "interim_sha256": sha256(blob),
        "text_chars": len(doc["text"]),
        "text_sha256": sha256(doc["text"].encode("utf-8")),
        "pdf_file": pdf.name if pdf else "",
        "pdf_sha256": sha256(pdf.read_bytes()) if pdf and pdf.exists() else "",
    }


def scan(interim: Path, raw: Path | None) -> dict[str, dict]:
    return {p.name: describe(p, raw) for p in sorted(interim.glob("*.json"))}


def compare(ref: dict[str, dict], new: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name in sorted(set(ref) | set(new)):
        a, b = ref.get(name), new.get(name)
        row = {"interim_file": name, "in_reference": a is not None, "in_new": b is not None}
        for field in ("interim_sha256", "text_sha256", "pdf_sha256"):
            if a is None or b is None or not a.get(field) or not b.get(field):
                row[field.replace("_sha256", "_match")] = "n/a"
            else:
                row[field.replace("_sha256", "_match")] = str(a[field] == b[field])
        row["text_chars_reference"] = a["text_chars"] if a else ""
        row["text_chars_new"] = b["text_chars"] if b else ""
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> None:
    both = df[df.in_reference & df.in_new]
    print(f"\n{label}: {len(df)} files ({len(both)} in both, "
          f"{int((~df.in_new).sum())} missing, {int((~df.in_reference).sum())} extra)")
    for col in ("interim_match", "text_match", "pdf_match"):
        counts = both[col].value_counts().to_dict()
        print(f"  {col:<14} {counts}")
    bad = both[(both.interim_match == "False") | (both.text_match == "False")
               | (both.pdf_match == "False")]
    for r in bad.itertuples():
        print(f"    differs: {r.interim_file}  file {r.interim_match}  text {r.text_match}  "
              f"pdf {r.pdf_match}  chars {r.text_chars_reference} -> {r.text_chars_new}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", type=Path, metavar="DIR")
    mode.add_argument("--compare", type=Path, nargs=2, metavar=("DIR_A", "DIR_B"))
    ap.add_argument("--raw", type=Path, default=None,
                    help="PDF directory for --check (pdf hashes compared when given).")
    ap.add_argument("--tag", default="clone")
    args = ap.parse_args()

    if args.write:
        if MANIFEST.exists():
            print(f"{MANIFEST} exists. It records the frozen extraction once; not overwritten.")
            return 1
        rows = scan(INTERIM_DIR, RAW_DIR)
        missing = [r["interim_file"] for r in rows.values() if not r["pdf_sha256"]]
        if missing:
            print(f"No PDF for {len(missing)} documents: {missing[:5]}")
            return 1
        with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows.values())
        print(f"{len(rows)} documents -> {MANIFEST}")
        return 0

    if args.check:
        with open(MANIFEST, encoding="utf-8", newline="") as fh:
            ref = {r["interim_file"]: r for r in csv.DictReader(fh)}
        df = compare(ref, scan(args.check, args.raw))
        label = f"{args.check} vs {MANIFEST.name}"
    else:
        a, b = args.compare
        df = compare(scan(a, None), scan(b, None))
        label = f"{a} vs {b}"
    out = TABLES_DIR / f"extraction_check_{args.tag}.csv"
    out.unlink(missing_ok=True)
    df.to_csv(out, index=False)
    record([out], gold=[MANIFEST] if args.check else [])
    report(df, label)
    print(f"\n  -> {out}")
    both = df[df.in_reference & df.in_new]
    clean = len(both) == len(df) and (both[["interim_match", "text_match"]] == "True").all().all()
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
