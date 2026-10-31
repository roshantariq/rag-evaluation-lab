"""How exact, and how repeatable, is dense retrieval from the Chroma index?

Chroma answers queries with an approximate HNSW search. This copies the
collection to a temporary directory (the real index is never opened for
writing), then for each ef_search setting queries the 58 scorable
questions' cached vectors for the top 50 in --processes separate Python
processes, and compares every run with exact cosine search over all
stored vectors. ef_search is a persisted collection setting in Chroma, not
a query argument, which is why it is changed on the copy only.

Writes results/tables/hnsw_check.csv: per ef_search, per depth k, the
number of questions whose top-k chunk order is identical across all runs,
identical to exact search in all runs, and whose gold spans found at k
match exact search in all runs.

  python scripts\\diagnose_hnsw.py
  python scripts\\diagnose_hnsw.py --ef 100 500 --processes 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from rageval.config import CHROMA_DIR, EVAL_DIR, TABLES_DIR
from rageval.provenance import record

COLLECTION = "baseline_fixed512_minilm"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEPTHS = (10, 13, 20, 30, 50)


def worker(path: str, out: str, ef: int | None, exact: bool) -> None:
    """One process: top-50 chunk ids per scorable question, as JSON."""
    import chromadb
    import numpy as np
    from chromadb.config import Settings

    from rageval.config import EMBED_CACHE
    from rageval.embed.encoder import EmbeddingCache
    from rageval.evaluation.gold import load_gold
    from rageval.evaluation.retrieval_metrics import is_scorable

    client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(COLLECTION)
    if ef is not None and col.configuration["hnsw"]["ef_search"] != ef:
        col.modify(configuration={"hnsw": {"ef_search": ef}})
        col = client.get_collection(COLLECTION)
    qs = [g for g in load_gold(EVAL_DIR / "gold_questions.jsonl") if is_scorable(g)]
    vecs = EmbeddingCache(EMBED_CACHE).get_many(MODEL, [q.question for q in qs])
    if len(vecs) != len(qs):
        raise SystemExit(f"only {len(vecs)} of {len(qs)} query vectors are cached")
    Q = np.vstack([vecs[i] for i in range(len(qs))]).astype(np.float32)
    if exact:
        stored = col.get(include=["embeddings"])
        E = np.asarray(stored["embeddings"], dtype=np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        sims = (Q / np.linalg.norm(Q, axis=1, keepdims=True)) @ E.T
        res = {q.id: [stored["ids"][j] for j in np.argsort(-sims[i], kind="stable")[:50]]
               for i, q in enumerate(qs)}
    else:
        r = col.query(query_embeddings=Q, n_results=50, include=[])
        res = {q.id: r["ids"][i] for i, q in enumerate(qs)}
    Path(out).write_text(json.dumps(res), encoding="utf-8")


def gold_found(gold: dict, qid: str, ids: list[str], k: int) -> frozenset[int]:
    found = set()
    for cid in ids[:k]:
        arxiv, span = cid.rsplit(":", 1)
        start, end = map(int, span.split("-"))
        for j, ev in enumerate(gold[qid]["evidence"]):
            if arxiv == ev["arxiv_id"] and start < ev["char_end"] and ev["char_start"] < end:
                found.add(j)
    return frozenset(found)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ef", type=int, nargs="+", default=[100, 500],
                    help="ef_search settings to test, each on its own fresh copy.")
    ap.add_argument("--processes", type=int, default=3)
    ap.add_argument("--worker", nargs=4, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        path, out, ef, exact = args.worker
        worker(path, out, None if ef == "-" else int(ef), exact == "exact")
        return 0

    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(EVAL_DIR / "gold_questions.jsonl", encoding="utf-8")}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def run(copy: Path, out: Path, ef, exact: bool) -> dict:
            subprocess.run([sys.executable, __file__, "--worker", str(copy), str(out),
                            "-" if ef is None else str(ef), "exact" if exact else "hnsw"],
                           check=True, capture_output=True)
            return json.loads(out.read_text(encoding="utf-8"))

        base = tmp / "copy_exact"
        shutil.copytree(CHROMA_DIR, base)
        exact = run(base, tmp / "exact.json", None, True)
        for ef in args.ef:
            copy = tmp / f"copy_ef{ef}"
            shutil.copytree(CHROMA_DIR, copy)
            runs = [run(copy, tmp / f"ef{ef}_{p}.json", ef, False) for p in range(args.processes)]
            for k in DEPTHS:
                rows.append({
                    "ef_search": ef, "k": k, "processes": args.processes, "questions": len(exact),
                    "identical_across_runs": sum(all(r[q][:k] == runs[0][q][:k] for r in runs)
                                                 for q in exact),
                    "identical_to_exact": sum(all(r[q][:k] == exact[q][:k] for r in runs)
                                              for q in exact),
                    "gold_spans_match_exact": sum(
                        all(gold_found(gold, q, r[q], k) == gold_found(gold, q, exact[q], k)
                            for r in runs) for q in exact),
                })
    df = pd.DataFrame(rows)
    out = TABLES_DIR / "hnsw_check.csv"
    out.unlink(missing_ok=True)
    df.to_csv(out, index=False)
    record([out], gold=[EVAL_DIR / "gold_questions.jsonl"])
    print(df.to_string(index=False))
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
