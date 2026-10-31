"""Which command wrote each results table, from which tree, on which gold set.

The build plan asked for runs "re-runnable by run ID". This project keeps
tagged CSVs instead of a tracking server (build log, Plan amendments), so
the run record is a ledger next to the tables: one appended row per table
written, holding the full command line, the git commit, whether the tree
was dirty, and a hash of every gold file the run read.

The dirty flag is the part not to drop. A commit recorded from a modified
tree names code that did not produce the table, which is worse than no
commit at all, and an unreported guard is this project's signature failure.
So the flag is computed, the paths behind it are listed, and when git
cannot answer, both read "unknown" rather than a guess.

Dirty means any uncommitted change (modified, staged or untracked and not
ignored) anywhere in the repo except the tables this run itself wrote and
the ledger. Other tables count: scripts 13 to 28 read them as inputs.

Every script that writes a table calls record(); there is one copy of this
logic, as with rageval.evaluation.spans.

`ef_search` is the HNSW search depth of every dense query the run made
(Chroma's approximate search; see rageval.store.chroma_store), "n/a" when
it made none. A table derived from other tables (paired tests, stability,
results) made no query and reads "n/a"; its inputs' rows carry the value.
Rows written before the column existed were annotated on 31 Oct 2026:
"100" for every table produced by a dense query, Chroma 1.5.9's default,
which every collection persisted from creation until then.

Tables written before the ledger existed carry one "retrofit" row each:
the regenerating command where it is on record or was confirmed by
re-running, otherwise "unknown", and the commit, flag and gold hash always
"unknown". A plausible commit inferred after the fact would fabricate
provenance in the one file whose job is to establish it.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from rageval.config import PROJECT_ROOT

LEDGER_NAME = "provenance.csv"
COLUMNS = ("table", "command", "git_sha", "dirty", "dirty_paths",
           "gold_files", "gold_sha256", "timestamp", "source", "note", "ef_search")
UNKNOWN = "unknown"
MAX_DIRTY_PATHS = 20


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, encoding="utf-8", timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def tree_state(exclude: Iterable[Path] = ()) -> tuple[str, str, str]:
    """(commit, dirty, dirty paths) for the working tree, ignoring `exclude`.

    Tracked changes come from `git diff HEAD`, which compares content after
    the repo's eol=lf normalisation, not from `git status`, which also flags
    a file rewritten with CRLF endings and identical content. On Windows
    pandas writes CRLF, so every re-run would otherwise mark the tree dirty.
    """
    sha = _git("rev-parse", "HEAD")
    changed = _git("diff", "HEAD", "--name-only")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    if sha is None or changed is None or untracked is None:
        return UNKNOWN, UNKNOWN, ""
    skip = {_rel(p) for p in exclude}
    paths = [p.strip().strip('"') for p in (changed + untracked).splitlines()]
    paths = [p for p in dict.fromkeys(paths) if p and p not in skip]
    shown = paths[:MAX_DIRTY_PATHS]
    if len(paths) > MAX_DIRTY_PATHS:
        shown.append(f"+{len(paths) - MAX_DIRTY_PATHS} more")
    return sha.strip(), str(bool(paths)), ";".join(shown)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_line(argv: Sequence[str] | None = None) -> str:
    """The invocation as typed, script path relative to the repo."""
    argv = list(sys.argv if argv is None else argv)
    if argv:
        argv[0] = _rel(Path(argv[0])).replace("/", "\\") if Path(argv[0]).exists() else argv[0]
    return subprocess.list2cmdline(["python", *argv])


def record(tables: Iterable[Path], gold: Iterable[Path] = (),
           argv: Sequence[str] | None = None, ef_search: int | str | None = None) -> None:
    """Append one ledger row per table, in the ledger beside each table.

    `ef_search`: the HNSW search depth the run's dense queries used, None
    when it made no dense query.
    """
    tables = [Path(t) for t in tables]
    gold = [Path(g) for g in gold]
    ledgers = {t.parent / LEDGER_NAME for t in tables}
    sha, dirty, dirty_paths = tree_state(exclude=[*tables, *ledgers])
    common = {
        "command": command_line(argv),
        "git_sha": sha,
        "dirty": dirty,
        "dirty_paths": dirty_paths,
        "gold_files": ";".join(_rel(g) for g in gold) or "n/a",
        "gold_sha256": ";".join(file_sha256(g) for g in gold) or "n/a",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "recorded",
        "note": "",
        "ef_search": "n/a" if ef_search is None else str(ef_search),
    }
    for table in tables:
        ledger = table.parent / LEDGER_NAME
        new = not ledger.exists()
        if not new:
            with open(ledger, encoding="utf-8", newline="") as fh:
                header = tuple(next(csv.reader(fh), ()))
            if header != COLUMNS:
                raise ValueError(f"{ledger} has columns {header}, expected {COLUMNS}; "
                                 "appending would misalign every field")
        with open(ledger, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow({"table": table.name, **common})
