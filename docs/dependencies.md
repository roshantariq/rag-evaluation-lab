# Dependencies

Why each non-obvious pin or workaround exists. `pyproject.toml` declares
the direct dependencies; `requirements.lock.txt` is the full resolved set.
Detail and history live in the Phase 0 section of `docs/build-log.md`.

Never bare-upgrade mid-project. A change here means a new section below:
what changed, why, and the symptom it fixes.

## Pins

### `langchain-community==0.4.1`

**Symptom.** Importing `ragas` fails at module load with an import error
for `langchain_community.chat_models.vertexai`.

**Cause.** `ragas` (every version through 0.4.3) imports that module at
load time. It was removed in `langchain-community` **0.4.2** and exists in
0.4.1 and earlier. `ragas` declares `langchain-community` with no upper
bound, so pip installs the newest release and produces a broken
combination. Upgrading `ragas` does not help: 0.4.3 has the same import.

**Fix.** Pin `langchain-community==0.4.1` in `pyproject.toml`.

### `pillow` (declared explicitly)

**Symptom.** `ragas` 0.3.1 fails importing `PIL`.

**Cause.** `ragas` 0.3.1 imports `PIL` without declaring `pillow` as a
dependency, so a clean environment may not have it.

**Fix.** Declare `pillow` directly in `pyproject.toml`.

**Verified working together:** langchain 1.3.17, langchain-core 1.6.0,
langchain-community 0.4.1, ragas 0.3.1 (versions as locked in
`requirements.lock.txt`). Both pins carry explanatory comments in
`pyproject.toml`.

## API changes worked around

### arxiv >= 4.0 removed downloading

`Result.download_pdf()` and `Result.download_source()` were removed in
arxiv 4.x; the package is now a metadata client only. PDFs are fetched
directly from `Result.pdf_url` via `requests`, which also avoids one
redundant API call per paper.

## The lockfile

`requirements.lock.txt` captures the full resolved set. It was originally
written by PowerShell redirection, which produced UTF-16 with CRLF line
endings, and it pinned the package itself to a git commit not in this
repository's history. On 28 October 2026 it was transcoded to UTF-8 with LF
and the self-pin replaced with `-e .`; no version changed. `.gitattributes`
now enforces LF.

`pywin32==312` is Windows-only and made `pip install -r` fail on Linux and
macOS. It now carries the marker `; sys_platform == "win32"`, so it
installs on Windows only. `pip freeze` drops markers, so re-add it after
regenerating.

Regenerate from the project root so the editable line stays `-e .`, and
check that the result is UTF-8 with LF line endings.

## torch: CUDA 12.8 wheel (resolved 31 October 2026)

**Symptom.** The RTX 5070 is Blackwell (sm_120) and needs a cu128 or later
torch wheel. A cu121 wheel installs cleanly and reports `cuda True`, then
fails with "no kernel image is available": the obvious check passes.

**What is installed.** `torch 2.11.0+cu128` (CUDA 12.8), on Python 3.11.9,
an RTX 5070 Ti Laptop GPU; `torch.cuda.is_available()` is True and a GPU op
succeeds. Its `dist-info` is dated 27 August 2026, 20:42, the day after the
lockfile was first written.

**Why the lockfile was stale.** It pinned `torch==2.13.0` with no CUDA
suffix, recorded before the CUDA wheel was installed. The cu128 index
(`https://download.pytorch.org/whl/cu128`) tops out at 2.11.0, so a cu128
2.13.0 never existed; the pin described a CPU build this environment does
not have. It was the only line of 215 that disagreed with `pip freeze`.

**Fix.** The lockfile now pins the installed wheel and names its index:

```
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.11.0+cu128; sys_platform != "darwin"
torch==2.11.0; sys_platform == "darwin"
```

macOS has no CUDA wheels, so it gets the plain build; that line is
untested. `pip install --dry-run --no-deps -r requirements.lock.txt`
against this environment reports every pin satisfied. A fresh install
from the lockfile is Phase 9's check, not yet run. `pip freeze` drops
the index line and the markers, so re-add both after regenerating, as for
`pywin32`.

## Open

- **Embedding cache provenance.** `.cache/embeddings.sqlite` is keyed only
  on `hash(model, text)`, with no record of device, torch version or date
  per vector. The Phase 5 dense baseline was committed at 20:11 on
  27 August, 31 minutes *before* the cu128 wheel was installed, so its
  vectors were computed without CUDA, under whatever torch was present
  then. Later dense work may have reused those cached vectors or added
  GPU-computed ones; the cache cannot say which. CPU and GPU vectors agree
  to about 1e-6, not bit for bit, which can at most reorder near-ties. It
  affects only dense arms (the baseline, chunk256/1024, hybrid,
  `para_dense`, the oracle tables); BM25, which every later decision used,
  reads no embeddings, and the reranker scores without the cache.
  Clearing the cache and rebuilding settles it, and needs the owner's
  approval (CLAUDE.md §14).
