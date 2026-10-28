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

Regenerate from the project root so the editable line stays `-e .`, and
check that the result is UTF-8 with LF line endings.

## Open

- **CUDA wheel.** The RTX 5070 is Blackwell (sm_120) and needs a cu128 or
  later torch wheel. A cu121 wheel installs cleanly and reports
  `cuda True`, then fails with "no kernel image is available". Whether the
  cu128 wheel was ever installed is unresolved (CLAUDE.md §9). The lockfile
  pins `torch==2.13.0` with no CUDA suffix. When this is settled, record
  the wheel and index URL here.
- **`pywin32==312`** is locked and is Windows-only, so
  `pip install -r requirements.lock.txt` fails on Linux or macOS. That
  matters for the Phase 9 fresh-clone check if it runs anywhere but
  Windows.
