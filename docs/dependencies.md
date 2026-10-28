### arxiv >= 4.0 removed downloading

`Result.download_pdf()` and `Result.download_source()` were removed in
arxiv 4.x; the package is now a metadata client only. PDFs are fetched
directly from `Result.pdf_url` via `requests`, which also avoids one
redundant API call per paper.