# Local data directory

This directory is intentionally empty in Git. Put private or licensed files in
the following local paths when running the project:

- `pdfs/` — source PDF documents
- `parsed/` — generated Markdown and structured JSON
- `chroma/` — generated vector database
Everything below `data/`, except this file, is ignored by Git. Never commit
source documents, extracted content, embeddings, or local databases.
