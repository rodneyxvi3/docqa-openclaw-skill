---
name: docqa
description: Answer questions grounded in a local folder of text/markdown documents using TF-IDF retrieval. Use when the user asks a question about files in a configured docs folder, says "search my docs", "what does X say about Y", "index this folder", or references company/project documentation that lives on disk.
version: 1.0.0
homepage: https://github.com/example/docqa-skill
metadata:
  openclaw:
    emoji: "📚"
    requires:
      bins: ["python3"]
    envVars:
      - name: DOCQA_DOCS_DIR
        required: false
        description: Path to the folder of documents to index. Defaults to {baseDir}/sample_docs if unset.
---

# Document Q&A

Grounded question-answering over a local folder of `.txt` / `.md` files. Retrieval
is local TF-IDF (no API key, no network) — you provide the answer, this skill
just finds the right passages.

## Workflow

1. Determine the docs folder: use `$DOCQA_DOCS_DIR` if set, else
   `{baseDir}/sample_docs`.
2. Check whether an index exists at `<docs_dir>/.docqa_index.json`. If it is
   missing, or older than any file in the docs folder, rebuild it:
   ```bash
   python3 {baseDir}/scripts/index_docs.py --docs-dir <docs_dir>
   ```
3. Run retrieval for the user's question:
   ```bash
   python3 {baseDir}/scripts/retrieve.py --docs-dir <docs_dir> --query "<question>" --top-k 5
   ```
   This prints JSON: `{"results": [{"source", "chunk_index", "score", "text"}, ...], "best_score": ...}`.
4. Answer using **only** the text in `results`. For every claim, cite the
   source file (e.g. "per `hr_policy.txt`"). Do not use outside knowledge to
   fill gaps.
5. Stop condition: if `best_score` is below `0.05`, or `results` is empty, tell
   the user the docs don't cover that question — do not guess.

## Rules

- Never fabricate a citation or a document that wasn't in `results`.
- If the user adds or edits files in the docs folder, re-run step 2 before
  retrieving again.
- Keep answers short; quote at most one short phrase per source, paraphrase
  the rest.
