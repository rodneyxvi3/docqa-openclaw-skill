#!/usr/bin/env python3
"""
index_docs.py — build a local TF-IDF search index over a folder of .txt/.md files.

No network access and no API key required. Writes <docs_dir>/.docqa_index.json.

Usage:
    python3 index_docs.py --docs-dir ./sample_docs [--chunk-size 180] [--chunk-overlap 40]
"""
import argparse
import json
import math
import os
import re
import sys
import time

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will",
    "with", "we", "you", "your", "i", "can", "do", "does", "did", "doing",
    "have", "has", "had", "am", "were", "been", "being", "should", "would",
    "could", "may", "might", "must", "shall", "what", "how", "why", "when",
    "where", "who", "which", "whom", "get", "got", "my", "me", "our",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def read_docs(docs_dir: str):
    files = []
    for root, _, names in os.walk(docs_dir):
        for name in sorted(names):
            if name.startswith("."):
                continue
            if name.lower().endswith((".txt", ".md")):
                files.append(os.path.join(root, name))
    return sorted(files)


def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        piece = words[start:start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def build_index(docs_dir: str, chunk_size: int, overlap: int):
    files = read_docs(docs_dir)
    if not files:
        raise FileNotFoundError(
            f"No .txt or .md files found under {docs_dir!r}. Nothing to index."
        )

    chunk_records = []  # {source, chunk_index, text, tokens}
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        rel = os.path.relpath(path, docs_dir)
        for i, piece in enumerate(chunk_text(raw, chunk_size, overlap)):
            chunk_records.append({
                "source": rel,
                "chunk_index": i,
                "text": piece,
                "tokens": tokenize(piece),
            })

    if not chunk_records:
        raise ValueError("Docs were found but produced zero chunks (empty files?).")

    # document frequency
    df = {}
    for rec in chunk_records:
        for tok in set(rec["tokens"]):
            df[tok] = df.get(tok, 0) + 1

    vocab = {tok: idx for idx, tok in enumerate(sorted(df.keys()))}
    n_chunks = len(chunk_records)
    # smoothed idf, same form as sklearn's default: ln((1+N)/(1+df)) + 1
    idf = [0.0] * len(vocab)
    for tok, idx in vocab.items():
        idf[idx] = math.log((1 + n_chunks) / (1 + df[tok])) + 1.0

    chunks_out = []
    for rec in chunk_records:
        counts = {}
        for tok in rec["tokens"]:
            if tok in vocab:
                counts[tok] = counts.get(tok, 0) + 1
        total = sum(counts.values()) or 1
        sparse = {}
        for tok, cnt in counts.items():
            idx = vocab[tok]
            tf = cnt / total
            sparse[str(idx)] = tf * idf[idx]
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in sparse.values())) or 1.0
        sparse = {k: v / norm for k, v in sparse.items()}
        chunks_out.append({
            "source": rec["source"],
            "chunk_index": rec["chunk_index"],
            "text": rec["text"],
            "vector": sparse,
        })

    index = {
        "version": 1,
        "built_at": time.time(),
        "docs_dir": os.path.abspath(docs_dir),
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "vocab": vocab,
        "idf": idf,
        "num_files": len(files),
        "num_chunks": n_chunks,
        "chunks": chunks_out,
    }
    return index


def main():
    ap = argparse.ArgumentParser(description="Build a local TF-IDF index for docqa.")
    ap.add_argument("--docs-dir", default=os.environ.get("DOCQA_DOCS_DIR", "./sample_docs"))
    ap.add_argument("--chunk-size", type=int, default=180)
    ap.add_argument("--chunk-overlap", type=int, default=40)
    ap.add_argument("--index-path", default=None, help="Override output path for the index JSON.")
    args = ap.parse_args()

    docs_dir = args.docs_dir
    index_path = args.index_path or os.path.join(docs_dir, ".docqa_index.json")

    try:
        index = build_index(docs_dir, args.chunk_size, args.chunk_overlap)
    except (FileNotFoundError, ValueError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(json.dumps({
        "status": "ok",
        "index_path": os.path.abspath(index_path),
        "num_files": index["num_files"],
        "num_chunks": index["num_chunks"],
        "vocab_size": len(index["vocab"]),
    }))


if __name__ == "__main__":
    main()
