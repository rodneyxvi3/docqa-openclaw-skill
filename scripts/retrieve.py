#!/usr/bin/env python3
"""
retrieve.py — query the local TF-IDF index built by index_docs.py.

No network access and no API key required. Prints a JSON object to stdout
that an agent (or a human) can read to answer a question with citations.

Usage:
    python3 retrieve.py --docs-dir ./sample_docs --query "What is the vacation policy?" --top-k 5
"""
import argparse
import json
import math
import os
import re
import sys

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Kept identical to index_docs.py on purpose — the query must be tokenized
# the same way the corpus was, or scores are meaningless.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will",
    "with", "we", "you", "your", "i", "can", "do", "does", "did", "doing",
    "have", "has", "had", "am", "were", "been", "being", "should", "would",
    "could", "may", "might", "must", "shall", "what", "how", "why", "when",
    "where", "who", "which", "whom", "get", "got", "my", "me", "our",
}


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def load_index(index_path: str):
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def query_vector(query: str, vocab: dict, idf: list):
    tokens = tokenize(query)
    counts = {}
    for tok in tokens:
        if tok in vocab:
            counts[tok] = counts.get(tok, 0) + 1
    total = sum(counts.values()) or 1
    sparse = {}
    for tok, cnt in counts.items():
        idx = vocab[tok]
        tf = cnt / total
        sparse[idx] = tf * idf[idx]
    norm = math.sqrt(sum(v * v for v in sparse.values())) or 1.0
    return {idx: v / norm for idx, v in sparse.items()}, len(tokens), len(counts)


def cosine_sim(query_vec: dict, chunk_vec: dict):
    # both are already L2-normalized, so cosine similarity = dot product
    if len(query_vec) > len(chunk_vec):
        query_vec, chunk_vec = chunk_vec, query_vec
    return sum(v * chunk_vec.get(idx, 0.0) for idx, v in query_vec.items())


def main():
    ap = argparse.ArgumentParser(description="Query the docqa TF-IDF index.")
    ap.add_argument("--docs-dir", default=os.environ.get("DOCQA_DOCS_DIR", "./sample_docs"))
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--index-path", default=None)
    args = ap.parse_args()

    index_path = args.index_path or os.path.join(args.docs_dir, ".docqa_index.json")

    if not os.path.exists(index_path):
        print(json.dumps({
            "error": f"No index found at {index_path!r}. Run index_docs.py first.",
        }), file=sys.stderr)
        sys.exit(1)

    index = load_index(index_path)
    vocab = index["vocab"]
    idf = index["idf"]

    q_vec, n_tokens, n_matched = query_vector(args.query, vocab, idf)

    if n_tokens == 0:
        print(json.dumps({"error": "Query had no usable tokens after stopword removal."}), file=sys.stderr)
        sys.exit(1)

    scored = []
    for chunk in index["chunks"]:
        cvec = {int(k): v for k, v in chunk["vector"].items()}
        score = cosine_sim(q_vec, cvec)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:args.top_k]
    best_score = top[0][0] if top else 0.0

    results = [{
        "source": c["source"],
        "chunk_index": c["chunk_index"],
        "score": round(s, 4),
        "text": c["text"],
    } for s, c in top if s > 0]

    out = {
        "query": args.query,
        "query_terms_matched_vocab": n_matched,
        "best_score": round(best_score, 4),
        "below_threshold": best_score < args.min_score,
        "results": results,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
