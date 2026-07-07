# docqa — a Document Q&A skill for OpenClaw

A custom [OpenClaw](https://github.com/openclaw/openclaw) skill that lets any
OpenClaw agent answer questions grounded in a local folder of `.txt`/`.md`
files, with citations — no API key, no network call, no vector database.

This is a follow-up to my RAG "chat with your PDF" project: same idea
(retrieve relevant chunks, answer only from them), but redesigned as an
**agent skill** instead of a standalone app. The skill's job is only
retrieval — the host agent already has an LLM, so it does the answering.
That's also what makes this fully offline: no OpenAI call inside the skill
at all.

## How it works

```
docqa/
├── SKILL.md              # instructions OpenClaw's agent reads
├── scripts/
│   ├── index_docs.py     # builds a local TF-IDF index (stdlib only)
│   └── retrieve.py       # queries the index, returns JSON + citations
├── sample_docs/          # demo corpus (HR policy, product FAQ, security guide)
└── tests/
    └── test_docqa.py     # pytest suite, 8 tests
```

1. `index_docs.py` walks a docs folder, chunks each file (180 words,
   40-word overlap), builds a TF-IDF vocabulary from scratch (no sklearn —
   just `math` + `re` from the standard library), and writes a JSON index.
2. `retrieve.py` tokenizes the question the same way, computes cosine
   similarity against every chunk, and returns the top-k matches as JSON
   with a `below_threshold` flag when nothing relevant was found.
3. `SKILL.md` tells the agent: rebuild the index if stale, run retrieval,
   answer **only** from the returned chunks, cite the source file, and say
   "the docs don't cover that" instead of guessing when scores are low.

Zero dependencies beyond Python 3 — `pip install` isn't even needed.

## Install into OpenClaw

```bash
cp -r docqa ~/.openclaw/skills/docqa
# or, scoped to one project:
cp -r docqa <your-project>/skills/docqa
```

Point it at your own docs (optional — defaults to the bundled `sample_docs/`):

```bash
export DOCQA_DOCS_DIR=~/Documents/company-handbook
```

Start a new OpenClaw session (skills load fresh per session) and ask it
something the docs actually cover, e.g. with the bundled sample corpus:

```bash
openclaw agent --message "How many vacation days do new employees get?"
```

## Testing

I don't have a live OpenClaw gateway in the sandbox I built this in (no
model credentials, no messaging platform tokens), so I tested the part
that's actually testable without one: the retrieval scripts themselves,
which is where all the real logic lives.

```bash
pip install pytest  # only needed for the test suite, not the skill itself
pytest tests/ -v
```

8 tests, all passing — including known-answer retrieval for all three
sample docs, correct rejection of an out-of-corpus query, and error
handling for a missing index and an empty docs folder. One real bug
surfaced during testing and got fixed: the word "what" was slipping past
the stopword filter, which let a nonsense question falsely match a
document just by sharing that one word. Worth mentioning in an interview —
it's a good example of why you test retrieval with adversarial queries, not
just the ones you expect to work.

The one thing I couldn't verify end-to-end here is the actual agent
conversation loop (`openclaw agent --message "..."`) — that needs a real
OpenClaw install with a model configured, which the docs recommend testing
locally before sharing. Worth running once on your machine before you
demo it.

## Design notes / talking points

- **Why TF-IDF instead of embeddings:** no API key, no network, deterministic,
  and good enough for small-to-medium doc sets. For a larger corpus or fuzzier
  natural-language queries, the next step would be swapping in embeddings
  (e.g. the same OpenAI embedding approach from the RAG project) behind the
  same `retrieve.py` interface — the JSON contract to the agent wouldn't
  need to change.
- **Why the skill doesn't call an LLM itself:** the host agent already is
  one. Keeping the skill to "retrieval only" means it works with whatever
  model the user has OpenClaw configured with, and keeps the skill's own
  test surface small and deterministic.
- **Sparse vectors as plain dicts:** the demo corpus is tiny, so this is
  simple and fast enough. At real scale you'd want a proper sparse matrix
  (scipy) or an actual vector index (FAISS/sqlite-vec).
