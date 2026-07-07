"""
Smoke / correctness tests for the docqa skill.

Run with:  pytest -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
SAMPLE_DOCS = os.path.join(ROOT, "sample_docs")
PY = sys.executable


def run(*args):
    result = subprocess.run(
        [PY, *args], capture_output=True, text=True, cwd=ROOT
    )
    return result


def index_sample_docs(docs_dir):
    r = run(os.path.join(SCRIPTS, "index_docs.py"), "--docs-dir", docs_dir)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def retrieve(docs_dir, query, top_k=5):
    r = run(
        os.path.join(SCRIPTS, "retrieve.py"),
        "--docs-dir", docs_dir, "--query", query, "--top-k", str(top_k),
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_index_builds_successfully():
    out = index_sample_docs(SAMPLE_DOCS)
    assert out["status"] == "ok"
    assert out["num_files"] == 3
    assert out["num_chunks"] >= 3
    assert os.path.exists(os.path.join(SAMPLE_DOCS, ".docqa_index.json"))


def test_retrieve_finds_correct_doc_for_vacation_question():
    index_sample_docs(SAMPLE_DOCS)
    out = retrieve(SAMPLE_DOCS, "How many vacation days do I accrue per year?")
    assert out["results"], "expected at least one result"
    assert out["results"][0]["source"] == "hr_policy.txt"
    assert out["best_score"] > 0.1


def test_retrieve_finds_correct_doc_for_pricing_question():
    index_sample_docs(SAMPLE_DOCS)
    out = retrieve(SAMPLE_DOCS, "How much does the Growth plan cost per user?")
    assert out["results"][0]["source"] == "product_faq.txt"


def test_retrieve_finds_correct_doc_for_security_question():
    index_sample_docs(SAMPLE_DOCS)
    out = retrieve(SAMPLE_DOCS, "Is two factor authentication mandatory for admins?")
    assert out["results"][0]["source"] == "security_guide.txt"


def test_retrieve_flags_out_of_corpus_query():
    index_sample_docs(SAMPLE_DOCS)
    out = retrieve(SAMPLE_DOCS, "What is the airspeed velocity of an unladen swallow?")
    assert out["below_threshold"] is True
    assert out["results"] == []


def test_retrieve_without_index_gives_helpful_error():
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(SAMPLE_DOCS, "hr_policy.txt"), tmp)
        r = run(os.path.join(SCRIPTS, "retrieve.py"), "--docs-dir", tmp, "--query", "vacation")
        assert r.returncode == 1
        assert "Run index_docs.py first" in r.stderr


def test_index_on_empty_folder_errors_cleanly():
    with tempfile.TemporaryDirectory() as tmp:
        r = run(os.path.join(SCRIPTS, "index_docs.py"), "--docs-dir", tmp)
        assert r.returncode == 1
        err = json.loads(r.stderr)
        assert "No .txt or .md files" in err["error"]


def test_index_is_stable_and_rerunnable():
    # rebuilding the index twice in a row shouldn't error or change file counts
    out1 = index_sample_docs(SAMPLE_DOCS)
    out2 = index_sample_docs(SAMPLE_DOCS)
    assert out1["num_chunks"] == out2["num_chunks"]


def teardown_module(module):
    idx = os.path.join(SAMPLE_DOCS, ".docqa_index.json")
    if os.path.exists(idx):
        os.remove(idx)
