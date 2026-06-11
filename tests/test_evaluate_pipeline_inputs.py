import importlib.util
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "comagraag"
sys.path.insert(0, str(PKG))


def _load_evaluate_module():
    spec = importlib.util.spec_from_file_location("comagraag_evaluate_impl", PKG / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bad_answer_detection_covers_common_pipeline_failures():
    evaluate = _load_evaluate_module()

    for answer in [
        "",
        "unknown",
        "Unknown",
        "insufficient information",
        "not specified",
        "Passage [8] gives full name",
        "The draft answer is \"Foo\"",
        "But per instructions",
    ]:
        assert evaluate._is_bad_answer(answer)


def test_2wiki_pipeline_passages_do_not_include_global_context():
    evaluate = _load_evaluate_module()
    context = [("Local Title", ["local evidence about the answer"])]
    global_passages = ["Global Title: unrelated founder evidence"]
    global_bm25 = BM25Okapi([global_passages[0].lower().split()])

    passages = evaluate._pipeline_passages_for(
        mode="full",
        dataset="2wiki",
        variant="evidence_first",
        question="Where does the founder work?",
        context=context,
        global_passages=global_passages,
        global_bm25=global_bm25,
    )

    assert passages == ["Local Title: local evidence about the answer"]


def test_hotpot_pipeline_passages_can_disable_global_context():
    evaluate = _load_evaluate_module()
    context = [("Local Title", ["local evidence about the answer"])]
    global_passages = ["Global Title: founder work evidence"]
    global_bm25 = BM25Okapi([global_passages[0].lower().split()])

    passages = evaluate._pipeline_passages_for(
        mode="full",
        dataset="hotpotqa",
        variant="evidence_first",
        question="Where does the founder work?",
        context=context,
        global_passages=global_passages,
        global_bm25=global_bm25,
        use_global_context=False,
    )

    assert passages == ["Local Title: local evidence about the answer"]


def test_naive_fallback_cache_lookup_can_reuse_existing_2wiki_naive_run(tmp_path):
    evaluate = _load_evaluate_module()
    cache_path = tmp_path / "cache_wise_2wiki_naive_rag_naive_rag.json"
    cache_path.write_text('{"qid-1": {"answer": "fallback answer"}}', encoding="utf-8")

    cache = evaluate._load_naive_fallback_cache(
        out_dir=str(tmp_path),
        dataset="2wiki",
        cache_tag="wise_2wiki_evidencefirst_v6_localctx_b0",
    )

    assert cache["qid-1"]["answer"] == "fallback answer"


def test_ablation_cache_path_is_distinct_from_full_model():
    evaluate = _load_evaluate_module()

    full_path = evaluate._cache_path(
        out_dir="results",
        dataset="hotpotqa",
        mode="full",
        variant="evidence_first",
        max_iter=3,
        cache_tag=None,
        ablation="none",
    )
    ablation_path = evaluate._cache_path(
        out_dir="results",
        dataset="hotpotqa",
        mode="full",
        variant="evidence_first",
        max_iter=3,
        cache_tag=None,
        ablation="without_repair",
    )

    assert full_path != ablation_path
    assert "without_repair" in ablation_path
