import importlib.util
import sys
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "comagraag"
sys.path.insert(0, str(PKG))


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("comagraag_pipeline_ablation_test", PKG / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_agents():
    spec = importlib.util.spec_from_file_location("comagraag_agents_ablation_test", PKG / "agents.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _graph():
    graph = nx.DiGraph()
    graph.add_edge("Alpha", "Beta", relations=["bridge"])
    return graph


def test_ablation_without_verification_skips_chain_check_and_repair(monkeypatch):
    pipeline = _load_pipeline()

    calls = {"checked": 0, "aga": 0}

    class Checker:
        last_answer_type = "entity"

        def check_evidence_chain_from_strings(self, triples, question):
            calls["checked"] += 1
            return False, {"gap_type": "missing_entities", "missing": ["Gamma"]}, []

    monkeypatch.setattr(pipeline, "EvidenceChainChecker", Checker)
    monkeypatch.setattr(pipeline, "run_qda", lambda ctx: ctx.setdefault("subqueries", [ctx["question"]]))
    monkeypatch.setattr(pipeline, "run_gra", lambda ctx: ctx.update(triples=["Alpha -- bridge --> Beta"]))

    def fake_aga(ctx):
        calls["aga"] += 1
        ctx["answer"] = "Beta"
        return "Beta"

    monkeypatch.setattr(pipeline, "run_aga", fake_aga)

    result = pipeline.run_pipeline(
        "What connects Alpha?",
        _graph(),
        mode="full",
        variant="evidence_first",
        ablation="without_verification",
    )

    assert result["answer"] == "Beta"
    assert calls == {"checked": 0, "aga": 1}
    assert result["diagnostics"]["evidence_first_ablation"] == "without_verification"
    assert result["diagnostics"]["evidence_first_verification_disabled"]
    assert result["diagnostics"]["evidence_first_repair_disabled"]
    assert result["diagnostics"]["evidence_first_repair_steps"] == []


def test_ablation_without_repair_keeps_gap_diagnostics_but_skips_repair(monkeypatch):
    pipeline = _load_pipeline()

    calls = {"checked": 0, "aga": 0}

    class Checker:
        last_answer_type = "entity"

        def check_evidence_chain_from_strings(self, triples, question):
            calls["checked"] += 1
            return False, {"gap_type": "missing_entities", "missing": ["Gamma"]}, []

    monkeypatch.setattr(pipeline, "EvidenceChainChecker", Checker)
    monkeypatch.setattr(pipeline, "run_qda", lambda ctx: ctx.setdefault("subqueries", [ctx["question"]]))
    monkeypatch.setattr(pipeline, "run_gra", lambda ctx: ctx.update(triples=["Alpha -- bridge --> Beta"]))

    def fake_aga(ctx):
        calls["aga"] += 1
        ctx["answer"] = "Beta"
        return "Beta"

    monkeypatch.setattr(pipeline, "run_aga", fake_aga)

    result = pipeline.run_pipeline(
        "What connects Alpha?",
        _graph(),
        mode="full",
        passages=["Gamma: supporting passage"],
        variant="evidence_first",
        ablation="without_repair",
    )

    assert result["answer"] == "Beta"
    assert calls == {"checked": 1, "aga": 1}
    assert result["diagnostics"]["evidence_first_gap_type"] == "missing_entities"
    assert result["diagnostics"]["evidence_first_missing_entities"] == ["Gamma"]
    assert result["diagnostics"]["evidence_first_ablation"] == "without_repair"
    assert result["diagnostics"]["evidence_first_repair_disabled"]
    assert result["diagnostics"]["evidence_first_repair_steps"] == []


def test_empty_triples_use_passage_fallback_and_record_step(monkeypatch):
    pipeline = _load_pipeline()

    calls = {"context_simple": 0, "aga": 0}

    monkeypatch.setattr(pipeline, "run_qda", lambda ctx: ctx.setdefault("subqueries", [ctx["question"]]))
    monkeypatch.setattr(pipeline, "run_gra", lambda ctx: ctx.update(triples=[]))

    def fake_context_simple(question, passages):
        calls["context_simple"] += 1
        return {"answer": "Fallback Beta"}

    def fake_aga(ctx):
        calls["aga"] += 1
        ctx["answer"] = "Should not run"
        return "Should not run"

    import agents

    monkeypatch.setattr(agents, "run_context_answer_simple", fake_context_simple)
    monkeypatch.setattr(pipeline, "run_aga", fake_aga)

    result = pipeline.run_pipeline(
        "What connects Alpha?",
        _graph(),
        mode="full",
        passages=["Alpha passage supports Beta"],
        variant="evidence_first",
    )

    assert result["answer"] == "Fallback Beta"
    assert calls == {"context_simple": 1, "aga": 0}
    assert result["history"][0]["step"] == "no_triples_fallback"
    assert result["diagnostics"]["evidence_first_fallback_steps"] == ["no_triples_fallback"]
    assert not result["diagnostics"]["evidence_first_chain_complete"]


def test_ablation_without_reader_context_disables_passage_reader_and_postprocess(monkeypatch):
    pipeline = _load_pipeline()

    calls = {"context": 0, "postprocess": 0}

    class Checker:
        last_answer_type = "entity"

        def check_evidence_chain_from_strings(self, triples, question):
            return True, None, triples

    monkeypatch.setattr(pipeline, "EvidenceChainChecker", Checker)
    monkeypatch.setattr(pipeline, "run_qda", lambda ctx: ctx.setdefault("subqueries", [ctx["question"]]))
    monkeypatch.setattr(pipeline, "run_gra", lambda ctx: ctx.update(triples=["Alpha -- bridge --> Beta"]))
    monkeypatch.setattr(pipeline, "run_aga", lambda ctx: ctx.update(answer="Beta") or "Beta")

    def fake_context(ctx):
        calls["context"] += 1
        return "Context Beta"

    def fake_postprocess(*args, **kwargs):
        calls["postprocess"] += 1
        return {"answer": "Post Beta", "source": "postprocess", "used": True}

    monkeypatch.setattr(pipeline, "run_context_answer", fake_context)
    monkeypatch.setattr(pipeline, "run_answer_postprocessor", fake_postprocess)

    result = pipeline.run_pipeline(
        "What connects Alpha?",
        _graph(),
        mode="full",
        passages=["Alpha passage"],
        variant="evidence_first",
        ablation="without_reader_context",
    )

    assert result["answer"] == "Beta"
    assert calls == {"context": 0, "postprocess": 0}
    assert result["diagnostics"]["evidence_first_ablation"] == "without_reader_context"
    assert result["diagnostics"]["evidence_first_reader_context_disabled"]
    assert not result["diagnostics"]["evidence_first_postprocess_used"]


def test_run_aga_does_not_include_passages_when_reader_context_is_disabled(monkeypatch):
    agents = _load_agents()
    prompts = []

    def fake_llm_call(prompt, max_tokens=800, retries=4):
        prompts.append(prompt)
        return "FINAL ANSWER: Beta"

    monkeypatch.setattr(agents, "llm_call", fake_llm_call)

    ctx = {
        "question": "What connects Alpha?",
        "variant": "evidence_first",
        "disable_reader_context": True,
        "triples": ["Alpha -- bridge --> Beta"],
        "evidence_chain": ["Alpha -- bridge --> Beta"],
        "passages": ["Alpha passage should be hidden from the reader"],
    }

    assert agents.run_aga(ctx) == "Beta"
    assert len(prompts) == 1
    assert "Alpha passage should be hidden" not in prompts[0]
    assert "Passages:" not in prompts[0]
