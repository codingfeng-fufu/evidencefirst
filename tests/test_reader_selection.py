import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "comagraag"
sys.path.insert(0, str(PKG))


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        return [[0.0] for _ in texts]


def _load_agents_module():
    sys.modules["sentence_transformers"] = types.SimpleNamespace(
        SentenceTransformer=_FakeSentenceTransformer
    )
    spec = importlib.util.spec_from_file_location("comagraag_agents_impl", PKG / "agents.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_pipeline_module():
    fake_agents = types.SimpleNamespace(
        run_aga=lambda ctx: "",
        run_answer_postprocessor=lambda *args, **kwargs: {},
        run_gra=lambda ctx: [],
        run_qda=lambda ctx: [],
        run_context_answer_simple=lambda *args, **kwargs: {"answer": "unknown"},
        run_va=lambda ctx: {"passed": False, "score": 0.0},
    )
    old_agents = sys.modules.get("agents")
    sys.modules["agents"] = fake_agents
    try:
        spec = importlib.util.spec_from_file_location("comagraag_pipeline_impl", PKG / "pipeline.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old_agents is None:
            sys.modules.pop("agents", None)
        else:
            sys.modules["agents"] = old_agents


def test_reader_passage_selection_keeps_small_local_context():
    agents = _load_agents_module()
    passages = [f"Title {i}: sentence {i}" for i in range(10)]

    selected = agents._select_reader_passages("Where did the founder work?", passages, top_k=5)

    assert selected == passages


def test_evidence_first_aga_uses_passages_even_when_many_triples():
    agents = _load_agents_module()
    captured = {}

    def fake_llm_call(prompt, max_tokens=800, retries=4):
        captured["prompt"] = prompt
        return "FINAL ANSWER: ok"

    agents.llm_call = fake_llm_call
    ctx = {
        "variant": "evidence_first",
        "question": "Where did the founder work?",
        "triples": [f"A{i} -- rel --> B{i}" for i in range(20)],
        "passages": ["Relevant title: relevant passage"],
    }

    agents.run_aga(ctx)

    assert "Additional Context Passages:" in captured["prompt"]
    assert "Relevant title: relevant passage" in captured["prompt"]


def test_pipeline_bad_answer_detection_covers_reader_meta_outputs():
    pipeline = _load_pipeline_module()

    for answer in [
        "Looking again",
        "Re-read",
        "Wait",
        "not determinable",
        "未提及",
        "FINAL ANSWER",
        "Passage 6 further states",
    ]:
        assert pipeline._looks_bad_answer(answer)


def test_yesno_postprocess_can_replace_bad_reader_output():
    pipeline = _load_pipeline_module()

    assert pipeline._should_select_answer_postprocess("Re-read", "yes", "yesno")
    assert pipeline._should_select_answer_postprocess("Wait", "no", "yesno")


def test_question_option_canonicalization_expands_truncated_option():
    agents = _load_agents_module()

    answer = agents._canonicalize_postprocess_answer(
        question=(
            "Which film has the director born later, "
            "Best Of The Best 3: No Turning Back or Ven Mi Corazón Te Llama?"
        ),
        answer="Best Of The Best 3",
        passages=[],
        answer_type="entity",
    )

    assert answer == "Best Of The Best 3: No Turning Back"


def test_question_option_canonicalization_preserves_comma_title():
    agents = _load_agents_module()

    answer = agents._canonicalize_postprocess_answer(
        question=(
            "Which film has the director born later, "
            "Occhio, Malocchio, Prezzemolo E Finocchio or Jailhouse Rock?"
        ),
        answer="Occhio, malocchio, prezzemolo e finocchio",
        passages=[],
        answer_type="entity",
    )

    assert answer == "Occhio, Malocchio, Prezzemolo E Finocchio"


def test_question_option_canonicalization_preserves_title_internal_or():
    agents = _load_agents_module()

    answer = agents._canonicalize_postprocess_answer(
        question=(
            "Which film has the director who died earlier, "
            "To See Or Not To See or Prince (1969 Film)?"
        ),
        answer="To See or Not to See",
        passages=[],
        answer_type="entity",
    )

    assert answer == "To See Or Not To See"


def test_person_title_canonicalization_removes_nonessential_rank():
    agents = _load_agents_module()

    assert (
        agents._canonicalize_postprocess_answer(
            question="Who lived longer, Neil Lloyd Macky or Domenico Cosselli?",
            answer="Colonel Neil Lloyd Macky",
            passages=[],
            answer_type="person",
        )
        == "Neil Lloyd Macky"
    )
    assert (
        agents._canonicalize_postprocess_answer(
            question="Who is the paternal grandfather of Margaret Drummond?",
            answer="Sir Malcolm Drummond",
            passages=[],
            answer_type="person",
        )
        == "Sir Malcolm Drummond"
    )


def test_award_canonicalization_removes_year_and_international_modifier():
    agents = _load_agents_module()

    assert (
        agents._canonicalize_postprocess_answer(
            question="Which award did X get?",
            answer="Sanremo Music Festival 2015",
            passages=[],
            answer_type="award",
        )
        == "Sanremo Music Festival"
    )
    assert (
        agents._canonicalize_postprocess_answer(
            question="Which award did X get?",
            answer="International Emmy Award",
            passages=[],
            answer_type="award",
        )
        == "Emmy Award"
    )
