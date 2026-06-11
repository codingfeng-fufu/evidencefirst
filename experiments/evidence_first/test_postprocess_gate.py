"""Regression tests for EvidenceFirst final-answer selection."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comagraag.agents import _canonicalize_postprocess_answer, _clean_final_answer
from comagraag.pipeline import (
    _looks_bad_answer,
    _select_answer_dual_path,
    _should_select_answer_postprocess,
)


def test_search_control_phrase_is_bad_answer():
    assert _looks_bad_answer("Let’s search again")


def test_clean_final_answer_extracts_acronym_from_definition():
    assert _clean_final_answer("The award won by only twelve people is the EGOT") == "EGOT"


def test_clean_final_answer_keeps_last_newline_answer_segment():
    assert _clean_final_answer("Gaydar.\nHayley Williams") == "Hayley Williams"


def test_clean_final_answer_keeps_last_sentence_answer_segment():
    assert _clean_final_answer("Epitaph Records.  Motion City Soundtrack") == "Motion City Soundtrack"


def test_clean_final_answer_keeps_last_newline_comparison_option():
    assert _clean_final_answer("ISIL.\nOperation Diadem") == "Operation Diadem"


def test_postprocess_selected_when_current_is_bad_and_post_is_canonical():
    assert _should_select_answer_postprocess(
        current_answer="Let’s search again",
        post_answer="EGOT",
        answer_type="award",
    )


def test_postprocess_selected_for_person_name_expansion():
    assert _should_select_answer_postprocess(
        current_answer="Dwight Schultz",
        post_answer="William Dwight Schultz",
        answer_type="person",
    )


def test_postprocess_selected_for_comparison_name_expansion():
    assert _should_select_answer_postprocess(
        current_answer="Edward Keonjian",
        post_answer="Dr. Edward Keonjian",
        answer_type="comparison",
    )


def test_postprocess_selected_when_non_yesno_question_has_yesno_answer():
    assert _should_select_answer_postprocess(
        current_answer="no",
        post_answer="alcoholic",
        answer_type="entity",
    )


def test_postprocess_selected_when_current_is_single_control_token():
    assert _should_select_answer_postprocess(
        current_answer="Wait",
        post_answer="Sapsali",
        answer_type="entity",
    )


def test_postprocess_not_selected_for_yesno_flip():
    assert not _should_select_answer_postprocess(
        current_answer="no",
        post_answer="yes",
        answer_type="yesno",
    )


def test_postprocess_not_selected_for_truncated_parenthetical_answer():
    assert not _should_select_answer_postprocess(
        current_answer="No",
        post_answer="Marc Predka (Tha Tradem",
        answer_type="profession",
    )


def test_population_choice_answer_keeps_supported_statement():
    question = "Did Qionghai or Suining have a population of 658,798 in 2002?"
    passages = [
        "Suining: Suining is a prefecture-level city. In 2002, Suining had a population of 658,798.",
    ]

    assert (
        _canonicalize_postprocess_answer(question, "Suining", passages, "choice")
        == "In 2002, Suining had a population of 658,798."
    )


def test_person_answer_expands_to_full_name_from_passages():
    passages = [
        "Dwight Schultz: William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.",
    ]

    assert (
        _canonicalize_postprocess_answer(
            "Who voices Chef Mung Daal?",
            "Dwight Schultz",
            passages,
            "person",
        )
        == "William Dwight Schultz"
    )


def test_comparison_person_answer_keeps_title_from_passages():
    passages = [
        "Edward Keonjian: Dr. Edward Keonjian (14 August 1909 - 6 September 1999) was a prominent engineer.",
    ]

    assert (
        _canonicalize_postprocess_answer(
            "Who was born first Edward Keonjian or Aram Saroyan?",
            "Edward Keonjian",
            passages,
            "comparison",
        )
        == "Dr. Edward Keonjian"
    )


def test_dual_path_selector_handles_missing_kg_score():
    result = _select_answer_dual_path(
        {"answer": "alpha", "score": None},
        {"answer": "beta", "confidence": 0.8},
        "What is the answer?",
    )

    assert result["answer"] == "beta"
