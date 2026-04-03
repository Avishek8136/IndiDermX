from __future__ import annotations

from hyperderm.services.medgemma_chat_service import MedGemmaChatService


def test_clean_answer_falls_back_when_top_candidate_missing_in_text() -> None:
    top = {
        "main_class": "Inflammatory Disorders",
        "sub_class": "Dermatitis",
        "disease": "Contact Dermatitis",
        "score": 2.5,
    }
    candidates = [top, {"disease": "Psoriasis", "score": 1.8}]

    cleaned = MedGemmaChatService._clean_answer(
        "Most likely condition is Onychomycosis due to fungal signs.",
        top,
        candidates=candidates,
        evidence=[],
    )

    assert "Contact Dermatitis" in cleaned
    assert "Onychomycosis" not in cleaned
    assert "Hierarchy:" in cleaned


def test_clean_answer_keeps_consistent_model_output() -> None:
    top = {
        "main_class": "Inflammatory Disorders",
        "sub_class": "Dermatitis",
        "disease": "Contact Dermatitis",
        "score": 4.0,
    }
    candidates = [top]
    text = "Most likely condition from available signals is Contact Dermatitis. Please consult a dermatologist."

    cleaned = MedGemmaChatService._clean_answer(text, top, candidates=candidates, evidence=[])

    assert cleaned == text
