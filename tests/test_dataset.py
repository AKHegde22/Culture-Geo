"""Tests for dataset construction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.concepts import UNTRANSLATABLE_CONCEPTS, TRANSLATABLE_CONCEPTS
from src.dataset.prompts import generate_prompts, summarize_prompts


def test_concepts_loaded():
    """Verify concepts are loaded."""
    assert len(UNTRANSLATABLE_CONCEPTS) > 50
    assert len(TRANSLATABLE_CONCEPTS) > 50


def test_concepts_have_required_fields():
    """Verify all concepts have required fields."""
    for concept in UNTRANSLATABLE_CONCEPTS + TRANSLATABLE_CONCEPTS:
        assert "word" in concept
        assert "language" in concept
        assert "language_code" in concept
        assert "meaning" in concept
        assert "translatability" in concept


def test_prompts_generation():
    """Verify prompt generation works."""
    concepts = UNTRANSLATABLE_CONCEPTS[:5] + TRANSLATABLE_CONCEPTS[:5]
    prompts = generate_prompts(concepts)
    assert len(prompts) == len(concepts) * 3  # 3 templates per concept


def test_prompt_metadata():
    """Verify prompts have correct metadata."""
    concepts = UNTRANSLATABLE_CONCEPTS[:5]
    prompts = generate_prompts(concepts)
    for p in prompts:
        assert p.language == concepts[0]["language"]
        assert p.translatability == "untranslatable"
        assert len(p.text) > 10


def test_prompt_summary():
    """Verify prompt summary statistics."""
    concepts = UNTRANSLATABLE_CONCEPTS[:5] + TRANSLATABLE_CONCEPTS[:5]
    prompts = generate_prompts(concepts)
    summary = summarize_prompts(prompts)
    assert summary["total"] == len(prompts)
    assert summary["by_translatability"]["untranslatable"] == 15  # 5 concepts * 3 templates
    assert summary["by_translatability"]["translatable"] == 15
