"""
Prompt templates for probing LLM representations of cultural concepts.

Three template types:
    1. DEFINITION: Asks the model to define the concept (tests knowledge retrieval)
    2. USAGE: Asks the model to use the concept in a sentence (tests generation)
    3. COMPARISON: Asks the model to explain to an English speaker (tests translation reasoning)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random


@dataclass
class PromptTemplate:
    """A single prompt template."""
    name: str
    template: str
    description: str

    def format(self, word: str, language: str, native_script: str, **kwargs) -> str:
        return self.template.format(
            word=word,
            language=language,
            native_script=native_script,
            **kwargs,
        )


# ─── Template Definitions ─────────────────────────────────────────────────

DEFINITION_TEMPLATE = PromptTemplate(
    name="definition",
    template=(
        "In {language}, the word \"{word}\" "
        "({native_script}) means"
    ),
    description="Probes whether the model knows the meaning of the concept",
)

USAGE_TEMPLATE = PromptTemplate(
    name="usage",
    template=(
        "Write a sentence in {language} using the word \"{word}\" "
        "({native_script}) in its proper cultural context."
    ),
    description="Probes whether the model can generate culturally appropriate usage",
)

COMPARISON_TEMPLATE = PromptTemplate(
    name="comparison",
    template=(
        "Explain the {language} concept \"{word}\" "
        "({native_script}) to someone who only speaks English. "
        "What is the closest English expression?"
    ),
    description="Probes how the model routes untranslatable concepts through English",
)

ALL_TEMPLATES = [DEFINITION_TEMPLATE, USAGE_TEMPLATE, COMPARISON_TEMPLATE]


@dataclass
class Prompt:
    """A fully formatted prompt with metadata."""
    text: str
    template_name: str
    concept_word: str
    language: str
    language_code: str
    translatability: str
    category: str
    concept_index: int


def generate_prompts(
    concepts: List[Dict],
    templates: Optional[List[PromptTemplate]] = None,
    seed: int = 42,
) -> List[Prompt]:
    """
    Generate formatted prompts for all concepts and templates.

    Args:
        concepts: List of concept dictionaries from concepts.py
        templates: List of PromptTemplate objects (defaults to ALL_TEMPLATES)
        seed: Random seed for reproducibility

    Returns:
        List of Prompt objects with full metadata
    """
    if templates is None:
        templates = ALL_TEMPLATES

    prompts = []
    for idx, concept in enumerate(concepts):
        for template in templates:
            text = template.format(
                word=concept["word"],
                language=concept["language"],
                native_script=concept.get("native_script", concept["word"]),
            )
            prompts.append(Prompt(
                text=text,
                template_name=template.name,
                concept_word=concept["word"],
                language=concept["language"],
                language_code=concept["language_code"],
                translatability=concept["translatability"],
                category=concept["category"],
                concept_index=idx,
            ))

    random.seed(seed)
    random.shuffle(prompts)

    return prompts


def get_prompts_by_language(
    prompts: List[Prompt],
    language_code: str,
) -> List[Prompt]:
    """Filter prompts by language code."""
    return [p for p in prompts if p.language_code == language_code]


def get_prompts_by_translatability(
    prompts: List[Prompt],
    translatability: str,
) -> List[Prompt]:
    """Filter prompts by translatability status."""
    return [p for p in prompts if p.translatability == translatability]


def get_prompts_by_template(
    prompts: List[Prompt],
    template_name: str,
) -> List[Prompt]:
    """Filter prompts by template name."""
    return [p for p in prompts if p.template_name == template_name]


def summarize_prompts(prompts: List[Prompt]) -> Dict:
    """Return summary statistics for a set of prompts."""
    by_language = {}
    by_translatability = {}
    by_template = {}
    by_category = {}

    for p in prompts:
        by_language[p.language_code] = by_language.get(p.language_code, 0) + 1
        by_translatability[p.translatability] = by_translatability.get(p.translatability, 0) + 1
        by_template[p.template_name] = by_template.get(p.template_name, 0) + 1
        by_category[p.category] = by_category.get(p.category, 0) + 1

    return {
        "total": len(prompts),
        "by_language": by_language,
        "by_translatability": by_translatability,
        "by_template": by_template,
        "by_category": by_category,
    }
