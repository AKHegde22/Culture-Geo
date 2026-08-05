"""
Dataset audit and control-matching diagnostics for Culture-Geo.

Drops clearly invalid / over-literal "untranslatable" labels and produces
language x category matching diagnostics. The cleaned concept set is frozen
before generation or probe significance tests.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from src.dataset.concepts import ALL_CONCEPTS, TRANSLATABLE_CONCEPTS, UNTRANSLATABLE_CONCEPTS

# (language, word) pairs removed from the untranslatable set.
# Reasons are recorded for the paper matching appendix.
EXCLUSIONS_UNTRANSLATABLE: Dict[Tuple[str, str], str] = {
    ("Japanese", "Kairos"): "Greek loanword mislabeled as Japanese untranslatable",
    ("Japanese", "Wabi"): "Redundant subset of Wabi-sabi",
    ("Hindi", "Dil"): "Direct English equivalent (heart)",
    ("Hindi", "Dosti"): "Direct English equivalent (friendship)",
    ("Hindi", "Filhal"): "Near-direct English equivalent (for now / currently)",
    ("Portuguese", "Despedida"): "Direct English equivalent (farewell)",
    ("Korean", "Gyeolhon"): "Direct English equivalent (marriage)",
    ("Russian", "Davai"): "Colloquial discourse marker with clear English glosses",
    ("Danish", "Lykke"): "Direct English equivalent (happiness)",
    ("Spanish", "Tutear"): "Has a clear English gloss (address with informal tu)",
    ("Arabic", "Taza"): "Near-direct English equivalent (freshness / fresh)",
    ("Portuguese", "Ilhado"): "Near-direct English equivalent (stranded / isolated)",
    ("Korean", "Jeongseon"): "Poorly attested / ambiguous lexical status",
    ("Korean", "Dinganeun"): "Poorly attested lexical item",
    ("German", "Zeitgeist"): "Fully borrowed into English dictionaries as Zeitgeist",
}

# Concepts that entered English as loanwords but remain culturally marked.
# Kept in the set; flagged in matching diagnostics.
ENGLISH_LOANWORD_FLAGS: Dict[Tuple[str, str], str] = {
    ("German", "Schadenfreude"): "Attested English loanword",
    ("Danish", "Hygge"): "Widely borrowed into English lifestyle discourse",
    ("Japanese", "Ikigai"): "Popularized as an English lifestyle loanword",
    ("German", "Weltschmerz"): "Attested literary English loanword",
    ("Portuguese", "Saudade"): "Occasionally borrowed; still primarily Portuguese",
}


def _key(c: Dict) -> Tuple[str, str]:
    return (c["language"], c["word"])


def script_type(native_script: str, language: str) -> str:
    """Coarse script family for matching diagnostics."""
    if not native_script:
        return "latin"
    sample = native_script[:8]
    for ch in sample:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF:
            return "cjk"
        if 0xAC00 <= o <= 0xD7AF:
            return "hangul"
        if 0x0600 <= o <= 0x06FF:
            return "arabic"
        if 0x0400 <= o <= 0x04FF:
            return "cyrillic"
        if 0x0900 <= o <= 0x097F:
            return "devanagari"
    return "latin"


def is_excluded(concept: Dict) -> bool:
    return _key(concept) in EXCLUSIONS_UNTRANSLATABLE


def cleaned_concepts(
    concepts: Optional[Sequence[Dict]] = None,
) -> List[Dict]:
    """Return concepts with invalid untranslatable labels removed."""
    src = list(concepts) if concepts is not None else list(ALL_CONCEPTS)
    out = []
    for c in src:
        if c.get("translatability") == "untranslatable" and is_excluded(c):
            continue
        # Also drop accidental Spanish Tutear if somehow in Korean list (handled above)
        enriched = dict(c)
        k = _key(c)
        if k in ENGLISH_LOANWORD_FLAGS:
            enriched["english_loanword"] = True
            enriched["loanword_note"] = ENGLISH_LOANWORD_FLAGS[k]
        else:
            enriched["english_loanword"] = False
        enriched["script_type"] = script_type(
            c.get("native_script", c.get("word", "")), c["language"]
        )
        enriched["char_len"] = len(c.get("word", ""))
        enriched["native_char_len"] = len(c.get("native_script") or c.get("word", ""))
        out.append(enriched)
    return out


def exclusion_report() -> List[Dict]:
    """Human-readable list of dropped concepts."""
    by_word = {(c["language"], c["word"]): c for c in UNTRANSLATABLE_CONCEPTS}
    rows = []
    for (lang, word), reason in sorted(EXCLUSIONS_UNTRANSLATABLE.items()):
        c = by_word.get((lang, word))
        rows.append(
            {
                "language": lang,
                "word": word,
                "reason": reason,
                "was_present": c is not None,
                "category": c.get("category") if c else None,
            }
        )
    return rows


def match_controls(concepts: List[Dict]) -> Dict:
    """
    Greedy within-language matching: prefer same category, else any control.

    Returns pairing stats and per-language coverage.
    """
    untrans = [c for c in concepts if c["translatability"] == "untranslatable"]
    trans = [c for c in concepts if c["translatability"] == "translatable"]

    pairs = []
    unmatched_u = []
    used_t = set()

    # Index controls by language then category
    by_lang_cat: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for t in trans:
        by_lang_cat[t["language"]][t["category"]].append(t)
        by_lang[t["language"]].append(t)

    for u in untrans:
        lang = u["language"]
        cat = u["category"]
        chosen = None
        # same category first
        for t in by_lang_cat[lang].get(cat, []):
            tid = id(t)
            if tid not in used_t:
                chosen = t
                break
        # any same language
        if chosen is None:
            for t in by_lang[lang]:
                tid = id(t)
                if tid not in used_t:
                    chosen = t
                    break
        if chosen is None:
            unmatched_u.append(u["word"])
            continue
        used_t.add(id(chosen))
        pairs.append(
            {
                "untranslatable": u["word"],
                "translatable": chosen["word"],
                "language": lang,
                "untrans_category": u["category"],
                "trans_category": chosen["category"],
                "same_category": u["category"] == chosen["category"],
                "char_len_u": u.get("char_len", len(u["word"])),
                "char_len_t": chosen.get("char_len", len(chosen["word"])),
                "script_u": u.get("script_type"),
                "script_t": chosen.get("script_type"),
            }
        )

    same_cat = sum(1 for p in pairs if p["same_category"])
    return {
        "n_pairs": len(pairs),
        "n_unmatched_untranslatable": len(unmatched_u),
        "unmatched_untranslatable": unmatched_u,
        "same_category_fraction": same_cat / max(len(pairs), 1),
        "pairs": pairs,
        "n_unused_controls": len(trans) - len(used_t),
    }


def matching_diagnostics(concepts: List[Dict]) -> Dict:
    """Full matching appendix payload."""
    untrans = [c for c in concepts if c["translatability"] == "untranslatable"]
    trans = [c for c in concepts if c["translatability"] == "translatable"]

    by_lang = defaultdict(lambda: {"untranslatable": 0, "translatable": 0})
    by_lang_cat = defaultdict(lambda: {"untranslatable": Counter(), "translatable": Counter()})
    for c in concepts:
        by_lang[c["language"]][c["translatability"]] += 1
        by_lang_cat[c["language"]][c["translatability"]][c["category"]] += 1

    char_lens = {
        "untranslatable_mean": sum(c["char_len"] for c in untrans) / max(len(untrans), 1),
        "translatable_mean": sum(c["char_len"] for c in trans) / max(len(trans), 1),
        "untranslatable_native_mean": sum(c["native_char_len"] for c in untrans)
        / max(len(untrans), 1),
        "translatable_native_mean": sum(c["native_char_len"] for c in trans)
        / max(len(trans), 1),
    }

    loanwords = [
        {"language": c["language"], "word": c["word"], "note": c.get("loanword_note")}
        for c in untrans
        if c.get("english_loanword")
    ]

    return {
        "n_concepts": len(concepts),
        "n_untranslatable": len(untrans),
        "n_translatable": len(trans),
        "n_excluded": len([r for r in exclusion_report() if r["was_present"]]),
        "exclusions": exclusion_report(),
        "by_language": {k: dict(v) for k, v in sorted(by_lang.items())},
        "by_language_category": {
            lang: {
                "untranslatable": dict(v["untranslatable"]),
                "translatable": dict(v["translatable"]),
            }
            for lang, v in sorted(by_lang_cat.items())
        },
        "char_length_summary": char_lens,
        "english_loanword_flags": loanwords,
        "control_matching": match_controls(concepts),
        "frozen": True,
        "seed_note": "Cleaned set frozen before generation/probe significance tests.",
    }


def keep_mask_from_metadata(metadata: List[Dict], keep_words_by_lang: Optional[set] = None) -> List[bool]:
    """
    Boolean mask over activation metadata rows for the cleaned concept set.

    Matching is by (language, concept_word).
    """
    if keep_words_by_lang is None:
        keep_words_by_lang = {_key(c) for c in cleaned_concepts()}
    return [
        (m["language"], m["concept_word"]) in keep_words_by_lang for m in metadata
    ]
