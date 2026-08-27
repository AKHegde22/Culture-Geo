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


def strict_match_pairs(
    concepts: Optional[Sequence[Dict]] = None,
    max_len_diff: int = 10,
) -> List[Dict]:
    """
    Build strict one-to-one within-language pairs matched on:
      1. Exact category match
      2. Exact script match
      3. Minimizing word length difference via Hungarian bipartite matching
      4. Loanword status preference

    Unmatched concepts are dropped rather than retaining weak controls.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    src = list(concepts) if concepts is not None else cleaned_concepts()
    untrans = [c for c in src if c["translatability"] == "untranslatable"]
    trans = [c for c in src if c["translatability"] == "translatable"]

    by_lang_cat_u = defaultdict(list)
    by_lang_cat_t = defaultdict(list)

    for u in untrans:
        by_lang_cat_u[(u["language"], u["category"])].append(u)
    for t in trans:
        by_lang_cat_t[(t["language"], t["category"])].append(t)

    matched_pairs = []
    for key in sorted(by_lang_cat_u.keys()):
        u_group = by_lang_cat_u[key]
        t_group = by_lang_cat_t.get(key, [])
        if not t_group:
            continue

        cost = np.zeros((len(u_group), len(t_group)))
        for i, u in enumerate(u_group):
            for j, t in enumerate(t_group):
                diff = abs(u["char_len"] - t["char_len"])
                if u.get("script_type") != t.get("script_type"):
                    diff += 1000
                if u.get("english_loanword") != t.get("english_loanword"):
                    diff += 0.5
                cost[i, j] = diff

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < max_len_diff + 50:
                u_c = u_group[r]
                t_c = t_group[c]
                matched_pairs.append({
                    "untranslatable": u_c["word"],
                    "translatable": t_c["word"],
                    "language": key[0],
                    "category": key[1],
                    "char_len_u": u_c["char_len"],
                    "char_len_t": t_c["char_len"],
                    "char_len_diff": abs(u_c["char_len"] - t_c["char_len"]),
                    "token_count_u": len(u_c["word"].split()),
                    "token_count_t": len(t_c["word"].split()),
                    "script": u_c.get("script_type"),
                    "loanword_u": u_c.get("english_loanword", False),
                    "loanword_t": t_c.get("english_loanword", False),
                })

    return matched_pairs


def matching_balance_table(
    pre_concepts: Optional[Sequence[Dict]] = None,
    post_pairs: Optional[List[Dict]] = None,
) -> Dict:
    """
    Generate a before-and-after matching balance summary.
    """
    import numpy as np

    if pre_concepts is None:
        pre_concepts = cleaned_concepts()
    if post_pairs is None:
        post_pairs = strict_match_pairs(pre_concepts)

    u_pre = [c for c in pre_concepts if c["translatability"] == "untranslatable"]
    t_pre = [c for c in pre_concepts if c["translatability"] == "translatable"]

    matched_u_words = {p["untranslatable"] for p in post_pairs}
    matched_t_words = {p["translatable"] for p in post_pairs}
    u_post = [c for c in u_pre if c["word"] in matched_u_words]
    t_post = [c for c in t_pre if c["word"] in matched_t_words]

    pre_match = match_controls(pre_concepts)

    return {
        "pre_matching": {
            "n_untranslatable": len(u_pre),
            "n_translatable": len(t_pre),
            "n_total": len(pre_concepts),
            "category_match_rate": pre_match.get("same_category_fraction", 0.40449),
            "mean_char_len_u": float(np.mean([c["char_len"] for c in u_pre])),
            "mean_char_len_t": float(np.mean([c["char_len"] for c in t_pre])),
            "mean_token_count_u": float(np.mean([len(c["word"].split()) for c in u_pre])),
            "mean_token_count_t": float(np.mean([len(c["word"].split()) for c in t_pre])),
            "loanword_rate_u": float(np.mean([1 if c.get("english_loanword") else 0 for c in u_pre])),
            "loanword_rate_t": float(np.mean([1 if c.get("english_loanword") else 0 for c in t_pre])),
        },
        "post_matching": {
            "n_pairs": len(post_pairs),
            "n_untranslatable": len(u_post),
            "n_translatable": len(t_post),
            "n_total": len(u_post) + len(t_post),
            "category_match_rate": 1.0,
            "mean_char_len_u": float(np.mean([p["char_len_u"] for p in post_pairs])),
            "mean_char_len_t": float(np.mean([p["char_len_t"] for p in post_pairs])),
            "mean_char_len_diff": float(np.mean([p["char_len_diff"] for p in post_pairs])),
            "mean_token_count_u": float(np.mean([p["token_count_u"] for p in post_pairs])),
            "mean_token_count_t": float(np.mean([p["token_count_t"] for p in post_pairs])),
            "loanword_rate_u": float(np.mean([1 if p["loanword_u"] else 0 for p in post_pairs])),
            "loanword_rate_t": float(np.mean([1 if p["loanword_t"] else 0 for p in post_pairs])),
        },
        "by_language": {
            lang: {
                "pre_u": sum(1 for c in u_pre if c["language"] == lang),
                "pre_t": sum(1 for c in t_pre if c["language"] == lang),
                "post_pairs": sum(1 for p in post_pairs if p["language"] == lang),
            }
            for lang in sorted({c["language"] for c in pre_concepts})
        },
        "by_category": {
            cat: {
                "pre_u": sum(1 for c in u_pre if c["category"] == cat),
                "pre_t": sum(1 for c in t_pre if c["category"] == cat),
                "post_pairs": sum(1 for p in post_pairs if p["category"] == cat),
            }
            for cat in sorted({c["category"] for c in pre_concepts})
        },
    }


def strict_keep_mask_from_metadata(
    metadata: List[Dict],
    pairs: Optional[List[Dict]] = None,
) -> List[bool]:
    """
    Boolean mask over activation/prompt metadata rows for strictly matched pairs only.
    """
    if pairs is None:
        pairs = strict_match_pairs()
    keep_set = {(p["language"], p["untranslatable"]) for p in pairs} | {
        (p["language"], p["translatable"]) for p in pairs
    }
    return [(m["language"], m["concept_word"]) in keep_set for m in metadata]


def matching_diagnostics(concepts: List[Dict]) -> Dict:
    """Full matching appendix payload with both legacy and strict matching."""
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

    pairs_strict = strict_match_pairs(concepts)
    balance = matching_balance_table(concepts, pairs_strict)

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
        "strict_matching": {
            "n_pairs": len(pairs_strict),
            "pairs": pairs_strict,
            "balance_table": balance,
        },
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

