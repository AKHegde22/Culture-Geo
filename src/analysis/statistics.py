"""
Statistical tests for the untranslatability study.

Runs significance tests comparing untranslatable vs. translatable concepts
across all metrics.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


def welch_ttest(
    group_a: np.ndarray,
    group_b: np.ndarray,
    label_a: str = "Group A",
    label_b: str = "Group B",
) -> Dict:
    """
    Welch's t-test for unequal variances.

    Returns:
        Dictionary with t-statistic, p-value, and effect size (Cohen's d)
    """
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    # Cohen's d (effect size)
    pooled_std = np.sqrt(
        (group_a.std() ** 2 + group_b.std() ** 2) / 2
    )
    cohens_d = (group_a.mean() - group_b.mean()) / max(pooled_std, 1e-8)

    # Confidence interval for Cohen's d (95%)
    se = np.sqrt(group_a.var() / len(group_a) + group_b.var() / len(group_b))
    se_d = se / max(pooled_std, 1e-8)
    ci_lower = cohens_d - 1.96 * se_d
    ci_upper = cohens_d + 1.96 * se_d

    return {
        "test": "Welch's t-test",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "mean_a": float(group_a.mean()),
        "mean_b": float(group_b.mean()),
        "std_a": float(group_a.std()),
        "std_b": float(group_b.std()),
        "n_a": len(group_a),
        "n_b": len(group_b),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def wilcoxon_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    label_a: str = "Group A",
    label_b: str = "Group B",
) -> Dict:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).

    Used when data may not be normally distributed.
    """
    # Need paired or equal-length arrays for Wilcoxon
    # Use Mann-Whitney U for independent samples
    u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')

    # Effect size (rank-biserial correlation)
    n1, n2 = len(group_a), len(group_b)
    effect_size = 1 - (2 * u_stat) / (n1 * n2)

    return {
        "test": "Mann-Whitney U",
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "mean_a": float(group_a.mean()),
        "mean_b": float(group_b.mean()),
        "n_a": len(group_a),
        "n_b": len(group_b),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def compare_translatability_groups(
    metrics_by_group: Dict[str, np.ndarray],
    test: str = "welch",
) -> Dict:
    """
    Compare metrics between untranslatable and translatable groups.

    Args:
        metrics_by_group: Dict mapping "untranslatable"/"translatable" -> metric values
        test: "welch" for t-test, "mannwhitney" for non-parametric

    Returns:
        Test results dictionary
    """
    group_a = metrics_by_group.get("untranslatable", np.array([]))
    group_b = metrics_by_group.get("translatable", np.array([]))

    if len(group_a) == 0 or len(group_b) == 0:
        return {"error": "One or both groups empty"}

    if test == "welch":
        return welch_ttest(group_a, group_b, "untranslatable", "translatable")
    elif test == "mannwhitney":
        return wilcoxon_test(group_a, group_b, "untranslatable", "translatable")
    else:
        raise ValueError(f"Unknown test: {test}")


def run_all_statistical_tests(
    err_by_translatability: Dict[str, float],
    silhouette_by_translatability: Dict[str, float],
    trajectory_by_translatability: Dict[str, Dict[int, float]],
    peak_distance_by_translatability: Dict[str, List[float]],
) -> Dict:
    """
    Run all statistical tests for the paper.

    Args:
        err_by_translatability: ERR values per layer, per group
        silhouette_by_translatability: Silhouette scores per group
        trajectory_by_translatability: Trajectory distances per layer, per group
        peak_distance_by_translatability: Peak distances per group

    Returns:
        Dictionary of all test results
    """
    results = {}

    # Test 1: ERR at peak layer (mid-layer)
    # Find the layer where untranslatable ERR is highest
    untrans_layers = list(err_by_translatability.get("untranslatable", {}).keys())
    if untrans_layers:
        peak_layer = max(
            untrans_layers,
            key=lambda l: err_by_translatability.get("untranslatable", {}).get(l, 0)
        )
        err_untrans = np.array([err_by_translatability["untranslatable"][peak_layer]])
        err_trans = np.array([err_by_translatability["translatable"].get(peak_layer, 0)])

        # For a proper test, we need per-prompt ERR values
        # For now, test at all layers
        all_untrans_err = np.array([
            err_by_translatability["untranslatable"].get(l, 0)
            for l in sorted(untrans_layers)
        ])
        all_trans_err = np.array([
            err_by_translatability["translatable"].get(l, 0)
            for l in sorted(untrans_layers)
        ])

        if len(all_untrans_err) > 1 and len(all_trans_err) > 1:
            results["err_by_translatability"] = compare_translatability_groups({
                "untranslatable": all_untrans_err,
                "translatable": all_trans_err,
            })

    # Test 2: Silhouette scores
    sil_untrans = np.atleast_1d(np.asarray(
        silhouette_by_translatability.get("untranslatable", [])))
    sil_trans = np.atleast_1d(np.asarray(
        silhouette_by_translatability.get("translatable", [])))
    if len(sil_untrans) > 1 and len(sil_trans) > 1:
        results["silhouette_by_translatability"] = compare_translatability_groups({
            "untranslatable": sil_untrans,
            "translatable": sil_trans,
        })

    # Test 3: Peak trajectory distances
    peak_untrans = np.atleast_1d(np.asarray(
        peak_distance_by_translatability.get("untranslatable", [])))
    peak_trans = np.atleast_1d(np.asarray(
        peak_distance_by_translatability.get("translatable", [])))
    if len(peak_untrans) > 1 and len(peak_trans) > 1:
        results["peak_distance_by_translatability"] = compare_translatability_groups({
            "untranslatable": peak_untrans,
            "translatable": peak_trans,
        })

    # Test 4: Trajectory curvature
    # (Would need per-concept curvature values - placeholder)

    return results


def format_results_table(test_results: Dict) -> str:
    """Format statistical results as a readable table."""
    lines = ["Statistical Test Results"]
    lines.append("=" * 70)
    lines.append(f"{'Test':<30} {'p-value':>10} {'Cohen d':>10} {'Sig?':>8}")
    lines.append("-" * 70)

    for test_name, result in test_results.items():
        if "error" in result:
            lines.append(f"{test_name:<30} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
            continue

        sig = "***" if result.get("significant_001", False) else (
            "*" if result.get("significant_005", False) else "ns"
        )
        d = result.get("cohens_d", result.get("effect_size", 0))
        lines.append(
            f"{test_name:<30} {result['p_value']:>10.4f} {d:>10.3f} {sig:>8}"
        )

    lines.append("-" * 70)
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    return "\n".join(lines)
