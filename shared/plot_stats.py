"""Shared statistical annotation helpers for distribution plots."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests


def significance_stars(p_value: float) -> str:
    """Return the standard star annotation for one p-value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def hue_offsets(hue_levels: list[str] | tuple[str, ...], group_width: float = 0.8) -> dict[str, float]:
    """Return horizontal offsets for seaborn hue groups."""
    if not hue_levels:
        return {}
    offsets = np.linspace(-group_width / 2, group_width / 2, len(hue_levels), endpoint=True)
    return dict(zip(hue_levels, offsets))


def box_centers(
    group_order: list[str] | tuple[str, ...],
    *,
    hue_levels: list[str] | tuple[str, ...] | None = None,
    group_width: float = 0.8,
) -> dict[tuple[str, str | None], float]:
    """Return x centers for each group and optional hue level."""
    hue_levels = list(hue_levels or [None])
    offsets = hue_offsets(hue_levels, group_width=group_width) if hue_levels != [None] else {None: 0.0}
    centers = {}
    for group_index, group_name in enumerate(group_order):
        for hue_name in hue_levels:
            centers[(group_name, hue_name)] = group_index + offsets[hue_name]
    return centers


def paired_or_unpaired_test(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    value_col: str,
    pair_cols: list[str] | tuple[str, ...] | None = None,
    test_mode: str = "auto",
) -> tuple[float, str] | None:
    """Run Wilcoxon for matched samples when possible, otherwise MWU."""
    if test_mode not in {"auto", "paired", "unpaired"}:
        raise ValueError(f"Unsupported test_mode: {test_mode}")

    if test_mode != "unpaired" and pair_cols:
        left = left_df[list(pair_cols) + [value_col]].rename(columns={value_col: "left"})
        right = right_df[list(pair_cols) + [value_col]].rename(columns={value_col: "right"})
        merged = left.merge(right, on=list(pair_cols), how="inner")
        if len(merged) >= 2:
            try:
                return float(wilcoxon(merged["left"], merged["right"]).pvalue), "wilcoxon"
            except ValueError:
                if test_mode == "paired":
                    return None

    if len(left_df) < 2 or len(right_df) < 2:
        return None
    return float(mannwhitneyu(left_df[value_col], right_df[value_col], alternative="two-sided").pvalue), "mannwhitneyu"


def pairwise_group_tests(
    panel_df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    group_order: list[str] | tuple[str, ...],
    hue_col: str | None = None,
    hue_order: list[str] | tuple[str, ...] | None = None,
    pair_cols: list[str] | tuple[str, ...] | None = None,
    test_mode: str = "auto",
) -> pd.DataFrame:
    """Run pairwise group comparisons within one panel and Bonferroni-correct them."""
    hue_levels = list(hue_order or ([None] if hue_col is None else panel_df[hue_col].dropna().unique().tolist()))
    rows: list[dict[str, object]] = []

    for hue_name in hue_levels:
        hue_df = panel_df if hue_col is None else panel_df[panel_df[hue_col] == hue_name]
        present_groups = [group_name for group_name in group_order if group_name in hue_df[group_col].unique()]
        for left_group, right_group in combinations(present_groups, 2):
            left_df = hue_df[hue_df[group_col] == left_group]
            right_df = hue_df[hue_df[group_col] == right_group]
            result = paired_or_unpaired_test(left_df, right_df, value_col, pair_cols=pair_cols, test_mode=test_mode)
            if result is None:
                continue
            p_value, test_name = result
            rows.append(
                {
                    "hue": hue_name,
                    "group_left": left_group,
                    "group_right": right_group,
                    "p_raw": p_value,
                    "test": test_name,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["hue", "group_left", "group_right", "p_raw", "p_corrected", "test"])

    result_df = pd.DataFrame(rows)
    result_df["p_corrected"] = multipletests(result_df["p_raw"], method="bonferroni")[1]
    return result_df.sort_values(["hue", "p_corrected", "group_left", "group_right"]).reset_index(drop=True)


def pairwise_hue_tests_within_group(
    panel_df: pd.DataFrame,
    *,
    group_col: str,
    group_order: list[str] | tuple[str, ...],
    hue_col: str,
    hue_order: list[str] | tuple[str, ...],
    value_col: str,
    pair_cols: list[str] | tuple[str, ...] | None = None,
    test_mode: str = "auto",
) -> pd.DataFrame:
    """Run pairwise hue comparisons within each group and Bonferroni-correct them."""
    rows: list[dict[str, object]] = []
    for group_name in group_order:
        group_df = panel_df[panel_df[group_col] == group_name]
        present_hues = [hue_name for hue_name in hue_order if hue_name in group_df[hue_col].unique()]
        for left_hue, right_hue in combinations(present_hues, 2):
            left_df = group_df[group_df[hue_col] == left_hue]
            right_df = group_df[group_df[hue_col] == right_hue]
            result = paired_or_unpaired_test(left_df, right_df, value_col, pair_cols=pair_cols, test_mode=test_mode)
            if result is None:
                continue
            p_value, test_name = result
            rows.append(
                {
                    "group": group_name,
                    "hue_left": left_hue,
                    "hue_right": right_hue,
                    "p_raw": p_value,
                    "test": test_name,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["group", "hue_left", "hue_right", "p_raw", "p_corrected", "test"])

    result_df = pd.DataFrame(rows)
    result_df["p_corrected"] = multipletests(result_df["p_raw"], method="bonferroni")[1]
    return result_df.sort_values(["p_corrected", "group", "hue_left", "hue_right"]).reset_index(drop=True)


def add_pairwise_annotations(
    ax,
    stats_df: pd.DataFrame,
    *,
    centers: dict[tuple[str, str | None], float],
    data_max: float,
    data_min: float,
    line_height_scale: float = 0.03,
) -> None:
    """Draw significance brackets above boxplots."""
    if stats_df.empty:
        return

    sig_df = stats_df[stats_df["p_corrected"] < 0.05]
    if sig_df.empty:
        return

    data_range = data_max - data_min
    step = line_height_scale * (data_range if data_range > 0 else 1.0)
    base_y = data_max + step

    for offset_index, row in enumerate(sig_df.itertuples(index=False), start=1):
        x1 = centers[(row.group_left, row.hue)]
        x2 = centers[(row.group_right, row.hue)]
        bracket_y = base_y + step * (offset_index - 1)
        star_text = significance_stars(float(row.p_corrected))
        ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + step / 2, bracket_y + step / 2, bracket_y], lw=1.2, c="black", clip_on=False)
        ax.text((x1 + x2) / 2, bracket_y + step / 2, star_text, ha="center", va="bottom", fontsize=22)


def add_within_group_hue_annotations(
    ax,
    stats_df: pd.DataFrame,
    *,
    centers: dict[tuple[str, str | None], float],
    data_max: float,
    data_min: float,
    line_height_scale: float = 0.03,
) -> None:
    """Draw significance brackets for hue comparisons within each x-axis group."""
    if stats_df.empty:
        return

    sig_df = stats_df[stats_df["p_corrected"] < 0.05]
    if sig_df.empty:
        return

    data_range = data_max - data_min
    step = line_height_scale * (data_range if data_range > 0 else 1.0)
    group_offsets: dict[str, int] = {}

    for row in sig_df.itertuples(index=False):
        x1 = centers[(row.group, row.hue_left)]
        x2 = centers[(row.group, row.hue_right)]
        offset = group_offsets.get(row.group, 0)
        bracket_y = data_max + step * (offset + 1)
        group_offsets[row.group] = offset + 1
        star_text = significance_stars(float(row.p_corrected))
        ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + step / 2, bracket_y + step / 2, bracket_y], lw=1.2, c="black", clip_on=False)
        ax.text((x1 + x2) / 2, bracket_y + step / 2, star_text, ha="center", va="bottom", fontsize=22)
