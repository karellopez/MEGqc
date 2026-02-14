"""Multi-sample dataset-level QA comparison plotting for MEGqc derivatives.

This module compares multiple BIDS samples using already computed MEGqc
derivatives (no raw recomputation). It is designed for two common scenarios:
1) Samples that share one or more tasks (task comparison across matched tasks).
2) Samples with different task sets (task-agnostic comparison).

Public entrypoint
-----------------
``make_multi_sample_group_plots_meg_qc(dataset_paths, ...)``
"""

from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import meg_qc
from meg_qc.calculation.meg_qc_pipeline import resolve_output_roots
from meg_qc.plotting.meg_qc_group_plots import (
    CH_TYPES,
    ChTypeAccumulator,
    _combine_accumulators,
    _discover_run_records,
    _load_settings_snapshot,
    _robust_bounds,
    _robust_normalize_array,
    _run_rows_dataframe,
    _update_accumulator_for_run,
)


MAX_SHARED_TASKS = 12
MAX_POINTS_VIOLIN = 3000
MAX_POINTS_SCATTER = 3500
_FIG_TOGGLE_COUNTER = count(1)


@dataclass
class SampleBundle:
    """Container with all information needed for one sample in comparisons."""

    sample_id: str
    dataset_path: str
    derivatives_root: str
    reports_dir: Path
    settings_snapshot: str
    tab_accumulators: Dict[str, ChTypeAccumulator]


def _figure_to_div(fig: Optional[go.Figure]) -> str:
    if fig is None:
        return "<p>No figure is available for this panel.</p>"
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


def _normalize_figure_axis(fig: go.Figure, *, mode: str) -> go.Figure:
    out = go.Figure(fig)
    axis_key = "y" if mode == "y" else "x"
    for trace in out.data:
        if not hasattr(trace, axis_key):
            continue
        vals = getattr(trace, axis_key)
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        norm = _robust_normalize_array(arr)
        if norm.size != arr.size:
            continue
        setattr(trace, axis_key, norm)
    return out


def _figure_block(
    fig: Optional[go.Figure],
    interpretation: str,
    normalized_variant: bool = False,
    norm_mode: str = "y",
) -> str:
    raw_div = _figure_to_div(fig)
    if not normalized_variant or fig is None:
        return (
            "<div class='fig'>"
            + raw_div
            + f"<div class='fig-note'><strong>How to interpret:</strong> {interpretation}</div>"
            + "</div>"
        )

    toggle_id = f"fig-toggle-{next(_FIG_TOGGLE_COUNTER)}"
    norm_fig = _normalize_figure_axis(fig, mode=norm_mode)
    norm_div = _figure_to_div(norm_fig)
    norm_note = (
        "Normalized variant uses robust z-scoring for readability of shape differences. "
        "Use Raw to inspect native units."
    )
    return (
        "<div class='fig'>"
        + f"<div class='fig-switch' data-fig-toggle='{toggle_id}'>"
        + f"<button class='fig-switch-btn active' data-target='{toggle_id}-raw'>Raw</button>"
        + f"<button class='fig-switch-btn' data-target='{toggle_id}-norm'>Normalized</button>"
        + "</div>"
        + f"<div id='{toggle_id}-raw' class='fig-view active'>{raw_div}</div>"
        + f"<div id='{toggle_id}-norm' class='fig-view'>{norm_div}</div>"
        + f"<div class='fig-note'><strong>How to interpret:</strong> {interpretation}</div>"
        + f"<div class='fig-note'><strong>Normalized view:</strong> {norm_note}</div>"
        + "</div>"
    )


def _build_subtabs_html(
    group_id: str,
    tabs: Sequence[Tuple[str, str]],
    *,
    level: int = 1,
) -> str:
    if not tabs:
        return "<p>No panels are available for this section.</p>"
    gid = _sanitize_token(group_id)
    buttons = []
    panels = []
    for idx, (label, html) in enumerate(tabs):
        panel_id = f"{gid}-panel-{idx}"
        active = " active" if idx == 0 else ""
        buttons.append(
            f"<button class='subtab-btn{active}' data-tab-group='{gid}' data-target='{panel_id}'>{label}</button>"
        )
        panels.append(
            f"<div id='{panel_id}' class='subtab-content{active}' data-tab-group='{gid}'>{html}</div>"
        )
    lvl = int(level) if int(level) in (1, 2, 3, 4) else 1
    return (
        f"<div class='subtab-group level-{lvl}'>"
        + f"<div class='subtab-row'>{''.join(buttons)}</div>"
        + "".join(panels)
        + "</div>"
    )


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return token or "sample"


def _safe_sample_id(dataset_path: str) -> str:
    return _sanitize_token(os.path.basename(os.path.normpath(dataset_path)))


def _collect_sample_bundle(
    dataset_path: str,
    derivatives_base: Optional[str] = None,
) -> Optional[SampleBundle]:
    """Load one sample into channel-type accumulators using streaming summaries."""
    _, derivatives_root = resolve_output_roots(dataset_path, derivatives_base)
    sample_id = _safe_sample_id(dataset_path)

    calculation_dir = Path(derivatives_root) / "Meg_QC" / "calculation"
    reports_dir = Path(derivatives_root) / "Meg_QC" / "reports"

    if not calculation_dir.exists():
        print(f"___MEGqc___: Multi-sample QA: skipping {sample_id}, calculation folder missing: {calculation_dir}")
        return None

    run_records = _discover_run_records(calculation_dir)
    if not run_records:
        print(f"___MEGqc___: Multi-sample QA: skipping {sample_id}, no run TSV derivatives found.")
        return None

    acc_by_type: Dict[str, ChTypeAccumulator] = {ch: ChTypeAccumulator() for ch in CH_TYPES}
    for run_key in sorted(run_records):
        _update_accumulator_for_run(acc_by_type, run_records[run_key])

    combined_acc = _combine_accumulators(acc_by_type)
    tab_accumulators: Dict[str, ChTypeAccumulator] = {
        "Combined (mag+grad)": combined_acc,
        "MAG": acc_by_type["mag"],
        "GRAD": acc_by_type["grad"],
    }
    if all(acc.run_count == 0 for acc in tab_accumulators.values()):
        print(f"___MEGqc___: Multi-sample QA: skipping {sample_id}, no usable run summaries.")
        return None

    return SampleBundle(
        sample_id=sample_id,
        dataset_path=dataset_path,
        derivatives_root=derivatives_root,
        reports_dir=reports_dir,
        settings_snapshot=_load_settings_snapshot(derivatives_root),
        tab_accumulators=tab_accumulators,
    )


def _tab_dataframe(bundles: Sequence[SampleBundle], tab_name: str) -> pd.DataFrame:
    """Stack run-level summaries for one tab across all samples."""
    frames: List[pd.DataFrame] = []
    for bundle in bundles:
        acc = bundle.tab_accumulators.get(tab_name)
        if acc is None or not acc.run_rows:
            continue
        df = _run_rows_dataframe(acc.run_rows)
        if df.empty:
            continue
        df = df.copy()
        df["sample_id"] = bundle.sample_id
        df["subject_original"] = df["subject"].astype(str)
        # Keep subject ids unique across samples for consistent coloring.
        df["subject"] = df["sample_id"].astype(str) + "::" + df["subject"].astype(str)
        # Guarantee globally unique run labels for cross-sample heatmaps.
        df["run_key"] = (
            df["sample_id"].astype(str)
            + "::"
            + df["run_key"].astype(str)
            + "::"
            + df["channel_type"].astype(str)
        )
        task_label = df["task"].astype(str)
        fallback = df["condition_label"].astype(str)
        df["task_label"] = np.where(task_label != "n/a", task_label, fallback)
        df["sample_task_label"] = df["sample_id"].astype(str) + " | task=" + df["task_label"].astype(str)
        df["hover_entities"] = "sample=" + df["sample_id"].astype(str) + "<br>" + df["hover_entities"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _metric_specs(amplitude_unit: str, is_combined: bool) -> List[Tuple[str, str]]:
    amp_label = "all channels" if is_combined else amplitude_unit
    return [
        ("std_upper_tail", f"STD upper tail ({amp_label})"),
        ("ptp_upper_tail", f"PtP upper tail ({amp_label})"),
        ("mains_ratio", "Mains relative power"),
        ("ecg_p95_abs_corr", "ECG |r| upper tail"),
        ("eog_p95_abs_corr", "EOG |r| upper tail"),
        ("muscle_p95", "Muscle upper tail"),
    ]


def _metric_value_label(metric_col: str, amplitude_unit: str, is_combined: bool) -> str:
    amp_label = "all channels" if is_combined else amplitude_unit
    labels = {
        "std_median": f"STD ({amp_label})",
        "std_upper_tail": f"STD ({amp_label})",
        "ptp_median": f"PtP ({amp_label})",
        "ptp_upper_tail": f"PtP ({amp_label})",
        "mains_ratio": "Mains relative power (unitless)",
        "mains_harmonics_ratio": "Harmonics relative power (unitless)",
        "ecg_mean_abs_corr": "|ECG correlation| (unitless)",
        "ecg_p95_abs_corr": "|ECG correlation| (unitless)",
        "eog_mean_abs_corr": "|EOG correlation| (unitless)",
        "eog_p95_abs_corr": "|EOG correlation| (unitless)",
        "muscle_median": "Muscle score (z-score)",
        "muscle_p95": "Muscle score (z-score)",
    }
    return labels.get(metric_col, metric_col)


def _values_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    if df.empty or value_col not in df.columns or group_col not in df.columns:
        return out
    for label, group in df.groupby(group_col):
        vals = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out[str(label)] = vals.tolist()
    return out


def _finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _downsample_indices(n_points: int, max_points: int) -> np.ndarray:
    if n_points <= 0:
        return np.array([], dtype=int)
    if n_points <= max_points:
        return np.arange(n_points, dtype=int)
    return np.linspace(0, n_points - 1, num=max_points, dtype=int)


def _kde_curve(values: np.ndarray, n_points: int = 220) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    vals = _finite_array(values)
    if vals.size < 2:
        return None
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    std = float(np.nanstd(vals))
    iqr = float(np.nanquantile(vals, 0.75) - np.nanquantile(vals, 0.25))
    scale = std if std > 0 else iqr / 1.34
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return None
    bw = 1.06 * scale * (vals.size ** (-1.0 / 5.0))
    bw = max(bw, np.finfo(float).eps)
    x = np.linspace(vmin, vmax, num=n_points)
    u = (x[:, None] - vals[None, :]) / bw
    y = np.exp(-0.5 * (u ** 2)).sum(axis=1) / (vals.size * bw * np.sqrt(2.0 * np.pi))
    return x, y


def plot_histogram_distribution(
    values_by_group: Dict[str, List[float]],
    *,
    title: str,
    x_title: str,
) -> Optional[go.Figure]:
    fig = go.Figure()
    density_curves = []
    for label in sorted(values_by_group):
        vals = _finite_array(values_by_group[label])
        if vals.size == 0:
            continue
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        density_curves.append((label, vals))
        fig.add_trace(
            go.Histogram(
                x=vals,
                name=f"{label} (n={vals.size})",
                histnorm="probability density",
                opacity=0.42,
                nbinsx=55,
            )
        )
    for label, vals in density_curves:
        kde = _kde_curve(vals)
        if kde is None:
            continue
        x, y = kde
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{label} density",
                line={"width": 2.0},
            )
        )
    if not fig.data:
        return None
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_title,
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def plot_density_distribution(
    values_by_group: Dict[str, List[float]],
    *,
    title: str,
    x_title: str,
) -> Optional[go.Figure]:
    fig = go.Figure()
    for label in sorted(values_by_group):
        vals = _finite_array(values_by_group[label])
        if vals.size < 2:
            continue
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        kde = _kde_curve(vals)
        if kde is None:
            continue
        x, y = kde
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{label} (n={vals.size})", line={"width": 2.0}))
    if not fig.data:
        return None
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_title,
        yaxis_title="Density",
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def _split_sample_task_label(label: str) -> Tuple[str, str]:
    txt = str(label)
    token = " | task="
    if token in txt:
        sample_id, task = txt.split(token, 1)
        return str(sample_id).strip(), str(task).strip()
    return txt.strip(), txt.strip()


def _task_from_condition_label(label: str) -> str:
    txt = str(label)
    match = re.search(r"task=([^,]+)", txt)
    if match:
        return str(match.group(1)).strip()
    return txt if txt else "all recordings"


def _condition_symbol_map(conditions: Sequence[str]) -> Dict[str, str]:
    symbols = ["circle", "diamond", "square", "cross", "triangle-up", "triangle-down", "x", "star"]
    return {str(cond): symbols[idx % len(symbols)] for idx, cond in enumerate(sorted(set(str(c) for c in conditions)))}


def _sample_color_map(sample_ids: Sequence[str]) -> Dict[str, str]:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600",
    ]
    return {sample_id: palette[idx % len(palette)] for idx, sample_id in enumerate(sorted(set(str(s) for s in sample_ids)))}


def _task_color_map(task_ids: Sequence[str]) -> Dict[str, str]:
    palette = [
        "#2a9d8f", "#e76f51", "#264653", "#f4a261", "#6a4c93",
        "#3a86ff", "#ff006e", "#ffbe0b", "#1982c4", "#8ac926",
    ]
    return {task_id: palette[idx % len(palette)] for idx, task_id in enumerate(sorted(set(str(t) for t in task_ids)))}


def _values_to_points_df(
    values: Dict[str, List[float]],
    point_col: str,
) -> pd.DataFrame:
    rows = []
    for label, vals in values.items():
        sample_id, task = _split_sample_task_label(str(label))
        arr = _finite_array(vals)
        if arr.size == 0:
            continue
        for val in arr:
            rows.append(
                {
                    "condition_label": str(label),
                    "sample_id": sample_id,
                    "task_label": task,
                    "subject": sample_id,
                    "hover_entities": f"sample={sample_id}<br>task={task}<br>group={label}",
                    point_col: float(val),
                }
            )
    return pd.DataFrame(rows)


def _subject_points_for_violin(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    out_col: str = "__subject_point__",
) -> pd.DataFrame:
    """Create one jitter point per subject within each displayed group."""
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    data = df.loc[np.isfinite(df[value_col])].copy()
    if data.empty:
        return pd.DataFrame()

    grouped = (
        data.groupby([group_col, "subject"], dropna=False)
        .agg(
            value=(value_col, "median"),
            sample_id_first=("sample_id", "first"),
            task_label_first=("task_label", "first"),
            hover_entities=("hover_entities", "first"),
        )
        .reset_index()
    )
    if grouped.empty:
        return pd.DataFrame()

    grouped["condition_label"] = grouped[group_col].astype(str)
    grouped["subject"] = grouped["subject"].astype(str)
    grouped[out_col] = pd.to_numeric(grouped["value"], errors="coerce")
    grouped = grouped.loc[np.isfinite(grouped[out_col])].copy()
    if grouped.empty:
        return pd.DataFrame()
    grouped["hover_entities"] = (
        "sample="
        + grouped["sample_id_first"].astype(str)
        + "<br>subject="
        + grouped["subject"].astype(str)
        + "<br>task="
        + grouped["task_label_first"].astype(str)
        + "<br>group="
        + grouped["condition_label"].astype(str)
    )
    return grouped[["condition_label", "subject", "hover_entities", out_col]]


def _legend_key_for_label(label: str, legend_mode: str) -> str:
    sample_id, task = _split_sample_task_label(label)
    if legend_mode == "sample":
        return sample_id
    if legend_mode == "task":
        return task
    return str(label)


def plot_violin_with_group_legend(
    values_by_group: Dict[str, List[float]],
    points_df: pd.DataFrame,
    *,
    point_col: str,
    title: str,
    x_title: str,
    y_title: str,
    legend_mode: str = "group",
) -> Optional[go.Figure]:
    """Violin with jitter points and legend filtering by group/sample/task."""
    labels = [k for k in sorted(values_by_group) if _finite_array(values_by_group[k]).size > 0]
    if not labels:
        return None

    xpos = {label: float(i) for i, label in enumerate(labels)}
    fig = go.Figure()
    legend_keys = [_legend_key_for_label(label, legend_mode) for label in labels]
    if legend_mode == "task":
        color_map = _task_color_map(legend_keys)
    else:
        color_map = _sample_color_map(legend_keys)
    legend_seen: set = set()

    for label in labels:
        vals = _finite_array(values_by_group[label])
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        legend_key = _legend_key_for_label(label, legend_mode)
        color = color_map.get(legend_key, "#1f77b4")
        fig.add_trace(
            go.Violin(
                x=np.full(vals.size, xpos[label], dtype=float),
                y=vals,
                name=legend_key,
                legendgroup=f"legend-{legend_key}",
                showlegend=legend_key not in legend_seen,
                box_visible=True,
                meanline_visible=False,
                points=False,
                line={"width": 1.0, "color": color},
                opacity=0.50,
            )
        )
        legend_seen.add(legend_key)
        kde = _kde_curve(vals)
        if kde is not None:
            kde_x, kde_y = kde
            y_max = float(np.nanmax(kde_y))
            if np.isfinite(y_max) and y_max > 0:
                ridge_x = xpos[label] + 0.18 + 0.22 * (kde_y / y_max)
                fig.add_trace(
                    go.Scatter(
                        x=ridge_x,
                        y=kde_x,
                        mode="lines",
                        line={"width": 1.4, "color": color},
                        opacity=0.88,
                        legendgroup=f"legend-{legend_key}",
                        showlegend=False,
                        hovertemplate=f"{label}<br>{y_title}=%{{y:.3g}}<br>density=%{{customdata:.3g}}<extra></extra>",
                        customdata=kde_y,
                    )
                )

    if points_df.empty or point_col not in points_df.columns:
        points_data = pd.DataFrame()
    else:
        points_data = points_df.loc[np.isfinite(points_df[point_col])].copy()
        points_data["condition_label"] = points_data["condition_label"].astype(str)
        points_data = points_data.loc[points_data["condition_label"].isin(labels)]

    if not points_data.empty:
        keep = _downsample_indices(len(points_data), min(MAX_POINTS_SCATTER, len(points_data)))
        points_data = points_data.iloc[keep].copy()
        points_data["legend_key"] = points_data["condition_label"].map(lambda s: _legend_key_for_label(str(s), legend_mode))
        has_subject = "subject" in points_data.columns
        if has_subject:
            points_data["subject_code"] = pd.Categorical(points_data["subject"].astype(str)).codes.astype(float)
        elif legend_mode == "task":
            points_color_map = _task_color_map(points_data["legend_key"].tolist())
        else:
            points_color_map = _sample_color_map(points_data["legend_key"].tolist())
        rng = np.random.default_rng(0)
        for label in labels:
            dlab = points_data.loc[points_data["condition_label"] == label].copy()
            if dlab.empty:
                continue
            legend_key = str(dlab["legend_key"].iloc[0])
            x_numeric = np.full(len(dlab), xpos[label], dtype=float)
            x_numeric = x_numeric + rng.uniform(-0.17, 0.17, size=x_numeric.size)
            if has_subject:
                marker = {
                    "size": 5.2,
                    "color": dlab["subject_code"],
                    "colorscale": "Turbo",
                    "showscale": False,
                    "opacity": 0.70,
                    "line": {"width": 0.35, "color": "rgba(20,20,20,0.45)"},
                }
            else:
                marker = {
                    "size": 5.2,
                    "color": points_color_map.get(legend_key, "#4F6F84"),
                    "opacity": 0.70,
                    "line": {"width": 0.35, "color": "rgba(20,20,20,0.45)"},
                }
            fig.add_trace(
                go.Scattergl(
                    x=x_numeric,
                    y=dlab[point_col],
                    mode="markers",
                    marker=marker,
                    legendgroup=f"legend-{legend_key}",
                    showlegend=False,
                    customdata=np.stack([dlab["hover_entities"].astype(str)], axis=-1),
                    hovertemplate="%{customdata[0]}<br>value=%{y:.3g}<extra></extra>",
                )
            )

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 70, "b": 55},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "groupclick": "togglegroup"},
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[xpos[l] for l in labels],
        ticktext=labels,
        range=[-0.5, len(labels) - 0.1],
    )
    return fig


def plot_run_fingerprint_scatter_by_sample(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> Optional[go.Figure]:
    """Run fingerprint scatter with sample-level color and task-level legend filter."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    data = df.loc[np.isfinite(df[x_col]) & np.isfinite(df[y_col])].copy()
    if data.empty:
        return None

    symbol_map = _condition_symbol_map(data["condition_label"].astype(str).tolist())
    sample_map = _sample_color_map(data["sample_id"].astype(str).tolist())
    fig = go.Figure()

    for cond in sorted(symbol_map):
        dcond = data.loc[data["condition_label"].astype(str) == cond].copy()
        if dcond.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=dcond[x_col],
                y=dcond[y_col],
                mode="markers",
                name=cond,
                legendgroup=f"task-{cond}",
                marker={
                    "size": 8,
                    "symbol": symbol_map[cond],
                    "color": [sample_map.get(str(s), "#4F6F84") for s in dcond["sample_id"].astype(str)],
                    "line": {"width": 0.6, "color": "#2b2d42"},
                    "opacity": 0.86,
                    "showscale": False,
                },
                customdata=np.stack([dcond["hover_entities"]], axis=-1),
                hovertemplate="%{customdata[0]}<br>task=" + cond + "<br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>",
                showlegend=True,
            )
        )

    if not fig.data:
        return None
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 70, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def _profiles_by_condition_from_runs(
    acc: ChTypeAccumulator,
    *,
    by_cond_attr: str,
    list_attr: str,
    run_metric_col: str,
) -> Dict[str, List]:
    """Return condition->profiles map, compatible with older accumulator schemas."""
    source = getattr(acc, by_cond_attr, None)
    if isinstance(source, dict) and source:
        return source

    profiles = list(getattr(acc, list_attr, []) or [])
    if not profiles:
        return {}

    rows = _run_rows_dataframe(getattr(acc, "run_rows", []))
    if rows.empty or run_metric_col not in rows.columns:
        return {"all tasks": profiles}

    cond_labels = (
        rows["condition_label"].astype(str).tolist()
        if "condition_label" in rows.columns
        else ["all recordings"] * len(rows)
    )
    metric_vals = pd.to_numeric(rows[run_metric_col], errors="coerce").to_numpy(dtype=float)
    labels_for_metric = [cond_labels[i] for i, val in enumerate(metric_vals) if np.isfinite(val)]

    out: Dict[str, List] = {}
    if labels_for_metric and len(labels_for_metric) == len(profiles):
        for label, prof in zip(labels_for_metric, profiles):
            out.setdefault(str(label), []).append(prof)
        return out

    if labels_for_metric:
        n = min(len(labels_for_metric), len(profiles))
        for i in range(n):
            out.setdefault(str(labels_for_metric[i]), []).append(profiles[i])
        if len(profiles) > n:
            out.setdefault("all tasks", []).extend(profiles[n:])
        return out

    return {"all tasks": profiles}


def _metric_variant_maps_for_tab(
    bundles: Sequence[SampleBundle],
    tab_name: str,
    metric_col: str,
    variant: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
    """Collect non-recording distribution maps for one metric across samples.

    Returns
    -------
    by_sample
        sample_id -> values
    by_sample_task
        "sample_id | task=<task>" -> values
    by_task_sample
        task -> {sample_id -> values}
    """
    by_sample: Dict[str, List[float]] = {}
    by_sample_task: Dict[str, List[float]] = {}
    by_task_sample: Dict[str, Dict[str, List[float]]] = {}

    for bundle in bundles:
        acc = bundle.tab_accumulators.get(tab_name)
        if acc is None:
            continue
        sample_id = bundle.sample_id

        if variant == "channels_over_epochs":
            if metric_col.startswith("std"):
                source = acc.std_dist_by_condition
            elif metric_col.startswith("ptp"):
                source = acc.ptp_dist_by_condition
            elif metric_col.startswith("mains"):
                source = acc.psd_ratio_by_condition
            elif metric_col.startswith("ecg"):
                source = acc.ecg_corr_by_condition
            elif metric_col.startswith("eog"):
                source = acc.eog_corr_by_condition
            elif metric_col.startswith("muscle"):
                source = acc.muscle_scalar_by_condition
            else:
                source = {}

            for cond_label, vals in source.items():
                arr = _finite_array(vals)
                if arr.size == 0:
                    continue
                task = _task_from_condition_label(cond_label)
                sample_task = f"{sample_id} | task={task}"
                by_sample.setdefault(sample_id, []).extend(arr.tolist())
                by_sample_task.setdefault(sample_task, []).extend(arr.tolist())
                by_task_sample.setdefault(task, {}).setdefault(sample_id, []).extend(arr.tolist())

        elif variant == "epochs_over_channels":
            if metric_col.startswith("std"):
                source_profiles = _profiles_by_condition_from_runs(
                    acc,
                    by_cond_attr="std_window_profiles_by_condition",
                    list_attr="std_window_profiles",
                    run_metric_col="std_median",
                )
                for cond_label, profiles in source_profiles.items():
                    task = _task_from_condition_label(cond_label)
                    sample_task = f"{sample_id} | task={task}"
                    vals = []
                    for prof in profiles:
                        vals.extend(_finite_array(np.asarray(prof.get("q50", []), dtype=float)).tolist())
                    if vals:
                        by_sample.setdefault(sample_id, []).extend(vals)
                        by_sample_task.setdefault(sample_task, []).extend(vals)
                        by_task_sample.setdefault(task, {}).setdefault(sample_id, []).extend(vals)

            elif metric_col.startswith("ptp"):
                source_profiles = _profiles_by_condition_from_runs(
                    acc,
                    by_cond_attr="ptp_window_profiles_by_condition",
                    list_attr="ptp_window_profiles",
                    run_metric_col="ptp_median",
                )
                for cond_label, profiles in source_profiles.items():
                    task = _task_from_condition_label(cond_label)
                    sample_task = f"{sample_id} | task={task}"
                    vals = []
                    for prof in profiles:
                        vals.extend(_finite_array(np.asarray(prof.get("q50", []), dtype=float)).tolist())
                    if vals:
                        by_sample.setdefault(sample_id, []).extend(vals)
                        by_sample_task.setdefault(sample_task, []).extend(vals)
                        by_task_sample.setdefault(task, {}).setdefault(sample_id, []).extend(vals)

            elif metric_col.startswith("muscle"):
                source_profiles = _profiles_by_condition_from_runs(
                    acc,
                    by_cond_attr="muscle_profiles_by_condition",
                    list_attr="muscle_profiles",
                    run_metric_col="muscle_median",
                )
                for cond_label, profiles in source_profiles.items():
                    task = _task_from_condition_label(cond_label)
                    sample_task = f"{sample_id} | task={task}"
                    vals = []
                    for arr in profiles:
                        vals.extend(_finite_array(arr).tolist())
                    if vals:
                        by_sample.setdefault(sample_id, []).extend(vals)
                        by_sample_task.setdefault(sample_task, []).extend(vals)
                        by_task_sample.setdefault(task, {}).setdefault(sample_id, []).extend(vals)

    return by_sample, by_sample_task, by_task_sample


def _distribution_variant_tabs_from_values(
    values: Dict[str, List[float]],
    *,
    title_prefix: str,
    value_label: str,
    block_id: str,
    points_df: Optional[pd.DataFrame] = None,
    point_col: Optional[str] = None,
    include_jitter: bool = False,
    x_title: str = "Group",
    legend_mode: str = "group",
) -> str:
    if not values:
        return "<p>No values are available for this distribution panel.</p>"

    if points_df is None:
        points_df = pd.DataFrame()
    pcol = point_col if point_col is not None else "__value__"
    if points_df.empty:
        points_df = _values_to_points_df(values, pcol)

    violin = plot_violin_with_group_legend(
        values,
        points_df,
        point_col=pcol,
        title=f"{title_prefix} - violin",
        x_title=x_title,
        y_title=value_label,
        legend_mode=legend_mode,
    )
    hist = plot_histogram_distribution(
        values,
        title=f"{title_prefix} - histogram",
        x_title=value_label,
    )
    dens = plot_density_distribution(
        values,
        title=f"{title_prefix} - density",
        x_title=value_label,
    )

    return _build_subtabs_html(
        block_id,
        [
            (
                "Violin",
                _figure_block(
                    violin,
                    (
                        "Each violin summarizes one comparison group. "
                        + ("Jittered points represent individual observations for the displayed grouping. " if include_jitter else "")
                        + ("Legend click toggles samples. " if legend_mode == "sample" else "")
                        + ("Legend click toggles tasks. " if legend_mode == "task" else "")
                        + f"Units: {value_label}."
                    ),
                    normalized_variant=True,
                    norm_mode="y",
                ),
            ),
            (
                "Histogram",
                _figure_block(
                    hist,
                    "Histogram + density variant of the same distribution.",
                    normalized_variant=True,
                    norm_mode="x",
                ),
            ),
            (
                "Density",
                _figure_block(
                    dens,
                    "Kernel density variant of the same distribution for shape comparison.",
                    normalized_variant=True,
                    norm_mode="x",
                ),
            ),
        ],
        level=4,
    )


def _distribution_variant_tabs(
    df: pd.DataFrame,
    *,
    value_col: str,
    group_col: str,
    title_prefix: str,
    value_label: str,
    block_id: str,
    x_title: str,
    legend_mode: str,
) -> str:
    values = _values_by_group(df, value_col, group_col)
    points_df = _subject_points_for_violin(
        df,
        group_col=group_col,
        value_col=value_col,
        out_col="__subject_point__",
    )
    return _distribution_variant_tabs_from_values(
        values,
        title_prefix=title_prefix,
        value_label=value_label,
        block_id=block_id,
        points_df=points_df,
        point_col="__subject_point__",
        include_jitter=True,
        x_title=x_title,
        legend_mode=legend_mode,
    )


def _task_coverage_figure(df: pd.DataFrame, title: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    run_df = df.drop_duplicates(subset=["sample_id", "run_key", "task_label"]).copy()
    if run_df.empty:
        return None

    pivot = run_df.pivot_table(
        index="sample_id",
        columns="task_label",
        values="run_key",
        aggfunc="nunique",
        fill_value=0,
    )
    if pivot.empty:
        return None

    z = pivot.to_numpy(dtype=float)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Blues",
            colorbar={"title": "N recordings"},
            customdata=np.stack([z], axis=-1),
            hovertemplate="sample=%{y}<br>task=%{x}<br>N=%{customdata[0]:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task label",
        yaxis_title="Sample",
        template="plotly_white",
        margin={"l": 72, "r": 20, "t": 65, "b": 65},
        height=max(360, 64 * max(1, len(pivot.index))),
    )
    fig.update_xaxes(tickangle=-22)
    return fig


def plot_recording_metric_heatmap(
    df: pd.DataFrame,
    metric_specs: Sequence[Tuple[str, str]],
    title: str,
) -> Optional[go.Figure]:
    """Recording-level heatmap with normalized color and raw hover values."""
    if df.empty:
        return None
    valid = [(col, label) for col, label in metric_specs if col in df.columns]
    if not valid:
        return None

    run_df = df.copy()
    sort_cols = [c for c in ["sample_id", "subject_original", "subject", "session", "task", "condition", "run"] if c in run_df.columns]
    if sort_cols:
        run_df = run_df.sort_values(sort_cols, kind="stable")

    row_labels = run_df["run_key"].astype(str).tolist() if "run_key" in run_df.columns else [f"recording-{i+1}" for i in range(len(run_df))]
    raw = np.full((len(run_df), len(valid)), np.nan, dtype=float)
    z = np.full_like(raw, np.nan, dtype=float)
    for idx, (col, _) in enumerate(valid):
        vals = pd.to_numeric(run_df[col], errors="coerce").to_numpy(dtype=float)
        raw[:, idx] = vals
        z[:, idx] = _robust_normalize_array(vals)

    bounds = _robust_bounds(z)
    zmin = bounds[0] if bounds is not None else None
    zmax = bounds[1] if bounds is not None else None
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[label for _, label in valid],
            y=row_labels,
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar={"title": "Normalized value (robust z)"},
            customdata=np.stack([raw], axis=-1),
            hovertemplate="recording=%{y}<br>metric=%{x}<br>raw=%{customdata[0]:.3g}<br>normalized=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Metric",
        yaxis_title="Recording",
        template="plotly_white",
        margin={"l": 92, "r": 20, "t": 65, "b": 60},
        height=max(420, 26 * max(1, len(row_labels))),
    )
    fig.update_xaxes(tickangle=-22)
    return fig


def _sample_metric_summary_heatmap(
    df: pd.DataFrame,
    metric_specs: Sequence[Tuple[str, str]],
    title: str,
) -> Optional[go.Figure]:
    if df.empty:
        return None
    valid = [(col, label) for col, label in metric_specs if col in df.columns]
    if not valid:
        return None

    agg = df.groupby("sample_id", dropna=False).agg({col: "median" for col, _ in valid})
    if agg.empty:
        return None

    raw = np.full((agg.shape[0], len(valid)), np.nan, dtype=float)
    z = np.full_like(raw, np.nan, dtype=float)
    for idx, (col, _) in enumerate(valid):
        raw[:, idx] = agg[col].to_numpy(dtype=float)
        z[:, idx] = _robust_normalize_array(raw[:, idx])

    bounds = _robust_bounds(z)
    zmin = bounds[0] if bounds is not None else None
    zmax = bounds[1] if bounds is not None else None

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[label for _, label in valid],
            y=agg.index.tolist(),
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar={"title": "Normalized value (robust z)"},
            customdata=np.stack([raw], axis=-1),
            hovertemplate="sample=%{y}<br>metric=%{x}<br>raw=%{customdata[0]:.3g}<br>normalized=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Metric",
        yaxis_title="Sample",
        template="plotly_white",
        margin={"l": 76, "r": 20, "t": 65, "b": 60},
    )
    fig.update_xaxes(tickangle=-22)
    return fig


def _shared_tasks(df: pd.DataFrame) -> List[str]:
    if df.empty or "task" not in df.columns:
        return []
    task_sets: List[set] = []
    for _, g in df.groupby("sample_id"):
        tasks = {str(v) for v in g["task"].astype(str).tolist() if str(v) != "n/a"}
        if tasks:
            task_sets.append(tasks)
    if len(task_sets) < 2:
        return []
    shared = set.intersection(*task_sets)
    return sorted(shared)


def _sample_task_count_map(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {}
    out: Dict[str, int] = {}
    for sample_id, g in df.groupby("sample_id"):
        tasks = sorted(set(g["task_label"].astype(str).tolist()))
        out[str(sample_id)] = len(tasks)
    return out


def _extract_task_values_for_sample(
    sample_task_values: Dict[str, List[float]],
    sample_id: str,
) -> Dict[str, List[float]]:
    task_map: Dict[str, List[float]] = {}
    prefix = f"{sample_id} | task="
    for label, vals in sample_task_values.items():
        txt = str(label)
        if not txt.startswith(prefix):
            continue
        task = txt[len(prefix):].strip()
        arr = _finite_array(vals)
        if arr.size == 0:
            continue
        task_map.setdefault(task, []).extend(arr.tolist())
    return task_map


def _coverage_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No run-level records were found for this tab.</p>"
    run_df = df.drop_duplicates(subset=["sample_id", "run_key", "task_label"]).copy()
    rows = []
    for sample_id, g in run_df.groupby("sample_id"):
        tasks = sorted(set(g["task_label"].astype(str).tolist()))
        rows.append(
            {
                "Sample": sample_id,
                "N recordings": int(g["run_key"].nunique()),
                "N tasks": int(len(tasks)),
                "Tasks": ", ".join(tasks[:10]) + (" ..." if len(tasks) > 10 else ""),
            }
        )
    tbl = pd.DataFrame(rows).sort_values("Sample")
    return tbl.to_html(index=False, classes="summary-table")


def _build_task_agnostic_section(
    bundles: Sequence[SampleBundle],
    tab_name: str,
    df: pd.DataFrame,
    *,
    amplitude_unit: str,
    is_combined: bool,
    tab_token: str,
) -> str:
    metric_specs = _metric_specs(amplitude_unit, is_combined)
    overview_heatmap = plot_recording_metric_heatmap(
        df,
        metric_specs=metric_specs,
        title=f"Recording-by-metric comparison overview ({tab_token})",
    )
    sample_heatmap = _sample_metric_summary_heatmap(
        df,
        metric_specs=metric_specs,
        title=f"Sample-by-metric comparison summary ({tab_token})",
    )

    fp_df = df.copy()
    fp_df["condition_label"] = fp_df["task_label"].astype(str)
    fingerprint_specs = [
        ("std_median", "std_upper_tail", "Run fingerprint (STD)"),
        ("ptp_median", "ptp_upper_tail", "Run fingerprint (PtP)"),
        ("mains_ratio", "mains_harmonics_ratio", "Run fingerprint (PSD mains/harmonics)"),
        ("ecg_mean_abs_corr", "ecg_p95_abs_corr", "Run fingerprint (ECG)"),
        ("eog_mean_abs_corr", "eog_p95_abs_corr", "Run fingerprint (EOG)"),
        ("muscle_median", "muscle_p95", "Run fingerprint (Muscle)"),
    ]
    fingerprint_tabs = []
    for x_col, y_col, label in fingerprint_specs:
        if x_col not in fp_df.columns or y_col not in fp_df.columns:
            continue
        x_label = _metric_value_label(x_col, amplitude_unit, is_combined)
        y_label = _metric_value_label(y_col, amplitude_unit, is_combined)
        fig = plot_run_fingerprint_scatter_by_sample(
            fp_df,
            x_col=x_col,
            y_col=y_col,
            title=f"{label} ({tab_token})",
            x_label=x_label,
            y_label=y_label,
        )
        fingerprint_tabs.append(
            (
                label,
                _figure_block(
                    fig,
                    (
                        "Each point is one recording-level summary. "
                        "Color is fixed per sample and legend entries are tasks; clicking a task filters all points for that task."
                    ),
                ),
            )
        )
    fp_html = _build_subtabs_html(f"fp-{tab_token}", fingerprint_tabs, level=3) if fingerprint_tabs else "<p>No fingerprint scatter panels are available.</p>"

    metric_tabs = []
    for metric_col, metric_label in metric_specs:
        rec_by_sample = _distribution_variant_tabs(
            df,
            value_col=metric_col,
            group_col="sample_id",
            title_prefix=f"{metric_label} recording-level by sample ({tab_token})",
            value_label=metric_label,
            block_id=f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-a-sample",
            x_title="Sample",
            legend_mode="sample",
        )
        rec_by_sample_task = _distribution_variant_tabs(
            df,
            value_col=metric_col,
            group_col="sample_task_label",
            title_prefix=f"{metric_label} recording-level by sample-task ({tab_token})",
            value_label=metric_label,
            block_id=f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-a-sampletask",
            x_title="Sample / task",
            legend_mode="task",
        )
        rec_by_task = _distribution_variant_tabs(
            df,
            value_col=metric_col,
            group_col="task_label",
            title_prefix=f"{metric_label} recording-level by task ({tab_token})",
            value_label=metric_label,
            block_id=f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-a-task",
            x_title="Task",
            legend_mode="task",
        )

        ch_sample, ch_sample_task, ch_task_sample = _metric_variant_maps_for_tab(
            bundles,
            tab_name=tab_name,
            metric_col=metric_col,
            variant="channels_over_epochs",
        )
        ch_by_sample = _distribution_variant_tabs_from_values(
            ch_sample,
            title_prefix=f"{metric_label} channels-over-epochs by sample ({tab_token})",
            value_label=metric_label,
            block_id=f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-b-sample",
            points_df=_subject_points_for_violin(
                df,
                group_col="sample_id",
                value_col=metric_col,
                out_col="__subject_point__",
            ),
            point_col="__subject_point__",
            include_jitter=True,
            x_title="Sample",
            legend_mode="sample",
        )
        ch_by_sample_task = _distribution_variant_tabs_from_values(
            ch_sample_task,
            title_prefix=f"{metric_label} channels-over-epochs by sample-task ({tab_token})",
            value_label=metric_label,
            block_id=f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-b-sampletask",
            points_df=_subject_points_for_violin(
                df,
                group_col="sample_task_label",
                value_col=metric_col,
                out_col="__subject_point__",
            ),
            point_col="__subject_point__",
            include_jitter=True,
            x_title="Sample / task",
            legend_mode="task",
        )
        ch_by_task = _distribution_variant_tabs_from_values(
            {task: [v for sample_vals in sample_map.values() for v in sample_vals] for task, sample_map in ch_task_sample.items()},
            title_prefix=f"{metric_label} channels-over-epochs by task ({tab_token})",
            value_label=metric_label,
            block_id=f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-b-task",
            points_df=_subject_points_for_violin(
                df,
                group_col="task_label",
                value_col=metric_col,
                out_col="__subject_point__",
            ),
            point_col="__subject_point__",
            include_jitter=True,
            x_title="Task",
            legend_mode="task",
        )

        ep_sample, ep_sample_task, ep_task_sample = _metric_variant_maps_for_tab(
            bundles,
            tab_name=tab_name,
            metric_col=metric_col,
            variant="epochs_over_channels",
        )
        ep_by_sample = _distribution_variant_tabs_from_values(
            ep_sample,
            title_prefix=f"{metric_label} epochs-over-channels by sample ({tab_token})",
            value_label=metric_label,
            block_id=f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-c-sample",
            points_df=_subject_points_for_violin(
                df,
                group_col="sample_id",
                value_col=metric_col,
                out_col="__subject_point__",
            ),
            point_col="__subject_point__",
            include_jitter=True,
            x_title="Sample",
            legend_mode="sample",
        )
        ep_by_sample_task = _distribution_variant_tabs_from_values(
            ep_sample_task,
            title_prefix=f"{metric_label} epochs-over-channels by sample-task ({tab_token})",
            value_label=metric_label,
            block_id=f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-c-sampletask",
            points_df=_subject_points_for_violin(
                df,
                group_col="sample_task_label",
                value_col=metric_col,
                out_col="__subject_point__",
            ),
            point_col="__subject_point__",
            include_jitter=True,
            x_title="Sample / task",
            legend_mode="task",
        )
        ep_by_task = _distribution_variant_tabs_from_values(
            {task: [v for sample_vals in sample_map.values() for v in sample_vals] for task, sample_map in ep_task_sample.items()},
            title_prefix=f"{metric_label} epochs-over-channels by task ({tab_token})",
            value_label=metric_label,
            block_id=f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-c-task",
            points_df=_subject_points_for_violin(
                df,
                group_col="task_label",
                value_col=metric_col,
                out_col="__subject_point__",
            ),
            point_col="__subject_point__",
            include_jitter=True,
            x_title="Task",
            legend_mode="task",
        )

        metric_tabs.append(
            (
                metric_label,
                _build_subtabs_html(
                    f"agnostic-{tab_token}-{_sanitize_token(metric_col)}",
                    [
                        (
                            "A: Recording-level",
                            _build_subtabs_html(
                                f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-a",
                                [("By sample", rec_by_sample), ("By task", rec_by_task), ("By sample-task", rec_by_sample_task)],
                                level=4,
                            ),
                        ),
                        (
                            "B: Channels over epochs",
                            _build_subtabs_html(
                                f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-b",
                                [("By sample", ch_by_sample), ("By task", ch_by_task), ("By sample-task", ch_by_sample_task)],
                                level=4,
                            ),
                        ),
                        (
                            "C: Epochs over channels",
                            _build_subtabs_html(
                                f"agnostic-{tab_token}-{_sanitize_token(metric_col)}-c",
                                [("By sample", ep_by_sample), ("By task", ep_by_task), ("By sample-task", ep_by_sample_task)],
                                level=4,
                            ),
                        ),
                    ],
                    level=3,
                ),
            )
        )
    dist_html = _build_subtabs_html(f"agnostic-metrics-{tab_token}", metric_tabs, level=2)

    return (
        "<section>"
        "<h2>Task-agnostic comparison</h2>"
        "<p>This section compares samples without assuming task overlap. "
        "Use it as the primary view when samples have different task sets.</p>"
        + _figure_block(
            overview_heatmap,
            (
                "Rows are recordings and columns are metric summaries. "
                "Color is normalized for readability across heterogeneous metrics; hover reports raw values."
            ),
        )
        + _figure_block(
            sample_heatmap,
            "Rows are samples and columns are robust metric summaries; this condenses the cohort-level profile per sample.",
        )
        + "<div class='metric-block'><h3>Run fingerprint scatter</h3>"
        + fp_html
        + "</div>"
        + "<div class='metric-block'><h3>Distribution comparisons</h3>"
        + dist_html
        + "</div>"
        + "</section>"
    )


def _build_task_comparison_section(
    bundles: Sequence[SampleBundle],
    tab_name: str,
    df: pd.DataFrame,
    *,
    amplitude_unit: str,
    is_combined: bool,
    tab_token: str,
) -> str:
    shared = _shared_tasks(df)
    metric_specs = _metric_specs(amplitude_unit, is_combined)
    if shared:
        clipped = shared[:MAX_SHARED_TASKS]
        clipped_msg = ""
        if len(shared) > len(clipped):
            clipped_msg = f"<p>Showing first {len(clipped)} shared tasks out of {len(shared)} for readability.</p>"

        metric_panels: List[Tuple[str, str]] = []
        for metric_col, metric_label in metric_specs:
            if metric_col not in df.columns:
                continue
            _, _, task_sample_ch = _metric_variant_maps_for_tab(
                bundles,
                tab_name=tab_name,
                metric_col=metric_col,
                variant="channels_over_epochs",
            )
            _, _, task_sample_ep = _metric_variant_maps_for_tab(
                bundles,
                tab_name=tab_name,
                metric_col=metric_col,
                variant="epochs_over_channels",
            )
            task_tabs = []
            for task in clipped:
                subset = df.loc[df["task"].astype(str) == task].copy()
                rec_values = _values_by_group(subset, metric_col, "sample_id") if not subset.empty else {}
                ch_values = task_sample_ch.get(task, {})
                ep_values = task_sample_ep.get(task, {})
                if (not rec_values) and (not ch_values) and (not ep_values):
                    continue
                points_df = _subject_points_for_violin(
                    subset,
                    group_col="sample_id",
                    value_col=metric_col,
                    out_col="__subject_point__",
                )

                rec_panel = _distribution_variant_tabs_from_values(
                    rec_values,
                    title_prefix=f"{metric_label} recording-level (task={task})",
                    value_label=metric_label,
                    block_id=f"matched-{tab_token}-{_sanitize_token(metric_col)}-{_sanitize_token(task)}-a",
                    points_df=points_df,
                    point_col="__subject_point__",
                    include_jitter=True,
                    x_title="Sample",
                    legend_mode="sample",
                )
                ch_panel = _distribution_variant_tabs_from_values(
                    ch_values,
                    title_prefix=f"{metric_label} channels-over-epochs (task={task})",
                    value_label=metric_label,
                    block_id=f"matched-{tab_token}-{_sanitize_token(metric_col)}-{_sanitize_token(task)}-b",
                    points_df=points_df,
                    point_col="__subject_point__",
                    include_jitter=True,
                    x_title="Sample",
                    legend_mode="sample",
                )
                ep_panel = _distribution_variant_tabs_from_values(
                    ep_values,
                    title_prefix=f"{metric_label} epochs-over-channels (task={task})",
                    value_label=metric_label,
                    block_id=f"matched-{tab_token}-{_sanitize_token(metric_col)}-{_sanitize_token(task)}-c",
                    points_df=points_df,
                    point_col="__subject_point__",
                    include_jitter=True,
                    x_title="Sample",
                    legend_mode="sample",
                )
                task_tabs.append(
                    (
                        f"task={task}",
                        _build_subtabs_html(
                            f"matched-{tab_token}-{_sanitize_token(metric_col)}-{_sanitize_token(task)}",
                            [
                                ("A: Recording-level", rec_panel),
                                ("B: Channels over epochs", ch_panel),
                                ("C: Epochs over channels", ep_panel),
                            ],
                            level=4,
                        ),
                    )
                )
            if task_tabs:
                metric_panels.append(
                    (
                        metric_label,
                        _build_subtabs_html(
                            f"matched-metric-{tab_token}-{_sanitize_token(metric_col)}",
                            task_tabs,
                            level=3,
                        ),
                    )
                )

        if not metric_panels:
            return (
                "<section><h2>Task comparison</h2>"
                "<p>Shared tasks exist but no numeric values were available for matched distributions.</p>"
                "</section>"
            )

        metric_tabs = _build_subtabs_html(
            f"matched-metrics-{tab_token}",
            metric_panels,
            level=2,
        )
        shared_list = ", ".join([f"task={t}" for t in clipped])
        return (
            "<section>"
            "<h2>Task comparison</h2>"
            f"<p>Shared tasks across all samples: {shared_list}. This view compares matched tasks between samples.</p>"
            + clipped_msg
            + metric_tabs
            + "</section>"
        )

    task_count = _sample_task_count_map(df)
    if not task_count:
        return (
            "<section><h2>Task comparison</h2>"
            "<p>No task labels were found for this tab.</p>"
            "</section>"
        )
    if not all(count >= 2 for count in task_count.values()):
        return (
            "<section><h2>Task comparison</h2>"
            "<p>No shared tasks were found across samples, and at least one sample has fewer than two tasks. "
            "Task comparison is therefore not shown for this tab.</p>"
            "</section>"
        )

    metric_panels: List[Tuple[str, str]] = []
    for metric_col, metric_label in metric_specs:
        if metric_col not in df.columns:
            continue
        _, sample_task_ch, _ = _metric_variant_maps_for_tab(
            bundles,
            tab_name=tab_name,
            metric_col=metric_col,
            variant="channels_over_epochs",
        )
        _, sample_task_ep, _ = _metric_variant_maps_for_tab(
            bundles,
            tab_name=tab_name,
            metric_col=metric_col,
            variant="epochs_over_channels",
        )
        sample_tabs = []
        for sample_id in sorted(task_count):
            subset = df.loc[df["sample_id"].astype(str) == str(sample_id)].copy()
            rec_values = _values_by_group(subset, metric_col, "task_label") if not subset.empty else {}
            ch_values = _extract_task_values_for_sample(sample_task_ch, str(sample_id))
            ep_values = _extract_task_values_for_sample(sample_task_ep, str(sample_id))
            if (not rec_values) and (not ch_values) and (not ep_values):
                continue
            points_df = _subject_points_for_violin(
                subset,
                group_col="task_label",
                value_col=metric_col,
                out_col="__subject_point__",
            )

            rec_panel = _distribution_variant_tabs_from_values(
                rec_values,
                title_prefix=f"{metric_label} recording-level ({sample_id})",
                value_label=metric_label,
                block_id=f"sampletasks-{tab_token}-{_sanitize_token(metric_col)}-{_sanitize_token(sample_id)}-a",
                points_df=points_df,
                point_col="__subject_point__",
                include_jitter=True,
                x_title="Task",
                legend_mode="task",
            )
            ch_panel = _distribution_variant_tabs_from_values(
                ch_values,
                title_prefix=f"{metric_label} channels-over-epochs ({sample_id})",
                value_label=metric_label,
                block_id=f"sampletasks-{tab_token}-{_sanitize_token(metric_col)}-{_sanitize_token(sample_id)}-b",
                points_df=points_df,
                point_col="__subject_point__",
                include_jitter=True,
                x_title="Task",
                legend_mode="task",
            )
            ep_panel = _distribution_variant_tabs_from_values(
                ep_values,
                title_prefix=f"{metric_label} epochs-over-channels ({sample_id})",
                value_label=metric_label,
                block_id=f"sampletasks-{tab_token}-{_sanitize_token(metric_col)}-{_sanitize_token(sample_id)}-c",
                points_df=points_df,
                point_col="__subject_point__",
                include_jitter=True,
                x_title="Task",
                legend_mode="task",
            )
            sample_tabs.append(
                (
                    sample_id,
                    _build_subtabs_html(
                        f"sampletasks-{tab_token}-{_sanitize_token(metric_col)}-{_sanitize_token(sample_id)}",
                        [
                            ("A: Recording-level", rec_panel),
                            ("B: Channels over epochs", ch_panel),
                            ("C: Epochs over channels", ep_panel),
                        ],
                        level=4,
                    ),
                )
            )
        if sample_tabs:
            metric_panels.append(
                (
                    metric_label,
                    _build_subtabs_html(
                        f"sampletasks-metric-{tab_token}-{_sanitize_token(metric_col)}",
                        sample_tabs,
                        level=3,
                    ),
                )
            )

    if not metric_panels:
        return (
            "<section><h2>Task comparison</h2>"
            "<p>No shared tasks were found, and within-sample task panels did not contain numeric values.</p>"
            "</section>"
        )

    return (
        "<section>"
        "<h2>Task comparison</h2>"
        "<p>No shared tasks were found across all samples. "
        "This fallback view compares tasks within each sample (all samples have at least two tasks).</p>"
        + _build_subtabs_html(
            f"sampletasks-metrics-{tab_token}",
            metric_panels,
            level=2,
        )
        + "</section>"
    )


def _build_coverage_section(df: pd.DataFrame, tab_token: str) -> str:
    coverage = _task_coverage_figure(df, f"Task coverage by sample ({tab_token})")
    return (
        "<section>"
        "<h2>Coverage</h2>"
        "<p>Coverage clarifies whether differences are potentially influenced by task composition and sampling depth.</p>"
        + _figure_block(
            coverage,
            "Cells report number of recordings per sample and task.",
        )
        + _coverage_table_html(df)
        + "</section>"
    )


def _build_metadata_section(bundles: Sequence[SampleBundle], tab_name: str) -> str:
    lines = []
    for bundle in bundles:
        acc = bundle.tab_accumulators.get(tab_name)
        n_runs = acc.run_count if acc is not None else 0
        lines.append(
            "<li>"
            f"<strong>{bundle.sample_id}</strong>: "
            f"dataset=<code>{bundle.dataset_path}</code>, "
            f"derivatives=<code>{bundle.derivatives_root}</code>, "
            f"runs_in_tab={n_runs}"
            "</li>"
        )

    settings_blocks = []
    for bundle in bundles:
        settings_blocks.append(
            f"<h4>{bundle.sample_id} settings snapshot</h4>"
            f"<pre>{bundle.settings_snapshot}</pre>"
        )

    return (
        "<section>"
        "<h2>Metadata & settings</h2>"
        "<h3>Samples</h3>"
        f"<ul>{''.join(lines)}</ul>"
        "<h3>Settings snapshots</h3>"
        + "".join(settings_blocks)
        + "</section>"
    )


def _build_tab_content(
    bundles: Sequence[SampleBundle],
    tab_name: str,
) -> str:
    is_combined = tab_name == "Combined (mag+grad)"
    if is_combined:
        amplitude_unit = "mixed MEG units (all channels)"
        tab_token = "combined"
    elif tab_name == "MAG":
        amplitude_unit = "Tesla (T)"
        tab_token = "mag"
    else:
        amplitude_unit = "Tesla/m (T/m)"
        tab_token = "grad"

    df = _tab_dataframe(bundles, tab_name)
    if df.empty:
        return (
            "<section><h2>No content</h2>"
            "<p>No run-level summaries are available for this tab.</p>"
            "</section>"
        )

    sections = [
        ("Coverage", _build_coverage_section(df, tab_token)),
        (
            "Task-agnostic comparison",
            _build_task_agnostic_section(
                bundles,
                tab_name,
                df,
                amplitude_unit=amplitude_unit,
                is_combined=is_combined,
                tab_token=tab_token,
            ),
        ),
        (
            "Task comparison",
            _build_task_comparison_section(
                bundles,
                tab_name,
                df,
                amplitude_unit=amplitude_unit,
                is_combined=is_combined,
                tab_token=tab_token,
            ),
        ),
        ("Metadata & settings", _build_metadata_section(bundles, tab_name)),
    ]
    return _build_subtabs_html(f"multi-main-{tab_token}", sections, level=1)


def _build_multi_sample_report_html(
    bundles: Sequence[SampleBundle],
    tab_order: Sequence[str],
) -> str:
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = getattr(meg_qc, "__version__", "unknown")
    sample_names = ", ".join(bundle.sample_id for bundle in bundles)

    tab_buttons = []
    tab_divs = []
    for idx, tab in enumerate(tab_order):
        tab_id = f"tab-{idx}"
        active_class = " active" if idx == 0 else ""
        tab_buttons.append(f"<button class='tab-btn{active_class}' data-target='{tab_id}'>{tab}</button>")
        content = _build_tab_content(bundles, tab)
        tab_divs.append(f"<div id='{tab_id}' class='tab-content{active_class}'>{content}</div>")

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QA multi-sample report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      margin: 0;
      color: #1a1a1a;
      background: linear-gradient(135deg, #f6fbff, #edf6ff 45%, #f9fcff);
      transition: background 0.25s ease, color 0.25s ease;
    }}
    main {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px 24px 56px;
    }}
    h1, h2, h3, h4 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 30px; letter-spacing: 0.2px; }}
    h2 {{ margin-top: 24px; font-size: 22px; }}
    h3 {{ margin-top: 12px; font-size: 18px; }}
    h4 {{ margin-top: 14px; font-size: 15px; color: #284b63; }}
    section {{
      background: rgba(255, 255, 255, 0.90);
      border: 1px solid #dce9f7;
      border-radius: 14px;
      padding: 16px 16px 10px;
      margin-top: 16px;
      box-shadow: 0 6px 24px rgba(5, 45, 79, 0.06);
    }}
    .report-header {{
      display: flex;
      gap: 14px;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
    .metric-block {{
      border-top: 1px solid #e7eef8;
      margin-top: 14px;
      padding-top: 12px;
    }}
    pre {{
      white-space: pre-wrap;
      background: #f4f8fd;
      border: 1px solid #d8e4f3;
      border-radius: 8px;
      padding: 10px;
      font-size: 12px;
    }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 13px;
    }}
    .summary-table th, .summary-table td {{
      border: 1px solid #d5e3f3;
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    .summary-table th {{ background: #eef5fd; }}
    .fig {{
      border-top: 1px solid #e7eef8;
      margin-top: 10px;
      padding-top: 10px;
      overflow: visible;
    }}
    .fig .js-plotly-plot {{
      width: 100% !important;
    }}
    .fig-note {{
      margin: 6px 2px 14px;
      font-size: 13px;
      line-height: 1.45;
      color: #2a425f;
    }}
    .tab-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .tab-btn {{
      border: 1px solid #c5d9f0;
      border-radius: 10px;
      background: #f4f9ff;
      padding: 8px 14px;
      cursor: pointer;
      font-size: 14px;
      color: #21415c;
    }}
    .tab-btn.active {{
      background: #e2f0ff;
      border-color: #8db5dd;
      font-weight: 600;
    }}
    .tab-content {{
      display: none;
      margin-top: 8px;
    }}
    .tab-content.active {{
      display: block;
    }}
    .subtab-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0 10px;
    }}
    .subtab-group {{
      border: 1px solid #d8e6f7;
      border-radius: 12px;
      padding: 10px 10px 8px;
      margin-top: 10px;
    }}
    .subtab-group.level-1 {{
      background: #f1f7ff;
      border-color: #c5daf1;
    }}
    .subtab-group.level-2 {{
      background: #f7fbff;
      border-color: #d4e4f7;
    }}
    .subtab-group.level-3 {{
      background: #fbfdff;
      border-color: #dce9f8;
    }}
    .subtab-group.level-4 {{
      background: #ffffff;
      border-color: #e6eef9;
    }}
    .subtab-group.level-1 > .subtab-row .subtab-btn {{
      background: #eaf3ff;
      border-color: #9ebfe4;
      font-weight: 600;
    }}
    .subtab-group.level-2 > .subtab-row .subtab-btn {{
      background: #f0f7ff;
      border-color: #bdd4ee;
    }}
    .subtab-group.level-3 > .subtab-row .subtab-btn {{
      background: #f6fbff;
      border-color: #cddff2;
    }}
    .subtab-btn {{
      border: 1px solid #c8daee;
      border-radius: 9px;
      background: #f8fbff;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 13px;
      color: #27475f;
    }}
    .subtab-btn.active {{
      background: #e9f3ff;
      border-color: #8db5dd;
      font-weight: 600;
    }}
    .subtab-content {{
      display: none;
    }}
    .subtab-content.active {{
      display: block;
    }}
    .fig-switch {{
      display: inline-flex;
      gap: 6px;
      margin: 6px 0 10px;
    }}
    .fig-switch-btn {{
      border: 1px solid #c8daee;
      border-radius: 8px;
      background: #f8fbff;
      padding: 5px 10px;
      font-size: 12px;
      color: #24445d;
      cursor: pointer;
    }}
    .fig-switch-btn.active {{
      background: #e9f3ff;
      border-color: #8db5dd;
      font-weight: 600;
    }}
    .fig-view {{
      display: none;
    }}
    .fig-view.active {{
      display: block;
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <div class="report-header">
        <div>
          <h1>QA multi-sample report</h1>
          <p><strong>Samples:</strong> {sample_names}</p>
          <p><strong>Generated:</strong> {generated}</p>
          <p><strong>MEGqc version:</strong> {version}</p>
          <p><strong>Scope:</strong> task-agnostic + task comparison</p>
        </div>
      </div>
      <div class="tab-row">
        {"".join(tab_buttons)}
      </div>
      {"".join(tab_divs)}
    </section>
  </main>
  <script>
    (function() {{
      const buttons = Array.from(document.querySelectorAll('.tab-btn'));
      const tabs = Array.from(document.querySelectorAll('.tab-content'));
      function resizePlots(targetId) {{
        if (typeof Plotly === 'undefined') {{
          return;
        }}
        const root = document.getElementById(targetId);
        if (!root) {{
          return;
        }}
        const plots = Array.from(root.querySelectorAll('.js-plotly-plot'));
        plots.forEach((plotEl) => {{
          try {{
            Plotly.Plots.resize(plotEl);
          }} catch (err) {{
            // no-op
          }}
        }});
      }}
      function activate(targetId) {{
        tabs.forEach(t => t.classList.toggle('active', t.id === targetId));
        buttons.forEach(b => b.classList.toggle('active', b.dataset.target === targetId));
        window.requestAnimationFrame(() => {{
          resizePlots(targetId);
          window.setTimeout(() => resizePlots(targetId), 120);
        }});
      }}
      buttons.forEach(btn => {{
        btn.addEventListener('click', () => activate(btn.dataset.target));
      }});
      if (buttons.length > 0) {{
        activate(buttons[0].dataset.target);
      }}
      function activateSubtab(groupId, targetId) {{
        const subButtons = Array.from(document.querySelectorAll(`.subtab-btn[data-tab-group="${{groupId}}"]`));
        const subPanels = Array.from(document.querySelectorAll(`.subtab-content[data-tab-group="${{groupId}}"]`));
        subPanels.forEach(p => p.classList.toggle('active', p.id === targetId));
        subButtons.forEach(b => b.classList.toggle('active', b.dataset.target === targetId));
        const activeTop = tabs.find(t => t.classList.contains('active'));
        if (activeTop) {{
          window.requestAnimationFrame(() => resizePlots(activeTop.id));
        }}
      }}
      const subButtonsAll = Array.from(document.querySelectorAll('.subtab-btn'));
      const seenGroups = new Set();
      subButtonsAll.forEach(btn => {{
        const groupId = btn.dataset.tabGroup;
        if (!groupId) {{
          return;
        }}
        btn.addEventListener('click', () => activateSubtab(groupId, btn.dataset.target));
        seenGroups.add(groupId);
      }});
      seenGroups.forEach(groupId => {{
        const firstBtn = document.querySelector(`.subtab-btn[data-tab-group="${{groupId}}"]`);
        if (firstBtn) {{
          activateSubtab(groupId, firstBtn.dataset.target);
        }}
      }});
      function activateFigVariant(toggleId, targetId) {{
        const root = document.querySelector(`.fig-switch[data-fig-toggle="${{toggleId}}"]`);
        if (!root) return;
        const btns = Array.from(root.querySelectorAll('.fig-switch-btn'));
        btns.forEach(b => b.classList.toggle('active', b.dataset.target === targetId));
        const raw = document.getElementById(`${{toggleId}}-raw`);
        const norm = document.getElementById(`${{toggleId}}-norm`);
        if (raw) raw.classList.toggle('active', `${{toggleId}}-raw` === targetId);
        if (norm) norm.classList.toggle('active', `${{toggleId}}-norm` === targetId);
        const activeTop = tabs.find(t => t.classList.contains('active'));
        if (activeTop) {{
          window.requestAnimationFrame(() => resizePlots(activeTop.id));
        }}
      }}
      Array.from(document.querySelectorAll('.fig-switch')).forEach((sw) => {{
        const toggleId = sw.dataset.figToggle;
        Array.from(sw.querySelectorAll('.fig-switch-btn')).forEach((btn) => {{
          btn.addEventListener('click', () => activateFigVariant(toggleId, btn.dataset.target));
        }});
        const first = sw.querySelector('.fig-switch-btn');
        if (first) activateFigVariant(toggleId, first.dataset.target);
      }});
      window.addEventListener('resize', () => {{
        const active = tabs.find(t => t.classList.contains('active'));
        if (active) {{
          resizePlots(active.id);
        }}
      }});
    }})();
  </script>
</body>
</html>
"""


def make_multi_sample_group_plots_meg_qc(
    dataset_paths: Sequence[str],
    derivatives_bases: Optional[Sequence[Optional[str]]] = None,
    output_report_path: Optional[str] = None,
) -> Dict[str, Path]:
    """Build one HTML report comparing multiple MEGqc samples.

    Parameters
    ----------
    dataset_paths
        Paths to BIDS datasets to compare.
    derivatives_bases
        Optional list (same length as ``dataset_paths``) of external derivatives
        parent folders. Use ``None`` for default in-dataset derivatives.
    output_report_path
        Optional explicit output path for the final HTML report.

    Returns
    -------
    dict
        Mapping ``{"report": Path(...)}`` for the generated multi-sample HTML.
    """
    if not dataset_paths:
        print("___MEGqc___: Multi-sample QA: no datasets were provided.")
        return {}
    if len(dataset_paths) < 2:
        print("___MEGqc___: Multi-sample QA: provide at least two datasets for comparison.")
        return {}

    if derivatives_bases is None:
        derivatives_bases = [None] * len(dataset_paths)
    if len(derivatives_bases) != len(dataset_paths):
        raise ValueError("derivatives_bases must be None or have same length as dataset_paths")

    bundles: List[SampleBundle] = []
    for ds_path, der_base in zip(dataset_paths, derivatives_bases):
        bundle = _collect_sample_bundle(ds_path, der_base)
        if bundle is not None:
            bundles.append(bundle)

    if len(bundles) < 2:
        print("___MEGqc___: Multi-sample QA: fewer than two datasets had usable derivatives.")
        return {}

    tab_order = ["Combined (mag+grad)", "MAG", "GRAD"]
    html = _build_multi_sample_report_html(bundles, tab_order)

    if output_report_path is not None:
        out_path = Path(output_report_path).expanduser().resolve()
    else:
        primary_reports_dir = bundles[0].reports_dir
        ids = [_sanitize_token(b.sample_id) for b in bundles]
        suffix = "_vs_".join(ids[:4])
        if len(ids) > 4:
            suffix = f"{suffix}_and_{len(ids) - 4}_more"
        out_name = f"QA_multi_sample_report_{suffix}.html"
        out_path = primary_reports_dir / out_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print("___MEGqc___: Multi-sample QA report created:")
    print(f"___MEGqc___:   report: {out_path}")
    return {"report": out_path}
