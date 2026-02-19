"""Dataset-level QC plotting from Global Quality Index summary tables.

This module builds an interactive HTML QC report from
``summary_reports/group_metrics/Global_Quality_Index_attempt_*.tsv`` produced
by the GQI calculation pipeline.

Public entrypoint
-----------------
``make_group_qc_plots_meg_qc(dataset_path, input_tsv=None, output_html=None,\n                              attempt=None, derivatives_base=None)``
"""

from __future__ import annotations

import configparser
import datetime as dt
import html
import json
import os
import re
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

import meg_qc
from meg_qc.calculation.meg_qc_pipeline import resolve_output_roots


METRIC_ORDER = ("GQI", "STD", "PtP", "PSD", "ECG", "EOG", "Muscle")
TOP_TABS = ("Combined (mag+grad)", "MAG", "GRAD")

_LAZY_PLOT_COUNTER = count(1)
_LAZY_PAYLOAD_COUNTER = count(1)
_LAZY_PLOT_PAYLOADS: Dict[str, str] = {}


@dataclass(frozen=True)
class ComponentSpec:
    """One plottable scalar component inside a QC metric tab."""

    key: str
    label: str
    unit: str
    mag_col: Optional[str] = None
    grad_col: Optional[str] = None
    global_col: Optional[str] = None


@dataclass
class QCDatasetBundle:
    """Resolved and loaded inputs for one dataset QC report source."""

    dataset_path: str
    dataset_name: str
    derivatives_root: str
    reports_dir: Path
    tsv_path: Path
    cfg_path: Optional[Path]
    attempt: Optional[int]
    df: pd.DataFrame
    general_settings_snapshot: str
    gqi_settings_snapshot: str


COMPONENTS: Dict[str, List[ComponentSpec]] = {
    "GQI": [
        ComponentSpec("gqi", "Global Quality Index", "% quality", global_col="GQI"),
        ComponentSpec("pen_ch", "Variability penalty", "penalty units", global_col="GQI_penalty_ch"),
        ComponentSpec("pen_corr", "Correlational penalty", "penalty units", global_col="GQI_penalty_corr"),
        ComponentSpec("pen_mus", "Muscle penalty", "penalty units", global_col="GQI_penalty_mus"),
        ComponentSpec("pen_psd", "PSD penalty", "penalty units", global_col="GQI_penalty_psd"),
        ComponentSpec("gqi_std", "GQI STD component", "%", global_col="GQI_std_pct"),
        ComponentSpec("gqi_ptp", "GQI PtP component", "%", global_col="GQI_ptp_pct"),
        ComponentSpec("gqi_ecg", "GQI ECG component", "%", global_col="GQI_ecg_pct"),
        ComponentSpec("gqi_eog", "GQI EOG component", "%", global_col="GQI_eog_pct"),
        ComponentSpec("gqi_mus", "GQI muscle component", "%", global_col="GQI_muscle_pct"),
        ComponentSpec("gqi_psd", "GQI PSD-noise component", "%", global_col="GQI_psd_noise_pct"),
    ],
    "STD": [
        ComponentSpec(
            "std_ts_noisy",
            "Time-series noisy channels",
            "% channels",
            mag_col="STD_ts_noisy_channels_mag_percentage",
            grad_col="STD_ts_noisy_channels_grad_percentage",
        ),
        ComponentSpec(
            "std_ts_flat",
            "Time-series flat channels",
            "% channels",
            mag_col="STD_ts_flat_channels_mag_percentage",
            grad_col="STD_ts_flat_channels_grad_percentage",
        ),
        ComponentSpec(
            "std_ep_noisy",
            "Noisy epochs",
            "% epochs",
            mag_col="STD_ep_mag_noisy_percentage",
            grad_col="STD_ep_grad_noisy_percentage",
        ),
        ComponentSpec(
            "std_ep_flat",
            "Flat epochs",
            "% epochs",
            mag_col="STD_ep_mag_flat_percentage",
            grad_col="STD_ep_grad_flat_percentage",
        ),
    ],
    "PtP": [
        ComponentSpec(
            "ptp_ts_noisy",
            "Time-series noisy channels",
            "% channels",
            mag_col="PTP_ts_noisy_channels_mag_percentage",
            grad_col="PTP_ts_noisy_channels_grad_percentage",
        ),
        ComponentSpec(
            "ptp_ts_flat",
            "Time-series flat channels",
            "% channels",
            mag_col="PTP_ts_flat_channels_mag_percentage",
            grad_col="PTP_ts_flat_channels_grad_percentage",
        ),
        ComponentSpec(
            "ptp_ep_noisy",
            "Noisy epochs",
            "% epochs",
            mag_col="PTP_ep_mag_noisy_percentage",
            grad_col="PTP_ep_grad_noisy_percentage",
        ),
        ComponentSpec(
            "ptp_ep_flat",
            "Flat epochs",
            "% epochs",
            mag_col="PTP_ep_mag_flat_percentage",
            grad_col="PTP_ep_grad_flat_percentage",
        ),
    ],
    "PSD": [
        ComponentSpec(
            "psd_noise",
            "PSD noise burden",
            "% relative power",
            mag_col="PSD_noise_mag_percentage",
            grad_col="PSD_noise_grad_percentage",
        ),
    ],
    "ECG": [
        ComponentSpec(
            "ecg_high_corr",
            "High-correlation channels",
            "% channels",
            mag_col="ECG_mag_high_corr_percentage",
            grad_col="ECG_grad_high_corr_percentage",
        ),
    ],
    "EOG": [
        ComponentSpec(
            "eog_high_corr",
            "High-correlation channels",
            "% channels",
            mag_col="EOG_mag_high_corr_percentage",
            grad_col="EOG_grad_high_corr_percentage",
        ),
    ],
    "Muscle": [
        ComponentSpec("muscle_events", "Muscle events", "event count", global_col="Muscle_events_num"),
        ComponentSpec("muscle_rate", "Muscle event rate", "% of events", global_col="__muscle_event_rate_pct"),
        ComponentSpec("gqi_muscle_component", "GQI muscle component", "%", global_col="GQI_muscle_pct"),
    ],
}


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def _attempt_from_name(path: Path) -> Optional[int]:
    match = re.search(r"attempt_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _latest_attempt_tsv(group_metrics_dir: Path) -> Optional[Path]:
    files = list(group_metrics_dir.glob("Global_Quality_Index_attempt_*.tsv"))
    if not files:
        return None
    files.sort(key=lambda p: (_attempt_from_name(p) or -1, p.stat().st_mtime), reverse=True)
    return files[0]


def _resolve_input_paths(
    dataset_path: str,
    derivatives_base: Optional[str],
    input_tsv: Optional[str],
    attempt: Optional[int],
) -> Tuple[str, Path, Optional[Path], Optional[int], Path]:
    """Resolve derivatives root, selected TSV, matching config, and reports dir."""
    _, derivatives_root = resolve_output_roots(dataset_path, derivatives_base)
    summary_root = Path(derivatives_root) / "Meg_QC" / "summary_reports"
    group_metrics_dir = summary_root / "group_metrics"
    config_dir = summary_root / "config"
    reports_dir = Path(derivatives_root) / "Meg_QC" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if input_tsv:
        tsv_path = Path(input_tsv)
        if not tsv_path.exists():
            raise FileNotFoundError(f"Input TSV does not exist: {tsv_path}")
    else:
        if attempt is not None:
            tsv_path = group_metrics_dir / f"Global_Quality_Index_attempt_{attempt}.tsv"
            if not tsv_path.exists():
                raise FileNotFoundError(
                    f"Requested attempt {attempt} not found: {tsv_path}"
                )
        else:
            tsv_path = _latest_attempt_tsv(group_metrics_dir)
            if tsv_path is None:
                raise FileNotFoundError(
                    f"No Global_Quality_Index_attempt_*.tsv found in {group_metrics_dir}"
                )

    resolved_attempt = _attempt_from_name(tsv_path)
    cfg_path = None
    if resolved_attempt is not None:
        candidate = config_dir / f"global_quality_index_{resolved_attempt}.ini"
        if candidate.exists():
            cfg_path = candidate

    return derivatives_root, tsv_path, cfg_path, resolved_attempt, reports_dir


def _load_one_dataset_bundle(
    dataset_path: str,
    *,
    derivatives_base: Optional[str],
    input_tsv: Optional[str],
    attempt: Optional[int],
) -> QCDatasetBundle:
    (
        derivatives_root,
        tsv_path,
        cfg_path,
        resolved_attempt,
        reports_dir,
    ) = _resolve_input_paths(
        dataset_path=dataset_path,
        derivatives_base=derivatives_base,
        input_tsv=input_tsv,
        attempt=attempt,
    )

    df_raw = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    df = _prepare_table(df_raw)
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    df["dataset"] = dataset_name
    df["dataset_path"] = dataset_path

    return QCDatasetBundle(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        derivatives_root=derivatives_root,
        reports_dir=reports_dir,
        tsv_path=tsv_path,
        cfg_path=cfg_path,
        attempt=resolved_attempt,
        df=df,
        general_settings_snapshot=_general_settings_snapshot(derivatives_root),
        gqi_settings_snapshot=_config_snapshot(cfg_path),
    )


# ---------------------------------------------------------------------------
# Data adapters
# ---------------------------------------------------------------------------


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "subject" not in out.columns:
        out["subject"] = "n/a"
    if "task" not in out.columns:
        out["task"] = "unknown"

    out["subject"] = out["subject"].fillna("n/a").astype(str)
    out["task"] = out["task"].fillna("unknown").astype(str)
    out["task_label"] = out["task"].map(lambda t: f"task={t}")
    out["recording_label"] = out["subject"] + "|" + out["task_label"]

    if {"Muscle_events_num", "Muscle_events_total"}.issubset(out.columns):
        num = _coerce_numeric(out["Muscle_events_num"])
        den = _coerce_numeric(out["Muscle_events_total"])
        out["__muscle_event_rate_pct"] = np.where(
            np.isfinite(num) & np.isfinite(den) & (den > 0),
            100.0 * num / den,
            np.nan,
        )
    else:
        out["__muscle_event_rate_pct"] = np.nan

    return out


def _series_for_component(df: pd.DataFrame, spec: ComponentSpec, view: str) -> pd.Series:
    """Return one numeric series for a metric component and channel-type view."""

    def _from_col(col: Optional[str]) -> pd.Series:
        if col is None or col not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return _coerce_numeric(df[col])

    if view == "mag":
        s = _from_col(spec.mag_col)
        if s.notna().any():
            return s
        if spec.global_col:
            return _from_col(spec.global_col)
        return s

    if view == "grad":
        s = _from_col(spec.grad_col)
        if s.notna().any():
            return s
        if spec.global_col:
            return _from_col(spec.global_col)
        return s

    # combined: average mag/grad if possible, else fallback.
    sm = _from_col(spec.mag_col)
    sg = _from_col(spec.grad_col)
    if sm.notna().any() and sg.notna().any():
        both = pd.concat([sm, sg], axis=1)
        return both.mean(axis=1, skipna=True)
    if sm.notna().any():
        return sm
    if sg.notna().any():
        return sg
    if spec.global_col:
        return _from_col(spec.global_col)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _component_frame(df: pd.DataFrame, spec: ComponentSpec, view: str) -> pd.DataFrame:
    vals = _series_for_component(df, spec, view)
    base_cols = ["subject", "task", "task_label", "recording_label"]
    if "dataset" in df.columns:
        base_cols.append("dataset")
    out = df[base_cols].copy()
    out["value"] = vals
    out = out.loc[np.isfinite(out["value"])].copy()
    return out


def _with_all_tasks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    all_rows = out.copy()
    all_rows["task_label"] = "all tasks"
    return pd.concat([all_rows, out], ignore_index=True)


def _with_all_tasks_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Expanded labels for multi-dataset distributions.

    Includes:
    - per-dataset all-tasks distributions
    - per-dataset per-task distributions
    - pooled all-datasets all-tasks distribution
    """
    if df.empty:
        return df
    out = df.copy()
    out["dataset"] = out.get("dataset", "dataset").astype(str)
    out["task_label"] = out.get("task_label", "task=unknown").astype(str)

    per_task = out.copy()
    per_task["task_label"] = per_task["dataset"] + " | " + per_task["task_label"]
    per_task["__group_kind"] = "dataset_task"
    per_task["__group_dataset"] = per_task["dataset"].astype(str)
    per_task["__group_task"] = per_task["task"].astype(str)

    per_dataset_all = out.copy()
    per_dataset_all["task_label"] = per_dataset_all["dataset"] + " | all tasks"
    per_dataset_all["__group_kind"] = "dataset_all"
    per_dataset_all["__group_dataset"] = per_dataset_all["dataset"].astype(str)
    per_dataset_all["__group_task"] = "all tasks"

    pooled = out.copy()
    pooled["task_label"] = "all datasets | all tasks"
    pooled["__group_kind"] = "pooled"
    pooled["__group_dataset"] = "all datasets"
    pooled["__group_task"] = "all tasks"
    return pd.concat([pooled, per_dataset_all, per_task], ignore_index=True)


def _has_channel_type_data(df: pd.DataFrame, view: str) -> bool:
    """True when a channel-type view has at least one finite metric column."""
    if view not in {"mag", "grad"}:
        return True
    cols: List[str] = []
    for specs in COMPONENTS.values():
        for spec in specs:
            col = spec.mag_col if view == "mag" else spec.grad_col
            if col:
                cols.append(col)
    for col in cols:
        if col in df.columns and np.isfinite(_coerce_numeric(df[col])).any():
            return True
    return False


def _group_order(df: pd.DataFrame) -> List[str]:
    if "__group_kind" in df.columns:
        keys = (
            df[["task_label", "__group_kind", "__group_dataset", "__group_task"]]
            .drop_duplicates()
            .copy()
        )
        rank = {"pooled": 0, "dataset_all": 1, "dataset_task": 2}
        keys["__rank"] = keys["__group_kind"].map(rank).fillna(99).astype(int)
        keys = keys.sort_values(
            by=["__rank", "__group_dataset", "__group_task", "task_label"],
            kind="stable",
        )
        return keys["task_label"].astype(str).tolist()

    groups = sorted(set(df["task_label"].astype(str).tolist()))
    if "all tasks" in groups:
        groups.remove("all tasks")
        return ["all tasks", *groups]
    return groups


def _palette(labels: Sequence[str]) -> Dict[str, str]:
    base = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
        "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#2CA02C", "#D62728",
    ]
    return {label: base[i % len(base)] for i, label in enumerate(labels)}


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------


def _gaussian_kde_line(values: np.ndarray, n_points: int = 240) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmax <= vmin:
        vmax = vmin + 1.0

    std = float(np.nanstd(vals))
    bw = 1.06 * std * (vals.size ** (-1.0 / 5.0)) if std > 0 else 0.3
    bw = max(float(bw), max(abs(vmin), abs(vmax), 1.0) * 1e-3)

    x = np.linspace(vmin, vmax, n_points)
    diff = (x[:, None] - vals[None, :]) / bw
    y = np.exp(-0.5 * diff ** 2).sum(axis=1) / (vals.size * bw * np.sqrt(2.0 * np.pi))
    return x, y


def plot_violin_with_subject_points(
    df: pd.DataFrame,
    title: str,
    y_label: str,
    multi_dataset_mode: bool = False,
) -> Optional[go.Figure]:
    if df.empty:
        return None
    expanded = _with_all_tasks_by_dataset(df) if multi_dataset_mode else _with_all_tasks(df)
    groups = _group_order(expanded)
    colors = _palette(groups)
    labels = [label for label in groups if np.isfinite(expanded.loc[expanded["task_label"] == label, "value"]).any()]
    if not labels:
        return None
    xpos = {label: float(i) for i, label in enumerate(labels)}

    fig = go.Figure()
    for group in labels:
        g = expanded.loc[expanded["task_label"] == group].copy()
        vals = pd.to_numeric(g["value"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        group_id = f"group::{group}"
        fig.add_trace(
            go.Violin(
                x=np.full(vals.size, xpos[group], dtype=float),
                y=vals,
                name=f"{group} (n={vals.size})",
                legendgroup=group_id,
                box_visible=True,
                meanline_visible=False,
                points=False,
                line={"color": colors[group], "width": 1.25},
                fillcolor=colors[group],
                opacity=0.38,
                width=0.82,
                spanmode="soft",
                hovertemplate="group=" + group + "<br>value=%{y:.3g}<extra></extra>",
            )
        )

    # Subject dots: one robust value per subject per task group, with independent subject colors.
    subject_rows: List[Tuple[str, float, str, str]] = []
    tmp = expanded.loc[np.isfinite(pd.to_numeric(expanded["value"], errors="coerce"))].copy()
    tmp["subject"] = tmp["subject"].fillna("n/a").astype(str)
    if "dataset" in tmp.columns:
        tmp["dataset"] = tmp["dataset"].fillna("dataset").astype(str)
    else:
        tmp["dataset"] = "dataset"
    tmp["task"] = tmp["task"].fillna("unknown").astype(str)
    tmp["recording_label"] = tmp["recording_label"].fillna("n/a").astype(str)
    for (group, dataset, subject), grp in tmp.groupby(["task_label", "dataset", "subject"], dropna=False):
        vals = pd.to_numeric(grp["value"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0 or group not in xpos:
            continue
        value = float(np.nanmedian(vals))
        runs = sorted(set(grp["recording_label"].astype(str).tolist()))
        subject_rows.append(
            (
                str(group),
                value,
                f"{dataset}|{subject}",
                f"dataset={dataset}<br>subject={subject}<br>task_group={group}<br>n_rows={len(grp)}"
                + (f"<br>records={', '.join(runs[:4])}" + ("..." if len(runs) > 4 else "") if runs else "")
                + f"<br>value={value:.3g}",
            )
        )
    points_df = pd.DataFrame(subject_rows, columns=["label", "value", "subject", "hover"])
    if not points_df.empty:
        points_df["subj_code"] = pd.Categorical(points_df["subject"]).codes.astype(float)
        rng = np.random.default_rng(7)
        for label in labels:
            gp = points_df.loc[points_df["label"] == label].copy()
            if gp.empty:
                continue
            x_numeric = np.full(gp.shape[0], xpos[label], dtype=float)
            x_numeric = x_numeric + rng.uniform(-0.17, 0.17, size=x_numeric.size)
            fig.add_trace(
                go.Scattergl(
                    x=x_numeric,
                    y=gp["value"].to_numpy(dtype=float),
                    mode="markers",
                    marker={
                        "size": 6.5,
                        "color": gp["subj_code"].to_numpy(dtype=float),
                        "colorscale": "Turbo",
                        "opacity": 0.78,
                        "line": {"width": 0.4, "color": "rgba(20,20,20,0.55)"},
                        "showscale": False,
                    },
                    customdata=np.stack([gp["hover"].to_numpy()], axis=-1),
                    hovertemplate="%{customdata[0]}<extra></extra>",
                    legendgroup=f"group::{label}",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title="Task / condition",
        yaxis_title=y_label,
        legend={"orientation": "h", "y": 1.08, "x": 0.0, "groupclick": "togglegroup"},
        margin={"l": 55, "r": 30, "t": 120, "b": 60},
        height=620,
        violinmode="group",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[xpos[l] for l in labels],
        ticktext=labels,
        range=[-0.5, len(labels) - 0.2],
    )
    return fig


def plot_histogram_by_task(df: pd.DataFrame, title: str, x_label: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    multi_dataset_mode = "dataset" in df.columns and (df["dataset"].astype(str).nunique() > 1)
    expanded = _with_all_tasks_by_dataset(df) if multi_dataset_mode else _with_all_tasks(df)
    groups = _group_order(expanded)
    colors = _palette(groups)

    fig = go.Figure()
    for group in groups:
        g = expanded.loc[expanded["task_label"] == group]
        vals = g["value"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        group_id = f"group::{group}"
        fig.add_trace(
            go.Histogram(
                x=vals,
                name=f"{group} (n={vals.size})",
                legendgroup=group_id,
                marker={"color": colors[group]},
                opacity=0.38,
                histnorm="probability density",
                hovertemplate="group=" + group + "<br>value=%{x:.3g}<br>density=%{y:.3g}<extra></extra>",
            )
        )
        kde = _gaussian_kde_line(vals)
        if kde is not None:
            kx, ky = kde
            fig.add_trace(
                go.Scatter(
                    x=kx,
                    y=ky,
                    mode="lines",
                    name=f"{group} density",
                    legendgroup=group_id,
                    showlegend=False,
                    line={"color": colors[group], "width": 2.2},
                    hovertemplate="group=" + group + "<br>value=%{x:.3g}<br>density=%{y:.3g}<extra></extra>",
                )
            )

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title="Density",
        barmode="overlay",
        legend={"orientation": "h", "y": 1.08, "x": 0.0, "groupclick": "togglegroup"},
        margin={"l": 55, "r": 30, "t": 120, "b": 60},
        height=620,
    )
    return fig


def plot_density_by_task(df: pd.DataFrame, title: str, x_label: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    multi_dataset_mode = "dataset" in df.columns and (df["dataset"].astype(str).nunique() > 1)
    expanded = _with_all_tasks_by_dataset(df) if multi_dataset_mode else _with_all_tasks(df)
    groups = _group_order(expanded)
    colors = _palette(groups)

    fig = go.Figure()
    for group in groups:
        vals = expanded.loc[expanded["task_label"] == group, "value"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        kde = _gaussian_kde_line(vals)
        if kde is None:
            continue
        x, y = kde
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{group} (n={vals.size})",
                legendgroup=group,
                line={"color": colors[group], "width": 2.2},
                hovertemplate="group=" + group + "<br>value=%{x:.3g}<br>density=%{y:.3g}<extra></extra>",
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title="Kernel density",
        legend={"orientation": "h", "y": 1.08, "x": 0.0, "groupclick": "togglegroup"},
        margin={"l": 55, "r": 30, "t": 120, "b": 60},
        height=620,
    )
    return fig


def plot_task_profiles(df: pd.DataFrame, title: str, y_label: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    ordered_tasks = sorted(set(df["task_label"].astype(str).tolist()))
    if not ordered_tasks:
        return None
    subject_ids = sorted(set(df["subject"].astype(str).tolist()))
    colors = _palette(subject_ids)

    fig = go.Figure()
    for subject in subject_ids:
        sub = df.loc[df["subject"] == subject]
        vals_by_task = {}
        for task in ordered_tasks:
            vals = _coerce_numeric(sub.loc[sub["task_label"] == task, "value"]).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            vals_by_task[task] = float(np.mean(vals)) if vals.size else np.nan
        y = np.array([vals_by_task[t] for t in ordered_tasks], dtype=float)
        if not np.any(np.isfinite(y)):
            continue
        fig.add_trace(
            go.Scatter(
                x=ordered_tasks,
                y=y,
                mode="lines+markers",
                line={"width": 1.6, "color": colors[subject]},
                marker={"size": 6},
                name=subject,
                legendgroup=subject,
                showlegend=False,
                hovertemplate="subject=" + subject + "<br>task=%{x}<br>value=%{y:.3g}<extra></extra>",
            )
        )

    med = []
    for task in ordered_tasks:
        vals = _coerce_numeric(df.loc[df["task_label"] == task, "value"]).to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        med.append(np.nanmedian(vals) if vals.size else np.nan)
    fig.add_trace(
        go.Scatter(
            x=ordered_tasks,
            y=med,
            mode="lines+markers",
            line={"width": 3.2, "color": "#0B3D91"},
            marker={"size": 7},
            name="Cohort median",
            showlegend=True,
            hovertemplate="task=%{x}<br>median=%{y:.3g}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title="Task / condition",
        yaxis_title=y_label,
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        margin={"l": 55, "r": 30, "t": 120, "b": 60},
        height=620,
    )
    return fig


def plot_subject_task_heatmap(df: pd.DataFrame, title: str, color_title: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    pivot = (
        df.pivot_table(index="subject", columns="task_label", values="value", aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if pivot.empty:
        return None
    z = pivot.to_numpy(dtype=float)
    x = list(pivot.columns)
    y = list(pivot.index)
    custom = np.empty(z.shape + (2,), dtype=object)
    for i, subj in enumerate(y):
        for j, task in enumerate(x):
            custom[i, j, 0] = subj
            custom[i, j, 1] = task

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            customdata=custom,
            colorscale="Viridis",
            colorbar={"title": color_title},
            hovertemplate="subject=%{customdata[0]}<br>task=%{customdata[1]}<br>value=%{z:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title="Task / condition",
        yaxis_title="Subject",
        margin={"l": 70, "r": 40, "t": 90, "b": 60},
        height=max(560, 28 * len(y) + 220),
    )
    return fig


def _subject_ranking_table_html(df: pd.DataFrame, title: str, value_label: str) -> str:
    if df.empty:
        return "<p>No subject ranking available.</p>"
    grp = df.groupby("subject", dropna=False)
    tab = pd.DataFrame(
        {
            "subject": grp["subject"].first(),
            "n_tasks": grp["task_label"].nunique(),
            "mean": grp["value"].mean(),
            "median": grp["value"].median(),
            "upper_tail_p95": grp["value"].quantile(0.95),
        }
    ).reset_index(drop=True)
    tab = tab.sort_values("upper_tail_p95", ascending=False)
    rows: List[str] = []
    for _, r in tab.iterrows():
        mean_txt = f"{r['mean']:.3g}" if np.isfinite(r["mean"]) else "n/a"
        median_txt = f"{r['median']:.3g}" if np.isfinite(r["median"]) else "n/a"
        p95_txt = f"{r['upper_tail_p95']:.3g}" if np.isfinite(r["upper_tail_p95"]) else "n/a"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(r['subject']))}</td>"
            f"<td>{int(r['n_tasks'])}</td>"
            f"<td>{mean_txt}</td>"
            f"<td>{median_txt}</td>"
            f"<td>{p95_txt}</td>"
            + "</tr>"
        )
    return (
        "<div class='fig'>"
        f"<h4>{html.escape(title)}</h4>"
        "<div class='ranking-scroll'>"
        "<table class='ranking-table'>"
        "<thead><tr>"
        "<th>Subject</th>"
        "<th>N tasks</th>"
        f"<th>Mean ({html.escape(value_label)})</th>"
        f"<th>Median ({html.escape(value_label)})</th>"
        f"<th>p95 ({html.escape(value_label)})</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# HTML helpers (lazy loading + tabs)
# ---------------------------------------------------------------------------


def _reset_lazy_figure_store() -> None:
    global _LAZY_PLOT_COUNTER, _LAZY_PAYLOAD_COUNTER
    _LAZY_PLOT_PAYLOADS.clear()
    _LAZY_PLOT_COUNTER = count(1)
    _LAZY_PAYLOAD_COUNTER = count(1)


def _register_lazy_figure(fig: go.Figure, *, height_px: str) -> str:
    fig_id = f"lazy-qc-plot-{next(_LAZY_PLOT_COUNTER)}"
    payload_id = f"lazy-qc-payload-{next(_LAZY_PAYLOAD_COUNTER)}"
    payload_json = json.dumps(
        {
            "figure": fig.to_plotly_json(),
            "config": {"responsive": True, "displaylogo": False},
        },
        cls=PlotlyJSONEncoder,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    _LAZY_PLOT_PAYLOADS[payload_id] = payload_json
    return (
        f"<div id='{fig_id}' class='js-lazy-plot' data-payload-id='{payload_id}' "
        f"style='height:{height_px}; width:100%;'></div>"
    )


def _lazy_payload_script_tags_html() -> str:
    return "".join(
        f"<script id='{pid}' type='application/json'>{pjson}</script>"
        for pid, pjson in _LAZY_PLOT_PAYLOADS.items()
    )


def _figure_to_div(fig: Optional[go.Figure]) -> str:
    if fig is None:
        return "<p>No data available for this panel.</p>"
    height = fig.layout.height
    if height is None or (isinstance(height, (float, int)) and not np.isfinite(height)):
        height = 620
    return _register_lazy_figure(fig, height_px=f"{int(max(420, float(height)))}px")


def _figure_block(fig: Optional[go.Figure], interpretation: str) -> str:
    return (
        f"<div class='fig'>{_figure_to_div(fig)}</div>"
        f"<p class='fig-note'><strong>How to interpret:</strong> {interpretation}</p>"
    )


def _build_subtabs_html(group_id: str, tabs: Sequence[Tuple[str, str]], *, level: int = 1) -> str:
    if not tabs:
        return "<p>No content available.</p>"
    btns = []
    panels = []
    for idx, (label, html) in enumerate(tabs):
        panel_id = f"{group_id}-panel-{idx}"
        active = " active" if idx == 0 else ""
        btns.append(
            f"<button class='subtab-btn{active}' data-tab-group='{group_id}' data-target='{panel_id}'>{label}</button>"
        )
        panels.append(
            f"<div id='{panel_id}' class='subtab-content{active}' data-tab-group='{group_id}'>{html}</div>"
        )
    lvl = max(1, int(level))
    return f"<div class='subtab-group level-{lvl}'><div class='subtab-row'>{''.join(btns)}</div>{''.join(panels)}</div>"


def _config_snapshot(cfg_path: Optional[Path]) -> str:
    if cfg_path is None or (not cfg_path.exists()):
        return "No attempt-matched GQI config file was found."
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    lines = [f"source={cfg_path}"]
    for section in cfg.sections():
        lines.append(f"[{section}]")
        for key, value in cfg[section].items():
            lines.append(f"  {key}={value}")
        lines.append("")
    return "\n".join(lines)


def _general_settings_snapshot(derivatives_root: str) -> str:
    """Load the latest general MEGqc settings ini (same source used by QA report)."""
    config_dir = Path(derivatives_root) / "Meg_QC" / "config"
    ini_files = sorted(config_dir.glob("*.ini"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ini_files:
        return "No general MEGqc settings ini was found."

    chosen = ini_files[0]
    cfg = configparser.ConfigParser()
    cfg.read(chosen)

    lines = [f"source={chosen}"]
    disallowed_tokens = ("bad", "good", "reject", "flag", "exceed", "pass", "fail", "threshold")
    sections = ["default", "Filtering", "Epoching", "STD", "PTP_manual", "PSD", "ECG", "EOG", "Muscle"]
    for section in sections:
        if section not in cfg:
            continue
        safe_items: List[str] = []
        for key, value in cfg[section].items():
            key_l = key.lower()
            val_l = str(value).lower()
            if any(tok in key_l for tok in disallowed_tokens):
                continue
            if any(tok in val_l for tok in disallowed_tokens):
                continue
            safe_items.append(f"{key}={value}")
        if safe_items:
            lines.append(f"[{section}]")
            for item in safe_items:
                lines.append(f"  {item}")
            lines.append("")
        else:
            lines.append(f"[{section}] QA-safe settings view active")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _summary_table_html(df: pd.DataFrame, selected_cols: Sequence[str]) -> str:
    n_subjects = int(df["subject"].nunique()) if "subject" in df.columns else 0
    n_rows = int(len(df))
    tasks = sorted(set(df.get("task_label", pd.Series(dtype=str)).astype(str).tolist()))

    rows = []
    for col in selected_cols:
        if col not in df.columns:
            rows.append(f"<tr><td>{col}</td><td>0</td><td>{n_rows}</td></tr>")
            continue
        s = _coerce_numeric(df[col])
        avail = int(np.isfinite(s).sum())
        rows.append(f"<tr><td>{col}</td><td>{avail}</td><td>{max(0, n_rows-avail)}</td></tr>")

    task_rows = []
    if "task_label" in df.columns and "subject" in df.columns:
        grouped = (
            df.groupby("task_label", dropna=False)
            .agg(n_subjects=("subject", "nunique"), n_rows=("task_label", "size"))
            .reset_index()
            .sort_values("task_label")
        )
        for _, row in grouped.iterrows():
            task_rows.append(
                f"<tr><td>{html.escape(str(row['task_label']))}</td>"
                f"<td>{int(row['n_subjects'])}</td>"
                f"<td>{int(row['n_rows'])}</td></tr>"
            )
    task_table = (
        "<table><thead><tr><th>Task / condition</th><th>N subjects</th><th>N rows</th></tr></thead>"
        f"<tbody>{''.join(task_rows) if task_rows else '<tr><td colspan=3>n/a</td></tr>'}</tbody></table>"
    )

    return (
        "<div class='summary-grid'>"
        "<div class='tile'>"
        f"<h3>Dataset summary</h3><p><strong>N subjects:</strong> {n_subjects}</p>"
        f"<p><strong>N task-level rows:</strong> {n_rows}</p>"
        f"<p><strong>Tasks:</strong> {', '.join(tasks) if tasks else 'n/a'}</p>"
        "</div>"
        "<div class='tile'><h3>Subjects per task / condition</h3>"
        f"{task_table}</div>"
        "<div class='tile'><h3>Column availability</h3>"
        "<table><thead><tr><th>Column</th><th>Available n</th><th>Missing n</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</div>"
    )


def _metric_component_panel(
    df_view: pd.DataFrame,
    spec: ComponentSpec,
    view_label: str,
    metric_name: str,
    dataset_order: Optional[Sequence[str]] = None,
) -> str:
    if df_view.empty:
        return "<p>No data available for this component.</p>"
    multi_dataset_mode = "dataset" in df_view.columns and (df_view["dataset"].astype(str).nunique() > 1)

    violin = plot_violin_with_subject_points(
        df_view,
        title=f"{spec.label} by task ({view_label}) - violin",
        y_label=f"{spec.label} ({spec.unit})",
        multi_dataset_mode=multi_dataset_mode,
    )
    hist = plot_histogram_by_task(
        df_view,
        title=f"{spec.label} by task ({view_label}) - histogram",
        x_label=f"{spec.label} ({spec.unit})",
    )
    density = plot_density_by_task(
        df_view,
        title=f"{spec.label} by task ({view_label}) - density",
        x_label=f"{spec.label} ({spec.unit})",
    )
    profile = plot_task_profiles(
        df_view,
        title=f"Within-subject profiles across tasks ({view_label})",
        y_label=f"{spec.label} ({spec.unit})",
    )
    ranking_html = _subject_ranking_table_html(
        df_view,
        title=f"Subject ranking by {spec.label} ({view_label})",
        value_label=spec.unit,
    )

    dist_tabs = _build_subtabs_html(
        f"dist-{spec.key}-{view_label.lower().replace(' ', '-')}",
        [
            (
                "Violin",
                _figure_block(
                    violin,
                    (
                        f"X-axis is task / condition. Y-axis is {spec.label} in {spec.unit}. "
                        f"Each violin shows the full task-level distribution across subject-task rows; wider segments indicate higher density. "
                        "Jittered dots are one robust value per subject per task group (median if multiple rows exist) and are colored by subject identity."
                    ),
                ),
            ),
            (
                "Histogram",
                _figure_block(
                    hist,
                    (
                        f"X-axis is {spec.label} ({spec.unit}); Y-axis is probability density. "
                        "Bars compare task-wise empirical distributions. "
                        "Separation between task histograms indicates task-linked differences in QC burden."
                    ),
                ),
            ),
            (
                "Density",
                _figure_block(
                    density,
                    (
                        f"X-axis is {spec.label} ({spec.unit}); Y-axis is kernel density estimate. "
                        "Curves summarize distribution shape per task. "
                        "Broad or multi-peaked curves indicate high between-subject variability."
                    ),
                ),
            ),
        ],
        level=4,
    )

    detail_panels: str
    if multi_dataset_mode:
        profile_ds_tabs: List[Tuple[str, str]] = []
        ranking_ds_tabs: List[Tuple[str, str]] = []
        ds_iter = list(dataset_order) if dataset_order else sorted(df_view["dataset"].astype(str).unique())
        for ds in ds_iter:
            ds_df = df_view.loc[df_view["dataset"].astype(str) == ds].copy()
            if ds_df.empty:
                msg = f"<p>Dataset <strong>{html.escape(str(ds))}</strong> has no values for {spec.label} in {view_label} view.</p>"
                profile_ds_tabs.append((str(ds), msg))
                ranking_ds_tabs.append((str(ds), msg))
                continue
            ds_profile = plot_task_profiles(
                ds_df,
                title=f"Within-subject profiles across tasks ({view_label}) - {ds}",
                y_label=f"{spec.label} ({spec.unit})",
            )
            ds_ranking_html = _subject_ranking_table_html(
                ds_df,
                title=f"Subject ranking by {spec.label} ({view_label}) - {ds}",
                value_label=spec.unit,
            )
            profile_ds_tabs.append(
                (
                    ds,
                    _figure_block(
                        ds_profile,
                        (
                            f"X-axis is task / condition; Y-axis is {spec.label} ({spec.unit}). "
                            "Thin lines are within-subject trajectories and thick line is cohort median for this dataset only."
                        ),
                    ),
                )
            )
            ranking_ds_tabs.append(
                (
                    ds,
                    ds_ranking_html
                    + (
                        "<p class='fig-note'><strong>How to interpret:</strong> "
                        f"Scroll the ranking table to inspect subjects with highest upper-tail (p95) {spec.label} within dataset {ds}.</p>"
                    ),
                )
            )
        detail_panels = _build_subtabs_html(
            f"detail-datasets-view-{spec.key}-{view_label.lower().replace(' ', '-')}",
            [
                (
                    "Within-subject profiles across tasks",
                    _build_subtabs_html(
                        f"detail-datasets-profile-{spec.key}-{view_label.lower().replace(' ', '-')}",
                        profile_ds_tabs,
                        level=5,
                    ),
                ),
                (
                    "Subject ranking table",
                    _build_subtabs_html(
                        f"detail-datasets-ranking-{spec.key}-{view_label.lower().replace(' ', '-')}",
                        ranking_ds_tabs,
                        level=5,
                    ),
                ),
            ],
            level=4,
        )
    else:
        detail_panels = (
            _figure_block(
                profile,
                (
                    f"X-axis is task / condition; Y-axis is {spec.label} ({spec.unit}). "
                    "Thin lines are within-subject trajectories across tasks; the thick line is cohort median per task. "
                    "Use this panel to separate subject-specific shifts from cohort-level task effects."
                ),
            )
            + ranking_html
            + (
                "<p class='fig-note'><strong>How to interpret:</strong> "
                f"The ranking table is scrollable and lists subjects ordered by upper-tail (p95) {spec.label} across tasks. "
                f"Columns show mean, median, and p95 in {spec.unit}; use it to prioritize subject-level QC follow-up.</p>"
            )
        )

    return (
        "<div class='metric-block'>"
        f"<h4>{spec.label}</h4>"
        f"<p><strong>Unit:</strong> {spec.unit}</p>"
        + dist_tabs
        + detail_panels
        + "</div>"
    )


def _metric_tab_content(df: pd.DataFrame, metric_name: str, view: str, view_label: str) -> str:
    specs = COMPONENTS.get(metric_name, [])
    if not specs:
        return "<p>No metric definition available.</p>"

    comp_tabs: List[Tuple[str, str]] = []
    dataset_order = (
        sorted(df["dataset"].astype(str).unique())
        if ("dataset" in df.columns and df["dataset"].astype(str).nunique() > 1)
        else None
    )
    for spec in specs:
        comp_df = _component_frame(df, spec, view=view)
        if comp_df.empty:
            continue
        comp_tabs.append(
            (
                spec.label,
                _metric_component_panel(comp_df, spec, view_label, metric_name, dataset_order=dataset_order),
            )
        )

    if not comp_tabs:
        return "<p>No available columns for this metric in the selected attempt.</p>"

    return _build_subtabs_html(
        f"metric-{metric_name.lower()}-{view}",
        comp_tabs,
        level=3,
    )


def _tab_content(
    df: pd.DataFrame,
    top_tab: str,
    general_settings_snapshots: Dict[str, str],
    gqi_settings_snapshots: Dict[str, str],
) -> str:
    if top_tab == "Combined (mag+grad)":
        view = "combined"
    elif top_tab == "MAG":
        view = "mag"
    else:
        view = "grad"

    has_view_data = _has_channel_type_data(df, view) if view in {"mag", "grad"} else True
    metric_tabs: List[Tuple[str, str]] = []
    if has_view_data:
        for metric in METRIC_ORDER:
            metric_tabs.append((metric, _metric_tab_content(df, metric, view=view, view_label=top_tab)))
    else:
        metric_tabs.append(
            (
                "No data",
                (
                    "<section><h2>Channel-type availability</h2>"
                    f"<p>This dataset has no {view.upper()} information in the loaded QC TSV columns, so metric panels are not available for this tab.</p>"
                    "</section>"
                ),
            )
        )

    selected_cols = sorted({
        c
        for specs in COMPONENTS.values()
        for spec in specs
        for c in (spec.mag_col, spec.grad_col, spec.global_col)
        if c
    })

    snapshot_tiles: List[str] = []
    for ds_name in sorted(general_settings_snapshots):
        snapshot_tiles.append(
            "<div class='tile'>"
            f"<h3>General MEGqc settings snapshot ({html.escape(ds_name)})</h3>"
            f"<pre>{html.escape(general_settings_snapshots[ds_name])}</pre>"
            "</div>"
        )
    for ds_name in sorted(gqi_settings_snapshots):
        snapshot_tiles.append(
            "<div class='tile'>"
            f"<h3>GQI attempt settings snapshot ({html.escape(ds_name)})</h3>"
            f"<pre>{html.escape(gqi_settings_snapshots[ds_name])}</pre>"
            "</div>"
        )

    overview_html = (
        "<section>"
        "<h2>Cohort QC overview</h2>"
        "<p><strong>Purpose:</strong> summarize QC percentages and burdens for the selected channel-type view before diving into per-metric panels.</p>"
        + _summary_table_html(df, selected_cols)
        + "<div class='summary-grid'>"
        + "".join(snapshot_tiles)
        + "</div>"
        + "</section>"
    )

    metrics_html = (
        "<section>"
        "<h2>QC metrics details</h2>"
        "<p><strong>Important:</strong> each metric panel provides distribution views, subject trajectories, and ranking to complement QA with explicit QC burden summaries.</p>"
        + _build_subtabs_html(f"top-{re.sub(r'[^a-z0-9]+','-',top_tab.lower())}", metric_tabs, level=2)
        + "</section>"
    )

    return _build_subtabs_html(
        f"qc-sections-{re.sub(r'[^a-z0-9]+','-',top_tab.lower())}",
        [("Cohort QC overview", overview_html), ("QC metrics details", metrics_html)],
        level=1,
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _build_report_html(
    report_name: str,
    df: pd.DataFrame,
    source_rows: Sequence[str],
    attempt_text: str,
    general_settings_snapshots: Dict[str, str],
    gqi_settings_snapshots: Dict[str, str],
) -> str:
    _reset_lazy_figure_store()
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = getattr(meg_qc, "__version__", "unknown")

    tab_buttons = []
    tab_panels = []
    for idx, top_tab in enumerate(TOP_TABS):
        tab_id = f"qc-top-tab-{idx}"
        active = " active" if idx == 0 else ""
        tab_buttons.append(f"<button class='tab-btn{active}' data-target='{tab_id}'>{top_tab}</button>")
        tab_panels.append(
            f"<div id='{tab_id}' class='tab-content{active}'>{_tab_content(df, top_tab, general_settings_snapshots, gqi_settings_snapshots)}</div>"
        )

    lazy_payload_scripts = _lazy_payload_script_tags_html()
    sources_html = "".join(f"<li>{html.escape(line)}</li>" for line in source_rows)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QC group report {report_name}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      margin: 0;
      color: #1a1a1a;
      background: linear-gradient(135deg, #f6fbff, #edf6ff 45%, #f9fcff);
    }}
    main {{ max-width: 1360px; margin: 0 auto; padding: 28px 24px 56px; }}
    h1, h2, h3, h4 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 30px; letter-spacing: 0.2px; }}
    h2 {{ margin-top: 24px; font-size: 22px; }}
    h3 {{ margin-top: 12px; font-size: 18px; }}
    h4 {{ margin-top: 14px; font-size: 15px; color: #284b63; }}
    .report-header {{ display: flex; gap: 14px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; }}
    section {{
      background: rgba(255, 255, 255, 0.90);
      border: 1px solid #dce9f7;
      border-radius: 14px;
      padding: 16px 16px 10px;
      margin-top: 16px;
      box-shadow: 0 6px 24px rgba(5, 45, 79, 0.06);
    }}
    .metric-block {{ border-top: 1px solid #e7eef8; margin-top: 14px; padding-top: 12px; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      background: #f4f8fd;
      border: 1px solid #d8e4f3;
      border-radius: 8px;
      padding: 10px;
      font-size: 12px;
    }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
    th, td {{ border: 1px solid #d5e3f3; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #e7f1ff; color: #224c78; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .tile {{ border: 1px solid #d9e7f8; border-radius: 10px; padding: 10px; background: #fbfdff; }}
    .fig {{ border-top: 1px solid #e7eef8; margin-top: 10px; padding-top: 10px; overflow: visible; }}
    .fig .js-plotly-plot {{ width: 100% !important; }}
    .fig-note {{ margin: 6px 2px 14px; font-size: 13px; line-height: 1.45; color: #2a425f; }}
    .tab-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .tab-btn {{
      border: 1px solid #99bbe5;
      border-radius: 10px;
      background: #e7f1ff;
      padding: 8px 14px;
      cursor: pointer;
      font-size: 14px;
      color: #1f3b63;
      font-weight: 600;
    }}
    .tab-btn.active {{ background: #1d4ed8; border-color: #1d4ed8; color: #ffffff; font-weight: 700; }}
    .tab-content {{ display: none; margin-top: 8px; }}
    .tab-content.active {{ display: block; }}
    .subtab-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 10px; }}
    .subtab-group {{ border: 1px solid #d8e6f7; border-radius: 12px; padding: 10px 10px 8px; margin-top: 10px; }}
    .subtab-group.level-1 {{ background: #dbeafe; border-color: #7fb0ea; }}
    .subtab-group.level-2 {{ background: #ecf5ff; border-color: #a8c8ee; }}
    .subtab-group.level-3 {{ background: #f5f9ff; border-color: #c1d8f2; }}
    .subtab-group.level-4 {{ background: #ffffff; border-color: #d9e6f7; }}
    .subtab-group.level-1 > .subtab-row .subtab-btn {{ background: #cfe3ff; border-color: #79a9e4; font-weight: 600; }}
    .subtab-group.level-2 > .subtab-row .subtab-btn {{ background: #e2efff; border-color: #9fc2e8; }}
    .subtab-group.level-3 > .subtab-row .subtab-btn {{ background: #edf5ff; border-color: #b2cfea; }}
    .subtab-btn {{
      border: 1px solid #9fc2e8;
      border-radius: 9px;
      background: #eaf3ff;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 13px;
      color: #1f3f61;
      font-weight: 600;
    }}
    .subtab-btn.active {{ background: #cfe3ff; border-color: #79a9e4; color: #16a34a; font-weight: 700; }}
    .ranking-scroll {{
      max-height: 520px;
      overflow-y: auto;
      border: 1px solid #d7e5f7;
      border-radius: 10px;
      background: #ffffff;
    }}
    .ranking-table {{ width: 100%; border-collapse: collapse; margin-top: 0; }}
    .ranking-table thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #e7f1ff;
      color: #1f3f61;
    }}
    .ranking-table td, .ranking-table th {{ padding: 7px 9px; border: 1px solid #d5e3f3; }}
    .subtab-content {{ display: none; }}
    .subtab-content.active {{ display: block; }}
    .report-tools {{ display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }}
    .tool-btn {{
      border: 1px solid #7da9dd;
      border-radius: 9px;
      background: #dbeafe;
      padding: 7px 12px;
      font-size: 13px;
      color: #143a63;
      font-weight: 700;
      cursor: pointer;
    }}
    .tool-btn.active {{ background: #1d4ed8; border-color: #1d4ed8; color: #ffffff; }}
    .loading-overlay {{
      position: fixed;
      inset: 0;
      background: rgba(241, 245, 249, 0.95);
      backdrop-filter: blur(2px);
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: opacity 220ms ease;
    }}
    .loading-overlay.hidden {{ opacity: 0; pointer-events: none; }}
    .loading-card {{
      min-width: 280px;
      max-width: 420px;
      padding: 18px 20px;
      border-radius: 14px;
      border: 1px solid #93c5fd;
      background: #ffffff;
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.14);
      text-align: center;
    }}
    .loading-spinner {{
      width: 34px;
      height: 34px;
      margin: 0 auto 10px;
      border: 4px solid #bfdbfe;
      border-top-color: #1d4ed8;
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }}
    .loading-title {{ font-weight: 700; color: #1e3a5f; margin-bottom: 4px; }}
    .loading-subtitle {{ color: #334e68; font-size: 13px; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>
  <div id="report-loading-overlay" class="loading-overlay">
    <div class="loading-card">
      <div class="loading-spinner"></div>
      <div class="loading-title">Loading QC report</div>
      <div class="loading-subtitle">Rendering visible figures...</div>
    </div>
  </div>
  <main>
    <section>
      <div class="report-header">
        <div>
          <h1>QC group report: {report_name}</h1>
          <p><strong>Generated:</strong> {generated}</p>
          <p><strong>MEGqc version:</strong> {version}</p>
          <p><strong>Attempt selection:</strong> {html.escape(attempt_text)}</p>
          <p><strong>Important:</strong> this QC report complements QA views with burden percentages and ranking summaries per metric.</p>
        </div>
      </div>
      <div class="report-tools">
        <button id="grid-toggle-btn" class="tool-btn active" type="button">Hide grids</button>
      </div>
      <div class="tab-row">
        {''.join(tab_buttons)}
      </div>
      {''.join(tab_panels)}
      <section>
        <h2>Machine-readable input</h2>
        <ul>
          {sources_html}
        </ul>
      </section>
    </section>
  </main>
  {lazy_payload_scripts}
  <script>
    (function() {{
      const buttons = Array.from(document.querySelectorAll('.tab-btn'));
      const tabs = Array.from(document.querySelectorAll('.tab-content'));
      const gridToggleBtn = document.getElementById('grid-toggle-btn');
      const loadingOverlay = document.getElementById('report-loading-overlay');
      const lazyPayloadCache = {{}};
      let gridsVisible = true;

      function getPayloadFromScript(payloadId) {{
        if (!payloadId) return null;
        if (lazyPayloadCache[payloadId]) return lazyPayloadCache[payloadId];
        const payloadEl = document.getElementById(payloadId);
        if (!payloadEl || !payloadEl.textContent) return null;
        try {{
          const payload = JSON.parse(payloadEl.textContent);
          lazyPayloadCache[payloadId] = payload;
          payloadEl.textContent = '';
          if (payloadEl.parentNode) payloadEl.parentNode.removeChild(payloadEl);
          return payload;
        }} catch (err) {{
          return null;
        }}
      }}

      function hideLoadingOverlay() {{
        if (!loadingOverlay || loadingOverlay.dataset.hidden === '1') return;
        loadingOverlay.dataset.hidden = '1';
        loadingOverlay.classList.add('hidden');
        window.setTimeout(() => {{
          if (loadingOverlay && loadingOverlay.parentNode) loadingOverlay.parentNode.removeChild(loadingOverlay);
        }}, 260);
      }}

      function renderLazyInScope(scopeRoot) {{
        if (typeof Plotly === 'undefined') return Promise.resolve();
        const scope = scopeRoot || document;
        const placeholders = Array.from(scope.querySelectorAll('.js-lazy-plot'));
        const renderPromises = [];
        placeholders.forEach((el) => {{
          if (el.dataset.rendered === '1') return;
          if (el.offsetParent === null) return;
          const payloadId = el.dataset.payloadId;
          const payload = getPayloadFromScript(payloadId);
          if (!payload || !payload.figure) return;
          try {{
            const p = Plotly.newPlot(el, payload.figure.data || [], payload.figure.layout || {{}}, payload.config || {{responsive: true, displaylogo: false}});
            el.dataset.rendered = '1';
            renderPromises.push((p && typeof p.then === 'function') ? p.catch(() => undefined) : Promise.resolve());
          }} catch (err) {{}}
        }});
        return renderPromises.length ? Promise.all(renderPromises).then(() => undefined) : Promise.resolve();
      }}

      function applyGridToPlot(plotEl, show) {{
        if (typeof Plotly === 'undefined' || !plotEl) return;
        const layout = plotEl.layout || {{}};
        const axisKeys = Object.keys(layout).filter((k) => k.startsWith('xaxis') || k.startsWith('yaxis'));
        const relayoutUpdate = {{}};
        if (axisKeys.length === 0) {{
          relayoutUpdate['xaxis.showgrid'] = show;
          relayoutUpdate['yaxis.showgrid'] = show;
        }} else {{
          axisKeys.forEach((k) => {{ relayoutUpdate[`${{k}}.showgrid`] = show; }});
        }}
        try {{ Plotly.relayout(plotEl, relayoutUpdate); }} catch (err) {{}}
      }}

      function applyGridState(show, scopeRoot) {{
        const scope = scopeRoot || document;
        const plots = Array.from(scope.querySelectorAll('.js-plotly-plot'));
        plots.forEach((plotEl) => applyGridToPlot(plotEl, show));
      }}

      function resizePlots(targetId) {{
        if (typeof Plotly === 'undefined') return Promise.resolve();
        const root = document.getElementById(targetId);
        if (!root) return Promise.resolve();
        return renderLazyInScope(root).then(() => {{
          const plots = Array.from(root.querySelectorAll('.js-plotly-plot'));
          plots.forEach((plotEl) => {{ try {{ Plotly.Plots.resize(plotEl); }} catch (err) {{}} }});
          if (!gridsVisible) applyGridState(false, root);
        }});
      }}

      function activate(targetId) {{
        tabs.forEach(t => t.classList.toggle('active', t.id === targetId));
        buttons.forEach(b => b.classList.toggle('active', b.dataset.target === targetId));
        window.requestAnimationFrame(() => {{
          resizePlots(targetId).then(() => window.setTimeout(() => resizePlots(targetId), 120));
        }});
      }}

      buttons.forEach(btn => btn.addEventListener('click', () => activate(btn.dataset.target)));
      if (buttons.length > 0) activate(buttons[0].dataset.target);

      if (gridToggleBtn) {{
        gridToggleBtn.addEventListener('click', () => {{
          gridsVisible = !gridsVisible;
          applyGridState(gridsVisible, document);
          gridToggleBtn.textContent = gridsVisible ? 'Hide grids' : 'Show grids';
          gridToggleBtn.classList.toggle('active', gridsVisible);
        }});
      }}

      function activateSubtab(groupId, targetId) {{
        const subButtons = Array.from(document.querySelectorAll(`.subtab-btn[data-tab-group="${{groupId}}"]`));
        const subPanels = Array.from(document.querySelectorAll(`.subtab-content[data-tab-group="${{groupId}}"]`));
        subPanels.forEach(p => p.classList.toggle('active', p.id === targetId));
        subButtons.forEach(b => b.classList.toggle('active', b.dataset.target === targetId));
        const activePanel = document.getElementById(targetId);
        if (activePanel) renderLazyInScope(activePanel);
        const activeTop = tabs.find(t => t.classList.contains('active'));
        if (activeTop) window.requestAnimationFrame(() => resizePlots(activeTop.id));
      }}

      const subButtonsAll = Array.from(document.querySelectorAll('.subtab-btn'));
      const seenGroups = new Set();
      subButtonsAll.forEach(btn => {{
        const groupId = btn.dataset.tabGroup;
        if (!groupId) return;
        btn.addEventListener('click', () => activateSubtab(groupId, btn.dataset.target));
        seenGroups.add(groupId);
      }});
      seenGroups.forEach(groupId => {{
        const firstBtn = document.querySelector(`.subtab-btn[data-tab-group="${{groupId}}"]`);
        if (firstBtn) activateSubtab(groupId, firstBtn.dataset.target);
      }});

      window.addEventListener('resize', () => {{
        const active = tabs.find(t => t.classList.contains('active'));
        if (active) resizePlots(active.id);
      }});

      window.requestAnimationFrame(() => {{
        const active = tabs.find(t => t.classList.contains('active'));
        const p = active ? resizePlots(active.id) : Promise.resolve();
        p.then(() => window.setTimeout(hideLoadingOverlay, 120)).catch(() => hideLoadingOverlay());
      }});
    }})();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_group_qc_plots_meg_qc(
    dataset_path: str,
    input_tsv: Optional[str] = None,
    output_html: Optional[str] = None,
    attempt: Optional[int] = None,
    derivatives_base: Optional[str] = None,
) -> Optional[Path]:
    """Build one dataset-level QC HTML report from GQI summary TSV.

    Parameters
    ----------
    dataset_path
        BIDS dataset root.
    input_tsv
        Optional explicit GQI TSV path. If omitted, latest attempt is used.
    output_html
        Optional explicit output HTML path.
    attempt
        Optional attempt number (used only when ``input_tsv`` is omitted).
    derivatives_base
        Optional external derivatives parent root (same convention as QA CLI).

    Returns
    -------
    Path or None
        Path to generated HTML report, or ``None`` when generation fails.
    """
    try:
        bundle = _load_one_dataset_bundle(
            dataset_path=dataset_path,
            derivatives_base=derivatives_base,
            input_tsv=input_tsv,
            attempt=attempt,
        )
    except Exception as exc:
        print(f"___MEGqc___: QC group report: unable to resolve inputs: {exc}")
        return None

    report_html = _build_report_html(
        report_name=bundle.dataset_name,
        df=bundle.df,
        source_rows=[
            f"{bundle.dataset_name}: tsv={bundle.tsv_path}",
            f"{bundle.dataset_name}: config={bundle.cfg_path if bundle.cfg_path is not None else 'No matched config file found'}",
        ],
        attempt_text=str(bundle.attempt) if bundle.attempt is not None else "n/a",
        general_settings_snapshots={bundle.dataset_name: bundle.general_settings_snapshot},
        gqi_settings_snapshots={bundle.dataset_name: bundle.gqi_settings_snapshot},
    )

    if output_html:
        out_path = Path(output_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        suffix = f"_attempt_{bundle.attempt}" if bundle.attempt is not None else ""
        out_path = bundle.reports_dir / f"QC_group_report_{bundle.dataset_name}{suffix}.html"

    out_path.write_text(report_html, encoding="utf-8")
    print(f"___MEGqc___: QC group report created: {out_path}")
    print(f"___MEGqc___: Source TSV used: {bundle.tsv_path}")
    if bundle.cfg_path is not None:
        print(f"___MEGqc___: Config snapshot used: {bundle.cfg_path}")
    else:
        print("___MEGqc___: Config snapshot used: none found for selected attempt")
    print(f"___MEGqc___: Derivatives root: {bundle.derivatives_root}")
    return out_path


def make_group_qc_plots_multi_meg_qc(
    dataset_paths: Sequence[str],
    output_html: Optional[str] = None,
    attempt: Optional[int] = None,
    derivatives_base: Optional[str] = None,
) -> Optional[Path]:
    """Build one multi-dataset QC HTML report from multiple GQI summary TSVs."""
    if not dataset_paths:
        print("___MEGqc___: QC multi report: no datasets provided.")
        return None

    bundles: List[QCDatasetBundle] = []
    for ds in dataset_paths:
        try:
            bundle = _load_one_dataset_bundle(
                dataset_path=ds,
                derivatives_base=derivatives_base,
                input_tsv=None,
                attempt=attempt,
            )
        except Exception as exc:
            print(f"___MEGqc___: QC multi report: failed to load dataset '{ds}': {exc}")
            return None
        bundles.append(bundle)

    df_all = pd.concat([b.df for b in bundles], ignore_index=True)
    names = [b.dataset_name for b in bundles]
    report_name = " + ".join(names)
    source_rows: List[str] = []
    for b in bundles:
        source_rows.append(f"{b.dataset_name}: tsv={b.tsv_path}")
        source_rows.append(
            f"{b.dataset_name}: config={b.cfg_path if b.cfg_path is not None else 'No matched config file found'}"
        )
    attempts = sorted({str(b.attempt) if b.attempt is not None else "n/a" for b in bundles})
    attempt_text = ", ".join(attempts)
    general_snapshots = {b.dataset_name: b.general_settings_snapshot for b in bundles}
    gqi_snapshots = {b.dataset_name: b.gqi_settings_snapshot for b in bundles}

    report_html = _build_report_html(
        report_name=report_name,
        df=df_all,
        source_rows=source_rows,
        attempt_text=attempt_text,
        general_settings_snapshots=general_snapshots,
        gqi_settings_snapshots=gqi_snapshots,
    )

    if output_html:
        out_path = Path(output_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        suffix = f"_attempt_{attempt}" if attempt is not None else ""
        safe_name = "_".join(names)
        out_path = bundles[0].reports_dir / f"QC_group_report_multi_{safe_name}{suffix}.html"

    out_path.write_text(report_html, encoding="utf-8")
    print(f"___MEGqc___: QC multi-dataset report created: {out_path}")
    for b in bundles:
        print(f"___MEGqc___:   {b.dataset_name} -> {b.tsv_path}")
    return out_path
