"""Dataset-level QA plotting for MEGqc derivatives.

This module builds dataset-level QA HTML reports from machine-readable outputs
already produced by the MEGqc calculation step. It reads derivative TSV files
from ``derivatives/Meg_QC/calculation`` and writes group-level HTML reports to
``derivatives/Meg_QC/reports``.

Public entrypoint
-----------------
``make_group_plots_meg_qc(dataset_path, derivatives_base=None)``
"""

from __future__ import annotations

import configparser
import datetime as dt
import os
import re
import warnings
from collections import Counter, defaultdict
from itertools import count
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pandas.errors import DtypeWarning

import meg_qc


MODULES = ("STD", "PTP", "PSD", "ECG", "EOG", "Muscle")
CH_TYPES = ("mag", "grad")
MAX_POINTS_ECDF = 2500
MAX_POINTS_PROFILE = 2000
MAX_POINTS_VIOLIN = 3000
MAX_POINTS_SCATTER = 3500
MAX_HEATMAP_WINDOWS = 900
MAX_HEATMAP_CHANNELS = 350
MAX_RECORDINGS_OVERVIEW = 2200
MAX_SUBJECT_LINES = 160
_FIG_TOGGLE_COUNTER = count(1)


@dataclass
class RunMeta:
    run_key: str
    subject: str = "n/a"
    session: str = "n/a"
    task: str = "n/a"
    run: str = "n/a"
    condition: str = "n/a"
    acquisition: str = "n/a"
    recording: str = "n/a"
    processing: str = "n/a"


@dataclass
class RunRecord:
    meta: RunMeta
    files: Dict[str, Path] = field(default_factory=dict)


@dataclass
class SensorLayout:
    x: np.ndarray
    y: np.ndarray
    names: List[str]


@dataclass
class TopomapPayload:
    layout: SensorLayout
    values: np.ndarray


@dataclass
class RunMetricRow:
    run_key: str
    subject: str
    session: str
    task: str
    run: str
    condition: str
    acquisition: str
    recording: str
    processing: str
    channel_type: str
    std_median: float = np.nan
    std_upper_tail: float = np.nan
    std_median_norm: float = np.nan
    std_upper_tail_norm: float = np.nan
    ptp_median: float = np.nan
    ptp_upper_tail: float = np.nan
    ptp_median_norm: float = np.nan
    ptp_upper_tail_norm: float = np.nan
    mains_ratio: float = np.nan
    mains_harmonics_ratio: float = np.nan
    ecg_mean_abs_corr: float = np.nan
    ecg_p95_abs_corr: float = np.nan
    eog_mean_abs_corr: float = np.nan
    eog_p95_abs_corr: float = np.nan
    muscle_median: float = np.nan
    muscle_p95: float = np.nan


@dataclass
class ChTypeAccumulator:
    subjects: set = field(default_factory=set)
    run_count: int = 0
    runs_by_condition: Counter = field(default_factory=Counter)
    module_present: Counter = field(default_factory=Counter)
    module_missing: Counter = field(default_factory=Counter)

    std_dist_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    std_window_profiles: List[Dict[str, np.ndarray]] = field(default_factory=list)
    std_window_profiles_by_condition: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))

    ptp_dist_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    ptp_window_profiles: List[Dict[str, np.ndarray]] = field(default_factory=list)
    ptp_window_profiles_by_condition: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))

    psd_ratio_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    psd_profiles: List[Dict[str, np.ndarray]] = field(default_factory=list)
    psd_profiles_by_condition: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))

    ecg_corr_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    eog_corr_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    muscle_scalar_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    muscle_profiles: List[np.ndarray] = field(default_factory=list)
    muscle_profiles_by_condition: Dict[str, List[np.ndarray]] = field(default_factory=lambda: defaultdict(list))

    std_subject_profiles: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))
    ptp_subject_profiles: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))
    run_rows: List[RunMetricRow] = field(default_factory=list)

    std_heatmap_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    std_heatmap_score_by_condition: Dict[str, float] = field(default_factory=dict)
    std_heatmap_score_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    std_heatmap_upper_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    std_heatmap_upper_score_by_condition: Dict[str, float] = field(default_factory=dict)
    std_heatmap_sum_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    std_heatmap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)

    ptp_heatmap_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_heatmap_score_by_condition: Dict[str, float] = field(default_factory=dict)
    ptp_heatmap_score_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    ptp_heatmap_upper_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_heatmap_upper_score_by_condition: Dict[str, float] = field(default_factory=dict)
    ptp_heatmap_sum_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_heatmap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)

    std_topomap_by_condition: Dict[str, TopomapPayload] = field(default_factory=dict)
    ptp_topomap_by_condition: Dict[str, TopomapPayload] = field(default_factory=dict)
    psd_topomap_by_condition: Dict[str, TopomapPayload] = field(default_factory=dict)
    ecg_topomap_by_condition: Dict[str, TopomapPayload] = field(default_factory=dict)
    eog_topomap_by_condition: Dict[str, TopomapPayload] = field(default_factory=dict)
    std_topomap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_topomap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    psd_topomap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ecg_topomap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    eog_topomap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)

    source_paths: set = field(default_factory=set)


def resolve_output_roots(dataset_path: str, external_derivatives_root: Optional[str]) -> Tuple[str, str]:
    """Return output root and derivatives root respecting optional override."""
    ds_name = os.path.basename(os.path.normpath(dataset_path))
    output_root = (
        dataset_path
        if external_derivatives_root is None
        else os.path.join(external_derivatives_root, ds_name)
    )
    derivatives_root = os.path.join(output_root, "derivatives")
    os.makedirs(derivatives_root, exist_ok=True)
    return output_root, derivatives_root


def _parse_entities_from_run_key(run_key: str) -> RunMeta:
    entities = {}
    for token in run_key.split("_"):
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        entities[key] = value

    condition = entities.get("acq") or entities.get("proc") or entities.get("recording") or "n/a"
    return RunMeta(
        run_key=run_key,
        subject=entities.get("sub", "n/a"),
        session=entities.get("ses", "n/a"),
        task=entities.get("task", "n/a"),
        run=entities.get("run", "n/a"),
        condition=condition,
        acquisition=entities.get("acq", "n/a"),
        recording=entities.get("recording", "n/a"),
        processing=entities.get("proc", "n/a"),
    )


def _extract_run_and_desc(file_name: str) -> Tuple[Optional[str], Optional[str]]:
    match = re.search(r"_desc-([^_]+)_meg\.tsv$", file_name, flags=re.IGNORECASE)
    if not match:
        return None, None
    desc = match.group(1)
    run_key = file_name.split("_desc-")[0]
    return run_key, desc


def _discover_run_records(calculation_dir: Path) -> Dict[str, RunRecord]:
    tmp_files: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    run_meta: Dict[str, RunMeta] = {}

    for tsv_path in calculation_dir.rglob("*.tsv"):
        run_key, desc = _extract_run_and_desc(tsv_path.name)
        if run_key is None or desc is None:
            continue
        tmp_files[run_key][desc].append(tsv_path)
        if run_key not in run_meta:
            run_meta[run_key] = _parse_entities_from_run_key(run_key)

    run_records: Dict[str, RunRecord] = {}
    for run_key, desc_map in tmp_files.items():
        record = RunRecord(meta=run_meta[run_key], files={})
        for desc, paths in desc_map.items():
            # Keep only the newest file for a given run + desc.
            chosen = sorted(paths, key=lambda p: p.stat().st_mtime)[-1]
            record.files[desc] = chosen
        run_records[run_key] = record

    return run_records


def _read_tsv(path: Path) -> Optional[pd.DataFrame]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)
            return pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
    except Exception:
        return None


def _col_sort_key(name: str) -> Tuple[int, str]:
    tail = name.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail), name
    return 10**9, name


def _finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _as_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _robust_normalize_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full(arr.shape, np.nan, dtype=float)
    core = arr[finite]
    center = float(np.nanmedian(core))
    q25, q75 = np.nanquantile(core, [0.25, 0.75])
    scale = float(q75 - q25)
    if (not np.isfinite(scale)) or scale <= np.finfo(float).eps:
        std = float(np.nanstd(core))
        scale = std if std > np.finfo(float).eps else 1.0
    out = np.full(arr.shape, np.nan, dtype=float)
    out[finite] = (arr[finite] - center) / scale
    return out


def _normalize_profile_quantiles(profile: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    fields = ["q05", "q25", "q50", "q75", "q95", "top5"]
    all_values = []
    for field in fields:
        if field in profile:
            all_values.append(np.asarray(profile[field], dtype=float).reshape(-1))
    if not all_values:
        return dict(profile)

    pool = _finite_array(np.concatenate(all_values))
    if pool.size == 0:
        return dict(profile)
    center = float(np.nanmedian(pool))
    q25, q75 = np.nanquantile(pool, [0.25, 0.75])
    scale = float(q75 - q25)
    if (not np.isfinite(scale)) or scale <= np.finfo(float).eps:
        std = float(np.nanstd(pool))
        scale = std if std > np.finfo(float).eps else 1.0

    out = {}
    for key, vals in profile.items():
        arr = np.asarray(vals, dtype=float)
        if key in fields:
            out[key] = (arr - center) / scale
        else:
            out[key] = arr
    return out


def _safe_channel_subset(df: pd.DataFrame, ch_type: str) -> pd.DataFrame:
    if "Type" not in df.columns:
        return pd.DataFrame()
    ch_series = df["Type"].astype(str).str.lower()
    return df.loc[ch_series == ch_type].copy()


def _load_sensor_layout(path: Path) -> Dict[str, SensorLayout]:
    df = _read_tsv(path)
    if df is None:
        return {}

    x_col = _find_column(df, ["Sensor_location_0", "sensor_location_0", "loc_x", "x"])
    y_col = _find_column(df, ["Sensor_location_1", "sensor_location_1", "loc_y", "y"])
    name_col = _find_column(df, ["Name", "name", "Channel", "channel"])
    if x_col is None or y_col is None:
        return {}

    out: Dict[str, SensorLayout] = {}
    for ch_type in CH_TYPES:
        df_ch = _safe_channel_subset(df, ch_type)
        if df_ch.empty:
            continue
        x = pd.to_numeric(df_ch[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df_ch[y_col], errors="coerce").to_numpy(dtype=float)
        if name_col is not None:
            names = df_ch[name_col].fillna("").astype(str).tolist()
        else:
            names = [f"{ch_type}_{idx}" for idx in range(len(df_ch))]
        out[ch_type] = SensorLayout(x=x, y=y, names=names)
    return out


def _store_topomap_payload_if_missing(
    payloads_by_condition: Dict[str, TopomapPayload],
    condition_label: str,
    layout: Optional[SensorLayout],
    values: np.ndarray,
) -> None:
    if condition_label in payloads_by_condition or layout is None:
        return
    vals = np.asarray(values, dtype=float).reshape(-1)
    n = min(vals.size, layout.x.size, layout.y.size, len(layout.names))
    if n < 3:
        return
    payloads_by_condition[condition_label] = TopomapPayload(
        layout=SensorLayout(
            x=np.asarray(layout.x[:n], dtype=float),
            y=np.asarray(layout.y[:n], dtype=float),
            names=list(layout.names[:n]),
        ),
        values=vals[:n],
    )


def _update_topomap_payload_mean(
    payloads_by_condition: Dict[str, TopomapPayload],
    counts_by_condition: Dict[str, np.ndarray],
    condition_label: str,
    layout: Optional[SensorLayout],
    values: np.ndarray,
) -> None:
    if layout is None:
        return
    vals = np.asarray(values, dtype=float).reshape(-1)
    n = min(vals.size, layout.x.size, layout.y.size, len(layout.names))
    if n < 3:
        return
    vals = vals[:n]
    finite = np.isfinite(vals).astype(float)

    if condition_label not in payloads_by_condition:
        init_vals = np.where(np.isfinite(vals), vals, np.nan)
        payloads_by_condition[condition_label] = TopomapPayload(
            layout=SensorLayout(
                x=np.asarray(layout.x[:n], dtype=float),
                y=np.asarray(layout.y[:n], dtype=float),
                names=list(layout.names[:n]),
            ),
            values=init_vals,
        )
        counts_by_condition[condition_label] = finite
        return

    payload = payloads_by_condition[condition_label]
    old_vals = np.asarray(payload.values, dtype=float).reshape(-1)
    old_cnt = np.asarray(counts_by_condition.get(condition_label, np.zeros_like(old_vals)), dtype=float).reshape(-1)
    m = min(old_vals.size, vals.size, old_cnt.size)
    if m < 3:
        return

    old_vals = old_vals[:m]
    old_cnt = old_cnt[:m]
    new_vals = vals[:m]
    new_finite = np.isfinite(new_vals)
    num = np.where(np.isfinite(old_vals), old_vals, 0.0) * old_cnt + np.where(new_finite, new_vals, 0.0)
    den = old_cnt + new_finite.astype(float)
    out = np.full(m, np.nan, dtype=float)
    np.divide(num, np.maximum(den, np.finfo(float).eps), out=out, where=den > 0)

    payloads_by_condition[condition_label] = TopomapPayload(
        layout=SensorLayout(
            x=np.asarray(payload.layout.x[:m], dtype=float),
            y=np.asarray(payload.layout.y[:m], dtype=float),
            names=list(payload.layout.names[:m]),
        ),
        values=out,
    )
    counts_by_condition[condition_label] = den


def _load_std_or_ptp_matrix(path: Path, epoch_prefix: str, overall_column: str) -> Dict[str, np.ndarray]:
    df = _read_tsv(path)
    if df is None:
        return {}

    epoch_cols = sorted([c for c in df.columns if c.startswith(epoch_prefix)], key=_col_sort_key)
    out: Dict[str, np.ndarray] = {}

    for ch_type in CH_TYPES:
        df_ch = _safe_channel_subset(df, ch_type)
        if df_ch.empty:
            continue

        if epoch_cols:
            matrix = df_ch[epoch_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        elif overall_column in df_ch.columns:
            matrix = (
                df_ch[[overall_column]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            )
        else:
            continue

        if matrix.size > 0:
            out[ch_type] = matrix

    return out


def _load_psd_matrix(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    df = _read_tsv(path)
    if df is None:
        return {}

    freq_cols = []
    for col in df.columns:
        if col.startswith("PSD_Hz_"):
            try:
                freq = float(col.replace("PSD_Hz_", ""))
                freq_cols.append((freq, col))
            except ValueError:
                continue
    if not freq_cols:
        return {}

    freq_cols.sort(key=lambda x: x[0])
    freqs = np.asarray([fc[0] for fc in freq_cols], dtype=float)
    col_names = [fc[1] for fc in freq_cols]

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for ch_type in CH_TYPES:
        df_ch = _safe_channel_subset(df, ch_type)
        if df_ch.empty:
            continue
        matrix = df_ch[col_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if matrix.size > 0:
            out[ch_type] = (freqs, matrix)
    return out


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _load_correlation_scalar(path: Path, metric: str) -> Dict[str, np.ndarray]:
    corr_values = _load_correlation_values(path, metric)
    out: Dict[str, np.ndarray] = {}
    for ch_type, values in corr_values.items():
        vals = _finite_array(values)
        if vals.size:
            out[ch_type] = vals
    return out


def _load_correlation_values(path: Path, metric: str) -> Dict[str, np.ndarray]:
    df = _read_tsv(path)
    if df is None:
        return {}

    metric = metric.lower()
    col = _find_column(df, [f"{metric}_corr_coeff", f"{metric}_similarity_score"])
    if col is None:
        return {}

    out: Dict[str, np.ndarray] = {}
    for ch_type in CH_TYPES:
        df_ch = _safe_channel_subset(df, ch_type)
        if df_ch.empty:
            continue
        vals = pd.to_numeric(df_ch[col], errors="coerce").to_numpy(dtype=float)
        out[ch_type] = np.abs(vals)
    return out


def _parse_muscle_ch_type(value: str) -> Optional[str]:
    value = str(value).lower()
    if "grad" in value:
        return "grad"
    if "mag" in value:
        return "mag"
    return None


def _load_muscle_scores(path: Path) -> Dict[str, np.ndarray]:
    df = _read_tsv(path)
    if df is None or "scores_muscle" not in df.columns:
        return {}

    scores = pd.to_numeric(df["scores_muscle"], errors="coerce").to_numpy(dtype=float)
    scores = _finite_array(scores)
    if not scores.size:
        return {}

    ch_type = None
    if "ch_type" in df.columns:
        ch_non_na = df["ch_type"].dropna()
        if not ch_non_na.empty:
            ch_type = _parse_muscle_ch_type(str(ch_non_na.iloc[0]))
    if ch_type is None:
        ch_type = "mag"

    return {ch_type: scores}


def _condition_label(meta: RunMeta) -> str:
    parts = []
    if meta.task != "n/a":
        parts.append(f"task={meta.task}")
    if meta.condition != "n/a":
        parts.append(f"condition={meta.condition}")
    if not parts:
        return "all recordings"
    return ", ".join(parts)


def _profile_quantiles(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    q = np.nanquantile(matrix, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    top5 = np.full(matrix.shape[1], np.nan, dtype=float)
    for idx in range(matrix.shape[1]):
        col = _finite_array(matrix[:, idx])
        if col.size == 0:
            continue
        k = min(5, col.size)
        top5[idx] = float(np.mean(np.sort(col)[-k:]))
    return {"q05": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q95": q[4], "top5": top5}


def _compute_mains_ratio(psd_matrix: np.ndarray, freqs: np.ndarray, dropna: bool = True) -> np.ndarray:
    if psd_matrix.size == 0 or freqs.size == 0:
        return np.array([], dtype=float)

    fmax = float(np.nanmax(freqs))
    if not np.isfinite(fmax) or fmax <= 0:
        return np.array([], dtype=float)

    targets = []
    for base in (50.0, 60.0):
        harmonic = base
        while harmonic <= fmax + 1.0:
            targets.append(harmonic)
            harmonic += base

    if not targets:
        return np.array([], dtype=float)

    mask = np.zeros(freqs.shape, dtype=bool)
    for target in targets:
        mask |= np.abs(freqs - target) <= 1.0
    if not np.any(mask):
        return np.array([], dtype=float)

    total = np.nansum(psd_matrix, axis=1)
    mains = np.nansum(psd_matrix[:, mask], axis=1)
    ratio = mains / np.maximum(total, np.finfo(float).eps)
    return _finite_array(ratio) if dropna else ratio


def _compute_mains_and_harmonics_ratio(psd_matrix: np.ndarray, freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    psd_matrix = np.asarray(psd_matrix, dtype=float)
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    if psd_matrix.size == 0 or freqs.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    ref_profile = np.nanmedian(psd_matrix, axis=0)
    mains_base = _infer_mains_frequency(freqs, ref_profile)
    if mains_base is None:
        fallback = _compute_mains_ratio(psd_matrix, freqs, dropna=False)
        return fallback, np.full(fallback.shape, np.nan, dtype=float)

    mains_mask = np.abs(freqs - mains_base) <= 1.0
    harmonic_mask = np.zeros(freqs.shape, dtype=bool)
    harmonic = mains_base * 2.0
    fmax = float(np.nanmax(freqs))
    while harmonic <= fmax + 0.5:
        harmonic_mask |= np.abs(freqs - harmonic) <= 1.0
        harmonic += mains_base

    total = np.nansum(psd_matrix, axis=1)
    denom = np.maximum(total, np.finfo(float).eps)
    mains_ratio = np.nansum(psd_matrix[:, mains_mask], axis=1) / denom
    harmonics_ratio = np.nansum(psd_matrix[:, harmonic_mask], axis=1) / denom
    return mains_ratio, harmonics_ratio


def _infer_mains_frequency(freqs: np.ndarray, psd_q50: np.ndarray) -> Optional[float]:
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    psd_q50 = np.asarray(psd_q50, dtype=float).reshape(-1)
    if freqs.size == 0 or psd_q50.size == 0:
        return None

    n = min(freqs.size, psd_q50.size)
    freqs = freqs[:n]
    psd_q50 = psd_q50[:n]
    finite = np.isfinite(freqs) & np.isfinite(psd_q50)
    if not np.any(finite):
        return None
    freqs = freqs[finite]
    psd_q50 = psd_q50[finite]

    candidate_scores: Dict[float, float] = {}
    for base in (50.0, 60.0):
        band = (freqs >= base - 3.0) & (freqs <= base + 3.0)
        if np.any(band):
            candidate_scores[base] = float(np.nanmax(psd_q50[band]))
    if not candidate_scores:
        return None
    return max(candidate_scores.items(), key=lambda item: item[1])[0]


def _aggregate_window_profiles(profiles: List[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
    if not profiles:
        return None

    max_len = max(len(p["q50"]) for p in profiles)
    if max_len == 0:
        return None

    def _stack(field: str) -> np.ndarray:
        arr = np.full((len(profiles), max_len), np.nan, dtype=float)
        for i, prof in enumerate(profiles):
            vals = np.asarray(prof[field], dtype=float).reshape(-1)
            if vals.size == 0:
                continue
            arr[i, : len(vals)] = vals
        return np.nanmedian(arr, axis=0)

    agg = {
        "x": np.arange(max_len, dtype=int),
        "q05": _stack("q05"),
        "q25": _stack("q25"),
        "q50": _stack("q50"),
        "q75": _stack("q75"),
        "q95": _stack("q95"),
    }
    if any("top5" in p for p in profiles):
        agg["top5"] = _stack("top5")
    return agg


def _aggregate_psd_profiles(profiles: List[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
    if not profiles:
        return None

    ref = max(profiles, key=lambda p: len(p["freqs"]))
    ref_freqs = _finite_array(ref["freqs"])
    if ref_freqs.size == 0:
        return None

    def _to_ref_grid(freqs: np.ndarray, values: np.ndarray) -> np.ndarray:
        freqs = _finite_array(freqs)
        values = np.asarray(values, dtype=float).reshape(-1)
        if freqs.size == 0 or values.size == 0:
            return np.full(ref_freqs.shape, np.nan, dtype=float)

        # Keep only finite pairs and ensure increasing x for interpolation.
        pair_mask = np.isfinite(freqs) & np.isfinite(values[: len(freqs)])
        x = freqs[pair_mask]
        y = values[: len(freqs)][pair_mask]
        if x.size == 0:
            return np.full(ref_freqs.shape, np.nan, dtype=float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if x.size == ref_freqs.size and np.allclose(x, ref_freqs):
            return y
        return np.interp(ref_freqs, x, y, left=np.nan, right=np.nan)

    def _stack(field: str) -> np.ndarray:
        arr = np.full((len(profiles), ref_freqs.size), np.nan, dtype=float)
        for i, prof in enumerate(profiles):
            arr[i, :] = _to_ref_grid(prof["freqs"], prof[field])
        return np.nanmedian(arr, axis=0)

    out = {
        "x": ref_freqs,
        "q05": _stack("q05"),
        "q25": _stack("q25"),
        "q50": _stack("q50"),
        "q75": _stack("q75"),
        "q95": _stack("q95"),
    }
    if any("top5" in p for p in profiles):
        out["top5"] = _stack("top5")
    return out


def _aggregate_muscle_profiles(profiles: List[np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    if not profiles:
        return None

    max_len = max(len(p) for p in profiles)
    if max_len == 0:
        return None

    arr = np.full((len(profiles), max_len), np.nan, dtype=float)
    for i, profile in enumerate(profiles):
        vals = np.asarray(profile, dtype=float).reshape(-1)
        if vals.size == 0:
            continue
        arr[i, : len(vals)] = vals

    return {
        "x": np.arange(max_len, dtype=int),
        "q05": np.nanquantile(arr, 0.05, axis=0),
        "q25": np.nanquantile(arr, 0.25, axis=0),
        "q50": np.nanquantile(arr, 0.50, axis=0),
        "q75": np.nanquantile(arr, 0.75, axis=0),
        "q95": np.nanquantile(arr, 0.95, axis=0),
    }


def _update_representative_matrix(
    matrix_by_condition: Dict[str, np.ndarray],
    score_by_condition: Dict[str, float],
    score_history: Dict[str, List[float]],
    condition_label: str,
    matrix: np.ndarray,
    run_score: float,
) -> None:
    if matrix.size == 0 or not np.isfinite(run_score):
        return

    history = score_history[condition_label]
    history.append(float(run_score))
    finite_history = _finite_array(history)
    if finite_history.size == 0:
        return
    target = float(np.nanmedian(finite_history))

    if condition_label not in matrix_by_condition:
        matrix_by_condition[condition_label] = np.asarray(matrix, dtype=float)
        score_by_condition[condition_label] = float(run_score)
        return

    prev_score = score_by_condition.get(condition_label, np.nan)
    if (not np.isfinite(prev_score)) or abs(run_score - target) <= abs(prev_score - target):
        matrix_by_condition[condition_label] = np.asarray(matrix, dtype=float)
        score_by_condition[condition_label] = float(run_score)


def _update_upper_tail_matrix(
    matrix_by_condition: Dict[str, np.ndarray],
    score_by_condition: Dict[str, float],
    condition_label: str,
    matrix: np.ndarray,
    run_score: float,
) -> None:
    """Keep the matrix with the largest run-level upper-tail score per condition."""
    if matrix.size == 0 or not np.isfinite(run_score):
        return
    prev = score_by_condition.get(condition_label, float("-inf"))
    if condition_label not in matrix_by_condition or (not np.isfinite(prev)) or run_score >= prev:
        matrix_by_condition[condition_label] = np.asarray(matrix, dtype=float)
        score_by_condition[condition_label] = float(run_score)


def _accumulate_matrix_mean(
    sum_by_condition: Dict[str, np.ndarray],
    count_by_condition: Dict[str, np.ndarray],
    condition_label: str,
    matrix: np.ndarray,
) -> None:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return
    vals = np.nan_to_num(arr, nan=0.0)
    cnt = np.isfinite(arr).astype(float)

    if condition_label not in sum_by_condition:
        sum_by_condition[condition_label] = vals
        count_by_condition[condition_label] = cnt
        return

    prev_sum = sum_by_condition[condition_label]
    prev_cnt = count_by_condition[condition_label]
    n_rows = max(prev_sum.shape[0], vals.shape[0])
    n_cols = max(prev_sum.shape[1], vals.shape[1])

    def _pad(a: np.ndarray) -> np.ndarray:
        if a.shape == (n_rows, n_cols):
            return a
        return np.pad(a, ((0, n_rows - a.shape[0]), (0, n_cols - a.shape[1])), mode="constant", constant_values=0.0)

    sum_by_condition[condition_label] = _pad(prev_sum) + _pad(vals)
    count_by_condition[condition_label] = _pad(prev_cnt) + _pad(cnt)


def _mean_matrix_from_sum_count(sum_matrix: np.ndarray, count_matrix: np.ndarray) -> np.ndarray:
    s = np.asarray(sum_matrix, dtype=float)
    c = np.asarray(count_matrix, dtype=float)
    if s.shape != c.shape or s.size == 0:
        return np.array([], dtype=float)
    out = np.full(s.shape, np.nan, dtype=float)
    np.divide(s, np.maximum(c, np.finfo(float).eps), out=out, where=c > 0)
    return out


def _ecdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = _finite_array(values)
    if arr.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    x = np.sort(arr)
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    return x, y


def _downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=int)
    idx = np.linspace(0, length - 1, num=max_points, dtype=int)
    return np.unique(idx)


def _make_ecdf_figure(values_by_condition: Dict[str, List[float]], title: str, x_title: str) -> Optional[go.Figure]:
    fig = go.Figure()
    for label in sorted(values_by_condition):
        x, y = _ecdf(values_by_condition[label])
        if x.size == 0:
            continue
        keep = _downsample_indices(x.size, MAX_POINTS_ECDF)
        x = x[keep]
        y = y[keep]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{label} (n={x.size})",
                line={"width": 2},
            )
        )

    if not fig.data:
        return None

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_title,
        yaxis_title="ECDF",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
    )
    return fig


def plot_violin_channel_distribution(
    values_by_condition: Dict[str, List[float]],
    title: str,
    y_title: str,
) -> Optional[go.Figure]:
    fig = go.Figure()
    for label in sorted(values_by_condition):
        vals = _finite_array(values_by_condition[label])
        if vals.size == 0:
            continue
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        fig.add_trace(
            go.Violin(
                x=np.repeat(label, vals.size),
                y=vals,
                name=f"{label} (n={vals.size})",
                box_visible=True,
                meanline_visible=False,
                points=False,
                line={"width": 1.1},
            )
        )

    if not fig.data:
        return None
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task / condition",
        yaxis_title=y_title,
        template="plotly_white",
        violinmode="group",
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
    )
    return fig


def plot_quantile_band_timecourse(
    quantiles: Dict[str, np.ndarray],
    title: str,
    x_title: str,
    y_title: str,
) -> Optional[go.Figure]:
    if quantiles is None:
        return None

    x = np.asarray(quantiles["x"], dtype=float)
    q05 = np.asarray(quantiles["q05"], dtype=float)
    q25 = np.asarray(quantiles["q25"], dtype=float)
    q50 = np.asarray(quantiles["q50"], dtype=float)
    q75 = np.asarray(quantiles["q75"], dtype=float)
    q95 = np.asarray(quantiles["q95"], dtype=float)
    top5 = np.asarray(quantiles["top5"], dtype=float) if "top5" in quantiles else None

    valid = np.isfinite(x) & (
        np.isfinite(q05) | np.isfinite(q25) | np.isfinite(q50) | np.isfinite(q75) | np.isfinite(q95)
    )
    if not np.any(valid):
        return None
    x = x[valid]
    q05 = q05[valid]
    q25 = q25[valid]
    q50 = q50[valid]
    q75 = q75[valid]
    q95 = q95[valid]
    if top5 is not None:
        top5 = top5[valid]

    keep = _downsample_indices(x.size, MAX_POINTS_PROFILE)
    x = x[keep]
    q05 = q05[keep]
    q25 = q25[keep]
    q50 = q50[keep]
    q75 = q75[keep]
    q95 = q95[keep]
    if top5 is not None:
        top5 = top5[keep]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=q95, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"},
            showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q05,
            mode="lines",
            line={"width": 0, "color": "rgba(0,0,0,0)"},
            fill="tonexty",
            fillcolor="rgba(88, 166, 255, 0.20)",
            name="Middle 90% of channels",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=q75, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"},
            showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q25,
            mode="lines",
            line={"width": 0, "color": "rgba(0,0,0,0)"},
            fill="tonexty",
            fillcolor="rgba(30, 136, 229, 0.28)",
            name="Middle 50% of channels (IQR)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q50,
            mode="lines",
            line={"width": 2.3, "color": "#164B84"},
            name="Median across channels",
        )
    )
    if top5 is not None and np.any(np.isfinite(top5)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=top5,
                mode="lines",
                line={"width": 1.8, "color": "#D1495B", "dash": "dash"},
                name="Mean of top 5 channels",
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
    )
    return fig


def _make_quantile_band_figure(
    quantiles: Dict[str, np.ndarray],
    title: str,
    x_title: str,
    y_title: str,
) -> Optional[go.Figure]:
    return plot_quantile_band_timecourse(quantiles, title, x_title, y_title)


def plot_psd_median_band(
    psd_quantiles: Dict[str, np.ndarray],
    title: str,
) -> Optional[go.Figure]:
    if psd_quantiles is None:
        return None
    fig = plot_quantile_band_timecourse(
        psd_quantiles,
        title=title,
        x_title="Frequency (Hz)",
        y_title="Relative power",
    )
    if fig is None:
        return None

    freqs = np.asarray(psd_quantiles["x"], dtype=float)
    q50 = np.asarray(psd_quantiles["q50"], dtype=float)
    mains = _infer_mains_frequency(freqs, q50)
    finite_freqs = _finite_array(freqs)
    if mains is not None and finite_freqs.size:
        max_freq = float(np.nanmax(finite_freqs))
        harmonic = mains
        first = True
        while harmonic <= max_freq + 0.5:
            fig.add_vline(
                x=float(harmonic),
                line_width=1.2,
                line_dash="dot",
                line_color="rgba(108,117,125,0.60)",
            )
            if first:
                fig.add_annotation(
                    x=float(harmonic),
                    y=1.02,
                    yref="paper",
                    showarrow=False,
                    text=f"Mains frequency ({int(round(mains))} Hz)",
                    font={"size": 11, "color": "#495057"},
                    xanchor="left",
                )
                first = False
            harmonic += mains

    positive = np.isfinite(q50) & (q50 > 0)
    if np.any(positive):
        fig.update_yaxes(type="log")
    return fig


def _robust_bounds(values: np.ndarray) -> Optional[Tuple[float, float]]:
    vals = _finite_array(values)
    if vals.size == 0:
        return None
    lo, hi = np.nanquantile(vals, [0.02, 0.98])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        hi = lo + np.finfo(float).eps
    return float(lo), float(hi)


def plot_heatmap_sorted_channels_windows(
    matrix: np.ndarray,
    title: str,
    color_title: str,
    summary_mode: str = "median",
) -> Optional[go.Figure]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None

    if summary_mode == "upper_tail":
        ch_summary = np.nanquantile(arr, 0.95, axis=1)
    else:
        ch_summary = np.nanmedian(arr, axis=1)
    order = np.argsort(np.nan_to_num(ch_summary, nan=-np.inf))[::-1]
    arr = arr[order, :]
    ch_summary = ch_summary[order]

    row_keep = _downsample_indices(arr.shape[0], MAX_HEATMAP_CHANNELS)
    col_keep = _downsample_indices(arr.shape[1], MAX_HEATMAP_WINDOWS)

    z = arr[row_keep][:, col_keep]
    x = np.arange(arr.shape[1], dtype=int)[col_keep]
    y = np.arange(arr.shape[0], dtype=int)[row_keep]
    top_q05 = np.nanquantile(arr[:, col_keep], 0.05, axis=0)
    top_q25 = np.nanquantile(arr[:, col_keep], 0.25, axis=0)
    top_q50 = np.nanmedian(arr[:, col_keep], axis=0)
    top_q75 = np.nanquantile(arr[:, col_keep], 0.75, axis=0)
    top_q95 = np.nanquantile(arr[:, col_keep], 0.95, axis=0)
    side_strip = ch_summary[row_keep]

    bounds = _robust_bounds(z)
    if bounds is None:
        return None
    zmin, zmax = bounds

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.30, 0.70],
        column_widths=[0.84, 0.16],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "heatmap"}, {"type": "xy"}],
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=top_q95,
            mode="lines",
            line={"width": 0.0, "color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=top_q05,
            mode="lines",
            line={"width": 0.0, "color": "rgba(0,0,0,0)"},
            fill="tonexty",
            fillcolor="rgba(88,166,255,0.16)",
            name="Middle 90% of channels",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=top_q75,
            mode="lines",
            line={"width": 0.0, "color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=top_q25,
            mode="lines",
            line={"width": 0.0, "color": "rgba(0,0,0,0)"},
            fill="tonexty",
            fillcolor="rgba(30,136,229,0.22)",
            name="Middle 50% of channels (IQR)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=top_q50,
            mode="lines",
            line={"width": 2.1, "color": "#184E77"},
            name="Median across channels",
            hovertemplate="epoch=%{x}<br>median=%{y:.3g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar={"title": color_title},
            hovertemplate="epoch=%{x}<br>channel=%{y}<br>value=%{z:.3g}<extra></extra>",
            showscale=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=side_strip,
            y=y,
            mode="lines",
            line={"width": 2.0, "color": "#2A9D8F"},
            name="Channel summary",
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Channel quantile bands", row=1, col=1)
    fig.update_xaxes(title_text="Epoch index", row=2, col=1)
    fig.update_yaxes(title_text="Sorted channel index", autorange="reversed", row=2, col=1)
    fig.update_xaxes(title_text="Channel summary", row=2, col=2)
    fig.update_yaxes(showticklabels=False, autorange="reversed", row=2, col=2)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        margin={"l": 55, "r": 30, "t": 74, "b": 55},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "xanchor": "left", "x": 0},
    )
    return fig


def plot_topomap_if_available(
    payload: Optional[TopomapPayload],
    title: str,
    color_title: str,
) -> Optional[go.Figure]:
    if payload is None:
        return None
    x = np.asarray(payload.layout.x, dtype=float).reshape(-1)
    y = np.asarray(payload.layout.y, dtype=float).reshape(-1)
    values = np.asarray(payload.values, dtype=float).reshape(-1)
    n = min(x.size, y.size, values.size, len(payload.layout.names))
    if n < 3:
        return None

    x = x[:n]
    y = y[:n]
    values = values[:n]
    names = np.asarray(payload.layout.names[:n], dtype=object)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if np.sum(mask) < 3:
        return None

    # Spread overlapping points slightly so Elekta triplets (1 mag + 2 grad) remain visible.
    x_plot = x[mask].copy()
    y_plot = y[mask].copy()
    v_plot = values[mask]
    n_plot = names[mask]
    xy_key = np.round(np.column_stack([x_plot, y_plot]), 6)
    unique, inv = np.unique(xy_key, axis=0, return_inverse=True)
    xrange = float(np.nanmax(x_plot) - np.nanmin(x_plot)) if x_plot.size else 1.0
    yrange = float(np.nanmax(y_plot) - np.nanmin(y_plot)) if y_plot.size else 1.0
    base_r = max(xrange, yrange) * 0.012
    if not np.isfinite(base_r) or base_r <= 0:
        base_r = 1e-3
    for k in range(unique.shape[0]):
        idx = np.where(inv == k)[0]
        if idx.size <= 1:
            continue
        angles = np.linspace(0.0, 2.0 * np.pi, num=idx.size, endpoint=False)
        x_plot[idx] += base_r * np.cos(angles)
        y_plot[idx] += base_r * np.sin(angles)

    fig = go.Figure(
        go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="markers",
            text=n_plot,
            customdata=v_plot,
            hovertemplate="%{text}<br>value=%{customdata:.3g}<extra></extra>",
            marker={
                "size": 11,
                "color": v_plot,
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": color_title},
                "line": {"width": 0.5, "color": "#2F3E46"},
            },
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        margin={"l": 40, "r": 40, "t": 70, "b": 35},
    )
    return fig


def _figure_to_div(fig: Optional[go.Figure]) -> str:
    if fig is None:
        return "<p>No distribution is available for this section.</p>"
    height = fig.layout.height
    if height is None or not np.isfinite(height):
        height = 640
    height_px = f"{int(max(420, float(height)))}px"
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        default_height=height_px,
        default_width="100%",
        config={"responsive": True, "displaylogo": False},
    )


def _trace_numeric(values) -> Optional[np.ndarray]:
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.any(np.isfinite(arr)):
        return None
    return arr


def _normalize_figure_for_mode(fig: Optional[go.Figure], mode: str) -> Optional[go.Figure]:
    if fig is None:
        return None
    out = go.Figure(fig)
    for tr in out.data:
        t = getattr(tr, "type", "")
        if mode in ("y", "auto"):
            if t in ("scatter", "scattergl", "violin", "box"):
                arr = _trace_numeric(getattr(tr, "y", None))
                if arr is not None:
                    tr.y = _robust_normalize_array(arr)
        if mode in ("x", "auto"):
            if t == "histogram":
                arr = _trace_numeric(getattr(tr, "x", None))
                if arr is not None:
                    tr.x = _robust_normalize_array(arr)
            elif mode == "x" and t in ("scatter", "scattergl"):
                arr = _trace_numeric(getattr(tr, "x", None))
                if arr is not None:
                    tr.x = _robust_normalize_array(arr)
        if mode in ("z", "auto") and t == "heatmap":
            z = np.asarray(getattr(tr, "z", np.array([])), dtype=float)
            if z.size:
                tr.z = _robust_normalize_array(z.reshape(-1)).reshape(z.shape)
        if mode == "color" and t in ("scatter", "scattergl"):
            marker = getattr(tr, "marker", None)
            if marker is not None and getattr(marker, "color", None) is not None:
                arr = _trace_numeric(marker.color)
                if arr is not None:
                    marker.color = _robust_normalize_array(arr)
            arr = _trace_numeric(getattr(tr, "customdata", None))
            if arr is not None:
                tr.customdata = _robust_normalize_array(arr)

    if mode in ("y", "auto"):
        for axis_name in ("yaxis", "yaxis2", "yaxis3", "yaxis4", "yaxis5", "yaxis6"):
            axis = getattr(out.layout, axis_name, None)
            if axis is not None and getattr(axis, "title", None) is not None and getattr(axis.title, "text", None):
                axis.title.text = f"{axis.title.text} [normalized]"
    if mode == "x":
        for axis_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4", "xaxis5", "xaxis6"):
            axis = getattr(out.layout, axis_name, None)
            if axis is not None and getattr(axis, "title", None) is not None and getattr(axis.title, "text", None):
                axis.title.text = f"{axis.title.text} [normalized]"
    return out


def _auto_figure_details(fig: Optional[go.Figure]) -> str:
    if fig is None:
        return ""

    json_fig = fig.to_plotly_json()
    layout = json_fig.get("layout", {})
    title_text = ""
    title_obj = layout.get("title", {})
    if isinstance(title_obj, dict):
        title_text = str(title_obj.get("text", "") or "")

    x_titles = []
    y_titles = []
    for key, val in layout.items():
        if key.startswith("xaxis") and isinstance(val, dict):
            t = val.get("title", {})
            if isinstance(t, dict) and t.get("text"):
                x_titles.append(str(t.get("text")))
        if key.startswith("yaxis") and isinstance(val, dict):
            t = val.get("title", {})
            if isinstance(t, dict) and t.get("text"):
                y_titles.append(str(t.get("text")))
    x_titles = sorted(set(x_titles))
    y_titles = sorted(set(y_titles))

    trace_types = [getattr(tr, "type", "") for tr in fig.data]
    has_heatmap = any(t == "heatmap" for t in trace_types)
    has_violin = any(t == "violin" for t in trace_types)
    has_hist = any(t == "histogram" for t in trace_types)
    has_table = any(t == "table" for t in trace_types)
    scatter_modes = [str(getattr(tr, "mode", "")) for tr in fig.data if getattr(tr, "type", "") in ("scatter", "scattergl")]
    has_lines = any("lines" in m for m in scatter_modes)
    has_markers = any("markers" in m for m in scatter_modes)

    color_title = None
    for tr in fig.data:
        marker = getattr(tr, "marker", None)
        if marker is not None and getattr(marker, "colorbar", None) is not None:
            cb = marker.colorbar
            if getattr(cb, "title", None) is not None and getattr(cb.title, "text", None):
                color_title = str(cb.title.text)
                break
        if getattr(tr, "colorbar", None) is not None:
            cb = tr.colorbar
            if getattr(cb, "title", None) is not None and getattr(cb.title, "text", None):
                color_title = str(cb.title.text)
                break

    all_axis_text = " | ".join(x_titles + y_titles + ([color_title] if color_title else []))
    context_blob = f"{title_text} | {all_axis_text}".lower()
    looks_global = any(tok in context_blob for tok in ("global", "across subjects", "all recordings", "pooled", "all channels"))
    looks_subject = any(tok in context_blob for tok in ("subject", "recording", "fingerprint", "drill-down", "within-subject"))

    details = []
    if x_titles:
        details.append(f"X-axis encodes {', '.join(x_titles)}.")
    if y_titles:
        details.append(f"Y-axis encodes {', '.join(y_titles)}.")
    if color_title:
        details.append(f"The color scale encodes {color_title}.")

    if has_violin:
        details.append(
            "In violin views, each violin is a full empirical distribution; violin width reflects density, and jittered points represent recording-level values with subject identity in hover text."
        )
    if has_hist:
        details.append(
            "In histogram views, bars show probability density and the smooth overlaid curve is a kernel density estimate of the same observations."
        )
    if has_heatmap:
        details.append(
            "In heatmaps, each cell is one channel-by-epoch value. Vertical stripe patterns indicate simultaneous across-channel shifts by epoch, while isolated horizontal structure indicates channel-specific burden."
        )
    if has_lines:
        details.append(
            "In line/band profiles, the central line is the median trend and shaded envelopes indicate spread (IQR and 5-95% where present), so widening bands reflect increased cross-channel variability."
        )
    if has_markers and not has_violin and not has_hist and not has_table:
        details.append(
            "In scatter views, each point is one recording-level summary. Outlying points indicate recordings with atypical metric combinations relative to the cohort cloud."
        )
    if has_table:
        details.append(
            "In tables, each row corresponds to one subject/recording entity and each column is a robust metric summary in native units."
        )

    if "std" in context_blob:
        details.append(
            "STD summarizes channel variability amplitude per epoch; stable cohorts usually show gradual epoch trends, while heavy upper tails indicate a subset of channels with larger variability."
        )
    if "ptp" in context_blob:
        details.append(
            "PtP summarizes excursion amplitude. A heavy upper tail indicates channels with larger peak-to-peak bursts than the cohort median."
        )
    if "psd" in context_blob or "frequency" in context_blob or "mains" in context_blob:
        details.append(
            "PSD-based plots use frequency in Hz on the x-axis. Relative-power peaks around mains frequency and harmonics indicate narrow-band dominance in spectral content."
        )
    if "ecg" in context_blob or "eog" in context_blob or "|r|" in context_blob or "corr" in context_blob:
        details.append(
            "Correlation magnitude plots are unitless (|r|). Higher values indicate stronger temporal similarity between channel signals and physiological reference traces."
        )
    if "muscle" in context_blob or "z-score" in context_blob:
        details.append(
            "Muscle score is unitless (z-score). Elevated upper-tail epochs indicate brief high-frequency burden, while persistently elevated profiles indicate sustained high-frequency activity."
        )

    if looks_global:
        details.append(
            "This is a global pooled view: it summarizes the cohort footprint and should not be interpreted as subject identification."
        )
    if looks_subject:
        details.append(
            "This is a subject-aware view: it supports locating outlying subjects/recordings, but it does not by itself determine downstream handling."
        )

    if not details:
        details.append(
            "Interpret values using axis titles and hover text: these define units, entities, and the aggregation level of each visual element."
        )
    return " ".join(details)


def _figure_block(
    fig: Optional[go.Figure],
    interpretation: str,
    *,
    normalized_variant: bool = False,
    norm_mode: str = "y",
) -> str:
    auto_details = _auto_figure_details(fig)
    detailed_note = f"{interpretation} {auto_details}".strip()
    if (not normalized_variant) or fig is None:
        return (
            f'<div class="fig">{_figure_to_div(fig)}</div>'
            f'<p class="fig-note"><strong>How to interpret:</strong> {detailed_note}</p>'
        )

    norm_fig = _normalize_figure_for_mode(fig, norm_mode)
    toggle_id = f"fig-toggle-{next(_FIG_TOGGLE_COUNTER)}"
    raw_div = _figure_to_div(fig)
    norm_div = _figure_to_div(norm_fig)
    return (
        f"<div class='fig-switch' data-fig-toggle='{toggle_id}'>"
        f"<button class='fig-switch-btn active' data-target='{toggle_id}-raw'>Raw</button>"
        f"<button class='fig-switch-btn' data-target='{toggle_id}-norm'>Normalized</button>"
        f"</div>"
        f"<div id='{toggle_id}-raw' class='fig-view active'>{raw_div}</div>"
        f"<div id='{toggle_id}-norm' class='fig-view'>{norm_div}</div>"
        f"<p class='fig-note'><strong>How to interpret:</strong> {detailed_note} "
        "The normalized view applies a robust scaling (median/IQR) to improve comparability of shape across conditions without changing rank structure.</p>"
    )


def _condition_figure_blocks(
    figures_by_condition: Dict[str, Optional[go.Figure]],
    interpretation: str,
    *,
    normalized_variant: bool = False,
    norm_mode: str = "y",
) -> str:
    chunks = []
    for condition in sorted(figures_by_condition):
        fig = figures_by_condition[condition]
        if fig is None:
            continue
        chunks.append(f"<h4>{condition}</h4>")
        chunks.append(
            _figure_block(
                fig,
                interpretation,
                normalized_variant=normalized_variant,
                norm_mode=norm_mode,
            )
        )
    return "".join(chunks) if chunks else "<p>No distribution is available for this section.</p>"


def _build_subtabs_html(group_id: str, tabs: Sequence[Tuple[str, str]], *, level: int = 1) -> str:
    if not tabs:
        return "<p>No content available.</p>"
    buttons = []
    contents = []
    for idx, (label, html) in enumerate(tabs):
        panel_id = f"{group_id}-panel-{idx}"
        active = " active" if idx == 0 else ""
        buttons.append(
            f"<button class='subtab-btn{active}' data-tab-group='{group_id}' data-target='{panel_id}'>{label}</button>"
        )
        contents.append(
            f"<div id='{panel_id}' class='subtab-content{active}' data-tab-group='{group_id}'>{html}</div>"
        )
    level = max(1, int(level))
    return f"<div class='subtab-group level-{level}'><div class='subtab-row'>{''.join(buttons)}</div>{''.join(contents)}</div>"


def _topomap_blocks(
    payloads_by_condition: Dict[str, TopomapPayload],
    title_prefix: str,
    color_title: str,
    interpretation: str,
    *,
    normalized_variant: bool = False,
) -> str:
    figures: Dict[str, Optional[go.Figure]] = {}
    for cond, payload in payloads_by_condition.items():
        figures[cond] = plot_topomap_if_available(
            payload,
            title=f"{title_prefix} ({cond})",
            color_title=color_title,
        )
    if not any(fig is not None for fig in figures.values()):
        return "<p>Topographic maps not shown: channel positions not available in stored outputs.</p>"
    return _condition_figure_blocks(
        figures,
        interpretation,
        normalized_variant=normalized_variant,
        norm_mode="color",
    )


def _load_settings_snapshot(derivatives_root: str) -> str:
    config_dir = Path(derivatives_root) / "Meg_QC" / "config"
    ini_files = sorted(config_dir.glob("*.ini"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ini_files:
        return "No settings snapshot file was found."

    chosen = ini_files[0]
    cfg = configparser.ConfigParser()
    cfg.read(chosen)

    sections = ["default", "Epoching", "STD", "PTP_manual", "PSD", "ECG", "EOG", "Muscle"]
    lines = [f"source={chosen}"]
    disallowed_tokens = ("bad", "good", "reject", "flag", "exceed", "pass", "fail", "threshold")
    for section in sections:
        if section in cfg:
            qa_safe_items = []
            for key, value in cfg[section].items():
                key_l = key.lower()
                val_l = str(value).lower()
                if any(tok in key_l for tok in disallowed_tokens):
                    continue
                if any(tok in val_l for tok in disallowed_tokens):
                    continue
                qa_safe_items.append(f"{key}={value}")
            if qa_safe_items:
                lines.append(f"[{section}] {', '.join(qa_safe_items)}")
            else:
                lines.append(f"[{section}] QA-safe settings view active")

    return "\n".join(lines) if lines else f"source={chosen}"


def _summary_table_html(acc: ChTypeAccumulator) -> str:
    rows = []
    for module in MODULES:
        rows.append(
            "<tr>"
            f"<td>{module}</td>"
            f"<td>{acc.module_present.get(module, 0)}</td>"
            f"<td>{acc.module_missing.get(module, 0)}</td>"
            "</tr>"
        )
    module_table = (
        "<table>"
        "<thead><tr><th>Module</th><th>Available n</th><th>Missing n</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )

    condition_rows = []
    for cond, count in sorted(acc.runs_by_condition.items(), key=lambda x: x[0]):
        condition_rows.append(f"<tr><td>{cond}</td><td>{count}</td></tr>")
    condition_table = (
        "<table>"
        "<thead><tr><th>Task / condition</th><th>Runs</th></tr></thead>"
        f"<tbody>{''.join(condition_rows)}</tbody>"
        "</table>"
    )

    return (
        "<div class='summary-grid'>"
        "<div class='tile'>"
        f"<h3>Dataset summary</h3>"
        f"<p><strong>N subjects:</strong> {len(acc.subjects)}</p>"
        f"<p><strong>N runs:</strong> {acc.run_count}</p>"
        "</div>"
        "<div class='tile'>"
        "<h3>Runs per task / condition</h3>"
        f"{condition_table}"
        "</div>"
        "<div class='tile'>"
        "<h3>Missingness per module</h3>"
        f"{module_table}"
        "</div>"
        "</div>"
    )


def _paths_html(source_paths: set) -> str:
    if not source_paths:
        return "<p>No machine-readable derivatives were consumed.</p>"

    paths = sorted(source_paths)
    max_lines = 80
    visible = paths[:max_lines]
    hidden = max(0, len(paths) - max_lines)
    items = "".join(f"<li><code>{p}</code></li>" for p in visible)
    if hidden > 0:
        items += f"<li>... {hidden} additional paths omitted for brevity.</li>"
    return f"<ul>{items}</ul>"


def _run_rows_dataframe(rows: List[RunMetricRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([row.__dict__ for row in rows])
    numeric_cols = [
        "std_median", "std_upper_tail", "std_median_norm", "std_upper_tail_norm",
        "ptp_median", "ptp_upper_tail", "ptp_median_norm", "ptp_upper_tail_norm",
        "mains_ratio", "mains_harmonics_ratio", "ecg_mean_abs_corr", "ecg_p95_abs_corr",
        "eog_mean_abs_corr", "eog_p95_abs_corr", "muscle_median", "muscle_p95",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["subject", "session", "task", "run", "condition", "acquisition", "recording", "processing", "channel_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("n/a").astype(str)

    def _cond_display(row: pd.Series) -> str:
        parts = []
        if row["task"] != "n/a":
            parts.append(f"task={row['task']}")
        if row["condition"] != "n/a":
            parts.append(f"condition={row['condition']}")
        return ", ".join(parts) if parts else "all recordings"

    df["condition_label"] = df.apply(_cond_display, axis=1)
    df["hover_entities"] = (
        "sub=" + df["subject"]
        + "<br>ses=" + df["session"]
        + "<br>task=" + df["task"]
        + "<br>run=" + df["run"]
        + "<br>acq=" + df["acquisition"]
        + "<br>recording=" + df["recording"]
        + "<br>proc=" + df["processing"]
        + "<br>channel_type=" + df["channel_type"]
        + "<br>run_key=" + df["run_key"].astype(str)
    )
    return df


def _subject_color_map(subjects: List[str]) -> Dict[str, str]:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#003f5c", "#58508d",
        "#bc5090", "#ff6361", "#ffa600", "#2f4b7c", "#665191", "#a05195",
        "#d45087", "#f95d6a", "#ff7c43", "#ffa600",
    ]
    out = {}
    for idx, subject in enumerate(sorted(subjects)):
        out[subject] = palette[idx % len(palette)]
    return out


def _condition_symbol_map(conditions: List[str]) -> Dict[str, str]:
    symbols = ["circle", "diamond", "square", "cross", "triangle-up", "triangle-down", "x", "star"]
    out = {}
    for idx, cond in enumerate(sorted(conditions)):
        out[cond] = symbols[idx % len(symbols)]
    return out


def plot_subject_condition_violin(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
) -> Optional[go.Figure]:
    if df.empty or value_col not in df.columns:
        return None
    data = df.loc[np.isfinite(df[value_col])].copy()
    if data.empty:
        return None

    fig = go.Figure()
    for label in sorted(data["condition_label"].unique()):
        vals = data.loc[data["condition_label"] == label, value_col].to_numpy(dtype=float)
        vals = _finite_array(vals)
        if vals.size == 0:
            continue
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        fig.add_trace(
            go.Violin(
                x=np.repeat(label, vals.size),
                y=vals,
                name=f"{label} (n={vals.size})",
                box_visible=True,
                meanline_visible=False,
                points=False,
                line={"width": 1.0},
                opacity=0.45,
            )
        )

    subject_codes = pd.Categorical(data["subject"]).codes.astype(float)
    keep_points = _downsample_indices(len(data), min(MAX_POINTS_SCATTER, len(data)))
    scatter_data = data.iloc[keep_points]
    scatter_codes = subject_codes[keep_points]
    fig.add_trace(
        go.Scattergl(
            x=scatter_data["condition_label"],
            y=scatter_data[value_col],
            mode="markers",
            name="Recording summaries",
            marker={
                "size": 6,
                "color": scatter_codes,
                "colorscale": "Turbo",
                "opacity": 0.72,
                "line": {"width": 0.35, "color": "rgba(24,24,24,0.45)"},
                "showscale": False,
            },
            customdata=np.stack([scatter_data["hover_entities"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>value=%{y:.3g}<extra></extra>",
            showlegend=False,
        )
    )

    if not fig.data:
        return None
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task / condition",
        yaxis_title=y_label,
        template="plotly_white",
        violinmode="group",
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
    )
    return fig


def plot_run_fingerprint_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> Optional[go.Figure]:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    data = df.loc[np.isfinite(df[x_col]) & np.isfinite(df[y_col])].copy()
    if data.empty:
        return None

    symbol_map = _condition_symbol_map(sorted(data["condition_label"].unique()))
    subject_map = _subject_color_map(sorted(data["subject"].astype(str).unique().tolist()))
    keep_points = _downsample_indices(len(data), min(MAX_POINTS_SCATTER, len(data)))
    data = data.iloc[keep_points].copy()

    fig = go.Figure()
    for cond in sorted(symbol_map):
        dcond = data.loc[data["condition_label"] == cond].copy()
        if dcond.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=dcond[x_col],
                y=dcond[y_col],
                mode="markers",
                name=cond,
                legendgroup=f"condition-{cond}",
                marker={
                    "size": 8,
                    "symbol": symbol_map[cond],
                    "color": [subject_map.get(str(s), "#4F6F84") for s in dcond["subject"].astype(str)],
                    "line": {"width": 0.6, "color": "#2b2d42"},
                    "opacity": 0.86,
                    "showscale": False,
                },
                customdata=np.stack([dcond["hover_entities"]], axis=-1),
                hovertemplate="%{customdata[0]}<br>condition="
                + cond
                + "<br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>",
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
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def plot_subject_ranking_table(df: pd.DataFrame, title: str) -> Optional[go.Figure]:
    if df.empty:
        return None

    def _p95(series: pd.Series) -> float:
        vals = _finite_array(series.to_numpy(dtype=float))
        return float(np.nanquantile(vals, 0.95)) if vals.size else np.nan

    grouped = df.groupby("subject", dropna=False).agg(
        n_recordings=("run_key", "nunique"),
        std_p95=("std_upper_tail", _p95),
        ptp_p95=("ptp_upper_tail", _p95),
        mains_mean=("mains_ratio", "mean"),
        muscle_p95=("muscle_p95", _p95),
        eog_mean=("eog_mean_abs_corr", "mean"),
        ecg_mean=("ecg_mean_abs_corr", "mean"),
    ).reset_index()

    if grouped.empty:
        return None

    grouped = grouped.sort_values(
        by=["std_p95", "ptp_p95", "mains_mean", "muscle_p95", "eog_mean", "ecg_mean"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))

    display_cols = ["rank", "subject", "n_recordings", "std_p95", "ptp_p95", "mains_mean", "muscle_p95", "eog_mean", "ecg_mean"]
    header = ["Rank", "Subject", "N recordings", "STD p95", "PtP p95", "Mains mean", "Muscle p95", "EOG |r| mean", "ECG |r| mean"]
    cells = []
    for col in display_cols:
        if col in ("rank", "subject", "n_recordings"):
            cells.append(grouped[col].tolist())
        else:
            cells.append([f"{v:.3g}" if np.isfinite(v) else "n/a" for v in grouped[col].to_numpy(dtype=float)])

    fig = go.Figure(
        data=[
            go.Table(
                header={"values": header, "fill_color": "#eef5fd", "align": "left"},
                cells={"values": cells, "fill_color": "#ffffff", "align": "left"},
            )
        ]
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def plot_subject_metric_heatmap(df: pd.DataFrame, title: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    metrics = {
        "STD p95": "std_upper_tail",
        "PtP p95": "ptp_upper_tail",
        "Mains mean": "mains_ratio",
        "Muscle p95": "muscle_p95",
        "EOG |r| mean": "eog_mean_abs_corr",
        "ECG |r| mean": "ecg_mean_abs_corr",
    }
    agg = df.groupby("subject", dropna=False).agg({col: "median" for col in metrics.values()})
    if agg.empty:
        return None

    raw = np.full((agg.shape[0], len(metrics)), np.nan, dtype=float)
    z = np.full((agg.shape[0], len(metrics)), np.nan, dtype=float)
    for idx, col in enumerate(metrics.values()):
        raw[:, idx] = agg[col].to_numpy(dtype=float)
        z[:, idx] = _robust_normalize_array(raw[:, idx])

    bounds = _robust_bounds(z)
    zmin = bounds[0] if bounds is not None else None
    zmax = bounds[1] if bounds is not None else None

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(metrics.keys()),
            y=agg.index.tolist(),
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "Normalized value (robust z)"},
            customdata=np.stack([raw], axis=-1),
            hovertemplate="subject=%{y}<br>metric=%{x}<br>raw=%{customdata[0]:.3g}<br>normalized=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Metric",
        yaxis_title="Subject",
        template="plotly_white",
        margin={"l": 70, "r": 20, "t": 65, "b": 50},
    )
    fig.update_xaxes(tickangle=-22)
    return fig


def plot_subject_condition_effect(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
) -> Optional[go.Figure]:
    if df.empty or metric_col not in df.columns:
        return None
    data = df.loc[np.isfinite(df[metric_col])].copy()
    if data.empty:
        return None
    pivot = data.pivot_table(
        index="subject",
        columns="condition_label",
        values=metric_col,
        aggfunc="median",
    )
    if pivot.shape[1] < 2:
        return None
    conditions = list(pivot.columns)
    color_map = _subject_color_map(list(pivot.index))

    fig = go.Figure()
    for subject in pivot.index:
        vals = pivot.loc[subject].to_numpy(dtype=float)
        if np.sum(np.isfinite(vals)) < 2:
            continue
        fig.add_trace(
            go.Scatter(
                x=conditions,
                y=vals,
                mode="lines+markers",
                name=f"sub-{subject}",
                line={"width": 1.2, "color": color_map[subject]},
                marker={"size": 6},
                showlegend=False,
                hovertemplate=f"sub={subject}<br>condition=%{{x}}<br>value=%{{y:.3g}}<extra></extra>",
            )
        )

    if not fig.data:
        return None
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task / condition",
        yaxis_title=y_label,
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
    )
    return fig


def _values_with_task_agnostic(values_by_condition: Dict[str, List[float]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"all tasks": []}
    for cond, vals in sorted(values_by_condition.items()):
        finite = _finite_array(vals).tolist()
        out[cond] = finite
        out["all tasks"].extend(finite)
    if not out["all tasks"]:
        del out["all tasks"]
    return out


def _run_values_by_condition(df: pd.DataFrame, value_col: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    if df.empty or value_col not in df.columns:
        return {}
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    labels = df["condition_label"].astype(str).tolist() if "condition_label" in df.columns else ["all recordings"] * len(df)
    for label, val in zip(labels, values):
        if np.isfinite(val):
            out[label].append(float(val))
    return dict(out)


def _epoch_values_from_profiles(profiles_by_condition: Dict[str, List[Dict[str, np.ndarray]]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    for cond, profiles in profiles_by_condition.items():
        for prof in profiles:
            if "q50" not in prof:
                continue
            out[cond].extend(_finite_array(prof["q50"]).tolist())
    return dict(out)


def _mean_matrices_by_condition(
    sum_by_condition: Dict[str, np.ndarray],
    count_by_condition: Dict[str, np.ndarray],
    include_all_tasks: bool = True,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for cond, sum_mat in sum_by_condition.items():
        cnt = count_by_condition.get(cond)
        if cnt is None:
            continue
        mean_mat = _mean_matrix_from_sum_count(sum_mat, cnt)
        if mean_mat.size:
            out[cond] = mean_mat

    if include_all_tasks and out:
        sum_total = None
        cnt_total = None
        for cond in out.keys():
            s = np.asarray(sum_by_condition[cond], dtype=float)
            c = np.asarray(count_by_condition[cond], dtype=float)
            if sum_total is None:
                sum_total = s
                cnt_total = c
                continue

            n_rows = max(sum_total.shape[0], s.shape[0])
            n_cols = max(sum_total.shape[1], s.shape[1])

            def _pad(a: np.ndarray) -> np.ndarray:
                if a.shape == (n_rows, n_cols):
                    return a
                return np.pad(a, ((0, n_rows - a.shape[0]), (0, n_cols - a.shape[1])), mode="constant", constant_values=0.0)

            sum_total = _pad(sum_total) + _pad(s)
            cnt_total = _pad(cnt_total) + _pad(c)

        if sum_total is not None and cnt_total is not None:
            all_mean = _mean_matrix_from_sum_count(sum_total, cnt_total)
            if all_mean.size:
                out = {"all tasks": all_mean, **out}
    return out


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


def plot_violin_with_subject_jitter(
    values_by_group: Dict[str, List[float]],
    points_df: pd.DataFrame,
    point_col: str,
    title: str,
    y_title: str,
) -> Optional[go.Figure]:
    labels = [k for k in values_by_group if _finite_array(values_by_group[k]).size > 0]
    if not labels:
        return None

    xpos = {label: float(i) for i, label in enumerate(labels)}
    fig = go.Figure()
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
    ]
    for idx, label in enumerate(labels):
        vals = _finite_array(values_by_group[label])
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Violin(
                x=np.full(vals.size, xpos[label], dtype=float),
                y=vals,
                name=f"{label} (n={vals.size})",
                box_visible=True,
                meanline_visible=False,
                points=False,
                line={"width": 1.0, "color": color},
                opacity=0.45,
                showlegend=False,
            )
        )
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
                        line={"width": 1.5, "color": color},
                        opacity=0.9,
                        showlegend=False,
                        hovertemplate=f"{label}<br>{y_title}=%{{y:.3g}}<br>density=%{{customdata:.3g}}<extra></extra>",
                        customdata=kde_y,
                    )
                )

    if points_df.empty or point_col not in points_df.columns:
        points_data = pd.DataFrame()
    else:
        tmp = points_df.loc[np.isfinite(points_df[point_col])].copy()
        recs = []
        for _, row in tmp.iterrows():
            label = str(row.get("condition_label", "all recordings"))
            recs.append((label, float(row[point_col]), str(row.get("subject", "n/a")), str(row.get("hover_entities", ""))))
            recs.append(("all tasks", float(row[point_col]), str(row.get("subject", "n/a")), str(row.get("hover_entities", ""))))
        points_data = pd.DataFrame(recs, columns=["label", "value", "subject", "hover"])
        points_data = points_data.loc[points_data["label"].isin(labels)]

    if not points_data.empty:
        keep = _downsample_indices(len(points_data), min(MAX_POINTS_SCATTER, len(points_data)))
        points_data = points_data.iloc[keep].copy()
        subj_codes = pd.Categorical(points_data["subject"]).codes.astype(float)
        rng = np.random.default_rng(0)
        x_numeric = np.asarray([xpos[l] for l in points_data["label"]], dtype=float)
        x_numeric = x_numeric + rng.uniform(-0.17, 0.17, size=x_numeric.size)
        fig.add_trace(
            go.Scattergl(
                x=x_numeric,
                y=points_data["value"],
                mode="markers",
                marker={
                    "size": 6,
                    "color": subj_codes,
                    "colorscale": "Turbo",
                    "opacity": 0.7,
                    "line": {"width": 0.35, "color": "rgba(20,20,20,0.5)"},
                    "showscale": False,
                },
                customdata=np.stack([points_data["hover"]], axis=-1),
                hovertemplate="%{customdata[0]}<br>value=%{y:.3g}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task / condition",
        yaxis_title=y_title,
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[xpos[l] for l in labels],
        ticktext=labels,
        range=[-0.5, len(labels) - 0.1],
    )
    return fig


def plot_recording_metric_heatmap(
    df: pd.DataFrame,
    metric_specs: Sequence[Tuple[str, str]],
    title: str,
) -> Optional[go.Figure]:
    if df.empty:
        return None

    valid_specs = [(col, label) for col, label in metric_specs if col in df.columns]
    if not valid_specs:
        return None

    data = df.copy()
    z_cols = []
    raw_cols = []
    labels = []
    for col, label in valid_specs:
        vals = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        raw_cols.append(vals)
        z_cols.append(_robust_normalize_array(vals))
        labels.append(label)

    z = np.column_stack(z_cols)
    raw = np.column_stack(raw_cols)
    row_score = np.nanmean(z, axis=1)
    finite_row = np.any(np.isfinite(z), axis=1)
    if not np.any(finite_row):
        return None

    idx = np.where(finite_row)[0]
    tmp = data.iloc[idx].copy()
    tmp["__row_score__"] = row_score[idx]
    tmp["__orig_idx__"] = idx
    sort_cols = [c for c in ["subject", "session", "task", "condition", "run"] if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols + ["__row_score__"], ascending=[True] * len(sort_cols) + [False], kind="mergesort")
    else:
        tmp = tmp.sort_values("__row_score__", ascending=False, kind="mergesort")
    if len(tmp) > MAX_RECORDINGS_OVERVIEW:
        tmp = tmp.iloc[:MAX_RECORDINGS_OVERVIEW]

    order = tmp["__orig_idx__"].to_numpy(dtype=int)
    z = z[order, :]
    raw = raw[order, :]
    ordered_data = data.iloc[order].reset_index(drop=True)
    y = ordered_data["run_key"].fillna("n/a").astype(str).tolist()
    if len(y) != z.shape[0]:
        y = [f"recording_{idx}" for idx in range(z.shape[0])]
    show_ticklabels = len(y) <= 120

    bounds = _robust_bounds(z)
    zmin = bounds[0] if bounds is not None else None
    zmax = bounds[1] if bounds is not None else None

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=y,
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "Normalized value (robust z)"},
            customdata=np.stack([raw], axis=-1),
            hovertemplate="recording=%{y}<br>metric=%{x}<br>raw=%{customdata[0]:.3g}<br>normalized=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="QA metric summary",
        yaxis_title="Recordings (organized by subject / task / run)",
        template="plotly_white",
        margin={"l": 65, "r": 20, "t": 65, "b": 55},
        height=620,
    )
    fig.update_xaxes(tickangle=-25)
    fig.update_yaxes(showticklabels=show_ticklabels)
    return fig


def plot_condition_effect_grid(
    df: pd.DataFrame,
    metric_specs: Sequence[Tuple[str, str]],
    title: str,
) -> Optional[go.Figure]:
    if df.empty:
        return None
    conditions = sorted(df["condition_label"].unique())
    if len(conditions) < 2:
        return None

    valid_specs = [(col, label) for col, label in metric_specs if col in df.columns]
    if not valid_specs:
        return None

    n = len(valid_specs)
    cols = 3
    rows = int(np.ceil(n / cols))
    all_subjects = sorted(df["subject"].dropna().astype(str).unique().tolist())
    color_map = _subject_color_map(all_subjects)
    show_subject_legend = len(all_subjects) <= 10
    subplot_titles = [re.sub(r"\s*\([^)]*\)", "", label).strip() for _, label in valid_specs]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.20,
    )

    for idx, (metric_col, y_label) in enumerate(valid_specs):
        pivot = df.pivot_table(
            index="subject",
            columns="condition_label",
            values=metric_col,
            aggfunc="median",
        )
        if pivot.empty:
            continue
        pivot = pivot.reindex(columns=conditions)
        pivot = pivot.dropna(how="all")
        if pivot.empty:
            continue

        subject_var = pivot.var(axis=1, skipna=True).fillna(-np.inf)
        selected_subjects = subject_var.sort_values(ascending=False).index.tolist()
        if len(selected_subjects) > MAX_SUBJECT_LINES:
            selected_subjects = selected_subjects[:MAX_SUBJECT_LINES]

        row_i = idx // cols + 1
        col_i = idx % cols + 1
        for subject in selected_subjects:
            vals = pivot.loc[subject].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if np.sum(finite) < 2:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[conditions[i] for i in range(len(conditions)) if finite[i]],
                    y=vals[finite],
                    mode="lines+markers",
                    line={"width": 1.0, "color": color_map.get(str(subject), "#4F6F84")},
                    marker={"size": 4.5, "color": color_map.get(str(subject), "#4F6F84")},
                    name=f"sub-{subject}",
                    legendgroup=f"sub-{subject}",
                    showlegend=(show_subject_legend and idx == 0),
                    hovertemplate=f"sub={subject}<br>condition=%{{x}}<br>value=%{{y:.3g}}<extra></extra>",
                ),
                row=row_i,
                col=col_i,
            )

        median_vals = np.nanmedian(pivot.to_numpy(dtype=float), axis=0)
        fig.add_trace(
            go.Scatter(
                x=conditions,
                y=median_vals,
                mode="lines+markers",
                line={"width": 2.8, "color": "#0A2F51"},
                marker={"size": 6, "color": "#0A2F51"},
                name="Median subject profile",
                legendgroup="median-profile",
                showlegend=(idx == 0),
                hovertemplate="condition=%{x}<br>median=%{y:.3g}<extra></extra>",
            ),
            row=row_i,
            col=col_i,
        )
        if row_i == rows:
            fig.update_xaxes(title_text="Task / condition", row=row_i, col=col_i)
        else:
            fig.update_xaxes(title_text="", row=row_i, col=col_i)

        short_y = y_label
        if col_i > 1:
            short_y = ""
        fig.update_yaxes(title_text=short_y, row=row_i, col=col_i)

    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        margin={"l": 54, "r": 20, "t": 126, "b": 230},
        height=max(640, 360 * rows),
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.28,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11},
            "tracegroupgap": 6,
        },
    )
    fig.update_annotations(font={"size": 13})
    return fig


def plot_subject_epoch_small_multiples(
    profiles_by_subject: Dict[str, List[Dict[str, np.ndarray]]],
    subject_scores: Dict[str, float],
    title: str,
    y_label: str,
    top_n: int = 12,
) -> Optional[go.Figure]:
    if not profiles_by_subject:
        return None

    subjects = [s for s, profiles in profiles_by_subject.items() if profiles]
    if not subjects:
        return None
    subjects = sorted(
        subjects,
        key=lambda s: subject_scores.get(s, float("-inf")),
        reverse=True,
    )
    selected = subjects[:top_n]
    n = len(selected)
    if n == 0:
        return None
    cols = 3
    rows = int(np.ceil(n / cols))
    subplot_titles = [
        f"sub-{sub} (n={len(profiles_by_subject[sub])})"
        for sub in selected
    ]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, horizontal_spacing=0.08, vertical_spacing=0.1)

    for idx, subject in enumerate(selected):
        profile = _aggregate_window_profiles(profiles_by_subject[subject])
        if profile is None:
            continue
        r = idx // cols + 1
        c = idx % cols + 1

        x = np.asarray(profile["x"], dtype=float)
        q05 = np.asarray(profile["q05"], dtype=float)
        q25 = np.asarray(profile["q25"], dtype=float)
        q50 = np.asarray(profile["q50"], dtype=float)
        q75 = np.asarray(profile["q75"], dtype=float)
        q95 = np.asarray(profile["q95"], dtype=float)
        keep = _downsample_indices(x.size, MAX_POINTS_PROFILE)
        x = x[keep]
        q05 = q05[keep]
        q25 = q25[keep]
        q50 = q50[keep]
        q75 = q75[keep]
        q95 = q95[keep]

        fig.add_trace(go.Scatter(x=x, y=q95, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"}, showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=q05, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"}, fill="tonexty", fillcolor="rgba(88,166,255,0.16)", showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=q75, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"}, showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=q25, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"}, fill="tonexty", fillcolor="rgba(30,136,229,0.22)", showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=q50, mode="lines", line={"width": 1.7, "color": "#164B84"}, showlegend=False), row=r, col=c)

        fig.update_xaxes(title_text="Epoch index", row=r, col=c)
        fig.update_yaxes(title_text=y_label, row=r, col=c)

    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 80, "b": 40},
        height=max(420, 260 * rows),
    )
    return fig


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    flat = _robust_normalize_array(arr.reshape(-1))
    return flat.reshape(arr.shape)


def _concat_topomap_payload(
    existing: Optional[TopomapPayload],
    incoming: TopomapPayload,
    name_prefix: str = "",
) -> TopomapPayload:
    in_names = [f"{name_prefix}{n}" for n in incoming.layout.names]
    if existing is None:
        return TopomapPayload(
            layout=SensorLayout(
                x=np.asarray(incoming.layout.x, dtype=float),
                y=np.asarray(incoming.layout.y, dtype=float),
                names=in_names,
            ),
            values=np.asarray(incoming.values, dtype=float),
        )
    return TopomapPayload(
        layout=SensorLayout(
            x=np.concatenate([np.asarray(existing.layout.x, dtype=float), np.asarray(incoming.layout.x, dtype=float)]),
            y=np.concatenate([np.asarray(existing.layout.y, dtype=float), np.asarray(incoming.layout.y, dtype=float)]),
            names=list(existing.layout.names) + in_names,
        ),
        values=np.concatenate([np.asarray(existing.values, dtype=float), np.asarray(incoming.values, dtype=float)]),
    )


def _vstack_nan_padded(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack 2D matrices by rows, padding shorter epoch dimension with NaN."""
    a = np.atleast_2d(np.asarray(left, dtype=float))
    b = np.atleast_2d(np.asarray(right, dtype=float))

    if a.size == 0:
        return b
    if b.size == 0:
        return a

    n_cols = max(a.shape[1], b.shape[1])
    if a.shape[1] < n_cols:
        a = np.pad(a, ((0, 0), (0, n_cols - a.shape[1])), mode="constant", constant_values=np.nan)
    if b.shape[1] < n_cols:
        b = np.pad(b, ((0, 0), (0, n_cols - b.shape[1])), mode="constant", constant_values=np.nan)

    return np.vstack([a, b])


def _combine_accumulators(acc_by_type: Dict[str, ChTypeAccumulator]) -> ChTypeAccumulator:
    combined = ChTypeAccumulator()
    for ch_type in CH_TYPES:
        if ch_type not in acc_by_type:
            continue
        acc = acc_by_type[ch_type]
        combined.subjects.update(acc.subjects)
        combined.module_present.update(acc.module_present)
        combined.module_missing.update(acc.module_missing)
        combined.source_paths.update(acc.source_paths)

        for cond, vals in acc.std_dist_by_condition.items():
            combined.std_dist_by_condition[cond].extend(_finite_array(vals).tolist())
        for cond, vals in acc.ptp_dist_by_condition.items():
            combined.ptp_dist_by_condition[cond].extend(_finite_array(vals).tolist())

        for profile in acc.std_window_profiles:
            combined.std_window_profiles.append(profile)
        for profile in acc.ptp_window_profiles:
            combined.ptp_window_profiles.append(profile)
        for cond, profiles in acc.std_window_profiles_by_condition.items():
            for profile in profiles:
                combined.std_window_profiles_by_condition[cond].append(profile)
        for cond, profiles in acc.ptp_window_profiles_by_condition.items():
            for profile in profiles:
                combined.ptp_window_profiles_by_condition[cond].append(profile)

        for cond, matrix in acc.std_heatmap_by_condition.items():
            if cond in combined.std_heatmap_by_condition:
                combined.std_heatmap_by_condition[cond] = _vstack_nan_padded(
                    combined.std_heatmap_by_condition[cond],
                    matrix,
                )
            else:
                combined.std_heatmap_by_condition[cond] = np.asarray(matrix, dtype=float)
        for cond, matrix in acc.std_heatmap_upper_by_condition.items():
            if cond in combined.std_heatmap_upper_by_condition:
                combined.std_heatmap_upper_by_condition[cond] = _vstack_nan_padded(
                    combined.std_heatmap_upper_by_condition[cond],
                    matrix,
                )
            else:
                combined.std_heatmap_upper_by_condition[cond] = np.asarray(matrix, dtype=float)
        for cond, sum_matrix in acc.std_heatmap_sum_by_condition.items():
            if cond in combined.std_heatmap_sum_by_condition:
                combined.std_heatmap_sum_by_condition[cond] = _vstack_nan_padded(
                    combined.std_heatmap_sum_by_condition[cond],
                    sum_matrix,
                )
                combined.std_heatmap_count_by_condition[cond] = _vstack_nan_padded(
                    combined.std_heatmap_count_by_condition[cond],
                    acc.std_heatmap_count_by_condition.get(cond, np.zeros_like(sum_matrix, dtype=float)),
                )
            else:
                combined.std_heatmap_sum_by_condition[cond] = np.asarray(sum_matrix, dtype=float)
                combined.std_heatmap_count_by_condition[cond] = np.asarray(
                    acc.std_heatmap_count_by_condition.get(cond, np.zeros_like(sum_matrix, dtype=float)),
                    dtype=float,
                )
        for cond, matrix in acc.ptp_heatmap_by_condition.items():
            if cond in combined.ptp_heatmap_by_condition:
                combined.ptp_heatmap_by_condition[cond] = _vstack_nan_padded(
                    combined.ptp_heatmap_by_condition[cond],
                    matrix,
                )
            else:
                combined.ptp_heatmap_by_condition[cond] = np.asarray(matrix, dtype=float)
        for cond, matrix in acc.ptp_heatmap_upper_by_condition.items():
            if cond in combined.ptp_heatmap_upper_by_condition:
                combined.ptp_heatmap_upper_by_condition[cond] = _vstack_nan_padded(
                    combined.ptp_heatmap_upper_by_condition[cond],
                    matrix,
                )
            else:
                combined.ptp_heatmap_upper_by_condition[cond] = np.asarray(matrix, dtype=float)
        for cond, sum_matrix in acc.ptp_heatmap_sum_by_condition.items():
            if cond in combined.ptp_heatmap_sum_by_condition:
                combined.ptp_heatmap_sum_by_condition[cond] = _vstack_nan_padded(
                    combined.ptp_heatmap_sum_by_condition[cond],
                    sum_matrix,
                )
                combined.ptp_heatmap_count_by_condition[cond] = _vstack_nan_padded(
                    combined.ptp_heatmap_count_by_condition[cond],
                    acc.ptp_heatmap_count_by_condition.get(cond, np.zeros_like(sum_matrix, dtype=float)),
                )
            else:
                combined.ptp_heatmap_sum_by_condition[cond] = np.asarray(sum_matrix, dtype=float)
                combined.ptp_heatmap_count_by_condition[cond] = np.asarray(
                    acc.ptp_heatmap_count_by_condition.get(cond, np.zeros_like(sum_matrix, dtype=float)),
                    dtype=float,
                )

        for cond, vals in acc.psd_ratio_by_condition.items():
            combined.psd_ratio_by_condition[cond].extend(vals)
        combined.psd_profiles.extend(acc.psd_profiles)
        for cond, profiles in acc.psd_profiles_by_condition.items():
            combined.psd_profiles_by_condition[cond].extend(profiles)
        for cond, vals in acc.ecg_corr_by_condition.items():
            combined.ecg_corr_by_condition[cond].extend(vals)
        for cond, vals in acc.eog_corr_by_condition.items():
            combined.eog_corr_by_condition[cond].extend(vals)
        for cond, vals in acc.muscle_scalar_by_condition.items():
            combined.muscle_scalar_by_condition[cond].extend(vals)
        combined.muscle_profiles.extend(acc.muscle_profiles)
        for cond, profiles in acc.muscle_profiles_by_condition.items():
            combined.muscle_profiles_by_condition[cond].extend(profiles)

        for subject, profiles in acc.std_subject_profiles.items():
            for profile in profiles:
                combined.std_subject_profiles[subject].append(profile)
        for subject, profiles in acc.ptp_subject_profiles.items():
            for profile in profiles:
                combined.ptp_subject_profiles[subject].append(profile)

        topomap_pairs = [
            (acc.std_topomap_by_condition, combined.std_topomap_by_condition),
            (acc.ptp_topomap_by_condition, combined.ptp_topomap_by_condition),
            (acc.psd_topomap_by_condition, combined.psd_topomap_by_condition),
            (acc.ecg_topomap_by_condition, combined.ecg_topomap_by_condition),
            (acc.eog_topomap_by_condition, combined.eog_topomap_by_condition),
        ]
        for src, dst in topomap_pairs:
            for cond, payload in src.items():
                dst[cond] = _concat_topomap_payload(dst.get(cond), payload, name_prefix=f"{ch_type}_")

    rows = []
    for ch_type in CH_TYPES:
        rows.extend(acc_by_type[ch_type].run_rows)
    df = _run_rows_dataframe(rows)
    if not df.empty:
        combined.run_rows.extend(rows)
        run_level = df.drop_duplicates(subset=["run_key", "condition_label"])
        combined.run_count = int(run_level["run_key"].nunique())
        cond_counts = run_level["condition_label"].value_counts().to_dict()
        for cond, count in cond_counts.items():
            combined.runs_by_condition[str(cond)] += int(count)
    else:
        combined.run_count = 0

    return combined


def _build_global_qa_section(acc: ChTypeAccumulator, tab_name: str, amplitude_unit: str, is_combined: bool) -> str:
    suffix = " (all channels)" if is_combined else ""

    std_violin = plot_violin_channel_distribution(
        acc.std_dist_by_condition,
        f"STD channel distribution by condition{suffix}",
        f"STD summary ({amplitude_unit})",
    )
    std_profile = plot_quantile_band_timecourse(
        _aggregate_window_profiles(acc.std_window_profiles),
        f"STD epoch profile (channel quantile bands){suffix}",
        "Global pooled epoch index",
        f"STD ({amplitude_unit})",
    )
    std_heatmaps = {
        cond: plot_heatmap_sorted_channels_windows(
            matrix,
            title=f"STD channel-by-epoch footprint ({cond}){suffix}",
            color_title=f"STD ({amplitude_unit})",
            summary_mode="median",
        )
        for cond, matrix in sorted(acc.std_heatmap_by_condition.items())
    }
    std_topomaps = _topomap_blocks(
        payloads_by_condition=acc.std_topomap_by_condition,
        title_prefix=f"STD spatial footprint (median over epochs){suffix}",
        color_title=f"STD ({amplitude_unit})",
        interpretation=(
            f"Axes: spatial coordinates from stored channel geometry; color in {amplitude_unit}. "
            "Each marker is one channel summary aggregated across recordings in this condition (Global QA). "
            "Typical appearance: smooth spatial gradients. Suspicious appearance: localized hotspots. "
            "Limitation: Global QA does not identify subjects."
        ),
    )

    ptp_violin = plot_violin_channel_distribution(
        acc.ptp_dist_by_condition,
        f"PtP channel distribution by condition{suffix}",
        f"PtP summary ({amplitude_unit})",
    )
    ptp_profile = plot_quantile_band_timecourse(
        _aggregate_window_profiles(acc.ptp_window_profiles),
        f"PtP epoch profile (channel quantile bands){suffix}",
        "Global pooled epoch index",
        f"PtP amplitude ({amplitude_unit})",
    )
    ptp_heatmaps = {
        cond: plot_heatmap_sorted_channels_windows(
            matrix,
            title=f"PtP channel-by-epoch footprint ({cond}){suffix}",
            color_title=f"PtP ({amplitude_unit})",
            summary_mode="upper_tail",
        )
        for cond, matrix in sorted(acc.ptp_heatmap_by_condition.items())
    }
    ptp_topomaps = _topomap_blocks(
        payloads_by_condition=acc.ptp_topomap_by_condition,
        title_prefix=f"PtP spatial footprint (95th percentile over epochs){suffix}",
        color_title=f"PtP ({amplitude_unit})",
        interpretation=(
            f"Axes: spatial coordinates from stored channel geometry; color in {amplitude_unit}. "
            "Marker color shows per-channel upper-tail PtP amplitude aggregated across recordings in this condition (Global QA). "
            "Typical appearance: moderate spatial spread. Suspicious appearance: persistent hotspots. "
            "Limitation: Global QA does not identify subjects."
        ),
    )

    psd_profile = plot_psd_median_band(
        _aggregate_psd_profiles(acc.psd_profiles),
        "PSD median profile with channel quantile bands",
    )
    psd_violin = plot_violin_channel_distribution(
        acc.psd_ratio_by_condition,
        "Mains relative power distribution by condition",
        "Mains relative power (unitless)",
    )
    psd_topomaps = _topomap_blocks(
        payloads_by_condition=acc.psd_topomap_by_condition,
        title_prefix="Mains relative power spatial footprint",
        color_title="Mains ratio",
        interpretation=(
            "Axes: spatial coordinates from stored channel geometry; color is unitless relative power. "
            "Each marker is one channel aggregated across recordings in this condition (Global QA). "
            "Typical appearance: mild spatial variation. Suspicious appearance: strong localized hotspots. "
            "Limitation: Global QA does not identify subjects."
        ),
    )

    ecg_violin = plot_violin_channel_distribution(
        acc.ecg_corr_by_condition,
        "ECG correlation magnitude distribution by condition",
        "|r| (unitless)",
    )
    eog_violin = plot_violin_channel_distribution(
        acc.eog_corr_by_condition,
        "EOG correlation magnitude distribution by condition",
        "|r| (unitless)",
    )
    muscle_profile = plot_quantile_band_timecourse(
        _aggregate_muscle_profiles(acc.muscle_profiles),
        "Muscle score epoch profile",
        "Global pooled epoch index",
        "Muscle score (z-score, unitless)",
    )
    muscle_violin = plot_violin_channel_distribution(
        acc.muscle_scalar_by_condition,
        "Muscle upper-tail distribution by condition",
        "Muscle score (z-score, unitless)",
    )

    std_ecdf = _make_ecdf_figure(acc.std_dist_by_condition, f"STD ECDF{suffix}", f"STD ({amplitude_unit})")
    ptp_ecdf = _make_ecdf_figure(acc.ptp_dist_by_condition, f"PtP ECDF{suffix}", f"PtP ({amplitude_unit})")
    psd_ecdf = _make_ecdf_figure(acc.psd_ratio_by_condition, "Mains relative power ECDF", "Mains ratio")
    ecg_ecdf = _make_ecdf_figure(acc.ecg_corr_by_condition, "ECG |r| ECDF", "|r|")
    eog_ecdf = _make_ecdf_figure(acc.eog_corr_by_condition, "EOG |r| ECDF", "|r|")
    muscle_ecdf = _make_ecdf_figure(acc.muscle_scalar_by_condition, "Muscle score ECDF", "Muscle score")

    std_heatmap_blocks = _condition_figure_blocks(
        std_heatmaps,
        (
            f"Axes: x is epoch index, y is sorted channel index; color is STD in {amplitude_unit}. "
            "Each heatmap cell represents the STD of one channel in one epoch, aggregated across recordings. "
            "Typical appearance: moderate contrast without persistent vertical stripes. "
            "Suspicious appearance: repeated vertical stripes and heavy upper tails. "
            "Global pooled epoch index is not aligned in time across recordings."
        ),
    )
    ptp_heatmap_blocks = _condition_figure_blocks(
        ptp_heatmaps,
        (
            f"Axes: x is epoch index, y is sorted channel index; color is PtP amplitude in {amplitude_unit}. "
            "Heatmap cell = PtP value for one channel in one epoch, aggregated across recordings. "
            "Typical appearance: moderate spread. Suspicious appearance: strong upper-tail bursts and vertical stripes. "
            "Global pooled epoch index is not aligned in time across recordings."
        ),
    )

    return (
        "<section>"
        "<h2>Global QA</h2>"
        "<p>Aggregated distributions/footprints across all recordings in this dataset (all subjects x all runs).</p>"
        "<div class='metric-block'>"
        "<h3>STD</h3>"
        + _figure_block(
            std_violin,
            (
                f"Y-axis is STD in {amplitude_unit}; x-axis is task/condition. "
                "Each observation represents one channel summarized across epochs for one recording. "
                "Typical appearance: compact distribution with moderate upper tail. "
                "Suspicious appearance: heavy upper tail where few channels dominate. Global QA does not identify subjects."
            ),
        )
        + _figure_block(
            std_profile,
            (
                f"Y-axis is STD in {amplitude_unit}; x-axis is global pooled epoch index. "
                "This profile is computed per epoch by pooling channels across recordings (global footprint). "
                "Typical appearance: stable median with moderate band width. Suspicious appearance: persistent wide bands or repeated peaks. "
                "Global pooled epoch index is not aligned in time across recordings."
            ),
        )
        + std_heatmap_blocks
        + std_topomaps
        + "</div>"
        "<div class='metric-block'>"
        "<h3>PtP</h3>"
        + _figure_block(
            ptp_violin,
            (
                f"Y-axis is PtP amplitude in {amplitude_unit}; x-axis is task/condition. "
                "Each observation represents one channel summarized across epochs for one recording. "
                "Typical appearance: moderate spread. Suspicious appearance: heavy upper tail and condition-dependent broadening."
            ),
        )
        + _figure_block(
            ptp_profile,
            (
                f"Y-axis is PtP amplitude in {amplitude_unit}; x-axis is global pooled epoch index. "
                "Line = median across channels per epoch; bands = IQR and 5-95% across channels. "
                "Typical appearance: stable median and moderate band width. Suspicious appearance: bursty upper tails."
            ),
        )
        + ptp_heatmap_blocks
        + ptp_topomaps
        + "</div>"
        "<div class='metric-block'>"
        "<h3>PSD</h3>"
        + _figure_block(
            psd_profile,
            (
                "X-axis is frequency in Hz; y-axis is power spectral density (relative power). "
                "Each profile summarizes channels pooled across recordings (Global QA). "
                "Typical appearance: smooth broad-band shape. Suspicious appearance: strong mains peaks and harmonics."
            ),
        )
        + _figure_block(
            psd_violin,
            (
                "Y-axis is mains relative power (unitless). "
                "Each observation represents one channel summarized for one recording. "
                "Typical appearance: moderate distribution. Suspicious appearance: heavy upper tail with strong mains contribution."
            ),
        )
        + psd_topomaps
        + "</div>"
        "<div class='metric-block'>"
        "<h3>ECG / EOG</h3>"
        + _figure_block(
            ecg_violin,
            (
                "Y-axis is |r| (unitless). Each observation represents one channel summary for one recording. "
                "Typical appearance: moderate correlation magnitude distribution. Suspicious appearance: broad upper tails."
            ),
        )
        + _figure_block(
            eog_violin,
            (
                "Y-axis is |r| (unitless). Each observation represents one channel summary for one recording. "
                "Typical appearance: moderate correlation magnitude distribution. Suspicious appearance: pronounced upper tails and spatial hotspots."
            ),
        )
        + "</div>"
        "<div class='metric-block'>"
        "<h3>Muscle</h3>"
        + _figure_block(
            muscle_profile,
            (
                "Y-axis is muscle score (z-score, unitless); x-axis is global pooled epoch index. "
                "Line = median across channels per epoch; bands = IQR and 5-95% across channels. "
                "Typical appearance: stable profile. Suspicious appearance: sustained high upper tail and bursts."
            ),
        )
        + _figure_block(
            muscle_violin,
            (
                "Y-axis is muscle score (z-score, unitless). "
                "Each observation is one recording-level upper-tail summary. "
                "Typical appearance: compact distribution. Suspicious appearance: heavy upper tail."
            ),
        )
        + "</div>"
        "<div class='metric-block'>"
        "<h3>QA Distributions (Statistical)</h3>"
        + _figure_block(std_ecdf, "ECDF of global channel summaries. Right-shifted curves indicate larger values.")
        + _figure_block(ptp_ecdf, "ECDF of global channel summaries. Right-shifted curves indicate larger values.")
        + _figure_block(psd_ecdf, "ECDF of global mains relative power by channel.")
        + _figure_block(ecg_ecdf, "ECDF of global ECG correlation magnitude by channel.")
        + _figure_block(eog_ecdf, "ECDF of global EOG correlation magnitude by channel.")
        + _figure_block(muscle_ecdf, "ECDF of global recording-level muscle summaries.")
        + "</div>"
        "</section>"
    )


def _subject_scores_from_df(df: pd.DataFrame, metric_col: str) -> Dict[str, float]:
    if df.empty or metric_col not in df.columns:
        return {}
    out = {}
    for subject, group in df.groupby("subject"):
        vals = _finite_array(group[metric_col].to_numpy(dtype=float))
        out[str(subject)] = float(np.nanmedian(vals)) if vals.size else float("nan")
    return out


def _build_subject_qa_section(acc: ChTypeAccumulator, tab_name: str, amplitude_unit: str) -> str:
    df = _run_rows_dataframe(acc.run_rows)
    if df.empty:
        return "<section><h2>Subject QA</h2><p>No recording-level summaries are available.</p></section>"

    std_violin = plot_subject_condition_violin(
        df, "std_median",
        "STD by condition with subject-aware points",
        f"STD ({amplitude_unit})",
    )
    ptp_violin = plot_subject_condition_violin(
        df, "ptp_upper_tail",
        "PtP by condition with subject-aware points",
        f"PtP ({amplitude_unit})",
    )
    ecg_violin = plot_subject_condition_violin(
        df, "ecg_mean_abs_corr",
        "ECG correlation magnitude by condition with subject-aware points",
        "|r| (unitless)",
    )
    eog_violin = plot_subject_condition_violin(
        df, "eog_mean_abs_corr",
        "EOG correlation magnitude by condition with subject-aware points",
        "|r| (unitless)",
    )
    mains_violin = plot_subject_condition_violin(
        df, "mains_ratio",
        "Mains relative power by condition with subject-aware points",
        "Mains ratio (unitless)",
    )

    std_scatter = plot_run_fingerprint_scatter(
        df,
        "std_median",
        "std_upper_tail",
        "Run fingerprint scatter (STD)",
        f"Median channel STD per recording ({amplitude_unit})",
        f"Upper-tail channel STD per recording ({amplitude_unit})",
    )
    ptp_scatter = plot_run_fingerprint_scatter(
        df,
        "ptp_median",
        "ptp_upper_tail",
        "Run fingerprint scatter (PtP)",
        f"Median channel PtP per recording ({amplitude_unit})",
        f"Upper-tail channel PtP per recording ({amplitude_unit})",
    )
    mains_scatter = plot_run_fingerprint_scatter(
        df,
        "mains_ratio",
        "mains_harmonics_ratio",
        "Run fingerprint scatter (PSD mains)",
        "Mains relative power (unitless)",
        "Harmonics relative power (unitless)",
    )
    ecg_scatter = plot_run_fingerprint_scatter(
        df,
        "ecg_mean_abs_corr",
        "ecg_p95_abs_corr",
        "Run fingerprint scatter (ECG)",
        "Mean |ECG correlation| per recording",
        "p95 |ECG correlation| per recording",
    )
    eog_scatter = plot_run_fingerprint_scatter(
        df,
        "eog_mean_abs_corr",
        "eog_p95_abs_corr",
        "Run fingerprint scatter (EOG)",
        "Mean |EOG correlation| per recording",
        "p95 |EOG correlation| per recording",
    )
    muscle_scatter = plot_run_fingerprint_scatter(
        df,
        "muscle_median",
        "muscle_p95",
        "Run fingerprint scatter (Muscle)",
        "Median muscle score per recording",
        "p95 muscle score per recording",
    )

    ranking = plot_subject_ranking_table(df, "Subject ranking by robust QA summaries")
    metric_heatmap = plot_subject_metric_heatmap(df, "Subject-by-metric normalized summary heatmap")
    condition_effect = plot_subject_condition_effect(
        df,
        metric_col="std_median",
        title="Condition effect per subject (STD median)",
        y_label=f"STD ({amplitude_unit})",
    )

    std_subject_profiles = plot_subject_epoch_small_multiples(
        acc.std_subject_profiles,
        _subject_scores_from_df(df, "std_upper_tail"),
        "Per-subject STD epoch profile small multiples (top 12 by upper-tail summary)",
        y_label=f"STD ({amplitude_unit})",
        top_n=12,
    )
    ptp_subject_profiles = plot_subject_epoch_small_multiples(
        acc.ptp_subject_profiles,
        _subject_scores_from_df(df, "ptp_upper_tail"),
        "Per-subject PtP epoch profile small multiples (top 12 by upper-tail summary)",
        y_label=f"PtP ({amplitude_unit})",
        top_n=12,
    )

    return (
        "<section>"
        "<h2>Subject QA</h2>"
        "<p>Subject-aware summaries highlight recordings/subjects with atypical distributions without applying thresholds.</p>"
        "<div class='metric-block'>"
        "<h3>Subject-aware distributions</h3>"
        + _figure_block(
            std_violin,
            (
                f"Y-axis is STD in {amplitude_unit}; x-axis is task/condition. "
                "Each point represents one recording summary and hover includes BIDS entities. "
                "Typical appearance: moderate spread per condition. Suspicious appearance: heavy upper tails or subject clusters away from peers."
            ),
        )
        + _figure_block(
            ptp_violin,
            (
                f"Y-axis is PtP in {amplitude_unit}; x-axis is task/condition. "
                "Each point represents one recording summary and is colored by subject. "
                "Typical appearance: compact subject overlap. Suspicious appearance: repeated subject-specific upper tails."
            ),
        )
        + _figure_block(ecg_violin, "Y-axis is |r| (unitless); each point is one recording summary. Look for consistent subject-level shifts.")
        + _figure_block(eog_violin, "Y-axis is |r| (unitless); each point is one recording summary. Look for condition-specific and subject-specific shifts.")
        + _figure_block(mains_violin, "Y-axis is mains ratio (unitless); each point is one recording summary. Heavy upper tails indicate stronger mains burden.")
        + "</div>"
        "<div class='metric-block'>"
        "<h3>Run fingerprint scatter</h3>"
        + _figure_block(
            std_scatter,
            (
                "Each point = one recording (sub x task x run), colored by subject and symbolized by condition. "
                "Axes represent central and upper-tail STD summaries. "
                "Typical appearance: compact cloud with moderate spread. Suspicious appearance: isolated points with strong upper-tail displacement."
            ),
        )
        + _figure_block(ptp_scatter, "Each point = one recording. Axes represent central and upper-tail PtP summaries.")
        + _figure_block(mains_scatter, "Each point = one recording. Axes summarize mains and harmonics relative power.")
        + _figure_block(ecg_scatter, "Each point = one recording. Axes summarize mean and upper-tail ECG correlation magnitude.")
        + _figure_block(eog_scatter, "Each point = one recording. Axes summarize mean and upper-tail EOG correlation magnitude.")
        + _figure_block(muscle_scatter, "Each point = one recording. Axes summarize median and upper-tail muscle scores.")
        + "</div>"
        "<div class='metric-block'>"
        "<h3>Subject ranking and footprints</h3>"
        + _figure_block(
            ranking,
            (
                "Table values are robust per-subject summaries with recording counts. "
                "This table ranks distribution footprints only and does not imply binary decisions."
            ),
        )
        + _figure_block(
            std_subject_profiles,
            (
                f"Each panel shows one subject: median and quantile bands across channels per epoch (STD in {amplitude_unit}). "
                "Panels are limited to top 12 subjects by upper-tail summary for scalability."
            ),
        )
        + _figure_block(
            ptp_subject_profiles,
            (
                f"Each panel shows one subject: median and quantile bands across channels per epoch (PtP in {amplitude_unit}). "
                "Panels are limited to top 12 subjects by upper-tail summary for scalability."
            ),
        )
        + _figure_block(metric_heatmap, "Rows are subjects and columns are metrics. Values are normalized summaries (z-score) for quick multi-metric comparison.")
        + _figure_block(condition_effect, "Paired condition lines per subject help visualize task-related shifts in summary metrics.")
        + "</div>"
        "</section>"
    )


def _build_cohort_overview_section(acc: ChTypeAccumulator, amplitude_unit: str, is_combined: bool) -> str:
    df = _run_rows_dataframe(acc.run_rows)
    if df.empty:
        return "<section><h2>Cohort Overview</h2><p>No recording-level summaries are available.</p></section>"

    suffix = " (all channels)" if is_combined else ""
    amp_label = "all channels" if is_combined else amplitude_unit
    metric_specs = [
        ("std_upper_tail", f"STD upper tail ({amp_label})"),
        ("ptp_upper_tail", f"PtP upper tail ({amp_label})"),
        ("mains_ratio", "Mains relative power"),
        ("ecg_p95_abs_corr", "ECG |r| upper tail"),
        ("eog_p95_abs_corr", "EOG |r| upper tail"),
        ("muscle_p95", "Muscle upper tail"),
    ]
    overview_heatmap = plot_recording_metric_heatmap(
        df,
        metric_specs=metric_specs,
        title=f"Recording-by-metric cohort overview{suffix}",
    )
    ranking = plot_subject_ranking_table(df, "Subject ranking by robust summary footprint")

    return (
        "<section>"
        "<h2>Cohort Overview</h2>"
        "<p>Rows are recordings and columns are metric summaries. This view answers who shows elevated burden patterns across the cohort.</p>"
        + _figure_block(
            overview_heatmap,
            (
                "Each row is one recording (subject x task x run) and each column is a robust metric summary. "
                "Cell color encodes normalized magnitude for cross-metric readability, while hover keeps raw values in native units. "
                "Use this as the first triage step to select recordings for deeper epoch/channel inspection."
            ),
        )
        + _figure_block(
            ranking,
            (
                "Rows are subjects ranked by robust aggregated summaries and recording count. "
                "Higher rank indicates larger relative burden across one or more metrics. "
                "This ranking supports prioritization and does not define decisions."
            ),
        )
        + "</section>"
    )


def _build_condition_effect_section(acc: ChTypeAccumulator, amplitude_unit: str, is_combined: bool) -> str:
    df = _run_rows_dataframe(acc.run_rows)
    if df.empty:
        return "<section><h2>Task/Condition Effects</h2><p>No condition comparison is available.</p></section>"

    suffix = " (all channels)" if is_combined else ""
    std_y = "STD median (all channels)" if is_combined else f"STD median ({amplitude_unit})"
    ptp_y = "PtP upper tail (all channels)" if is_combined else f"PtP upper tail ({amplitude_unit})"
    metric_specs = [
        ("std_median", std_y),
        ("ptp_upper_tail", ptp_y),
        ("mains_ratio", "Mains relative power"),
        ("ecg_mean_abs_corr", "ECG |r| mean"),
        ("eog_mean_abs_corr", "EOG |r| mean"),
        ("muscle_median", "Muscle median"),
    ]
    cond_effect = plot_condition_effect_grid(
        df,
        metric_specs=metric_specs,
        title=f"Within-subject task/condition profiles{suffix}",
    )

    return (
        "<section>"
        "<h2>Task/Condition Effects</h2>"
        "<p>Each thin line is one subject profile across conditions. The dark line is the cohort median profile. Subject identity is always available in hover text.</p>"
        + _figure_block(
            cond_effect,
            (
                "Panels summarize within-subject condition effects for each metric. "
                "Thin lines indicate subject trajectories; the dark line indicates median trajectory across subjects. "
                "Consistent separation between conditions suggests task-related shifts in distribution burden."
            ),
        )
        + "</section>"
    )


def _build_metric_details_section(acc: ChTypeAccumulator, amplitude_unit: str, is_combined: bool) -> str:
    suffix = " (all channels)" if is_combined else ""
    df = _run_rows_dataframe(acc.run_rows)
    tab_token = "combined" if is_combined else ("mag" if amplitude_unit.startswith("Tesla (T)") else "grad")

    def _variant_tabs(
        group_id: str,
        values_map: Dict[str, List[float]],
        point_col: str,
        title_prefix: str,
        value_label: str,
    ) -> str:
        vals = _values_with_task_agnostic(values_map)
        violin = plot_violin_with_subject_jitter(
            vals,
            df,
            point_col=point_col,
            title=f"{title_prefix} - violin",
            y_title=value_label,
        )
        hist = plot_histogram_distribution(
            vals,
            title=f"{title_prefix} - histogram",
            x_title=value_label,
        )
        dens = plot_density_distribution(
            vals,
            title=f"{title_prefix} - density",
            x_title=value_label,
        )
        return _build_subtabs_html(
            group_id,
            [
                ("Violin", _figure_block(violin, f"Distribution by task/condition (plus all-tasks aggregate) with subject jitter points. Units: {value_label}.", normalized_variant=True, norm_mode="y")),
                ("Histogram", _figure_block(hist, f"Histogram variant for the same distribution with overlaid density curves. Units: {value_label}.", normalized_variant=True, norm_mode="x")),
                ("Density", _figure_block(dens, f"Density-curve variant for the same distribution. Units: {value_label}.", normalized_variant=True, norm_mode="x")),
            ],
            level=4,
        )

    def _metric_panel(
        metric_name: str,
        rec_values: Dict[str, List[float]],
        ch_values: Dict[str, List[float]],
        epoch_values: Dict[str, List[float]],
        point_col: str,
        unit_label: str,
        mean_sum_map: Optional[Dict[str, np.ndarray]] = None,
        mean_count_map: Optional[Dict[str, np.ndarray]] = None,
        topomap_payloads: Optional[Dict[str, TopomapPayload]] = None,
    ) -> str:
        dist_tabs = _build_subtabs_html(
            f"dist-{metric_name.lower()}-{tab_token}",
            [
                ("A: Recording distributions", _variant_tabs(f"var-{metric_name.lower()}-a-{tab_token}", rec_values, point_col, f"{metric_name} recording distribution{suffix}", unit_label)),
                ("B: Epochs per channel", _variant_tabs(f"var-{metric_name.lower()}-b-{tab_token}", ch_values, point_col, f"{metric_name} epochs-per-channel distribution{suffix}", unit_label)),
                ("C: Channels per epoch", _variant_tabs(f"var-{metric_name.lower()}-c-{tab_token}", epoch_values, point_col, f"{metric_name} channels-per-epoch distribution{suffix}", unit_label)),
            ],
            level=3,
        )

        blocks = [f"<div class='metric-block'><h3>{metric_name}</h3>{dist_tabs}</div>"]

        if mean_sum_map is not None and mean_count_map is not None:
            mean_mats = _mean_matrices_by_condition(mean_sum_map, mean_count_map, include_all_tasks=True)
            heatmaps = {
                cond: plot_heatmap_sorted_channels_windows(
                    matrix,
                    title=f"{metric_name} mean channel-epoch map across subjects ({cond}){suffix}",
                    color_title=unit_label,
                    summary_mode="median",
                )
                for cond, matrix in sorted(mean_mats.items())
            }
            blocks.append(
                _condition_figure_blocks(
                    heatmaps,
                    (
                        "Heatmap cell = mean metric value across subjects for one channel and one epoch. "
                        "The top strip in the same figure shows median and quantile bands across channels per epoch, "
                        "so the envelope and heatmap share the same epoch axis."
                    ),
                    normalized_variant=True,
                    norm_mode="z",
                )
            )

        if topomap_payloads is not None:
            blocks.append(
                _topomap_blocks(
                    payloads_by_condition=topomap_payloads,
                    title_prefix=f"{metric_name} topographic footprint{suffix}",
                    color_title=unit_label,
                    interpretation=(
                        "Each point is a channel-level summary value. "
                        "For Elekta systems, overlapping triplets are visually spread so 1 magnetometer and 2 gradiometer values can be inspected."
                    ),
                    normalized_variant=True,
                )
            )

        return "".join(blocks)

    std_panel = _metric_panel(
        metric_name="STD",
        rec_values=_run_values_by_condition(df, "std_median"),
        ch_values=acc.std_dist_by_condition,
        epoch_values=_epoch_values_from_profiles(acc.std_window_profiles_by_condition),
        point_col="std_median",
        unit_label=f"STD ({amplitude_unit})",
        mean_sum_map=acc.std_heatmap_sum_by_condition,
        mean_count_map=acc.std_heatmap_count_by_condition,
        topomap_payloads=acc.std_topomap_by_condition,
    )
    ptp_panel = _metric_panel(
        metric_name="PtP",
        rec_values=_run_values_by_condition(df, "ptp_upper_tail"),
        ch_values=acc.ptp_dist_by_condition,
        epoch_values=_epoch_values_from_profiles(acc.ptp_window_profiles_by_condition),
        point_col="ptp_upper_tail",
        unit_label=f"PtP ({amplitude_unit})",
        mean_sum_map=acc.ptp_heatmap_sum_by_condition,
        mean_count_map=acc.ptp_heatmap_count_by_condition,
        topomap_payloads=acc.ptp_topomap_by_condition,
    )
    psd_panel = _metric_panel(
        metric_name="PSD mains ratio",
        rec_values=_run_values_by_condition(df, "mains_ratio"),
        ch_values=acc.psd_ratio_by_condition,
        epoch_values={},
        point_col="mains_ratio",
        unit_label="Mains relative power (unitless)",
        topomap_payloads=acc.psd_topomap_by_condition,
    ) + _condition_figure_blocks(
        {
            cond: plot_psd_median_band(_aggregate_psd_profiles(profiles), title=f"PSD profile ({cond})")
            for cond, profiles in sorted(acc.psd_profiles_by_condition.items())
        },
        "Line and quantile bands summarize channel PSD profiles by condition (frequency in Hz).",
        normalized_variant=True,
        norm_mode="y",
    )
    ecg_panel = _metric_panel(
        metric_name="ECG correlation",
        rec_values=_run_values_by_condition(df, "ecg_mean_abs_corr"),
        ch_values=acc.ecg_corr_by_condition,
        epoch_values={},
        point_col="ecg_mean_abs_corr",
        unit_label="|r| (unitless)",
        topomap_payloads=acc.ecg_topomap_by_condition,
    )
    eog_panel = _metric_panel(
        metric_name="EOG correlation",
        rec_values=_run_values_by_condition(df, "eog_mean_abs_corr"),
        ch_values=acc.eog_corr_by_condition,
        epoch_values={},
        point_col="eog_mean_abs_corr",
        unit_label="|r| (unitless)",
        topomap_payloads=acc.eog_topomap_by_condition,
    )
    muscle_panel = _metric_panel(
        metric_name="Muscle score",
        rec_values=_run_values_by_condition(df, "muscle_p95"),
        ch_values=acc.muscle_scalar_by_condition,
        epoch_values=_epoch_values_from_profiles(
            {k: [{"q50": np.asarray(v, dtype=float)} for v in vals] for k, vals in acc.muscle_profiles_by_condition.items()}
        ),
        point_col="muscle_p95",
        unit_label="Muscle score (z-score)",
    ) + _condition_figure_blocks(
        {
            cond: plot_quantile_band_timecourse(
                _aggregate_muscle_profiles(profiles),
                title=f"Muscle epoch envelope ({cond})",
                x_title="Epoch index",
                y_title="Muscle score (z-score)",
            )
            for cond, profiles in sorted(acc.muscle_profiles_by_condition.items())
        },
        "Muscle epoch envelope by condition (median and quantile bands).",
        normalized_variant=True,
        norm_mode="y",
    )

    metric_tabs = _build_subtabs_html(
        f"metric-main-{tab_token}",
        [
            ("STD", std_panel),
            ("PtP", ptp_panel),
            ("PSD", psd_panel),
            ("ECG", ecg_panel),
            ("EOG", eog_panel),
            ("Muscle", muscle_panel),
        ],
        level=2,
    )

    return (
        "<section>"
        "<h2>Metric Details</h2>"
        "<p>Each metric is organized into subtabs with three distribution views (recording, epochs-per-channel, channels-per-epoch), each with violin/histogram/density variants, followed by mean channel-epoch maps whose top strip includes median and quantile bands.</p>"
        + metric_tabs
        + "</section>"
    )


def _build_subject_drilldown_section(acc: ChTypeAccumulator, amplitude_unit: str) -> str:
    df = _run_rows_dataframe(acc.run_rows)
    if df.empty:
        return "<section><h2>Subject Drill-down</h2><p>No subject-aware details are available.</p></section>"

    std_scatter = plot_run_fingerprint_scatter(
        df,
        "std_median",
        "std_upper_tail",
        "Run fingerprint (STD)",
        f"Median channel STD per recording ({amplitude_unit})",
        f"Upper-tail channel STD per recording ({amplitude_unit})",
    )
    ptp_scatter = plot_run_fingerprint_scatter(
        df,
        "ptp_median",
        "ptp_upper_tail",
        "Run fingerprint (PtP)",
        f"Median channel PtP per recording ({amplitude_unit})",
        f"Upper-tail channel PtP per recording ({amplitude_unit})",
    )
    mains_scatter = plot_run_fingerprint_scatter(
        df,
        "mains_ratio",
        "mains_harmonics_ratio",
        "Run fingerprint (PSD mains/harmonics)",
        "Mains relative power",
        "Harmonics relative power",
    )
    ecg_scatter = plot_run_fingerprint_scatter(
        df,
        "ecg_mean_abs_corr",
        "ecg_p95_abs_corr",
        "Run fingerprint (ECG)",
        "Mean |ECG correlation| per recording",
        "p95 |ECG correlation| per recording",
    )
    eog_scatter = plot_run_fingerprint_scatter(
        df,
        "eog_mean_abs_corr",
        "eog_p95_abs_corr",
        "Run fingerprint (EOG)",
        "Mean |EOG correlation| per recording",
        "p95 |EOG correlation| per recording",
    )
    muscle_scatter = plot_run_fingerprint_scatter(
        df,
        "muscle_median",
        "muscle_p95",
        "Run fingerprint (Muscle)",
        "Median muscle score per recording",
        "p95 muscle score per recording",
    )
    metric_heatmap = plot_subject_metric_heatmap(df, "Subject-by-metric normalized summary matrix (raw values on hover)")
    std_profiles = plot_subject_epoch_small_multiples(
        acc.std_subject_profiles,
        _subject_scores_from_df(df, "std_upper_tail"),
        "Subject epoch profiles (STD, top 12 by upper-tail summary)",
        y_label=f"STD ({amplitude_unit})",
        top_n=12,
    )
    ptp_profiles = plot_subject_epoch_small_multiples(
        acc.ptp_subject_profiles,
        _subject_scores_from_df(df, "ptp_upper_tail"),
        "Subject epoch profiles (PtP, top 12 by upper-tail summary)",
        y_label=f"PtP ({amplitude_unit})",
        top_n=12,
    )

    return (
        "<section>"
        "<h2>Subject Drill-down</h2>"
        "<p>These plots isolate recording and subject fingerprints after cohort-level triage.</p>"
        + _figure_block(
            std_scatter,
            "Each point is one recording. X is central summary and Y is upper-tail summary for STD.",
        )
        + _figure_block(
            ptp_scatter,
            "Each point is one recording. X is central summary and Y is upper-tail summary for PtP.",
        )
        + _figure_block(
            mains_scatter,
            "Each point is one recording. Axes summarize mains and harmonics relative power.",
        )
        + _figure_block(
            ecg_scatter,
            "Each point is one recording. Axes summarize central and upper-tail ECG correlation magnitude.",
        )
        + _figure_block(
            eog_scatter,
            "Each point is one recording. Axes summarize central and upper-tail EOG correlation magnitude.",
        )
        + _figure_block(
            muscle_scatter,
            "Each point is one recording. Axes summarize central and upper-tail muscle score burden.",
        )
        + _figure_block(
            metric_heatmap,
            "Rows are subjects; columns are metrics; color is normalized for readability and hover shows raw metric summaries.",
        )
        + _figure_block(
            std_profiles,
            "Each panel shows one subject. Line and bands summarize channel quantiles across epochs (STD).",
        )
        + _figure_block(
            ptp_profiles,
            "Each panel shows one subject. Line and bands summarize channel quantiles across epochs (PtP).",
        )
        + "</section>"
    )


def _build_statistical_appendix_section(acc: ChTypeAccumulator, amplitude_unit: str, is_combined: bool) -> str:
    suffix = " (all channels)" if is_combined else ""
    std_ecdf = _make_ecdf_figure(acc.std_dist_by_condition, f"STD ECDF{suffix}", f"STD ({amplitude_unit})")
    ptp_ecdf = _make_ecdf_figure(acc.ptp_dist_by_condition, f"PtP ECDF{suffix}", f"PtP ({amplitude_unit})")
    psd_ecdf = _make_ecdf_figure(acc.psd_ratio_by_condition, "Mains relative power ECDF", "Mains ratio")
    ecg_ecdf = _make_ecdf_figure(acc.ecg_corr_by_condition, "ECG |r| ECDF", "|r|")
    eog_ecdf = _make_ecdf_figure(acc.eog_corr_by_condition, "EOG |r| ECDF", "|r|")
    muscle_ecdf = _make_ecdf_figure(acc.muscle_scalar_by_condition, "Muscle score ECDF", "Muscle score")

    return (
        "<section>"
        "<h2>Statistical Appendix</h2>"
        "<p>Supplementary distribution views for readers who prefer cumulative representations.</p>"
        + _figure_block(std_ecdf, "Cumulative distribution of channel-level STD summaries.")
        + _figure_block(ptp_ecdf, "Cumulative distribution of channel-level PtP summaries.")
        + _figure_block(psd_ecdf, "Cumulative distribution of mains relative power summaries.")
        + _figure_block(ecg_ecdf, "Cumulative distribution of ECG correlation magnitude summaries.")
        + _figure_block(eog_ecdf, "Cumulative distribution of EOG correlation magnitude summaries.")
        + _figure_block(muscle_ecdf, "Cumulative distribution of recording-level muscle summaries.")
        + "</section>"
    )


def _build_tab_content(tab_name: str, acc: ChTypeAccumulator, is_combined: bool) -> str:
    if is_combined:
        amplitude_unit = "mixed MEG units (all channels)"
    elif tab_name.upper() == "MAG":
        amplitude_unit = "Tesla (T)"
    else:
        amplitude_unit = "Tesla/m (T/m)"

    sections = [
        ("Cohort Overview", _build_cohort_overview_section(acc, amplitude_unit=amplitude_unit, is_combined=is_combined)),
        ("Task/Condition Effects", _build_condition_effect_section(acc, amplitude_unit=amplitude_unit, is_combined=is_combined)),
        ("Metric Details", _build_metric_details_section(acc, amplitude_unit=amplitude_unit, is_combined=is_combined)),
        ("Subject Drill-down", _build_subject_drilldown_section(acc, amplitude_unit=amplitude_unit)),
        ("Statistical Appendix", _build_statistical_appendix_section(acc, amplitude_unit=amplitude_unit, is_combined=is_combined)),
        (
            "Metadata & Missingness",
            "<section><h2>Metadata & Missingness</h2>"
            + _summary_table_html(acc)
            + "<h3>Machine-readable derivatives used</h3>"
            + _paths_html(acc.source_paths)
            + "</section>",
        ),
    ]
    group_id = f"main-{re.sub(r'[^a-z0-9]+', '-', tab_name.lower())}"
    return _build_subtabs_html(group_id, sections, level=1)


def _build_report_html(
    dataset_name: str,
    tab_accumulators: Dict[str, ChTypeAccumulator],
    settings_snapshot: str,
) -> str:
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = getattr(meg_qc, "__version__", "unknown")
    tab_order = ["Combined (mag+grad)", "MAG", "GRAD"]
    available_tabs = [tab for tab in tab_order if tab in tab_accumulators and tab_accumulators[tab].run_count > 0]
    if not available_tabs:
        available_tabs = [tab for tab in tab_order if tab in tab_accumulators]

    tab_buttons = []
    tab_divs = []
    for idx, tab in enumerate(available_tabs):
        tab_id = f"tab-{idx}"
        active_class = " active" if idx == 0 else ""
        tab_buttons.append(f"<button class='tab-btn{active_class}' data-target='{tab_id}'>{tab}</button>")
        content = _build_tab_content(tab, tab_accumulators[tab], is_combined=(tab == "Combined (mag+grad)"))
        tab_divs.append(f"<div id='{tab_id}' class='tab-content{active_class}'>{content}</div>")

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QA group report {dataset_name}</title>
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
    .report-header {{
      display: flex;
      gap: 14px;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
    .theme-toggle {{
      border: 1px solid #9bbce0;
      border-radius: 10px;
      background: #ecf4ff;
      color: #21415c;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }}
    section {{
      background: rgba(255, 255, 255, 0.90);
      border: 1px solid #dce9f7;
      border-radius: 14px;
      padding: 16px 16px 10px;
      margin-top: 16px;
      box-shadow: 0 6px 24px rgba(5, 45, 79, 0.06);
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
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #d5e3f3;
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef5fd; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 12px;
    }}
    .tile {{
      border: 1px solid #d9e7f8;
      border-radius: 10px;
      padding: 10px;
      background: #fbfdff;
    }}
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
    body[data-theme="dark"] {{
      color: #dbe6f3;
      background: linear-gradient(140deg, #111723, #162434 45%, #10212f);
    }}
    body[data-theme="dark"] section {{
      background: rgba(16, 26, 39, 0.88);
      border-color: #2b3f56;
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.33);
    }}
    body[data-theme="dark"] h4,
    body[data-theme="dark"] .fig-note {{
      color: #b8cbe0;
    }}
    body[data-theme="dark"] .tab-btn,
    body[data-theme="dark"] .subtab-btn,
    body[data-theme="dark"] .fig-switch-btn,
    body[data-theme="dark"] .theme-toggle {{
      background: #1a3046;
      color: #d8e5f2;
      border-color: #3f5f80;
    }}
    body[data-theme="dark"] .tab-btn.active,
    body[data-theme="dark"] .subtab-btn.active,
    body[data-theme="dark"] .fig-switch-btn.active {{
      background: #244766;
      border-color: #6ea0cf;
    }}
    body[data-theme="dark"] .subtab-group.level-1 {{
      background: #162637;
      border-color: #2d4560;
    }}
    body[data-theme="dark"] .subtab-group.level-2 {{
      background: #132233;
      border-color: #294158;
    }}
    body[data-theme="dark"] .subtab-group.level-3 {{
      background: #11202f;
      border-color: #243a52;
    }}
    body[data-theme="dark"] .subtab-group.level-4 {{
      background: #0f1d2b;
      border-color: #1f3449;
    }}
    body[data-theme="dark"] pre {{
      background: #142436;
      border-color: #2b4058;
      color: #d5e3f2;
    }}
    body[data-theme="dark"] table,
    body[data-theme="dark"] th,
    body[data-theme="dark"] td {{
      border-color: #2b4058;
      color: #d5e3f2;
      background: #132233;
    }}
    body[data-theme="dark"] th {{
      background: #1b3248;
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <div class="report-header">
        <div>
          <h1>QA group report: {dataset_name}</h1>
          <p><strong>Generated:</strong> {generated}</p>
          <p><strong>MEGqc version:</strong> {version}</p>
          <p><strong>Epoch label:</strong> epochs</p>
        </div>
        <button id="theme-toggle" class="theme-toggle" type="button">Dark mode</button>
      </div>
      <h3>Settings snapshot</h3>
      <pre>{settings_snapshot}</pre>
      <p><strong>Important:</strong> Cohort views summarize global footprints, while subject drill-down views preserve recording identity for interpretation.</p>
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
      function stylePlotsForTheme(theme) {{
        if (typeof Plotly === 'undefined') {{
          return;
        }}
        const plots = Array.from(document.querySelectorAll('.js-plotly-plot'));
        const dark = theme === 'dark';
        const update = dark
          ? {{
              paper_bgcolor: '#0f1d2b',
              plot_bgcolor: '#0f1d2b',
              font: {{ color: '#d8e5f2' }},
            }}
          : {{
              paper_bgcolor: '#ffffff',
              plot_bgcolor: '#ffffff',
              font: {{ color: '#1a1a1a' }},
            }};
        plots.forEach((plotEl) => {{
          try {{
            Plotly.relayout(plotEl, update);
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
      const themeBtn = document.getElementById('theme-toggle');
      function applyTheme(themeName) {{
        const theme = themeName === 'dark' ? 'dark' : 'light';
        document.body.setAttribute('data-theme', theme);
        if (themeBtn) {{
          themeBtn.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
        }}
        stylePlotsForTheme(theme);
        const active = tabs.find(t => t.classList.contains('active'));
        if (active) {{
          window.requestAnimationFrame(() => resizePlots(active.id));
        }}
      }}
      if (themeBtn) {{
        const savedTheme = window.localStorage ? localStorage.getItem('megqc-theme') : null;
        applyTheme(savedTheme || 'light');
        themeBtn.addEventListener('click', () => {{
          const current = document.body.getAttribute('data-theme') || 'light';
          const next = current === 'dark' ? 'light' : 'dark';
          applyTheme(next);
          if (window.localStorage) {{
            localStorage.setItem('megqc-theme', next);
          }}
        }});
      }}
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


def _update_accumulator_for_run(acc_by_type: Dict[str, ChTypeAccumulator], record: RunRecord) -> None:
    files = record.files
    condition_label = _condition_label(record.meta)

    std_data: Dict[str, np.ndarray] = {}
    ptp_data: Dict[str, np.ndarray] = {}
    psd_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    ecg_raw_data: Dict[str, np.ndarray] = {}
    eog_raw_data: Dict[str, np.ndarray] = {}
    muscle_data: Dict[str, np.ndarray] = {}

    ptp_desc = None

    if "STDs" in files:
        std_data = _load_std_or_ptp_matrix(files["STDs"], "STD epoch_", "STD all")
    if "PtPsManual" in files:
        ptp_desc = "PtPsManual"
        ptp_data = _load_std_or_ptp_matrix(files["PtPsManual"], "PtP epoch_", "PtP all")
    elif "PtPsAuto" in files:
        ptp_desc = "PtPsAuto"
        ptp_data = _load_std_or_ptp_matrix(files["PtPsAuto"], "PtP epoch_", "PtP all")
    if "PSDs" in files:
        psd_data = _load_psd_matrix(files["PSDs"])
    if "ECGs" in files:
        ecg_raw_data = _load_correlation_values(files["ECGs"], "ecg")
    if "EOGs" in files:
        eog_raw_data = _load_correlation_values(files["EOGs"], "eog")
    if "Muscle" in files:
        muscle_data = _load_muscle_scores(files["Muscle"])

    layout_cache: Dict[str, Dict[str, SensorLayout]] = {}

    def _layout_by_desc(desc: str) -> Dict[str, SensorLayout]:
        if desc not in files:
            return {}
        if desc not in layout_cache:
            layout_cache[desc] = _load_sensor_layout(files[desc])
        return layout_cache[desc]

    run_ch_types = set()
    run_ch_types.update(std_data.keys())
    run_ch_types.update(ptp_data.keys())
    run_ch_types.update(psd_data.keys())
    run_ch_types.update(ecg_raw_data.keys())
    run_ch_types.update(eog_raw_data.keys())
    run_ch_types.update(muscle_data.keys())

    for ch_type in run_ch_types:
        if ch_type not in CH_TYPES:
            continue

        acc = acc_by_type[ch_type]
        acc.run_count += 1
        if record.meta.subject != "n/a":
            acc.subjects.add(record.meta.subject)
        acc.runs_by_condition[condition_label] += 1

        module_seen = {module: False for module in MODULES}
        row = RunMetricRow(
            run_key=record.meta.run_key,
            subject=record.meta.subject,
            session=record.meta.session,
            task=record.meta.task,
            run=record.meta.run,
            condition=record.meta.condition,
            acquisition=record.meta.acquisition,
            recording=record.meta.recording,
            processing=record.meta.processing,
            channel_type=ch_type,
        )

        if ch_type in std_data:
            matrix = std_data[ch_type]
            ch_summary_all = np.nanmedian(matrix, axis=1)
            ch_summary_norm = _robust_normalize_array(ch_summary_all)
            ch_summary = _finite_array(ch_summary_all)
            if ch_summary.size:
                acc.std_dist_by_condition[condition_label].extend(ch_summary.tolist())
                std_central_score = float(np.nanmedian(ch_summary))
                std_upper_score = float(np.nanquantile(ch_summary, 0.95))
                _update_representative_matrix(
                    acc.std_heatmap_by_condition,
                    acc.std_heatmap_score_by_condition,
                    acc.std_heatmap_score_history,
                    condition_label,
                    matrix,
                    std_central_score,
                )
                _update_upper_tail_matrix(
                    acc.std_heatmap_upper_by_condition,
                    acc.std_heatmap_upper_score_by_condition,
                    condition_label,
                    matrix,
                    std_upper_score,
                )
                row.std_median = _as_float(std_central_score)
                row.std_upper_tail = _as_float(std_upper_score)
            ch_summary_norm_finite = _finite_array(ch_summary_norm)
            if ch_summary_norm_finite.size:
                row.std_median_norm = _as_float(np.nanmedian(ch_summary_norm_finite))
                row.std_upper_tail_norm = _as_float(np.nanquantile(ch_summary_norm_finite, 0.95))
            std_profile = _profile_quantiles(matrix)
            acc.std_window_profiles.append(std_profile)
            acc.std_window_profiles_by_condition[condition_label].append(std_profile)
            _accumulate_matrix_mean(
                acc.std_heatmap_sum_by_condition,
                acc.std_heatmap_count_by_condition,
                condition_label,
                matrix,
            )
            if record.meta.subject != "n/a":
                acc.std_subject_profiles[record.meta.subject].append(std_profile)
            _update_topomap_payload_mean(
                acc.std_topomap_by_condition,
                acc.std_topomap_count_by_condition,
                condition_label,
                _layout_by_desc("STDs").get(ch_type),
                ch_summary_all,
            )
            acc.source_paths.add(str(files["STDs"]))
            module_seen["STD"] = True

        if ch_type in ptp_data:
            matrix = ptp_data[ch_type]
            ch_summary_all = np.nanquantile(matrix, 0.95, axis=1)
            ch_summary_norm = _robust_normalize_array(ch_summary_all)
            ch_summary = _finite_array(ch_summary_all)
            if ch_summary.size:
                acc.ptp_dist_by_condition[condition_label].extend(ch_summary.tolist())
                ptp_central_score = float(np.nanmedian(ch_summary))
                ptp_upper_score = float(np.nanquantile(ch_summary, 0.95))
                _update_representative_matrix(
                    acc.ptp_heatmap_by_condition,
                    acc.ptp_heatmap_score_by_condition,
                    acc.ptp_heatmap_score_history,
                    condition_label,
                    matrix,
                    ptp_central_score,
                )
                _update_upper_tail_matrix(
                    acc.ptp_heatmap_upper_by_condition,
                    acc.ptp_heatmap_upper_score_by_condition,
                    condition_label,
                    matrix,
                    ptp_upper_score,
                )
                row.ptp_median = _as_float(ptp_central_score)
                row.ptp_upper_tail = _as_float(ptp_upper_score)
            ch_summary_norm_finite = _finite_array(ch_summary_norm)
            if ch_summary_norm_finite.size:
                row.ptp_median_norm = _as_float(np.nanmedian(ch_summary_norm_finite))
                row.ptp_upper_tail_norm = _as_float(np.nanquantile(ch_summary_norm_finite, 0.95))
            ptp_profile = _profile_quantiles(matrix)
            acc.ptp_window_profiles.append(ptp_profile)
            acc.ptp_window_profiles_by_condition[condition_label].append(ptp_profile)
            _accumulate_matrix_mean(
                acc.ptp_heatmap_sum_by_condition,
                acc.ptp_heatmap_count_by_condition,
                condition_label,
                matrix,
            )
            if record.meta.subject != "n/a":
                acc.ptp_subject_profiles[record.meta.subject].append(ptp_profile)
            if ptp_desc is not None:
                _update_topomap_payload_mean(
                    acc.ptp_topomap_by_condition,
                    acc.ptp_topomap_count_by_condition,
                    condition_label,
                    _layout_by_desc(ptp_desc).get(ch_type),
                    ch_summary_all,
                )
            if "PtPsManual" in files:
                acc.source_paths.add(str(files["PtPsManual"]))
            elif "PtPsAuto" in files:
                acc.source_paths.add(str(files["PtPsAuto"]))
            module_seen["PTP"] = True

        if ch_type in psd_data:
            freqs, matrix = psd_data[ch_type]
            ratios_all, harmonics_all = _compute_mains_and_harmonics_ratio(matrix, freqs)
            ratios = _finite_array(ratios_all)
            if ratios.size:
                acc.psd_ratio_by_condition[condition_label].extend(ratios.tolist())
                row.mains_ratio = _as_float(np.nanmean(ratios))
            harmonics = _finite_array(harmonics_all)
            if harmonics.size:
                row.mains_harmonics_ratio = _as_float(np.nanmean(harmonics))
            quant = _profile_quantiles(matrix)
            quant["freqs"] = freqs
            acc.psd_profiles.append(quant)
            acc.psd_profiles_by_condition[condition_label].append(quant)
            _update_topomap_payload_mean(
                acc.psd_topomap_by_condition,
                acc.psd_topomap_count_by_condition,
                condition_label,
                _layout_by_desc("PSDs").get(ch_type),
                ratios_all,
            )
            acc.source_paths.add(str(files["PSDs"]))
            module_seen["PSD"] = True

        if ch_type in ecg_raw_data:
            vals_all = np.asarray(ecg_raw_data[ch_type], dtype=float)
            vals = _finite_array(vals_all)
            if vals.size:
                acc.ecg_corr_by_condition[condition_label].extend(vals.tolist())
                row.ecg_mean_abs_corr = _as_float(np.nanmean(vals))
                row.ecg_p95_abs_corr = _as_float(np.nanquantile(vals, 0.95))
            _update_topomap_payload_mean(
                acc.ecg_topomap_by_condition,
                acc.ecg_topomap_count_by_condition,
                condition_label,
                _layout_by_desc("ECGs").get(ch_type),
                vals_all,
            )
            acc.source_paths.add(str(files["ECGs"]))
            module_seen["ECG"] = True

        if ch_type in eog_raw_data:
            vals_all = np.asarray(eog_raw_data[ch_type], dtype=float)
            vals = _finite_array(vals_all)
            if vals.size:
                acc.eog_corr_by_condition[condition_label].extend(vals.tolist())
                row.eog_mean_abs_corr = _as_float(np.nanmean(vals))
                row.eog_p95_abs_corr = _as_float(np.nanquantile(vals, 0.95))
            _update_topomap_payload_mean(
                acc.eog_topomap_by_condition,
                acc.eog_topomap_count_by_condition,
                condition_label,
                _layout_by_desc("EOGs").get(ch_type),
                vals_all,
            )
            acc.source_paths.add(str(files["EOGs"]))
            module_seen["EOG"] = True

        if ch_type in muscle_data:
            scores = _finite_array(muscle_data[ch_type])
            if scores.size:
                acc.muscle_profiles.append(scores)
                acc.muscle_profiles_by_condition[condition_label].append(scores)
                acc.muscle_scalar_by_condition[condition_label].append(
                    float(np.nanquantile(scores, 0.95))
                )
                row.muscle_median = _as_float(np.nanmedian(scores))
                row.muscle_p95 = _as_float(np.nanquantile(scores, 0.95))
            acc.source_paths.add(str(files["Muscle"]))
            module_seen["Muscle"] = True

        acc.run_rows.append(row)

        for module in MODULES:
            if module_seen[module]:
                acc.module_present[module] += 1
            else:
                acc.module_missing[module] += 1


def make_group_plots_meg_qc(
    dataset_path: str,
    derivatives_base: Optional[str] = None,
) -> Dict[str, Path]:
    """Build dataset-level QA reports from saved per-run derivatives.

    Parameters
    ----------
    dataset_path : str
        Path to the BIDS dataset.
    derivatives_base : str, optional
        Optional external parent directory for derivatives, matching the
        behavior of the main MEGqc plotting pipeline.

    Returns
    -------
    dict
        Mapping ``{"report": Path(...)}`` for the generated dataset-level HTML.
    """

    _, derivatives_root = resolve_output_roots(dataset_path, derivatives_base)
    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    calculation_dir = Path(derivatives_root) / "Meg_QC" / "calculation"
    reports_dir = Path(derivatives_root) / "Meg_QC" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not calculation_dir.exists():
        print(f"___MEGqc___: Group QA: calculation folder not found: {calculation_dir}")
        return {}

    run_records = _discover_run_records(calculation_dir)
    if not run_records:
        print(f"___MEGqc___: Group QA: no run-level TSV derivatives found under {calculation_dir}")
        return {}

    acc_by_type: Dict[str, ChTypeAccumulator] = {ch: ChTypeAccumulator() for ch in CH_TYPES}

    for run_key in sorted(run_records):
        _update_accumulator_for_run(acc_by_type, run_records[run_key])

    settings_snapshot = _load_settings_snapshot(derivatives_root)
    combined_acc = _combine_accumulators(acc_by_type)

    tab_accumulators: Dict[str, ChTypeAccumulator] = {
        "Combined (mag+grad)": combined_acc,
        "MAG": acc_by_type["mag"],
        "GRAD": acc_by_type["grad"],
    }
    if all(acc.run_count == 0 for acc in tab_accumulators.values()):
        print("___MEGqc___: Group QA: no reports were generated.")
        return {}

    report_html = _build_report_html(
        dataset_name=dataset_name,
        tab_accumulators=tab_accumulators,
        settings_snapshot=settings_snapshot,
    )
    out_name = f"QA_group_report_{dataset_name}.html"
    out_path = reports_dir / out_name
    out_path.write_text(report_html, encoding="utf-8")

    print("___MEGqc___: Group QA report created:")
    print(f"___MEGqc___:   report: {out_path}")
    return {"report": out_path}
