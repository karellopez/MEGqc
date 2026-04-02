# Pull Request: Full EEG Support & Integrated QC Viewer

## Summary

This PR introduces **full EEG data support** across the entire MEGqc pipeline and adds a new **integrated QC Viewer** for interactive inspection of QC results. The toolbox evolves from MEG-only to a unified MEG+EEG quality control solution.

**Version bump: 0.7.9 → 0.9.0**

---

##  Major Features

### 1. Full EEG Data Support

The pipeline can now process standalone EEG recordings in BIDS format, in addition to the existing MEG (FIF/CTF) support. All QC metrics have been extended to handle EEG channels natively.

#### Supported EEG Formats
- **EDF** (`.edf`) — European Data Format
- **BDF** (`.bdf`) — BioSemi Data Format
- **BrainVision** (`.vhdr`) — Brain Products
- **EEGLAB** (`.set`) — EEGLAB datasets
- **EGI/MFF** (`.mff`) — EGI NetStation
- **Neuroscan** (`.cnt`) — Neuroscan CNT
- **FIF** (`.fif`) — EEG stored in FIF containers

#### EEG-Specific Processing Pipeline
- **BIDS modality detection**: Files with BIDS suffix `_eeg` are automatically classified as independent EEG recordings; `_meg` files remain as MEG. A safeguard reclassifies `_meg` files that contain zero MEG channels.
- **Channel type correction from BIDS `channels.tsv`**: New `_apply_bids_channel_types()` function reads the companion sidecar and corrects channel types (EOG, EMG, ECG, MISC) that MNE may auto-detect as EEG.
- **Automatic montage detection**: `apply_eeg_montage()` tries `standard_1005`, `standard_1010`, and `standard_1020` montages with a 50% channel-name overlap threshold. A second pass strips common reference suffixes (e.g., `Fp1-M2` → `Fp1`) before retrying.
- **EEG re-referencing**: `apply_eeg_reference()` supports `average`, `REST`, or `none` methods, configurable via the new `[EEG]` settings section.
- **Projector conflict handling**: ECG/EOG channel retyping now safely removes conflicting SSP projectors (e.g., "Average EEG reference") before retrying `set_channel_types()`.

#### Metric Adaptations for EEG
| Metric | EEG Adaptation |
|--------|---------------|
| **STD** | EEG channels processed with Volts units |
| **PTP (Peak-to-Peak)** | New thresholds: `peak_eeg = 200µV`, `flat_eeg = 1µV` |
| **PSD** | Nyquist clamping for low-sample-rate EEG; dynamic dict initialization for `eeg` key |
| **Muscle** | Separate frequency band: `muscle_freqs_eeg = 20, 100 Hz` (vs. 110–140 Hz for MEG) |
| **ECG** | Graceful skip for EEG-only data without ECG channel; name-based ECG channel search |
| **EOG** | Projector-safe channel retyping |
| **Head Motion** | Automatically skipped for EEG (requires MEG cHPI coils) |
| **Peaks (auto/manual)** | Extended to handle EEG channel objects |
| **GQI** | Summary metrics now include "EEG CHANNELS" column |

### 2. Integrated QC Viewer (`qc_viewer/`)

A brand-new PyQt6 + pyqtgraph-based interactive viewer for inspecting QC results, launchable directly from the MEGqc GUI via the **"QC Viewer"** button.

**~3,900 lines of new code** across 6 modules:

| Module | Lines | Description |
|--------|-------|-------------|
| `viewer_window.py` | 272 | Main window shell with menu bar and status bar |
| `file_panel.py` | 146 | BIDS-aware file system explorer tree |
| `content_panel.py` | 543 | Renders HTML reports (via QWebEngineView or system browser fallback), JSON metrics, TSV tables |
| `timeseries_widget.py` | 2,188 | Real-time scrollable time-series viewer with channel selection, event markers, annotation overlays |
| `annotation_manager.py` | 740 | Loads/manages QC annotations from derivative directories; supports ECG/EOG event peaks, BIDS events.tsv, stimulus markers |
| `__init__.py` | 16 | Package exports |

**Key Viewer Capabilities:**
- Browse BIDS derivative directories and open any QC output file
- Render interactive Plotly HTML reports inline (QWebEngineView) or in system browser
- Display JSON metrics in formatted view
- Load raw MEG/EEG data and display scrollable time series with pyqtgraph
- Overlay ECG/EOG detected events, BIDS stimulus events, and annotation markers
- Auto-load annotations from the nearest matching derivative profile
- Theme synchronization with the main MEGqc GUI

### 3. GUI Enhancements

- **EEG channel type checkbox**: The settings editor now shows `mag`, `grad`, and `eeg` checkboxes for `ch_types` selection.
- **EEG settings section**: New dropdown widgets for `reference_method` (average/REST/none) and `montage` (auto/standard_1020/biosemi64/…).
- **Multi-directory dataset selection**: The "Browse" button for dataset input now supports selecting multiple BIDS dataset folders at once.
- **Theme propagation**: Theme changes in the main GUI are forwarded to the QC Viewer window.

---

## 📁 Changed Files

### Core Pipeline (`calculation/`)

| File | +/- Lines | Changes |
|------|-----------|---------|
| `initial_meg_qc.py` | +672 / -89 | EEG format loaders, montage/reference/channel-type helpers, `SUPPORTED_CH_TYPES` constant, EEG epoching, `load_data()` returns `modality`, `_apply_bids_channel_types()`, `apply_eeg_montage()`, `apply_eeg_reference()` |
| `meg_qc_pipeline.py` | +259 / -82 | BIDS `suffix='eeg'` file discovery, multimodal dataset resolution, per-modality derivative folders, EEG-aware config snapshots, `_sensor_key()` for summary CSV, safe variable cleanup |
| `objects.py` | +8 / -4 | `MEG_channel` → `QC_channel` rename (with backward-compat alias), `eeg` type in docstring |

### Metric Modules (`calculation/metrics/`)

| File | +/- Lines | Changes |
|------|-----------|---------|
| `ECG_EOG_meg_qc.py` | +169 / -24 | `no_ecg` method for EEG-only data, name-based ECG channel search, projector-safe EOG retyping, epoch length alignment, correlation failure guards, richer `mean_good=False` metadata |
| `Head_meg_qc.py` | +12 / -2 | Skip with informative message for EEG data |
| `PSD_meg_qc.py` | +20 / -12 | Nyquist clamping, dynamic dict keys from `m_or_g_chosen`, EEG key in simple metrics |
| `Peaks_auto_meg_qc.py` | +4 / -2 | Handle EEG channel objects |
| `Peaks_manual_meg_qc.py` | +7 / -5 | Handle EEG channel objects |
| `Peaks_manual_meg_qc_numba.py` | +7 / -5 | Handle EEG channel objects |
| `STD_meg_qc.py` | +5 / -5 | EEG-aware threshold/unit handling |
| `muscle_meg_qc.py` | +22 / -7 | EEG-specific frequency band (20–100 Hz), Nyquist clamping, EEG-only fallback |
| `summary_report_GQI.py` | +108 / -56 | "EEG CHANNELS" column in all summary tables, `eeg` key throughout |

### Plotting (`plotting/`)

| File | +/- Lines | Changes |
|------|-----------|---------|
| `meg_qc_plots.py` | +513 / -48 | EEG topomap support, EEG-aware plot generation for all metric types |
| `meg_qc_group_plots.py` | +268 / -165 | EEG group-level plots, dynamic channel type handling |
| `meg_qc_group_qc_plots.py` | +402 / -59 | EEG group QC plots across all metrics |
| `meg_qc_multi_sample_group_plots.py` | +131 / -34 | Multi-sample EEG support |
| `universal_html_report.py` | +30 / -7 | EEG-aware report generation |
| `universal_plots.py` | +143 / -23 | EEG units (Volts, Volts/Hz), enhanced ECG/EOG plot metadata, shape-check failure annotations |

### GUI (`miscellaneous/GUI/`)

| File | +/- Lines | Changes |
|------|-----------|---------|
| `megqcGUI.py` | +79 / -9 | EEG checkbox, EEG settings dropdowns, QC Viewer button, multi-directory browse, theme propagation |
| `qc_viewer/` | +3,905 (new) | Complete QC Viewer package (6 modules) |

### Configuration

| File | +/- Lines | Changes |
|------|-----------|---------|
| `settings.ini` | +25 / -1 | `eeg` in `ch_types` docs, `peak_eeg`/`flat_eeg` thresholds, `muscle_freqs_eeg`, new `[EEG]` section with `reference_method` and `montage` |
| `pyproject.toml` | version bump, description update |
| `requirements.txt` | +2 new deps: `pyqtgraph>=0.13.3`, `PyQt6-WebEngine>=6.6.0` |

### Other

| File | Changes |
|------|---------|
| `miscellaneous/optimizations/artifact_detection_ancp.py` | Minor updates for EEG compatibility |

---

## 🔧 New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pyqtgraph` | ≥ 0.13.3 | High-performance time-series plotting in the QC Viewer |
| `PyQt6-WebEngine` | ≥ 6.6.0 | Inline HTML report rendering in the QC Viewer (optional — falls back to system browser) |

---

## ⚙️ Configuration Changes

### New Settings in `settings.ini`

```ini
[GENERAL]
# ch_types now supports: mag, grad, eeg (or any combination)
ch_types = mag, grad

[PTP_auto]
# New EEG-specific thresholds
peak_eeg = 200e-6
flat_eeg = 1e-6

[Muscle]
# New EEG muscle artifact frequency band
muscle_freqs_eeg = 20, 100

[EEG]  # ← Entirely new section
reference_method = average    # average | REST | none
montage = auto                # auto | standard_1020 | standard_1005 | biosemi64 | ...
```

---

## 📂 BIDS Derivative Output Tree

Derivatives are now organized with a **modality subfolder** (`meg/` or `eeg/`) under `calculation/`, `reports/`, and `summary_reports/group_metrics/`. This prevents filename collisions in multimodal datasets and keeps outputs cleanly separated.

### MEG-only dataset (e.g., `ds000247`)

```
derivatives/Meg_QC/profiles/analysis_YYYYMMDD_HHMMSS/
├── config/
│   ├── desc-UsedSettings…_settings.ini
│   └── desc-UsedSettings…_settings.json
├── calculation/
│   └── meg/                                          ← modality subfolder
│       ├── sub-0002/
│       │   ├── sub-0002_…_desc-ECGchannel_meg.tsv
│       │   ├── sub-0002_…_desc-SimpleMetrics_meg.json
│       │   ├── sub-0002_…_desc-STDs_meg.tsv
│       │   ├── sub-0002_…_desc-PSDs_meg.tsv
│       │   ├── sub-0002_…_desc-Muscle_meg.tsv
│       │   └── …                                     (all derivatives use _meg suffix)
│       └── sub-0003/
│           └── …
├── reports/
│   └── meg/
│       ├── sub-0002/
│       │   └── sub-0002_desc-subject_qa_report_meg.html
│       └── sub-0003/
│           └── …
└── summary_reports/
    ├── config/
    ├── global_quality_index_1/
    │   ├── sub-0002/
    │   └── sub-0003/
    └── group_metrics/
        └── meg/
            └── Global_Quality_Index_attempt_1_meg.tsv
```

### EEG-only dataset (e.g., `ds007315`)

```
derivatives/Meg_QC/profiles/analysis_YYYYMMDD_HHMMSS/
├── config/
│   ├── desc-UsedSettings…_settings.ini
│   └── desc-UsedSettings…_settings.json
├── calculation/
│   └── eeg/                                          ← modality subfolder
│       ├── sub-G01/
│       │   ├── sub-G01_…_desc-EventSummary_eeg.json
│       │   ├── sub-G01_…_desc-SimpleMetrics_eeg.json
│       │   ├── sub-G01_…_desc-STDs_eeg.tsv
│       │   ├── sub-G01_…_desc-PSDs_eeg.tsv
│       │   ├── sub-G01_…_desc-PSDnoiseEeg_eeg.tsv
│       │   ├── sub-G01_…_desc-Muscle_eeg.tsv
│       │   ├── sub-G01_…_desc-Noisy_epochs_on_std_base_eeg_eeg.tsv
│       │   └── …                                     (all derivatives use _eeg suffix)
│       └── sub-G02/
│           └── …
├── reports/
│   └── eeg/
│       ├── sub-G01/
│       │   └── sub-G01_desc-subject_qa_report_eeg.html
│       └── sub-G02/
│           └── …
└── summary_reports/
    ├── config/
    ├── global_quality_index_1/
    │   ├── sub-G01/
    │   └── sub-G02/
    └── group_metrics/
        └── eeg/
            └── Global_Quality_Index_attempt_1_eeg.tsv
```

### Multimodal dataset — MEG + EEG (e.g., `ds_007353`)

When `ch_types = mag, grad, eeg` is configured and the BIDS dataset contains both `_meg` and `_eeg` files, **both modalities are processed** and their outputs are placed in parallel subfolder trees:

```
derivatives/Meg_QC/profiles/analysis_YYYYMMDD_HHMMSS/
├── config/
│   ├── desc-UsedSettings…_settings.ini
│   └── desc-UsedSettings…_settings.json
├── calculation/
│   ├── meg/                                          ← MEG derivatives
│   │   ├── sub-01/
│   │   │   ├── sub-01_ses-meg_task-action_…_desc-ECGchannel_meg.tsv
│   │   │   ├── sub-01_ses-meg_task-action_…_desc-PSDnoiseMag_meg.tsv
│   │   │   └── …
│   │   └── sub-02/
│   │       └── …
│   └── eeg/                                          ← EEG derivatives
│       ├── sub-01/
│       │   ├── sub-01_ses-eeg_task-action_…_desc-EOGchannel_eeg.tsv
│       │   ├── sub-01_ses-eeg_task-action_…_desc-PSDnoiseEeg_eeg.tsv
│       │   └── …
│       └── sub-02/
│           └── …
├── reports/
│   ├── meg/
│   │   ├── sub-01/
│   │   │   └── sub-01_desc-subject_qa_report_meg.html
│   │   └── sub-02/
│   └── eeg/
│       ├── sub-01/
│       │   └── sub-01_desc-subject_qa_report_eeg.html
│       └── sub-02/
└── summary_reports/
    ├── config/
    ├── global_quality_index_1/
    │   ├── sub-01/
    │   └── sub-02/
    └── group_metrics/
        ├── meg/
        │   └── Global_Quality_Index_attempt_1_meg.tsv
        └── eeg/
            └── Global_Quality_Index_attempt_1_eeg.tsv
```

### Key changes from previous output tree

| Aspect | Before (v0.7.9) | After (v0.9.0) |
|--------|-----------------|-----------------|
| **Modality subfolder** | None — `calculation/sub-XX/` directly | `calculation/meg/sub-XX/` or `calculation/eeg/sub-XX/` |
| **File BIDS suffix** | Always `_meg` | `_meg` for MEG data, `_eeg` for EEG data |
| **Config snapshot suffix** | `_meg.ini` / `_meg.json` | `_settings.ini` / `_settings.json` |
| **Group metrics folder** | `group_metrics/meg/` | `group_metrics/meg/` or `group_metrics/eeg/` |
| **Report subfolder** | `reports/meg/sub-XX/` | `reports/meg/sub-XX/` or `reports/eeg/sub-XX/` |

---

## 🔄 Breaking Changes

- `load_data()` now returns 4 values instead of 3: `raw, shielding_str, meg_system, modality`
- `MEG_channel` class renamed to `QC_channel` (backward-compat alias `MEG_channel = QC_channel` preserved)
- `get_files_list()` has new optional parameter `m_or_g_chosen`
- BIDS derivative suffix changed from `_meg` to `_settings` for config snapshot files
- `simple_metric_basic()` now accepts `metric_global_content_eeg` and `metric_local_content_eeg` keyword arguments
- BIDS derivative folder structure: derivatives are now organized under `calculation/meg/sub-XX/` or `calculation/eeg/sub-XX/` instead of directly under `calculation/sub-XX/`

---

## ✅ Testing Notes

- Tested with BIDS-compliant EEG datasets in EDF, BDF, and BrainVision formats
- Tested with multimodal MEG+EEG datasets (correct modality routing)
- Backward compatibility verified with existing MEG-only (FIF and CTF) datasets
- QC Viewer tested with Plotly HTML reports, JSON metrics, and raw data loading
- GUI EEG settings verified in the settings editor

---

## 📊 Stats

- **22 modified files** + **1 new package** (6 new files)
- **~2,900 lines added** across existing files
- **~3,900 lines** in the new QC Viewer package
- **Total: ~6,800 new lines of code**



