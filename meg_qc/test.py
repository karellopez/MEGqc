import argparse
import os
import sys
import shutil
from typing import List, Union

def hello_world():
    """
    Simple example function that prints the --subs argument from the command line.
    Not directly related to MEG QC, but provided as an example.
    """
    dataset_path_parser = argparse.ArgumentParser(description="parser for string to print")
    dataset_path_parser.add_argument("--subs", nargs='+', type=str, required=False, help="path to config file")
    args = dataset_path_parser.parse_args()
    print(args.subs)


def run_megqc():
    """
    Main entry point for launching the MEG QC pipeline from the command line.

    Command line usage example:
        run-megqc --inputdata /path/to/BIDS_dataset [--config /path/to/config.ini] [--subs 001 002] [--n_jobs 4]

    After parsing arguments, it calls make_derivative_meg_qc() with the chosen config,
    dataset path, subject list, and number of parallel jobs.
    """
    from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc
    import time

    import argparse

    # Create an ArgumentParser for MEG QC
    dataset_path_parser = argparse.ArgumentParser(
        description=(
            "Command-line argument parser for MEGqc.\n"
            "--inputdata (required): path to a BIDS dataset.\n"
            "--config (optional): path to a config file.\n"
            "--subs (optional): list of subject IDs (defaults to all).\n"
            "--n_jobs (optional): number of parallel jobs (defaults to 1)."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    dataset_path_parser.add_argument(
        "--inputdata",
        type=str,
        required=True,
        help=(
            "Path to the root of your BIDS MEG dataset.\n"
            "This is a required argument.\n"
            "Example: /path/to/dataset"
        )
    )

    dataset_path_parser.add_argument(
        "--config",
        type=str,
        required=False,
        help=(
            "Path to a INI config file with user-defined parameters.\n"
            "Optional: If not provided, default parameters are used.\n"
            "Example: /path/to/config.ini"
        )
    )

    dataset_path_parser.add_argument(
        "--subs",
        nargs='+',
        type=str,
        required=False,
        help=(
            "List of subject IDs to run the pipeline on.\n"
            "Optional: If not provided, the pipeline will run on all subjects.\n"
            "Example: --subs 009 012 013"
        )
    )

    dataset_path_parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help=(
            "Number of parallel jobs to use during processing.\n"
            "Default is 1. Use -1 to utilize all available CPU cores.\n"
            "\n"
            "⚠️ Recommendation based on system memory:\n"
            "  - 8 GB RAM → up to 1 parallel jobs (default)\n"
            "  - 16 GB RAM → up to 2 parallel jobs\n"
            "  - 32 GB RAM → up to 6 parallel jobs\n"
            "  - 64 GB RAM → up to 16 parallel jobs\n"
            "  - 128 GB RAM → up to 30 parallel jobs\n"
            "\n"
            "Using --n_jobs -1 will use all available CPU cores.\n"
            "Note: this may not always be optimal, especially when processing many subjects\n"
            "on systems with limited memory.\n"
            "⚠️ If you have many CPU cores but low RAM, this can lead to crashes.\n"
            "As a rule of thumb, your available RAM (in GB) should be at least\n"
            "3.5 times the number of CPUs. For example, using 16 CPUs\n"
            "requires at least 56 GB of total system memory (46 GB of available memory)."
        )
    )

    # Parse arguments
    args = dataset_path_parser.parse_args()

    # ----------------------------------------------------------------
    # Prepare internal and default user config file paths
    # ----------------------------------------------------------------
    path_to_megqc_installation = os.path.abspath(
        os.path.join(os.path.abspath(__file__), os.pardir)
    )
    relative_path_to_internal_config = "settings/settings_internal.ini"
    relative_path_to_config = "settings/settings.ini"

    # Normalize both relative paths
    relative_path_to_internal_config = os.path.normpath(relative_path_to_internal_config)
    relative_path_to_config = os.path.normpath(relative_path_to_config)

    # Join paths to form absolute paths
    internal_config_file_path = os.path.join(
        path_to_megqc_installation,
        relative_path_to_internal_config
    )

    # Print for debug, showing which directory is in use
    print("MEG QC installation directory:", path_to_megqc_installation)

    data_directory = args.inputdata
    print("Data directory:", data_directory)

    # Check if --subs was provided
    if args.subs is None:
        sub_list = 'all'
    else:
        sub_list = args.subs
        print("Subjects to process:", sub_list)

    # Decide how to handle the config file
    if args.config is None:
        url_megqc_book = 'https://aaronreer.github.io/docker_workshop_setup/settings_explanation.html'
        text = 'The settings explanation section of our MEGqc User Jupyterbook'

        print(
            'You called the MEGqc pipeline without the optional\n\n'
            '--config <path/to/custom/config> argument.\n\n'
            'MEGqc will proceed with the default parameter settings.\n'
            'Detailed information on the user parameters in MEGqc and their default values '
            f'can be found here: \n\n\033]8;;{url_megqc_book}\033\\{text}\033]8;;\033\\\n\n'
        )
        user_confirm = input('Do you want to proceed with the default settings? (y/n): ').lower().strip() == 'y'
        if user_confirm:
            config_file_path = os.path.join(path_to_megqc_installation, relative_path_to_config)
        else:
            print(
                "Use the following command to copy the default config file:\n"
                "   get-megqc-config --target_directory <path/to/directory>\n\n"
                "Then edit the copied file (e.g., to adjust parameters) and run the pipeline again with:\n"
                "   run-megqc --inputdata <path> --config <path/to/modified_config.ini>\n\n"
            )
            return
    else:
        config_file_path = args.config

    # ----------------------------------------------------------------
    # Number of parallel jobs
    # ----------------------------------------------------------------
    n_jobs_used = args.n_jobs
    print(f"Running MEG QC in parallel with n_jobs={n_jobs_used}")

    # ----------------------------------------------------------------
    # Optionally measure time for the pipeline execution
    # ----------------------------------------------------------------
    start_time = time.time()

    # ----------------------------------------------------------------
    # Run the MEG QC pipeline
    # ----------------------------------------------------------------
    make_derivative_meg_qc(
        default_config_file_path=config_file_path,
        internal_config_file_path=internal_config_file_path,
        ds_paths=data_directory,
        sub_list=sub_list,
        n_jobs=n_jobs_used
    )

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    print(f"MEGqc has completed the calculation of metrics in {elapsed_seconds:.2f} seconds.")
    print(
        f"Results can be found in {data_directory}/derivatives/Meg_QC/calculation"
    )

    # ----------------------------------------------------------------
    # Optionally prompt the user to run the plotting module
    # ----------------------------------------------------------------
    user_input = input('Do you want to run the MEGqc plotting module on the MEGqc results? (y/n): ').lower().strip() == 'y'
    if user_input:
        from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc
        make_plots_meg_qc(data_directory)
        return
    else:
        return


def get_config():
    """
    Copies the default config file (settings.ini) to the user-specified target directory.
    Allows the user to customize the config before running MEG QC.
    """
    target_directory_parser = argparse.ArgumentParser(
        description="parser for MEGqc get_config: "
                    "--target_directory (mandatory) path/to/directory to store the config"
    )
    target_directory_parser.add_argument(
        "--target_directory",
        type=str,
        required=True,
        help="Path to which the default MEG QC config file (settings.ini) will be copied."
    )
    args = target_directory_parser.parse_args()
    destination_directory = args.target_directory + '/settings.ini'
    print("Destination directory for config:", destination_directory)

    path_to_megqc_installation = os.path.abspath(
        os.path.join(os.path.abspath(__file__), os.pardir)
    )
    print("MEG QC installation directory:", path_to_megqc_installation)

    config_file_path = os.path.join(path_to_megqc_installation, 'settings', 'settings.ini')
    print("Source of default config file:", config_file_path)

    shutil.copy(config_file_path, destination_directory)
    print('The config file has been copied to ' + destination_directory)

    return


def get_plots():
    """
    Unified plotting entry point for QA/QC report generation.

    This command supports explicit QA and QC modes with one-shot execution.
    It can run subject-level, group-level, and multi-sample reports depending
    on provided flags and number of datasets.
    """
    from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc
    from meg_qc.plotting.meg_qc_group_plots import make_group_plots_meg_qc
    from meg_qc.plotting.meg_qc_multi_sample_group_plots import make_multi_sample_group_plots_meg_qc
    from meg_qc.plotting.meg_qc_group_qc_plots import (
        make_group_qc_plots_meg_qc,
        make_group_qc_plots_multi_meg_qc,
    )
    from meg_qc.calculation.meg_qc_pipeline import resolve_output_roots

    dataset_path_parser = argparse.ArgumentParser(
        description=(
            "MEGqc plotting dispatcher for QA/QC reports.\n\n"
            "Use explicit flags to choose report scope:\n"
            "  QA flags: --qa-subject, --qa-group, --qa-multisample, --qa-all\n"
            "  QC flags: --qc-group, --qc-multisample, --qc-all\n"
            "  Combined: --all (runs QA+QC in one shot)\n\n"
            "Examples:\n"
            "  Subject QA (single dataset):\n"
            "    run-megqc-plotting --inputdata /path/ds --qa-subject\n\n"
            "  Group QA (single dataset):\n"
            "    run-megqc-plotting --inputdata /path/ds --qa-group\n\n"
            "  Multi-sample QA (2+ datasets):\n"
            "    run-megqc-plotting --inputdata /path/ds1 /path/ds2 --qa-multisample\n\n"
            "  Group QC with selected attempt:\n"
            "    run-megqc-plotting --inputdata /path/ds --qc-group --attempt 2\n\n"
            "  All QA+QC reports (one shot):\n"
            "    run-megqc-plotting --inputdata /path/ds1 /path/ds2 --all"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    dataset_path_parser.add_argument(
        "--inputdata",
        nargs="+",
        type=str,
        required=True,
        help="Path(s) to one or more BIDS MEG datasets.",
    )
    dataset_path_parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help=(
            "Optional external derivatives parent folder. For single-dataset "
            "modes, dataset-specific derivatives are resolved from this root."
        ),
    )
    dataset_path_parser.add_argument(
        "--output_report",
        type=str,
        required=False,
        help=(
            "Optional explicit output HTML path. Allowed only when exactly one "
            "single-report mode is selected: --qa-multisample, --qc-group, "
            "or --qc-multisample."
        ),
    )
    dataset_path_parser.add_argument(
        "--attempt",
        type=int,
        required=False,
        help=(
            "QC only: attempt number used to select "
            "Global_Quality_Index_attempt_<n>.tsv when --input_tsv is not given."
        ),
    )
    dataset_path_parser.add_argument(
        "--input_tsv",
        type=str,
        required=False,
        help=(
            "QC only: explicit Global_Quality_Index_attempt_*.tsv path. "
            "Allowed only with --qc-group and a single dataset."
        ),
    )
    dataset_path_parser.add_argument(
        "-njobs",
        "--njobs",
        type=int,
        default=1,
        required=False,
        help="Parallel workers for supported QA plotting tasks (1=sequential, -1=all cores).",
    )
    dataset_path_parser.add_argument(
        "--qa-subject",
        action="store_true",
        help="Run subject-level QA plotting for each selected dataset.",
    )
    dataset_path_parser.add_argument(
        "--qa-group",
        action="store_true",
        help="Run dataset-level QA group report for each selected dataset.",
    )
    dataset_path_parser.add_argument(
        "--qa-multisample",
        action="store_true",
        help="Run one multi-sample QA report across selected datasets (requires >=2 datasets).",
    )
    dataset_path_parser.add_argument(
        "--qa-all",
        action="store_true",
        help="Run all QA modes valid for the provided datasets.",
    )
    dataset_path_parser.add_argument(
        "--qc-group",
        action="store_true",
        help=(
            "Run dataset-level QC report(s). For one dataset, uses --input_tsv "
            "or --attempt when provided; otherwise selects latest attempt."
        ),
    )
    dataset_path_parser.add_argument(
        "--qc-multisample",
        action="store_true",
        help="Run one multi-sample QC report across selected datasets (requires >=2 datasets).",
    )
    dataset_path_parser.add_argument(
        "--qc-all",
        action="store_true",
        help="Run all QC modes valid for the provided datasets.",
    )
    dataset_path_parser.add_argument(
        "--all",
        action="store_true",
        help="Run all QA and QC modes valid for the provided datasets in one shot.",
    )
    args = dataset_path_parser.parse_args()

    dataset_paths = args.inputdata
    n_datasets = len(dataset_paths)

    qa_subject = args.qa_subject
    qa_group = args.qa_group
    qa_multisample = args.qa_multisample
    qc_group = args.qc_group
    qc_multisample = args.qc_multisample

    # Expand aggregate flags.
    if args.qa_all or args.all:
        qa_subject = True
        qa_group = True
        qa_multisample = True
    if args.qc_all or args.all:
        qc_group = True
        qc_multisample = True

    # Backward compatibility: default to subject-level QA when no action flags.
    if not any([qa_subject, qa_group, qa_multisample, qc_group, qc_multisample]):
        print("No QA/QC plotting mode selected; defaulting to --qa-subject.")
        qa_subject = True

    # Validate dataset count requirements for multi-sample modes.
    if qa_multisample and n_datasets < 2:
        dataset_path_parser.error("--qa-multisample requires at least two datasets in --inputdata.")
    if qc_multisample and n_datasets < 2:
        dataset_path_parser.error("--qc-multisample requires at least two datasets in --inputdata.")

    # Validate --input_tsv usage.
    if args.input_tsv:
        if not qc_group:
            dataset_path_parser.error("--input_tsv is only valid with --qc-group.")
        if n_datasets != 1:
            dataset_path_parser.error("--input_tsv is only valid with --qc-group and a single dataset.")
        if qc_multisample:
            dataset_path_parser.error("--input_tsv cannot be combined with --qc-multisample.")

    # Validate --output_report usage to avoid path collisions.
    if args.output_report:
        enabled_modes = [
            ("qa_subject", qa_subject),
            ("qa_group", qa_group),
            ("qa_multisample", qa_multisample),
            ("qc_group", qc_group),
            ("qc_multisample", qc_multisample),
        ]
        enabled_count = sum(1 for _, on in enabled_modes if on)
        if enabled_count != 1:
            dataset_path_parser.error(
                "--output_report is only allowed when exactly one plotting mode is selected."
            )
        if qa_subject or qa_group:
            dataset_path_parser.error(
                "--output_report is not used by --qa-subject/--qa-group. "
                "Use it with --qa-multisample, --qc-group, or --qc-multisample."
            )

    # Inform user about derivatives roots that will be used.
    for ds in dataset_paths:
        _, derivatives_root = resolve_output_roots(ds, args.derivatives_output)
        print(f"Using derivatives from: {derivatives_root}")

    # --------------------------
    # QA execution
    # --------------------------
    if qa_subject:
        for ds in dataset_paths:
            print(f"Running QA subject plotting for dataset: {ds}")
            make_plots_meg_qc(ds, n_jobs=args.njobs, derivatives_base=args.derivatives_output)

    if qa_group:
        for ds in dataset_paths:
            print(f"Running QA group plotting for dataset: {ds}")
            make_group_plots_meg_qc(
                ds,
                derivatives_base=args.derivatives_output,
                n_jobs=args.njobs,
            )

    if qa_multisample:
        print("Running QA multisample plotting...")
        derivatives_bases = (
            [args.derivatives_output] * n_datasets if args.derivatives_output else None
        )
        make_multi_sample_group_plots_meg_qc(
            dataset_paths=dataset_paths,
            derivatives_bases=derivatives_bases,
            output_report_path=args.output_report,
            n_jobs=args.njobs,
        )

    # --------------------------
    # QC execution
    # --------------------------
    if qc_group:
        if n_datasets == 1:
            print(f"Running QC group plotting for dataset: {dataset_paths[0]}")
            make_group_qc_plots_meg_qc(
                dataset_path=dataset_paths[0],
                input_tsv=args.input_tsv,
                output_html=args.output_report,
                attempt=args.attempt,
                derivatives_base=args.derivatives_output,
            )
        else:
            if args.input_tsv:
                dataset_path_parser.error(
                    "--input_tsv is not supported when --qc-group is used with multiple datasets."
                )
            for ds in dataset_paths:
                print(f"Running QC group plotting for dataset: {ds}")
                make_group_qc_plots_meg_qc(
                    dataset_path=ds,
                    input_tsv=None,
                    output_html=None,
                    attempt=args.attempt,
                    derivatives_base=args.derivatives_output,
                )

    if qc_multisample:
        print("Running QC multisample plotting...")
        make_group_qc_plots_multi_meg_qc(
            dataset_paths=dataset_paths,
            output_html=args.output_report,
            attempt=args.attempt,
            derivatives_base=args.derivatives_output,
        )

    return


def run_gqi():
    """Recalculate Global Quality Index reports using existing metrics."""
    from meg_qc.calculation.metrics.summary_report_GQI import generate_gqi_summary
    from meg_qc.calculation.meg_qc_pipeline import resolve_output_roots

    parser = argparse.ArgumentParser(
        description="Recompute Global Quality Index using previously calculated metrics"
    )
    parser.add_argument(
        "--inputdata",
        type=str,
        required=True,
        help="Path to the root of your BIDS MEG dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to a config file with GQI parameters",
    )
    parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help="Optional folder to store derivatives outside the BIDS dataset",
    )
    args = parser.parse_args()

    install_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
    default_config = os.path.join(install_path, "settings", "settings.ini")
    cfg_path = args.config if args.config else default_config

    # Use the same resolver as the calculation module so that external
    # derivatives directories are handled consistently for GQI regeneration.
    _, derivatives_root = resolve_output_roots(args.inputdata, args.derivatives_output)

    generate_gqi_summary(args.inputdata, derivatives_root, cfg_path)
    return
