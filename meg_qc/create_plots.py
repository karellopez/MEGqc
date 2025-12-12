import os
import argparse

def get_plots():
    from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc

    dataset_path_parser = argparse.ArgumentParser(description= "parser for MEGqc: --inputdata(mandatory) path/to/your/BIDSds)")
    dataset_path_parser.add_argument("--inputdata", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    dataset_path_parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help="Optional folder to store MEGqc derivatives outside the BIDS dataset",
    )
    args=dataset_path_parser.parse_args()
    data_directory = args.inputdata

    print(data_directory)
    print(type(data_directory))

    make_plots_meg_qc(data_directory, derivatives_base=args.derivatives_output)


get_plots()