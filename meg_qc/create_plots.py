import os
import argparse

def get_plots():
    from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc

    dataset_path_parser = argparse.ArgumentParser(description= "parser for MEGqc: --inputdata(mandatory) path/to/your/BIDSds)")
    dataset_path_parser.add_argument("--inputdata", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    args=dataset_path_parser.parse_args()
    data_directory = args.inputdata

    print(data_directory)
    print(type(data_directory))

    make_plots_meg_qc(data_directory)


get_plots()