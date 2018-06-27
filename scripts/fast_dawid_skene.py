import argparse
import os
import sys
sys.path.insert(0, '..')
from fast_dawid_skene.main import run


def main():
    parser = argparse.ArgumentParser(
        description='Run the Dawid-Skene, Fast Dawid-Skene, the Hybrid, or the Majority Voting Algorithm')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to use', dest='dataset_name')
    parser.add_argument('--k', type=int, required=True,
                        help='Number of annotators to use. Each data point must have at least K annotators. If more annotators are available, the first K annotators are used.', dest='k')
    parser.add_argument('--algorithm', type=str, choices=['DS', 'FDS', 'H', 'MV'], required=True,
                        help='Algorithm to use - DS: Dawid-Skene, FDS: Fast-Dawid Skene, H: Hybrid, MV: Majority Voting.')
    parser.add_argument('--mode', default='aggregate', type=str, choices=[
                        'aggregate', 'test'], required=False, help='The mode to run this program - aggregate: obtain aggregated dataset, test: aggregate data and compare with ground truths. Default is aggregate.')
    parser.add_argument('--crowd_annotations_path', default=None, type=str, required=False,
                        help='Path to crowdsourced annotations. Default is crowd.csv inside the dataset directory')
    parser.add_argument('--ground_truths_path', default=None, type=str, required=False,
                        help='Path to ground truths, if using test mode. Default is gold.csv inside the dataset directory')
    parser.add_argument('--dataset_path', default=None, type=str,
                        required=False, help='Custom path to dataset, to override default')
    parser.add_argument('--v', '--verbose', default=False,
                        type=bool, required=False, help='Run in verbose mode')
    parser.print_help()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
