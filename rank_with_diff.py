import pandas as pd
import argparse
import os
from collections import defaultdict
import warnings
import shutil

# Ignore all warnings
warnings.filterwarnings("ignore")

def create_intervals(min_val, max_val, interval_points):
    bins = [min_val] + interval_points + [max_val]
    labels = [f"{bins[i]:.10f}-{bins[i+1]:.10f}" for i in range(len(bins)-1)]
    return bins, labels

def extract_queries(main_data, inputs, param_to_categorize, bins):
    extracted_queries = pd.DataFrame(columns=main_data.columns)

    for input_str in inputs:
        bin_num, num_queries, distribution = input_str.split()
        num_queries = int(num_queries)

        try:
            bin_index = int(bin_num.replace('bin', ''))
            if bin_index < 0 or bin_index >= len(bins) - 1:
                raise ValueError("Bin number out of range")
        except ValueError:
            print(f"Warning: Invalid bin number '{bin_num}', skipping.")
            continue

        bin_interval = f"{bins[bin_index]:.10f}-{bins[bin_index + 1]:.10f}"
        print(f"Processing bin {bin_interval} for {num_queries} queries with '{distribution}' distribution.")
        bin_queries = main_data[main_data[f"{param_to_categorize}_class"] == bin_interval]
        distribution_queries = bin_queries[bin_queries['distribution'] == distribution]

        if distribution_queries.empty:
            print(f"Warning: No queries found with '{distribution}' distribution in bin {bin_interval}, skipping.")
            continue

        num_distribution_queries = len(distribution_queries)
        if num_queries > num_distribution_queries:
            print(f"Warning: Only {num_distribution_queries} queries found with '{distribution}' distribution in bin {bin_interval}, although {num_queries} were requested.")
            num_queries = num_distribution_queries

        queries = distribution_queries.head(num_queries)
        if not queries.empty:
            extracted_queries = pd.concat([extracted_queries, queries], ignore_index=True)
            print(f"Queries for {bin_interval} are processed.")
        else:
            print(f"Warning: No queries found in bin {bin_interval}, skipping.")

    return extracted_queries

def main(param_to_categorize):
    min_values = {
        'cardinality': 0.0, 
        'executionTime': 0, 
        'mbrTests': 0 
    }

    max_values = {
        'cardinality': 0.05, 
        'executionTime': 1000, 
        'mbrTests': 1000000 
    }

    interval_points = {
        'cardinality': [0.0000000005, 0.000000005, 0.00000005, 0.0000005, 0.000005, 0.00005, 0.0005, 0.005], 
        'executionTime': [10, 15, 20, 25, 30, 35, 40, 45, 50], 
        'mbrTests': [10000, 50000, 100000, 200000, 500000, 700000, 900000] 
    }

    if param_to_categorize not in min_values:
        raise ValueError(f"Parameter to categorize must be one of {list(min_values.keys())}")

    min_val = min_values[param_to_categorize]
    max_val = max_values[param_to_categorize]
    intervals = interval_points[param_to_categorize]
    bins, labels = create_intervals(min_val, max_val, intervals)

    main_data = pd.read_csv('rq_result.csv', delimiter=';') 
    summary_data = pd.read_csv('dataset-summaries.csv', delimiter=';') 
    main_data = main_data.merge(summary_data[['datasetName', 'distribution']], on='datasetName', how='left')

    distribution_counts = main_data['distribution'].value_counts()
    print("Distribution Counts:")
    print(distribution_counts)

    main_data[f'{param_to_categorize}_class'] = pd.cut(main_data[param_to_categorize], bins=bins, labels=labels, include_lowest=True)
    print(f"\n{param_to_categorize.capitalize()} Counts:")
    print(main_data[f'{param_to_categorize}_class'].value_counts().sort_index())

    print("\nBin Associations:")
    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
        print(f"Bin {i}: {bin_start:.10f} - {bin_end:.10f}")

    inputs = []

    print("\nEnter your queries in the format 'bin_num num_queries distribution'. Type 'end' to finish.")
    while True:
        user_input = input("Input: ")
        if user_input.lower() == 'end':
            break
        inputs.append(user_input)

    print("\nProcessing user inputs:")
    extracted_queries = extract_queries(main_data, inputs, param_to_categorize, bins)

    max_index = 0
    for folder_name in os.listdir("."):
        if folder_name.startswith("training_set_"):
            try:
                index = int(folder_name.split("_")[-1])
                max_index = max(max_index, index)
            except ValueError:
                pass

    new_folder_name = f"training_set_{max_index + 1}"
    os.makedirs(new_folder_name, exist_ok=True)
    shutil.copy("augm/augmentation.py", new_folder_name)                      ###### script

    extracted_queries.drop(columns=['distribution']).to_csv(os.path.join(new_folder_name, "rq_result_ts.csv"), index=False, sep=';')
    used_datasets = extracted_queries['datasetName'].unique()
    filtered_summaries = summary_data[summary_data['datasetName'].isin(used_datasets)]
    filtered_summaries.to_csv(os.path.join(new_folder_name, "dataset-summaries_ts.csv"), index=False, sep=';')

    # New step to create fd2_ts.csv
    fd2_data = pd.read_csv('fd2_geom_allds.csv', delimiter=',')
    filtered_fd2 = fd2_data[fd2_data['datasetName'].isin(used_datasets)]
    filtered_fd2.to_csv(os.path.join(new_folder_name, "fd2_geom_allds_ts.csv"), index=False)

    # Generate the training_set_N_diff folder
    new_diff_folder_name = f"training_set_{max_index + 1}_diff"
    os.makedirs(new_diff_folder_name, exist_ok=True)

    # Generate differences for rq_result
    rq_result_diff = pd.concat([main_data, extracted_queries]).drop_duplicates(keep=False)
    rq_result_diff.drop(columns=['distribution']).to_csv(os.path.join(new_diff_folder_name, "rq_result_diff.csv"), index=False, sep=';')

    # Generate differences for dataset-summaries
    dataset_summaries_diff = summary_data[~summary_data['datasetName'].isin(used_datasets)]
    dataset_summaries_diff.to_csv(os.path.join(new_diff_folder_name, "dataset-summaries_diff.csv"), index=False, sep=';')

    # Generate differences for fd2_geom_allds
    fd2_geom_diff = fd2_data[~fd2_data['datasetName'].isin(used_datasets)]
    fd2_geom_diff.to_csv(os.path.join(new_diff_folder_name, "fd2_geom_allds_diff.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some CSV files and output the results.")
    parser.add_argument('param_to_categorize', type=str, choices=['cardinality', 'executionTime', 'mbrTests'], help='The parameter to categorize.')
    args = parser.parse_args()
    main(args.param_to_categorize)

