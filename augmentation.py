import pandas as pd
import sys
import argparse
import os
import subprocess
from collections import defaultdict
import warnings
import random

# Ignore all warnings
warnings.filterwarnings("ignore")

def remove_geometry(file, query_mbr):
    """
    Removes a single geometry from the specified dataset file that falls outside the
    provided query MBR (Minimum Bounding Rectangle).

    Args:
        file (file object): File object to the file containing geometries.
        query_mbr (tuple[float, float, float, float]): Coordinates of the query MBR
            as (xmin, ymin, xmax, ymax).

    Returns:
        bool: True if a geometry was removed, False otherwise.
    """
    geometry_removed = False

    # Store the current position of the file pointer
    current_position = file.tell()

    # Read lines until a geometry outside the query MBR is found
    while True:
        # Read the next line
        line = file.readline()
        if not line:
            # End of file reached
            break

        # Parse the geometry coordinates
        xmin, ymin, xmax, ymax = map(float, line.strip().split(','))

        # Check if the geometry falls within the query MBR
        if not mbr_overlaps(query_mbr, (xmin, ymin, xmax, ymax)):
            # Seek back to the start of the line
            file.seek(current_position)
            # Truncate the file from the current position to remove the line
            file.truncate()
            # Geometry removed, exit the loop
            geometry_removed = True
            break

        # Update the current position to the start of the next line
        current_position = file.tell()

    return geometry_removed

def mbr_overlaps(mbr1, mbr2):
    """
    Helper function to check if two MBRs overlap.

    Args:
        mbr1 (tuple[float, float, float, float]): Coordinates of the first MBR.
        mbr2 (tuple[float, float, float, float]): Coordinates of the second MBR.

    Returns:
        bool: True if the MBRs overlap, False otherwise.
    """

    (xmin1, ymin1, xmax1, ymax1) = mbr1
    (xmin2, ymin2, xmax2, ymax2) = mbr2

    return not (xmin2 > xmax1 or xmax2 < xmin1 or ymin2 > ymax1 or ymax2 < ymin1)


def generate_random_box(mbr_dataset, w, h):

  xmin_d, ymin_d, xmax_d, ymax_d = mbr_dataset.iloc[0]
  # print(xmin_d, ymin_d, xmax_d, ymax_d)

  box_width = float(w)
  box_height = float(h)

  while True:
    xmin = random.uniform(xmin_d, xmax_d - box_width)
    ymin = random.uniform(ymin_d, ymax_d - box_height)

    xmax = xmin + box_width
    ymax = ymin + box_height

    if xmax <= xmax_d and xmin >= xmin_d and ymax <= ymax_d and ymin >= ymin_d:
      return xmin, ymin, xmax, ymax

def remove_distribution_column(file_path):
    # Read the CSV file into a DataFrame with explicit delimiter (semicolon)
    df = pd.read_csv(file_path, delimiter=';')
    
    # Check if the 'distribution' column exists and drop it if it does
    if 'distribution' in df.columns:
        df.drop(columns=['distribution'], inplace=True)
        print(f"Column 'distribution' has been removed.")
    else:
        print(f"Column 'distribution' does not exist in the file.")
    
    # Save the modified DataFrame to the same CSV file
    df.to_csv(file_path, sep=';', index=False)
    print(f"Modified file saved to {file_path}")

def extract_numbers(dataset_name):
    # Extract numbers from dataset name
    numbers = [int(s) for s in dataset_name.split("-")[-1].split("_")]
    return numbers

def get_dataset_info(dataset_summary_file, dataset_name):
    df_summary = pd.read_csv(dataset_summary_file, delimiter=';')

    # Filter rows based on dataset name
    dataset_info = df_summary[df_summary['datasetName'] == dataset_name]

    # Check if any rows match the dataset name
    if len(dataset_info) == 0:
        raise ValueError(f"No dataset information found for {dataset_name} in the summary file.")

    # Calculate size
    avg_side_length_0 = dataset_info.iloc[0]['avg_side_length_0']
    avg_side_length_1 = dataset_info.iloc[0]['avg_side_length_1']

    # Extract num_features
    num_features = dataset_info.iloc[0]['num_features']

    return avg_side_length_0, avg_side_length_1, num_features

def get_coordinates(dataset_summary_file, dataset_name):
    df_summary = pd.read_csv(dataset_summary_file, delimiter=';')
    summary_row = df_summary[df_summary['datasetName'] == dataset_name]
    if len(summary_row) == 0:
        raise ValueError(f"No dataset information found for {dataset_name} in the summary file.")
    x1, y1, x2, y2 = summary_row.iloc[0]['x1'], summary_row.iloc[0]['y1'], summary_row.iloc[0]['x2'], summary_row.iloc[0]['y2']
    return x1, y1, x2, y2
    
def get_features(dataset_summary_file, dataset_name):
    df_summary = pd.read_csv(dataset_summary_file, delimiter=';')
    summary_row = df_summary[df_summary['datasetName'] == dataset_name]
    if len(summary_row) == 0:
        raise ValueError(f"No dataset information found for {dataset_name} in the summary file.")
    n = summary_row.iloc[0]['num_features']
    return n


def get_coordinates_rq(dataset_summary_file, index):
    # Read the dataset summary file
    df_summary = pd.read_csv(dataset_summary_file, delimiter=';')

    # Check if the index is within the range of the DataFrame
    if index < 0 or index >= len(df_summary):
        raise IndexError("Index is out of range.")

    # Get the coordinates from the row at the specified index
    x1 = df_summary.iloc[index]['minX']
    y1 = df_summary.iloc[index]['minY']
    x2 = df_summary.iloc[index]['maxX']
    y2 = df_summary.iloc[index]['maxY']

    return x1, y1, x2, y2

def format_number(num):
    # Format the number to a fixed number of decimal places (e.g., 10 decimal places)
    return f"{num:.10f}"

def create_intervals(min_val, max_val, intervals):
    bins = [min_val] + intervals + [max_val]
    labels = [f"{format_number(bins[i])}-{format_number(bins[i+1])}" for i in range(len(bins)-1)]
    return bins, labels

def summarize_bin_content(main_data, param_to_categorize, bins, labels):
    bin_summary = {}
    for label in labels:
        bin_data = main_data[main_data[f'{param_to_categorize}_class'] == label]
        if bin_data.empty:
            bin_summary[label] = "Empty bin"
        else:
            dist_counts = bin_data['distribution'].value_counts().to_dict()
            bin_summary[label] = dist_counts
    return bin_summary

def validate_input(user_input):
    parts = user_input.split()
    if len(parts) < 4:
        raise ValueError("Input must be in the format 'bin_num num_queries distribution augmentation_technique1 [augmentation_technique2] [augmentation_technique3]'")
    bin_num = parts[0]
    if not bin_num.startswith("bin") or not bin_num[3:].isdigit():
        raise ValueError("The bin number must start with 'bin' followed by a number (e.g., 'bin0').")
    num_queries = parts[1]
    if not num_queries.isdigit():
        raise ValueError("The number of queries must be an integer.")
    distribution = parts[2]
    augmentation_techniques = parts[3:]
    if not augmentation_techniques:
        raise ValueError("At least one augmentation technique must be specified.")
    for technique in augmentation_techniques:
        if technique not in ['rotation', 'noise', 'merge']:
            raise ValueError("Invalid augmentation technique. Allowed values are 'rotation', 'noise', 'merge'.")
    return bin_num, int(num_queries), distribution, augmentation_techniques

def generate_random_degree():
    intervals = [88, 89, 90, 91, 92, 178, 179, 180, 181, 182, 268, 269, 270, 271, 272]
    base_value = random.choice(intervals)
    decimal_part = random.uniform(0, 0.9)  
    degree = round(base_value + decimal_part, 1)  
    return degree

def rotate_dataset(dataset_name, output_csv, angle_degrees):
    input_csv = f"../datasets/{dataset_name}.csv"

    if not os.path.exists(input_csv):
        print(f"Error: Dataset file '{input_csv}' not found.")
        return False

    try:
        # Define the command and its arguments
        command = ["python3", "../augm/rotate_data.py", input_csv, output_csv, str(angle_degrees)]

        # Run the command and capture the output
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Rotated dataset by {angle_degrees}: {input_csv}")
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Error rotating dataset '{input_csv}': {e}")
        return False


def rotate_query_window(dataset_name, xmin, ymin, xmax, ymax, degree):
    try:
        # Define the command and its arguments
        command = ["python3", "../augm/rot_mbr.py", str(xmin), str(ymin), str(xmax), str(ymax), str(degree)]

        # Run the command and capture the output and error
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Error rotating query window for dataset '{dataset_name}': {e}")
        return None

def update_dataset_summary(original_dataset_name, rotated_dataset_name, x1, y1, x2, y2, number):
    # Read the existing dataset summaries file
    df_summary = pd.read_csv('dataset-summaries_ts.csv', delimiter=';')
    
    # Check if the rotated dataset already exists in the dataset summaries
    if rotated_dataset_name in df_summary['datasetName'].values:
        # Update the existing row with the rotated dataset information
        df_summary.loc[df_summary['datasetName'] == rotated_dataset_name, ['x1', 'y1', 'x2', 'y2']] = [x1, y1, x2, y2]
    else:
        # Create a new row with the rotated dataset information
        new_row = pd.DataFrame({
            'datasetName': [rotated_dataset_name],
            'distribution': df_summary.loc[df_summary['datasetName'] == original_dataset_name, 'distribution'].values,
            'x1': [x1],
            'y1': [y1],
            'x2': [x2],
            'y2': [y2],
            'num_features': number,
            'size': df_summary.loc[df_summary['datasetName'] == original_dataset_name, 'size'].values,
            'num_points': df_summary.loc[df_summary['datasetName'] == original_dataset_name, 'num_points'].values,
            'avg_area': df_summary.loc[df_summary['datasetName'] == original_dataset_name, 'avg_area'].values,
            'avg_side_length_0': df_summary.loc[df_summary['datasetName'] == original_dataset_name, 'avg_side_length_0'].values,
            'avg_side_length_1': df_summary.loc[df_summary['datasetName'] == original_dataset_name, 'avg_side_length_1'].values,
            'E0': df_summary.loc[df_summary['datasetName'] == original_dataset_name, 'E0'].values,
            'E2': df_summary.loc[df_summary['datasetName'] == original_dataset_name, 'E2'].values
        })

        # Concatenate the new row with the existing DataFrame
        df_summary = pd.concat([df_summary, new_row], ignore_index=True)

    # Write the updated DataFrame back to the dataset summaries file
    df_summary.to_csv('dataset-summaries_ts.csv', index=False, sep=';')
    df_summary.to_csv('new_datasets.csv', index=False, sep=';')

def update_dataset_param(dataset_name, new_dataset_name, num_features):
    # Read the existing dataset summaries file
    df_summary = pd.read_csv('dataset-summaries_ts.csv', delimiter=';')

    # Check if the new dataset already exists in the dataset summaries
    if new_dataset_name in df_summary['datasetName'].values:
        # Update the existing row with the new dataset information
        df_summary.loc[df_summary['datasetName'] == new_dataset_name, 'num_features'] = num_features
    else:
        # Create a new row with the new dataset information
        new_row = pd.DataFrame({
            'datasetName': [new_dataset_name],
            'distribution': df_summary.loc[df_summary['datasetName'] == dataset_name, 'distribution'].values,
            'x1': df_summary.loc[df_summary['datasetName'] == dataset_name, 'x1'].values,
            'y1': df_summary.loc[df_summary['datasetName'] == dataset_name, 'y1'].values,
            'x2': df_summary.loc[df_summary['datasetName'] == dataset_name, 'x2'].values,
            'y2': df_summary.loc[df_summary['datasetName'] == dataset_name, 'y2'].values,
            'num_features': [num_features],
            'size': df_summary.loc[df_summary['datasetName'] == dataset_name, 'size'].values,
            'num_points': df_summary.loc[df_summary['datasetName'] == dataset_name, 'num_points'].values,
            'avg_area': df_summary.loc[df_summary['datasetName'] == dataset_name, 'avg_area'].values,
            'avg_side_length_0': df_summary.loc[df_summary['datasetName'] == dataset_name, 'avg_side_length_0'].values,
            'avg_side_length_1': df_summary.loc[df_summary['datasetName'] == dataset_name, 'avg_side_length_1'].values,
            'E0': df_summary.loc[df_summary['datasetName'] == dataset_name, 'E0'].values,
            'E2': df_summary.loc[df_summary['datasetName'] == dataset_name, 'E2'].values
        })

        # Concatenate the new row with the existing DataFrame
        df_summary = pd.concat([df_summary, new_row], ignore_index=True)

    # Write the updated DataFrame back to the dataset summaries file
    df_summary.to_csv('dataset-summaries_ts.csv', index=False, sep=';')
    df_summary.to_csv('new_datasets.csv', index=False, sep=';')


def update_range_query_file(index, rotated_dataset_name, minX, minY, maxX, maxY):
    # Read the existing range query file
    df_range_query = pd.read_csv('rq_result_ts.csv', delimiter=';')

    # Retrieve the row at the specified index
    original_row = df_range_query.iloc[index]

    # Create a new DataFrame with the rotated dataset information
    new_row = {
        'datasetName': rotated_dataset_name,
        'numQuery': original_row['numQuery'],
        'queryArea': original_row['queryArea'],
        'minX': minX,
        'minY': minY,
        'maxX': maxX,
        'maxY': maxY,
        'areaint': original_row['areaint'],
        'cardinality': original_row['cardinality'],
        'executionTime': original_row['executionTime'],
        'mbrTests': original_row['mbrTests'],
        'cardinality_class': original_row['cardinality_class']
    }

    # Concatenate the new row with the existing DataFrame
    df_range_query = pd.concat([df_range_query, pd.DataFrame([new_row])], ignore_index=True)

    # Write the updated DataFrame back to the dataset summaries file
    df_range_query.to_csv('rq_result_ts.csv', index=False, sep=';')
    
def update_range_query_file_label(index, rotated_dataset_name, minX, minY, maxX, maxY, label):
    # Read the existing range query file
    df_range_query = pd.read_csv('rq_result_ts.csv', delimiter=';')

    # Retrieve the row at the specified index
    original_row = df_range_query.iloc[index]
    
    # List of potential last column names
    potential_last_columns = ['cardinality_class', 'mbrTests', 'elapsedTime']

    # Find the actual last column name that exists in original_row
    last_column_name = next((col for col in potential_last_columns if col in original_row), None)

    # Raise an error if no valid last column name is found
    if last_column_name is None:
        raise ValueError("None of the potential last column names were found in original_row")

    # Create a new DataFrame with the rotated dataset information
    new_row = {
        'datasetName': rotated_dataset_name,
        'numQuery': original_row['numQuery'],
        'queryArea': original_row['queryArea'],
        'minX': minX,
        'minY': minY,
        'maxX': maxX,
        'maxY': maxY,
        'areaint': original_row['areaint'],
        'cardinality': original_row['cardinality'],
        'executionTime': original_row['executionTime'],
        'mbrTests': original_row['mbrTests'],
        last_column_name: label 
    }

    # Concatenate the new row with the existing DataFrame
    df_range_query = pd.concat([df_range_query, pd.DataFrame([new_row])], ignore_index=True)

    # Write the updated DataFrame back to the dataset summaries file
    df_range_query.to_csv('rq_result_ts.csv', index=False, sep=';')    
    
    
def update_range_query_param(index, new_dataset_name, cardinality):
    # Read the existing range query file
    df_range_query = pd.read_csv('rq_result_ts.csv', delimiter=';')

    # Retrieve the row at the specified index
    original_row = df_range_query.iloc[index]

    # Create a new DataFrame with the new dataset information
    new_row = {
        'datasetName': new_dataset_name,
        'numQuery': original_row['numQuery'],
        'queryArea': original_row['queryArea'],
        'minX': original_row['minX'],
        'minY': original_row['minY'],
        'maxX': original_row['maxX'],
        'maxY': original_row['maxY'],
        'areaint': original_row['areaint'],
        'cardinality': cardinality,
        'executionTime': original_row['executionTime'],
        'mbrTests': original_row['mbrTests'],
        'cardinality_class': original_row['cardinality_class']
    }

    # Concatenate the new row with the existing DataFrame
    df_range_query = pd.concat([df_range_query, pd.DataFrame([new_row])], ignore_index=True)

    # Write the updated DataFrame back to the range query file
    df_range_query.to_csv('rq_result_ts.csv', index=False, sep=';')

def update_range_query_param2(index, new_dataset_name, cardinality, label):
    # Read the existing range query file
    df_range_query = pd.read_csv('rq_result_ts.csv', delimiter=';')

    # Retrieve the row at the specified index
    original_row = df_range_query.iloc[index]

    # List of potential last column names
    potential_last_columns = ['cardinality_class', 'mbrTests', 'elapsedTime']

    # Find the actual last column name that exists in original_row
    last_column_name = next((col for col in potential_last_columns if col in original_row), None)

    # Raise an error if no valid last column name is found
    if last_column_name is None:
        raise ValueError("None of the potential last column names were found in original_row")

    # Create the new row with the identified last column
    new_row = {
        'datasetName': new_dataset_name,
        'numQuery': original_row['numQuery'],
        'queryArea': original_row['queryArea'],
        'minX': original_row['minX'],
        'minY': original_row['minY'],
        'maxX': original_row['maxX'],
        'maxY': original_row['maxY'],
        'areaint': original_row['areaint'],
        'cardinality': cardinality,
        'executionTime': original_row['executionTime'],
        'mbrTests': original_row['mbrTests'],
        last_column_name: label 
    }

    # Concatenate the new row with the existing DataFrame
    df_range_query = pd.concat([df_range_query, pd.DataFrame([new_row])], ignore_index=True)

    # Write the updated DataFrame back to the range query file
    df_range_query.to_csv('rq_result_ts.csv', index=False, sep=';')


def get_next_filename(dataset_name):
    counter = 1
    while True:
        filename = f"{dataset_name}_noise_{counter:04d}.csv"
        if not os.path.exists(filename):
            return filename
        counter += 1

def are_disjoint(mbr1, mbr2):
    # Unpack the coordinates
    xmin1, ymin1, xmax1, ymax1 = mbr1
    xmin2, ymin2, xmax2, ymax2 = mbr2
    
    # Check for disjoint conditions
    if xmax1 < xmin2:  # mbr1 is to the left of mbr2
        return True
    if xmin1 > xmax2:  # mbr1 is to the right of mbr2
        return True
    if ymax1 < ymin2:  # mbr1 is below mbr2
        return True
    if ymin1 > ymax2:  # mbr1 is above mbr2
        return True
    
    # If none of the conditions are met, they are not disjoint
    return False


# Function to check if two rectangles are disjoint
def are_disjoint2(row, minx, miny, maxx, maxy):
    return row['xmax'] < minx or row['xmin'] > maxx or row['ymax'] < miny or row['ymin'] > maxy

def format_number(num):
    # Format the number to a fixed number of decimal places (e.g., 10 decimal places)
    return f"{num:.10f}"

def create_intervals(min_val, max_val, intervals):
    bins = [min_val] + intervals + [max_val]
    labels = [f"{format_number(bins[i])}-{format_number(bins[i+1])}" for i in range(len(bins)-1)]
    return bins, labels

def summarize_bin_content(data, param, bins, labels):
    summary = {}
    for label in labels:
        bin_data = data[data[f'{param}_class'] == label]
        if bin_data.empty:
            summary[label] = "Empty bin"
        else:
            dist_counts = bin_data['distribution'].value_counts().to_dict()
            summary[label] = dist_counts
    return summary
    
def count_lines_efficient(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for line in f)







def main():
    # Read the rq_result.csv file
    rq_result = pd.read_csv('rq_result_ts.csv', delimiter=';')
    last_column = rq_result.columns[-1]
    
    if not last_column.endswith('_class'):
        raise ValueError("The last column of rq_result_ts.csv does not match the expected format.")
    
    # Extract the parameter to categorize from the column name
    param_to_categorize = last_column.rsplit('_', 1)[0]
    valid_params = ['cardinality', 'executionTime', 'mbrTests']
    if param_to_categorize not in valid_params:
        raise ValueError(f"Parameter to categorize must be one of {valid_params}")
    
    # Extract all label values from the last column
    label_values = rq_result[last_column].unique()
    
    # Extract all single values that compose a label
    all_values = set()
    for label in label_values:
        start, end = label.split('-')
        all_values.update([float(start), float(end)])
    
    all_values = sorted(all_values)
    
    # Define min, max and intervals
    min_val = all_values[0]
    max_val = all_values[-1]
    intervals = all_values[1:-1]

    # Define min_values, max_values and interval_points dynamically
    min_values = {param_to_categorize: min_val}
    max_values = {param_to_categorize: max_val}
    interval_points = {param_to_categorize: intervals}
    
    bins, labels = create_intervals(min_val, max_val, intervals)
    
    main_data = pd.read_csv('rq_result_ts.csv', delimiter=';')
    summary_data = pd.read_csv('dataset-summaries_ts.csv', delimiter=';')
    
    # Check if 'distribution' column is already present
    if 'distribution' not in main_data.columns:
        if 'distribution' not in summary_data.columns:
            raise KeyError("'distribution' column is missing in the summary data.")
        main_data = main_data.merge(summary_data[['datasetName', 'distribution']], on='datasetName', how='left')
    
    main_data[f'{param_to_categorize}_class'] = pd.cut(main_data[param_to_categorize], bins=bins, labels=labels, include_lowest=True)
    
    distribution_counts = main_data['distribution'].value_counts()
    print("Distribution Counts:")
    print(distribution_counts)
    
    print(f"\n{param_to_categorize.capitalize()} Counts:")
    bin_summary = summarize_bin_content(main_data, param_to_categorize, bins, labels)
    counts = main_data[f'{param_to_categorize}_class'].value_counts().sort_index()
    
    # Ensure all bins are included, even if they are empty
    for label in labels:
        count = counts.get(label, 0)
        print(f"{label}    {count}")
        if bin_summary[label] == "Empty bin":
            print("- empty bin")
        else:
            for dist, dist_count in bin_summary[label].items():
                print(f"- {dist}: {dist_count}")
    
    print("\nBin Associations:")
    for index, label in enumerate(labels):
        print(f"bin{index} -- {label}")
        
        
    inputs = []
    print("\nEnter your queries in the format 'bin_num num_queries distribution augmentation_technique1 [augmentation_technique2] [augmentation_technique3]'. Type 'end' to finish.")
    while True:
        user_input = input("Input: ")
        if user_input.lower() == 'end':
            break
        try:
            validated_input = validate_input(user_input)
            inputs.append(validated_input)
        except ValueError as e:
            print(f"Invalid input: {e}")

    print("\nProcessing user inputs:")
    for bin_num, num_queries, distribution, augmentation_techniques in inputs:
        bin_index = int(bin_num[3:])
        bin_label = labels[bin_index]
        bin_data = main_data[main_data[f'{param_to_categorize}_class'] == bin_label]
        dist_data = bin_data[bin_data['distribution'] == distribution]

        if not dist_data.empty:
            if len(dist_data) > num_queries:
                # Case 1: More than required queries
                to_keep = dist_data.head(num_queries)
                to_remove = dist_data.tail(len(dist_data) - num_queries)
                main_data = main_data.drop(to_remove.index)

                # Remove datasets only if they are not used in other queries
                datasets_to_remove = to_remove['datasetName'].unique()
                remaining_datasets = main_data['datasetName'].unique()
                datasets_to_remove = [ds for ds in datasets_to_remove if ds not in remaining_datasets]

                summary_data = summary_data[~summary_data['datasetName'].isin(datasets_to_remove)]
                print(f"Kept {num_queries} queries with distribution {distribution} in bin {bin_index}")

                # Save the modified data back to the CSV files
                main_data.to_csv('rq_result_ts.csv', index=False, sep=';')
                summary_data.to_csv('dataset-summaries_ts.csv', index=False, sep=';')
            elif len(dist_data) < num_queries:
                # Case 2: Less than required queries
                print(f"Bin {bin_index} needs augmentation to reach {num_queries} queries with distribution {distribution}. Techniques: {augmentation_techniques}")
                with open('new_datasets.csv', 'w') as file:
                    pass  # The `pass` statement means do nothing, effectively creating an empty file

                # Get distribution name from dist_data
                distribution_name = dist_data['distribution'].iloc[0]

                # Get count of the specific distribution
                named_distribution_count = bin_data[bin_data['distribution'] == distribution].shape[0]

                # Calculate the number of additional queries needed
                num_queries = num_queries - named_distribution_count

                # Get the number of rows in dist_data after filtering
                num_rows = len(dist_data)

                # Initialize the index of the current row
                num_queries_inserted = 0
                
                
                while True:
                
                    # Define the folder name and the output file name
                    folder_name = "datasets_augmentation"

                    # Create the folder if it does not exist
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                        
                    # Fetch a row by its row number
                    row = dist_data.sample(n=1).iloc[0]
                    file_index = row.name
                    dataset_name = row['datasetName']
                    
                                       
                    if param_to_categorize == "cardinality":
                        element_to_pop = "merge"  

                        for i in range(3, len(augmentation_techniques)):
                          if augmentation_techniques[i] == element_to_pop:
                            element = augmentation_techniques.pop(i)
                            break  # Exit the loop after finding the element
                        
                    chosen_technique = random.choice(augmentation_techniques)
                    print("AEEE") 
                    print(chosen_technique)
                    print("AEEE")                    
                    
                    # Depending on the chosen technique, execute the corresponding code block
                    if chosen_technique == 'rotation':
                        try:
                            degree = generate_random_degree()

                            # Formatting output degree
                            degree_str = str(degree).replace('.', '_')
                            output_ds = f"datasets_augmentation/{dataset_name}_rotated_{degree_str}.csv"
                            rotated_dataset_name = f"{dataset_name}_rotated_{degree_str}"

                            if os.path.exists(output_ds): # No rotation required 
                                print(f"Skipped: Rotated dataset file '{rotated_dataset_name}' already exists.")
                            else:
                                # Rotate the dataset
                                nrem = rotate_dataset(dataset_name, output_ds, degree)

                                # Get the coordinates from the dataset summaries file
                                x1, y1, x2, y2 = get_coordinates('dataset-summaries_ts.csv', dataset_name)

                                # Get number of features from dataset summaries file
                                nf = get_features('dataset-summaries_ts.csv', dataset_name)

                                res = float(nf) - float(nrem)
                                print(f"Figures removed: '{nrem}'")

                                # Rotate the mbr of the dataset
                                rotated_query_window = rotate_query_window(dataset_name, x1, y1, x2, y2, degree)

                                # Extract the rotated coordinates
                                rotated_x1, rotated_y1, rotated_x2, rotated_y2 = map(float, rotated_query_window.split(','))

                                # Update the dataset summaries file with the rotated mbr values
                                update_dataset_summary(dataset_name, rotated_dataset_name, rotated_x1, rotated_y1, rotated_x2, rotated_y2, res)

                            # Get the coordinates for range query
                            minX, minY, maxX, maxY = get_coordinates_rq('rq_result_ts.csv', file_index)

                            # Rotate the range query coordinates
                            rotated_query_window_rq = rotate_query_window(dataset_name, minX, minY, maxX, maxY, degree)

                            # Extract the rotated coordinates for range query
                            rotated_x1r, rotated_y1r, rotated_x2r, rotated_y2r = map(float, rotated_query_window_rq.split(','))

                            if param_to_categorize == "cardinality":
                                # Update range query file with the new query
                                update_range_query_file(file_index, rotated_dataset_name, rotated_x1r, rotated_y1r, rotated_x2r, rotated_y2r)
                            else:
                                update_range_query_file_label(file_index, rotated_dataset_name, rotated_x1r, rotated_y1r, rotated_x2r, rotated_y2r, bin_label)                    

                        except Exception as e:
                            print(f"An error occurred: {e}") 
                            num_queries_inserted -= 1                  
                        
                    elif chosen_technique == 'noise':	                                                                                                ###### NOISE CODE ###################
                        try:                       
                            # Fetch a row with the same distribution but different cardinality_class and selected_cardinality != 0.0
                            if param_to_categorize == "cardinality": 
                                noise_data = main_data[
                                    (main_data['distribution'] == distribution) & 
                                    (main_data['cardinality_class'] != bin_label) & 
                                    (main_data['cardinality'] != 0.0) &
                                    (main_data['datasetName'].str.len() == 12)
                                ]                        
                            else:
                                noise_data = main_data[
                                    (main_data['distribution'] == distribution) &
                                    (main_data['cardinality'] != 0.0) & 
                                    (main_data['datasetName'].str.len() == 12)
                                ]
                            
                
                            if not noise_data.empty:
                                selected_row = noise_data.sample(n=1).iloc[0]
                                file_index = selected_row.name
                                selected_cardinality = selected_row['cardinality']
                                

                                # Extract the datasetName correctly
                                dataset_name = selected_row['datasetName']
                                # Load the CSV file into a DataFrame
                                file_path = f"../index/{dataset_name}_spatial_index/_master.rsgrove"  
                                while True:
                                    try:
                                        df = pd.read_csv(file_path)
                                        break  # If the file is successfully loaded, exit the loop
                                    except FileNotFoundError:
                                        # If the file is not found, try another name
                                        file_path = f"../index/{dataset_name}_spatial_index/_master.cells"
                                        if not os.path.exists(file_path):
                                            # If the alternative file path doesn't exist, break the loop
                                            print("File not found. Exiting.")
                                            break  # Break out of the inner loop
                                
                                #_master.cells 
                                df = pd.read_csv(file_path, delim_whitespace=True, skiprows=0)

                                # Extract the xmin, ymin, xmax, ymax columns
                                coordinates_df = df[['xmin', 'ymin', 'xmax', 'ymax']]
                                                                                    
                                minX = selected_row['minX']
                                minY = selected_row['minY']
                                maxX = selected_row['maxX']
                                maxY = selected_row['maxY']
                                
                                disjoint_df = coordinates_df[coordinates_df.apply(are_disjoint2, axis=1, args=(minX, minY, maxX, maxY))]
                                
                                input_dataset = f"../datasets/{dataset_name}.csv"
                                
                                max_index = 0  # Initialize max_index

                                for file_name_d in os.listdir("datasets_augmentation"):
                                    if file_name_d.startswith(selected_row['datasetName']):
                                        try:
                                            index = int(file_name_d.split("_noise_")[-1].split(".")[0])
                                            max_index = max(max_index, index)
                                        except ValueError:
                                            pass

                                output_dataset = f"{dataset_name}_noise_{max_index + 1}.csv"
                                
                                # Read content from input file
                                with open(input_dataset, 'r') as input_file:
                                    content = input_file.read()
                                output_path = os.path.join(folder_name, output_dataset)
                                # Write content to output file
                                with open(output_path, 'w') as output_file:
                                    output_file.write(content)
                                
                                print(output_path)   
                                    
                                                        
                                if param_to_categorize == "cardinality":
                                    if selected_cardinality > bin_data['cardinality'].iloc[0]:
                                        operation = "decrease"
                                    else:
                                        operation = "increase"
                                
                                b, h, num_features = get_dataset_info('dataset-summaries_ts.csv', dataset_name)
                                
                                
                                
                                try: 
                                    # Get the coordinates from the dataset summaries file
                                    x1, y1, x2, y2 = get_coordinates('dataset-summaries_ts.csv', dataset_name)
                                    coordinates_summary = (x1, y1, x2, y2)

                                    # Get the coordinates for range query
                                    coordinates_rq = (minX, minY, maxX, maxY)
                                    
                                    
                                    
                                    if param_to_categorize == "cardinality":
                                        print(f"Noised dataset by {operation} geometries: {output_path}")
                                    
                                        # Get bin_label from the current row
                                        bin_label = row['cardinality_class']
                                        print("Bin Label:", bin_label)
                                        lower_bound, upper_bound = bin_label.split('-')
                                        
                                        
                                        
                                        if operation == "decrease":
                                            count_geom_tot = count_lines_efficient(output_path)
            
                                            count_inside = count_geom_tot * selected_cardinality      
                                            while float(selected_cardinality) > float(upper_bound):
                                                random_row_df = disjoint_df.sample(n=1) 
                                                new_box = generate_random_box(random_row_df, b, h)
                                                with open(output_path, 'r+') as file:
                                                    file.write(','.join(map(str, new_box)) + '\n')
                                                count_geom_tot += 1
                                                selected_cardinality = count_inside / count_geom_tot
                                                print(f"{selected_cardinality},{count_geom_tot}") 
                                            print(f"{selected_cardinality},{count_geom_tot}")

                                        else:  # increase
                                            with open(output_path, 'r+') as file:
                                                count_geom_tot = count_lines_efficient(output_path)
                                                count_inside = count_geom_tot * selected_cardinality

                                                while float(selected_cardinality) < float(lower_bound):
                                                    random_row_df = disjoint_df.sample(n=1)
                                                    remove_geometry(file, coordinates_rq)
                                                    count_geom_tot -= 1
                                                    selected_cardinality = count_inside / count_geom_tot
                                                    print(f"{selected_cardinality},{count_geom_tot}")
                                            print(f"{selected_cardinality},{count_geom_tot}")

                                    else:
                                        ops = ["increase", "decrease"]
                                        operation = random.choice(ops)
                                        print(f"Noised dataset by {operation} geometries: {output_path}")
                                        random_percentage = random.uniform(5, 20)
                                        number = int(num_features * (random_percentage / 100))
                                        print(f"{random_percentage},{number}")
                                    
                                        
                                        if operation == "decrease":
                                            count_geom_tot = count_lines_efficient(output_path)
            
                                            count_inside = count_geom_tot * selected_cardinality      
                                            while number > 0:
                                                random_row_df = disjoint_df.sample(n=1) 
                                                new_box = generate_random_box(random_row_df, b, h)
                                                with open(output_path, 'r+') as file:
                                                    file.write(','.join(map(str, new_box)) + '\n')
                                                count_geom_tot += 1
                                                selected_cardinality = count_inside / count_geom_tot 
                                                number = number - 1
                                                print(f"{selected_cardinality},{count_geom_tot}, {number}")
                                            print(f"{selected_cardinality},{count_geom_tot}")

                                        else:  # increase
                                            count_geom_tot = count_lines_efficient(output_path)
                                            count_inside = count_geom_tot * selected_cardinality      

                                            while number > 0:
                                                with open(output_path, 'r+') as file:
                                                    random_row_df = disjoint_df.sample(n=1)
                                                    remove_geometry(file, coordinates_rq)
                                                    count_geom_tot -= 1
                                                    selected_cardinality = count_inside / count_geom_tot
                                                    number -= 1
                                                    print(f"{selected_cardinality},{count_geom_tot}, {number}")
                                            print(f"{selected_cardinality},{count_geom_tot}")
                                        
                                    new_dataset_name = str(output_dataset).replace('.csv', '')
                                    
                                    
                                    update_dataset_param(dataset_name, new_dataset_name, count_geom_tot)
                                    if param_to_categorize == "cardinality":
                                        update_range_query_param(file_index, new_dataset_name, selected_cardinality)
                                    else:
                                        update_range_query_param2(file_index, new_dataset_name, selected_cardinality, bin_label)
                                except subprocess.CalledProcessError as e:
                                    print(f"Error applying noise to dataset '{output_path}': {e}")
                            
                            else:
                                print("No appropriate noise data found.")
                        except Exception as e:
                            print(f"An error occurred: {e}")    
                            num_queries_inserted -= 1
                    
                    elif chosen_technique == 'merge':
                        try:												####### MERGE CODE #######
                            
                            print("Collecting data")
                            # Code block for merge augmentation
                            merge_data = main_data
                        
                            # Group by 'datasetName'
                            grouped = merge_data.groupby('datasetName')

                            # Select one random row from each group
                            sampled_groups = grouped.apply(lambda x: x.sample(1)).reset_index(drop=True)

                            # Check if there are at least two different dataset names
                            if len(sampled_groups) < 2:
                                raise ValueError("Not enough unique dataset names to sample from.")
                                
                            print("Fetching datasets...")

                            while True:
                                while True:
                                    # Select 2 random rows with different 'datasetName'
                                    random_rows = sampled_groups.sample(n=2, replace=False)

                                    # Extract the datasetName from the 2 random rows
                                    dataset_names = random_rows['datasetName'].tolist()

                                    # Select rows from summary_data where datasetName is equal to the extracted datasetNames
                                    summary_data_rows = summary_data[summary_data['datasetName'].isin(dataset_names)]

                                    # Extract x1, y1, x2, and y2 from the summary_data rows for both datasetNames
                                    row1 = summary_data_rows.iloc[0][['x1', 'y1', 'x2', 'y2']].tolist()
                                    row2 = summary_data_rows.iloc[1][['x1', 'y1', 'x2', 'y2']].tolist()
                        
                                    if are_disjoint(row1, row2):
                                        print("Disjoint found")
                                        break
                        
                                # Dataset merge
                                name_d1 = summary_data_rows.iloc[0]['datasetName']
                                name_d2 = summary_data_rows.iloc[1]['datasetName']
                            
                                nf_d1 = summary_data_rows.iloc[0]['num_features']
                                nf_d2 = summary_data_rows.iloc[1]['num_features']
                                sum_nf = nf_d1 + nf_d2
                        
                                card_d1 = random_rows.iloc[0]['cardinality']
                                card_d2 = random_rows.iloc[1]['cardinality']
                        
                                #card = inside/tot -> inside = card * tot
                                inside_d1 = card_d1 * nf_d1
                                inside_d2 = card_d2 * nf_d2
                        
                                # Select a random row from random_rows
                                random_row = random_rows.sample(n=1).iloc[0]
                        
                                if random_row.equals(random_rows.iloc[0]):
                                    card = random_rows.iloc[0]['cardinality']
                                    inside = card_d1 * nf_d1
                        
                                else:
                                    card = random_rows.iloc[1]['cardinality']
                                    inside = card_d2 * nf_d2
                        
                                new_card = inside/sum_nf
                        
                                if param_to_categorize == "cardinality": 
                                    bin_label = row['cardinality_class']
                                    print("Bin Label:", bin_label)
                                    # Split the bin_label string at the dash
                                    bounds = bin_label.split('-')

                                    # Convert the split strings to floats
                                    lower_bound = float(bounds[0])
                                    upper_bound = float(bounds[1])
                            
                                    if lower_bound <= new_card <= upper_bound:
                                        break
                                    else:
                                        print("New cardinality was not in the bin. Nuovo tentativo in corso...")
                                        continue
                        
                                else: 
                                    break     
                        
                            print("Combination found, proceding to merge")
                            # Updating stuff
                            # Load dataset1.csv and dataset2.csv 
                            in1 = f"../datasets/{name_d1}.csv"
                            in2 = f"../datasets/{name_d2}.csv"
                            dataset1 = pd.read_csv(in1)
                            dataset2 = pd.read_csv(in2)
                    
                            # Extract numbers from dataset names
                            n1, n2 = extract_numbers(name_d1), extract_numbers(name_d2)

                            # Compose the combined dataset filename
                            combined_filename = f"{folder_name}/dataset_{n1[0]}_{n2[0]}_combined.csv"
                            comb = f"dataset_{n1[0]}_{n2[0]}_combined"

                            # Concatenate the two datasets
                            combined_dataset = pd.concat([dataset1, dataset2], ignore_index=True)

                            # Write the combined dataset to a new file
                            combined_dataset.to_csv(combined_filename, index=False, header=True)
                    
                            update_dataset_param(random_row['datasetName'], comb, sum_nf)
                            if param_to_categorize == "cardinality":
                                update_range_query_param(file_index, comb, card)
                            else:
                                update_range_query_param2(file_index, comb, new_card, bin_label)
                            
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            num_queries_inserted -= 1
                    
                    # Increment the count of new queries inserted
                    num_queries_inserted += 1
                    
                    # Check if all rows have been processed
                    if num_queries_inserted >= num_queries:
                        break  

                    print("\n")
            print("\n")
            remove_distribution_column("rq_result_ts.csv")
            print(f"Augmented queries with {augmentation_techniques} techniques.")

        elif len(dist_data) == num_queries:
            print(f"Bin {bin_index} already has {num_queries} queries with distribution {distribution}")
        else:
            # Case 3: No queries with the specified distribution
            print(f"Bin {bin_index} does not contain any query with distribution {distribution}")

if __name__ == "__main__":
    main()
