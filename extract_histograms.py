import os
import numpy as np
import argparse
import datetime

def process_input_file(input_filename, x1, y1, x2, y2, num_rows, num_columns):
    hist = np.zeros((num_rows, num_columns))
    box_data = {}

    d1 = (x2 - x1) / num_columns
    d2 = (y2 - y1) / num_rows

    with open(input_filename, 'r') as input_f:
        for line in input_f:
            data = line.strip().split(',')
            xmin = float(data[0])
            ymin = float(data[1])
            xmax = float(data[2])
            ymax = float(data[3])
            x_centroid = (xmin + xmax) / 2
            y_centroid = (ymin + ymax) / 2
            col = int((x_centroid - x1) / d1)
            row = int((y_centroid - y1) / d2)

            if 0 <= row < num_rows and 0 <= col < num_columns:
                hist[row, col] += 1
                if (row, col) not in box_data:
                    box_data[(row, col)] = []
                box_data[(row, col)].append((xmin, ymin, xmax, ymax))

    return hist, box_data

def calculate_bin_metrics(box_data):
    output_data = []
    for (row, col), boxes in box_data.items():
        num_features = len(boxes)
        size = num_features * 16
        num_points = num_features * 4
        total_area = 0
        total_side_length_0 = 0
        total_side_length_1 = 0

        for xmin, ymin, xmax, ymax in boxes:
            area = (xmax - xmin) * (ymax - ymin)
            total_area += area
            total_side_length_0 += (xmax - xmin)
            total_side_length_1 += (ymax - ymin)

        avg_area = total_area / num_features
        avg_side_length_0 = total_side_length_0 / num_features
        avg_side_length_1 = total_side_length_1 / num_features

        output_data.append((row, col, num_features, size, num_points, avg_area, avg_side_length_0, avg_side_length_1))
    return output_data

def extract_histogram(input_filename, output_filename, num_rows, num_columns):
    # Using fixed values for x1, y1, x2, y2
    x1, y1, x2, y2 = 0, 0, 10, 10
    hist, box_data = process_input_file(input_filename, x1, y1, x2, y2, num_rows, num_columns)
    output_data = calculate_bin_metrics(box_data)

    with open(output_filename, 'w') as output_f:
        output_f.write('i0,i1,num_features,size,num_points,avg_area,avg_side_length_0,avg_side_length_1\n')
        for row, col, num_features, size, num_points, avg_area, avg_side_length_0, avg_side_length_1 in output_data:
            output_f.write(f'{row},{col},{num_features},{size},{num_points},{avg_area},{avg_side_length_0},{avg_side_length_1}\n')

def extract_histograms(single_file=None):
    histogram_dirs = ['128x128']
    csv_files = [single_file] if single_file else [file for file in os.listdir() if file.endswith('.csv') and file != 'dataset-summaries_ts.csv' and file != 'rq_result_ts.csv' and file != 'fd2_geom_allds_ts.csv' and file != 'new_datasets.csv']
    input_dir = './datasets_augmentation'

    # Original folder name
    output_dir = "../output_hist_new"

    # Get current timestamp with specific format (YYYY-MM-DD_HH-MM-SS)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Combine original name with underscore and timestamp
    modified_dir = f"{output_dir}_{timestamp}
    os.makedirs(output_dir, exist_ok=True)

    for histogram_dir in histogram_dirs:
        num_rows, num_columns = map(int, histogram_dir.split('x'))
        for filename in csv_files:
            output_filename = os.path.join(output_dir, filename.replace('.csv', '_summary.csv'))
            extract_histogram(os.path.join(input_dir, filename), output_filename, num_rows, num_columns)

def main():
    parser = argparse.ArgumentParser(description='Process histogram extraction.')
    parser.add_argument('--file', type=str, help='Single CSV file to process.')
    args = parser.parse_args()

    extract_histograms(single_file=args.file)

if __name__ == '__main__':
    main()

