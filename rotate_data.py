import pandas as pd
import numpy as np
import math
import csv
import sys

def rotate_points(cx, cy, angle, px, py):
    """Rotate points around a center (cx, cy) by a given angle in radians."""
    s = np.sin(angle)
    c = np.cos(angle)

    # Translate points back to origin
    px -= cx
    py -= cy

    # Rotate points
    xnew = px * c - py * s
    ynew = px * s + py * c

    # Translate points back
    px = xnew + cx
    py = ynew + cy
    return px, py

def rotate_boxes(df, cx, cy, angle):
    """Rotate bounding boxes around the center (cx, cy) by a given angle in radians."""
    # Extract coordinates
    xmin = df['xmin'].values
    ymin = df['ymin'].values
    xmax = df['xmax'].values
    ymax = df['ymax'].values

    # Get all four corners
    points = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ])

    # Rotate all points
    rotated_points = [rotate_points(cx, cy, angle, points[i][0], points[i][1]) for i in range(4)]

    # Stack and calculate new bounding boxes
    all_x_coords = np.vstack([rotated_points[i][0] for i in range(4)])
    all_y_coords = np.vstack([rotated_points[i][1] for i in range(4)])

    new_xmin = np.min(all_x_coords, axis=0)
    new_ymin = np.min(all_y_coords, axis=0)
    new_xmax = np.max(all_x_coords, axis=0)
    new_ymax = np.max(all_y_coords, axis=0)

    return new_xmin, new_ymin, new_xmax, new_ymax

def process_csv(input_csv, output_csv, angle_degrees, space_bounds=(0, 0, 10, 10)):
    angle_radians = math.radians(angle_degrees)
    cx, cy = (space_bounds[2] - space_bounds[0]) / 2, (space_bounds[3] - space_bounds[1]) / 2

    # Read the CSV file without a header
    df = pd.read_csv(input_csv, header=None, names=['xmin', 'ymin', 'xmax', 'ymax'])

    # Rotate all bounding boxes
    new_xmin, new_ymin, new_xmax, new_ymax = rotate_boxes(df, cx, cy, angle_radians)

    # Create a new DataFrame with rotated bounding boxes
    rotated_df = pd.DataFrame({
        'xmin': new_xmin,
        'ymin': new_ymin,
        'xmax': new_xmax,
        'ymax': new_ymax
    })

    # Filter out bounding boxes that are not fully inside the space bounds
    inside_bounds = (
        (rotated_df['xmin'] >= space_bounds[0]) &
        (rotated_df['ymin'] >= space_bounds[1]) &
        (rotated_df['xmax'] <= space_bounds[2]) &
        (rotated_df['ymax'] <= space_bounds[3])
    )
    filtered_df = rotated_df[inside_bounds]
    removed_count = len(rotated_df) - len(filtered_df)

    # Save to CSV without quoting
    filtered_df.to_csv(output_csv, index=False, header=None, quoting=csv.QUOTE_NONE)

    print(f"{removed_count}")

    return df, filtered_df

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 rotate.py input_csv output_csv angle_degrees")
        return

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    angle_degrees = float(sys.argv[3])

    df, rotated_df = process_csv(input_csv, output_csv, angle_degrees)

if __name__ == "__main__":
    main()

