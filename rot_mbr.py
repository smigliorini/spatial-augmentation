import pandas as pd
import numpy as np
import math
import time
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

def rotate_boxes(xmin, ymin, xmax, ymax, angle_degrees):
    """Rotate bounding boxes around the center by a given angle in degrees."""
    angle_radians = math.radians(angle_degrees)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    # Get all four corners
    points = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ])

    # Rotate all points
    rotated_points = [rotate_points(cx, cy, angle_radians, points[i][0], points[i][1]) for i in range(4)]
    
    # Stack and calculate new bounding boxes
    all_x_coords = np.vstack([rotated_points[i][0] for i in range(4)])
    all_y_coords = np.vstack([rotated_points[i][1] for i in range(4)])

    new_xmin = np.min(all_x_coords)
    new_ymin = np.min(all_y_coords)
    new_xmax = np.max(all_x_coords)
    new_ymax = np.max(all_y_coords)

    return new_xmin, new_ymin, new_xmax, new_ymax

def main():
    if len(sys.argv) != 6:
        print("Usage: python3 rot_mbr.py xmin ymin xmax ymax degree")
        sys.exit(1)

    xmin, ymin, xmax, ymax, angle_degrees = map(float, sys.argv[1:])
    
    new_minX, new_minY, new_maxX, new_maxY = rotate_boxes(xmin, ymin, xmax, ymax, angle_degrees)
    print(f"{new_minX}, {new_minY}, {new_maxX}, {new_maxY}")

if __name__ == "__main__":
    main()

