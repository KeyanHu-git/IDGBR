import os

import cv2
import numpy as np
from tqdm import tqdm


def extract_edge_map(label_index, dilate_kernel=None):
    label_index_uint8 = np.uint8(label_index)

    sobel_x = cv2.Sobel(label_index_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(label_index_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    edge_map = np.uint8(gradient_magnitude > 0)

    if dilate_kernel is not None:
        result = cv2.dilate(edge_map, dilate_kernel, iterations=1)
    else:
        result = edge_map

    result_label = np.copy(label_index)
    result_label[result == 0] = 255
    return result_label


def extract_edge_map_path(input_dir, edge_size=3, output_dir=None, overwrite=False):
    if output_dir is None:
        output_dir = input_dir + f"_edge_map{edge_size}"

    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if not (filename.endswith(".tif") or filename.endswith(".png")):
            continue
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path) and not overwrite:
            continue
        label_index = cv2.imread(os.path.join(input_dir, filename), -1)
        edge_map = extract_edge_map(label_index, dilate_kernel=np.ones((edge_size, edge_size), np.uint8))
        cv2.imwrite(output_path, edge_map)
