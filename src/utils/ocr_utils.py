"""OCR utils."""

import rootutils

ROOT = rootutils.autosetup()

from typing import Tuple

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def get_mini_boxes(contour: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Get mini boxes from contour.

    Args:
        contour (np.ndarray): Contour.

    Returns:
        Tuple(np.ndarray, int): Bounding box and min side length.
    """
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0

    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]

    return np.array(box, dtype=np.int32), min(bounding_box[1])


def unclip(box: np.ndarray, ratio: float = 2.0) -> np.ndarray:
    """
    Unclip or enlarge box.

    Args:
        box (np.ndarray): Bounding box.

    Returns:
        np.ndarray: Unclipped box.
    """
    poly = Polygon(box)
    distance = poly.area * ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    return np.array(offset.Execute(distance))
