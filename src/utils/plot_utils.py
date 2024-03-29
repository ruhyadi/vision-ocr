"""Plotting utils."""

import rootutils

ROOT = rootutils.autosetup()

import base64
import io
from typing import List

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def draw_ocr(
    img: np.ndarray, boxes: List[int], texts: List[str], oris: List[str]
) -> np.ndarray:
    """Draw OCR results."""
    for text, box, ori in zip(texts, boxes, oris):
        cv2.polylines(
            img,
            [np.array(box).reshape(-1, 1, 2).astype(np.int32)],
            True,
            (0, 255, 0),
            2,
        )
        color = (0, 255, 0) if ori == "up" else (0, 0, 255)
        cv2.putText(
            img, text, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return img


def draw_ocr_comparisson(
    img: np.ndarray, boxes: List[List[int]], texts: List[str], oris: List[str]
) -> np.ndarray:
    """Draw OCR results."""
    img_white = np.ones_like(img) * 255
    for text, box, ori in zip(texts, boxes, oris):
        cv2.polylines(
            img_white,
            [np.array(box).reshape(-1, 1, 2).astype(np.int32)],
            True,
            (0, 255, 0),
            2,
        )
        cv2.polylines(
            img,
            [np.array(box).reshape(-1, 1, 2).astype(np.int32)],
            True,
            (0, 255, 0),
            2,
        )
    for text, box, ori in zip(texts, boxes, oris):
        color = (0, 0, 0) if ori == "up" else (0, 0, 150)
        cv2.putText(
            img_white, text, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    img = cv2.hconcat([img, img_white])

    return img


def generate_st_table(
    img: np.ndarray, boxes: List[int], texts: List[str], oris: List[str]
) -> pd.DataFrame:
    """Generate Streamlit table."""
    crop_imgs: List[str] = []
    for box in boxes:
        box = np.array(box).reshape(-1, 1, 2).astype(np.int32)
        crop_img = rotated_crop(img, box)
        crop_imgs.append(np_to_base64(crop_img))

    df = pd.DataFrame(
        {
            "Image": crop_imgs,
            "Text": texts,
            "Orientation": oris,
        }
    )

    return df


def rotated_crop(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop image with rotated box (points)."""
    assert len(points) == 4, "Rotated box must have 4 points"
    width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ]
    )
    M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
    dst_image = cv2.warpPerspective(
        img,
        M,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_LINEAR,
    )

    # rotate image
    dst_img_h, dst_img_w = dst_image.shape[:2]
    if dst_img_h * 1.0 / dst_img_w > 1.5:
        dst_image = np.rot90(dst_image)

    return dst_image


def np_to_base64(img: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    img_pil = Image.fromarray(img)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    img_str = base64.b64encode(img_bytes.read()).decode()

    return f"<img src='data:image/jpeg;base64,{img_str}' width='80'>"
