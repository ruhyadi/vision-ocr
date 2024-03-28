"""OCR utils."""

import rootutils

ROOT = rootutils.autosetup()

import string
from typing import List, Optional, Tuple

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


class BaseRecLabelDecode(object):
    """Convert between text-label and text-index"""

    def __init__(
        self, character_dict_path=None, character_type="ch", use_space_char=False
    ):
        # fmt: off
        support_character_type = [
            'ch', 'en', 'EN_symbol', 'french', 'german', 'japan', 'korean',
            'it', 'xi', 'pu', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs', 'oc',
            'rsc', 'bg', 'uk', 'be', 'te', 'ka', 'chinese_cht', 'hi', 'mr',
            'ne', 'EN', 'latin', 'arabic', 'cyrillic', 'devanagari'
        ]
        # fmt: on

        assert (
            character_type in support_character_type
        ), "Only {} are supported now but get {}".format(
            support_character_type, character_type
        )

        self.beg_str = "sos"
        self.end_str = "eos"

        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == "EN_symbol":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        elif character_type in support_character_type:
            self.character_str = []
            assert (
                character_dict_path is not None
            ), "character_dict_path should not be None when character_type is {}".format(
                character_type
            )
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        else:
            raise NotImplementedError
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list))) if conf_list else None
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(
        self,
        character_dict_path=None,
        character_type="ch",
        use_space_char=False,
        **kwargs
    ):
        super(CTCLabelDecode, self).__init__(
            character_dict_path, character_type, use_space_char
        )

    def __call__(
        self, preds, label=None, *args, **kwargs
    ) -> List[Optional[Tuple[str, np.ndarray]]]:
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character
