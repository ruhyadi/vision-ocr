"""OCR page."""

import rootutils

ROOT = rootutils.autosetup()

from io import BytesIO

import requests
import streamlit as st
from PIL import Image

from src.schema.ocr_schema import OcrResultSchema
from src.utils.logger import get_logger

log = get_logger()

BASE_URL = "http://localhost:4700"


def page():
    """OCR page."""
    st.set_page_config(
        page_title="OCR",
        page_icon="🔤",
    )

    st.header("Optical Character Recognition")
    st.write("OCR Engine using PaddleOCR, ONNXRuntime and FastAPI")

    st.subheader("Upload Image")
    img_uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    submit_btn = st.button("Submit")

    if submit_btn and img_uploaded:
        st.image(img_uploaded, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")
        ocr_result = post_image(Image.open(img_uploaded))
        st.write(ocr_result)


def post_image(img: Image) -> OcrResultSchema:
    """Post image to OCR API."""
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
    res = requests.post(
        f"{BASE_URL}/api/v1/engine/ocr/snapshot",
        files=files,
    )
    res.raise_for_status()

    return OcrResultSchema(**res.json())


if __name__ == "__main__":
    page()
