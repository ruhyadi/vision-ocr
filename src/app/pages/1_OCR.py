"""OCR page."""

import rootutils

ROOT = rootutils.autosetup()

import time
from io import BytesIO

import os
import numpy as np
import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from src.schema.ocr_schema import OcrResultSchema
from src.utils.logger import get_logger
from src.utils.plot_utils import draw_ocr, draw_ocr_comparisson, generate_st_table

log = get_logger()

load_dotenv()

API_PORT = os.getenv("API_PORT")
API_HOST = os.getenv("API_HOST")
BASE_URL = f"http://{API_HOST}:{API_PORT}"


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
        with st.spinner("Processing..."):
            img_pil = Image.open(img_uploaded)

            # predict OCR
            t0 = time.time()
            ocr_result = post_image(img_pil)
            t_ms = (time.time() - t0) * 1000

            ocr_result_img = draw_ocr_comparisson(
                np.array(img_pil),
                ocr_result.boxes,
                ocr_result.texts,
                ocr_result.oris,
            )
            
        st.success(f"Success generating OCR result in {t_ms:.2f} ms")
        st.warning(
            f"Image quality reduced. Please download the image to view the result."
        )
        st.download_button(
            label="Download Full Resolution OCR Result",
            data=np_to_bytes_pil(ocr_result_img),
            file_name="ocr_result.jpg",
            mime="image/jpeg",
        )
        st.image(ocr_result_img, caption="OCR Result", use_column_width=True)

        with st.spinner("Generating table..."):
            st.subheader("OCR Result Table")
            df = generate_st_table(
                img=np.array(img_pil),
                boxes=ocr_result.boxes,
                texts=ocr_result.texts,
                oris=ocr_result.oris,
            )
            st.write(df.to_html(escape=False), unsafe_allow_html=True)


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


def np_to_bytes_pil(img: np.ndarray) -> bytes:
    """Convert numpy array to PIL Image."""
    img_pil = Image.fromarray(img)
    img_bytes = BytesIO()
    img_pil.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    return img_bytes


if __name__ == "__main__":
    page()
