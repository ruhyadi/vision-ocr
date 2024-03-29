"""Streamlit home."""

import rootutils

ROOT = rootutils.autosetup()

import streamlit as st

from src.utils.logger import get_logger

log = get_logger()

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

st.header("OCR Engine")

st.write(
    """
    Welcome to the OCR Engine! This engine is built using PaddleOCR, ONNXRuntime and FastAPI.
    """
)
