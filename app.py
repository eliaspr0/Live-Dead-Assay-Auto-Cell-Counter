import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# UI: Upload image
st.title("Live/Dead Cell Counter (Otsu Thresholding)")
uploaded_file = st.file_uploader("Upload a fluorescent cell image", type=["tif", "jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    if img_np.dtype != np.uint8:
        img_np = (255 * (img_np / np.max(img_np))).astype(np.uint8)

    st.image(image, caption="Original Image", use_column_width=True)

    # Split channels
    red = img_np[:, :, 0]
    green = img_np[:, :, 1]

    # Otsu thresholding
    _, red_thresh = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, green_thresh = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # UI sliders
    st.sidebar.header("Area Filters")
    live_min = st.sidebar.slider("Live Min Area", 10, 300, 80, 10)
    live_max = st.sidebar.slider("Live Max Area", 500, 8000, 3000, 100)
    dead_min = st.sidebar.slider("Dead Min Area", 10, 300, 30, 10)
    dead_max = st.sidebar.slider("Dead Max Area", 500, 8000, 2000, 100)

    def count_cells(binary_img, min_area, max_area):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
        vis = np.zeros_like(binary_img)
        cv2.drawContours(vis, filtered, -1, 255, 1)
        return len(filtered), vis

    live_cells, live_vis = count_cells(green_thresh, live_min, live_max)
    dead_cells, dead_vis = count_cells(red_thresh, dead_min, dead_max)
    total = live_cells + dead_cells
    live_pct = (live_cells / total) * 100 if total else 0
    dead_pct = (dead_cells / total) * 100 if total else 0

    st.markdown(f"### 🧬 Results")
    st.write(f"**Live Cells:** {live_cells} ({live_pct:.2f}%)")
    st.write(f"**Dead Cells:** {dead_cells} ({dead_pct:.2f}%)")

    # Show masks
    st.image(live_vis, caption="Live Cell Contours", clamp=True)
    st.image(dead_vis, caption="Dead Cell Contours", clamp=True)
