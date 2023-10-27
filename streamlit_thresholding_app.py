import streamlit as st
import cv2
import numpy as np
import io
import base64
from PIL import Image

# Generating a link to download a particular image file.
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format = 'JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Set title.
st.title('Image Thresholding using OpenCV')

# Specify canvas parameters in application
uploaded_file = st.file_uploader("Upload Image to perform thresholding:", type=["png", "jpg"])

# Create a Slider and get the threshold from the slider.
threshold = st.slider("SET Threshold", min_value=0.0, max_value=255.0, step=1.0, value=150.0)

if uploaded_file is not None:
    # Convert the file to an opencv image.
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    input_col, output_col = st.columns(2)
    with input_col:
        st.header('Original')
        # Display uploaded image.
        st.image(img, channels='BGR', use_column_width=True)

    # Flag for showing output image.
    output_flag = 1
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, out_image = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

    with output_col:
        if output_flag == 1:
            st.header('Output')
            st.image(out_image)
            result = Image.fromarray(out_image)
            # Display link.
            st.markdown(get_image_download_link(result,'output.png','Download '+'Output'),
                        unsafe_allow_html=True)
