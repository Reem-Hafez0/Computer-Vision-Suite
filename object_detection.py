import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Visualization function
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def visualize(image, detection_result):
    """Draws bounding boxes on the input image and return it."""
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

st.title("Object Detection with mediapipe")

st.write("Upload An image")

uploadfile=st.file_uploader("Choose file",type=["jpg","png","jpeg"])

if uploadfile is not None:
    file_bytes=np.asarray(bytearray(uploadfile.read()),dtype=np.uint8)
    image=cv2.imdecode(file_bytes,1)
    img_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    st.image(img_rgb,caption="this is image",use_column_width=True)
    
    
    
    base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    detection_result = detector.detect(mp_image)
    
    annotated_img=visualize(img_rgb,detection_result)
    # rgb_annt=cv2.cvtColor(annotated_img,cv2.COLOR_BGR2RGB)
    st.image(annotated_img,caption="Results",use_column_width=True)