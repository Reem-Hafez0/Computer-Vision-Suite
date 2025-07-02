import tempfile
import streamlit as st
import cv2
import numpy as np
import openai
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Visualization function
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0) # red

def visualize(image, detection_result):
    for detection in detection_result.detections:

        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

st.title("Object Detection")

input_type = st.selectbox("Select input type", ("Image", "Video", "Camera"))

if input_type == "Image":
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
        st.image(annotated_img,caption="Results",use_column_width=True)


elif input_type == "Video":
    st.write("Upload a video")
    uploadfile = st.file_uploader("Choose file", type=["mp4", "mov", "avi"])
    
    if uploadfile is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploadfile.read())
        cap = cv2.VideoCapture(tfile.name)

        base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
        options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
        detector = vision.ObjectDetector.create_from_options(options)
        
        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            detection_result = detector.detect(mp_image)
            annotated_frame = visualize(img_rgb, detection_result)

            st.image(annotated_frame, caption="Video Frame", use_column_width=True)
        
        cap.release()


elif input_type == "Camera":
    st.write("Opening camera...")
    cap = cv2.VideoCapture(0)

    base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = detector.detect(mp_image)
        annotated_frame = visualize(img_rgb, detection_result)

        st.image(annotated_frame, caption="Camera Frame", use_column_width=True)

    cap.release()