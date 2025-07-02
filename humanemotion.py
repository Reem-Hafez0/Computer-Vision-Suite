import streamlit as st
import numpy as np
import cv2
from deepface import DeepFace
from PIL import Image
import tempfile 


st.title("Human Emotion Detection")
st.write("Upload an image")

def analyze_emotion(image):
    analysis=DeepFace.analyze(image,actions=["emotion"]) #age race ...
    # print("analysis",analysis)
    return analysis[0]["dominant_emotion"]

option=st.selectbox("choose which analysis",("image","video","live"))
if option=="image":
    upload=st.file_uploader("choose file",type=["png","jpg","jpeg"])
    if upload is not None:
        img=Image.open(upload)
        img_np=np.array(img)
        st.image(img_np,caption="uploaded image",use_column_width=True)
        emotion=analyze_emotion(img_np)
        st.write("Emotion: ",emotion)

elif option=="video":
    upload=st.file_uploader("choose file",type=["mp4","mov","avi"])
    if upload is not None:
        tfile=tempfile.NamedTemporaryFile(delete=False)
        tfile.write(upload.read())
        tfile.close()

        video=cv2.VideoCapture(tfile.name)
        stFrame=st.empty()
        while video.isOpened():
            ret,frame=video.read()
            if not ret:
                break
            frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            emotion=analyze_emotion(frame_rgb)
            cv2.putText(frame,f"{emotion}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            stFrame.image(frame,channels="BGR")