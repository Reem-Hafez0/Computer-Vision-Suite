# Computer Vision Suite 

A collection of computer vision apps including object detection, emotion analysis, and video processing using Streamlit.

## Available Projects

### 1. Human Emotion Detection (`humanemotion.py`)
Detect human emotions from images using DeepFace.

### 2. Object Detection (`object_detection.py`)
Detect objects in static images using MediaPipe and EfficientDet.

### 3. Video Object Detection (`obj_vid.py`)
Detect objects in videos or live camera feed.

### 4. EfficientDet Model (`efficientdet.tflite`)
TensorFlow Lite model used by MediaPipe for object detection.

## Features
- Real-time emotion detection
- High-accuracy object detection
- Video and camera processing
- User-friendly Streamlit interfaces
- Optimized AI models for speed

## Requirements
- Python 3.8 or higher
- TensorFlow
- OpenCV
- MediaPipe
- PyTorch (optional)

## Installation

### 1. Clone the repository

### 2. Create a virtual environment

### 3. Install the dependencies

### 4. Run the Applications

#### Emotion Analysis:
```bash
streamlit run humanemotion.py
```

#### Object Detection (Image):
```bash
streamlit run object_detection.py
```

#### Object Detection (Video or Camera):
```bash
streamlit run obj_vid.py
```

## How to Use

### Emotion Detection:
1. Upload an image or use camera
2. Get real-time emotion analysis

### Object Detection:
1. Upload an image
2. View detected objects
3. See bounding box labels and confidence scores

### Video Detection:
1. Upload a video file
2. Watch real-time detection results
3. View statistics or frame-by-frame analysis

## Future Enhancements
- Add more AI models
- Improve detection accuracy
- Support more object types
- Scene analysis support
- Enhanced performance and speed
