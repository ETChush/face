import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

img_path = "test_images/z-image_00189_.png"
img = cv2.imread(img_path)
if img is None:
    print("Failed to load image")
    exit(1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

detection_result = detector.detect(mp_image)

if detection_result.face_landmarks:
    print("Face landmarks detected!")
    print(f"Number of landmarks: {len(detection_result.face_landmarks[0])}")
    print(f"First landmark: x={detection_result.face_landmarks[0][0].x}, y={detection_result.face_landmarks[0][0].y}")
else:
    print("No face landmarks detected")