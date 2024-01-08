import cv2
import numpy as np
from insightface.app import FaceAnalysis
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--image1", type=str, help="Path of the first image")
ap.add_argument("--image2", type=str, help="Path of the second image")

args = ap.parse_args()

image_1_path = args.image1
image_2_path = args.image2

app = FaceAnalysis(providers=["CPUExecutionProvider"], name="buffalo_s")
app.prepare(ctx_id=0, det_size=(640, 640))

image_1 = cv2.imread(image_1_path)
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
result1 = app.get(image_1)
embedding1 = result1[0]["embedding"]

image_2 = cv2.imread(image_2_path)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
result2 = app.get(image_2)
embedding2 = result2[0]["embedding"]

distance = np.sqrt(np.sum((embedding1 - embedding2)**2))

threshold = 25

if distance < threshold:
    print("Same Person")
else:
    print("Diffrent Persons")