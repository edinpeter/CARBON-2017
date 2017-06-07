import cv2
import capture as cp 
import numpy as np
import scipy, scipy.misc


# get ground truth image from assets folder
# get captured image from assets folder

filepath_to_assets = ''
filename_img_truth = '3.jpg'
filename_img_capture = 'dom2.jpg'
casc_path = "FaceDetect/haarcascade_frontalface_default.xml"

face_casc = cv2.CascadeClassifier(casc_path)

img_truth   = cv2.imread(filepath_to_assets + filename_img_truth, 1)
img_capture = cv2.imread(filepath_to_assets + filename_img_capture, 1)

img_truth = cv2.cvtColor(img_truth, cv2.COLOR_BGR2GRAY)
img_capture = cv2.cvtColor(img_capture, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_casc.detectMultiScale(
    img_truth,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(30, 30)
)

print(faces)

img_crops = []

for (x, y, w, h) in faces:
	crop = img_truth[x : x+w, y : y+h]
	img_crops.append(crop)
	cv2.imshow("", crop)
	cv2.waitKey(0)




