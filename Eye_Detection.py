# A code which detects the closing of eyes and plays a music when closed

import cv2 as cv
import numpy as num 
import pygame as py 
import mediapipe as mp 


full_face_mesh = mp.solutions.face_mesh
face_mesh = full_face_mesh.FaceMesh(refine_landmarks = True)

cap = cv.VideoCapture(0)

#coordinates of the left and right eye.

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_frame)

#now we tell it to only detect th eyes and not the whole face.        