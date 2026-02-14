# A code which detects the closing of eyes and plays a music when closed

# Libaries -------------------
import cv2 as cv
import numpy as num 
import pygame as py 
import mediapipe as mp 
#-----------------------------



#calling the face mesh model from mediapipe --------
# full_face_mesh = mp.solutions.face_mesh
# face_mesh = full_face_mesh.FaceMesh(refine_landmarks = True)
#or
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True)
#----------------------------------------------------



#videocapture -----------
cap = cv.VideoCapture(0) 
#------------------------



#coordinates of the left and right eye.
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE = [33, 160, 158, 133, 153, 144]


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)


#----------hitting the pixels on;y for the eye region and not the whole face.-------------
    #if result.multi_face_landmarks[0]: for one face that is indexing
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
#now we tell it to only detect th eyes and not the whole face.
            for pixel_position in LEFT_EYE + RIGHT_EYE:
                x = int(face_landmarks.landmark[pixel_position].x * w)
                y = int(face_landmarks.landmark[pixel_position].y * h)
                cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
#---------------------------------------------------------------------------------



    cv.imshow("EYE DETECTION", frame)
    k = cv.waitKey(1)
    if k == ord('q'):
        break

cv.release()
cv.destroyAllWindows()

#till now it detects the eyes and color it red.            