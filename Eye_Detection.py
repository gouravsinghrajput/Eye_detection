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


#----------initial values for EAR calculations-------------
ear_threshold = 0.18
frame_treshold = 3
blink_counter = 0
total_blinks = 0
#---------------------------------------------------------

#num.linalg.norm is used to calculate the distance between the points

#-------------function to calculate the EAR ----------------
def ear_calculation(points_of_eye):
    vertical_1 = num.linalg.norm(num.array(points_of_eye[1]) - num.array(points_of_eye[5]))
    vertical_2 = num.linalg.norm(num.array(points_of_eye[2]) - num.array(points_of_eye[4]))
    horizontal = num.linalg.norm(num.array(points_of_eye[0]) - num.array(points_of_eye[3]))
    ear = (vertical_1 + vertical_2) / (2 * horizontal)
    return ear
#---------------------------------------------------------



#coordinates of the left and right eye.
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE = [33, 160, 158, 133, 153, 144]


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)


#--------------------------------------------------------------------------------
    #if result.multi_face_landmarks[0]: for one face that is indexing
    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[1]

        right_eye_points = [] 

 #--------For right eye--------------------------------------------------------   
        for pixel_position in RIGHT_EYE:
            x = int(face_landmarks.landmark[pixel_position].x * w)
            y = int(face_landmarks.landmark[pixel_position].y * h)
            right_eye_points.append((x,y))
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
#------------------------------------------------------------------------------
#---------For left eye --------------------------------------------------------
        left_eye_points = []
        for pixel_position in LEFT_EYE:
            x = int(face_landmarks.landmark[pixel_position].x * w)
            y = int(face_landmarks.landmark[pixel_position].y * h)
            left_eye_points.append((x, y))
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
#-----------------------------------------------------------------------------
#---------------------------------------------------------------------------------


    cv.imshow("EYE DETECTION", frame)
    k = cv.waitKey(1)
    if k == ord('q'):
        break

cv.release()
cv.destroyAllWindows()

#till now it detects the eyes and color it red.            