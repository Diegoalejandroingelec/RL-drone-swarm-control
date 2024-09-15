"""
Created on Tue Aug 20 18:29:57 2024

@author: lenovo
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp




gesture_class_left = "no_action"
gesture_class_right = "take_off"



cap = cv2.VideoCapture(0)  # Use 0 for default camera or pass the video file path

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def draw_face_and_hands(frame, face_landmarks, hand_landmarks,i,j):
    h, w, c = frame.shape
    dummy_frame = np.zeros(frame.shape)
    hand_landmarks = hand_landmarks.multi_hand_landmarks
    handedness = hand_results.multi_handedness

    # Draw the hands landmarks and bounding box
    if hand_landmarks:
        for landmarks, hand_handedness in zip(hand_landmarks, handedness):
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y

            # Get the label (Left or Right)
            hand_label = hand_handedness.classification[0].label


            # Draw hand bounding box with label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, hand_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
           

            
            
            #if(handedness)
            # Draw hand landmarks
            
            if (hand_label=='Left'):
                mp_drawing.draw_landmarks(
                    dummy_frame, landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=5, circle_radius=1),
                    
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=15, circle_radius=1),
                )
            else:
                mp_drawing.draw_landmarks(
                    dummy_frame, landmarks, mp_hands.HAND_CONNECTIONS,
                    # mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=25, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=5, circle_radius=1)
                )
            
            cv2.imshow('Dummy Hands Detection', dummy_frame)
            
            if(hand_label == 'Left'):
                
                y_crop_min = y_min-10 if y_min-10 > 0 else 0
                x_crop_min = x_min-10 if x_min-10 > 0 else 0
                
                y_crop_max = y_max+10 if y_max+10 < dummy_frame.shape[0] else dummy_frame.shape[0]
                x_crop_max = x_max+10 if x_max+10 < dummy_frame.shape[1] else dummy_frame.shape[1]
                
                i+=1

                if(i%20==0):
                    cv2.imwrite(f'./rl_dataset_left_hand/{gesture_class_left}_{i}.jpg', dummy_frame[y_crop_min: y_crop_max,x_crop_min: x_crop_max,:])
                    with open(f'./rl_dataset_left_hand/{gesture_class_left}_{i}.txt', 'w') as file:
                        # Write the text to the file
                        file.write(gesture_class_left)
                
            if(hand_label == 'Right'):
                
                y_crop_min = y_min-10 if y_min-10 > 0 else 0
                x_crop_min = x_min-10 if x_min-10 > 0 else 0
                
                y_crop_max = y_max+10 if y_max+10 < dummy_frame.shape[0] else dummy_frame.shape[0]
                x_crop_max = x_max+10 if x_max+10 < dummy_frame.shape[1] else dummy_frame.shape[1]
                
                j+=1
                if(j%5==0):
                    cv2.imwrite(f'./rl_dataset_right_hand/{gesture_class_right}_{j}.jpg', dummy_frame[y_crop_min: y_crop_max,x_crop_min: x_crop_max,:])
                    with open(f'./rl_dataset_right_hand/{gesture_class_right}_{j}.txt', 'w') as file:
                        # Write the text to the file
                        file.write(gesture_class_right)
                        print(j)

    return dummy_frame, i, j

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    i=0
    j=0     
    while True:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        face_results = face_mesh.process(image)
        hand_results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        dummy_frame, i, j=draw_face_and_hands(image, face_results.multi_face_landmarks, hand_results,i,j)
        cv2.imshow('Face and Hands Detection', image)
        # cv2.imshow('Dummy Hands Detection', dummy)/
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
