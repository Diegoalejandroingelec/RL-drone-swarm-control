from real_time_agent_testing import initialize_agents, RL_image_classifier
import cv2
import mediapipe as mp
from djitellopy import TelloSwarm

# Open webcam
cap = cv2.VideoCapture(0)
dqn_left,dqn_right = initialize_agents()


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        predicted_class_left_hand,predicted_class_right_hand, annotated_frame = RL_image_classifier(hands, mp_drawing, mp_hands, frame, dqn_left, dqn_right)
        
        print(f"Predicted class left hand: {predicted_class_left_hand} Predicted class right hand: {predicted_class_right_hand}")
        # Display the combined frame
        # cv2.imshow('CAMERA Window', annotated_frame)

        # # Press 'q' to exit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break 