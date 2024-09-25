import cv2  # OpenCV for webcam access
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Dropout
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
import mediapipe as mp



def get_hands_landmarks(frame, hand_landmarks,mp_drawing,mp_hands):
    h, w, c = frame.shape
    dummy_frame = np.zeros(frame.shape)
    hand_landmarks_1 = hand_landmarks.multi_hand_landmarks
    
    handedness = hand_landmarks.multi_handedness

    # Draw the hands landmarks and bounding box
    left_hand_cropped = None
    right_hand_cropped = None
    if hand_landmarks_1:
        for landmarks, hand_handedness in zip(hand_landmarks_1, handedness):
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
            
            
            if(hand_label == 'Left'):
                
                y_crop_min = y_min-10 if y_min-10 > 0 else 0
                x_crop_min = x_min-10 if x_min-10 > 0 else 0
                
                y_crop_max = y_max+10 if y_max+10 < dummy_frame.shape[0] else dummy_frame.shape[0]
                x_crop_max = x_max+10 if x_max+10 < dummy_frame.shape[1] else dummy_frame.shape[1]
                
                left_hand_cropped = dummy_frame[y_crop_min: y_crop_max,x_crop_min: x_crop_max,:]

                
            if(hand_label == 'Right'):
                
                y_crop_min = y_min-10 if y_min-10 > 0 else 0
                x_crop_min = x_min-10 if x_min-10 > 0 else 0
                
                y_crop_max = y_max+10 if y_max+10 < dummy_frame.shape[0] else dummy_frame.shape[0]
                x_crop_max = x_max+10 if x_max+10 < dummy_frame.shape[1] else dummy_frame.shape[1]

                right_hand_cropped = dummy_frame[y_crop_min: y_crop_max,x_crop_min: x_crop_max,:]
                

    if(left_hand_cropped is None):
        left_hand_cropped = np.zeros((64, 64, 3))
    if(right_hand_cropped is None):
        right_hand_cropped = np.zeros((64, 64, 3))


    return cv2.resize(left_hand_cropped, (64, 64)), cv2.resize(right_hand_cropped, (64, 64)), frame



# Webcam inference function
def capture_and_classify_webcam(dqn, height, width, channels):
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    try:
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture image")
                    break

                # Preprocess the image (resize to 64x64 and normalize)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                hand_results = hands.process(image)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                left_hand_cropped, right_hand_cropped, CamFrame = get_hands_landmarks(image, hand_results,mp_drawing,mp_hands)

                # cv2.imshow('left landmarks', left_hand_cropped)
        
                # cv2.imshow('right landmarks', right_hand_cropped)


                left_hand_cropped = left_hand_cropped.astype(np.float32) / 255.0
                # left_hand_cropped = np.expand_dims(left_hand_cropped, axis=0)  # Add batch dimension
                # left_hand_cropped = np.expand_dims(left_hand_cropped, axis=0)  # Add window length dimension for DQN

                # Get prediction (action) from the agent
                action = dqn.forward(left_hand_cropped)

                # Map the action to class names
                class_names = ['swarm_1', 'swarm_2', 'no_action']
                predicted_class = class_names[action]
                print(f"Predicted class: {predicted_class}")

                cv2.imshow('left hand landmarks', left_hand_cropped)

                # Display the webcam frame with the predicted class
                cv2.putText(CamFrame, f"CONTOL:",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(CamFrame, f"{predicted_class}",(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('CAMERA Window', CamFrame)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break 
    finally:
        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

# Build the model (same as your original code)
def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

# Build and compile the agent (same as your original code)
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=50000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn

# Set up the environment parameters
height, width, channels = 64, 64, 3
actions = 3

# Create the model and agent
model = build_model(height, width, channels, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))

# Load the pre-trained weights
dqn.load_weights('dqn_weights_with_back_data.h5f')

# Run the webcam and classify each frame
capture_and_classify_webcam(dqn, height, width, channels)
