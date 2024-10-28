import cv2  # OpenCV for webcam access
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Dropout
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def draw_outlined_text(image, text, position, font, font_scale, color, thickness, outline_color, outline_thickness):
    image = cv2.putText(image, text, position, font, font_scale, outline_color, outline_thickness, lineType=cv2.LINE_AA)
    image = cv2.putText(image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return image

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
            # cv2.putText(frame, hand_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
            #if(handedness)
            # Draw hand landmarks
            
            if (hand_label=='Left'):
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
                mp_drawing.draw_landmarks(
                    dummy_frame, landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=5, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=15, circle_radius=1),
                )
            else:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
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
def capture_and_classify_webcam(hands, mp_drawing, mp_hands, frame, dqn_left,dqn_right):

    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        hand_results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        left_hand_cropped, right_hand_cropped, CamFrame = get_hands_landmarks(image, hand_results,mp_drawing,mp_hands)

        left_hand_cropped = left_hand_cropped.astype(np.float32) / 255.0
        right_hand_cropped = right_hand_cropped.astype(np.float32) / 255.0

        # Get prediction (action) from the agent
        action_left_hand = dqn_left.forward(left_hand_cropped)
        action_right_hand = dqn_right.forward(right_hand_cropped)
        # Map the action to class names
        class_names_left_hand = ['swarm_1', 'swarm_2', 'no_action']
        predicted_class_left_hand = class_names_left_hand[action_left_hand]

        class_names_right_hand = ['up', 'down', 'left','right','backwards','forward','take_off','land','no_action']
        predicted_class_right_hand = class_names_right_hand[action_right_hand]

        # Get the dimensions of the frame
        height, width, channels = CamFrame.shape

        # Check if the frame is large enough to place the cropped images
        if height >= 64 and width >= 64:
            translateX, translateY = 16, 16
            # Merge left_hand_cropped into CamFrame at bottom left corner
            CamFrame[height - 64 - translateY : height - translateY, 0 + translateX : 64 + translateX] = left_hand_cropped*255

            translateX, translateY = 86, 16
            # Merge right_hand_cropped into CamFrame at bottom right corner
            CamFrame[height - 64 - translateY : height - translateY, 0 + translateX : 64 + translateX] = right_hand_cropped*255
        else:
            print("CamFrame is too small to overlay the images.")

        # Define font parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6

        # Left corner text (Top Left)
        CamFrame = draw_outlined_text(CamFrame, f"{predicted_class_left_hand}", (16, 30), font, font_scale, (0, 0,255), 2, (255, 255, 255), 2)
        # cv2.putText(CamFrame, "SWARM", (10, 30), font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(CamFrame, f"{predicted_class_left_hand}", (10, 60), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)

        # Calculate text sizes for right corner text
        (text_width1, _), _ = cv2.getTextSize("ACTION", font, font_scale, 1)
        (text_width2, _), _ = cv2.getTextSize(predicted_class_right_hand, font, font_scale, 2)

        # Right corner text positions
        x1 = width - text_width1 - 10  # 10 pixels from the right edge
        x2 = width - text_width2 - 10

        # Right corner text (Top Right)
        CamFrame = draw_outlined_text(CamFrame, f"{predicted_class_right_hand}", (16, 50), font, font_scale, (0, 255, 0), 2, (255, 255, 255), 2)
        # cv2.putText(CamFrame, "ACTION", (x1, 30), font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(CamFrame, f"{predicted_class_right_hand}", (x2, 60), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)

        return predicted_class_left_hand,predicted_class_right_hand,CamFrame
    except:
        print('SOMETHING WENT TERRIBLY WRONG')


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


def initialize_agents():
    # Set up the environment parameters
    height, width, channels = 64, 64, 3
    actions_left_hand = 3

    # Create the model and agent
    model_left_hand = build_model(height, width, channels, actions_left_hand)
    dqn_left = build_agent(model_left_hand, actions_left_hand)
    dqn_left.compile(Adam(lr=1e-4))

    # Load the pre-trained weights
    dqn_left.load_weights('dqn_weights_with_back_data.h5f')

    actions_right_hand = 9

    # Create the model and agent
    model_right_hand = build_model(height, width, channels, actions_right_hand)
    dqn_right = build_agent(model_right_hand, actions_right_hand)
    dqn_right.compile(Adam(lr=1e-4))

    # Load the pre-trained weights
    dqn_right.load_weights('dqn_weights_for_right_hand_gestures.h5f')

    return dqn_left, dqn_right


def RL_image_classifier(hands, mp_drawing, mp_hands, frame, dqn_left, dqn_right):

    # Run the webcam and classify each frame
    predicted_class_left_hand,predicted_class_right_hand, annotated_frame= capture_and_classify_webcam(hands,
                                                                                                       mp_drawing,
                                                                                                       mp_hands,
                                                                                                       frame,
                                                                                                       dqn_left,
                                                                                                       dqn_right)

    return predicted_class_left_hand,predicted_class_right_hand, annotated_frame