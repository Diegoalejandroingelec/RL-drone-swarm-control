import gym
from gym import spaces
import numpy as np
import os
import random
from PIL import Image


from rl.agents import SARSAAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D,Dropout
from tensorflow.keras.optimizers import Adam
from rl.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
# Compute and display the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix


test_agent_for_left_hand = True
class ImageClassificationEnv(gym.Env):
    def __init__(self, image_folder, hand):
        super(ImageClassificationEnv, self).__init__()
        self.hand = hand
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        
        # Observation space: A 3D array representing the image (e.g., 64x64x3 for RGB images)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        
        # Action space: Discrete actions corresponding to the classes
        if self.hand == 'left':
            self.action_space = spaces.Discrete(3)
        if self.hand == 'right':
            self.action_space = spaces.Discrete(9)
        
        self.current_step = 0
        self.max_steps = 30
        self.current_image = None
        self.image_label = None
        self.load_images()

    def load_images(self):
        # Randomly select images for the game
        self.selected_images = random.sample(self.image_files, self.max_steps + 1)

    def reset(self):
        self.current_step = 0
        self.load_images()
        return self._next_observation()

    def _next_observation(self):
        # Load the next image
        image_file = self.selected_images[self.current_step]
        image_path = os.path.join(self.image_folder, image_file)
        self.current_image = np.array(Image.open(image_path).resize((64, 64))) / 255.0
        
        # Load the corresponding label from the txt file
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(self.image_folder, label_file)
        with open(label_path, 'r') as file:
            label = file.read().strip()
        
        if self.hand == 'left':
            # Convert label to action space format
            if label == 'swarm_1':
                self.image_label = 0
            elif label == 'swarm_2':
                self.image_label = 1
            else:
                self.image_label = 2
        
        if self.hand == 'right':
            label_mapping = {
                'up': 0,
                'down': 1,
                'left': 2,
                'right': 3,
                'backwards': 4,
                'forward': 5,
                'take_off': 6,
                'land': 7
            }
            self.image_label = label_mapping.get(label, 8)  # Default to 8 if label not found
        
        return self.current_image

    def step(self, action):

        # Include true label in the info dictionary
        info = {'true_label': self.image_label}
        # Reward logic
        if action == self.image_label:
            reward = 1
        else:
            reward = -1
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        if not done:
            obs = self._next_observation()
        else:
            obs = self._next_observation()
        
        
        return obs, reward, done, info

    def render(self, mode='human'):
        pass


if(test_agent_for_left_hand==True):
    env = ImageClassificationEnv(image_folder='rl_train_dataset_left_hand',hand='left')
else:
    env = ImageClassificationEnv(image_folder='rl_train_dataset_right_hand',hand='right')


obs = env.reset()

height, width, channels = env.observation_space.shape

actions = env.action_space.n

def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu', input_shape=(1, height, width, channels)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(Convolution2D(256, (3, 3), activation='relu'))  # Add more layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))  # Adding Dropout
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model



model = build_model(height, width, channels, actions)

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    sarsa = SARSAAgent(model=model, policy=policy, nb_actions=actions, nb_steps_warmup=1000)
    return sarsa


sarsa = build_agent(model, actions)
sarsa.compile(Adam(lr=1e-4))


# Load the weights into the agent
if(test_agent_for_left_hand==False):
    sarsa.load_weights('sarsa_weights_for_right_hand_gestures.h5f')
else:
    sarsa.load_weights('sarsa_weights_for_left_hand_gestures.h5f')

class TestLogger(Callback):
    def __init__(self):
        self.true_labels = []
        self.predicted_labels = []

    def on_step_end(self, step, logs={}):
        action = logs['action']
        info = logs.get('info')
        if info is not None:
            true_label = info.get('true_label')
            if true_label is not None:
                self.true_labels.append(true_label)
                self.predicted_labels.append(action)

test_logger = TestLogger()
callbacks = [test_logger]

# Test the agent
sarsa.test(env, nb_episodes=30, visualize=False, callbacks=callbacks)

avg_reward = np.mean(sarsa.test(env, nb_episodes=30, visualize=False).history['episode_reward'])

print(f'Average Reward: {avg_reward}')

# Compute and display the classification report and confusion matrix




print("Classification Report:")
print(classification_report(test_logger.true_labels, test_logger.predicted_labels))


confusion_matrix=confusion_matrix(test_logger.true_labels, test_logger.predicted_labels)



def plot_confusion_matrix(confusion_matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

if(test_agent_for_left_hand==False):
    plot_confusion_matrix(confusion_matrix, ['up','down','left','right','backwards','forward','take_off','land','no_action'])
else:
    plot_confusion_matrix(confusion_matrix, ['swarm_1','swarm_2','no_action'])