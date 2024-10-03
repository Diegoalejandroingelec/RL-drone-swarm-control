import gym
from gym import spaces
import numpy as np
import os
import random
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D,Dropout
from tensorflow.keras.optimizers import Adam



from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.applications import VGG16



class ImageClassificationEnv(gym.Env):
    def __init__(self, image_folder,hand):
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
        # Randomly select 30 images for the game
        self.selected_images = random.sample(self.image_files, self.max_steps+1)

    def reset(self):
        self.current_step = 0
        self.load_images()
        return self._next_observation()

    def _next_observation(self):
        # Load the next image
        image_file = self.selected_images[self.current_step]
        image_path = os.path.join(self.image_folder, image_file)
        self.current_image = np.array(Image.open(image_path).resize((64, 64)))/255.0
        
        # Load the corresponding label from the txt file
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(self.image_folder, label_file)
        with open(label_path, 'r') as file:
            label = file.read().strip()
        
        if self.hand == 'left':
            #Convert label to action space format (0 for swarm_1, 1 for swarm_2)
            if label == 'swarm_1':
                self.image_label = 0
            elif label == 'swarm_2':
                self.image_label = 1
            else:
                self.image_label = 2
        
        if self.hand == 'right':
            if label == 'up':
                self.image_label = 0
            elif label == 'down':
                self.image_label = 1
            elif label == 'left':
                self.image_label = 2
            elif label == 'right':
                self.image_label = 3
            elif label == 'backwards':
                self.image_label = 4
            elif label == 'forward':
                self.image_label = 5
            elif label == 'take_off':
                self.image_label = 6
            elif label == 'land':
                self.image_label = 7
            else: ## no action
                self.image_label = 8
        
        return self.current_image

    def step(self, action):
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
        
        return obs, reward, done, {}

    def render(self, mode='human'):
      pass

train_agent_for_left_hand = False
if(train_agent_for_left_hand==True):
    env = ImageClassificationEnv(image_folder='rl_train_dataset_left_hand',hand='left')
else:
    env = ImageClassificationEnv(image_folder='rl_train_dataset_right_hand',hand='right')

obs = env.reset()

height, width, channels = env.observation_space.shape

actions = env.action_space.n


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu', input_shape=(3, height, width, channels)))
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

model.summary()





def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=50000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=1000
                  )
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))


dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)


print('debugging')

scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history['episode_reward']))


# Save the weights
dqn.save_weights('dqn_weights_for_right_hand_gestures.h5f', overwrite=True)




# ###Random selection
# episodes = 10
# for _ in range(episodes):
#     total_reward = 0
#     while True:
#         #env.render()
#         action = env.action_space.sample()  # Replace with your agent's action
#         obs, reward, done, info = env.step(action)

#         total_reward+=reward
#         if done:
#             print(f'total reward for this episode: {total_reward}')
#             env.reset()
#             break

