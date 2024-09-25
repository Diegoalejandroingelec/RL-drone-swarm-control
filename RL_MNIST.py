import gym
from gym import spaces
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

# Function to load MNIST dataset
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test

# Custom environment for MNIST classification
class MNISTClassificationEnv(gym.Env):
    def __init__(self):
        super(MNISTClassificationEnv, self).__init__()
        
        # Load MNIST data
        x_train, y_train, x_test, y_test = load_mnist()
        self.x_data = np.concatenate([x_train, x_test], axis=0)
        self.y_data = np.concatenate([y_train, y_test], axis=0)
        self.num_samples = self.x_data.shape[0]
        self.current_step = 0
        self.max_steps = 30  # Set a limit for an episode
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(28, 28, 1), dtype=np.float32)
        self.action_space = spaces.Discrete(10)  # Digits 0-9
        
        # Shuffle data indices
        self.indices = np.arange(self.num_samples)
        np.random.shuffle(self.indices)
        
    def reset(self):
        self.current_step = 0
        np.random.shuffle(self.indices)
        return self._next_observation()
        
    def _next_observation(self):
        idx = self.indices[self.current_step]
        self.current_label = self.y_data[idx]
        return self.x_data[idx]
    
    def step(self, action):
        reward = 1 if action == self.current_label else -1
        self.current_step += 1
        done = self.current_step >= self.max_steps
        # obs = self._next_observation() if not done else None

        if not done:
            obs = self._next_observation()
        else:
            obs = self._next_observation()
        

        return obs, reward, done, {}
    
    def render(self, mode='human'):
        pass

# Initialize environment
env = MNISTClassificationEnv()
obs = env.reset()

# Get shape parameters
height, width, channels = env.observation_space.shape
actions = env.action_space.n

# Build the model
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

# Build the agent
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1.0, value_min=0.1, value_test=0.2, nb_steps=10000)
    memory = SequentialMemory(limit=50000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))

# Train the agent
dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)

# Test the agent
scores = dqn.test(env, nb_episodes=10, visualize=False)
print(f'Average reward over 10 episodes: {np.mean(scores.history["episode_reward"])}')

# Save the weights
dqn.save_weights('dqn_mnist_weights.h5f', overwrite=True)
