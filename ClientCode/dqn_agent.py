"""
This agent will be used for  Deep Q-Network. DQNs blend Q-learning (a model free RL algo) with
deep neural netowrks. In DQN, the agent stores it's experiences in a memory buffer. During training,
it randomly samples minibatches from these experiences to make updates to the neural network.

The random sampling helps stabilize the learning process and break correlations between
consecutive learning samples.

DQNs use target networks. A target network is a separate neural net that is architecturally
identical to the primary network, but with it's weights update less frequently. This helps to stabilize
the learning by keeping the target Q-values fixed for a number of updates.

This code is adapted from the [RL with CARLA](https://pythonprogramming.net/reinforcement-learning-self-driving-autonomous-cars-carla-python/?completed=/reinforcement-learning-agent-self-driving-autonomous-cars-carla-python/)
tutorial written by @sentdex.
"""

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import tensorflow as tf

from modified_tensor_board import ModifiedTensorBoard

from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model

# TODO - can I delete?
# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
# import carla

# Image Parameters
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Training Parameters
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

DISCOUNT = 0.99
epsilon = 1


class DQNAgent:

    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        # This does not work with TensorFlow 2.0
        # TODO - update to work with TF 2.0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # This is a 3-neuron output representing the 3 actions the agent could take
        predictions = Dense(3, activation="linear")(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
