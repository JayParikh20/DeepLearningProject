import random
# from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from tensorflow.keras.models import load_model
from environment import CarEnv
from PIL import Image
import cv2
import carla


def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# MODEL_PATH = 'models/DATA_1.h5'
MODEL_PATH = 'models/1.h5'
speed_limit = 30
if __name__ == '__main__':

    # Load the model
    model = load_model(MODEL_PATH)

    # Create environment
    env = CarEnv()

    model.predict(np.ones((1, 160,320, 3)))

    # Loop over episodes
    while True:

        print('Restarting episode')

        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []

        done = False

        # Loop over steps
        while True:

            # Show current frame
            cv2.imshow(f'Agent - preview', current_state)
            cv2.waitKey(1)

            # Predict an action based on current observation space
            image = np.array(current_state)
            image = img_preprocess(image)
            image = np.array([image])
            speed = env.vehicle.get_velocity()
            steering_angle = float(model.predict(image))
            if steering_angle < 0.01:
                action = 0
            elif steering_angle > 0.01:
                action = 2
            else:
                action = 1
           

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, done, _ = env.step(action)

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                break
            
        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()