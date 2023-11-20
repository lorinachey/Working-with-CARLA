"""
VehicleEnvironment contains the typical RL methods (e.g. step, reset) that
are commonly used by convention. This class also contains environment
variables and the list of actions.

"""
import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import numpy as np
import cv2


# This can be used for testing purposes to observe the camera image.
# Set to False when doing RL to avoid hogging CPU/GPU resources.
DISPLAY_PREVIEW = True

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

STEER_AMOUNT = 0.9

# Training Parameters
SECONDS_PER_EPISODE = 10

class VehicleEnvironment:

    actor_list = []

    front_camera = None
    collision_history = []

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        self.im_height = IMAGE_HEIGHT
        self.im_width = IMAGE_WIDTH

        # Retrieve the currently running world
        self.world = self.client.get_world()

        blueprint_library = self.world.get_blueprint_library()
        self.mini_cooper = blueprint_library.filter('cooper_s_2021')[0]


    def reset(self):

        # Clear the collision and actors lists
        self.collision_history = []
        self.actor_list = []

        # Spawn a new vehicle in the world
        spawn_successful = False
        while not spawn_successful:
            try:
                spawn_point = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.mini_cooper, spawn_point)
                self.actor_list.append(self.vehicle)
                spawn_successful = True
            except Exception as e:
                print(e)

        # Set up the camera sensor
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')

        self.rgb_cam.set_attribute('image_size_x', f'{IMAGE_WIDTH}')
        self.rgb_cam.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
        self.rgb_cam.set_attribute('fov', '110')

        # Use this transform to spawn the camera somewhere on the car
        transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.camera_sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)

        self.actor_list.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: self.process_image(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        collision_sensor_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_sensor_bp, transform, attach_to=self.vehicle)

        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        # Make sure the camera sensor is initiated and processing images
        while self.front_camera is None:
            time.sleep(0.1)

        return self.front_camera


    def step(self, action):
        """
        The action space will be left, center, and right. We'll represent these actions
        using the values 1, 0, 2, respectively.
        """
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*STEER_AMOUNT))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=STEER_AMOUNT))

        # Penalize getting into a collision with a hefty negative reward
        if len(self.collision_history) != 0:
            done = True
            reward = -10
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None


    def process_image(self, image):
        """
        CARLA image data from the camera sensor is RGB-alpha. We'll reshape the raw
        data, then extract just the RGB channels, ignoring the alpha data.
        """
        raw_image_data = np.array(image.raw_data)
        reshaped_image = raw_image_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
        rgb_extracted_image = reshaped_image[:, :, :3]
        if DISPLAY_PREVIEW:
            cv2.imshow("", rgb_extracted_image)
            cv2.waitKey(1)
        self.front_camera = rgb_extracted_image


    def collision_data(self, event):
        self.collision_history.append(event)






