"""
    Implements the particle which has motion model, sensor model and EKFs for landmarks.
"""

import random
import math
from slam_helper import *
from world import WINDOWWIDTH, WINDOWHEIGHT

class Particle(object):
    """Represents the robot and particles"""
    def __init__(self, is_robot=False):
        """pos_x: from left to right
           pos_y: from up to down
           orientation: [0,2*pi)
        """
        self.pos_x = random.random() * WINDOWWIDTH
        self.pos_y = random.random() * WINDOWHEIGHT
        self.orientation = random.random() * 2. * math.pi
        self.dick_length = 10
        self.is_robot = is_robot
        self.landmarks =[]
        self.set_noise()


    def set_noise(self):
        if self.is_robot:
            self.bearing_noise = 0
            self.distance_noise = 0
            self.motion_noise = 0
            self.turning_nosie = 0
        else:
            self.bearing_noise = 50
            self.distance_noise = 3
            self.motion_noise = 0.1
            self.turning_nosie = 5 # unit: degree

    def set_pos(self, x, y, orien):
        if x >= WINDOWWIDTH:
            x = WINDOWWIDTH - 1
        if y >= WINDOWHEIGHT:
            y = WINDOWHEIGHT - 1
        self.pos_x = x
        self.pos_y = y
        self.orientation = orien

    def reset_pos(self):
        self.set_pos(random.random() * WINDOWWIDTH, random.random() * WINDOWHEIGHT, random.random() * 2. * math.pi)

    def check_pos(self, x, y):
        """Checks if a particle is in the invalid place"""
        if x >= WINDOWWIDTH or y >= WINDOWHEIGHT or x <=0 or y <= 0:
            return True

    def forward(self, d):
        """Motion model.
           Moves robot forward of distance d plus gaussian noise
        """
        x = self.pos_x + d * math.cos(self.orientation) + gauss_noise(0, self.motion_noise)
        y = self.pos_y - d * math.sin(self.orientation) + gauss_noise(0, self.motion_noise)
        if self.check_pos(x, y):
            if self.is_robot:
                return
            else:
                self.reset_pos()
                return
        else:
            self.set_pos(x, y, self.orientation)

    def turn_left(self, angle):
        self.orientation = (self.orientation + (angle + gauss_noise(0, self.turning_nosie)) / 180. * math.pi) % (2 * math.pi)

    def turn_right(self, angle):
        self.orientation = (self.orientation - (angle + gauss_noise(0, self.turning_nosie)) / 180. * math.pi) % (2 * math.pi)

    def dick(self):
        return [(int(self.pos_x), int(self.pos_y)), (int(self.pos_x + self.dick_length * math.cos(self.orientation)), int(self.pos_y - self.dick_length * math.sin(self.orientation)))]

    def sense(self, landmarks):
        """Given the existing landmarks, generates a random number of obs (distance, direction)"""
        num_obs = random.randint(0, len(landmarks)-1)
        obs_list = []
        for i in random.sample(num_obs, range(len(landmarks))):
            l = landmarks[i]
            dis = euclidean_distance(l, (self.pos_x, self.pos_y))
            if (dis + gauss_noise(0, self.distance_noise)) > 0:
                dis = (dis + gauss_noise(0, self.distance_noise))
            direction = cal_direction((self.pos_x, self.pos_y), l)
            obs_list.append((dis, direction))
        return obs_list

    def update_landmarks(self):
        pass



