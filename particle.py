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
        self.weight = 1

    def pos(self):
        return (self.pos_x, self.pos_y)

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
        """The arguments x, y are associated with the origin on the top left, we need to transform the coordinates
        so that the origin is at bottom left.
        """
        if x > WINDOWWIDTH:
            x = WINDOWWIDTH
        if y > WINDOWHEIGHT:
            y = WINDOWHEIGHT
        self.pos_x = x
        self.pos_y = WINDOWHEIGHT - y
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
        y = self.pos_y + d * math.sin(self.orientation) + gauss_noise(0, self.motion_noise)
        if self.check_pos(x, y):
            if self.is_robot:
                return
            else:
                self.reset_pos()
                return
        else:
            self.pos_x = x
            self.pos_y = y

    def turn_left(self, angle):
        self.orientation = (self.orientation + (angle + gauss_noise(0, self.turning_nosie)) / 180. * math.pi) % (2 * math.pi)

    def turn_right(self, angle):
        self.orientation = (self.orientation - (angle + gauss_noise(0, self.turning_nosie)) / 180. * math.pi) % (2 * math.pi)

    def dick(self):
        return [(self.pos_x, self.pos_y), (self.pos_x + self.dick_length * math.cos(self.orientation), self.pos_y + self.dick_length * math.sin(self.orientation))]

    def update(self, obs):
        """After the motion, update the weight of the particle and its EKFs based on the sensor data"""
        for o in obs:
            prob = 0
            if self.landmarks:
                for landmark in self.landmarks:
                    continue
                    # find the data association with ML
                    # update corresponding EKF
                    # prob =
            else: # no initial landmarks
                x, y = guess_landmark(self.pos_x, self.pos_y, obs)
                landmark = Landmark(x, y)
                self.landmarks.append(landmark)
                # prob
            self.weight *= prob

    def sense(self, landmarks):
        """
        Only for robot.
        Given the existing landmarks, generates a random number of obs (distance, direction)
        """
        num_obs = random.randint(1, len(landmarks)-1)
        obs_list = []
        for i in random.sample(range(len(landmarks)), num_obs):
            l = landmarks[i].pos()
            dis = euclidean_distance(l, (self.pos_x, self.pos_y))
            noise = gauss_noise(0, self.distance_noise)
            if (dis + noise) > 0:
                dis += noise
            direction = cal_direction((self.pos_x, self.pos_y), (l[0], l[1]))
            obs_list.append((dis, direction))
        return obs_list

    def update_landmarks(self, obs):
        pass



