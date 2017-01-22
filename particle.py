"""
    Implements the particle which has motion model, sensor model and EKFs for landmarks.
"""

import random
import math
import numpy as np
from slam_helper import *
from scipy import linalg
from world import WINDOWWIDTH, WINDOWHEIGHT
from landmark import Landmark

class Particle(object):
    """Represents the robot and particles"""
    TOL = 1E-5

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
        self.obs_noise = np.zeros((2,2))

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
            ass_obs = np.zeros((2,1))
            ass_jacobian = np.zeros((2,2))
            ass_adjcov = np.zeros((2,2))
            landmark_idx = -1
            if self.landmarks:
                # find the data association with ML
                prob, landmark_idx, ass_obs, ass_jacobian, ass_adjcov = self.find_data_association(o)
                if prob < TOL:
                    # create new landmark
                    self.create_landmark(o)
                else:
                    # update corresponding EKF
                    self.update_landmark(np.transpose(np.array([o])), landmark_idx, ass_obs, ass_jacobian, ass_adjcov)
            else:
                # no initial landmarks
                self.create_landmark(o)
                # prob
            self.weight *= prob

    def sense(self, landmarks):
        """
        Only for robot.
        Given the existing landmarks, generates a random number of obs (distance, direction)
        """
        num_obs = random.randint(1, len(landmarks))
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

    def compute_jacobians(self, landmark):
        dx = landmark.pos_x - self.pos_x
        dy = landmark.pos_y - self.pos_y
        d2 = dx**2 + dy**2
        d = math.sqrt(d2)

        predicted_obs = np.array([[d],[(math.atan2(dy, dx) + 2 * math.pi) % (2 * math.pi)]])
        jacobian = np.array([[dx/d,   dy/d],
                             [-dy/d2, dx/d2]])
        adj_cov = jacobian.dot(landmark.sig).dot(np.transpose(jacobian)) + self.obs_noise
        return predicted_obs, jacobian, adj_cov

    def guess_landmark(self, obs):
        """Based on the particle position and observation, guess the location of the landmark. Origin at top left"""
        distance, direction = obs
        lm_x = self.pos_x + distance * math.cos(direction)
        lm_y = self.pos_y + distance * math.sin(direction)
        return Landmark(lm_x, lm_y)

    def find_data_association(self, obs):
        """Using maximum likelihood to find data association"""
        prob = 0
        ass_obs = np.zeros((2,1))
        ass_jacobian = np.zeros((2,2))
        ass_adjcov = np.zeros((2,2))
        landmark_idx = -1
        for idx, landmark in enumerate(self.landmarks):
            predicted_obs, jacobian, adj_cov = self.compute_jacobians(landmark)
            p = multi_normal(np.transpose(np.array([obs])), predicted_obs, adj_cov)
            if p > prob:
                prob = p
                ass_obs = predicted_obs
                ass_jacobian = jacobian
                ass_adjcov = adj_cov
                landmark_idx = idx
        return prob, landmark_idx, ass_obs, ass_jacobian, ass_adjcov

    def create_landmark(self, obs):
        landmark = guess_landmark(self.pos_x, self.pos_y, obs)
        self.landmarks.append(landmark)

    def update_landmark(self, obs, landmark_idx, ass_obs, ass_jacobian, ass_adjcov):
        landmark = self.landmarks[landmark_idx]
        K = landmark.sig.dot(np.transpose(ass_jacobian)).dot(linalg.inv(ass_adjcov))
        new_mu = landmark.mu + K.dot(obs - ass_obs)
        new_sig = (np.eye(2) - K.dot(ass_jacobian)).dot(landmark.sig)
        landmark.update(new_mu, new_sig)




