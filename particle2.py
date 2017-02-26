import random
import math
from operator import itemgetter
import numpy as np
from slam_helper import *
from scipy import linalg
from world import WINDOWWIDTH, WINDOWHEIGHT
from landmark import Landmark
from particle import Particle

class Particle2(Particle):
    """Inherit from Particle. Incorporates latest obs in the proposal distribution"""
    def __init__(self, x, y, orien, is_robot=False):
        super(Particle2, self).__init__(x, y, orien, is_robot)
        self.control_noise = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, (3.0*math.pi/180)**2]])
        self.obs_noise = np.array([[0.1, 0], [0, (3.0*math.pi/180)**2]])

    def update(self, obs):
        """After the motion, update the weight of the particle and its EKFs based on the sensor data"""
        # Find data association first
        data_association = []
        for o in obs:
            prob = np.exp(-70)
            landmark_idx = -1
            if self.landmarks:
                # find the data association with ML
                prob, landmark_idx = self.pre_compute_data_association(o)
                if prob < self.TOL:
                    # create new landmark
                    landmark_idx = -1
            data_association.append((o, landmark_idx, prob))
        # Incorporates obs that creates new features last
        data_association.sort(key=itemgetter(1), reverse=True)
        # incorporate multiple obs to get the proposal distribution
        initial_pose = np.array([[self.pos_x], [self.pos_y], [self.orientation]])
        pose_mean = initial_pose
        pose_cov = self.control_noise
        for da in data_association:
            if da[1] > -1:
                # Using EKF to update the robot pose
                predicted_obs, featurn_jacobian, pose_jacobian, adj_cov = self.compute_jacobians(self.landmarks[da[1]])
                pose_cov = linalg.inv( np.transpose(pose_jacobian).dot(linalg.inv(adj_cov).dot(pose_jacobian)) + linalg.inv(pose_cov) )
                pose_mean = np.array([[self.pos_x], [self.pos_y], [self.orientation]]) + pose_cov.dot(np.transpose(pose_jacobian)).dot(linalg.inv(adj_cov)).dot(np.transpose(np.array([da[0]])) - predicted_obs)
                new_pose = np.random.multivariate_normal(pose_mean[:,0], pose_cov)
                self.set_pos(*new_pose)
            else:
                # Using the latest pose to create the landmark
                self.create_landmark(da[0])

        # update the landmarks EKFs
        for da in data_association:
            if da[1] > -1:
                predicted_obs, featurn_jacobian, pose_jacobian, adj_cov = self.compute_jacobians(self.landmarks[da[1]])
                self.weight *= multi_normal(np.transpose(np.array([da[0]])), predicted_obs, adj_cov)
                self.update_landmark(np.transpose(np.array([da[0]])), da[1], predicted_obs, featurn_jacobian, adj_cov)
            else:
                self.weight *= da[2]
        prior = multi_normal(np.array([[self.pos_x], [self.pos_y], [self.orientation]]), initial_pose, self.control_noise)
        prop = multi_normal(np.array([[self.pos_x], [self.pos_y], [self.orientation]]), pose_mean, pose_cov)
        self.weight = self.weight * prior / prop

    def compute_jacobians(self, landmark):
        dx = landmark.pos_x - self.pos_x
        dy = landmark.pos_y - self.pos_y
        d2 = dx**2 + dy**2
        d = math.sqrt(d2)

        predicted_obs = np.array([[d],[math.atan2(dy, dx)]])
        feature_jacobian = np.array([[dx/d,   dy/d],
                                     [-dy/d2, dx/d2]])
        pose_jacobian = np.array([[-dx/d, -dy/d, 0],
                                  [dy/d2, -dx/d2, -1]])
        adj_cov = feature_jacobian.dot(landmark.sig).dot(np.transpose(feature_jacobian)) + self.obs_noise
        return predicted_obs, feature_jacobian, pose_jacobian, adj_cov

    def pre_compute_data_association(self, obs):
        """Tries all the landmarks to incorporate the obs to get the proposal distribution and get the one with maximum likelihood"""
        prob = 0
        ass_obs = np.zeros((2,1))
        ass_jacobian = np.zeros((2,2))
        ass_adjcov = np.zeros((2,2))
        landmark_idx = -1
        for idx, landmark in enumerate(self.landmarks):
            # Sample a new particle pose from the proposal distribution
            predicted_obs, featurn_jacobian, pose_jacobian, adj_cov = self.compute_jacobians(landmark)
            pose_cov = linalg.inv( np.transpose(pose_jacobian).dot(linalg.inv(adj_cov).dot(pose_jacobian)) + linalg.inv(self.control_noise) )
            pose_mean = np.array([[self.pos_x], [self.pos_y], [self.orientation]]) + pose_cov.dot(np.transpose(pose_jacobian)).dot(linalg.inv(adj_cov)).dot(np.transpose(np.array([obs])) - predicted_obs)
            new_pose = np.random.multivariate_normal(pose_mean[:,0], pose_cov)

            distance = euclidean_distance((landmark.pos_x, landmark.pos_y), (new_pose[0], new_pose[1]))
            direction = cal_direction((new_pose[0], new_pose[1]), (landmark.pos_x, landmark.pos_y))
            p = multi_normal(np.transpose(np.array([obs])), np.array([[distance],[direction]]), adj_cov)
            if p > prob:
                prob = p
                landmark_idx = idx
        return prob, landmark_idx

    def update_landmark(self, obs, landmark_idx, ass_obs, ass_jacobian, ass_adjcov):
        landmark = self.landmarks[landmark_idx]
        K = landmark.sig.dot(np.transpose(ass_jacobian)).dot(linalg.inv(ass_adjcov))
        new_mu = landmark.mu + K.dot(obs - ass_obs)
        new_sig = (np.eye(2) - K.dot(ass_jacobian)).dot(landmark.sig)
        landmark.update(new_mu, new_sig)
