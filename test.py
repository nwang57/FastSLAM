import unittest
import math
from particle import Particle
from landmark import Landmark
from slam_helper import *
import numpy as np

class FastSLAMTest(unittest.TestCase):
    def test_guess_landmark(self):
        robot = Particle(is_robot=True)
        robot.set_pos(100, 100, 0)

        landmark = Landmark(20, 20)
        landmarks = [landmark]
        obs = robot.sense(landmarks)
        self.assertEqual(obs[0][0], math.hypot(100-20, 100-20))

        landmark = robot.guess_landmark(obs[0])
        self.assertAlmostEqual(landmark.pos_x, 20)
        self.assertAlmostEqual(landmark.pos_y, 20)

    def test_data_association(self):
        robot = Particle(is_robot=True)
        robot.set_pos(100, 100, 0)

        landmark1 = Landmark(20, 20)
        landmark2 = Landmark(40, 60)
        landmark_fake2 = Landmark(45, 60)
        landmark3 = Landmark(20, 110)
        robot.landmarks = [landmark1, landmark_fake2, landmark3]
        obs = robot.sense([landmark2])
        for o in obs:
            prob, idx, ass_obs, ass_jacobian, ass_adjcov = robot.find_data_association(o)
        self.assertEqual(idx, 1)

    def test_update_EKF(self):
        robot = Particle(is_robot=True)
        robot.set_pos(100, 100, 0)

        landmark1 = Landmark(20, 20)
        landmark2 = Landmark(40, 60)
        landmark_fake2 = Landmark(45, 60)
        landmark3 = Landmark(20, 110)
        robot.landmarks = [landmark1, landmark_fake2, landmark3]

        obs = robot.sense([landmark2])
        for o in obs:
            prob, idx, ass_obs, ass_jacobian, ass_adjcov = robot.find_data_association(o)
            self.assertEqual(robot.landmarks[idx].pos(), (45, 60))
            robot.update_landmark(np.transpose(np.array([o])), idx, ass_obs, ass_jacobian, ass_adjcov)
            self.assertAlmostEqual(robot.landmarks[idx].pos_x, 39.89615124708)
            self.assertAlmostEqual(robot.landmarks[idx].pos_y, 60.04079368286)
