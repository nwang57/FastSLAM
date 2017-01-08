import numpy as np
import math
import random

def gauss_noise(mu, sig):
    return random.gauss(mu, sig)

def euclidean_distance(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])

def cal_direction(a, b):
    """Calculate the angle of the vector a to b, [0, 2*pi)"""
    return (math.atan2(b[1]-a[1], b[0]-a[0]) + 2 * math.pi) % (2 * math.pi)

def guess_landmark(x, y, obs):
    """Based on the particle position and observation, guess the location of the landmark. Origin at top left"""
    distance, direction = obs
    lm_x = x + distance * math.cos(direction)
    lm_y = y - distance * math.sin(direction)
    return lm_x, lm_y
