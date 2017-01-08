import numpy as np
import math
import random

def gauss_noise(mu, sig):
    return random.gauss(mu, sig)

def euclidean_distance(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])

def cal_direction(a, b):
    """Calculate the angle of the vector a to b, [0, 2*pi)"""
    return (math.atan2(b[0]-a[0], b[1]-a[1]) + 2 * math.pi) % (2 * math.pi)
