import numpy as np

class Landmark(object):
    """Data structure of a landmark associated with a particle.
       Origin is the left-bottom point
    """
    def __init__(self, x, y):
        self.mu = np.array([[x],[y]])
        self.sig = np.eye(2) * 999

