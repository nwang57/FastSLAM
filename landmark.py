import numpy as np

class Landmark(object):
    """Data structure of a landmark associated with a particle.
       Origin is the left-bottom point
    """
    def __init__(self, x, y):
        self.pos_x = x
        self.pos_y = y
        self.mu = np.array([[self.pos_x],[self.pos_y]])
        self.sig = np.eye(2) * 99

    def pos(self):
        return (self.pos_x, self.pos_y)

    def update(self, mu, sig):
        self.mu = mu
        self.sig = sig
        self.pos_x = self.mu[0][0]
        self.pos_y = self.mu[1][0]

    def __str__(self):
        return str(self.pos())
