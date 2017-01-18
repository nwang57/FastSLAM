import numpy as np

class Landmark(object):
    """Data structure of a landmark associated with a particle.
       Origin is the left-bottom point
    """
    def __init__(self, x, y):
        self.pos_x = x
        self.pos_y = y
        self.mu = np.array([[self.pos_x],[self.pos_y]])
        self.sig = np.eye(2) * 999

    def pos(self):
        return (self.pos_x, self.pos_y)
