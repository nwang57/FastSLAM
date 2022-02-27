"""
    This is the main file that starts the simulation.
    It contains a "World" object specifying the world settings, and a set of particles.
    Every time the robot moves, it generates random observations usded to update the particle sets
    to estimate the robot path as well as the landmarks locations.
"""
import sys
import random
import math
from copy import deepcopy
from world import World
from particle2 import Particle2
from fast_slam import FastSlam

class FastSlam2(FastSlam):
    """Inherit from FastSlam"""
    def __init__(self, x, y, orien, particle_size = 50):
        self.world = World()
        self.particles = [Particle2(x, y, random.random()* 2.*math.pi) for i in range(particle_size)]
        self.robot = Particle2(x, y, orien, is_robot=True)
        self.particle_size = particle_size

if __name__=="__main__":
    random.seed(5)
    simulator = FastSlam2(80, 140, 0, particle_size=200)
    simulator.run_simulation()
