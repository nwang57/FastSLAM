"""
    This is the main file that starts the simulation.
    It contains a "World" object specifying the world settings, and a set of particles.
    Every time the robot moves, it generates random observations usded to update the particle sets
    to estimate the robot path as well as the landmarks locations.
"""
import sys
from world import World
from particle import Particle

class FastSlam(object):
    """Main class that implements the FastSLAM1.0 algorithm"""
    def __init__(self, particle_size = 500):
        self.world = World()
        self.particles = [Particle() for i in xrange(particle_size)]
        self.robot = Particle(is_robot=True)

    def run_simulation(self):
        self.robot.set_pos(100, 100, 0)
        while True:
            for event in self.world.pygame.event.get():
                self.world.test_end(event)
            self.world.clear()
            key_pressed = self.world.pygame.key.get_pressed()
            if self.world.move_forward(key_pressed):
                self.robot.forward(1)
            if self.world.turn_left(key_pressed):
                self.robot.turn_left(5)
            if self.world.turn_right(key_pressed):
                self.robot.turn_right(5)
            self.world.render(self.robot, self.particles)

if __name__=="__main__":
    simulator = FastSlam()
    simulator.run_simulation()
