import pygame
import sys
from pygame.locals import *

FPS=30
WINDOWWIDTH = 800
WINDOWHEIGHT = 600
COLOR = { "white": (255, 255, 255),
          "black": (0, 0, 0),
          "green": (0, 255, 0),
          "blue": (0, 0, 255),
          "red": (255, 0, 0)
        }

class World(object):
    """Implement the pygame simulator, drawing and rendering stuff"""
    def __init__(self):
        self.pygame = pygame
        self.fpsClock = self.pygame.time.Clock()
        self.main_clock = self.pygame.time.Clock()
        self.window = self.pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), 0, 32)
        self.pygame.display.set_caption("FastSLAM")
        self.setup_world()

    def setup_world(self):
        """Set up landmarks"""
        l1 = (10, 10)
        l2 = (25, 60)
        l3 = (100, 50)
        l4 = (110, 100)
        self.landmarks = [l1, l2, l3, l4]

    def draw(self, robot, particles):
        """Draw the objects in the window"""
        for landmark in self.landmarks:
            self.pygame.draw.circle(self.window, COLOR["green"], landmark, 3)
        pygame.draw.circle(self.window, COLOR["blue"], (int(robot.pos_x), int(robot.pos_y)), 7)
        pygame.draw.line(self.window, COLOR["green"], *robot.dick())

    def test_end(self, event):
        if event.type == QUIT:
            self.pygame.quit()
            sys.exit()

    def clear(self):
        """Fill the background white"""
        self.window.fill(COLOR["white"])

    def move_forward(self, key_pressed):
        """Test the motion command UP"""
        return key_pressed[K_UP]

    def turn_left(self, key_pressed):
        """Test the motion command LEFT"""
        return key_pressed[K_LEFT]

    def turn_right(self, key_pressed):
        """Test the motion command RIGHT"""
        return key_pressed[K_RIGHT]

    def render(self, robot, particles):
        self.draw(robot, particles)
        self.fpsClock.tick(FPS)
        self.pygame.display.update()

