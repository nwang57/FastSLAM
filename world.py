import pygame
import sys
from pygame.locals import *
from landmark import Landmark

FPS=30
WINDOWWIDTH = 200
WINDOWHEIGHT = 200
COLOR = { "white": (255, 255, 255),
          "black": (0, 0, 0),
          "green": (0, 255, 0),
          "blue": (0, 0, 255),
          "red": (255, 0, 0),
          "purple": (128, 0, 128)
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
        """Set up landmarks, origin is the left-bottom point"""
        l1 = Landmark(10, 10)
        l2 = Landmark(25, 60)
        l3 = Landmark(100, 50)
        l4 = Landmark(110, 100)
        l5 = Landmark(160, 88)
        l6 = Landmark(170, 180)
        self.landmarks = [l1, l2, l3, l4, l5, l6]

    def draw(self, robot, particles, landmarks):
        """Draw the objects in the window"""
        for landmark in self.landmarks:
            self.pygame.draw.circle(self.window, COLOR["green"], self.convert_coordinates(landmark.pos()), 3)
        self.pygame.draw.circle(self.window, COLOR["blue"], self.convert_coordinates(robot.pos()), 7)
        self.pygame.draw.line(self.window, COLOR["green"], *[self.convert_coordinates(pos) for pos in robot.dick()])
        for p in particles:
            self.pygame.draw.circle(self.window, COLOR["red"], self.convert_coordinates(p.pos()), 2)
        for l in landmarks:
            self.pygame.draw.circle(self.window, COLOR["purple"], self.convert_coordinates(l.pos()), 2)

    def convert_coordinates(self, pos):
        """Change the origin from bottom left to top left"""
        return (int(pos[0]), int(WINDOWHEIGHT - pos[1]))

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

    def render(self, robot, particles, landmarks):
        self.draw(robot, particles, landmarks)
        self.fpsClock.tick(FPS)
        self.pygame.display.update()

