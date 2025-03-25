import random
import math
from config import SIGMA, EPSILON

#circle.py
class Circle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    @classmethod
    def random(cls):
        x = random.uniform(-0.5, 0.5)
        y = random.uniform(-0.5, 0.5)
        return cls(x, y, 0.05)

