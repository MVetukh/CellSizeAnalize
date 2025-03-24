import random
import math
from config import SIGMA, EPSILON


class Circle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.box_size = 800
        self.charge = 1

    @classmethod
    def random(cls):
        x = random.uniform(-0.5, 0.5)
        y = random.uniform(-0.5, 0.5)
        radius = 0.05  # abs(random.gauss(0.05, 0.02))  # Радиусы из гауссовского распределения
        return cls(x, y, radius)

    def distance_squared(self, other):
        """Вычисление квадрата расстояния между двумя частицами"""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def lennard_jones_potential(self, other):
        """Вычисление электрического потенциала между двумя частицами"""
        r = self.distance_squared(other)
        delta = 1e-6
        if r <= 0:  # Исключение для одинаковых точек (если рентгеновский эффект, как в потенциале Леннард-Джонса)
            return float('inf')

        # Потенциал по закону Кулона
        potential =  self.charge * other.charge / r
        return potential

    def lj_force(self, other):
        """Вычисление силы между двумя частицами через закон Кулона"""
        r = self.distance_squared(other)
        delta = 1e-6
        if r <= delta:  # Исключение для одинаковых точек
            return 0, 0

        # Сила по закону Кулона
        force_magnitude = self.charge * other.charge / (r ** 2)

        # Направление силы
        dx = self.x - other.x
        dy = self.y - other.y

        # Нормализация направления силы
        dist = self.distance_squared(other)
        fx = force_magnitude * dx / dist
        fy = force_magnitude * dy / dist

        return fx, fy