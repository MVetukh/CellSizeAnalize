# system.py

from circle import Circle
from config import LEARNING_RATE, BETA1, BETA2, EPSILON_ADAM, STEPS
import math

class System:
    def __init__(self, num_circles):
        self.circles = [Circle.random() for _ in range(num_circles)]
        self.m = [[0, 0] for _ in range(num_circles)]
        self.v = [[0, 0] for _ in range(num_circles)]
        self.t = 0

    def compute_energy(self):
        energy = 0.0
        for i in range(len(self.circles)):
            for j in range(i + 1, len(self.circles)):
                energy += self.circles[i].lennard_jones_potential(self.circles[j])
        return energy

    def update(self):
        self.t += 1
        forces = [[0, 0] for _ in range(len(self.circles))]
        for i in range(len(self.circles)):
            for j in range(i + 1, len(self.circles)):
                fx, fy = self.circles[i].lj_force(self.circles[j])
                forces[i][0] += fx
                forces[i][1] += fy
                forces[j][0] -= fx
                forces[j][1] -= fy

        # Применение оптимизации Adam
        for i in range(len(self.circles)):
            for d in range(2):
                self.m[i][d] = BETA1 * self.m[i][d] + (1 - BETA1) * forces[i][d]
                self.v[i][d] = BETA2 * self.v[i][d] + (1 - BETA2) * (forces[i][d] ** 2)
                m_hat = self.m[i][d] / (1 - BETA1 ** self.t)
                v_hat = self.v[i][d] / (1 - BETA2 ** self.t)
                self.circles[i].x -= LEARNING_RATE * m_hat / (math.sqrt(v_hat) + EPSILON_ADAM)
                self.circles[i].y -= LEARNING_RATE * m_hat / (math.sqrt(v_hat) + EPSILON_ADAM)

    def run_optimization(self, steps=STEPS):
        energy_history = []
        for _ in range(steps):
            self.update()
            energy_history.append(self.compute_energy())
        return energy_history
