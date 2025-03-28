import numpy as np
import matplotlib.pyplot as plt

# Параметры Леннард-Джонса
EPSILON = 1.0
SIGMA = 2.0
CUTOFF = 2.5 * SIGMA  # Радиус отсечения
DT = 0.1  # Шаг интегрирования
DAMPING = 0.98  # Коэффициент затухания скорости


class ParticleSystem:
    def __init__(self, num_particles, box_size):
        self.num_particles = num_particles
        self.box_size = box_size
        self.positions = np.random.rand(num_particles, 2) * box_size
        self.velocities = np.random.randn(num_particles, 2) * 0.1

    def lennard_jones_force(self, r):
        """Вычисляет силу Леннард-Джонса."""
        if r > CUTOFF:
            return 0.0
        r6 = (SIGMA / r) ** 6
        r12 = r6 ** 2
        force = 24 * EPSILON * (2 * r12 - r6) / r
        return force

    def compute_forces(self):
        """Вычисляет силы между всеми частицами с учетом периодических условий."""
        forces = np.zeros((self.num_particles, 2))
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                r_vec = self.positions[j] - self.positions[i]
                r_vec -= np.round(r_vec / self.box_size) * self.box_size  # Периодические границы
                r = np.linalg.norm(r_vec)
                if r > 1e-5:
                    f = self.lennard_jones_force(r) * (r_vec / r)
                    forces[i] += f
                    forces[j] -= f
        return forces

    def update(self):
        """Обновляет положение частиц."""
        forces = self.compute_forces()
        self.velocities += forces * DT
        self.velocities *= DAMPING  # Затухание скорости
        self.positions += self.velocities * DT

        # Применение периодических граничных условий
        self.positions %= self.box_size

    def run_simulation(self, steps):
        """Запускает симуляцию и отображает визуализацию."""
        plt.ion()
        fig, ax = plt.subplots()
        for _ in range(steps):
            self.update()
            ax.clear()
            ax.set_xlim(0, self.box_size)
            ax.set_ylim(0, self.box_size)
            ax.scatter(self.positions[:, 0], self.positions[:, 1])
            plt.pause(0.01)
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    # Запуск симуляции
    system = ParticleSystem(num_particles=20, box_size=10)
    system.run_simulation(steps=50)
