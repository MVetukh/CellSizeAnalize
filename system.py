import torch
import torch.optim as optim
import math
from OpenGL.GL import *

class System:
    def __init__(self, num_circles, width, height, particle_radius=0.05):
        self.num_circles = num_circles
        self.width = width
        self.height = height
        self.particle_radius = particle_radius  # Радиус частиц

        # Инициализация частиц

        self.positions = torch.nn.Parameter(torch.rand(num_circles, 2) * 2 - 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Оптимизатор
        self.optimizer = optim.Adam([self.positions], lr=0.001)

        # Параметры потенциала Леннард-Джонса
        self.sigma = 1.0
        self.epsilon = 1.0

    def lj_potential(self, r):
        """Потенциал Леннард-Джонса"""
        sig_r = self.sigma / r
        return 4 * self.epsilon * (sig_r ** 12 - sig_r ** 6)

    def total_energy(self):
        """Вычисление полной энергии системы"""
        energy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        num_particles = len(self.positions)

        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                dx = self.positions[i, 0] - self.positions[j, 0]
                dy = self.positions[i, 1] - self.positions[j, 1]
                r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)  # Добавляем малое число для стабильности
                energy = energy + self.lj_potential(r)

        return energy

    def update(self):
        """Шаг оптимизации"""
        self.optimizer.zero_grad()
        energy = self.total_energy()
        energy.backward()
        self.optimizer.step()
        return energy.item()

    def optimize(self, steps=1000):
        """Оптимизация системы"""
        energy_history = []
        for step in range(steps):
            energy = self.update()
            energy_history.append(energy)
            if step % 100 == 0:
                print(f"Step {step}, Energy: {energy:.6f}")

        print("Optimization finished.")
        return energy_history

    def update_projection(self):
        """Обновление проекции OpenGL"""
        positions = self.positions.detach().cpu().numpy()
        if len(positions) == 0:
            return

        min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
        min_y, max_y = positions[:, 1].min(), positions[:, 1].max()

        margin = 0.1
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(
            min_x - margin, max_x + margin,
            min_y - margin, max_y + margin,
            -1, 1
        )
