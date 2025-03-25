import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import threading

# Для визуализации через OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

class ParticleSystem:
    def __init__(self, num_particles, box_size, radius):
        """
        Инициализация системы частиц.
        :param num_particles: количество частиц
        :param box_size: размер бокса (считаем квадратным)
        :param radius: радиус частицы (будет использован как σ в потенциале)
        """
        self.num_particles = num_particles
        self.box_size = box_size
        self.radius = radius
        self.positions = self.generate_particles()

    def generate_particles(self):
        """Генерирует начальные позиции частиц случайным образом внутри бокса."""
        return np.random.rand(self.num_particles, 2) * self.box_size

    def compute_energy(self, positions_flat):
        """
        Вычисляет суммарную энергию системы по потенциалу Леннарда-Джонса.
        :param positions_flat: одномерный массив позиций (для передачи в оптимизатор)
        :return: суммарная энергия системы
        """
        positions = positions_flat.reshape((-1, 2))
        energy = 0.0
        # Для каждой пары частиц
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                dx = positions[i] - positions[j]
                # Реализация периодических граничных условий через метод минимального изображения
                dx = dx - self.box_size * np.rint(dx / self.box_size)
                r = np.linalg.norm(dx)
                if r == 0:
                    continue
                sigma = self.radius
                energy += 4 * ((sigma / r) ** 12 - (sigma / r) ** 6)
        return energy

    def compute_gradient(self, positions_flat):
        """
        Вычисляет градиент энергии системы.
        :param positions_flat: одномерный массив позиций
        :return: градиент (одномерный массив той же размерности)
        """
        positions = positions_flat.reshape((-1, 2))
        grad = np.zeros_like(positions)
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                dx = positions[i] - positions[j]
                dx = dx - self.box_size * np.rint(dx / self.box_size)
                r = np.linalg.norm(dx)
                if r == 0:
                    continue
                sigma = self.radius
                factor = 4 * (-12 * (sigma ** 12) / r ** 14 + 6 * (sigma ** 6) / r ** 8)
                grad_i = factor * dx
                grad[i] += grad_i
                grad[j] -= grad_i
        return grad.flatten()


class EnergyMinimizer:
    def __init__(self, particle_system):
        """
        Инициализация минимизатора энергии.
        :param particle_system: объект класса ParticleSystem
        """
        self.system = particle_system
        self.energy_history = []

    def callback(self, xk):
        """Callback-функция, сохраняющая энергию на каждой итерации минимизации."""
        energy = self.system.compute_energy(xk)
        self.energy_history.append(energy)

    def minimize_energy(self):
        """Минимизирует энергию системы с использованием метода BFGS."""
        result = minimize(
            self.system.compute_energy,
            self.system.positions.flatten(),
            jac=self.system.compute_gradient,
            method='BFGS',
            callback=self.callback,
            options={'disp': True}
        )
        self.system.positions = result.x.reshape((-1, 2))
        return result


class Visualizer:
    def __init__(self, particle_system, energy_history):
        """
        Инициализация визуализатора.
        :param particle_system: объект ParticleSystem с финальной конфигурацией
        :param energy_history: история изменения энергии системы
        """
        self.system = particle_system
        self.energy_history = energy_history

    def plot_energy(self):
        """Строит график изменения энергии по итерациям минимизации (Matplotlib)."""
        plt.figure("Энергия")
        plt.plot(self.energy_history, marker='o')
        plt.xlabel("Итерация")
        plt.ylabel("Энергия")
        plt.title("Изменение энергии системы")
        plt.grid(True)
        plt.show(block=False)

    def plot_delaunay(self):
        """Строит триангуляцию Делоне для финальной конфигурации (Matplotlib)."""
        tri = Delaunay(self.system.positions)
        plt.figure("Делоне")
        plt.triplot(self.system.positions[:, 0], self.system.positions[:, 1], tri.simplices.copy(), color='blue')
        plt.plot(self.system.positions[:, 0], self.system.positions[:, 1], 'ro')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Триангуляция Делоне")
        plt.grid(True)
        plt.show(block=False)

    def draw_circle(self, x, y, radius, segments=20):
        """Рисует круг с центром (x,y) и заданным радиусом."""
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            glVertex2f(x + np.cos(angle) * radius, y + np.sin(angle) * radius)
        glEnd()

    def opengl_visualization(self):
        """
        Визуализация с использованием OpenGL (через pygame).
        Отрисовываются круги (частицы) и линии триангуляции Делоне.
        """
        pygame.init()
        display = (600, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.system.box_size, 0, self.system.box_size)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Отрисовка частиц как кругов
            glColor3f(1.0, 0.0, 0.0)
            for pos in self.system.positions:
                self.draw_circle(pos[0], pos[1], self.system.radius)

            # Триангуляция Делоне и отрисовка линий
            tri = Delaunay(self.system.positions)
            glColor3f(0.0, 1.0, 0.0)
            glBegin(GL_LINES)
            for simplex in tri.simplices:
                for i in range(3):
                    j = (i + 1) % 3
                    glVertex2f(self.system.positions[simplex[i], 0], self.system.positions[simplex[i], 1])
                    glVertex2f(self.system.positions[simplex[j], 0], self.system.positions[simplex[j], 1])
            glEnd()

            pygame.display.flip()
            pygame.time.wait(10)
        pygame.quit()


if __name__ == '__main__':
    # Параметры системы
    num_particles = 10     # число частиц
    box_size = 10.0        # размер бокса
    radius = 0.2          # радиус частицы (используется как σ в LJ-потенциале)

    # Создаем систему частиц
    system = ParticleSystem(num_particles, box_size, radius)

    # Минимизируем энергию системы
    minimizer = EnergyMinimizer(system)
    result = minimizer.minimize_energy()
    print("Минимизация завершена. Итоговая энергия:", result.fun)

    # Создаем объект визуализации
    visualizer = Visualizer(system, minimizer.energy_history)

    # Запуск графиков Matplotlib в неблокирующем режиме
    visualizer.plot_energy()    # окно с графиком изменения энергии
    visualizer.plot_delaunay()  # окно с триангуляцией Делоне

    # Запуск OpenGL визуализации в отдельном потоке
    opengl_thread = threading.Thread(target=visualizer.opengl_visualization)
    opengl_thread.start()

    # Чтобы Matplotlib окна не закрылись сразу, можно использовать цикл ожидания
    # или просто запустить бесконечный цикл с периодической проверкой.
    try:
        while opengl_thread.is_alive():
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass
