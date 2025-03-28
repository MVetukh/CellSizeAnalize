import matplotlib

matplotlib.use('TkAgg')
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import threading
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
from scipy.optimize import check_grad

class ParticleSystem:
    def __init__(self, num_particles, box_size, radius):
        self.num_particles = num_particles
        self.box_size = box_size
        self.radius = radius
        self.positions = self.generate_particles()


    def generate_particles(self):
        return np.random.rand(self.num_particles, 2) * self.box_size

    def compute_energy(self, positions_flat, DEBUG=False):
        positions = positions_flat.reshape((-1, 2))
        energy = 0.0
        epsilon = 1e-6  # Маленькое число для стабильности

        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                dx = positions[i] - positions[j]
                dx -= self.box_size * np.rint(dx / self.box_size)  # PBC
                r = np.linalg.norm(dx) + epsilon

                if r < 1e-3:
                    continue

                sigma = self.radius
                lj_potential = 4 * ((sigma / r) ** 12 - (sigma / r) ** 6)

                # # Добавление экспоненциального отталкивания при r < 2 * sigma
                # if r < 2 * self.radius:
                #     repulsion = np.exp(2 * self.radius - r)
                # else:
                #     repulsion = 0

                energy += lj_potential #+ repulsion

                if DEBUG:
                    print(
                        f"Pair ({i}, {j}): r = {r:.5f}, LJ = {lj_potential:.5f},, Total E = {energy:.5f}")
        #  Repulsion = {repulsion:.5f}
        return energy

    def compute_gradient(self, positions_flat, DEBUG=False):
        positions = positions_flat.reshape((-1, 2))
        grad = np.zeros_like(positions)
        epsilon = 1e-8  # Маленькое число для стабильности
        r_min = 1e-3  # Минимальное расстояние для расчета

        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                dx = positions[i] - positions[j]
                dx -= self.box_size * np.rint(dx / self.box_size)  # PBC
                r = np.linalg.norm(dx) + epsilon

                if r < 1e-6:
                    continue

                sigma = self.radius
                lj_force = 4 * (-12 * (sigma ** 12) / r ** 14 + 6 * (sigma ** 6) / r ** 8)

                # Производная отталкивающего потенциала
                repulsion_force = 0
                # if r < 2 * self.radius:
                #     repulsion_force = np.exp(2 * self.radius - r) / r

                total_force = lj_force + repulsion_force
                grad_i = total_force * dx
                grad[i] += grad_i
                grad[j] -= grad_i

                if DEBUG:
                    print(
                        f"Pair ({i}, {j}): r = {r:.5f}, LJ_force = {lj_force:.5f}, Repulsion_force = {repulsion_force:.5f}, Total_force = {total_force:.5f}")

        return grad.flatten()


class EnergyMinimizer:
    def __init__(self, particle_system):
        self.system = particle_system
        self.energy_history = []

    def callback(self, xk):
        energy = self.system.compute_energy(xk,DEBUG=True)
        grad_norm = np.linalg.norm(self.system.compute_gradient(xk,DEBUG=True))
        print(f"Iteration {len(self.energy_history)}: Energy = {energy:.6f}, ||Grad|| = {grad_norm:.6f}")
        self.energy_history.append(energy)

    def minimize_energy(self):
        # Начальные настройки (константы заданы внутри функции)
        x = self.system.positions.flatten()
        lr = 1e-3  # начальный learning rate
        momentum_coef = 0.6  # коэффициент momentum
        velocity = np.zeros_like(x)  # инициализация скорости (momentum)
        maxiter = 1000  # максимальное число итераций
        tol = 1e-9 # порог изменения энергии для адаптации lr
        gtol = 1e-8  # порог нормы градиента для сходимости

        energy_prev = self.system.compute_energy(x)

        for i in range(maxiter):
            grad = self.system.compute_gradient(x)
            grad_norm = np.linalg.norm(grad)

            # Динамическое условие выхода: если норма градиента мала
            if grad_norm < gtol:
                print(f"Converged at iteration {i}: grad_norm = {grad_norm:.2e}")
                break

            # Обновление с учетом momentum
            velocity = momentum_coef * velocity - lr * grad
            x_new = x + velocity
            energy_new = self.system.compute_energy(x_new)

            # Если энергия возросла, уменьшаем шаг и не обновляем положение
            if energy_new > energy_prev:
                lr *= 0.5
                print(
                    f"Iteration {i}: energy increased from {energy_prev:.6e} to {energy_new:.6e}, reducing lr to {lr:.2e}")
            else:
                # Если изменение энергии очень мало, слегка уменьшаем lr
                if abs(energy_prev - energy_new) < tol:
                    lr *= 0.9
                x = x_new
                energy_prev = energy_new
                self.callback(x)

        # Обновляем позиции системы
        self.system.positions = x.reshape((-1, 2))

        # Формируем результат в виде словаря (аналог OptimizeResult)
        result = {
            "fun": energy_prev,
            "nit": i + 1,
            "message": "Converged" if grad_norm < gtol else "Max iterations reached"
        }

        print(f"Статус сходимости: {result['message']}")
        print(f"Итераций выполнено: {result['nit']}")

        return result


class Visualizer:
    def __init__(self, particle_system, energy_history):
        self.system = particle_system
        self.energy_history = energy_history

    def plot_energy(self):
        """Строит график энергии с явным заданием осей."""
        plt.figure("Энергия системы")

        # Явно задаем данные для осей
        iterations =  list(range(len(self.energy_history)))
        energy_values = self.energy_history

        plt.plot(iterations, energy_values, 'b-', linewidth=2)
        plt.xlabel("Номер итерации", fontsize=12)
        plt.ylabel("Энергия системы", fontsize=12)
        plt.title("Зависимость энергии от итерации", fontsize=14)

        # Добавляем сетку и логарифмический масштаб при необходимости
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        if max(energy_values) / min(energy_values) > 1e3:
            plt.yscale('log')
            plt.ylabel("Энергия (log scale)")

        # Добавляем аннотацию финального значения
        final_energy = energy_values[-1]
        plt.annotate(f'Финальная энергия: {final_energy:.2e}',
                     xy=(0.7, 0.1),
                     xycoords='axes fraction',
                     bbox=dict(boxstyle="round", fc="w"))

        plt.tight_layout()
        plt.show(block=False)

    def _create_periodic_copies(self, positions, box_size):
        copies = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                copies.append(positions + [dx * box_size, dy * box_size])
        return np.vstack(copies)

    def plot_delaunay(self):
        """Строит триангуляцию Делоне без периодических копий."""
        # Нормализуем позиции частиц
        normalized_pos = self.system.positions % self.system.box_size

        # Строим триангуляцию только для основных частиц
        tri = Delaunay(normalized_pos)

        # Визуализация
        plt.figure("Делоне")
        plt.triplot(normalized_pos[:, 0], normalized_pos[:, 1], tri.simplices.copy(), color='blue')
        plt.plot(normalized_pos[:, 0], normalized_pos[:, 1], 'ro', markersize=4)
        plt.xlim(0, self.system.box_size)
        plt.ylim(0, self.system.box_size)
        plt.title("Триангуляция Делоне")
        plt.grid(True)
        plt.show(block=False)

    def draw_circle(self, x, y, radius, segments=32):
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            glVertex2f(x + np.cos(angle) * radius, y + np.sin(angle) * radius)
        glEnd()

    def draw_circle_outline(self, x, y, radius, segments=32):
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            glVertex2f(x + np.cos(angle) * radius, y + np.sin(angle) * radius)
        glEnd()

    def opengl_visualization(self):
        try:
            pygame.init()
            box_size = self.system.box_size
            window_size = 800
            screen = pygame.display.set_mode((window_size, window_size), DOUBLEBUF | OPENGL)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluOrtho2D(0, box_size, 0, box_size)
            glMatrixMode(GL_MODELVIEW)
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            running = True
            clock = pygame.time.Clock()
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                glClear(GL_COLOR_BUFFER_BIT)
                normalized_pos = self.system.positions % box_size
                render_radius = self.system.radius * 0.4

                # Отрисовка частиц
                glColor3f(1.0, 0.0, 0.0)
                for pos in normalized_pos:
                    self.draw_circle(pos[0], pos[1], render_radius)

                # Контуры
                glColor3f(0.0, 0.0, 0.0)
                for pos in normalized_pos:
                    self.draw_circle_outline(pos[0], pos[1], render_radius)

                # Триангуляция
                extended_pos = self._create_periodic_copies(normalized_pos, box_size)
                tri = Delaunay(extended_pos)
                glColor4f(0.0, 1.0, 0.0, 0.3)
                glBegin(GL_LINES)
                for simplex in tri.simplices:
                    for i in range(3):
                        j = (i + 1) % 3
                        p1 = extended_pos[simplex[i]] % box_size
                        p2 = extended_pos[simplex[j]] % box_size
                        glVertex2f(p1[0], p1[1])
                        glVertex2f(p2[0], p2[1])
                glEnd()

                pygame.display.flip()
                clock.tick(60)
            pygame.quit()
        except Exception as e:
            print(f"OpenGL error: {str(e)}")
            pygame.quit()


if __name__ == '__main__':
    num_particles = 10

    box_size = 2.0
    radius = 0.1

    system = ParticleSystem(num_particles, box_size, radius)
    minimizer = EnergyMinimizer(system)
    result = minimizer.minimize_energy()
    x0 = system.positions.flatten()
    err = check_grad(system.compute_energy, system.compute_gradient, x0)
    print(f"Ошибка градиента: {err:.2e} (должна быть < 1e-4)")
    # if err > 1e-4:
    #     raise ValueError("Градиент вычислен некорректно!")
    print(f"Final energy: {result['fun']:.4f}")


    visualizer = Visualizer(system, minimizer.energy_history)
    visualizer.plot_energy()
    visualizer.plot_delaunay()

    opengl_thread = threading.Thread(target=visualizer.opengl_visualization)
    opengl_thread.daemonreached = True
    opengl_thread.start()

    try:
        while opengl_thread.is_alive():
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass