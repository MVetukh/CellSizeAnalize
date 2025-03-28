

import numpy as np
import numba
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


@numba.jit(nopython=True)
def apply_pbc(positions):
    """Применяет периодические граничные условия."""
    for i in range(positions.shape[0]):
        for d in range(2):  # x и y
            if positions[i, d] > 1:
                positions[i, d] -= 2
            elif positions[i, d] < -1:
                positions[i, d] += 2
    return positions


@numba.jit(nopython=True)
def pairwise_potential_energy(positions):
    """Вычисляет электростатическую энергию системы с учетом PBC."""
    N = positions.shape[0]
    energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]

            # Находим кратчайшее расстояние с учетом PBC
            dx -= round(dx / 2) * 2
            dy -= round(dy / 2) * 2
            r = np.sqrt(dx ** 2 + dy ** 2)
            energy += 1.0 / r  # Потенциал Кулона
    return energy


@numba.jit(nopython=True)
def coordinate_descent(positions, lr=0.01, tol=1e-5, max_iters=10000):
    """Оптимизация методом покоординатного спуска с PBC."""
    N = positions.shape[0]
    for iteration in range(max_iters):
        prev_energy = pairwise_potential_energy(positions)

        for i in range(N):
            for d in range(2):  # Оптимизируем x и y отдельно
                old_value = positions[i, d]

                # Двигаем вправо
                positions[i, d] = old_value + lr
                apply_pbc(positions)  # Применяем PBC
                energy_right = pairwise_potential_energy(positions)

                # Двигаем влево
                positions[i, d] = old_value - lr
                apply_pbc(positions)  # Применяем PBC
                energy_left = pairwise_potential_energy(positions)

                # Выбираем лучшее положение
                if energy_right < energy_left and energy_right < prev_energy:
                    positions[i, d] = old_value + lr
                elif energy_left < prev_energy:
                    positions[i, d] = old_value - lr
                else:
                    positions[i, d] = old_value  # Оставляем без изменений

        new_energy = pairwise_potential_energy(positions)
        if abs(new_energy - prev_energy) < tol:
            print(f"Сошлось на {iteration} итерации")
            break

    return positions


@numba.jit(nopython=True)
def gradient_descent(positions, lr=0.005, tol=1e-5, max_iters=10000):
    """Метод градиентного спуска с PBC для быстрого улучшения."""
    N = positions.shape[0]
    for iteration in range(max_iters):
        gradients = np.zeros_like(positions)

        # Рассчитываем градиенты
        for i in range(N):
            for d in range(2):  # Для x и y
                gradient = 0.0
                for j in range(N):
                    if i != j:
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dx -= round(dx / 2) * 2
                        dy -= round(dy / 2) * 2
                        r = np.sqrt(dx ** 2 + dy ** 2)
                        gradient += (1.0 / r ** 2) * (dx if d == 0 else dy)  # Потенциал Кулона

                gradients[i, d] = gradient

        # Обновляем позиции зарядов
        positions -= lr * gradients
        apply_pbc(positions)  # Применяем PBC

        # Проверка на сходимость
        if np.linalg.norm(gradients) < tol:
            print(f"Градиентный спуск сошелся на {iteration} итерации")
            break

    return positions


# ---- OpenGL Визуализация ----
def init_gl():
    glEnable(GL_POINT_SMOOTH)
    glPointSize(10)
    glClearColor(0, 0, 0, 1)
    glMatrixMode(GL_PROJECTION)
    gluOrtho2D(-1.2, 1.2, -1.2, 1.2)


def draw_points(positions):
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(1, 1, 1)  # Белые точки

    glBegin(GL_POINTS)
    for x, y in positions:
        glVertex2f(x, y)
    glEnd()

    pygame.display.flip()


# Основной цикл с визуализацией
def main():
    global positions
    pygame.init()
    screen = pygame.display.set_mode((600, 600), DOUBLEBUF | OPENGL)
    init_gl()

    running = True
    iteration = 0
    energy = pairwise_potential_energy(positions)
    gradient_step_done = False  # Флаг, чтобы следить за тем, когда произошел переход

    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        # Если энергия системы высока, продолжаем использовать градиентный спуск
        if energy > threshold_energy and not gradient_step_done:
            positions = gradient_descent(positions, learning_rate, tolerance, 1)
            energy = pairwise_potential_energy(positions)
            if energy < threshold_energy:  # Переход на покоординатный спуск
                gradient_step_done = True
                print("Переключение на покоординатный спуск.")
        # Когда градиентный спуск завершился, используем покоординатный спуск
        elif gradient_step_done:
            positions = coordinate_descent(positions, learning_rate, tolerance, 1)
            energy = pairwise_potential_energy(positions)

        # Рисуем заряды
        draw_points(positions)

    pygame.quit()


if __name__ == '__main__':
    N = 10  # Количество зарядов
    learning_rate = 0.005  # Меньший шаг для градиентного спуска
    tolerance = 1e-5  # Критерий остановки
    max_iters = 10000  # Максимум итераций
    threshold_energy = 1e-2  # Порог для переключения на метод покоординатного спуска
    decay_factor = 0.99  # Фактор для адаптивного уменьшения шага

    # Генерируем случайные координаты зарядов
    positions = np.random.rand(N, 2) * 2 - 1  #\
    main()
