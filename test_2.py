import numpy as np
import numba
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial import Delaunay  # Для вычисления триангуляции Делоне

N = 50  # Количество частиц
learning_rate = 0.02  # Шаг оптимизации
tolerance = 1e-8  # Критерий остановки
max_iters = 2000  # Максимум итераций
num_neighbors = 6  # Количество ближайших соседей для локального анализа

# Генерируем случайные координаты частиц
positions = np.random.rand(N, 2) * 2 - 1  # В пределах [-1, 1] для OpenGL

# Параметры системы
N = 50
learning_rate = 0.01  # Уменьшенный шаг обучения
tolerance = 1e-8
max_iters = 2000
num_neighbors = 6

# Генерация начальных позиций
positions = np.random.rand(N, 2) * 2 - 1


@numba.jit(nopython=True)
def apply_pbc(positions):
    """Корректные периодические граничные условия"""
    return (positions + 1) % 2 - 1


@numba.jit(nopython=True)
def local_energy(i, positions, num_neighbors=6):
    N = positions.shape[0]

    # Добавляем проверку количества соседей
    if num_neighbors >= N - 1:
        num_neighbors = N - 2  # Оставляем минимум 1 частицу

    epsilon = 1.0
    sigma = 0.28

    # Правильный расчёт смещений с PBC
    deltas = positions - positions[i]
    deltas -= np.round(deltas / 2) * 2  # Корректный учёт периодичности

    dists_sq = (deltas ** 2).sum(axis=1)
    dists_sq[i] = np.inf

    # Исправленный выбор соседей
    k = min(num_neighbors, len(dists_sq) - 1)
    neighbor_indices = np.argpartition(dists_sq, k)[:k]

    r = np.sqrt(dists_sq[neighbor_indices])
    r = np.maximum(r, 1e-9)

    lj = 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    return np.sum(lj)

@numba.jit(nopython=True)
def local_gradient_descent(positions, lr=0.001, tol=1e-5, max_iters=10000, num_neighbors=6):
    N = positions.shape[0]
    step_scale = 1.0
    prev_energy = np.inf

    for iteration in range(max_iters):
        moved = False
        total_energy = 0.0

        # Создаем копию позиций для одновременного обновления
        new_positions = positions.copy()

        for i in range(N):
            # Вычисление градиента
            grad = np.zeros(2)
            current_energy = local_energy(i, positions, num_neighbors)

            for d in range(2):
                # Метод центральной разности
                delta = np.zeros_like(positions)
                delta[i, d] = 1e-5

                pos_plus = apply_pbc(positions + delta)
                energy_plus = local_energy(i, pos_plus, num_neighbors)

                pos_minus = apply_pbc(positions - delta)
                energy_minus = local_energy(i, pos_minus, num_neighbors)

                grad[d] = (energy_plus - energy_minus) / (2e-5)

            # Обновление позиции
            new_pos = positions[i] - step_scale * lr * grad
            new_pos = apply_pbc(new_pos.reshape(1, -1))[0]

            # Проверка улучшения энергии
            new_energy = local_energy(i, new_pos.reshape(1, -1), num_neighbors)
            if new_energy < current_energy:
                new_positions[i] = new_pos
                moved = True
                total_energy += new_energy
            else:
                total_energy += current_energy

        positions = new_positions

        # Критерий остановки
        if abs(prev_energy - total_energy) < tol:
            print(f"Сходимость достигнута на итерации {iteration}")
            break
        prev_energy = total_energy

        # Регулировка шага
        if not moved:
            step_scale /= 2
            print(f"Уменьшение шага: {step_scale}")
            if step_scale < 1e-6:
                break

        if iteration % 100 == 0:
            print(f"Iter: {iteration}, Energy: {total_energy}")

    return positions

# ---- OpenGL Визуализация ----
def init_gl():
    glEnable(GL_POINT_SMOOTH)
    glPointSize(6)
    glClearColor(0, 0, 0, 1)
    glMatrixMode(GL_PROJECTION)
    gluOrtho2D(-1.2, 1.2, -1.2, 1.2)


def draw_points(positions):
    glColor3f(1, 1, 1)  # Белые точки
    glBegin(GL_POINTS)
    for x, y in positions:
        glVertex2f(x, y)
    glEnd()



def draw_triangulation(positions):
    try:
        tri = Delaunay(positions)
        print(f"Триангуляция содержит {len(tri.simplices)} симплексов")
    except Exception as e:
        print("Ошибка триангуляции:", e)
        return
    # Создаём 3x3 копий системы для учёта PBC
    expanded = np.concatenate([
        positions + [dx, dy]
        for dx in [-2, 0, 2]
        for dy in [-2, 0, 2]
    ])

    # Вычисляем триангуляцию для расширенной системы
    tri = Delaunay(expanded)

    # Отрисовываем только центральную копию
    glColor3f(1, 0, 0)
    for simplex in tri.simplices:
        if np.all((expanded[simplex] >= -1) & (expanded[simplex] <= 1)):
            glBegin(GL_LINE_LOOP)
            for vertex in simplex:
                x, y = expanded[vertex]
                glVertex2f(x, y)
            glEnd()

def draw_final_state(positions):
    glClear(GL_COLOR_BUFFER_BIT)
    draw_triangulation(positions)
    draw_points(positions)
    pygame.display.flip()


def main():
    global positions
    positions = apply_pbc(positions)  # Нормализация начальных позиций

    # Запуск оптимизации
    positions = local_gradient_descent(
        positions,
        learning_rate,
        tolerance,
        max_iters,
        num_neighbors
    )

    # Проверка результатов
    print("Проверка позиций:")
    print("Min:", np.min(positions, axis=0))
    print("Max:", np.max(positions, axis=0))
    print("Unique:", len(np.unique(positions.round(decimals=3), axis=0)))

    # Визуализация
    pygame.init()
    screen = pygame.display.set_mode((600, 600), DOUBLEBUF | OPENGL)
    init_gl()
    draw_final_state(positions)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
        pygame.time.wait(100)
    pygame.quit()


if __name__ == "__main__":
    main()