import numpy as np
import numba
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial import Delaunay

# Параметры системы
N = 100
learning_rate = 0.03
tolerance = 1e-6
max_iters = 1000
num_neighbors = 6

# Параметры размеров частиц
mean_size = 15.0    # Средний размер
std_size = 6.0     # Стандартное отклонение
min_size = 4.0     # Минимальный размер

# Генерация начальных позиций и размеров
positions = np.random.rand(N, 2) * 2 - 1
sizes = np.abs(np.random.normal(mean_size, std_size, N))  # Нормальное распределение
sizes = np.clip(sizes, min_size, None)  # Защита от отрицательных размеров




@numba.jit(nopython=True)
def apply_pbc(positions):
    """Периодические граничные условия"""
    return (positions + 1) % 2 - 1


@numba.jit(nopython=True)
def local_energy(i, positions, num_neighbors=6):
    N = positions.shape[0]
    if num_neighbors >= N - 1:
        num_neighbors = N - 2

    epsilon = 1.0
    sigma = 0.28

    deltas = positions - positions[i]
    deltas -= np.round(deltas / 2) * 2

    dists_sq = (deltas ** 2).sum(axis=1)
    dists_sq[i] = np.inf

    k = min(num_neighbors, len(dists_sq) - 1)
    neighbors = np.argpartition(dists_sq, k)[:k]
    r = np.sqrt(dists_sq[neighbors])
    r = np.maximum(r, 1e-9)

    lj = 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    return np.sum(lj)


@numba.jit(nopython=True)
def coordinate_descent(positions, lr=0.01, tol=1e-5, max_iters=1000, num_neighbors=6):
    """Покоординатный спуск с адаптивным шагом"""
    N = positions.shape[0]
    step_scale = 1.0
    prev_energy = np.inf

    for iteration in range(max_iters):
        total_energy = 0.0
        moved = False

        for i in range(N):
            for dim in [0, 1]:  # x и y по очереди
                # Сохраняем исходное значение БЕЗ .copy()
                old_val = positions[i, dim]  # <-- Исправлено здесь
                current_energy = local_energy(i, positions, num_neighbors)

                # Пробуем шаг в положительном направлении
                positions[i, dim] = old_val + step_scale * lr
                positions = apply_pbc(positions)
                energy_plus = local_energy(i, positions, num_neighbors)

                # Пробуем шаг в отрицательном направлении
                positions[i, dim] = old_val - step_scale * lr
                positions = apply_pbc(positions)
                energy_minus = local_energy(i, positions, num_neighbors)

                # Возвращаем исходное значение для проверки
                positions[i, dim] = old_val
                positions = apply_pbc(positions)

                # Выбираем лучшее направление
                if energy_plus < current_energy and energy_plus < energy_minus:
                    positions[i, dim] = old_val + step_scale * lr
                    total_energy += energy_plus
                    moved = True
                elif energy_minus < current_energy:
                    positions[i, dim] = old_val - step_scale * lr
                    total_energy += energy_minus
                    moved = True
                else:
                    total_energy += current_energy

        # Остальной код без изменений
        if abs(prev_energy - total_energy) < tol:
            print(f"Сходимость на итерации {iteration}")
            break
        prev_energy = total_energy

        if not moved:
            step_scale /= 2.0
            if step_scale < 1e-4:
                break

        if iteration % 50 == 0:
            print(f"Iter: {iteration}, Energy: {float(total_energy)}, Step: {float(step_scale)}")

    return positions




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


# ---- OpenGL Визуализация ----
def init_gl():
    glEnable(GL_POINT_SMOOTH)
    glClearColor(0, 0, 0, 1)
    glMatrixMode(GL_PROJECTION)
    gluOrtho2D(-1.2, 1.2, -1.2, 1.2)


def draw_points(positions, sizes):
    """Отрисовка точек с разными размерами"""
    for i in range(len(positions)):
        glPointSize(sizes[i])
        glColor3f(1, 1, 1)  # Белый цвет
        glBegin(GL_POINTS)
        x, y = positions[i]
        glVertex2f(x, y)
        glEnd()




def draw_final_state(positions, sizes):
    glClear(GL_COLOR_BUFFER_BIT)
    draw_triangulation(positions)
    draw_points(positions, sizes)
    pygame.display.flip()


def main():
    global positions, sizes

    # Оптимизация позиций
    positions = apply_pbc(positions)
    positions = coordinate_descent(
        positions,
        learning_rate,
        tolerance,
        max_iters,
        num_neighbors
    )

    # Визуализация
    pygame.init()
    screen = pygame.display.set_mode((600, 600), DOUBLEBUF | OPENGL)
    init_gl()
    draw_final_state(positions, sizes)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
        pygame.time.wait(100)
    pygame.quit()


if __name__ == "__main__":
    main()