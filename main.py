import numpy as np
import numba
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial import Delaunay,Voronoi
from scipy import stats
from libpysal.weights import W
from esda.moran import Moran, Moran_BV
import matplotlib.pyplot as plt
import networkx as nx


# Параметры системы
N = 200
learning_rate = 0.005
tolerance = 1e-7
max_iters = 3000
num_neighbors = 8

# Параметры размеров
mean_size = 0.07
std_size = 0.015
min_size = 0.04
# Генерация начальных позиций и радиусов
positions = np.random.rand(N, 2) * 2 - 1
sizes = np.abs(np.random.normal(mean_size, std_size, N))
sizes = np.clip(sizes, min_size, None)


@numba.jit(nopython=True)
def apply_pbc(positions):
    return (positions + 1) % 2 - 1


@numba.jit(nopython=True)
def local_energy(i, positions, sizes, num_neighbors=6):
    N = positions.shape[0]
    epsilon = 1.0
    energy = 0.0

    deltas = positions - positions[i]
    deltas -= np.round(deltas / 2) * 2

    dists_sq = (deltas ** 2).sum(axis=1)
    dists_sq[i] = np.inf

    k = min(num_neighbors, N - 1)
    neighbors = np.argpartition(dists_sq, k)[:k]

    for j in neighbors:
        r = np.sqrt(dists_sq[j])
        sigma = sizes[i] + sizes[j]  # Сумма радиусов
        r = max(r, 1e-9)
        energy += 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    return energy


@numba.jit(nopython=True)
def coordinate_descent(positions, sizes, lr=0.005, tol=1e-5, max_iters=1000):
    N = positions.shape[0]
    step_scale = 1.0
    prev_energy = np.inf

    for iteration in range(max_iters):
        total_energy = 0.0
        moved = False

        for i in range(N):
            for dim in [0, 1]:
                old_val = positions[i, dim]
                current_energy = local_energy(i, positions, sizes, num_neighbors)

                # Positive direction
                positions[i, dim] = old_val + step_scale * lr
                positions = apply_pbc(positions)
                e_plus = local_energy(i, positions, sizes, num_neighbors)

                # Negative direction
                positions[i, dim] = old_val - step_scale * lr
                positions = apply_pbc(positions)
                e_minus = local_energy(i, positions, sizes, num_neighbors)

                # Reset position
                positions[i, dim] = old_val
                positions = apply_pbc(positions)

                # Update logic
                if e_plus < current_energy and e_plus < e_minus:
                    positions[i, dim] = old_val + step_scale * lr
                    total_energy += e_plus
                    moved = True
                elif e_minus < current_energy:
                    positions[i, dim] = old_val - step_scale * lr
                    total_energy += e_minus
                    moved = True
                else:
                    total_energy += current_energy

        # Convergence check
        if abs(prev_energy - total_energy) < tol:
            break
        prev_energy = total_energy

        if not moved:
            step_scale /= 2
            if step_scale < 1e-4:
                break

        print("Total energy:", total_energy, "step_scale:", step_scale, "iteration:", iteration)

    return positions


# ---- OpenGL Визуализация ----
def init_gl():
    glEnable(GL_POINT_SMOOTH)
    glClearColor(0, 0, 0, 1)
    glMatrixMode(GL_PROJECTION)
    gluOrtho2D(-1.2, 1.2, -1.2, 1.2)


# ---- OpenGL Визуализация ----
def draw_points(positions, sizes):
    """Отрисовка белых кружков с разными размерами"""
    glEnable(GL_PROGRAM_POINT_SIZE)
    for i in range(len(positions)):
        scaled_size = sizes[i] * 500  # Масштабирование размера
        glPointSize(scaled_size)
        glColor3f(1, 1, 1)  # Белый цвет
        glBegin(GL_POINTS)
        glVertex2f(positions[i, 0], positions[i, 1])
        glEnd()


def draw_triangulation(positions):
    """Отрисовка триангуляции поверх частиц"""
    try:
        # Создаем расширенную систему для учёта PBC
        expanded = np.concatenate([
            positions + [dx, dy]
            for dx in [-2, 0, 2]
            for dy in [-2, 0, 2]
        ])

        tri = Delaunay(expanded)

        glColor3f(1, 0, 0)
        glLineWidth(1.5)
        for simplex in tri.simplices:
            # Отрисовываем только центральную копию
            if np.all((expanded[simplex] >= -1) & (expanded[simplex] <= 1)):
                glBegin(GL_LINE_LOOP)
                for vertex in simplex:
                    x, y = expanded[vertex]
                    glVertex2f(x, y)
                glEnd()
    except Exception as e:
        print("Ошибка триангуляции:", e)

#
# def voronoi_finite_polygons_2d(vor, radius=None):
#     """
#     Перестраивает бесконечные регионы диаграммы Вороного в конечные полигоны.
#
#     Parameters:
#         vor (scipy.spatial.Voronoi): исходная диаграмма Вороного.
#         radius (float, optional): расстояние, на которое «удаляется» бесконечное ребро.
#                                    Если не задано, вычисляется как 2*макс.размах точек.
#
#     Returns:
#         regions (list of lists): список регионов, каждый регион — список индексов вершин.
#         vertices (ndarray): массив вершин, включая добавленные для бесконечных областей.
#     """
#     if vor.points.shape[1] != 2:
#         raise ValueError("Функция работает только для 2D данных.")
#
#     new_regions = []
#     new_vertices = vor.vertices.tolist()
#
#     center = vor.points.mean(axis=0)
#     if radius is None:
#         radius = np.ptp(vor.points, axis=0).max()
#
#     # Для каждой точки формируем список рёбер (ridge)
#     all_ridges = {}
#     for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
#         all_ridges.setdefault(p1, []).append((p2, v1, v2))
#         all_ridges.setdefault(p2, []).append((p1, v1, v2))
#
#     # Перебираем регионы для каждой точки
#     for p1, region_idx in enumerate(vor.point_region):
#         vertices_idx = vor.regions[region_idx]
#         if all(v >= 0 for v in vertices_idx):
#             # Регион конечный, добавляем как есть
#             new_regions.append(vertices_idx)
#             continue
#
#         # Для бесконечного региона: начинаем с уже известных конечных вершин
#         new_region = [v for v in vertices_idx if v >= 0]
#         # Обрабатываем рёбра, связанные с точкой p1
#         for p2, v1, v2 in all_ridges[p1]:
#             # Если одно из рёбер бесконечно (отмечается -1)
#             if v1 < 0 or v2 < 0:
#                 # Гарантируем, что v2 — конечная вершина
#                 if v2 < 0:
#                     v1, v2 = v2, v1
#
#                 # Вычисляем единичный вектор вдоль направления от p1 к p2 (тангенс)
#                 t = vor.points[p2] - vor.points[p1]
#                 t /= np.linalg.norm(t)
#                 # Нормаль к вектору
#                 n = np.array([-t[1], t[0]])
#
#                 # Определяем направление: от центра диаграммы
#                 midpoint = vor.points[[p1, p2]].mean(axis=0)
#                 direction = np.sign(np.dot(midpoint - center, n)) * n
#                 # Вычисляем «далёкую» точку вдоль направления
#                 far_point = vor.vertices[v2] + direction * radius
#                 new_region.append(len(new_vertices))
#                 new_vertices.append(far_point.tolist())
#
#         # Сортируем вершины многоугольника по углу относительно центра
#         vs = np.asarray([new_vertices[v] for v in new_region])
#         angles = np.arctan2(vs[:, 1] - center[1], vs[:, 0] - center[0])
#         new_region = [v for _, v in sorted(zip(angles, new_region))]
#         new_regions.append(new_region)
#
#     return new_regions, np.array(new_vertices)






def draw_final_state(positions, sizes):
    glClear(GL_COLOR_BUFFER_BIT)
    draw_points(positions, sizes)  # Частицы
    vor = Voronoi(positions)

    draw_triangulation(positions)  # Триангуляция поверх
    pygame.display.flip()



def main():
    global positions, sizes
    positions = apply_pbc(positions)
    positions = coordinate_descent(positions, sizes, learning_rate, tolerance, max_iters)

    # Визуализация
    pygame.init()
    screen = pygame.display.set_mode((600, 600), DOUBLEBUF | OPENGL)
    init_gl()
    draw_final_state(positions, sizes)


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        pygame.time.wait(100)
    pygame.quit()
if __name__ == "__main__":
    main()