
import numba
import pandas as pd
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial import Delaunay, Voronoi, distance_matrix
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, ConvexHull, distance_matrix
from itertools import product
import numpy as np


# Новые параметры системы
N = 410               # Увеличили число частиц
num_neighbors = 6    # Оптимизируем для 6 ближайших соседей

# Параметры оптимизации
learning_rate = 0.03   # Увеличили скорость обучения
max_iters = 1000       # Увеличили число итераций
tolerance = 1e-7
# Оптимальные параметры (в единицах 10e-3):
mean_size = 50 * 1e-3  # 50 → 0.05 (золотая середина)
std_size = 15 * 1e-3   # 15 → 0.015 (умеренный разброс)
min_size = 20 * 1e-3   # 20 → 0.02 (минимальный видимый размер)
max_size = 100 * 1e-3  # 100 → 0.1 (максимум без слипания)   # Установить реалистичный максимум

# Генерация начальных позиций и радиусов
positions = np.random.rand(N, 2) * 2 - 1
sizes = np.abs(np.random.normal(mean_size, std_size, N))
sizes = np.clip(sizes, min_size, max_size)  # Ограничили максимальный размер


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
        sigma = sizes[i] + sizes[j]
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

        if abs(prev_energy - total_energy) < tol:
            break
        prev_energy = total_energy

        if not moved:
            step_scale /= 2
            if step_scale < 1e-4:
                break

    return positions

def init_gl():
    glEnable(GL_POINT_SMOOTH)
    glClearColor(0, 0, 0, 1)
    glMatrixMode(GL_PROJECTION)
    gluOrtho2D(-1.2, 1.2, -1.2, 1.2)


def draw_points(positions, sizes):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    num_segments = 30  # Количество сегментов для аппроксимации круга

    for i in range(len(positions)):
        scaled_size = sizes[i]*2
        radius = scaled_size / 2.0

        # Отрисовка заполненного круга (основной частицы)
        glColor3f(1, 1, 1)
        glBegin(GL_POLYGON)
        for j in range(num_segments):
            angle = 2 * np.pi * j / num_segments
            x = positions[i, 0] + radius * np.cos(angle)
            y = positions[i, 1] + radius * np.sin(angle)
            glVertex2f(x, y)
        glEnd()

        # Отрисовка обводки круга
        glLineWidth(2.0)  # Толщина границы
        glColor3f(1, 1, 1)
        glBegin(GL_LINE_LOOP)
        for j in range(num_segments):
            angle = 2 * np.pi * j / num_segments
            x = positions[i, 0] + radius * np.cos(angle)
            y = positions[i, 1] + radius * np.sin(angle)
            glVertex2f(x, y)
        glEnd()

def draw_triangulation(positions):
    try:
        expanded = np.concatenate([
            positions + [dx, dy]
            for dx in [-2, 0, 2]
            for dy in [-2, 0, 2]
        ])
        tri = Delaunay(expanded)
        glColor3f(1, 0, 0)
        glLineWidth(1.5)
        for simplex in tri.simplices:
            if np.all((expanded[simplex] >= -1) & (expanded[simplex] <= 1)):
                glBegin(GL_LINE_LOOP)
                for vertex in simplex:
                    x, y = expanded[vertex]
                    glVertex2f(x, y)
                glEnd()
    except Exception as e:
        print("Ошибка триангуляции:", e)

def voronoi_finite_polygons_2d(vor):
    """Фильтруем только конечные регионы (исправленная сигнатура)"""
    regions = []
    for region in vor.regions:
        if not region:
            continue
        if all(v >= 0 for v in region):
            regions.append(region)
    return regions, vor.vertices


def draw_voronoi(positions):
    """Отрисовка диаграммы Вороного с цветовым кодированием"""
    try:
        vor = Voronoi(positions)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        # Цветовая схема для количества граней
        face_colors = {
            4: (0.2, 0.8, 0.2),  # зеленый
            5: (0.9, 0.6, 0.1),  # оранжевый
            6: (0.1, 0.4, 0.9),  # синий
            7: (0.8, 0.1, 0.8),  # фиолетовый
            8: (0.7, 0.7, 0.1),  # желтый
            9: (0.5, 0.2, 0.2)  # коричневый
        }

        glLineWidth(2.0)
        for region in regions:
            polygon = vertices[region]
            if np.all((polygon >= -1) & (polygon <= 1)):
                # Определяем количество граней
                num_faces = len(region)
                color = face_colors.get(num_faces, (0.5, 0.5, 0.5))  # серый для других

                # Отрисовка заполненной ячейки
                glColor3f(*color)
                glBegin(GL_POLYGON)
                for (x, y) in polygon:
                    glVertex2f(x, y)
                glEnd()

                # Контур ячейки
                glColor3f(0, 0, 0)
                glBegin(GL_LINE_LOOP)
                for (x, y) in polygon:
                    glVertex2f(x, y)
                glEnd()

    except Exception as e:
        print("Ошибка Вороного:", e)

def draw_triangulation_window(positions, sizes):
    pygame.init()
    screen = pygame.display.set_mode((600, 600), DOUBLEBUF | OPENGL)
    init_gl()

    running = True
    while running:
        glClear(GL_COLOR_BUFFER_BIT)
        draw_points(positions, sizes)
        draw_triangulation(positions)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        pygame.time.wait(30)

def draw_voronoi_window(positions, sizes):
    pygame.init()
    screen = pygame.display.set_mode((600, 600), DOUBLEBUF | OPENGL)
    init_gl()

    running = True
    while running:
        glClear(GL_COLOR_BUFFER_BIT)
        draw_points(positions, sizes)
        draw_voronoi(positions)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        pygame.time.wait(30)

def plot_neighbor_distribution(positions):
    vor = Voronoi(positions)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    neighbor_counts = []

    for idx in range(len(positions)):
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]

        if not region or not all(v >= 0 for v in region):
            continue

        polygon = vertices[region]
        if np.all((polygon >= -1) & (polygon <= 1)):
            count = sum(1 for ridge in vor.ridge_points if idx in ridge)
            neighbor_counts.append(count)

    unique, counts = np.unique(neighbor_counts, return_counts=True)
    proportions = counts / counts.sum()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(unique, proportions, width=0.5, color='steelblue', edgecolor='white')

    # Добавляем значения поверх столбцов
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.xlabel("Число соседей", fontsize=12)
    plt.ylabel("Доля клеток", fontsize=12)
    plt.title(f"Распределение по кличеству соседей (N={len(neighbor_counts)} клеток)", fontsize=14)
    plt.xticks(unique)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def calculate_areas_and_faces(positions):
    """Вычисление площадей и количества сторон ячеек"""
    vor = Voronoi(positions)
    data = []

    for idx in range(len(positions)):
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]

        if not region or -1 in region:
            continue

        polygon = vor.vertices[region]
        if np.all((polygon >= -1.2) & (polygon <= 1.2)):
            # Вычисляем площадь
            area = 0.5 * np.abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) -
                                np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
            # Количество сторон = количество вершин
            num_faces = len(region)
            data.append((area, num_faces))


    return np.array(data, dtype=[('area', float), ('faces', int)])

def calc_half_width(areas):
    """Расчет относительной полуширины распределения площадей"""
    areas = np.sort(areas)  # Упорядочиваем по возрастанию
    mean = np.mean(areas)
    n = len(areas)
    half_n = n // 2

    # Массив модулей разностей
    delta_areas = np.abs(areas - mean)

    # Скользящие суммы по окну half_n
    sum_deltas = np.convolve(
        delta_areas,
        np.ones(half_n),
        mode='valid'
    )

    # Находим минимальную сумму
    min_idx = np.argmin(sum_deltas)

    # Границы интервала
    S_i = areas[min_idx]
    S_k = areas[min_idx + half_n - 1]

    return abs(S_k - S_i) / mean


def plot_area_distribution(data):
    if len(data) == 0:
        print("Нет данных для визуализации")
        return
        # Расчет относительной полуширины
    half_width = calc_half_width(data['area'])
    print(f"\nОтносительная полуширина распределения: {half_width:.4f}")
    # Создаем DataFrame
    df = pd.DataFrame({
        'area': data['area'],
        'faces': data['faces']
    })

    # Фильтруем грани и задаем цвета
    face_colors = {
        4: '#33CC33', 5: '#FF9933', 6: '#3366FF',
        7: '#CC33CC', 8: '#FFFF33', 9: '#993300'
    }
    df = df[df['faces'].isin(face_colors.keys())]

    # Автоматические бины и группировка
    bins = np.histogram_bin_edges(df['area'], bins='auto')
    df['area_bin'] = pd.cut(df['area'], bins=bins, include_lowest=True)

    # Подсчет и нормировка данных
    hist_data = df.groupby(['area_bin', 'faces'], observed=False).size().unstack(fill_value=0)
    total = hist_data.sum().sum()
    hist_data = hist_data / total

    # Визуализация
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    bottom = np.zeros(len(hist_data))

    # Построение столбцов
    for face in sorted(face_colors.keys()):
        if face in hist_data.columns:
            values = hist_data[face].values
            ax.bar(
                x=range(len(hist_data)),
                height=values,
                width=0.8,
                bottom=bottom,
                color=face_colors[face],
                edgecolor='black',
                linewidth=0.5,
                label=f'{face}-угольники'
            )
            bottom += values

    # Динамическое масштабирование осей
    y_max = 1.05 * np.max(bottom)  # +5% запаса сверху
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.linspace(0, y_max, 11))  # 10 делений

    # Форматирование оси X
    bin_labels = [f"{b.left:.2f}-{b.right:.2f}" for b in hist_data.index]
    ax.set_xticks(range(len(hist_data)))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')

    # Дополнительные элементы
    ax.plot(range(len(hist_data)), bottom, 'k--', alpha=0.5, label='Общее распределение')
    ax.set_xlabel("Интервал площади", fontsize=12)
    ax.set_ylabel("Доля всех ячеек", fontsize=12)
    ax.set_title("Распределение площадей ячеек Вороного", fontsize=14)
    ax.legend(title='Тип ячейки', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

# Добавляем вызов в main()
def main():
    global positions, sizes
    positions_pbc = apply_pbc(positions)
    positions = coordinate_descent(positions_pbc, sizes, learning_rate, tolerance, max_iters)

    draw_triangulation_window(positions, sizes)
    draw_voronoi_window(positions, sizes)
    voro_data = calculate_areas_and_faces(positions)
    plot_neighbor_distribution(positions)
    plot_area_distribution(voro_data)  # Новый график
    print("Тип данных:", type(voro_data))
    print("Поля данных:", voro_data.dtype.names if hasattr(voro_data, 'dtype') else 'нет данных')




if __name__ == "__main__":
    main()
