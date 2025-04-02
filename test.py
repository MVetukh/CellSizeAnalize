from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import threading
from scipy.spatial import Delaunay

# Параметры окна
WIDTH, HEIGHT = 600, 600
NUM_CIRCLES = 100  # Количество кругов
CIRCLE_RADIUS = 0.04  # Фиксированный радиус для всех кругов
KE = 1.0  # Уменьшенная электрическая постоянная
CHARGE = 1e-3  # Уменьшенный заряд
LEARNING_RATE = 0.0001  # Уменьшенная скорость обучения
BETA1 = 0.9
BETA2 = 0.999
EPSILON_ADAM = 1e-8
STEPS = 500  # Увеличенное количество шагов
REGULARITY_WEIGHT = 10.0  # Увеличенный вес штрафа за регулярность

# Желаемое расстояние для гексагональной упаковки
DESIRED_DISTANCE = 0.1

# Глобальные переменные для обмена данными между потоками
circles = []
energy_history = []


def electric_potential(r):
    """
    Вычисление электрического потенциала между двумя зарядами.
    Если расстояние меньше DELTA, возвращаем потенциал для DELTA.
    """
    DELTA = 1e-2   # Локальная переменная для сравнения
    if r < DELTA:
        r = DELTA  # Избегаем деления на ноль или слишком маленьких значений
    return KE * CHARGE ** 2 / r


def electric_force(r, dx, dy):
    """
    Вычисление силы по закону Кулона.
    Если расстояние меньше DELTA, возвращаем силу для DELTA.
    """
    DELTA = 1e-2  # Локальная переменная для сравнения
    if r < DELTA:
        r = DELTA  # Избегаем деления на ноль или слишком маленьких значений
    force_magnitude = KE * CHARGE ** 2 / (r ** 2)
    return force_magnitude * dx / r, force_magnitude * dy / r


def regularity_penalty(circles):
    """Штраф для регулярности расположения частиц, заставляющий их стремиться к гексагональной упаковке"""
    penalty = 0
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            r = math.sqrt(dx ** 2 + dy ** 2)
            penalty += (r - DESIRED_DISTANCE) ** 2
    return REGULARITY_WEIGHT * penalty


def compute_energy(circles):
    energy = 0.0
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            r = math.sqrt(dx ** 2 + dy ** 2)
            if r > 0:
                energy += electric_potential(r)
    # Добавляем штраф за отклонение от желаемой гексагональной упаковки
    energy += regularity_penalty(circles)
    return energy


def adam_optimization(circles, steps=STEPS, lr=LEARNING_RATE):
    global energy_history
    m, v = [[0, 0] for _ in circles], [[0, 0] for _ in circles]
    t = 0
    energy_history = []
    for _ in range(steps):
        t += 1
        forces = [[0, 0] for _ in circles]
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                r = math.sqrt(dx ** 2 + dy ** 2)
                if r > 0:
                    fx, fy = electric_force(r, dx, dy)
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy

        for i in range(len(circles)):
            for d in range(2):
                m[i][d] = BETA1 * m[i][d] + (1 - BETA1) * forces[i][d]
                v[i][d] = BETA2 * v[i][d] + (1 - BETA2) * (forces[i][d] ** 2)
                m_hat = m[i][d] / (1 - BETA1 ** t)
                v_hat = v[i][d] / (1 - BETA2 ** t)
                circles[i][d] -= lr * m_hat / (math.sqrt(v_hat) + EPSILON_ADAM)

        energy_history.append(compute_energy(circles))
    return energy_history


def generate_initial_positions(num_circles, width, height):
    positions = []
    spacing = DESIRED_DISTANCE * 1.2  # Расстояние между частицами
    rows = int(math.sqrt(num_circles))
    cols = rows
    for i in range(rows):
        for j in range(cols):
            x = (i - rows / 2) * spacing
            y = (j - cols / 2) * spacing
            positions.append([x, y, CIRCLE_RADIUS])
    return positions


def draw_circle(x, y, radius, aspect_ratio, segments=50):
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        glVertex2f(x + math.cos(angle) * radius / aspect_ratio, y + math.sin(angle) * radius)
    glEnd()


def draw_delaunay_triangulation(circles):
    points = np.array([(circle[0], circle[1]) for circle in circles])  # только координаты x, y
    tri = Delaunay(points)  # Вычисление триангуляции Делоне
    glColor3f(1.0, 0.0, 0.0)  # Красный цвет для треугольников
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                x1, y1 = points[simplex[i]]
                x2, y2 = points[simplex[j]]
                glBegin(GL_LINES)
                glVertex2f(x1, y1)
                glVertex2f(x2, y2)
                glEnd()


def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(1.0, 1.0, 1.0)
    aspect_ratio = WIDTH / HEIGHT
    for x, y, radius in circles:
        draw_circle(x, y, radius, aspect_ratio)
    draw_delaunay_triangulation(circles)  # Отображение триангуляции Делоне
    glutSwapBuffers()


def reshape(w, h):
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = w, h
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    # Определяем размер окна в зависимости от круга
    min_x = min(circles, key=lambda c: c[0])[0]
    max_x = max(circles, key=lambda c: c[0])[0]
    min_y = min(circles, key=lambda c: c[1])[1]
    max_y = max(circles, key=lambda c: c[1])[1]

    margin = 0.1  # Немного пространства для масштаба

    # Масштабируем видимую область
    glOrtho(min_x - margin, max_x + margin, min_y - margin, max_y + margin, -1.0, 1.0)

    glutPostRedisplay()


def plot_energy(energy_history):
    plt.figure("System Energy")
    plt.plot(energy_history, label="System Energy")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Energy Minimization via Adam Optimizer")
    plt.legend()
    plt.grid()
    plt.show()


def start_glut():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow(b"Particle Interaction and Optimization")
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMainLoop()


def main():
    global circles, energy_history

    # Генерация начального расположения частиц
    circles = generate_initial_positions(NUM_CIRCLES, WIDTH, HEIGHT)

    # Оптимизация частиц с использованием Adam
    energy_history = adam_optimization(circles)

    # Запуск OpenGL в отдельном потоке
    glut_thread = threading.Thread(target=start_glut)
    glut_thread.start()

    # Отрисовка графика энергии в основном потоке
    plot_energy(energy_history)

    # Ожидание завершения потока OpenGL
    glut_thread.join()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-whitegrid')

    # Параметры потенциала
    epsilon = 1.0  # Глубина потенциальной ямы
    sigma = 1.0  # Расстояние, где потенциал равен нулю


    # Функция потенциала Леннарда-Джонса
    def lennard_jones(r):
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


    # Создание массива расстояний
    r = np.linspace(0.9, 3.0, 500)
    U = lennard_jones(r)

    # Создание фигуры с настройками
    plt.figure(figsize=(10, 6), dpi=100)
    plt.rcParams.update({'font.size': 12})

    # Построение графика
    plt.plot(r, U, lw=3, color='#2E86C1', label='Потенциал Леннарда-Джонса')

    # Разметка осей и заголовок
    plt.title("Потенциал Леннарда-Джонса", pad=20, fontsize=18, fontweight='bold')
    plt.xlabel(r'Расстояние $r/\sigma$', fontsize=14, labelpad=10)
    plt.ylabel(r'Энергия $U/\epsilon$', fontsize=14, labelpad=10)
    plt.ylim(-1.5, 5)

    # Вертикальные и горизонтальные линии
    plt.axhline(0, color='gray', lw=1, ls='--')
    plt.axvline(2 ** (1 / 6), color='#E74C3C', lw=2, ls='--',
                label=r'Равновесие ($r=2^{1/6}\sigma$)')

    # Аннотации
    plt.annotate('Зона отталкивания',
                 xy=(1.0, 3),
                 xytext=(0.95, 4),
                 arrowprops=dict(arrowstyle="->", color='black'),
                 fontsize=12)

    plt.annotate('Зона притяжения',
                 xy=(1.5, -0.7),
                 xytext=(1.7, -1.2),
                 arrowprops=dict(arrowstyle="->", color='black'),
                 fontsize=12)

    plt.annotate(r'Глубина ямы: $-\epsilon$',
                 xy=(2 ** (1 / 6), -1),
                 xytext=(1.8, -1.3),
                 arrowprops=dict(arrowstyle="->", color='black'),
                 fontsize=12)

    # Легенда и сетка
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.4)

    # Сохранение в высоком разрешении
    plt.savefig('LJ_potential.png', bbox_inches='tight', dpi=300)
    plt.show()