import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from scipy.spatial import Delaunay
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
import scipy.spatial

class Visualizer:
    def __init__(self, system, width=800, height=600):
        self.system = system
        self.width = width
        self.height = height

    def init_gl(self):
        """Инициализация OpenGL"""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)

    def display(self):
        """Финальная отрисовка"""
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        glColor3f(0.4, 0.6, 1.0)
        positions = self.system.positions.detach().cpu().numpy()

        for x, y in positions:
            self.draw_circle(x, y, self.system.particle_radius)

        # Триангуляция
        if len(positions) > 2:
            self.draw_delaunay(positions)

        glutSwapBuffers()

    def draw_circle(self, cx, cy, radius, num_segments=20):
        """Отрисовка круга"""
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for i in range(num_segments + 1):
            angle = 2.0 * np.pi * i / num_segments
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            glVertex2f(x, y)
        glEnd()

    def draw_delaunay(self, points):
        """Отрисовка триангуляции"""
        tri = scipy.spatial.Delaunay(points)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        for simplex in tri.simplices:
            for i in range(3):
                p1, p2 = points[simplex[i]], points[simplex[(i + 1) % 3]]
                glVertex2f(p1[0], p1[1])
                glVertex2f(p2[0], p2[1])
        glEnd()

    def start(self):
        """Запуск OpenGL"""
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(b"Particle System")
        self.init_gl()
        glutDisplayFunc(self.display)
        glutIdleFunc(glutPostRedisplay)  # Обновление кадра
        glutMainLoop()
