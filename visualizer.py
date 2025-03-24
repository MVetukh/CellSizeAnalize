# visualizer.py

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
from scipy.spatial import Delaunay
import numpy as np

class Visualizer:
    def __init__(self, system, width=800, height=600):
        self.system = system
        self.width = width
        self.height = height

    def draw_circle(self, x, y, radius, aspect_ratio, segments=50):
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex2f(x + math.cos(angle) * radius / aspect_ratio, y + math.sin(angle) * radius)
        glEnd()

    def draw_delaunay_triangulation(self):
        points = np.array([(circle.x, circle.y) for circle in self.system.circles])
        tri = Delaunay(points)
        glColor3f(1.0, 0.0, 0.0)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    x1, y1 = points[simplex[i]]
                    x2, y2 = points[simplex[j]]
                    glBegin(GL_LINES)
                    glVertex2f(x1, y1)
                    glVertex2f(x2, y2)
                    glEnd()

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glColor3f(1.0, 1.0, 1.0)
        aspect_ratio = self.width / self.height
        for circle in self.system.circles:
            self.draw_circle(circle.x, circle.y, circle.radius, aspect_ratio)
        self.draw_delaunay_triangulation()
        glutSwapBuffers()

    def reshape(self, w, h):
        self.width, self.height = w, h
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        min_x = min(self.system.circles, key=lambda c: c.x).x
        max_x = max(self.system.circles, key=lambda c: c.x).x
        min_y = min(self.system.circles, key=lambda c: c.y).y
        max_y = max(self.system.circles, key=lambda c: c.y).y
        margin = 0.1
        glOrtho(min_x - margin, max_x + margin, min_y - margin, max_y + margin, -1.0, 1.0)
        glutPostRedisplay()

    def start(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(b"OpenGL Circles")
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glutMainLoop()
