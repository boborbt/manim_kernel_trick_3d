from manim import *
import numpy as np
from numpy.linalg import inv


class Kernel3D(ThreeDScene):
    NUM_EXAMPLES = 5
    QUAD_SP = [0.3, 0.3] # Quadratic surface scaling parameters
    LABELING_THRESHOLD = 1.5

    def construct(self):
        self.add_axes()
        self.add_data_points()

        self.animate_move_camera()
        kernel_animations = self.apply_kernel()
        quad_surf_animation = self.add_quadratic_surface()

        self.play(*kernel_animations, quad_surf_animation, run_time=5)
        
        linear_classifier = self.add_linear_classifier()
        self.play(linear_classifier)

    def animate_move_camera(self):
        # Static movement of the camera (for debugging purposes)
        # self.camera.set_phi(0.8*np.pi/2)
        # self.camera.set_theta(-0.45*np.pi)
 
        self.move_camera(0.8*np.pi/2, -0.45*np.pi)

    def add_axes(self):
        self.axes = ThreeDAxes()
        self.add(self.axes)

    def add_data_points(self):
        self.data_points = []
        self.labels = []
        self.data_coords = np.random.normal(0.0, 1.0, (self.NUM_EXAMPLES, 2))
        self.data_coords = np.hstack((self.data_coords, np.zeros((self.NUM_EXAMPLES,1))))

        def labeling_rule(x, y): return 1 if (x**2 + y**2) > self.LABELING_THRESHOLD else -1

        for coord in self.data_coords:
            self.data_points.append(Sphere(center = coord,radius = 0.05))

            label = labeling_rule(coord[0], coord[1])
            self.labels.append(label)
            self.data_points[-1].set_color(BLUE  if label == 1 else RED)

        self.add(*self.data_points)

    def quadratic_kernel(self, x, y):
        return self.QUAD_SP[0] * x**2 + self.QUAD_SP[1] * y**2

    def apply_kernel(self):
        animations = []
        for point in self.data_points:
            coord = point.get_center()
            animations.append(point.animate.move_to([coord[0], coord[1], self.quadratic_kernel(coord[0], coord[1])]))
        return animations

    def add_quadratic_surface(self):
        linear_surface = Surface(
            lambda x, y: self.axes.c2p(x, y,  0),
            v_range=[-5, 5],
            u_range=[-5, 5]
        )
        linear_surface.set_style(fill_opacity=0.3)
        linear_surface.set_fill_by_value(
            axes=self.axes, colors=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)])

        quadratic_surface = Surface(
            lambda x, y: self.axes.c2p(x, y,  self.quadratic_kernel(x,y)),
            v_range=[-5, 5],
            u_range=[-5, 5]
            )
        quadratic_surface.set_style(fill_opacity=0.3)
        quadratic_surface.set_fill_by_value(axes=self.axes, colors=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)])

        self.add(linear_surface)
        return Transform(linear_surface, quadratic_surface)

        # self.play(self.add(self.axes, quadratic_surface))

    def add_linear_classifier(self):
        classify = lambda x, y: self.LABELING_THRESHOLD

        linear_classifier = Surface(
            lambda x, y: self.axes.c2p(x, y, classify(x, y)),
            v_range=[-5, 5],
            u_range=[-5, 5]
        )

        linear_classifier.set_style(fill_opacity=0.3)
        linear_classifier.set_fill_by_value(
            axes=self.axes, colors=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)])

        xy_plane = Surface(
            lambda x, y: self.axes.c2p(x, y, 0),
        )

        xy_plane.set_style(fill_opacity=0.3)
        xy_plane.set_fill_by_value(
            axes=self.axes, colors=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)])
        self.add(xy_plane)

        return Transform(xy_plane, linear_classifier)


