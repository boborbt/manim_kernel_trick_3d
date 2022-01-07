from manim import *
import numpy as np
from numpy.linalg import inv
import math


class Kernel3D(ThreeDScene):
    NUM_EXAMPLES = 50
    QUAD_SP = 0.3 # Quadratic surface scaling parameters
    LABELING_THRESHOLD = 1
    RANDOM_SEED=42

    def construct(self):
        np.random.seed(self.RANDOM_SEED)

        # Initial setup
        self.add_axes()
        self.add_data_points()
        self.wait(2)

        # Showing 3d scene
        self.move_camera(0.8*np.pi/2, -0.45*np.pi)

        # Applying the kernel trick
        kernel_text = Tex(r"$(x,y) \rightarrow (x^2, y^2)$", font_size=72)
        kernel_text.to_edge(UP, buff=0.5)
        kernel_text.to_edge(RIGHT, buff=0.5)
        self.add_fixed_in_frame_mobjects(kernel_text)
        self.play(Write(kernel_text))

        kernel_animations = self.apply_kernel()
        quad_surf_animation = self.add_quadratic_surface()
        self.play(*kernel_animations, quad_surf_animation, run_time=5)

        self.play(Unwrite(kernel_text))
        self.wait(1)
        
        # Showing plane intersecting with the kernel
        self.move_camera(np.pi/2, -0.45*np.pi)
        self.wait(1)

        linear_classifier_anim = self.add_linear_classifier()
        self.play(linear_classifier_anim)

        self.move_camera(0.8*np.pi/2, -0.45*np.pi)

        intersection_line = self.add_intersection_line()
        self.play(intersection_line)

        self.wait(1)

        # Switching back to 2d scene and removing 3d surfaces
        self.move_camera(0, 0)

        self.play(Uncreate(self.quadratic_surface), Uncreate(self.linear_surface))
        self.remove(self.quadratic_surface)
        self.play(Uncreate(self.linear_classifier))
        self.remove(self.linear_classifier)
        self.wait(2) 

    def add_axes(self):
        self.axes = ThreeDAxes()
        self.add(self.axes)

    def add_data_points(self):
        self.data_points = []
        self.labels = []
        inner_examples = np.random.normal(0.0, 0.5, (int(self.NUM_EXAMPLES/2), 2))
        outer_examples = np.random.normal(0.0, 2.0, (int(self.NUM_EXAMPLES/2), 2))
        self.data_coords = np.vstack((inner_examples, outer_examples))
        self.data_coords = np.hstack(
            (self.data_coords, np.zeros((self.data_coords.shape[0], 1))))

        def labeling_rule(x, y): return 1 if (x**2 + y**2) > self.LABELING_THRESHOLD else -1

        for coord in self.data_coords:
            self.data_points.append(Sphere(center = coord, radius = 0.05))

            label = labeling_rule(coord[0], coord[1])
            self.labels.append(label)
            self.data_points[-1].set_color(BLUE  if label == 1 else RED)

        self.add(*self.data_points)

    def quadratic_kernel(self, x, y):
        return self.QUAD_SP * x**2 + self.QUAD_SP * y**2

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

        self.linear_surface = linear_surface
        self.play(Create(linear_surface))

        quadratic_surface = Surface(
            lambda x, y: self.axes.c2p(x, y,  self.quadratic_kernel(x,y)),
            v_range=[-5, 5],
            u_range=[-5, 5]
            )
        quadratic_surface.set_style(fill_opacity=0.3)
        quadratic_surface.set_fill_by_value(axes=self.axes, colors=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)])

        self.quadratic_surface = quadratic_surface
        return Transform(linear_surface, quadratic_surface)

        # self.play(self.add(self.axes, quadratic_surface))

    def add_linear_classifier(self):
        classify = lambda x, y: self.LABELING_THRESHOLD * self.QUAD_SP 

        linear_classifier = Surface(
            lambda x, y: self.axes.c2p(x, y, classify(x, y)),
            v_range=[-5, 5],
            u_range=[-5, 5]
        )

        linear_classifier.set_style(fill_opacity=0.3)
        linear_classifier.set_fill_by_value(
            axes=self.axes, colors=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)])

        self.linear_classifier = linear_classifier
        return Create(linear_classifier)

    def add_intersection_line(self):
        intersection_line = Circle(
            radius = math.sqrt(self.LABELING_THRESHOLD),
            color=RED
        )

        intersection_line.shift(self.axes.c2p(0, 0, self.LABELING_THRESHOLD * self.QUAD_SP))

        self.add(intersection_line)
        return DrawBorderThenFill(intersection_line)

