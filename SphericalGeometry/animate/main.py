from manim import *
import numpy as np


class PlaneShow(VectorScene):
    def construct(self):
        pln = NumberPlane()
        axs = Axes(
            x_length=14,
            y_length=8,
            axis_config={
                "include_ticks": False
            }
        )

        v1 = Vector([2, 1], color=RED)
        v2 = Vector([-1, 2], color=YELLOW)

        tl = Text("平面", font="STKaiti", color=YELLOW).shift(6*RIGHT+3*UP)

        self.play(Create(pln), run_time=3)
        self.play(Write(tl))
        self.play(Create(axs), run_time=3)
        self.wait()
        self.play(Create(v1), Create(v2))
        self.wait(3)


class SpaceShow(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=2*PI/5, theta=PI/5)

        axs = ThreeDAxes(
            axis_config={
                "include_ticks": False
            }
        )

        tl = Text("三维空间", font="STKaiti", color=YELLOW)
        tl.rotate(PI/2, axis=np.array([1, 0, 0]))
        tl.rotate(PI/5+PI/2)
        tl.shift(np.array([-3, 3, 3]))

        helix = ParametricFunction(
            lambda t: 1*np.array([np.cos(t), np.sin(t), t*0.1]),
            color=RED,
            t_range=np.array([-20, 20])
        ).set_shade_in_3d(True)

        self.play(Create(axs))
        self.play(Write(tl))
        self.play(Create(helix), run_time=5)
        self.wait(5)


class SphereShow(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI / 6, theta=PI / 6)

        sph = Sphere(
            radius=2,
            resolution=(20, 20),
            fill_opacity=0.6
        )
        sph.set_color(BLUE)

        c1 = ParametricFunction(
                lambda t: 2*np.array([np.cos(0)*np.sin(t), np.sin(0)*np.sin(t), np.cos(t)]),
                color=WHITE,
                t_range=np.array([0, 2*PI])
            ).set_shade_in_3d(True)

        rotate_phi = np.array([[np.cos(PI/6), 0, -1*np.sin(PI/6)],
                               [0, 1, 0],
                               [np.sin(PI/6), 0, np.cos(PI/6)]])
        c2 = ParametricFunction(
            lambda t: 2 * np.dot(rotate_phi,
                                 np.array([np.cos(t) * np.sin(PI/2), np.sin(t) * np.sin(PI/2), np.cos(PI/2)])),
            color=BLUE,
            t_range=np.array([0, 2 * PI])
        ).set_shade_in_3d(True)

        coord_p = 2*np.dot(rotate_phi, np.array([np.cos(0)*np.sin(PI/2), np.sin(0)*np.sin(PI/2), np.cos(PI/2)]))
        p1 = Dot3D(point=coord_p)
        p2 = Dot3D(point=-1*coord_p)

        self.play(Create(sph))
        self.play(Create(c1))
        self.play(Create(c2))
        self.play(Create(p1), Create(p2))
        self.wait(5)


class PtLnDstOfPlane(Scene):
    def construct(self):
        axs = Axes(
            x_length=14,
            y_length=8,
            axis_config={
                "include_ticks": False
            }
        )
        pln = NumberPlane()

        pa = Dot(point=np.array([1, 2, 0]), color=RED)
        t1 = MathTex(r'A').next_to(pa, direction=LEFT)
        pb = Dot(point=np.array([2, 1, 0]), color=RED)
        t2 = MathTex(r'B').next_to(pb, direction=LEFT)
        # pt = VGroup(pa, pb)

        ln = ParametricFunction(
            lambda t: 1*np.array([t, 3-t, 0]),
            t_range=np.array([-1, 6]),
            color=YELLOW
        )
        t3 = MathTex(r'l').shift(np.array([4, -0.5, 0]))

        dist_brace = BraceBetweenPoints(pa.get_point_mobject(), pb.get_point_mobject(), direction=np.array([1, 1, 0]))
        t4 = MathTex(r"d = \sqrt{2}").next_to(dist_brace, direction=np.array([1, 1, 0]))
        t4.shift(np.array([-0.4, -0.4, 0]))

        self.add(axs, pln)
        self.play(Create(pa))
        self.play(Create(pb))
        self.play(Create(t1), Create(t2))
        self.play(Create(ln))
        self.play(Write(t3))
        self.play(Create(dist_brace))
        self.play(Write(t4))
        self.wait(5)
