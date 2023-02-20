from manimlib import *
import numpy as np


def sph_to_coord(theta, phi):
    return 2*np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])


class PlaneShow(Scene):
    def construct(self):
        pln = NumberPlane()
        axs = Axes(
            x_length=14,
            y_length=8,
            axis_config={
                "include_ticks": False
            }
        )

        v1 = Vector(np.array([2, 1]), color=RED)
        v2 = Vector(np.array([-1, 2]), color=YELLOW)

        tl = Text("平面", font="STKaiti", color=YELLOW).shift(6*RIGHT+3*UP)

        self.play(ShowCreation(pln), run_time=3)
        self.play(Write(tl))
        self.play(ShowCreation(axs), run_time=3)
        self.wait()
        self.play(ShowCreation(v1), ShowCreation(v2))
        self.wait(3)


class SpaceShow(Scene):
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

        helix = ParametricSurface(
            lambda t: 1*np.array([np.cos(t), np.sin(t), t*0.1]),
            color=RED,
            t_range=np.array([-20, 20])
        )

        self.play(ShowCreation(axs))
        self.play(Write(tl))
        self.play(ShowCreation(helix), run_time=5)
        self.wait(5)


class SphereShow(Scene):
    def construct(self):
        self.set_camera_orientation(phi=PI / 6, theta=PI / 6)

        sph = Sphere(
            radius=2,
            resolution=(20, 20),
            fill_opacity=0.6
        )
        # noinspection PyTypeChecker
        sph.set_color(BLUE)

        c1 = ParametricSurface(
                lambda t: 2*np.array([np.cos(0)*np.sin(t), np.sin(0)*np.sin(t), np.cos(t)]),
                color=WHITE,
                t_range=np.array([0, 2*PI])
            )

        rotate_phi = np.array([[np.cos(PI/6), 0, -1*np.sin(PI/6)],
                               [0, 1, 0],
                               [np.sin(PI/6), 0, np.cos(PI/6)]])
        c2 = ParametricSurface(
            lambda t: 2 * np.dot(rotate_phi,
                                 np.array([np.cos(t) * np.sin(PI/2), np.sin(t) * np.sin(PI/2), np.cos(PI/2)])),
            color=BLUE,
            t_range=np.array([0, 2 * PI])
        )

        coord_p = 2*np.dot(rotate_phi, sph_to_coord(0, PI/2))
        p1 = Dot3D(point=coord_p)
        p2 = Dot3D(point=-1*coord_p)

        self.play(ShowCreation(sph))
        self.play(ShowCreation(c1))
        self.play(ShowCreation(c2))
        self.play(ShowCreation(p1), ShowCreation(p2))
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
        t1 = Tex(r'A').next_to(pa, direction=LEFT)
        pb = Dot(point=np.array([2, 1, 0]), color=RED)
        t2 = Tex(r'B').next_to(pb, direction=LEFT)
        # pt = VGroup(pa, pb)

        ln = ParametricSurface(
            lambda t: 1*np.array([t, 3-t, 0]),
            t_range=np.array([-1, 6]),
            color=YELLOW
        )
        t3 = Tex(r'l').shift(np.array([4, -0.5, 0]))

        dist_brace = BraceBetweenPoints(pa.get_point_mobject(), pb.get_point_mobject(), direction=np.array([1, 1, 0]))
        t4 = Tex(r"d = \sqrt{2}").next_to(dist_brace, direction=np.array([1, 1, 0]))
        t4.shift(np.array([-0.4, -0.4, 0]))

        self.add(axs, pln)
        self.play(ShowCreation(pa))
        self.play(ShowCreation(pb))
        self.play(ShowCreation(t1), ShowCreation(t2))
        self.play(ShowCreation(ln))
        self.play(Write(t3))
        self.play(ShowCreation(dist_brace))
        self.play(Write(t4))
        self.wait(5)


class PtLnDstOfSphere(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI / 6, theta=PI / 6)

        temp_axs = ThreeDAxes()

        '''
        sph = Sphere(
            radius=2,
            resolution=(20, 20),
            fill_opacity=0.6
        )
        # noinspection PyTypeChecker
        sph.set_color(BLUE)
        '''

        sph = Surface(
            sph_to_coord,
            u_range=(0, TAU), v_range=(0, PI),
            color=BLUE
        )

        # noinspection PyTypeChecker
        pa = Dot3D(point=sph_to_coord(PI/8, PI/6), color=RED, radius=0.05)
        # noinspection PyTypeChecker
        pb = Dot3D(point=sph_to_coord(-PI/8, PI/6), color=RED, radius=0.05)
        # arc1 = ArcBetweenPoints(start=pa, end=pb, angle=PI/8+PI/6)

        arc1 = ParametricSurface(
            lambda t: 2 * np.array([np.cos(t) * np.sin(PI / 6), np.sin(t) * np.sin(PI / 6), np.cos(PI / 6)]),
            color=YELLOW,
            t_range=np.array([-PI / 8, PI / 8])
        )

        self.add(temp_axs)
        self.add(sph)
        self.add(pa, pb)
        self.play(ShowCreation(arc1))
        self.wait(5)
