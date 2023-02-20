from manimlib import *
import numpy as np


# 用转轴axis和转角theta计算四元数
def axis_theta_to_quaternion(axis, theta):
    norm = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    v = list((np.sin(theta/2)/norm)*np.array(axis))
    return [*v, np.cos(theta/2)]


# 四元数的乘法
def prod_quaternion(q, p):
    return [
        q[0]*p[3]+q[3]*p[0]-q[2]*p[1]+q[1]*p[2],
        q[1]*p[3]+q[2]*p[0]+q[3]*p[1]-q[0]*p[2],
        q[2]*p[3]-q[1]*p[0]+q[0]*p[1]+q[3]*p[2],
        q[3]*p[3]-q[1]*p[1]-q[2]*p[2]-q[0]*q[0]
    ]


# 将球坐标转换为numpy坐标数组
def sph_to_coord(theta, phi):
    return 3*np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])


# 开篇展示平面，空间和球面
class PSSShow(Scene):
    def construct(self):
        # 平面部分的展示
        plane_grid = NumberPlane()
        plane_axis = Axes(
            axis_config={
                "include_tip": True,
                "include_ticks": False
            }
        )

        vector1 = Vector(np.array([2, 1]), color=RED)
        vector2 = Vector(np.array([-1, 2]), color=YELLOW)

        plane_text = Text("平面", font="STKaiti", color=YELLOW).to_edge(RIGHT+UP)

        self.play(ShowCreation(plane_grid), run_time=3)
        self.play(Write(plane_text), ShowCreation(plane_axis))
        self.play(ShowCreation(vector1), ShowCreation(vector2))
        self.wait(3)

        # 三维空间部分的展示
        space_axis = ThreeDAxes(
            width=8,
            depth=6,
            axis_config={
                "include_tip": True,
                "include_ticks": False
            }
        )
        vector3 = Vector(np.array([0, 1, 2]), color=BLUE)

        # 转动相机之前定义好所有文本
        space_text = Text("三维空间", font='STKaiti', color=YELLOW).to_edge(UP+RIGHT)
        space_text.fix_in_frame()
        sphere_text = Text('球面', font='STKaiti', color=YELLOW).to_edge(UP+RIGHT)
        sphere_text.fix_in_frame()
        title_text = MarkupText(
            """
            球面几何
            """,
            font='STKaiti',
            color=YELLOW,
            isolate=['球面', '几何']
        ).shift(RIGHT*3)
        title_text.fix_in_frame()

        # 获取相机帧的引用
        camera = self.camera.frame
        # 使用四元数旋转（欧拉旋转经常会检测到万向节锁死）
        quaternion1 = axis_theta_to_quaternion([1, 0, 0], PI/2)
        quaternion2 = axis_theta_to_quaternion([0, 0, 1], 3*PI/4)
        quaternion3 = axis_theta_to_quaternion([-0.2, 1, 0], -PI/20)
        quaternion_a = prod_quaternion(quaternion2, quaternion1)
        quaternion_a = prod_quaternion(quaternion3, quaternion_a)

        self.play(
            camera.animate.set_orientation(Rotation(quaternion_a)),
            FadeOut(plane_grid),
            FadeOut(plane_text),
            Write(space_text)
        )
        self.play(FadeOut(plane_axis), ShowCreation(space_axis))
        self.play(ShowCreation(vector3))
        self.wait()

        # 球面展示
        sphere = Sphere(
            radius=2,
            color=BLUE,
            opacity=0.8,
        )
        sphere_mesh = SurfaceMesh(sphere)

        self.play(
            FadeOut(space_text),
            FadeOut(space_axis),
            FadeOut(vector1),
            FadeOut(vector2),
            FadeOut(vector3)
        )
        self.play(ShowCreation(sphere))
        self.wait()
        self.play(ShowCreation(sphere_mesh), Write(sphere_text))
        self.wait()
        self.play(sphere.animate.shift(RIGHT*3+DOWN*1), sphere_mesh.animate.shift(RIGHT*3+DOWN*1))
        self.play(
            TransformMatchingStrings(
                sphere_text, title_text
            )
        )


class Apply(Scene):
    def construct(self):
        text1 = Text(
            """
            对球面几何的
            研究最早
            开始于古希腊,
            """,
            font='STKaiti',
            font_size=30
        ).shift(RIGHT*3)
        text1.fix_in_frame()

        pic = ImageMobject("material\\Greece.jpg").shift(LEFT*3)

        text2 = Text(
            """
            球面几何在大地
            测量, 航海, 
            飞机飞行, 卫星定位
            方面有重要作用.
            """,
            font='STKaiti',
            font_size=30
        ).shift(RIGHT*3)
        text2.fix_in_frame()

        day_texture = "material\\1280px-Whole_world_-_land_and_oceans.jpg"
        night_texture = "material\\The_earth_at_night.jpg"

        sphere = Sphere(radius=2.71)
        earth = TexturedSurface(sphere, day_texture, night_texture).shift(LEFT*3)

        # 地球旋转的函数
        def update_rotate_func(mob, alpha):
            mob.rotate(1*DEGREES*alpha)

        self.play(Write(text1), FadeIn(pic, shift=RIGHT), run_time=3)
        self.wait()
        self.play(FadeOut(text1), FadeOut(pic, shift=LEFT), run_time=3)

        camera = self.camera.frame
        camera.set_euler_angles(
            theta=-30 * DEGREES,
            phi=70 * DEGREES,
        )
        self.play(ShowCreation(earth))
        self.play(
            Write(text2),
            UpdateFromAlphaFunc(earth, update_rotate_func),
            run_time=3
        )
        self.play(
            FadeOut(text2),
            FadeOut(earth)
        )
        self.wait()


class ConstructionOfSphere(Scene):
    def construct(self):
        plane_grid = NumberPlane()
        plane_axis = Axes(
            axis_config={
                "include_tip": True,
                "include_ticks": False
            },
            height=7
        )

        point_a = Dot(point=np.array([1, 2, 0]), color=RED)
        text1 = Tex(r'A').next_to(point_a, direction=LEFT)
        point_b = Dot(point=np.array([2, 1, 0]), color=RED)
        text2 = Tex(r'B').next_to(point_b, direction=LEFT)
        point = VGroup(point_a, point_b)

        line = Line(start=np.array([-0.5, 3.5, 0]), end=np.array([4, -1, 0]), color=YELLOW)
        text3 = Tex(r'l').shift(np.array([4, -0.5, 0]))

        brace = BraceLabel(point, text=r'd = \sqrt{2}', brace_direction=np.array([1, 1, 0]))

        self.add(plane_axis, plane_grid)
        self.play(ShowCreation(point_a))
        self.play(ShowCreation(point_b))
        self.play(Write(text1), Write(text2))
        self.play(ShowCreation(line))
        self.play(Write(text3))
        self.play(ShowCreation(brace))
        self.wait()

        sphere = Sphere(
            radius=3,
            color=BLUE_E,
            opacity=0.8,
        )
        sphere_mesh = SurfaceMesh(sphere)

        sphere_point_a = Dot(point=sph_to_coord(PI / 8-PI/2, PI / 6), color=RED)
        sphere_point_b = Dot(point=sph_to_coord(-PI / 8-PI/2, PI / 6), color=RED)

        arc1 = ParametricCurve(
            lambda t: sph_to_coord(t, PI/6),
            color=YELLOW,
            t_range=np.array([-PI/2-PI / 8, -PI/2+PI / 8])
        )

        camera = self.camera.frame
        quaternion1 = axis_theta_to_quaternion([1, 0, 0], PI / 2)
        quaternion2 = axis_theta_to_quaternion([0.5, 1, 0], -PI / 8)
        quaternion3 = axis_theta_to_quaternion([0, 0, 1], PI / 2)
        quaternion4 = axis_theta_to_quaternion([0, 1, 0], -PI/4)
        quaternion5 = axis_theta_to_quaternion([0, 0, 1], -PI/2)
        quaternion_a = prod_quaternion(quaternion2, quaternion1)
        quaternion_a = prod_quaternion(quaternion3, quaternion_a)
        quaternion_a = prod_quaternion(quaternion4, quaternion_a)
        quaternion_a = prod_quaternion(quaternion5, quaternion_a)
        light = self.camera.light_source
        light.move_to(np.array([0, 0, 10]))

        self.play(
            FadeOut(brace),
            FadeOut(line),
            FadeOut(text1),
            FadeOut(text2),
            FadeOut(text3),
            FadeOut(point),
            FadeOut(plane_axis)
        )
        self.play(
            camera.animate.set_orientation(Rotation(quaternion_a)),
            ReplacementTransform(plane_grid, sphere_mesh),
            ShowCreation(sphere)
        )
        # self.play(FadeOut(sphere_mesh))
        self.play(ShowCreation(sphere_point_a), ShowCreation(sphere_point_b))
        self.wait()
        self.play(ShowCreation(arc1))
        self.wait()
