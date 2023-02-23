import manimlib.utils.space_ops
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
def sph_to_coord(r, theta, phi):
    return r*np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])


# 由两个点生成球面直线
def sphere_line(pa, pb):
    normal = manimlib.utils.space_ops.cross(pa, pb)
    axis = manimlib.utils.space_ops.cross(np.array(normal), np.array([0, 0, 1]))
    r = np.sqrt((pa ** 2).sum())
    if axis == [0, 0, 0]:
        rotate = np.eye(3)
    else:
        rotate = manimlib.utils.space_ops.z_to_vector(np.array(normal))
    line = ParametricCurve(
        lambda t: np.matmul(rotate, sph_to_coord(r, t, PI/2)),
        t_range=[0, 2*PI]
    )
    return line


# 连接两点球面线段
def sphere_segment(pa, pb):
    normal = manimlib.utils.space_ops.cross(pa, pb)
    axis = manimlib.utils.space_ops.cross(np.array(normal), np.array([0, 0, 1]))
    if axis == [0, 0, 0]:
        rotate = np.eye(3)
    else:
        rotate = manimlib.utils.space_ops.z_to_vector(np.array(normal))
    r = np.sqrt((pa**2).sum())
    # 将输入的两点转至赤道后转为复数求得幅角， 作为t的范围
    z_a = manimlib.utils.space_ops.R3_to_complex(np.matmul(np.linalg.inv(rotate), pa)/r)
    z_b = manimlib.utils.space_ops.R3_to_complex(np.matmul(np.linalg.inv(rotate), pb)/r)
    # noinspection PyTypeChecker
    d_angle = np.angle(z_b/z_a)
    if d_angle > PI:
        d_angle = 2*PI - d_angle
        # noinspection PyTypeChecker
        b_angle = np.angle(z_b)
    else:
        # noinspection PyTypeChecker
        b_angle = np.angle(z_a)
    segment = ParametricCurve(
        lambda t: np.matmul(rotate, sph_to_coord(r, t, PI / 2)),
        t_range=[b_angle, b_angle+d_angle]
    )
    return segment


# 球面上连接两点的球小圆的劣弧
def sphere_small_circle(pa, pb, pc):
    normal = manimlib.utils.space_ops.cross(pa - pc, pb - pc)
    axis = manimlib.utils.space_ops.cross(np.array(normal), np.array([0, 0, 1]))
    if axis == [0, 0, 0]:
        rotate = np.eye(3)
    else:
        rotate = manimlib.utils.space_ops.z_to_vector(np.array(normal))
    ra = np.matmul(np.linalg.inv(rotate), pa)
    rb = np.matmul(np.linalg.inv(rotate), pb)
    r = np.sqrt((pa ** 2).sum())
    phi = np.arccos(ra[2]/r)

    # 将输入的两点转至纬线后转为复数求得幅角， 作为t的范围
    z_a = manimlib.utils.space_ops.R3_to_complex(ra)
    z_b = manimlib.utils.space_ops.R3_to_complex(rb)
    # noinspection PyTypeChecker
    d_angle = np.angle(z_b / z_a)
    if d_angle > PI:
        d_angle = 2 * PI - d_angle
        # noinspection PyTypeChecker
        b_angle = np.angle(z_b)
    else:
        # noinspection PyTypeChecker
        b_angle = np.angle(z_a)

    segment = ParametricCurve(
        lambda t: np.matmul(rotate, sph_to_coord(r, t, phi)),
        t_range=[b_angle, b_angle+d_angle]
    )
    return segment


# 球面上的月形, 以a为一个顶点, a_b, a_c为月形的两边
def moon_shape(pa, pb, pc):
    if manimlib.utils.space_ops.cross(np.array(pa), np.array([0, 0, 1])) == [0, 0, 0]:
        rotate = np.eye(3)
    else:
        rotate = manimlib.utils.space_ops.z_to_vector(pa)
    rb = np.matmul(np.linalg.inv(rotate), pb)
    rc = np.matmul(np.linalg.inv(rotate), pc)
    r = np.sqrt((pa ** 2).sum())

    # 将输入的两点转至纬线后转为复数求得幅角， 作为theta的范围
    z_b = manimlib.utils.space_ops.R3_to_complex(rb)
    z_c = manimlib.utils.space_ops.R3_to_complex(rc)
    # noinspection PyTypeChecker
    d_theta = np.angle(z_c / z_b)
    if d_theta > PI:
        d_theta = 2 * PI - d_theta
        # noinspection PyTypeChecker
        b_theta = np.angle(z_c)
    else:
        # noinspection PyTypeChecker
        b_theta = np.angle(z_b)

    moon = ParametricSurface(
        lambda u, v: np.matmul(rotate, sph_to_coord(r, u, v)),
        u_range=(b_theta, b_theta+d_theta),
        v_range=(0, PI)
    )
    return moon


# 数学对象:球面三角形
class SphereTriangle(Group):
    def __init__(self, pa: np.array, pb: np.array, pc: np.array):
        self.pa = pa
        self.pb = pb
        self.pc = pc
        self.surf = None
        self.sigment_a = sphere_segment(pb, pc)
        self.sigment_b = sphere_segment(pc, pa)
        self.sigment_c = sphere_segment(pa, pb)
        Group.__init__(self, self.sigment_a, self.sigment_b, self.sigment_c)

    # 显示顶点
    def add_vertex(self):
        self.add(Dot(self.pa))
        self.add(Dot(self.pb))
        self.add(Dot(self.pc))

    # 显示三角内部的曲面
    def add_surf(self):
        # 判断一个点是否在给定三点构成的以pa为顶点的月形中
        if manimlib.utils.space_ops.cross(np.array(self.pa), np.array([0, 0, 1])) == [0, 0, 0]:
            rotate1 = np.eye(3)
        else:
            rotate1 = manimlib.utils.space_ops.z_to_vector(self.pa)
        rb1 = np.matmul(np.linalg.inv(rotate1), self.pb)
        rc1 = np.matmul(np.linalg.inv(rotate1), self.pc)
        z_b1 = manimlib.utils.space_ops.R3_to_complex(rb1)
        z_c1 = manimlib.utils.space_ops.R3_to_complex(rc1)
        # noinspection PyTypeChecker
        d_theta1 = np.angle(z_c1 / z_b1)
        if d_theta1 > PI:
            d_theta1 = 2 * PI - d_theta1
            # noinspection PyTypeChecker
            b_theta1 = np.angle(z_c1)
        else:
            # noinspection PyTypeChecker
            b_theta1 = np.angle(z_b1)

        def in_moon(pd):
            rd = np.matmul(np.linalg.inv(rotate1), pd)
            # 将输入的两点转至纬线后转为复数求得幅角， 作为theta的范围
            z_d = manimlib.utils.space_ops.R3_to_complex(rd)
            # noinspection PyTypeChecker
            theta = np.angle(z_d)
            if b_theta1 <= theta <= d_theta1+b_theta1:
                return True
            else:
                return False

        # 在以pb为顶点的月形上改动
        if manimlib.utils.space_ops.cross(np.array(self.pb), np.array([0, 0, 1])) == [0, 0, 0]:
            rotate = np.eye(3)
        else:
            rotate = manimlib.utils.space_ops.z_to_vector(self.pb)
        rb = np.matmul(np.linalg.inv(rotate), self.pa)
        rc = np.matmul(np.linalg.inv(rotate), self.pc)
        r = np.sqrt((self.pb ** 2).sum())

        # 将输入的两点转至纬线后转为复数求得幅角， 作为theta的范围
        z_b = manimlib.utils.space_ops.R3_to_complex(rb)
        z_c = manimlib.utils.space_ops.R3_to_complex(rc)
        # noinspection PyTypeChecker
        d_theta = np.angle(z_c / z_b)
        if d_theta > PI:
            d_theta = 2 * PI - d_theta
            # noinspection PyTypeChecker
            b_theta = np.angle(z_c)
        else:
            # noinspection PyTypeChecker
            b_theta = np.angle(z_b)

        def func_triangle(u: float, v: float):
            pt = np.matmul(rotate, sph_to_coord(r, u, v))
            if in_moon(pt):
                return pt
            else:
                return None

        triangle = ParametricSurface(
            func_triangle,
            u_range=(b_theta, b_theta+d_theta),
            v_range=(0, PI)
        )
        self.surf = triangle
        self.add(self.surf)


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
        # 平面上的部分
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

        # 球面上的部分
        sphere = Sphere(
            radius=3,
            color=BLUE_E,
            opacity=0.5,
        )
        sphere_mesh = SurfaceMesh(sphere)

        npa = sph_to_coord(3, PI / 8-PI/2, 2*PI/3)
        npb = sph_to_coord(3, -PI / 8-PI/2, PI / 6)
        sphere_point_a = Dot(point=npa, color=RED)
        sphere_point_b = Dot(point=npb, color=sphere_point_a.get_color())
        arc1 = sphere_segment(npb, npa).set_color(YELLOW)
        arc2 = sphere_small_circle(npb, npa, sph_to_coord(3, PI, PI/2)).set_color(arc1.get_color())
        arc3 = sphere_small_circle(npa, npb, sph_to_coord(3, -PI/4, PI / 3)).set_color(arc1.get_color())
        arc4 = sphere_small_circle(npa, npb, sph_to_coord(3, -PI/2-PI/8, PI / 3)).set_color(arc1.get_color())
        arc5 = sphere_small_circle(npa, npb, sph_to_coord(3, -3*PI/8, PI / 6)).set_color(arc1.get_color())
        v_arc = VGroup(arc1, arc2, arc3, arc4, arc5)
        s_line1 = sphere_line(npb, npa).set_color(YELLOW_E)
        s_line2 = sphere_line(np.array([0, 0, 3]), npa).set_color(s_line1.get_color())
        s_line3 = sphere_line(np.array([0, 0, 3]), npb).set_color(s_line1.get_color())
        s_line4 = sphere_line(sph_to_coord(3, -PI/4, PI/2), np.array([0, 3, 0])).set_color(s_line1.get_color())
        s_line5 = sphere_line(npb, sph_to_coord(3, -PI/4, -PI/4)).set_color(s_line1.get_color())

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

        # 制作时作参考
        axes = ThreeDAxes(
            axis_config={
                'include_tip': True
            }
        )
        self.add(axes)

        self.play(
            camera.animate.set_orientation(Rotation(quaternion_a)),
            ReplacementTransform(plane_grid, sphere_mesh),
            ShowCreation(sphere)
        )

        self.play(
            ShowCreation(sphere_point_a),
            ShowCreation(sphere_point_b)
        )
        self.wait()
        self.play(ShowCreation(arc1))
        self.play(ShowCreation(arc2))
        self.play(ShowCreation(arc3))
        self.play(ShowCreation(arc4))
        self.play(ShowCreation(arc5))
        self.add(sphere_point_a, sphere_point_b)
        self.play(ShowCreation(s_line1))
        self.add(sphere_point_a, sphere_point_b)
        self.wait()
        self.play(
            FadeOut(v_arc),
            FadeOut(sphere_point_a),
            FadeOut(sphere_point_b)
        )
        self.wait()
        self.play(FadeOut(s_line1))
        self.play(ShowCreation(s_line2))
        self.play(FadeOut(s_line2))
        self.play(ShowCreation(s_line3))
        self.wait()
        self.play(ShowCreation(s_line1))
        self.play(FadeOut(s_line1))
        self.play(ShowCreation(s_line4))
        self.play(FadeOut(s_line4))
        self.play(ShowCreation(s_line5))
        self.wait()

        self.play(
            FadeOut(sphere_mesh),
            FadeOut(sphere),
            FadeOut(s_line5),
            FadeOut(s_line3)
        )
        self.wait()


class DefinitionOfLast(Scene):
    def construct(self):
        definition_line = TexText(
            r'称$l$为球面直线, 若$l$是球面的一个大圆.',
            tex_to_color_map={'球面直线': YELLOW}
        ).shift(UP)
        definition_dist1 = TexText(
            r'球面上两点$A,B$间的球面距离$d_{S}(A, B)$',
            tex_to_color_map={"球面距离": YELLOW}
        ).next_to(definition_line, direction=DOWN).shift(DOWN)
        definition_dist2 = TexText(
            r'定义为过$A,B$的球大圆的劣弧弧长.'
        ).next_to(definition_dist1, direction=DOWN)
        definition_dist = VGroup(definition_dist1, definition_dist2)

        self.play(ShowCreation(definition_line), run_time=3)
        self.play(ShowCreation(definition_dist), run_time=4)
        self.wait()


class TriangleOnSphere(Scene):
    def construct(self):
        sphere = Sphere(
            radius=3,
            color=BLUE,
            opacity=0.8
        )

        npa = sph_to_coord(3, PI, 0)
        npb = sph_to_coord(3, -PI/2, PI/2)
        npc = sph_to_coord(3, PI/6-PI/2, PI/2)
        '''
        triangle = SphereTriangle(npa, npb, npc)
        triangle.add_vertex()
        triangle.add_surf()
        triangle.surf.set_color(RED_A)
        '''

        moon = moon_shape(npa, npb, npc)

        camera = self.camera.frame
        quaternion1 = axis_theta_to_quaternion([1, 0, 0], PI / 2)
        quaternion2 = axis_theta_to_quaternion([0.5, 1, 0], -PI / 8)
        quaternion3 = axis_theta_to_quaternion([0, 0, 1], PI / 2)
        quaternion4 = axis_theta_to_quaternion([0, 1, 0], -PI / 4)
        quaternion5 = axis_theta_to_quaternion([0, 0, 1], -PI / 2)
        quaternion_a = prod_quaternion(quaternion2, quaternion1)
        quaternion_a = prod_quaternion(quaternion3, quaternion_a)
        quaternion_a = prod_quaternion(quaternion4, quaternion_a)
        quaternion_a = prod_quaternion(quaternion5, quaternion_a)
        light = self.camera.light_source
        light.move_to(np.array([0, 0, 10]))
        camera.set_orientation(Rotation(quaternion_a)),

        self.play(ShowCreation(sphere))
        # self.play(ShowCreation(triangle))
        self.play(ShowCreation(moon))
        self.wait()
