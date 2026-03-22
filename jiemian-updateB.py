#jiemian-updateB.py:在jiemian-update.py基础上增加
#1.python代码中请将混凝土、钢筋和钢材三种材料的本构关系，用def专门设定，以便后续有更多本构可以调用。
#2.每行增加代码解释。

#jiemian-update.py:在jiemian.py基础上增加
#1.python代码和界面中增加圆形截面，并考虑中间部位是否有设置型钢，如果有型钢，应计算型钢的内力贡献。
#2.python代码中应增加型钢本构，内力计算应包含型钢贡献。

import streamlit as st  # 导入 Streamlit 库，用于快速构建和渲染 Web 交互界面
import numpy as np  # 导入 NumPy 库，用于执行高效的矩阵运算和向量化数学计算
import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块，用于绘制 2D 图表
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图工具包，用于渲染三维 PMM 空间曲面
import matplotlib.patches as mpatches  # 导入图例色块工具，用于在图表中生成自定义图例标识

# =====================================================================
# 页面基础设置
# =====================================================================
# 配置 Streamlit 页面的全局属性：标题、浏览器标签页图标、以及采用宽屏布局模式
st.set_page_config(page_title="P-M-M 截面分析系统 (本构模块化版)", page_icon="🏗️", layout="wide")


# =====================================================================
# 独立材料本构关系库 (Constitutive Models)
# 优势：将材料力学属性与几何积分剥离，支持单根纤维或矩阵批量并行计算
# =====================================================================

def get_stress_concrete(strain, fc, ecu=0.003):
    """
    混凝土本构：采用 ACI 318 兼容的 Hognestad 抛物线-矩形模型
    """
    e0 = 0.002  # 混凝土达到峰值应力时对应的应变 (通常取 0.002)
    strain_arr = np.asarray(strain)  # 将输入的应变(可能是单个数字或数组)统一转换为 NumPy 数组格式
    stress = np.zeros_like(strain_arr, dtype=float)  # 初始化一个与应变数组大小相同、初值全为 0 的应力数组

    # 利用布尔掩码(Mask)筛选出处于不同受力阶段的纤维：
    mask_para = (strain_arr > 0) & (strain_arr <= e0)  # 掩码 1：处于上升段抛物线的应变区间 (0 到 0.002)
    mask_rect = (strain_arr > e0) & (strain_arr <= ecu)  # 掩码 2：处于极限破坏平段的矩形应变区间 (0.002 到 0.003)

    # 计算抛物线段的应力：公式为 0.85fc * [ 2*(e/e0) - (e/e0)^2 ]
    stress[mask_para] = 0.85 * fc * (2 * (strain_arr[mask_para] / e0) - (strain_arr[mask_para] / e0) ** 2)
    # 计算矩形平段的应力：达到最大等效压应力，恒定为 0.85fc
    stress[mask_rect] = 0.85 * fc

    if np.isscalar(strain):  # 判断：如果一开始传入的是单个数字(标量)
        return stress.item()  # 则将结果剥离出数组，返回标量，保证数据类型一致性
    return stress  # 如果传入的是矩阵，则直接返回应力矩阵


def get_stress_rebar(strain, fy, Es):
    """
    钢筋本构：理想弹塑性双折线模型
    """
    # 核心逻辑：应变乘以弹性模量得到弹性应力，然后用 np.clip 强制截断在 [-fy, fy] 之间，模拟拉压双向屈服
    return np.clip(strain * Es, -fy, fy)


def get_stress_steel(strain, fya, Es):
    """
    型钢本构：理想弹塑性双折线模型
    """
    # 同钢筋逻辑，使用型钢专用的屈服强度 fya 进行截断限制
    return np.clip(strain * Es, -fya, fya)


# =====================================================================
# 核心类：钢筋混凝土及型钢截面定义
# 作用：根据用户在界面的输入，一次性生成所有的几何坐标、面积和规范极值
# =====================================================================
class RCSection:
    def __init__(self, sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                 has_steel, hs, bs, tw, tf, fya):
        self.sec_type = sec_type  # 记录截面类型（'矩形' 或 '圆形'）
        # 统一尺寸变量：如果是矩形使用 b 和 h，如果是圆形则宽高均等效为直径 D
        self.b = float(b) if sec_type == '矩形' else float(D)
        self.h = float(h) if sec_type == '矩形' else float(D)
        self.D = float(D)  # 记录直径
        self.fc = float(fc)  # 混凝土圆柱体抗压强度
        self.fy = float(fy)  # 钢筋屈服强度
        self.Es = float(Es)  # 钢筋弹性模量
        self.ecu = 0.003  # 混凝土极限压应变 (ACI 规范标准)
        self.ey = self.fy / self.Es  # 计算钢筋屈服应变

        # 计算截面毛面积 Ag：矩形为 b*h，圆形为 π*R^2
        self.Ag = self.b * self.h if sec_type == '矩形' else np.pi * (self.D / 2) ** 2

        # ================= 1. 钢筋阵列生成 =================
        A_bar = np.pi * (d_bar ** 2) / 4  # 计算单根钢筋的横截面积
        xs_list, ys_list = [], []  # 初始化用于存储所有钢筋 X 和 Y 坐标的列表

        if sec_type == '矩形':  # 【矩形截面排筋逻辑】
            if n_top > 0:  # 如果顶部有钢筋
                xs_list.extend(np.linspace(-self.b / 2 + cover, self.b / 2 - cover, n_top))  # 在宽度内均匀打点生成 X 坐标
                ys_list.extend([self.h / 2 - cover] * n_top)  # 顶部钢筋的 Y 坐标固定为 h/2 - cover
            if n_bot > 0:  # 如果底部有钢筋
                xs_list.extend(np.linspace(-self.b / 2 + cover, self.b / 2 - cover, n_bot))  # 同理生成底部 X 坐标
                ys_list.extend([-self.h / 2 + cover] * n_bot)  # 底部钢筋 Y 坐标固定为 -h/2 + cover
            if n_side > 0:  # 如果有侧面中部钢筋
                # 在顶排和底排之间，均匀生成 n_side 层 Y 坐标
                y_side = np.linspace(self.h / 2 - cover, -self.h / 2 + cover, n_side + 2)[1:-1]
                for y in y_side:  # 遍历每一层
                    xs_list.extend([-self.b / 2 + cover, self.b / 2 - cover])  # 每层左右各放一根钢筋
                    ys_list.extend([y, y])  # 赋予对应的 Y 坐标
        else:  # 【圆形截面排筋逻辑】
            R_rebar = self.D / 2 - cover  # 计算钢筋所在环的半径
            # 在 0 到 360 度之间均匀划分 n_circ 个角度 (极坐标法)
            angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
            xs_list = R_rebar * np.sin(angles)  # 通过 sin 计算 X 坐标 (以正上方为起点)
            ys_list = R_rebar * np.cos(angles)  # 通过 cos 计算 Y 坐标

        self.xs_coords = np.array(xs_list)  # 将列表转为 Numpy 数组，供向量化计算使用
        self.ys_coords = np.array(ys_list)  # 同上
        self.As_array = np.full(len(xs_list), A_bar)  # 生成一个数组，表示每根钢筋的面积均为 A_bar
        self.Ast_total = np.sum(self.As_array)  # 对数组求和，得到截面总钢筋面积 Ast

        # ================= 2. H型钢纤维阵列生成 =================
        self.has_steel = has_steel  # 布尔值：判断是否包含内置型钢
        self.fya = float(fya)  # 型钢的屈服强度
        s_xs, s_ys, s_As = [], [], []  # 初始化型钢微纤维的 X, Y 坐标及面积列表

        if has_steel:  # 如果包含型钢，开始将其离散化为细小的矩形纤维
            # (1) 离散上翼缘
            nx_f = max(5, int(bs / 20));
            ny_f = max(2, int(tf / 10))  # 根据尺寸决定网格密度：X向至少5格，Y向至少2格
            x_f = np.linspace(-bs / 2 + bs / (2 * nx_f), bs / 2 - bs / (2 * nx_f), nx_f)  # 上翼缘 X 坐标分布
            y_f_top = np.linspace(hs / 2 - tf + tf / (2 * ny_f), hs / 2 - tf / (2 * ny_f), ny_f)  # 上翼缘 Y 坐标分布
            X, Y = np.meshgrid(x_f, y_f_top)  # 生成网格矩阵
            s_xs.extend(X.flatten());
            s_ys.extend(Y.flatten())  # 展平网格并存入列表
            s_As.extend([bs * tf / (nx_f * ny_f)] * len(X.flatten()))  # 计算并赋予每个微纤维的面积 (总面积/网格数)

            # (2) 离散下翼缘
            y_f_bot = np.linspace(-hs / 2 + tf / (2 * ny_f), -hs / 2 + tf - tf / (2 * ny_f), ny_f)  # 下翼缘 Y 坐标
            X, Y = np.meshgrid(x_f, y_f_bot)  # 利用相同的 X 坐标分布生成下翼缘网格
            s_xs.extend(X.flatten());
            s_ys.extend(Y.flatten())
            s_As.extend([bs * tf / (nx_f * ny_f)] * len(X.flatten()))

            # (3) 离散腹板
            nx_w = max(2, int(tw / 10));
            ny_w = max(10, int((hs - 2 * tf) / 20))  # 腹板 X向极窄，Y向极长
            x_w = np.linspace(-tw / 2 + tw / (2 * nx_w), tw / 2 - tw / (2 * nx_w), nx_w)  # 腹板 X 坐标
            y_w = np.linspace(-hs / 2 + tf + (hs - 2 * tf) / (2 * ny_w), hs / 2 - tf - (hs - 2 * tf) / (2 * ny_w),
                              ny_w)  # 腹板 Y 坐标
            X, Y = np.meshgrid(x_w, y_w)  # 生成腹板网格
            s_xs.extend(X.flatten());
            s_ys.extend(Y.flatten())
            s_As.extend([tw * (hs - 2 * tf) / (nx_w * ny_w)] * len(X.flatten()))  # 赋予腹板纤维面积

        self.steel_xs = np.array(s_xs)  # 将型钢纤维 X 坐标转为数组
        self.steel_ys = np.array(s_ys)  # 将型钢纤维 Y 坐标转为数组
        self.steel_As = np.array(s_As)  # 将型钢纤维面积转为数组
        self.A_steel = np.sum(self.steel_As) if has_steel else 0.0  # 计算型钢总面积

        # ================= 3. 规范极值更新 =================
        # 理论纯压点 Po 承载力 = 混凝土受压 + 钢筋受压 + 型钢受压
        self.Po = 0.85 * self.fc * (self.Ag - self.Ast_total - self.A_steel) + \
                  self.fy * self.Ast_total + self.fya * self.A_steel
        self.Pn_max = 0.80 * self.Po  # ACI 318 规定：箍筋柱偶然偏心截断上限为 0.8Po
        self.Pd_max = 0.65 * self.Pn_max  # ACI 318 规定：受压控制区强度折减系数 φ = 0.65
        # 等效矩形应力块深度系数 beta1，按 ACI 公式计算
        self.beta1 = 0.85 if self.fc <= 28.0 else max(0.65, 0.85 - 0.05 * (self.fc - 28.0) / 7.0)


# =====================================================================
# 计算引擎 1：3D PMM 曲面积分 (纤维法，用于生成“飞艇”包络面)
# =====================================================================
@st.cache_data  # 启用缓存，当用户界面参数不变时，直接返回上一次计算的三维矩阵以提升响应速度
def compute_3d_data(sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                    has_steel, hs, bs, tw, tf, fya):
    _section = RCSection(sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                         has_steel, hs, bs, tw, tf, fya)  # 实例化截面对象

    # 全截面混凝土网格离散化 (X向 40 格，Y向 40 格)
    nx, ny = 40, 40
    X_grid, Y_grid = np.meshgrid(
        np.linspace(-_section.b / 2 + _section.b / (2 * nx), _section.b / 2 - _section.b / (2 * nx), nx),
        np.linspace(-_section.h / 2 + _section.h / (2 * ny), _section.h / 2 - _section.h / (2 * ny), ny))

    if _section.sec_type == '圆形':  # 如果是圆形截面
        mask = X_grid ** 2 + Y_grid ** 2 <= (_section.D / 2) ** 2  # 利用圆方程 x^2+y^2<=R^2 生成裁剪掩码
        xf, yf = X_grid[mask], Y_grid[mask]  # 剔除圆外的网格点，保留真实混凝土纤维
        dA = (_section.D / nx) * (_section.D / ny)  # 定义单根混凝土纤维的微元面积
    else:  # 如果是矩形截面
        xf, yf = X_grid.flatten(), Y_grid.flatten()  # 直接展平全部 1600 个网格点
        dA = (_section.b / nx) * (_section.h / ny)  # 矩形微元面积

    # 定义三维包络面的两条扫描线：
    alpha_angles = np.linspace(0, 2 * np.pi, 36)  # 扫描角 alpha：从 0 到 360 度，等分 36 份 (控制圆周)
    c_values = np.append(np.logspace(np.log10(_section.h * 100), np.log10(1), 60), -1e-6)  # 中和轴 c：从极大(纯压)到负(纯拉) (控制高度)

    # 初始化 6 个大矩阵，用于存储三维空间的坐标点
    Pn_mesh, Mnx_mesh, Mny_mesh = (np.zeros((len(c_values), len(alpha_angles))) for _ in range(3))  # 名义曲面网格
    Pd_mesh, Mdx_mesh, Mdy_mesh = (np.zeros((len(c_values), len(alpha_angles))) for _ in range(3))  # 设计曲面网格

    for j, alpha in enumerate(alpha_angles):  # 外层循环：遍历每一个弯曲角度 (空间旋转)
        # 核心几何变换：将坐标系旋转 alpha 角。此时对于倾斜弯曲，转化为相对旋转坐标系 y' 轴的单向受压
        yf_prime = xf * np.sin(alpha) + yf * np.cos(alpha)  # 混凝土纤维投影 Y 坐标
        ys_prime = _section.xs_coords * np.sin(alpha) + _section.ys_coords * np.cos(alpha)  # 钢筋投影 Y 坐标

        if _section.has_steel:  # 寻找当前旋转角度下，截面的最外侧受压边缘位置
            yst_prime = _section.steel_xs * np.sin(alpha) + _section.steel_ys * np.cos(alpha)  # 型钢投影 Y 坐标
            y_prime_max = np.max(np.concatenate([yf_prime, ys_prime, yst_prime])) + 1e-5  # 取所有坐标最大值
            df_steel = y_prime_max - yst_prime  # 计算每根型钢纤维距受压边缘的深度
        else:
            y_prime_max = np.max(np.concatenate([yf_prime, ys_prime])) + 1e-5

        df_fibers, ds_rebars = y_prime_max - yf_prime, y_prime_max - ys_prime  # 计算混凝土和钢筋的受压深度

        for i, c in enumerate(c_values):  # 内层循环：遍历每一个中和轴高度
            if c < 0:  # 极值处理：如果 c < 0 代表纯受拉状态
                strain_c, strain_s = np.full_like(df_fibers, -0.01), np.full_like(ds_rebars, -0.01)  # 强行赋予极度拉应变
            else:  # 正常受弯状态：平截面假定
                strain_c, strain_s = _section.ecu * (c - df_fibers) / c, _section.ecu * (
                            c - ds_rebars) / c  # e = ecu*(c-d)/c

            # 1. 计算混凝土的力学贡献
            force_c = get_stress_concrete(strain_c, _section.fc, _section.ecu) * dA  # 调用本构获应力，乘面积得合力
            # 对微元力求和得轴力 P，分别乘以未旋转的原生 x,y 坐标求得双向弯矩 Mx, My
            P_c, Mx_c, My_c = np.sum(force_c), np.sum(force_c * yf), np.sum(force_c * xf)

            # 2. 计算普通钢筋的力学贡献
            # 调用钢筋本构获应力，且强制扣除受压区混凝土本构(防止混凝土和钢筋在同一位置被重复计算面积)
            stress_s = get_stress_rebar(strain_s, _section.fy, _section.Es) - get_stress_concrete(strain_s, _section.fc,
                                                                                                  _section.ecu)
            force_s = stress_s * _section.As_array  # 应力乘各自面积得钢筋内力
            P_s = np.sum(force_s)  # 钢筋总轴力
            Mx_s = np.sum(force_s * _section.ys_coords)  # 钢筋对 X 轴弯矩
            My_s = np.sum(force_s * _section.xs_coords)  # 钢筋对 Y 轴弯矩

            # 3. 计算内置型钢的力学贡献 (逻辑同普通钢筋)
            if _section.has_steel:
                strain_st = _section.ecu * (c - df_steel) / c if c > 0 else np.full_like(df_steel, -0.01)
                stress_st = get_stress_steel(strain_st, _section.fya, _section.Es) - get_stress_concrete(strain_st,
                                                                                                         _section.fc,
                                                                                                         _section.ecu)
                force_st = stress_st * _section.steel_As
                P_s += np.sum(force_st)  # 将型钢内力直接叠加到钢筋内力变量 P_s 中
                Mx_s += np.sum(force_st * _section.steel_ys)
                My_s += np.sum(force_st * _section.steel_xs)

            # 汇总当前状态下截面的名义内力
            Pn, Mnx, Mny = P_c + P_s, Mx_c + Mx_s, My_c + My_s

            # ACI 规范折减系数 phi 动态计算逻辑
            et = np.max(-strain_s) if len(strain_s) > 0 else 0  # 找出截面中最外层受拉钢筋的应变 (拉定义为正)
            if et <= _section.ey:
                phi = 0.65  # 受压控制区：较危险，折减大
            elif et >= 0.005:
                phi = 0.90  # 受拉控制区：延性好，折减小
            else:
                phi = 0.65 + 0.25 * (et - _section.ey) / (0.005 - _section.ey)  # 过渡区线性插值

            # 轴压上限平顶截断逻辑
            Pd, Mdx, Mdy = phi * Pn, phi * Mnx, phi * Mny  # 理论设计内力
            Pd_plot = min(Pd, _section.Pd_max)  # 强制轴压上限不得超过规范规定的 Pd_max
            # 如果发生了截断，弯矩也要按比例向心缩减，形成完美的平顶锥体
            Mdx_plot = Mdx * (_section.Pd_max / Pd) if Pd > _section.Pd_max and Pd != 0 else Mdx
            Mdy_plot = Mdy * (_section.Pd_max / Pd) if Pd > _section.Pd_max and Pd != 0 else Mdy

            # 将内力由 N, N*mm 转换为 kN, kN*m 后存入网格矩阵
            Pn_mesh[i, j], Mnx_mesh[i, j], Mny_mesh[i, j] = Pn / 1000, Mnx / 1e6, Mny / 1e6
            Pd_mesh[i, j], Mdx_mesh[i, j], Mdy_mesh[i, j] = Pd_plot / 1000, Mdx_plot / 1e6, Mdy_plot / 1e6

    return Pn_mesh, Mnx_mesh, Mny_mesh, Pd_mesh, Mdx_mesh, Mdy_mesh


# =====================================================================
# 计算引擎 2：严谨 2D 双向扫描 (用于主次轴高精度验证，包含弓形解析解)
# =====================================================================
@st.cache_data
def compute_2d_pm_strict(sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                         has_steel, hs, bs, tw, tf, fya, method, bending_axis='X'):
    _section = RCSection(sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                         has_steel, hs, bs, tw, tf, fya)

    # 通过交换几何和坐标变量，实现 0度和90度方向的通用计算，无需写两套代码
    if bending_axis == 'X':
        h_dim, b_dim = _section.h, _section.b  # 绕 X 轴弯曲，高度为 Y 向长度
        y_coords, steel_y_coords = _section.ys_coords, _section.steel_ys
    else:
        h_dim, b_dim = _section.b, _section.h  # 绕 Y 轴弯曲，高度为 X 向长度
        y_coords, steel_y_coords = _section.xs_coords, _section.steel_xs

    dy = h_dim / 100  # 为 2D 纤维法划分 100 层薄片
    y_fibers = np.linspace(-h_dim / 2 + dy / 2, h_dim / 2 - dy / 2, 100)  # 纤维中心 Y 坐标

    def calc_point(c, comp_side):
        """单点内力计算子函数，comp_side 决定是从顶部往下压还是从底部往上压"""
        P_c, M_c = 0.0, 0.0

        # --- 混凝土受压计算 ---
        if method == 'stress_block':  # 【方法一：等效应力块法】
            a = min(_section.beta1 * c, h_dim)  # 计算等效应力块高度 a
            if _section.sec_type == '矩形':  # 矩形截面计算极其简单
                P_c = 0.85 * _section.fc * a * b_dim  # Pc = 应力 * 宽 * a
                y_pc = h_dim / 2 - a / 2 if comp_side == 'top' else -h_dim / 2 + a / 2  # 形心在 a/2 处
                M_c = P_c * y_pc
            else:
                R = _section.D / 2  # 【圆形截面复杂解析计算：弓形面积与形心】
                if a >= 2 * R:  # 全截面受压
                    A_c = np.pi * R ** 2;
                    y_pc = 0.0
                else:  # 截面部分受压，形成一个弓形
                    dist = R - a  # 圆心到受压边界的距离
                    alpha = np.arccos(dist / R)  # 弓形半圆心角
                    A_c = R ** 2 * (alpha - np.sin(alpha) * np.cos(alpha))  # 弓形面积解析公式
                    y_pc = (2 * R ** 3 * np.sin(alpha) ** 3) / (3 * A_c) if A_c > 0 else 0  # 弓形形心距圆心距离解析公式
                P_c = 0.85 * _section.fc * A_c
                y_pc = y_pc if comp_side == 'top' else -y_pc  # 判断受压区在上还是下
                M_c = P_c * y_pc

        elif method == 'fiber':  # 【方法二：二维纤维积分法】
            for y in y_fibers:  # 遍历 100 层微元
                depth = h_dim / 2 - y if comp_side == 'top' else y - (-h_dim / 2)  # 当前纤维距受压边缘深度
                strain = _section.ecu * (c - depth) / c  # 根据平截面假定得应变
                if strain > 0:
                    stress = get_stress_concrete(strain, _section.fc, _section.ecu)  # 调本构算应力
                    # 关键步：圆形截面的水平宽度会随深度发生非线性变化 (弦长公式)
                    width = b_dim if _section.sec_type == '矩形' else 2 * np.sqrt(
                        max((_section.D / 2) ** 2 - y ** 2, 0))
                    force = stress * width * dy  # 求本层合力
                    P_c += force;
                    M_c += force * y  # 累加轴力和弯矩

        # --- 调用外部本构函数：钢筋 ---
        F_s, M_s = 0.0, 0.0
        strains_s = []
        for y, As in zip(y_coords, _section.As_array):  # 遍历每根钢筋
            depth = h_dim / 2 - y if comp_side == 'top' else y - (-h_dim / 2)
            strain = _section.ecu * (c - depth) / c
            strains_s.append(strain)

            stress = get_stress_rebar(strain, _section.fy, _section.Es)  # 取钢筋应力
            if strain > 0:  # 如果处于受压区，必须扣除该位置混凝土应力
                if method == 'stress_block' and depth < _section.beta1 * c:
                    stress -= 0.85 * _section.fc  # 扣除矩形应力块的值
                elif method == 'fiber':
                    stress -= get_stress_concrete(strain, _section.fc, _section.ecu)  # 扣除抛物线应力的值
            F_s += stress * As;
            M_s += stress * As * y

        # --- 调用外部本构函数：型钢 ---
        if _section.has_steel:  # 型钢的逻辑与普通钢筋完全一致
            for y, As in zip(steel_y_coords, _section.steel_As):
                depth = h_dim / 2 - y if comp_side == 'top' else y - (-h_dim / 2)
                strain = _section.ecu * (c - depth) / c

                stress = get_stress_steel(strain, _section.fya, _section.Es)
                if strain > 0:
                    if method == 'stress_block' and depth < _section.beta1 * c:
                        stress -= 0.85 * _section.fc
                    elif method == 'fiber':
                        stress -= get_stress_concrete(strain, _section.fc, _section.ecu)
                F_s += stress * As;
                M_s += stress * As * y

        Pn, Mn = P_c + F_s, M_c + M_s  # 汇总得到总名义内力
        et = -np.min(strains_s) if len(strains_s) > 0 else 0  # Phi计算逻辑
        phi = 0.65 if et <= _section.ey else (
            0.90 if et >= 0.005 else 0.65 + 0.25 * (et - _section.ey) / (0.005 - _section.ey))

        Pd, Md = phi * Pn, phi * Mn  # 设计内力及平顶截断
        Pd_plot = min(Pd, _section.Pd_max)
        Md_plot = Md * (_section.Pd_max / Pd) if Pd > _section.Pd_max and Pd != 0 else Md

        return Pn, Mn, Pd_plot, Md_plot

    # 执行 360 度双向闭合扫描，构建二维包络线阵列
    Pn_list, Mn_list, Pd_list, Md_list = [], [], [], []

    # 手动计算极其特殊的起止点 (全截面纯拉与纯压)，考虑到非对称截面时可能伴生的微小弯矩 M_Po
    Pt = -_section.fy * _section.Ast_total - (_section.fya * _section.A_steel if _section.has_steel else 0)
    stress_c_pure = _section.fy - 0.85 * _section.fc
    stress_st_pure = _section.fya - 0.85 * _section.fc
    M_Po = sum([stress_c_pure * As * y for y, As in zip(y_coords, _section.As_array)])
    M_Pt = sum([-_section.fy * As * y for y, As in zip(y_coords, _section.As_array)])
    if _section.has_steel:
        M_Po += sum([stress_st_pure * As * y for y, As in zip(steel_y_coords, _section.steel_As)])
        M_Pt += sum([-_section.fya * As * y for y, As in zip(steel_y_coords, _section.steel_As)])

    def add_pt(pn, mn, pd, md):  # 工具函数：追加坐标点
        Pn_list.append(pn);
        Mn_list.append(mn);
        Pd_list.append(pd);
        Md_list.append(md)

    # 1. 压入终极纯拉点
    add_pt(Pt, M_Pt, 0.9 * Pt, 0.9 * M_Pt)
    # 2. 扫过底部受压(负弯矩)区域，c 从极小(趋近纯拉)增大到极大(纯压)
    for c in np.logspace(np.log10(1), np.log10(h_dim * 10000), 100):
        add_pt(*calc_point(c, 'bottom'))
    # 3. 压入终极纯压点 (用于连接正负弯矩区，如果不对称此点将不在 Y 轴上)
    add_pt(_section.Po, M_Po, _section.Pd_max, 0.65 * M_Po)
    # 4. 扫过顶部受压(正弯矩)区域，c 从极大(纯压)变回极小(趋近纯拉)
    for c in np.logspace(np.log10(h_dim * 10000), np.log10(1), 100):
        add_pt(*calc_point(c, 'top'))
    # 5. 压入终极纯拉点，形成完美闭合环
    add_pt(Pt, M_Pt, 0.9 * Pt, 0.9 * M_Pt)

    return np.array(Pn_list) / 1000, np.array(Mn_list) / 1e6, np.array(Pd_list) / 1000, np.array(Md_list) / 1e6


# =====================================================================
# 后处理工具：从 3D 网格中提取指定水平标高(即指定轴力) 的 Mx-My 切片
# =====================================================================
def extract_mx_my_contour(target_P, P_mesh, Mx_mesh, My_mesh):
    contour_Mx, contour_My = [], []  # 初始化用来存放插值结果的坐标列表
    for j in range(P_mesh.shape[1]):  # 遍历空间曲面的 36 个旋转经度角
        sort_idx = np.argsort(P_mesh[:, j])  # 找到该经度线上，让 P 值由小到大单调递增的排序索引
        P_sorted, Mx_sorted, My_sorted = P_mesh[:, j][sort_idx], Mx_mesh[:, j][sort_idx], My_mesh[:, j][sort_idx]

        # np.interp 执行一维线性插值。利用已知的递增 P 序列，查找 target_P 对应的 Mx 和 My
        Mx_val = np.interp(target_P, P_sorted, Mx_sorted, left=np.nan, right=np.nan)
        My_val = np.interp(target_P, P_sorted, My_sorted, left=np.nan, right=np.nan)
        contour_Mx.append(Mx_val);
        contour_My.append(My_val)  # 保存当前经线产生的切点

    contour_Mx.append(contour_Mx[0]);
    contour_My.append(contour_My[0])  # 首尾相连，闭合多边形
    return np.array(contour_Mx), np.array(contour_My)


# =====================================================================
# Web 界面布局与交互逻辑 (基于 Streamlit)
# =====================================================================
st.title("🏗️ P-M-M 截面分析系统 (含型钢 SRC 及圆形支持)")  # 网页大标题

with st.sidebar:  # 【侧边栏组件域】用户控制面板
    st.header("⚙️ 参数设定区")
    # 单选框：选择核心算法引擎
    engine_mode = st.radio("1. 选择计算引擎：", ("3D PMM 曲面", "2D 等效应力块法", "2D 纤维法"), index=0)

    custom_alpha = 45;
    target_N = 1000.0  # 初始化默认值
    if engine_mode == "3D PMM 曲面":  # 只有在 3D 模式下才激活这几个控件
        custom_alpha = st.slider("自定义 2D 切片角度 $\\alpha$ (度)", 0, 180, 45, 5)
        target_N = st.number_input("指定设计轴力 $N$ (kN)", value=1000.0, step=100.0)

    st.divider()  # 界面分隔线
    # 截面形状开关：选择后动态隐藏或展示对应的长宽/直径输入框
    sec_type = st.radio("2. 截面形状：", ["矩形 (Rectangular)", "圆形 (Circular)"], horizontal=True)

    col1, col2 = st.columns(2)  # 并排两列布局
    if '矩形' in sec_type:
        b = col1.number_input("宽度 $b$ (mm)", value=400.0, step=50.0)
        h = col2.number_input("高度 $h$ (mm)", value=600.0, step=50.0)
        D = 0.0;
        n_circ = 0
    else:
        b = 0.0;
        h = 0.0
        D = col1.number_input("直径 $D$ (mm)", value=600.0, step=50.0)
        n_circ = col2.number_input("环绕钢筋根数", value=12, min_value=4, step=1)

    fc = col1.number_input("砼强度 $f'_c$ (MPa)", value=30.0, step=5.0)
    fy = col2.number_input("钢筋强度 $f_y$ (MPa)", value=400.0, step=50.0)

    st.subheader("3. 钢筋排布参数")
    cover = st.number_input("保护层厚度 (mm)", value=50, step=5)
    d_bar = st.number_input("钢筋直径 (mm)", value=20.0, step=2.0)

    if '矩形' in sec_type:  # 若为矩形，则定义具体的阵列排数
        col3, col4 = st.columns(2)
        n_top = col3.number_input("顶部排钢筋", value=3, step=1)
        n_bot = col4.number_input("底部排钢筋", value=3, step=1)
        n_side = st.number_input("单侧中部钢筋", value=2, step=1)
    else:
        n_top = n_bot = n_side = 0

    st.divider()
    st.subheader("4. 内部型钢配置 (SRC)")
    has_steel = st.toggle("内置 H 型钢")  # 拨动开关：是否启用内置型钢
    if has_steel:
        col_s1, col_s2 = st.columns(2)
        hs = col_s1.number_input("型钢高 $h_s$ (mm)", value=300.0, step=50.0)
        bs = col_s2.number_input("翼缘宽 $b_s$ (mm)", value=200.0, step=50.0)
        tw = col_s1.number_input("腹板厚 $t_w$ (mm)", value=10.0, step=2.0)
        tf = col_s2.number_input("翼缘厚 $t_f$ (mm)", value=14.0, step=2.0)
        fya = st.number_input("型钢屈服 $f_{ya}$ (MPa)", value=355.0, step=10.0)
    else:
        hs = bs = tw = tf = fya = 0.0

    run_btn = st.button("🚀 开始计算", use_container_width=True, type="primary")  # 执行按钮

if run_btn:  # 【主执行流】当用户点击计算按钮后
    with st.spinner(f"正在进行核心分析，请稍候..."):  # 显示加载中菊花动画
        # 用输入参数实例化截面，主要用于获取坐标点以便画预览图
        sec = RCSection('矩形' if '矩形' in sec_type else '圆形', b, h, D, fc, fy, 200000.0, cover, int(n_top),
                        int(n_bot), int(n_side), int(n_circ), d_bar, has_steel, hs, bs, tw, tf, fya)

        # ====== 画左侧的截面俯视预览图 ======
        fig_sec, ax_sec = plt.subplots(figsize=(4, 4))
        if sec.sec_type == '矩形':  # 画矩形外轮廓边框
            ax_sec.plot([-sec.b / 2, sec.b / 2, sec.b / 2, -sec.b / 2, -sec.b / 2],
                        [sec.h / 2, sec.h / 2, -sec.h / 2, -sec.h / 2, sec.h / 2], 'k-', linewidth=2)
        else:  # 画圆形外轮廓边框
            theta = np.linspace(0, 2 * np.pi, 100)
            ax_sec.plot(sec.D / 2 * np.cos(theta), sec.D / 2 * np.sin(theta), 'k-', linewidth=2)

        ax_sec.scatter(sec.xs_coords, sec.ys_coords, s=60, c='red', zorder=5)  # 用红点打出所有的钢筋位置

        if has_steel:  # 如果有型钢，绘制漂亮的蓝色 H 轮廓
            # 严格顺时针排列的点集，解决漏斗交叉 Bug
            h_poly = [-bs / 2, bs / 2, bs / 2, tw / 2, tw / 2, bs / 2, bs / 2, -bs / 2, -bs / 2, -tw / 2, -tw / 2,
                      -bs / 2, -bs / 2]
            v_poly = [hs / 2, hs / 2, hs / 2 - tf, hs / 2 - tf, -hs / 2 + tf, -hs / 2 + tf, -hs / 2, -hs / 2,
                      -hs / 2 + tf, -hs / 2 + tf, hs / 2 - tf, hs / 2 - tf, hs / 2]
            ax_sec.plot(h_poly, v_poly, 'b-', linewidth=1.5)
            ax_sec.fill(h_poly, v_poly, 'blue', alpha=0.2)

        ax_sec.set_aspect('equal');
        ax_sec.axis('off')  # 锁定比例并关闭坐标轴网格
        title_str = f"Section Preview\nAs={sec.Ast_total:.0f} mm²" + (
            f", Aa={sec.A_steel:.0f} mm²" if has_steel else "")
        ax_sec.set_title(title_str, fontsize=10)

        # 开始主页面排版：左边极值统计区(占宽1)，右侧图表图(占宽4)
        col_res1, col_res2 = st.columns([1, 4])
        with col_res1:
            st.pyplot(fig_sec)  # 渲染预览图
            st.info(
                f"**总含钢率**: {((sec.Ast_total + sec.A_steel) / sec.Ag) * 100:.2f}%\n\n**理论最大轴压 $P_o$**: {sec.Po / 1000:.1f} kN\n\n**设计最大轴压 $N_{{u,max}}$**: {sec.Pd_max / 1000:.1f} kN")  # 渲染规范极值信息框

        with col_res2:
            # ==========================================
            # 分支一：执行并渲染 3D 面板
            # ==========================================
            if engine_mode == "3D PMM 曲面":
                # 将 UI 所有的原始参数传入缓存引擎，获取生成 3D 面板用的六大矩阵
                Pn_mesh, Mnx_mesh, Mny_mesh, Pd_mesh, Mdx_mesh, Mdy_mesh = compute_3d_data(
                    sec.sec_type, b, h, D, fc, fy, 200000.0, cover, int(n_top), int(n_bot), int(n_side), int(n_circ),
                    d_bar, has_steel, hs, bs, tw, tf, fya)

                tab1, tab2 = st.tabs(["📊 3D 曲面与主轴切片", "🎯 指定轴力 Mx-My 交互曲线"])  # 创建双页签

                with tab1:  # 第 1 页内容
                    fig_main = plt.figure(figsize=(15, 10))
                    # 绘制上半部分的巨大 3D 曲面
                    ax_3d = fig_main.add_subplot(2, 3, (1, 3), projection='3d')
                    ax_3d.plot_surface(Mnx_mesh, Mny_mesh, Pn_mesh, color='cyan', edgecolor='c', linewidth=0.1,
                                       alpha=0.15)  # 半透明的庞大名义曲面
                    ax_3d.plot_surface(Mdx_mesh, Mdy_mesh, Pd_mesh, color='red', edgecolor='darkred', linewidth=0.3,
                                       alpha=0.7)  # 实体被截断的微小设计曲面
                    ax_3d.set_title('3D P-Mx-My Interaction Surface', fontsize=14)
                    ax_3d.set_xlabel('Mx (kN·m)');
                    ax_3d.set_ylabel('My (kN·m)');
                    ax_3d.set_zlabel('P (kN)')
                    nom_patch = mpatches.Patch(color='cyan', alpha=0.3, label='Theoretical Nominal ($P_o$)')
                    des_patch = mpatches.Patch(color='red', alpha=0.7, label='Design ($\phi P_n$, Capped)')
                    ax_3d.legend(handles=[nom_patch, des_patch], loc='upper left')
                    ax_3d.view_init(elev=20, azim=40)  # 设定默认观察仰角


                    def extract_2d_slice(alpha_deg):  # 闭包辅助函数：从 3D 网格中提取指定竖向剖面的 2D 环
                        idx_pos = int((alpha_deg / 360.0) * 36)  # 寻找正侧面数组索引
                        idx_neg = int(((alpha_deg + 180) / 360.0) * 36) % 36  # 寻找负侧面数组索引
                        Mres_n_nom = np.sqrt(Mnx_mesh[:, idx_neg] ** 2 + Mny_mesh[:, idx_neg] ** 2) * -1
                        Mres_p_nom = np.sqrt(Mnx_mesh[:, idx_pos] ** 2 + Mny_mesh[:, idx_pos] ** 2)
                        Mres_n_des = np.sqrt(Mdx_mesh[:, idx_neg] ** 2 + Mdy_mesh[:, idx_neg] ** 2) * -1
                        Mres_p_des = np.sqrt(Mdx_mesh[:, idx_pos] ** 2 + Mdy_mesh[:, idx_pos] ** 2)
                        return (np.concatenate([Mres_n_nom[::-1], Mres_p_nom]),
                                np.concatenate([Pn_mesh[:, idx_neg][::-1], Pn_mesh[:, idx_pos]]),
                                np.concatenate([Mres_n_des[::-1], Mres_p_des]),
                                np.concatenate([Pd_mesh[:, idx_neg][::-1], Pd_mesh[:, idx_pos]]))


                    # 绘制底部的三个关联二维护航切片视图
                    for idx, (ang, title) in enumerate(zip([0, 90, custom_alpha],
                                                           ['Slice @ 0° (P-Mx)', 'Slice @ 90° (P-My)',
                                                            f'Slice @ {custom_alpha}° (P-M$\\alpha$)'])):
                        ax = fig_main.add_subplot(2, 3, 4 + idx)
                        M_nom, P_nom, M_des, P_des = extract_2d_slice(ang)
                        ax.plot(M_nom, P_nom, 'c-.', linewidth=2, label='Nominal')
                        ax.plot(M_des, P_des, 'r-', linewidth=2, label='Design')
                        ax.axhline(0, color='k', linewidth=0.8);
                        ax.axvline(0, color='k', linewidth=0.8)
                        ax.axhline(sec.Po / 1000, color='cyan', linestyle=':', alpha=0.6)
                        ax.axhline(sec.Pd_max / 1000, color='red', linestyle=':', alpha=0.6)
                        ax.set_title(title, fontsize=11);
                        ax.set_xlabel('Moment (kN·m)');
                        ax.set_ylabel('Axial (kN)')
                        ax.grid(True, linestyle=':', alpha=0.7);
                        ax.legend(fontsize=8)

                    fig_main.tight_layout()  # 让图表元素不要挤压重叠
                    st.pyplot(fig_main)  # 渲染该大图

                with tab2:  # 第 2 页内容：双偏压横截面剖分图
                    st.subheader(f"🔍 截面在指定轴向荷载 $N = {target_N}$ kN 下的抗弯包络线")
                    max_N_des = sec.Pd_max / 1000;
                    max_N_nom = sec.Po / 1000
                    min_N_ten = (-sec.fy * sec.Ast_total - (sec.fya * sec.A_steel if has_steel else 0)) * 0.9 / 1000
                    plot_N_des = target_N;
                    plot_N_nom = target_N

                    # 安全纠错：对异常或过大的轴力进行警告并截断替代，防止插值函数崩溃崩溃
                    if target_N >= max_N_des:
                        st.warning(
                            f"⚠️ **警示**：指定轴力超出设计上限 $N_{{u,max}} = {max_N_des:.1f}$ kN。设计曲线降级至最大边界。")
                        plot_N_des = max_N_des - 1e-3
                    if target_N >= max_N_nom:
                        plot_N_nom = max_N_nom - 1e-3
                    if target_N <= min_N_ten:
                        st.warning(f"⚠️ **警示**：指定拉力超限。按纯拉状态显示。")
                        plot_N_des = min_N_ten + 1e-3;
                        plot_N_nom = min_N_ten + 1e-3

                    # 调用插值算法，切出目标平面的二维坐标数组
                    mx_nom, my_nom = extract_mx_my_contour(plot_N_nom, Pn_mesh, Mnx_mesh, Mny_mesh)
                    mx_des, my_des = extract_mx_my_contour(plot_N_des, Pd_mesh, Mdx_mesh, Mdy_mesh)

                    # 绘制横断面等高线图
                    fig_contour, ax_c = plt.subplots(figsize=(5.5, 5.5))
                    ax_c.plot(mx_nom, my_nom, 'c-.', linewidth=2, label=f'Nominal Curve ($P_n = {plot_N_nom:.1f}$ kN)')
                    ax_c.plot(mx_des, my_des, 'r-', linewidth=2.5, label=f'Design Curve ($P_d = {plot_N_des:.1f}$ kN)')
                    ax_c.axhline(0, color='k', linewidth=1);
                    ax_c.axvline(0, color='k', linewidth=1)
                    ax_c.set_xlabel('Mx (kN·m)', fontsize=11);
                    ax_c.set_ylabel('My (kN·m)', fontsize=11)
                    ax_c.set_title(f'Mx-My Contour at N = {target_N:.1f} kN', fontsize=12)
                    ax_c.grid(True, linestyle=':', alpha=0.7);
                    ax_c.set_aspect('equal', adjustable='datalim');
                    ax_c.legend(fontsize=10)

                    # 为了让该图居中且不要过大，利用空白列进行夹逼占位
                    col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
                    with col_chart:
                        st.pyplot(fig_contour)

            # ==========================================
            # 分支二：执行纯 2D 降维独立引擎 (手算对账最爱)
            # ==========================================
            else:
                # 判别选用哪种微积分手段
                method = 'stress_block' if '等效应力块' in engine_mode else 'fiber'
                # 执行独立的纯正双向扫描算法，分别跑出沿 X 轴和 Y 轴偏压的精准结果
                Pn_x, Mn_x, Pd_x, Md_x = compute_2d_pm_strict(
                    sec.sec_type, b, h, D, fc, fy, 200000.0, cover, int(n_top), int(n_bot), int(n_side), int(n_circ),
                    d_bar, has_steel, hs, bs, tw, tf, fya, method, bending_axis='X')
                Pn_y, Mn_y, Pd_y, Md_y = compute_2d_pm_strict(
                    sec.sec_type, b, h, D, fc, fy, 200000.0, cover, int(n_top), int(n_bot), int(n_side), int(n_circ),
                    d_bar, has_steel, hs, bs, tw, tf, fya, method, bending_axis='Y')

                fig_2d, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12, 6))  # 并排画两张图

                # 画 0 度主轴包络线
                ax_x.plot(Mn_x, Pn_x, 'c-.', linewidth=2, label='Nominal');
                ax_x.plot(Md_x, Pd_x, 'r-', linewidth=2, label='Design')
                ax_x.set_title(f'P-Mx (0°) Main Axis Bending\nMethod: {method}', fontsize=12)

                # 画 90 度次轴包络线
                ax_y.plot(Mn_y, Pn_y, 'c-.', linewidth=2, label='Nominal');
                ax_y.plot(Md_y, Pd_y, 'r-', linewidth=2, label='Design')
                ax_y.set_title(f'P-My (90°) Minor Axis Bending\nMethod: {method}', fontsize=12)

                # 为两个视图统一添加参考标线与网格美化
                for ax in (ax_x, ax_y):
                    ax.axhline(0, color='k', linewidth=0.8);
                    ax.axvline(0, color='k', linewidth=0.8)  # 原点十字
                    ax.axhline(sec.Po / 1000, color='cyan', linestyle=':', alpha=0.6)  # 理论名义纯压轴力线
                    ax.axhline(sec.Pd_max / 1000, color='red', linestyle=':', alpha=0.6)  # 规范截断轴力线
                    ax.set_xlabel('Moment (kN·m)');
                    ax.set_ylabel('Axial Load (kN)')
                    ax.grid(True, linestyle=':', alpha=0.7);
                    ax.legend()

                fig_2d.tight_layout()
                st.pyplot(fig_2d)  # 渲染大图
else:
    st.info("👈 请在侧边栏配置各项参数，然后点击 **开始计算**。")  # 当程序刚启动，还没点击按钮时给用户的贴心提示