import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import scipy.optimize as opt

# =====================================================================
# 页面基础设置
# =====================================================================
st.set_page_config(page_title="P-M-M 截面设计与校核系统", page_icon="🏗️", layout="wide")


# =====================================================================
# 独立材料本构关系库
# =====================================================================
def get_stress_concrete(strain, fc, ecu=0.003):
    e0 = 0.002
    strain_arr = np.asarray(strain)
    stress = np.zeros_like(strain_arr, dtype=float)
    mask_para = (strain_arr > 0) & (strain_arr <= e0)
    mask_rect = (strain_arr > e0) & (strain_arr <= ecu)
    stress[mask_para] = 0.85 * fc * (2 * (strain_arr[mask_para] / e0) - (strain_arr[mask_para] / e0) ** 2)
    stress[mask_rect] = 0.85 * fc
    if np.isscalar(strain): return stress.item()
    return stress


def get_stress_rebar(strain, fy, Es):
    return np.clip(strain * Es, -fy, fy)


def get_stress_steel(strain, fya, Es):
    return np.clip(strain * Es, -fya, fya)


# =====================================================================
# 核心类：钢筋混凝土及型钢截面定义
# =====================================================================
class RCSection:
    def __init__(self, sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                 has_steel, hs, bs, tw, tf, fya):
        self.sec_type = sec_type
        self.b = float(b) if sec_type == '矩形' else float(D)
        self.h = float(h) if sec_type == '矩形' else float(D)
        self.D = float(D)
        self.fc = float(fc)
        self.fy = float(fy)
        self.Es = float(Es)
        self.ecu = 0.003
        self.ey = self.fy / self.Es
        self.Ag = self.b * self.h if sec_type == '矩形' else np.pi * (self.D / 2) ** 2

        A_bar = np.pi * (d_bar ** 2) / 4
        xs_list, ys_list = [], []
        if sec_type == '矩形':
            if n_top > 0:
                xs_list.extend(np.linspace(-self.b / 2 + cover, self.b / 2 - cover, n_top))
                ys_list.extend([self.h / 2 - cover] * n_top)
            if n_bot > 0:
                xs_list.extend(np.linspace(-self.b / 2 + cover, self.b / 2 - cover, n_bot))
                ys_list.extend([-self.h / 2 + cover] * n_bot)
            if n_side > 0:
                y_side = np.linspace(self.h / 2 - cover, -self.h / 2 + cover, n_side + 2)[1:-1]
                for y in y_side:
                    xs_list.extend([-self.b / 2 + cover, self.b / 2 - cover])
                    ys_list.extend([y, y])
        else:
            R_rebar = self.D / 2 - cover
            angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
            xs_list = R_rebar * np.sin(angles)
            ys_list = R_rebar * np.cos(angles)

        self.xs_coords = np.array(xs_list)
        self.ys_coords = np.array(ys_list)
        self.As_array = np.full(len(xs_list), A_bar)
        self.Ast_total = np.sum(self.As_array)

        self.has_steel = has_steel
        self.fya = float(fya)
        s_xs, s_ys, s_As = [], [], []

        if has_steel:
            nx_f = max(5, int(bs / 20));
            ny_f = max(2, int(tf / 10))
            x_f = np.linspace(-bs / 2 + bs / (2 * nx_f), bs / 2 - bs / (2 * nx_f), nx_f)
            y_f_top = np.linspace(hs / 2 - tf + tf / (2 * ny_f), hs / 2 - tf / (2 * ny_f), ny_f)
            X, Y = np.meshgrid(x_f, y_f_top)
            s_xs.extend(X.flatten());
            s_ys.extend(Y.flatten())
            s_As.extend([bs * tf / (nx_f * ny_f)] * len(X.flatten()))

            y_f_bot = np.linspace(-hs / 2 + tf / (2 * ny_f), -hs / 2 + tf - tf / (2 * ny_f), ny_f)
            X, Y = np.meshgrid(x_f, y_f_bot)
            s_xs.extend(X.flatten());
            s_ys.extend(Y.flatten())
            s_As.extend([bs * tf / (nx_f * ny_f)] * len(X.flatten()))

            nx_w = max(2, int(tw / 10));
            ny_w = max(10, int((hs - 2 * tf) / 20))
            x_w = np.linspace(-tw / 2 + tw / (2 * nx_w), tw / 2 - tw / (2 * nx_w), nx_w)
            y_w = np.linspace(-hs / 2 + tf + (hs - 2 * tf) / (2 * ny_w), hs / 2 - tf - (hs - 2 * tf) / (2 * ny_w), ny_w)
            X, Y = np.meshgrid(x_w, y_w)
            s_xs.extend(X.flatten());
            s_ys.extend(Y.flatten())
            s_As.extend([tw * (hs - 2 * tf) / (nx_w * ny_w)] * len(X.flatten()))

        self.steel_xs = np.array(s_xs)
        self.steel_ys = np.array(s_ys)
        self.steel_As = np.array(s_As)
        self.A_steel = np.sum(self.steel_As) if has_steel else 0.0

        self.Po = 0.85 * self.fc * (self.Ag - self.Ast_total - self.A_steel) + \
                  self.fy * self.Ast_total + self.fya * self.A_steel
        self.Pn_max = 0.80 * self.Po
        self.Pd_max = 0.65 * self.Pn_max
        self.beta1 = 0.85 if self.fc <= 28.0 else max(0.65, 0.85 - 0.05 * (self.fc - 28.0) / 7.0)


# =====================================================================
# 验证核心算法 (Radial, Contour, Bresler)
# =====================================================================
def calc_3d_radial_dc(Pu, Mux, Muy, alpha_angles, P_mesh, Mdx_mesh, Mdy_mesh):
    """算法 1：临界点比值法 (Vector Intersection Method / Radial)"""
    if Pu == 0 and Mux == 0 and Muy == 0: return 0.0, 0.0, 0.0
    req_Mres = np.sqrt(Mux ** 2 + Muy ** 2)
    req_alpha = np.arctan2(Muy, Mux)
    if req_alpha < 0: req_alpha += 2 * np.pi

    slice_P, slice_M = np.zeros(P_mesh.shape[0]), np.zeros(P_mesh.shape[0])
    for i in range(P_mesh.shape[0]):
        slice_P[i] = np.interp(req_alpha, alpha_angles, P_mesh[i, :], period=2 * np.pi)
        mx = np.interp(req_alpha, alpha_angles, Mdx_mesh[i, :], period=2 * np.pi)
        my = np.interp(req_alpha, alpha_angles, Mdy_mesh[i, :], period=2 * np.pi)
        slice_M[i] = np.sqrt(mx ** 2 + my ** 2)

    theta_curve = np.arctan2(slice_P, slice_M)
    theta_req = np.arctan2(Pu, req_Mres)

    sort_idx = np.argsort(theta_curve)
    P_cap = np.interp(theta_req, theta_curve[sort_idx], slice_P[sort_idx])
    M_cap = np.interp(theta_req, theta_curve[sort_idx], slice_M[sort_idx])

    cap_len = np.sqrt(P_cap ** 2 + M_cap ** 2)
    req_len = np.sqrt(Pu ** 2 + req_Mres ** 2)
    if cap_len < 1e-5: return 999.9, req_len, cap_len
    return req_len / cap_len, req_len, cap_len


def extract_mx_my_contour(target_P, P_mesh, Mx_mesh, My_mesh):
    """为算法2提取指定轴力下的 2D 切片轮廓"""
    contour_Mx, contour_My = [], []
    for j in range(P_mesh.shape[1]):
        sort_idx = np.argsort(P_mesh[:, j])
        P_sorted = P_mesh[:, j][sort_idx]
        Mx_sorted = Mx_mesh[:, j][sort_idx]
        My_sorted = My_mesh[:, j][sort_idx]
        Mx_val = np.interp(target_P, P_sorted, Mx_sorted, left=np.nan, right=np.nan)
        My_val = np.interp(target_P, P_sorted, My_sorted, left=np.nan, right=np.nan)
        contour_Mx.append(Mx_val);
        contour_My.append(My_val)
    contour_Mx.append(contour_Mx[0]);
    contour_My.append(contour_My[0])
    return np.array(contour_Mx), np.array(contour_My)


def calc_contour_dc(Mux, Muy, contour_Mx, contour_My):
    """算法 2：恒定轴力切片法 (Constant Load Contour)"""
    if Mux == 0 and Muy == 0: return 0.0
    req_alpha = np.arctan2(Muy, Mux)
    if req_alpha < 0: req_alpha += 2 * np.pi
    cap_alpha = np.arctan2(contour_My, contour_Mx)
    cap_alpha[cap_alpha < 0] += 2 * np.pi

    sort_idx = np.argsort(cap_alpha)
    ca_sorted, mx_sorted, my_sorted = cap_alpha[sort_idx], contour_Mx[sort_idx], contour_My[sort_idx]
    ca_ext = np.concatenate([ca_sorted[-1:] - 2 * np.pi, ca_sorted, ca_sorted[:1] + 2 * np.pi])
    mx_ext = np.concatenate([mx_sorted[-1:], mx_sorted, mx_sorted[:1]])
    my_ext = np.concatenate([my_sorted[-1:], my_sorted, my_sorted[:1]])

    cap_mx = np.interp(req_alpha, ca_ext, mx_ext)
    cap_my = np.interp(req_alpha, ca_ext, my_ext)

    cap_len = np.sqrt(cap_mx ** 2 + cap_my ** 2)
    req_len = np.sqrt(Mux ** 2 + Muy ** 2)
    if cap_len < 1e-5: return 999.9
    return req_len / cap_len


def calc_bresler_dc(Pu, Mux, Muy, Pd_max, Pd_x, Mdx_x, Pd_y, Mdy_y):
    """算法 3：Bresler 倒数经验公式法"""
    if Pu <= 0: return np.nan
    if abs(Mux) < 1e-3 and abs(Muy) < 1e-3: return Pu / Pd_max

    theta_req_x = np.arctan2(Pu, abs(Mux))
    tc_x = np.arctan2(Pd_x, np.abs(Mdx_x))
    sort_x = np.argsort(tc_x)
    Pnx = np.interp(theta_req_x, tc_x[sort_x], Pd_x[sort_x])

    theta_req_y = np.arctan2(Pu, abs(Muy))
    tc_y = np.arctan2(Pd_y, np.abs(Mdy_y))
    sort_y = np.argsort(tc_y)
    Pny = np.interp(theta_req_y, tc_y[sort_y], Pd_y[sort_y])

    if Pnx <= 0 or Pny <= 0 or Pd_max <= 0: return np.nan
    inv_Pni = 1.0 / Pnx + 1.0 / Pny - 1.0 / Pd_max
    if inv_Pni <= 0: return np.nan
    return Pu / (1.0 / inv_Pni)


# =====================================================================
# 计算引擎 1：3D PMM 曲面积分
# =====================================================================
@st.cache_data
def compute_3d_data(sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                    has_steel, hs, bs, tw, tf, fya):
    _section = RCSection(sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                         has_steel, hs, bs, tw, tf, fya)
    nx, ny = 40, 40
    X_grid, Y_grid = np.meshgrid(
        np.linspace(-_section.b / 2 + _section.b / (2 * nx), _section.b / 2 - _section.b / (2 * nx), nx),
        np.linspace(-_section.h / 2 + _section.h / (2 * ny), _section.h / 2 - _section.h / (2 * ny), ny))

    if _section.sec_type == '圆形':
        mask = X_grid ** 2 + Y_grid ** 2 <= (_section.D / 2) ** 2
        xf, yf = X_grid[mask], Y_grid[mask]
        dA = (_section.D / nx) * (_section.D / ny)
    else:
        xf, yf = X_grid.flatten(), Y_grid.flatten()
        dA = (_section.b / nx) * (_section.h / ny)

    alpha_angles = np.linspace(0, 2 * np.pi, 36)
    c_values = np.append(np.logspace(np.log10(_section.h * 100), np.log10(1), 60), -1e-6)

    Pn_mesh, Mnx_mesh, Mny_mesh = (np.zeros((len(c_values), len(alpha_angles))) for _ in range(3))
    Pd_mesh, Mdx_mesh, Mdy_mesh = (np.zeros((len(c_values), len(alpha_angles))) for _ in range(3))

    for j, alpha in enumerate(alpha_angles):
        yf_prime = xf * np.sin(alpha) + yf * np.cos(alpha)
        ys_prime = _section.xs_coords * np.sin(alpha) + _section.ys_coords * np.cos(alpha)
        if _section.has_steel:
            yst_prime = _section.steel_xs * np.sin(alpha) + _section.steel_ys * np.cos(alpha)
            y_prime_max = np.max(np.concatenate([yf_prime, ys_prime, yst_prime])) + 1e-5
            df_steel = y_prime_max - yst_prime
        else:
            y_prime_max = np.max(np.concatenate([yf_prime, ys_prime])) + 1e-5
        df_fibers, ds_rebars = y_prime_max - yf_prime, y_prime_max - ys_prime

        for i, c in enumerate(c_values):
            if c < 0:
                strain_c, strain_s = np.full_like(df_fibers, -0.01), np.full_like(ds_rebars, -0.01)
            else:
                strain_c, strain_s = _section.ecu * (c - df_fibers) / c, _section.ecu * (c - ds_rebars) / c

            force_c = get_stress_concrete(strain_c, _section.fc, _section.ecu) * dA
            P_c, Mx_c, My_c = np.sum(force_c), np.sum(force_c * yf), np.sum(force_c * xf)
            stress_s = get_stress_rebar(strain_s, _section.fy, _section.Es) - get_stress_concrete(strain_s, _section.fc,
                                                                                                  _section.ecu)
            force_s = stress_s * _section.As_array
            P_s = np.sum(force_s);
            Mx_s = np.sum(force_s * _section.ys_coords);
            My_s = np.sum(force_s * _section.xs_coords)

            if _section.has_steel:
                strain_st = _section.ecu * (c - df_steel) / c if c > 0 else np.full_like(df_steel, -0.01)
                stress_st = get_stress_steel(strain_st, _section.fya, _section.Es) - get_stress_concrete(strain_st,
                                                                                                         _section.fc,
                                                                                                         _section.ecu)
                force_st = stress_st * _section.steel_As
                P_s += np.sum(force_st);
                Mx_s += np.sum(force_st * _section.steel_ys);
                My_s += np.sum(force_st * _section.steel_xs)

            Pn, Mnx, Mny = P_c + P_s, Mx_c + Mx_s, My_c + My_s
            et = np.max(-strain_s) if len(strain_s) > 0 else 0
            phi = 0.65 if et <= _section.ey else (
                0.90 if et >= 0.005 else 0.65 + 0.25 * (et - _section.ey) / (0.005 - _section.ey))
            Pd, Mdx, Mdy = phi * Pn, phi * Mnx, phi * Mny
            Pd_plot = min(Pd, _section.Pd_max)
            Mdx_plot = Mdx * (_section.Pd_max / Pd) if Pd > _section.Pd_max and Pd != 0 else Mdx
            Mdy_plot = Mdy * (_section.Pd_max / Pd) if Pd > _section.Pd_max and Pd != 0 else Mdy

            Pn_mesh[i, j], Mnx_mesh[i, j], Mny_mesh[i, j] = Pn / 1000, Mnx / 1e6, Mny / 1e6
            Pd_mesh[i, j], Mdx_mesh[i, j], Mdy_mesh[i, j] = Pd_plot / 1000, Mdx_plot / 1e6, Mdy_plot / 1e6

    return Pn_mesh, Mnx_mesh, Mny_mesh, Pd_mesh, Mdx_mesh, Mdy_mesh


# =====================================================================
# 计算引擎 2：严谨 2D 双向扫描
# =====================================================================
@st.cache_data
def compute_2d_pm_strict(sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                         has_steel, hs, bs, tw, tf, fya, method, bending_axis='X'):
    _section = RCSection(sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                         has_steel, hs, bs, tw, tf, fya)
    if bending_axis == 'X':
        h_dim, b_dim = _section.h, _section.b
        y_coords, steel_y_coords = _section.ys_coords, _section.steel_ys
    else:
        h_dim, b_dim = _section.b, _section.h
        y_coords, steel_y_coords = _section.xs_coords, _section.steel_xs

    dy = h_dim / 100
    y_fibers = np.linspace(-h_dim / 2 + dy / 2, h_dim / 2 - dy / 2, 100)

    def calc_point(c, comp_side):
        P_c, M_c = 0.0, 0.0
        if method == 'stress_block':
            a = min(_section.beta1 * c, h_dim)
            if _section.sec_type == '矩形':
                P_c = 0.85 * _section.fc * a * b_dim
                y_pc = h_dim / 2 - a / 2 if comp_side == 'top' else -h_dim / 2 + a / 2
                M_c = P_c * y_pc
            else:
                R = _section.D / 2
                if a >= 2 * R:
                    A_c = np.pi * R ** 2;
                    y_pc = 0.0
                else:
                    dist = R - a
                    alpha = np.arccos(dist / R)
                    A_c = R ** 2 * (alpha - np.sin(alpha) * np.cos(alpha))
                    y_pc = (2 * R ** 3 * np.sin(alpha) ** 3) / (3 * A_c) if A_c > 0 else 0
                P_c = 0.85 * _section.fc * A_c
                y_pc = y_pc if comp_side == 'top' else -y_pc
                M_c = P_c * y_pc
        elif method == 'fiber':
            for y in y_fibers:
                depth = h_dim / 2 - y if comp_side == 'top' else y - (-h_dim / 2)
                strain = _section.ecu * (c - depth) / c
                if strain > 0:
                    stress = get_stress_concrete(strain, _section.fc, _section.ecu)
                    width = b_dim if _section.sec_type == '矩形' else 2 * np.sqrt(
                        max((_section.D / 2) ** 2 - y ** 2, 0))
                    force = stress * width * dy
                    P_c += force;
                    M_c += force * y

        F_s, M_s = 0.0, 0.0
        strains_s = []
        for y, As in zip(y_coords, _section.As_array):
            depth = h_dim / 2 - y if comp_side == 'top' else y - (-h_dim / 2)
            strain = _section.ecu * (c - depth) / c
            strains_s.append(strain)
            stress = get_stress_rebar(strain, _section.fy, _section.Es)
            if strain > 0:
                if method == 'stress_block' and depth < _section.beta1 * c:
                    stress -= 0.85 * _section.fc
                elif method == 'fiber':
                    stress -= get_stress_concrete(strain, _section.fc, _section.ecu)
            F_s += stress * As;
            M_s += stress * As * y

        if _section.has_steel:
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

        Pn, Mn = P_c + F_s, M_c + M_s
        et = -np.min(strains_s) if len(strains_s) > 0 else 0
        phi = 0.65 if et <= _section.ey else (
            0.90 if et >= 0.005 else 0.65 + 0.25 * (et - _section.ey) / (0.005 - _section.ey))
        Pd, Md = phi * Pn, phi * Mn
        Pd_plot = min(Pd, _section.Pd_max)
        Md_plot = Md * (_section.Pd_max / Pd) if Pd > _section.Pd_max and Pd != 0 else Md
        return Pn, Mn, Pd_plot, Md_plot

    Pn_list, Mn_list, Pd_list, Md_list = [], [], [], []
    Pt = -_section.fy * _section.Ast_total - (_section.fya * _section.A_steel if _section.has_steel else 0)
    stress_c_pure = _section.fy - 0.85 * _section.fc
    stress_st_pure = _section.fya - 0.85 * _section.fc
    M_Po = sum([stress_c_pure * As * y for y, As in zip(y_coords, _section.As_array)])
    M_Pt = sum([-_section.fy * As * y for y, As in zip(y_coords, _section.As_array)])
    if _section.has_steel:
        M_Po += sum([stress_st_pure * As * y for y, As in zip(steel_y_coords, _section.steel_As)])
        M_Pt += sum([-_section.fya * As * y for y, As in zip(steel_y_coords, _section.steel_As)])

    def add_pt(pn, mn, pd, md):
        Pn_list.append(pn);
        Mn_list.append(mn);
        Pd_list.append(pd);
        Md_list.append(md)

    add_pt(Pt, M_Pt, 0.9 * Pt, 0.9 * M_Pt)
    for c in np.logspace(np.log10(1), np.log10(h_dim * 10000), 100): add_pt(*calc_point(c, 'bottom'))
    add_pt(_section.Po, M_Po, _section.Pd_max, 0.65 * M_Po)
    for c in np.logspace(np.log10(h_dim * 10000), np.log10(1), 100): add_pt(*calc_point(c, 'top'))
    add_pt(Pt, M_Pt, 0.9 * Pt, 0.9 * M_Pt)

    return np.array(Pn_list) / 1000, np.array(Mn_list) / 1e6, np.array(Pd_list) / 1000, np.array(Md_list) / 1e6


# =====================================================================
# 六个特殊特征点提取 (用于生成名义与设计详表)
# =====================================================================
@st.cache_data
def get_6_characteristic_points(sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                                has_steel, hs, bs, tw, tf, fya, method, bending_axis='X'):
    """针对单向主轴提取六大受力特征点"""
    Pn, Mn, Pd, Md = compute_2d_pm_strict(sec_type, b, h, D, fc, fy, Es, cover, n_top, n_bot, n_side, n_circ, d_bar,
                                          has_steel, hs, bs, tw, tf, fya, method, bending_axis)

    idx_max_p = np.argmax(Pn)
    idx_min_p = np.argmin(Pn)
    idx_pure_m = np.argmin(np.abs(Pn))

    pts = [
        {"特征点": "1. 纯压点 (Pure Compression)", "名义轴力 Pn (kN)": Pn[idx_max_p],
         "名义弯矩 Mn (kN·m)": Mn[idx_max_p], "设计轴力 Pd (kN)": Pd[idx_max_p], "设计弯矩 Md (kN·m)": Md[idx_max_p]},
        {"特征点": "2. 界限破坏点 (Balanced)", "名义轴力 Pn (kN)": Pn[idx_max_p - 40],
         "名义弯矩 Mn (kN·m)": Mn[idx_max_p - 40], "设计轴力 Pd (kN)": Pd[idx_max_p - 40],
         "设计弯矩 Md (kN·m)": Md[idx_max_p - 40]},
        {"特征点": "3. 受拉控制点 (Tension Ctrl)", "名义轴力 Pn (kN)": Pn[idx_max_p - 65],
         "名义弯矩 Mn (kN·m)": Mn[idx_max_p - 65], "设计轴力 Pd (kN)": Pd[idx_max_p - 65],
         "设计弯矩 Md (kN·m)": Md[idx_max_p - 65]},
        {"特征点": "4. 纯弯点 (Pure Bending)", "名义轴力 Pn (kN)": Pn[idx_pure_m], "名义弯矩 Mn (kN·m)": Mn[idx_pure_m],
         "设计轴力 Pd (kN)": Pd[idx_pure_m], "设计弯矩 Md (kN·m)": Md[idx_pure_m]},
        {"特征点": "5. 大偏心受拉 (Small Comp)", "名义轴力 Pn (kN)": Pn[idx_pure_m - 10],
         "名义弯矩 Mn (kN·m)": Mn[idx_pure_m - 10], "设计轴力 Pd (kN)": Pd[idx_pure_m - 10],
         "设计弯矩 Md (kN·m)": Md[idx_pure_m - 10]},
        {"特征点": "6. 纯拉点 (Pure Tension)", "名义轴力 Pn (kN)": Pn[idx_min_p], "名义弯矩 Mn (kN·m)": Mn[idx_min_p],
         "设计轴力 Pd (kN)": Pd[idx_min_p], "设计弯矩 Md (kN·m)": Md[idx_min_p]}
    ]
    return pd.DataFrame(pts)


# =====================================================================
# Web 界面布局与数据集成
# =====================================================================
with st.sidebar:
    st.header("⚙️ 引擎与几何参数")
    engine_mode = st.radio("1. 核心引擎：", ("3D PMM 曲面", "2D 等效应力块法", "2D 纤维法"), index=0)

    sec_type = st.radio("2. 截面形状：", ["矩形 (Rectangular)", "圆形 (Circular)"], horizontal=True)
    col1, col2 = st.columns(2)
    if '矩形' in sec_type:
        b = col1.number_input("宽度 $b$ (mm)", value=400.0, step=50.0);
        h = col2.number_input("高度 $h$ (mm)", value=600.0, step=50.0)
        D = 0.0;
        n_circ = 0
    else:
        b = 0.0;
        h = 0.0
        D = col1.number_input("直径 $D$ (mm)", value=600.0, step=50.0)
        n_circ = col2.number_input("环绕钢筋根数", value=12, step=1)

    fc = col1.number_input("砼强度 $f'_c$ (MPa)", value=30.0, step=5.0);
    fy = col2.number_input("钢筋强度 $f_y$ (MPa)", value=400.0, step=50.0)
    st.subheader("3. 钢筋排布参数")
    cover = st.number_input("保护层厚度 (mm)", value=50, step=5);
    d_bar = st.number_input("钢筋直径 (mm)", value=20.0, step=2.0)

    if '矩形' in sec_type:
        col3, col4 = st.columns(2)
        n_top = col3.number_input("顶排钢筋", value=3);
        n_bot = col4.number_input("底排钢筋", value=3)
        n_side = st.number_input("单侧中部钢筋", value=2)
    else:
        n_top = n_bot = n_side = 0

    st.subheader("4. 内部型钢配置 (SRC)")
    has_steel = st.toggle("内置 H 型钢")
    if has_steel:
        col_s1, col_s2 = st.columns(2)
        hs = col_s1.number_input("型钢高 $h_s$", value=300.0);
        bs = col_s2.number_input("翼缘宽 $b_s$", value=200.0)
        tw = col_s1.number_input("腹板厚 $t_w$", value=10.0);
        tf = col_s2.number_input("翼缘厚 $t_f$", value=14.0)
        fya = st.number_input("型钢屈服 $f_{ya}$", value=355.0)
    else:
        hs = bs = tw = tf = fya = 0.0

    #st.divider()
    #st.subheader("4. 内部型钢配置 (SRC)")
    st.header("📥 设计内力 (Demands)")
    st.markdown("**单工况手动输入**")
    cd1, cd2 = st.columns(2)
    Pu_man = cd1.number_input("轴力 $P_u$ (kN)", value=1000.0, step=100.0)
    Mux_man = cd2.number_input("弯矩 $M_{ux}$ (kN·m)", value=150.0, step=10.0)
    Muy_man = cd1.number_input("弯矩 $M_{uy}$ (kN·m)", value=100.0, step=10.0)
    Vx_man = cd2.number_input("剪力 $V_x$ (kN)", value=0.0)
    Vy_man = cd1.number_input("剪力 $V_y$ (kN)", value=0.0)
    Tu_man = cd2.number_input("扭矩 $T_u$ (kN·m)", value=0.0)

    custom_alpha = 45;
    target_N = 1000.0  # 初始化默认值
    if engine_mode == "3D PMM 曲面":  # 只有在 3D 模式下才激活这几个控件
        custom_alpha = st.slider("自定义 2D 切片角度 $\\alpha$ (度)", 0, 180, 45, 5)
        target_N = st.number_input("指定设计轴力 $N$ (kN)", value=1000.0, step=100.0)

    uploaded_file = st.file_uploader("或导入 ETABS 内力 CSV 文件", type=['csv'])
    run_btn = st.button("🚀 开始分析与验算", use_container_width=True, type="primary")

if run_btn:
    with st.spinner("正在进行核心计算与多维空间映射..."):
        sec = RCSection('矩形' if '矩形' in sec_type else '圆形', b, h, D, fc, fy, 200000.0, cover, int(n_top),
                        int(n_bot), int(n_side), int(n_circ), d_bar, has_steel, hs, bs, tw, tf, fya)

        demands_list = [{"LoadCombo": "Manual", "P": Pu_man, "Mx": Mux_man, "My": Muy_man, "Vx": Vx_man, "Vy": Vy_man,
                         "Tu": Tu_man}]
        if uploaded_file is not None:
            try:
                df_up = pd.read_csv(uploaded_file)
                col_map = {}
                for c in df_up.columns:
                    c_up = c.upper().strip()
                    if 'P' in c_up and 'P' not in col_map.values():
                        col_map[c] = 'P'
                    elif ('M3' in c_up or 'MX' in c_up) and 'Mx' not in col_map.values():
                        col_map[c] = 'Mx'
                    elif ('M2' in c_up or 'MY' in c_up) and 'My' not in col_map.values():
                        col_map[c] = 'My'
                    elif ('V2' in c_up or 'VX' in c_up) and 'Vx' not in col_map.values():
                        col_map[c] = 'Vx'
                    elif ('V3' in c_up or 'VY' in c_up) and 'Vy' not in col_map.values():
                        col_map[c] = 'Vy'
                    elif ('T' in c_up or 'TU' in c_up) and 'Tu' not in col_map.values():
                        col_map[c] = 'Tu'
                    elif 'COMBO' in c_up or 'CASE' in c_up or 'LOAD' in c_up:
                        col_map[c] = 'LoadCombo'

                df_mapped = df_up.rename(columns=col_map)
                for needed in ["LoadCombo", "P", "Mx", "My", "Vx", "Vy", "Tu"]:
                    if needed not in df_mapped.columns:
                        df_mapped[needed] = f"Combo_{len(df_mapped)}" if needed == "LoadCombo" else 0.0

                records = df_mapped[["LoadCombo", "P", "Mx", "My", "Vx", "Vy", "Tu"]].to_dict('records')
                demands_list.extend(records)
            except Exception as e:
                st.error(f"文件解析失败: {e}")

        df_demands = pd.DataFrame(demands_list)
        max_N_val = df_demands['P'].max()
        min_N_val = df_demands['P'].min()

        # ================== 左侧固定展示区：截面预览 ==================
        col_res1, col_res2 = st.columns([1, 4])
        with col_res1:
            fig_sec, ax_sec = plt.subplots(figsize=(4, 5))
            if sec.sec_type == '矩形':
                ax_sec.plot([-sec.b / 2, sec.b / 2, sec.b / 2, -sec.b / 2, -sec.b / 2],
                            [sec.h / 2, sec.h / 2, -sec.h / 2, -sec.h / 2, sec.h / 2], 'k-', linewidth=2.5)
            else:
                theta = np.linspace(0, 2 * np.pi, 100)
                ax_sec.plot(sec.D / 2 * np.cos(theta), sec.D / 2 * np.sin(theta), 'k-', linewidth=2.5)

            ax_sec.scatter(sec.xs_coords, sec.ys_coords, s=90, c='red', zorder=5)

            if has_steel:
                h_poly = [-bs / 2, bs / 2, bs / 2, tw / 2, tw / 2, bs / 2, bs / 2, -bs / 2, -bs / 2, -tw / 2, -tw / 2,
                          -bs / 2, -bs / 2]
                v_poly = [hs / 2, hs / 2, hs / 2 - tf, hs / 2 - tf, -hs / 2 + tf, -hs / 2 + tf, -hs / 2, -hs / 2,
                          -hs / 2 + tf, -hs / 2 + tf, hs / 2 - tf, hs / 2 - tf, hs / 2]
                ax_sec.plot(h_poly, v_poly, 'b-', linewidth=2)
                ax_sec.fill(h_poly, v_poly, 'blue', alpha=0.25)

            ax_sec.set_aspect('equal');
            ax_sec.axis('off')
            ax_sec.set_title(
                f"Section Preview\nAs={sec.Ast_total:.0f} mm²" + (f", Aa={sec.A_steel:.0f} mm²" if has_steel else ""),
                fontsize=12, pad=15)
            st.pyplot(fig_sec)

            st.info(
                f"**总含钢率**: {((sec.Ast_total + sec.A_steel) / sec.Ag) * 100:.2f}%\n\n**理论最大轴压 $P_o$**: {sec.Po / 1000:.1f} kN\n\n**设计最大轴压 $N_{{u,max}}$**: {sec.Pd_max / 1000:.1f} kN")

        # ================== 右侧动态页签区 ==================
        with col_res2:
            if engine_mode == "3D PMM 曲面":
                Pn_mesh, Mnx_mesh, Mny_mesh, Pd_mesh, Mdx_mesh, Mdy_mesh = compute_3d_data(
                    sec.sec_type, b, h, D, fc, fy, 200000.0, cover, int(n_top), int(n_bot), int(n_side), int(n_circ),
                    d_bar, has_steel, hs, bs, tw, tf, fya)

                alpha_angles = np.linspace(0, 2 * np.pi, 36)
                df_demands[['D/C (3D Radial)', 'Req_Len', 'Cap_Len']] = df_demands.apply(
                    lambda row: pd.Series(
                        calc_3d_radial_dc(row['P'], row['Mx'], row['My'], alpha_angles, Pd_mesh, Mdx_mesh, Mdy_mesh)),
                    axis=1)

                max_dc_3d = df_demands['D/C (3D Radial)'].max()
                critical_row = df_demands.loc[df_demands['D/C (3D Radial)'].idxmax()]

                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["📊 3D 曲面与主轴切片", "🎯 极值轴力 Mx-My 切片验算", "📋 Bresler 验算报告单", "🗄️ PMM 曲面数据输出",
                     "📄 计算书输出 (Report)"])

                with tab1:
                    col_3d, col_dc = st.columns([3, 1])
                    with col_3d:
                        fig_3d = plt.figure(figsize=(10, 8))
                        ax_3d = fig_3d.add_subplot(111, projection='3d')
                        ax_3d.plot_surface(Mnx_mesh, Mny_mesh, Pn_mesh, color='cyan', edgecolor='c', linewidth=0.1,
                                           alpha=0.15)
                        ax_3d.plot_surface(Mdx_mesh, Mdy_mesh, Pd_mesh, color='red', edgecolor='darkred', linewidth=0.3,
                                           alpha=0.7)

                        ax_3d.scatter(df_demands['Mx'], df_demands['My'], df_demands['P'], color='black', s=40,
                                      marker='x', label='Demands (Pu, Mux, Muy)', zorder=10)

                        ax_3d.set_title('3D P-Mx-My Interaction Surface', fontsize=15)
                        ax_3d.set_xlabel('Mx (kN·m)');
                        ax_3d.set_ylabel('My (kN·m)');
                        ax_3d.set_zlabel('P (kN)')
                        nom_patch = mpatches.Patch(color='cyan', alpha=0.3, label='Theoretical Nominal ($P_o$)')
                        des_patch = mpatches.Patch(color='red', alpha=0.7, label='Design ($\phi P_n$, Capped)')
                        ax_3d.legend(handles=[nom_patch, des_patch], loc='upper left')
                        ax_3d.view_init(elev=20, azim=40)
                        st.pyplot(fig_3d)

                    with col_dc:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="background-color:#f0f6ff; padding:20px; border-radius:10px; margin-bottom:15px;">
                            <p style="color:#004085; font-size:16px; font-weight:bold; margin-bottom:5px;">最高 3D D/C Ratio:</p>
                            <h1 style="color:#0056b3; margin-top:0px;">{max_dc_3d:.3f}</h1>
                        </div>
                        """, unsafe_allow_html=True)

                        if max_dc_3d <= 1.0:
                            st.markdown(
                                """<div style="background-color:#e6f4ea; padding:15px; border-radius:10px; color:#1e7e34; font-weight:bold;">✅ 截面安全</div>""",
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                """<div style="background-color:#fce8e6; padding:15px; border-radius:10px; color:#c5221f; font-weight:bold;">❌ 截面不安全</div>""",
                                unsafe_allow_html=True)

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown("**最不利工况 Radial 数据：**")
                        st.write(f"工况名: {critical_row['LoadCombo']}")
                        st.write(f"分子 (Demand Vector): **{critical_row['Req_Len']:.1f}**")
                        st.write(f"分母 (Capacity Vector): **{critical_row['Cap_Len']:.1f}**")

                    st.markdown("<br>", unsafe_allow_html=True)
                    col_s1, col_s2, col_s3 = st.columns(3)


                    def extract_2d_slice(alpha_deg):
                        idx_pos = int((alpha_deg / 360.0) * 36);
                        idx_neg = int(((alpha_deg + 180) / 360.0) * 36) % 36
                        Mres_n_nom = np.sqrt(Mnx_mesh[:, idx_neg] ** 2 + Mny_mesh[:, idx_neg] ** 2) * -1
                        Mres_p_nom = np.sqrt(Mnx_mesh[:, idx_pos] ** 2 + Mny_mesh[:, idx_pos] ** 2)
                        Mres_n_des = np.sqrt(Mdx_mesh[:, idx_neg] ** 2 + Mdy_mesh[:, idx_neg] ** 2) * -1
                        Mres_p_des = np.sqrt(Mdx_mesh[:, idx_pos] ** 2 + Mdy_mesh[:, idx_pos] ** 2)
                        return (np.concatenate([Mres_n_nom[::-1], Mres_p_nom]),
                                np.concatenate([Pn_mesh[:, idx_neg][::-1], Pn_mesh[:, idx_pos]]),
                                np.concatenate([Mres_n_des[::-1], Mres_p_des]),
                                np.concatenate([Pd_mesh[:, idx_neg][::-1], Pd_mesh[:, idx_pos]]))


                    for col, ang, title, m_col in zip([col_s1, col_s2, col_s3], [0, 90, custom_alpha],
                                                      ['Slice @ 0° (P-Mx)', 'Slice @ 90° (P-My)',
                                                       f'Slice @ {custom_alpha}° (P-M$\\alpha$)'],
                                                      ['Mx', 'My', None]):
                        with col:
                            fig_sl = plt.figure(figsize=(6, 5))
                            ax_sl = fig_sl.add_subplot(111)
                            M_nom, P_nom, M_des, P_des = extract_2d_slice(ang)
                            ax_sl.plot(M_nom, P_nom, 'c-.', linewidth=2, label='Nominal')
                            ax_sl.plot(M_des, P_des, 'r-', linewidth=2, label='Design')

                            if m_col == 'Mx':
                                ax_sl.scatter(df_demands['Mx'], df_demands['P'], color='black', s=25, marker='x',
                                              zorder=5)
                            elif m_col == 'My':
                                ax_sl.scatter(df_demands['My'], df_demands['P'], color='black', s=25, marker='x',
                                              zorder=5)
                            else:
                                a_rad = np.radians(custom_alpha)
                                m_proj = df_demands['Mx'] * np.cos(a_rad) + df_demands['My'] * np.sin(a_rad)
                                ax_sl.scatter(m_proj, df_demands['P'], color='black', s=25, marker='x', zorder=5)

                            ax_sl.axhline(0, color='k', linewidth=0.8);
                            ax_sl.axvline(0, color='k', linewidth=0.8)
                            ax_sl.axhline(sec.Po / 1000, color='cyan', linestyle=':', alpha=0.6);
                            ax_sl.axhline(sec.Pd_max / 1000, color='red', linestyle=':', alpha=0.6)
                            ax_sl.set_title(title, fontsize=12);
                            ax_sl.set_xlabel('Moment (kN·m)');
                            ax_sl.set_ylabel('Axial (kN)')
                            ax_sl.grid(True, linestyle=':', alpha=0.7);
                            ax_sl.legend(fontsize=9)
                            st.pyplot(fig_sl)

                with tab2:
                    st.subheader("🎯 恒定轴力 Mx-My 等高线切片 (Constant Load Contour)")
                    st.write("利用水平截面截取 3D 曲面，得到在指定轴向力 $P_u$ 下的双向抗弯闭合曲线。")
                    col_p1, col_p2, col_p3 = st.columns(3)
                    eval_N_list = [("User Input N", target_N, col_p1), ("Max N in Demands", max_N_val, col_p2),
                                   ("Min N in Demands", min_N_val, col_p3)]
                    max_N_des = sec.Pd_max / 1000;
                    min_N_ten = (-sec.fy * sec.Ast_total - (sec.fya * sec.A_steel if has_steel else 0)) * 0.9 / 1000

                    for title, plot_N, col in eval_N_list:
                        safe_N = plot_N
                        if plot_N >= max_N_des: safe_N = max_N_des - 1e-3
                        if plot_N <= min_N_ten: safe_N = min_N_ten + 1e-3

                        mx_nom, my_nom = extract_mx_my_contour(safe_N, Pn_mesh, Mnx_mesh, Mny_mesh)
                        mx_des, my_des = extract_mx_my_contour(safe_N, Pd_mesh, Mdx_mesh, Mdy_mesh)

                        fig_c, ax_c = plt.subplots(figsize=(6, 6))
                        ax_c.plot(mx_nom, my_nom, 'c-.', label=f'Nominal ($P_n = {plot_N:.1f}$ kN)')
                        ax_c.plot(mx_des, my_des, 'r-', linewidth=2.5, label=f'Design ($P_d = {plot_N:.1f}$ kN)')

                        nearby_demands = df_demands[np.abs(df_demands['P'] - plot_N) < 50.0]
                        if not nearby_demands.empty:
                            ax_c.scatter(nearby_demands['Mx'], nearby_demands['My'], color='black', marker='x', s=80,
                                         label='Demands')
                            max_contour_dc = max([calc_contour_dc(row['Mx'], row['My'], mx_des, my_des) for _, row in
                                                  nearby_demands.iterrows()])
                            ax_c.set_title(f"{title}: {plot_N:.1f} kN\nMax D/C = {max_contour_dc:.2f}", fontsize=12,
                                           color='red' if max_contour_dc > 1 else 'green')
                        else:
                            ax_c.set_title(f"{title}: {plot_N:.1f} kN\n(No points nearby)", fontsize=12)

                        ax_c.axhline(0, color='k', linewidth=0.5);
                        ax_c.axvline(0, color='k', linewidth=0.5)
                        ax_c.grid(True, linestyle=':', alpha=0.5);
                        ax_c.set_aspect('equal', adjustable='datalim');
                        ax_c.legend(loc='upper right')
                        with col:
                            st.pyplot(fig_c)

                with tab3:
                    st.subheader("📋 Bresler 倒数公式与综合设计报表")
                    st.markdown("基于 ACI 318 的 **Bresler Reciprocal Method** 经验校核。")
                    st.latex(r"\frac{1}{P_{ni}} = \frac{1}{P_{nx}} + \frac{1}{P_{ny}} - \frac{1}{P_{o}}")

                    _, _, Pd_x, Mdx_x = compute_2d_pm_strict(sec.sec_type, b, h, D, fc, fy, 200000.0, cover, int(n_top),
                                                             int(n_bot), int(n_side), int(n_circ), d_bar, has_steel, hs,
                                                             bs, tw, tf, fya, 'fiber', 'X')
                    _, _, Pd_y, Mdy_y = compute_2d_pm_strict(sec.sec_type, b, h, D, fc, fy, 200000.0, cover, int(n_top),
                                                             int(n_bot), int(n_side), int(n_circ), d_bar, has_steel, hs,
                                                             bs, tw, tf, fya, 'fiber', 'Y')
                    df_demands['D/C (Bresler)'] = df_demands.apply(
                        lambda row: calc_bresler_dc(row['P'], row['Mx'], row['My'], sec.Pd_max / 1000, Pd_x, Mdx_x,
                                                    Pd_y, Mdy_y), axis=1)

                    st.dataframe(df_demands.style.format({
                        'P': '{:.1f}', 'Mx': '{:.1f}', 'My': '{:.1f}', 'Vx': '{:.1f}', 'Vy': '{:.1f}', 'Tu': '{:.1f}',
                        'D/C (3D Radial)': '{:.3f}', 'Req_Len': '{:.1f}', 'Cap_Len': '{:.1f}', 'D/C (Bresler)': '{:.3f}'
                    }).background_gradient(subset=['D/C (3D Radial)'], cmap='Reds', vmin=0.5, vmax=1.2),
                                 use_container_width=True)

                with tab4:
                    st.subheader("🗄️ PMM 曲面空间数据输出")
                    st.markdown("以下为系统积分生成的完整 3D 空间曲面包络面离散节点数据。")

                    surface_data = []
                    for j, alpha in enumerate(alpha_angles):
                        alpha_deg = np.degrees(alpha)
                        for i in range(Pn_mesh.shape[0]):
                            surface_data.append({
                                "Alpha 角 (°)": alpha_deg,
                                "名义轴力 Pn (kN)": Pn_mesh[i, j],
                                "名义弯矩 Mnx (kN·m)": Mnx_mesh[i, j],
                                "名义弯矩 Mny (kN·m)": Mny_mesh[i, j],
                                "设计轴力 Pd (kN)": Pd_mesh[i, j],
                                "设计弯矩 Mdx (kN·m)": Mdx_mesh[i, j],
                                "设计弯矩 Mdy (kN·m)": Mdy_mesh[i, j]
                            })
                    df_surface = pd.DataFrame(surface_data)
                    st.dataframe(df_surface.style.format(precision=1), use_container_width=True)

                with tab5:
                    st.subheader("📄 构件计算书输出 (Report)")
                    st.info("计算书生成模块正在建设中。后续将根据具体的工程文档格式排版并导出 PDF/Word 详单。")

            else:
                method = 'stress_block' if '等效应力块' in engine_mode else 'fiber'
                Pn_x, Mn_x, Pd_x, Md_x = compute_2d_pm_strict(
                  sec.sec_type, b, h, D, fc, fy, 200000.0, cover, int(n_top), int(n_bot), int(n_side), int(n_circ),
                  d_bar, has_steel, hs, bs, tw, tf, fya, method, bending_axis='X')
                Pn_y, Mn_y, Pd_y, Md_y = compute_2d_pm_strict(
                  sec.sec_type, b, h, D, fc, fy, 200000.0, cover, int(n_top), int(n_bot), int(n_side), int(n_circ),
                  d_bar, has_steel, hs, bs, tw, tf, fya, method, bending_axis='Y')

                fig_2d, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(14, 6))

                ax_x.plot(Mn_x, Pn_x, 'c-.', label='Nominal')
                ax_x.plot(Md_x, Pd_x, 'r-', linewidth=2.5, label='Design')
                ax_x.scatter(df_demands['Mx'], df_demands['P'], color='black', s=40, marker='x', label='Demands')
                ax_x.set_title(f'P-Mx (0°) Bending\nMethod: {method}', fontsize=12)

                ax_y.plot(Mn_y, Pn_y, 'c-.', label='Nominal')
                ax_y.plot(Md_y, Pd_y, 'r-', linewidth=2.5, label='Design')
                ax_y.scatter(df_demands['My'], df_demands['P'], color='black', s=40, marker='x', label='Demands')
                ax_y.set_title(f'P-My (90°) Bending\nMethod: {method}', fontsize=12)

                for ax in (ax_x, ax_y):
                 ax.axhline(0, color='k', linewidth=0.8);
                 ax.axvline(0, color='k', linewidth=0.8)
                 ax.set_xlabel('Moment (kN·m)');
                 ax.set_ylabel('Axial Load (kN)')
                 ax.grid(True, linestyle=':', alpha=0.7);
                 ax.legend()

                 st.pyplot(fig_2d)

else:
    st.info("👈 请在侧边栏配置各项参数及输入设计内力，然后点击 **开始分析与验算**。")