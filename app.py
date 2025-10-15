# app.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from scipy.special import erfc

# =============== Streamlit 基本设置 ===============
st.set_page_config(page_title="fib Chloride Ingress – Reliability", layout="wide")

# =============== 工具函数与模型内核（保持你原公式与逻辑） ===============
def beta_from_pf(Pf: np.ndarray) -> np.ndarray:
    return -norm.ppf(Pf)

def lognorm_from_mu_sd(rng, n, mu, sd):
    """Sample lognormal given arithmetic mean mu and std sd."""
    sigma2 = math.log(1 + (sd**2) / (mu**2))
    mu_log = math.log(mu) - 0.5 * sigma2
    sigma = math.sqrt(sigma2)
    return rng.lognormal(mu_log, sigma, n)

def beta01_shapes_from_mean_sd(mu, sd):
    """Convert mean/sd on [0,1] to Beta(a,b) shapes with clamps for stability."""
    mu = max(min(mu, 1 - 1e-9), 1e-9)
    var = max(sd**2, 1e-12)
    t = mu * (1 - mu) / var - 1
    a = max(mu * t, 1e-6)
    b = max((1 - mu) * t, 1e-6)
    return a, b

def beta_interval_from_mean_sd(rng, n, mu, sd, L, U):
    """Sample Beta on [L, U] given arithmetic mean mu and sd on that interval."""
    if U <= L:
        raise ValueError("Upper bound must be greater than lower bound.")
    mu = max(min(mu, U - 1e-12), L + 1e-12)
    sd = max(sd, 1e-12)
    mu01 = (mu - L) / (U - L)
    sd01 = sd / (U - L)
    a, b = beta01_shapes_from_mean_sd(mu01, sd01)
    return L + (U - L) * rng.beta(a, b, n)

def run_fib_chloride(params, N=100000, seed=42, t_start=0.9, t_end=50.0, t_points=200):
    """Simulate fib chloride ingress reliability over time. Returns DataFrame with t, Pf, beta."""
    rng = np.random.default_rng(seed)
    t_years = np.linspace(float(t_start), float(t_end), int(t_points))

    # Unpack
    mu_Cs, sd_Cs       = params["Cs_mu"], params["Cs_sd"]
    mu_alpha, sd_alpha = params["alpha_mu"], params["alpha_sd"]
    alpha_L, alpha_U   = params["alpha_L"], params["alpha_U"]
    mu_D0, sd_D0       = params["D0_mu"], params["D0_sd"]
    mu_cover, sd_cover = params["cover_mu"], params["cover_sd"]
    mu_Ccrit, sd_Ccrit = params["Ccrit_mu"], params["Ccrit_sd"]
    Ccrit_L, Ccrit_U   = params["Ccrit_L"], params["Ccrit_U"]
    mu_be, sd_be       = params["be_mu"], params["be_sd"]
    mu_Treal, sd_Treal = params["Treal_mu"], params["Treal_sd"]
    Tref, t0_year, C0  = params["Tref"], params["t0"], params["C0"]

    # Δx modes
    dx_mode = params["dx_mode"]
    if dx_mode == "zero":
        dx_mm = np.zeros(N)
    elif dx_mode in ("beta_submerged", "beta_tidal"):
        dx_mm = beta_interval_from_mean_sd(
            rng, N,
            mu=params["dx_mu"], sd=params["dx_sd"],
            L=params["dx_L"], U=params["dx_U"]
        )
    else:
        raise ValueError("dx_mode must be one of: zero, beta_submerged, beta_tidal")

    # Parameters distribution
    Cs = lognorm_from_mu_sd(rng, N, mu_Cs, sd_Cs)
    alpha = beta_interval_from_mean_sd(rng, N, mu_alpha, sd_alpha, alpha_L, alpha_U)
    Ccrit = beta_interval_from_mean_sd(rng, N, mu_Ccrit, sd_Ccrit, Ccrit_L, Ccrit_U)

    D0 = np.maximum(rng.normal(mu_D0, sd_D0, N), 1e-3) * 1e-12
    cover_m = np.maximum(rng.normal(mu_cover, sd_cover, N), 1.0) / 1000.0
    be = np.maximum(rng.normal(mu_be, sd_be, N), 1.0)
    Treal = np.maximum(rng.normal(mu_Treal, sd_Treal, N), 250.0)

    # Formula
    t0_sec = t0_year * 365.25 * 24 * 3600.0
    temp_fac = np.exp(be * (1.0 / Tref - 1.0 / Treal))

    Pf = []
    for t in t_years:
        t_sec = t * 365.25 * 24 * 3600.0
        Dapp = temp_fac * D0 * (t0_sec / t_sec) ** alpha
        arg = (cover_m - dx_mm / 1000.0) / (2.0 * np.sqrt(Dapp * t_sec))
        C_at = C0 + (Cs - C0) * erfc(arg)
        Pf.append(np.mean(C_at >= Ccrit))

    Pf = np.clip(np.array(Pf), 1e-12, 1 - 1e-12)
    beta = beta_from_pf(Pf)
    return pd.DataFrame({"t_years": t_years, "Pf": Pf, "beta": beta})

def plot_beta(df_window, t_end, axes_cfg=None, show_pf=True, beta_target=None, show_beta_target=False):
    x_abs = df_window["t_years"].to_numpy()
    y_beta = df_window["beta"].to_numpy()
    y_pf   = df_window["Pf"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x_abs, y_beta, lw=2, label="β(t)")
    ax1.set_xlabel("Time (yr)")
    ax1.set_ylabel("Reliability index β(-)")
    ax1.grid(True)

    ax2 = None
    if show_pf:
        ax2 = ax1.twinx()
        ax2.plot(x_abs, y_pf, linestyle="--", lw=1.6, label="Pf(t)")
        ax2.set_ylabel("Failure probability Pf(t)")

    if axes_cfg is None:
        axes_cfg = {}

    ax1.set_xlim(0, float(t_end))
    x_tick = axes_cfg.get("x_tick", None)
    if x_tick is not None and x_tick > 0:
        ax1.set_xticks(np.arange(0, float(t_end) + 1e-12, x_tick))

    y1_min = axes_cfg.get("y1_min", None)
    y1_max = axes_cfg.get("y1_max", None)
    y1_tick = axes_cfg.get("y1_tick", None)
    if y1_min is not None and y1_max is not None and y1_max > y1_min:
        ax1.set_ylim(y1_min, y1_max)
    if y1_tick is not None and y1_tick > 0:
        ymin, ymax = ax1.get_ylim()
        ax1.set_yticks(np.arange(ymin, ymax + 1e-12, y1_tick))

    if show_pf and ax2 is not None:
        y2_min = axes_cfg.get("y2_min", None)
        y2_max = axes_cfg.get("y2_max", None)
        y2_tick = axes_cfg.get("y2_tick", None)
        if y2_min is not None and y2_max is not None and y2_max > y2_min:
            ax2.set_ylim(y2_min, y2_max)
        if y2_tick is not None and y2_tick > 0:
            ymin2, ymax2 = ax2.get_ylim()
            ax2.set_yticks(np.arange(ymin2, ymax2 + 1e-12, y2_tick))

    ax1.axvline(float(t_end), linestyle=":", lw=1.5)

    # Target beta line + annotation
    if show_beta_target and beta_target is not None:
        ax1.axhline(beta_target, linestyle='--', lw=1.5, label=f'Target β = {beta_target}')
        crossing_year = None
        for i in range(len(y_beta) - 1):
            if (y_beta[i] - beta_target) * (y_beta[i+1] - beta_target) <= 0:
                t1, t2 = x_abs[i], x_abs[i+1]
                b1, b2 = y_beta[i], y_beta[i+1]
                if (b2 - b1) != 0:
                    crossing_year = t1 + (beta_target - b1) * (t2 - t1) / (b2 - b1)
                break

        textstr = f'Target β = {beta_target:.2f}'
        if crossing_year is not None:
            textstr += f'\nYear reached: {crossing_year:.2f} yr'
            ax1.axvline(crossing_year, linestyle=':', lw=1.0, alpha=0.7)
        else:
            textstr += '\nNot reached in time range'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                 va='top', ha='right', bbox=props)
        ax1.legend(loc='upper left')

    fig.tight_layout()
    return fig

# =============== 轻量 “? 帮助” 组件 ===============
def help_badge(title_key: str, default_text: str = ""):
    """
    右侧显示一个 “?” 小按钮，点击/悬停（取决于Streamlit版本）展开内容区。
    里边允许输入文字描述和上传一张图片做说明。
    """
    # 尝试使用新版本的 popover；不支持则回退到 expander
    use_popover = hasattr(st, "popover")
    if use_popover:
        with st.popover("?", use_container_width=False):
            st.caption(f"Help – {title_key}")
            st.session_state.setdefault(f"help_text_{title_key}", default_text)
            st.session_state[f"help_text_{title_key}"] = st.text_area(
                "说明文字（可选）", value=st.session_state[f"help_text_{title_key}"], height=120, label_visibility="collapsed"
            )
            img = st.file_uploader("上传一张图片（可选）", type=["png", "jpg", "jpeg"], key=f"help_img_{title_key}")
            if img:
                st.image(img, use_container_width=True)
    else:
        with st.expander("？"):
            st.caption(f"Help – {title_key}")
            st.session_state.setdefault(f"help_text_{title_key}", default_text)
            st.session_state[f"help_text_{title_key}"] = st.text_area(
                "说明文字（可选）", value=st.session_state[f"help_text_{title_key}"], height=120, label_visibility="collapsed"
            )
            img = st.file_uploader("上传一张图片（可选）", type=["png", "jpg", "jpeg"], key=f"help_img_{title_key}")
            if img:
                st.image(img, use_container_width=True)

# =============== 温度双向同步（°C ↔ K）的小工具 ===============
def sync_ck_pair(label_left: str, label_right: str, key_c: str, key_k: str, default_c=None, default_k=None):
    """
    在同一行放两个输入框：左 °C，右 K，双向同步。
    - 如果用户改 °C：K = C + 273.15
    - 如果用户改 K：C = K - 273.15
    """
    c1, c2 = st.columns(2)
    # 初始化
    if key_c not in st.session_state and default_c is not None:
        st.session_state[key_c] = float(default_c)
    if key_k not in st.session_state and default_k is not None:
        st.session_state[key_k] = float(default_k)

    def _on_c_change():
        try:
            c = st.session_state[key_c]
            if c is not None:
                st.session_state[key_k] = float(c) + 273.15
        except Exception:
            pass

    def _on_k_change():
        try:
            k = st.session_state[key_k]
            if k is not None:
                st.session_state[key_c] = float(k) - 273.15
        except Exception:
            pass

    with c1:
        st.number_input(label_left, key=key_c, value=st.session_state.get(key_c, default_c), on_change=_on_c_change, help="输入或调整摄氏温度（°C）")
    with c2:
        st.number_input(label_right, key=key_k, value=st.session_state.get(key_k, default_k), on_change=_on_k_change, help="输入或调整开尔文温度（K）")

# =============== 页面主体布局 ===============
st.title("fib chloride ingress – reliability index vs time")

# 顶层左右两列：左（主要参数），右（Δx + Temperature + 运行/显示）
left_col, right_col = st.columns([1.1, 1.0], vertical_alignment="top")

# ---------------- 左列：主要随机参数 + 目标与坐标轴 ----------------
with left_col:
    # === Ageing Exponent (α) ===
    header_cols = st.columns([0.85, 0.15])
    with header_cols[0]:
        st.subheader("Ageing Exponent (α)")
    with header_cols[1]:
        help_badge("Ageing Exponent (α)", "在此放置关于 α 选择依据的说明或图示。")

    alpha_presets = {
        "Please select": None,
        "Portland Cement (PCC)  μ=0.30, σ=0.12": (0.30, 0.12, 0.0, 1.0),
        "PCC + ≥20% Fly Ash    μ=0.60, σ=0.15": (0.60, 0.15, 0.0, 1.0),
        "PCC + BFS              μ=0.45, σ=0.20": (0.45, 0.20, 0.0, 1.0),
        "All types (atmos.)     μ=0.65, σ=0.12": (0.65, 0.12, 0.0, 1.0),
        "Custom – enter values": None,
    }
    alpha_choice = st.selectbox("α Preset", list(alpha_presets.keys()), index=0)
    alpha_cols = st.columns(4)
    if alpha_presets.get(alpha_choice):
        mu_def, sd_def, L_def, U_def = alpha_presets[alpha_choice]
        alpha_mu = alpha_cols[0].number_input("α μ", value=float(mu_def))
        alpha_sd = alpha_cols[1].number_input("α σ", value=float(sd_def))
        alpha_L  = alpha_cols[2].number_input("α lower L", value=float(L_def))
        alpha_U  = alpha_cols[3].number_input("α upper U", value=float(U_def))
        disable_alpha = True
    else:
        # 自定义
        alpha_mu = alpha_cols[0].number_input("α μ", value=0.50)
        alpha_sd = alpha_cols[1].number_input("α σ", value=0.15)
        alpha_L  = alpha_cols[2].number_input("α lower L", value=0.0)
        alpha_U  = alpha_cols[3].number_input("α upper U", value=1.0)
        disable_alpha = False

    st.divider()

    # === Reference Age (t0) ===
    header_cols = st.columns([0.85, 0.15])
    with header_cols[0]:
        st.subheader("Reference Age (t0)")
    with header_cols[1]:
        help_badge("Reference Age (t0)", "将 DRCM0 的参考龄期设为常见龄期之一，或自定义。")

    t0_map = {
        "Please select": None,
        "0.0767 – 28 days": 0.0767,
        "0.1533 – 56 days": 0.1533,
        "0.2464 – 90 days": 0.2464,
    }
    t0_choice = st.selectbox("t0 Preset", list(t0_map.keys()), index=0)
    if t0_map.get(t0_choice) is not None:
        t0_year = st.number_input("t0 (year)", value=float(t0_map[t0_choice]), disabled=True)
    else:
        t0_year = st.number_input("t0 (year)", value=0.0767)

    st.divider()

    # === Editable Parameters（Cs, C0, D0, cover, Ccrit, b_e） ===
    st.subheader("Material / Exposure Parameters")

    # Chloride surface & initial
    c1, c2, c3 = st.columns(3)
    with c1:
        C0 = st.number_input("Initial chloride C0 (wt-%/binder)", value=0.0)
    with c2:
        Cs_mu = st.number_input("Surface chloride μ (wt-%/binder)", value=1.8)
    with c3:
        Cs_sd = st.number_input("Surface chloride σ", value=0.3)

    # D0 with auto-sd = 0.2*mu
    c1, c2 = st.columns(2)
    with c1:
        D0_mu = st.number_input("DRCM0 μ (×1e-12 m²/s)", value=10.0)
    with c2:
        D0_sd = st.number_input("DRCM0 σ (=0.2×μ)", value=max(0.2*D0_mu, 0.0), disabled=True)

    # cover
    c1, c2 = st.columns(2)
    with c1:
        cover_mu = st.number_input("Cover μ (mm)", value=50.0)
    with c2:
        cover_sd = st.number_input("Cover σ (mm)", value=7.0)

    # Ccrit beta interval
    st.markdown("**Critical chloride content Ccrit (Beta on [L, U])**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        Ccrit_mu = st.number_input("Ccrit μ (wt-%/binder)", value=0.60)
    with c2:
        Ccrit_sd = st.number_input("Ccrit σ", value=0.15)
    with c3:
        Ccrit_L  = st.number_input("Ccrit L", value=0.20)
    with c4:
        Ccrit_U  = st.number_input("Ccrit U", value=2.00)

    # temperature coefficient b_e
    c1, c2 = st.columns(2)
    with c1:
        be_mu = st.number_input("Temperature coeff (b_e) μ", value=4800.0)
    with c2:
        be_sd = st.number_input("Temperature coeff (b_e) σ", value=700.0)

    st.divider()

    # === Target Reliability Index ===
    header_cols = st.columns([0.85, 0.15])
    with header_cols[0]:
        st.subheader("Target Reliability Index")
    with header_cols[1]:
        help_badge("Target Reliability Index", "设定目标 β，用于图中参考线与触达年份标注。")

    c1, c2 = st.columns(2)
    with c1:
        beta_target = st.number_input("Target β value (可留空)", value=3.80)
    with c2:
        show_beta_target = st.checkbox("在图中显示目标 β", value=True)

    st.divider()

    # === Plot Axes Controls ===
    header_cols = st.columns([0.85, 0.15])
    with header_cols[0]:
        st.subheader("Plot Axes Controls")
    with header_cols[1]:
        help_badge("Plot Axes Controls", "横纵轴范围/刻度（留空=自动）。建议先运行一次再微调。")

    c1, c2, c3 = st.columns(3)
    with c1:
        x_tick = st.number_input("X tick step (years)", value=10.0)
    with c2:
        y1_min = st.number_input("Y₁ = β min", value=-2.0)
    with c3:
        y1_max = st.number_input("Y₁ = β max", value=5.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        y1_tick = st.number_input("Y₁ = β tick step", value=1.0)
    with c2:
        y2_min = st.number_input("Y₂ = Pf min", value=0.0)
    with c3:
        y2_max = st.number_input("Y₂ = Pf max", value=1.0)

    c1, c2 = st.columns(2)
    with c1:
        y2_tick = st.number_input("Y₂ = Pf tick step", value=0.1)
    with c2:
        show_pf = st.checkbox("显示 Pf (failure probability) 曲线", value=True)

# ---------------- 右列：Δx、Temperature、时间窗&MC、运行按钮与图 ----------------
with right_col:
    # === Convection Zone (Δx) ===
    header_cols = st.columns([0.85, 0.15])
    with header_cols[0]:
        st.subheader("Convection Zone (Δx)")
    with header_cols[1]:
        help_badge("Convection Zone (Δx)", "选择 Δx 模式；潮差区可编辑参数，沉没/喷淋可使用预设。")

    dx_display_to_code = {
        "Please select": None,
        "Zero – submerged/spray (Δx = 0)": "zero",
        "Beta – submerged (locked)": "beta_submerged",
        "Beta – tidal (editable)": "beta_tidal",
    }
    dx_choice = st.selectbox("Δx mode", list(dx_display_to_code.keys()), index=0)
    dx_mode_internal = dx_display_to_code.get(dx_choice, None)

    # Δx 参数区
    if dx_mode_internal in ("beta_submerged", "beta_tidal"):
        if dx_mode_internal == "beta_submerged":
            # 锁定为给定常用值
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                dx_mu = st.number_input("Δx μ (mm)", value=8.9, disabled=True)
            with col2:
                dx_sd = st.number_input("Δx σ (mm)", value=5.6, disabled=True)
            with col3:
                dx_L  = st.number_input("Δx L (mm)", value=0.0, disabled=True)
            with col4:
                dx_U  = st.number_input("Δx U (mm)", value=50.0, disabled=True)
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                dx_mu = st.number_input("Δx μ (mm)", value=10.0)
            with col2:
                dx_sd = st.number_input("Δx σ (mm)", value=5.0)
            with col3:
                dx_L  = st.number_input("Δx L (mm)", value=0.0)
            with col4:
                dx_U  = st.number_input("Δx U (mm)", value=50.0)

    st.divider()

    # === Temperature（放在 Δx 下方；行内双列 C 与 K 同时可编辑） ===
    header_cols = st.columns([0.85, 0.15])
    with header_cols[0]:
        st.subheader("Temperature (Actual & Reference)")
    with header_cols[1]:
        help_badge("Temperature", "可同时编辑 °C 与 K，自动双向换算。σ 不变，°C 与 K 的 σ 数值相同。")

    # 实际温度 μ
    sync_ck_pair("Actual temperature μ (°C)", "Actual temperature μ (K)",
                 key_c="Treal_mu_C", key_k="Treal_mu_K", default_c=23.0, default_k=296.15)
    # 实际温度 σ（°C 与 K 的标准差数值一致）
    c1, c2 = st.columns(2)
    with c1:
        Treal_sd_C = st.number_input("Actual temperature σ (°C)", value=3.0, help="标准差不做单位换算，数值一致")
    with c2:
        # 同步填写
        Treal_sd_K = st.number_input("Actual temperature σ (K)", value=Treal_sd_C, help="= σ(°C)")

    # 参考温度
    sync_ck_pair("Reference temperature Tref (°C)", "Reference temperature Tref (K)",
                 key_c="Tref_C", key_k="Tref_K", default_c=23.0, default_k=296.15)

    st.divider()

    # === Time Window & Monte Carlo（两列） ===
    header_cols = st.columns([0.85, 0.15])
    with header_cols[0]:
        st.subheader("Time Window & Monte Carlo")
    with header_cols[1]:
        help_badge("Time Window & Monte Carlo", "将时间窗参数与 Monte Carlo 总量分成两列，便于快速录入。")

    tw_col1, tw_col2 = st.columns(2)
    with tw_col1:
        t_start_disp = st.number_input("Plot start time (yr)", value=0.9)
        t_end = st.number_input("Plot end time (Target yr)", value=50.0, min_value=0.0)
        t_points = st.number_input("Number of time points", value=200, min_value=10, step=10)
    with tw_col2:
        N = st.number_input("Monte Carlo samples N", value=100000, min_value=1000, step=1000)
        seed = st.number_input("Random seed", value=42, step=1)

    st.divider()

    # === 运行 & 绘图 ===
    run_btn = st.button("Run Simulation", type="primary")
    if run_btn:
        # 基本校验
        if alpha_choice == "Please select":
            st.error("请先选择 α Preset（或用 Custom 自填）。")
        elif dx_mode_internal is None:
            st.error("请先选择 Δx 模式。")
        elif t_end <= t_start_disp:
            st.error("Plot end time 必须大于 Plot start time。")
        else:
            try:
                # 组装参数
                params = {
                    "Cs_mu": float(Cs_mu),
                    "Cs_sd": float(Cs_sd),
                    "alpha_mu": float(alpha_mu),
                    "alpha_sd": float(alpha_sd),
                    "alpha_L":  float(alpha_L),
                    "alpha_U":  float(alpha_U),
                    "D0_mu": float(D0_mu),
                    "D0_sd": float(max(0.2*D0_mu, 0.0)),  # 自动 = 0.2 × μ
                    "cover_mu": float(cover_mu),
                    "cover_sd": float(cover_sd),
                    "Ccrit_mu": float(Ccrit_mu),
                    "Ccrit_sd": float(Ccrit_sd),
                    "Ccrit_L":  float(Ccrit_L),
                    "Ccrit_U":  float(Ccrit_U),
                    "be_mu": float(be_mu),
                    "be_sd": float(be_sd),
                    "Treal_mu": float(st.session_state.get("Treal_mu_K", 296.15)),  # 用 K
                    "Treal_sd": float(Treal_sd_K),  # σ 数值相同
                    "t0": float(t0_year),
                    "Tref": float(st.session_state.get("Tref_K", 296.15)),  # 用 K
                    "C0": float(C0),
                    "dx_mode": dx_mode_internal,
                }
                if dx_mode_internal in ("beta_submerged", "beta_tidal"):
                    params.update({
                        "dx_mu": float(dx_mu),
                        "dx_sd": float(dx_sd),
                        "dx_L":  float(dx_L),
                        "dx_U":  float(dx_U),
                    })

                # 先算全域（从 0 → t_end），再裁切显示窗
                df_full = run_fib_chloride(params, N=int(N), seed=int(seed),
                                           t_start=0.0, t_end=float(t_end), t_points=int(t_points))
                df_window = df_full[(df_full["t_years"] >= float(t_start_disp)) &
                                    (df_full["t_years"] <= float(t_end))].copy()
                if df_window.empty:
                    st.error("显示窗口内没有点；请增加时间点数量或调整时间范围。")
                else:
                    # 画图
                    axes_cfg = {
                        "x_tick":  float(x_tick) if x_tick is not None else None,
                        "y1_min":  float(y1_min) if y1_min is not None else None,
                        "y1_max":  float(y1_max) if y1_max is not None else None,
                        "y1_tick": float(y1_tick) if y1_tick is not None else None,
                        "y2_min":  float(y2_min) if y2_min is not None else None,
                        "y2_max":  float(y2_max) if y2_max is not None else None,
                        "y2_tick": float(y2_tick) if y2_tick is not None else None,
                    }
                    fig = plot_beta(
                        df_window,
                        t_end=float(t_end),
                        axes_cfg=axes_cfg,
                        show_pf=bool(show_pf),
                        beta_target=float(beta_target) if beta_target is not None else None,
                        show_beta_target=bool(show_beta_target),
                    )
                    st.pyplot(fig)

                    # 数据下载
                    csv_bytes = df_window.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "下载当前窗口数据 (CSV)",
                        data=csv_bytes,
                        file_name="fib_output_window.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"输入无效或计算出错：{e}")

# ===============（可选）在最底部附上开关：显示/隐藏更多细节 ===============
with st.expander("Advanced notes / 调试信息（可选）"):
    st.markdown("""
- 本应用保持你原始数学模型与随机分布设定（Cs~Lognormal, α/Ccrit~Beta[L,U]，Δx 三模式）。
- 温度（实际与参考）统一以 K 进入计算，但提供 °C↔K 双向同步输入。
- Time Window & Monte Carlo 两列布置；坐标控制与目标 β 提供图上指示。
- 右侧每个小节标题旁 “?” 可录入图文帮助说明；若部署在云端，请将图片一并推送到仓库或使用上传功能。
""")
