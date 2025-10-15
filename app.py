import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erfc
import streamlit as st

# =============================
# Core math functions
# =============================

def beta_from_pf(Pf):
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
    mu_Cs, sd_Cs = params["Cs_mu"], params["Cs_sd"]
    mu_alpha, sd_alpha = params["alpha_mu"], params["alpha_sd"]
    alpha_L, alpha_U = params["alpha_L"], params["alpha_U"]
    mu_D0, sd_D0 = params["D0_mu"], params["D0_sd"]
    mu_cover, sd_cover = params["cover_mu"], params["cover_sd"]
    mu_Ccrit, sd_Ccrit = params["Ccrit_mu"], params["Ccrit_sd"]
    Ccrit_L, Ccrit_U = params["Ccrit_L"], params["Ccrit_U"]
    mu_be, sd_be = params["be_mu"], params["be_sd"]
    mu_Treal, sd_Treal = params["Treal_mu"], params["Treal_sd"]
    Tref, t0_year, C0 = params["Tref"], params["t0"], params["C0"]

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
    y_pf = df_window["Pf"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x_abs, y_beta, lw=2, label="β(t)", color='#1f77b4')
    ax1.set_xlabel("Time (yr)", fontsize=12)
    ax1.set_ylabel("Reliability index β(-)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = None
    if show_pf:
        ax2 = ax1.twinx()
        ax2.plot(x_abs, y_pf, linestyle="--", lw=1.6, label="Pf(t)", color='#ff7f0e')
        ax2.set_ylabel("Failure probability Pf(t)", fontsize=12)

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

    ax1.axvline(float(t_end), linestyle=":", lw=1.5, color='gray', alpha=0.7)
    
    # Add target beta line and annotation
    if show_beta_target and beta_target is not None:
        ax1.axhline(beta_target, color='red', linestyle='--', lw=1.5, label=f'Target β = {beta_target}')
        crossing_year = None
        for i in range(len(y_beta) - 1):
            if (y_beta[i] >= beta_target and y_beta[i+1] < beta_target) or \
               (y_beta[i] <= beta_target and y_beta[i+1] > beta_target):
                t1, t2 = x_abs[i], x_abs[i+1]
                b1, b2 = y_beta[i], y_beta[i+1]
                crossing_year = t1 + (beta_target - b1) * (t2 - t1) / (b2 - b1)
                break
        
        textstr = f'Target β = {beta_target:.2f}'
        if crossing_year is not None:
            textstr += f'\nYear reached: {crossing_year:.2f} yr'
            ax1.axvline(crossing_year, color='red', linestyle=':', lw=1.0, alpha=0.7)
        else:
            textstr += '\nNot reached in time range'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        ax1.legend(loc='upper left')
    
    fig.tight_layout()
    return fig


# =============================
# Streamlit helpers: temperature sync & D0 auto-sigma
# =============================

def _ensure_default(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

def _apply_temp_sync(prefix):
    last = st.session_state.get(f"{prefix}_last", "C")
    if last == "C":
        st.session_state[f"{prefix}_K"] = st.session_state[f"{prefix}_C"] + 273.15
    else:
        st.session_state[f"{prefix}_C"] = st.session_state[f"{prefix}_K"] - 273.15

def _edit_C(prefix):
    st.session_state[f"{prefix}_last"] = "C"
    _apply_temp_sync(prefix)

def _edit_K(prefix):
    st.session_state[f"{prefix}_last"] = "K"
    _apply_temp_sync(prefix)

def _copy_sigma(prefix):  # σ(°C) == σ(K)
    last = st.session_state.get(f"{prefix}_sd_last", "C")
    if last == "C":
        st.session_state[f"{prefix}_sd_K"] = st.session_state[f"{prefix}_sd_C"]
    else:
        st.session_state[f"{prefix}_sd_C"] = st.session_state[f"{prefix}_sd_K"]

def _edit_sd_C(prefix):
    st.session_state[f"{prefix}_sd_last"] = "C"
    _copy_sigma(prefix)

def _edit_sd_K(prefix):
    st.session_state[f"{prefix}_sd_last"] = "K"
    _copy_sigma(prefix)

def _d0_mu_changed():
    st.session_state["d0_sd_val"] = round(st.session_state["d0_mu_val"] * 0.2, 6)


# =============================
# Streamlit App
# =============================

def main():
    st.set_page_config(page_title="fib Chloride Model", layout="wide", initial_sidebar_state="expanded")
    
    # ---- CSS: 更窄、更大字体、间距更紧 ----
    st.markdown("""
        <style>
        /* 数字/文本输入更窄更大 */
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            width: 5% !important;
            min-width: 80px !important;
            font-size: 18px !important;
            padding: 6px 8px !important;
        }
        /* select 更窄 */
        div[data-testid="stSelectbox"] > div > div {
            width: 18% !important;
            min-width: 160px !important;
        }
        div[data-testid="stSelectbox"] input {
            font-size: 16px !important;
        }
        /* label 与文本更大 */
        label, .stMarkdown, .stText {
            font-size: 16px !important;
        }
        /* 压缩上下间距 */
        section.main > div.block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; }
        .stMarkdown h2, .stMarkdown h3 { margin-top: 0.4rem; margin-bottom: 0.2rem; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔬 fib Chloride Ingress Model - Reliability Analysis")
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📋 Model Parameters")
        
        # Locked parameters (Ccrit)
        st.subheader("🔒 Critical Chloride Content (Ccrit) - Locked")
        Ccrit_mu = st.number_input("Ccrit μ (wt-%/binder)", value=0.60, disabled=True, key="ccrit_mu")
        Ccrit_sd = st.number_input("Ccrit σ", value=0.15, disabled=True, key="ccrit_sd")
        Ccrit_L = st.number_input("Ccrit lower bound L", value=0.20, disabled=True, key="ccrit_l")
        Ccrit_U = st.number_input("Ccrit upper bound U", value=2.00, disabled=True, key="ccrit_u")
        
        # Temperature coefficient (locked)
        st.subheader("🔒 Temperature Coefficient (be) - Locked")
        be_mu = st.number_input("Temperature coeff (b_e) μ", value=4800.0, disabled=True, key="be_mu")
        be_sd = st.number_input("Temperature coeff (b_e) σ", value=700.0, disabled=True, key="be_sd")
        
        # Alpha preset
        alpha_header = st.columns([0.9, 0.1])
        with alpha_header[0]:
            st.subheader("⚙️ Ageing Exponent (α)")
        with alpha_header[1]:
            st.markdown("")  # spacing
            with st.popover("❓"):
                st.markdown("### Ageing Exponent (α) Information")
                st.write("Ageing exponent α 描述扩散系数随时间减小的速率（老化效应）。")
                st.write("常见范围：0.3~0.6 视体系而定。")
        
        alpha_presets = {
            "Please select": None,
            "Portland Cement (PCC) 0.30 0.12": (0.30, 0.12, 0.0, 1.0),
            "PCC w/ ≥ 20% Fly Ash 0.60 0.15": (0.60, 0.15, 0.0, 1.0),
            "PCC w/ Blast Furnace Slag 0.45 0.20": (0.45, 0.20, 0.0, 1.0),
            "All types (atmospheric zone) 0.65 0.12": (0.65, 0.12, 0.0, 1.0),
            "Custom – enter values": "custom",
        }
        alpha_choice = st.selectbox("α preset", list(alpha_presets.keys()), key="alpha_preset")
        
        # Determine if fields should be locked
        preset_data = alpha_presets[alpha_choice]
        is_locked = (preset_data is not None and preset_data != "custom")
        is_custom = (preset_data == "custom")
        
        if preset_data is None:
            alpha_mu = 0.0; alpha_sd = 0.0; alpha_L = 0.0; alpha_U = 0.0
            st.number_input("α μ", value=alpha_mu, disabled=True, key="alpha_empty_mu")
            st.number_input("α σ", value=alpha_sd, disabled=True, key="alpha_empty_sd")
            st.number_input("α lower bound L", value=alpha_L, disabled=True, key="alpha_empty_L")
            st.number_input("α upper bound U", value=alpha_U, disabled=True, key="alpha_empty_U")
        elif is_custom:
            alpha_mu = st.number_input("α μ", value=0.30, min_value=0.0, max_value=1.0, key="alpha_custom_mu")
            alpha_sd = st.number_input("α σ", value=0.12, min_value=0.0, key="alpha_custom_sd")
            alpha_L  = st.number_input("α lower bound L", value=0.0, min_value=0.0, key="alpha_custom_L")
            alpha_U  = st.number_input("α upper bound U", value=1.0, min_value=0.0, key="alpha_custom_U")
        else:
            alpha_mu, alpha_sd, alpha_L, alpha_U = preset_data
            st.number_input("α μ", value=float(alpha_mu), disabled=True, key="alpha_locked_mu")
            st.number_input("α σ", value=float(alpha_sd), disabled=True, key="alpha_locked_sd")
            st.number_input("α lower bound L", value=float(alpha_L), disabled=True, key="alpha_locked_L")
            st.number_input("α upper bound U", value=float(alpha_U), disabled=True, key="alpha_locked_U")
        
        # Reference age
        t0_header = st.columns([0.9, 0.1])
        with t0_header[0]:
            st.subheader("📅 Reference Age (t0)")
        with t0_header[1]:
            st.markdown("")  # spacing
            with st.popover("❓"):
                st.markdown("### Reference Age (t0) Information")
                st.write("t0 是进行 DRCM 测试时的混凝土龄期（折算为年）。")
        
        t0_options = {
            "Please select": None,
            "0.0767 – 28 days": 0.0767,
            "0.1533 – 56 days": 0.1533,
            "0.2464 – 90 days": 0.2464,
        }
        t0_choice = st.selectbox("Reference age t0 (yr)", list(t0_options.keys()), key="t0_select")
        
        # Show t0 value in input box
        if t0_options[t0_choice] is None:
            t0 = st.number_input("t0 value (years)", value=0.0, disabled=True, key="t0_value", 
                                help="Select a reference age option above")
        else:
            t0 = st.number_input("t0 value (years)", value=t0_options[t0_choice], disabled=True, 
                                key="t0_value", help="Reference age in years")
        
        # Editable parameters
        st.subheader("✏️ Editable Parameters")
        
        C0 = st.number_input("Initial chloride C0 (wt-%/binder)", value=0.0, min_value=0.0)
        Cs_mu = st.number_input("Surface chloride μ (wt-%/binder)", value=3.5, min_value=0.0)
        Cs_sd = st.number_input("Surface chloride σ", value=1.0, min_value=0.0)
        
        # D0: μ 改变触发 σ=0.2×μ
        _ensure_default("d0_mu_val", 10.0)
        _ensure_default("d0_sd_val", round(st.session_state["d0_mu_val"] * 0.2, 6))
        D0_mu = st.number_input("DRCM0 μ (×1e-12 m²/s)", min_value=0.0, key="d0_mu_val",
                                on_change=_d0_mu_changed)
        st.number_input("DRCM0 σ (×1e-12 m²/s)", value=st.session_state["d0_sd_val"],
                        disabled=True, key="d0_sd_view",
                        help="Auto = 0.2 × μ（随 μ 变化即刻更新）")
        D0_sd = float(st.session_state["d0_sd_val"])
        
        cover_mu = st.number_input("Cover μ (mm)", value=50.0, min_value=0.0)
        cover_sd = st.number_input("Cover σ (mm)", value=10.0, min_value=0.0)
    
    with col2:
        st.header("⚙️ Simulation Settings")
        
        # Convection zone
        dx_header = st.columns([0.9, 0.1])
        with dx_header[0]:
            st.subheader("🌊 Convection Zone (Δx)")
        with dx_header[1]:
            st.markdown("")  # spacing
            with st.popover("❓"):
                st.markdown("### Convection Zone (Δx) Information")
                st.write("Δx = 0（完全淹没/喷淋）或 Beta 区间（部分淹没/潮汐区）。")
        
        dx_options = {
            "Please select": None,
            "Zero – submerged/spray (Δx = 0)": "zero",
            "Beta – submerged (locked)": "beta_submerged",
            "Beta – tidal (editable)": "beta_tidal",
        }
        dx_choice = st.selectbox("Δx mode", list(dx_options.keys()), key="dx_mode_select")
        dx_mode = dx_options[dx_choice]

        # 准备 Δx 变量的默认值，便于 params.update
        dx_mu = dx_sd = dx_L = dx_U = 0.0
        
        # Determine if fields should be locked or editable
        if dx_mode is None:
            st.number_input("Δx Beta mean μ (mm)", value=0.0, disabled=True, key="dx_empty_mu")
            st.number_input("Δx Beta SD σ (mm)", value=0.0, disabled=True, key="dx_empty_sd")
            st.number_input("Δx lower bound L (mm)", value=0.0, disabled=True, key="dx_empty_L")
            st.number_input("Δx upper bound U (mm)", value=0.0, disabled=True, key="dx_empty_U")
        elif dx_mode == "zero":
            st.number_input("Δx Beta mean μ (mm)", value=0.0, disabled=True, key="dx_zero_mu")
            st.number_input("Δx Beta SD σ (mm)", value=0.0, disabled=True, key="dx_zero_sd")
            st.number_input("Δx lower bound L (mm)", value=0.0, disabled=True, key="dx_zero_L")
            st.number_input("Δx upper bound U (mm)", value=0.0, disabled=True, key="dx_zero_U")
            st.info("Δx = 0 for submerged/spray conditions")
        elif dx_mode == "beta_submerged":
            dx_mu, dx_sd, dx_L, dx_U = 8.9, 5.6, 0.0, 50.0
            st.number_input("Δx Beta mean μ (mm)", value=float(dx_mu), disabled=True, key="dx_locked_mu")
            st.number_input("Δx Beta SD σ (mm)", value=float(dx_sd), disabled=True, key="dx_locked_sd")
            st.number_input("Δx lower bound L (mm)", value=float(dx_L), disabled=True, key="dx_locked_L")
            st.number_input("Δx upper bound U (mm)", value=float(dx_U), disabled=True, key="dx_locked_U")
        else:  # beta_tidal - editable
            dx_mu = st.number_input("Δx Beta mean μ (mm)", value=10.0, min_value=0.0, key="dx_tidal_mu")
            dx_sd = st.number_input("Δx Beta SD σ (mm)", value=5.0, min_value=0.0, key="dx_tidal_sd")
            dx_L  = st.number_input("Δx lower bound L (mm)", value=0.0, min_value=0.0, key="dx_tidal_L")
            dx_U  = st.number_input("Δx upper bound U (mm)", value=50.0, min_value=0.0, key="dx_tidal_U")
        
        # ---------------- Temperature Parameters (bidirectional binding) --------------
        temp_header = st.columns([0.9, 0.1])
        with temp_header[0]:
            st.subheader("🌡️ Temperature Parameters")
        with temp_header[1]:
            st.markdown("")
            with st.popover("❓"):
                st.markdown("### Temperature Information")
                st.write("K = °C + 273.15；标准差在 °C 与 K 下数值相同。")

        # 初始化默认值
        _ensure_default("Treal_C", 20.0)
        _ensure_default("Treal_K", st.session_state["Treal_C"] + 273.15)
        _ensure_default("Treal_last", "C")
        _ensure_default("Treal_sd_C", 5.0)
        _ensure_default("Treal_sd_K", st.session_state["Treal_sd_C"])
        _ensure_default("Treal_sd_last", "C")

        _ensure_default("Tref_C", 23.0)
        _ensure_default("Tref_K", st.session_state["Tref_C"] + 273.15)
        _ensure_default("Tref_last", "C")

        st.write("**Actual Temperature (mean)**")
        temp_mu_cols = st.columns(2)
        with temp_mu_cols[0]:
            st.number_input("°C", key="Treal_C", on_change=_edit_C, args=("Treal",))
        with temp_mu_cols[1]:
            st.number_input("K",  key="Treal_K", on_change=_edit_K, args=("Treal",))

        st.write("**Actual Temperature (std dev)**")
        temp_sd_cols = st.columns(2)
        with temp_sd_cols[0]:
            st.number_input("σ (°C)", min_value=0.0, key="Treal_sd_C",
                            on_change=_edit_sd_C, args=("Treal",))
        with temp_sd_cols[1]:
            st.number_input("σ (K)",  min_value=0.0, key="Treal_sd_K",
                            on_change=_edit_sd_K, args=("Treal",))

        st.write("**Reference Temperature**")
        temp_ref_cols = st.columns(2)
        with temp_ref_cols[0]:
            st.number_input("Tref (°C)", key="Tref_C", on_change=_edit_C, args=("Tref",))
        with temp_ref_cols[1]:
            st.number_input("Tref (K)",  key="Tref_K", on_change=_edit_K, args=("Tref",))

        # 供模型使用（全部用 K）
        Treal_mu = float(st.session_state["Treal_K"])
        Treal_sd = float(st.session_state["Treal_sd_K"])
        Tref     = float(st.session_state["Tref_K"])
        
        # ---------------- Time window & Monte Carlo (two columns) ----------------
        time_header = st.columns([0.9, 0.1])
        with time_header[0]:
            st.subheader("⏱️ Time Window & Monte Carlo")
        with time_header[1]:
            st.markdown("")
            with st.popover("❓"):
                st.markdown("### Time Window & Monte Carlo Information")
                st.write("Start/End/Points 控制时间窗口与采样点；N/Seed 控制蒙特卡洛。")
        
        time_col1, time_col2 = st.columns(2)
        with time_col1:
            t_start = st.number_input("Plot start time (yr)", value=0.9, min_value=0.0, key="t_start")
            t_end   = st.number_input("Plot end time (Target yr)", value=50.0, min_value=0.1, key="t_end")
            t_points = st.number_input("Number of time points", value=200, min_value=10, key="t_points")
        with time_col2:
            N    = st.number_input("Monte Carlo samples N", value=100000, min_value=1000, key="n_samples")
            seed = st.number_input("Random seed", value=42, min_value=0, key="seed")
        
        # Target reliability
        target_header = st.columns([0.9, 0.1])
        with target_header[0]:
            st.subheader("🎯 Target Reliability Index")
        with target_header[1]:
            st.markdown("")
            with st.popover("❓"):
                st.markdown("### Target Reliability Index (β) Information")
                st.write("典型：β=1.5/2.3/3.8 等。")
        
        beta_target = st.number_input("Target β value", value=1.5, min_value=0.0)
        show_beta_target = st.checkbox("Show target β on plot", value=True)
        
        # Axes controls
        axes_header = st.columns([0.9, 0.1])
        with axes_header[0]:
            st.subheader("📊 Plot Axes Controls")
        with axes_header[1]:
            st.markdown("")
            with st.popover("❓"):
                st.markdown("### Plot Axes Controls Information")
                st.write("留空或 0 使用自动刻度；先跑图再微调更直观。")
        
        st.info("Leave blank or 0 for auto-scaling. Run simulation first, then adjust if needed.")
        
        col_ax1, col_ax2 = st.columns(2)
        with col_ax1:
            x_tick = st.number_input("X tick step (years)", value=10.0, min_value=0.0)
            y1_min = st.number_input("Y₁ (β) min", value=-2.0)
            y1_max = st.number_input("Y₁ (β) max", value=5.0)
            y1_tick = st.number_input("Y₁ (β) tick step", value=1.0, min_value=0.0)
        
        with col_ax2:
            y2_min = st.number_input("Y₂ (Pf) min", value=0.0, min_value=0.0)
            y2_max = st.number_input("Y₂ (Pf) max", value=1.0, min_value=0.0)
            y2_tick = st.number_input("Y₂ (Pf) tick step", value=0.1, min_value=0.0)
        
        show_pf = st.checkbox("Show Pf (failure probability) curve", value=True)
    
    # Run simulation button
    st.markdown("---")
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        try:
            # Validation checks
            if alpha_choice == "Please select":
                st.error("❌ Please select an α preset (Ageing exponent)")
                return
            if t0_choice == "Please select":
                st.error("❌ Please select a reference age t0")
                return
            if dx_choice == "Please select":
                st.error("❌ Please select a Δx mode")
                return
            
            # Get the actual t0 value
            if t0_options[t0_choice] is None:
                st.error("❌ Please select a valid reference age t0")
                return
            t0 = t0_options[t0_choice]
            
            with st.spinner("Running simulation..."):
                # Prepare parameters
                params = {
                    "Cs_mu": Cs_mu, "Cs_sd": Cs_sd,
                    "alpha_mu": alpha_mu, "alpha_sd": alpha_sd,
                    "alpha_L": alpha_L, "alpha_U": alpha_U,
                    "D0_mu": D0_mu, "D0_sd": D0_sd,
                    "cover_mu": cover_mu, "cover_sd": cover_sd,
                    "Ccrit_mu": Ccrit_mu, "Ccrit_sd": Ccrit_sd,
                    "Ccrit_L": Ccrit_L, "Ccrit_U": Ccrit_U,
                    "be_mu": be_mu, "be_sd": be_sd,
                    "Treal_mu": Treal_mu, "Treal_sd": Treal_sd,
                    "t0": t0, "Tref": Tref, "C0": C0,
                    "dx_mode": dx_mode,
                }
                
                if dx_mode in ("beta_submerged", "beta_tidal"):
                    params.update({"dx_mu": dx_mu, "dx_sd": dx_sd, "dx_L": dx_L, "dx_U": dx_U})
                
                # Run simulation
                df_full = run_fib_chloride(params, N=int(N), seed=int(seed), 
                                          t_start=0.0, t_end=t_end, t_points=int(t_points))
                df_window = df_full[(df_full["t_years"] >= t_start) & (df_full["t_years"] <= t_end)].copy()
                
                if df_window.empty:
                    st.error("Display window has no points; increase number of time points or adjust times.")
                else:
                    # Prepare axes config
                    axes_cfg = {
                        "x_tick": x_tick if x_tick > 0 else None,
                        "y1_min": y1_min, "y1_max": y1_max,
                        "y1_tick": y1_tick if y1_tick > 0 else None,
                        "y2_min": y2_min if y2_min > 0 else None,
                        "y2_max": y2_max if y2_max > 0 else None,
                        "y2_tick": y2_tick if y2_tick > 0 else None,
                    }
                    
                    # Plot results
                    fig = plot_beta(df_window, t_end=t_end, axes_cfg=axes_cfg, 
                                   show_pf=show_pf, beta_target=beta_target if show_beta_target else None,
                                   show_beta_target=show_beta_target)
                    
                    st.pyplot(fig)
                    
                    # Show results table
                    st.subheader("📈 Results Data")
                    st.dataframe(df_window.head(20), use_container_width=True)
                    
                    # Download button
                    csv = df_window.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="fib_chloride_results.csv",
                        mime="text/csv",
                    )
                    
                    st.success("✅ Simulation completed successfully!")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
