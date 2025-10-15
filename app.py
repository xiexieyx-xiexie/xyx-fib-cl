import numpy as np, math, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erfc
import streamlit as st

# =============================
# Core math (与你给的一致)
# =============================

def beta_from_pf(Pf):
    return -norm.ppf(Pf)

def lognorm_from_mu_sd(rng, n, mu, sd):
    sigma2 = math.log(1 + (sd**2) / (mu**2))
    mu_log = math.log(mu) - 0.5 * sigma2
    sigma = math.sqrt(sigma2)
    return rng.lognormal(mu_log, sigma, n)

def beta01_shapes_from_mean_sd(mu, sd):
    mu = max(min(mu, 1 - 1e-9), 1e-9)
    var = max(sd**2, 1e-12)
    t = mu * (1 - mu) / var - 1
    a = max(mu * t, 1e-6)
    b = max((1 - mu) * t, 1e-6)
    return a, b

def beta_interval_from_mean_sd(rng, n, mu, sd, L, U):
    if U <= L:
        raise ValueError("Upper bound must be greater than lower bound.")
    mu = max(min(mu, U - 1e-12), L + 1e-12)
    sd = max(sd, 1e-12)
    mu01 = (mu - L) / (U - L)
    sd01 = sd / (U - L)
    a, b = beta01_shapes_from_mean_sd(mu01, sd01)
    return L + (U - L) * rng.beta(a, b, n)

def run_fib_chloride(params, N=100000, seed=42, t_start=0.9, t_end=50.0, t_points=200):
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

    # Distributions
    Cs    = lognorm_from_mu_sd(rng, N, mu_Cs, sd_Cs)
    alpha = beta_interval_from_mean_sd(rng, N, mu_alpha, sd_alpha, alpha_L, alpha_U)
    Ccrit = beta_interval_from_mean_sd(rng, N, mu_Ccrit, sd_Ccrit, Ccrit_L, Ccrit_U)

    D0      = np.maximum(rng.normal(mu_D0, sd_D0, N), 1e-3) * 1e-12
    cover_m = np.maximum(rng.normal(mu_cover, sd_cover, N), 1.0) / 1000.0
    be      = np.maximum(rng.normal(mu_be, sd_be, N), 1.0)
    Treal   = np.maximum(rng.normal(mu_Treal, sd_Treal, N), 250.0)

    # Model
    t0_sec  = t0_year * 365.25 * 24 * 3600.0
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

# =============================
# Plot（与你Tkinter图一致的风格）
# =============================

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
# Helpers：稳态同步 + 无spinner
# =============================

def _ensure_default(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

def _two_way_temp(prefix, add=273.0):
    """
    稳定双向：优先以本轮输入为准；返回 (C, K)
    """
    _ensure_default(f"{prefix}_C", 20.0)
    _ensure_default(f"{prefix}_K", st.session_state[f"{prefix}_C"] + add)

    c1, c2 = st.columns(2)
    with c1:
        new_C = st.number_input("°C" if prefix=="Treal" else "Tref (°C)", value=float(st.session_state[f"{prefix}_C"]), key=f"{prefix}_C_in")
    with c2:
        new_K = st.number_input("K" if prefix=="Treal" else "Tref (K)",  value=float(st.session_state[f"{prefix}_K"]), key=f"{prefix}_K_in")

    # 谁变了用谁
    if float(new_C) != float(st.session_state[f"{prefix}_C"]):
        st.session_state[f"{prefix}_C"] = float(new_C)
        st.session_state[f"{prefix}_K"] = float(new_C) + add
    elif float(new_K) != float(st.session_state[f"{prefix}_K"]):
        st.session_state[f"{prefix}_K"] = float(new_K)
        st.session_state[f"{prefix}_C"] = float(new_K) - add

    return st.session_state[f"{prefix}_C"], st.session_state[f"{prefix}_K"]

def _sigma_equal(prefix, default=5.0):
    """
    σ(°C) == σ(K) 强制一致；返回 (σC, σK)
    """
    _ensure_default(f"{prefix}_sd_C", default)
    _ensure_default(f"{prefix}_sd_K", default)

    s1, s2 = st.columns(2)
    with s1:
        new_sC = st.number_input("σ (°C)", value=float(st.session_state[f"{prefix}_sd_C"]), min_value=0.0, key=f"{prefix}_sd_C_in")
    with s2:
        new_sK = st.number_input("σ (K)",  value=float(st.session_state[f"{prefix}_sd_K"]), min_value=0.0, key=f"{prefix}_sd_K_in")

    if float(new_sC) != float(st.session_state[f"{prefix}_sd_C"]):
        st.session_state[f"{prefix}_sd_C"] = float(new_sC)
        st.session_state[f"{prefix}_sd_K"] = float(new_sC)
    elif float(new_sK) != float(st.session_state[f"{prefix}_sd_K"]):
        st.session_state[f"{prefix}_sd_K"] = float(new_sK)
        st.session_state[f"{prefix}_sd_C"] = float(new_sK)

    return st.session_state[f"{prefix}_sd_C"], st.session_state[f"{prefix}_sd_K"]

# =============================
# App（抄你喜欢的布局）
# =============================

def main():
    st.set_page_config(page_title="fib Chloride Model", layout="wide")

    # 隐藏所有 number_input 的 +/- 按钮
    st.markdown("""
        <style>
        /* Chrome/Safari/Edge */
        input[type=number]::-webkit-outer-spin-button,
        input[type=number]::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type=number] { -moz-appearance: textfield; } /* Firefox */
        </style>
    """, unsafe_allow_html=True)

    st.title("fib Chloride Ingress – Reliability (Streamlit)")
    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.header("Model Parameters")

        # --- Ccrit (locked) ---
        st.subheader("Critical Chloride Content (Ccrit) – locked")
        Ccrit_mu = st.number_input("Ccrit μ (wt-%/binder)", value=0.60, disabled=True, key="Ccrit_mu")
        Ccrit_sd = st.number_input("Ccrit σ", value=0.15, disabled=True, key="Ccrit_sd")
        Ccrit_L  = st.number_input("Ccrit lower bound L", value=0.20, disabled=True, key="Ccrit_L")
        Ccrit_U  = st.number_input("Ccrit upper bound U", value=2.00, disabled=True, key="Ccrit_U")

        # --- be (locked) ---
        st.subheader("Temperature coeff (b_e) – locked")
        be_mu = st.number_input("Temperature coeff (b_e) μ", value=4800.0, disabled=True, key="be_mu")
        be_sd = st.number_input("Temperature coeff (b_e) σ", value=700.0, disabled=True, key="be_sd")

        # --- α preset ---
        st.subheader("Ageing exponent α preset")
        alpha_presets = {
            "Please select": None,
            "Portland Cement (PCC) 0.30 0.12": (0.30, 0.12, 0.0, 1.0),
            "PCC w/ ≥ 20% Fly Ash 0.60 0.15": (0.60, 0.15, 0.0, 1.0),
            "PCC w/ Blast Furnace Slag 0.45 0.20": (0.45, 0.20, 0.0, 1.0),
            "All types (atmospheric zone) 0.65 0.12": (0.65, 0.12, 0.0, 1.0),
            "Custom – enter values": "custom",
        }
        alpha_choice = st.selectbox("α preset", list(alpha_presets.keys()), key="alpha_preset")
        preset = alpha_presets[alpha_choice]

        if preset is None:
            alpha_mu = st.number_input("α μ", value=0.0, disabled=True, key="alpha_mu_empty")
            alpha_sd = st.number_input("α σ", value=0.0, disabled=True, key="alpha_sd_empty")
            alpha_L  = st.number_input("α lower bound L", value=0.0, disabled=True, key="alpha_L_empty")
            alpha_U  = st.number_input("α upper bound U", value=0.0, disabled=True, key="alpha_U_empty")
        elif preset == "custom":
            alpha_mu = st.number_input("α μ", value=0.30, min_value=0.0, max_value=1.0, key="alpha_mu_custom")
            alpha_sd = st.number_input("α σ", value=0.12, min_value=0.0, key="alpha_sd_custom")
            alpha_L  = st.number_input("α lower bound L", value=0.0, min_value=0.0, key="alpha_L_custom")
            alpha_U  = st.number_input("α upper bound U", value=1.0, min_value=0.0, key="alpha_U_custom")
        else:
            a_mu, a_sd, a_L, a_U = preset
            alpha_mu = st.number_input("α μ", value=float(a_mu), disabled=True, key="alpha_mu_lock")
            alpha_sd = st.number_input("α σ", value=float(a_sd), disabled=True, key="alpha_sd_lock")
            alpha_L  = st.number_input("α lower bound L", value=float(a_L), disabled=True, key="alpha_L_lock")
            alpha_U  = st.number_input("α upper bound U", value=float(a_U), disabled=True, key="alpha_U_lock")

        # --- t0 preset ---
        st.subheader("Reference age t0 (yr)")
        t0_options = {
            "Please select": None,
            "0.0767 – 28 days": 0.0767,
            "0.1533 – 56 days": 0.1533,
            "0.2464 – 90 days": 0.2464,
        }
        t0_choice = st.selectbox("t0 preset", list(t0_options.keys()), key="t0_choice")
        if t0_options[t0_choice] is None:
            t0 = st.number_input("t0 value (years)", value=0.0, disabled=True, key="t0_val_empty")
        else:
            t0 = st.number_input("t0 value (years)", value=t0_options[t0_choice], disabled=True, key="t0_val_lock")

        st.markdown("---")
        st.subheader("Editable Parameters")
        C0    = st.number_input("Initial chloride C0 (wt-%/binder)", value=0.0, min_value=0.0, key="C0")
        Cs_mu = st.number_input("Surface chloride μ (wt-%/binder)", value=3.5, min_value=0.0, key="Cs_mu")
        Cs_sd = st.number_input("Surface chloride σ", value=1.0, min_value=0.0, key="Cs_sd")

        # D0：σ=0.2×μ（实时）
        D0_mu = st.number_input("DRCM0 μ (×1e-12 m²/s)", value=10.0, min_value=0.0, key="D0_mu")
        D0_sd = round(0.2 * float(D0_mu), 6)
        st.number_input("DRCM0 σ (×1e-12 m²/s)", value=D0_sd, disabled=True, key="D0_sd_view")

        cover_mu = st.number_input("Cover μ (mm)", value=50.0, min_value=0.0, key="cover_mu")
        cover_sd = st.number_input("Cover σ (mm)", value=10.0, min_value=0.0, key="cover_sd")

    with right:
        st.header("Δx, Plot Settings")

        # --- Δx preset ---
        st.subheader("Convection zone Δx")
        dx_display_to_code = {
            "Please select": None,
            "Zero – submerged/spray (Δx = 0)": "zero",
            "Beta – submerged (locked)": "beta_submerged",
            "Beta – tidal (editable) – please enter": "beta_tidal",
        }
        dx_choice = st.selectbox("Δx mode", list(dx_display_to_code.keys()), key="dx_choice")
        dx_mode = dx_display_to_code[dx_choice]

        dx_mu = dx_sd = dx_L = dx_U = 0.0
        if dx_mode is None:
            st.number_input("Δx Beta mean μ (mm)", value=0.0, disabled=True, key="dx_mu_empty")
            st.number_input("Δx Beta SD σ (mm)", value=0.0, disabled=True, key="dx_sd_empty")
            st.number_input("Δx lower bound L (mm)", value=0.0, disabled=True, key="dx_L_empty")
            st.number_input("Δx upper bound U (mm)", value=0.0, disabled=True, key="dx_U_empty")
        elif dx_mode == "zero":
            st.number_input("Δx Beta mean μ (mm)", value=0.0, disabled=True, key="dx_mu_zero")
            st.number_input("Δx Beta SD σ (mm)", value=0.0, disabled=True, key="dx_sd_zero")
            st.number_input("Δx lower bound L (mm)", value=0.0, disabled=True, key="dx_L_zero")
            st.number_input("Δx upper bound U (mm)", value=0.0, disabled=True, key="dx_U_zero")
            st.info("Δx = 0 for submerged/spray conditions")
        elif dx_mode == "beta_submerged":
            dx_mu, dx_sd, dx_L, dx_U = 8.9, 5.6, 0.0, 50.0
            st.number_input("Δx Beta mean μ (mm)", value=dx_mu, disabled=True, key="dx_mu_lock")
            st.number_input("Δx Beta SD σ (mm)",   value=dx_sd, disabled=True, key="dx_sd_lock")
            st.number_input("Δx lower bound L (mm)", value=dx_L, disabled=True, key="dx_L_lock")
            st.number_input("Δx upper bound U (mm)", value=dx_U, disabled=True, key="dx_U_lock")
        else:  # beta_tidal
            dx_mu = st.number_input("Δx Beta mean μ (mm)", value=10.0, min_value=0.0, key="dx_mu_edit")
            dx_sd = st.number_input("Δx Beta SD σ (mm)",   value=5.0,  min_value=0.0, key="dx_sd_edit")
            dx_L  = st.number_input("Δx lower bound L (mm)", value=0.0, min_value=0.0, key="dx_L_edit")
            dx_U  = st.number_input("Δx upper bound U (mm)", value=50.0, min_value=0.0, key="dx_U_edit")

        # --- 温度（C↔K + σ一致；按你代码用 +273） ---
        st.subheader("Temperature")
        st.write("**Actual temperature (mean)**")
        Treal_C, Treal_K = _two_way_temp("Treal", add=273.0)

        st.write("**Actual temperature (std dev)**")
        Treal_sd_C, Treal_sd_K = _sigma_equal("Treal", default=5.0)

        st.write("**Reference temperature**")
        Tref_C, Tref_K = _two_way_temp("Tref", add=273.0)

        # 模型用 K
        Treal_mu = float(Treal_K)
        Treal_sd = float(Treal_sd_K)
        Tref     = float(Tref_K)

        st.markdown("---")
        st.subheader("Time window & Monte Carlo")
        c1, c2 = st.columns(2)
        with c1:
            t_start = st.number_input("Plot start time (yr)", value=0.9, min_value=0.0, key="t_start")
            t_end   = st.number_input("Plot end time (Target yr)", value=50.0, min_value=0.1, key="t_end")
            t_points = st.number_input("Number of time points", value=200, min_value=10, key="t_points")
        with c2:
            N    = st.number_input("Monte Carlo samples N", value=100000, min_value=1000, key="N")
            seed = st.number_input("Random seed", value=42, min_value=0, key="seed")

        st.markdown("---")
        st.subheader("Target Reliability Index")
        beta_target = st.number_input("Target β value (0 = ignore)", value=0.0, min_value=0.0, key="beta_target")
        show_beta_target = st.checkbox("Show target β on plot", value=False, key="show_beta")

        st.markdown("---")
        st.subheader("Axes controls (0 = auto) – RUN FIRST – adjust if needed")
        ax1, ax2 = st.columns(2)
        with ax1:
            x_tick = st.number_input("X tick step (years)", value=10.0, min_value=0.0, key="x_tick")
            y1_min = st.number_input("Y₁ = β min", value=-2.0, key="y1_min")
            y1_max = st.number_input("Y₁ = β max", value=5.0,  key="y1_max")
            y1_tick = st.number_input("Y₁ = β tick step", value=1.0, min_value=0.0, key="y1_tick")
        with ax2:
            y2_min = st.number_input("Y₂ = Pf min", value=0.0, min_value=0.0, key="y2_min")
            y2_max = st.number_input("Y₂ = Pf max", value=1.0, min_value=0.0, key="y2_max")
            y2_tick = st.number_input("Y₂ = Pf tick step", value=0.1, min_value=0.0, key="y2_tick")

        show_pf = st.checkbox("Show Pf (failure probability) curve", value=True, key="show_pf")

    st.markdown("---")
    if st.button("Run Simulation", type="primary"):
        try:
            # 必选校验
            if alpha_choice == "Please select":
                st.error("Please select an α preset.")
                st.stop()
            if t0_choice == "Please select":
                st.error("Please select a reference age t0.")
                st.stop()
            if dx_choice == "Please select":
                st.error("Please select a Δx mode.")
                st.stop()

            # 取 α（locked/custom 正确对应）
            preset = alpha_presets[alpha_choice]
            if preset == "custom":
                alpha_mu_val = float(st.session_state["alpha_mu_custom"])
                alpha_sd_val = float(st.session_state["alpha_sd_custom"])
                alpha_L_val  = float(st.session_state["alpha_L_custom"])
                alpha_U_val  = float(st.session_state["alpha_U_custom"])
            elif preset is None:
                st.error("Invalid α state.")
                st.stop()
            else:
                a_mu, a_sd, a_L, a_U = preset
                alpha_mu_val = float(a_mu)
                alpha_sd_val = float(a_sd)
                alpha_L_val  = float(a_L)
                alpha_U_val  = float(a_U)

            # Δx（locked 的值用我们上面设的常数；editable 用输入）
            if dx_mode == "beta_submerged":
                dx_mu_val, dx_sd_val, dx_L_val, dx_U_val = 8.9, 5.6, 0.0, 50.0
            elif dx_mode == "beta_tidal":
                dx_mu_val = float(st.session_state["dx_mu_edit"])
                dx_sd_val = float(st.session_state["dx_sd_edit"])
                dx_L_val  = float(st.session_state["dx_L_edit"])
                dx_U_val  = float(st.session_state["dx_U_edit"])
            elif dx_mode == "zero":
                dx_mu_val = dx_sd_val = dx_L_val = dx_U_val = 0.0
            else:
                st.error("Invalid Δx mode.")
                st.stop()

            if t_end <= t_start:
                st.error("Plot end time must be greater than plot start time.")
                st.stop()

            # 组装参数（与Tkinter版一致）
            params = {
                "Cs_mu": float(Cs_mu), "Cs_sd": float(Cs_sd),
                "alpha_mu": alpha_mu_val, "alpha_sd": alpha_sd_val,
                "alpha_L": alpha_L_val, "alpha_U": alpha_U_val,
                "D0_mu": float(D0_mu), "D0_sd": float(D0_sd),
                "cover_mu": float(cover_mu), "cover_sd": float(cover_sd),
                "Ccrit_mu": float(Ccrit_mu), "Ccrit_sd": float(Ccrit_sd),
                "Ccrit_L": float(Ccrit_L), "Ccrit_U": float(Ccrit_U),
                "be_mu": float(be_mu), "be_sd": float(be_sd),
                "Treal_mu": float(Treal_mu), "Treal_sd": float(Treal_sd),
                "t0": float(t0 if t0_options[t0_choice] is not None else 0.0),
                "Tref": float(Tref), "C0": float(C0),
                "dx_mode": dx_mode,
            }
            if dx_mode in ("beta_submerged", "beta_tidal"):
                params.update({"dx_mu": dx_mu_val, "dx_sd": dx_sd_val, "dx_L": dx_L_val, "dx_U": dx_U_val})

            # 运行
            df_full = run_fib_chloride(params, N=int(N), seed=int(seed),
                                       t_start=0.0, t_end=float(t_end), t_points=int(t_points))
            df_window = df_full[(df_full["t_years"] >= float(t_start)) & (df_full["t_years"] <= float(t_end))].copy()
            if df_window.empty:
                st.error("Display window has no points; increase number of time points or adjust times.")
                st.stop()

            # 轴配置（0为自动）
            axes_cfg = {
                "x_tick": st.session_state["x_tick"] if st.session_state["x_tick"] > 0 else None,
                "y1_min": st.session_state["y1_min"] if not (st.session_state["y1_min"] == 0 and st.session_state["y1_max"] == 0) else None,
                "y1_max": st.session_state["y1_max"] if not (st.session_state["y1_min"] == 0 and st.session_state["y1_max"] == 0) else None,
                "y1_tick": st.session_state["y1_tick"] if st.session_state["y1_tick"] > 0 else None,
                "y2_min": st.session_state["y2_min"] if st.session_state["y2_min"] > 0 else None,
                "y2_max": st.session_state["y2_max"] if st.session_state["y2_max"] > 0 else None,
                "y2_tick": st.session_state["y2_tick"] if st.session_state["y2_tick"] > 0 else None,
            }

            beta_target_val = None if (not show_beta_target or beta_target == 0.0) else float(beta_target)

            fig = plot_beta(
                df_window, t_end=float(t_end), axes_cfg=axes_cfg, show_pf=show_pf,
                beta_target=beta_target_val, show_beta_target=show_beta_target
            )
            st.pyplot(fig)

            st.subheader("Results (first 20 rows)")
            st.dataframe(df_window.head(20), use_container_width=True)

            csv = df_window.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="fib_output.csv", mime="text/csv")

            st.success("Simulation completed.")
        except Exception as e:
            st.error(f"Invalid input: {e}")

if __name__ == "__main__":
    main()
