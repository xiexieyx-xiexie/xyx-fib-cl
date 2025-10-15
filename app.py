# app.py — integrated updates (red target line, white/shadow annotation, Pf display options)
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import streamlit as st
from scipy.stats import norm
from scipy.special import erfc

st.set_page_config(page_title="fib Chloride Ingress – Reliability", layout="wide")

# Hide +/- spinners on number_inputs (incl. disabled)
st.markdown("""
<style>
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button{ -webkit-appearance:none; margin:0; }
input[type=number]{ -moz-appearance:textfield; }
</style>
""", unsafe_allow_html=True)

# ---------- Core model ----------
def beta_from_pf(Pf: np.ndarray) -> np.ndarray:
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
    if U <= L: raise ValueError("Upper bound must be greater than lower bound.")
    mu = max(min(mu, U - 1e-12), L + 1e-12)
    sd = max(sd, 1e-12)
    mu01 = (mu - L) / (U - L)
    sd01 = sd / (U - L)
    a, b = beta01_shapes_from_mean_sd(mu01, sd01)
    return L + (U - L) * rng.beta(a, b, n)

def run_fib_chloride(params, N=100000, seed=42, t_start=0.9, t_end=50.0, t_points=200):
    rng = np.random.default_rng(seed)
    t_years = np.linspace(float(t_start), float(t_end), int(t_points))

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

    dx_mode = params["dx_mode"]
    if dx_mode == "zero":
        dx_mm = np.zeros(N)
    elif dx_mode in ("beta_submerged", "beta_tidal"):
        dx_mm = beta_interval_from_mean_sd(
            rng, N, mu=params["dx_mu"], sd=params["dx_sd"], L=params["dx_L"], U=params["dx_U"]
        )
    else:
        raise ValueError("dx_mode must be one of: zero, beta_submerged, beta_tidal")

    Cs    = lognorm_from_mu_sd(rng, N, mu_Cs, sd_Cs)
    alpha = beta_interval_from_mean_sd(rng, N, mu_alpha, sd_alpha, alpha_L, alpha_U)
    Ccrit = beta_interval_from_mean_sd(rng, N, mu_Ccrit, sd_Ccrit, Ccrit_L, Ccrit_U)

    D0      = np.maximum(rng.normal(mu_D0, sd_D0, N), 1e-3) * 1e-12
    cover_m = np.maximum(rng.normal(mu_cover, sd_cover, N), 1.0) / 1000.0
    be      = np.maximum(rng.normal(mu_be, sd_be, N), 1.0)
    Treal   = np.maximum(rng.normal(mu_Treal, sd_Treal, N), 250.0)

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

# ---------- Plotting ----------
def plot_beta(df_window, t_end, axes_cfg=None,
              show_pf=True, beta_target=None, show_beta_target=False,
              pf_mode="overlay"):
    """Return (fig_beta, (x_years, y_pf))"""
    x_abs = df_window["t_years"].to_numpy()
    y_beta = df_window["beta"].to_numpy()
    y_pf   = df_window["Pf"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x_abs, y_beta, lw=2, label="β(t)")
    ax1.set_xlabel("Time (yr)")
    ax1.set_ylabel("Reliability index β(-)")
    ax1.grid(True)

    # Pf overlay mode
    overlay_pf = (pf_mode == "overlay")
    ax2 = None
    if show_pf and overlay_pf:
        ax2 = ax1.twinx()
        ax2.plot(x_abs, y_pf, linestyle="--", lw=1.6, label="Pf(t)")
        ax2.set_ylabel("Failure probability Pf(t)")

    axes_cfg = axes_cfg or {}
    ax1.set_xlim(0, float(t_end))
    x_tick = axes_cfg.get("x_tick", None)
    if x_tick and x_tick > 0:
        ax1.set_xticks(np.arange(0, float(t_end) + 1e-12, x_tick))

    y1_min = axes_cfg.get("y1_min", None)
    y1_max = axes_cfg.get("y1_max", None)
    y1_tick = axes_cfg.get("y1_tick", None)
    if y1_min is not None and y1_max is not None and y1_max > y1_min:
        ax1.set_ylim(y1_min, y1_max)
    if y1_tick and y1_tick > 0:
        ymin, ymax = ax1.get_ylim()
        ax1.set_yticks(np.arange(ymin, ymax + 1e-12, y1_tick))

    if ax2 is not None:
        y2_min = axes_cfg.get("y2_min", None)
        y2_max = axes_cfg.get("y2_max", None)
        y2_tick = axes_cfg.get("y2_tick", None)
        if y2_min is not None and y2_max is not None and y2_max > y2_min:
            ax2.set_ylim(y2_min, y2_max)
        if y2_tick and y2_tick > 0:
            ymin2, ymax2 = ax2.get_ylim()
            ax2.set_yticks(np.arange(ymin2, ymax2 + 1e-12, y2_tick))

    ax1.axvline(float(t_end), linestyle=":", lw=1.5)

    # Target β line + annotation (red line, white box with shadow, only "Year reached")
    if show_beta_target and (beta_target is not None):
        ax1.axhline(beta_target, color="red", linestyle="--", lw=1.6)  # red target line
        crossing_year = None
        for i in range(len(y_beta) - 1):
            if (y_beta[i] - beta_target) * (y_beta[i+1] - beta_target) <= 0:
                t1, t2 = x_abs[i], x_abs[i+1]
                b1, b2 = y_beta[i], y_beta[i+1]
                if (b2 - b1) != 0:
                    crossing_year = t1 + (beta_target - b1) * (t2 - t1) / (b2 - b1)
                break

        msg = f"Year reached: {crossing_year:.2f} yr" if crossing_year is not None else "Year reached: —"
        txt = ax1.text(
            0.98, 0.98, msg, transform=ax1.transAxes, fontsize=10,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#333333', alpha=0.98)
        )
        # subtle drop shadow
        txt.set_path_effects([pe.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace=(0, 0, 0), alpha=0.18)])
        if crossing_year is not None:
            ax1.axvline(crossing_year, color='red', linestyle=':', lw=1.0, alpha=0.7)

    fig.tight_layout()
    return fig, (x_abs, y_pf)

def plot_pf_only(x_years, y_pf):
    fig_pf, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x_years, y_pf, lw=2)
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Failure probability Pf(t)")
    ax.grid(True)
    fig_pf.tight_layout()
    return fig_pf

# ---------- Helpers ----------
def help_badge(title_key: str, default_text: str = ""):
    if hasattr(st, "popover"):
        with st.popover("?", use_container_width=False):
            st.caption(f"Help – {title_key}")
            st.session_state.setdefault(f"help_text_{title_key}", default_text)
            st.session_state[f"help_text_{title_key}"] = st.text_area(
                "Notes (optional)", value=st.session_state[f"help_text_{title_key}"],
                height=120, label_visibility="collapsed"
            )
            img = st.file_uploader("Upload an image (optional)", type=["png","jpg","jpeg"],
                                   key=f"help_img_{title_key}")
            if img: st.image(img, use_container_width=True)
    else:
        with st.expander("Help"):
            st.caption(f"Help – {title_key}")
            st.session_state.setdefault(f"help_text_{title_key}", default_text)
            st.session_state[f"help_text_{title_key}"] = st.text_area(
                "Notes (optional)", value=st.session_state[f"help_text_{title_key}"],
                height=120, label_visibility="collapsed"
            )
            img = st.file_uploader("Upload an image (optional)", type=["png","jpg","jpeg"],
                                   key=f"help_img_{title_key}")
            if img: st.image(img, use_container_width=True)

def sync_ck_pair(label_left, label_right, key_c, key_k, default_c=None, default_k=None):
    c1, c2 = st.columns(2)
    if key_c not in st.session_state and default_c is not None:
        st.session_state[key_c] = float(default_c)
    if key_k not in st.session_state and default_k is not None:
        st.session_state[key_k] = float(default_k)
    def _on_c_change():
        try:
            c = st.session_state[key_c]
            if c is not None: st.session_state[key_k] = float(c) + 273.15
        except Exception: pass
    def _on_k_change():
        try:
            k = st.session_state[key_k]
            if k is not None: st.session_state[key_c] = float(k) - 273.15
        except Exception: pass
    with c1:
        st.number_input(label_left, key=key_c,
                        value=st.session_state.get(key_c, default_c),
                        on_change=_on_c_change)
    with c2:
        st.number_input(label_right, key=key_k,
                        value=st.session_state.get(key_k, default_k),
                        on_change=_on_k_change)

# ---------- Page layout (same as your latest version above this point) ----------
st.title("fib chloride ingress – reliability index vs time")
left_col, right_col = st.columns([1.1, 1.0], vertical_alignment="top")

# ===== LEFT: Model → α (locks) → t0 → Editable (all single-column) =====
with left_col:
    hdr = st.columns([0.9, 0.1])
    with hdr[0]: st.subheader("Model Parameters")
    with hdr[1]: help_badge("Model Parameters", "Ccrit and b_e are locked.")
    st.markdown("**🔒 Critical Chloride Content (Ccrit) – Locked**")
    Ccrit_mu = st.number_input("Ccrit μ (wt-%/binder)", value=0.60, disabled=True)
    Ccrit_sd = st.number_input("Ccrit σ", value=0.15, disabled=True)
    Ccrit_L  = st.number_input("Ccrit lower bound L", value=0.20, disabled=True)
    Ccrit_U  = st.number_input("Ccrit upper bound U", value=2.00, disabled=True)

    st.markdown("**🔒 Temperature Coefficient (b_e) – Locked**")
    be_mu = st.number_input("Temperature coeff (b_e) μ", value=4800.0, disabled=True)
    be_sd = st.number_input("Temperature coeff (b_e) σ", value=700.0, disabled=True)

    st.divider()

    hdr = st.columns([0.9, 0.1])
    with hdr[0]: st.subheader("Ageing Exponent (α)")
    with hdr[1]: help_badge("Ageing Exponent (α)", "Choose a preset to lock fields; Custom unlocks.")
    alpha_presets = {
        "Please select": None,
        "Portland Cement (PCC)  μ=0.30, σ=0.12": (0.30, 0.12, 0.0, 1.0),
        "PCC + ≥20% Fly Ash     μ=0.60, σ=0.15": (0.60, 0.15, 0.0, 1.0),
        "PCC + BFS               μ=0.45, σ=0.20": (0.45, 0.20, 0.0, 1.0),
        "All types (atmos.)      μ=0.65, σ=0.12": (0.65, 0.12, 0.0, 1.0),
        "Custom – enter values": None,
    }
    alpha_choice = st.selectbox("α preset", list(alpha_presets.keys()), index=0)
    preset_vals = alpha_presets.get(alpha_choice)
    is_locked = preset_vals is not None and alpha_choice != "Please select"
    if is_locked:
        mu_def, sd_def, L_def, U_def = preset_vals
        alpha_mu = st.number_input("α μ", value=float(mu_def), disabled=True)
        alpha_sd = st.number_input("α σ", value=float(sd_def), disabled=True)
        alpha_L  = st.number_input("α lower bound L", value=float(L_def), disabled=True)
        alpha_U  = st.number_input("α upper bound U", value=float(U_def), disabled=True)
    else:
        alpha_mu = st.number_input("α μ", value=0.50)
        alpha_sd = st.number_input("α σ", value=0.15)
        alpha_L  = st.number_input("α lower bound L", value=0.0)
        alpha_U  = st.number_input("α upper bound U", value=1.0)

    st.divider()

    hdr = st.columns([0.9, 0.1])
    with hdr[0]: st.subheader("Reference Age (t0)")
    with hdr[1]: help_badge("Reference Age (t0)", "Pick a common age or enter custom.")
    t0_map = {
        "Please select": None,
        "0.0767 – 28 days": 0.0767,
        "0.1533 – 56 days": 0.1533,
        "0.2464 – 90 days": 0.2464,
    }
    t0_choice = st.selectbox("Reference age t0 (yr)", list(t0_map.keys()), index=0)
    if t0_map.get(t0_choice) is not None:
        t0_year = st.number_input("t0 value (years)", value=float(t0_map[t0_choice]),
                                  disabled=True, format="%.4f")
    else:
        t0_year = st.number_input("t0 value (years)", value=0.0767, format="%.4f")

    st.divider()

    hdr = st.columns([0.9, 0.1])
    with hdr[0]: st.subheader("Editable Parameters")
    with hdr[1]: help_badge("Editable Parameters", "C0, surface chloride, DRCM0, cover.")
    C0       = st.number_input("Initial chloride C0 (wt-%/binder)", value=0.0)
    Cs_mu    = st.number_input("Surface chloride μ (wt-%/binder)", value=1.8)
    Cs_sd    = st.number_input("Surface chloride σ", value=0.3)
    D0_mu    = st.number_input("DRCM0 μ (×1e-12 m²/s)", value=10.0)
    D0_sd    = st.number_input("DRCM0 σ (=0.2×μ)", value=max(0.2*D0_mu, 0.0), disabled=True)
    cover_mu = st.number_input("Cover μ (mm)", value=50.0)
    cover_sd = st.number_input("Cover σ (mm)", value=7.0)

# ===== RIGHT: Δx (always shown) → Temperature (2-col) → Time Window (2-col) → Target β → Axes (2-col rows) =====
with right_col:
    hdr = st.columns([0.9, 0.1])
    with hdr[0]: st.subheader("Convection Zone (Δx)")
    with hdr[1]: help_badge("Convection Zone (Δx)", "Zero shows disabled zeros; submerged locked; tidal editable.")
    dx_display_to_code = {
        "Please select": None,
        "Zero – submerged/spray (Δx = 0)": "zero",
        "Beta – submerged (locked)": "beta_submerged",
        "Beta – tidal (editable)": "beta_tidal",
    }
    dx_choice = st.selectbox("Δx mode", list(dx_display_to_code.keys()), index=0)
    dx_mode_internal = dx_display_to_code.get(dx_choice, None)
    if dx_mode_internal == "zero":
        dx_mu_val, dx_sd_val, dx_L_val, dx_U_val = 0.0, 0.0, 0.0, 0.0
        disabled = True
    elif dx_mode_internal == "beta_submerged":
        dx_mu_val, dx_sd_val, dx_L_val, dx_U_val = 8.9, 5.6, 0.0, 50.0
        disabled = True
    elif dx_mode_internal == "beta_tidal":
        dx_mu_val, dx_sd_val, dx_L_val, dx_U_val = 10.0, 5.0, 0.0, 50.0
        disabled = False
    else:
        dx_mu_val, dx_sd_val, dx_L_val, dx_U_val = 0.0, 0.0, 0.0, 0.0
        disabled = True
    dx_mu = st.number_input("Δx mean μ (mm)", value=float(dx_mu_val), disabled=disabled)
    dx_sd = st.number_input("Δx SD σ (mm)", value=float(dx_sd_val), disabled=disabled)
    dx_L  = st.number_input("Δx lower bound L (mm)", value=float(dx_L_val), disabled=disabled)
    dx_U  = st.number_input("Δx upper bound U (mm)", value=float(dx_U_val), disabled=disabled)

    st.divider()

    hdr = st.columns([0.9, 0.1])
    with hdr[0]: st.subheader("Temperature Parameters")
    with hdr[1]: help_badge("Temperature Parameters", "Two-column C/K inputs with sync; σ same value.")
    sync_ck_pair("Actual Temperature (mean) – °C", "Actual Temperature (mean) – K",
                 key_c="Treal_mu_C", key_k="Treal_mu_K", default_c=23.0, default_k=296.15)
    c1, c2 = st.columns(2)
    with c1: Treal_sd_C = st.number_input("Actual Temperature (std dev) σ (°C)", value=3.0)
    with c2: Treal_sd_K = st.number_input("Actual Temperature (std dev) σ (K)",  value=Treal_sd_C)
    sync_ck_pair("Reference Temperature Tref (°C)", "Reference Temperature Tref (K)",
                 key_c="Tref_C", key_k="Tref_K", default_c=23.0, default_k=296.15)

    st.divider()

    hdr = st.columns([0.9, 0.1])
    with hdr[0]: st.subheader("Time Window & Monte Carlo")
    with hdr[1]: help_badge("Time Window & Monte Carlo", "Left: window; Right: samples/seed.")
    tw1, tw2 = st.columns(2)
    with tw1:
        t_start_disp = st.number_input("Plot start time (yr)", value=0.9)
        t_end        = st.number_input("Plot end time (target yr)", value=50.0, min_value=0.0)
        t_points     = st.number_input("Number of time points", value=200, min_value=10, step=10)
    with tw2:
        N    = st.number_input("Monte Carlo samples N", value=100000, min_value=1000, step=1000)
        seed = st.number_input("Random seed", value=42, step=1)

    st.divider()

    hdr = st.columns([0.9, 0.1])
    with hdr[0]: st.subheader("Target Reliability Index")
    with hdr[1]: help_badge("Target Reliability Index", "Red target line; annotation shows only the year.")
    beta_target      = st.number_input("Target β value (optional)", value=3.80)
    show_beta_target = st.checkbox("Show target β line", value=True)

    st.divider()

    # --- Pf display choice ---
    pf_mode = st.radio(
        "Pf display mode",
        options=["Overlay on β figure", "Separate Pf figure"],
        index=0, horizontal=True
    )

    # --- Plot Axes Controls (two-column rows) ---
    st.subheader("Plot Axes Controls")
    # Row 1
    x_tick = st.number_input("X tick step (years)", value=10.0)
    # Row 2
    y1_tick = st.number_input("Y₁ (β) tick step", value=1.0)
    # Row 3
    r3c1, r3c2 = st.columns(2)
    with r3c1: y1_min = st.number_input("Y₁ (β) min", value=-2.0)
    with r3c2: y1_max = st.number_input("Y₁ (β) max", value=5.0)
    # Row 4
    y2_tick = st.number_input("Y₂ (Pf) tick step", value=0.1)
    # Row 5
    r5c1, r5c2 = st.columns(2)
    with r5c1: y2_min = st.number_input("Y₂ (Pf) min", value=0.0)
    with r5c2: y2_max = st.number_input("Y₂ (Pf) max", value=1.0)

    st.divider()

    # Run + Plot
    run_btn = st.button("Run Simulation", type="primary")
    if run_btn:
        if alpha_choice == "Please select":
            st.error("Please choose an α preset (or use Custom).")
        elif dx_mode_internal is None:
            st.error("Please choose a Δx mode.")
        elif t_end <= t_start_disp:
            st.error("Plot end time must be greater than plot start time.")
        else:
            try:
                params = {
                    "Cs_mu": float(Cs_mu),
                    "Cs_sd": float(Cs_sd),
                    "alpha_mu": float(alpha_mu),
                    "alpha_sd": float(alpha_sd),
                    "alpha_L":  float(alpha_L),
                    "alpha_U":  float(alpha_U),
                    "D0_mu": float(D0_mu),
                    "D0_sd": float(max(0.2*D0_mu, 0.0)),
                    "cover_mu": float(cover_mu),
                    "cover_sd": float(cover_sd),
                    # locked:
                    "Ccrit_mu": float(Ccrit_mu),
                    "Ccrit_sd": float(Ccrit_sd),
                    "Ccrit_L":  float(Ccrit_L),
                    "Ccrit_U":  float(Ccrit_U),
                    "be_mu": float(be_mu),
                    "be_sd": float(be_sd),
                    # temperature in K
                    "Treal_mu": float(st.session_state.get("Treal_mu_K", 296.15)),
                    "Treal_sd": float(Treal_sd_K),
                    "t0": float(t0_year),
                    "Tref": float(st.session_state.get("Tref_K", 296.15)),
                    "C0": float(C0),
                    "dx_mode": dx_mode_internal,
                }
                if dx_mode_internal in ("beta_submerged", "beta_tidal"):
                    params.update({
                        "dx_mu": float(dx_mu), "dx_sd": float(dx_sd),
                        "dx_L":  float(dx_L),  "dx_U":  float(dx_U),
                    })

                df_full = run_fib_chloride(params, N=int(N), seed=int(seed),
                                           t_start=0.0, t_end=float(t_end), t_points=int(t_points))
                df_window = df_full[(df_full["t_years"] >= float(t_start_disp)) &
                                    (df_full["t_years"] <= float(t_end))].copy()
                if df_window.empty:
                    st.error("No points in display window; increase time points or adjust times.")
                else:
                    axes_cfg = {
                        "x_tick":  float(x_tick) if x_tick is not None else None,
                        "y1_min":  float(y1_min) if y1_min is not None else None,
                        "y1_max":  float(y1_max) if y1_max is not None else None,
                        "y1_tick": float(y1_tick) if y1_tick is not None else None,
                        "y2_min":  float(y2_min) if y2_min is not None else None,
                        "y2_max":  float(y2_max) if y2_max is not None else None,
                        "y2_tick": float(y2_tick) if y2_tick is not None else None,
                    }
                    fig, (x_pf, y_pf) = plot_beta(
                        df_window,
                        t_end=float(t_end),
                        axes_cfg=axes_cfg,
                        show_pf=True,  # overlay is controlled by pf_mode inside
                        beta_target=float(beta_target) if beta_target is not None else None,
                        show_beta_target=bool(show_beta_target),
                        pf_mode=("overlay" if pf_mode == "Overlay on β figure" else "separate"),
                    )
                    st.pyplot(fig)

                    if pf_mode == "Separate Pf figure":
                        fig_pf = plot_pf_only(x_pf, y_pf)
                        st.pyplot(fig_pf)

                    st.download_button(
                        "Download window data (CSV)",
                        data=df_window.to_csv(index=False).encode("utf-8"),
                        file_name="fib_output_window.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Invalid input or computation error: {e}")
