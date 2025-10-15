# app.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erfc
import streamlit as st

st.set_page_config(page_title="fib Chloride Ingress – Reliability", layout="wide")

# ===== Little helper: title + "?" help (optional refs) =====
def title_with_help(title: str, help_md: str | None = None, items: list[dict] | None = None, level="subheader"):
    items = items or []
    n = len(items)
    cols = st.columns([0.74, 0.08] + ([0.18 / n] * n if n else []), vertical_alignment="center") if n \
        else st.columns([0.90, 0.10], vertical_alignment="center")
    with cols[0]:
        getattr(st, level)(title) if level in ("title", "header", "subheader") else st.subheader(title)
    with cols[1]:
        with st.popover("❓", use_container_width=True):
            st.markdown(help_md or "No help text provided.")
    for i, item in enumerate(items, start=2):
        with cols[i]:
            with st.popover(f"🔗 {item.get('label','Reference')}", use_container_width=True):
                src = item.get("image")
                if src:
                    st.image(src, caption=item.get("caption"), use_container_width=True)
                else:
                    st.info("Add image path/URL in code.")

# =============================
# Core math
# =============================
def beta_from_pf(Pf): return -norm.ppf(Pf)

def lognorm_from_mu_sd(rng, n, mu, sd):
    sigma2 = math.log(1 + (sd**2)/(mu**2))
    mu_log = math.log(mu) - 0.5*sigma2
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
            rng, N,
            mu=params["dx_mu"], sd=params["dx_sd"],
            L=params["dx_L"], U=params["dx_U"]
        )
    else:
        raise ValueError("dx_mode must be one of: zero, beta_submerged, beta_tidal")

    Cs    = lognorm_from_mu_sd(rng, N, mu_Cs, sd_Cs)
    alpha = beta_interval_from_mean_sd(rng, N, mu_alpha, sd_alpha, alpha_L, alpha_U)
    Ccrit = beta_interval_from_mean_sd(rng, N, mu_Ccrit, sd_Ccrit, Ccrit_L, Ccrit_U)

    D0 = np.maximum(rng.normal(mu_D0, sd_D0, N), 1e-3) * 1e-12
    cover_m = np.maximum(rng.normal(mu_cover, sd_cover, N), 1.0) / 1000.0
    be = np.maximum(rng.normal(mu_be, sd_be, N), 1.0)
    Treal = np.maximum(rng.normal(mu_Treal, sd_Treal, N), 250.0)

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

    axes_cfg = axes_cfg or {}
    ax1.set_xlim(0, float(t_end))

    x_tick = axes_cfg.get("x_tick")
    if x_tick is not None and x_tick > 0:
        ax1.set_xticks(np.arange(0, float(t_end) + 1e-12, x_tick))

    y1_min = axes_cfg.get("y1_min")
    y1_max = axes_cfg.get("y1_max")
    y1_tick = axes_cfg.get("y1_tick")
    if y1_min is not None and y1_max is not None and y1_max > y1_min:
        ax1.set_ylim(y1_min, y1_max)
    if y1_tick is not None and y1_tick > 0:
        ymin, ymax = ax1.get_ylim()
        ax1.set_yticks(np.arange(ymin, ymax + 1e-12, y1_tick))

    if show_pf and ax2 is not None:
        y2_min = axes_cfg.get("y2_min")
        y2_max = axes_cfg.get("y2_max")
        y2_tick = axes_cfg.get("y2_tick")
        if y2_min is not None and y2_max is not None and y2_max > y2_min:
            ax2.set_ylim(y2_min, y2_max)
        if y2_tick is not None and y2_tick > 0:
            ymin2, ymax2 = ax2.get_ylim()
            ax2.set_yticks(np.arange(ymin2, ymax2 + 1e-12, y2_tick))

    ax1.axvline(float(t_end), linestyle=":", lw=1.5)

    # --- Target β overlay ---
    if show_beta_target and (beta_target is not None):
        ax1.axhline(beta_target, color='red', linestyle='--', lw=1.5, label=f'Target β = {beta_target}')
        crossing_year = None
        for i in range(len(y_beta) - 1):
            if (y_beta[i] >= beta_target and y_beta[i+1] < beta_target) or \
               (y_beta[i] <= beta_target and y_beta[i+1] > beta_target):
                t1, t2 = x_abs[i], x_abs[i+1]
                b1, b2 = y_beta[i], y_beta[i+1]
                crossing_year = t1 + (beta_target - b1) * (t2 - t1) / (b2 - b1)
                break
        text = f'Target β = {beta_target:.2f}'
        if crossing_year is not None:
            text += f'\nYear reached: {crossing_year:.2f} yr'
            ax1.axvline(crossing_year, color='red', linestyle=':', lw=1.0, alpha=0.7)
        else:
            text += '\nNot reached in time range'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.98, 0.98, text, transform=ax1.transAxes, fontsize=10,
                 va='top', ha='right', bbox=props)
        ax1.legend(loc='upper left')

    fig.tight_layout()
    return fig

# =============================
# LAYOUT
# =============================
left, right = st.columns(2)

with left:
    title_with_help(
        "Ageing exponent α preset",
        help_md="Pick a preset or choose Custom to input μ/σ/L/U.",
        items=[{"label": "Reference", "image": "assets/alpha_diagram.png", "caption": "How to choose α"}],
    )

    # === Updated α presets (labels include values) ===
    alpha_presets = {
        "Please select": None,
        "Portland Cement (PCC) 0.30 0.12": (0.30, 0.12, 0.0, 1.0),
        "PCC w/ ≥ 20% Fly Ash 0.60 0.15": (0.60, 0.15, 0.0, 1.0),
        "PCC w/ Blast Furnace Slag 0.45 0.20": (0.45, 0.20, 0.0, 1.0),
        "All types (atmospheric zone) 0.65 0.12": (0.65, 0.12, 0.0, 1.0),
        "Custom – enter values": None,
    }
    alpha_choice = st.selectbox("Ageing exponent α preset", list(alpha_presets.keys()), index=0)

    if alpha_presets[alpha_choice] is None:
        alpha_mu = st.number_input("α mean (μ)", value=0.65, step=0.01)
        alpha_sd = st.number_input("α SD (σ)", value=0.12, step=0.01)
        alpha_L  = st.number_input("α lower bound L", value=0.0, step=0.01)
        alpha_U  = st.number_input("α upper bound U", value=1.0, step=0.01)
    else:
        mu, sd, L, U = alpha_presets[alpha_choice]
        alpha_mu = st.number_input("α mean (μ)", value=float(mu), disabled=True)
        alpha_sd = st.number_input("α SD (σ)", value=float(sd), disabled=True)
        alpha_L  = st.number_input("α lower bound L", value=float(L), disabled=True)
        alpha_U  = st.number_input("α upper bound U", value=float(U), disabled=True)

    # t0
    title_with_help("Reference age t0 (yr)", help_md="28/56/90 days ≈ 0.0767/0.1533/0.2464 yr.")
    t0_options = {
        "Please select": None,
        "0.0767 – 28 days": 0.0767,
        "0.1533 – 56 days": 0.1533,
        "0.2464 – 90 days": 0.2464,
    }
    t0_choice = st.selectbox("Reference age t0", list(t0_options.keys()), index=0)
    t0_value = t0_options[t0_choice]
    st.text_input("", value=("" if t0_value is None else str(t0_value)), disabled=True, label_visibility="collapsed")

    # Ccrit (locked)
    title_with_help("Critical chloride content Ccrit (locked)", help_md="Fixed for this version for comparison.")
    Ccrit_mu = st.number_input("Ccrit μ (wt-%/binder)", value=0.60, disabled=True)
    Ccrit_sd = st.number_input("Ccrit σ", value=0.15, disabled=True)
    Ccrit_L  = st.number_input("Ccrit lower bound L", value=0.20, disabled=True)
    Ccrit_U  = st.number_input("Ccrit upper bound U", value=2.00, disabled=True)

    # be (locked)
    title_with_help("Temperature coefficient b_e (locked)", help_md="Arrhenius-like temp factor.")
    be_mu = st.number_input("b_e μ", value=4800.0, disabled=True)
    be_sd = st.number_input("b_e σ", value=700.0, disabled=True)

    st.divider()
    title_with_help("Editable Parameters", help_md="General input parameters.")

    C0    = st.number_input("Initial chloride C0 (wt-%/binder)", value=0.0, step=0.01)
    Cs_mu = st.number_input("Surface chloride μ (wt-%/binder)", value=3.0, step=0.01)
    Cs_sd = st.number_input("Surface chloride σ", value=0.5, step=0.01)

    D0_mu = st.number_input("DRCM0 μ (×1e-12 m²/s)", value=8.0, step=0.1)
    D0_sd = st.number_input("DRCM0 σ", value=1.5, step=0.1)

    cover_mu = st.number_input("Cover μ (mm)", value=50.0, step=0.5)
    cover_sd = st.number_input("Cover σ (mm)", value=8.0, step=0.5)

    # Temperatures are already in K in Streamlit version (editable)
    Treal_mu = st.number_input("Actual temperature μ (K)", value=302.0, step=0.5)
    Treal_sd = st.number_input("Actual temperature σ (K)", value=2.0, step=0.5)
    Tref     = st.number_input("Reference temperature Tref (K)", value=296.0, step=0.5)

with right:
    # ===== Δx with presets & locking like Tkinter =====
    title_with_help(
        "Convection zone Δx",
        help_md="Choose mode; values auto-fill and lock/unlock accordingly.",
        items=[{"label": "Reference", "image": "assets/dx_modes.png", "caption": "Δx option"}],
    )

    dx_display_to_code = {
        "Please select": None,
        "Zero – submerged/spray (Δx = 0)": "zero",
        "Beta – submerged (locked)": "beta_submerged",
        "Beta – tidal (editable) – please enter": "beta_tidal",
    }
    dx_choice = st.selectbox("Δx mode", list(dx_display_to_code.keys()), index=0, key="dx_choice")
    dx_code = dx_display_to_code[dx_choice]

    # presets from your Tk version
    DX_PRESETS = {
        "zero":           {"dx_mu": 0.0,  "dx_sd": 0.0, "dx_L": 0.0, "dx_U": 0.0},
        "beta_submerged": {"dx_mu": 8.9,  "dx_sd": 5.6, "dx_L": 0.0, "dx_U": 50.0},
        "beta_tidal":     {"dx_mu": 10.0, "dx_sd": 5.0, "dx_L": 0.0, "dx_U": 50.0},
    }

    # Use session_state to auto-fill when mode changes
    if "prev_dx_code" not in st.session_state:
        st.session_state.prev_dx_code = None
    if dx_code != st.session_state.prev_dx_code:
        preset = DX_PRESETS.get(dx_code, None)
        if preset:
            st.session_state.dx_mu = preset["dx_mu"]
            st.session_state.dx_sd = preset["dx_sd"]
            st.session_state.dx_L  = preset["dx_L"]
            st.session_state.dx_U  = preset["dx_U"]
        st.session_state.prev_dx_code = dx_code

    editable_dx = (dx_code == "beta_tidal")
    locked_dx   = (dx_code == "beta_submerged")
    disabled    = (dx_code in (None, "zero", "beta_submerged"))

    dx_mu = st.number_input("Δx Beta mean μ (mm)", value=st.session_state.get("dx_mu", 10.0),
                            step=0.1, disabled=disabled and not editable_dx, key="dx_mu")
    dx_sd = st.number_input("Δx Beta SD σ (mm)", value=st.session_state.get("dx_sd", 5.0),
                            step=0.1, disabled=disabled and not editable_dx, key="dx_sd")
    dx_L  = st.number_input("Δx lower bound L (mm)", value=st.session_state.get("dx_L", 0.0),
                            step=0.1, disabled=disabled and not editable_dx, key="dx_L")
    dx_U  = st.number_input("Δx upper bound U (mm)", value=st.session_state.get("dx_U", 50.0),
                            step=0.1, disabled=disabled and not editable_dx, key="dx_U")

    st.divider()
    title_with_help("Time window & Monte Carlo", help_md="Computation grid & sampling.")
    t_start_disp = st.number_input("Plot start time (yr)", min_value=0.0, value=0.9, step=0.1)
    t_end        = st.number_input("Plot end time (Target yr)", min_value=t_start_disp + 1e-6, value=50.0, step=1.0)
    t_points     = st.number_input("Number of time points", min_value=10, value=200, step=10)
    N            = st.number_input("Monte Carlo samples N", min_value=1000, value=100000, step=1000)
    seed         = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.divider()
    title_with_help("Target Reliability", help_md="Optional overlay of a target β with crossing year.")
    show_beta_target = st.checkbox("Show target β on plot", value=False)
    beta_target = st.number_input("Target β value", value=3.8, step=0.1, disabled=not show_beta_target)

    st.divider()
    title_with_help("Axes controls — tune if needed", help_md="Leave defaults unless plot looks off.")
    x_tick  = st.number_input("X tick step (years)", value=10.0, step=1.0)
    y1_min  = st.number_input("Y₁ = β min", value=-2.0, step=0.5)
    y1_max  = st.number_input("Y₁ = β max", value=5.0, step=0.5)
    y1_tick = st.number_input("Y₁ = β tick step", value=1.0, step=0.1)
    y2_min  = st.number_input("Y₂ = Pf min", value=0.0, step=0.01)
    y2_max  = st.number_input("Y₂ = Pf max", value=1.0, step=0.01)
    y2_tick = st.number_input("Y₂ = Pf tick step", value=0.1, step=0.01)

    show_pf = st.checkbox("Show Pf (failure probability) curve", value=True)

# ===== Run button =====
c1, c2, c3 = st.columns([1,2,1])
with c2:
    run_button = st.button("Run Simulation", type="primary", use_container_width=True)

# =============================
# Compute + Plot
# =============================
if run_button:
    if alpha_presets[alpha_choice] is None:
        pass  # manual α ok
    if t0_value is None:
        st.error("Please select a reference age t0.")
        st.stop()
    if dx_code is None:
        st.error("Please select a Δx mode.")
        st.stop()
    if t_end <= t_start_disp:
        st.error("Plot end time must be greater than plot start time.")
        st.stop()

    try:
        params = {
            "Cs_mu": float(Cs_mu),
            "Cs_sd": float(Cs_sd),

            "alpha_mu": float(alpha_mu),
            "alpha_sd": float(alpha_sd),
            "alpha_L":  float(alpha_L),
            "alpha_U":  float(alpha_U),

            "D0_mu": float(D0_mu),
            "D0_sd": float(D0_sd),

            "cover_mu": float(cover_mu),
            "cover_sd": float(cover_sd),

            "Ccrit_mu": float(Ccrit_mu),
            "Ccrit_sd": float(Ccrit_sd),
            "Ccrit_L":  float(Ccrit_L),
            "Ccrit_U":  float(Ccrit_U),

            "be_mu": float(be_mu),
            "be_sd": float(be_sd),
            "Treal_mu": float(Treal_mu),
            "Treal_sd": float(Treal_sd),

            "t0": float(t0_value),
            "Tref": float(Tref),

            "C0": float(C0),

            "dx_mode": dx_code,
        }
        if dx_code in ("beta_submerged", "beta_tidal"):
            params.update({
                "dx_mu": float(dx_mu),
                "dx_sd": float(dx_sd),
                "dx_L": float(dx_L),
                "dx_U": float(dx_U),
            })

        df_full = run_fib_chloride(
            params,
            N=int(N),
            seed=int(seed),
            t_start=0.0,
            t_end=float(t_end),
            t_points=int(t_points)
        )
        df_window = df_full[(df_full["t_years"] >= float(t_start_disp)) & (df_full["t_years"] <= float(t_end))].copy()
        if df_window.empty:
            st.error("Display window has no points; increase number of time points or adjust times.")
            st.stop()

        axes_cfg = {
            "x_tick":  float(x_tick) if x_tick > 0 else None,
            "y1_min":  float(y1_min),
            "y1_max":  float(y1_max),
            "y1_tick": float(y1_tick) if y1_tick > 0 else None,
            "y2_min":  float(y2_min),
            "y2_max":  float(y2_max),
            "y2_tick": float(y2_tick) if y2_tick > 0 else None,
        }

        fig = plot_beta(
            df_window,
            t_end=float(t_end),
            axes_cfg=axes_cfg,
            show_pf=bool(show_pf),
            beta_target=(float(beta_target) if show_beta_target else None),
            show_beta_target=bool(show_beta_target),
        )
        st.pyplot(fig, clear_figure=True)

        st.markdown("### Download")
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("Download CSV (windowed)", df_window.to_csv(index=False).encode("utf-8"),
                               file_name="fib_output_window.csv", mime="text/csv")
        with col_b:
            st.download_button("Download CSV (full)", df_full.to_csv(index=False).encode("utf-8"),
                               file_name="fib_output_full.csv", mime="text/csv")

        with st.expander("Preview data (window)"):
            st.dataframe(df_window.head(20))

    except Exception as e:
        st.error(f"Invalid input: {e}")
