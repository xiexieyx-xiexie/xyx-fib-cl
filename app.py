# app.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erfc
import streamlit as st

st.set_page_config(page_title="fib Chloride Ingress – Reliability", layout="wide")

# =============================================================
# Helper: section title + "?" help + optional reference buttons
# =============================================================
def title_with_help(
    title: str,
    help_md: str | None = None,
    items: list[dict] | None = None,
    level: str = "subheader",
):
    """Render title with a right-side '?' help popover (or expander fallback)."""
    items = items or []
    n = len(items)
    col_widths = [0.90, 0.10] if n == 0 else [0.74, 0.08] + [0.18 / n] * n
    cols = st.columns(col_widths)

    # Title
    with cols[0]:
        if level == "title":
            st.title(title)
        elif level == "header":
            st.header(title)
        else:
            st.subheader(title)

    # Help popover (fallback safe)
    with cols[1]:
        if hasattr(st, "popover"):
            with st.popover("❓"):
                st.markdown(help_md or "No help text provided.")
        else:
            with st.expander("❓ Help"):
                st.markdown(help_md or "No help text provided.")

    # Optional reference image buttons
    for i, item in enumerate(items, start=2 if n > 0 else 2):
        label = item.get("label", "Reference")
        src = item.get("image", "")
        caption = item.get("caption", None)
        with cols[i]:
            if hasattr(st, "popover"):
                with st.popover(f"🔗 {label}"):
                    if src:
                        st.image(src, caption=caption, use_container_width=True)
                    else:
                        st.info("Add an image path/URL.")
            else:
                with st.expander(f"🔗 {label}"):
                    if src:
                        st.image(src, caption=caption, use_container_width=True)
                    else:
                        st.info("Add an image path/URL.")

# =============================================================
# Core math
# =============================================================
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

# =============================================================
# Plot
# =============================================================
def plot_beta(df_window, t_end, axes_cfg=None, show_pf=True, beta_target=None, show_beta_target=False):
    x_abs = df_window["t_years"].to_numpy()
    y_beta = df_window["beta"].to_numpy()
    y_pf   = df_window["Pf"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x_abs, y_beta, lw=2, label="β(t)")
    ax1.set_xlabel("Time (yr)")
    ax1.set_ylabel("Reliability index β(-)")
    ax1.grid(True)

    if show_pf:
        ax2 = ax1.twinx()
        ax2.plot(x_abs, y_pf, "--", lw=1.5, label="Pf(t)")
        ax2.set_ylabel("Failure probability Pf(t)")

    ax1.set_xlim(0, float(t_end))
    if axes_cfg:
        xt = axes_cfg.get("x_tick")
        if xt: ax1.set_xticks(np.arange(0, t_end + 1e-12, xt))
        y1min, y1max = axes_cfg.get("y1_min"), axes_cfg.get("y1_max")
        if y1min is not None and y1max is not None: ax1.set_ylim(y1min, y1max)
    ax1.axvline(float(t_end), ":", lw=1.5)

    # Target β overlay
    if show_beta_target and beta_target is not None:
        ax1.axhline(beta_target, color='r', ls='--', lw=1.5, label=f'Target β={beta_target}')
        crossing = None
        for i in range(len(y_beta) - 1):
            if (y_beta[i] >= beta_target > y_beta[i+1]) or (y_beta[i] <= beta_target < y_beta[i+1]):
                t1, t2 = x_abs[i], x_abs[i+1]
                b1, b2 = y_beta[i], y_beta[i+1]
                crossing = t1 + (beta_target - b1)*(t2 - t1)/(b2 - b1)
                break
        txt = f'Target β={beta_target:.2f}'
        if crossing:
            txt += f'\nReached: {crossing:.2f} yr'
            ax1.axvline(crossing, color='r', ls=':', lw=1)
        else:
            txt += '\nNot reached'
        ax1.text(0.98, 0.98, txt, transform=ax1.transAxes, ha='right', va='top',
                 fontsize=10, bbox=dict(facecolor='wheat', alpha=0.8))
        ax1.legend(loc='upper left')
    fig.tight_layout()
    return fig

# =============================================================
# LAYOUT
# =============================================================
left, right = st.columns(2)

with left:
    title_with_help("Ageing exponent α preset", "Pick a preset or choose Custom to input μ/σ/L/U.",
        items=[{"label": "Reference", "image": "assets/alpha_diagram.png"}])
    alpha_presets = {
        "Please select": None,
        "Portland Cement (PCC) 0.30 0.12": (0.30, 0.12, 0.0, 1.0),
        "PCC w/ ≥ 20% Fly Ash 0.60 0.15": (0.60, 0.15, 0.0, 1.0),
        "PCC w/ Blast Furnace Slag 0.45 0.20": (0.45, 0.20, 0.0, 1.0),
        "All types (atmospheric zone) 0.65 0.12": (0.65, 0.12, 0.0, 1.0),
        "Custom – enter values": None,
    }
    α_choice = st.selectbox("Ageing exponent preset", list(alpha_presets.keys()))
    if alpha_presets[α_choice]:
        μ, σ, L, U = alpha_presets[α_choice]
        αμ = st.number_input("α mean μ", value=μ, disabled=True)
        ασ = st.number_input("α SD σ", value=σ, disabled=True)
        αL = st.number_input("α lower bound", value=L, disabled=True)
        αU = st.number_input("α upper bound", value=U, disabled=True)
    else:
        αμ = st.number_input("α mean μ", value=0.65)
        ασ = st.number_input("α SD σ", value=0.12)
        αL = st.number_input("α lower bound", value=0.0)
        αU = st.number_input("α upper bound", value=1.0)

    title_with_help("Reference age t0", "Typical values: 28/56/90 days ≈ 0.0767/0.1533/0.2464 yr.")
    t0_opts = {"Please select": None, "0.0767 – 28 days": 0.0767, "0.1533 – 56 days": 0.1533, "0.2464 – 90 days": 0.2464}
    t0_choice = st.selectbox("t0", list(t0_opts.keys()))
    t0_val = t0_opts[t0_choice]
    st.text_input("", value=("" if t0_val is None else str(t0_val)), disabled=True, label_visibility="collapsed")

    title_with_help("Critical chloride content Ccrit (locked)", "Fixed for comparison.")
    Ccrit_mu = st.number_input("Ccrit μ", 0.6, disabled=True)
    Ccrit_sd = st.number_input("Ccrit σ", 0.15, disabled=True)
    Ccrit_L = st.number_input("Ccrit L", 0.2, disabled=True)
    Ccrit_U = st.number_input("Ccrit U", 2.0, disabled=True)
    be_mu = st.number_input("b_e μ", 4800.0, disabled=True)
    be_sd = st.number_input("b_e σ", 700.0, disabled=True)

    st.divider()
    title_with_help("Editable Parameters", "General inputs for model.")
    C0 = st.number_input("Initial chloride C0", 0.0)
    Cs_mu = st.number_input("Surface chloride μ", 3.0)
    Cs_sd = st.number_input("Surface chloride σ", 0.5)
    D0_mu = st.number_input("DRCM0 μ", 8.0)
    D0_sd = st.number_input("DRCM0 σ", 1.6)
    cover_mu = st.number_input("Cover μ (mm)", 50.0)
    cover_sd = st.number_input("Cover σ (mm)", 8.0)
    Treal_mu = st.number_input("Actual T μ (K)", 302.0)
    Treal_sd = st.number_input("Actual T σ (K)", 2.0)
    Tref = st.number_input("Ref T (K)", 296.0)

with right:
    title_with_help("Convection zone Δx", "Mode auto-fills and locks inputs.",
        items=[{"label": "Reference", "image": "assets/dx_modes.png"}])
    dx_modes = {"Please select": None, "Zero – submerged/spray (Δx=0)": "zero",
                "Beta – submerged (locked)": "beta_submerged",
                "Beta – tidal (editable)": "beta_tidal"}
    dx_choice = st.selectbox("Δx mode", list(dx_modes.keys()))
    dx_code = dx_modes[dx_choice]
    DX_PRESETS = {"zero": (0,0,0,0), "beta_submerged": (8.9,5.6,0,50), "beta_tidal": (10,5,0,50)}
    preset = DX_PRESETS.get(dx_code, (0,0,0,0))
    editable = dx_code == "beta_tidal"
    dx_mu = st.number_input("Δx μ (mm)", preset[0], disabled=not editable)
    dx_sd = st.number_input("Δx σ (mm)", preset[1], disabled=not editable)
    dx_L  = st.number_input("Δx L (mm)", preset[2], disabled=not editable)
    dx_U  = st.number_input("Δx U (mm)", preset[3], disabled=not editable)

    st.divider()
    title_with_help("Monte Carlo & Time Window", "Computation grid.")
    t_start = st.number_input("Plot start (yr)", 0.9)
    t_end = st.number_input("Plot end (yr)", 50.0)
    t_pts = st.number_input("Time points", 200)
    N = st.number_input("Samples N", 100000)
    seed = st.number_input("Seed", 42)

    st.divider()
    title_with_help("Target Reliability β", "Optional overlay of β target.")
    show_target = st.checkbox("Show target β", value=False)
    β_target = st.number_input("Target β", 3.8, disabled=not show_target)

    st.divider()
    x_tick = st.number_input("X tick (yr)", 10.0)
    y1_min = st.number_input("β min", -2.0)
    y1_max = st.number_input("β max", 5.0)
    y1_tick = st.number_input("β tick", 1.0)
    show_pf = st.checkbox("Show Pf curve", True)

# =============================================================
# Run simulation
# =============================================================
if st.button("Run Simulation", type="primary", use_container_width=True):
    if dx_code is None or t0_val is None:
        st.error("Please complete all selections."); st.stop()

    p = dict(Cs_mu=Cs_mu, Cs_sd=Cs_sd, alpha_mu=αμ, alpha_sd=ασ, alpha_L=αL, alpha_U=αU,
             D0_mu=D0_mu, D0_sd=D0_sd, cover_mu=cover_mu, cover_sd=cover_sd,
             Ccrit_mu=Ccrit_mu, Ccrit_sd=Ccrit_sd, Ccrit_L=Ccrit_L, Ccrit_U=Ccrit_U,
             be_mu=be_mu, be_sd=be_sd, Treal_mu=Treal_mu, Treal_sd=Treal_sd,
             t0=t0_val, Tref=Tref, C0=C0, dx_mode=dx_code)
    if dx_code in ("beta_submerged", "beta_tidal"):
        p.update(dict(dx_mu=dx_mu, dx_sd=dx_sd, dx_L=dx_L, dx_U=dx_U))

    df = run_fib_chloride(p, N=int(N), seed=int(seed), t_start=0.0, t_end=t_end
