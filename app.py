# app.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erfc
import streamlit as st

st.set_page_config(page_title="fib Chloride Ingress – Reliability", layout="wide")

# ===== Title =====
st.title("fib Chloride induced corrosion Full probabilistic Model")
st.caption("")

# =============================
# Helper: title with popovers
# =============================
def title_with_popovers(title: str, items: list[dict]):
    """
    Render a section title with small popover buttons to show images inline.

    items: list of {"label": str, "image": str, "caption": Optional[str]}
           - 'image' 
    """
    n = max(1, len(items))
    cols = st.columns([0.78] + [0.22 / n] * n)
    with cols[0]:
        st.subheader(title)
    for i, item in enumerate(items, start=1):
        label = item.get("label", "Diagram")
        src = item.get("image", "")
        caption = item.get("caption", None)
        with cols[i]:
            with st.popover(f"🔗 {label}"):
                if src:
                    st.image(src, caption=caption, use_container_width=True)
                else:
                    st.info("Add an image path/URL in the code (see comments).")

# =============================
# Core math
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

def plot_beta(df_window, t_end, axes_cfg=None, show_pf=True):
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
    fig.tight_layout()
    return fig

# =============================
# LAYOUT: Two columns like Tkinter
# =============================
left, right = st.columns(2)

with left:
    # ---- Ageing exponent section with popovers ----
    title_with_popovers(
        "Ageing exponent α preset",
        items=[
            {"label": "Reference", "image": "assets/alpha_diagram.png", "caption": "How to choose α"},
    ],
    )

    alpha_presets = {
        "Please select": None,
        "Portland Cement (PCC)": (0.30, 0.12, 0.0, 1.0),
        "PCC w/ ≥ 20% Fly Ash": (0.60, 0.15, 0.0, 1.0),
        "PCC w/ Blast Furnace Slag": (0.45, 0.20, 0.0, 1.0),
        "Normally used – All types (atmospheric zone)": (0.65, 0.12, 0.0, 1.0),
    }
    alpha_choice = st.selectbox("Ageing exponent α preset", list(alpha_presets.keys()), index=0)

    if alpha_presets[alpha_choice] is None:
        alpha_mu = st.number_input("Ageing exponent α mean", value=0.65, step=0.01)
        alpha_sd = st.number_input("Ageing exponent α SD", value=0.12, step=0.01)
        alpha_L  = st.number_input("α lower bound L", value=0.0, step=0.01)
        alpha_U  = st.number_input("α upper bound U", value=1.0, step=0.01)
    else:
        mu, sd, L, U = alpha_presets[alpha_choice]
        alpha_mu = st.number_input("Ageing exponent α mean", value=float(mu), disabled=True)
        alpha_sd = st.number_input("Ageing exponent α SD", value=float(sd), disabled=True)
        alpha_L  = st.number_input("α lower bound L", value=float(L), disabled=True)
        alpha_U  = st.number_input("α upper bound U", value=float(U), disabled=True)

    # t0
    st.markdown("**Reference age t0 (yr)**")
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
    Ccrit_mu = st.number_input("Ccrit mean μ", value=0.60, disabled=True)
    Ccrit_sd = st.number_input("Ccrit SD σ", value=0.15, disabled=True)
    Ccrit_L  = st.number_input("Ccrit lower bound L", value=0.20, disabled=True)
    Ccrit_U  = st.number_input("Ccrit upper bound U", value=2.00, disabled=True)

    # be (locked)
    be_mu = st.number_input("Temperature coeff mean (b_e)", value=4800.0, disabled=True)
    be_sd = st.number_input("Temperature coeff SD (b_e)", value=700.0, disabled=True)

    st.divider()
    st.markdown("**Editable Parameters**")
    C0    = st.number_input("Initial chloride C0 (wt-%/binder)", value=0.0, step=0.01)
    Cs_mu = st.number_input("Surface chloride mean (wt-%/binder)", value=3.0, step=0.01)
    Cs_sd = st.number_input("Surface chloride SD", value=0.5, step=0.01)

    D0_mu = st.number_input("DRCM0 mean (×1e-12 m²/s)", value=8.0, step=0.1)
    D0_sd = st.number_input("DRCM0 SD", value=1.5, step=0.1)

    cover_mu = st.number_input("Cover mean (mm)", value=50.0, step=0.5)
    cover_sd = st.number_input("Cover SD (mm)", value=8.0, step=0.5)

    Treal_mu = st.number_input("Actual temperature mean (K)", value=302.0, step=0.5)
    Treal_sd = st.number_input("Actual temperature SD (K)", value=2.0, step=0.5)
    Tref     = st.number_input("Reference temperature Tref (K)", value=296.0, step=0.5)

with right:
    # ---- Δx section with popovers ----
    title_with_popovers(
        "Convection zone Δx",
        items=[
            {"label": "Reference",  "image": "assets/dx_modes.png",  "caption": "Δx option"},
        ],
    )

    dx_display_to_code = {
        "Please select": None,
        "Zero – submerged/spray (Δx = 0)": "zero",
        "Beta – submerged (locked)": "beta_submerged",
        "Beta – tidal (editable) – please enter": "beta_tidal",
    }
    dx_choice = st.selectbox("Δx mode", list(dx_display_to_code.keys()), index=0)
    dx_code = dx_display_to_code[dx_choice]

    editable_dx = (dx_code == "beta_tidal")
    dx_mu = st.number_input("Δx Beta mean μ (mm)", value=8.9, step=0.1, disabled=not editable_dx)
    dx_sd = st.number_input("Δx Beta SD σ (mm)", value=5.6, step=0.1, disabled=not editable_dx)
    dx_L  = st.number_input("Δx lower bound L (mm)", value=0.0, step=0.1, disabled=not editable_dx)
    dx_U  = st.number_input("Δx upper bound U (mm)", value=50.0, step=0.1, disabled=not editable_dx)

    st.divider()
    st.markdown("**Time window & Monte Carlo**")
    t_start_disp = st.number_input("Plot start time (yr)", min_value=0.0, value=0.9, step=0.1)
    t_end        = st.number_input("Plot end time (Target yr)", min_value=t_start_disp + 1e-6, value=50.0, step=1.0)
    t_points     = st.number_input("Number of time points", min_value=10, value=200, step=10)
    N            = st.number_input("Monte Carlo samples N", min_value=1000, value=100000, step=1000)
    seed         = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.divider()
    st.markdown("**Axes controls (leave blank = auto) — RUN FIRST — Adjust only if graph not good**")
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
        st.error("Plot end time T must be greater than plot start time.")
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

        fig = plot_beta(df_window, t_end=float(t_end), axes_cfg=axes_cfg, show_pf=bool(show_pf))
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