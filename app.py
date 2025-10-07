import numpy as np, math, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from scipy.special import erfc

st.set_page_config(page_title="fib Chloride Ingress – Reliability", layout="wide")
st.title("fib chloride ingress – reliability index vs time")

# ----------------------- Core math (from your final code) -----------------------
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

def run_fib_chloride(params, N=30000, seed=42, t_start=0.0, t_end=100.0, t_points=200):
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

    dx_mode = params["dx_mode"]  # "zero" or "beta"
    if dx_mode == "zero":
        dx_mm = np.zeros(N)
    elif dx_mode == "beta":
        dx_mm = beta_interval_from_mean_sd(
            rng, N,
            mu=params["dx_mu"], sd=params["dx_sd"],
            L=params["dx_L"], U=params["dx_U"]
        )
    else:
        raise ValueError("dx_mode must be 'zero' or 'beta'.")

    Cs = lognorm_from_mu_sd(rng, N, mu_Cs, sd_Cs)
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
    ax1.set_xlabel("Time t (years)")
    ax1.set_ylabel("Reliability index β(t)")
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
    fig.tight_layout()
    return fig

# ----------------------- UI (Streamlit, two columns) -----------------------
left, right = st.columns(2)

with left:
    st.subheader("Material & Model Parameters")
    Cs_mu   = st.number_input("Surface chloride mean (wt-%/binder)",  value=1.5)
    Cs_sd   = st.number_input("Surface chloride SD",                  value=1.1)

    alpha_mu= st.number_input("Ageing exponent α mean",               value=0.65)
    alpha_sd= st.number_input("Ageing exponent α SD",                 value=0.12)
    alpha_L = st.number_input("α lower bound L",                      value=0.0)
    alpha_U = st.number_input("α upper bound U",                      value=1.0)

    D0_mu   = st.number_input("DRCM0 mean (×1e-12 m²/s)",            value=15.8)
    D0_sd   = st.number_input("DRCM0 SD",                             value=3.16)

    cover_mu= st.number_input("Cover mean (mm)",                      value=55.0)
    cover_sd= st.number_input("Cover SD (mm)",                        value=6.0)

    be_mu   = st.number_input("Temperature coeff mean",               value=4800.0)
    be_sd   = st.number_input("Temperature coeff SD",                 value=700.0)
    Treal_mu= st.number_input("Actual temperature mean (K)",          value=288.0)
    Treal_sd= st.number_input("Actual temperature SD (K)",            value=8.0)

    t0      = st.number_input("Reference age t0 (years)",             value=0.0767, format="%.5f")
    Tref    = st.number_input("Reference temperature Tref (K)",       value=293.0)

with right:
    st.subheader("Bounds, Time & Plot Settings")

    st.markdown("**Critical chloride content Ccrit ~ Beta[L,U]**")
    Ccrit_mu= st.number_input("Ccrit mean μ (wt-%/binder)",           value=0.60)
    Ccrit_sd= st.number_input("Ccrit SD σ",                           value=0.15)
    Ccrit_L = st.number_input("Ccrit lower bound L",                  value=0.00)
    Ccrit_U = st.number_input("Ccrit upper bound U",                  value=2.00)

    st.markdown("---")
    st.markdown("**Convection zone Δx**")
    dx_mode = st.selectbox("Δx mode", ["zero","beta"], index=1)
    if dx_mode == "beta":
        dx_mu = st.number_input("Δx Beta mean μ (mm)", value=5.0)
        dx_sd = st.number_input("Δx Beta SD σ (mm)",   value=2.0)
        dx_L  = st.number_input("Δx lower bound L (mm)", value=0.0)
        dx_U  = st.number_input("Δx upper bound U (mm)", value=15.0)

    st.markdown("---")
    st.markdown("**Time window & Monte Carlo**")
    t_start_disp = st.number_input("Plot start time (years)", value=0.1)
    t_end        = st.number_input("Plot end time T (years)", value=100.0)
    t_points     = st.number_input("Number of time points",   value=200, step=10)
    N            = st.number_input("Monte Carlo samples N",   value=30000, step=5000)
    seed         = st.number_input("Random seed",             value=42, step=1)

    st.markdown("---")
    st.markdown("**Axes controls (leave blank for auto)**")
    x_tick  = st.number_input("X tick step (years)", value=0.0, help="0 = auto")
    y1_min  = st.text_input("Y₁ = β min", "")
    y1_max  = st.text_input("Y₁ = β max", "")
    y1_tick = st.text_input("Y₁ = β tick step", "")
    y2_min  = st.text_input("Y₂ = Pf min", "")
    y2_max  = st.text_input("Y₂ = Pf max", "")
    y2_tick = st.text_input("Y₂ = Pf tick step", "")
    show_pf = st.checkbox("Show Pf (failure probability) curve", value=True)

# helper to parse axis text inputs
def _parse_or_none(s):
    s = s.strip()
    return None if s == "" else float(s)

if st.button("Run Simulation", type="primary"):
    try:
        params = {
            "Cs_mu": Cs_mu, "Cs_sd": Cs_sd,
            "alpha_mu": alpha_mu, "alpha_sd": alpha_sd, "alpha_L": alpha_L, "alpha_U": alpha_U,
            "D0_mu": D0_mu, "D0_sd": D0_sd,
            "cover_mu": cover_mu, "cover_sd": cover_sd,
            "Ccrit_mu": Ccrit_mu, "Ccrit_sd": Ccrit_sd, "Ccrit_L": Ccrit_L, "Ccrit_U": Ccrit_U,
            "be_mu": be_mu, "be_sd": be_sd, "Treal_mu": Treal_mu, "Treal_sd": Treal_sd,
            "t0": t0, "Tref": Tref,
            "C0": 0.0, "dx_mode": dx_mode
        }
        if dx_mode == "beta":
            params.update({"dx_mu": dx_mu, "dx_sd": dx_sd, "dx_L": dx_L, "dx_U": dx_U})

        if t_end <= t_start_disp:
            st.error("Plot end time T must be greater than plot start time.")
        else:
            df_full = run_fib_chloride(params, N=int(N), seed=int(seed),
                                       t_start=0.0, t_end=float(t_end), t_points=int(t_points))
            df_window = df_full[(df_full["t_years"] >= float(t_start_disp)) & (df_full["t_years"] <= float(t_end))].copy()
            if df_window.empty:
                st.error("Display window has no points; increase number of time points or adjust times.")
            else:
                axes_cfg = {
                    "x_tick":  x_tick if x_tick > 0 else None,
                    "y1_min":  _parse_or_none(y1_min),
                    "y1_max":  _parse_or_none(y1_max),
                    "y1_tick": _parse_or_none(y1_tick),
                    "y2_min":  _parse_or_none(y2_min),
                    "y2_max":  _parse_or_none(y2_max),
                    "y2_tick": _parse_or_none(y2_tick),
                }
                fig = plot_beta(df_window, t_end=float(t_end), axes_cfg=axes_cfg, show_pf=show_pf)
                st.pyplot(fig, clear_figure=True)
                st.download_button("Download CSV (windowed)", df_window.to_csv(index=False).encode(), "fib_output.csv", "text/csv")
    except Exception as e:
        st.error(f"Invalid input: {e}")
