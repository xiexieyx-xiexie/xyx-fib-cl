import numpy as np, math, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from scipy.special import erfc

st.set_page_config(page_title="fib Chloride Ingress – Reliability", layout="wide")
st.title("fib chloride ingress – reliability index vs time")

# ----------------------- Core math -----------------------
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

    # Δx mode
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

    # Samples
    Cs = lognorm_from_mu_sd(rng, N, mu_Cs, sd_Cs)
    alpha = beta_interval_from_mean_sd(rng, N, mu_alpha, sd_alpha, alpha_L, alpha_U)
    Ccrit = beta_interval_from_mean_sd(rng, N, mu_Ccrit, sd_Ccrit, Ccrit_L, Ccrit_U)

    D0 = np.maximum(rng.normal(mu_D0, sd_D0, N), 1e-3) * 1e-12
    cover_m = np.maximum(rng.normal(mu_cover, sd_cover, N), 1.0) / 1000.0
    be = np.maximum(rng.normal(mu_be, sd_be, N), 1.0)
    Treal = np.maximum(rng.normal(mu_Treal, sd_Treal, N), 250.0)

    # Constants
    t0_sec = t0_year * 365.25 * 24 * 3600.0
    temp_fac = np.exp(be * (1.0 / Tref - 1.0 / Treal))

    # Time stepping
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
    x_abs = df_window["t_years"].to_numpy()   # absolute time
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

    # X axis: 0 → t_end (absolute)
    ax1.set_xlim(0, float(t_end))
    x_tick = axes_cfg.get("x_tick", None)
    if x_tick is not None and x_tick > 0:
        ax1.set_xticks(np.arange(0, float(t_end) + 1e-12, x_tick))

    # Left Y (β)
    y1_min = axes_cfg.get("y1_min", None)
    y1_max = axes_cfg.get("y1_max", None)
    y1_tick = axes_cfg.get("y1_tick", None)
    if y1_min is not None and y1_max is not None and y1_max > y1_min:
        ax1.set_ylim(y1_min, y1_max)
    if y1_tick is not None and y1_tick > 0:
        ymin, ymax = ax1.get_ylim()
        ax1.set_yticks(np.arange(ymin, ymax + 1e-12, y1_tick))

    # Right Y (Pf)
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

# ----------------------- UI helpers -----------------------
def _parse_or_none(s: str):
    s = s.strip()
    return None if s == "" else float(s)

# α presets with on-change autofill via session_state
ALPHA_PRESETS = {
    "Portland Cement (PCC)  (μ=0.30, σ=0.12, L=0, U=1)": (0.30, 0.12, 0.0, 1.0),
    "PCC w/ ≥ 20% Fly Ash  (μ=0.60, σ=0.15, L=0, U=1)": (0.60, 0.15, 0.0, 1.0),
    "PCC w/ Blast Furnace Slag  (μ=0.45, σ=0.20, L=0, U=1)": (0.45, 0.20, 0.0, 1.0),
    "Normally used – All types (atmospheric)  (μ=0.65, σ=0.15, L=0, U=1)": (0.65, 0.15, 0.0, 1.0),
}
PRESET_NONE = "(none)"

def apply_alpha_preset():
    sel = st.session_state.get("alpha_preset", PRESET_NONE)
    if sel in ALPHA_PRESETS:
        mu, sd, L, U = ALPHA_PRESETS[sel]
        st.session_state["alpha_mu"] = mu
        st.session_state["alpha_sd"] = sd
        st.session_state["alpha_L"]  = L
        st.session_state["alpha_U"]  = U

# Initialize session_state defaults (so on_change can write into existing keys)
for k, v in {
    "alpha_mu": 0.65, "alpha_sd": 0.12, "alpha_L": 0.0, "alpha_U": 1.0,
    "Cs_mu": 1.5, "Cs_sd": 1.1,
    "D0_mu": 15.8, "D0_sd": 3.16,
    "cover_mu": 55.0, "cover_sd": 6.0,
    "be_mu": 4800.0, "be_sd": 700.0,
    "Treal_mu": 288.0, "Treal_sd": 8.0,
    "t0": 0.0767, "Tref": 296.0,
    "Ccrit_mu": 0.60, "Ccrit_sd": 0.15, "Ccrit_L": 0.20, "Ccrit_U": 2.00,
    "dx_mode": "beta", "dx_mu": 5.0, "dx_sd": 2.0, "dx_L": 0.0, "dx_U": 15.0,
    "t_start_disp": 0.1, "t_end": 100.0, "t_points": 200, "N": 30000, "seed": 42,
    "x_tick": 0.0, "show_pf": True,
}.items():
    st.session_state.setdefault(k, v)

# ----------------------- UI (two columns) -----------------------
left, right = st.columns(2)

with left:
    st.subheader("Material & Model Parameters")

    # α preset + fields (Ccrit moved to left too)
    st.selectbox(
        "α preset (auto-fill below)",
        options=[PRESET_NONE] + list(ALPHA_PRESETS.keys()),
        index=0,
        key="alpha_preset",
        on_change=apply_alpha_preset
    )

    alpha_mu = st.number_input("Ageing exponent α mean",
                               value=st.session_state["alpha_mu"], key="alpha_mu")
    alpha_sd = st.number_input("Ageing exponent α SD",
                               value=st.session_state["alpha_sd"], key="alpha_sd")
    alpha_L  = st.number_input("α lower bound L",
                               value=st.session_state["alpha_L"], key="alpha_L")
    alpha_U  = st.number_input("α upper bound U",
                               value=st.session_state["alpha_U"], key="alpha_U")

    Cs_mu    = st.number_input("Surface chloride mean (wt-%/binder)",
                               value=st.session_state["Cs_mu"], key="Cs_mu")
    Cs_sd    = st.number_input("Surface chloride SD",
                               value=st.session_state["Cs_sd"], key="Cs_sd")

    D0_mu    = st.number_input("DRCM0 mean (×1e-12 m²/s)",
                               value=st.session_state["D0_mu"], key="D0_mu")
    D0_sd    = st.number_input("DRCM0 SD",
                               value=st.session_state["D0_sd"], key="D0_sd")

    cover_mu = st.number_input("Cover mean (mm)",
                               value=st.session_state["cover_mu"], key="cover_mu")
    cover_sd = st.number_input("Cover SD (mm)",
                               value=st.session_state["cover_sd"], key="cover_sd")

    be_mu    = st.number_input("Temperature coeff mean",
                               value=st.session_state["be_mu"], key="be_mu")
    be_sd    = st.number_input("Temperature coeff SD",
                               value=st.session_state["be_sd"], key="be_sd")

    Treal_mu = st.number_input("Actual temperature mean (K)",
                               value=st.session_state["Treal_mu"], key="Treal_mu")
    Treal_sd = st.number_input("Actual temperature SD (K)",
                               value=st.session_state["Treal_sd"], key="Treal_sd")

    t0       = st.number_input("Reference age t0 (years)",
                               value=st.session_state["t0"], format="%.5f", key="t0")
    Tref     = st.number_input("Reference temperature Tref (K)",
                               value=st.session_state["Tref"], key="Tref")

    st.markdown("---")
    st.markdown("**Critical chloride content Ccrit ~ Beta[L,U]**")
    Ccrit_mu = st.number_input("Ccrit mean μ (wt-%/binder)",
                               value=st.session_state["Ccrit_mu"], key="Ccrit_mu")
    Ccrit_sd = st.number_input("Ccrit SD σ",
                               value=st.session_state["Ccrit_sd"], key="Ccrit_sd")
    Ccrit_L  = st.number_input("Ccrit lower bound L",
                               value=st.session_state["Ccrit_L"], key="Ccrit_L")
    Ccrit_U  = st.number_input("Ccrit upper bound U",
                               value=st.session_state["Ccrit_U"], key="Ccrit_U")

with right:
    st.subheader("Bounds, Time & Plot Settings")

    st.markdown("**Convection zone Δx**")
    dx_mode = st.selectbox("Δx mode", ["zero","beta"],
                           index=0 if st.session_state["dx_mode"]=="zero" else 1, key="dx_mode")
    if dx_mode == "beta":
        dx_mu = st.number_input("Δx Beta mean μ (mm)",
                                value=st.session_state["dx_mu"], key="dx_mu")
        dx_sd = st.number_input("Δx Beta SD σ (mm)",
                                value=st.session_state["dx_sd"], key="dx_sd")
        dx_L  = st.number_input("Δx lower bound L (mm)",
                                value=st.session_state["dx_L"], key="dx_L")
        dx_U  = st.number_input("Δx upper bound U (mm)",
                                value=st.session_state["dx_U"], key="dx_U")

    st.markdown("---")
    st.markdown("**Time window & Monte Carlo**")
    t_start_disp = st.number_input("Plot start time (years)",
                                   value=st.session_state["t_start_disp"], key="t_start_disp")
    t_end        = st.number_input("Plot end time T (years)",
                                   value=st.session_state["t_end"], key="t_end")
    t_points     = st.number_input("Number of time points",
                                   value=st.session_state["t_points"], step=10, key="t_points")
    N            = st.number_input("Monte Carlo samples N",
                                   value=st.session_state["N"], step=5000, key="N")
    seed         = st.number_input("Random seed",
                                   value=st.session_state["seed"], step=1, key="seed")

    st.markdown("---")
    st.markdown("**Axes controls (leave blank for auto)**")
    x_tick  = st.number_input("X tick step (years)", value=st.session_state["x_tick"],
                              help="0 = auto", key="x_tick")
    y1_min  = st.text_input("Y₁ = β min", "")
    y1_max  = st.text_input("Y₁ = β max", "")
    y1_tick = st.text_input("Y₁ = β tick step", "")
    y2_min  = st.text_input("Y₂ = Pf min", "")
    y2_max  = st.text_input("Y₂ = Pf max", "")
    y2_tick = st.text_input("Y₂ = Pf tick step", "")
    show_pf = st.checkbox("Show Pf (failure probability) curve",
                          value=st.session_state["show_pf"], key="show_pf")

# ----------------------- Run -----------------------
if st.button("Run Simulation", type="primary"):
    try:
        params = {
            "Cs_mu": float(Cs_mu), "Cs_sd": float(Cs_sd),

            "alpha_mu": float(alpha_mu), "alpha_sd": float(alpha_sd),
            "alpha_L":  float(alpha_L),  "alpha_U":  float(alpha_U),

            "D0_mu": float(D0_mu), "D0_sd": float(D0_sd),

            "cover_mu": float(cover_mu), "cover_sd": float(cover_sd),

            "Ccrit_mu": float(Ccrit_mu), "Ccrit_sd": float(Ccrit_sd),
            "Ccrit_L":  float(Ccrit_L),  "Ccrit_U":  float(Ccrit_U),

            "be_mu": float(be_mu), "be_sd": float(be_sd),
            "Treal_mu": float(Treal_mu), "Treal_sd": float(Treal_sd),

            "t0": float(t0), "Tref": float(Tref),

            "C0": 0.05, "dx_mode": dx_mode
        }
        if dx_mode == "beta":
            params.update({
                "dx_mu": float(st.session_state["dx_mu"]),
                "dx_sd": float(st.session_state["dx_sd"]),
                "dx_L":  float(st.session_state["dx_L"]),
                "dx_U":  float(st.session_state["dx_U"]),
            })

        if float(t_end) <= float(t_start_disp):
            st.error("Plot end time T must be greater than plot start time.")
        else:
            df_full = run_fib_chloride(params,
                                       N=int(N), seed=int(seed),
                                       t_start=0.0, t_end=float(t_end),
                                       t_points=int(t_points))
            # Windowed display (keep absolute x)
            df_window = df_full[(df_full["t_years"] >= float(t_start_disp)) &
                                (df_full["t_years"] <= float(t_end))].copy()
            if df_window.empty:
                st.error("Display window has no points; increase number of time points or adjust times.")
            else:
                axes_cfg = {
                    "x_tick":  float(x_tick) if float(x_tick) > 0 else None,
                    "y1_min":  _parse_or_none(y1_min),
                    "y1_max":  _parse_or_none(y1_max),
                    "y1_tick": _parse_or_none(y1_tick),
                    "y2_min":  _parse_or_none(y2_min),
                    "y2_max":  _parse_or_none(y2_max),
                    "y2_tick": _parse_or_none(y2_tick),
                }
                fig = plot_beta(df_window, t_end=float(t_end),
                                axes_cfg=axes_cfg, show_pf=bool(show_pf))
                st.pyplot(fig, clear_figure=True)
                st.download_button("Download CSV (windowed)",
                                   df_window.to_csv(index=False).encode(),
                                   "fib_output.csv", "text/csv")
    except Exception as e:
        st.error(f"Invalid input: {e}")
