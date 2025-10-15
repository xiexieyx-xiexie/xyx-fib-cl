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
# Streamlit App
# =============================

def main():
    st.set_page_config(page_title="fib Chloride Model", layout="wide", initial_sidebar_state="expanded")
    
    st.title("🔬 fib Chloride Ingress Model - Reliability Analysis")
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📋 Model Parameters")
        
        # Locked parameters (Ccrit)
        with st.expander("🔒 Critical Chloride Content (Ccrit) - Locked", expanded=False):
            Ccrit_mu = st.number_input("Ccrit μ (wt-%/binder)", value=0.60, disabled=True)
            Ccrit_sd = st.number_input("Ccrit σ", value=0.15, disabled=True)
            Ccrit_L = st.number_input("Ccrit lower bound L", value=0.20, disabled=True)
            Ccrit_U = st.number_input("Ccrit upper bound U", value=2.00, disabled=True)
        
        # Temperature coefficient (locked)
        with st.expander("🔒 Temperature Coefficient (be) - Locked", expanded=False):
            be_mu = st.number_input("Temperature coeff (b_e) μ", value=4800.0, disabled=True)
            be_sd = st.number_input("Temperature coeff (b_e) σ", value=700.0, disabled=True)
        
        # Alpha preset
        st.subheader("⚙️ Ageing Exponent (α)")
        alpha_presets = {
            "Portland Cement (PCC) 0.30 0.12": (0.30, 0.12, 0.0, 1.0),
            "PCC w/ ≥ 20% Fly Ash 0.60 0.15": (0.60, 0.15, 0.0, 1.0),
            "PCC w/ Blast Furnace Slag 0.45 0.20": (0.45, 0.20, 0.0, 1.0),
            "All types (atmospheric zone) 0.65 0.12": (0.65, 0.12, 0.0, 1.0),
            "Custom – enter values": None,
        }
        alpha_choice = st.selectbox("α preset", list(alpha_presets.keys()))
        
        if alpha_presets[alpha_choice] is not None:
            alpha_mu, alpha_sd, alpha_L, alpha_U = alpha_presets[alpha_choice]
            st.text(f"α μ = {alpha_mu}")
            st.text(f"α σ = {alpha_sd}")
            st.text(f"α bounds: [{alpha_L}, {alpha_U}]")
        else:
            alpha_mu = st.number_input("α μ", value=0.30, min_value=0.0, max_value=1.0)
            alpha_sd = st.number_input("α σ", value=0.12, min_value=0.0)
            alpha_L = st.number_input("α lower bound L", value=0.0, min_value=0.0)
            alpha_U = st.number_input("α upper bound U", value=1.0, min_value=0.0)
        
        # Reference age
        st.subheader("📅 Reference Age (t0)")
        t0_options = {
            "0.0767 – 28 days": 0.0767,
            "0.1533 – 56 days": 0.1533,
            "0.2464 – 90 days": 0.2464,
        }
        t0_choice = st.selectbox("Reference age t0 (yr)", list(t0_options.keys()))
        t0 = t0_options[t0_choice]
        st.text(f"t0 = {t0} years")
        
        # Editable parameters
        st.subheader("✏️ Editable Parameters")
        
        C0 = st.number_input("Initial chloride C0 (wt-%/binder)", value=0.0, min_value=0.0)
        Cs_mu = st.number_input("Surface chloride μ (wt-%/binder)", value=3.5, min_value=0.0)
        Cs_sd = st.number_input("Surface chloride σ", value=1.0, min_value=0.0)
        
        D0_mu = st.number_input("DRCM0 μ (×1e-12 m²/s)", value=10.0, min_value=0.0)
        D0_sd = D0_mu * 0.2
        st.text(f"DRCM0 σ = {D0_sd:.2f} (auto: 0.2×μ)")
        
        cover_mu = st.number_input("Cover μ (mm)", value=50.0, min_value=0.0)
        cover_sd = st.number_input("Cover σ (mm)", value=10.0, min_value=0.0)
        
        Treal_C = st.number_input("Actual temperature μ (°C)", value=20.0)
        Treal_mu = Treal_C + 273
        st.text(f"Actual temperature = {Treal_mu:.2f} K")
        
        Treal_sd_C = st.number_input("Actual temperature σ (°C)", value=5.0, min_value=0.0)
        Treal_sd = Treal_sd_C
        st.text(f"Actual temperature σ = {Treal_sd:.2f} K")
        
        Tref_C = st.number_input("Reference temperature (°C)", value=23.0)
        Tref = Tref_C + 273
        st.text(f"Reference temperature = {Tref:.2f} K")
    
    with col2:
        st.header("⚙️ Simulation Settings")
        
        # Convection zone
        st.subheader("🌊 Convection Zone (Δx)")
        dx_options = {
            "Zero – submerged/spray (Δx = 0)": "zero",
            "Beta – submerged (locked)": "beta_submerged",
            "Beta – tidal (editable)": "beta_tidal",
        }
        dx_choice = st.selectbox("Δx mode", list(dx_options.keys()))
        dx_mode = dx_options[dx_choice]
        
        if dx_mode == "zero":
            dx_mu, dx_sd, dx_L, dx_U = 0, 0, 0, 0
            st.info("Δx = 0 for submerged/spray conditions")
        elif dx_mode == "beta_submerged":
            dx_mu, dx_sd, dx_L, dx_U = 8.9, 5.6, 0, 50
            st.text(f"Δx Beta: μ={dx_mu}, σ={dx_sd}, bounds=[{dx_L}, {dx_U}] mm")
        else:  # beta_tidal
            dx_mu = st.number_input("Δx Beta mean μ (mm)", value=10.0, min_value=0.0)
            dx_sd = st.number_input("Δx Beta SD σ (mm)", value=5.0, min_value=0.0)
            dx_L = st.number_input("Δx lower bound L (mm)", value=0.0, min_value=0.0)
            dx_U = st.number_input("Δx upper bound U (mm)", value=50.0, min_value=0.0)
        
        # Time window & Monte Carlo
        st.subheader("⏱️ Time Window & Monte Carlo")
        t_start = st.number_input("Plot start time (yr)", value=0.9, min_value=0.0)
        t_end = st.number_input("Plot end time (Target yr)", value=50.0, min_value=0.1)
        t_points = st.number_input("Number of time points", value=200, min_value=10)
        N = st.number_input("Monte Carlo samples N", value=100000, min_value=1000)
        seed = st.number_input("Random seed", value=42, min_value=0)
        
        # Target reliability
        st.subheader("🎯 Target Reliability Index")
        beta_target = st.number_input("Target β value", value=1.5, min_value=0.0)
        show_beta_target = st.checkbox("Show target β on plot", value=True)
        
        # Axes controls
        st.subheader("📊 Plot Axes Controls")
        st.info("Leave blank for auto-scaling. Run simulation first, then adjust if needed.")
        
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
            with st.spinner("Running simulation... This may take a moment."):
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