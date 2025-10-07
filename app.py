import numpy as np, math, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from scipy.special import erfc

st.set_page_config(page_title="fib Chloride Reliability", layout="wide")
st.title("fib Chloride Ingress – Reliability App")

left, right = st.columns(2)

with left:
    st.subheader("Chloride / Transport")
    Cs_mu   = st.number_input("Surface chloride mean Cs (wt-%/binder)",  value=2.0)
    Cs_sd   = st.number_input("Surface chloride SD",                     value=0.5)
    alpha_mu= st.number_input("Ageing exponent mean α",                  value=0.45)
    alpha_sd= st.number_input("Ageing exponent SD",                      value=0.2)
    D0_mu   = st.number_input("DRCM0 mean (×1e-12 m²/s)",               value=1.9)
    D0_sd   = st.number_input("DRCM0 SD",                                value=0.38)
    cover_mu= st.number_input("Cover mean c_nom (mm)",                   value=45.0)
    cover_sd= st.number_input("Cover SD (mm)",                           value=3.0)
    Ccrit_mu= st.number_input("Critical chloride mean Ccrit", value=0.60)
    Ccrit_sd= st.number_input("Critical chloride SD",   value=0.15)

with right:
    st.subheader("Temperature / Time")
    be_mu   = st.number_input("Temperature coeff b_e mean",              value=4800.0)
    be_sd   = st.number_input("Temperature coeff b_e SD",                value=700.0)
    Treal_mu= st.number_input("Actual temperature mean T_real (K)",      value=288.0)
    Treal_sd= st.number_input("Actual temperature SD (K)",               value=5.0)
    t0      = st.number_input("Reference age t₀ (years)",                value=0.0767)
    Tref    = st.number_input("Reference temperature T_ref (K)",         value=293.0)
    t_start = st.number_input("Plot start time (years)",                 value=4.0)
    t_end   = st.number_input("Plot end time (years)",                   value=50.0)
    t_points= st.number_input("Number of time points",                   value=100)
    N       = st.number_input("Monte Carlo samples N",                   value=50000)
    seed    = st.number_input("Random seed",                             value=44)
    dx_mode = st.selectbox("Convection zone Δx mode", ["zero","beta"])
    if dx_mode == "beta":
        dx_mu = st.number_input("Δx Beta mean μ (mm)", value=5.0)
        dx_sd = st.number_input("Δx Beta sd σ (mm)",   value=2.0)
        dx_L  = st.number_input("Δx lower bound L (mm)", value=0.0)
        dx_U  = st.number_input("Δx upper bound U (mm)", value=15.0)

def lognorm_from_mu_sd(rng, n, mu, sd):
    sigma2 = math.log(1+(sd**2)/(mu**2))
    mu_log = math.log(mu) - 0.5*sigma2
    sigma = math.sqrt(sigma2)
    return rng.lognormal(mu_log, sigma, n)

def beta01_shapes_from_mean_sd(mu, sd):
    mu = max(min(mu, 1-1e-12), 1e-12)
    var = max(sd**2, 1e-12)
    t = mu*(1-mu)/var - 1
    return max(mu*t,1e-6), max((1-mu)*t,1e-6)

def beta_interval_from_mean_sd(rng, n, mu, sd, L, U):
    mu01 = (mu-L)/(U-L); sd01 = sd/(U-L)
    a,b = beta01_shapes_from_mean_sd(mu01, sd01)
    return L + (U-L)*rng.beta(a,b,n)

def simulate():
    rng = np.random.default_rng(int(seed))
    t = np.linspace(float(t_start), float(t_end), int(t_points))
    dx_mm = np.zeros(int(N)) if dx_mode=="zero" else beta_interval_from_mean_sd(rng, int(N), dx_mu, dx_sd, dx_L, dx_U)
    Cs = lognorm_from_mu_sd(rng, int(N), Cs_mu, Cs_sd)
    aA,bA = beta01_shapes_from_mean_sd(alpha_mu, alpha_sd); alpha = rng.beta(aA,bA,int(N))
    aC,bC = beta01_shapes_from_mean_sd(Ccrit_mu, Ccrit_sd); Ccrit = rng.beta(aC,bC,int(N))
    D0 = np.maximum(rng.normal(D0_mu, D0_sd, int(N)),1e-6)*1e-12
    cover_m = np.maximum(rng.normal(cover_mu, cover_sd, int(N)),1)/1000.0
    be = np.maximum(rng.normal(be_mu, be_sd, int(N)),1.0)
    Treal = np.maximum(rng.normal(Treal_mu, Treal_sd, int(N)),250.0)
    t0_sec = float(t0)*365.25*24*3600.0
    temp_fac = np.exp(be*(1.0/float(Tref) - 1.0/Treal))
    Pf=[]
    for ti in t:
        ti_sec = ti*365.25*24*3600.0
        Dapp = temp_fac * D0 * (t0_sec/ti_sec)**alpha
        arg = (cover_m - dx_mm/1000.0) / (2*np.sqrt(Dapp*ti_sec))
        C_at = (Cs) * erfc(arg)
        Pf.append(np.mean(C_at >= Ccrit))
    Pf = np.clip(np.array(Pf), 1e-12, 1-1e-12)
    beta = -norm.ppf(Pf)
    return t, Pf, beta

if st.button("Run Simulation", type="primary"):
    with st.spinner("Running Monte Carlo..."):
        t, Pf, beta = simulate()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Reliability index β(t)")
        fig, ax = plt.subplots()
        ax.plot(t, beta); ax.set_xlabel("t (years)"); ax.set_ylabel("β(t)"); ax.grid(True)
        st.pyplot(fig)
    with c2:
        st.subheader("Failure probability Pf(t)")
        fig2, ax2 = plt.subplots()
        ax2.plot(t, Pf, "--"); ax2.set_xlabel("t (years)"); ax2.set_ylabel("Pf(t)"); ax2.grid(True)
        st.pyplot(fig2)
