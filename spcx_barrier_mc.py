"""
Streamlit app: Monte Carlo Pricing of a Knock-Out Barrier Call (SPCX)
---------------------------------------------------------------------
Prices an up-and-out call option using Monte Carlo simulation under
Geometric Brownian Motion (a Wiener process for log-price):

    dS = mu * S dt + sigma * S dW
    =>  S_{t+dt} = S_t * exp[(mu - 0.5*sigma^2) dt + sigma * sqrt(dt) * Z]

Two knockout conventions are supported:
  1. AT-EXPIRY KO: option pays only if S_T < Barrier at expiry.
     Payoff = max(S_T - K, 0) * 1{S_T < B}
  2. DAILY-MONITORED KO (up-and-out): option dies if the stock touches
     the barrier on ANY day before expiry.
     Payoff = max(S_T - K, 0) * 1{max(S_t) < B for all t}

For the at-expiry case an analytic Black-Scholes cross-check is shown:
     KO call = C(K) - C(B) - (B - K) * DigitalCall(B)

Run with:
    streamlit run spcx_barrier_mc.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="SPCX Barrier Option — Monte Carlo", layout="wide")

TRADING_DAYS = 252


@st.cache_data(ttl=600)
def fetch_live_spot(ticker: str = "SPCX"):
    """Fetch the latest traded price for SPCX (NASDAQ) via yfinance.
    Cached for 10 minutes. Returns None if the fetch fails."""
    try:
        import yfinance as yf
        data = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
        if len(data) == 0:
            return None
        return float(data.iloc[-1])
    except Exception:
        return None


# ============================= Sidebar inputs =============================
# All inputs are number boxes — type any value you want.
st.sidebar.header("Option Parameters")

use_live = st.sidebar.checkbox("Fetch live SPCX spot price", value=True,
                               help="Pulls the latest NASDAQ:SPCX close via yfinance. "
                                    "Untick to type your own spot price.")

live_spot = fetch_live_spot() if use_live else None
if use_live and live_spot is None:
    st.sidebar.error("Couldn't fetch SPCX price — enter the spot manually below.")

S0 = st.sidebar.number_input(
    "Spot price S₀ ($)", min_value=0.01,
    value=round(live_spot, 2) if live_spot else 150.0,
    step=1.0, format="%.2f",
    help="Auto-filled from the live SPCX price when the box above is ticked. "
         "You can still override it by typing.",
)
if live_spot:
    st.sidebar.caption(f"📡 Live SPCX: ${live_spot:,.2f} (delayed; cached 10 min)")
K  = st.sidebar.number_input("Strike K ($)",      min_value=0.01, value=150.0, step=1.0, format="%.2f")
B  = st.sidebar.number_input("Knockout barrier B ($)", min_value=0.01, value=250.0, step=1.0, format="%.2f")

expiry_days = st.sidebar.number_input("Days to expiry (trading days)",
                                      min_value=1, value=100, step=1)

st.sidebar.header("Model Parameters")
sigma = st.sidebar.number_input("Volatility σ (annualised, e.g. 0.80 = 80%)",
                                min_value=0.0001, value=0.80, step=0.05, format="%.4f")
mu    = st.sidebar.number_input("Drift μ (annualised, e.g. 0.01 = 1%)",
                                value=0.01, step=0.01, format="%.4f")
r     = st.sidebar.number_input("Discount rate r (annualised)",
                                value=0.01, step=0.01, format="%.4f",
                                help="Rate used to discount the expected payoff back to today. "
                                     "For risk-neutral pricing set μ = r.")

st.sidebar.header("Simulation Settings")
barrier_mode = st.sidebar.radio(
    "Knockout monitoring",
    ["At expiry only", "Daily (any day breaches barrier)"],
    help="At-expiry: knocked out only if the FINAL price is ≥ B. "
         "Daily: knocked out if the price touches B on ANY simulated day.",
)

n_paths = st.sidebar.number_input("Number of simulated paths",
                                  min_value=100, max_value=50_000_000,
                                  value=500_000, step=100_000,
                                  help="Type any number up to 50 million. "
                                       "Large counts are processed in memory-safe chunks.")
seed = st.sidebar.number_input("Random seed (0 = random each run)", min_value=0, value=42, step=1)

n_chart_paths = st.sidebar.number_input("Sample paths to draw on chart",
                                        min_value=10, max_value=2_000, value=200, step=10)

run = st.sidebar.button("Run Simulation", type="primary")

# ============================= Pricing functions =============================

def bs_call(S, K, r, sigma, T):
    """Plain Black-Scholes call price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_digital_call(S, K, r, sigma, T):
    """Cash-or-nothing digital call paying $1 if S_T > K."""
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(d2)


def analytic_at_expiry_ko(S, K, B, r, sigma, T):
    """Static replication: long C(K), short C(B), short (B-K) digitals at B."""
    return bs_call(S, K, r, sigma, T) - bs_call(S, B, r, sigma, T) \
           - (B - K) * bs_digital_call(S, B, r, sigma, T)


def mc_at_expiry_ko(S0, K, B, mu, sigma, T, r, n_paths, seed,
                    chunk=2_000_000, progress_cb=None):
    """
    At-expiry KO: only the terminal price matters, so we draw S_T directly
    from the closed-form GBM solution (no day-by-day stepping needed):
        S_T = S0 * exp[(mu - 0.5 sigma^2) T + sigma sqrt(T) Z]
    Chunked so 50M paths never blow up memory. Returns (price, std_error,
    knockout_probability, sample_of_payoffs).
    """
    rng = np.random.default_rng(seed if seed != 0 else None)
    disc = np.exp(-r * T)
    drift = (mu - 0.5 * sigma**2) * T
    vol_t = sigma * np.sqrt(T)

    total, total_sq, ko_count, done = 0.0, 0.0, 0, 0
    payoff_sample = None
    while done < n_paths:
        m = min(chunk, n_paths - done)
        z = rng.standard_normal(m)
        ST = S0 * np.exp(drift + vol_t * z)
        payoff = np.where(ST < B, np.maximum(ST - K, 0.0), 0.0)
        total += payoff.sum()
        total_sq += (payoff**2).sum()
        ko_count += int((ST >= B).sum())
        if payoff_sample is None:
            payoff_sample = payoff[:200_000].copy()
        done += m
        if progress_cb:
            progress_cb(done / n_paths)

    mean = total / n_paths
    var = total_sq / n_paths - mean**2
    se = disc * np.sqrt(max(var, 0.0) / n_paths)
    return disc * mean, se, ko_count / n_paths, disc * payoff_sample


def mc_daily_ko(S0, K, B, mu, sigma, T, n_days, r, n_paths, seed,
                chunk=200_000, progress_cb=None):
    """
    Daily-monitored up-and-out: simulate the FULL path day by day and kill
    the option if the price is >= B on any day. Chunked over paths:
    memory per chunk = n_days * chunk * 8 bytes (200k paths x 100 days
    ~ 160MB), so total path count can be large without exhausting RAM.
    """
    rng = np.random.default_rng(seed if seed != 0 else None)
    dt = T / n_days
    disc = np.exp(-r * T)
    drift = (mu - 0.5 * sigma**2) * dt
    vol_dt = sigma * np.sqrt(dt)

    total, total_sq, ko_count, done = 0.0, 0.0, 0, 0
    payoff_sample = None
    while done < n_paths:
        m = min(chunk, n_paths - done)
        z = rng.standard_normal((n_days, m))
        log_paths = np.cumsum(drift + vol_dt * z, axis=0)
        S = S0 * np.exp(log_paths)                 # days 1..n_days
        knocked = (S.max(axis=0) >= B) | (S0 >= B)
        payoff = np.where(~knocked, np.maximum(S[-1] - K, 0.0), 0.0)
        total += payoff.sum()
        total_sq += (payoff**2).sum()
        ko_count += int(knocked.sum())
        if payoff_sample is None:
            payoff_sample = payoff[:200_000].copy()
        done += m
        if progress_cb:
            progress_cb(done / n_paths)

    mean = total / n_paths
    var = total_sq / n_paths - mean**2
    se = disc * np.sqrt(max(var, 0.0) / n_paths)
    return disc * mean, se, ko_count / n_paths, disc * payoff_sample


def simulate_chart_paths(S0, mu, sigma, T, n_days, n_paths, seed):
    """Small full-path simulation purely for visualisation."""
    rng = np.random.default_rng(seed if seed != 0 else None)
    dt = T / n_days
    z = rng.standard_normal((n_days, n_paths))
    log_paths = np.vstack([np.zeros(n_paths),
                           np.cumsum((mu - 0.5 * sigma**2) * dt
                                     + sigma * np.sqrt(dt) * z, axis=0)])
    return S0 * np.exp(log_paths)


# ============================= Main =============================
st.title("🚀 SPCX Knock-Out Barrier Call — Monte Carlo Pricer")
st.markdown(
    f"Pricing a **call, strike ${K:,.0f}, knockout ${B:,.0f}**, "
    f"{expiry_days} trading days to expiry, under GBM with "
    f"σ = {sigma:.0%}, μ = {mu:.1%}."
)

T = expiry_days / TRADING_DAYS

if run or "price" not in st.session_state:
    progress = st.progress(0.0, text=f"Simulating {n_paths:,} paths…")
    cb = lambda f: progress.progress(f, text=f"Simulating {n_paths:,} paths… {f:.0%}")

    if barrier_mode == "At expiry only":
        price, se, ko_prob, payoff_sample = mc_at_expiry_ko(
            S0, K, B, mu, sigma, T, r, int(n_paths), int(seed), progress_cb=cb)
    else:
        price, se, ko_prob, payoff_sample = mc_daily_ko(
            S0, K, B, mu, sigma, T, int(expiry_days), r, int(n_paths), int(seed),
            progress_cb=cb)
    progress.empty()

    chart_paths = simulate_chart_paths(S0, mu, sigma, T, int(expiry_days),
                                       int(n_chart_paths), int(seed) + 1)

    st.session_state.update(
        price=price, se=se, ko_prob=ko_prob, payoff_sample=payoff_sample,
        chart_paths=chart_paths, mode=barrier_mode,
        params=(S0, K, B, mu, sigma, r, T, int(n_paths)),
    )

price         = st.session_state["price"]
se            = st.session_state["se"]
ko_prob       = st.session_state["ko_prob"]
payoff_sample = st.session_state["payoff_sample"]
chart_paths   = st.session_state["chart_paths"]
mode          = st.session_state["mode"]
S0_, K_, B_, mu_, sigma_, r_, T_, n_ = st.session_state["params"]

# ============================= Results =============================
st.subheader("Monte Carlo Price")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Option price", f"${price:,.4f}")
c2.metric("Std. error", f"±${se:,.4f}")
c3.metric("95% CI", f"${price - 1.96*se:,.3f} – ${price + 1.96*se:,.3f}")
c4.metric("Knockout probability", f"{ko_prob:.2%}")

# Analytic cross-check + vanilla comparison
vanilla = bs_call(S0_, K_, r_, sigma_, T_)
st.subheader("Sanity checks")
cc1, cc2, cc3 = st.columns(3)
cc1.metric("Vanilla BS call (no barrier)", f"${vanilla:,.4f}",
           help="What the option would cost WITHOUT the knockout. The KO price must be below this.")
if mode == "At expiry only":
    analytic = analytic_at_expiry_ko(S0_, K_, B_, r_, sigma_, T_)
    cc2.metric("Analytic at-expiry KO (BS)", f"${analytic:,.4f}",
               help="Static replication: C(K) − C(B) − (B−K)·Digital(B), priced with r as drift. "
                    "Matches MC when μ = r.")
    cc3.metric("MC − Analytic", f"${price - analytic:+,.4f}",
               help="Should be within ~2 standard errors of zero when μ = r.")
    if abs(mu_ - r_) > 1e-9:
        st.info("μ ≠ r: the analytic value uses the risk-neutral drift r, so a gap vs MC is expected. "
                "Set μ = r to check convergence.")
else:
    cc2.metric("Analytic at-expiry KO (BS)", f"${analytic_at_expiry_ko(S0_, K_, B_, r_, sigma_, T_):,.4f}",
               help="Upper bound reference — daily monitoring gives MORE chances to knock out, "
                    "so the daily-KO price should be BELOW this.")
    cc3.metric("Discount: barrier haircut", f"{(1 - price / vanilla):.1%}",
               help="How much value the knockout feature removes vs the vanilla call.")

# ============================= Charts =============================
st.subheader("Sample simulated paths")
st.caption(
    "Red paths breach the barrier (knocked out under daily monitoring); "
    "blue paths survive. The dashed lines mark the strike and barrier."
)

days_axis = np.arange(chart_paths.shape[0])
breached = chart_paths.max(axis=0) >= B_

fig = go.Figure()
for i in range(chart_paths.shape[1]):
    color = "rgba(220,20,60,0.25)" if breached[i] else "rgba(70,130,180,0.25)"
    fig.add_trace(go.Scatter(x=days_axis, y=chart_paths[:, i], mode="lines",
                             line=dict(width=0.6, color=color),
                             showlegend=False, hoverinfo="skip"))
fig.add_hline(y=B_, line=dict(color="crimson", dash="dash", width=2),
              annotation_text=f"Barrier ${B_:,.0f}")
fig.add_hline(y=K_, line=dict(color="green", dash="dash", width=2),
              annotation_text=f"Strike ${K_:,.0f}")
fig.add_hline(y=S0_, line=dict(color="black", dash="dot", width=1),
              annotation_text="Spot")
fig.update_layout(xaxis_title="Trading day", yaxis_title="Price ($)",
                  height=520, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Discounted payoff distribution (sample)")
nonzero = payoff_sample[payoff_sample > 0]
zero_frac = 1 - len(nonzero) / len(payoff_sample)
st.caption(
    f"{zero_frac:.1%} of paths pay **zero** (knocked out or finished below strike) — "
    "shown as the note below rather than a giant bar. Histogram shows the non-zero payoffs."
)
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=nonzero, nbinsx=80, marker_color="steelblue"))
fig2.update_layout(xaxis_title="Discounted payoff ($)", yaxis_title="Frequency",
                   height=380, template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# ============================= Explanation =============================
with st.expander("📖 How this works — the maths"):
    st.markdown(r"""
**1. The stock model (GBM / Wiener process)**

$$dS = \mu S\,dt + \sigma S\,dW$$

By Itô's Lemma this has the exact solution
$$S_T = S_0 \exp\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\,Z\right], \quad Z \sim N(0,1)$$

The $-\tfrac{1}{2}\sigma^2$ is the Itô correction: log-returns have a lower
mean than the drift because volatility drags the geometric average down.

**2. The payoff**

- *At-expiry KO*: $\;\text{payoff} = \max(S_T - K, 0)\cdot\mathbf{1}\{S_T < B\}$
  — only the final price matters, so we can draw $S_T$ in one shot per path.
- *Daily KO (up-and-out)*: $\;\text{payoff} = \max(S_T - K, 0)\cdot\mathbf{1}\{\max_t S_t < B\}$
  — we must simulate every day and check the running maximum.

**3. The price**

$$V_0 = e^{-rT}\,\mathbb{E}[\text{payoff}] \approx e^{-rT}\cdot\frac{1}{N}\sum_{i=1}^{N}\text{payoff}_i$$

The Monte Carlo standard error is $\;e^{-rT}\,\hat\sigma_{\text{payoff}}/\sqrt{N}$ —
it shrinks with $\sqrt{N}$, so 4× more paths = half the error.

**4. Analytic cross-check (at-expiry KO only)**

The at-expiry KO call decomposes into vanilla instruments:
$$\text{KO}(K,B) = C(K) - C(B) - (B-K)\cdot\text{DigitalCall}(B)$$

Intuition: own the call struck at $K$; if the stock ends above $B$ you must
give back all of that payoff, which equals a call struck at $B$ plus
$(B-K)$ paid at the digital boundary.

**5. μ vs r — an important subtlety**

Proper *no-arbitrage pricing* is done in the **risk-neutral measure**: the
drift is replaced by the risk-free rate $r$, regardless of the stock's
real-world drift. This task fixes μ = 1% by assumption — if you also set
the discount rate $r$ = 1%, the simulation is exactly a risk-neutral
pricing with $r$ = 1%, and the MC result converges to the analytic
Black-Scholes value above.
""")
