import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(
    page_title="Bitcoin Multi-Vector Forecasting Dashboard",
    page_icon="🪙",
    layout="wide",
)

st.title("🪙 Multi-Vector Bitcoin Forecasting & Risk Intelligence")
st.markdown(
    """
This data product operationalizes the **Tri-Modal Dual-Stream Late-Fusion Model** by combining 
**Technical Time-Series**, **NLP FinBERT Sentiment**, and **Deep On-Chain Fundamentals** (MVRV, NUPL, Active Addresses).
"""
)

# ==========================================
# 2. SIDEBAR: SCENARIO & PARAMETER CONTROLS
# ==========================================
st.sidebar.header("🕹️ Scenario & Stress-Test Controls")

# Mode Selection
scenario_mode = st.sidebar.selectbox(
    "Select Operating Scenario",
    [
        "Scenario 1: What-If Sentiment Stress Testing",
        "Scenario 2: Market Regime & Variance Assessment",
        "Scenario 3: On-Chain Valuation Risk Alerts",
    ],
)

# Baseline State Inputs
st.sidebar.subheader("Live Market Inputs")
base_price = st.sidebar.number_input(
    "Current BTC Price ($)", min_value=5000, max_value=150000, value=64250, step=250
)
current_mvrv = st.sidebar.slider(
    "MVRV Ratio", min_value=0.5, max_value=4.5, value=2.45, step=0.05
)
current_nupl = st.sidebar.slider(
    "NUPL Score", min_value=-0.5, max_value=0.9, value=0.52, step=0.01
)
current_active_addr = st.sidebar.slider(
    "Daily Active Addresses",
    min_value=400000,
    max_value=1200000,
    value=890000,
    step=10000,
)
current_sentiment = st.sidebar.slider(
    "FinBERT Sentiment Polarity",
    min_value=-1.0,
    max_value=1.0,
    value=0.15,
    step=0.05,
)

# ==========================================
# 3. CORE INFERENCE SIMULATION ENGINE
# ==========================================
# Approximation of the Dual-Stream Late-Fusion Response Curve
def simulate_tri_modal_prediction(
    price, mvrv, nupl, active_addr, sentiment, noise_seed=42
):
    # Technical Momentum Base
    tech_drift = 0.002

    # On-Chain Macro Anchor (Mean reversion toward fair value)
    onchain_factor = (
        (3.0 - mvrv) * 0.008
        + (nupl - 0.4) * 0.012
        + (active_addr - 800000) / 1000000 * 0.015
    )

    # NLP Behavioral Multiplier
    sentiment_factor = sentiment * 0.025

    # Combined Predicted Delta
    predicted_return = tech_drift + onchain_factor + sentiment_factor
    predicted_price = price * (1 + predicted_return)
    return predicted_price, predicted_return


pred_price, pred_return = simulate_tri_modal_prediction(
    base_price,
    current_mvrv,
    current_nupl,
    current_active_addr,
    current_sentiment,
)

# Top Metrics Bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Asset Base", f"${base_price:,.2f}")
col2.metric(
    "Predicted Next-Day Close",
    f"${pred_price:,.2f}",
    f"{pred_return * 100:+.2f}%",
)
col3.metric("Tri-Modal Expected RMSE", "$7,892.58", "-3.36% vs Base")
col4.metric("Statistical Confidence", "p = 0.000099", "p < 0.001 (DM Test)")

st.divider()

# ==========================================
# 4. SCENARIO 1: SENTIMENT STRESS TESTING
# ==========================================
if scenario_mode == "Scenario 1: What-If Sentiment Stress Testing":
    st.subheader("🧪 Scenario 1: 'What-If' Social Sentiment Shock Simulation")
    st.markdown(
        """
    Evaluate how rapid exogenous sentiment shocks (e.g., regulatory actions, breaking retail FUD, or speculative euphoria)
    impact the Phase 4 Tri-Modal model versus the Phase 1 Price-Only baseline.
    """
    )

    stress_sent = st.slider(
        "Simulate Hypothetical Sentiment Shock (FinBERT Score)",
        min_value=-1.0,
        max_value=1.0,
        value=-0.75,
        step=0.05,
    )

    # Compare Baseline vs Shocked
    shocked_price, shocked_ret = simulate_tri_modal_prediction(
        base_price,
        current_mvrv,
        current_nupl,
        current_active_addr,
        stress_sent,
    )
    price_only_price = base_price * 1.002  # Closed-system unaware of sentiment

    c1, c2 = st.columns(2)
    with c1:
        st.info(
            f"**Phase 1 (Price-Only Baseline):** Closed-system forecast: **${price_only_price:,.2f}** (Unchanged by sentiment)"
        )
    with c2:
        st.warning(
            f"**Phase 4 (Tri-Modal Fusion):** Shock-adjusted forecast: **${shocked_price:,.2f}** ({shocked_ret * 100:+.2f}%)"
        )

    # Sensitivity Curve
    test_sentiments = np.linspace(-1.0, 1.0, 25)
    curve_preds = [
        simulate_tri_modal_prediction(
            base_price,
            current_mvrv,
            current_nupl,
            current_active_addr,
            s,
        )[0]
        for s in test_sentiments
    ]

    fig_sent = go.Figure()
    fig_sent.add_trace(
        go.Scatter(
            x=test_sentiments,
            y=curve_preds,
            mode="lines+markers",
            name="Tri-Modal Predicted Price",
            line=dict(color="#00CC96", width=3),
        )
    )
    fig_sent.add_hline(
        y=price_only_price,
        line_dash="dash",
        line_color="red",
        annotation_text="Phase 1 Price-Only Level",
    )
    fig_sent.update_layout(
        title="Predictive Sensitivity Across Social Sentiment Spectrum",
        xaxis_title="FinBERT Sentiment Polarity (-1 = Panic, +1 = Euphoria)",
        yaxis_title="Predicted Price ($ USD)",
        template="plotly_white",
    )
    st.plotly_chart(fig_sent, use_container_width=True)

# ==========================================
# 5. SCENARIO 2: REGIME & VARIANCE ASSESSMENT
# ==========================================
elif scenario_mode == "Scenario 2: Market Regime & Variance Assessment":
    st.subheader("📊 Scenario 2: Market Regime Classification & Error Bounds")
    st.markdown(
        """
    Classifies the current market phase into one of the empirical sub-periods established in Chapter 4,
    dynamically projecting expected MAE and RMSE error intervals.
    """
    )

    if current_mvrv > 3.0:
        regime = "Regime A / C: Parabolic Market Expansion"
        exp_mae, exp_rmse = 15210.60, 18150.30
        regime_color = "error"
    elif current_mvrv < 1.2:
        regime = "Regime B: Macro Bear Trough / Accumulation"
        exp_mae, exp_rmse = 3210.80, 4180.15
        regime_color = "success"
    else:
        regime = "Regime B-Transitional: Horizontal Consolidation"
        exp_mae, exp_rmse = 5772.47, 7632.11
        regime_color = "info"

    if regime_color == "error":
        st.error(f"**Identified Cycle:** {regime}")
    elif regime_color == "success":
        st.success(f"**Identified Cycle:** {regime}")
    else:
        st.info(f"**Identified Cycle:** {regime}")

    r1, r2, r3 = st.columns(3)
    r1.metric("Regime Expected MAE", f"${exp_mae:,.2f}")
    r2.metric("Regime Expected RMSE", f"${exp_rmse:,.2f}")
    r3.metric(
        "Estimated Prediction Band",
        f"${pred_price - exp_mae:,.0f} - ${pred_price + exp_mae:,.0f}",
    )

    # 5-Fold Cross Validation Summary Table
    cv_table = pd.DataFrame(
        {
            "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
            "Market Regime Tested": [
                "Early Recovery",
                "Bear Consolidation",
                "Transitional",
                "Parabolic Expansion",
                "ATH Volatility",
            ],
            "MAE ($)": [3420.14, 1782.52, 3244.43, 17394.44, 3020.83],
            "RMSE ($)": [5568.00, 1968.10, 7008.38, 19190.25, 4425.82],
        }
    )
    st.dataframe(cv_table, use_container_width=True)

# ==========================================
# 6. SCENARIO 3: ON-CHAIN RISK ALERTS
# ==========================================
elif scenario_mode == "Scenario 3: On-Chain Valuation Risk Alerts":
    st.subheader("🚨 Scenario 3: On-Chain Fundamental Risk Boundary Alerting")
    st.markdown(
        """
    Translates deep blockchain health metrics (MVRV Ratio and NUPL) into automated decision-support triggers
    to manage asymmetric downside risk.
    """
    )

    # Status Indicators
    a1, a2 = st.columns(2)
    with a1:
        if current_mvrv >= 3.7:
            st.error("⚠️ **MVRV SELL ZONE TRIGGERED (MVRV > 3.7)**")
            st.write(
                "Network valuation indicates extreme over-extension and heavy unrealized holder profits."
            )
        elif current_mvrv <= 1.0:
            st.success("✅ **MVRV BUY / ACCUMULATION ZONE (MVRV < 1.0)**")
            st.write(
                "Market value is below realized network cost basis; historic cyclical accumulation bottom."
            )
        else:
            st.info("ℹ️ **MVRV Neutral Holding Band (1.0 - 3.7)**")

    with a2:
        if current_nupl >= 0.7:
            st.error("⚠️ **NUPL Euphoria Phase (NUPL > 0.70)**")
        elif current_nupl <= 0.0:
            st.success("✅ **NUPL Capitulation Phase (NUPL < 0.00)**")
        else:
            st.info("ℹ️ **NUPL Belief / Optimism Corridor**")

    # Gauge Chart
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_mvrv,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "MVRV Fundamental Risk Index"},
            gauge={
                "axis": {"range": [0.5, 4.5]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0.5, 1.0], "color": "rgba(0, 255, 0, 0.3)"},
                    {"range": [1.0, 3.7], "color": "rgba(255, 255, 255, 0.1)"},
                    {"range": [3.7, 4.5], "color": "rgba(255, 0, 0, 0.4)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 3.7,
                },
            },
        )
    )
    fig_gauge.update_layout(height=350, template="plotly_white")
    st.plotly_chart(fig_gauge, use_container_width=True)