🪙 Multi-Vector Bitcoin Forecasting & Risk Intelligence Dashboard

An interactive Streamlit application that operationalizes the Tri-Modal Dual-Stream Late-Fusion Model for Bitcoin price forecasting and risk intelligence. The dashboard integrates Technical Time-Series, NLP FinBERT Sentiment, and On-Chain Fundamentals (MVRV, NUPL, Daily Active Addresses) into real-time scenario simulations.

✨ Features

Real-Time Multi-Vector Inference: Dynamically calculates predicted next-day BTC closing prices and expected returns based on live market inputs and behavioral multipliers.

Scenario 1 — What-If Sentiment Stress Testing: Simulates exogenous social sentiment shocks (e.g., regulatory actions, retail panic, or market euphoria) and compares the Tri-Modal fusion forecast against a closed-system Price-Only baseline.

Scenario 2 — Market Regime & Variance Assessment: Classifies current market conditions into empirical cycle regimes (Parabolic Expansion, Bear Trough / Accumulation, or Horizontal Consolidation), projecting expected MAE/RMSE error intervals alongside a 5-fold cross-validation summary.

Scenario 3 — On-Chain Valuation Risk Alerts: Evaluates blockchain health metrics (MVRV Ratio and NUPL) against automated buy/sell risk boundaries with an interactive risk gauge.

🛠️ Tech Stack

Frontend & App Framework: Streamlit

Data Manipulation & Math: NumPy, Pandas

Interactive Visualizations: Plotly

🚀 Getting Started

1. Clone the Repository

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


2. Set Up a Virtual Environment (Optional)

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Run the Application

streamlit run app.py


The dashboard will automatically open in your default browser at http://localhost:8501.


⚠️ Disclaimer

This application is built for research, simulation, and educational decision-support purposes only. It does not constitute financial or investment advice.