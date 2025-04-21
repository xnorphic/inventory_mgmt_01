import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO
from inventory_optimizer import InventoryOptimizer  # Import the logic module

st.set_page_config(page_title="AI Inventory Optimizer", layout="wide")
st.title("🧠 AI-Powered Inventory Optimization Dashboard")

# Sidebar Inputs
st.sidebar.header("Upload Your Data")
file = st.sidebar.file_uploader("Upload Inventory CSV", type=["csv"])

revenue_target = st.sidebar.number_input("Revenue Target", min_value=0, value=12000000)
forecast_periods = st.sidebar.slider("Forecast Months", min_value=1, max_value=6, value=3)
run_button = st.sidebar.button("Run Optimization")

if run_button:
    if file is None:
        st.warning("Please upload a CSV file.")
        st.stop()

    df = pd.read_csv(file)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Replace #N/A and other non-numeric with NaN
    df.replace(to_replace=['#N/A', 'NA', 'N/A', '', ' '], value=np.nan, inplace=True)

    # Convert necessary columns to numeric
    for col in ['forecast_avg', 'cogs_per_unit', 'selling_price_per_unit', 'lead_time_days', 'moq',
                'last_3month_sold', 'last_6month_sold', 'current_stock']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    required_cols = ['selling_price_per_unit', 'cogs_per_unit']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)} in uploaded file.")
        st.stop()

    optimizer = InventoryOptimizer().load_dataframe(df)

    optimizer.calculate_monthly_avg_sales()
    optimizer.forecast_sales(forecast_periods)
    optimizer.calculate_stock_needed(revenue_target)
    optimizer.calculate_reorder_points()

    optimizer.data['forecast_profit'] = optimizer.data['forecast_avg'] * (
        optimizer.data['selling_price_per_unit'] - optimizer.data['cogs_per_unit'])
    optimizer.data['holding_cost'] = optimizer.data['current_stock'] * optimizer.data['cogs_per_unit']
    optimizer.data['gross_margin_per_unit'] = optimizer.data['selling_price_per_unit'] - optimizer.data['cogs_per_unit']
    optimizer.data['reorder_priority'] = optimizer.data['needs_reorder'].astype(int) * optimizer.data['gross_margin_per_unit']

    report = optimizer.generate_report(revenue_target)

    st.subheader("📊 Dashboard Metrics")
    st.write(report['metrics'])

    def plot_bar(data, x, y, title, color=None):
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=data, x=x, y=y, ax=ax, color=color)
        plt.xticks(rotation=45, ha='right')
        plt.title(title)
        st.pyplot(fig)

    top_profit = optimizer.data.sort_values('forecast_profit', ascending=False).head(10)
    top_holding = optimizer.data.sort_values('holding_cost', ascending=False).head(10)
    reorder_priority = optimizer.data[optimizer.data['needs_reorder']].sort_values('reorder_priority', ascending=False).head(10)

    st.subheader("💰 Top Forecast Profit SKUs")
    plot_bar(top_profit, 'product_name', 'forecast_profit', 'Top Forecast Profit')

    st.subheader("🏬 High Holding Cost SKUs")
    plot_bar(top_holding, 'product_name', 'holding_cost', 'Top Holding Cost')

    st.subheader("⚠️ Reorder Priority SKUs")
    plot_bar(reorder_priority, 'product_name', 'reorder_priority', 'Top Reorder Priority')

    st.subheader("🛒 Reorder Recommendations")
    st.dataframe(report['reorder_recommendations'].head(10))

    st.subheader("📦 Stock Recommendations")
    st.dataframe(report['stock_recommendations'].head(10))

    st.sidebar.download_button("Download Reorder CSV", data=report['reorder_recommendations'].to_csv(index=False),
                               file_name="reorder_recommendations.csv")
    st.sidebar.download_button("Download Stock CSV", data=report['stock_recommendations'].to_csv(index=False),
                               file_name="stock_recommendations.csv")
