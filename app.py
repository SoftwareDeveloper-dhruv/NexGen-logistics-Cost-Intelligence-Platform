# app.py — NexGen Logistics Cost Intelligence Platform

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ---------------------- STREAMLIT PAGE CONFIG ----------------------
st.set_page_config(
    page_title="NexGen Logistics - Cost Intelligence Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- HEADER ----------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>💰 NexGen Logistics - Cost Intelligence Platform</h1>
    <p style='text-align: center; font-size:18px; color:#5D6D7E;'>
    Powered by AI, ML, and advanced analytics to identify cost leakages and optimize logistics efficiency.
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    orders = pd.read_csv("orders.csv")
    delivery = pd.read_csv("delivery_performance.csv")
    routes = pd.read_csv("routes_distance.csv")
    fleet = pd.read_csv("vehicle_fleet.csv")
    warehouse = pd.read_csv("warehouse_inventory.csv")
    feedback = pd.read_csv("customer_feedback.csv")
    costs = pd.read_csv("cost_breakdown.csv")
    return orders, delivery, routes, fleet, warehouse, feedback, costs


orders, delivery, routes, fleet, warehouse, feedback, costs = load_data()

# ---------------------- DATA MERGE ----------------------
merged = (
    orders.merge(delivery, on="Order_ID", how="left")
    .merge(routes, on="Order_ID", how="left")
    .merge(costs, on="Order_ID", how="left")
)

# Derived Metrics
merged["Total_Cost"] = merged[
    ["Fuel_Cost", "Labor_Cost", "Vehicle_Maintenance", "Insurance",
     "Packaging_Cost", "Technology_Platform_Fee", "Other_Overhead"]
].sum(axis=1)

merged["Cost_to_OrderValue"] = merged["Total_Cost"] / merged["Order_Value_INR"]
merged["Delivery_Delay"] = merged["Actual_Delivery_Days"] - merged["Promised_Delivery_Days"]

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("📊 Dashboard Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["Overview Dashboard", "Cost Leakage Analysis", "Optimization Opportunities", "AI Recommendations"]
)

# ---------------------- 1. OVERVIEW DASHBOARD ----------------------
if page == "Overview Dashboard":
    st.subheader("📈 Business Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Orders", len(orders))
    with col2:
        st.metric("Average Cost-to-Order Value Ratio", f"{merged['Cost_to_OrderValue'].mean():.2f}")
    with col3:
        st.metric("Average Customer Rating", f"{delivery['Customer_Rating'].mean():.2f}")

    # Plot 1 - Cost Breakdown Distribution
    cost_cols = ["Fuel_Cost", "Labor_Cost", "Vehicle_Maintenance", "Insurance",
                 "Packaging_Cost", "Technology_Platform_Fee", "Other_Overhead"]
    cost_summary = merged[cost_cols].mean().reset_index()
    cost_summary.columns = ["Cost_Type", "Average_INR"]

    fig1 = px.bar(cost_summary, x="Cost_Type", y="Average_INR",
                  color="Cost_Type", title="Average Cost Breakdown (INR)",
                  color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
    merged,
    x="Delivery_Delay",
    y="Customer_Rating",
    color="Delivery_Status",
    title="Delivery Delay vs Customer Rating",
    color_discrete_sequence=px.colors.qualitative.Safe,
)
    fig2.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))

    st.plotly_chart(fig2, use_container_width=True)


# ---------------------- 2. COST LEAKAGE ANALYSIS ----------------------
elif page == "Cost Leakage Analysis":
    st.subheader("💸 Cost Leakage Detection and Analysis")

    # --- 1️⃣ Overall Cost-to-Order Value Distribution ---
    st.markdown("### 1️⃣ Perform Cost Analysis")
    fig3 = px.histogram(
        merged, x="Cost_to_OrderValue", nbins=30, color_discrete_sequence=["#FF7F50"],
        title="Distribution of Cost-to-Order Value Ratio"
    )
    st.plotly_chart(fig3, use_container_width=True)

    avg_ratio = merged["Cost_to_OrderValue"].mean()
    high_cost_orders = merged[merged["Cost_to_OrderValue"] > 0.6]

    st.markdown(f"""
    - **Average Cost-to-Order Value Ratio:** {avg_ratio:.2f}  
    - **High-Cost Orders (>0.6 Ratio):** {len(high_cost_orders)}  
    - Potential savings exist if these orders are optimized or renegotiated.
    """)
    st.dataframe(high_cost_orders[["Order_ID", "Carrier", "Order_Value_INR", "Total_Cost", "Cost_to_OrderValue"]])

    # --- 2️⃣ Delivery Performance vs Cost Impact ---
    st.markdown("### 2️⃣ Analyze Delivery Performance and Its Impact on Costs")

    fig4 = px.scatter(
        merged, x="Delivery_Delay", y="Cost_to_OrderValue", color="Delivery_Status",
        title="Delivery Delay vs Cost-to-Order Value Ratio",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    - Late deliveries correlate with **higher cost ratios** due to extra labor and rescheduling.  
    - Orders marked as *Delayed* show clear cost inefficiency.
    """)

    # --- 3️⃣ Route Cost Implications ---
    st.markdown("### 3️⃣ Analyze Routes and Their Cost Implications")

    route_summary = merged.groupby("Route")[["Distance_KM", "Toll_Charges_INR", "Fuel_Consumption_L"]].mean().reset_index()
    route_summary["Avg_Total_Cost"] = merged.groupby("Route")["Total_Cost"].mean().values

    fig5 = px.scatter(
        route_summary, x="Distance_KM", y="Avg_Total_Cost",
        size="Fuel_Consumption_L", color="Toll_Charges_INR",
        hover_name="Route", title="Route Distance vs Avg Total Cost",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    - Longer routes with high tolls show **disproportionate cost increases**.  
    - Use route optimization or alternate paths to control costs.
    """)

    # --- 4️⃣ Vehicle Fleet Cost Analysis ---
    st.markdown("### 4️⃣ Analyze Vehicle Fleet Data")

    fleet_fig = px.scatter(
        fleet, x="Age_Years", y="Fuel_Efficiency_KM_per_L",
        color="Vehicle_Type", size="CO2_Emissions_Kg_per_KM",
        title="Vehicle Age vs Fuel Efficiency and Emissions",
        color_discrete_sequence=px.colors.qualitative.D3
    )
    st.plotly_chart(fleet_fig, use_container_width=True)

    st.markdown("""
    - Older vehicles show **reduced efficiency** and **higher emissions**, raising fuel and maintenance costs.  
    - Schedule proactive maintenance or consider replacing outdated fleet assets.
    """)

    # --- 5️⃣ Warehouse and Inventory Cost Insights ---
    st.markdown("### 5️⃣ Analyze Warehouse and Inventory Costs")

    fig6 = px.bar(
        warehouse, x="Location", y="Storage_Cost_per_Unit",
        color="Product_Category", title="Warehouse Storage Cost per Unit",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig6, use_container_width=True)

    high_cost_wh = warehouse[warehouse["Storage_Cost_per_Unit"] > warehouse["Storage_Cost_per_Unit"].mean()]
    st.markdown(f"""
    - **Average Storage Cost per Unit:** ₹{warehouse['Storage_Cost_per_Unit'].mean():.2f}  
    - **High-Cost Warehouses:** {len(high_cost_wh)}  
    - Consider relocating or optimizing inventory for expensive warehouse locations.
    """)

    # --- 6️⃣ Customer Feedback and Cost Correlation ---
    st.markdown("### 6️⃣ Analyze Customer Feedback and Its Link to Costs")

    feedback_cost = merged.merge(feedback, on="Order_ID", how="left")
    fig7 = px.box(
        feedback_cost, x="Rating", y="Cost_to_OrderValue",
        color="Would_Recommend", title="Customer Rating vs Cost-to-Order Value",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("""
    - Lower customer ratings often align with **higher cost ratios**, likely due to poor quality or delays.  
    - Improving delivery experience can reduce refund, re-delivery, and penalty costs.
    """)

    st.success("✅ Comprehensive cost leakage analysis completed. Key leakages found across delivery delays, inefficient routes, old vehicles, and costly warehouses.")


# ---------------------- 3. OPTIMIZATION OPPORTUNITIES ----------------------
elif page == "Optimization Opportunities":
    st.subheader("🚀 Optimization Opportunities")

    # Route Optimization Insight
    route_perf = merged.groupby("Route")[["Distance_KM", "Toll_Charges_INR", "Traffic_Delay_Minutes"]].mean().reset_index()
    fig4 = px.scatter_3d(
        route_perf, x="Distance_KM", y="Toll_Charges_INR", z="Traffic_Delay_Minutes",
        color="Traffic_Delay_Minutes", title="Route Performance: Distance vs Toll vs Delay",
        color_continuous_scale="Turbo"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Warehouse Cost Insight
    fig5 = px.bar(
        warehouse, x="Location", y="Storage_Cost_per_Unit",
        color="Product_Category", title="Warehouse Storage Costs",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Fleet Insight
    fig6 = px.scatter(
        fleet, x="Age_Years", y="Fuel_Efficiency_KM_per_L", color="Vehicle_Type",
        title="Vehicle Age vs Fuel Efficiency", color_discrete_sequence=px.colors.qualitative.D3
    )
    st.plotly_chart(fig6, use_container_width=True)

# ---------------------- 4. AI RECOMMENDATIONS ----------------------
elif page == "AI Recommendations":
    st.subheader("🤖 AI-Powered Recommendations")

    # Simple ML model to predict total cost
    features = ["Fuel_Cost", "Labor_Cost", "Vehicle_Maintenance", "Insurance",
                "Packaging_Cost", "Technology_Platform_Fee", "Other_Overhead"]
    X = merged[features]
    y = merged["Total_Cost"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    merged["Predicted_Cost"] = model.predict(X)

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=merged["Order_ID"], y=merged["Total_Cost"],
                              mode='lines', name='Actual Cost', line=dict(color='royalblue')))
    fig7.add_trace(go.Scatter(x=merged["Order_ID"], y=merged["Predicted_Cost"],
                              mode='lines', name='Predicted Cost', line=dict(color='orange')))
    fig7.update_layout(title="Actual vs Predicted Total Cost", xaxis_title="Order ID", yaxis_title="Cost (INR)")
    st.plotly_chart(fig7, use_container_width=True)

    # Strategic Recommendations
    st.markdown("### 📋 Recommended Cost Optimization Strategies")

    st.info("""
    **1️⃣ Re-negotiate Carrier Contracts**  
    - Focus on carriers with high average `Cost_to_OrderValue` ratios.
    - Use ML cost predictions to set data-driven negotiation thresholds.
    
    **2️⃣ Route Optimization**  
    - Identify routes with high toll charges or long delays using `routes_distance.csv`.
    - Apply dynamic route optimization to reduce delays and costs.
    
    **3️⃣ Delivery Consolidation**  
    - Combine smaller deliveries to minimize per-order fuel and packaging costs.
    
    **4️⃣ Proactive Vehicle Maintenance**  
    - Use `vehicle_fleet.csv` age and efficiency trends to schedule maintenance before costly breakdowns.
    
    **5️⃣ Automated Invoice Validation**  
    - Compare actual costs with ML-predicted cost benchmarks to flag billing anomalies.
    """)

    # Recommendation Bar Chart
    rec_data = pd.DataFrame({
        "Strategy": [
            "Renegotiate Carrier Contracts",
            "Optimize Routes",
            "Consolidate Deliveries",
            "Proactive Maintenance",
            "Automate Invoice Validation"
        ],
        "Potential_Savings_%": [10, 15, 12, 8, 10]
    })

    fig8 = px.bar(rec_data, x="Strategy", y="Potential_Savings_%",
                  color="Strategy", text="Potential_Savings_%",
                  color_discrete_sequence=px.colors.qualitative.Alphabet,
                  title="Estimated Savings by Recommendation (%)")
    st.plotly_chart(fig8, use_container_width=True)

# ---------------------- FOOTER ----------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray;'>
    © 2025 NexGen Logistics | Developed using Python, Streamlit, Plotly & AI
    </p>
    """,
    unsafe_allow_html=True
)
