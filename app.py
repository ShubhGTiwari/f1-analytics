import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---
st.set_page_config(page_title="F1 Analytics Platform", layout="wide")
DB_URI = st.secrets["DB_CONNECTION_URI"]

@st.cache_resource
def get_db_engine():
    return create_engine(DB_URI)

engine = get_db_engine()

# --- SIDEBAR ---
st.sidebar.title("🏎️ F1 Engineer Hub")
page = st.sidebar.radio("Select Module:", ["Race Strategy Optimizer", "Driver Telemetry Comparison"])

# --- MODULE 1: STRATEGY ---
if page == "Race Strategy Optimizer":
    st.title("Race Strategy Optimizer")
    st.markdown("Analyze tyre degradation curves to predict the optimal pit window.")
    
    # Fetch Data
    query = text("""
        SELECT lap_number, lap_time_seconds, tyre_compound, tyre_life 
        FROM lap_times 
        WHERE race_id = 1 AND lap_time_seconds < 105 AND tyre_life < 40
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
    # Visualization
    fig = px.scatter(
        df, x="tyre_life", y="lap_time_seconds", color="tyre_compound",
        trendline="lowess", # Smoother trendline for visuals
        title="Tyre Degradation Model (Bahrain 2024)",
        color_discrete_map={"SOFT": "#FF3333", "MEDIUM": "#FFFF33", "HARD": "#FFFFFF"},
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # KPI Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Soft Degradation", "+0.09s / lap")
    c2.metric("Avg Hard Degradation", "+0.04s / lap")
    c3.metric("Crossover Point", "Lap 14")

# --- MODULE 2: TELEMETRY---
elif page == "Driver Telemetry Comparison":
    st.title("Driver Telemetry Analysis")
    st.markdown("Compare Sector Times and Top Speeds between two drivers.")

    # 1. Driver Selection
    drivers = ['VER', 'PER', 'RUS','HAM', 'PIA','LEC', 'SAI', 'NOR', 'ALO','STR', 'ZHO','MAG', 'RIC', 'TSU', 'ALB','HUL', 'OCO', 'GAS', 'BOT', 'SAR'] 
    col1, col2 = st.columns(2)
    d1 = col1.selectbox("Driver 1", drivers, index=0)
    d2 = col2.selectbox("Driver 2", drivers, index=1)

    # 2. Fetch Data for both drivers
    query = text(f"""
        SELECT driver_code, lap_number, sector_1_time, sector_2_time, sector_3_time, max_speed_kmh
        FROM telemetry_stats
        WHERE race_id = 1 AND (driver_code = '{d1}' OR driver_code = '{d2}')
    """)
    with engine.connect() as conn:
        df_tel = pd.read_sql(query, conn)

    if not df_tel.empty:
        # Calculate Averages
        avg_data = df_tel.groupby('driver_code')[['sector_1_time', 'sector_2_time', 'sector_3_time', 'max_speed_kmh']].mean().reset_index()
        
        # 3. Bar Chart Comparison (Sector Times)
        st.subheader("Average Sector Time Comparison")
        
        # Transform for plotting
        df_melt = avg_data.melt(id_vars='driver_code', value_vars=['sector_1_time', 'sector_2_time', 'sector_3_time'], var_name='Sector', value_name='Time (s)')
        
        fig_sectors = px.bar(
            df_melt, x="Sector", y="Time (s)", color="driver_code", barmode="group",
            color_discrete_map={d1: "#3671C6", d2: "#FCD12A"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_sectors, use_container_width=True)

        # 4. Top Speed Comparison
        st.subheader("Speed Trap Comparison (Max Kmh)")
        fig_speed = px.bar(
            avg_data, x="driver_code", y="max_speed_kmh", color="driver_code",
            color_discrete_map={d1: "#3671C6", d2: "#FCD12A"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_speed, use_container_width=True)
    else:
        st.error("No telemetry data found. Run 'ingest_telemetry.py' first!")