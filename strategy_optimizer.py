import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import statsmodels.api as sm
from turtle import st

# --- CONFIGURATION ---
DB_CONNECTION_URI = st.secrets["DB_CONNECTION_URI"]

# Setup DB Connection
engine = create_engine(DB_CONNECTION_URI)

def fetch_race_data(race_id=1):
    """Fetches clean lap times from our Cloud DB"""
    print("Fetching data from Supabase...")
    
    query = text(f"""
        SELECT driver_code, lap_number, lap_time_seconds, tyre_compound, tyre_life 
        FROM lap_times 
        WHERE race_id = {race_id} 
        AND lap_time_seconds IS NOT NULL 
        AND is_pit_in = FALSE 
        AND is_pit_out = FALSE
        AND tyre_life < 40 -- Filter out extreme outliers
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    return df

def analyze_strategy(df):
    """Calculates the degradation curve for each tyre compound"""
    print("Analyzing Tyre Degradation...")
    
    # 1. Clean outliers (laps slower than 105s are likely yellow flags/errors)
    df_clean = df[df['lap_time_seconds'] < 105]

    # 2. Visualize: Tyre Age vs. Lap Time
    fig = px.scatter(
        df_clean, 
        x="tyre_life", 
        y="lap_time_seconds", 
        color="tyre_compound",
        trendline="ols", 
        title="Bahrain 2024: Tyre Degradation Analysis (Strategy Model)",
        labels={"tyre_life": "Tyre Age (Laps)", "lap_time_seconds": "Lap Time (s)"},
        color_discrete_map={"SOFT": "red", "MEDIUM": "yellow", "HARD": "white"}
    )
    
    # Update layout to look like F1 TV
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#15151e",
        paper_bgcolor="#15151e",
        font=dict(color="white")
    )
    
    # 3. Show the interactive plot
    fig.show()
    
    # 4. Print the Mathematical Slope (Degradation per Lap)
    print("\n--- STRATEGY INSIGHTS (Degradation per Lap) ---")
    results = px.get_trendline_results(fig)
    for i, row in results.iterrows():
        compound = row['tyre_compound']
        model = row['px_fit_results']
        slope = model.params[1]
        print(f"Compound: {compound} | Time Loss per Lap: {slope:.3f}s")

if __name__ == "__main__":
    # 1. Get Data
    df = fetch_race_data(race_id=1) 
    # 2. Run Strategy Model
    analyze_strategy(df)