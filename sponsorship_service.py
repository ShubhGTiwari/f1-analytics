import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st
from fastf1.ergast import Ergast

# --- CONFIGURATION ---
DB_URI = st.secrets["DB_CONNECTION_URI"]

engine = create_engine(DB_URI)

# 1. Define Sponsor Mappings
SPONSOR_MAP = {
    "Red Bull Racing": ["Oracle", "Bybit"],
    "Ferrari": ["Santander", "Shell"],
    "Mercedes": ["Petronas", "Ineos"],
    "McLaren": ["Google Chrome", "OKX"],
    "Aston Martin": ["Aramco", "Cognizant"],
    "Alpine": ["BWT", "Castrol"],
    "Williams": ["Komatsu", "Gulf"],
    "RB": ["Visa", "Cash App"],
    "Kick Sauber": ["Stake", "Kick"],
    "Haas F1 Team": ["MoneyGram", "Chipotle"]
}

def get_team_popularity_weights():
    print("Fetching Team Standings for weighting...")
    try:
        ergast = Ergast()
        standings = ergast.get_constructor_standings(season='current')
        data = standings.content[0]['constructorStandings']
        
        weights = {}
        for entry in data:
            team_name = entry['constructor']['name']
            rank = int(entry['position'])
            # Logic: Rank 1 = 1.5x multiplier, Rank 10 = 0.8x
            multiplier = 1.5 - (rank * 0.07)
            weights[team_name] = max(0.5, multiplier)
        return weights
    except:
        print("API Error. Using default weights.")
        return {}

def calculate_media_value(race_id=1):
    # 1. Get Lap Data
    query = text(f"""
        SELECT l.driver_code, d.team, l.position 
        FROM lap_times l 
        JOIN drivers d ON l.driver_code = d.code 
        WHERE l.race_id = {race_id}
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
    # 2. Get Real Weights
    team_weights = get_team_popularity_weights()
    BASE_LAP_VALUE = 5000 
    
    roi_data = []
    
    for team, sponsors in SPONSOR_MAP.items():
        team_laps = df[df['team'].str.contains(team, case=False, na=False)]
        total_value = 0
        
        # Get Team Weight (Default to 1.0 if not found)
        t_weight = 1.0
        for real_name, weight in team_weights.items():
            if team.lower() in real_name.lower():
                t_weight = weight
                break
        
        for _, row in team_laps.iterrows():
            pos = row['position']
            # Position Multiplier
            pos_mult = 10.0 if pos == 1 else (5.0 if pos <= 3 else (1.0 if pos <= 10 else 0.1))
            
            # ROI = Base * Position Bonus * Team Popularity
            lap_value = BASE_LAP_VALUE * pos_mult * t_weight
            total_value += lap_value
            
        for sponsor in sponsors:
            roi_data.append({
                "Sponsor": sponsor,
                "Team": team,
                "Estimated Media Value ($)": round(total_value, 2)
            })
            
    return pd.DataFrame(roi_data).sort_values(by="Estimated Media Value ($)", ascending=False)