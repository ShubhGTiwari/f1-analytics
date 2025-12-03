import fastf1
import pandas as pd
from sqlalchemy import create_engine, text
from turtle import st

# --- CONFIGURATION ---
DB_CONNECTION_URI = st.secrets["DB_CONNECTION_URI"] 

# Setup DB Connection
engine = create_engine(DB_CONNECTION_URI)

def convert_timedelta(td):
    """Converts F1 timing (timedelta) to total seconds (float)"""
    if pd.isna(td):
        return None
    return td.total_seconds()

def ingest_telemetry(year, race_name):
    print(f"--- Ingesting Telemetry Stats for {year} {race_name} ---")
    
    # 1. Load Data (Cached)
    session = fastf1.get_session(year, race_name, 'R')
    session.load()
    
    # 2. Get the Race ID from DB (to link foreign keys)
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT race_id FROM races WHERE year={year} AND circuit_name='{race_name}'"))
        race_id_row = result.fetchone()
        
        if not race_id_row:
            print("Error: Race not found in DB. Run ingestion_to_db.py first!")
            return
        race_id = race_id_row[0]

    # 3. Prepare Telemetry Data
    laps = session.laps
    telemetry_rows = []

    print("Processing sector times and top speeds...")
    
    for index, row in laps.iterrows():
        # Only process valid laps
        if pd.isna(row['LapTime']):
            continue

        telemetry_rows.append({
            'race_id': race_id,
            'driver_code': row['Driver'],
            'lap_number': row['LapNumber'],
            'sector_1_time': convert_timedelta(row['Sector1Time']),
            'sector_2_time': convert_timedelta(row['Sector2Time']),
            'sector_3_time': convert_timedelta(row['Sector3Time']),
            'max_speed_kmh': row['SpeedST'], 
            'avg_throttle_pct': None 
        })

    # 4. Bulk Upload
    if telemetry_rows:
        print(f"Uploading {len(telemetry_rows)} telemetry rows to Supabase...")
        df_upload = pd.DataFrame(telemetry_rows)
        df_upload.to_sql('telemetry_stats', engine, if_exists='append', index=False, method='multi')
        print("Telemetry Ingestion Complete!")
    else:
        print("No telemetry rows generated.")

if __name__ == "__main__":
    ingest_telemetry(2024, 'Bahrain')