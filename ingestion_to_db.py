from turtle import st
import fastf1
import pandas as pd
from sqlalchemy import create_engine, text  

# --- CONFIGURATION ---
DB_CONNECTION_URI = st.secrets["DB_CONNECTION_URI"] 

# Setup DB Connection
engine = create_engine(DB_CONNECTION_URI)

def convert_timedelta(td):
    """Converts F1 timing (timedelta) to total seconds (float) for the DB"""
    if pd.isna(td):
        return None
    return td.total_seconds()

def process_race(year, race_name):
    print(f"--- Processing {year} {race_name} ---")
    
    # 1. Load Data (Using Service 1 Logic)
    session = fastf1.get_session(year, race_name, 'R')
    session.load()
    
    # 2. Insert Race Metadata
    race_date = session.date.strftime('%Y-%m-%d')
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO races (year, round, circuit_name, date)
            VALUES ({year}, {session.event.RoundNumber}, '{race_name}', '{race_date}')
            ON CONFLICT (year, round) DO NOTHING;
        """))
        result = conn.execute(text(f"SELECT race_id FROM races WHERE year={year} AND round={session.event.RoundNumber}"))
        race_id = result.fetchone()[0]

        # 3. Process Laps (The Complex Part)
        laps = session.laps
        clean_laps = []
        drivers_seen = set()
        for index, row in laps.iterrows():
            driver = row['Driver']
            team = row['Team']
            if driver not in drivers_seen:
                conn.execute(text(f"""
                    INSERT INTO drivers (code, name, team) 
                    VALUES ('{driver}', '{driver}', '{team}')
                    ON CONFLICT (code) DO NOTHING;
                """))
                drivers_seen.add(driver)
            clean_laps.append({
                'race_id': race_id,
                'driver_code': driver,
                'lap_number': row['LapNumber'],
                'position': row['Position'],
                'lap_time_seconds': convert_timedelta(row['LapTime']),
                'tyre_compound': row['Compound'],
                'tyre_life': row['TyreLife'],
                'is_pit_in': row['PitInTime'] is not pd.NaT,
                'is_pit_out': row['PitOutTime'] is not pd.NaT
            })

    # 4. Bulk Upload to DB 
    print("Uploading Laps to Database...")
    df_upload = pd.DataFrame(clean_laps)
    df_upload.to_sql('lap_times', engine, if_exists='append', index=False, method='multi')
    print(f"Success! {len(df_upload)} laps uploaded for {race_name}.")

if __name__ == "__main__":
    process_race(2024, 'Bahrain')