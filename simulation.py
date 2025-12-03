import pandas as pd
import random
from fastf1.ergast import Ergast

def get_live_standings():
    """Fetches real-world points from the Ergast API"""
    ergast = Ergast()
    # Fetch current season standings
    standings = ergast.get_driver_standings(season='current')
    
    # Parse the complex API response
    drivers = {}
    if standings.content and standings.content[0].get('driverStandings'):
        for entry in standings.content[0]['driverStandings']:
            name = entry['driver']['familyName']
            points = float(entry['points'])
            weight = points / 10 if points > 0 else 0.5
            
            drivers[name] = {"points": points, "win_prob": weight}
    return drivers

def run_championship_simulation(n_simulations=1000):
    print("--- 📡 Fetching Live Standings... ---")
    
    # 1. GET REAL DATA
    try:
        drivers = get_live_standings()
        if not drivers: raise ValueError("No data returned")
    except Exception as e:
        print(f"API Warning: {e}. Using cached 2024 fallback data.")
        drivers = {
            "Verstappen": {"points": 437, "win_prob": 40}, 
            "Norris": {"points": 374, "win_prob": 35},
            "Leclerc": {"points": 371, "win_prob": 25}
        }
        
    remaining_races = 5
    results = {driver: 0 for driver in drivers}

    # 2. Run Simulations
    for _ in range(n_simulations):
        sim_standings = {k: v['points'] for k, v in drivers.items()}
        
        for race in range(remaining_races):
            winner = random.choices(
                list(drivers.keys()), 
                weights=[d['win_prob'] for d in drivers.values()]
            )[0]
            sim_standings[winner] += 25
            
        season_winner = max(sim_standings, key=sim_standings.get)
        results[season_winner] += 1

    # 3. Format Output
    data = []
    for driver, wins in results.items():
        if wins > 0:
            win_pct = (wins / n_simulations) * 100
            data.append({"Driver": driver, "Title Probability": win_pct})
        
    return pd.DataFrame(data).sort_values(by="Title Probability", ascending=False)