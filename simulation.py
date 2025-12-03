import pandas as pd
import random

def run_championship_simulation(n_simulations=1000):
    """
    Monte Carlo Championship Predictor
    Simulates the remainder of the season 1,000 times to determine win probability.
    """
    # 1. Current Standings (Mock Data)
    drivers = {
        "Verstappen": {"points": 303, "win_prob": 0.45},
        "Norris":     {"points": 241, "win_prob": 0.30},
        "Leclerc":    {"points": 217, "win_prob": 0.15},
        "Piastri":    {"points": 197, "win_prob": 0.08},
        "Hamilton":   {"points": 164, "win_prob": 0.02}
    }
    
    remaining_races = 5
    results = {driver: 0 for driver in drivers} 

    # 2. Run the Simulations
    for _ in range(n_simulations):
        sim_standings = {k: v['points'] for k, v in drivers.items()}
        
        # Simulate each remaining race
        for race in range(remaining_races):
            winner = random.choices(
                list(drivers.keys()), 
                weights=[d['win_prob'] for d in drivers.values()]
            )[0]
            sim_standings[winner] += 25
        season_winner = max(sim_standings, key=sim_standings.get)
        results[season_winner] += 1

    # 3. Convert to Percentages
    data = []
    for driver, wins in results.items():
        win_pct = (wins / n_simulations) * 100
        data.append({"Driver": driver, "Title Probability": win_pct})
        
    return pd.DataFrame(data).sort_values(by="Title Probability", ascending=False)