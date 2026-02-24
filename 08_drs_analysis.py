import pandas as pd
import numpy as np

# =============================
# CONFIG
# =============================
LAP_LEN_M = 5183.7
TOTAL_RACE_M = LAP_LEN_M * 8  # 41469.6m
GAP_FACTOR = 16.646  # Calibration factor
DP1 = 3510
DP2 = 4550

# =============================
# PROCESS DATA
# =============================
# Load the smooth interpolation file
df = pd.read_csv("race_time_interpolated.csv")
df = df.sort_values(["time_sec", "distance_km"], ascending=[True, False])

drivers = df['driver'].unique()
driver_state = {d: {'prev_m': 0} for d in drivers}
drs_results = []

print("Analyzing DRS Detection Points...")

# Group by time to handle gaps between cars correctly
for ti, frame in df.groupby('time_sec'):
    sorted_drivers = frame['driver'].tolist()
    
    for rank, d in enumerate(sorted_drivers):
        dist_km = frame.loc[frame['driver'] == d, 'distance_km'].values[0]
        dist_m = dist_km * 1000
        lap_m = dist_m % LAP_LEN_M
        lap_num = int(dist_m // LAP_LEN_M) + 1
        
        if dist_m >= TOTAL_RACE_M:
            driver_state[d]['prev_m'] = lap_m
            continue
        # Check if they crossed a DP this frame
        crossed_dp1 = (driver_state[d]['prev_m'] < DP1 <= lap_m)
        crossed_dp2 = (driver_state[d]['prev_m'] < DP2 <= lap_m)
        
        # Logic starts after Lap 1
        if lap_num > 1 and (crossed_dp1 or crossed_dp2):
            point_name = "DP1 (3510m)" if crossed_dp1 else "DP2 (4550m)"
            
            status = "DENIED"
            gap_s = 0.0
            
            if rank > 0: # If there is a car ahead
                driver_ahead = sorted_drivers[rank-1]
                dist_ahead = frame.loc[frame['driver'] == driver_ahead, 'distance_km'].values[0]
                gap_s = (dist_ahead - dist_km) * GAP_FACTOR
                
                if gap_s <= 1.0:
                    status = "ELIGIBLE"
            
            # Log the result
            drs_results.append({
                "time_sec": ti,
                "lap": lap_num,
                "driver": d,
                "detection_point": point_name,
                "gap_to_ahead": round(gap_s, 3),
                "status": status
            })
            
            # Optional: Print to console for immediate check
            if status == "ELIGIBLE":
                print(f"Time {ti:.2f}s | Lap {lap_num} | {d} -> {status} at {point_name} (Gap: {gap_s:.3f}s)")

        # Update persistent state
        driver_state[d]['prev_m'] = lap_m

# =============================
# SAVE RESULTS
# =============================
output = pd.DataFrame(drs_results)
output.to_csv("drs_eligibility_log.csv", index=False)

print("\n✔ Analysis Complete.")
print(f"✔ Found {len(output[output['status'] == 'ELIGIBLE'])} DRS activations.")
print("✔ Log saved to: drs_eligibility_log.csv")