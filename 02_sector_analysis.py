import os
from itertools import combinations
import fastf1
import pandas as pd

# -----------------------------
# SETTINGS
# -----------------------------
year = 2025
reference_speed = 216.3  # km/h
cache_dir = "cache"

tracks_to_load = {
    "Miami": "Miami",
    "Yas Marina": "Abu Dhabi",
    "Zandvoort": "Netherlands"
}

# -----------------------------
# CACHE SETUP
# -----------------------------
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir)

# -----------------------------
# STEP 1: LOAD DATA FROM FASTF1
# -----------------------------
track_data = {}

for label, gp_name in tracks_to_load.items():
    print(f"\n===== Loading {label} =====")

    session = fastf1.get_session(year, gp_name, 'R')
    session.load()

    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry().add_distance()

    # Sector timestamps
    s1_end = lap['Sector1SessionTime']
    s2_end = lap['Sector2SessionTime']

    # Sector times (seconds)
    s1_time = lap['Sector1Time'].total_seconds()
    s2_time = lap['Sector2Time'].total_seconds()
    s3_time = lap['Sector3Time'].total_seconds()

    # Split telemetry by sector
    s1_tel = tel[tel['SessionTime'] <= s1_end]
    s2_tel = tel[(tel['SessionTime'] > s1_end) & (tel['SessionTime'] <= s2_end)]
    s3_tel = tel[tel['SessionTime'] > s2_end]

    # Sector distances (meters)
    s1_dist = s1_tel['Distance'].max() - s1_tel['Distance'].min()
    s2_dist = s2_tel['Distance'].max() - s2_tel['Distance'].min()
    s3_dist = s3_tel['Distance'].max() - s3_tel['Distance'].min()

    track_data[label] = [
        ("S1", s1_time, s1_dist),
        ("S2", s2_time, s2_dist),
        ("S3", s3_time, s3_dist),
    ]

    print("Sector Times (s):", round(s1_time, 3),
          round(s2_time, 3), round(s3_time, 3))

    print("Sector Lengths (m):", round(s1_dist, 1),
          round(s2_dist, 1), round(s3_dist, 1))

# -----------------------------
# STEP 2: CALCULATE SPEEDS
# -----------------------------
results = []

for track, sectors in track_data.items():
    for sector_name, time_s, distance_m in sectors:
        speed_mps = distance_m / time_s
        speed_kmh = speed_mps * 3.6
        difference = abs(speed_kmh - reference_speed)

        results.append({
            "track": track,
            "sector": sector_name,
            "speed_kmh": speed_kmh,
            "difference_from_reference": difference
        })

print("\n===== Sector Speeds =====")
for r in results:
    print(f"{r['track']} {r['sector']}: {r['speed_kmh']:.2f} km/h")

# -----------------------------
# STEP 3: BEST SECTOR 1
# -----------------------------
sector1_options = [r for r in results if r["sector"] == "S1"]
best_sector1 = min(sector1_options,
                   key=lambda x: x["difference_from_reference"])

print("\nSelected Sector 1:")
print(f"{best_sector1['track']} {best_sector1['sector']} "
      f"{best_sector1['speed_kmh']:.2f} km/h")

# -----------------------------
# STEP 4: OPTIMISE REMAINING SECTORS
# -----------------------------
remaining_tracks = [t for t in track_data.keys()
                    if t != best_sector1["track"]]

remaining_candidates = [
    r for r in results
    if r["track"] in remaining_tracks and r["sector"] in ["S2", "S3"]
]

best_combo = None
best_score = float("inf")

for combo in combinations(remaining_candidates, 2):

    if combo[0]["track"] == combo[1]["track"]:
        continue

    speeds = [
        best_sector1["speed_kmh"],
        combo[0]["speed_kmh"],
        combo[1]["speed_kmh"]
    ]

    avg_speed = sum(speeds) / 3
    avg_diff = abs(avg_speed - reference_speed)
    variation = abs(combo[0]["speed_kmh"] - combo[1]["speed_kmh"])

    score = avg_diff - (variation * 0.01)

    if score < best_score:
        best_score = score
        best_combo = {
            "sectors": combo,
            "average": avg_speed,
            "variation": variation,
            "avg_diff": avg_diff
        }

# -----------------------------
# FINAL OUTPUT
# -----------------------------
print("\n===== Final Combination =====")
print(f"{best_sector1['track']} {best_sector1['sector']} "
      f"{best_sector1['speed_kmh']:.2f} km/h")

for sector in best_combo["sectors"]:
    print(f"{sector['track']} {sector['sector']} "
          f"{sector['speed_kmh']:.2f} km/h")

print(f"\nCombined Average Speed: {best_combo['average']:.2f} km/h")
print(f"Difference from Reference: {best_combo['avg_diff']:.2f} km/h")
print(f"Speed Variation: {best_combo['variation']:.2f} km/h")