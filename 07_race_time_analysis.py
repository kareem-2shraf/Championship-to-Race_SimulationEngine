import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
# =============================
# CONFIG
# =============================
FPS = 30
LAP_LENGTH_KM = 5.1837
ROUND_TIME = 86.288 / 3   # seconds per round

# =============================
# SPEED PROFILES (Derived from Telemetry, Fastf1 & Tracinginsights)
# Distance vs speed data

# Zandvoort — 1st lap
# 0m to 198m (0 to 222 km/h)
# 198m to 342m (222 to 99 km/h)
# 342m to 573m (99 to 237 km/h)
# 573m to 679m (237 to 162 km/h)
# 679m to 732m (162 to 177 km/h)
# 732m to 819m (177 to 127 km/h)
# 819m to 1145m (127 to 263 km/h)
# 1145m to 1503m (263 to 280 km/h)

# Zandvoort — Fastest lap
# 0m to 122m (295 to 299 km/h)
# 122m to 217m (299 to 295 km/h)
# 217m to 373m (295 to 115 km/h)
# 373m to 605m (115 to 240 km/h)
# 605m to 704m (240 to 195 km/h)
# 704m to 738m (195 to 203 km/h)
# 738m to 834m (203 to 138 km/h)
# 834m to 1195m (138 to 274 km/h)
# 1195m to 1503m (274 to 280 km/h)

# Yas Marina
# 3090m to 3370m (280 to 301 km/h)
# 3370m to 3520m (301 to 297 km/h)
# 3520m to 3669m (297 to 167 km/h)
# 3669m to 4053m (167 to 271 km/h)
# 4053m to 4194m (271 to 270 km/h)
# 4194m to 4324m (270 to 104 km/h)
# 4324m to 4510m (104 to 164 km/h)
# 4510m to 4558m (164 to 155 km/h)
# 4558m to 4783m (155 to 270 km/h)
# 4783m to 4896m (270 to 140 km/h)
# 4896m to 5000m (140 to 167 km/h)

# Miami
# 3770m to 4688m (167 to 303 km/h)
# 4688m to 4831m (303 to 70 km/h)
# 4831m to 5412m (70 to 295 km/h)

# Final F1 2025 circuit physics

# Sector 1 (Zandvoort-style)
# 0m to 122m (295 to 299 km/h)
# 122m to 217m (299 to 295 km/h)
# 217m to 373m (295 to 115 km/h)
# 373m to 605m (115 to 240 km/h)
# 605m to 704m (240 to 195 km/h)
# 704m to 738m (195 to 203 km/h)
# 738m to 834m (203 to 138 km/h)
# 834m to 1195m (138 to 274 km/h)
# 1195m to 1503m (274 to 280 km/h)

# Transition
# 1503m to 1580m (280 to 250 km/h)
# 1580m to 1658m (250 to 280 km/h)

# Sector 2 (Yas Marina-style)
# 1658m to 1938m (280 to 301 km/h)
# 1938m to 2088m (301 to 297 km/h)
# 2088m to 2237m (297 to 167 km/h)
# 2237m to 2621m (167 to 271 km/h)
# 2621m to 2762m (271 to 270 km/h)
# 2762m to 2892m (270 to 104 km/h)
# 2892m to 3078m (104 to 164 km/h)
# 3078m to 3126m (164 to 155 km/h)
# 3126m to 3351m (155 to 270 km/h)
# 3351m to 3438m (270 to 140 km/h)
# 3438m to 3542m (140 to 167 km/h)

# Sector 3 (Miami-style)
# 3542m to 4460m (167 to 303 km/h)
# 4460m to 4603m (303 to 70 km/h)
# 4603m to 5183.7m (70 to 295 km/h)
# =============================
lap1_profile = np.array([
    (0,0),(256,222),(403,99),(708,237),(808,162),(861,177),
    (952,127),(1295,263),(1700,280),(2088,297),(2235,167),
    (2568,268),(2740,267),(2848,104),(3000,164),(3100,154),
    (3235,245),(3360,170),(4350,315),(4500,70),(5183.7,190)
])

lapN_profile = np.array([
    (0,295),(297,288),(431,115),(740,240),(833,195),(877,303),
    (964,138),(1322,270),(1700,280),(2088,297),(2235,167),
    (2568,268),(2740,267),(2848,104),(3000,164),(3100,154),
    (3235,245),(3360,170),(4350,315),(4500,70),(5183.7,190)
])
lap1_interp = PchipInterpolator(
    lap1_profile[:, 0],
    lap1_profile[:, 1]
)

lapN_interp = PchipInterpolator(
    lapN_profile[:, 0],
    lapN_profile[:, 1]
)
def speed_at(dist_m, lap):
    if lap == 1:
        return float(lap1_interp(dist_m))
    else:
        return float(lapN_interp(dist_m))

# =============================
# LOAD SNAPSHOT DATA
# =============================
race = pd.read_csv("output/f1_race.csv")
race = race.sort_values(["round", "position"])

drivers = race["driver"].unique()
rounds = sorted(race["round"].unique())

snapshots = {
    r: race[race["round"] == r].set_index("driver")["distance_km"]
    for r in rounds
}

times_by_round = {
    r: race[race["round"] == r]["time_sec"].iloc[0]
    for r in rounds
}

records = []

# =============================
# NORMAL INTERPOLATION (PRE-FINISH)
# =============================
for r in range(len(rounds) - 1):
    r0, r1 = rounds[r], rounds[r+1]
    d0 = snapshots[r0]
    d1 = snapshots[r1]

    t0 = times_by_round[r0]
    t1 = times_by_round[r1]

    times = np.arange(t0, t1, 1/FPS)

    for d in drivers:
        start_km = d0[d]
        end_km   = d1[d]
        delta_km = end_km - start_km

        if abs(delta_km) < 1e-6:
            for ti in times:
                records.append([ti, d, start_km])
            continue

        lap = int(start_km // LAP_LENGTH_KM) + 1

        SAMPLES = 300
        meters = np.linspace(0, abs(delta_km) * 1000, SAMPLES)
        speeds = np.array([
            speed_at((start_km*1000 + m) % (LAP_LENGTH_KM*1000), lap)
            for m in meters
        ])

        weights = speeds / speeds.sum()
        cum = np.cumsum(weights)

        for i, ti in enumerate(times):
            frac = i / (len(times) - 1)
            idx = np.searchsorted(cum, frac)
            progressed = delta_km * cum[min(idx, len(cum)-1)]
            records.append([ti, d, start_km + progressed])

# =============================
# POST-FINISH (CORRECT LOGIC)
# =============================
finish_time = times_by_round[rounds[-1]]
final_snapshot = snapshots[rounds[-1]]

leader = final_snapshot.idxmax()
leader_finish_dist = final_snapshot[leader]

# gap to leader at finish
gaps = {
    d: leader_finish_dist - final_snapshot[d]
    for d in drivers
}

# leader pace (km/sec) from last round
leader_prev = snapshots[rounds[-2]][leader]
leader_speed = (leader_finish_dist - leader_prev) / ROUND_TIME

# time until last car finishes
max_gap = max(gaps.values())
extra_time = max_gap / leader_speed
end_time = finish_time + extra_time

times = np.arange(finish_time, end_time, 1/FPS)

for i, ti in enumerate(times):
    leader_dist = leader_finish_dist + leader_speed * (ti - finish_time)

    for d in drivers:
        records.append([
            ti,
            d,
            leader_dist - gaps[d]
        ])

# =============================
# SAVE
# =============================
interp = pd.DataFrame(
    records,
    columns=["time_sec","driver","distance_km"]
)

interp = interp.sort_values(["time_sec","driver"])
interp.to_csv("race_time_interpolated.csv", index=False)

print("✔ Correct finish-line physics applied")
print(f"✔ Race ends at {end_time:.2f}s")
print(f"✔ Total frames: {len(interp)}")