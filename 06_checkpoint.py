import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================
# CONFIG (DO NOT CHANGE)
# =============================
LAP_LENGTH_KM = 5.1837          # derived circuit
CHECK_ROUND = 5                # snapshot round to visualize

# =============================
# LOAD TRACK WAYPOINTS
# =============================
track = pd.read_csv("track_waypoints.csv")
points = track[['x', 'y']].values

# =============================
# COMPUTE TRACK LENGTH (PIXELS)
# =============================
segment_lengths = np.sqrt(
    np.sum(np.diff(points, axis=0) ** 2, axis=1)
)

pixel_distances = np.insert(np.cumsum(segment_lengths), 0, 0)
total_track_pixels = pixel_distances[-1]

KM_PER_PIXEL = LAP_LENGTH_KM / total_track_pixels
distance_km_along_track = pixel_distances * KM_PER_PIXEL

print("Track loaded")
print(f"Total pixels: {int(total_track_pixels)}")
print(f"KM per pixel: {KM_PER_PIXEL:.6f}")

# =============================
# FUNCTION: KM → PIXEL INDEX
# =============================
def km_to_pixel_index(distance_km):
    d = distance_km % LAP_LENGTH_KM
    return np.argmin(np.abs(distance_km_along_track - d))

# =============================
# LOAD RACE DATA
# =============================
race = pd.read_csv("output/f1_race.csv")

drivers = race['driver'].unique()
snapshot = race[race['round'] == CHECK_ROUND]

print(f"Loaded race data for round {CHECK_ROUND}")

# =============================
# STATIC VISUAL CHECK
# =============================
plt.figure(figsize=(7, 7))

# draw track
plt.plot(points[:, 0], -points[:, 1], color='black', linewidth=1)

# draw cars
for _, row in snapshot.iterrows():
    idx = km_to_pixel_index(row['distance_km'])
    x, y = points[idx]
    plt.plot(x, -y, 'ro', markersize=4)

plt.title(f"Round {CHECK_ROUND} – Static Position Check")
plt.axis('equal')
plt.axis('off')
plt.show()

print("If cars sit ON the track → mapping is correct.")
