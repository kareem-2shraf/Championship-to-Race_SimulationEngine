import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data
waypoints = pd.read_csv('track_waypoints.csv')
pts = waypoints.values
pts[:, 1] = -pts[:, 1] # Flip Y

# 2. Calculate Raw Pixel Distance
dx = np.diff(pts[:, 0])
dy = np.diff(pts[:, 1])
pixel_segments = np.sqrt(dx**2 + dy**2)
total_pixel_length = np.sum(pixel_segments)

# 3. CALIBRATION (The Fix)
# Change TARGET_METERS to the actual length you want your track to be
TARGET_METERS = 5183.7  
scale_factor = TARGET_METERS / total_pixel_length

# Apply scale to our "tape measure"
cumulative_dist = np.concatenate(([0], np.cumsum(pixel_segments * scale_factor)))

print(f"--- CALIBRATION ---")
print(f"Total Pixels: {total_pixel_length:.2f}")
print(f"Scaling Factor: {scale_factor:.4f}")
print(f"Final Track Length: {cumulative_dist[-1]:.2f} meters")

# 4. Interactive Tool
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#111')
ax.plot(pts[:, 0], pts[:, 1], color='#444', lw=3, zorder=1)
ax.scatter(pts[0, 0], pts[0, 1], color='lime', s=100, label='Start (0m)', zorder=5)

def on_click(event):
    if event.xdata is None or event.ydata is None: return
    
    # Find closest point
    dists = np.sqrt((pts[:, 0] - event.xdata)**2 + (pts[:, 1] - event.ydata)**2)
    idx = np.argmin(dists)
    
    meters = cumulative_dist[idx]
    percentage = (meters / TARGET_METERS) * 100
    
    print(f"Point: {idx} | Dist: {meters:.1f}m | {percentage:.1f}% of lap")
    
    ax.scatter(pts[idx, 0], pts[idx, 1], color='cyan', s=30, zorder=10)
    plt.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.title("Track Ruler (Calibrated to Meters)", color='white')
plt.show()