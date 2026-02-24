import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import euclidean

# =========================
# 1. LOAD CENTERLINE
# =========================
img = cv2.imread("final_centerline.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

pixels = np.column_stack(np.where(binary == 255)[::-1])
pixel_set = set(map(tuple, pixels))

# =========================
# 2. BUILD GRAPH
# =========================
neighbors = defaultdict(list)
dirs = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1) if not (dx==0 and dy==0)]

for x,y in pixel_set:
    for dx,dy in dirs:
        nx, ny = x+dx, y+dy
        if (nx,ny) in pixel_set:
            neighbors[(x,y)].append((nx,ny))

# =========================
# 3. FIND DEAD-ENDS (TUNNEL)
# =========================
dead_ends = [p for p in neighbors if len(neighbors[p]) == 1]

print(f"Detected dead-ends: {len(dead_ends)}")

if len(dead_ends) != 2:
    raise RuntimeError(
        f"Expected exactly 2 tunnel dead-ends, found {len(dead_ends)}"
    )

# =========================
# 4. VIRTUALLY RECONNECT TUNNEL
# =========================
a, b = dead_ends
neighbors[a].append(b)
neighbors[b].append(a)

print("Tunnel reconnected logically.")

# =========================
# 5. CLICK START + DIRECTION
# =========================
fig, ax = plt.subplots(figsize=(12,8))
ax.imshow(img, cmap="gray")
ax.set_title("LEFT CLICK start → direction\nClose after 2 clicks")

clicks = []

def onclick(event):
    if event.xdata is None: return
    clicks.append((int(event.xdata), int(event.ydata)))
    ax.plot(event.xdata, event.ydata, "ro" if len(clicks)==1 else "go")
    plt.draw()

fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()

start = min(pixel_set, key=lambda p:(p[0]-clicks[0][0])**2 + (p[1]-clicks[0][1])**2)
second = min(pixel_set, key=lambda p:(p[0]-clicks[1][0])**2 + (p[1]-clicks[1][1])**2)

# =========================
# 6. PURE GRAPH WALK
# =========================
path = [start]
visited = {start}
prev = start
curr = second

while True:
    path.append(curr)
    visited.add(curr)

    next_nodes = [n for n in neighbors[curr] if n != prev]

    if not next_nodes:
        break

    for n in next_nodes:
        if n not in visited:
            prev, curr = curr, n
            break
    else:
        break

    if curr == start:
        break

path = np.array(path)

# =========================
# 7. SAVE & VERIFY
# =========================
pd.DataFrame(path, columns=["x","y"]).to_csv("track_waypoints.csv", index=False)
print(f"Final track length: {len(path)} points")

plt.figure(figsize=(10,10))
plt.scatter(path[:,0], -path[:,1], c=range(len(path)), s=1, cmap="turbo")
plt.title("Final CLOSED Track (Tunnel Correct)")
plt.axis("equal")
plt.show()
