import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw

# =============================
# GLOBAL DATA LOADING
# =============================
race = pd.read_csv("race_time_interpolated.csv")
track = pd.read_csv("track_waypoints.csv")

# Timing setup
TARGET_UI_FPS = 1
unique_times = np.sort(race["time_sec"].unique())
total_race_sec = unique_times.max() - unique_times.min()
actual_data_fps = len(unique_times) / total_race_sec
SKIP_VAL = max(1, int(actual_data_fps / TARGET_UI_FPS))

PRE_RACE_STAY_SEC = 5
pre_race_frames = PRE_RACE_STAY_SEC * TARGET_UI_FPS
race_times = unique_times[::SKIP_VAL]
times = np.concatenate([np.full(pre_race_frames, race_times[0]), race_times])
race_dict = {t: df for t, df in race.groupby("time_sec")}

drivers = race["driver"].unique()
TRUE_LEN = 5183.7
TIME = 86.288
ROUND_TIME = TIME / 3
TOTAL_RACE_DIST = (TRUE_LEN * 8) / 1000
ROUND_DIST = TRUE_LEN / 3
SURNAME_MAP = {d.split()[-1].upper(): d for d in drivers}

# Track setup
x_coords = track['x'].values
y_coords = -track['y'].values
points = np.column_stack([x_coords, y_coords])
dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
cum_dist_raw = np.insert(np.cumsum(dists), 0, 0)
cum_dist = (cum_dist_raw / cum_dist_raw[-1]) * TRUE_LEN

TOTAL_LAPS = 8
RACE_COMPLETION_KM = (TRUE_LEN * TOTAL_LAPS) / 1000
INSTANT_VANISH_KM = RACE_COMPLETION_KM + (400 / 1000)

dx = np.gradient(x_coords)
dy = np.gradient(y_coords)
normals = np.column_stack([-dy, dx])
normals /= (np.linalg.norm(normals, axis=1)[:, None] + 1e-6)
segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)

sec1_m = cum_dist[:-1] < TRUE_LEN / 3
sec2_m = (cum_dist[:-1] >= TRUE_LEN / 3) & (cum_dist[:-1] < 2 * TRUE_LEN / 3)
sec3_m = cum_dist[:-1] >= 2 * TRUE_LEN / 3

# =============================
# SHARED DATA
# =============================
# Name codes (3-letter codes)
NAME_CODES = {
    "Verstappen": "VER", "Lando Norris": "NOR", "Charles Leclerc": "LEC",
    "Oscar Piastri": "PIA", "Sainz": "SAI", "George Russell": "RUS",
    "Lewis Hamilton": "HAM", "Alonso": "ALO", "Gasly": "GAS",
    "Hulkenberg": "HUL", "Yuki": "TSU", "Stroll": "STR",
    "Ocon": "OCO", "Albon": "ALB", "Bearman": "BEA",
    "Colapinto": "COL", "Lawson": "LAW", "Bortoleto": "BOR",
    "Hadjar": "HAD", "Antonelli": "ANT"
}

# Add PRE-RACE TEAM DATA (2024 season final standings)
PRE_RACE_TEAM_DATA = {
    "McLaren": {"points": 666, "wins": 6, "podiums": 21},
    "Ferrari": {"points": 652, "wins": 5, "podiums": 22},
    "Red Bull": {"points": 589, "wins": 9, "podiums": 18},
    "Mercedes": {"points": 468, "wins": 4, "podiums": 9},
    "Aston Martin": {"points": 94, "wins": 0, "podiums": 0},
    "Alpine": {"points": 65, "wins": 0, "podiums": 2},
    "Haas": {"points": 58, "wins": 0, "podiums": 0},
    "Racing Bulls": {"points": 46, "wins": 0, "podiums": 0},
    "Williams": {"points": 17, "wins": 0, "podiums": 0},
    "Sauber": {"points": 4, "wins": 0, "podiums": 0}
}

# Pre-race team order (by 2024 final standings)
PRE_RACE_TEAM_ORDER = [
    "McLaren", "Ferrari", "Red Bull", "Mercedes", "Aston Martin",
    "Alpine", "Haas", "Racing Bulls", "Williams", "Sauber"
]

GRID_DATA = {
    "Verstappen": (437, "F1-24"), "Lando Norris": (374, "F1-24"), "Charles Leclerc": (356, "F1-24"),
    "Oscar Piastri": (292, "F1-24"), "Sainz": (290, "F1-24"), "George Russell": (245, "F1-24"),
    "Lewis Hamilton": (223, "F1-24"), "Alonso": (70, "F1-24"), "Gasly": (42, "F1-24"),
    "Hulkenberg": (41, "F1-24"), "Yuki": (30, "F1-24"), "Stroll": (24, "F1-24"),
    "Ocon": (23, "F1-24"), "Albon": (12, "F1-24"), "Bearman": (7, "F1-24"),
    "Colapinto": (5, "F1-24"), "Lawson": (4, "F1-24"), "Bortoleto": (214.5, "F2-24"),
    "Hadjar": (192, "F2-24"), "Antonelli": (113, "F2-24")
}

# DRS CONFIG
DP1 = 3510
DP2 = 4550
DRS_ZONE_1 = (3660, 4450)
DRS_ZONE_2_START = (4880, 5183.7)
DRS_ZONE_2_END = (0, 355)

# Load DRS eligibility log and create a simple lookup
try:
    drs_log = pd.read_csv("drs_eligibility_log.csv")
    drs_enabled = {}
    
    for _, row in drs_log.iterrows():
        if row['status'] != 'ELIGIBLE':
            continue
            
        driver = row['driver']
        lap = row['lap']
        zone = 1 if "DP1" in row['detection_point'] else 2
        
        if driver not in drs_enabled:
            drs_enabled[driver] = {}
        if lap not in drs_enabled[driver]:
            drs_enabled[driver][lap] = {}
        
        drs_enabled[driver][lap][zone] = True
    
    print(f"✔ DRS log loaded: {len(drs_log[drs_log['status']=='ELIGIBLE'])} eligible activations")
except Exception as e:
    drs_enabled = {}
    print(f"⚠ Warning: Could not load DRS data: {e}")

POINTS_DATA = {
    "Lando Norris": (25,44,62,77,89,115,133,158,176,176,201,226,250,275,275,293,299,314,332,357,390,390,408,423),
    "Verstappen": (18,36,61,69,87,99,124,136,137,155,155,165,185,187,205,230,255,273,306,321,341,366,396,421),
    "Oscar Piastri": (2,34,49,74,99,131,146,161,186,198,216,234,266,284,309,324,324,336,346,356,366,366,392,410),
    "George Russell": (15,35,45,63,73,93,99,99,111,136,146,147,157,172,184,194,212,237,252,258,276,294,309,319),
    "Charles Leclerc": (4,8,20,32,47,53,61,79,94,104,119,119,139,151,151,163,165,173,192,210,214,226,230,242),
    "Lewis Hamilton": (1,9,15,25,31,41,53,63,71,79,91,103,109,109,109,117,121,125,142,146,148,152,152,156),
    "Antonelli": (12,22,30,30,38,48,48,48,48,63,63,63,63,64,64,66,78,88,89,97,122,137,150,150),
    "Albon": (10,16,18,18,20,30,40,42,42,42,42,46,54,54,64,70,70,70,73,73,73,73,73,73),
    "Sainz": (0,1,1,1,5,7,11,12,12,13,13,13,16,16,16,16,31,32,38,38,38,48,64,64),
    "Alonso": (0,0,0,0,0,0,0,0,2,8,14,16,16,26,30,30,30,36,37,37,40,40,48,56),
    "Hulkenberg": (6,6,6,6,6,6,6,6,16,20,22,37,37,37,37,37,37,37,41,41,43,49,49,51),
    "Hadjar": (0,0,4,4,5,5,7,15,21,21,21,21,22,22,35,36,37,37,37,37,41,49,49,51),
    "Bearman": (0,4,5,6,6,6,6,6,6,6,6,6,8,8,16,16,16,18,20,32,40,41,41,41),
    "Lawson": (0,0,0,0,0,0,0,4,4,4,12,12,16,20,20,20,30,30,30,30,36,36,38,38),
    "Ocon": (0,10,10,14,14,14,14,20,20,22,23,23,27,27,28,28,28,28,28,30,30,32,32,38),
    "Stroll": (8,10,10,10,10,14,14,14,14,14,14,20,20,26,32,32,32,32,32,32,32,32,32,33),
    "Yuki": (0,3,3,5,5,9,10,10,10,10,10,10,10,10,12,12,20,20,28,28,28,28,33,33),
    "Gasly": (0,0,0,6,6,7,7,7,11,11,11,19,20,20,20,20,20,20,20,20,22,22,22,22),
    "Bortoleto": (0,0,0,0,0,0,0,0,0,0,4,4,6,14,14,18,18,18,18,19,19,19,19,19),
    "Colapinto": (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
}

HISTORICAL_PODIUMS = {
    1: ["NORRIS", "VERSTAPPEN", "RUSSELL"], 2: ["PIASTRI", "NORRIS", "RUSSELL"],
    3: ["VERSTAPPEN", "NORRIS", "PIASTRI"], 4: ["PIASTRI", "NORRIS", "RUSSELL"],
    5: ["PIASTRI", "VERSTAPPEN", "LECLERC"], 6: ["PIASTRI", "NORRIS", "RUSSELL"],
    7: ["VERSTAPPEN", "NORRIS", "PIASTRI"], 8: ["NORRIS", "LECLERC", "PIASTRI"],
    9: ["PIASTRI", "NORRIS", "LECLERC"], 10: ["RUSSELL", "VERSTAPPEN", "ANTONELLI"],
    11: ["NORRIS", "PIASTRI", "LECLERC"], 12: ["NORRIS", "PIASTRI", "HULKENBERG"],
    13: ["PIASTRI", "NORRIS", "LECLERC"], 14: ["NORRIS", "PIASTRI", "RUSSELL"],
    15: ["PIASTRI", "VERSTAPPEN", "HADJAR"], 16: ["VERSTAPPEN", "NORRIS", "PIASTRI"],
    17: ["VERSTAPPEN", "RUSSELL", "SAINZ"], 18: ["RUSSELL", "VERSTAPPEN", "NORRIS"],
    19: ["VERSTAPPEN", "NORRIS", "LECLERC"], 20: ["NORRIS", "LECLERC", "VERSTAPPEN"],
    21: ["NORRIS", "ANTONELLI", "VERSTAPPEN"], 22: ["VERSTAPPEN", "RUSSELL", "ANTONELLI"],
    23: ["VERSTAPPEN", "PIASTRI", "SAINZ"], 24: ["VERSTAPPEN", "PIASTRI", "NORRIS"]
}

COLORS = {
    "Verstappen": "#3671C6", "Lando Norris": "#FF8700", "Charles Leclerc": "#E8002D",
    "Oscar Piastri": "#FF8700", "Sainz": "#005AFF", "George Russell": "#27F4D2",
    "Lewis Hamilton": "#E8002D", "Antonelli": "#27F4D2", "Albon": "#005AFF",
    "Alonso": "#229971", "Gasly": "#0090FF", "Hulkenberg": "#2DFF00",
    "Yuki": "#3671C6", "Stroll": "#229971", "Ocon": "#BDC3C7", "Bearman": "#BDC3C7",
    "Colapinto": "#0090FF", "Lawson": "#6692FF", "Bortoleto": "#2DFF00", "Hadjar": "#6692FF"
}

TEAM_MAP = {
    "Verstappen": "Red Bull", "Yuki": "Red Bull",
    "Lando Norris": "McLaren", "Oscar Piastri": "McLaren",
    "Charles Leclerc": "Ferrari", "Lewis Hamilton": "Ferrari",
    "George Russell": "Mercedes", "Antonelli": "Mercedes",
    "Sainz": "Williams", "Albon": "Williams",
    "Alonso": "Aston Martin", "Stroll": "Aston Martin",
    "Gasly": "Alpine", "Colapinto": "Alpine",
    "Ocon": "Haas", "Bearman": "Haas",
    "Lawson": "Racing Bulls", "Hadjar": "Racing Bulls",
    "Bortoleto": "Sauber", "Hulkenberg": "Sauber"
}

DRIVER_TO_TEAM = {
    "VERSTAPPEN": "Red Bull", "YUKI": "Red Bull",
    "NORRIS": "McLaren", "PIASTRI": "McLaren",
    "LECLERC": "Ferrari", "HAMILTON": "Ferrari",
    "RUSSELL": "Mercedes", "ANTONELLI": "Mercedes",
    "SAINZ": "Williams", "ALBON": "Williams",
    "ALONSO": "Aston Martin", "STROLL": "Aston Martin",
    "GASLY": "Alpine", "COLAPINTO": "Alpine",
    "OCON": "Haas", "BEARMAN": "Haas",
    "LAWSON": "Racing Bulls", "HADJAR": "Racing Bulls",
    "BORTOLETO": "Sauber", "HULKENBERG": "Sauber"
}

TEAM_COLORS = {
    "Red Bull": "#3671C6", "McLaren": "#FF8700", "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2", "Williams": "#64C4FF", "Aston Martin": "#229971",
    "Alpine": "#0093CC", "Haas": "#B6BABD", "Racing Bulls": "#6692FF", "Sauber": "#52E252"
}

INITIAL_TEAM_ORDER = [
    "McLaren", "Ferrari", "Red Bull", "Mercedes", "Aston Martin", 
    "Alpine", "Haas", "Racing Bulls", "Williams", "Sauber"
]

LOGO_PATHS = {
    "Red Bull": "logos/redbull.jpg", "McLaren": "logos/mclaren.jpg", "Ferrari": "logos/ferrari.jpg",
    "Mercedes": "logos/mercedes.jpg", "Williams": "logos/williams.jpg", "Aston Martin": "logos/astonmartin.jpg",
    "Alpine": "logos/alpine.jpg", "Haas": "logos/haas.jpg", "Racing Bulls": "logos/racingbulls.jpg", "Sauber": "logos/sauber.jpg"
}
DRIVER_PATHS = {
    "Verstappen": "driver/ver.png", "Lando Norris": "driver/nor.png",
    "Oscar Piastri": "driver/pia.png", "Charles Leclerc": "driver/lec.png",
    "George Russell": "driver/rus.png"
}
CAR_PATHS = {
    "Verstappen": "cars/redbull_car.png", "Lando Norris": "cars/mclaren_car.png",
    "Oscar Piastri": "cars/mclaren_car.png", "Charles Leclerc": "cars/ferrari_car.png",
    "George Russell": "cars/mercedes_car.png"
}

DRIVER_NAME_MAP = {
    "Lando Norris": "NORRIS", "Verstappen": "VERSTAPPEN", "Oscar Piastri": "PIASTRI",
    "Charles Leclerc": "LECLERC", "George Russell": "RUSSELL"
}

TOP_DRIVERS = ["Lando Norris", "Verstappen", "Oscar Piastri", "Charles Leclerc", "George Russell"]
DRIVER_TEAMS = {d: TEAM_MAP[d] for d in TOP_DRIVERS}

POINTS_ADJUSTMENT = {"Red Bull": -3, "Racing Bulls": +3}
MAX_SEASON_POINTS = 423

GP_NAMES = {
    1: "AUSTRALIAN GP", 2: "CHINESE GP & SPRINT", 3: "JAPANESE GP", 4: "BAHRAIN GP",
    5: "SAUDI ARABIAN GP", 6: "MIAMI GP & SPRINT", 7: "EMILIA ROMAGNA GP", 8: "MONACO GP",
    9: "SPANISH GP", 10: "CANADIAN GP", 11: "AUSTRIAN GP", 12: "BRITISH GP",
    13: "BELGIAN GP & SPRINT", 14: "HUNGARIAN GP", 15: "DUTCH GP", 16: "ITALIAN GP",
    17: "AZERBAIJAN GP", 18: "SINGAPORE GP", 19: "US GP & SPRINT", 20: "MEXICAN GP",
    21: "BRAZILIAN GP & SPRINT", 22: "LAS VEGAS GP", 23: "QATAR GP & SPRINT", 24: "ABU DHABI GP"
}

CORNERS = [(412, 1), (780, -1), (935, 1), (1205, -1), (1400, -1), (1590, 1), (2266, -1), (2600, -1), 
           (2720, -1), (2835, 1), (2940, -1), (3045, -1), (3300, -1), (3420, 1), (4477, 1), (4650, 1), (4880, 1)]
DRS_ZONES = [(3560, 4450), (4880, 5183.7), (0, 355)]

# =============================
# HELPER FUNCTIONS
# =============================
def get_pts(driver_name, time):
    pts_tuple = POINTS_DATA.get(driver_name)
    if not pts_tuple: return 0
    raw_rnd = time / ROUND_TIME
    idx = int(raw_rnd)
    frac = raw_rnd - idx
    if idx >= 23: return pts_tuple[23]
    p_start = pts_tuple[idx-1] if idx > 0 else 0
    return p_start + (pts_tuple[idx] - p_start) * frac

def get_historical_stats(completed_rounds):
    stats = {d.upper(): {'wins': 0, 'pods': 0} for d in SURNAME_MAP}
    for r in range(1, completed_rounds + 1):
        if r in HISTORICAL_PODIUMS:
            podium = HISTORICAL_PODIUMS[r]
            stats[podium[0]]['wins'] += 1
            for driver in podium:
                stats[driver]['pods'] += 1
    return stats

def calculate_team_stats(current_round):
    team_wins, team_podiums = {}, {}
    for round_num in range(1, current_round):
        if round_num in HISTORICAL_PODIUMS:
            podium = HISTORICAL_PODIUMS[round_num]
            if len(podium) > 0:
                winner_team = DRIVER_TO_TEAM.get(podium[0])
                if winner_team:
                    team_wins[winner_team] = team_wins.get(winner_team, 0) + 1
            for driver in podium:
                team = DRIVER_TO_TEAM.get(driver)
                if team:
                    team_podiums[team] = team_podiums.get(team, 0) + 1
    return team_wins, team_podiums

def calculate_driver_stats(current_round):
    driver_wins, driver_podiums = {}, {}
    for round_num in range(1, min(current_round + 1, 25)):
        if round_num in HISTORICAL_PODIUMS:
            podium = HISTORICAL_PODIUMS[round_num]
            if len(podium) > 0:
                driver_wins[podium[0]] = driver_wins.get(podium[0], 0) + 1
            for driver in podium:
                driver_podiums[driver] = driver_podiums.get(driver, 0) + 1
    return driver_wins, driver_podiums

def smooth_transition(progress, start, peak, end):
    if progress < start:
        return progress / start if start > 0 else 0
    elif progress < peak:
        return 1.0
    elif progress < end:
        return 1.0 - ((progress - peak) / (end - peak))
    else:
        return 0

def get_text_color(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
    return "black" if brightness > 0.5 else "white"

def create_circular_logo(path, size=200):
    try:
        img = Image.open(path).convert("RGBA")
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        output = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        output.paste(img, (0, 0))
        output.putalpha(mask)
        return output
    except:
        blank = Image.new('RGBA', (size, size), (255, 255, 255, 50))
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        blank.putalpha(mask)
        return blank

def load_car_image(path, size=200):
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return img
    except:
        return Image.new('RGBA', (size, int(size/2)), (255, 255, 255, 200))

def create_checkered_pattern(height, square_size=0.15):
    pattern = []
    num_squares = int(height / square_size) + 1
    for i in range(num_squares):
        y_pos = i * square_size
        color = 'white' if i % 2 == 0 else 'black'
        pattern.append((y_pos, square_size, color))
    return pattern

def check_drs_active(driver, lap_m, current_lap):
    """Check if driver has DRS active at current position."""
    if not drs_enabled or driver not in drs_enabled:
        return False
    
    if current_lap not in drs_enabled[driver]:
        return False
    
    # DRS Zone 1: 3660-4450m (activated by DP1 at 3510m)
    if DRS_ZONE_1[0] <= lap_m <= DRS_ZONE_1[1]:
        return drs_enabled[driver][current_lap].get(1, False)
    
    # DRS Zone 2: 4880-5183.7m, then wraps to 0-355m (activated by DP2 at 4550m)
    if DRS_ZONE_2_START[0] <= lap_m <= DRS_ZONE_2_START[1]:
        return drs_enabled[driver][current_lap].get(2, False)
    
    if DRS_ZONE_2_END[0] <= lap_m <= DRS_ZONE_2_END[1]:
        prev_lap = current_lap - 1
        if prev_lap in drs_enabled[driver]:
            return drs_enabled[driver][prev_lap].get(2, False)
    
    return False

# Helper function to load driver photo
def load_driver_photo(path, size=200):
    """Load driver photo or create team color placeholder"""
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return img
    except:
        # Create simple placeholder
        img = Image.new('RGBA', (size, size), (100, 100, 100, 255))
        return img

# =============================
# MAIN FIGURE (16:9 RATIO)
# =============================
fig = plt.figure(figsize=(16, 9), facecolor='black')

# Region 1: Leaderboard (x=[0, 48/160], y=[0, 90/90])
ax1 = fig.add_axes([0, 0, 48/160, 1])
ax1.set_facecolor('#000000')
ax1.axis([0, 100, 0, 100])
ax1.axis('off')

# Region 2: Track Map (x=[48/160, 1], y=[21/90, 1])
ax2 = fig.add_axes([48/160, 21/90, 112/160, 69/90])
ax2.set_facecolor('black')
ax2.axis('off')

# Region 3: Constructor's Championship (x=[48/160, 80/160], y=[0, 45/90])
ax3 = fig.add_axes([48/160, 0, 32/160, 45/90])
ax3.set_facecolor('black')

# Region 4: Racing Track (x=[80/160, 1], y=[0, 21/90])
ax4 = fig.add_axes([80/160, 0, 80/160, 21/90])
ax4.set_facecolor('black')

# =============================
# REGION 1 SETUP: LEADERBOARD
# =============================
ax1.add_patch(plt.Rectangle((1, 1), 98, 98, color='#0a0a0a', alpha=0.95, zorder=0))
ax1.add_patch(plt.Rectangle((1, 1), 98, 98, color='none', ec='#333', lw=2.5, zorder=1))
ax1.add_patch(plt.Rectangle((1, 92), 98, 7, color='#1a1a1a', zorder=1))
ax1.axhline(y=92, color='#FF1E00', linewidth=4, alpha=0.9, zorder=2)

# Header text - SIMPLIFIED (no round info)
lap_label = ax1.text(4, 94, "F1 2025", color="#ffffff", fontsize=32, weight='heavy', zorder=3)
time_label = ax1.text(96, 95, "00:00.000", color="#999", fontsize=20, family='monospace', ha='right', zorder=3)

h_x = [5, 16, 31.5, 50.5, 71, 80, 90]
headers = ["POS", "DRIVER", "GAP", "INT", "PTS", "Q", "EVT"]
header_objs = [ax1.text(x, 89.2, txt, color="#555", fontsize=10.5, weight='bold', zorder=2) for txt, x in zip(headers, h_x)]

rows = []
BASE_Y = 86.0
ROW_SPACING = 4.25
finish_times = {}

# Position animation tracking
prev_positions = {}
target_positions = {}
current_y_positions = {}
POSITION_ANIMATION_SPEED = 0.7

# Gap/Interval update throttling
gap_update_counter = 0
GAP_UPDATE_INTERVAL = 6
cached_gaps = {}

for i in range(20):
    y = BASE_Y - (i * ROW_SPACING)
    bg = ax1.add_patch(plt.Rectangle((2, y-1.5), 96, 4.0, color='#0d0d0d', alpha=0.6, zorder=1))
    bg_h = ax1.add_patch(plt.Rectangle((2, y-1.5), 96, 4.0, color='#ffffff', alpha=0, zorder=2))
    border = ax1.add_patch(plt.Rectangle((2, y-1.5), 96, 4.0, color='none', ec='#222', lw=1.5, zorder=10))
    cbar = ax1.add_patch(plt.Rectangle((12.2, y-1.5), 1.3, 4.0, color='#333', zorder=3))
    pos = ax1.text(h_x[0]+1.5, y, "", color='#fff', fontsize=13, weight='bold', ha='center', zorder=4)
    name = ax1.text(h_x[1]+1, y, "", color='#fff', fontsize=15, weight='bold', zorder=4)
    gap = ax1.text(h_x[2], y, "", color='#aaa', fontsize=11.5, family='monospace', weight='bold', zorder=4)
    inter = ax1.text(h_x[3], y, "", color='#aaa', fontsize=11.5, family='monospace', weight='bold', zorder=4)
    pts = ax1.text(h_x[4]+2, y, "", color='#fff', fontsize=13, weight='bold', ha='center', zorder=4)
    wins = ax1.text(h_x[5]+1, y, "", color='#ccc', fontsize=11, ha='center', zorder=4)
    pod = ax1.text(h_x[6]+1, y, "", color='#ccc', fontsize=11, ha='center', zorder=4)
    rows.append({'pos': pos, 'name': name, 'gap': gap, 'int': inter, 'pts': pts, 'wins': wins, 'pod': pod, 
                 'cbar': cbar, 'bg_highlight': bg_h, 'border': border, 'bg': bg, 'base_y': y,
                 'last_gap_text': '', 'last_int_text': ''})

# =============================
# REGION 2 SETUP: TRACK MAP
# =============================
ax2.add_collection(LineCollection(segments, linewidths=16, color='white', alpha=0.08, zorder=1))
sector_colors = ['#ce161b' if d < TRUE_LEN/3 else '#169cc5' if d < 2*TRUE_LEN/3 else '#efd035' for d in cum_dist[:-1]]
lc_sector = LineCollection(segments, linewidths=3.5, colors=sector_colors, zorder=3)
ax2.add_collection(lc_sector)

glow_colors = ['#ce161b', '#169cc5', '#efd035']
sector_glows = [LineCollection(segments[m], linewidths=12, colors=glow_colors[i], alpha=0, zorder=2) 
                for i, m in enumerate([sec1_m, sec2_m, sec3_m])]
for glow in sector_glows: 
    ax2.add_collection(glow)

full_track_glow = LineCollection(segments, linewidths=14, color='white', alpha=0.04, zorder=2)
ax2.add_collection(full_track_glow)

for start, end in DRS_ZONES:
    idx = np.searchsorted(cum_dist, start)
    px, py = x_coords[idx] + normals[idx, 0]*40, y_coords[idx] + normals[idx, 1]*40
    ax2.text(px, py, "DRS", color='#00fb0c', fontsize=10, weight='black', ha='center', va='center', zorder=10, alpha=0.8)
    mask = (cum_dist >= start) & (cum_dist <= end)
    ax2.plot(x_coords[mask] + normals[mask][:, 0]*10, y_coords[mask] + normals[mask][:, 1]*10, 
             color='#00fb0c', lw=2, alpha=0.4, zorder=2)

for i, (dist_m, side) in enumerate(CORNERS, 1):
    idx = np.searchsorted(cum_dist, dist_m)
    px, py = x_coords[idx] + normals[idx, 0]*(side*24), y_coords[idx] + normals[idx, 1]*(side*24)
    ax2.add_patch(plt.Circle((px, py), 9, color='black', ec='white', lw=1, zorder=5))
    ax2.text(px, py, str(i), color='white', fontsize=8, weight='bold', ha='center', va='center', zorder=6)

ax2.scatter(x_coords[0], y_coords[0], color='white', s=100, marker='|', zorder=10, lw=2)
margin = 50
ax2.set_xlim(x_coords.min() - margin, x_coords.max() + margin)
ax2.set_ylim(y_coords.min() - margin, y_coords.max() + margin)

title_main = ax2.text(0.95, 0.88, "FORMULA 1 2025 SEASON", color="white", transform=ax2.transAxes, 
                      fontsize=28, weight='bold', ha='right')
title_round = ax2.text(0.95, 0.84, "", color="white", transform=ax2.transAxes, fontsize=18, weight='bold', ha='right')
title_gp = ax2.text(0.95, 0.80, "", color="white", transform=ax2.transAxes, fontsize=20, weight='bold', ha='right')

car_dots = {d: ax2.plot([], [], "o", ms=24, color=COLORS.get(d, "white"), mec='black', mew=1.5, zorder=100)[0] 
            for d in drivers}
car_labs = {d: ax2.text(0, 0, d.split()[-1][:3].upper(), color=get_text_color(COLORS.get(d, "#FFFFFF")), 
                        fontsize=9, weight='bold', ha='center', va='center', zorder=101) for d in drivers}

# =============================
# REGION 3 SETUP: CONSTRUCTOR'S
# =============================
bar_width_team = 0.7
ax3.axis('off')

bars_team = {t: ax3.add_patch(plt.Rectangle((0, 0), bar_width_team, 0, color=TEAM_COLORS[t], alpha=0.9, zorder=3)) 
             for t in INITIAL_TEAM_ORDER}
point_labels_team = {t: ax3.text(0, 5, "0", color='white', ha='center', va='bottom', fontsize=11, weight='bold') 
                     for t in INITIAL_TEAM_ORDER}

logo_boxes_team = {}
LOGO_SIZE_TEAM = 200
fig_width_inches = 16
axes_width_inches = fig_width_inches * (32/160)
dpi = fig.dpi
axes_width_pixels = axes_width_inches * dpi
pixels_per_data_unit = axes_width_pixels / 10
desired_logo_pixels = bar_width_team * pixels_per_data_unit
LOGO_ZOOM_TEAM = desired_logo_pixels / LOGO_SIZE_TEAM

for team in INITIAL_TEAM_ORDER:
    circular_logo = create_circular_logo(LOGO_PATHS[team], size=LOGO_SIZE_TEAM)
    logo_img = OffsetImage(circular_logo, zoom=LOGO_ZOOM_TEAM)
    ab = AnnotationBbox(logo_img, (0, 0), frameon=False, zorder=5, xycoords='data', box_alignment=(0.5, 0.5))
    logo_boxes_team[team] = ax3.add_artist(ab)

ax3.set_xlim(-0.5, 9.5)

stats_boxes_team = {}
stats_texts_team = {}
connection_lines_team = {}
prev_team_positions = {}
current_team_x_positions = {}
TEAM_POSITION_ANIMATION_SPEED = 0.5
pre_race_team_texts = {}

# =============================
# REGION 4 SETUP: RACING TRACK
# =============================
ax4.axis('off')

# Track constants (in km)
TRACK_LENGTH_KM = 1.75
TOTAL_RACE_KM = 41.4696
LEADER_POSITION_KM = 1.54  # Leader stays at this position on track during race

# Layout constants (normalized 0-1 within Region 4)
BOTTOM_BORDER_H = 1/21
TRACK_BORDER_START = 1/21
TRACK_BORDER_H = 1.5/21
LANE_START = 2.5/21
LANE_HEIGHT = 4/21
CARD_SPACE = 1/21
CARD_START = 17/21
CARD_HEIGHT = 4/21
TOP_BORDER_START = 20/21

# Horizontal constants (within Region 4 width)
LEFT_BORDER_W = 2/80
TRACK_LENGTH_NORM = 76/80
RIGHT_BORDER_W = 2/80
CARD_WIDTH_NORM = 24/80
START_LINE_NORM = (110-80)/80  # Starting line position
LEADER_RACE_NORM = (154-80)/80  # Leader position during race

# Set axis limits
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

# Track border stripes (top and bottom of track, will change color based on position)
track_border_stripes_top = []
track_border_stripes_bottom = []
num_stripes = 80
stripe_width = TRACK_LENGTH_NORM / num_stripes

for i in range(num_stripes):
    x_pos = LEFT_BORDER_W + (i * stripe_width)
    color = 'white' if i % 2 == 0 else '#ce161b'  # Default sector 1 colors
    
    # Top stripe (above track lanes)
    stripe_top = ax4.add_patch(Rectangle((x_pos, LANE_START + (LANE_HEIGHT * 3)), 
                                         stripe_width, TRACK_BORDER_H,
                                         color=color, zorder=2, alpha=0.5))
    track_border_stripes_top.append(stripe_top)
    
    # Bottom stripe (below track lanes)
    stripe_bottom = ax4.add_patch(Rectangle((x_pos, TRACK_BORDER_START), 
                                            stripe_width, TRACK_BORDER_H,
                                            color=color, zorder=2, alpha=0.5))
    track_border_stripes_bottom.append(stripe_bottom)

# Three lanes
lane_patches = []
lane_center_lines = []
for i in range(3):
    lane_y = LANE_START + (i * LANE_HEIGHT)
    lane = ax4.add_patch(Rectangle((LEFT_BORDER_W, lane_y), TRACK_LENGTH_NORM, LANE_HEIGHT,
                                  color='#2a2a2a', ec='#444', linewidth=1, zorder=3, alpha=0.5))
    lane_patches.append(lane)
    
    # Center line for each lane
    center_y = lane_y + (LANE_HEIGHT / 2)
    lane_center_lines.append(center_y)

# Starting/finish line (vertical white line) - starts at P1 grid position
P1_GRID_CENTER = (21/80 + 30/80) / 2  # Center of P1 grid position
start_finish_line = ax4.add_patch(Rectangle((P1_GRID_CENTER - 0.002, LANE_START), 0.004, 
                                            LANE_HEIGHT * 3, color='white', zorder=15, alpha=0.5))

# Progress bars storage
progress_bars = {d: None for d in TOP_DRIVERS}

# Driver card elements
driver_cards = {}
driver_photos = {}
driver_name_texts = {}
driver_stats_texts = {}
driver_car_images = {}

# Initialize driver elements (will be positioned dynamically)
for driver in TOP_DRIVERS:
    # Card background (positioned at top)
    card = ax4.add_patch(Rectangle((0, CARD_START), CARD_WIDTH_NORM, CARD_HEIGHT,
                                   color='#0d0d0d', ec=TEAM_COLORS[DRIVER_TEAMS[driver]],
                                   linewidth=2, alpha=0.9, zorder=20))
    driver_cards[driver] = card
    
    # Driver photo (left side of card)
    photo_path = DRIVER_PATHS.get(driver, "")
    photo_img = load_driver_photo(photo_path, size=200)
    photo_offset = OffsetImage(photo_img, zoom=0.12)
    photo_box = AnnotationBbox(photo_offset, (0, CARD_START + CARD_HEIGHT/2),
                              frameon=False, zorder=21, xycoords='data', 
                              box_alignment=(0.5, 0.5))
    driver_photos[driver] = ax4.add_artist(photo_box)
    
    # Driver surname (centered in card height)
    surname = driver.split()[-1].upper()
    name_text = ax4.text(0, CARD_START + CARD_HEIGHT*0.65, surname,
                        color=TEAM_COLORS[DRIVER_TEAMS[driver]], fontsize=10, weight='bold',
                        ha='left', va='center', zorder=22)
    driver_name_texts[driver] = name_text
    
    # Stats (1st/2nd/3rd positions) on same line as name
    stats_text = ax4.text(0, CARD_START + CARD_HEIGHT*0.25, "",
                         color='#aaa', fontsize=8, weight='normal',
                         ha='left', va='center', zorder=22)
    driver_stats_texts[driver] = stats_text
    
    # Car image (on track)
    car_img = load_car_image(CAR_PATHS.get(driver), size=300)
    car_offset = OffsetImage(car_img, zoom=0.10)
    car_box = AnnotationBbox(car_offset, (0, 0), frameon=False, zorder=100,
                            xycoords='data', box_alignment=(0.5, 0.5))
    driver_car_images[driver] = ax4.add_artist(car_box)

# Animation state
track_scroll_offset = 0
finish_line_x = None  # Will move from right when race is ending
start_line_x = P1_GRID_CENTER  # Track start line position for smooth transition
start_line_target_x = P1_GRID_CENTER  # Target position for start line
START_LINE_TRANSITION_SPEED = 0.15  # Slower transition for smoother movement

# Points display (at car tip instead of card)
car_points_texts = {}
for driver in TOP_DRIVERS:
    pts_text = ax4.text(0, 0, "", color='white', fontsize=8, weight='bold',
                       ha='left', va='center', zorder=102,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                edgecolor='white', alpha=0.8, linewidth=1))
    car_points_texts[driver] = pts_text

# DRS bars behind cars
drs_bars = {}
for driver in TOP_DRIVERS:
    drs_bar = ax4.add_patch(Rectangle((0, 0), 0, 0, color='#00fb0c', 
                                     alpha=0, zorder=99))
    drs_bars[driver] = drs_bar

# Position labels for cards (moved to right side to avoid photo overlap)
card_position_labels = {}
for driver in TOP_DRIVERS:
    pos_label = ax4.text(0, CARD_START + CARD_HEIGHT*0.60, "", 
                        color='white', fontsize=9, weight='heavy',
                        ha='right', va='center', zorder=23,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                                 edgecolor='white', alpha=0.9, linewidth=1.5))
    card_position_labels[driver] = pos_label

def calculate_podium_finishes(driver_surname, current_round):
    """Calculate number of 1st, 2nd, 3rd finishes for a driver"""
    first = second = third = 0
    for round_num in range(1, min(current_round + 1, 25)):
        if round_num in HISTORICAL_PODIUMS:
            podium = HISTORICAL_PODIUMS[round_num]
            if len(podium) > 0 and podium[0] == driver_surname:
                first += 1
            if len(podium) > 1 and podium[1] == driver_surname:
                second += 1
            if len(podium) > 2 and podium[2] == driver_surname:
                third += 1
    return first, second, third

print("✔ Region 4 racing track setup complete")

# =============================
# ANIMATION UPDATE
# =============================
START_DELAY_FRAMES = int(5 * 5)

def update(frame):
    # Declare all global variables at the start
    global gap_update_counter, cached_gaps, prev_positions, target_positions, current_y_positions
    global prev_team_positions, current_team_x_positions, pre_race_team_texts
    global track_scroll_offset, finish_line_x, start_line_x, start_line_target_x
    
    # Determine time and race state
    if frame < START_DELAY_FRAMES:
        t = times[0]
        is_countdown = True
        is_pre_race = True
    else:
        race_frame = frame - START_DELAY_FRAMES
        if race_frame < pre_race_frames:
            t = times[race_frame]
            is_pre_race = True
            is_countdown = False
        else:
            idx = race_frame - pre_race_frames
            if idx >= len(race_times):
                idx = len(race_times) - 1
            t = race_times[idx]
            is_pre_race = False
            is_countdown = False
    
    snap = race_dict[t].set_index("driver")
    
    # ==================
    # REGION 1: LEADERBOARD
    # ==================
    
    if is_pre_race:
        standings = snap.reindex(GRID_DATA.keys())
        lap_label.set_text("GRID PREVIEW")
        time_label.set_text("T=")
        header_objs[5].set_text("Q")
        header_objs[6].set_text("EVT")
    else:
        standings = snap.sort_values("distance_km", ascending=False)
        header_objs[5].set_text("W")
        header_objs[6].set_text("POD")
    
    leader_dist = standings.iloc[0]["distance_km"]
    leader_finished = leader_dist >= TOTAL_RACE_DIST
    
    raw_progress = (leader_dist * 1000) / ROUND_DIST
    completed_rounds = max(0, int(raw_progress))
    current_round = completed_rounds + 1
    round_progress = raw_progress - completed_rounds
    
    if not is_pre_race:
        lap = min(8, int((leader_dist * 1000) // TRUE_LEN) + 1)
        lap_label.set_text(f"LAP {lap}/8")
        mins, secs = divmod(t, 60)
        time_label.set_text(f"T={int(mins):02d}:{int(secs):02d}.{int((secs % 1) * 1000):03d}")
    
    # Gap update throttling
    should_update_gaps = (gap_update_counter % GAP_UPDATE_INTERVAL == 0)
    gap_update_counter += 1
    
    # Podium animation
    popup_intensity = smooth_transition(round_progress, 0.15, 0.70, 0.98)
    p1_reveal = max(0, min(1, (round_progress - 0.70) / 0.08))
    p2_reveal = max(0, min(1, (round_progress - 0.78) / 0.08))
    p3_reveal = max(0, min(1, (round_progress - 0.86) / 0.08))
    p_reveals = [p1_reveal, p2_reveal, p3_reveal]
    p_colors = ['#FFD700', '#C0C0C0', '#CD7F32']
    
    stats_data = get_historical_stats(completed_rounds)
    current_podium = HISTORICAL_PODIUMS.get(current_round, [])
    
    # Position tracking for smooth animation
    for idx, d in enumerate(standings.index):
        new_pos = idx + 1
        if d not in prev_positions:
            prev_positions[d] = new_pos
            current_y_positions[d] = BASE_Y - (idx * ROW_SPACING)
        target_positions[d] = BASE_Y - (idx * ROW_SPACING)
    
    for idx, (d, row) in enumerate(standings.iterrows()):
        r = rows[idx]
        surname = d.split()[-1].upper()
        name_code = NAME_CODES.get(d, d[:3].upper())
        dist_km = row["distance_km"]
        time_s = row["time_sec"]
        is_finished = dist_km >= TOTAL_RACE_DIST
        driver_color = COLORS.get(d, "#fff")
        
        # DRS check
        lap_m = (dist_km * 1000) % TRUE_LEN
        d_lap = int((dist_km * 1000) // TRUE_LEN) + 1
        has_drs = False
        if not is_pre_race and not is_finished and d_lap > 1:
            has_drs = check_drs_active(d, lap_m, d_lap)
        
        # SMOOTH POSITION ANIMATION
        if d in current_y_positions and d in target_positions:
            current_y = current_y_positions[d]
            target_y = target_positions[d]
            diff = target_y - current_y
            current_y_positions[d] = current_y + (diff * POSITION_ANIMATION_SPEED)
            current_y = current_y_positions[d]
        else:
            current_y = BASE_Y - (idx * ROW_SPACING)
            current_y_positions[d] = current_y
        
        position_changed = (prev_positions.get(d, idx+1) != idx+1)
        
        r['bg'].set_xy((2, current_y - 1.5))
        r['bg_highlight'].set_xy((2, current_y - 1.5))
        r['border'].set_xy((2, current_y - 1.5))
        r['cbar'].set_xy((12.2, current_y - 1.5))
        
        r['pos'].set_y(current_y)
        r['name'].set_y(current_y)
        r['gap'].set_y(current_y)
        r['int'].set_y(current_y)
        r['pts'].set_y(current_y)
        r['wins'].set_y(current_y)
        r['pod'].set_y(current_y)
        
        # Reset
        r['bg_highlight'].set_alpha(0)
        r['border'].set_edgecolor('#222')
        r['border'].set_linewidth(1.5)
        r['name'].set_fontsize(15)
        r['name'].set_weight('bold')
        r['bg'].set_alpha(0.6)
        r['gap'].set_color('#aaa')
        r['int'].set_color('#aaa')
        
        r['pos'].set_text(f"{idx+1}")
        r['name'].set_text(name_code)
        r['name'].set_color(driver_color)
        r['cbar'].set_color(driver_color)
        
        if is_pre_race:
            r['gap'].set_text("—")
            r['int'].set_text("—")
            r['pts'].set_text("0")
            r['wins'].set_text(f"{GRID_DATA[d][0]}")
            r['pod'].set_text(f"{GRID_DATA[d][1]}")
        else:
            r['pts'].set_text(f"{int(get_pts(d, time_s))}")
            r['wins'].set_text(f"{stats_data[surname]['wins']}")
            r['pod'].set_text(f"{stats_data[surname]['pods']}")
            
            if is_finished:
                if d not in finish_times:
                    finish_times[d] = t
                if idx == 0:
                    m, s = divmod(finish_times[d], 60)
                    gap_text = f"{int(m):02d}:{s:06.3f}"
                else:
                    time_delta = finish_times[d] - finish_times[standings.index[0]]
                    gap_text = f"+{time_delta:.3f}"
                int_text = "FINISHED"
                r['gap'].set_text(gap_text)
                r['int'].set_text(int_text)
                cached_gaps[d] = {'gap': gap_text, 'int': int_text}
            else:
                if idx == 0:
                    gap_text = "LEADER"
                    int_text = "—"
                    r['gap'].set_text(gap_text)
                    r['int'].set_text(int_text)
                    cached_gaps[d] = {'gap': gap_text, 'int': int_text}
                else:
                    # Calculate/update gaps on throttle
                    if should_update_gaps or d not in cached_gaps:
                        dist_gap = leader_dist - dist_km
                        time_gap = dist_gap * 18
                        
                        # +LAP logic
                        if time_gap > TIME:
                            laps_down = int(time_gap / TIME)
                            gap_text = f"+{laps_down} LAP" if laps_down == 1 else f"+{laps_down} LAPS"
                        else:
                            gap_text = f"+{time_gap:.3f}"
                        
                        prev_dist = standings.iloc[idx-1]["distance_km"]
                        int_gap = prev_dist - dist_km
                        int_time = int_gap * 18
                        int_text = f"+{int_time:.3f}"
                        
                        cached_gaps[d] = {'gap': gap_text, 'int': int_text}
                    
                    r['gap'].set_text(cached_gaps[d]['gap'])
                    r['int'].set_text(cached_gaps[d]['int'])
                
                # Add subtle flicker to gaps after leader finishes
                if leader_finished and idx > 0 and not is_finished:
                    if d in cached_gaps:
                        if 'LAP' not in cached_gaps[d]['gap']:
                            gap_str = cached_gaps[d]['gap']
                            if gap_str.startswith('+'):
                                try:
                                    base_gap = float(gap_str[1:])
                                    if base_gap < TIME:
                                        jitter = (np.random.random() - 0.5) * 0.006
                                        flickered_gap = base_gap + jitter
                                        r['gap'].set_text(f"+{flickered_gap:.3f}")
                                except:
                                    pass
                        
                        int_str = cached_gaps[d]['int']
                        if int_str.startswith('+'):
                            try:
                                base_int = float(int_str[1:])
                                jitter = (np.random.random() - 0.5) * 0.006
                                flickered_int = base_int + jitter
                                r['int'].set_text(f"+{flickered_int:.3f}")
                            except:
                                pass
                
                # DRS HIGHLIGHTING
                if has_drs:
                    r['gap'].set_color('#00fb0c')
                    r['int'].set_color('#00fb0c')
                
                # Last lap highlight
                if d_lap >= 8:
                    r['border'].set_edgecolor('#C5C6C7')
                    r['border'].set_linewidth(1.5)
                
                # Podium animation
                if surname in current_podium and not leader_finished:
                    pod_pos = current_podium.index(surname)
                    base_alpha = 0.08 * popup_intensity
                    
                    if p_reveals[pod_pos] > 0:
                        r['bg_highlight'].set_color(p_colors[pod_pos])
                        r['bg_highlight'].set_alpha(base_alpha + 0.18 * p_reveals[pod_pos])
                    else:
                        if not has_drs:
                            r['bg_highlight'].set_color('#ffffff')
                            r['bg_highlight'].set_alpha(base_alpha)
                    
                    size_boost = popup_intensity * 2.5
                    r['name'].set_fontsize(15 + size_boost)
                    r['name'].set_weight('heavy' if popup_intensity > 0.5 else 'bold')
    
    # Update previous positions
    for idx, d in enumerate(standings.index):
        prev_positions[d] = idx + 1
    
    # ==================
    # REGION 2: TRACK MAP
    # ==================
    curr_round_num = int((leader_dist * 1000) // (TRUE_LEN / 3)) + 1
    rnd_clamped = min(24, curr_round_num)
    active_sec = (rnd_clamped - 1) % 3
    
    if not is_countdown:
        if curr_round_num > 24:
            for glow in sector_glows: glow.set_alpha(0)
            full_track_glow.set_alpha(0.6)
            title_round.set_color("white")
        else:
            title_round.set_color(glow_colors[active_sec])
        
        title_round.set_text(f"ROUND {rnd_clamped}/24")
        title_gp.set_text(GP_NAMES.get(rnd_clamped, ""))
        for idx, glow in enumerate(sector_glows):
            glow.set_alpha(0.8 if idx == active_sec else 0.05)
    else:
        title_round.set_text("STARTING GRID")
        title_gp.set_text("QUALIFYING RESULTS")
        for glow in sector_glows: glow.set_alpha(0.0)
    
    for idx, (d, row) in enumerate(standings.iterrows()):
        total_dist_km = row["distance_km"]
        curr_px_idx = np.searchsorted(cum_dist, (total_dist_km % (TRUE_LEN/1000)) * 1000)
        curr_px_idx = min(curr_px_idx, len(points)-1)
        x, y = points[curr_px_idx]
        
        is_lap_8 = (total_dist_km * 1000 >= 36285.9) and (total_dist_km * 1000 < 41469.6)
        size = 34 if is_lap_8 else 24
        f_size = 12 if is_lap_8 else 9
        
        alpha_val = 1.0 if total_dist_km < INSTANT_VANISH_KM else 0.0
        z_base = 100 + (len(drivers) - idx) * 2
        car_dots[d].set_data([x], [y])
        car_dots[d].set_alpha(alpha_val)
        car_dots[d].set_markersize(size)
        car_dots[d].set_zorder(z_base)
        car_labs[d].set_position((x, y))
        car_labs[d].set_alpha(alpha_val)
        car_labs[d].set_fontsize(f_size)
        car_labs[d].set_zorder(z_base + 1)
    
    # ==================
    # REGION 3: CONSTRUCTOR'S
    # ==================
    
    if is_pre_race:
        # Clear any previous pre-race texts
        for team in pre_race_team_texts:
            for text_obj in pre_race_team_texts[team]:
                text_obj.remove()
        pre_race_team_texts.clear()
        
        # Clear race-mode stats
        for box in stats_boxes_team.values():
            box.remove()
        for text_obj in stats_texts_team.values():
            if isinstance(text_obj, tuple):
                for txt in text_obj:
                    txt.remove()
            else:
                text_obj.remove()
        for line in connection_lines_team.values():
            line.remove()
        stats_boxes_team.clear()
        stats_texts_team.clear()
        connection_lines_team.clear()
        
        # Set y-axis to accommodate 2024 points + space for vertical text
        max_pre_race_pts = 700
        ax3.set_ylim(0, max_pre_race_pts)
        
        # Display each team
        for rank, team in enumerate(PRE_RACE_TEAM_ORDER):
            pts_2024 = PRE_RACE_TEAM_DATA[team]["points"]
            wins = PRE_RACE_TEAM_DATA[team]["wins"]
            podiums = PRE_RACE_TEAM_DATA[team]["podiums"]
            
            bars_team[team].set_x(rank - bar_width_team/2)
            bars_team[team].set_height(pts_2024)
            
            if team in logo_boxes_team:
                logo_boxes_team[team].xybox = (rank, pts_2024)
                logo_boxes_team[team].set_visible(True)
            
            point_labels_team[team].set_x(rank)
            point_labels_team[team].set_y(pts_2024 + (max_pre_race_pts * 0.04))
            point_labels_team[team].set_text("0")
            
            text_parts = [f"{team}: {pts_2024}"]
            if wins > 0:
                text_parts.append(f"W:{wins}")
            if podiums > 0:
                text_parts.append(f"P:{podiums}")
            text_string = " ".join(text_parts)
            
            text_y_start = pts_2024 + (max_pre_race_pts * 0.1)
            
            vertical_text = ax3.text(rank, text_y_start, text_string,
                                    color=TEAM_COLORS[team], ha='left', va='bottom',
                                    fontsize=9, weight='normal', rotation=90, zorder=6)
            
            pre_race_team_texts[team] = [vertical_text]
    
    else:
        # RACE MODE: Normal bars with stats boxes
        
        # Clear pre-race texts
        for team in pre_race_team_texts:
            for text_obj in pre_race_team_texts[team]:
                text_obj.remove()
        pre_race_team_texts.clear()
        
        snap_copy = snap.copy()
        snap_copy['p_calc'] = snap_copy.apply(lambda r: get_pts(r.name, t), axis=1)
        snap_copy['team'] = snap_copy.index.map(TEAM_MAP)
        team_data = snap_copy.groupby('team')['p_calc'].sum().to_dict()
        
        adjustment_applied = False
        current_round_team = completed_rounds + 1
        if current_round_team == 2 and not adjustment_applied:
            for team, adjustment in POINTS_ADJUSTMENT.items():
                if team in team_data:
                    team_data[team] += adjustment
            adjustment_applied = True
        
        team_wins, team_podiums = calculate_team_stats(current_round_team)
        sorted_teams = sorted(team_data.keys(), key=lambda x: team_data[x], reverse=True)
        max_pts_team = max(team_data.values()) if max(team_data.values()) > 0 else 100
        ax3.set_ylim(0, max_pts_team * 1.1)
        
        # Clear previous stats boxes
        for box in stats_boxes_team.values():
            box.remove()
        for text_obj in stats_texts_team.values():
            if isinstance(text_obj, tuple):
                for txt in text_obj:
                    txt.remove()
            else:
                text_obj.remove()
        for line in connection_lines_team.values():
            line.remove()
        stats_boxes_team.clear()
        stats_texts_team.clear()
        connection_lines_team.clear()
        
        bar_positions_team = {}
        
        # Initialize position tracking
        for rank, team in enumerate(sorted_teams):
            if team not in prev_team_positions:
                prev_team_positions[team] = rank
                current_team_x_positions[team] = rank
        
        # Update bar positions with smooth animation
        for rank, team in enumerate(sorted_teams):
            pts = team_data[team]
            
            # Smooth position animation
            if team in current_team_x_positions:
                current_x = current_team_x_positions[team]
                target_x = rank
                diff = target_x - current_x
                current_team_x_positions[team] = current_x + (diff * TEAM_POSITION_ANIMATION_SPEED)
                animated_x = current_team_x_positions[team]
            else:
                animated_x = rank
                current_team_x_positions[team] = rank
            
            bar_positions_team[team] = (animated_x, pts)
            
            # Main bar
            bars_team[team].set_x(animated_x - bar_width_team/2)
            bars_team[team].set_height(pts)
            
            # Point label
            point_labels_team[team].set_x(animated_x)
            point_labels_team[team].set_y(pts + (max_pts_team * 0.04))
            point_labels_team[team].set_text(f"{int(pts)}")
            
            # Update logo position
            if team in logo_boxes_team:
                logo_boxes_team[team].xybox = (animated_x, pts)
                logo_boxes_team[team].set_visible(True)
        
        # Update previous positions
        for rank, team in enumerate(sorted_teams):
            prev_team_positions[team] = rank
        
        # STATS BOXES
        stats_x_left = 5.5
        stats_x_right = 8.75
        stats_y_bottom_ratio = 11/45
        stats_y_top_ratio = 44/45
        
        stats_y_bottom = max_pts_team * stats_y_bottom_ratio
        stats_y_top = max_pts_team * stats_y_top_ratio
        
        y_offset_team = 0
        box_width_team_stat = stats_x_right - stats_x_left
        available_height = stats_y_top - stats_y_bottom
        
        teams_with_stats = sum(1 for team in sorted_teams 
                              if team_wins.get(team, 0) > 0 or team_podiums.get(team, 0) > 0)
        
        if teams_with_stats > 0:
            box_height_team = min(available_height / (teams_with_stats * 1), available_height * 0.15)
        else:
            box_height_team = available_height * 0.1
        
        # Draw stats boxes
        for team in sorted_teams:
            wins = team_wins.get(team, 0)
            podiums = team_podiums.get(team, 0)
            
            if wins > 0 or podiums > 0:
                box_y = stats_y_top - y_offset_team
                
                box = FancyBboxPatch((stats_x_left, box_y - box_height_team), 
                                    box_width_team_stat, box_height_team,
                                    boxstyle="round,pad=0.05", edgecolor=TEAM_COLORS[team], 
                                    facecolor='black', linewidth=1.5, alpha=0.9, zorder=4)
                ax3.add_patch(box)
                stats_boxes_team[team] = box
                
                team_name_text = ax3.text(stats_x_left + box_width_team_stat/2, 
                                         box_y - box_height_team/2 + box_height_team*0.25, 
                                         team, color=TEAM_COLORS[team], ha='center', va='center', 
                                         fontsize=11, weight='heavy', alpha=0.95, zorder=5)
                
                stats_line = ""
                if wins > 0:
                    stats_line += f"W:{wins}"
                if podiums > 0:
                    if stats_line:
                        stats_line += " "
                    stats_line += f"P:{podiums}"
                
                if stats_line:
                    stats_text_obj = ax3.text(stats_x_left + box_width_team_stat/2, 
                                             box_y - box_height_team/2 - box_height_team*0.15, 
                                             stats_line, color='white', ha='center', va='center', 
                                             fontsize=10, weight='bold', alpha=0.95, zorder=5)
                    stats_texts_team[team] = (team_name_text, stats_text_obj)
                else:
                    stats_texts_team[team] = (team_name_text,)
                
                # Connection line
                if team in bar_positions_team:
                    bar_x, bar_y = bar_positions_team[team]
                    line = plt.Line2D([stats_x_left, bar_x], 
                                     [box_y - box_height_team/2, bar_y],
                                     color=TEAM_COLORS[team], linewidth=1, 
                                     alpha=0.6, linestyle='--', zorder=2)
                    ax3.add_line(line)
                    connection_lines_team[team] = line
                
                y_offset_team += box_height_team * 1
    
    # ==================
    # REGION 4: RACING TRACK
    # ==================
    
    # Get top 3 drivers based on distance
    driver_data = {}
    for driver in TOP_DRIVERS:
        if driver in standings.index:
            driver_data[driver] = {
                'dist_km': standings.loc[driver, 'distance_km'],
                'rank': list(standings.index).index(driver) + 1,
                'points': int(get_pts(driver, t))
            }
    
    # Sort by distance to get actual top 3
    sorted_drivers = sorted(driver_data.keys(), 
                           key=lambda d: driver_data[d]['dist_km'], 
                           reverse=True)[:3]
    
    # Get driver stats
    driver_wins, driver_podiums = calculate_driver_stats(current_round)
    
    # Define stationary card positions for P1, P2, P3 (left to right)
    CARD_POSITIONS = {
        0: LEFT_BORDER_W,  # P1 at left
        1: LEFT_BORDER_W + (TRACK_LENGTH_NORM / 2) - (CARD_WIDTH_NORM / 2),  # P2 at center
        2: LEFT_BORDER_W + TRACK_LENGTH_NORM - CARD_WIDTH_NORM  # P3 at right
    }
    
    if is_pre_race:
        # PRE-RACE: Grid positions
        # Position cars BEHIND the starting line
        P1_CAR_OFFSET = -0.015  # Distance behind the line
        grid_positions = {
            0: P1_GRID_CENTER + P1_CAR_OFFSET,  # 1st place behind line
            1: ((12/80 + 21/80) / 2) + P1_CAR_OFFSET,  # 2nd place behind line
            2: ((3/80 + 12/80) / 2) + P1_CAR_OFFSET    # 3rd place behind line
        }
        
        # Smoothly transition start line to grid position during pre-race
        start_line_target_x = P1_GRID_CENTER
        diff = start_line_target_x - start_line_x
        start_line_x += diff * START_LINE_TRANSITION_SPEED
        
        start_finish_line.set_x(start_line_x - 0.002)
        start_finish_line.set_alpha(1.0)
        
        # Update all stripes to sector 1 colors (pre-race default)
        for i in range(num_stripes):
            color = 'white' if i % 2 == 0 else '#ce161b'
            track_border_stripes_top[i].set_color(color)
            track_border_stripes_bottom[i].set_color(color)
        
        for rank, driver in enumerate(sorted_drivers):
            lane_idx = rank  # 0 = bottom lane, 1 = middle, 2 = top
            lane_y = lane_center_lines[lane_idx]
            car_x = grid_positions[rank]
            
            # Position car at grid
            driver_car_images[driver].xybox = (car_x, lane_y)
            driver_car_images[driver].set_visible(True)
            
            # STATIONARY card at top
            card_x = CARD_POSITIONS[rank]
            driver_cards[driver].set_x(card_x)
            driver_cards[driver].set_visible(True)
            
            # Update card contents
            photo_x = card_x + (CARD_WIDTH_NORM * 0.15)
            driver_photos[driver].xybox = (photo_x, CARD_START + CARD_HEIGHT/2)
            driver_photos[driver].set_visible(True)
            
            # Driver name
            name_x = card_x + (CARD_WIDTH_NORM * 0.35)
            driver_name_texts[driver].set_x(name_x)
            driver_name_texts[driver].set_visible(True)
            
            # Stats in card (grid position info only)
            driver_stats_texts[driver].set_x(name_x)
            grid_pts, grid_series = GRID_DATA.get(driver, (0, ""))
            driver_stats_texts[driver].set_text(f"{grid_series}: {grid_pts}pts")
            driver_stats_texts[driver].set_visible(True)
            
            # Position label (right side of card)
            position_labels = {0: "P1", 1: "P2", 2: "P3"}
            pos_x = card_x + CARD_WIDTH_NORM - 0.01  # Right edge minus small margin
            card_position_labels[driver].set_x(pos_x)
            card_position_labels[driver].set_text(position_labels[rank])
            card_position_labels[driver].set_visible(True)
            
            # No progress bars or points display in pre-race
            if progress_bars[driver]:
                progress_bars[driver].set_visible(False)
            car_points_texts[driver].set_visible(False)
            drs_bars[driver].set_alpha(0)
    
    else:
        # RACE MODE: Scrolling track animation
        
        # Smoothly transition start line to left border at race start
        start_line_target_x = LEFT_BORDER_W
        diff = start_line_target_x - start_line_x
        start_line_x += diff * START_LINE_TRANSITION_SPEED
        
        # Get leader distance
        leader = sorted_drivers[0] if sorted_drivers else None
        if leader:
            leader_km = driver_data[leader]['dist_km']
            
            # Calculate track scroll offset
            if leader_km <= LEADER_POSITION_KM:
                # Leader hasn't reached racing position yet
                leader_x = LEFT_BORDER_W + ((leader_km / TRACK_LENGTH_KM) * TRACK_LENGTH_NORM)
                track_offset_km = 0
            else:
                # Leader at racing position, track scrolls
                leader_x = LEFT_BORDER_W + ((LEADER_POSITION_KM / TRACK_LENGTH_KM) * TRACK_LENGTH_NORM)
                track_offset_km = leader_km - LEADER_POSITION_KM
            
            # Update track border stripes based on scrolling position
            # One round = 5.1837 km, divided into 3 sectors of 1.72790 km each
            SECTOR_LENGTH_KM = 5.1837 / 3  # 1.72790 km per sector
            
            # Calculate which part of track is visible
            visible_start_km = track_offset_km
            visible_end_km = track_offset_km + TRACK_LENGTH_KM
            
            # Update each stripe based on its position in the visible track
            for i, stripe_top in enumerate(track_border_stripes_top):
                # Calculate the actual track position for this stripe
                stripe_relative_pos = (i / num_stripes) * TRACK_LENGTH_NORM
                stripe_km = visible_start_km + (stripe_relative_pos / TRACK_LENGTH_NORM) * TRACK_LENGTH_KM
                
                # Determine which sector this stripe represents (modulo for repeating sectors)
                position_in_round = stripe_km % 5.1837
                
                if position_in_round < SECTOR_LENGTH_KM:
                    # Sector 1: Red & White
                    base_color, alt_color = '#ce161b', 'white'
                elif position_in_round < 2 * SECTOR_LENGTH_KM:
                    # Sector 2: Blue & White
                    base_color, alt_color = '#169cc5', 'white'
                else:
                    # Sector 3: Yellow & Black
                    base_color, alt_color = '#efd035', 'black'
                
                # Alternate colors for stripe pattern
                color = base_color if i % 2 == 0 else alt_color
                track_border_stripes_top[i].set_color(color)
                track_border_stripes_bottom[i].set_color(color)
            
            # Start/finish line logic
            # Calculate where the start/finish line should be relative to leader
            # Start line is at km=0, km=5.1837, km=10.3674, etc.
            start_line_positions = [n * 5.1837 for n in range(20)]  # Multiple laps
            
            # Find the nearest start line ahead of or behind the visible track
            closest_start_line = None
            min_distance = float('inf')
            
            for sl_km in start_line_positions:
                if visible_start_km - 0.5 <= sl_km <= visible_end_km + 0.5:
                    distance = abs(sl_km - leader_km)
                    if distance < min_distance:
                        min_distance = distance
                        closest_start_line = sl_km
            
            # Position start/finish line if it's visible
            if closest_start_line is not None:
                # Calculate line position relative to visible track
                line_offset_from_visible_start = closest_start_line - visible_start_km
                line_x = LEFT_BORDER_W + ((line_offset_from_visible_start / TRACK_LENGTH_KM) * TRACK_LENGTH_NORM)
                
                # Only show if within track bounds
                if LEFT_BORDER_W <= line_x <= LEFT_BORDER_W + TRACK_LENGTH_NORM:
                    start_finish_line.set_x(line_x - 0.002)
                    start_finish_line.set_alpha(1.0)
                else:
                    start_finish_line.set_alpha(0.0)
            else:
                start_finish_line.set_alpha(0.0)
            
            # Finish line animation (special case for final lap)
            if leader_km >= (TOTAL_RACE_KM - 0.3):
                # Last 300m - finish line comes from right
                distance_to_finish = TOTAL_RACE_KM - leader_km
                finish_line_travel = 0.3 - distance_to_finish
                finish_x = 1.0 - ((finish_line_travel / 0.3) * (1.0 - leader_x))
                start_finish_line.set_x(finish_x - 0.002)
                start_finish_line.set_alpha(1.0)
            
            # Position all 3 drivers
            for rank, driver in enumerate(sorted_drivers):
                dist_km = driver_data[driver]['dist_km']
                driver_points = driver_data[driver]['points']
                
                # Calculate car x position based on track scroll
                relative_dist = dist_km - track_offset_km
                car_x = LEFT_BORDER_W + ((relative_dist / TRACK_LENGTH_KM) * TRACK_LENGTH_NORM)
                
                # Clamp to track bounds
                car_x = max(LEFT_BORDER_W, min(car_x, LEFT_BORDER_W + TRACK_LENGTH_NORM))
                
                # Lane assignment (0=bottom, 1=middle, 2=top)
                lane_idx = rank
                lane_y = lane_center_lines[lane_idx]
                
                # Check DRS status for this driver
                lap_m = (dist_km * 1000) % TRUE_LEN
                d_lap = int((dist_km * 1000) // TRUE_LEN) + 1
                has_drs = False
                if d_lap > 1:
                    has_drs = check_drs_active(driver, lap_m, d_lap)
                
                # DRS bar (green strip behind car)
                if has_drs:
                    drs_bar_width = 0.03  # Width of DRS bar
                    drs_bar_x = car_x - drs_bar_width
                    drs_bar_height = LANE_HEIGHT * 0.5
                    drs_bar_y = lane_y - (drs_bar_height / 2)
                    
                    drs_bars[driver].set_xy((drs_bar_x, drs_bar_y))
                    drs_bars[driver].set_width(drs_bar_width)
                    drs_bars[driver].set_height(drs_bar_height)
                    drs_bars[driver].set_alpha(0.6)
                else:
                    drs_bars[driver].set_alpha(0)
                
                # Position car
                driver_car_images[driver].xybox = (car_x, lane_y)
                driver_car_images[driver].set_visible(True)
                
                # Progress bar (from left border to car position)
                bar_width = car_x - LEFT_BORDER_W
                bar_y = lane_y - (LANE_HEIGHT * 0.15)
                bar_height = LANE_HEIGHT * 0.3
                
                if progress_bars[driver]:
                    progress_bars[driver].remove()
                
                progress_bar = ax4.add_patch(Rectangle((LEFT_BORDER_W, bar_y), bar_width, bar_height,
                                                       color=TEAM_COLORS[DRIVER_TEAMS[driver]],
                                                       alpha=0.4, zorder=4))
                progress_bars[driver] = progress_bar
                
                # Points display at car tip (right side of car with spacing)
                points_x = car_x + 0.035  # Increased offset for spacing from car tip
                car_points_texts[driver].set_position((points_x, lane_y))
                car_points_texts[driver].set_text(f"{driver_points}")
                car_points_texts[driver].set_visible(True)
                
                # STATIONARY card at top (position based on rank)
                card_x = CARD_POSITIONS[rank]
                driver_cards[driver].set_x(card_x)
                driver_cards[driver].set_visible(True)
                
                # Update card contents
                photo_x = card_x + (CARD_WIDTH_NORM * 0.15)
                driver_photos[driver].xybox = (photo_x, CARD_START + CARD_HEIGHT/2)
                driver_photos[driver].set_visible(True)
                
                # Driver name
                name_x = card_x + (CARD_WIDTH_NORM * 0.35)
                driver_name_texts[driver].set_x(name_x)
                driver_name_texts[driver].set_visible(True)
                
                # Stats: 1st/2nd/3rd finish counts
                driver_stats_texts[driver].set_x(name_x)
                driver_key = DRIVER_NAME_MAP.get(driver, "")
                first, second, third = calculate_podium_finishes(driver_key, current_round)
                driver_stats_texts[driver].set_text(f"1st:{first} 2nd:{second} 3rd:{third}")
                driver_stats_texts[driver].set_visible(True)
                
                # Position label (right side of card)
                position_labels = {0: "P1", 1: "P2", 2: "P3"}
                pos_x = card_x + CARD_WIDTH_NORM - 0.01  # Right edge minus small margin
                card_position_labels[driver].set_x(pos_x)
                card_position_labels[driver].set_text(position_labels[rank])
                card_position_labels[driver].set_visible(True)
    
    # Hide non-top-3 drivers
    for driver in TOP_DRIVERS:
        if driver not in sorted_drivers:
            driver_car_images[driver].set_visible(False)
            driver_cards[driver].set_visible(False)
            driver_photos[driver].set_visible(False)
            driver_name_texts[driver].set_visible(False)
            driver_stats_texts[driver].set_visible(False)
            car_points_texts[driver].set_visible(False)
            card_position_labels[driver].set_visible(False)
            if progress_bars[driver]:
                progress_bars[driver].set_visible(False)
            drs_bars[driver].set_alpha(0)
    
    return []

# =============================
# RUN ANIMATION
# =============================
total_frames = len(times) + START_DELAY_FRAMES
ani = FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)
plt.show()