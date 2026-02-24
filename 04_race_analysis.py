import csv, os

# =============================
# DERIVED F1 REFERENCE CIRCUIT
# =============================
LAP_LENGTH_KM = 5.1837
FASTEST_LAP_SEC = 86.288
LAPS = 8

TOTAL_RACE_KM = LAP_LENGTH_KM * LAPS
TOTAL_RACE_TIME = FASTEST_LAP_SEC * LAPS

ROUNDS = 24                         # 3 rounds per lap
ROUND_DISTANCE = LAP_LENGTH_KM / 3
ROUND_TIME = FASTEST_LAP_SEC / 3

GRID_GAP_KM = 0.01                  # 10 m on grid
MIN_GAP_KM  = 0.01  # 10 m while racing

TOTAL_GP=24
TOTAL_SPRINT=6
POINT_DIST_PER_GP=101
POINT_DIST_PER_SPRINT=36
TOTALPOINTS_GP=TOTAL_GP*POINT_DIST_PER_GP
TOTALPOINTS_SPRINT=TOTAL_SPRINT*POINT_DIST_PER_SPRINT
TOTAL_POINTS=TOTALPOINTS_GP+TOTALPOINTS_SPRINT
TOTAL_DRIVERS=20

NORMALIZATION = TOTAL_POINTS/TOTAL_DRIVERS
GAP_SCALE = LAP_LENGTH_KM * (LAPS / 20)

# =============================
# STARTING GRID
# =============================
#starting grid position (f1 2024 standings  Verstappen 437-p1 Norris 374-p2 Leclerc 356-p3 Piastri 292-p4 Sainz 290-p5 Russell 245-p6 Hamilton 223-p7 Alonso 70-p8 Gasly 42-p9 Hulkenberg 41-p10 Yuki 30-p11 Stroll 24-p12 Ocon 23-p13 Albon 12-p14 Bearman 7-p15 Colapinto 5-p16 Lawson 4-p17(Rookies:Bortoleto 0,Hadjar 0, Antonelli 0 so, F2 2024 standings used Bortoleto 214.5-p18, Hadjar 192-p19, Antonelli 113-p20))
starting_grid = [
    "Verstappen","Lando Norris","Charles Leclerc","Oscar Piastri","Sainz",
    "George Russell","Lewis Hamilton","Alonso","Gasly","Hulkenberg",
    "Yuki","Stroll","Ocon","Albon","Bearman","Colapinto","Lawson",
    "Bortoleto","Hadjar","Antonelli"
]

# =============================
# CUMULATIVE POINTS DATA
# =============================
drivers = {
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

# =============================
# DISTANCE SIMULATION
# =============================
distances = {d: [] for d in drivers}

# Round 0 – starting grid
for i,d in enumerate(starting_grid):
    distances[d].append(-i * GRID_GAP_KM)

prev = {d: distances[d][0] for d in drivers}

for r in range(ROUNDS):
    unlocked = (r+1) * ROUND_DISTANCE
    leader_points = max(drivers[d][r] for d in drivers)

    raw = {}
    for d in drivers:
        gap = ((leader_points - drivers[d][r]) / NORMALIZATION) * GAP_SCALE
        raw[d] = max(prev[d], unlocked - gap)

    order = sorted(raw, key=lambda d: raw[d], reverse=True)

    fixed = {}
    last = None
    for d in order:
        if last is None:
            fixed[d] = raw[d]
        else:
            fixed[d] = min(raw[d], fixed[last] - MIN_GAP_KM)
        last = d

    for d in drivers:
        distances[d].append(fixed[d])
        prev[d] = fixed[d]

# =============================
# CSV OUTPUT
# =============================
os.makedirs("output", exist_ok=True)

with open("output/f1_race.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "round",
        "time_sec",
        "position",
        "driver",
        "distance_km",
        "gap_to_leader_km"
    ])

    for r in range(ROUNDS+1):
        # sort drivers by distance (leader first)
        order = sorted(drivers, key=lambda d: distances[d][r], reverse=True)

        leader_dist = distances[order[0]][r]

        for pos, d in enumerate(order, start=1):
            dist = distances[d][r]
            gap_km = leader_dist - dist

            w.writerow([
                r,
                round(r * ROUND_TIME, 3),
                pos,
                d,
                round(dist, 4),
                round(gap_km, 4)
            ])

print("Simulation complete → output/f1_race.csv")
