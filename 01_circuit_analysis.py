import math

total_round = 24

#Round 1: Albert Park Circuit (weather factor) Length: 5.278 km, 58 laps, fastest lap: 1:20.000, track speed: 233 km/h Corners: 14, Right Turn: 9, DRS straight: 2
#Round 2: Shanghai International Circuit Length: 5.451 km, 56 laps, fastest lap: 1:35.454, track speed: 205.58 km/h Corners: 16, Right Turn: 9, DRS straight: 2
#Round 3: Suzuka International Racing Course Length: 5.807 km, 53 laps, fastest lap: 1:30.965, track speed: 229.8 km/h Corners: 18, Right Turn: 10, DRS straight: 1
#Round 4: Bahrain International Circuit Length: 5.412 km, 57 laps, fastest lap: 1:35.140, track speed: 204.7845 km/h Corners: 15, Right Turn: 9, DRS straight: 3
#Round 5: Jeddah Corniche Circuit Length: 6.174 km, 50 laps, fastest lap: 1:31.77, track speed: 242 km/h Corners: 27, Right Turn: 11, DRS straight: 3
#Round 6: Miami International Autodrome Length: 5.412 km, 57 laps, fastest lap: 1:29.746, track speed: 217 km/h Corners: 19, Right Turn: 7, DRS straight: 3
#Round 7: Autodromo Enzo e Dino Ferrari Length: 4.909 km, 63 laps, fastest lap: 1:17.988, track speed: 226.6 km/h Corners: 19, Right Turn: 9, DRS straight: 1
#Round 8: Circuit de Monaco Length: 3.337 km, 78 laps, fastest lap: 1:13.221, track speed: 164 km/h Corners: 19, Right Turn: 12, DRS straight: 1
#Round 9: Circuit de Barcelona-Catalunya Length: 4.657 km, 66 laps, fastest lap: 1:15.743, track speed: 221.343 km/h Corners: 14, Right Turn: 8, DRS straight: 2
#Round 10: Circuit Gilles-Villeneuve Length: 4.361 km, 70 laps, fastest lap: 1:14.119, track speed: 211.8 km/h Corners: 14, Right Turn: 8, DRS straight: 2
#Round 11: Red Bull Ring Length: 4.326 km, 71 laps, fastest lap: 1:07.924, track speed: 229 km/h Corners: 10, Right Turn: 7, DRS straight: 3
#Round 12: Silverstone Circuit (weather factor) Length: 5.891 km, 52 laps, fastest lap: 1:29.337, track speed: 237 km/h Corners: 18, Right Turn: 10, DRS straight: 3
#Round 13: Circuit de Spa-Francorchamps (weather factor) Length: 7.004 km, 44 laps, fastest lap: 1:44.861, track speed: 240 km/h Corners: 19, Right Turn: 9, DRS straight: 2
#Round 14: Hungaroring Length: 4.381 km, 70 laps, fastest lap: 1:19.409, track speed: 198.6 km/h Corners: 14, Right Turn: 8, DRS straight: 1
#Round 15: Circuit Zandvoort Length: 4.259 km, 72 laps, fastest lap: 1:12.271, track speed: 212.15 km/h Corners: 14, Right Turn: 10, DRS straight: 2
#Round 16: Autodromo Nazionale Monza Length: 5.793 km, 53 laps, fastest lap: 1:20.901, track speed: 257.78 km/h Corners: 11, Right Turn: 7, DRS straight: 2
#Round 17: Baku City Circuit Length: 6.003 km, 51 laps, fastest lap: 1:43.388, track speed: 209 km/h Corners: 20, Right Turn: 8, DRS straight: 2
#Round 18: Marina Bay Street Circuit Length: 4.927 km, 62 laps, fastest lap: 1:33.808, track speed: 189 km/h Corners: 19, Right Turn: 7, DRS straight: 3
#Round 19: Circuit of The Americas Length: 5.513 km, 56 laps, fastest lap: 1:37.577, track speed: 203.4 km/h Corners: 20, Right Turn: 9, DRS straight: 2
#Round 20: Autódromo Hermanos Rodríguez Length: 4.304 km, 71 laps, fastest lap: 1:20.052, track speed: 193.554 km/h Corners: 17, Right Turn: 10, DRS straight: 2
#Round 21: Autódromo José Carlos Pace (weather factor) Length: 4.309 km, 71 laps, fastest lap: 1:12.400, track speed: 214.26 km/h Corners: 15, Right Turn: 5, DRS straight: 2
#Round 22: Las Vegas Strip Circuit Length: 6.201 km, 50 laps, fastest lap: 1:33.365, track speed: 239.1 km/h Corners: 17, Right Turn: 6, DRS straight: 2
#Round 23: Losail International Circuit Length: 5.419 km, 57 laps, fastest lap: 1:22.996, track speed: 235 km/h Corners: 16, Right Turn: 10, DRS straight: 1
#Round 24: Yas Marina Circuit Length: 5.281 km, 58 laps, fastest lap: 1:26.725, track speed: 219.2 km/h Corners: 16, Right Turn: 7, DRS straight: 2

circuits = [
    # length, corners, right_turns, drs, fastest_lap_seconds, name, official_speed, weather_factor
    (5.278, 14, 9, 2, 80.000, "Albert Park", 233.0, True),
    (5.451, 16, 9, 2, 95.454, "Shanghai", 205.58, False),
    (5.807, 18, 10, 1, 90.965, "Suzuka", 229.8, False),
    (5.412, 15, 9, 3, 95.140, "Bahrain", 204.7845, False),
    (6.174, 27, 11, 3, 91.770, "Jeddah", 242.0, False),
    (5.412, 19, 7, 3, 89.746, "Miami", 217.0, False),
    (4.909, 19, 9, 1, 77.988, "Imola", 226.6, False),
    (3.337, 19, 12, 1, 73.221, "Monaco", 164.0, False),
    (4.657, 14, 8, 2, 75.743, "Barcelona", 221.343, False),
    (4.361, 14, 8, 2, 74.119, "Canada", 211.8, False),
    (4.326, 10, 7, 3, 67.924, "Red Bull Ring", 229.0, False),
    (5.891, 18, 10, 3, 89.337, "Silverstone", 237.0, True),
    (7.004, 19, 9, 2, 104.861, "Spa", 240.0, True),
    (4.381, 14, 8, 1, 79.409, "Hungaroring", 198.6, False),
    (4.259, 14, 10, 2, 72.271, "Zandvoort", 212.15, False),
    (5.793, 11, 7, 2, 80.901, "Monza", 257.78, False),
    (6.003, 20, 8, 2, 103.388, "Baku", 209.0, False),
    (4.927, 19, 7, 3, 93.808, "Singapore", 189.0, False),
    (5.513, 20, 9, 2, 97.577, "COTA", 203.4, False),
    (4.304, 17, 10, 2, 80.052, "Mexico", 193.554, False),
    (4.309, 15, 5, 2, 72.400, "Interlagos", 214.26, True),
    (6.201, 17, 6, 2, 93.365, "Las Vegas", 239.1, False),
    (5.419, 16, 10, 1, 82.996, "Losail", 235.0, False),
    (5.281, 16, 7, 2, 86.725, "Yas Marina", 219.2, False)
]

# -------------------------
# 1) Compute total km
# -------------------------
total_km = sum(c[0] for c in circuits)

# -------------------------
# 2) Property per km
# -------------------------
corners_per_km = sum(c[1] for c in circuits) / total_km
right_turns_per_km = sum(c[2] for c in circuits) / total_km
drs_per_km = sum(c[3] for c in circuits) / total_km
seconds_per_km = sum(c[4] / c[0] for c in circuits) / len(circuits)

# -------------------------
# 3) Derived circuit
# -------------------------
L = total_km / total_round

derived_corners = round(corners_per_km * L)
derived_right = round(right_turns_per_km * L)
derived_left = derived_corners - derived_right
derived_drs = round(drs_per_km * L)
derived_fastest_lap = seconds_per_km * L
track_speed = (L / derived_fastest_lap) * 3600

# -------------------------
# Output reference circuit
# -------------------------
print("Derived F1 Reference Circuit")
print("-----------------------------")
print(f"Length: {L:.4f} km")
print(f"Corners: {derived_corners}")
print(f"Right turns: {derived_right}")
print(f"Left turns: {derived_left}")
print(f"DRS zones: {derived_drs}")
print(f"Fastest lap: {derived_fastest_lap:.3f} seconds")
print(f"TRACK SPEED: {track_speed:.1f} km/h")

# -------------------------
# 4) Ranking similarity
# -------------------------
ranking = sorted(
    [(c[5], c[6], c[7]) for c in circuits],
    key=lambda x: abs(x[1] - track_speed)
)

print("\nSpeed-Class Similarity Ranking:\n")
for i, (name, speed, weather) in enumerate(ranking, 1):
    diff = abs(speed - track_speed)
    weather_tag = " WEATHER RISK" if weather else ""
    print(f"{i:2d}. {name:<15} speed={speed:7.2f} km/h   Δ={diff:6.2f}{weather_tag}")

# -------------------------
# 5) List weather-factor tracks
# -------------------------
weather_tracks = [c[5] for c in circuits if c[7]]

print("\nWeather-Sensitive Circuits:")
for track in weather_tracks:
    print(f"• {track}")