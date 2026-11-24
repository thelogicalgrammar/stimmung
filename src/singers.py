"""The six voices: each sings one harmonic of B-flat 1."""
F0_BASE = 58.27047    # B-flat 1

SINGERS = [
    dict(name="bass",     h=2, speaker="M01"),
    dict(name="tenor1",   h=3, speaker="JD2"),
    dict(name="tenor2",   h=4, speaker="JD2"),
    dict(name="alto",     h=5, speaker="W02"),
    dict(name="soprano1", h=7, speaker="W02"),
    dict(name="soprano2", h=9, speaker="W02"),
]
for s in SINGERS:
    s["f0"] = F0_BASE * s["h"]
    s["male"] = s["h"] <= 4
