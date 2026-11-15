# Controlled grouping ablation; point counts and compute differ by occupancy.
_base_ = ["./semseg-spq-v2-cylindrical.py"]
grouping = "grid"
model = dict(backbone=dict(grouping="grid"))
