# Inherit the same training/data recipe; change only the local positional encoding.
_base_ = ["./semseg-pt-v3m1-pq-cylindrical.py"]
embedding_mode = "spherical"
model = dict(backbone=dict(embedding_mode="spherical"))
