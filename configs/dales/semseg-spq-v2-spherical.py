_base_ = ["./semseg-spq-v2-cylindrical.py"]
embedding_mode = "spherical"
model = dict(backbone=dict(embedding_mode="spherical"))
