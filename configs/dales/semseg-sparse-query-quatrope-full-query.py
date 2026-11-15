_base_ = ["./semseg-sparse-query-quatrope.py"]

model = dict(backbone=dict(query_ratios=(1.,1.,1.,1.)))
