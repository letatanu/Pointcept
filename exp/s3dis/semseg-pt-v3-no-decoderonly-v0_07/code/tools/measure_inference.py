import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, time, torch
import numpy as np
from pointcept.engines.defaults import (
    default_config_parser, default_setup
)
from pointcept.models import build_model
from pointcept.datasets import build_dataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs",   type=int, default=50)
    parser.add_argument("--save-results", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = default_config_parser(args.config_file, options=None)
    default_setup(cfg)

    # Build model and load checkpoint
    model = build_model(cfg.model).cuda().eval()
    ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    # Single real S3DIS scene (Area 5) — no synthetic tensor
    val_cfg = cfg.data.val
    dataset = build_dataset(val_cfg)
    sample = dataset[0]

    data_dict = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.cuda()
        else:
            data_dict[k] = v

    # offset = [total number of points] — required by offset2batch
    data_dict["offset"] = torch.tensor(
        [data_dict["coord"].shape[0]], dtype=torch.long
    ).cuda()

    # ---------- Warm-up ----------
    with torch.no_grad():
        for _ in range(args.num_warmup):
            _ = model(data_dict)
    torch.cuda.synchronize()

    # ---------- Latency ----------
    latencies = []
    with torch.no_grad():
        for _ in range(args.num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(data_dict)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    # ---------- Peak Memory ----------
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(data_dict)
    torch.cuda.synchronize()
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    results = {
        "latency_mean_ms":   float(np.mean(latencies)),
        "latency_std_ms":    float(np.std(latencies)),
        "latency_median_ms": float(np.median(latencies)),
        "peak_memory_mb":    float(peak_mem_mb),
        "num_warmup":        args.num_warmup,
        "num_runs":          args.num_runs,
        "checkpoint":        args.checkpoint,
    }

    print("\n===== Inference Metrics =====")
    for k, v in results.items():
        print(f"  {k}: {v}")

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.save_results}")


if __name__ == "__main__":
    main()