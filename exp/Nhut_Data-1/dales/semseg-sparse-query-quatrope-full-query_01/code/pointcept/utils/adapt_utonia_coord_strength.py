import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("destination")
args = parser.parse_args()

destination = Path(args.destination)
if destination.exists():
    raise FileExistsError(f"Output already exists: {destination}")

checkpoint = torch.load(
    args.source, map_location="cpu", weights_only=False
)
state = checkpoint["state_dict"]
changed = []

for key, value in list(state.items()):
    if key.endswith("embedding.stem.linear.weight"):
        if value.ndim != 2 or value.shape[1] != 9:
            raise ValueError(f"Unexpected shape: {key}: {value.shape}")

        adapted = value.new_zeros(value.shape[0], 4)
        adapted[:, :3] = value[:, :3]  # Coordinate weights.
        # Column 4 learns strength during fine-tuning.
        state[key] = adapted
        changed.append(key)

if not changed:
    raise RuntimeError("No input embedding found; checkpoint unchanged.")

destination.parent.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": state}, destination)

for key in changed:
    print(f"Converted {key}: 9 -> 4 input channels")
print(f"Saved: {destination}")