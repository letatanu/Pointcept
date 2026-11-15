"""Pointcept dataset adapter exposing full-coverage SO3 spatial blocks.

Copy beside so3_transforms.py and import SO3BlockDataset in datasets/__init__.py.
First use materializes blocks in cache_dir; later visits read individual blocks.
Source datasets must expose untransformed get_data() and filesystem data_list.
"""

import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from .builder import DATASETS, build_dataset
from .transform import Compose
from .so3_transforms import SO3SpatialPartition


@DATASETS.register_module()
class SO3BlockDataset(Dataset):
    def __init__(
        self,
        source,
        cache_dir,
        max_points=16384,
        transform=None,
        loop=1,
        pad_to_multiple=1,
        test_mode=False,
    ):
        if test_mode:
            raise ValueError(
                "Use the original dataset plus SO3SpatialPartition for testing"
            )
        for name, value in [("loop", loop), ("pad_to_multiple", pad_to_multiple)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            source.get("transform")
            or source.get("test_mode", False)
            or source.get("loop", 1) != 1
        ):
            raise ValueError("source must use transform=[], test_mode=False and loop=1")
        partition = SO3SpatialPartition(max_points=max_points, center=False)
        dataset = build_dataset(source)
        if not getattr(dataset, "data_list", None):
            raise ValueError("Source has no scenes")
        if not isinstance(dataset.data_list, (list, tuple)):
            raise ValueError("Source data_list must contain scene directory paths")
        # Fingerprint source files and implementation so stale caches are not reused.
        fingerprint = hashlib.sha256()
        fingerprint.update(json.dumps(dict(source), sort_keys=True).encode())
        fingerprint.update(str(max_points).encode())
        fingerprint.update(Path(__file__).read_bytes())
        fingerprint.update(Path(__file__).with_name("so3_transforms.py").read_bytes())
        for scene in dataset.data_list:
            folder = Path(scene).resolve()
            assets = sorted(folder.glob("*.npy"))
            if not assets:
                raise ValueError(f"No numpy assets under {folder}")
            fingerprint.update(str(folder).encode())
            for asset in assets:
                info = asset.stat()
                fingerprint.update(
                    f"{asset.name}:{info.st_size}:{info.st_mtime_ns}".encode()
                )
        self.cache_root = Path(cache_dir).resolve() / fingerprint.hexdigest()[:24]
        self.cache_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.cache_root / "manifest.json"
        # Coordinate concurrent loader/rank initialization on a shared Linux filesystem.
        with (self.cache_root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            complete = False
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                complete = all(
                    (self.cache_root / item["file"]).is_file()
                    for item in manifest["blocks"]
                )
            if not complete:
                records = []
                for scene_id in range(len(dataset.data_list)):
                    raw = dataset.get_data(scene_id)
                    blocks = partition(raw)
                    for block_id, block in enumerate(blocks):
                        name = f"{scene_id:06d}-{block_id:06d}"
                        filename = name + ".npz"
                        # Persist aligned arrays only; np.load never needs pickle.
                        keys = set(block["index_valid_keys"])
                        arrays = {key: block[key] for key in keys if key in block}
                        destination = self.cache_root / filename
                        temporary = destination.with_suffix(".tmp")
                        with temporary.open("wb") as handle:
                            np.savez(handle, **arrays)
                        os.replace(temporary, destination)
                        records.append(
                            dict(
                                file=filename,
                                name=name,
                                scene=scene_id,
                                points=len(block["coord"]),
                            )
                        )
                    logging.getLogger("pointcept").info(
                        "SO3 blocks: scene %d/%d, %d points -> %d blocks",
                        scene_id + 1,
                        len(dataset.data_list),
                        len(raw["coord"]),
                        len(blocks),
                    )
                manifest = dict(blocks=records, max_points=max_points)
                temporary = manifest_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(manifest))
                os.replace(temporary, manifest_path)
        self.blocks = manifest["blocks"]
        self.loop = loop
        self.pad_to_multiple = pad_to_multiple
        self.transform = Compose(transform or [])
        self.test_mode = False
        self.split = source.get("split")
        self.data_list = [str(self.cache_root / item["file"]) for item in self.blocks]
        logging.getLogger("pointcept").info(
            "SO3BlockDataset: %d blocks, %d source points, %d padded samples/pass",
            len(self.blocks),
            sum(item["points"] for item in self.blocks),
            len(self),
        )

    def __len__(self):
        count = len(self.blocks) * self.loop
        return (
            (count + self.pad_to_multiple - 1) // self.pad_to_multiple
        ) * self.pad_to_multiple

    def get_data_name(self, idx):
        return self.blocks[idx % len(self.blocks)]["name"]

    def get_data(self, idx):
        record = self.blocks[idx % len(self.blocks)]
        with np.load(self.cache_root / record["file"], allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        data["index_valid_keys"] = list(data)
        data["name"] = record["name"]
        data["split"] = self.split
        return data

    def __getitem__(self, idx):
        return self.transform(self.get_data(idx))
