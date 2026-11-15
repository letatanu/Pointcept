"""Export OccuFly voxel-center arrays to colored binary PLY for verification.

--mode verify exports semantic GT, camera RGB, and color-validity views.
Validity colors: green = measured RGB available; magenta = unavailable.
RGB is exported unchanged (unavailable colors normally zero/black).
PLY vertices preserve training labels and, when present, color_valid.
This exports points at voxel centers, not voxel cubes or reconstructed surfaces.

Standalone mode needs only NumPy. Optional --config-file mode requires Pointcept.
Labels must use the 0..20 mapping from preprocess_occufly.py; -1 is ignore.

Export predictions and ground truth together:
python pointcept/datasets/preprocessing/occufly/npy2ply.py --dataset_root data/occufly/pointcept --split train --mode verify --output-dir data/occufly/viz
"""

import argparse
from pathlib import Path
import os
import tempfile

import numpy as np

# Official OccuFly palette, reordered into contiguous training ID order.
# Source: https://github.com/markus-42/OccuFly/blob/main/docs/dataset_notes.md
RAW_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 21, 22, 33, 34, 35, 36)
COLOR_MAP = np.array([
    [128, 0, 128],     # 0 road (raw 1)
    [204, 163, 72],    # 1 walkway (raw 2)
    [128, 0, 0],       # 2 dirt (raw 3)
    [192, 192, 192],   # 3 gravel (raw 4)
    [246, 120, 40],    # 4 rock (raw 5)
    [0, 255, 0],       # 5 grass (raw 6)
    [112, 148, 32],    # 6 vegetation (raw 7)
    [64, 64, 0],       # 7 tree (raw 8)
    [255, 255, 0],     # 8 ground obstacle (raw 9)
    [255, 16, 255],    # 9 person (raw 11)
    [255, 204, 153],   # 10 bicycle (raw 12)
    [0, 128, 128],     # 11 vehicle (raw 13)
    [0, 0, 255],       # 12 water (raw 14)
    [255, 0, 0],       # 13 building (raw 16)
    [64, 160, 120],    # 14 roof (raw 17)
    [255, 160, 0],     # 15 cable (raw 21)
    [106, 0, 255],     # 16 cable tower (raw 22)
    [128, 64, 128],    # 17 parking lot (raw 33)
    [240, 120, 120],   # 18 construction (raw 34)
    [255, 255, 128],   # 19 crane (raw 35)
    [128, 128, 64],    # 20 truck (raw 36)
], dtype=np.uint8)


def validate_labels(labels, count):
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1 or len(labels) != count:
        raise ValueError(f"Expected ({count},) labels, got {labels.shape}; predictions must match original point order")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integer class IDs, not logits or probabilities")
    valid = (labels >= 0) & (labels < len(COLOR_MAP))
    if np.any(~valid & (labels != -1)):
        raise ValueError("Expected training IDs 0..20 or ignore -1; raw OccuFly IDs are not supported")
    return labels.astype(np.int32), valid



def validate_mask(mask, count):
    mask = np.asarray(mask)
    if mask.shape != (count,) or not np.isin(mask, [0, 1]).all():
        raise ValueError(f"color_valid must have shape ({count},) with boolean or 0/1 values")
    return mask.astype(bool)


def write_ply(path, coord, labels, overwrite=False, colors=None, color_valid=None):
    coord = np.asarray(coord)
    if coord.ndim != 2 or coord.shape[1] != 3 or not np.all(np.isfinite(coord)):
        raise ValueError("Coordinates must be finite with shape (N, 3)")
    labels, valid = validate_labels(labels, len(coord))
    if colors is None:
        colors = np.full((len(coord), 3), 128, dtype=np.uint8)
        colors[valid] = COLOR_MAP[labels[valid]]
    else:
        colors = np.asarray(colors)
        if colors.shape != (len(coord), 3) or not np.isfinite(colors).all():
            raise ValueError("Colors must be finite with shape (N, 3)")
        if np.any((colors < 0) | (colors > 255)):
            raise ValueError("Expected RGB in 0..255, not normalized training features")
        colors = np.rint(colors).astype(np.uint8)
    if color_valid is not None:
        color_valid = validate_mask(color_valid, len(coord))
    fields = [
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', '<i4'),
    ]
    if color_valid is not None:
        fields.append(('color_valid', 'u1'))
    vertex = np.empty(len(coord), dtype=fields)
    for i, axis in enumerate(('x', 'y', 'z')):
        vertex[axis] = coord[:, i]
        if not np.all(np.isfinite(vertex[axis])):
            raise ValueError("Coordinates exceed float32 range")
    for i, channel in enumerate(('red', 'green', 'blue')):
        vertex[channel] = colors[:, i]
    vertex['label'] = labels
    if color_valid is not None:
        vertex['color_valid'] = color_valid
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        'comment OccuFly voxel centers; label is training ID 0..20 or ignore -1\n'
        f'element vertex {len(coord)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'property int label\n'
        + ('property uchar color_valid\n' if color_valid is not None else '')
        + 'end_header\n'
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite to replace")
    fd, temporary = tempfile.mkstemp(prefix='.ply-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(header.encode('ascii'))
            vertex.tofile(handle)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Exclusive publication also protects against concurrent exporters.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset_root', '--dataset-root', type=Path)
    source.add_argument('--config-file', help='Pointcept config; run in your Pointcept environment')
    parser.add_argument('--options', nargs='+', help='Pointcept config overrides: key=value')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--mode', choices=('pred', 'gt', 'both', 'rgb', 'validity', 'verify'), default='pred')
    result = parser.add_mutually_exclusive_group()
    result.add_argument('--result-dir', type=Path)
    result.add_argument('--exp-name', help='Uses exp/<exp-name>/result, matching your reference')
    parser.add_argument('--output_dir', '--output-dir', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--valid-colors-only', action='store_true',
                        help='Filter every exported view to points with color_valid=True')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.options and not args.config_file:
        parser.error('--options requires --config-file')
    result_dir = args.result_dir or (Path('exp') / args.exp_name / 'result' if args.exp_name else None)
    if args.mode in ('pred', 'both') and result_dir is None:
        parser.error('Prediction export requires --result-dir or --exp-name')
    output = args.output_dir or (result_dir / 'ply_viz' if result_dir else None)
    if output is None:
        parser.error('Ground-truth export requires --output-dir when no result directory is given')

    if args.config_file:
        from pointcept.utils.config import Config, DictAction
        from pointcept.datasets import build_dataset
        cfg = Config.fromfile(args.config_file)
        if args.options:
            overrides = argparse.ArgumentParser()
            overrides.add_argument('--options', nargs='+', action=DictAction)
            cfg.merge_from_dict(overrides.parse_args(['--options'] + args.options).options)
        dataset = build_dataset(cfg.data[args.split])
        # Iterate each original sample once and bypass all transforms.
        count = len(dataset.data_list)
        def get_data(index):
            data = dataset.get_data(index)
            # Original dataset VALID_ASSETS may omit the new validity mask.
            sample = Path(dataset.data_list[index])
            for key in ('color', 'color_valid'):
                if key not in data and (sample / f'{key}.npy').is_file():
                    data[key] = np.load(sample / f'{key}.npy', allow_pickle=False)
            return data
    else:
        samples = sorted(p.parent for p in (args.dataset_root / args.split).glob('*/coord.npy'))
        count = len(samples)

        def get_data(index):
            sample = samples[index]
            data = {'name': sample.name, 'coord': np.load(sample / 'coord.npy', allow_pickle=False)}
            if args.mode != 'pred':
                data['segment'] = np.load(sample / 'segment.npy', allow_pickle=False)
            for key in ('color', 'color_valid'):
                if (sample / f'{key}.npy').is_file():
                    data[key] = np.load(sample / f'{key}.npy', allow_pickle=False)
            return data

    if count == 0:
        parser.error('No samples found; dataset_root must point to the converted Pointcept dataset')
    count = min(count, args.limit) if args.limit else count
    written = errors = 0
    for index in range(count):
        try:
            data = get_data(index)
            name = str(data['name'])
            if not name or name in ('.', '..') or any(c in name for c in '/\\:'):
                raise ValueError(f'Invalid sample name: {name!r}')
            kinds = ('gt', 'rgb', 'validity') if args.mode == 'verify' else (('gt', 'pred') if args.mode == 'both' else (args.mode,))
            mask = data.get('color_valid')
            if mask is not None:
                mask = validate_mask(mask, len(data['coord']))
                print(f'{name}: {int(mask.sum())}/{len(mask)} points have valid camera RGB ({100 * mask.mean() if len(mask) else 0:.2f}%)', flush=True)
            if (args.mode in ('rgb', 'validity', 'verify') or args.valid_colors_only) and mask is None:
                raise ValueError('Missing color_valid.npy; use the revised RGB preprocessor')
            selection = mask if args.valid_colors_only else slice(None)
            for kind in kinds:
                try:
                    labels = data['segment'] if kind != 'pred' else np.load(result_dir / f'{name}_pred.npy', allow_pickle=False)
                    labels, _ = validate_labels(labels, len(data['coord']))
                    colors = None
                    if kind == 'rgb':
                        colors = np.asarray(data['color'])
                        if colors.shape != (len(data['coord']), 3) or not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 255)):
                            raise ValueError('Expected original color.npy: finite N x 3 RGB in 0..255')
                    elif kind == 'validity':
                        colors = np.where(mask[:, None], np.array([0, 200, 0]), np.array([255, 0, 255])).astype(np.uint8)
                    suffix = '_valid_only' if args.valid_colors_only else ''
                    path = output / f'{name}_{kind}{suffix}.ply'
                    write_ply(path, data['coord'][selection], labels[selection], args.overwrite,
                              colors=None if colors is None else colors[selection],
                              color_valid=None if mask is None else mask[selection])
                    written += 1
                    print(f'[{index+1}/{count}] Saved {path}', flush=True)
                except Exception as exc:
                    errors += 1
                    print(f'ERROR {name} ({kind}): {exc}', flush=True)
        except Exception as exc:
            errors += 1
            print(f'ERROR sample {index}: {exc}', flush=True)
    print(f'Finished: {written} PLY files saved, {errors} errors.')
    return 1 if errors or not written else 0


if __name__ == '__main__':
    raise SystemExit(main())
