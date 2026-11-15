"""Generate eight fully commented, standalone DALES configs beside this script.

Also generates a grouped-mode config matching the original cvpr.py backbone.
Run before copying configs into Pointcept/configs/dales/. No PyTorch required.
"""
from pathlib import Path
from itertools import product
import ast


def generate():
    root = Path(__file__).resolve().parent
    text = (root / "cvpr_three_attention.py").read_text(encoding="utf-8")
    flags = ("enable_local_attention", "enable_cross_attention", "enable_coarse_attention")
    outputs = []
    for values in product((False, True), repeat=3):
        source = text
        for flag, value in zip(flags, values):
            # Exact top-level assignment; preserves parameter comments and nested references.
            source = source.replace(f"{flag} = True", f"{flag} = {value}", 1)
        tag = "L%d_X%d_G%d" % tuple(map(int, values))
        path = root / f"cvpr_{tag}.py"
        ast.parse(source)
        path.write_text(source, encoding="utf-8")
        outputs.append(path.name)
    legacy = text.replace("enable_coarse_attention = True", "enable_coarse_attention = False", 1)
    legacy = legacy.replace('cross_mode="central",', 'cross_mode="grouped",', 1)
    legacy = legacy.replace('cross_queries=(16, 16, 16, 16),', 'cross_queries=(32, 32, 32, 32),', 1)
    legacy = legacy.replace('cross_keys=(32, 32, 32, 32),', 'cross_keys=(128, 128, 128, 128),', 1)
    legacy = legacy.replace('save_path = f"exp/dales/cvpr_three_attention_{ablation_name}"',
                            'save_path = "exp/dales/cvpr_legacy_matched"', 1)
    legacy = ('# LEGACY CONTROL: grouped mode, Q=32/group, K=128/patch, coarse OFF.\n'
              '# Explanatory comments below describe the central-mode template unless stated otherwise.\n'
              + legacy)
    ast.parse(legacy)
    path = root / "cvpr_legacy_matched.py"
    path.write_text(legacy, encoding="utf-8")
    outputs.append(path.name)
    def variant(filename, replacements):
        source = text
        for before, after in replacements:
            if before not in source:
                raise ValueError(f"Template setting not found: {before}")
            source = source.replace(before, after, 1)
        source = source.replace('save_path = f"exp/dales/cvpr_three_attention_{ablation_name}"',
                                f'save_path = "exp/dales/{filename}"', 1)
        source = '# Generated experiment variant; inspect the setting values below.\n' + source
        ast.parse(source)
        path = root / f"{filename}.py"
        path.write_text(source, encoding="utf-8")
        outputs.append(path.name)
    # Query sweep isolates Q: keys/window/dilation fixed; coarse context OFF.
    for budget in (16, 32, 64, 128):
        variant(f"cvpr_queries_{budget}", [
            ("enable_coarse_attention = True", "enable_coarse_attention = False"),
            ("cross_queries=(16, 16, 16, 16),", f"cross_queries=({budget}, {budget}, {budget}, {budget}),")])
    # Placement sweep isolates placement: Q=16 in all cases, same token grid.
    variant("cvpr_coarse_last_stage", [])
    variant("cvpr_coarse_late_stages", [("coarse_stages=None,", "coarse_stages=(2, 3),")])
    variant("cvpr_coarse_all_stages", [("coarse_stages=None,", "coarse_stages=(0, 1, 2, 3),")])
    variant("cvpr_coarse_every_block", [
        ("coarse_stages=None,", "coarse_stages=(0, 1, 2, 3),"),
        ('coarse_placement="stage_end",', 'coarse_placement="every_block",')])
    return outputs


if __name__ == "__main__":
    print("\n".join(generate()))
