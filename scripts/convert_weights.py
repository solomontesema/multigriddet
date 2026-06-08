#!/usr/bin/env python3
"""
Convert MultiGridDet checkpoints between HDF5 and portable NumPy formats.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multigriddet.config import ConfigLoader, build_model_from_config
from multigriddet.config.model_builder import (
    load_model_weights,
    save_weights_npz,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MultiGridDet weight files between .weights.h5 and .npz formats."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/infer_config.yaml",
        help="Path to inference config file.",
    )
    parser.add_argument(
        "--input-weights",
        type=str,
        required=True,
        help="Source weights file (.weights.h5, .h5, or .npz).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output weights file (.npz or .weights.h5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = ConfigLoader.load_config(args.config)
    model_config = ConfigLoader.load_config(config["model_config"])
    full_config = ConfigLoader.merge_configs(model_config, config)

    model = build_model_from_config(full_config, for_training=False)
    load_mode = load_model_weights(model, args.input_weights)
    print(f"Loaded source weights: {args.input_weights} ({load_mode})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".npz":
        save_weights_npz(model, str(output_path))
        print(f"Saved portable weights to: {output_path}")
        return 0

    model.save_weights(str(output_path))
    print(f"Saved Keras weights to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
