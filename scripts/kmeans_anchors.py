#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate dataset-specific anchors for MultiGridDet.

The active MultiGridDet anchor loader expects one detection scale per line:

    112,74 149,190 370,328
    28,17 56,112 57,35
    9,10 13,28 28,55

Each line contains the anchors assigned to one prediction head. The default
order is large, medium, small to match model outputs y1, y2, y3
(stride 32, stride 16, stride 8).
"""

import argparse
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np


class MultiGridAnchorKMeans:
    """IoU/IoL K-means clustering for MultiGridDet anchor generation."""

    def __init__(
        self,
        cluster_number: int,
        annotation_file: str,
        anchors_file: str,
        input_shape: Tuple[int, int] = (608, 608),
        anchors_per_scale: int = 3,
        metric: str = "iol",
        scale_order: str = "large-to-small",
        seed: int = 42,
    ):
        self.cluster_number = cluster_number
        self.annotation_file = annotation_file
        self.anchors_file = anchors_file
        self.input_shape = input_shape
        self.anchors_per_scale = anchors_per_scale
        self.metric = metric
        self.scale_order = scale_order
        self.seed = seed

    def overlap(self, boxes: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """Calculate IoU or IoL between box sizes and cluster sizes."""
        boxes = boxes.astype(np.float32)
        clusters = clusters.astype(np.float32)

        box_area = boxes[:, 0:1] * boxes[:, 1:2]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        intersection_wh = np.minimum(boxes[:, None, :], clusters[None, :, :])
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

        if self.metric == "iou":
            denominator = box_area + cluster_area[None, :] - intersection
        else:
            denominator = np.maximum(box_area, cluster_area[None, :])

        return intersection / np.maximum(denominator, 1e-8)

    def avg_overlap(self, boxes: np.ndarray, clusters: np.ndarray) -> float:
        """Average best overlap between every box and the generated anchors."""
        return float(np.mean(np.max(self.overlap(boxes, clusters), axis=1)))

    def kmeans(self, boxes: np.ndarray) -> np.ndarray:
        """Run overlap-distance K-means."""
        box_number = boxes.shape[0]
        if box_number < self.cluster_number:
            raise ValueError(
                f"Need at least {self.cluster_number} boxes, found {box_number}"
            )

        rng = np.random.default_rng(self.seed)
        clusters = boxes[rng.choice(box_number, self.cluster_number, replace=False)]
        last_nearest = np.full((box_number,), -1, dtype=np.int32)

        while True:
            distances = 1.0 - self.overlap(boxes, clusters)
            current_nearest = np.argmin(distances, axis=1)
            if np.array_equal(last_nearest, current_nearest):
                break

            for cluster_idx in range(self.cluster_number):
                assigned = boxes[current_nearest == cluster_idx]
                if len(assigned) > 0:
                    clusters[cluster_idx] = np.median(assigned, axis=0)

            last_nearest = current_nearest

        return clusters

    def parse_annotations(self) -> np.ndarray:
        """Parse MultiGridDet annotation txt files and extract box sizes."""
        boxes = []
        input_h, input_w = self.input_shape

        with open(self.annotation_file, "r") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                for box_str in parts[1:]:
                    coords = box_str.split(",")
                    if len(coords) < 5:
                        continue
                    try:
                        x1, y1, x2, y2 = map(float, coords[:4])
                    except ValueError:
                        warnings.warn(f"Skipping invalid box at line {line_no}: {box_str}")
                        continue

                    width = max(0.0, x2 - x1)
                    height = max(0.0, y2 - y1)
                    if width < 1.0 or height < 1.0:
                        continue

                    width = min(width, float(input_w))
                    height = min(height, float(input_h))
                    boxes.append([width, height])

        if not boxes:
            raise ValueError("No valid bounding boxes found in annotation file")

        return np.asarray(boxes, dtype=np.float32)

    def sort_and_group(self, anchors: np.ndarray) -> np.ndarray:
        """Sort anchors by area and group them in model output order."""
        areas = anchors[:, 0] * anchors[:, 1]
        anchors = anchors[np.argsort(areas)]

        if self.cluster_number % self.anchors_per_scale != 0:
            raise ValueError(
                "cluster_number must be divisible by anchors_per_scale for MultiGridDet output"
            )

        num_scales = self.cluster_number // self.anchors_per_scale
        grouped = anchors.reshape(num_scales, self.anchors_per_scale, 2)

        if self.scale_order == "large-to-small":
            grouped = grouped[::-1]
        elif self.scale_order != "small-to-large":
            raise ValueError(f"Unsupported scale_order: {self.scale_order}")

        return grouped

    def layer_label(self, scale_idx: int, num_scales: int) -> str:
        """Return a human-readable prediction-head label."""
        if num_scales == 3 and self.scale_order == "large-to-small":
            labels = [
                "layer 0 / stride 32 / large",
                "layer 1 / stride 16 / medium",
                "layer 2 / stride 8 / small",
            ]
            return labels[scale_idx]
        if num_scales == 3 and self.scale_order == "small-to-large":
            labels = [
                "layer 0 / small",
                "layer 1 / medium",
                "layer 2 / large",
            ]
            return labels[scale_idx]
        return f"scale {scale_idx}"

    def save_anchors(self, grouped_anchors: np.ndarray, legacy_yolo_file: str = None) -> None:
        """Save anchors in active MultiGridDet format."""
        output_path = Path(self.anchors_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            for scale_anchors in grouped_anchors:
                pairs = [
                    f"{int(round(anchor[0]))},{int(round(anchor[1]))}"
                    for anchor in scale_anchors
                ]
                f.write(" ".join(pairs) + "\n")

        if legacy_yolo_file:
            legacy_path = Path(legacy_yolo_file)
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            flat = grouped_anchors.reshape(-1, 2)
            pairs = [
                f"{int(round(anchor[0]))},{int(round(anchor[1]))}"
                for anchor in flat
            ]
            legacy_path.write_text(", ".join(pairs) + "\n")

    def generate_anchors(self, legacy_yolo_file: str = None) -> Tuple[np.ndarray, float]:
        """Generate, save, and report anchor clusters."""
        boxes = self.parse_annotations()
        anchors = self.kmeans(boxes)
        grouped_anchors = self.sort_and_group(anchors)
        flat_anchors = grouped_anchors.reshape(-1, 2)
        score = self.avg_overlap(boxes, flat_anchors)

        self.save_anchors(grouped_anchors, legacy_yolo_file=legacy_yolo_file)

        print(f"Parsed boxes: {len(boxes):,}")
        print(f"Metric: {self.metric.upper()}")
        print(f"Scale order: {self.scale_order}")
        print(f"Average best {self.metric.upper()}: {score * 100:.2f}%")
        print(f"Saved MultiGridDet anchors to: {self.anchors_file}")
        num_scales = grouped_anchors.shape[0]
        for scale_idx, scale_anchors in enumerate(grouped_anchors):
            pairs = " ".join(
                f"{int(round(anchor[0]))},{int(round(anchor[1]))}"
                for anchor in scale_anchors
            )
            print(f"  {self.layer_label(scale_idx, num_scales)}: {pairs}")

        if legacy_yolo_file:
            print(f"Saved legacy YOLO one-line anchors to: {legacy_yolo_file}")

        return grouped_anchors, score


def parse_input_shape(value: str) -> Tuple[int, int]:
    """Parse H,W or HxW input shape."""
    normalized = value.lower().replace("x", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("input shape must be H,W or HxW")
    h, w = map(int, parts)
    if h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError("input dimensions must be positive")
    return h, w


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate MultiGridDet-ready anchors using overlap K-means"
    )
    parser.add_argument(
        "--annotation_file",
        "--annotation-file",
        dest="annotation_file",
        required=True,
        help="Training annotation txt file: image_path x1,y1,x2,y2,class ...",
    )
    parser.add_argument(
        "--cluster_number",
        "--cluster-number",
        dest="cluster_number",
        type=int,
        default=9,
        help="Total number of anchors to generate (default: 9)",
    )
    parser.add_argument(
        "--anchors_file",
        "--anchors-file",
        dest="anchors_file",
        required=True,
        help="Output MultiGridDet anchor file path",
    )
    parser.add_argument(
        "--input-shape",
        type=parse_input_shape,
        default=(608, 608),
        help="Model input shape as H,W or HxW (default: 608,608)",
    )
    parser.add_argument(
        "--anchors-per-scale",
        type=int,
        default=3,
        help="Anchors per detection scale/head (default: 3)",
    )
    parser.add_argument(
        "--metric",
        choices=["iol", "iou"],
        default="iol",
        help="Overlap metric for clustering (default: iol, matching MultiGridDet assignment)",
    )
    parser.add_argument(
        "--scale-order",
        choices=["large-to-small", "small-to-large"],
        default="large-to-small",
        help=(
            "Order of output lines. Default large-to-small matches MultiGridDet "
            "outputs y1/y2/y3: stride32, stride16, stride8."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for K-means initialization (default: 42)",
    )
    parser.add_argument(
        "--legacy-yolo-file",
        default=None,
        help="Optional one-line YOLO-format output file for compatibility",
    )

    args = parser.parse_args()

    if args.cluster_number != 9:
        warnings.warn(
            "MultiGridDet Darknet configs expect 9 anchors split as 3 scales x 3 anchors. "
            f"You requested {args.cluster_number} anchors."
        )

    generator = MultiGridAnchorKMeans(
        cluster_number=args.cluster_number,
        annotation_file=args.annotation_file,
        anchors_file=args.anchors_file,
        input_shape=args.input_shape,
        anchors_per_scale=args.anchors_per_scale,
        metric=args.metric,
        scale_order=args.scale_order,
        seed=args.seed,
    )
    generator.generate_anchors(legacy_yolo_file=args.legacy_yolo_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
