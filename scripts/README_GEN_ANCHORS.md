# MultiGridDet Anchor Generation

Use `scripts/kmeans_anchors.py` to generate dataset-specific anchors from a
MultiGridDet annotation file.

## Annotation Input

The script expects the same txt format used by training:

```text
/path/to/image.jpg x1,y1,x2,y2,class x1,y1,x2,y2,class ...
```

Box coordinates should be in pixels after the same resize/letterbox convention
used by the training annotation files.

## Generate Anchors

From the repository root:

```bash
python scripts/kmeans_anchors.py \
  --annotation-file data/coco_train2017.txt \
  --cluster-number 9 \
  --anchors-per-scale 3 \
  --input-shape 608,608 \
  --metric iol \
  --scale-order large-to-small \
  --anchors-file configs/coco_custom_anchors.txt
```

`--metric iol` is the default and matches MultiGridDet's anchor assignment logic.
Use `--metric iou` only if you explicitly want YOLO-style IoU clustering.
`--scale-order large-to-small` is also the default and matches the current
MultiGridDet model output order.

## Output Format

The active MultiGridDet loader expects one detection scale per line, with
whitespace-separated `width,height` pairs:

```text
112,74 149,190 370,328
28,17 56,112 57,35
9,10 13,28 28,55
```

Line order matters. The current Darknet model returns predictions as
`[y1, y2, y3]`, where `y1` is the coarse stride-32 head for large objects,
`y2` is the stride-16 head for medium objects, and `y3` is the stride-8 head for
small objects. Therefore the anchor file must be ordered:

```text
line 0: large anchors  -> layer 0 / y1 / stride 32
line 1: medium anchors -> layer 1 / y2 / stride 16
line 2: small anchors  -> layer 2 / y3 / stride 8
```

This is the same ordering used by `configs/yolov3_coco_anchor.txt`. For the
current Darknet config, keep `9` total anchors split as `3` per scale.

## Use The Anchors

Update the model config:

```yaml
model:
  preset:
    anchors_path: configs/coco_custom_anchors.txt
```

Then train from the same model config. If you are fine-tuning from an old
checkpoint, changing anchors changes the target encoding and anchor logits, so
expect a short adaptation period and consider a conservative learning rate.

## Optional Legacy YOLO Output

If you also want a one-line comma-separated file:

```bash
python scripts/kmeans_anchors.py \
  --annotation-file data/coco_train2017.txt \
  --anchors-file configs/coco_custom_anchors.txt \
  --legacy-yolo-file configs/coco_custom_anchors_legacy.txt
```

The legacy file is not the preferred format for this repository's active
`load_anchors()` path.
