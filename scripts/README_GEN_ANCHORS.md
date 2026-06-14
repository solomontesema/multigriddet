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
  --anchors-file configs/coco_custom_anchors.txt
```

`--metric iol` is the default and matches MultiGridDet's anchor assignment logic.
Use `--metric iou` only if you explicitly want YOLO-style IoU clustering.

## Output Format

The active MultiGridDet loader expects one detection scale per line, with
whitespace-separated `width,height` pairs:

```text
9,10 13,28 16,36
31,19 36,45 64,73
108,131 196,226 411,369
```

The lines are ordered from small to large anchors, matching the three prediction
heads. For the current Darknet config, keep `9` total anchors split as `3` per
scale.

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
