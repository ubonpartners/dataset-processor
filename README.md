# Dataset Processor

**Dataset Processor** is a config-driven toolbox for **building training datasets for object detectors** (Ultralytics `ubon26` format), with support for:

- **Bounding boxes** (YOLO-format labels)
- **Keypoints**: 17×3 COCO pose + 5×3 RetinaFace-style face keypoints (combined 22×3 per person)
- **Dataset completion** via strong detectors: add missing boxes, add pose/face keypoints to existing GTs
- **Hard example mining** and hard-negative/background mining
- **Binary attributes** via a vision LLM, plus FIQA-style face quality attributes

The pipeline produces datasets in the format expected by the [`ubon26` branch](https://github.com/ubonpartners/ultralytics/tree/ubon26) of Ultralytics — one row per object with a separate attribute vector.

---

## Quick start

### Install

```bash
conda env create -f environment.yml
conda activate dataset-processor
```

### Data locations

By default, everything is under `/mldata`:

- **Downloaded/raw datasets**: `/mldata/downloaded_datasets`
- **Generated output datasets**: `/mldata/output_datasets`

Override the root with:

```bash
export MLDATA_LOCATION=/your/mldata
```

### Build a dataset

```bash
python build_dataset.py --config data/dataset_test.yaml
```

For a small smoke test:

```bash
python build_dataset.py --config data/dataset_test.yaml --test
```

### Evaluate a model (mAP)

```bash
python map.py --config data/map_ultralytics.yaml
```

---

## Mental model

There are three main moving parts:

1. **Loaders** (`src/loaders/*_loader.py`) — know how to read a source dataset (COCO, OpenImages, WiderFace, …) and return per-image annotations.

2. **DatasetProcessor** (`src/dataset_processor.py`) — a wrapper around a generated dataset folder that can read/write YOLO label files, run a detector, and apply pipeline stages.

3. **Pipeline runner** (`build_dataset.py`) — reads a YAML/JSON config and executes stages in order to create datasets under `output_datasets/<name>`.

---

## Dataset format

Datasets produced by this pipeline target the **`ubon26` Ultralytics branch**. See [`NEW_FORMAT.md`](NEW_FORMAT.md) for the full specification.

**Dataset YAML**:
- `nc: 5`, `names`: the 5 main classes (person, face, vehicle, animal, weapon)
- `attributes: true`, `attr_nc: 50`, `attr_label_format: "split"`
- `kpt_shape: [22, 3]` (5 face + 17 pose keypoints) when keypoints are present
- Optionally `attr_names`: list of the 50 attribute names

**Label files** — one row per object:

```
cls  x  y  w  h  attr_0 … attr_49  [kpt_x kpt_y kpt_v] × 22
```

- `cls` ∈ {0,1,2,3,4} — main class only (person=0, face=1, vehicle=2, animal=3, weapon=4)
- `attr_0 … attr_49` — binary attribute vector in [0, 1]
- Keypoints: 22×3 = 66 floats (only on person/face boxes)
- **Total columns**: 121 (pose) or 55 (detect-only)

**Internal pipeline format**: during processing, labels use one row per object with `cls`, bbox, keypoints, then a literal `A` marker followed by the attribute vector. This is what `load_ground_truth_labels` reads and `write_annotations` writes. The final output step converts this to the `ubon26`-compatible format above.

---

## Configuration basics

Pipeline configs are YAML (or JSON):

```yaml
datasets:
  <dataset_name>:
    general: { ... }
    import:  { ... }
    <optional_stage>: { ... }
    <optional_stage>: [ ... ]
```

### Inheritance

```yaml
general:
  inherit_config: dataset_test_default.yaml/default
```

Copies defaults from another config entry, then overlays your dataset-specific values.

### Task-specific parameters

Many stage params can be split by dataset split (`max_images_train` / `max_images_val`). Internally handled by `dsu.get_param()`.

---

## Pipeline stage reference

Stages are optional — enable them by including the stage key in your config.

Typical flow:

1. `import`
2. optional filtering/mining: `llm_filter`, `sample`, `make_hard`
3. chunked augmentation stages (run per chunk): `add_objects`, `mask_objects`, `add_pose`, `add_faces`, `add_fiqa`, `add_attributes`; `normalise` always runs once per chunk
4. post-processing: `add_backgrounds`, `merge_facepose`, `filter_classes`
5. optional `merge`

---

### `import` (required for non-merge datasets)

Loads a source dataset, maps categories into your `general.class_names`, copies images, and writes initial label files.

| Key | Description |
|-----|-------------|
| `loader` | Loader class name (e.g. `CocoLoader`) |
| `max_images_train` / `max_images_val` | Cap imported images |
| `max_width`, `max_height`, `max_bpp` | Resize/transcode large JPEGs on import |
| `fix_orientation` | Apply EXIF rotation |
| `dedup` | Image deduplication after import |
| `enable_ignore_regions` | Support an extra `ignore` class for masking regions |

---

### `llm_filter` (optional)

Uses a vision model prompt to decide whether to keep each image; deletes those that fail.

| Key | Description |
|-----|-------------|
| `prompt` | Path to a system prompt file |
| `llm` | Model name passed to `stuff.simple_llm` |
| `llm_batch` | Batch size |
| `max_images` | Limit how many images are evaluated |

---

### `sample` (optional)

Keeps a reproducible subset of images after import — useful for fast iteration.

| Key | Description |
|-----|-------------|
| `max_images_train` / `max_images_val` | Number of images to keep |
| `method` | `random` \| `first` \| `last` (default `random`) |
| `seed` | RNG seed |
| `rename` | Rename kept images sequentially (default `true`) |
| `base_name` | Prefix for renamed images |

---

### `make_hard` (optional)

Runs a reference detector and keeps the top N hardest images, scored by false negatives, false positives, and optional rare-class weighting.

| Key | Description |
|-----|-------------|
| `model` | Detector model path |
| `max_images` | Number of images to keep |
| `hardness_cache_folder` | Cache of hardness scores per origin EXIF tag |
| `rare_classes` | Class names to up-weight |

---

### `add_objects` (optional, per chunk)

Runs an object detector and adds detections that don't match existing GTs — used to fill missing classes in partially-labelled datasets. This stage is a list so you can chain multiple detectors:

```yaml
add_objects:
  - model: "/mldata/weights/yolo26l.pt"
    sz: 640
    thr: 0.90
    iou_thr: 0.50
    min_sz: 0.0025
    per_class_thr: [0.90, 0.95]
```

---

### `mask_objects` (optional, per chunk)

Similar to `add_objects` but instead of adding labels, **masks** image regions where unlabeled objects were found — useful to suppress spurious regions or for privacy masking.

---

### `add_pose` (optional, per chunk)

Runs a pose model and copies pose keypoints into existing person GT boxes.

```yaml
add_pose:
  - model: "/mldata/weights/yolo26l-pose.pt"
    sz: 640
    thr: 0.20
```

---

### `add_faces` (optional, per chunk)

Runs a face detector to add missing face boxes and/or 5-point face keypoints.

| Key | Description |
|-----|-------------|
| `box_thr` | Add a new face box above this confidence |
| `kp_thr` | Copy face keypoints above this confidence |

---

### `add_fiqa` (optional, per chunk)

Uses a detector that outputs a `fiqa_score` and turns that score into binary attributes. Thresholds are encoded in `general.attributes` as e.g. `person:fiqa_0.65`.

---

### `normalise` (runs per chunk, always)

- Clips coordinates, confidence, and keypoints to [0, 1].
- Deduplicates overlapping same-class GTs.

---

### `add_attributes` (optional, per chunk)

Crops person boxes, sends them to a vision LLM, and writes a binary attribute vector into each GT.

| Key | Description |
|-----|-------------|
| `llm-model` | Model name passed to `stuff.simple_llm` |
| `prompt` | Path to a system prompt file |
| `cache_path` | Per-origin attribute cache directory |
| `min_pose` | Optionally require N pose points before running LLM |

Set `OPENAI_API_KEY` if using OpenAI-backed models. Cost is per object crop.

---

### `add_backgrounds` (optional, post-stage)

Scores background images by detector false positives (hard negatives) and copies the hardest into your dataset.

| Key | Description |
|-----|-------------|
| `loader` | Source of background images |
| `model` | Detector for scoring |
| `check_model` | Optional second detector to reject suspected mislabels |
| `max_images_train` / `max_images_val` | How many to keep |

---

### `merge` (optional)

Builds multiple named sub-datasets and merges them into a single output:

```yaml
merge:
  - coco
  - widerface
  - openimages
```

---

### `merge_facepose` (optional, post-stage)

Merges 5-point face keypoints from face boxes into the corresponding person boxes, producing the combined 22-point representation used in training.

---

### `filter_classes` (optional, post-stage)

Removes classes from a dataset and remaps indices. Can also filter small instances or strip keypoints.

---

## Worked example

See [`data/dataset_example.yaml`](data/dataset_example.yaml) for a heavily-commented example demonstrating most stages.

```bash
python build_dataset.py --config data/dataset_example.yaml --test
```

---

## mAP evaluation (`map.py`)

Evaluates models on a dataset; computes box mAP/precision/recall per class, and keypoint AP for person pose and face keypoints. Supports cached results and multiprocessing.

```bash
python map.py --config data/map_ultralytics.yaml
```

---

## License

This project is dual-licensed under:

- **AGPL v3.0**
- The **[Ubon Cooperative License](https://github.com/ubonpartners/license/blob/main/LICENSE)**

Please also respect the licenses of any third-party datasets you use.
