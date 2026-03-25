# Dataset Format

This document specifies the label and YAML format produced by Dataset Processor for training with the [`ubon26` Ultralytics branch](https://github.com/ubonpartners/ultralytics/tree/ubon26).

---

## Dataset YAML

```yaml
nc: 5
names:
  - person    # 0
  - face      # 1
  - vehicle   # 2
  - animal    # 3
  - weapon    # 4

attributes: true
attr_nc: 50
attr_label_format: "split"

# Optional: human-readable attribute names (same order as attr_0..attr_49)
attr_names:
  - person_is_male
  - person_is_female
  - person_is_child
  - person_is_teen
  - person_is_adult
  - person_is_senior
  - person_is_wearing_hat_or_head_covering
  - person_is_wearing_a_mask_or_face_covering
  - person_wearing_glasses_or_sunglasses
  - person_has_facial_hair
  - person_has_shoulder_length_hair
  - person_has_buzz_cut_or_bald_head
  - person_has_bag_or_backpack
  - person_is_holding_mobile_phone
  - person_is_wearing_a_uniform
  - person_has_long_sleeves
  - person_is_wearing_shorts
  - person_is_wearing_a_coat_or_jacket
  - person_is_wearing_hoodie
  - person_is_wearing_bright_colored_clothing
  - person_is_wearing_hi_vis_clothes
  - person_has_patterned_clothing
  - person_clothing_has_prominent_logo
  - person_has_heavy_build
  - person_is_lying_down
  - person_has_a_threatening_posture
  - person_is_running
  - person_is_fighting
  - person_is_behind_camera
  - person_is_carrying_a_weapon
  - person_weapon_held_in_hands
  - person_has_visible_tattoos
  - person_is_smoking_or_vaping
  - person_top_is_white_or_light
  - person_top_is_black_or_gray_or_dark
  - person_top_is_blue_or_purple
  - person_top_is_green
  - person_top_is_red_or_pink
  - person_top_is_orange_or_beige_or_yellow
  - person_bottom_is_white_or_light
  - person_bottom_is_black_or_gray_or_dark
  - person_bottom_is_blue_or_purple
  - person_bottom_is_green
  - person_bottom_is_red_or_pink
  - person_bottom_is_orange_or_beige_or_yellow
  - person_fiqa_0
  - person_fiqa_1
  - person_fiqa_2
  - person_fiqa_3
  - person_fiqa_4

# Keypoints (when present)
kpt_shape: [22, 3]   # 5 face + 17 pose keypoints per person box
flip_idx: [...]      # mirror pairs for augmentation
```

---

## Label files

One `.txt` file per image. Each line is one object:

```
cls  x  y  w  h  attr_0 … attr_49  kpt_x0 kpt_y0 kpt_v0 … kpt_x21 kpt_y21 kpt_v21
```

| Field | Count | Description |
|-------|-------|-------------|
| `cls` | 1 | Main class index, 0–4 |
| `x y w h` | 4 | Normalized bounding box (center x, center y, width, height) |
| `attr_0 … attr_49` | 50 | Binary attribute vector; values in [0, 1] |
| Keypoints | 66 | 22 × (x, y, visibility); omitted for detect-only rows |

**Total columns**: 121 with keypoints, 55 without.

- `cls` is the main class only — never an attribute index.
- Attribute values are 0.0 or 1.0 (binary), or soft labels in [0, 1] when applicable.
- Keypoints are normalized to [0, 1]. Visibility: 0 = not labeled, 1 = labeled but possibly occluded, 2 = labeled and visible.
- Face keypoints occupy slots 0–4 (R-eye, L-eye, nose, R-mouth, L-mouth); pose keypoints occupy slots 5–21 (COCO order).

---

## Internal pipeline format

During processing, Dataset Processor uses a different internal representation — one row per object with `cls`, bbox, keypoints, then a literal `A` marker, then the attribute vector. This is what `load_ground_truth_labels` reads and `write_annotations` writes throughout the pipeline stages. The final output step converts from this internal format to the `ubon26`-compatible format above.
