import json
import logging
import os
import tempfile

import cv2
import stuff
from PIL import Image


class PersonPath22Loader:
    """
    Loader for PersonPath22 MOT annotations.

    Default dataset root expected:
      /mldata/downloaded_datasets/other/personpath22/dataset/personpath22

    It keeps frames that have at least one "person" annotation and can apply
    per-video visual near-duplicate filtering while decoding frames.
    """

    def __init__(self,
                 personpath22_path="/mldata/downloaded_datasets/other/personpath22",
                 task="val",
                 class_names=None,
                 ds_params=None):
        if ds_params is None:
            ds_params = {}
        if class_names is None:
            class_names = ["person"]

        # Keep compatibility with older configs that may use only the constructor arg.
        configured_path = ds_params.get("personpath22_path", personpath22_path)
        self.dataset_root = self._resolve_dataset_root(configured_path)

        gdrive_sync = ds_params.get("gdrive_sync", False)
        if gdrive_sync:
            stuff.gdrive_sync_mldata(configured_path, direction="from_drive")

        self.class_names = class_names
        self.personpath22_names = ["person", "crowd", "reflection"]
        self.class_mapping = [-1] * len(self.personpath22_names)
        for i, name in enumerate(self.personpath22_names):
            if name in self.class_names:
                self.class_mapping[i] = self.class_names.index(name)

        annotation_type = str(ds_params.get("annotation_type", "visible")).lower()
        annotation_folder = "anno_visible_2022" if annotation_type != "amodal" else "anno_amodal_2022"
        self.annotation_dir = os.path.join(self.dataset_root, "annotation", annotation_folder)
        self.split_file = os.path.join(self.dataset_root, "annotation", "splits.json")
        self.raw_video_dir = os.path.join(self.dataset_root, "raw_data")
        self.frame_stride = max(1, int(ds_params.get("frame_stride", 1)))
        self.max_samples_per_second = float(ds_params.get("max_samples_per_second", 1.0))
        default_min_interval_ms = 0.0
        if self.max_samples_per_second > 0:
            default_min_interval_ms = 1000.0 / self.max_samples_per_second
        self.min_frame_interval_ms = max(
            0.0, float(ds_params.get("min_frame_interval_ms", default_min_interval_ms))
        )
        self.similarity_filter = bool(ds_params.get("similarity_filter", True))
        self.similarity_threshold = float(ds_params.get("similarity_threshold", 0.98))
        self.similarity_max_side = max(64, int(ds_params.get("similarity_max_side", 320)))

        self.task = task
        self.split_task = self._map_task_to_split(task)
        self.info = f"PersonPath22 ({annotation_folder}, split={self.split_task})"

        self.tmpdir = tempfile.mkdtemp()
        self.current_frame_file = os.path.join(self.tmpdir, "personpath22_frame.jpg")
        self.current_video = None
        self.video_loader = None
        self.previous_similarity_frame = None

        self.frames = []
        self._build_frame_index()

    def __del__(self):
        remove_dir = getattr(stuff, "rmdir", None)
        if callable(remove_dir):
            remove_dir(self.tmpdir)

    def _resolve_dataset_root(self, base_path):
        path = base_path
        candidate = os.path.join(path, "dataset", "personpath22")
        if os.path.isdir(candidate):
            return candidate
        if os.path.isdir(os.path.join(path, "annotation")) and os.path.isdir(os.path.join(path, "raw_data")):
            return path
        raise FileNotFoundError(
            f"PersonPath22 path not found: {base_path}. "
            "Expected either <path>/dataset/personpath22 or a direct dataset root with annotation/raw_data."
        )

    def _map_task_to_split(self, task):
        if task == "val":
            return "test"
        if task == "train":
            return "train"
        if task == "test":
            return "test"
        raise ValueError(f"PersonPath22Loader: unsupported task '{task}' (expected train/val/test)")

    def _source_class_from_labels(self, labels):
        if not isinstance(labels, dict):
            return None
        if labels.get("person", 0):
            return "person"
        if labels.get("crowd", 0):
            return "crowd"
        if labels.get("reflection", 0):
            return "reflection"
        return None

    def _frame_time_ms(self, frame_idx, frame_time_lookup, fps):
        time_ms = frame_time_lookup.get(frame_idx)
        if time_ms is not None:
            return float(time_ms)
        if fps > 0:
            return (1000.0 * float(frame_idx)) / float(fps)
        return None

    def _limit_frames_by_time(self, frame_indices, frame_time_lookup, fps):
        if self.min_frame_interval_ms <= 0:
            return frame_indices

        selected = []
        last_time_ms = None
        for frame_idx in frame_indices:
            time_ms = self._frame_time_ms(frame_idx, frame_time_lookup, fps)
            if time_ms is None:
                selected.append(frame_idx)
                continue
            if last_time_ms is None or (time_ms - last_time_ms) >= self.min_frame_interval_ms:
                selected.append(frame_idx)
                last_time_ms = time_ms
        return selected

    def _build_frame_index(self):
        if not os.path.isfile(self.split_file):
            raise FileNotFoundError(f"PersonPath22 splits file not found: {self.split_file}")
        if not os.path.isdir(self.annotation_dir):
            raise FileNotFoundError(f"PersonPath22 annotation dir not found: {self.annotation_dir}")

        with open(self.split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        split_videos = list(splits.get(self.split_task, []))

        total_loaded_frames = 0
        for video_name in split_videos:
            ann_path = os.path.join(self.annotation_dir, f"{video_name}.json")
            video_path = os.path.join(self.raw_video_dir, video_name)
            if not os.path.isfile(ann_path):
                logging.warning(f"PersonPath22: missing annotation file {ann_path}")
                continue
            if not os.path.isfile(video_path):
                logging.warning(f"PersonPath22: missing video file {video_path}")
                continue

            with open(ann_path, "r", encoding="utf-8") as f:
                anno = json.load(f)

            metadata = anno.get("metadata", {})
            resolution = metadata.get("resolution", {})
            width = float(resolution.get("width", 0))
            height = float(resolution.get("height", 0))
            fps = float(metadata.get("fps", 0) or 0)
            if width <= 0 or height <= 0:
                logging.warning(f"PersonPath22: invalid resolution in {ann_path}")
                continue

            frame_dets = {}
            person_frames = set()
            frame_time_lookup = {}
            for entity in anno.get("entities", []):
                source_name = self._source_class_from_labels(entity.get("labels", {}))
                if source_name is None:
                    continue

                bb = entity.get("bb", None)
                if not isinstance(bb, list) or len(bb) != 4:
                    continue
                x, y, w, h = [float(v) for v in bb]
                if w <= 0 or h <= 0:
                    continue

                blob = entity.get("blob", {})
                frame_idx = int(blob.get("frame_idx", -1))
                if frame_idx < 0:
                    continue

                det = {
                    "box": [
                        stuff.clip01(x / width),
                        stuff.clip01(y / height),
                        stuff.clip01((x + w) / width),
                        stuff.clip01((y + h) / height),
                    ],
                    "class": self.personpath22_names.index(source_name),
                    "confidence": float(entity.get("confidence", 1.0)),
                }
                frame_dets.setdefault(frame_idx, []).append(det)
                if source_name == "person":
                    person_frames.add(frame_idx)
                    entity_time = entity.get("time", None)
                    if entity_time is not None:
                        try:
                            time_ms = float(entity_time)
                            prev_time = frame_time_lookup.get(frame_idx, None)
                            if prev_time is None or time_ms < prev_time:
                                frame_time_lookup[frame_idx] = time_ms
                        except (TypeError, ValueError):
                            pass

            frame_indices = sorted(person_frames)
            frame_indices = frame_indices[::self.frame_stride]
            frame_indices = self._limit_frames_by_time(frame_indices, frame_time_lookup, fps)
            for frame_idx in frame_indices:
                self.frames.append({
                    "video": video_path,
                    "frame_idx": frame_idx,
                    "gts": frame_dets.get(frame_idx, []),
                    "time_ms": self._frame_time_ms(frame_idx, frame_time_lookup, fps),
                })
            total_loaded_frames += len(frame_indices)

        logging.info(
            f"PersonPath22: loaded {total_loaded_frames} person-labelled frames from "
            f"{len(split_videos)} videos for split {self.split_task}; "
            f"min_frame_interval_ms={self.min_frame_interval_ms}; "
            f"similarity_filter={self.similarity_filter} threshold={self.similarity_threshold}"
        )

    def get_info(self):
        return self.info

    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        if source_class not in self.personpath22_names:
            return
        if dest_class not in self.class_names:
            return
        self.class_mapping[self.personpath22_names.index(source_class)] = self.class_names.index(dest_class)

    def get_image_ids(self):
        return range(len(self.frames))

    def _switch_video_if_needed(self, video_path):
        if video_path == self.current_video:
            return
        self.current_video = video_path
        self.video_loader = stuff.RandomAccessVideoReader(video_path)
        self.previous_similarity_frame = None

    def _make_similarity_frame(self, frame):
        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side <= self.similarity_max_side:
            return frame
        scale = self.similarity_max_side / float(max_side)
        target_w = max(8, int(w * scale))
        target_h = max(8, int(h * scale))
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def get_img_path(self, img_id):
        frame = self.frames[img_id]
        self._switch_video_if_needed(frame["video"])
        decoded, _ = self.video_loader.get_frame_at_index(frame["frame_idx"])
        if decoded is None:
            return None

        if self.similarity_filter:
            curr_similarity_frame = self._make_similarity_frame(decoded)
            if self.previous_similarity_frame is not None:
                similarity = stuff.image_ssim(curr_similarity_frame, self.previous_similarity_frame)
                if similarity >= self.similarity_threshold:
                    return None
            self.previous_similarity_frame = curr_similarity_frame

        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(self.current_frame_file, subsampling=0, quality=95)
        return self.current_frame_file

    def get_annotations(self, img_id):
        out = []
        for gt in self.frames[img_id]["gts"]:
            mapped_class = self.class_mapping[gt["class"]]
            if mapped_class == -1:
                continue
            out.append({
                "box": list(gt["box"]),
                "class": mapped_class,
                "confidence": float(gt.get("confidence", 1.0)),
            })
        return out

    def get_category_maps(self):
        return {
            "person": ["person"],
            "vehicle": [],
            "animal": [],
            "face": [],
            "weapon": [],
            "ignore": ["crowd", "reflection"],
        }
