import logging
import os
import tempfile
import xml.etree.ElementTree as ET

import cv2
import stuff
from PIL import Image


class JAADLoader:
    """
    Loader for JAAD clips + XML annotations.

    Expected dataset root:
      /mldata/downloaded_datasets/other/JAAD
    containing:
      - annotations/*.xml
      - JAAD_clips/*.mp4
      - split_ids/<subset>/{train,val,test}.txt
    """

    def __init__(self,
                 jaad_path="/mldata/downloaded_datasets/other/JAAD",
                 task="val",
                 class_names=None,
                 ds_params=None):
        if ds_params is None:
            ds_params = {}
        if class_names is None:
            class_names = ["person"]

        configured_path = ds_params.get("jaad_path", jaad_path)
        self.dataset_root = self._resolve_dataset_root(configured_path)

        gdrive_sync = ds_params.get("gdrive_sync", False)
        if gdrive_sync:
            stuff.gdrive_sync_mldata(configured_path, direction="from_drive")

        self.class_names = class_names
        # "people" means group/crowd region in JAAD.
        self.jaad_names = ["person", "people"]
        self.class_mapping = [-1] * len(self.jaad_names)
        for i, name in enumerate(self.jaad_names):
            if name in self.class_names:
                self.class_mapping[i] = self.class_names.index(name)

        self.task = task
        self.split_task = self._map_task_to_split(task)
        self.split_subset = str(ds_params.get("split_subset", "default"))
        self.split_file = os.path.join(
            self.dataset_root, "split_ids", self.split_subset, f"{self.split_task}.txt"
        )
        self.annotation_dir = os.path.join(self.dataset_root, "annotations")
        self.video_dir = os.path.join(self.dataset_root, "JAAD_clips")

        self.frame_stride = max(1, int(ds_params.get("frame_stride", 1)))
        self.max_samples_per_second = float(ds_params.get("max_samples_per_second", 1.0))
        default_min_interval_ms = 0.0
        if self.max_samples_per_second > 0:
            default_min_interval_ms = 1000.0 / self.max_samples_per_second
        self.min_frame_interval_ms = max(
            0.0, float(ds_params.get("min_frame_interval_ms", default_min_interval_ms))
        )

        # Keep only frames that contain at least one individual person.
        # Group regions ("people") still come through on those frames as ignore GT.
        self.require_person_frame = bool(ds_params.get("require_person_frame", True))

        self.similarity_filter = bool(ds_params.get("similarity_filter", True))
        self.similarity_threshold = float(ds_params.get("similarity_threshold", 0.98))
        self.similarity_max_side = max(64, int(ds_params.get("similarity_max_side", 320)))

        self.info = f"JAAD (subset={self.split_subset}, split={self.split_task})"

        self.tmpdir = tempfile.mkdtemp()
        self.current_frame_file = os.path.join(self.tmpdir, "jaad_frame.jpg")
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
        if (
            os.path.isdir(os.path.join(base_path, "annotations"))
            and os.path.isdir(os.path.join(base_path, "JAAD_clips"))
            and os.path.isdir(os.path.join(base_path, "split_ids"))
        ):
            return base_path
        raise FileNotFoundError(
            f"JAAD path not found: {base_path}. "
            "Expected a dataset root with annotations/, JAAD_clips/, and split_ids/."
        )

    def _map_task_to_split(self, task):
        if task == "train":
            return "train"
        if task == "val":
            return "val"
        if task == "test":
            return "test"
        raise ValueError(f"JAADLoader: unsupported task '{task}' (expected train/val/test)")

    def _source_class_from_track_label(self, track_label):
        label = str(track_label or "").strip().lower()
        if label in ("pedestrian", "ped"):
            return "person"
        if label == "people":
            return "people"
        return None

    def _frame_time_ms(self, frame_idx, fps):
        if fps <= 0:
            return None
        return (1000.0 * float(frame_idx)) / float(fps)

    def _limit_frames_by_time(self, frame_indices, fps):
        if self.min_frame_interval_ms <= 0:
            return frame_indices

        selected = []
        last_time_ms = None
        for frame_idx in frame_indices:
            time_ms = self._frame_time_ms(frame_idx, fps)
            if time_ms is None:
                selected.append(frame_idx)
                continue
            if last_time_ms is None or (time_ms - last_time_ms) >= self.min_frame_interval_ms:
                selected.append(frame_idx)
                last_time_ms = time_ms
        return selected

    def _probe_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0, 0.0, 0.0, 0
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, width, height, frame_count

    def _build_frame_index(self):
        if not os.path.isfile(self.split_file):
            raise FileNotFoundError(f"JAAD split file not found: {self.split_file}")
        if not os.path.isdir(self.annotation_dir):
            raise FileNotFoundError(f"JAAD annotation dir not found: {self.annotation_dir}")
        if not os.path.isdir(self.video_dir):
            raise FileNotFoundError(f"JAAD video dir not found: {self.video_dir}")

        with open(self.split_file, "r", encoding="utf-8") as f:
            split_videos = [x.strip() for x in f.readlines() if x.strip()]

        total_loaded_frames = 0
        for video_name in split_videos:
            ann_path = os.path.join(self.annotation_dir, f"{video_name}.xml")
            video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
            if not os.path.isfile(ann_path):
                logging.warning(f"JAAD: missing annotation file {ann_path}")
                continue
            if not os.path.isfile(video_path):
                logging.warning(f"JAAD: missing video file {video_path}")
                continue

            try:
                tree = ET.parse(ann_path)
            except Exception as e:
                logging.warning(f"JAAD: failed parsing {ann_path}: {e}")
                continue
            root = tree.getroot()

            num_frames = 0
            width = 0.0
            height = 0.0
            try:
                num_frames = int(root.find("./meta/task/size").text)
            except Exception:
                num_frames = 0
            try:
                width = float(root.find("./meta/task/original_size/width").text)
                height = float(root.find("./meta/task/original_size/height").text)
            except Exception:
                width = 0.0
                height = 0.0

            fps, video_w, video_h, video_frames = self._probe_video(video_path)
            if fps <= 0:
                fps = 30.0
            if width <= 0:
                width = video_w
            if height <= 0:
                height = video_h
            if width <= 0 or height <= 0:
                logging.warning(f"JAAD: invalid frame size for {video_name}")
                continue
            if num_frames <= 0:
                num_frames = video_frames

            frame_dets = {}
            person_frames = set()
            any_labelled_frames = set()
            for track in root.findall("./track"):
                source_name = self._source_class_from_track_label(track.attrib.get("label"))
                if source_name is None:
                    continue

                source_class = self.jaad_names.index(source_name)
                for box in track.findall("./box"):
                    try:
                        frame_idx = int(box.get("frame"))
                    except (TypeError, ValueError):
                        continue
                    if frame_idx < 0:
                        continue
                    if num_frames > 0 and frame_idx >= num_frames:
                        continue
                    try:
                        outside = int(box.get("outside", "0"))
                    except (TypeError, ValueError):
                        outside = 0
                    if outside != 0:
                        continue
                    try:
                        xtl = float(box.get("xtl"))
                        ytl = float(box.get("ytl"))
                        xbr = float(box.get("xbr"))
                        ybr = float(box.get("ybr"))
                    except (TypeError, ValueError):
                        continue
                    if xbr <= xtl or ybr <= ytl:
                        continue

                    det = {
                        "box": [
                            stuff.clip01(xtl / width),
                            stuff.clip01(ytl / height),
                            stuff.clip01(xbr / width),
                            stuff.clip01(ybr / height),
                        ],
                        "class": source_class,
                        "confidence": 1.0,
                    }
                    frame_dets.setdefault(frame_idx, []).append(det)
                    any_labelled_frames.add(frame_idx)
                    if source_name == "person":
                        person_frames.add(frame_idx)

            frame_indices = sorted(person_frames if self.require_person_frame else any_labelled_frames)
            frame_indices = frame_indices[::self.frame_stride]
            frame_indices = self._limit_frames_by_time(frame_indices, fps)

            for frame_idx in frame_indices:
                self.frames.append({
                    "video": video_path,
                    "frame_idx": frame_idx,
                    "gts": frame_dets.get(frame_idx, []),
                    "time_ms": self._frame_time_ms(frame_idx, fps),
                })
            total_loaded_frames += len(frame_indices)

        logging.info(
            f"JAAD: loaded {total_loaded_frames} labelled frames from {len(split_videos)} videos "
            f"for subset={self.split_subset} split={self.split_task}; "
            f"min_frame_interval_ms={self.min_frame_interval_ms}; "
            f"similarity_filter={self.similarity_filter} threshold={self.similarity_threshold}; "
            f"require_person_frame={self.require_person_frame}"
        )

    def get_info(self):
        return self.info

    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        if source_class not in self.jaad_names:
            return
        if dest_class not in self.class_names:
            return
        self.class_mapping[self.jaad_names.index(source_class)] = self.class_names.index(dest_class)

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
            "ignore": ["people"],
        }
