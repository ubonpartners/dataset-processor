import os
import json
import stuff

class CrowdPoseLoader:
    """
    Loader for the CrowdPose dataset (COCO-style format).

    Expects:
        base_path/
            images/                        # Directory with all image files
            CrowdPose/crowdpose_<split>.json # JSON annotations for train/val/test
    """
    def __init__(self,
                 crowdpose_base_path="/mldata/downloaded_datasets/crowdpose",
                 task="train",
                 class_names=None,
                 ds_params=None):
        self.base_path = crowdpose_base_path

        gdrive_sync=ds_params.get("gdrive_sync", True)
        if gdrive_sync:
            stuff.gdrive_sync_mldata(self.base_path, direction="from_drive")

        self.task = task  # 'train', 'val', or 'test'
        self.images_dir = os.path.join(self.base_path, "images")
        self.ann_file = os.path.join(self.base_path, "CrowdPose", f"crowdpose_{task}.json")
        self.class_names = class_names
        self.crowdpose_names=["person","crowd"]
        self.class_mapping=[-1]*len(self.crowdpose_names)
        self._load_annotations()

    def _load_annotations(self):
        with open(self.ann_file, 'r') as f:
            data = json.load(f)
        self.image_dict = {img['id']: img for img in data['images']}
        self.annotations_by_image = {}
        for ann in data['annotations']:
            if ann['image_id'] not in self.annotations_by_image:
                self.annotations_by_image[ann['image_id']] = []
            self.annotations_by_image[ann['image_id']].append(ann)
        self.image_ids = list(self.image_dict.keys())

    def get_info(self):
        return "CrowdPose: https://github.com/Jeff-sjtu/CrowdPose"

    def get_image_ids(self):
        return self.image_ids

    def get_img_path(self, img_id):
        filename = self.image_dict[img_id]['file_name']
        return os.path.join(self.images_dir, filename)

    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        #print(f"add category mapping {source_class} -> {dest_class}")
        self.class_mapping[self.crowdpose_names.index(source_class)]=self.class_names.index(dest_class)

    def get_annotations(self, img_id):
        img_info = self.image_dict[img_id]
        width, height = img_info['width'], img_info['height']
        annotations = self.annotations_by_image.get(img_id, [])
        detections = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            iscrowd = ann.get('iscrowd', 0)
            x1n = x / width
            y1n = y / height
            x2n = (x + w) / width
            y2n = (y + h) / height
            cl=0 # person
            if iscrowd:
                cl=1
            det = {
                "box": [x1n, y1n, x2n, y2n],
                "class": self.class_mapping[cl],  # "person"
                "confidence": 1.0
            }
            detections.append(det)
        return detections

    def get_category_maps(self):
        return { 'person': ["person"],
                 'vehicle': [],
                 'animal': [],
                 'face': [],
                 'weapon': [],
                 'ignore': ["crowd"]
               }
