import os
import cv2
import src.dataset_util as du
import stuff

class WiderPersonLoader:
    """
    Loader for the WiderPerson pedestrian detection dataset.

    Expects a directory structure:
        <base_path>/Images/      # contains all .jpg images
        <base_path>/Annotations/ # contains annotation files named <image>.jpg.txt
        <base_path>/<split>.txt  # train.txt, val.txt, test.txt listing images (with or without .jpg)
    """
    def __init__(self,
                 widerperson_base_path="/mldata/downloaded_datasets/widerperson",
                 task="train",
                 class_names=None,
                 ds_params=None):

        gdrive_sync=ds_params.get("gdrive_sync", True)
        if gdrive_sync:
            stuff.gdrive_sync_mldata(widerperson_base_path, direction="from_drive")

        self.base_path = widerperson_base_path
        self.task = task  # 'train', 'val', or 'test'
        # Default class names for labels 1-5
        self.class_names = class_names
        self.widerperson_names=["unused", "pedestrian","rider","partial","ignore","crowd"]
        self.class_mapping=[-1]*len(self.widerperson_names)
        self._load_split_list()
        # Directories
        self.images_dir = os.path.join(self.base_path, "Images")
        self.ann_dir = os.path.join(self.base_path, "Annotations")

    def _load_split_list(self):
        split_file = os.path.join(self.base_path, f"{self.task}.txt")
        self.image_ids = []
        with open(split_file, 'r') as f:
            for line in f:
                img = line.strip()
                # allow lines with or without .jpg
                if img.lower().endswith('.jpg'):
                    self.image_ids.append(img)
                else:
                    self.image_ids.append(f"{img}.jpg")

    def get_info(self):
        return "WiderPerson: https://github.com/shaohua0116/WiderPerson"

    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        #print(f"add category mapping {source_class} -> {dest_class}")
        self.class_mapping[self.widerperson_names.index(source_class)]=self.class_names.index(dest_class)

    def get_image_ids(self):
        return list(self.image_ids)

    def get_img_path(self, img_id):
        return os.path.join(self.images_dir, img_id)

    def get_annotations(self, img_id):
        """
        Read the annotation file for img_id and return a list of detections:
            { 'box': [x1, y1, x2, y2] normalized, 'class': int, 'confidence': 1.0 }
        Returns None if the annotation file is missing or unreadable.
        """
        # Determine annotation filename
        ann_name = img_id + ".txt" if img_id.lower().endswith('.jpg') else img_id + ".jpg.txt"
        ann_path = os.path.join(self.ann_dir, ann_name)
        if not os.path.exists(ann_path):
            return None
        # Load image size
        img_path = self.get_img_path(img_id)
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        detections = []
        with open(ann_path, 'r') as f:
            try:
                num = int(f.readline().strip())
            except ValueError:
                return None
            for _ in range(num):
                parts = f.readline().strip().split()
                if len(parts) < 5:
                    continue
                label = int(parts[0])
                x1, y1, x2, y2 = map(float, parts[1:5])
                # normalize
                x1n = stuff.clip01(x1 / w)
                y1n = stuff.clip01(y1 / h)
                x2n = stuff.clip01(x2 / w)
                y2n = stuff.clip01(y2 / h)
                cl=self.class_mapping[label]
                det = {
                    "box": [x1n, y1n, x2n, y2n],
                    "class": cl,
                    "confidence": 1.0
                }
                if cl!=-1:
                    detections.append(det)
        return detections

    def get_category_maps(self):
        """
        Provide groups of original class labels for higher-level categories.
        """
        return { 'person': ["pedestrian", "partial", "rider"],
                 'vehicle':[],
                 'animal':[],
                 'face': [],
                 'weapon': [],
                 'ignore': ["crowd", "ignore"]
                 }
