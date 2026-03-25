import src.dataset_util as du
import logging
import numpy as np

class FiftyOneLoader:
    cached_fiftyone_dataset = None
    cached_fifyone_ds_name = None

    def __init__(self,
                 task="val", class_names=["person"],
                 ds_params=None):
        try:
            from fiftyone.utils.huggingface import load_from_hub
        except ImportError:
            load_from_hub = None
        assert load_from_hub is not None, "try pip install fiftyone"
        fifyone_ds_name=ds_params["fiftyone_dataset"]
        self.label_field = ds_params["fifyone_label_field"]
        self.train_split=0.8
        if "train_split" in ds_params:
            self.train_split=ds_params["train_split"]

        if FiftyOneLoader.cached_fiftyone_dataset is None or FiftyOneLoader.cached_fifyone_ds_name!=fifyone_ds_name:
            logging.info(f"fiftyone load from hub")
            FiftyOneLoader.cached_fiftyone_dataset = load_from_hub(fifyone_ds_name)
            FiftyOneLoader.cached_fifyone_ds_name=fifyone_ds_name

        self.fiftyone_dataset=FiftyOneLoader.cached_fiftyone_dataset

        self.fiftyone_names = self.fiftyone_dataset.distinct(f"{self.label_field}.detections.label")
        logging.info(f"fiftyone import class names {self.fiftyone_names}")
        self.class_names=class_names
        self.info="FifytyOne dataset: "+fifyone_ds_name

        self.class_mapping=[-1]*len(self.fiftyone_names)
        for i,n in enumerate(self.fiftyone_names):
            if n in self.class_names:
                self.class_mapping[i]=self.class_names.index(n)

        self.num_images=len(self.fiftyone_dataset)

        self.image_ids=[sample.id for sample in self.fiftyone_dataset]

        rng = np.random.default_rng(seed=42)
        self.image_ids=rng.permutation(self.image_ids)

        split=int(len(self.image_ids)*self.train_split)
        if task=='train':
            self.image_ids=self.image_ids[0:split]
        else:
            self.image_ids=self.image_ids[split:]

    def get_info(self):
        return self.info

    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        logging.debug(f"{source_class} -> {dest_class}")
        self.class_mapping[self.fiftyone_names.index(source_class)]=self.class_names.index(dest_class)

    def get_annotations(self, img_id):
        sample = self.fiftyone_dataset[img_id]
        detections = sample[self.label_field].detections
        out_gts=[]
        for det in detections:
            label = det.label
            bbox = det.bounding_box  # (x, y, w, h), normalized [0, 1]
            g={"class":self.fiftyone_names.index(label),
               "box":[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]}
            out_gts.append(g)

        return out_gts

    def get_image_ids(self):
        return self.image_ids

    def get_img_path(self, img_id):
        return self.fiftyone_dataset[img_id].filepath

    def get_category_maps(self):
        return { 'person': [x for x in self.fiftyone_names if x in ["person"]],
                 'vehicle':[],
                 'animal':[],
                 'face': [],
                 'weapon': [],
                 'ignore':[],
                 }