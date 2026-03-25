import os
import numpy as np
import stuff

class RawimageLoader:
    def __init__(self,
                 task="val", class_names=["face","person"],
                 ds_params=None):

        self.images_path = ds_params["images_path"]

        gdrive_sync=ds_params.get("gdrive_sync", True)
        if gdrive_sync:
            stuff.gdrive_sync_mldata(self.images_path, direction="from_drive")

        self.task = task
        self.train_split=ds_params.get("train_split", 1.0)

        self.all_images=os.listdir(self.images_path)

        rng = np.random.default_rng(seed=42)
        self.all_images=rng.permutation(self.all_images)

        split=int(len(self.all_images)*self.train_split)
        if task=='train':
            self.all_images=self.all_images[0:split]
        else:
            self.all_images=self.all_images[split:]

    def get_info(self):
        return "Rawimages loader"

    def add_category_mapping(self, source_class, dest_class):
        pass

    def get_image_ids(self):
        return self.all_images

    def get_img_path(self, img_id):
        return self.images_path+"/"+img_id

    def get_annotations(self, img_id):
        return []

    def get_category_maps(self):
        return {}
