import src.dataset_util as du
import os
import stuff

# This code assumes you have already downloaded the widerface dataset
# and labels
# http://shuoyang1213.me/WIDERFACE/
# https://drive.google.com/file/d/1FsZ0ACah386yUufi0E_PVsRW_0VtZ1bd/view?usp=sharing
#
# ls /mldata/downloaded_datasets/widerface/
# WIDER_test  WIDER_train  WIDER_val  yolov7-face-label

class WiderfaceLoader:
    def __init__(self,
                 widerface_path="/mldata/downloaded_datasets/widerface",
                 task="val", class_names=["face","person"], ds_params=None):

        gdrive_sync=ds_params.get("gdrive_sync", True)
        if gdrive_sync:
            stuff.gdrive_sync_mldata(widerface_path, direction="from_drive")

        self.class_names=class_names
        self.face_class_index=None
        if "face" in self.class_names:
            self.face_class_index=self.class_names.index("face")
        self.task=task
        if task=="val":
            base_folder=widerface_path+"/WIDER_val/images"
            self.labels=widerface_path+"/yolov7-face-label/val"
        else:
            base_folder=widerface_path+"/WIDER_train/images"
            self.labels=widerface_path+"/yolov7-face-label/train"
        self.all_files = []
        for root, _, files in os.walk(base_folder):
            for file in files:
                self.all_files.append(os.path.join(root, file))

    def get_info(self):
        return "widerface"

    def add_category_mapping(self, source_class, dest_class):
        self.face_class_index=None
        if "face" in self.class_names:
            self.face_class_index=self.class_names.index("face")

    def get_annotations(self, img_id):
        img=self.all_files[img_id]
        label=self.labels+"/"+du.name_from_file(img)+".txt"
        if self.face_class_index is None:
            return []
        gts=du.load_source_yolo_labels(label)
        for g in gts:
            g["class"]=self.face_class_index
        return gts

    def get_image_ids(self):
        return range(len(self.all_files))

    def get_img_path(self, img_id):
        return self.all_files[img_id]

    def get_category_maps(self):
        return { 'person':[],
                 'vehicle':[],
                 'animal':[],
                 'ignore':[],
                 'face':['face']}
