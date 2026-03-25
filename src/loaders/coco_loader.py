import src.dataset_util as du
import os
import stuff

# This code assumes you have downloaded the coco and coco-pose datasets/annotations
# each of the 'paths' should point to a folder containing:
# annotations  images  labels  train2017.txt  val2017.txt

class CocoLoader:
    def __init__(self,
                 coco_path="/mldata/downloaded_datasets/coco",
                 coco_pose_path="/mldata/downloaded_datasets/coco-pose",
                 task="val", class_names=["face","person"], ds_params=None):
        # Paths to MS COCO stuff
        gdrive_sync=ds_params.get("gdrive_sync", True)
        if gdrive_sync:
            stuff.gdrive_sync_mldata(coco_path, direction="from_drive")
            stuff.gdrive_sync_mldata(coco_pose_path, direction="from_drive")

        self.task=task
        coco_annotation_path = coco_path+'/annotations/instances_'+self.task+'2017.json'
        self.coco_images_dir = coco_path+'/images/'+self.task+'2017'
        coco_kpt_path = coco_pose_path+'/annotations/person_keypoints_'+self.task+'2017.json'
        # Load COCO dataset
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError("Please install pycocotools: pip install pycocotools")
        self.coco = COCO(coco_annotation_path)
        self.coco_kp = COCO(coco_kpt_path)
        self.class_names=class_names
        self.class_mappings={}
        self.category_list=[]

    def get_info(self):
        return "coco https://cocodataset.org/"

    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        coco_category_id = self.coco.getCatIds(catNms=[source_class])[0]
        self.class_mappings[coco_category_id]=self.class_names.index(dest_class)
        self.category_list.append(coco_category_id)

    def get_coco_anns(self, img_id, anns):
        img_info = self.coco.loadImgs([img_id])[0]
        img_width=img_info['width']
        img_height=img_info['height']
        out_det=[]
        for ann in anns:
            bbox=ann['bbox']
            x=bbox[0]/img_width
            y=bbox[1]/img_height
            w=bbox[2]/img_width
            h=bbox[3]/img_height
            an_box=[stuff.clip01(x),stuff.clip01(y),stuff.clip01(x+w),stuff.clip01(y+h)]
            det={"box":an_box,
                "class":self.class_mappings[ann['category_id']],
                "confidence":1.0}
            if 'keypoints' in ann:
                keypoints=ann['keypoints']
                for i in range(len(keypoints)//3):
                    keypoints[i*3+0]=keypoints[i*3+0]/img_width
                    keypoints[i*3+1]=keypoints[i*3+1]/img_height
                    keypoints[i*3+2]=0 if keypoints[i*3+2]==0 else 1
                det["pose_points"]=keypoints
            out_det.append(det)
        return out_det

    def get_non_pose_annotations(self, img_id):
        ann_ids_crowd = self.coco.getAnnIds(imgIds=img_id, catIds=self.category_list, iscrowd=True)
        if len(ann_ids_crowd)!=0:
            return None
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.category_list, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        return self.get_coco_anns(img_id, anns)

    def get_pose_annotations(self, img_id):
        ann_ids = self.coco_kp.getAnnIds(imgIds=img_id, catIds=self.category_list, iscrowd=False)
        anns = self.coco_kp.loadAnns(ann_ids)
        return self.get_coco_anns(img_id, anns)

    def get_annotations(self, img_id):
        gts=self.get_pose_annotations(img_id)
        gts2=self.get_non_pose_annotations(img_id)
        if gts2==None:
            return None
        for g in gts2:
            best_iou, best_match, idx=du.best_iou_match(g, gts)
            if best_iou<0.8:
                gts.append(g)
        return gts

    def get_image_ids(self):
        image_ids=set()
        for c in self.category_list:
            image_ids=image_ids.union(set(self.coco.getImgIds(catIds=[c])))
        return list(image_ids)

    def get_img_path(self, img_id):
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.coco_images_dir, img_info['file_name'])
        return img_path

    def get_category_maps(self):
        return { 'person':['person'],
                 'vehicle':['car','bicycle','motorcycle','train','truck','bus','airplane','boat'],
                 'animal':['cat','dog','horse','sheep','cow','bird','elephant','bear','zebra','giraffe'],
                 'face':[],
                 'weapon':[],
                 'confused':["knife", "baseball bat", "teddy bear"],
                 'ignore':[],
                 'background':["fire hydrant", "stop sign", "stop sign", "parking meter", "bench", "umbrella",
                               "suitcase", "handbag","bottle", "wine glass", "fork", "spoon", "bowl", "sandwich",
                               "broccoli", "carrot", "hot dog","pizza", "donut", "chair", "couch", "potted plant",
                               "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
                               "sink","refrigerator", "book", "clock", "vase", "hair drier", "toothbrush"]}
