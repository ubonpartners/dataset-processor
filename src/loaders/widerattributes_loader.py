import src.dataset_util as du
import json
import cv2

# https://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html

class WiderAttributesLoader:
    def __init__(self,
                 widerface_path="/mldata/downloaded_datasets/wider_attributes",
                 task="val", class_names=["face","person"], ds_params=None):
        with open(widerface_path+"/"+"wider_attribute_trainval.json") as json_file:
            self.labels=json.load(json_file)
        self.class_names=class_names
        self.image_path=widerface_path+"/Image"
        self.classes=[  "person",
                        "male","female",
                        "longhair", "no-longhair",
                        "sunglasses", "no-sunglasses",
                        "hat", "no-hat",
                        "tshirt", "no-tshirt",
                        "longsleeves","no-longsleeves",
                        "formal", "no-formal",
                        "shorts", "no-shorts",
                        "jeans", "no-jeans",
                        "longtrousers", "no-longtrousers",
                        "skirt", "no-skirt",
                        "facemask", "no-facemask",
                        "logo", "no-logo",
                        "stripes", "no-stripes"]
        self.class_mapping=[-1]*len(self.classes)
        self.images=[]
        for i in self.labels["images"]:
            if i["file_name"].startswith(task):
                self.images.append(i)
        print(f"WiderAttributeLoader: Loaded {len(self.images)} images")
    
    def get_info(self):
        return "widerattributes: https://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html"
    
    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        if source_class in self.classes and dest_class in self.class_names:
            self.class_mapping[self.classes.index(source_class)]=self.class_names.index(dest_class)

    def get_annotations(self, img_id):
        img=cv2.imread(self.get_img_path(img_id))
        height, width, c = img.shape
        gts=[]
        targets=self.images[img_id]["targets"]
        for t in targets:
            x=t["bbox"][0]/width
            y=t["bbox"][1]/height
            w=t["bbox"][2]/width
            h=t["bbox"][3]/height
            an_box=[x,y,x+w,y+h]
            gt={"box":an_box, 
                "class":self.class_mapping[0],
                "confidence":1.0,
                "face_points":[0]*15,
                "pose_points":[0]*51}

            attributes=[]
            for i in range(len(t["attribute"])):
                a=t["attribute"][i]
                if a==0:
                    continue
                if a==1:
                    cl=1+2*i
                else:
                    cl=1+2*i+1
                gt_attr={"box":an_box, 
                        "class":self.class_mapping[cl],
                        "confidence":1.0,
                        "face_points":[0]*15,
                        "pose_points":[0]*51}
                attributes.append(self.classes[cl])
                if gt_attr["class"]!=-1:
                    gts.append(gt_attr)
            gt["attributes"]=attributes
            if gt["class"]!=-1:
                gts.append(gt)
        return gts

    def get_image_ids(self):
        return range(len(self.images))

    def get_img_path(self, img_id):
        return self.image_path+"/"+self.images[img_id]["file_name"]
    
    def get_category_maps(self):
        return { 'person':['person'],
                 'vehicle':[],
                 'animal':[],
                 'face':[''],
                 'person_attr_male':['male'],
                 'person_attr_female':['female'],
                 'person_attr_hat': ['hat'],
                 'person_attr_no_hat': ['no-hat'],
                 'person_attr_sunglasses': ['sunglasses'],
                 'person_attr_no_sunglasses': ['no-sunglasses'],
                 'person_attr_facemask': ['facemask'],
                 'person_attr_no_facemask': ['no-facemask']
        }