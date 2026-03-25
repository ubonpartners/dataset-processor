import src.dataset_util as du
import os
import yaml
import stuff

# This code assumes you have already downloaded the roboflow dataset in 'yolov9' format

class RoboflowLoader:
    def __init__(self,
                 task="val", class_names=["face","person"],
                 ds_params=None):

        yaml_path=ds_params["yaml_path"]

        gdrive_sync=ds_params.get("gdrive_sync", True)
        if gdrive_sync:
            dir_path = os.path.dirname(yaml_path)
            stuff.gdrive_sync_mldata(dir_path, direction="from_drive")

        with open(yaml_path) as stream:
            try:
                ds=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()

        self.img_path=ds[task]
        self.roboflow_names=ds["names"]
        self.class_names=class_names
        self.info=ds["roboflow"]

        self.class_mapping=[-1]*len(self.roboflow_names)
        for i,n in enumerate(self.roboflow_names):
            if n in self.class_names:
                self.class_mapping[i]=self.class_names.index(n)

        if self.img_path.startswith(".."):
            self.img_path=os.path.split(yaml_path)[0]+self.img_path[2:]

        if os.path.isdir(self.img_path):
            self.all_img_files=os.listdir(self.img_path)
        else:
            print("Error: path "+self.img_path+" does not exist")
            self.all_img_files=[]
        self.all_img_files.sort()

        # try to remove multiple augmented copies of the same image
        out=[]
        prev=""
        for i,_ in enumerate(self.all_img_files):
            curr=self.all_img_files[i]

            j_curr=curr.find(".rf") # original name of image seems to be before .rf in filename
            dup=False
            if prev[0:j_curr]==curr[0:j_curr]:
                dup=True
            if not dup:
                out.append(curr)
            prev=curr

        #print(f" {len(self.all_img_files)} reduced to {len(out)} after de-duplication")
        self.all_img_files=out

    def get_info(self):
        return "roboflow :"+str(self.info)

    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        #print(f"{source_class} -> {dest_class}")
        self.class_mapping[self.roboflow_names.index(source_class)]=self.class_names.index(dest_class)

    def get_annotations(self, img_id):
        img=self.img_path+"/"+self.all_img_files[img_id]
        label=img.replace("/images","/labels")
        label=os.path.splitext(label)[0]+".txt"
        gts=du.load_ground_truth_labels(label)
        if gts==None:
            print("Error: no gts for "+label)
            return None

        out_gts=[]
        for g in gts:
            g["class"]=self.class_mapping[g["class"]]
            if g["class"]!=-1:
                out_gts.append(g)
        return out_gts

    def get_image_ids(self):
        return range(len(self.all_img_files))

    def get_img_path(self, img_id):
        #print(img_id)
        ret=self.img_path+"/"+self.all_img_files[img_id]
        if os.path.isfile(ret)==False:
            print("Error: image file "+ret+" does not exist")
            return None
        #print(ret)
        return ret

    def get_category_maps(self):
        return { 'person': [x for x in self.roboflow_names if x in ["person","Person"]],
                 'vehicle':[],
                 'animal':[],
                 'ignore':[],
                 'face': [x for x in self.roboflow_names if x in ["face"]],
                 'weapon': [x for x in self.roboflow_names if x in ["gun", "guns", "knife","Knife_Weapon","weapon", "pistol"]],
                 }