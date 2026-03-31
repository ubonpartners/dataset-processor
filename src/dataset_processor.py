import src.dataset_util as dsu
import cv2
import os
import shutil
import stuff

class DatasetProcessor:
    def __init__(self, ds_yaml, task="val",
                 class_names=None,
                 chunk_size=0,
                 append_log=False, print_log=False, face_kp=True, pose_kp=True,
                 facepose_kp=False, fold_attributes=True):
        self.chunk_size=chunk_size
        self.num_chunks=0
        self.ds_yaml=ds_yaml
        self.task=task
        self.reload_files()
        self.config_name=dsu.get_dataset_config_name(ds_yaml)
        assert self.ds_files is not None
        self.face_kp=face_kp
        self.pose_kp=pose_kp
        self.facepose_kp=facepose_kp
        assert self.facepose_kp==False, "facepose_kp not supported in dataset-processor"
        self.dataset_path=os.path.dirname(ds_yaml)
        self.append_log=append_log
        self.print_log=print_log
        if class_names is None:
            # if no class_names are passed we assume we want to set up the DP based
            # on whatever is in the yaml config
            class_names=self.ds_class_names
            self.face_kp, self.pose_kp, self.facepose_kp=dsu.map_keypoint_shape(self.kp_shape)
            if fold_attributes and self.attributes is None:
                self.attributes=dsu.attributes_from_class_names(self.ds_class_names)
                print(f"Inferred dataset attributes: {self.attributes}")

        self.fold_attributes=fold_attributes
        self.class_names=class_names
        self.od=None
        self.od_nms_thr=0.6
        self.od_max_det=600
        self.od_det_conf=0.001

        assert self.fold_attributes==True, "expanded attributed deprecated in dataset-processor"

    def reload_files(self):
        self.ds_files, self.ds_class_names, self.dataset_name, self.attributes, self.kp_shape, self.attr_nc=dsu.load_dataset_yaml(self.ds_yaml, self.task, "image")
        self.num_files=len(self.ds_files)
        if self.chunk_size!=0:
            self.num_chunks=(self.num_files+self.chunk_size-1)//self.chunk_size
            self.current_chunk=0
            self.ds_files_all=self.ds_files
            self.num_files_all=self.num_files
            self.set_chunk(0)

    def __len__(self):
        return len(self.ds_files)

    def __getitem__(self, index):
        return self.ds_files[index]

    def log(self, txt):
        if self.print_log:
            print("log: "+str(txt))
        if self.append_log:
            dsu.append_comments(self.ds_yaml, txt)

    def basic_stats(self):
        class_counts=[0]*len(self.class_names)
        image_count=0
        background_count=0
        face_point_count=0
        pose_point_count=0
        small_persons_count=0
        attributes_count=0
        person_class=self.get_class_index("person")
        for i in range(self.num_files):
            gts=self.get_gt(i)
            has_small_persons=False
            for g in gts:
                class_counts[g["class"]]+=1
                if stuff.has_face_points(g):
                    face_point_count+=1
                if stuff.has_pose_points(g):
                    pose_point_count+=1
                if g["class"]==person_class:
                    if not stuff.is_large(g):
                        has_small_persons=True
                if "attrs" in g:
                    attributes_count+=1
            if has_small_persons:
                small_persons_count+=1
            if len(gts)==0:
                background_count+=1
            image_count+=1

        ret={"task":self.task,
            "class_counts":class_counts,
            "image_count":image_count,
            "backgrounds":background_count,
            "image_small_person_count":small_persons_count,
            "has_face_kp":face_point_count,
            "has_pose_kp":pose_point_count,
            "has_attributes":attributes_count}
        return ret

    def log_stats(self):
        stats=self.basic_stats()
        self.log(str(stats))

    def set_chunk(self, cn):
        self.current_chunk=cn
        start=cn*self.chunk_size
        end=min((cn+1)*self.chunk_size, self.num_files_all)
        self.num_files=end-start
        self.ds_files=self.ds_files_all[start:end]

    def get_gt(self, index):
        if index>=len(self.ds_files):
            print(f"ERROR: get_gt index {index} not in image list {len(self.ds_files)}")
            return None
        label_file=self.ds_files[index]["label"]
        gts=dsu.load_ground_truth_labels(label_file, attr_nc=getattr(self, "attr_nc", 0), kpt_shape=getattr(self, "kp_shape", None))
        if gts is None:
            return None
        gt_class_remap=[self.class_names.index(x) if x in self.class_names else -1 for x in self.ds_class_names]
        for g in gts:
            ci=g["class"]
            if ci>=len(gt_class_remap) or ci<0:
                print(f"ERROR: bad GT class index {ci} (max {len(gt_class_remap)} {self.ds_class_names})")
                g["class"]=-1
                continue
            g["class"]=gt_class_remap[ci]
        ret=[g for g in gts if g["class"]!=-1]
        if self.fold_attributes:
            ret=stuff.fold_detections_to_attributes(ret, self.class_names, self.attributes)
        #print(f"remap keypoints {self.facepose_kp}")
        #print(label_file, ret)
        stuff.map_keypoints(ret, self.face_kp, self.pose_kp, self.facepose_kp)
        return ret

    def get_gt_file_path(self, index):
        label_file=self.ds_files[index]["label"]
        return label_file

    def get_class_names(self):
        return self.class_names

    def get_class_index(self, class_name):
        if class_name in self.class_names:
            return self.class_names.index(class_name)
        return -1

    def add(self, name, img_file, dets, add_exif=False, ignore_class=None,
            max_width=4096, max_height=4096, max_bpp=4, fix_orientation=False, yuv420=False):
        stuff.map_keypoints(dets, self.face_kp, self.pose_kp, self.facepose_kp)
        dst_img=self.dataset_path+"/"+self.task+"/images/"+name+".jpg"
        dst_label=self.dataset_path+"/"+self.task+"/labels/"+name+".txt"
        shutil.copyfile(img_file, dst_img)
        stuff.image_copy_resize(dst_img, max_width, max_height, max_bpp, fix_orientation=fix_orientation, yuv420=yuv420) # transcode jpeg if silly size

        dets_no_ignore=[]
        ignore_boxes=[]
        unignore_boxes=[]
        if ignore_class is not None:
            for d in dets:
                if d["class"]==ignore_class:
                    ignore_boxes.append(d["box"])
                else:
                    dets_no_ignore.append(d)
                    unignore_boxes.append(d["box"])
        else:
            dets_no_ignore=dets

        if len(ignore_boxes)!=0:
            #print("rendering ignore boxes", dst_img)
            dsu.render_ignore_boxes(dst_img, ignore_boxes, unignore_boxes)

        if add_exif:
            ok=stuff.image_append_exif_comment(dst_img, f"config={self.config_name};origin={img_file}")
            if ok is False:
                print(f"dataset_processor add: rejecting file {img_file}")
                stuff.rm(dst_img)
                return False

        an_txt=dsu.write_annotations(dets_no_ignore,
                                     include_face=self.face_kp,
                                     include_pose=self.pose_kp,
                                     include_facepose=self.facepose_kp,
                                     include_attrs=True)

        with open(dst_label, 'w') as file:
            file.write(an_txt)
        return True

    def delete(self, index):
        stuff.rm(self.ds_files[index]["label"])
        stuff.rm(self.ds_files[index]["image"])
        self.ds_files[index]["label"]=None
        self.ds_files[index]["image"]=None

    def rename(self, index, name):
        dst_img=self.dataset_path+"/"+self.task+"/images/"+name+".jpg"
        dst_label=self.dataset_path+"/"+self.task+"/labels/"+name+".txt"
        stuff.rename(self.ds_files[index]["label"], dst_label)
        stuff.rename(self.ds_files[index]["image"], dst_img)
        self.ds_files[index]["label"]=dst_label
        self.ds_files[index]["image"]=dst_img

    def get_exif_data(self, index):
        return dsu.image_get_exif_data(self.ds_files[index]["image"])

    def get_image_origin(self, index):
        exif_data=self.get_exif_data(index)
        origin=None
        if "origin" in exif_data:
            origin=exif_data["origin"]
        return origin

    def append_exif_comment(self, index, comment):
        stuff.image_append_exif_comment(self.ds_files[index]["image"], comment)

    def set_object_detector(self, od_model,
                          imgsz=640,
                          thr=0.001,
                          rect=False,
                          half=True,
                          batch_size=32,
                          remap_pose=True,
                          max_det=600,
                          get_feats=False):

        if self.od!=None:
            del self.od
            self.od=None

        stuff.platform_stuff.platform_clear_caches()
        self.od_cache={}

        if od_model==None:
            return

        self.imgsz=imgsz
        self.od_batch_size=batch_size
        self.od_det_conf=max(thr, 0.001)
        self.od_rect=rect
        self.od_half=half
        self.num_gpus=stuff.platform_stuff.platform_num_gpus()
        self.od_remap_pose=remap_pose
        self.rf_detr_model=None
        self.augment=False

        self.od=stuff.InferenceWrapper(od_model,
                                       half=half,
                                       rect=rect,
                                       thr=thr,
                                       batch_size=batch_size,
                                       nms_thr=self.od_nms_thr,
                                       imgsz=imgsz,
                                       max_det=max_det,
                                       class_names=self.class_names,
                                       attributes=self.attributes,
                                       face_kp=self.face_kp,
                                       pose_kp=self.pose_kp,
                                       facepose_kp=self.facepose_kp,
                                       fold_attributes=self.fold_attributes,
                                       get_feats=get_feats)

        self.od_num_params=self.od.yolo_num_params
        self.od_num_flops=self.od.yolo_num_flops


    def get_img(self, index):
        if 0 <= index < len(self.ds_files):
             return cv2.imread(self.ds_files[index]["image"])
        else:
            return None

    def replace_img(self, index, image):
        cv2.imwrite(self.ds_files[index]["image"], image)

    def get_img_path(self, index):
        if 0 <= index < len(self.ds_files):
            return self.ds_files[index]["image"]
        else:
            return None

    def get_label_path(self, index):
        return self.ds_files[index]["label"]

    def get_img_name(self, index):
        label_file=self.ds_files[index]["label"]
        return dsu.name_from_file(label_file)

    def replace_annotations(self, index, an):
        txt=dsu.write_annotations(an, include_face=self.face_kp, include_pose=self.pose_kp, include_facepose=self.facepose_kp, include_attrs=True)
        label_file=self.get_gt_file_path(index)
        with open(label_file, 'w') as file:
            file.write(txt)

    def get_detections(self, index):
        out_det=[]
        if self.od!=None:
            out_det=self.od.infer_cached(index, self.num_files, self.get_img_path)
        return out_det

def dataset_merge_and_delete(src, dest):
    dsu.append_comments(dest, "====== Merge dataset "+src+" ======")
    dsu.dataset_copy_comments(src, dest)

    dest_path=dsu.get_dataset_path(dest)

    for task in ["val","train"]:
        x=DatasetProcessor(src, task=task)
        for i in range(x.num_files):
            img=x.get_img_path(i)
            label=x.get_label_path(i)
            stuff.rename(img, dest_path+"/"+task+"/images/"+os.path.basename(img))
            stuff.rename(label, dest_path+"/"+task+"/labels/"+os.path.basename(label))

    stuff.rmdir(os.path.dirname(src))

def dataset_delete(dataset_yaml):
    stuff.rmdir(os.path.dirname(dataset_yaml))
