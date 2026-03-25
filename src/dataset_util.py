import os
import yaml
import cv2
import math
import copy
import stuff
import base64
import numpy as np
from PIL import Image, ImageDraw

from datetime import datetime
from scipy.optimize import linear_sum_assignment
from src.loaders.coco_loader import CocoLoader
from src.loaders.openimages_loader import OpenImagesLoader
from src.loaders.o365_loader import O365Loader
from src.loaders.widerface_loader import WiderfaceLoader
from src.loaders.roboflow_loader import RoboflowLoader
from src.loaders.weapondetection_loader import WeaponDetectionLoader
from src.loaders.widerattributes_loader import WiderAttributesLoader
from src.loaders.fiftyone_loader import FiftyOneLoader
from src.loaders.trackset_loader import TracksetLoader
from src.loaders.widerperson_loader import WiderPersonLoader
from src.loaders.crowdpose_loader import CrowdPoseLoader
from src.loaders.rawimage_loader import RawimageLoader

ATTR_NC_V10 = 50

def mldata_folder():
    d=os.environ.get('MLDATA_LOCATION')
    if d==None:
        return "/mldata"
    return d

def unique_dataset_name(dataset_name):
    ver=0
    while True:
        name=dataset_name
        if not os.path.isdir(mldata_folder()+"/"+name):
            break
        name+="-"+datetime.today().strftime('%y%m%d')
        if ver!=0:
            name+="-v"+str(ver)
        if not os.path.isdir(mldata_folder()+"/"+name):
            break
        ver+=1
    return name

def name_from_file(x):
    ext=""
    if isinstance(x, str):
        if "+" in x:
            t=x.split("+")
            x=t[0]
            ext+=t[1]
        if ":" in x:
            t=x.split(":")
            x=t[0]
            ext+=t[1]+" "
        if x.endswith(".engine"):
            ext+="TRT "
        if len(ext)>0:
            ext="("+ext[:-1]+")"
    return os.path.splitext(os.path.basename(x))[0]+ext

def map_keypoint_shape(kp_shape):
    # returns face_points, pose_points, facepose_points
    if kp_shape==[19,3]:
        return False, False, True
    if kp_shape==[22,3]:
        return True, True, False
    if kp_shape==[17,3]:
        return False, True, False
    if kp_shape==[5,3]:
        return True, False, False
    assert kp_shape==None
    return False, False, False

def load_dataset_yaml(dataset, task="val", file_key="both"):
    """
    'Load' a yaml dataset and return arrays of all the corresponding image and label files

    Args:
        dataset: path of yaml file
        task: train or val
        file_key: if 'both' returns img/label where both exist, if 'image' returns all images, if 'label' returns all labels

    Returns:
        list of dict, each containing "label" and "image" keys with paths
        list of class names
    """

    assert(file_key=="both" or file_key=="image" or file_key=="label")

    with open(dataset) as stream:
        try:
            ds=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None, None, None, None, None, 0

    if not "names" in ds:
        print(f"Could not class names in dataset {dataset} yaml")
        return None, None, None, None, None, 0

    names=ds["names"]

    if "dataset_name" in ds:
        dataset_name=ds["dataset_name"]
    else:
        dataset_name=name_from_file(dataset)

    attributes=None
    if "attributes" in ds:
        attributes=ds["attributes"]

    # v10 (split) format: attributes: true and/or attr_nc > 0, optionally attr_names
    attr_nc=int(ds.get("attr_nc", 0))
    if ds.get("attributes") is True and attr_nc == 0:
        attr_nc=50
    if "attr_names" in ds:
        attributes=ds["attr_names"]
    elif attributes is True and attr_nc > 0:
        attributes=[f"attr_{i}" for i in range(attr_nc)]

    if not task in ds:
        print(f"Could not find {task} in dataset {dataset} yaml")
        return None, None, None, None, None, 0

    kp_shape=None
    if "kpt_shape" in ds:
        kp_shape=ds["kpt_shape"]

    ds[task]=str(ds[task])

    val=ds[task].strip()

    if val.startswith('[') and val.endswith(']'):
        val=val[1:-1].split(",")
        val=[v.strip() for v in val]
    else:
        val=[val]

    for i,v in enumerate(val):
        if v.startswith('"') and v.endswith('"'):
            v=v[1:-1]
        if v.startswith("'") and v.endswith("'"):
            v=v[1:-1]
        val[i]=v

    if "path" in ds:
        path=ds["path"]
        if path==".":
            path=os.path.dirname(dataset)
        val=[os.path.join(path, v) for v in val]

    files=[]

    for v in val:
        for x in ["/images","/labels"]:
            if v.endswith(x):
                v=v[:-len(x)]
        img_path=os.path.join(v, "images")
        label_path=os.path.join(v, "labels")
        if file_key=="image" or file_key=="both":
            if os.path.isdir(img_path):
                imgs=sorted(os.listdir(img_path))
                for i in imgs:
                    l=os.path.splitext(i)[0]+".txt"
                    img=os.path.join(img_path, i)
                    lab=os.path.join(label_path,l)
                    f={}
                    if os.path.isfile(img) and os.path.isfile(lab):
                        f={"image":img, "label":lab}
                        files.append(f)
                    elif file_key=="image" and os.path.isfile(img):
                        f={"image":img, "label":None}
                        files.append(f)
        else:
            if os.path.isdir(label_path):
                labels=sorted(os.listdir(label_path))
                for l in labels:
                    # Build the corresponding image path from the label filename.
                    img_name=os.path.splitext(l)[0]+".jpg"
                    img=os.path.join(img_path, img_name)
                    lab=os.path.join(label_path,l)
                    if os.path.isfile(img) and os.path.isfile(lab):
                        f={"image":img, "label":lab}
                        files.append(f)
                    elif file_key=="label" and os.path.isfile(lab):
                        f={"image":None, "label":lab}
                        files.append(f)

    subsample=ds.get("subsample", None)
    if subsample!=None:
        files=files[::subsample]

    return files, names, dataset_name, attributes, kp_shape, attr_nc

def make_dataset_yaml(name, config_name=None,
                      class_names=None, face_kp=True, pose_kp=True,
                      facepose_kp=False,
                      attributes=None,
                      poseattr=None):
    """
    Write out a basic dataset yaml description
    """
    path=mldata_folder()+"/"+name

    txt="# autogenerated dataset YAML file for "+name+"\n"
    txt+="# created: "+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+"\n"
    txt+="dataset_name: "+os.path.split(name)[1]+"\n"
    if config_name is not None:
        txt+="config_name: "+config_name+"\n"
    txt+="path: "+path+"\n"
    txt+="train: train/images\n"
    txt+="val: val/images\n"
    txt+="nc: "+str(len(class_names))+"\n"
    num_kpt=0
    flip_idx=[]
    if face_kp and pose_kp:
        txt+="# key points are POSE+FACE\n"
        txt+="# [0x3:5x3] FACE are only valid for face boxes\n"
        txt+="# [5x3:22x3] POSE are only valid for person boxes; coco-style\n"
        num_kpt=22
        flip_idx=[1, 0, 2, 4, 3, 5, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20]
    elif face_kp:
        txt+="# key points are FACE only, 5x3\n"
        num_kpt=5
        flip_idx=[1, 0, 2, 4, 3]
    elif pose_kp:
        txt+="# key points are POSE only, same format 17x3 as coco-pose\n"
        num_kpt=17
        flip_idx=[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    elif facepose_kp:
        txt+="# key points are FACEPOSE = 19 sets of [x,y,conf]\n"
        txt+="# first 17 points are as coco-pose, 18/19 are mouth\n"
        num_kpt=19
        flip_idx=[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]
    if poseattr is not None:
        for i in range(len(poseattr)):
            flip_idx.append(num_kpt+i)
        num_kpt+=len(poseattr)
    if num_kpt!=0:
        txt+=f"kpt_shape: [{num_kpt}, 3]\n"
        txt+=f"flip_idx: {flip_idx}\n"
    txt+="names:\n"
    for c in class_names:
        txt+="    - "+c+"\n"
    if poseattr is not None:
        txt+="poseattr:\n"
        for p in poseattr:
            txt+="    - "+p+"\n"
    if attributes is not None:
        txt+="attributes:\n"
        for a in attributes:
            txt+="    - "+a+"\n"

    stuff.makedir(path)
    stuff.makedir(path+"/train/images")
    stuff.makedir(path+"/train/labels")
    stuff.makedir(path+"/val/images")
    stuff.makedir(path+"/val/labels")
    yaml_file_name=path+"/dataset.yaml"
    with open(yaml_file_name, 'w') as file:
        file.write(txt)
    return yaml_file_name


def make_dataset_yaml_v10(name, config_name=None,
                           class_names=None, attr_names=None,
                           face_kp=True, pose_kp=True, facepose_kp=False,
                           attr_nc=ATTR_NC_V10):
    """
    Write dataset YAML in v10 (split) format for Ultralytics attributes branch:
    nc=5, names (main classes), attributes: true, attr_nc=50, attr_label_format: "split", attr_names.
    Creates directory structure. class_names should be the 5 main classes; attr_names the 50 attribute names.
    """
    path = mldata_folder() + "/" + name
    txt = "# v10 (split) dataset YAML for attributes branch\n"
    txt += "path: " + path + "\n"
    txt += "train: train/images\n"
    txt += "val: val/images\n"
    txt += "nc: 5\n"
    txt += "attributes: true\n"
    txt += "attr_nc: " + str(attr_nc) + "\n"
    txt += 'attr_label_format: "split"\n'
    num_kpt = 0
    flip_idx = []
    if face_kp and pose_kp:
        num_kpt = 22
        flip_idx = [1, 0, 2, 4, 3, 5, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20]
    elif face_kp:
        num_kpt = 5
        flip_idx = [1, 0, 2, 4, 3]
    elif pose_kp:
        num_kpt = 17
        flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    elif facepose_kp:
        num_kpt = 19
        flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]
    if num_kpt != 0:
        txt += "kpt_shape: [" + str(num_kpt) + ", 3]\n"
        txt += "flip_idx: " + str(flip_idx) + "\n"
    txt += "names:\n"
    main_names = (class_names or [])[:5]
    while len(main_names) < 5:
        main_names.append("unknown")
    for c in main_names:
        txt += "    - " + c + "\n"
    txt += "attr_names:\n"
    attrs = (attr_names or [])[:attr_nc]
    while len(attrs) < attr_nc:
        attrs.append("attr_" + str(len(attrs)))
    for a in attrs:
        txt += "    - " + str(a) + "\n"
    if config_name is not None:
        txt += "config_name: " + config_name + "\n"
    stuff.makedir(path)
    stuff.makedir(path + "/train/images")
    stuff.makedir(path + "/train/labels")
    stuff.makedir(path + "/val/images")
    stuff.makedir(path + "/val/labels")
    yaml_file_name = path + "/dataset.yaml"
    with open(yaml_file_name, 'w') as file:
        file.write(txt)
    return yaml_file_name


import os

def load_ground_truth_labels(label_file, attr_nc=None, kpt_shape=None):
    """
    Load label file. Supports:
    - Internal format: rows with 'A' separator (bbox/keypoints before A, attrs after).
    - v10 (split) format: when attr_nc>0 and kpt_shape set, rows are cls, bbox(4), attr_0..attr_{attr_nc-1}, keypoints.
    """
    # 0) Early exits
    if not label_file or not os.path.isfile(label_file):
        return None
    if os.path.getsize(label_file) == 0:
        return []

    # 1) Read all non-blank lines, split once to detect 'A' index
    lines = []
    a_idx = None
    with open(label_file, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            lines.append(toks)
            if a_idx is None and 'A' in toks:
                a_idx = toks.index('A')
    # If no data lines at all
    if not lines:
        return []

    # v10 format: no 'A', fixed length 5+attr_nc+nkpt*ndim (or 5+attr_nc if no keypoints)
    nkpt = 0
    ndim = 0
    if kpt_shape is not None and len(kpt_shape) >= 2:
        nkpt = int(kpt_shape[0])
        ndim = int(kpt_shape[1])
    v10_len = (5 + (attr_nc or 0) + nkpt * ndim) if (attr_nc and attr_nc > 0) else 0

    results = []

    for toks in lines:
        # split tokens into numeric part and attrs part (internal format)
        if a_idx is not None:
            pre = toks[:a_idx]
            post = toks[a_idx+1:]
        else:
            pre = toks
            post = []

        # convert pre-A tokens to floats (or full line for v10)
        try:
            nums = [float(x) for x in pre]
        except ValueError as e:
            print(f"Skipping line (non-numeric before A): {toks}")
            continue

        # v10 (split) format: one row = cls, bbox(4), attrs(attr_nc), keypoints
        if a_idx is None and v10_len > 0 and len(nums) == v10_len:
            try:
                cls = int(nums[0])
                x_center, y_center, width, height = nums[1:5]
                x1 = x_center - 0.5 * width
                y1 = y_center - 0.5 * height
                x2 = x_center + 0.5 * width
                y2 = y_center + 0.5 * height
                gt = {
                    "box": [x1, y1, x2, y2],
                    "class": cls,
                    "confidence": 1.0,
                    "attrs": nums[5:5+attr_nc],
                }
                kpt_start = 5 + attr_nc
                kpt_end = kpt_start + nkpt * ndim
                kpt_slice = nums[kpt_start:kpt_end] if nkpt > 0 else []
                face_kp, pose_kp, facepose_kp = map_keypoint_shape(kpt_shape) if nkpt > 0 else (False, False, False)
                if face_kp and pose_kp and len(kpt_slice) >= 66:  # 22*3
                    gt["face_points"] = kpt_slice[0:15]
                    gt["pose_points"] = kpt_slice[15:66]
                elif pose_kp and len(kpt_slice) >= 51:  # 17*3
                    gt["pose_points"] = kpt_slice[0:51]
                elif face_kp and len(kpt_slice) >= 15:  # 5*3
                    gt["face_points"] = kpt_slice[0:15]
                elif facepose_kp and len(kpt_slice) >= 57:  # 19*3
                    gt["facepose_points"] = kpt_slice[0:57]
                results.append(gt)
            except Exception as e:
                print(f"Error parsing v10 line {toks[:10]}...: {e}")
            continue

        # apply original logic on nums (internal or legacy)
        try:
            cls = int(nums[0])
            x_center, y_center, width, height = nums[1:5]
            x1 = x_center - 0.5 * width
            y1 = y_center - 0.5 * height
            x2 = x_center + 0.5 * width
            y2 = y_center + 0.5 * height

            gt = {
                "box": [x1, y1, x2, y2],
                "class": cls,
                "confidence": 1.0
            }

            L = len(nums)
            if L in (20, 71):
                gt["face_points"] = nums[5:20]
            if L == 56:
                gt["pose_points"] = nums[5:56]
            if L == 62:
                gt["facepose_points"] = nums[5:62]
            if L == 71:
                gt["pose_points"] = nums[20:71]

            # parse attrs (post-A) to floats, if any
            if post:
                try:
                    attrs = [float(x) for x in post]
                    if attrs:
                        gt["attrs"] = attrs
                except ValueError:
                    print(f"Ignoring non-numeric attrs: {post}")

            results.append(gt)

        except Exception as e:
            print(f"Error parsing numeric values {nums}: {e}")

    return results

def load_ground_truth_labels_old(label_file):
    if not label_file or not os.path.isfile(label_file):
        return None
    if os.path.getsize(label_file) == 0:
        return []  # Empty file: return empty list
    try:
        data = np.loadtxt(label_file, ndmin=2)  # ensure 2D even if single line
    except Exception as e:
        print(f"Failed to load {label_file}: {e}")
        return None

    results = []
    for row in data:
        try:
            cls = int(row[0])
            x_center, y_center, width, height = row[1:5]
            x1 = x_center - 0.5 * width
            y1 = y_center - 0.5 * height
            x2 = x_center + 0.5 * width
            y2 = y_center + 0.5 * height

            gt = {
                "box": [x1, y1, x2, y2],
                "class": cls,
                "confidence": 1.0
            }

            val_len = len(row)
            if val_len == 20 or val_len == 71:
                gt["face_points"] = row[5:20].tolist()
            if val_len == 56:
                gt["pose_points"] = row[5:56].tolist()
            if val_len == 74:
                gt["facepose_points"] = row[5:74].tolist()
            if val_len == 71:
                gt["pose_points"] = row[20:71].tolist()

            results.append(gt)

        except Exception as e:
            print(f"Error parsing line in {label_file}: {e}")

    return results

def load_cached_detections(model_path, dataset, f):
    """
    load_cached_detections

    Args:
        path to model .pt file
        path to dataset yaml file
        path to where a label .txt file would be if it existed

    Returns:
        flat list of boxes co-ords x1,x2,y1,y2
        list of box classes
        list of confidences
    """
    model=name_from_file(model_path)
    dataset=name_from_file(dataset)
    f=os.path.basename(f)
    path=os.path.join("/mldata/detections",dataset,model,f)

    if not os.path.isfile(path):
        return None, None, None

    boxes=[]
    classes=[]
    confidences=[]

    with open(path, 'r') as f:
        for line in f:
            vals=[float(x) for x in line.strip().split()]
            box=[vals[1]-0.5*vals[3], vals[2]-0.5*vals[4], vals[1]+0.5*vals[3], vals[2]+0.5*vals[4]]
            boxes.append(box)
            classes.append(vals[0])
            confidences.append(vals[5])
    return boxes, classes, confidences

def dedup_gt(gts, iou_thr=0.6):
    """
    Remove multiple overlapping instances of the same class
    This can happen because things that are different in the original dataset may be mapped to the same label
    e.g. head,face or boy,person
    """
    for i,g in enumerate(gts):
        for j,g2 in enumerate(gts):
            if i==j or g2["confidence"]==0 or g["confidence"]==0:
                continue
            if g["class"]==g2["class"]:
                if stuff.box_iou(g["box"], g2["box"])>iou_thr:
                    if stuff.better_annotation(g, g2): # delete the 'worst' one
                        g2["confidence"]=0
                    else:
                        g["confidence"]=0
    out=[]
    for g in gts:
        if g["confidence"]>0:
            out.append(g)
    return out

def best_iou_match(x, to_check):
    """
    check against list and find match of same class with largest IOU. Retun triple iou, matching item, item index
    """
    best_iou=0
    best_g=None
    best_i=-1
    for i,g in enumerate(to_check):
        if x["class"]==g["class"]:
            iou=stuff.box_iou(x["box"],g["box"])
            if iou>best_iou:
                best_iou=iou
                best_g=g
                best_i=i
    return best_iou, best_g, best_i

def kp_iou(kp_gt, kp_det, s, num_pt):
    """
    Computes the keypoint iou between two boxes

    Args:
        kpgt, gpdet: list [x,y,conf]*npoint
        s: GT box area
    Returns:
        float iou 0-1
    """
    # https://learnopencv.com/object-keypoint-similarity/
    if num_pt==5:
        scales=[0.025, 0.025, 0.026, 0.025, 0.025]
    elif num_pt==17:
        scales=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    elif num_pt==22:
        scales=[0.025, 0.025, 0.026, 0.025, 0.025, 0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    elif num_pt==3: # face triangle
        scales=[0.026, 0.025, 0.025]
    else:
        scales=[0.025]*num_pt

    scales=[x*2 for x in scales] # scale=2*sigma

    ss=s*0.53 # approximation of shape area from box area
    num=0
    denom=0
    for i in range(num_pt):
        if kp_gt[i*3+2]>0.3: # is point labelled
            dx=kp_gt[i*3+0]-kp_det[i*3+0]
            dy=kp_gt[i*3+1]-kp_det[i*3+1]
            num+=math.exp(-(dx*dx+dy*dy)/(2.0*ss*scales[i]*scales[i]+1e-7))
            denom+=1.0
    iou=num/(denom+1e-7)

    return iou

def kp_iou2(kp_gt, kp_det, s, num_pt):
    """
    Computes the keypoint iou between two boxes

    Args:
        kpgt, gpdet: list [x,y,conf]*npoint
        s: GT box area
    Returns:
        float iou 0-1
    """
    # https://learnopencv.com/object-keypoint-similarity/
    if num_pt==5:
        scales=[0.025, 0.025, 0.026, 0.025, 0.025]
    elif num_pt==17:
        scales=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    elif num_pt==22:
        scales=[0.025, 0.025, 0.026, 0.025, 0.025, 0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    elif num_pt==3: # face triangle
        scales=[0.026, 0.025, 0.025]
    else:
        scales=[0.025]*num_pt

    scales=[x*2 for x in scales] # scale=2*sigma

    ss=s*0.53 # approximation of shape area from box area
    num=0
    denom=0
    for i in range(num_pt):
        if kp_gt[i*3+2]>0.01 and kp_det[i*3+2]>0.01: # BOTH points labelled
            dx=kp_gt[i*3+0]-kp_det[i*3+0]
            dy=kp_gt[i*3+1]-kp_det[i*3+1]
            num+=math.exp(-(dx*dx+dy*dy)/(2.0*ss*scales[i]*scales[i]+1e-7))
            denom+=1.0
    iou=num/(denom+1e-7)
    return iou

def match_boxes(dets, gts, iou_thr=0.5):
    """
    Matches detections to ground truths by class and IOU.

    Returns:
        det_matched: list of GT index for each detection (-1 if unmatched)
        gt_matched: list of DET index for each ground truth (-1 if unmatched)
    """
    num_dets, num_gts = len(dets), len(gts)
    det_matched = [-1] * num_dets
    gt_matched = [-1] * num_gts

    if num_dets == 0 or num_gts == 0:
        return det_matched, gt_matched

    for j, det in enumerate(dets):
        det_cls = det['class']
        det_box = det['box']

        best_iou = 0
        best_gt = -1

        for i, gt in enumerate(gts):
            if gt_matched[i] != -1:
                continue  # already matched
            if gt['class'] != det_cls:
                continue

            iou = stuff.box_iou(gt['box'], det_box)
            if iou > iou_thr and iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_gt != -1:
            det_matched[j] = best_gt
            gt_matched[best_gt] = j

    return det_matched, gt_matched

def match_boxes_kp(dets, gts, iou_thr=0.5):
    """
    Match detections to GTs using keypoint IOU
    """
    gt_matched=[-1]*len(gts)
    det_matched=[-1]*len(dets)
    for j,_ in enumerate(dets):
        for i,_ in enumerate(gts):
            if gt_matched[i]==-1 and gts[i]['class']==dets[j]['class']:
                box_iou=stuff.box_iou(dets[j]["box"],gts[i]["box"])
                if box_iou==0:
                    continue
                a=stuff.box_a(gts[i]['box'])
                if 'face_points' in gts[i]:
                    if 'face_points' in dets[j] and kp_iou(gts[i]['face_points'], dets[j]['face_points'], a, 5)>iou_thr:
                        gt_matched[i]=j
                        det_matched[j]=i
                        break
                if 'pose_points' in gts[i]:
                    if 'pose_points' in dets[j] and kp_iou(gts[i]['pose_points'], dets[j]['pose_points'], a, 17)>iou_thr:
                        gt_matched[i]=j
                        det_matched[j]=i
                        break
    return det_matched, gt_matched

def match_boxes_pose(dets, gts, iou_thr=0.5):
    """
    Match detections to GTs using keypoint IOU
    """
    gt_matched=[-1]*len(gts)
    det_matched=[-1]*len(dets)
    for j,_ in enumerate(dets):
        for i,_ in enumerate(gts):
            assert gts[i]['class']==dets[j]['class']
            if gt_matched[i]==-1:
                if stuff.box_iou(dets[j]["box"],gts[i]["box"])==0:
                    continue
                assert 'pose_points' in gts[i]
                assert 'pose_points' in dets[j]
                if kp_iou(gts[i]['pose_points'], dets[j]['pose_points'], stuff.box_a(gts[i]['box']), 17)>iou_thr:
                    gt_matched[i]=j
                    det_matched[j]=i
                    break
    return det_matched, gt_matched

def match_boxes_face(dets, gts, iou_thr=0.5):
    """
    Match detections to GTs using keypoint IOU
    """
    gt_matched=[-1]*len(gts)
    det_matched=[-1]*len(dets)
    for j,_ in enumerate(dets):
        for i,_ in enumerate(gts):
            assert gts[i]['class']==dets[j]['class']
            if gt_matched[i]==-1:
                if stuff.box_iou(dets[j]["box"],gts[i]["box"])==0:
                    continue
                assert 'face_points' in gts[i]
                assert 'face_points' in dets[j]
                if kp_iou(gts[i]['face_points'], dets[j]['face_points'], stuff.box_a(gts[i]['box']), 5)>iou_thr:
                    gt_matched[i]=j
                    det_matched[j]=i
                    break
    return det_matched, gt_matched

def match_boxes_facepose(dets, gts, iou_thr=0.5):
    """
    Match detections to GTs using keypoint IOU
    """
    gt_matched=[-1]*len(gts)
    det_matched=[-1]*len(dets)

    for j,_ in enumerate(dets):
        assert 'facepose_points' in dets[j]
        assert 'face_points' in dets[j]
        for i,_ in enumerate(gts):
            assert gts[i]['class']==dets[j]['class']
            if gt_matched[i]==-1:
                if stuff.box_iou(dets[j]["box"],gts[i]["box"])==0:
                    continue
                if kp_iou(gts[i]['face_points'], dets[j]['face_points'], stuff.box_a(gts[i]['box']), 5)>iou_thr:
                    gt_matched[i]=j
                    det_matched[j]=i
                    break
    return det_matched, gt_matched

def default_match(det, gt, context):
    if det["class"]!=gt["class"]:
        return 0
    return stuff.box_iou(det["box"], gt["box"])

def match_boxes_greedy(dets, gts, mfn=default_match, mfn_context=None):
    n_gts=len(gts)
    n_dets=len(dets)

    if n_gts==0 or n_dets==0:
        return [], []

    gt_matched=[False]*n_gts
    out_det_index=[]
    out_gt_index=[]
    for i,det in enumerate(dets):
        best_v=0
        best_match=None
        for j,gt in enumerate(gts):
            v=mfn(det,gt,mfn_context)
            if gt_matched[j]==False and v>best_v:
                best_v=v
                best_match=j
        if best_match is not None:
            gt_matched[best_match]=True
            out_det_index.append(i)
            out_gt_index.append(best_match)
    return out_det_index, out_gt_index

def match_boxes_lsa(dets, gts, mfn=default_match, mfn_context=None):
    n_gts=len(gts)
    n_dets=len(dets)

    if n_gts==0 or n_dets==0:
        return [], []

    costs = [[0 for x in range(n_dets)] for y in range(n_gts)]
    for ii in range(n_gts):
        for jj in range(n_dets):
            costs[ii][jj]=mfn(dets[jj], gts[ii], mfn_context)

    costs_np = np.array(costs)
    gt_ind, det_ind = linear_sum_assignment(costs_np, maximize=True)

    # Filter matches by positive cost
    matched = [(d, g) for d, g in zip(det_ind, gt_ind) if costs_np[g, d] > 0]

    if not matched:
        return [], []

    det_filtered, gt_filtered = zip(*matched)
    return list(det_filtered), list(gt_filtered)

def append_comments(fn, x):
    x="#"+x
    # Ensure multi-line comments stay commented in YAML.
    x = x.replace('\n', '\n#')
    if not os.path.isfile(fn):
        print(f"Error: file {fn} does not exist")
        return
    with open(fn, "a") as f:
        f.write(x+"\n")

def dataset_copy_comments(src, dest):
    with open(src, 'r') as file:
        for line in file:
            if line.startswith("#"):
                append_comments(dest, line[1:].strip())

def attribute_text(a, attributes):
    text=""
    if "attrs" in a:
        l=list(range(len(a["attrs"])))
        l=[x for _, x in sorted(zip(a["attrs"], l), reverse=True)]
        for i in l:
            attr=a["attrs"][i]
            if attr>0.1:
                name=attributes[i]
                if name.startswith("person:"):
                    name=name[len("person:"):]
                text+=f"{name}={attr:0.2f}\n"
    return text

def add_pose_points_to_gts(dets, gts, iou_thr=0.6, min_sz=0.5*0.5, class_names=None):
    """
    Given existing GTs and detections from a pose point detector, add the pose values into the GTs
    does this by matching the box to find a good box in the GTs and copying the pose points to that box
    -i.e. assumption is the GTs already have full and complete labelling for the person boxes
    """
    person_class=class_names.index("person")
    added=0

    if True:
        det_ind, gt_ind=match_boxes_lsa(dets, gts)

        for i,_ in enumerate(gt_ind):
            gt=gts[gt_ind[i]]
            det=dets[det_ind[i]]
            iou=stuff.box_iou(gt["box"],det["box"])
            if det["class"]==person_class and gt["class"]==person_class and iou>iou_thr:
                if stuff.has_pose_points(gt) is False:
                    if "pose_points" in det:
                        gt["pose_points"]=det["pose_points"]
                    if "facepose_points" in det:
                        gt["facepose_points"]=det["facepose_points"]
                    added+=1
    return added

def face_person_kp_iou(f, p):
    face_tri_iou=0
    if stuff.has_face_points(f) and (stuff.has_pose_points(p) or stuff.has_facepose_points(p)):
        face_tri_face=stuff.get_face_triangle_points(f)
        face_tri_person=stuff.get_face_triangle_points(p)
        face_tri_iou=kp_iou2(face_tri_face, face_tri_person, stuff.box_a(f["box"])*2, 3)
    return face_tri_iou

def face_match_function(f, p, context):
    if p["class"]!=context["person_class"]:
        return 0
    if f["class"]!=context["face_class"]:
        return 0

    intersection_area=stuff.box_i(f["box"], p["box"])
    if intersection_area==0:
        return 0

    face_tri_iou=0
    if stuff.has_face_points(f) and stuff.has_pose_points(p):
        face_tri_iou=face_person_kp_iou(f, p)

    # face must be not too small relative to body width
    if stuff.box_w(f["box"])<0.10*stuff.box_w(p["box"]):
        return face_tri_iou
    # top of face has to be in top half of body
    if f["box"][1]>(0.5*p["box"][1]+0.5*p["box"][3]):
        return face_tri_iou
    # face must not be too small relative to body
    if intersection_area/(stuff.box_a(p["box"])+1e-10)<0.005:
        return face_tri_iou
    ioa=intersection_area/(stuff.box_a(f["box"])+1e-10)
    return ioa+f["confidence"]/100.0+p["confidence"]/10.0+face_tri_iou

def filter_faces_by_persons(faces, persons, ioa_thr=0.9, class_names=None):
    assert class_names is not None
    if not "person" in class_names:
        return faces # can't do any filtering

    assert "face" in class_names
    face_class=class_names.index("face")
    person_class=class_names.index("person")
    context={"person_class":person_class, "face_class":face_class}
    f_ind, p_ind=match_boxes_lsa(faces, persons, mfn=face_match_function, mfn_context=context)

    ret=[]
    ret_p_ind=[]
    ret_f_ind=[]
    for i,_ in enumerate(f_ind):
        p=persons[p_ind[i]]
        f=faces[f_ind[i]]
        ioa=stuff.box_i(f["box"], p["box"])/(stuff.box_a(f["box"])+1e-10)
        if ioa>ioa_thr:
            ret_p_ind.append(p_ind[i])
            ret_f_ind.append(f_ind[i])
            ret.append(f)
    return ret, ret_f_ind, ret_p_ind

def filter_persons_by_faces(dets, ioa_thr=0.9, add_anyway_conf=0.5, class_names=None):
    assert class_names is not None
    assert "person" in class_names
    assert "face" in class_names

    face_class=class_names.index("face")
    person_class=class_names.index("person")

    context={"person_class":person_class, "face_class":face_class}

    persons=[]
    faces=[]
    other=[]
    for d in dets:
        if d["class"]==person_class:
            persons.append(d)
        elif d["class"]==face_class:
            faces.append(d)
        else:
            other.append(d)

    f_ind, p_ind=match_boxes_lsa(faces, persons, mfn=face_match_function, mfn_context=context)

    for i,_ in enumerate(f_ind):
        p=persons[p_ind[i]]
        f=faces[f_ind[i]]
        ioa=stuff.box_i(f["box"], p["box"])/(stuff.box_a(f["box"])+1e-10)
        if ioa>ioa_thr:
            p["face_filter_ok"]=True

    persons_out=[]
    for p in persons:
        if "face_filter_ok" in p or p["confidence"]>add_anyway_conf:
            include=True
            for p2 in persons_out:
                if stuff.box_iou(p["box"],p2["box"])>0.5:
                    include=False
            if include:
                persons_out.append(p)

    return persons_out+faces+other

def sstr(x):
    if x==int(x):
        return str(int(x))+" "
    return "{:4.3f}  ".format(x)

def write_annotations(an,
                      include_face=True,
                      include_pose=False,
                      include_facepose=False,
                      include_attrs=True):
    an_txt=""
    for a in an:
        b=a["box"]
        cx=(b[2]+b[0])*0.5
        cy=(b[3]+b[1])*0.5
        w=b[2]-b[0]
        h=b[3]-b[1]
        an_txt+=str(a["class"])+" {0:.4f} {1:.4f} {2:.4f} {3:.4f} ".format(cx,cy,w,h)

        if include_facepose:
            assert(include_pose==False)
            assert(include_face==False)
            if "facepose_points" in a:
                fpp=a["facepose_points"]
            else:
                fpp=[0]*19*3
            for i in range(19*3):
                an_txt+=sstr(stuff.clip01(fpp[i]))
        if include_face:
            if "face_points" in a:
                fp=a["face_points"]
            else:
                fp=[0]*15
            for i in range(5*3):
                #if fp[i]<0 or fp[i]>1.0:
                #    print(f"Warning: face point {i} out of range {fp[i]}")
                an_txt+=sstr(stuff.clip01(fp[i]))
        if include_pose:
            if "pose_points" in a:
                pp=a["pose_points"]
            else:
                pp=[0]*17*3
            for i in range(17*3):
                if pp[i]<0 or pp[i]>1.0:
                    print(f"Warning: pose point {i} out of range {pp[i]}")
                an_txt+=sstr(stuff.clip01(pp[i]))
        if include_attrs:
            if "attrs" in a and a["attrs"] is not None:
                nonzero=0
                for x in a["attrs"]:
                    nonzero+=(x!=0)
                if nonzero!=0:
                    an_txt+="A "
                    for i in a["attrs"]:
                        an_txt+=sstr(stuff.clip01(i))
        an_txt+="\n"
    return an_txt


def write_annotations_v10(an,
                          include_face=True,
                          include_pose=False,
                          include_facepose=False,
                          attr_nc=ATTR_NC_V10):
    """
    Write annotations in v10 (split) format for Ultralytics attributes branch:
    cls, bbox (4), attr_0..attr_{attr_nc-1}, then keypoints. One row per object.
    """
    an_txt = ""
    for a in an:
        b = a["box"]
        cx = (b[2] + b[0]) * 0.5
        cy = (b[3] + b[1]) * 0.5
        w = b[2] - b[0]
        h = b[3] - b[1]
        an_txt += str(a["class"]) + " {0:.4f} {1:.4f} {2:.4f} {3:.4f} ".format(cx, cy, w, h)
        # attributes: fixed length attr_nc, between bbox and keypoints
        attrs = [0.0] * attr_nc
        if "attrs" in a and a["attrs"] is not None:
            for i, v in enumerate(a["attrs"]):
                if i < attr_nc:
                    attrs[i] = stuff.clip01(v)
        for x in attrs:
            an_txt += sstr(x)
        # keypoints: same order as write_annotations (face then pose when both)
        if include_facepose:
            assert not include_pose and not include_face
            fpp = a.get("facepose_points", [0] * (19 * 3))
            for i in range(19 * 3):
                an_txt += sstr(stuff.clip01(fpp[i] if i < len(fpp) else 0))
        if include_face:
            fp = a.get("face_points", [0] * 15)
            for i in range(5 * 3):
                an_txt += sstr(stuff.clip01(fp[i] if i < len(fp) else 0))
        if include_pose:
            pp = a.get("pose_points", [0] * 51)
            for i in range(17 * 3):
                an_txt += sstr(stuff.clip01(pp[i] if i < len(pp) else 0))
        an_txt += "\n"
    return an_txt


def get_dataset_path(ds_yaml):
    return os.path.dirname(ds_yaml)

def get_dataset_name(ds_yaml):
    ds=stuff.load_dictionary(ds_yaml)
    if ds!=None:
        if "dataset_name" in ds:
            return ds["dataset_name"]
    return name_from_file(ds_yaml)

def get_dataset_config_name(ds_yaml):
    ds=stuff.load_dictionary(ds_yaml)
    if ds is not None:
        if "config_name" in ds:
            return ds["config_name"]
        if "dataset_name" in ds:
            config_name=ds["dataset_name"]
        else:
            config_name=name_from_file(ds_yaml)
    else:
        # If the YAML can't be loaded, fall back to a name derived from the filename.
        config_name=name_from_file(ds_yaml)
    if "-" in config_name:
        config_name=config_name[0:config_name.find("-")]
    return config_name

def get_loader(name):
    loaders={"CocoLoader":CocoLoader,
             "OpenImagesLoader": OpenImagesLoader,
             "O365Loader": O365Loader,
             "WiderfaceLoader":WiderfaceLoader,
             "RoboflowLoader":RoboflowLoader,
             "WeaponDetectionLoader":WeaponDetectionLoader,
             "WiderAttributesLoader":WiderAttributesLoader,
             "FiftyOneLoader":FiftyOneLoader,
             "TracksetLoader":TracksetLoader,
             "WiderpersonLoader": WiderPersonLoader,
             "CrowdPoseLoader": CrowdPoseLoader,
             "RawimageLoader": RawimageLoader}

    if not name in loaders:
        print("Could not find loader "+name)
        exit()
    return loaders[name]

def loader_name(loader):
    name=loader.lower()
    if "loader" in name:
        name=name[0:name.find("loader")]
    return name

def image_get_exif_data(image_file):
    comment=stuff.image_get_exif_comment(image_file)
    if comment is None:
        return None
    comment_lines=comment.split(";")
    ret={}
    items=["origin","config_name"]
    for c in comment_lines:
        for i in items:
            if i+"=" in c:
                ret[i]=c.split("=")[1]
    return ret

def box_to_coords(box, w, h):
    x0=int(box[0]*w)
    x1=int(box[2]*w)
    y0=int(box[1]*h)
    y1=int(box[3]*h)
    return x0,y0,x1,y1

def mask_detections(img, gts, dets, thr=0.5):
    h, w, c = img.shape
    orig = img.copy()
    for d in dets:
        if d["confidence"]>thr:
            x0,y0,x1,y1=box_to_coords(d["box"],w,h)
            img[y0:y1, x0:x1]=(80,80,80)
    for g in gts:
        x0,y0,x1,y1=box_to_coords(g["box"],w,h)
        img[y0:y1, x0:x1]=orig[y0:y1, x0:x1]

def facepose_iou(pose_points, pose_box, face_points, face_box):
    test_iou=stuff.box_iou(pose_box, face_box)
    if test_iou==0:
        return 0

    pose_xmin=1
    pose_xmax=0
    pose_ymin=1
    pose_ymax=0
    np=0
    for i in [0,1,2,5,6]:
        if pose_points[3*i+2]==0:
            continue
        pose_xmin=min(pose_points[3*i+0], pose_xmin)
        pose_xmax=max(pose_points[3*i+0], pose_xmax)
        pose_ymin=min(pose_points[3*i+1], pose_ymin)
        pose_ymax=max(pose_points[3*i+1], pose_ymax)
        np+=1
        if i==2 and np==3:
            break

    face_xmin=1
    face_xmax=0
    face_ymin=1
    face_ymax=0

    np=0
    for i in [0,1,2,3,4]:
        if face_points[3*i+2]==0:
            continue
        face_xmin=min(face_points[3*i+0], face_xmin)
        face_xmax=max(face_points[3*i+0], face_xmax)
        face_ymin=min(face_points[3*i+1], face_ymin)
        face_ymax=max(face_points[3*i+1], face_ymax)
        np+=1
        if i==2 and np==3:
            break

    face_point_box=[face_xmin, face_ymin, face_xmax, face_ymax]
    iou=stuff.box_iou([pose_xmin, pose_ymin, pose_xmax, pose_ymax], face_point_box)
    if iou>0:
        return iou
    ret=stuff.box_iou(pose_box, face_point_box)
    return ret

def match_facepose(gts, person_class, face_class, map_unmatched_faces=False):
    person_indices=[]
    face_indices=[]
    for ii,g in enumerate(gts):
        if g["class"]==person_class:
            person_indices.append(ii)
        elif g["class"]==face_class:
            face_indices.append(ii)

    n_p=len(person_indices)
    n_f=len(face_indices)
    if n_p!=0 and n_f!=0:
        costs = [[0 for x in range(n_f)] for y in range(n_p)]
        for ii in range(n_p):
            for jj in range(n_f):
                costs[ii][jj]=facepose_iou(gts[person_indices[ii]]["pose_points"],
                                           gts[person_indices[ii]]["box"],
                                           gts[face_indices[jj]]["face_points"],
                                           gts[face_indices[jj]]["box"])
        row_ind, col_ind = linear_sum_assignment(np.array(costs), maximize=True)

        for k,_ in enumerate(row_ind):
            p_gt=gts[person_indices[row_ind[k]]]
            f_gt=gts[face_indices[col_ind[k]]]
            iou=stuff.box_iou(p_gt["box"],f_gt["box"])
            if iou==0:
                continue
            p_gt["face_points"]=copy.copy(f_gt["face_points"])
            stuff.map_one_gt_keypoints(p_gt, False, False, True)
            del f_gt["face_points"]
            del f_gt["pose_points"]

    for gt in gts:
        if gt["class"]==person_class:
            if not "facepose_points" in gt:
                stuff.map_one_gt_keypoints(gt, False, False, True)
            assert "facepose_points" in gt
            assert not "pose_points" in gt
        if map_unmatched_faces and gt["class"]==face_class:
            if "face_points" in gt or "pose_points" in gt:
                gt["class"]=person_class
                stuff.map_one_gt_keypoints(gt, False, False, True)
                assert not "pose_points" in gt
    for gt in gts:
        if "pose_points" in gt:
            del gt["pose_points"]

def print_gts(gts, classes=None):
    for g in gts:
        if g==None:
            continue
        if classes==None:
            s="{:10s}".format("class "+str(g["class"]))
        else:
            s="{:10s}".format(classes[g["class"]])
        s+=" C={:3.2f}".format(g["confidence"])
        s+=" B={:3.2f},{:3.2f}->{:3.2f},{:3.2f}".format(g["box"][0], g["box"][1], g["box"][2], g["box"][3])
        print(s)

def object_img(img, gt, scale_width=None, scale_height=None, expand=1.0, no_enlarge=False):
    height, width, _ = img.shape
    x0=width*gt["box"][0]
    x1=width*gt["box"][2]
    y0=height*gt["box"][1]
    y1=height*gt["box"][3]
    if expand!=1.0:
        w=x1-x0
        h=y1-y0
        w=0.5*w*(expand-1.0)
        h=0.5*h*(expand-1.0)
        x0-=w
        x1+=w
        y0-=h
        y1+=h
    x0=int(x0)
    x1=int(x1+0.99)
    y0=int(y0)
    y1=int(y1+0.99)
    x0=max(0,x0)
    y0=max(0,y0)
    x1=min(width,x1)
    y1=min(height,y1)
    person_img=img[y0:y1, x0:x1]

    if scale_width is not None or scale_height is not None:
        f=10
        if scale_width is not None:
            f=min(f, scale_width/(x1-x0))
        if scale_height is not None:
            f=min(f, scale_height/(y1-y0))
        if no_enlarge:
            f=min(f, 1.0)
        person_img=cv2.resize(person_img, None, fx=f, fy=f, interpolation = cv2.INTER_CUBIC)
    return person_img

def object_jpeg(img, gt, scale_width=None, scale_height=None, expand=1.0, no_enlarge=False):
    img=object_img(img, gt, scale_width, scale_height, expand, no_enlarge)
    success, encoded=cv2.imencode(".jpg", img)
    assert success
    return encoded

def base64_encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_param(task, config, param, default_value):
    if config is None:
        return default_value
    if param+"_val" in config and task=="val":
        return config[param+"_val"]
    if param+"_train" in config and task=="train":
        return config[param+"_train"]
    if param in config:
        return config[param]
    return default_value

def attributes_from_class_names(class_names):
    """
    Detect class names that correspond to attributes of a base
    class. Returns a list of attribtues
    e.g. person_male class -> person:male attribute
    """
    attributes=[]
    for c in class_names:
        if c.startswith("person_"):
            attributes.append(c.replace("person_", "person:"))
    for c in class_names:
        if c.startswith("face_"):
            attributes.append(c.replace("face_", "face:"))
    return attributes

def timestr(delta):
    delta=int(delta)
    s=delta%60
    m=(delta//60)%60
    h=delta//3600
    r=f"{m:02d}:{s:02d}"
    if h!=0:
        r=f"{h:02d}:"+r
    return r


def render_ignore_boxes(jpeg_path, ignore_boxes, unignore_boxes):
    """
    Draws grey rectangles over 'ignore_boxes', then restores areas in 'unignore_boxes'
    from the original image.

    Args:
        jpeg_path (str): Path to the JPEG file.
        ignore_boxes (List[Tuple[float, float, float, float]]): Boxes to grey out.
        unignore_boxes (List[Tuple[float, float, float, float]]): Boxes to restore.
    """
    # Load original image
    original = Image.open(jpeg_path).convert("RGB")
    img = original.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # First pass: draw grey rectangles
    for box in ignore_boxes:
        x0, y0, x1, y1 = box
        px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        draw.rectangle([px0, py0, px1, py1], fill=(128, 128, 128))

    # Second pass: restore unignored areas
    for box in unignore_boxes:
        x0, y0, x1, y1 = box
        px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        region = original.crop((px0, py0, px1, py1))
        img.paste(region, (px0, py0))

    # Save the modified image
    img.save(jpeg_path, format="JPEG")