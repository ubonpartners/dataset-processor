from tqdm import tqdm
import stuff
import src.dataset_util as dsu
from src.llm_attributes import generate_person_attributes

def build_task_add_objects(processor, add_object_config):
    """
    Check dataset with a high quality object detector. If sufficiently confident detections are
    found that are not labelled then we add those objects into the GT dataset, i.e. we are
    assuming it is the object detector that is correct.
    """
    if not "model" in add_object_config:
        return
    model=add_object_config["model"]
    imgsz=add_object_config["sz"]
    thr=dsu.get_param(processor.task, add_object_config, "thr", 0.9)
    iou_thr=dsu.get_param(processor.task, add_object_config, "iou_thr", 0.5)
    min_sz=dsu.get_param(processor.task, add_object_config, "min_sz", 0.05*0.05)
    filter_persons_by_face=dsu.get_param(processor.task, add_object_config, "filter_persons_by_face", False)
    per_class_thr=dsu.get_param(processor.task, add_object_config, "per_class_thr",
                                [thr]*len(processor.class_names))

    processor.set_object_detector(model, imgsz=imgsz, thr=min(per_class_thr), half=True, rect=False, batch_size=16)
    added=[0]*len(processor.class_names)
    filtered=0
    desc=processor.config_name+"/"+processor.task+"/add objects"
    colour="#ff1010" if imgsz>640 else "#d04040"
    for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour=colour, smoothing=0.01):
        gts=processor.get_gt(i)
        if gts is None:
            gts=[]
        detections=processor.get_detections(i)
        missing_detections=[]
        for d in detections:
            best_iou, _, _=dsu.best_iou_match(d, gts)
            thr_use=per_class_thr[d["class"]]
            if best_iou<iou_thr and d["confidence"]>thr_use and stuff.box_a(d["box"])>min_sz:
                missing_detections.append(d)
                added[d["class"]]+=1
        dets=gts+missing_detections
        if filter_persons_by_face and "face" in processor.get_class_names():
            before=len(dets)
            dets=dsu.filter_persons_by_faces(dets, class_names=processor.get_class_names())
            filtered+=(before-len(dets))
        processor.replace_annotations(i, dets)
    processor.log(f" add_objects: det={model} sz={imgsz} thr={per_class_thr}: {added} detections added {filtered} filtered")
    processor.log_stats()

def build_task_mask_objects(processor, add_object_config):
    """
    Check dataset with an object detector. If confident detections are found that are not labelled
    then we 'mask out' that area in the image.
    """
    model=add_object_config["model"]
    imgsz=add_object_config["sz"]
    thr=dsu.get_param(processor.task, add_object_config, "thr", 0.5)
    iou_thr=dsu.get_param(processor.task, add_object_config, "iou_thr", 0.5)
    per_class_thr=dsu.get_param(processor.task, add_object_config, "per_class_thr", [thr]*len(processor.class_names))
    processor.set_object_detector(model, imgsz=imgsz, thr=min(per_class_thr), half=True, rect=False, batch_size=16)
    masked=[0]*len(processor.class_names)
    desc=processor.config_name+"/"+processor.task+"/mask objects"
    colour="#ff1010" if imgsz>640 else "#d04040"
    for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour=colour, smoothing=0.01):
        gts=processor.get_gt(i)
        if gts is None:
            continue
        detections=processor.get_detections(i)
        missing_detections=[]
        for d in detections:
            best_iou, _, _=dsu.best_iou_match(d, gts)
            thr_use=per_class_thr[d["class"]]
            if best_iou<iou_thr and d["confidence"]>thr_use:
                missing_detections.append(d)
                masked[d["class"]]+=1

        if len(missing_detections)!=0:
            img=processor.get_img(i)
            dsu.mask_detections(img, gts, missing_detections, thr=0)
            processor.replace_img(i, img)

    processor.log(f" mask_objects: det={model} sz={imgsz} thr={per_class_thr}: {masked} objects masked")

def build_task_add_pose(processor, add_object_config):
    """
    Use a pose point detector to add missing pose points to the GTs
    """
    model=add_object_config["model"]
    imgsz=add_object_config["sz"]
    thr=dsu.get_param(processor.task, add_object_config, "thr", 0.2)
    min_sz=dsu.get_param(processor.task, add_object_config, "min_sz", 0.05*0.05)

    processor.set_object_detector(model, imgsz=imgsz, batch_size=16, thr=thr)
    added=0
    desc=processor.config_name+"/"+processor.task+"/add pose"
    colour="#ff8080" if imgsz>640 else "#d0a0a0"
    for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour=colour, smoothing=0.01):
        gts=processor.get_gt(i)
        if gts is None:
            continue
        detections=processor.get_detections(i)
        added+=dsu.add_pose_points_to_gts(detections, gts, min_sz=min_sz, class_names=processor.get_class_names())
        processor.replace_annotations(i, gts)
    processor.log(desc+f" added {added} pose point sets with det {model} sz {imgsz}")
    processor.log_stats()

def build_task_add_faces(processor, add_object_config):
    """
    Use a face box / face point detector to add missing face boxes
    and/or face points to the GTs
    """

    assert "face" in processor.class_names

    model=add_object_config["model"]
    imgsz=add_object_config["sz"]
    kp_thr=dsu.get_param(processor.task, add_object_config, "kp_thr", 0.9)
    box_thr=dsu.get_param(processor.task, add_object_config, "box_thr", 0.9)

    processor.set_object_detector(model, imgsz=imgsz, thr=min(kp_thr, box_thr), batch_size=16)

    faces_added=0
    face_kp_added=0
    desc=processor.config_name+"/"+processor.task+f"/add faces bthr:{box_thr} kpthr:{kp_thr}"
    colour="#8080ff" if imgsz>640 else "#a0a0d0"
    for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour=colour, smoothing=0.01):
        gts=processor.get_gt(i)
        if gts is None:
            continue

        detections=processor.get_detections(i)
        faces, face_idx, person_idx=dsu.filter_faces_by_persons(detections, gts, class_names=processor.get_class_names())

        for f in faces:
            # check for existing face
            iou, gt, _=dsu.best_iou_match(f, gts)
            if iou<0.5:
                # new face
                if f["confidence"]>box_thr:
                    gts.append(f)
                    faces_added+=1
            else:
                # existing face; consider adding the face points from detection into that face
                # if it doesn't already have face points
                if not stuff.has_face_points(gt) and f["confidence"]>kp_thr:
                    gt["face_points"]=f["face_points"]
                    face_kp_added+=1

        processor.replace_annotations(i, gts)

    processor.log(f" add_faces det={model} sz={imgsz} thrs={box_thr},{kp_thr}: {faces_added} faces added; {face_kp_added} face keypoint sets added")
    processor.log_stats()

def build_task_normalise(processor):
    """
    Normalise does some random cleanup
    - make sure all the co-ordinates are in range
    - make sure there aren't multiple overlapping instances of the same class
    """
    desc=processor.config_name+"/"+processor.task+"/normalise"
    for idx in tqdm(range(processor.num_files), desc=desc.ljust(45), smoothing=0.01):
        gts=processor.get_gt(idx)
        if gts is None:
            continue
        for g in gts:
            if "confidence" in g:
                g["confidence"]=stuff.clip01(g["confidence"])
            if "box" in g:
                for i in range(4):
                    g["box"][i]=stuff.clip01(g["box"][i])
            if "face_points" in g:
                for i in range(15):
                    g["face_points"][i]=stuff.clip01(g["face_points"][i])
            if "pose_points" in g:
                for i in range(51):
                    g["pose_points"][i]=stuff.clip01(g["pose_points"][i])
        # dedup does a kind of NMS style removal of multiple overlapping instances of the same class
        gts=dsu.dedup_gt(gts, iou_thr=0.5)
        processor.replace_annotations(idx, gts)

def build_task_add_attributes(processor, config):
    """
    Extract image boxes from GT image and pass to a vision-LLM with a suitable prompt to
    determing 'attributes' of the detections
    """
    processor.log(f" Add attributes: loader {config['loader']} LLM {config['llm_model']}")
    attributes=processor.attributes
    if attributes is None:
        return

    generate_person_attributes(processor, config)
    processor.log_stats()

def build_task_add_fiqa(processor, config):
    """
    Use FIQA model to add face quality attributes to person detections
    """

    desc=processor.config_name+"/"+processor.task+f"/add FIQA"
    colour="#8080ff"
    processor.set_object_detector("upyc_track", batch_size=32)
    attributes=processor.attributes
    fiqa_attr=[]

    match_face=False
    match_person=False
    for i,a in enumerate(attributes):
        if a.startswith("face:fiqa_"):
            thr=float(a.split("face:fiqa_")[1])
            fiqa_attr.append({"index":i, "thr":thr})
            match_face=True
        if a.startswith("person:fiqa_"):
            thr=float(a.split("person:fiqa_")[1])
            fiqa_attr.append({"index":i, "thr":thr})
            match_person=True

    #for a in fiqa_attr:
    #    assert a is not None, "could not find face::xxx_quality attribute"
    num=[0]*len(fiqa_attr)
    for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour=colour, smoothing=0.01):
        gts=processor.get_gt(i)
        if gts is None:
            continue
        for gt in gts:
            if not "attrs" in gt:
                gt["attrs"]=[0]*len(attributes)

        dets=processor.get_detections(i)
        fiqa_dets=[]
        person_class=processor.get_class_index("person")
        face_class=processor.get_class_index("face")
        for d in dets:
            if "fiqa_score" in d:
                if match_face:
                    fiqa_dets.append({"box":d["subbox"], "confidence":1, "class":face_class, "fiqa_score":d["fiqa_score"]})
                if match_person:
                    fiqa_dets.append({"box":d["box"], "confidence":1, "class":person_class, "fiqa_score":d["fiqa_score"]})

        det_ind, gt_ind=dsu.match_boxes_lsa(fiqa_dets, gts)
        for k in range(len(det_ind)):
            det=fiqa_dets[det_ind[k]]
            gt=gts[gt_ind[k]]
            score=det["fiqa_score"]
            for j,v in enumerate(fiqa_attr):
                if score>v["thr"]:
                    gt["attrs"][v["index"]]=1
                    num[j]+=1

        processor.replace_annotations(i, gts)

    processor.log(f"FIQA added {num} annotations")
    processor.log_stats()
