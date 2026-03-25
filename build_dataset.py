from src.dataset_processor import DatasetProcessor
from src.dataset_processor import dataset_merge_and_delete, dataset_delete
from src.llm_attributes import generate_person_attributes, llm_test
import src.dataset_util as dsu
import os
import argparse
import math
import time
import copy
import random
import pickle
from tqdm import tqdm
import stuff

def build_task_import(processor, import_config, test=False):
    """
    import
    - use one of the import functions to load the initial images and GTs from a dataset
    - the classes/annotation are remapped to our target set
    - the files are renamed
    """
    task=processor.task
    class_names=processor.class_names
    name=processor.dataset_name

    processor.log(f"import; class_names={class_names}")

    loader=import_config["loader"]

    max_width=dsu.get_param(processor.task, import_config, "max_width", 1920)
    max_height=dsu.get_param(processor.task, import_config, "max_height", 1920)
    max_bpp=dsu.get_param(processor.task, import_config, "max_bpp", 3.0)
    yuv420_frac=dsu.get_param(processor.task, import_config, "yuv420", 0.0)
    fix_orientation=dsu.get_param(processor.task, import_config, "fix_orientation", False)
    yuv420=random.random()<yuv420_frac

    max_images=dsu.get_param(processor.task, import_config, "max_images", 10000000)
    add_exif=dsu.get_param(processor.task, import_config, "add_exif", True)
    filter_single_large=dsu.get_param(processor.task, import_config, "filter_single_large", False)
    enable_ignore_regions=dsu.get_param(processor.task, import_config, "enable_ignore_regions", True)

    if enable_ignore_regions:
        class_name_ignore=class_names+["ignore"]
        ignore_class=class_name_ignore.index("ignore")
    else:
        class_name_ignore=-1
        ignore_class=None

    loader_fn=dsu.get_loader(loader)
    o=loader_fn(class_names=class_name_ignore, task=task, ds_params=import_config)

    processor.log("origin info: "+o.get_info())
    maps=o.get_category_maps()
    for c in class_name_ignore:
        if c in maps:
            o.add_category_mapping(maps[c], c)

    ids=o.get_image_ids()
    if len(ids) != len(set(ids)):
        print("WARNING: Duplicate ids")
    processor.log(f"found {len(ids)} ids, limiting to {max_images}")

    base=name+"_"+task[0]

    index=0
    no_annotations=0
    filtered_single_large=0

    max_index=min(len(ids), max_images)
    if test:
        max_index=min(max_index, 250)

    desc=name+"/"+task+"/Generate"
    add_failed=0
    with tqdm(total=max_index, desc=desc.ljust(45), colour="#008000", smoothing=0.01) as pbar:
        for k,i in enumerate(ids):
            pbar.update(max(index, (k*max_index)//len(ids))-pbar.n)
            img_path=o.get_img_path(i)
            if img_path is None or not os.path.isfile(img_path):
                continue
            dets=o.get_annotations(i)
            if dets is None:
                no_annotations+=1
                continue

            if filter_single_large is True:
                if len(dets)==1:
                    if stuff.box_a(dets[0]["box"])>0.20:
                        filtered_single_large+=1
                        continue

            n=base+f"{index:07d}"
            if False==processor.add(n, img_path, dets,
                                    add_exif=add_exif or test,
                                    ignore_class=ignore_class,
                                    max_width=max_width,
                                    max_height=max_height,
                                    max_bpp=max_bpp,
                                    yuv420=yuv420,
                                    fix_orientation=fix_orientation):
                add_failed+=1
            index+=1
            pbar.update(1)

            if index>=max_index:
                break
        pbar.update(max_index-pbar.n)

    processor.log(f"imported {index} images {add_failed} failed")
    if no_annotations!=0:
        processor.log(f"import_dataset: filtered {no_annotations} images due to 'None' annotations")
    if filtered_single_large!=0:
        processor.log(f"import_dataset: filtered {filtered_single_large} images due to 'filtered_single_large'")
    processor.reload_files()
    processor.log_stats()

    dedup=dsu.get_param(processor.task, import_config, "dedup", False)
    if dedup:
        build_task_dedup(processor)

def box_size_scale(det):
    """
    penalise larger false positives / missed detections more
    we do this by raising the box area (0<a<=1) to a small power
    and multiplying the score by that
    """
    return math.pow(float(stuff.box_a(det["box"])), float(0.4))

def build_task_make_hard(processor, hard_config, test=False):
    """
    make_hard
    - measure 'hardness' of GTs in dataset by seeing how well a reference object detector does
    - extract the hardest N to use
    """
    max_images=dsu.get_param(processor.task, hard_config, "max_images", 20000)
    add_exif=dsu.get_param(processor.task, hard_config, "add_exif", True)
    hardness_cache_folder=dsu.get_param(processor.task, hard_config, "hardness_cache_folder", "/mldata/hardness_cache")
    if test:
        max_images=min(max_images, 50)
    model=hard_config["model"]
    rare_classes=[]
    if "rare_classes" in hard_config:
        rare_classes=hard_config["rare_classes"]

    processor.log(f"make_hard: reduce {processor.num_files} images to {max_images}")
    hardness=[]
    num=processor.num_files
    if num<=max_images:
        processor.log("make_hard: nothing to do")
        return

    processor.set_object_detector(model, imgsz=480, thr=0.01, batch_size=48)
    class_names=processor.class_names
    nc=len(class_names)
    rare_class_map=[0.0]*nc
    for index,c in enumerate(class_names):
        if c in rare_classes:
            rare_class_map[index]=1.0

    desc=processor.config_name+"/"+processor.task+"/measure hardness"

    hardness_cache_folder=hardness_cache_folder+"/"+processor.config_name+"/"+processor.task
    hardness_dict={}
    stuff.makedir(hardness_cache_folder)
    hardness_cache_file=hardness_cache_folder+"/hardness.pkl"
    if os.path.isfile(hardness_cache_file):
        with open(hardness_cache_file, 'rb') as handle:
            hardness_dict = pickle.load(handle)
            processor.log(f"Loaded {len(hardness_dict)} hardness cache entries")
    out_hardness_dict_added=0
    out_hardness_dict={}
    out_hardness_dict.update(hardness_dict)

    classnames=processor.get_class_names()
    can_detect=[False]*len(classnames)
    for n in ["person","face","vehicle","animal","weapon"]:
        if n in classnames:
            can_detect[classnames.index(n)]=True

    for i in tqdm(range(num), desc=desc.ljust(45), colour="#10a010", smoothing=0.01):

        exif_data=processor.get_exif_data(i)
        origin=None
        if "origin" in exif_data:
            origin=dsu.name_from_file(exif_data["origin"])
            if origin in hardness_dict:
                hardness.append(hardness_dict[origin])
                continue

        gts=processor.get_gt(i)
        dets=processor.get_detections(i)
        det_matched, gt_matched=dsu.match_boxes(dets, gts, 0.5)
        hardness_fn=0.0
        hardness_fp=0.0
        hardness_rare_class=0.0
        for g in gts:
            hardness_rare_class+=(rare_class_map[g["class"]]*box_size_scale(g))
        num_gt=0
        for j,_ in enumerate(gt_matched):
            cls=gts[j]["class"]
            # Don't count a penalty for GTs we couldn't possibly match.
            if can_detect[cls] is False:
                assert gt_matched[j]==-1
                continue
            num_gt+=1
            fn=1.0
            if gt_matched[j]!=-1:
                fn-=dets[gt_matched[j]]["confidence"]
            fn*=box_size_scale(gts[j])
            hardness_fn+=fn
        min_fp_thr=0.3
        fp_scale=1.0/(1.0-min_fp_thr)
        for j,_ in enumerate(det_matched):
            if det_matched[j]==-1:
                if dets[j]["confidence"]>min_fp_thr:
                    hardness_fp+=((dets[j]["confidence"]-min_fp_thr)*fp_scale*box_size_scale(dets[j]))
        hardness_tot=hardness_fp+hardness_fn+4*hardness_rare_class
        hardness.append(hardness_tot)
        if origin is not None:
            out_hardness_dict[origin]=hardness_tot
            out_hardness_dict_added+=1
            if out_hardness_dict_added>2000:
                with open(hardness_cache_file+".tmp", 'wb') as handle:
                    pickle.dump(out_hardness_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    stuff.rename(hardness_cache_file+".tmp", hardness_cache_file)
                    out_hardness_dict_added=0

    with open(hardness_cache_file+".tmp", 'wb') as handle:
        pickle.dump(out_hardness_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        stuff.rename(hardness_cache_file+".tmp", hardness_cache_file)

    l=list(range(processor.num_files))
    l=[x for _, x in sorted(zip(hardness, l), reverse=True)]

    base_name=processor.config_name+"_"+processor.task[0]+"h"
    desc=processor.config_name+"/"+processor.task+"/deleting"
    for i in tqdm(range(max_images, processor.num_files),
                  desc=desc.ljust(45),
                  colour="#20c020",
                  smoothing=0.01):
        processor.delete(l[i])

    desc=processor.config_name+"/"+processor.task+"/renaming"
    for i in tqdm(range(max_images), desc=desc.ljust(45), colour="#30e030", smoothing=0.01):
        new_name=base_name+f"{i:07d}"
        if add_exif or test:
            processor.append_exif_comment(l[i], f"hardness={hardness[l[i]]:0.2f}")
        processor.rename(l[i], new_name)

    processor.reload_files()
    processor.log_stats()

def build_task_dedup(processor):
    desc=processor.config_name+"/"+processor.task+"/dedup"
    num=processor.num_files
    deduper=stuff.ImgDedup()
    num_deleted=0
    for i in tqdm(range(num), desc=desc.ljust(45), colour="#10a010", smoothing=0.01):
        img=processor.get_img_path(i)
        r=deduper.test(img)
        if r.duplicate==True:
            processor.delete(i)
            num_deleted+=1
    processor.log(f"Deup deleted {num_deleted} images from {num}, {100.0*num_deleted/(num+1e-7):3.2f}%")
    processor.reload_files()
    processor.log_stats()

def build_task_sample(processor, sample_config, test=False):
    """
    sample
    - Keep only a subset of images in the dataset (useful for quick experiments).
    - This stage is deterministic when `seed` is set and `method: random` is used.

    Config options:
      max_images: int (required)            # number of images to keep
      method: "random"|"first"|"last"       # selection strategy (default: random)
      seed: int                             # RNG seed for method=random (default: 0)
      rename: bool                          # if true, rename kept images sequentially (default: true)
      base_name: str                        # output name prefix for renaming (default: <config>_<t>s)
    """
    if sample_config is None:
        return

    max_images=dsu.get_param(processor.task, sample_config, "max_images", None)
    if max_images is None:
        return

    if test:
        max_images=min(int(max_images), 50)

    num=processor.num_files
    if num<=max_images:
        processor.log(f"sample: nothing to do ({num} <= {max_images})")
        return

    method=dsu.get_param(processor.task, sample_config, "method", "random")
    seed=int(dsu.get_param(processor.task, sample_config, "seed", 0))
    do_rename=bool(dsu.get_param(processor.task, sample_config, "rename", True))

    indices=list(range(num))
    if method=="random":
        rng=random.Random(seed)
        rng.shuffle(indices)
        keep=set(indices[:max_images])
    elif method=="first":
        keep=set(indices[:max_images])
    elif method=="last":
        keep=set(indices[-max_images:])
    else:
        raise ValueError(f"sample: unknown method '{method}' (expected random|first|last)")

    desc=processor.config_name+"/"+processor.task+f"/sample({method})"
    for i in tqdm(range(num), desc=desc.ljust(45), colour="#10a0ff", smoothing=0.01):
        if i not in keep:
            processor.delete(i)

    processor.reload_files()

    if do_rename:
        base_name=dsu.get_param(processor.task, sample_config, "base_name", None)
        if base_name is None:
            base_name=processor.config_name+"_"+processor.task[0]+"s"
        desc=processor.config_name+"/"+processor.task+"/sample renaming"
        for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour="#20b0ff", smoothing=0.01):
            processor.rename(i, base_name+f"{i:07d}")
        processor.reload_files()

    processor.log(f"sample: reduced {num} -> {processor.num_files} (method={method}, seed={seed}, rename={do_rename})")
    processor.log_stats()

def build_task_llm_filter(processor, llm_filter_config, test=False):
    """
    llm_filter
    - filter input images by a vision model query
    """
    max_images=dsu.get_param(processor.task, llm_filter_config, "max_images", 100000)
    llm_name=dsu.get_param(processor.task, llm_filter_config, "llm", "gpt-5-mini")
    llm=stuff.simple_llm(llm_name)
    batch_size=dsu.get_param(processor.task, llm_filter_config, "llm_batch", 2000)

    if test:
        max_images=min(max_images, 250)

    system_prompt_file=llm_filter_config["prompt"]
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    #processor.log(f"llm_filter: prompt {system_prompt}")
    llm.set_system_prompt(system_prompt)

    prompt="""
        Task for vision model:
        1 . Study the supplied image carefully and evaluate the image per system-prompt rules.
        2 . Produce a single JSON object that satisfies the OUTPUT CONTRACT above.
        3 . Return nothing except the minified JSON inside a fenced code-block
        """

    desc=processor.config_name+"/"+processor.task+"/llm filter"
    num=min(max_images, processor.num_files)

    batch_list=[]
    num_deleted=0
    for i in tqdm(range(num), desc=desc.ljust(45), colour="#10a010", smoothing=0.01):
        #gts=processor.get_gt(i)

        s={"prompt": prompt,
           "img_path": processor.get_img_path(i),
           "id":i,
           "img_uid": processor.get_image_origin(i),
           "cache_name": desc}
        batch_list.append(s)

        if len(batch_list)>=batch_size or i==num-1:
            results=llm.infer_cached_batch(batch_list, attempts=2)
            for j,r in enumerate(results):
                if "false" in r:
                    if False:
                        import cv2
                        img = cv2.imread(processor.get_img_path(id_list[j]))
                        if img is not None:
                            # Display the image
                            cv2.imshow("Image", img)
                            # Wait for 2000 milliseconds (2 seconds)
                            cv2.waitKey(2000)
                            # Close the display window
                            cv2.destroyAllWindows()
                    processor.delete(batch_list[j]["id"])
                    num_deleted+=1
            print(f"COSTS {llm.get_stats()}")


            batch_list=[]
    processor.log(f"LLM stats {llm.get_stats()}")
    processor.log(f"Deleted {num_deleted} images")
    processor.reload_files()
    processor.log_stats()


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
        #if gts is None:
        #    continue
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

    # if we are adding faces we are either adding missing faces/face kps to
    # a dataset with a separate face class OR we are adding mouth points etc
    # to a dataset with facepose keypoints
    assert "face" in processor.class_names or processor.facepose_kp==True

    # we need dataset_processor to undestand/detect 'face' so
    # temporarily add it to classes of interest if not already present

    added_face_class=False
    if not "face" in processor.class_names:
        processor.class_names.append("face")
        added_face_class=True

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

        # case we have/want a separate face class
        if added_face_class==False:
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
                    if stuff.has_face_points(gt) is False and f["confidence"]>kp_thr:
                        gt["face_points"]=f["face_points"]
                        face_kp_added+=1
        else:
            # copy across the detected face points into the person
            for j in range(len(face_idx)):
                face=detections[face_idx[j]]
                gt=gts[person_idx[j]]

                assert "facepose_points" in gt
                assert "facepose_points" in face # will have been remapped for us

                face_to_facepose_map=[2, 1, 0, 18, 17]
                for ii in range(5):
                    dp=face_to_facepose_map[ii]
                    for jj in range(3):
                        gt["facepose_points"][dp*3+jj]=face['facepose_points'][dp*3+jj]
                face_kp_added+=1

        processor.replace_annotations(i, gts)

    if added_face_class:
        processor.class_names=processor.class_names[:-1] # remove 'face' to restore how we were

    processor.log(f" add_faces det={model} sz={imgsz} thrs={box_thr},{kp_thr}: {faces_added} faces added; {face_kp_added} face keypoint sets added")
    processor.log_stats()

def build_task_merge_facepose(dataset_yaml_folder, config=None):
    """
    For a dataset that has person GTs with standard 17-point pose points
    and Face GTs with 5-point face points, merge the face points into the
    person detections - the persons now have 19-point 'facepose' points.
    19 because 3 of the points (nose/eyes) are common
    """

    ds=os.listdir(dataset_yaml_folder)

    for d in ds:
        dataset_yaml=dataset_yaml_folder+"/"+d+"/dataset.yaml"
        print(f"Processing {dataset_yaml}...")
        processor=DatasetProcessor(dataset_yaml, task="val", append_log=False)
        output_dataset_name=dsu.get_param(processor.task, config, "output_dataset_name", None)
        do_delete=dsu.get_param(processor.task, config, "delete_existing", False)
        assert do_delete==False
        class_names=processor.get_class_names()
        class_names_no_face=copy.copy(class_names)
        class_names_no_face.pop(class_names_no_face.index("face"))
        if output_dataset_name is None:
            output_dataset_name="output_datasets/"+os.path.basename(dataset_yaml_folder)+"-fp/"+processor.config_name
            output_dataset_name=dsu.unique_dataset_name(output_dataset_name)
        out_yaml_path=dsu.make_dataset_yaml(output_dataset_name,
                                            config_name=processor.config_name,
                                            class_names=class_names_no_face,
                                            face_kp=False,
                                            pose_kp=False,
                                            facepose_kp=True)

        dsu.dataset_copy_comments(dataset_yaml, out_yaml_path)

        person_class=processor.get_class_index("person")
        face_class=processor.get_class_index("face")
        assert person_class!=-1
        assert face_class!=-1

        for task in ["val", "train"]:
            processor=DatasetProcessor(dataset_yaml, task=task, append_log=False)
            out_processor=DatasetProcessor(out_yaml_path,
                                        task=task,
                                        append_log=False,
                                        face_kp=False,
                                        pose_kp=False,
                                        facepose_kp=True)
            out_processor.log(f"======== {task} merge facepose =======")
            desc=processor.config_name+"/"+processor.task+"/merge facepose"
            class_names=processor.get_class_names()
            class_names_no_face=copy.copy(class_names)
            class_names_no_face.pop(class_names_no_face.index("face"))

            for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour="#8080ff", smoothing=0.01):
                gts=processor.get_gt(i)
                if gts is not None:
                    dsu.match_facepose(gts, person_class, face_class)
                out_gts=[]
                for g in gts:
                    if g["class"]==face_class:
                        continue
                    if g["class"]>face_class:
                        g["class"]-=1
                    out_gts.append(g)
                out_processor.add(processor.get_img_name(i), processor.get_img_path(i), out_gts)
            out_processor.reload_files()
            out_processor.log_stats()
        if do_delete:
            dataset_delete(dataset_yaml)
    return out_yaml_path

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


def build_task_expand_attribute(dataset_yaml,
                                config=None,
                                face_kp=False,
                                pose_kp=True,
                                facepose_kp=False):
    """
    Create a new object class for each binary attribute
    Expand attributes stored as a binary vector to whole GT objects
    of the new types
    """
    processor=DatasetProcessor(dataset_yaml,
                               task="val",
                               append_log=False,
                               face_kp=face_kp,
                               pose_kp=pose_kp,
                               facepose_kp=facepose_kp)

    output_dataset_name=dsu.get_param(processor.task, config, "output_dataset_name", None)
    do_delete=dsu.get_param(processor.task, config, "delete_existing", False)
    class_names=processor.get_class_names()
    attributes=processor.attributes
    num_attrs=len(attributes)

    class_names_extended=copy.copy(class_names)
    for a in attributes:
        class_names_extended.append(a.replace(":", "_"))

    if output_dataset_name is None:
        output_dataset_name="output_datasets/"+processor.config_name+"_exp"
        output_dataset_name=dsu.unique_dataset_name(output_dataset_name)
    out_yaml_path=dsu.make_dataset_yaml(output_dataset_name,
                                        config_name=processor.config_name,
                                        class_names=class_names_extended,
                                        face_kp=processor.face_kp,
                                        pose_kp=processor.pose_kp,
                                        facepose_kp=processor.facepose_kp,
                                        attributes=None)

    dsu.dataset_copy_comments(dataset_yaml, out_yaml_path)

    attributes=processor.attributes
    attribute_classes=[]
    for a in attributes:
        cl=a.split(":")[0]
        cl_idx=processor.get_class_index(cl)
        if not cl_idx in attribute_classes:
            attribute_classes.append(cl_idx)

    for task in ["val", "train"]:
        processor=DatasetProcessor(dataset_yaml, task=task, append_log=False)
        out_processor=DatasetProcessor(out_yaml_path, task=task, append_log=True)
        out_processor.log(f"======= {task} expand attributes ============")
        desc=processor.config_name+"/"+processor.task+"/expand attributes"

        for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour="#8080ff", smoothing=0.01):
            gts=processor.get_gt(i)
            out_gts=[]
            for g in gts:
                out_gts.append(g)
                if g["class"] in attribute_classes:
                    if "attrs" in g:
                        attrs=copy.copy(g["attrs"])
                        del g["attrs"]
                        for j in range(num_attrs):
                            if attrs[j]>0.5:
                                new_gt=copy.copy(g) # remember; keep same pose_points etc
                                new_gt["class"]=len(class_names)+j
                                out_gts.append(new_gt)
            out_processor.add(processor.get_img_name(i),
                              processor.get_img_path(i),
                              out_gts)
        out_processor.reload_files()
        out_processor.log_stats()
    if do_delete:
        dataset_delete(dataset_yaml)
    return out_yaml_path


def build_task_write_v10(dataset_yaml,
                         config=None,
                         face_kp=False,
                         pose_kp=True,
                         facepose_kp=False):
    """
    Write dataset in v10 (split) format: one row per object with cls (0-4), bbox, 50 attr columns, keypoints.
    Replaces the expand_attributes step when targeting the Ultralytics attributes branch.
    Reads internal format (one row per object, 'A' + attrs) and writes v10 labels + v10 YAML.
    """
    processor = DatasetProcessor(dataset_yaml,
                                task="val",
                                append_log=False,
                                face_kp=face_kp,
                                pose_kp=pose_kp,
                                facepose_kp=facepose_kp)
    output_dataset_name = dsu.get_param(processor.task, config or {}, "output_dataset_name", None)
    do_delete = dsu.get_param(processor.task, config or {}, "delete_existing", False)
    class_names = processor.get_class_names()
    attributes = processor.attributes
    if attributes is None:
        attributes = []
    attr_names = [a.replace(":", "_") for a in attributes]
    while len(attr_names) < dsu.ATTR_NC_V10:
        attr_names.append("attr_" + str(len(attr_names)))

    if output_dataset_name is None:
        output_dataset_name = "output_datasets/" + processor.config_name + "_v10"
        output_dataset_name = dsu.unique_dataset_name(output_dataset_name)

    out_yaml_path = dsu.make_dataset_yaml_v10(
        output_dataset_name,
        config_name=processor.config_name,
        class_names=class_names,
        attr_names=attr_names[:dsu.ATTR_NC_V10],
        face_kp=processor.face_kp,
        pose_kp=processor.pose_kp,
        facepose_kp=processor.facepose_kp,
        attr_nc=dsu.ATTR_NC_V10,
    )
    dsu.dataset_copy_comments(dataset_yaml, out_yaml_path)

    out_path = dsu.get_dataset_path(out_yaml_path)

    for task in ["val", "train"]:
        proc = DatasetProcessor(dataset_yaml, task=task, append_log=False,
                               face_kp=face_kp, pose_kp=pose_kp, facepose_kp=facepose_kp)
        out_proc = DatasetProcessor(out_yaml_path, task=task, append_log=(task == "train"))
        if task == "train":
            out_proc.log("======= write_v10 (split format) ============")
        desc = proc.config_name + "/" + task + "/write v10"
        for i in tqdm(range(proc.num_files), desc=desc.ljust(45), colour="#8080ff", smoothing=0.01):
            gts = proc.get_gt(i)
            if gts is None:
                gts = []
            for g in gts:
                if "attrs" not in g or g["attrs"] is None:
                    g["attrs"] = [0.0] * dsu.ATTR_NC_V10
                elif len(g["attrs"]) < dsu.ATTR_NC_V10:
                    g["attrs"] = list(g["attrs"]) + [0.0] * (dsu.ATTR_NC_V10 - len(g["attrs"]))
            an_txt = dsu.write_annotations_v10(
                gts,
                include_face=proc.face_kp,
                include_pose=proc.pose_kp,
                include_facepose=proc.facepose_kp,
                attr_nc=dsu.ATTR_NC_V10,
            )
            name = proc.get_img_name(i)
            dst_img = out_path + "/" + task + "/images/" + name + ".jpg"
            dst_label = out_path + "/" + task + "/labels/" + name + ".txt"
            shutil.copy2(proc.get_img_path(i), dst_img)
            with open(dst_label, "w") as f:
                f.write(an_txt)
        if task == "train":
            out_proc.reload_files()
            out_proc.log_stats()
    if do_delete:
        dataset_delete(dataset_yaml)
    return out_yaml_path


def build_task_filter_classes(dataset_yaml,
                              config=None,
                              output_dataset_name=None,
                              face_kp=False,
                              pose_kp=True,
                              facepose_kp=False):
    """
    Delete a list of classes from a dataset
    """
    processor=DatasetProcessor(dataset_yaml,
                               task="val",
                               append_log=False,
                               face_kp=face_kp,
                               pose_kp=pose_kp,
                               facepose_kp=facepose_kp)
    class_names=processor.get_class_names()
    classes_to_delete=dsu.get_param(processor.task, config, "classes_to_delete", [])
    classes_to_filter_small=dsu.get_param(processor.task, config, "classes_to_filter_small", [])
    classes_to_strip_pose=dsu.get_param(processor.task, config, "classes_to_strip_pose", [])
    do_delete=dsu.get_param(processor.task, config, "delete_existing", False)
    pose_kp=dsu.get_param(processor.task, config, "pose_kp", processor.pose_kp)
    face_kp=dsu.get_param(processor.task, config, "face_kp", processor.face_kp)
    classes_poseattr=dsu.get_param(processor.task, config, "classes_poseattr", None)
    facepose_kp=dsu.get_param(processor.task, config, "facepose_kp", processor.facepose_kp)

    output_dataset_name=dsu.get_param(processor.task, config, "output_dataset_name", output_dataset_name)

    classes_out=[]
    class_index_remap=[-1]*len(class_names)
    filter_small_mask=[False]*len(class_names)
    strip_pose_mask=[False]*len(class_names)

    assert classes_poseattr is None

    n_out=0
    deleted_classes=[]
    for i,c in enumerate(class_names):
        if c in classes_to_filter_small:
            filter_small_mask[i]=True
        if c in classes_to_strip_pose:
            strip_pose_mask[i]=True

        if c in classes_to_delete:
            deleted_classes.append(c)
        else:
            classes_out.append(c)
            class_index_remap[i]=n_out
            n_out+=1

    if output_dataset_name is None:
        name=processor.config_name
        output_dataset_name="output_datasets/"+name+"_del"
    output_dataset_name=dsu.unique_dataset_name(output_dataset_name)
    out_yaml_path=dsu.make_dataset_yaml(output_dataset_name,
                                        config_name=processor.config_name,
                                        class_names=classes_out,
                                        face_kp=face_kp,
                                        pose_kp=pose_kp,
                                        facepose_kp=facepose_kp,
                                        attributes=processor.attributes)

    dsu.dataset_copy_comments(dataset_yaml, out_yaml_path)

    for task in ["val", "train"]:
        processor=DatasetProcessor(dataset_yaml, task=task, append_log=False)
        out_processor=DatasetProcessor(out_yaml_path, task=task, append_log=True,
                                       pose_kp=pose_kp,
                                       face_kp=face_kp,
                                       facepose_kp=facepose_kp)
        out_processor.log(f"======= {task} filter classes {deleted_classes} ============")
        desc=processor.config_name+"/"+processor.task+"/filter classes"

        for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour="#8080ff", smoothing=0.01):
            gts=processor.get_gt(i)
            out_gts=[]
            for g in gts:
                ci=g["class"]
                g["class"]=class_index_remap[ci]
                if g["class"]==-1:
                    continue

                if filter_small_mask[ci] is True:
                    if stuff.is_large(g) is False:
                        continue
                if strip_pose_mask[ci] is True:
                    if "pose_points" in g:
                        g["pose_points"]=[0]*len(g["pose_points"])
                    if "face_points" in g:
                        g["face_points"]=[0]*len(g["face_points"])
                    if "facepose_points" in g:
                        g["facepose_points"]=[0]*len(g["facepose_points"])
                out_gts.append(g)
            out_processor.add(processor.get_img_name(i),
                              processor.get_img_path(i),
                              out_gts)
        out_processor.reload_files()
        out_processor.log_stats()
    if do_delete:
        dataset_delete(dataset_yaml)
    return out_yaml_path

def build_task_generate_backgrounds(class_names, config, face_kp=True, pose_kp=True, facepose_kp=False, test=False, config_name=None):

    loader=config["loader"]
    model=config["model"]
    check_model=None
    if "check_model" in config:
        check_model=config["check_model"]
    check_thr=0.55
    add_exif=True
    if "add_exif" in config:
        add_exif=config["add_exif"]
    if "check_thr" in config:
        check_thr=config["check_thr"]
    all_classes=class_names+["confused", "background"]
    if config_name is None:
        config_name=dsu.loader_name(loader)
    unique_dataset_name=dsu.unique_dataset_name("output_datasets/"+config_name+"_bg")

    yaml_path=dsu.make_dataset_yaml(unique_dataset_name,
                                    config_name=config_name,
                                    class_names=class_names,
                                    face_kp=face_kp,
                                    pose_kp=pose_kp,
                                    facepose_kp=facepose_kp)
    for task in ["val","train"]:

        processor=DatasetProcessor(yaml_path,
                                   task=task,
                                   append_log=False,
                                   face_kp=face_kp,
                                   pose_kp=pose_kp,
                                   facepose_kp=facepose_kp)
        max_images=dsu.get_param(processor.task, config, "max_images", 1000)
        if test:
            max_images=min(max_images, 50)
        if max_images==0:
            continue

        loader_fn=dsu.get_loader(loader)
        o=loader_fn(class_names=all_classes, task=task, ds_params=None)

        maps=o.get_category_maps()
        for c in all_classes:
            if c in maps:
                o.add_category_mapping(maps[c], c)

        ids=o.get_image_ids()
        num=min(len(ids), max_images)
        max_index=min(len(ids), 10*num)
        print(f"generate_backgrounds: max_images {max_images} imported {len(ids)} images; max_index={max_index}")
        index=0
        len_ids=len(ids)
        desc=config_name+"/"+task+"/bg generate"
        with tqdm(total=max_index, desc=desc.ljust(45), colour="#ffcc00", smoothing=0.01) as pbar:
            for n,i in enumerate(ids):
                pbar.update(max((n*max_index+len_ids-1)//len_ids, index)-pbar.n)

                img_path=o.get_img_path(i)
                if not os.path.isfile(img_path):
                    continue
                dets=o.get_annotations(i)
                if dets is None:
                    continue

                non_bg=0
                bg_class=len(all_classes)-1
                for d in dets:
                    if d["class"]!=bg_class:
                        non_bg+=1
                if non_bg!=0:
                    continue

                n="t"+f"{index:07d}"
                index+=1
                processor.add(n, img_path, [], add_exif=add_exif)

                if index>=max_index:
                    break
            pbar.update(max_index-pbar.n)

        processor.reload_files()

        processor.set_object_detector(model, imgsz=640, thr=0.1, half=True, rect=True, batch_size=32)

        fp_score=[0.0]*processor.num_files
        desc=config_name+"/"+task+"/bg measure"
        for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour="#ffdd20", smoothing=0.01):
            dets=processor.get_detections(i)
            fp_tot=random.random()*0.005 # just so we get a random mix of BGs with same score
            for d in dets:
                fp_tot+=d["confidence"]
            fp_score[i]=fp_tot

        l=list(range(processor.num_files))
        l=[x for _, x in sorted(zip(fp_score, l), reverse=True)]

        base_name="bg_"+config_name+"_"+processor.task[0]

        desc=(config_name+"/"+processor.task+"/bg rename").ljust(45)
        for i in tqdm(range(processor.num_files), desc=desc, colour="#ffee40", smoothing=0.01):
            new_name=base_name+f"{i:07d}"
            index=l[i]
            if add_exif or test:
                processor.append_exif_comment(index, f"bghardness={fp_score[index]:.2f}")
            processor.rename(index, new_name)

        processor.reload_files()

        base_name="bgc_"+config_name+"_"+processor.task[0]

        do_check="person" in class_names and check_model!=None
        if do_check:
            processor.set_object_detector(check_model, imgsz=640, thr=0.1, half=True, rect=True, batch_size=24)
        num_deleted=0
        num_ok=0
        desc=config_name+"/"+task+"/bg check"
        for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour="#ffff60", smoothing=0.01):
            if num_ok>=max_images:
                processor.delete(i)
                continue
            if do_check:
                dets=processor.get_detections(i)
                max_conf=0
                num_high=0
                for d in dets:
                    max_conf=max(max_conf, d["confidence"])
                    if d["confidence"]>0.35:
                        num_high+=1
                if max_conf>check_thr or num_high>=3:
                    processor.delete(i)
                    num_deleted+=1
                    continue
                if add_exif or test:
                    processor.append_exif_comment(i, f"check={max_conf:0.3f}:{num_high}")
            new_name=base_name+f"{num_ok:07d}"
            processor.rename(i, new_name)
            num_ok+=1

        processor.reload_files()
        processor.log(f"deleted {num_deleted} of {processor.num_files} (concern of mis-labelling) backgrounds output {num_ok}")
    return yaml_path

def merge(a: dict, b: dict, path=None):
    """
    Recursively merge two dictionaries
    Values in b take precedence, b is merged 'over' a
    """
    if path is None:
        path=[]
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

def get_expanded_config(path, config_filename, dataset_name):
    """
    Recursively expand configuration
    'inherit_config' lets one specify a configuration in the same or
    another file to set default options
    """
    this_config = stuff.load_dictionary(os.path.join(path, config_filename))
    dataset_config=this_config["datasets"][dataset_name]
    if "general" in dataset_config:
        if "inherit_config" in dataset_config["general"]:
            inherit_from=dataset_config["general"]["inherit_config"]
            if "/" in inherit_from:
                ih_config_filename, ih_dataset_name=inherit_from.split("/")
            else:
                ih_config_filename, ih_dataset_name=config_filename, inherit_from
            ih_config=get_expanded_config(path, ih_config_filename, ih_dataset_name)
            ih_config=copy.deepcopy(ih_config)
            merge(ih_config, dataset_config)
            return ih_config
    return dataset_config

def generate_dataset(path, config_filename, dataset_name, force_generate=False, test=False):
    config = stuff.load_dictionary(os.path.join(path, config_filename))

    # create new empty dataset
    dataset_config=get_expanded_config(path, config_filename, dataset_name)
    # check for OPENAI_API_KEY so it doesn't annoyingly crash later
    if "add_attributes" in dataset_config:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is None:
            print("Error: OPENAI_API_KEY is not set")
            exit()

    general_config=dataset_config["general"]
    class_names=general_config["class_names"]
    attributes=general_config.get("attributes", None)
    face_kp=general_config.get("face_kp", False)
    pose_kp=general_config.get("pose_kp", False)
    facepose_kp=general_config.get("facepose_kp", False)
    tasks=["val","train"]
    chunk_size=10000
    if "generate" in general_config:
        if general_config["generate"] is False and force_generate is False:
            return None
    if "tasks" in general_config:
        tasks=general_config["tasks"]

    if "chunk_size" in general_config:
        chunk_size=general_config["chunk_size"]

    unique_dataset_name=dsu.unique_dataset_name("output_datasets/"+dataset_name)

    yaml_path=None

    if "merge" in dataset_config:
        datasets_to_merge=dataset_config["merge"]
        for d in datasets_to_merge:
            to_merge_yaml=generate_dataset(path, config_filename, d, force_generate=True, test=test)
            processor=DatasetProcessor(to_merge_yaml)

            if yaml_path is None:
                # no classes are passed, this DP will be configured according to the yaml file
                # then we use it's values in creating our output yaml
                # long winded way of making sure the merged yaml has the same properties as the
                # original ones
                processor=DatasetProcessor(to_merge_yaml)
                merged_config_class_names=processor.class_names
                merged_config_attributes=processor.attributes
                merged_config_face_kp=processor.face_kp
                merged_config_pose_kp=processor.pose_kp
                merged_config_facepose_kp=processor.facepose_kp
                yaml_path=dsu.make_dataset_yaml(unique_dataset_name,
                                    config_name=dataset_name,
                                    class_names=merged_config_class_names,
                                    face_kp=merged_config_face_kp,
                                    pose_kp=merged_config_pose_kp,
                                    facepose_kp=merged_config_facepose_kp,
                                    attributes=merged_config_attributes)
            assert processor.class_names==merged_config_class_names
            assert processor.attributes==merged_config_attributes
            assert processor.face_kp==merged_config_face_kp
            assert processor.pose_kp==merged_config_pose_kp
            assert processor.facepose_kp==merged_config_facepose_kp

            dataset_merge_and_delete(to_merge_yaml, yaml_path)

        for task in ["val","train"]:
            x=DatasetProcessor(yaml_path, task=task, append_log=True)
            build_task_dedup(x)
            x.log("======================")
            x.log(" Final merged stats: "+str(x.basic_stats()))
        return yaml_path

    yaml_path=dsu.make_dataset_yaml(unique_dataset_name,
                                    config_name=dataset_name,
                                    class_names=class_names,
                                    face_kp=face_kp,
                                    pose_kp=pose_kp,
                                    facepose_kp=facepose_kp,
                                    attributes=attributes)
    # Overall wall-clock timer (across both val/train tasks, backgrounds, etc).
    overall_start_time=time.time()
    for task in tasks:
        processor=DatasetProcessor(yaml_path,
                                   task=task,
                                   append_log=True,
                                   face_kp=face_kp,
                                   pose_kp=pose_kp,
                                   facepose_kp=facepose_kp)

        # import
        import_config=dataset_config["import"]
        build_task_import(processor, import_config, test=test)
        # hard subset

        if "llm_filter" in dataset_config:
            llm_filter_config=dataset_config["llm_filter"]
            build_task_llm_filter(processor, llm_filter_config, test=test)

        if "make_hard" in dataset_config:
            hard_config=dataset_config["make_hard"]
            build_task_make_hard(processor, hard_config, test=test)

        # Optional: keep only a subset of images (useful for fast iteration).
        if "sample" in dataset_config:
            sample_config=dataset_config["sample"]
            build_task_sample(processor, sample_config, test=test)

        # Per-task timer used for chunk progress estimation.
        start_time=time.time()
        processor.chunk_size=chunk_size
        processor.reload_files()

        for chunk in range(processor.num_chunks):
            print(f"\n==== {processor.config_name} {processor.task} : chunk {chunk} of {processor.num_chunks} ====")
            t=time.time()
            processor.set_chunk(chunk)

            # add objects

            if "add_objects" in dataset_config:
                add_object_config=dataset_config["add_objects"]
                if add_object_config is not None:
                    for detector_config in add_object_config:
                        build_task_add_objects(processor, detector_config)

            # mask objects

            if "mask_objects" in dataset_config:
                mask_object_config=dataset_config["mask_objects"]
                for detector_config in mask_object_config:
                    build_task_mask_objects(processor, detector_config)

            # add pose points

            if "add_pose" in dataset_config:
                add_object_config=dataset_config["add_pose"]
                for detector_config in add_object_config:
                    build_task_add_pose(processor, detector_config)

            # add faces

            if "add_faces" in dataset_config:
                add_object_config=dataset_config["add_faces"]
                for detector_config in add_object_config:
                    build_task_add_faces(processor, detector_config)

            # add FIQA

            if "add_fiqa" in dataset_config:
                add_fiqa_config=dataset_config["add_fiqa"]
                if add_fiqa_config is not None:
                    build_task_add_fiqa(processor, add_fiqa_config)

            # normalise

            build_task_normalise(processor)

            # add attributes

            if "add_attributes" in dataset_config:
                add_attributes_config=dataset_config["add_attributes"]
                # copy loader from import config unless otherwise specified....
                if not "loader" in add_attributes_config or add_attributes_config["loader"] is None:
                    loader=dataset_config["import"]["loader"]
                    add_attributes_config["loader"]=loader
                build_task_add_attributes(processor, add_attributes_config)

            elapsed=time.time()-t
            total_elapsed=time.time()-start_time
            remaining=(processor.num_chunks-chunk-1)*total_elapsed/(chunk+1.0)

            print(f"chunk took {dsu.timestr(elapsed)};"
                  +f" total {dsu.timestr(total_elapsed)};"
                  +f" remaining {dsu.timestr(remaining)}")
    print(f"\n==== {processor.config_name} {processor.task} : all chunks complete ====\n")

    # generate background images

    if "add_backgrounds" in dataset_config:
        bg_config=dataset_config["add_backgrounds"]
        if bg_config["loader"]!="None":
            bg_yaml=build_task_generate_backgrounds(class_names,
                                                    bg_config,
                                                    face_kp=face_kp,
                                                    pose_kp=pose_kp,
                                                    facepose_kp=facepose_kp,
                                                    test=test,
                                                    config_name=dataset_name)

            for task in tasks:
                processor=DatasetProcessor(bg_yaml, task=task, append_log=False, class_names=class_names)
                dest_path=dsu.get_dataset_path(yaml_path)+"/"+task
                desc="BG"+"/"+task+"/copying"
                for i in tqdm(range(processor.num_files), desc=desc.ljust(45), smoothing=0.01):
                    src_img=processor.get_img_path(i)
                    src_label=processor.get_label_path(i)
                    dst_img=dest_path+"/images/"+os.path.split(src_img)[1]
                    dst_label=dest_path+"/labels/"+os.path.split(src_label)[1]
                    stuff.rename(src_img, dst_img)
                    stuff.rename(src_label, dst_label)
            path=dsu.get_dataset_path(bg_yaml)
            print(f"generate_backgrounds: delete folder {path}...")
            stuff.rmdir(path)

    if "merge_facepose" in dataset_config:
        merge_facepose_config=dataset_config["merge_facepose"]
        yaml_path=build_task_merge_facepose(yaml_path, merge_facepose_config)

    if "write_v10" in dataset_config:
        write_v10_config=dataset_config["write_v10"]
        yaml_path=build_task_write_v10(yaml_path,
                                       config=write_v10_config,
                                       face_kp=face_kp,
                                       pose_kp=pose_kp,
                                       facepose_kp=facepose_kp)
    elif "expand_attributes" in dataset_config:
        expand_config=dataset_config["expand_attributes"]
        yaml_path=build_task_expand_attribute(yaml_path,
                                              expand_config,
                                              face_kp=face_kp,
                                              pose_kp=pose_kp,
                                              facepose_kp=facepose_kp)

    if "filter_classes" in dataset_config:
        filter_config=dataset_config["filter_classes"]
        yaml_path=build_task_filter_classes(yaml_path,
                                            filter_config,
                                            face_kp=face_kp,
                                            pose_kp=pose_kp,
                                            facepose_kp=facepose_kp)

    total_elapsed=int(time.time()-overall_start_time)
    for task in ["val", "train"]:
        x=DatasetProcessor(yaml_path, task=task, append_log=True)
        x.log(" Final merged stats: "+str(x.basic_stats()))
        if task=="train":
            x.log("======================")
            x.log(f" Total elapsed time {dsu.timestr(total_elapsed)}")
    return yaml_path

def postprocess_datasets(config):
    pp_config=config["postprocess"]
    in_datasets=[]
    out_datasets=[]
    if "input_folder" in pp_config:
        input_folder=pp_config["input_folder"]
        if "output_folder" in pp_config:
            output_folder=pp_config["output_folder"]
        else:
            output_folder=input_folder
        output_folder=dsu.unique_dataset_name(output_folder)

        folders=os.listdir(dsu.mldata_folder()+"/"+input_folder)
        for f in folders:
            in_datasets.append(input_folder+"/"+f)
            out_datasets.append(output_folder+"/"+f)
    for i,_ in enumerate(in_datasets):
        build_task_filter_classes(dsu.mldata_folder()+"/"+in_datasets[i]+"/dataset.yaml",
                                  config=pp_config,
                                  output_dataset_name=out_datasets[i])



def process_dataset(config_filename, test=False, postprocess_only=False):
    config = stuff.load_dictionary(config_filename)
    if "datasets" in config and postprocess_only is False:
        for dataset_name in config["datasets"]:
            generate_dataset(os.path.split(config_filename)[0],
                            os.path.split(config_filename)[1],
                            dataset_name, test=test)
    if "postprocess" in config:
        postprocess_datasets(config)

def process_dataset_single_task(config_yaml):
    config = stuff.load_dictionary(config_yaml)
    # Convenience helper for inspecting a single config file.
    path=os.path.split(config_yaml)[0]
    config_filename=os.path.split(config_yaml)[1]

    print(config_yaml)
    print(path)
    print(config_filename)
    print(os.path.join(path,config_filename))

    for dataset_name in config["datasets"]:
        dataset_config=get_expanded_config(path, config_filename, dataset_name)
        print(dataset_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='process_dataset.py')
    parser.add_argument('--logging', type=str, default='info', help="Logging config: level[:console|file]")
    parser.add_argument('--config', type=str, default="dataset_config.yaml", help='Configuration to use (json/yaml)')
    parser.add_argument('--test', action='store_true', help='run in test mode (limit number of images)')
    parser.add_argument('--llm-test',  type=str, default=None, help='run in LLM test')
    parser.add_argument('--facepose', action='store_true', help='convert to facepose')
    parser.add_argument('--filter', action='store_true', help='filter datasets')
    parser.add_argument('--postprocess', action='store_true', help='postprocess datasets')
    parser.add_argument('--expand-attr', type=str, default=None, help='expand attributes to objects')
    opt = parser.parse_args()
    stuff.configure_root_logger(opt.logging)
    if opt.facepose:
        build_task_merge_facepose("/mldata/v5-attr", None)
        exit()

    if opt.filter:
        config1=stuff.load_dictionary("/mldata/config/dataset/dataset_attributes_default.yaml")
        for s in ["coco", "openimages", "hoang", "weapons", "widerface"]:

            to_filter_small=config1["datasets"]["default"]["filter_classes"]["classes_to_filter_small"]
            config={"classes_to_filter_small":to_filter_small}
            build_task_filter_classes("/mldata/attr-v2-large/"+s+"-attr-v2-large/dataset.yaml", config)
        exit()

    if opt.llm_test!=None:
        llm_test(opt.llm_test)
        exit()

    if opt.expand_attr is not None:
        config={}
        config["classes_to_delete"]=["person_no_person_visible"]
        new_dataset=build_task_expand_attribute(opt.expand_attr)
        build_task_filter_classes(new_dataset, config)
        exit()

    stuff.gdrive_sync_mldata("/mldata/models/standard_od/pt", "from_drive")
    stuff.gdrive_sync_mldata("/mldata/models/specialist/pt", "from_drive")
    stuff.gdrive_sync_mldata("/mldata/models/v8/pt", "from_drive")

    process_dataset(opt.config, test=opt.test, postprocess_only=opt.postprocess)