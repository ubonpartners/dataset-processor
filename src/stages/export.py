import os
import shutil
from tqdm import tqdm
import stuff
import src.dataset_util as dsu
from src.dataset_processor import DatasetProcessor, dataset_delete

def _normalise_attribute_name(name):
    if not isinstance(name, str):
        return name
    return name.replace(":", "_")

def _resolve_attribute_deletes(attributes, attributes_to_delete):
    if attributes_to_delete is None:
        return [], [], []
    if not isinstance(attributes_to_delete, list):
        raise ValueError("filter_classes.attributes_to_delete must be a list")
    if any(not isinstance(a, str) for a in attributes_to_delete):
        raise ValueError("filter_classes.attributes_to_delete must contain strings")

    attrs=attributes or []
    attr_lookup={}
    for i, attr in enumerate(attrs):
        key=_normalise_attribute_name(attr)
        if key not in attr_lookup:
            attr_lookup[key]=[]
        attr_lookup[key].append(i)

    missing=[]
    delete_indices=set()
    for attr in attributes_to_delete:
        key=_normalise_attribute_name(attr)
        if key not in attr_lookup:
            missing.append(attr)
            continue
        delete_indices.update(attr_lookup[key])

    delete_indices=sorted(delete_indices)
    deleted_attributes=[attrs[i] for i in delete_indices]
    return delete_indices, deleted_attributes, sorted(missing)

def build_task_export_dataset(dataset_yaml, config=None, face_kp=False, pose_kp=True):
    """
    Write dataset in v10 (split) format: one row per object with cls (0-4), bbox, 50 attr columns, keypoints.
    Copy an existing split-v10 dataset to a new output dataset.
    """
    processor = DatasetProcessor(dataset_yaml,
                                task="val",
                                append_log=False,
                                face_kp=face_kp,
                                pose_kp=pose_kp)
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
        attr_nc=dsu.ATTR_NC_V10,
    )
    dsu.dataset_copy_comments(dataset_yaml, out_yaml_path)

    out_path = dsu.get_dataset_path(out_yaml_path)

    for task in ["val", "train"]:
        proc = DatasetProcessor(dataset_yaml, task=task, append_log=False,
                               face_kp=face_kp, pose_kp=pose_kp)
        out_proc = DatasetProcessor(out_yaml_path, task=task, append_log=(task == "train"))
        if task == "train":
            out_proc.log("======= export split-v10 dataset ============")
        desc = proc.config_name + "/" + task + "/export"
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

def build_task_filter_classes(dataset_yaml, config=None, output_dataset_name=None, face_kp=False, pose_kp=True):
    """
    Delete a list of classes and/or attributes from a dataset.
    """
    processor=DatasetProcessor(dataset_yaml,
                               task="val",
                               append_log=False,
                               face_kp=face_kp,
                               pose_kp=pose_kp)
    class_names=processor.get_class_names()
    attributes=processor.attributes or []
    classes_to_delete=dsu.get_param(processor.task, config, "classes_to_delete", [])
    attributes_to_delete=dsu.get_param(processor.task, config, "attributes_to_delete", [])
    classes_to_filter_small=dsu.get_param(processor.task, config, "classes_to_filter_small", [])
    classes_to_strip_pose=dsu.get_param(processor.task, config, "classes_to_strip_pose", [])
    do_delete=dsu.get_param(processor.task, config, "delete_existing", False)
    pose_kp=dsu.get_param(processor.task, config, "pose_kp", processor.pose_kp)
    face_kp=dsu.get_param(processor.task, config, "face_kp", processor.face_kp)
    classes_poseattr=dsu.get_param(processor.task, config, "classes_poseattr", None)

    output_dataset_name=dsu.get_param(processor.task, config, "output_dataset_name", output_dataset_name)

    classes_out=[]
    class_index_remap=[-1]*len(class_names)
    filter_small_mask=[False]*len(class_names)
    strip_pose_mask=[False]*len(class_names)

    assert classes_poseattr is None

    delete_attr_indices, deleted_attributes, missing_attributes=_resolve_attribute_deletes(attributes, attributes_to_delete)
    delete_attr_index_set=set(delete_attr_indices)
    attributes_out=[a for i, a in enumerate(attributes) if i not in delete_attr_index_set]

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
                                        attributes=attributes_out)

    dsu.dataset_copy_comments(dataset_yaml, out_yaml_path)

    for task in ["val", "train"]:
        processor=DatasetProcessor(dataset_yaml, task=task, append_log=False)
        out_processor=DatasetProcessor(out_yaml_path, task=task, append_log=True,
                                       pose_kp=pose_kp,
                                       face_kp=face_kp)
        out_processor.log(
            f"======= {task} filter classes {deleted_classes} attributes {deleted_attributes} ============"
        )
        if missing_attributes:
            out_processor.log(
                f" filter_classes: requested attributes not present and skipped: {missing_attributes}"
            )
        desc=processor.config_name+"/"+processor.task+"/filter classes"

        for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour="#8080ff", smoothing=0.01):
            gts=processor.get_gt(i)
            out_gts=[]
            for g in gts:
                ci=g["class"]
                g["class"]=class_index_remap[ci]
                if g["class"]==-1:
                    continue

                if filter_small_mask[ci]:
                    if not stuff.is_large(g):
                        continue
                if strip_pose_mask[ci]:
                    if "pose_points" in g:
                        g["pose_points"]=[0]*len(g["pose_points"])
                    if "face_points" in g:
                        g["face_points"]=[0]*len(g["face_points"])
                if delete_attr_index_set and g.get("attrs") is not None:
                    g["attrs"]=[v for ai, v in enumerate(g["attrs"]) if ai not in delete_attr_index_set]
                out_gts.append(g)
            out_processor.add(processor.get_img_name(i),
                              processor.get_img_path(i),
                              out_gts)
        out_processor.reload_files()
        out_processor.log_stats()
    if do_delete:
        dataset_delete(dataset_yaml)
    return out_yaml_path

def build_task_generate_backgrounds(class_names, config, face_kp=True, pose_kp=True, test=False, config_name=None):

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
                                    pose_kp=pose_kp)
    for task in ["val","train"]:

        processor=DatasetProcessor(yaml_path,
                                   task=task,
                                   append_log=False,
                                   face_kp=face_kp,
                                   pose_kp=pose_kp)
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

        do_check="person" in class_names and check_model is not None
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
