import math
import os
import pickle
from tqdm import tqdm
import stuff
import src.dataset_util as dsu

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
            if not can_detect[cls]:
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
