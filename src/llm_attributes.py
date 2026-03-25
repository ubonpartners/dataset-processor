
import os
import cv2
from src.dataset_processor import DatasetProcessor
import src.dataset_util as dsu
from tqdm import tqdm
import stuff
import numpy as np
from src.llm_generic import LLMGeneric

def parse_response(r, stats, attrs_reduced, i=None):
    stats["num"]+=1

    if "unable" in r or "sorry" in r or "cant" in r or "too blurry" in r:
        if "too blurry" in r:
            if not "num_too_blurry" in stats:
                stats["num_too_blurry"]=0
            stats["num_too_blurry"]+=1
        else:
            if not "num_fail" in stats:
                stats["num_fail"]=0
            stats["num_fail"]+=1
            print(f"LLM_attributes {i}: Fail with response {r}")
        return None

    r=r.replace('"','')
    r=r.replace("'","")
    r=r.replace('\n','')
    lines=r.split(",")
    mask=[]
    not_found=0
    for a in attrs_reduced:
        result=None
        for l in lines:
            if a in l:
                if "true" in l and "false" in l:
                    print(f"LLM_attributes {i} Weird line {l}")
                    stats["num_weird_line"]+=1
                elif not "true" in l and not "false" in l:
                    print(f"LLM_attributes {i} Weird line2 {l}")
                    stats["num_weird_line2"]+=1
                else:
                    if "true" in l:
                        result=1
                    else:
                        result=0
        if result is None:
            not_found+=1
            mask.append(0)
        else:
            mask.append(result)
    if not_found==len(attrs_reduced):
        print(f"LLM_attributes {i}: img total {not_found} of {len(attrs_reduced)} not found or error; LLM response was {r}")
        stats["num_fail"]+=1
        return None
    return mask

def generate_results(llm, attrs, out_temp, stats, attr_indexes):
    jpegs=[]

    # change e.g. 'person:has_beard to just 'has_beard'
    attrs_reduced=[a.split(":")[1] for a in attrs]

    for x in out_temp:
        jpegs.append(x["jpg"])
    responses=llm.generate_attributes(attrs_reduced, jpegs)

    stat_types=["num","num_fail","num_too_blurry","num_weird_line","num_weird_line2","num_attr_fail"]

    for s in stat_types:
        if not s in stats:
            stats[s]=0

    for i,r in enumerate(responses):
        mask=parse_response(r, stats, attrs_reduced, i=i)
        s=""
        gt_index=out_temp[i]["gt_index"]
        gt=out_temp[i]["gts"][gt_index]
        for c in gt["box"]:
            s+=dsu.sstr(c)

        if mask is None:
            k=stats['num_fail']+stats['num_too_blurry']
            #w=int(out_temp[i]['pix_w'])
            #h=int(out_temp[i]['pix_h'])
            #jpg_file=f"/tmp/fail{k:05d}_{w}x{h}.jpg"
            #with open(jpg_file, 'wb') as f:
            #    f.write(out_temp[i]["jpg"])
            #print(f"Written failed file to {jpg_file}")
            s+="FAILED"
        else:
            for k in range(len(mask)):
                gt["attrs"][attr_indexes[k]]=mask[k]

            for j,c in enumerate(mask):
                if c>0.5:
                    stats[attrs[j]]+=1
                    s+="1 "
                else:
                    s+="0 "
        s+="\n"
        filename=out_temp[i]["out_filename"]
        with open(filename, "a", encoding="utf-8") as myfile:
            myfile.write(s)

        del out_temp[i]["jpg"]

def check_attr_file(processor, attr_def_file, attrs):
    if os.path.isfile(attr_def_file):
        with open(attr_def_file, "r", encoding="utf-8") as f:
            read_attr=f.readlines()
        read_attr=[x.strip() for x in read_attr]
        if read_attr!=attrs:
            print(f"attribute mismatch {attrs} and {read_attr}")
            exit()
        processor.log(f"existing class attrs {attr_def_file} ok")
    else:
        processor.log(f"creating new attr cache {attr_def_file}")
        with open(attr_def_file, "w", encoding="utf-8") as f:
            for a in attrs:
                f.write(a+"\n")

def is_llm_attribute(attr_name):
    if attr_name.startswith("person:")==False:
        return False
    # FIQA attributes (face quality) done by separate NN
    if attr_name.startswith("person:fiqa"):
        return False
    # unused are 'slots' left for future use
    if attr_name.startswith("person:unused"):
        return False
    return True

def generate_person_attributes(processor, config):

    task=processor.task
    all_attrs=processor.attributes
    all_attrs_len=len(all_attrs)
    attrs=[a for a in all_attrs if is_llm_attribute(a)]
    attrs_len=len(attrs)
    attr_indexes=[all_attrs.index(a) for a in attrs]

    llm_model=dsu.get_param(task, config, "llm-model", "gpt-5-mini")
    system_prompt_file=dsu.get_param(task, config, "prompt", None)
    min_pose=dsu.get_param(task, config, "min_pose", None)
    output_path_base=dsu.get_param(task, config, "cache_path", None)
    loader_name=dsu.get_param(task, config, "loader", None)
    cache_folder=dsu.get_param(task, config, "cache_folder", processor.config_name)
    person_class=processor.get_class_index("person")

    out_path=output_path_base+"/"+cache_folder
    stuff.makedir(out_path)

    # if cache path specified, check attributes there match ones we are targetting
    check_attr_file(processor, out_path+"/attributes.txt", attrs)

    out_path=out_path+"/"+task
    stuff.makedir(out_path)

    # create inference engine
    llm=LLMGeneric(system_prompt_file, model=llm_model)
    batch=llm.get_batch()
    scale_width, scale_height=llm.get_max_size()

    out_temp=[]
    stats={"already_done":0, "already_done_failed":0}
    for a in attrs:
        stats[a]=0

    desc=processor.config_name+"/"+processor.task+"/attributes"
    for i in tqdm(range(processor.num_files),
                  desc=desc.ljust(45),
                  colour="#ffa500",
                  smoothing=0.001):

        img_path=processor.get_img_path(i)
        if not os.path.isfile(img_path):
            print(f"LLM_Attributes error: no file at {img_path}")
            continue

        gts=processor.get_gt(i)
        if gts is None:
            print(f"LLM_Attributes error: No GTs {img_path}")
            continue

        exif_data=dsu.image_get_exif_data(img_path)
        if "origin" not in exif_data:
            print("LLM_Attributes error: Could not process exif data from file "+img_path)
            continue

        out_filename=out_path+"/"+dsu.name_from_file(exif_data["origin"])+".txt"

        existing_boxes=[]
        existing_attr=[]
        if os.path.isfile(out_filename):
            with open(out_filename, 'r', encoding="utf-8") as f:
                for line in f:
                    if "FAILED" in line:
                        line=line[0:line.find("FAILED")]
                        vals=[float(x) for x in line.strip().split()]
                        assert len(vals)==4
                        box=vals[0:4]
                        existing_boxes.append(box)
                        existing_attr.append(None)
                    else:
                        vals=[float(x) for x in line.strip().split()]
                        assert len(vals)==4+len(attrs)
                        box=vals[0:4]
                        existing_boxes.append(box)
                        existing_attr.append(vals[4:])
                        for j,_ in enumerate(attrs):
                            stats[attrs[j]]+=vals[4+j]
        img=None
        for gt_index,g in enumerate(gts):
            if not "attrs" in g:
                g["attrs"]=[0]*all_attrs_len

            if g["class"] is not person_class:
                continue

            best_iou=0
            best_index=0
            min_iou=0.8
            for index,b in enumerate(existing_boxes):
                iou=stuff.box_iou(g["box"], b)+(1e-7)*index
                # in case there are multiple entries lets take ones that worked preferentially
                if existing_attr[index] is None:
                    iou=min_iou
                if iou>=best_iou:
                    best_iou=iou
                    best_index=index
            if best_iou>=min_iou:
                if existing_attr[best_index] is None:
                    stats["already_done_failed"]+=1
                    g["attrs"][attr_indexes[0]]=1
                else:
                    ex=existing_attr[best_index]
                    assert(len(ex)==attrs_len)
                    for k in range(len(ex)):
                        g["attrs"][attr_indexes[k]]=ex[k]
                    stats["already_done"]+=1
                continue

            g["attrs"][attr_indexes[0]]=0.95

            if min_pose is not None:
                # Pose gating: only consider detections with enough labelled pose points.
                if not stuff.has_pose_points(g):
                    continue
                n=0
                nface=0
                for ii in range(17):
                    if g["pose_points"][3*ii+2]!=0:
                        n+=1
                        if ii<5:
                            nface+=1
                if n<min_pose and nface<4:
                    continue

            if stuff.box_w(g["box"])>0.05 or stuff.box_h(g["box"])>0.12:
                if img is None:
                    img=cv2.imread(img_path)
                h, w, _= img.shape
                pix_w=w*stuff.box_w(g["box"])
                pix_h=h*stuff.box_h(g["box"])
                if pix_w<48 or pix_h<64:
                    continue
                jpg=dsu.object_jpeg(img, g, scale_width=scale_width, scale_height=scale_height, expand=1.05)

                r={ "index":i,
                    "gts":gts,
                    "path":img_path,
                    "gt_index":gt_index,
                    "jpg":jpg,
                    "pix_w":pix_w,
                    "pix_h":pix_h,
                    "out_filename":out_filename}
                out_temp.append(r)

        img=None
        processor.replace_annotations(i, gts)

        while len(out_temp)>=batch:
            generate_results(llm, attrs, out_temp[0:batch], stats, attr_indexes)
            already_written=[]
            # yuk, have to rewrite the GTs files after the async LLM stuff has finished
            for o in out_temp[0:batch]:
                if not o["index"] in already_written:
                    processor.replace_annotations(o["index"], o["gts"])
                    already_written.append(o["index"])
            for i in out_temp[0:batch]:
                del i
            out_temp=out_temp[batch:]

    generate_results(llm, attrs, out_temp, stats, attr_indexes)
    already_written=[]
    # yuk, have to rewrite the GTs files after the async LLM stuff has finished
    for o in out_temp:
        if not o["index"] in already_written:
            processor.replace_annotations(o["index"], o["gts"])
            already_written.append(o["index"])

    processor.reload_files()
    processor.log(f"Add attribute stats {stats}")
    processor.log(f"LLM stats {llm.get_stats()}")

def llm_test(model):

    test_path="/mldata/attribute_test"
    test=stuff.load_dictionary(test_path+"/attribute_test_v9.yaml")
    attributes=test["attributes"]
    files=test["files"]

    llm=LLMGeneric(test["prompt"], model=model)

    na=len(attributes)

    flip=True

    names=[]
    jpegs=[]
    attr_vec=[]
    scale_width, scale_height=llm.get_max_size()
    for f in files:
        jpeg_file=test_path+"/"+f+".jpg"
        if False:
            with open(jpeg_file, "rb") as fi:
                jpeg_bytes=fi.read()
                jpegs.append(jpeg_bytes)
        else:
            img=cv2.imread(jpeg_file)
            h,w,_=img.shape
            if True:#w>=scale_width or h>=scale_height:
                sf=min(scale_width/w, scale_height/h)
                img=cv2.resize(img, None, fx = sf, fy = sf, interpolation = cv2.INTER_CUBIC)
                _, encoded=cv2.imencode(".jpg", img)
                jpegs.append(encoded)
            else:
                with open(jpeg_file, "rb") as f1:
                    byte_array=f1.read()
                jpegs.append(np.frombuffer(byte_array, dtype=np.uint8))
            if flip:
                flipped_image = cv2.flip(img, 1)
                _, encoded=cv2.imencode(".jpg", flipped_image)
                jpegs.append(encoded)
        set_attrs=files[f]
        vec=[0]*na
        for a in set_attrs:
            conf=1.0
            if ";" in a:
                conf=float(a.split(";")[1])
                a=a.split(";")[0]
            if not a in attributes:
                print(f"ERROR attribute {a} not found")
                exit()
            index=attributes.index(a)
            vec[index]=conf
        attr_vec.append(vec)
        names.append(f)
        if flip:
            attr_vec.append(vec)
            names.append(f)

    stat_types=["num","num_fail","num_weird_line","num_weird_line2","num_attr_fail"]
    stats={}
    for s in stat_types:
        if not s in stats:
            stats[s]=0
    attrs_reduced=[a.split(":")[1] for a in attributes]

    response=llm.generate_attributes(attributes, jpegs)
    #print(response)

    FP=[0]*na
    FN=[0]*na
    TP=[0]*na
    TN=[0]*na
    P=[0]*na
    R=[0]*na
    F=[0]*na
    print(names)
    print(len(names))
    score_out=[]
    error_images=[""]*na
    images=[""]*len(names)

    for i,r in enumerate(response):
        img_score=0
        mismatch=""
        images[i]+=f"{names[i]}: "
        mask=parse_response(r, stats, attrs_reduced, i=names[i])
        if mask is None:
            print(f"parse_response failed for {names[i]:20s}")
            mask=[0]*na

        for j in range(na):
            d=abs(mask[j]-attr_vec[i][j])
            if mask[j]>=0.5:
                images[i]+=f"{attrs_reduced[j]},"
            img_score+=(d*d)
            if d>0.5:
                mismatch+=(f"\t\t{names[i]:20s} {attributes[j]:30s} GT={attr_vec[i][j]:3.1f} LLM={mask[j]:3.1f}\n")
                if mask[j]>attr_vec[i][j]:
                    c="+"
                else:
                    c="-"
                error_images[j]+=f"{c}{names[i]} "
            else:
                if mask[j]>0.5:
                    TP[j]+=1
                else:
                    TN[j]+=1
            if mask[j]>attr_vec[i][j]:
                FP[j]+=(d*d)
            else:
                FN[j]+=(d*d)
        score_out.append({"name":names[i], "score":img_score, "mismatch":mismatch})

    FN_tot=sum(FN)
    FP_tot=sum(FP)
    TP_tot=sum(TP)
    score=FN_tot+FP_tot

    eps=1e-7

    p=TP_tot/(TP_tot+FP_tot+eps)
    r=TP_tot/(TP_tot+FN_tot+eps)
    f=2*p*r/(p+r+eps)

    print("----Images------")
    for i in images:
        print(i)
    score_out=sorted(score_out, key=lambda x: x["score"], reverse=True)
    print("----WORST IMAGES-----")
    for i in range(min(5,len(score_out))):
        print(f"image {score_out[i]["name"]:20s} score {score_out[i]["score"]:5.2f}")
        print(score_out[i]["mismatch"])
    print("---------------------")
    for i in range(na):
        P[i]=TP[i]/(TP[i]+FP[i]+eps)
        R[i]=TP[i]/(TP[i]+FN[i]+eps)
        F[i]=2*P[i]*R[i]/(P[i]+R[i]+eps)
        print(f"{attrs_reduced[i]:36s} TP={TP[i]:4.1f} FP={FP[i]:4.1f} FN={FN[i]:4.1f} p={P[i]:0.3f} r={R[i]:0.3f} F={F[i]:0.3f} e={error_images[i]}")
    print("---------------------")
    print(f"{"Overall":20s} TP={TP_tot:4.1f}  FP={FP_tot:4.1f} FN={FN_tot:4.1f}  p={p:0.3f} r={r:0.3f} F={f:0.3f}")
    print(f" Score {score}")
    print(llm.get_stats())

# March 18 2026
# Overall              TP=361.0  FP=48.5 FN=100.0  p=0.882 r=0.783 F=0.829    GPT4.1-mini cost/Minf($)': '687.18
# Overall              TP=409.0  FP=43.7 FN=52.7  p=0.903 r=0.886 F=0.895     GPT-5-mini cost/Minf($)': '307.42'
# Overall              TP=381.0  FP=78.6 FN=80.2  p=0.829 r=0.826 F=0.828     GPT-5.4-mini cost/Minf($)': '872.94
# Overall              TP=395.0  FP=63.1 FN=64.7  p=0.862 r=0.859 F=0.861     GPT-5 cost/Minf($)': '2667.64
# Overall              TP=412.0  FP=37.4 FN=48.5  p=0.917 r=0.895 F=0.906     gemini-3.1-flash-lite cost/Minf($)': '700.36
# Overall              TP=428.0  FP=63.5 FN=33.5  p=0.871 r=0.927 F=0.898     gemini-3-flash cost/Minf($)': '1400.20
