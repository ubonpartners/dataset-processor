from ultralytics import YOLO
import argparse
import os
from tqdm import tqdm
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
import src.dataset_util as dsu
import src.dataset_processor as dp
from datetime import datetime, timedelta
import stuff
import logging

def fstr(x, cw=8):
    if x is None:
        return " "*cw
    x=float(x)
    if x<0.0001:
        return " "*cw
    if cw<8:
        r=f"{x:4.3f}"
    else:
        r=f"{x:5.4f}"
    r=r+" "*(cw-len(r))
    return r

def match_python(dets, gts, context):
    if context["type"]=="normal":
        det_matched, _=dsu.match_boxes(dets, gts, context["iou_thr"])
    elif context["type"]=="pose":
        det_matched, _=dsu.match_boxes_pose(dets, gts, context["iou_thr"])
    elif context["type"]=="face":
        det_matched, _=dsu.match_boxes_face(dets, gts, context["iou_thr"])
    elif context["type"]=="facepose":
        det_matched, _=dsu.match_boxes_facepose(dets, gts, context["iou_thr"])
    else:
        assert False, f"unknown context type {context['type']}"
    return det_matched

def match_upyc(dets, gts, context):
    import ubon_pycstuff.ubon_pycstuff as upyc
    if context["type"]=="normal":
        t=upyc.MATCH_TYPE_BOX_IOU
    elif context["type"]=="pose":
        t=upyc.MATCH_TYPE_POSE_KP
    elif context["type"]=="face":
        t=upyc.MATCH_TYPE_FACE_KP
    elif context["type"]=="facepose":
        t=upyc.MATCH_TYPE_FACEPOSE_KP
    else:
        assert False, f"unknown context type {context['type']}"
    det_matched=[-1]*len(dets)
    if len(dets)!=0 and len(gts)!=0:
        dm,gm=upyc.match_box_iou(dets, gts, context["iou_thr"], t)
        for i in range(len(dm)):
            det_matched[dm[i]]=gm[i]
    return det_matched

def match_dets_to_gts(ret, name, cl, dets, gts, match_function, match_fn_ctx, attr_base_class_index=None):
    det_matched=[]
    det_matched=match_function(dets, gts, match_fn_ctx)

    # attributes are treaded in map score as extra clases after the main classes

    for large in [False, True]:
        target_class=[]
        pred_class=[]
        conf=[]
        tp=[]
        fname=name
        if large:
            fname+="_large"
        if not fname in ret:
            ret[fname]={"target_class":[], "conf":[], "tp":[], "pred_class": []}

        for j,gt in enumerate(gts):
            if large==False or stuff.is_large(gt):
                target_class.append(cl)

                if attr_base_class_index is not None and "attrs" in gt:
                    for k,c in enumerate(gt["attrs"]):
                        if float(c)>0.5:
                            target_class.append(k+attr_base_class_index)

        for j,det in enumerate(dets):
            if large==False or stuff.is_large(det):
                pred_class.append(cl)
                conf.append(det['confidence'])
                dm=0 if det_matched[j]==-1 else 1
                tp.append(dm)
                if attr_base_class_index is not None and "attrs" in det:
                    matched_attrs=None
                    # get the attributes from the matching ground truth, if there is a match
                    if dm==1:
                        if "attrs" in gts[det_matched[j]]:
                            matched_attrs=gts[det_matched[j]]['attrs']
                        else:
                            print("Warning: matched gt has no attrs")
                    # now add the class,tp,conf for true for each attribute
                    for k,c in enumerate(det['attrs']):
                        if c>0.001:
                            pred_class.append(k+attr_base_class_index)
                            conf.append(c)
                            is_tp=0
                            if matched_attrs is not None and k<len(matched_attrs):
                                if float(matched_attrs[k])>0.5:
                                    is_tp=1
                            tp.append(is_tp)

        ret[fname]["target_class"]+=target_class
        ret[fname]["conf"]+=conf
        ret[fname]["tp"]+=tp
        ret[fname]["pred_class"]+=pred_class
    return ret

def compute_one_ap(dataset,
                   model_path,
                   yolo_model=None,
                   batch_size=32,
                   max_det=1200,
                   iou_thr=0.5,
                   nms_iou=0.6,
                   det_conf=0.001,
                   classes=None,
                   augment=False,
                   min_gt=500,
                   mpwq_context=None,
                   mpwq_progress_fn=None):

    upyc=None
    try:
        import ubon_pycstuff.ubon_pycstuff as upyc
    except:
        pass

    x=dp.DatasetProcessor(dataset, task="val", class_names=classes, face_kp=True, pose_kp=True, fold_attributes=True)
    nc=len(classes)
    if not yolo_model==None:
        x.set_object_detector(yolo_model, batch_size=batch_size, max_det=max_det, thr=det_conf)
    else:
        x.set_object_detector(model_path, batch_size=batch_size, max_det=max_det, thr=det_conf)

    measure_kp=True

    desc=x.dataset_name+" / "+stuff.infer_model_name(model_path)
    if augment:
        desc+=" / AUG"
    else:
        desc+=" / NOAUG"

    pbar=None
    if mpwq_progress_fn is None:
        pbar=tqdm(total=x.num_files,
                  desc=desc,
                  smoothing=0.8)
    else:
        mpwq_progress_fn(mpwq_context, desc=desc, total=x.num_files)

    match_stats={}
    if upyc is None:
        matcher=match_python
    else:
        matcher=match_upyc

    for i in range(x.num_files):
        if pbar is not None:
            pbar.update(1)
        else:
            mpwq_progress_fn(mpwq_context, update=1)

        all_gts=x.get_gt(index=i)
        all_dets=x.get_detections(index=i)
        if all_gts==None or all_dets==None:
            continue

        dets_by_class=[[] for _ in range(nc)]
        gts_by_class=[[] for _ in range(nc)]
        for d in all_dets:
            dets_by_class[d["class"]].append(d)
        for gt in all_gts:
            gts_by_class[gt["class"]].append(gt)

        for cl in range(nc):
            dets=dets_by_class[cl]
            gts=gts_by_class[cl]

            # fixme: hack; assumes all attributes are for person class and are after the main classes in the class list
            attr_base_index=None
            if cl==x.get_class_index("person"):
                attr_base_index=next((i for i, s in enumerate(classes) if s.startswith(("person_", "person:"))), None)

            match_dets_to_gts(match_stats, "normal", cl, dets, gts, matcher, {"iou_thr":iou_thr, "type":"normal"}, attr_base_index)
            if measure_kp and (cl==x.get_class_index("person")):
                # measuring pose point mAP from pose or facepose detection points on person class
                gts_f=[g for g in gts if stuff.has_pose_points(g)]
                dets_f=[d for d in dets if stuff.has_pose_points(d)]
                match_dets_to_gts(match_stats, "kp", cl, dets_f, gts_f, matcher, {"iou_thr":iou_thr, "type":"pose"})
            if measure_kp and (cl==x.get_class_index("person")):
                # measuring face point mAP from facepose detection points on person class
                gts_f=[g for g in gts if stuff.has_face_points(g)]
                dets_f=[d for d in dets if stuff.has_face_points(d)]
                match_dets_to_gts(match_stats, "kp_face", cl, dets_f, gts_f, matcher, {"iou_thr":iou_thr, "type":"facepose"})
            if measure_kp and (cl==x.get_class_index("face")):
                # measuring face point mAP from face detection points on face class
                gts_f=[g for g in gts if stuff.has_face_points(g)]
                dets_f=[d for d in dets if stuff.has_face_points(d)]
                match_dets_to_gts(match_stats, "kp", cl, dets_f, gts_f, matcher, {"iou_thr":iou_thr, "type":"face"})

    for m in match_stats:
        s=match_stats[m]
        ap, p, r =stuff.ap_calc(s["conf"], s["tp"], s["pred_class"], s["target_class"], nc, min_gt=min_gt)
        match_stats[m]["ap"]=ap
        match_stats[m]["p"]=p
        match_stats[m]["r"]=r

    augstr="+aug" if augment else ""
    result={"model":stuff.infer_model_name(model_path),
            "dataset": x.dataset_name,
            "datasetaug": x.dataset_name+augstr,
            "augment": augment,
            "time": datetime.now(),
            "params":x.od_num_params/1000000.0,
            "flops":x.od_num_flops}

    for i,_ in enumerate(classes):
        c=classes[i]
        for j in match_stats:
            m=match_stats[j]
            if j=="normal":
                postfix=""
            elif j=="normal_large":
                postfix="_large"
            elif j=="kp":
                postfix="_kp"
            elif j=="kp_large":
                postfix="_kp_large"
            elif j=="kp_face":
                postfix="_kp_face"
            elif j=="kp_face_large":
                postfix="_kp_face_large"
            else:
                assert False, f"unknown match stats {j}"
            result[c+"_ap50"+postfix]=float(m['ap'][i])
            result[c+"-p"+postfix]=float(m['p'][i])
            result[c+"-r"+postfix]=float(m['r'][i])

    return result

def compare(scorea, scoreb):
    #return scorea['person_ap50']-scoreb['person_ap50']
    sa=scorea['person_ap50']
    sb=scoreb['person_ap50']
    if sa>sb:
        return 1
    elif sb>sa:
        return -1
    else:
        return 0

def get_avg_scores(results, model, param):
    t=1.0
    n=0
    for r in results:
        if r["model"]==model:
            if param in r:
                if isinstance(r[param], int) or isinstance(r[param], float):
                    if r[param]>0.001:
                        t*=r[param]
                        n=n+1
    if n>0:
        t=pow(t, 1.0/n)
        return t
    else:
        return 0

def add_averages(results):
    datasets = sorted(list(set(entry['dataset'] for entry in results)), reverse=True)

    models = list(set(entry['model'] for entry in results))
    paramset=set([])
    for r in results:
        paramset=paramset.union(set(r.keys()))

    params = list(paramset)
    params.remove("model")
    params.remove("dataset")
    params.remove("datasetaug")
    params.remove("augment")
    params.remove("time")

    results2=[]
    if len(datasets)>1:
        for m in models:
            e={"model":m, "dataset":"_geomean", "datasetaug":"_geomean", "augment":False, "time":datetime.now()}
            for p in params:
                e[p]=get_avg_scores(results, m, p)
            results2.append(e)
    results+=results2

def add_mean_columns(results, config):
    if "mean_columns" in config:
        for mc in config["mean_columns"]:
            cols=config["mean_columns"][mc]
            for r in results:
                tot=0
                n=0
                for c in cols:
                    if c in r:
                        tot+=r[c]
                        n+=1
                if n>0:
                    tot/=n
                r[mc]=tot

def show_results(results, columns, column_text, sort_key, cw=8):
    unique_datasets = list(dict.fromkeys(r["dataset"] for r in results))

    def sort_fn(r):
        return unique_datasets.index(r['dataset']) *1000 + r.get(sort_key, 0) #r[sort_key]
    stuff.show_data(results, ["dataset","model"]+columns, ["Dataset","Model"]+column_text, sort_fn)

def map_subprocess(work_item, mpwq_context, mpwq_progress_fn=None):
    result=compute_one_ap(work_item["dataset"],
                          work_item["model"],
                          classes=work_item["classes"],
                          augment=work_item["augment"],
                          min_gt=work_item["min_gt"],
                          batch_size=work_item["batch_size"],
                          mpwq_context=mpwq_context,
                          mpwq_progress_fn=mpwq_progress_fn
                          )
    return result

def on_result_callback(context, result):
    context["results"].append(result)
    context["cached_results"].append(result)
    add_mean_columns(context["results"], context["config"])
    show_results(context["results"], context["columns"], context["column_txt"], context["sort_key"], cw=context["cw"])
    logging.info("Writing "+str(len(context["cached_results"]))+" cached results")
    stuff.save_atomic_pickle(context["cached_results"], context["resultfile"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging', type=str, default='info', help="Logging config: level[:console|file]")
    parser.add_argument('--config', type=str, default='data/map_ultralytics.json', help='config file to use (json or yml)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--delete-dataset', type=str, default=None, help='delete a dataset from cached results')
    parser.add_argument('--delete-model', type=str, default=None, help='delete a model from cached results')
    opt = parser.parse_args()
    stuff.configure_root_logger(opt.logging)
    config = stuff.load_dictionary(opt.config)

    start_time=time.time()

    result_location=config["results_location"]
    resultfile=config["results_cache_file"]
    classes=config["classes"]
    datasets=config["datasets"]
    models=config["models"]
    regen_models=config["regen_models"]
    regen_datasets=config["regen_datasets"]
    max_age_days=config["max_age_days"]
    columns=config["columns"]
    sort_key=config["sort_key"]
    chart_list=config["chart_list"]
    cw=8
    if "cw" in config:
        cw=config["cw"]
    min_gt=500
    if "min_gt" in config:
        min_gt=config["min_gt"]
    workers=4
    if "workers" in config:
        workers=config["workers"]

    batch_size=config.get("batch_size", opt.batch_size)
    if stuff.platform_stuff.is_jetson():
        if batch_size>8:
            logging.info("Limiting batch size to 8 as Jetson")
            batch_size=8
    logging.info(f"Batch size is {batch_size}")
    cached_results=[]
    if os.path.isfile(resultfile):
        with open(resultfile, 'rb') as handle:
            cached_results = pickle.load(handle)

    augs=[False]

    column_txt=[]
    for i,c in enumerate(columns):
        if "," in c:
            columns[i]=c.split(",")[0]
            ctxt=c.split(",")[1]
            ctxt=ctxt.replace("\\n","\n")
            column_txt.append(ctxt)
            continue
        txt=c.replace("_","\n")
        for cl in classes:
            dcl=cl
            if "attr_" in dcl:
                dcl=dcl[dcl.find("attr_")+len("attr_"):]
            if c==cl+'_ap50':
                txt=dcl+"\n"+"AP50"
            elif c==cl+'-p':
                txt=dcl+"\n"+"Prec"
            elif c==cl+'-r':
                txt=dcl+"\n"+"Rec"
            elif c==cl+'_ap50_kp':
                if cl=='person':
                    txt="Pose KP\nAP50"
                elif cl=='face':
                    txt="Face KP\nAP50"
                else:
                    txt=cl+"\n"+"KP"
            elif c==cl+'_ap50_large':
                txt=dcl+"\nAP50 L"
            elif c==cl+'_ap50_kp_large':
                if c=='person':
                    txt="Pose KP\nAP L"
                elif c=='face':
                    txt="Face KP\nAP L"
                else:
                    txt=cl+"\n"+"KP L"
        column_txt.append(txt[0].upper()+txt[1:])

    for r in cached_results:
        if not "augment" in r:
            r["augment"]=False
        if not "datasetaug" in r:
            if r["augment"] is True:
                r["datasetaug"]=r["dataset"]+"+aug"
            else:
                r["datasetaug"]=r["dataset"]
        if not "inference" in r:
            r["inference"]="pytorch"
        if not "precision" in r:
            r["precision"]="fp16"
        if not "time" in r:
            r["time"]=datetime.now()
        if not "classes" in r:
            r["classes"]=classes

    if opt.delete_dataset is not None or opt.delete_model is not None:
        results_out=[]
        for r in cached_results:
            if r["dataset"]==opt.delete_dataset or r["model"]==opt.delete_model:
                logging.info(f"Delete {r}")
            else:
                results_out.append(r)
        logging.info("Writing "+str(len(cached_results))+" cached results")
        stuff.save_atomic_pickle(results_out, resultfile)
        exit()

    results=[]
    #for r in cached_results:
    #    print(f"{r['dataset']} {r['model']} {r['augment']} {r['datasetaug']}")

    min_time=datetime.now() - timedelta(days=max_age_days)

    results_to_generate=[]

    #for model in models:
    for dataset in datasets:
        #for dataset in datasets:
        for model in models:
            for augment in augs:
                dataset_name=dsu.get_dataset_name(dataset)
                model_name=stuff.infer_model_name(model)

                cached_result_to_use=None
                for r in cached_results:
                    if r["model"]==model_name and r["dataset"]==dataset_name and r["augment"]==augment and set(classes)<=set(r["classes"]):
                        if cached_result_to_use==None or r["time"]>cached_result_to_use["time"]:
                            if r["time"]>min_time:
                                cached_result_to_use=r

                if model_name in regen_models or dataset_name in regen_datasets:
                    cached_result_to_use=None

                if cached_result_to_use is not None:
                    results.append(cached_result_to_use)
                else:
                    work={"dataset":dataset,
                          "model":model,
                          "classes":classes,
                          "augment":augment,
                          "min_gt":min_gt,
                          "batch_size":batch_size}
                    results_to_generate.append(work)


    logging.info(f"Retrieved {len(results)} results from cache {resultfile}")
    logging.info(f"Need to generate {len(results_to_generate)} new results")

    on_result_context={"results":results,
                       "cached_results":cached_results,
                       "config":config,
                       "columns":columns,
                       "column_txt":column_txt,
                       "sort_key":sort_key,
                       "cw":cw,
                       "resultfile":resultfile}

    logging.info(f"Using {workers} multiprocessing workers")
    _ = stuff.mp_workqueue_run(results_to_generate,
                               map_subprocess,
                               desc="MAP calculation",
                               min_update_interval=1,
                               num_workers=workers,
                               result_callback_context=on_result_context,
                               result_callback=on_result_callback)

    add_mean_columns(results, config)
    add_averages(results)
    show_results(results, columns, column_txt, sort_key, cw=cw)

    elapsed=time.time()-start_time
    print(f"All done; took {dsu.timestr(elapsed)}")
