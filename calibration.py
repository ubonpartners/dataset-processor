import src.dataset_processor as dp
import src.dataset_util as dsu
import argparse
import src.ultralytics_ap as uap
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def compute_one_ap(dataset, model_path, yolo_model=None, batch_size=32, max_det=600, iou_thr=0.5, nms_iou=0.6, det_conf=0.001, classes=None, augment=False, min_gt=500):
    x=dp.DatasetProcessor(dataset, task="val", class_names=classes)
    nc=len(classes)
    x.set_object_detector(model_path, batch_size=batch_size)

    target_class=[]
    conf=[]
    tp=[]
    pred_class=[]

    desc=x.dataset_name+" / "+dsu.name_from_file(model_path)

    for i in tqdm(range(x.num_files), desc=desc, smoothing=0.01):
        gts=x.get_gt(index=i)
        dets=x.get_detections(index=i,det_thr=det_conf)
        if gts==None or dets==None:
            continue

        det_matched, _=dsu.match_boxes(dets, gts, iou_thr)

        for j,_ in enumerate(gts):
            target_class.append(gts[j]['class'])

        for j,_ in enumerate(dets):
            pred_class.append(dets[j]['class'])
            conf.append(dets[j]['confidence'])
            tp.append(0 if det_matched[j]==-1 else 1)

    ap, p, r, p_curve, r_curve = uap.ap_calc(conf, tp, pred_class, target_class, nc, min_gt=min_gt, pr_curves=True)
    return p_curve, r_curve

def calibration(models, dataset):
    classes=["person"]

    # Create the plot
    plt.figure(figsize=(10, 6))

    for m in models:
        p_curve, r_curve=compute_one_ap(dataset, m, classes=classes)
        n=dsu.name_from_file(m)
        plt.plot(p_curve[0], label='Precision '+n)
        plt.plot(r_curve[0], label='Recall '+n)

    # Add labels, title, and legend
    plt.xlabel('Index')
    plt.ylabel('Score')
    plt.title('Precision, Recall Curves')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='process_dataset.py')
    parser.add_argument('--dataset', type=str, default="/mldata/v5-attr/coco-v5-attr/dataset.yaml",
                         help='Dataset to use for quantization')
    parser.add_argument('--model1',  type=str, default="/mldata/weights/yolo11l-dpa-131224.pt", help='model to use')
    parser.add_argument('--model2',  type=str, default="/mldata/weights/yolo11l-dpa-131224_int8.engine", help='model to use')
    opt = parser.parse_args()
    calibration([opt.model1, opt.model2], opt.dataset)