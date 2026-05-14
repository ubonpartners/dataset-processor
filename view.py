from src.dataset_processor import DatasetProcessor
import src.dataset_util as dsu
import stuff
import argparse
import random
import cv2
from tqdm import tqdm
import numpy as np
import time
import ultralytics

def show_results_grid(results, search_str):
    w=1280
    h=720

    img = np.zeros((h, w, 3), np.uint8)

    for i in range(15):
        if i>=len(results):
            continue
        posx=(i%5)*220
        posy=30+(200 * (i//5))
        img_person=results[i]["img"]
        ph, pw, _ = img_person.shape
        assert posy+ph<=h
        assert posx+pw<=w
        img[posy:(posy+ph), posx:(posx+pw)]=img_person[0:ph, 0:pw]

    cv2.putText(img,
                search_str,
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255,255,255),
                2,
                2)

    cv2.imshow("results", img)
    cv2.waitKey(20)

def do_search(processor, search_str):

    attr_mask=[0]*len(processor.attributes)
    for i,a in enumerate(processor.attributes):
        if a in search_str:
            attr_mask[i]=1.0

    min_score=0.25
    num_results=20
    added=0
    results=[]

    last_show=time.time()
    for i in tqdm(range(processor.num_files)):
        img=None
        dets=processor.get_detections(i)
        for d in dets:
            if "attrs" in d:
                score=0
                for j in range(len(d["attrs"])):
                    score+=d["attrs"][j]*attr_mask[j]
                if score>min_score:
                    if img is None:
                        img=processor.get_img(i)
                    person_img=dsu.object_img(img, d, scale_height=160, expand=1.2, no_enlarge=True)
                    results.append({"img":person_img, "score":score})
                    added+=1
        if added!=0:
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            results=results[0:num_results]
            min_score=results[len(results)-1]["score"]
            added=0

        now=time.time()
        if now>last_show+5:
            last_show=now
            show_results_grid(results, search_str)

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    results=results[0:num_results]

    show_results_grid(results, search_str)
    cv2.waitKey(0)

def do_video(video, model, output_path=None, display=True):

    yolo=ultralytics.YOLO(model, verbose=False)
    class_names=[yolo.names[i] for i in range(len(yolo.names))]

    if video=="webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot access the webcam")
            exit()
        width=1280
        height=720
        fps=30
        frame_count=0
        # Step 2: Set video properties (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Set frame width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # Set frame height
        cap.set(cv2.CAP_PROP_FPS, fps)  # Set FPS (frames per second)
    else:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"Error: Cannot open the video file {video}")
            exit()

        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

    print(f"Video {width}x{height} {fps}fps {frame_count} frames")

    attributes=[]
    for c in class_names:
        if c.startswith("person_"):
            attributes.append("person:"+c[len("person_"):])

    paused=False

    display_width=1600
    display_height=900
    highlight_pos=None
    if display:
        display=stuff.Display(width=display_width, height=display_height, output=output_path)

    done=False
    while done is False:
        if paused==False:
            ret, frame = cap.read()  # Read a frame from the video

        result=yolo(frame, conf=0.2, max_det=500, half=True, iou=0.45)
        #print(result)

        if not ret:
            break  # Break if no frame is read (end of video)

        out_det=stuff.yolo_results_to_dets(result[0],
                                         det_thr=0.4,
                                         yolo_class_names=class_names,
                                         class_names=class_names,
                                         attributes=attributes,
                                         face_kp=True,
                                         pose_kp=True,
                                         fold_attributes=True)
        display.clear()

        highlight_index=None
        if highlight_pos is not None:
             highlight_index, dist1=stuff.find_gt_from_point(out_det, highlight_pos[0], highlight_pos[1])
        stuff.draw_boxes(display,
                         out_det,
                         attributes=attributes,
                         highlight_index=highlight_index,
                         class_names=class_names)

        if display:
            display.show(frame, title="results")
            events=display.get_events(10)
            for e in events:
                if e['key']=='p':
                    paused=not paused
                if e['key']=='q':
                    done=True
                if e['lbutton']:
                    highlight_pos=[e['x'], e['y']]
    print("Quitting!")
    display.close()

def do_img(image_file, model, display=True, infer_path="yolo"):

    infer_path=infer_path.lower()
    assert infer_path in ["yolo", "wrapper", "inference_wrapper"], f"Unsupported infer_path {infer_path}"

    yolo=None
    infer=None
    if infer_path=="yolo":
        yolo=ultralytics.YOLO(model)
        class_names=[yolo.names[i] for i in range(len(yolo.names))]
        attributes=[]
        for c in class_names:
            if c.startswith("person_"):
                attributes.append("person:"+c[len("person_"):])
    else:
        infer=stuff.inference_wrapper(model,
                                      thr=0.05,
                                      nms_thr=0.45,
                                      half=True,
                                      max_det=500,
                                      face_kp=True,
                                      pose_kp=True,
                                      fold_attributes=True)
        class_names=infer.class_names
        attributes=infer.attributes if infer.attributes is not None else []

    display_width=1600
    display_height=900
    highlight_pos=None
    if display:
        display=stuff.Display(width=display_width, height=display_height)

    while True:
        frame = cv2.imread(image_file)  # Read image
        if infer_path=="yolo":
            result=yolo(image_file, conf=0.1, half=True, max_det=500, iou=0.45)
            out_det=stuff.yolo_results_to_dets(result[0],
                                             det_thr=0.05,
                                             yolo_class_names=class_names,
                                             class_names=class_names,
                                             attributes=attributes,
                                             face_kp=True,
                                             pose_kp=True,
                                             fold_attributes=True)
        else:
            # Use the decoded frame path here (instead of a JPEG filename) to keep
            # preprocessing consistent with the YOLO path and avoid confidence drift.
            out_det=infer.infer([frame])[0]
        display.clear()

        highlight_index=None
        if highlight_pos is not None:
             highlight_index, dist1=stuff.find_gt_from_point(out_det, highlight_pos[0], highlight_pos[1])
        stuff.draw_boxes(display,
                         out_det,
                         attributes=attributes,
                         highlight_index=highlight_index,
                         class_names=class_names)

        if display:
            display.show(frame, title="results")
            events=display.get_events(1000)
            for e in events:
                if e['key']=='p':
                    paused=not paused
                if e['lbutton']:
                    highlight_pos=[e['x'], e['y']]

def view(x, class_filter=None, model=None):
    print(f"{x.num_files} files in dataset")
    print(f"{x.get_class_names()} -classes")
    print(f"{x.attributes}")
    i=0

    display=stuff.Display(width=1280, height=720)

    show_gts=True
    show_dets=False
    find_mode=False
    thr=0.4
    gt_highlight_index=None
    det_highlight_index=None
    jpg_num=0
    last_i=-1
    while True:
        display.clear()
        gts=x.get_gt(i)
        if i!=last_i:
            gt_highlight_index=None
            det_highlight_index=None
        if class_filter!=None:
            j=x.class_names.index(class_filter)
            present=False
            for g in gts:
                if g["class"]==j:
                    present=True
            if present==False:
                continue
        img=x.get_img(i)
        if img is None:
            i=0
            continue
        display_img=img
        display.show(display_img, title="View "+x.get_gt_file_path(i))
        if show_gts:
            stuff.draw_boxes(display,
                             gts,
                             class_names=x.get_class_names(),
                             attributes=x.attributes,
                             highlight_index=gt_highlight_index)
            if find_mode:
                found=False
                attr_index=0
                attributes=x.attributes
                attr="person:has_visible_tattoos"
                if attr in attributes:
                    attr_index=attributes.index(attr)

                for g in gts:
                    if "attrs" in g:
                        if g["attrs"][attr_index]>0:
                            found=True
                if found is False:
                    i=i+1
                    continue

            if False:
                for gt in gts:
                    if gt["class"]!=0:
                        continue
                    jpg_file=f"/mldata/output_datasets/attribute_test/person{jpg_num:05d}.jpg"
                    jpg_num+=1
                    jpg=dsu.object_jpeg(img, gt)
                    with open(jpg_file, 'wb') as f:
                        f.write(jpg.tobytes())
        if show_dets and model!=None:
            dets=x.get_detections(i)
            dets=[d for d in dets if d["confidence"]>thr]
            if i!=last_i:
                for k,d in enumerate(dets):
                    print(f" {k:3d}: conf {d['confidence']:0.3f} {d['box']}")
            #print(dets)
            #print("-------")
            if find_mode:
                found=False
                max_thr=0
                for g in dets:
                    max_thr=max(max_thr, d["confidence"])
                if max_thr<0.9:
                    i=i+1
                    print(f"Find {i} {max_thr}...")
                    continue

            stuff.draw_boxes(display,
                             dets,
                             class_names=x.get_class_names(),
                             alt_clr=True,
                             attributes=x.attributes,
                             highlight_index=det_highlight_index)

        display.show(display_img, title="View "+x.get_gt_file_path(i))
        last_i=i
        while True:
            events=display.get_events(10)
            for e in events:
                if e['key']=='g':
                    show_gts=not show_gts
                if e['key']=='d':
                    show_dets=not show_dets
                if e['key']=='f':
                    find_mode=not find_mode
                if e['key']=='r':
                    i=random.randint(0, x.num_files)
                if e['key']==' ':
                    i=i+1
                if e['key']=='-':
                    i=i-1
                if e['lbutton']:
                    new_gt_highlight_index=None
                    dist1=1000000
                    new_det_highlight_index=None
                    dist2=1000000
                    if show_gts:
                        new_gt_highlight_index, dist1=stuff.find_gt_from_point(gts, e['x'], e['y'])
                    if show_dets:
                        new_det_highlight_index, dist2=stuff.find_gt_from_point(dets, e['x'], e['y'])
                    if dist1>dist2:
                        new_gt_highlight_index=None
                    if dist2>dist1:
                        new_det_highlight_index=None
                    if new_gt_highlight_index==gt_highlight_index:
                        gt_highlight_index=None
                    else:
                        gt_highlight_index=new_gt_highlight_index
                    if new_det_highlight_index==det_highlight_index:
                        det_highlight_index=None
                    else:
                        det_highlight_index=new_det_highlight_index
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='view.py')
    parser.add_argument('--dataset', type=str, default="/mldata/v10/all/dataset.yaml", help='dataset to use')
    parser.add_argument('--model', type=str, default='/mldata/models/v11/engine/yolo26l-e2e-v11r-260426-qat-int8.engine', help='model to use')
    parser.add_argument('--task', type=str, default="val", help='val, train, or both')
    parser.add_argument('--search', type=str, default=None, help='search attributes to display matching crops')
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--img-infer-path',
                        type=str,
                        default="yolo",
                        choices=["yolo", "wrapper", "inference_wrapper"],
                        help='inference path for --img: yolo or stuff.inference_wrapper')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    opt = parser.parse_args()

    if False:
        model=ultralytics.YOLO("/mldata/weights/yolo11l-attributes-081224.pt")
        model.export(format="tflite")
        exit()

    if False:
        #do_video("/mldata/video/bc2.mp4", opt.model, "/mldata/results/video/bc2_081224.mp4", display=False)
        #do_video("/mldata/video/mall_escalators.264", opt.model, "/mldata/results/video/mall_081224.mp4", display=False)
        #do_video("/mldata/video/operahouse.264", opt.model, "/mldata/results/video/operahouse_081224.mp4", display=False)
        #do_video("/mldata/video/bourne.264", opt.model, "/mldata/results/video/bourne_081224.mp4", display=False)
        do_video("/mldata/video/cam22.h264", opt.model, "/mldata/results/video/cam22_081224.mp4", display=False)
        exit()

    if opt.img!=None:
        do_img(opt.img, opt.model, infer_path=opt.img_infer_path)
        exit()

    if opt.video!=None:
        do_video(opt.video, opt.model, output_path=opt.output, fold_attributes=True)
        exit()

    x=DatasetProcessor(opt.dataset, task=opt.task, fold_attributes=True)
    if opt.model!=None:
        x.set_object_detector(opt.model, imgsz=640, thr=0.01, batch_size=opt.batch_size, get_feats=True)

    if opt.search!=None:
        do_search(x, opt.search)
        exit()

    view(x, model=opt.model)
    exit()
