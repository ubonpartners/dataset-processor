import glob
import os
import cv2
import stuff
import argparse
import cv2

def load_detections_from_csv(csv_path):
    csv_files = glob.glob(os.path.join(csv_path, "*.txt"))
    res={}
    max_x=0
    max_y=0
    max_conf=0
    histo=[0]*200
    for file in csv_files:
        name=os.path.splitext(os.path.basename(file))[0]
        with open(file, "r") as f:
            dets=[]
            for line in f:
                row=[float(x) for x in line.strip().split(",")]
                box=row[0:4]
                b=[stuff.clip01(x) for x in box]
                max_x=max(max_x, box[2])
                max_y=max(max_y, box[3])
                det={"box":b, "class":int(row[4]), "confidence":row[5]}
                index=int(row[5]*100)
                histo[index]+=1
                max_conf=max(max_conf, row[5])
                dets.append(det)
        res[name]=dets
    print("Max ",max_x,max_y,max_conf)
    for i in range(200):
        print(f"{i:3d} {histo[i]:4d}")
    return res

def visualise(res, images):
    display=stuff.Display(width=1280, height=720)

    for r in res:
        image_file=images+"/"+r+".jpg"
        print(image_file)
        frame = cv2.imread(image_file)  # Read image
        display.clear()
        dets=res[r]
        dets_filtered=[d for d in dets if d["confidence"]>0.5 and d["class"]==0]
        #for d in dets_filtered:
        #    print(d["box"], d["class"], d["confidence"])
        display.show(frame, title="results")
        stuff.draw_boxes(display,
                         dets_filtered,
                         class_names=["person","face","vehicle","animal","weapon"])
        done=False
        while(done==False):
            display.show(frame, title="results")
            events=display.get_events(50)
            for e in events:
                if e['key']==' ':
                    done=True
                if e['key']=='q':
                    exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='map_csv.py')
    parser.add_argument('--images', type=str, default="/mldata/v6-attr/widerface/val/images", help='dataset to use')
    parser.add_argument('--csv-path', type=str, default='/home/mark/Downloads/out4', help='path to csv files')
    opt = parser.parse_args()
    dets=load_detections_from_csv(opt.csv_path)
    visualise(dets,opt.images)
