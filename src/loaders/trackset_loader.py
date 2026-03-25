
import os
import logging
import cv2
import stuff
import tempfile
import src.dataset_util as dsu
from PIL import Image

class TracksetLoader:
    def __init__(self,
                 task="val", class_names=["face","person"],
                 ds_params=None):

        self.info="fix me"
        self.task=task
        self.tmpdir = tempfile.mkdtemp()
        self.class_names=class_names
        self.face_class_index=self.class_names.index("face") if "face" in self.class_names else None
        self.person_class_index=self.class_names.index("person") if "person" in self.class_names else None
        self.vehicle_class_index=self.class_names.index("vehicle") if "vehicle" in self.class_names else None
        self.current_video=None
        self.subsample=2
        if "subsample" in ds_params:
            self.subsample=ds_params["subsample"]
        self.scale=None
        if "scale" in ds_params:
            self.scale=ds_params["scale"]
        self.augment=None
        if "augment" in ds_params:
            self.augment=ds_params["augment"]
        annotation_paths=ds_params["annotation_paths"]
        annotations=[]
        for path in annotation_paths:
            files=os.listdir(path)
            for f in files:
                if f.endswith(".json"):
                    an=path+"/"+f
                    annotations.append(an)
        logging.info(f"Found {len(annotations)} annotation files; ss={self.subsample} scale={self.scale} augment={self.augment}")

        self.frames=[]
        num_gts=0
        if self.task=="val":
            for an in annotations:
                d=stuff.load_dictionary(an)
                this_an_classes=d["metadata"]["classes"]
                class_mapping=[None]*len(this_an_classes)
                if "person" in this_an_classes:
                    class_mapping[this_an_classes.index("person")]=self.person_class_index
                if "face" in this_an_classes:
                    class_mapping[this_an_classes.index("face")]=self.face_class_index
                if "vehicle" in this_an_classes:
                    class_mapping[this_an_classes.index("vehicle")]=self.vehicle_class_index
                name=dsu.name_from_file(an)
                name.replace(".","_")
                for f in d["frames"]:
                    frame={}
                    frame["video"]=d["metadata"]["original_video"]
                    frame["time"]=f["frame_time"]
                    frame_name=name+"_"+f"{f['frame_time']:.2f}"
                    frame_name=frame_name.replace(".","_")
                    frame["name"]=frame_name
                    out_gts=[]
                    for o in f["objects"]:
                        gt=f["objects"][o]
                        cl=class_mapping[gt["class"]]
                        if cl is None:
                            continue
                        out_gts.append({"box":gt["box"], "class":cl, "confidence":gt["conf"]})
                        num_gts+=1
                    frame["gts"]=out_gts
                    self.frames.append(frame)
        prev_len=len(self.frames)
        self.frames=self.frames[::self.subsample]
        logging.info(f"Loaded {prev_len} frames {num_gts} total GTs; subsampled to {len(self.frames)} frames")


    def __del__(self):
        stuff.rmdir(self.tmpdir)

    def get_info(self):
        return "trackset :"+str(self.info)

    def add_category_mapping(self, source_class, dest_class):
        self.face_class_index=self.class_names.index("face")
        self.person_class_index=self.class_names.index("person")

    def get_annotations(self, img_id):
        return self.frames[img_id]["gts"]

    def get_image_ids(self):
        return range(len(self.frames))

    def get_img_path(self, img_id):
        video=self.frames[img_id]["video"]
        time=self.frames[img_id]["time"]
        if video!=self.current_video:
            self.current_video=video
            self.video_loader=stuff.RandomAccessVideoReader(video)
        assert self.video_loader is not None, f"Could not create video reader for {video}"
        frame,_=self.video_loader.get_frame_at_time(time)
        if frame is None:
            return None

        if self.scale is not None:
            h, w = frame.shape[:2]
            max_w, max_h=self.scale
            # Compute scale factor and clamp to 1.0 (no up-scaling)
            scale = min(max_w / w, max_h / h, 1.0)
            # Compute new size
            new_w, new_h = int(w * scale), int(h * scale)
            frame=cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if self.augment is not None:
            if "yuv420" in self.augment:
                frame=stuff.bt709_yuv420_augment_single(frame)
            else:
                assert False, f"unknown augment {self.augment}"

        filename=self.tmpdir+"/"+self.frames[img_id]["name"]+".jpg"
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # lets be very careful to have no chroma downsamping here!
        Image.fromarray(frame).save(filename, subsampling=0, quality=95)

        return filename

    def get_category_maps(self):
        return { 'person':['person'],
                 'vehicle':['vehicle'],
                 'animal':[],
                 'ignore':[],
                 'face':['face']}