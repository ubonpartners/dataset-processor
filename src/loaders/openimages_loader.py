import src.dataset_util as du
import os
import stuff
import pandas as pd

# This code assumes you have already downloaded the OpenImages dataset
# and labels
#
# ls /mldata/downloaded_datasets/openimages/
#test-annotations-bbox.csv                      train
#oidv6-train-annotations-bbox.csv               train-images-boxable-with-rotation.csv
#oidv7-class-descriptions-boxable.csv           validation
#oidv7-train-annotations-human-imagelabels.csv  validation-annotations-bbox.csv
#oidv7-val-annotations-machine-imagelabels.csv  validation-annotations-human-imagelabels-boxable.csv
#test                                           validation-images-with-rotation.csv


class OpenImagesLoader:
    def __init__(self,
                 oi_path="/mldata/downloaded_datasets/openimages",
                 task="val", class_names=["face","person"], ds_params=None):

        gdrive_sync=ds_params.get("gdrive_sync", True)
        if gdrive_sync:
            stuff.gdrive_sync_mldata(oi_path, direction="from_drive")

        self.task=task
        self.class_names=class_names
        if self.task=="train":
            self.oi_bbox_annotations_path = oi_path+"/oidv6-train-annotations-bbox.csv"
            self.oi_source_images_dir = oi_path+"/train"
            self.oi_image_annotations_path = oi_path+"/oidv7-train-annotations-human-imagelabels.csv"
            self.oi_image_info_path = oi_path+"/train-images-boxable-with-rotation.csv"
        else:
            self.oi_bbox_annotations_path = oi_path+"/validation-annotations-bbox.csv"
            self.oi_source_images_dir = oi_path+"/validation"
            self.oi_image_annotations_path = oi_path+"/validation-annotations-human-imagelabels-boxable.csv"
            self.oi_image_info_path = oi_path+"/validation-images-with-rotation.csv"
        self.oi_class_description_path = oi_path+"/oidv7-class-descriptions-boxable.csv"
        self.oi_bbox_df = pd.read_csv(self.oi_bbox_annotations_path)
        self.oi_image_annotations = pd.read_csv(self.oi_image_annotations_path)
        self.oi_class_descriptions = pd.read_csv(self.oi_class_description_path, header=None, names=['LabelName', 'ClassName'])
        self.oi_image_info = pd.read_csv(self.oi_image_info_path)
        self.oi_rotation_dict=dict(zip(self.oi_image_info['ImageID'], self.oi_image_info['Rotation']))
        self.oi_class_dict = pd.Series(self.oi_class_descriptions.LabelName.values, index=self.oi_class_descriptions.ClassName).to_dict()
        self.category_list=[]
        self.class_mappings={}

    def get_info(self):
        return "openimages https://storage.googleapis.com/openimages/web/index.html"

    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        try:
            oi_category_id = self.oi_class_dict[source_class]
            self.class_mappings[oi_category_id]=self.class_names.index(dest_class)
            self.category_list.append(oi_category_id)
        except KeyError as e:
            print(f"Error: Could not add openimages category map {source_class}->{dest_class}")
            print(f"keys {self.oi_class_dict.keys()}")

    def get_annotations(self, image_id):
        group=self.grouped.get_group(image_id)
        class_ids=group['LabelName']
        out_det=[]
        for i, row in group.iterrows():
            # Get bounding box coordinates
            xmin, xmax = row['XMin'], row['XMax']
            ymin, ymax = row['YMin'], row['YMax']

            an_box=[stuff.clip01(xmin),stuff.clip01(ymin),stuff.clip01(xmax),stuff.clip01(ymax)]
            det={"box":an_box,
                "class":self.class_mappings[class_ids[i]],
                "confidence":1.0}
            out_det.append(det)
        return du.dedup_gt(out_det, iou_thr=0.5)

    def get_image_ids(self):
        # Filter for only the rows that correspond to the 'Person' class
        self.person_bboxes = self.oi_bbox_df[self.oi_bbox_df['LabelName'].isin(self.category_list)]

        # Group by image, since there might be multiple persons per image
        self.grouped = self.person_bboxes.groupby('ImageID')

        image_ids=[]
        for image_id, group in self.grouped:
            skip=False
            if group['IsGroupOf'].any():
                skip=True
            if group['IsDepiction'].any():
                skip=True
            rotation=self.oi_rotation_dict.get(image_id)
            if rotation!=0:
                skip=True
            if skip==False:
                image_ids.append(image_id)
        return image_ids

    def get_img_path(self, image_id):
        return os.path.join(self.oi_source_images_dir, f"{image_id}.jpg")

    def get_category_maps(self):
        return { 'person':['Person','Man','Woman','Boy','Girl','Human body'],
                 'vehicle':['Aircraft','Ambulance','Car','Bus','Golf cart','Barge','Bicycle','Motorcycle','Land vehicle','Snowplow','Taxi','Truck','Train','Van'],
                 'animal':['Animal','Chicken','Fox','Eagle','Dolphin','Duck','Pig','Squirrel','Raccoon','Rabbit','Animal','Dog','Cat','Bird','Blue jay'],
                 'face':['Human face'], # Human head
                 'weapon':['Handgun', 'Rifle', 'Shotgun', 'Sword', 'Dagger'],
                 'confused':['Knife','Kitchen knife','Golf cart','Marine mammal'],
                 'ignore':[],
                 'background':['Accordion','Adhesive tape','Alarm clock','Backpack','Bagel','Balloon','Banjo','Barrel','Bathroom accessory',
                                'Beaker','Bed','Beer','Billiard table','Binoculars','Blender','Book','Bookcase','Bottle','Bottle opener',
                                'Bowl','Box','Briefcase','Building','Bust','Cake','Cake stand','Calculator','Carrot','Castle','Ceiling fan',
                                'Cello','Chainsaw','Chest of drawers','Chime','Chisel','Christmas tree','Clock','Closet','Clothing','Coat',
                                'Cocktail shaker','Coffee cup','Coffee table','Coin','Computer keyboard','Computer monitor','Computer mouse',
                                'Container','Convenience store','Cookie','Corded phone','Cosmetics','Couch','Countertop','Cowboy hat',
                                'Cricket ball','Croissant','Crutch','Cucumber','Cupboard','Curtain','Cutting board','Desk','Digital clock',
                                'Dishwasher','Doll','Door handle','Drawer','Dress','Drinking straw','Drum','Envelope','Fashion accessory',
                                'Fedora','Filing cabinet','Fire hydrant','Flag','Flower','Flowerpot','Flute','Food','Football','Footwear',
                                'Fork','Frying pan','Furniture','Glasses','Glove','Hair dryer','Hammer','Hand dryer','Handbag','Hat','Headphones',
                                'Hiking equipment','Home appliance','Horizontal bar','House','Houseplant','Indoor rower','Infant bed',
                                'Ipod','Jacket','Jug','Kettle','Kitchen & dining room table','Kitchen appliance','Kitchenware',
                                'Ladder','Lamp','Laptop','Lipstick','Luggage and bags','Mechanical fan','Medical equipment','Microphone',
                                'Mirror','Mixing bowl','Mobile phone','Mug','Office building','Office supplies','Paddle','Palm tree','Parking meter',
                                'Pasta','Pastry','Pen','Pencil case','Personal flotation device','Picture frame','Pizza','Plant','Plastic bag',
                                'Plate','Plumbing fixture','Porch','Power plugs and sockets','Printer','Punching bag','Refrigerator','Remote control',
                                'Rose','Rugby ball','Salt and pepper shakers','Sandal','Sandwich','Scarf','Screwdriver','Sculpture','Seat belt',
                                'Sewing machine','Shelf','Shirt','Shorts','Sink','Skateboard','Skyscraper','Snack','Snowboard','Sofa bed','Sombrero',
                                'Spoon','Sports equipment','Stairs','Stapler','Stop sign','Stool','Street light','Studio couch','Sunglasses','Sun hat',
                                'Swimming pool','Table','Table tennis racket','Tablet computer','Tableware','Tap','Tennis ball','Tennis racket',
                                'Tie','Tire','Toaster','Toilet','Tool','Tower','Toy','Traffic light','Traffic sign','Training bench',
                                'Treadmill','Tree','Tree house','Umbrella','Washing machine','Wardrobe','Waste container','Wheel','Wheelchair',
                                'Whiteboard','Window','Window blind','Wine glass','Wok','Wrench']
        }