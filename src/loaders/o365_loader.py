import src.dataset_util as du
import os
import stuff

# This code assumes you have already downloaded the O365 dataset
# and labels
#
#ls /mldata/downloaded_datasets/object365/
# annotations.json         annotations_val.json   val       zhiyuan_objv2_val.json
# annotations_train.json   train     zhiyuan_objv2_train.json

class O365Loader:
    def __init__(self,
                 o365_path="/mldata/downloaded_datasets/object365",
                 task="val", class_names=["face","person"], ds_params=None):

        gdrive_sync=ds_params.get("gdrive_sync", True)
        if gdrive_sync:
            stuff.gdrive_sync_mldata(o365_path, direction="from_drive")

        self.task=task
        annotation_path = o365_path+"/zhiyuan_objv2_"+task+".json"
        # Load COCO dataset
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError("Please install pycocotools: pip install pycocotools")
        self.o365 = COCO(annotation_path)
        self.o365_images_dir = o365_path+"/"+task
        self.class_names=class_names
        self.class_mappings={}
        self.category_list=[]

    def get_info(self):
        return "o365 https://www.objects365.org/"

    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        try:
            o365_category_id = self.o365.getCatIds(catNms=[source_class])[0]
            self.class_mappings[o365_category_id]=self.class_names.index(dest_class)
            self.category_list.append(o365_category_id)
        except IndexError as e:
            print(f"Warning: Could not add o365 category map {source_class}->{dest_class}")

    def get_o365_anns(self, img_id, anns):
        img_info = self.o365.loadImgs([img_id])[0]
        img_width=img_info['width']
        img_height=img_info['height']
        out_det=[]
        for ann in anns:
            bbox=ann['bbox']
            x=bbox[0]/img_width
            y=bbox[1]/img_height
            w=bbox[2]/img_width
            h=bbox[3]/img_height
            an_box=[stuff.clip01(x),stuff.clip01(y),stuff.clip01(x+w),stuff.clip01(y+h)]
            det={"box":an_box,
                "class":self.class_mappings[ann['category_id']],
                "confidence":1.0}
            out_det.append(det)
        return out_det

    def get_annotations(self, img_id):
        ann_ids = self.o365.getAnnIds(imgIds=img_id, catIds=self.category_list, iscrowd=False)
        ann_ids_crowd = self.o365.getAnnIds(imgIds=img_id, catIds=self.category_list, iscrowd=True)

        if len(ann_ids_crowd)!=0:
            return None

        anns = self.o365.loadAnns(ann_ids)

        gts=du.dedup_gt(self.get_o365_anns(img_id, anns))

        #if (int(img_id)==68559) or int(img_id)==56107:
        #    print(img_id)print(ann_ids)
        #    print(ann_ids_crowd)
        #
        #    for a in anns:
        #        print(a['category_id'])
        #        category=self.o365.loadCats(a['category_id'])
        #        if category:
        #            print(category[0]['name'])
        #    print(gts)
        #    exit()

        return gts

    def get_image_ids(self):
        image_ids=set()
        for c in self.category_list:
            image_ids=image_ids.union(set(self.o365.getImgIds(catIds=[c])))
        return list(image_ids)

    def get_img_path(self, img_id):
        img_info = self.o365.loadImgs([img_id])[0]
        img_path = os.path.join(self.o365_images_dir, img_info['file_name'])
        return img_path

    def get_category_maps(self):
        return { 'person':['Person'],
                 'vehicle':['Car','Truck','Train','Airplane','Carriage','Helicopter','Van','Pickup Truck','Tricycle','Boat','Bus','Machinery Vehicle','Sports Car','Fire Truck','Sailboat','Ship','SUV','Ambulance','Bicycle','Motorcycle','Scooter','Heavy Truck'],
                 'animal':['Wild Bird','Cow','Swan','Rabbit','Horse','Lion','Sheep','Campel','Deer','Yak','Bear','Dog','Pigeon','Donkey','Cat','Parrot','Goose','Chicken','Antelope','Zebra','Duck','Pig','Giraffe','Monkey'],
                 'face':[],
                 'weapon':['Gun'],
                 'confused':['Knife'],
                 'ignore':[],
                 'background':['Sneakers','Chair','Other Shoes','Hat','Lamp','Glasses','Bottle','Desk','Cup','Street Lights',
                               'Cabinet/shelf','Handbag/Satchel','Bracelet','Plate','Picture/Frame','Helmet','Book','Gloves',
                               'Storage box','Flower','Bench','Potted Plant','Bowl/Basin','Flag','Pillow','Boots','Vase',
                               'Microphone','Wine Glass','Belt','Moniter/TV','Backpack','Umbrella','Traffic Light','Speaker',
                               'Watch','Tie','Trash bin Can','Stool','Barrel/bucket','Couch','Sandals','Bakset','Drum','Pen/Pencil',
                               'Guitar','Carpet','Cell Phone','Camera','Canned','Traffic cone','Stuffed Toy','Candle','Bed',
                               'Faucet','Tent','Mirror','Power outlet','Sink','Apple','Air Conditioner','Hockey Stick',
                               'Paddle','Fork','Traffic Sign','Ballon','Tripod','Spoon','Clock','Pot','Cake','Dinning Table',
                               'Hanger','Blackboard/Whiteboard','Napkin','Keyboard','Lantern','Fan','Banana','Baseball Glove',
                               'Pumpkin','Skiboard','Luggage','Nightstand','Tea pot','Telephone','Trolley','Head Phone','Dessert',
                               'Stroller','Crane','Remote','Refrigerator','Oven','Lemon','Baseball Bat','Surveillance Camera',
                               'Jug','Pizza','Skateboard','Surfboard','Skating and Skiing shoes','Gas stove','Donut','Carrot',
                               'Toilet','Kite','Strawberry','Other Balls','Toilet','Shovel','Computer Box','Toilet Paper',
                               'Cleaning Products','Chopsticks','Microwave','Baseball','Cutting/chopping Board','Coffee Table',
                               'Side Table','Scissors','Marker','Pie','Ladder','Snowboard','Cookies','Radiator','Fire Hydrant',
                               'Basketball','Violin','Egg','Fire Extinguisher','Candy','Bathtub','Wheelchair','Golf Club',
                               'Briefcase','Cucumber','Cigar/Cigarette','Paint brush','Extractor','Extention Cord','Tong',
                               'Tennis Racket','American Football','earphone','Mask','Kettle','Tennis','Swing','Coffee Machine',
                               'Slide','Green beans','Projector','Washing Machine/Drying Machine','Printer','Toothbrush','Hotair ballon',
                               'Cello','Scale','Trophy','Cabbage','Hot dog','Blender','Wallet/Purse','Volleyball','Tablet',
                               'Cosmetics','Trumpet','Pineapple','Golf Ball','Parking meter','Mango','Key','Hurdle','Fishing Rod',
                               'Medal','Flute','Brush','Megaphone','Corn','Lettuce','Nuts','Speed Limit Sign','Broom','Router/modem',
                               'Poker Card','Toaster','Cheese','Notepad','Pliers','CD','Hammer','Cue','Flask','Mushroon',
                               'Screwdriver','Soap','Recorder','Board Eraser','Tape Measur/ Ruler','Showerhead','Globe',
                               'Chips','Steak','Crosswalk Sign','Stapler','Formula 1','Dishwasher','Hoverboard','Rice Cooker',
                               'Tuba','Calculator','Dumbell','Electric Drill','Hair Dryer','Treadmill','Lighter','Mop',
                               'Target','Pencil Case','Red Cabbage','Barbell','Asparagus','Comb','Table Teniis paddle',
                               'Chainsaw','Eraser','Durian','Okra','Lipstick','Cosmetics Mirror','Curling','Table Tennis']}