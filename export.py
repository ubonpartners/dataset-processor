from ultralytics import YOLO
import argparse

def do_quant(model, dataset):
    model = YOLO(model, task="detect")
    model.export(
        format="engine",
        dynamic=False,  
        batch=1,  
        workspace=8,  
        half=True,
        verbose=False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='process_dataset.py')
    parser.add_argument('--dataset', type=str, default="/mldata/v5-attr/coco-v5-attr/dataset.yaml",
                         help='Dataset to use for quantization')
    parser.add_argument('--model',  type=str, default="/mldata/weights/good/yolo11l-dpa-250525.pt", help='model to use')
    opt = parser.parse_args()
    do_quant(opt.model, opt.dataset)
