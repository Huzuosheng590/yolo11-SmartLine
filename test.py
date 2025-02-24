import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r'runs\train\1 Our dataset\C3K2-AKConv-EMA-Ghost/weights/best.pt')
    model.val(data=r'ultralytics\cfg\datasets\first-train.yaml',
              split='test',
              imgsz=640,
              batch=16,
              iou=0.6,
              conf=0.25,
              rect=True,
              save_json=False,
              project='runs/val',
              name='exp',
              )