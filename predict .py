import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r'runs\train\yolov10n/weights/best.pt')
    model.predict(source=r'ultralytics\cfg\datasets\Ours\test\images',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  conf=0.2,
                  iou=0.7,
                )