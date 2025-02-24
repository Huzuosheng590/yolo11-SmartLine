# 导包
from ultralytics import YOLO
import cv2




if __name__ == '__main__':
# 加载模型
   model = YOLO(r"runs\train\train\weights\best.pt") 
 

   results = model.val(data=r"C:\Users\16030\Desktop\ultralytics-main\ultralytics-main\ultralytics\cfg\datasets\first-train.yaml", split='test',imgsz=640,conf=0.001) 
# 获取绘制了边界框的图像
