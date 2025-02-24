import multiprocessing
from ultralytics import YOLO
 
# 设置多进程启动方法为 'spawn'（适用于 Windows）
multiprocessing.set_start_method('spawn', force=True)
 
if __name__ == "__main__":
    # 加载模型权重
    model = YOLO(r'D:\ultralytics-main\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml')
 
    # 开始训练
    model.train(
        # 指定 .yaml 配置文件
        data=r"D:\ultralytics-main\ultralytics-main\ultralytics\cfg\datasets\first-train.yaml",
        # 训练轮次
        epochs=100,
        batch=16,
        imgsz=640,
        # 保存训练结果路径
        project='runs/train',
        # 指定使用的设备（'0'代表第一张GPU，如果没有GPU使用'cpu'）
        device='0'
    )
    