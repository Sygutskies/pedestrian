from ultralytics import YOLO
model = YOLO("yolov8s-pose.pt")
results = model('converted_dataset\walking\walking (1).mp4', save=True)