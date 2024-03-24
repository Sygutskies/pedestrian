from ultralytics import YOLO

# model = YOLO("yolov8n-seg.pt")

# train_results = model.train(data='data.yaml', epochs=5, imgsz=640, batch=4)

model_best = YOLO("runs/segment/train3/weights/best.pt")
results = model_best.track('IMG_5315.mp4', save=True)
# results = model_best('IMG_5315.MOV', save=True, stream=True)

# print(results[0].masks)

