from is_models.har_model import HarModel
from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from is_utils.common_row_elements import common_row_elements

id_to_str = {0: "walking", 1: "standing", 2:"phone", 3:"running"}

def run(poseweights:str = "yolov8s-pose.pt", lstm_weights:str = "tmp/checkpoint", video_source:str="test_movies/crowd1.webm"):
    lstm_model = HarModel()
    lstm_model.build(input_shape=(None, 10, 34))

    lstm_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    lstm_model.load_weights('tmp/checkpoint')    
    model = YOLO(poseweights)
    results = model.track(video_source, persist=True)
    rows = []
    counter = 0 
    for result in results:
        if result.boxes.xyxy.size(dim=0) == 0:
            print('no person')
            counter = 0
        else:
            row = {}

            for box_id, box in enumerate(result.keypoints.xy):
                keypoints = []
                for cords in box.numpy():
                    for i, cord in enumerate(cords):
                        if i == 0:
                            keypoints.append((cord - result.boxes.xyxy[box_id].numpy()[0]) / (result.boxes.xyxy[box_id].numpy()[2]- result.boxes.xyxy[box_id].numpy()[0]))
                        else:
                            keypoints.append((cord - result.boxes.xyxy[box_id].numpy()[1]) / (result.boxes.xyxy[box_id].numpy()[3]- result.boxes.xyxy[box_id].numpy()[1]))
                if result.boxes.id is not None:
                    row[result.boxes.id.numpy()[box_id]] = keypoints
            rows.append(row)
            counter += 1

        if counter >= 10:
            data = rows[-10:]
            data_keys = []
            for row in data:
                data_keys.append(list(row.keys()))
            possible_ids = common_row_elements(data_keys)
            if possible_ids:
                videos = []
                for i in possible_ids:
                    video = []
                    for row in data:
                        video.append(row[i])
                    videos.append(video)
                videos = tf.convert_to_tensor(videos, dtype=tf.float32)
                predictions = lstm_model.predict(videos)
                y_pred = np.argmax(predictions, axis=1)
                y_pred = [id_to_str[x] for x in y_pred]
                print(y_pred)

            


run()