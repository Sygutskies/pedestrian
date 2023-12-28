import csv
from ultralytics import YOLO
import os
import numpy as np

def movie(results, name):
    rows = []
    for result in results:
        if result.boxes.xyxy.size(dim=0) == 0:
            print('no person')
        else:
            row = []
            row.append(name)
            for cord in result.boxes.xyxy[0].numpy():
                row.append(cord)
            for cords in result.keypoints.xy[0].numpy():
                for cord in cords:
                    row.append(cord)
            rows.append(row)

    return rows

model = YOLO("yolov8s-pose.pt")
directory = 'converted_dataset/running'
    
header = ['name','left', 'top', 'right', 'bottom', 'nose_x', 'nose_y', 'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y', 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y', 'left_elbow_x', 'left_elbow_y', 'right_elbow_x', 'right_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x', 'right_wrist_y', 'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y', 'left_knee_x', 'left_knee_y', 'right_knee_x', 'right_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_ankle_x', 'right_ankle_y']
with open('converted_dataset/running/running.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for filename in os.listdir(directory):
        if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".webm"):
            f = os.path.join(directory, filename)
            results = model(f)
            rows = movie(results, filename)
            writer.writerows(rows)