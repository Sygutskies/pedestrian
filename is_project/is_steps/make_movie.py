import cv2
import os
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from is_utils.helper_functions import common_row_elements, normalize_keypoints
from is_steps.resclass import Result

# Constants for drawing
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
FONT_THICKNESS = 2
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GRAY_COLOR = (128, 128, 128)
INT_TO_STATE = {True: "active", False: "not active"}

class MakeMovie:
    def __init__(self, motion_model, phone_model, poseweights="yolov8s-pose.pt",
                 motion_weights="logs/2024114177/weights/best/checkpoint",
                 phone_weights="logs/2024114177/weights/best/checkpoint",
                 video_source="pedestrian/is_project/test_movies/crowd1.webm",
                 phone_id_to_str={0: "no phone", 1: "phone"},
                 motion_id_to_str={0: "walking", 1: "running", 2: "standing"},
                 frames_number=12, log_name=""):
        self.motion_model = motion_model
        self.phone_model = phone_model
        self.poseweights = poseweights
        self.motion_weights = motion_weights
        self.phone_weights = phone_weights
        self.video_source = video_source
        self.phone_id_to_str = phone_id_to_str
        self.motion_id_to_str = motion_id_to_str
        self.frames_number = frames_number
        self.log_name = log_name

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pose_model = YOLO(self.poseweights)
        car_model = YOLO("yolov8n.pt")
        
        # Load the motion and phone models
        self.loadModels(device)
        
        rows = deque(maxlen=self.frames_number)
        filename = os.path.splitext(os.path.basename(self.video_source))[0]
        writer = cv2.VideoWriter(f'logs/{self.log_name}/{filename}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1920, 1080))
        output = Result()

        # Process each frame of the video
        for result in pose_model.track(self.video_source, stream=True):
            frame = result.orig_img
            cars_output = car_model(frame)
            car_boxes = self.getCarBoxes(cars_output)
            if result.boxes.xyxy.size(0) > 0:
                rows.append(self.extractKeypoints(result))
                if len(rows) == self.frames_number:
                    self.processFrames(rows, output, device)
            else:
                output.reset()

            self.drawPredictions(frame, result, output)
            writer.write(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) == 27:
                break

        writer.release()
        cv2.destroyAllWindows()
        print("The video was successfully saved")

    def loadModels(self, device):
        """Load the motion and phone models onto the specified device."""
        self.phone_model.load_state_dict(torch.load(self.phone_weights))
        self.motion_model.load_state_dict(torch.load(self.motion_weights))
        self.phone_model.to(device)
        self.motion_model.to(device)

    def getCarBoxes(self, cars_output):
        """Extract bounding boxes for cars from the model output."""
        indexes = np.argwhere(np.isin(cars_output[0].boxes.cls.cpu(), [2, 3, 5, 7]))
        car_boxes = cars_output[0].boxes.xyxy[indexes.squeeze()]
        return car_boxes

    def extractKeypoints(self, result):
        """Extract and normalize keypoints from the model result."""
        row = {}
        for box_id, box in enumerate(result.keypoints.xy):
            keypoints = box.cpu().numpy().flatten()
            normalized_keypoints = normalize_keypoints(keypoints, result.boxes.xyxy[box_id].cpu().numpy())
            if result.boxes.id is not None:
                row[result.boxes.id[box_id].item()] = normalized_keypoints
        return row

    def processFrames(self, rows, output, device):
        """Process a sequence of frames to predict motion and phone usage."""
        data = list(rows)
        possible_ids = common_row_elements([list(row.keys()) for row in data])
        if possible_ids:
            videos = {i: [row.get(i, []) for row in data] for i in possible_ids}
            videos_array = np.array(list(videos.values()))
            phone_preds = self.predict(self.phone_model, videos_array[:, :, 10:22], device)
            motion_preds = self.predict(self.motion_model, videos_array[:, :, 22:], device)
            self.updateOutput(videos, phone_preds, motion_preds, output)
        else:
            output.reset()

    def predict(self, model, data, device):
        """Predict using the specified model."""
        tensor = torch.from_numpy(data).to(device)
        predictions = model(tensor).detach().cpu().numpy()
        return predictions

    def updateOutput(self, videos, phone_preds, motion_preds, output):
        """Update the output object with predictions."""
        y_pred_phone = [self.phone_id_to_str[int(pred > 0.5)] for pred in phone_preds[:, 0]]
        y_pred_motion = [self.motion_id_to_str[idx] for idx in np.argmax(motion_preds, axis=1)]
        phone_duration = {str(k): 1 if v == "phone" else -1 for k, v in zip(videos.keys(), y_pred_phone)}
        motion_duration = {str(k): 1 if v == "running" else -1 for k, v in zip(videos.keys(), y_pred_motion)}
        y_pred = list(zip(y_pred_motion, y_pred_phone))
        output.set(list(videos.keys()), y_pred)
        output.set_duration(motion_duration, phone_duration)

    def drawPredictions(self, frame, result, output):
        """Draw predictions on the frame."""
        if output.ids and output.predictions:
            y_offset = 45
            for box_id, (idx, (motion_status, phone_status)) in enumerate(zip(output.ids, output.predictions)):
                x1, y1, x2, y2 = result.boxes.xyxy[box_id].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE_COLOR, 4)
                cv2.putText(frame, f"ID: {idx}", (x1 + 20, y1 - 30), FONT, FONT_SCALE, BLUE_COLOR, FONT_THICKNESS)
                
                if (x2 - x1) > (y2 - y1):
                    cv2.putText(frame, f"ID: {idx}", (10, y_offset), FONT, FONT_SCALE, BLUE_COLOR, FONT_THICKNESS)
                    cv2.putText(frame, "Fall detected", (10, y_offset + 40), FONT, FONT_SCALE, RED_COLOR, FONT_THICKNESS)
                    print("Fall detected")
                else:
                    phone_alert = output.phone_alert[str(idx)] and 50 < x1 and x2 < 1870
                    running_alert = output.motion_alert[str(idx)] and 50 < x1 and x2 < 1870
                    phone_color = RED_COLOR if phone_alert else GREEN_COLOR
                    motion_color = RED_COLOR if running_alert else GREEN_COLOR
                    if not (50 < x1 and x2 < 1870):
                        phone_color = motion_color = GRAY_COLOR

                    cv2.putText(frame, f"ID: {idx}", (10, y_offset), FONT, FONT_SCALE, BLUE_COLOR, FONT_THICKNESS)
                    cv2.putText(frame, f"Motion: {motion_status}, duration: {output.dur_motion[str(idx)]}, running alert: {INT_TO_STATE[running_alert]}",
                                (10, y_offset + 40), FONT, FONT_SCALE, motion_color, FONT_THICKNESS)
                    cv2.putText(frame, f"Phone: {phone_status}, duration: {output.dur_phone[str(idx)]}, phone alert: {INT_TO_STATE[phone_alert]}",
                                (10, y_offset + 80), FONT, FONT_SCALE, phone_color, FONT_THICKNESS)

                y_offset += 120
        else:
            cv2.putText(frame, "Not found", (10, 30), FONT, FONT_SCALE, GRAY_COLOR, FONT_THICKNESS)
