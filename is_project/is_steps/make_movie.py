from ultralytics import YOLO
import numpy as np
from is_utils.common_row_elements import common_row_elements
import cv2

class MakeMovie:
    def __init__(self, motion_model, phone_model, poseweights: str = "yolov8s-pose.pt",
                 motion_weights: str = "logs/2024114177/weights/best/checkpoint",
                 phone_weights: str = "logs/2024114177/weights/best/checkpoint",
                 video_source: str = "pedestrian/is_project/test_movies/crowd1.webm",
                 phone_id_to_str: dict = {0: "no phone", 1: "phone"},
                 motion_id_to_str: dict = {0: "walking", 1: "running", 2: "standing"},
                 frames_number: int = 12,
                 log_name :str = "") -> None:
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
        model = YOLO(self.poseweights)
        results = model.track(self.video_source, persist=True, stream=True)
        self.phone_model.load_weights(self.phone_weights)   
        self.motion_model.load_weights(self.motion_weights)  
        rows = []
        counter = 0
        output = {}
        bbs = {} 
        for frame_number, result in enumerate(results):
            import tensorflow as tf
            if result.boxes.xyxy.size(dim=0) == 0:
                print('no person')
                counter = 0
                bbs[frame_number] = "no boxes"
            else:
                row = {}
                bbs[frame_number] = (result.boxes.xyxy, result.boxes.id) 
                for box_id, box in enumerate(result.keypoints.xy):
                    keypoints = []
                    for cords in box.cpu().numpy():
                        for i, cord in enumerate(cords):
                            if i == 0:
                                keypoints.append((cord - result.boxes.xyxy[box_id].cpu().numpy()[0]) / (result.boxes.xyxy[box_id].cpu().numpy()[2]- result.boxes.xyxy[box_id].cpu().numpy()[0]))
                            else:
                                keypoints.append((cord - result.boxes.xyxy[box_id].cpu().numpy()[1]) / (result.boxes.xyxy[box_id].cpu().numpy()[3]- result.boxes.xyxy[box_id].cpu().numpy()[1]))
                    if result.boxes.id is not None:
                        row[result.boxes.id.cpu().numpy()[box_id]] = keypoints
                rows.append(row)
                counter += 1

            if counter >= self.frames_number:
                data = rows[-self.frames_number:]
                data_keys = []
                for row in data:
                    data_keys.append(list(row.keys()))
                possible_ids = common_row_elements(data_keys)
                if possible_ids:
                    videos = {}
                    for i in possible_ids:
                        video = []
                        for row in data:
                            video.append(row[i])
                        videos[i] = video
                    videos_tensor_phone = tf.convert_to_tensor(np.array(list(videos.values()))[:,:,10:22], dtype=tf.float32)
                    videos_tensor_motion = tf.convert_to_tensor(np.array(list(videos.values()))[:,:,22:], dtype=tf.float32)
                    predictions_phone = self.phone_model.predict(videos_tensor_phone)
                    y_pred_phone = np.argmax(predictions_phone, axis=1)
                    y_pred_phone = [self.phone_id_to_str[x] for x in y_pred_phone]
                    predictions_motion = self.motion_model.predict(videos_tensor_motion)
                    y_pred_motion = np.argmax(predictions_motion, axis=1)
                    y_pred_motion = [self.motion_id_to_str[x] for x in y_pred_motion]
                    y_pred = list(zip(y_pred_motion,y_pred_phone))
                    output[frame_number] = list(zip(list(videos.keys()), y_pred))
                else:
                    output[frame_number] = 0
            else:
                output[frame_number] = 0


        cap = cv2.VideoCapture(self.video_source)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create VideoWriter object to save the output video
        output_path = 'logs/' + self.log_name + '/output_video.mp4'  # Replace with the desired output video path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for video writing
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (1, 1, 1)  # White color
        red_color = (255, 1, 1)

        # Process each frame and add text
        frame_number = 0 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add text to the frame
            result = output[frame_number]
            if result != 0:
                ids, v = zip(*result)
            else:
                ids = None
                v = None
            if bbs[frame_number] != "no boxes":
                boxes, id_list = bbs[frame_number]
                boxes = boxes.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    if id_list is not None:
                        if id_list[i] is not None:
                            if ids and v:
                                if id_list.cpu().numpy()[i] in ids:
                                    if v[ids.index(id_list.cpu().numpy()[i])][1] == 'phone':
                                        cv2.putText(frame, str(v[ids.index(id_list.cpu().numpy()[i])]), (int(x1), int(y1 - 20)), font, font_scale, red_color, font_thickness)
                                    else:    
                                        cv2.putText(frame, str(v[ids.index(id_list.cpu().numpy()[i])]), (int(x1), int(y1 - 20)), font, font_scale, font_color, font_thickness)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 4)
            
            # Write the frame to the output video
            out.write(frame)
            frame_number += 1
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()
                    