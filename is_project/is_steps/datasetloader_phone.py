import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .base_datasetloader import BaseDatasetLoader
from typing_extensions import override

class DatasetLoaderPhone(BaseDatasetLoader):
    def __init__(self, dir_path:str = "pedestrian/is_project/datasets_csv/", 
                 max_frames_number:int = 10, 
                 class_2_int: dict = {"walking.csv": 0, "standing.csv": 2, "running.csv": 1, "phone.csv":0, "standing_phone.csv": 2},
                 prefix: dict = {"walking.csv": 'walking (', "standing.csv": 'standing (', "running.csv": 'running (', "phone.csv": 'phone (', "standing_phone.csv": 'stand_phone ('},
                 postfix: dict = {"walking.csv": ').mp4', "standing.csv": ').mp4', "running.csv": ').mp4', "phone.csv": ').mp4', "standing_phone.csv": ').mp4'},
                 phone: dict = {"walking.csv": False, "standing.csv": False, "running.csv": False, "phone.csv": True, "standing_phone.csv": True},
                 split: bool = True,
                 train_images_percent: float = 0.8) -> None:
        super().__init__(dir_path, max_frames_number, class_2_int, prefix, postfix, phone, split, train_images_percent, model_type="phone")

    @override
    def run(self):
        data_x = []
        data_y_motion = []
        data_y_phone = []
        for filename in os.listdir(self.dir_path):
            list_x, list_y_motion, list_y_phone = self.csv_to_lists_x_y(self.dir_path + filename, self.max_frames_number, self.prefix[filename], self.postfix[filename], self.class_2_int[filename], self.phone[filename])
            data_x = data_x + list_x
            data_y_motion = data_y_motion + list_y_motion
            data_y_phone = data_y_phone + list_y_phone
        data_x = np.array(data_x)
        data_y_phone = np.array(tf.keras.utils.to_categorical(data_y_phone, num_classes=2))
        if self.split:
            X_train, X_test, y_train, y_test = train_test_split(data_x, data_y_phone, test_size= 1 - self.train_images_percent)
            X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
            X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
            y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

            return X_train, X_test, y_train, y_test
        else:
            X = tf.convert_to_tensor(data_x, dtype=tf.float32)
            y = tf.convert_to_tensor(data_y_phone, dtype=tf.float32)

            return X, y