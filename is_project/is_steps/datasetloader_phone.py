import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from typing_extensions import override
from .base_datasetloader import BaseDatasetLoader

class DatasetLoaderPhone(BaseDatasetLoader):
    def __init__(self, 
                 dir_path: str = "pedestrian/is_project/datasets_csv/", 
                 max_frames_number: int = 10, 
                 class_2_int: dict = {"walking.csv": 0, "standing.csv": 2, "running.csv": 1, "phone.csv": 0, "standing_phone.csv": 2},
                 prefix: dict = {"walking.csv": 'walking (', "standing.csv": 'standing (', "running.csv": 'running (', "phone.csv": 'phone (', "standing_phone.csv": 'stand_phone ('},
                 postfix: dict = {"walking.csv": ').mp4', "standing.csv": ').mp4', "running.csv": ').mp4', "phone.csv": ').mp4', "standing_phone.csv": ').mp4'},
                 phone: dict = {"walking.csv": False, "standing.csv": False, "running.csv": False, "phone.csv": True, "standing_phone.csv": True},
                 split: bool = True,
                 train_images_percent: float = 0.8) -> None:
        """
        Initialize DatasetLoaderPhone.

        Args:
            dir_path (str): Path to the directory containing CSV files.
            max_frames_number (int): Maximum number of frames.
            class_2_int (dict): Dictionary mapping class CSV filenames to integer labels.
            prefix (dict): Dictionary mapping class CSV filenames to prefix strings.
            postfix (dict): Dictionary mapping class CSV filenames to postfix strings.
            phone (dict): Dictionary indicating whether the activity involves a phone.
            split (bool): Whether to split the dataset into training and testing sets.
            train_images_percent (float): Percentage of data to use for training.
        """
        super().__init__(dir_path, max_frames_number, class_2_int, prefix, postfix, phone, split, train_images_percent, model_type="phone")

    @override
    def run(self):
        """
        Load the dataset, process it, and optionally split it into training and testing sets.

        Returns:
            If split is True, returns X_train, X_test, y_train, y_test.
            Otherwise, returns X, y.
        """
        data_x = []
        data_y_motion = []
        data_y_phone = []

        # Iterate over each file in the directory
        for filename in os.listdir(self.dir_path):
            list_x, list_y_motion, list_y_phone = self.csv_to_lists_x_y(
                os.path.join(self.dir_path, filename), 
                self.max_frames_number, 
                self.prefix[filename], 
                self.postfix[filename], 
                self.class_2_int[filename], 
                self.phone[filename]
            )
            data_x.extend(list_x)
            data_y_motion.extend(list_y_motion)
            data_y_phone.extend(list_y_phone)

        # Convert lists to numpy arrays
        data_x = np.array(data_x, dtype=np.float32)
        data_y_phone = np.array(data_y_phone, dtype=np.float32)
        data_y_phone = np.expand_dims(data_y_phone, axis=1)

        if self.split:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(data_x, data_y_phone, test_size=1 - self.train_images_percent)
            X_train = torch.from_numpy(X_train)
            X_test = torch.from_numpy(X_test)
            y_train = torch.from_numpy(y_train)
            y_test = torch.from_numpy(y_test)

            return X_train, X_test, y_train, y_test
        else:
            X = torch.from_numpy(data_x)
            y = torch.from_numpy(data_y_phone)

            return X, y
