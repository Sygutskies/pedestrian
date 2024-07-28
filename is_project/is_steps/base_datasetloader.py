import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class BaseDatasetLoader:
    def __init__(self, 
                 dir_path: str = "pedestrian/is_project/datasets_csv/", 
                 max_frames_number: int = 10, 
                 class_2_int: dict = {"walking.csv": 0, "standing.csv": 1, "running.csv": 2, "phone.csv": 0, "standing_phone.csv": 1},
                 prefix: dict = {"walking.csv": 'walking (', "standing.csv": 'standing (', "running.csv": 'running (', "phone.csv": 'phone (', "standing_phone.csv": 'stand_phone ('},
                 postfix: dict = {"walking.csv": ').mp4', "standing.csv": ').mp4', "running.csv": ').mp4', "phone.csv": ').mp4', "standing_phone.csv": ').mp4'},
                 phone: dict = {"walking.csv": False, "standing.csv": False, "running.csv": False, "phone.csv": True, "standing_phone.csv": True},
                 split: bool = True,
                 train_images_percent: float = 0.8,
                 model_type: str = "phone") -> None:
        """
        Initialize BaseDatasetLoader.

        Args:
            dir_path (str): Path to the directory containing CSV files.
            max_frames_number (int): Maximum number of frames.
            class_2_int (dict): Dictionary mapping class CSV filenames to integer labels.
            prefix (dict): Dictionary mapping class CSV filenames to prefix strings.
            postfix (dict): Dictionary mapping class CSV filenames to postfix strings.
            phone (dict): Dictionary indicating whether the activity involves a phone.
            split (bool): Whether to split the dataset into training and testing sets.
            train_images_percent (float): Percentage of data to use for training.
            model_type (str): Type of the model ("phone" or "motion").
        """
        self.dir_path = dir_path
        self.max_frames_number = max_frames_number
        self.class_2_int = class_2_int
        self.prefix = prefix
        self.postfix = postfix
        self.phone = phone
        self.split = split
        self.train_images_percent = train_images_percent
        self.model_type = model_type

    def csv_to_lists_x_y(self, csv_path: str, max_frames_number: int, prefix: str, postfix: str, class_to_int: int, phone: bool):
        """
        Convert CSV data to lists of features (x) and labels (y).

        Args:
            csv_path (str): Path to the CSV file.
            max_frames_number (int): Maximum number of frames.
            prefix (str): Prefix for the video name.
            postfix (str): Postfix for the video name.
            class_to_int (int): Integer label for the class.
            phone (bool): Whether the activity involves a phone.

        Returns:
            tuple: Lists of features (x) and labels (y_motion, y_phone).
        """
        df = pd.read_csv(csv_path)
        number_of_videos = len(df)

        # Normalize coordinates
        for column in df.columns[5:]:
            if column.endswith('_x'):
                df[column] = (df[column] - df['left']) / (df['right'] - df['left'])
            else:
                df[column] = (df[column] - df['top']) / (df['bottom'] - df['top'])

        list_x = []

        # Process each video
        for i in range(number_of_videos):
            name = f"{prefix}{i+1}{postfix}"
            movie = df.loc[df['name'] == name].to_numpy()
            num_sets = movie.shape[0] // max_frames_number

            if num_sets > 0:
                if self.model_type == "motion":
                    movie = np.delete(movie, np.s_[0:27], axis=1)
                elif self.model_type == "phone":
                    movie = np.delete(movie, np.s_[0:15], axis=1)
                    movie = np.delete(movie, np.s_[12:], axis=1)
                else:
                    raise ValueError("Unknown model type")

                temp_list = [movie[i:(i+max_frames_number), :] for i in range(movie.shape[0] - max_frames_number)]
                list_x.extend(temp_list)

        list_y_motion = [class_to_int] * len(list_x)
        list_y_phone = [1 if phone else 0] * len(list_x)

        return list_x, list_y_motion, list_y_phone

    def run(self):
        """
        Load and process the dataset.
        This method should be overridden in subclasses.
        """
        pass