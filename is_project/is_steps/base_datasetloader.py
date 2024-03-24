import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class BaseDatasetLoader:
    def __init__(self, dir_path:str = "pedestrian/is_project/datasets_csv/", 
                 max_frames_number:int = 10, 
                 class_2_int: dict = {"walking.csv": 0, "standing.csv": 2, "running.csv": 1, "phone.csv":0, "standing_phone.csv": 2},
                 prefix: dict = {"walking.csv": 'walking (', "standing.csv": 'standing (', "running.csv": 'running (', "phone.csv": 'phone (', "standing_phone.csv": 'stand_phone ('},
                 postfix: dict = {"walking.csv": ').mp4', "standing.csv": ').mp4', "running.csv": ').mp4', "phone.csv": ').mp4', "standing_phone.csv": ').mp4'},
                 phone: dict = {"walking.csv": False, "standing.csv": False, "running.csv": False, "phone.csv": True, "standing_phone.csv": True},
                 split: bool = True,
                 train_images_percent: float = 0.8,
                 model_type: str = "phone") -> None:
        self.dir_path = dir_path
        self.max_frames_number = max_frames_number
        self.class_2_int = class_2_int
        self.prefix = prefix
        self.postfix = postfix
        self.phone = phone
        self.split = split
        self.train_images_percent = train_images_percent
        self.model_type = model_type

    def csv_to_lists_x_y(self, csv_path:str, max_frames_number:int, prefix:str, postfix:str, class_to_int:int, phone:bool):
        df = pd.read_csv(csv_path)
        number_of_videos = len(df)
        for column in df.columns[5:]:
            if column.endswith('_x'):
                df[column] = (df[column] - df['left']) / (df['right'] - df['left'])
            else:
                df[column] = (df[column] - df['top']) / (df['bottom'] - df['top'])
        list_x = []

        for i in range(number_of_videos):
            name = prefix + str(i+1) + postfix
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
                temp_list = []
                for i in range (movie.shape[0] - max_frames_number):
                    temp_list.append(movie[i:(i+max_frames_number), :])
                list_x.append(temp_list)
                # movie = movie[:num_sets * max_frames_number, :]
                # movie = np.delete(movie, np.s_[0:15], axis=1)
                # list_x.append(np.array_split(movie, num_sets))
        list_x = [item for sublist in list_x for item in sublist]
        list_y_motion = [class_to_int for i in range(len(list_x))]
        if phone:
            list_y_phone = [1 for i in range(len(list_x))]
        else:
            list_y_phone = [0 for i in range(len(list_x))]
        return list_x, list_y_motion, list_y_phone

    def run(self):
        pass

