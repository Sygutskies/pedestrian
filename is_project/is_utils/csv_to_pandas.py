import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from is_models.motion_model import MotionModel
import time
from is_steps.train import TrainStep
from is_steps.kpi import KpiStep

def csv_to_lists_x_y(csv_path:str, max_frames_number:int, prefix:str, postfix:str, class_to_int:int, phone:bool):
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
            movie = np.delete(movie, np.s_[0:5], axis=1)
            temp_list = []
            for i in range (movie.shape[0] - max_frames_number):
                temp_list.append(movie[i:(i+max_frames_number), :])
            list_x.append(temp_list)
            # movie = movie[:num_sets * max_frames_number, :]
            # movie = np.delete(movie, np.s_[0:5], axis=1)
            # list_x.append(np.array_split(movie, num_sets))

    list_x = [item for sublist in list_x for item in sublist]
    list_y_motion = [class_to_int for i in range(len(list_x))]
    if phone:
        list_y_phone = [1 for i in range(len(list_x))]
    else:
        list_y_phone = [0 for i in range(len(list_x))]
    return list_x, list_y_motion, list_y_phone

walking_kth_list_x, walking_kth_list_y_motion, walking_kth_list_y_phone = csv_to_lists_x_y("pedestrian/is_project/datasets_csv/walking.csv", 10, 'walking (', ').mp4', 0, False)
running_list_x, running_list_y_motion, running_list_y_phone = csv_to_lists_x_y("pedestrian/is_project/datasets_csv/running.csv", 10, 'running (', ').mp4', 2, False)
standing_list_x, standing_list_y_motion, standing_list_y_phone = csv_to_lists_x_y("pedestrian/is_project/datasets_csv/standing.csv", 10, 'standing (', ').mp4', 1, False)
phone_list_x, phone_list_y_motion, phone_list_y_phone = csv_to_lists_x_y("pedestrian/is_project/datasets_csv/phone.csv", 10, 'phone (', ').mp4', 0, True)

data_x = np.array(standing_list_x + phone_list_x + running_list_x + walking_kth_list_x)
data_y_motion = np.array(tf.keras.utils.to_categorical(standing_list_y_motion + phone_list_y_motion + running_list_y_motion + walking_kth_list_y_motion, num_classes=3))
data_y_phone = np.array(tf.keras.utils.to_categorical(standing_list_y_phone + phone_list_y_phone + running_list_y_phone + walking_kth_list_y_phone, num_classes=2))

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y_phone, test_size=0.2)
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
gmt = time.gmtime()
log_name = str(gmt.tm_year) + str(gmt.tm_mon) + str(gmt.tm_mday) + str(gmt.tm_hour) + str(gmt.tm_min)
train_step = TrainStep(epochs=2, model = "phone")
model = train_step.run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, log_name=log_name)
kpi_step = KpiStep(model=model, weights="./logs/" + log_name + "/weights/best/checkpoint")
kpi_step.predict(X_test=X_test, y_test=y_test)




