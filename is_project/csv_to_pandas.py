import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from is_models.har_model import HarModel
import time

def csv_to_lists_x_y(csv_path:str, number_of_videos:int, max_frames_number:int, prefix:str, postfix:str, class_to_int:int, phone:bool):
    df = pd.read_csv(csv_path)
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
            movie = movie[:num_sets * max_frames_number, :]
            movie = np.delete(movie, np.s_[0:5], axis=1)
            list_x.append(np.array_split(movie, num_sets))

    list_x = [item for sublist in list_x for item in sublist]
    list_y_1 = [class_to_int for i in range(len(list_x))]
    if phone:
        list_y_2 = [1 for i in range(len(list_x))]
    else:
        list_y_2 = [0 for i in range(len(list_x))]
    return list_x, list_y_1, list_y_2

#walking_list_x, walking_list_y = csv_to_lists_x_y("walking.csv", 171, 20, 'Walking (', ').mp4', 0)
walking_kth_list_x, walking_kth_list_y_1, walking_kth_list_y_2 = csv_to_lists_x_y("datasets_csv/walking.csv", 207, 10, 'walking (', ').mp4', 0, False)
#jogging_list_x, jogging_list_y = csv_to_lists_x_y("kth_jogging.csv", 100, 20, 'jogging (', ').avi', 3)
running_list_x, running_list_y_1, running_list_y_2 = csv_to_lists_x_y("datasets_csv/running.csv", 216, 10, 'running (', ').mp4', 2, False)
standing_list_x, standing_list_y_1, standing_list_y_2 = csv_to_lists_x_y("datasets_csv/standing.csv", 174, 10, 'standing (', ').mp4', 1, False)
phone_list_x, phone_list_y_1, phone_list_y_2 = csv_to_lists_x_y("datasets_csv/phone.csv", 20, 10, 'phone (', ').mp4', 0, True)

data_x = np.array(standing_list_x + phone_list_x + running_list_x + walking_kth_list_x)
data_y_1 = np.array(tf.keras.utils.to_categorical(standing_list_y_1 + phone_list_y_1 + running_list_y_1 + walking_kth_list_y_1, num_classes=3))
data_y_2 = np.array(tf.keras.utils.to_categorical(standing_list_y_2 + phone_list_y_2 + running_list_y_2 + walking_kth_list_y_2, num_classes=2))

data_y = np.column_stack((data_y_1, data_y_2))
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
model = HarModel()
model.build(input_shape=(None, 10, 34))
checkpoint_filepath = 'tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_mae',
    mode='min',
    save_best_only=True)
gmt = time.gmtime()
log_name = str(gmt.tm_year) + str(gmt.tm_mon) + str(gmt.tm_mday) + str(gmt.tm_hour) + str(gmt.tm_min) 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/" + log_name)

model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[model_checkpoint_callback, tensorboard_callback])

model.load_weights(checkpoint_filepath)
predictions = model.predict(X_test)
y_true = np.argmax(y_test.numpy()[:, :3], axis=1)
y_pred = np.argmax(predictions[:, :3], axis=1)

conf_matrix = confusion_matrix(y_true, y_pred)

print(conf_matrix)

y_true = np.argmax(y_test.numpy()[:, 3:], axis=1)
y_pred = np.argmax(predictions[:, 3:], axis=1)

conf_matrix = confusion_matrix(y_true, y_pred)

print(conf_matrix)


