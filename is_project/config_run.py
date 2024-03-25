from is_steps.datasetloader_motion import DatasetLoaderMotion
from is_steps.datasetloader_phone import DatasetLoaderPhone
from is_steps.kpi import KpiStep
from is_steps.train import TrainStep
from is_steps.make_movie import MakeMovie
from is_steps.build import Build
from datetime import datetime

frames_number = 15
log_name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_name_phone = log_name + "/phone"
log_name_motion = log_name + "/motion"

datasetloader = DatasetLoaderPhone(dir_path="pedestrian/is_project/dataset_all/", max_frames_number= frames_number, split=False)
X_train, y_train = datasetloader.run()
datasetloader = DatasetLoaderPhone(dir_path="pedestrian/is_project/dataset_valid/", max_frames_number = frames_number, split=False)
X_test, y_test = datasetloader.run()
build_step_phone = Build(model_type="phone", input_shape = (None, frames_number, 12))
model_phone = build_step_phone.run()
train_step = TrainStep(epochs=2)
model_phone = train_step.run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, log_name=log_name_phone, model=model_phone)
kpi_step = KpiStep(model=model_phone, weights="./logs/" + log_name_phone + "/weights/best/checkpoint", log_dir="./logs/" + log_name_phone + "/Kpi/conf_matrix", model_type="phone")
kpi_step.predict(X_test=X_test, y_test=y_test)

datasetloader = DatasetLoaderMotion(dir_path="pedestrian/is_project/motion_data/", max_frames_number = frames_number, split=False)
X_train, y_train = datasetloader.run()
datasetloader = DatasetLoaderMotion(dir_path="pedestrian/is_project/dataset_valid/", max_frames_number = frames_number, split=False)
X_test, y_test = datasetloader.run()
build_step_motion = Build(model_type ="motion", input_shape = (None, frames_number, 12))
model_motion = build_step_motion.run()
train_step = TrainStep(epochs=2)
model_motion = train_step.run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, log_name=log_name_motion, model=model_motion)
kpi_step = KpiStep(model=model_motion, weights="./logs/" + log_name_motion + "/weights/best/checkpoint", log_dir="./logs/" + log_name_motion + "/Kpi/conf_matrix", model_type="motion")
kpi_step.predict(X_test=X_test, y_test=y_test)

make_movie = MakeMovie(phone_model=model_phone, 
                       motion_model=model_motion, 
                       phone_weights="./logs/" + log_name_phone + "/weights/best/checkpoint", 
                       motion_weights="./logs/" + log_name_motion + "/weights/best/checkpoint",
                       video_source="test_movies/IMG_5315.MOV",
                       frames_number = frames_number,
                       log_name=log_name)
make_movie.run()
