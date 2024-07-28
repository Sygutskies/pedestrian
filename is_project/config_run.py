from datetime import datetime
import torch
from is_steps.datasetloader_motion import DatasetLoaderMotion
from is_steps.datasetloader_phone import DatasetLoaderPhone
from is_steps.kpi import KpiStep
from is_steps.train import TrainStep
from is_steps.make_movie import MakeMovie
from is_steps.build import Build
from is_datasets.custom_datasets import CustomDataset

# Constants
frames_number = 12
log_name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_name_phone = f"logs/{log_name}/phone"
log_name_motion = f"logs/{log_name}/motion"
batch_size = 256  # Adjust according to your needs

# DataLoader for Phone Model
phone_prefix_train = {"walking.csv": 'walkinga (', "standing.csv": 'standing (', "running.csv": 'runninga (', "phone.csv": 'phone (', "standing_phone.csv": 'standphone ('}
phone_postfix_train = {"walking.csv": ')', "standing.csv": ')', "running.csv": ')', "phone.csv": ')', "standing_phone.csv": ')'}
phone_prefix_test = {"walking.csv": 'walking (', "standing.csv": 'standing (', "running.csv": 'running (', "phone.csv": 'phone (', "standing_phone.csv": 'standphone ('}
phone_postfix_test = {"walking.csv": ')', "standing.csv": ')', "running.csv": ')', "phone.csv": ')', "standing_phone.csv": ')'}

# Load training data for phone model
datasetloader = DatasetLoaderPhone(dir_path="pedestrian/is_project/training_movies/", max_frames_number=frames_number, split=False,
                                   prefix=phone_prefix_train, postfix=phone_postfix_train)
X_train, y_train = datasetloader.run()

# Load testing data for phone model
datasetloader = DatasetLoaderPhone(dir_path="kpi_movies/", max_frames_number=frames_number, split=False,
                                   prefix=phone_prefix_test, postfix=phone_postfix_test)
X_test, y_test = datasetloader.run()

# Print data shapes
print(X_train.shape)
print(X_test.shape)

# Create custom datasets
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Build, train, and evaluate phone model
build_step_phone = Build(model_type="phone", features_number=12, log_dir=log_name_phone)
model_phone = build_step_phone.run()
train_step = TrainStep(epochs=100, l1_lambda=0.005)
model_phone = train_step.run(train_loader=train_loader, test_loader=test_loader, log_name=log_name_phone, model=model_phone)
kpi_step = KpiStep(model=model_phone, weights=log_name_phone + "/best_model.pth", log_dir=log_name_phone + "/Kpi/conf_matrix", model_type="phone")
kpi_step.predict(test_loader)

# DataLoader for Motion Model
motion_prefix_train = {"walking.csv": 'walkinga (', "standing.csv": 'standing (', "running.csv": 'runninga (', "phone.csv": 'phone (', "standing_phone.csv": 'standphone ('}
motion_postfix_train = {"walking.csv": ')', "standing.csv": ')', "running.csv": ')', "phone.csv": ')', "standing_phone.csv": ')'}
motion_prefix_test = {"walking.csv": 'walking (', "standing.csv": 'standing (', "running.csv": 'running (', "phone.csv": 'phone (', "standing_phone.csv": 'standphone ('}
motion_postfix_test = {"walking.csv": ')', "standing.csv": ')', "running.csv": ')', "phone.csv": ')', "standing_phone.csv": ')'}

# Load training data for motion model
datasetloader = DatasetLoaderMotion(dir_path="pedestrian/is_project/training_movies_motion/", max_frames_number=frames_number, split=False,
                                    prefix=motion_prefix_train, postfix=motion_postfix_train)
X_train, y_train = datasetloader.run()

# Load testing data for motion model
datasetloader = DatasetLoaderMotion(dir_path="kpi_movies/", max_frames_number=frames_number, split=False,
                                    prefix=motion_prefix_test, postfix=motion_postfix_test)
X_test, y_test = datasetloader.run()

# Print data shapes
print(X_train.shape)
print(X_test.shape)

# Create custom datasets
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Build, train, and evaluate motion model
build_step_motion = Build(model_type="motion", features_number=12, log_dir=log_name_motion)
model_motion = build_step_motion.run()
train_step = TrainStep(epochs=100, model_type="motion", l1_lambda=0.00003)
model_motion = train_step.run(train_loader=train_loader, test_loader=test_loader, log_name=log_name_motion, model=model_motion)
kpi_step = KpiStep(model=model_motion, weights=log_name_motion + "/best_model.pth", log_dir=log_name_motion + "/Kpi/conf_matrix", model_type="motion")
kpi_step.predict(test_loader)

# Create a movie with predictions
make_movie = MakeMovie(phone_model=model_phone,
                       motion_model=model_motion,
                       phone_weights=log_name_phone + "/best_model.pth",
                       motion_weights=log_name_motion + "/best_model.pth",
                       video_source="final_test/_test13.MOV",
                       frames_number=frames_number,
                       log_name=log_name)
make_movie.run()
