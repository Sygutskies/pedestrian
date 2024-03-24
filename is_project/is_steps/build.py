from is_models.motion_model import MotionModel
from is_models.phone_model import PhoneModel
import tensorflow as tf

class Build:
    def __init__(self, model_type: str = "motion",
                 input_shape: tuple = (None, 12, 24),
                 init_weights: str = None) -> None:
        self.model_type = model_type
        self.input_shape = input_shape
        self.init_weights = init_weights

    def run(self):
        if self.model_type == "motion":
            model = MotionModel()
        elif self.model_type == "phone":
            model = PhoneModel()
        else:
            raise ValueError("Wrong model")
        model.build(input_shape=self.input_shape)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        if self.init_weights:
            model.load_weights(self.init_weights)

        return model
