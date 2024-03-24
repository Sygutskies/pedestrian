from is_models.motion_model import MotionModel
from is_models.phone_model import PhoneModel
import tensorflow as tf

class TrainStep:
    def __init__(self, epochs: int = 10, 
                 tensorboard_callback: bool = True, 
                 save_best_checkpoint: bool = True) -> None:
        self.epochs = epochs
        self.tensorboard_callback = tensorboard_callback
        self.save_best_checkpoint = save_best_checkpoint
    def run(self, model, X_train, X_test, y_train, y_test, log_name):
        callbacks = []
        if self.tensorboard_callback:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/" + log_name)
            callbacks.append(tensorboard_callback)
        if self.save_best_checkpoint:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="./logs/" + log_name + "/weights/best/checkpoint",
            save_weights_only=True,
            monitor='val_categorical_accuracy',
            mode='max',
            save_best_only=True)
            callbacks.append(model_checkpoint_callback)

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs, callbacks=callbacks)
        
        return model