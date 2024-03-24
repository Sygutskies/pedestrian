import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
import io

class KpiStep:
    def __init__(self, model, log_dir, weights:str = "", tensorboard: bool = True, model_type: str = "motion") -> None:
        self.model = model
        self.weights = weights
        self.tensorboard = tensorboard
        self.log_dir = log_dir
        self.model_type = model_type
    def predict(self, X_test, y_test):
        def plot_to_image(figure):
            """Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call."""
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image
        
        def perf_measure(y_actual, y_pred):
            TP = 0
            FP = 0
            TN = 0
            FN = 0

            for i in range(len(y_pred)): 
                if y_actual[i]==y_pred[i]==1:
                    TP += 1
                if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
                    FP += 1
                if y_actual[i]==y_pred[i]==0:
                    TN += 1
                if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
                    FN += 1

            return(TP, FP, TN, FN)

        self.model.load_weights(self.weights)
        predictions = self.model.predict(X_test)
        y_true = np.argmax(y_test.numpy()[:, :3], axis=1)
        y_pred = np.argmax(predictions[:, :3], axis=1)

        conf_matrix = confusion_matrix(y_true, y_pred)
        TP, FP, TN, FN = perf_measure(y_true, y_pred)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 / (1/precision + 1/recall)

        print(conf_matrix)
        if self.model_type == "motion":
            df_cm = pd.DataFrame(conf_matrix, index = ["walk", "stand", "run"], columns = ["walk", "stand", "run"])
        elif self.model_type == "phone":
            df_cm = pd.DataFrame(conf_matrix, index = ["no phone", "phone"], columns = ["no phone", "phone"])
        else:
            raise ValueError("Wrong model type")
        figure = plt.figure(figsize=(10,10))
        sn.heatmap(df_cm, annot=True, fmt='.10g')

        if self.tensorboard:
            file_writer = tf.summary.create_file_writer(self.log_dir)
            with file_writer.as_default():
                tf.summary.image("Training data", plot_to_image(figure), step=0)
                tf.summary.scalar(self.model_type + "TP", TP, step=0)
                tf.summary.scalar(self.model_type + "TN", TN, step=0)
                tf.summary.scalar(self.model_type + "FP", FP, step=0)
                tf.summary.scalar(self.model_type + "FN", FN, step=0)
                tf.summary.scalar(self.model_type + "precision", precision, step=0)
                tf.summary.scalar(self.model_type + "recall", recall, step=0)
                tf.summary.scalar(self.model_type + "f1", f1, step=0)
                tf.summary.scalar(self.model_type + "accuracy", accuracy, step=0)

        
