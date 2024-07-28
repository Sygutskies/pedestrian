import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KpiStep:
    def __init__(self, model, log_dir, weights="", tensorboard=True, model_type="motion"):
        """
        Initialize KpiStep.

        Args:
            model: PyTorch model to be evaluated.
            log_dir: Directory to save logs.
            weights: Path to model weights file.
            tensorboard: Boolean to enable TensorBoard logging.
            model_type: Type of the model ("motion" or "phone").
        """
        self.model = model
        self.weights = weights
        self.tensorboard = tensorboard
        self.log_dir = log_dir
        self.model_type = model_type

    def predict(self, test_loader):
        """
        Predict the labels for the test dataset and generate metrics.

        Args:
            test_loader: DataLoader for the test dataset.
        """
        # Load model weights
        self.model.load_state_dict(torch.load(self.weights))
        self.model.to(device)
        self.model.eval()

        true_labels, predicted_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)

                if self.model_type == "phone":
                    predicted = torch.round(outputs)
                    true_labels.extend(labels.cpu().numpy().astype(int))
                    predicted_labels.extend(predicted.cpu().numpy().astype(int))
                elif self.model_type == "motion":
                    true_labels.extend(labels.argmax(dim=1).cpu().numpy())
                    predicted_labels.extend(outputs.argmax(dim=1).cpu().numpy())
                else:
                    raise ValueError("Invalid model type")

        self.generate_metrics(true_labels, predicted_labels)

    def generate_metrics(self, true_labels, predicted_labels):
        """
        Generate and log metrics including confusion matrix.

        Args:
            true_labels: True labels from the test dataset.
            predicted_labels: Predicted labels from the model.
        """
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        labels = ["walk", "stand", "run"] if self.model_type == "motion" else ["no phone", "phone"]
        df_cm = pd.DataFrame(conf_matrix / conf_matrix.sum(axis=1)[:, None], index=labels, columns=labels)
        plt.figure(figsize=(12, 7))
        heat_map = sn.heatmap(df_cm, annot=True).get_figure()

        if self.tensorboard:
            writer = SummaryWriter(log_dir=self.log_dir)
            writer.add_figure("Confusion matrix", heat_map, 0)
            if self.model_type == "phone":
                metrics = self.compute_metrics_phone(true_labels, predicted_labels)
                for name, value in metrics.items():
                    writer.add_scalar(f"Phone_{name}", value, 0)
            elif self.model_type == "motion":
                metrics = self.compute_metrics_motion(true_labels, predicted_labels)
                for name, value in metrics.items():
                    writer.add_scalar(f"Motion_{name}", value, 0)
            writer.close()

    def compute_metrics_phone(self, true_labels, predicted_labels):
        """
        Compute evaluation metrics for phone model.

        Args:
            true_labels: True labels from the test dataset.
            predicted_labels: Predicted labels from the model.

        Returns:
            dict: Dictionary with accuracy, precision, recall, and F1 score.
        """
        TP = sum((np.array(true_labels) == 1) & (np.array(predicted_labels) == 1))
        TN = sum((np.array(true_labels) == 0) & (np.array(predicted_labels) == 0))
        FP = sum((np.array(true_labels) == 0) & (np.array(predicted_labels) == 1))
        FN = sum((np.array(true_labels) == 1) & (np.array(predicted_labels) == 0))
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    def compute_metrics_motion(self, true_labels, predicted_labels):
        """
        Compute evaluation metrics for motion model.

        Args:
            true_labels: True labels from the test dataset.
            predicted_labels: Predicted labels from the model.

        Returns:
            dict: Dictionary with accuracy, precision, recall, and F1 score for each class.
        """
        metrics = {}
        classes = [0, 1, 2]
        labels = ['walk', 'stand', 'run']

        for i, label in enumerate(labels):
            TP = sum((np.array(true_labels) == classes[i]) & (np.array(predicted_labels) == classes[i]))
            TN = sum((np.array(true_labels) != classes[i]) & (np.array(predicted_labels) != classes[i]))
            FP = sum((np.array(true_labels) != classes[i]) & (np.array(predicted_labels) == classes[i]))
            FN = sum((np.array(true_labels) == classes[i]) & (np.array(predicted_labels) != classes[i]))
            
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP) if TP + FP != 0 else 0
            recall = TP / (TP + FN) if TP + FN != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

            metrics[f'accuracy_{label}'] = accuracy
            metrics[f'precision_{label}'] = precision
            metrics[f'recall_{label}'] = recall
            metrics[f'f1_{label}'] = f1_score

        return metrics