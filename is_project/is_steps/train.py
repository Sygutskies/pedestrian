import torch
from torch.utils.tensorboard import SummaryWriter
from is_steps.metrics import multiclass_accuracy, binary_accuracy
import csv

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainStep:
    """
    A class to handle the training process for a given model, including 
    logging, validation, and saving the best model checkpoint.
    """

    def __init__(self, epochs=10, tensorboard_callback=True, save_best_checkpoint=True, model_type="phone", l1_lambda=0.001):
        """
        Initializes the training parameters.

        Args:
            epochs (int): Number of epochs to train the model.
            tensorboard_callback (bool): Whether to log training progress to TensorBoard.
            save_best_checkpoint (bool): Whether to save the best model checkpoint.
            model_type (str): Type of model ("phone" or "motion").
            l1_lambda (float): Lambda value for L1 regularization.
        """
        self.epochs = epochs
        self.tensorboard_callback = tensorboard_callback
        self.save_best_checkpoint = save_best_checkpoint
        self.model_type = model_type
        self.l1_lambda = l1_lambda

    def run(self, model, train_loader, test_loader, log_name):
        """
        Runs the training and validation process.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            test_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            log_name (str): Directory name for saving logs and checkpoints.

        Returns:
            torch.nn.Module: The trained model.
        """
        # Move the model to the appropriate device
        model.to(device)

        # Initialize TensorBoard writer if logging is enabled
        writer = SummaryWriter(log_dir=log_name) if self.tensorboard_callback else None

        # Get the criterion, optimizer, and accuracy function based on the model type
        criterion, optimizer, accuracy = self.get_model_specifics(model)

        best_accuracy = 0.0
        best_model_path = f"{log_name}/best_model.pth"

        # Open a CSV file to log training and validation metrics
        with open(f'{log_name}/training.csv', 'w', newline='') as file:
            csv_writer = csv.writer(file)
            field = ["Loss/train", "Accuracy/train", "Loss/val", "Accuracy/val"]
            csv_writer.writerow(field)

            # Training loop
            for epoch in range(self.epochs):
                # Train the model for one epoch
                epoch_loss, epoch_accuracy = self.train_one_epoch(model, train_loader, criterion, optimizer, accuracy)
                
                # Validate the model
                val_loss, val_accuracy = self.validate(model, test_loader, criterion, accuracy)

                # Log metrics to TensorBoard
                if writer:
                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                    writer.add_scalar("Accuracy/train", epoch_accuracy, epoch)
                    writer.add_scalar("Loss/val", val_loss, epoch)
                    writer.add_scalar("Accuracy/val", val_accuracy, epoch)

                # Log metrics to CSV file
                csv_writer.writerow([epoch_loss, epoch_accuracy, val_loss, val_accuracy])

                # Save the best model checkpoint
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    if self.save_best_checkpoint:
                        torch.save(model.state_dict(), best_model_path)
                    print(f"Validation accuracy increased. Saving the best model...")

                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if writer:
            writer.close()

        return model

    def get_model_specifics(self, model):
        """
        Gets the criterion, optimizer, and accuracy function based on the model type.

        Args:
            model (torch.nn.Module): The model to train.

        Returns:
            tuple: Criterion, optimizer, and accuracy function.
        """
        if self.model_type == "phone":
            criterion = torch.nn.BCELoss()
            accuracy = binary_accuracy
        elif self.model_type == "motion":
            criterion = torch.nn.CrossEntropyLoss()
            accuracy = multiclass_accuracy
        else:
            raise ValueError("Invalid model type")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        return criterion, optimizer, accuracy

    def train_one_epoch(self, model, data_loader, criterion, optimizer, accuracy_func):
        """
        Trains the model for one epoch.

        Args:
            model (torch.nn.Module): The model to train.
            data_loader (torch.utils.data.DataLoader): DataLoader for training data.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            accuracy_func (function): Accuracy function.

        Returns:
            tuple: Average loss and accuracy for the epoch.
        """
        model.train()
        total_loss, total_accuracy = 0.0, 0.0

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) + self.l1_regularization_loss(model)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_accuracy += accuracy_func(outputs, labels).item()

        avg_loss = total_loss / len(data_loader.dataset)
        avg_accuracy = total_accuracy / len(data_loader)
        return avg_loss, avg_accuracy

    def validate(self, model, data_loader, criterion, accuracy_func):
        """
        Validates the model.

        Args:
            model (torch.nn.Module): The model to validate.
            data_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            criterion (torch.nn.Module): Loss function.
            accuracy_func (function): Accuracy function.

        Returns:
            tuple: Average loss and accuracy for the validation.
        """
        model.eval()
        total_loss, total_accuracy = 0.0, 0.0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_accuracy += accuracy_func(outputs, labels).item()

        avg_loss = total_loss / len(data_loader.dataset)
        avg_accuracy = total_accuracy / len(data_loader)
        return avg_loss, avg_accuracy

    def l1_regularization_loss(self, model):
        """
        Computes the L1 regularization loss for the model.

        Args:
            model (torch.nn.Module): The model to compute the regularization loss for.

        Returns:
            torch.Tensor: L1 regularization loss.
        """
        l1_loss = torch.tensor(0., device=device)
        for param in model.parameters():
            l1_loss += torch.norm(param, 1)
        return self.l1_lambda * l1_loss
