import torch

def binary_accuracy(predictions, labels):
    """
    Compute the accuracy for binary classification.

    Args:
        predictions (torch.Tensor): Predictions from the model.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: Accuracy of the predictions.
    """
    # Convert predictions to binary values (0 or 1) using threshold 0.5
    rounded_preds = (predictions >= 0.5).float()
    
    # Compare predicted values with true labels
    correct = (rounded_preds == labels).float()
    
    # Compute accuracy
    acc = correct.sum() / len(correct)
        
    return acc

def multiclass_accuracy(predictions, labels):
    """
    Compute the accuracy for multiclass classification.

    Args:
        predictions (torch.Tensor): Predictions from the model.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: Accuracy of the predictions.
    """
    # Get the predicted class by taking the index with the maximum value
    predicted = torch.argmax(predictions, dim=1)
    
    # Get the ground truth class by taking the index with the maximum value
    gt = torch.argmax(labels, dim=1)
    
    # Compare predicted values with true labels
    correct = torch.eq(predicted, gt)
    
    # Compute accuracy
    acc = torch.sum(correct) / len(correct)
    
    return acc
