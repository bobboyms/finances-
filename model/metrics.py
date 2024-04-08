import torch
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_accuracy


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the accuracy of multiclass classification outputs.

    This function computes the softmax of the model's outputs to turn them into probabilities,
    finds the class with the highest probability for each instance, and then calculates
    the accuracy by comparing these predictions to the true labels.

    Args:
        outputs (torch.Tensor): The raw output tensor from the model (logits).
        labels (torch.Tensor): The true labels tensor. Labels are assumed to be in one-hot encoded format.

    Returns:
        torch.Tensor: The accuracy of the predictions as a single-element tensor.
    """
    
    # [0,1,0] Mercado
    # [1,0,0] Restaurante
    # [0,0,1] Saude
    
    # Find the class with the highest probability in the true labels
    target = torch.argmax(labels, dim=1)

    # [0.0255, 0.2544, 0.044]
    # [0.2, 0.7, 0.1]
    # 1 = 1
    # Apply softmax to the model output and find the predicted class
    probabilities = F.softmax(outputs, dim=1)
    output = torch.argmax(probabilities, dim=1)

    # Calculate and return the accuracy
    return multiclass_accuracy(output, target).cpu()
