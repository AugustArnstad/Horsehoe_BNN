from torch import nn
import torch
import properscoring as ps
import matplotlib.pyplot as plt
import numpy as np
discrimination_loss = nn.functional.cross_entropy
from models import HorseNet

def regression_objective(output, target, kl_divergence, p=2.0):
    """
    Computes the variational bound using either classification loss (cross-entropy) 
    or regression loss (interpolated between MSE and MAE).
    
    Args:
        output (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth labels.
        kl_divergence (torch.Tensor): KL divergence term.
        p (float): p-norm parameter for regression loss.
    
    Returns:
        torch.Tensor: Computed loss (variational bound).
    """
    epsilon = 1e-6
    
    reg_loss = torch.mean((torch.abs(output - target) + epsilon)**p)**(1/p)

    variational_bound = reg_loss + kl_divergence
    return variational_bound


def poisson_objective(output, target, kl_divergence, classification=False, p=2.0):
    """
    Computes the variational bound using either classification loss (cross-entropy) 
    or regression loss (p-norm or Poisson loss for count data).
    """
    if classification:
        discrimination_error = discrimination_loss(output, target)
        return discrimination_error + kl_divergence

    reg_loss_fn = nn.PoissonNLLLoss(log_input=True)  # Define loss function
    reg_loss = reg_loss_fn(output, target)  # Compute actual loss value

    variational_bound = reg_loss + kl_divergence
    return variational_bound


def binary_classification_objective(output, target, kl_divergence, p=0.0):
    """
    Binary Cross-Entropy Loss + KL Divergence for Bayesian Logistic Regression.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    reg_loss = loss_fn(output, target)  # Compute loss
    
    variational_bound = reg_loss + kl_divergence  # Add KL divergence term
    return variational_bound

def multiclass_classification_objective(output, target, kl_divergence, p=0.0):
    """
    Cross-Entropy Loss + KL Divergence for Bayesian Multi-Class Classification.
    """
    loss_fn = nn.CrossEntropyLoss()  
    reg_loss = loss_fn(output, target)  # Compute loss
    
    variational_bound = reg_loss + kl_divergence  # Add KL divergence term
    return variational_bound

