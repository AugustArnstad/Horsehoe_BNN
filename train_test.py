import torch
from torch.autograd import Variable
from torch import nn
import properscoring as ps
import matplotlib.pyplot as plt
import numpy as np
discrimination_loss = nn.functional.cross_entropy
from models import HorseNet
from objectives import regression_objective, poisson_objective, binary_classification_objective, multiclass_classification_objective

def train(epoch, model, train_loader, optimizer, N, objective, **kwargs):
    """
    Train for one epoch.
    
    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): Model being trained.
        train_loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer for updating weights.
        N (int): Normalization factor for KL divergence.
        objective (function): Objective function (loss function).
        **kwargs: Additional parameters (classification, p, track, MC_sample, device).
    
    Returns:
        Tuple: (kl, loss, neg_elbo, kl/neg_elbo ratio).
    """
    model.train()
    epoch_kl = 0
    epoch_loss = 0
    epoch_neg_elbo = 0
    num_batches = 0

    p = kwargs.get("p", 2.0)
    track = kwargs.get("track", False)
    MC_sample = kwargs.get("MC_sample", False)
    device = kwargs.get("device", 'cpu')
    link = kwargs.get("link", None)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        
        kl_div = model.MC_KL()/N if MC_sample else model.kl_divergence()/N
        neg_elbo = objective(output, target, kl_div, p=p)
        neg_elbo.backward()
        optimizer.step()

        epoch_kl += kl_div.item()
        epoch_loss += (neg_elbo - kl_div).item()
        epoch_neg_elbo += neg_elbo.item()
        num_batches += 1

    if track:
        kl = epoch_kl / num_batches
        loss = epoch_loss / num_batches
        neg_elbo_history = epoch_neg_elbo / num_batches
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Negative ELBO = {neg_elbo_history:.6f}')
        return kl, loss, neg_elbo_history, kl_div/neg_elbo
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Negative ELBO = {epoch_neg_elbo / num_batches:.6f}')

def train_wrap(model, optimizer, train_loader, N, objective, **kwargs):
    """
    Generalized training loop with additional options.
    
    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_loader (DataLoader): Training data loader.
        N (int): Normalization factor for KL divergence.
        objective (function): Objective function (loss function).
        **kwargs: Additional options like `epochs`, `cosine_annealing`, `classification`, `p`, etc.
    
    Returns:
        Dict containing loss history, KL history, and shrinkage/dropout rates.
    """
    epochs = kwargs.get("epochs", 50)
    check_list = kwargs.get("check_list", [1, 10, 50, 100, 150, 200])
    cosine_annealing = kwargs.get("cosine_annealing", False)

    kl_history, loss_history, neg_elbo_history, ratio_history = [], [], [], []
    
    # Initialize learning rate scheduler if needed
    if cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    for epoch in range(1, epochs + 1):
        results = train(epoch, model, train_loader, optimizer, N, objective, **kwargs)

        if kwargs.get("track", False):
            kl, loss, neg_elbo, ratio = results
            kl_history.append(kl)
            loss_history.append(loss)
            neg_elbo_history.append(neg_elbo)
            if epoch in check_list:
                ratio_history.append(ratio)

        if cosine_annealing:
            scheduler.step()

    # Extract shrinkage/dropout rates from model
    weight_mus = [layer.w_mu_q.unsqueeze(0) for layer in model.kl_list]
    if isinstance(model, HorseNet):
        kappas = [layer.get_shrinkage_rates() for layer in model.kl_list]
    # elif isinstance(model, JeffreyNet):
    #     kappas = [layer.get_log_dropout_rates() for layer in model.kl_list]

    return {
        "kl_history": kl_history,
        "loss_history": loss_history,
        "neg_elbo_history": neg_elbo_history,
        "ratio_history": ratio_history,
        "weight_means": weight_mus,
        "shrinkage/dropout_rates": kappas if isinstance(model, (HorseNet)) else None
        #"shrinkage/dropout_rates": kappas if isinstance(model, (HorseNet, JeffreyNet)) else None
    }


def test(model, test_loader, classification=False, deterministic=True, device='cpu', link=None):
    if deterministic:
        model.eval()
    else:
        model.train()
        
    test_loss = 0
    correct = 0

    with torch.no_grad():  # Ensures gradients are not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)

            if classification:
                test_loss += discrimination_loss(output, target, size_average=False).item()
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                test_loss += nn.MSELoss(reduction="sum")(output, target).item()

        test_loss /= len(test_loader.dataset)

    if classification:
        print('Test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    else:
        print('Test loss (MSE): {:.4f}\n'.format(test_loss))
    