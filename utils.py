import torch
from torch.autograd import Variable
from torch import nn
import properscoring as ps
import matplotlib.pyplot as plt
import numpy as np
discrimination_loss = nn.functional.cross_entropy
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import torch.nn.functional as F




def crps_full_test_set(model, test_loader, n_samples=100):
    """
    Computes the mean CRPS over the full test set using Monte Carlo sampling.
    
    Parameters:
        model: Bayesian neural network (Horseshoe)
        test_loader: DataLoader for the test set
        n_samples: Number of Monte Carlo samples per test input
    
    Returns:
        mean_crps: Mean CRPS over the entire test set
    """
    crps_scores = []

    for data, target in test_loader:
        data, target = Variable(data), target

        # Generate predictive samples for the entire batch
        model.train()
        samples = torch.stack([model(data) for _ in range(n_samples)], dim=0).cpu().detach().numpy()
        model.eval()

        # Compute mean and variance of predictive distribution
        y_mu = samples.mean(axis=0)  # Shape: (batch_size,)
        y_var = samples.var(axis=0)  # Shape: (batch_size,)

        # Compute CRPS for the batch
        batch_crps = ps.crps_gaussian(target.cpu().numpy(), mu=y_mu, sig=np.sqrt(y_var + 1e-8))  # Avoid zero variance
        crps_scores.extend(batch_crps)

    return np.mean(crps_scores)  # Return mean CRPS over dataset


def plot_training_progress(kl_history, loss_history, neg_elbo_history, N, title_suffix=""):
    epochs_range = range(1, len(kl_history) + 1)

    # Convert lists to numpy arrays for plotting
    kl_array = np.array(kl_history) #/ N
    loss_array = torch.tensor(loss_history).cpu().numpy()
    elbo_array = np.array(neg_elbo_history)

    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # KL Divergence Plot
    axs[0].plot(epochs_range, kl_array, label="KL Divergence", marker="o", color='r')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("KL Divergence")
    axs[0].set_title("KL Divergence ↓")
    axs[0].legend()
    axs[0].grid(True)

    # Loss Plot
    axs[1].plot(epochs_range, loss_array, label="Loss", marker="s", color='b')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_title(f"Loss ↓, {title_suffix}")
    axs[1].legend()
    axs[1].grid(True)

    # Negative ELBO Plot
    axs[2].plot(epochs_range, elbo_array, label="Negative ELBO", marker="d", color='g')
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Negative ELBO")
    axs[2].set_title("Negative ELBO ↓")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

def plot_shrinkage(model, model_type="Jeffrey"):
    """
    Plots histograms of log alphas (Jeffrey) or kappa values (Horseshoe) for all layers dynamically.
    
    Parameters:
        model: The Bayesian neural network model.
        model_type: "Jeffrey" for log alphas, "Horseshoe" for shrinkage kappas.
    """
    num_layers = len(model.layers)
    
    if model_type == "Jeffrey":
        # Extract log alphas and convert to numpy
        log_alphas = [layer.get_log_dropout_rates() for layer in model.layers]
        alpha_np = [torch.exp(log_alpha).detach().numpy() for log_alpha in log_alphas]

        # Set up grid size dynamically
        rows = (num_layers + 1) // 2  # Ensure enough rows
        fig, axes = plt.subplots(rows, 2, figsize=(10, 5 * rows))
        axes = axes.flatten() if num_layers > 1 else [axes]
        
        for i, ax in enumerate(axes[:num_layers]):
            counts, bins = np.histogram(alpha_np[i], bins=20)
            ax.bar(bins[:-1], counts, width=(bins[1] - bins[0]), align='edge')
            ax.set_xlabel("Alpha Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Layer {i}: Alpha Values")
    
    elif model_type == "Horseshoe":
        # Extract shrinkage rates and convert to numpy
        kappas = [layer.get_shrinkage_rates() for layer in model.layers]
        kappas_np = [kappa.detach().numpy() for kappa in kappas]

        # Set up grid size dynamically
        rows = (num_layers + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(10, 5 * rows))
        axes = axes.flatten() if num_layers > 1 else [axes]
        
        for i, ax in enumerate(axes[:num_layers]):
            counts, bins = np.histogram(kappas_np[i], bins=20)
            ax.bar(bins[:-1], counts, width=(bins[1] - bins[0]), align='edge')
            ax.set_xlabel("Kappa Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Layer {i}: Kappa Values")
    
    else:
        raise ValueError("Invalid model_type. Choose 'Jeffrey' or 'Horseshoe'.")
    
    plt.tight_layout()
    plt.show()  

    

def plot_regression_results(model, model_name, test_loader, ax=None):
    """
    Plots regression results for a single model with 1D or 2D input.
    
    If the input is 1D, it creates a scatter plot.
    If the input is 2D, it creates a 3D surface plot.

    Parameters:
        model (PyTorch Model): Trained model.
        model_name (str): Name of the model.
        test_loader (DataLoader): DataLoader containing test samples.
        ax (matplotlib axis, optional): Axis to plot on (used for subplots).

    Returns:
        None (displays the plot if `ax` is None).
    """

    # Extract test data
    x_test, y_test = [], []
    
    for x_batch, y_batch in test_loader:
        x_test.append(x_batch.numpy())  # Input features
        y_test.append(y_batch.numpy())  # Ground truth outputs

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Get predictions
    with torch.no_grad():
        y_pred = model(torch.from_numpy(x_test)).cpu().numpy()

    # Check input shape: 1D or 2D function
    input_dim = x_test.shape[1] if len(x_test.shape) > 1 else 1
    
    print(input_dim)

    # **1D Regression Case**
    if input_dim == 1:
        if ax is None:
            plt.figure(figsize=(8, 6))
            plt.scatter(x_test, y_test, label="True Data", color="black", marker=".")
            plt.scatter(x_test, y_pred, label=f"Predictions {model_name}", alpha=0.7, marker="x")
            plt.legend()
            plt.grid()
            plt.xlabel("Input (x)")
            plt.ylabel("Output (y)")
            plt.title(f"Regression Model Predictions (1D) - {model_name}")
            plt.show()
        else:
            ax.scatter(x_test, y_test, label="True Data", color="black", marker=".")
            ax.scatter(x_test, y_pred, label=f"Predictions {model_name}", alpha=0.7, marker="x")
            ax.legend()
            ax.grid()
            ax.set_xlabel("Input (x)")
            ax.set_ylabel("Output (y)")
            ax.set_title(f"Regression (1D) - {model_name}")

    # **2D Regression Case (Bivariate Normal Sampling)**
    elif input_dim == 2:
        x_vals = x_test[:, 0]  # First feature
        y_vals = x_test[:, 1]  # Second feature

        # Create a structured grid for visualization
        num_grid_points = 100
        x_grid = np.linspace(x_vals.min(), x_vals.max(), num_grid_points)
        y_grid = np.linspace(y_vals.min(), y_vals.max(), num_grid_points)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate predictions onto the structured grid
        Z = griddata((x_vals, y_vals), y_pred.flatten(), (X, Y), method='cubic')

        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.7)
            ax.set_xlabel("Input x1")
            ax.set_ylabel("Input x2")
            ax.set_zlabel("Output y")
            ax.set_title(f"Regression Model Predictions (2D) - {model_name}")
            plt.show()
        else:
            ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.7)
            ax.set_xlabel("Input x1")
            ax.set_ylabel("Input x2")
            ax.set_zlabel("Output y")
            ax.set_title(f"Regression (2D) - {model_name}")

    else:
        raise ValueError("This function only supports 1D or 2D input data!")


def generate_y_samples(model, x, n_samples=100):
    """
    Generates predictive samples Y = WH + Bias by running multiple stochastic forward passes.
    
    Parameters:
        model: Bayesian neural network
        x: Input sample (single instance)
        n_samples: Number of Monte Carlo samples
    
    Returns:
        y_samples: Array of predicted outputs
    """
    model.train()
    samples = torch.stack([model(x) for _ in range(n_samples)], dim=0).cpu().detach().numpy()
    model.eval()
    return samples.flatten()  # Convert shape (n_samples, 1) to (n_samples,)

def ecdf(data):
    """ Compute empirical CDF """
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def get_random_test_sample(test_loader):
    """
    Selects a random data sample from the test set.
    
    Parameters:
        test_loader: DataLoader for the test set
    
    Returns:
        x_sample: Random input sample
        y_true: Corresponding target value
    """
    data, target = next(iter(test_loader))  # Load a batch
    idx = random.randint(0, len(data) - 1)  # Random index within the batch
    return Variable(data[idx].unsqueeze(0)), target[idx]

def plot_ecdf(models, model_names, test_loader, n_samples=1000):
    """
    Plots the Empirical Cumulative Distribution Function (ECDF) for Bayesian regression models.
    
    Parameters:
        models (list or torch.nn.Module): List of models or a single model.
        model_names (list or str): Names corresponding to the models.
        test_loader (DataLoader): Test data loader.
        n_samples (int): Number of Monte Carlo samples for ECDF estimation.

    Returns:
        None (Displays the plot)
    """
    # Ensure models and model_names are iterable
    if not isinstance(models, list):
        models = [models]
    if not isinstance(model_names, list):
        model_names = [model_names]

    # Select a random test sample
    x_sample, y_true = get_random_test_sample(test_loader)

    # Store ECDF data
    ecdf_data = {}
    predictive_means = {}

    # Generate predictions and compute ECDF
    for model, name in zip(models, model_names):
        y_samples = generate_y_samples(model, x_sample, n_samples=n_samples)
        x_ecdf, y_ecdf = ecdf(y_samples)
        ecdf_data[name] = (x_ecdf, y_ecdf)
        predictive_means[name] = y_samples.mean()

    # Compute x-axis limits for alignment
    min_x = min(np.min(data[0]) for data in ecdf_data.values())
    max_x = max(np.max(data[0]) for data in ecdf_data.values())

    # Set up the figure
    fig, axs = plt.subplots(len(models), 1, figsize=(12, 2 * len(models)))
    plt.subplots_adjust(hspace=0.5)  # Increase space between subplots

    # Ensure axs is iterable for a single model
    if len(models) == 1:
        axs = [axs]

    # Plot each model's ECDF
    for ax, (name, (x_ecdf, y_ecdf)) in zip(axs, ecdf_data.items()):
        ax.plot(x_ecdf, y_ecdf, label="Predictive ECDF", marker="_")
        ax.axvline(y_true.cpu().numpy(), color='r', linestyle='dashed', label="True y")
        ax.axvline(predictive_means[name], color='b', linestyle='dashed', label="Predictive Mean")
        ax.set_xlim(min_x, max_x)  # Align x-axis across all plots
        ax.set_xlabel("Predicted y")
        ax.set_ylabel("Empirical CDF")
        ax.set_title(f"ECDF of Predictive Distribution - {name}")
        ax.legend()
        ax.grid(True)

    plt.show()
    

def get_predictions(model, dataloader, device="cpu"):
    """
    Compute predictions and ground truth labels in a memory-efficient way.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (str): Device to run inference on ('cpu', 'mps', or 'cuda').

    Returns:
        tuple: (y_true, y_pred) - ground truth labels and predicted labels.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.cpu().numpy()

            outputs = model(images)  # Get logits
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Get predicted class

            y_true.append(labels)
            y_pred.append(preds)

    return np.concatenate(y_true), np.concatenate(y_pred)

def plot_confusion_matrix(models, model_names, dataloader, device="cpu", cmap="BuGn"):
    """
    Plots confusion matrices for multiple models.

    Args:
        models (list): List of trained models.
        model_names (list): List of model names (strings).
        dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (str): Device to run inference on ('cpu', 'mps', or 'cuda').
        cmap (str): Color map for heatmap (default: 'BuGn').

    Returns:
        None (Displays the confusion matrices).
    """
    if len(models) != len(model_names):
        raise ValueError("Number of models and model names must be the same")

    # Compute confusion matrices
    confusion_matrices = []
    for model in models:
        y_true, y_pred = get_predictions(model, dataloader, device)
        cm = confusion_matrix(y_true, y_pred)
        confusion_matrices.append(cm)

    # Plot
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))

    if len(models) == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, cm, name in zip(axes, confusion_matrices, model_names):
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.show()



def plot_roc_auc(model, dataloader, class_index, device="cpu"):
    """
    Compute and plot the ROC-AUC curve for a given class using the One-vs-Rest approach.
    
    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader containing test data.
        class_index: Class index to compute ROC-AUC for (e.g., 3 for digit "3").
        device: "mps", "cuda", or "cpu".
    """
    model.eval()
    y_true, y_scores = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.cpu().numpy()

            outputs = model(images)  # Get logits
            probs = F.softmax(outputs, dim=1).cpu().numpy()  # Convert logits to probabilities

            y_true.append(labels)
            y_scores.append(probs)

    # Convert lists to numpy arrays
    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)

    # Convert labels to binary (One-vs-Rest)
    y_binary = (y_true == class_index).astype(int)
    class_probs = y_scores[:, class_index]

    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_binary, class_probs)
    auc_score = roc_auc_score(y_binary, class_probs)

    # Plot ROC Curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'Class {class_index} (AUC = {auc_score:.6f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line for random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-AUC Curve for Class {class_index}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    

def layer_unitwise_weight_distribution(models, layer_idx=0, pre_activation=True, thresholds=None):
    """
    Generates a violin plot of weight distributions per unit (neuron) in a specific layer, 
    with units sorted by deviation from zero.
    """
    plt.close('all')  
    fig, ax = plt.subplots(figsize=(15, 8))

    for model in models:
        weights = model.layers[layer_idx].w_mu_q.detach().cpu().numpy()  

        if pre_activation:
            weight_distributions = [weights[i, :] for i in range(weights.shape[0])]
            unit_means = np.array([np.mean(w) for w in weight_distributions])
        else:
            weight_distributions = [weights[:, j] for j in range(weights.shape[1])]
            unit_means = np.array([np.mean(w) for w in weight_distributions])

        sorted_indices = np.argsort(-np.abs(unit_means))  
        weight_distributions = [weight_distributions[i] for i in sorted_indices]
        unit_positions = np.arange(len(sorted_indices))

        ax.violinplot(weight_distributions, positions=unit_positions, vert=False, showmedians=True)

        if thresholds is not None:
            model_thresholds = np.array(thresholds[0])[sorted_indices]  
            ax.scatter(model_thresholds, unit_positions, color='black', marker='o', s=10)

    ax.set_xlabel("Weight Distribution")
    ax.set_ylabel("Neurons in Layer (Sorted)")
    ax.set_yticks(unit_positions)
    ax.set_yticklabels([str(i+1) for i in sorted_indices])
    plt.show()
    

def prune_model(model, threshold=1e-2):
    """
    Prune small weights in the Bayesian Neural Network after training.

    Args:
    - model (HorseNet_flex): The trained BNN model.
    - threshold (float): The absolute weight value below which weights will be set to zero.
    
    Returns:
    - None (modifies model in-place)
    """
    with torch.no_grad():  # Ensure no gradients are tracked
        for i, layer in enumerate(model.layers):
            #if isinstance(layer, MeanFieldHorseshoeLayer_Post_Activation):
            # Get the absolute weights
            abs_weights = torch.abs(layer.w_mu_q.data)

            # Create mask: 1 if above threshold, 0 otherwise
            mask = (abs_weights >= threshold).float()

            # Apply mask to set pruned weights to zero
            layer.w_mu_q.data *= mask

            # Print pruning summary
            total_weights = layer.w_mu_q.numel()
            pruned_weights = (mask == 0).sum().item()
            prune_ratio = pruned_weights / total_weights * 100
            print(f"Layer {i}: Pruned {prune_ratio:.2f}% weights ({pruned_weights}/{total_weights})")
            


def plot_weights(model):
    for i, layer in enumerate(model.layers):
        #if isinstance(layer, MeanFieldHorseshoeLayer_Post_Activation):
        print(f"Layer {i} weights:")
        #print(layer.w_mu_q.data)  # This should show zeroed-out weights
        plt.hist(layer.w_mu_q.data.numpy().flatten(), bins=100)
        plt.show()
        
    

def compare_models(models_dict, results, X_test, y_test, test_loader, link):
    """
    Compare Bayesian Linear Regression and multiple Bayesian Neural Networks.

    Parameters:
    - models_dict: A dictionary where keys are model names and values are PyTorch models.
    - results: The output dictionary from Bayesian Linear Regression (contains 'beta').
    - X_test: Test feature matrix (NumPy array).
    - y_test: Test target values (NumPy array).
    - test_loader: DataLoader for the test set.
    
    Returns:
    - None (plots predictions and prints MSE)
    """

    if link == "identity":
        apply_link = lambda x: x
    elif link == "log":
        apply_link = lambda x: np.exp(x)
    elif link == "logit":
        apply_link = lambda x: 1 / (1 + np.exp(-x)) 
    else:
        raise ValueError("Invalid link function. Choose either 'identity' or 'log'.")
    
    # Compute posterior mean of beta coefficients for Bayesian Linear Regression
    beta_mean = np.mean(results["beta"], axis=0)
    
    # Compute predictions for Bayesian Linear Regression
    y_pred_lr = apply_link(X_test @ beta_mean)
    
    # Function to generate predictions from a PyTorch model
    def get_predictions(model, test_loader):
        model.eval()
        y_pred_list = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                y_pred_temp = model(X_batch).squeeze()
                y_pred_list.append(y_pred_temp.cpu().numpy())
        return np.concatenate(y_pred_list, axis=0)

    # Compute predictions for all models in `models_dict`
    y_pred_models = {name: get_predictions(model, test_loader) for name, model in models_dict.items()}

    # Create subplots for visualization
    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), sharey=True)

    if num_models == 1:
        axes = [axes]  # Ensure `axes` is iterable if there's only one model

    # Plot each model's predictions
    for ax, (model_name, y_pred_nn) in zip(axes, y_pred_models.items()):
        ax.scatter(y_test, y_pred_lr, alpha=0.5, label="Predicted_LR vs True")
        ax.scatter(y_test, y_pred_nn, alpha=0.5, label=f"Predicted_{model_name} vs True")
        #ax.plot([min(y_pred_lr), max(y_pred_lr)], [min(y_pred_lr), max(y_pred_lr)], 'r--', label="Perfect Fit")
        #ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Perfect Fit")
        ax.set_xlabel("True Values")
        ax.set_title(f"{model_name}")
        ax.legend()
        ax.grid(True)

    # Set shared ylabel
    axes[0].set_ylabel("Predicted Values")

    # Show the figure
    plt.suptitle("Comparison of Bayesian Linear Regression and Bayesian Neural Networks")
    plt.tight_layout()
    plt.show()

    # Print MSE for each model
    print("MSE (Bayesian Linear Regression):", np.mean((y_test - y_pred_lr) ** 2))
    for model_name, y_pred_nn in y_pred_models.items():
        print(f"MSE ({model_name}):", np.mean((y_test - y_pred_nn) ** 2))



def compare_shrinkage_rates(models_dict, results):
    """
    Compare shrinkage rates (kappa) from Bayesian Linear Regression and Bayesian Neural Networks.

    Parameters:
    - models_dict: A dictionary where keys are model names and values are PyTorch models.
    - results: The output dictionary from Bayesian Linear Regression (contains 'lambda').

    Returns:
    - None (plots histograms of shrinkage rates)
    """
    
    # Compute shrinkage rate (kappa) for Bayesian Linear Regression
    kappa_linreg = 1 / (1 + results["lambda"]**2)

    # Function to extract shrinkage rates from a model
    def get_kappa_from_model(model):
        return [layer.get_shrinkage_rates().detach().numpy() for layer in model.layers]

    # Compute shrinkage rates for all models in `models_dict`
    kappa_models = {name: get_kappa_from_model(model) for name, model in models_dict.items()}

    # Create subplots for visualization
    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), sharey=True)

    if num_models == 1:
        axes = [axes]  # Ensure `axes` is iterable if there's only one model

    # Plot each model's shrinkage rates
    for ax, (model_name, kappa_bnn) in zip(axes, kappa_models.items()):
        ax.hist(kappa_linreg[1499], bins=10, alpha=0.5, label="Linear Regression")
        ax.hist(kappa_bnn[0], bins=50, alpha=0.5, label=model_name)
        ax.set_xlabel("Shrinkage Rate (kappa)")
        ax.set_title(f"{model_name}")
        ax.legend()
        ax.grid(True)

    # Set shared ylabel
    axes[0].set_ylabel("Frequency")

    # Show the figure
    plt.suptitle("Comparison of Shrinkage Rates: Bayesian Linear Regression vs BNNs")
    plt.tight_layout()
    plt.show()
