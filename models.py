import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
from scipy.linalg import solve
from Layer_classes_clean import MeanFieldHorseshoeLayer_Ghosh, MeanFieldHorseshoeLayer_Ullrich, MeanFieldHorseshoeConv2D

class HorseNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU(), pre_activations=True, tau_0=np.sqrt(1e-1), prune=False):
        """
        A flexible Bayesian neural network with a variable number of hidden layers.
        
        Args:
        - input_dim (int): Dimension of the input features.
        - hidden_dims (list of int): List specifying the number of neurons in each hidden layer.
        - output_dim (int): Dimension of the output layer.
        - activation (torch.nn.Module): Activation function (default: ReLU).
        - pre_activations (bool): Whether to use pre-activation or post-activation layers.
        """
        super(HorseNet, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        self.kl_list = []
        self.thresholds = []

        # Select the correct layer type
        layer_type = MeanFieldHorseshoeLayer_Ghosh if pre_activations else MeanFieldHorseshoeLayer_Ullrich

        # Construct hidden layers dynamically
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = layer_type(prev_dim, hidden_dim, prior_precision=1., tau_0=tau_0,
                               k_a_z=1/2, k_a_s=1/2, k_b_z=1., clip_var=None, prune=prune)
            threshold = layer.get_threshold()
            self.thresholds.append(threshold)
            # with torch.no_grad():
            #     mask = torch.abs(layer.w_mu_q) > threshold
            #     layer.w_mu_q *= mask
            self.layers.append(layer)
            self.kl_list.append(layer)
            
            prev_dim = hidden_dim  # Update input dimension for next layer

        # Output layer
        output_layer = layer_type(prev_dim, output_dim, prior_precision=1., tau_0=np.sqrt(1e-1),
                                  k_a_z=1/2, k_a_s=1/2, k_b_z=1., clip_var=None)
        self.layers.append(output_layer)
        self.kl_list.append(output_layer)

    def forward(self, x):
        self.thresholds = []
        x = x.view(x.size(0), -1)  # Flatten input
        x = x.unsqueeze(0)
        for layer in self.layers[:-1]:  # Apply activation after all hidden layers
            self.thresholds.append(layer.get_threshold())
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # No activation in the output layer
        #x = self.activation(self.layers[-1](x))
        return x.squeeze(0)

    def kl_divergence(self):
        return sum(layer.kl_divergence() for layer in self.kl_list)

    def MC_KL(self):
        return sum(layer.mc_kl_divergence() for layer in self.kl_list)


class Convolutional_Horse(nn.Module):
    """
    A Bayesian Convolutional Neural Network (BCNN) with Horseshoe priors.

    Uses a flexible number of convolutional layers, followed by a fully connected output layer.

    Args:
    - input_channels (int): Number of input image channels (e.g., 3 for RGB images).
    - conv_dims (list of tuples): List of convolutional layer configurations in the form:
      (out_channels, kernel_size, stride, padding).
    - fc_hidden_dims (list of int): Number of neurons in each fully connected hidden layer.
    - output_dim (int): Number of output neurons (e.g., classification classes).
    - activation (torch.nn.Module): Activation function (default: ReLU).
    """
    def __init__(self, input_channels, conv_dims, fc_hidden_dims, output_dim, activation=nn.ReLU(), pre_activations=False, tau_0=np.sqrt(1e-1)):
        super(Convolutional_Horse, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        self.kl_list = []
        self.fc_hidden_dims = fc_hidden_dims

        prev_channels = input_channels

        # Add convolutional layers dynamically
        for out_channels, kernel_size, stride, padding in conv_dims:
            conv_layer = MeanFieldHorseshoeConv2D(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.layers.append(conv_layer)
            self.kl_list.append(conv_layer)
            prev_channels = out_channels  # Update input channels for next layer
            

        # Adaptive pooling to flatten convolutional output to a fixed size (1x1 per channel)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        layer_type = MeanFieldHorseshoeLayer_Ghosh if pre_activations else MeanFieldHorseshoeLayer_Ullrich

        # Fully connected layers
        prev_dim = prev_channels  # The number of channels left after convolution
        for hidden_dim in fc_hidden_dims:
            fc_layer = layer_type(prev_dim, hidden_dim, prior_precision=1., tau_0=tau_0,
                               k_a_z=1/2, k_a_s=1/2, k_b_z=1., clip_var=None)
            self.layers.append(fc_layer)
            self.kl_list.append(fc_layer)
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = layer_type(prev_dim, output_dim, prior_precision=1., tau_0=tau_0,
                               k_a_z=1/2, k_a_s=1/2, k_b_z=1., clip_var=None)
        self.layers.append(self.output_layer)
        self.kl_list.append(self.output_layer)

    def forward(self, x):
        """
        Forward pass:
        1. Applies convolutional layers with activations.
        2. Uses global average pooling to reduce spatial dimensions.
        3. Passes the result through fully connected layers dynamically.
        """
        batch_size = x.shape[0]

        # Apply convolutional layers
        for layer in self.layers[:len(self.kl_list) - len(self.fc_hidden_dims) - 1]:  # Only conv layers
            x = self.activation(layer(x))

        # Global average pooling to flatten feature maps
        x = self.global_avg_pool(x)  # Shape: (batch_size, num_channels, 1, 1)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, num_channels)

        # Apply all fully connected layers
        for layer in self.layers[-(len(self.fc_hidden_dims) + 1):]:  # Select all FC layers + output layer
            x = self.activation(layer(x))

        return x


    def kl_divergence(self):
        """Compute total KL divergence across all layers."""
        return sum(layer.kl_divergence() for layer in self.kl_list)
    
# Sample based horsehoe regression from https://github.com/MansMeg/hslm

class HorseshoeRegression:
    def __init__(self, y, X, iter=2000, intercept=False, ab=(1, 1)):
        self.y = np.asarray(y).reshape(-1, 1)
        self.X = np.asarray(X)
        self.iter = iter
        self.intercept = intercept
        self.ab = ab
        
        if self.intercept:
            self.X = np.column_stack((np.ones(self.X.shape[0]), self.X))
        
        self.n, self.p = self.X.shape
        
        # Initialize storage
        self.beta_samples = np.zeros((self.iter, self.p))
        self.lambda_samples = np.ones((self.iter, self.p))
        self.tau_samples = np.ones(self.iter)
        self.sigma_samples = np.ones(self.iter) if self.ab else np.ones(self.iter) * 1.0
    
    def sample_beta(self, mu_n, Lambda_n_inv, sigma):
        return np.random.multivariate_normal(mean=mu_n.flatten(), cov=(sigma ** 2) * Lambda_n_inv)
    
    def sample_sigma(self, a_n, b_0, yty, mu_n, Lambda_n):
        b_n = b_0 + 0.5 * (yty - mu_n.T @ Lambda_n @ mu_n)
        return np.sqrt(1 / np.random.gamma(a_n, 1 / b_n))
    
    def sample_lambda(self, beta, sigma, tau):
        mu2_j = (beta / (sigma * tau)) ** 2
        gamma_l = np.random.exponential(scale=2 / np.maximum(mu2_j, 1e-6))  # Prevent zero
        return 1 / np.sqrt(np.maximum(gamma_l, 1e-6))  # Avoid division by zero
    
    def sample_tau(self, lambda_, beta, sigma):
        shape_tau = 0.5 * (len(lambda_) + 1)
        mu2_tau = np.sum((beta / (sigma * lambda_)) ** 2)
        gamma_t = np.random.gamma(shape=shape_tau, scale=2 / np.maximum(mu2_tau, 1e-6))  # Prevent zero
        return 1 / np.sqrt(np.maximum(gamma_t, 1e-6))  # Avoid division by zero


    
    def fit(self):
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y
        yty = self.y.T @ self.y
        a_0, b_0 = self.ab
        a_n = a_0 + self.n / 2 if self.ab else None
        
        for it in range(1, self.iter):
            Lambda0 = np.diag(1 / (self.lambda_samples[it - 1, :] ** 2 * self.tau_samples[it - 1] ** 2 + 1e-4)) #np.diag(1 / (self.lambda_samples[it - 1, :] ** 2 * self.tau_samples[it - 1] ** 2))
            Lambda_n = XtX + Lambda0
            Lambda_n_inv = Lambda_n_inv = solve(Lambda_n, np.eye(self.p), assume_a='pos') #np.linalg.pinv(Lambda_n + 1e-6 * np.eye(self.p))
            mu_n = Lambda_n_inv @ Xty
            
            self.beta_samples[it, :] = self.sample_beta(mu_n, Lambda_n_inv, self.sigma_samples[it - 1])
            
            if self.ab:
                self.sigma_samples[it] = self.sample_sigma(a_n, b_0, yty, mu_n, Lambda_n)
            
            self.lambda_samples[it, :] = self.sample_lambda(self.beta_samples[it, :], self.sigma_samples[it], self.tau_samples[it - 1])
            self.tau_samples[it] = self.sample_tau(self.lambda_samples[it, :], self.beta_samples[it, :], self.sigma_samples[it])
        

        return {
            "beta": self.beta_samples[self.iter//4:, :],
            "lambda": self.lambda_samples[self.iter//4:, :],
            "tau": self.tau_samples[self.iter//4:],
            "sigma": self.sigma_samples[self.iter//4:] if self.ab else None,
        }


class HorseshoeGLM:
    def __init__(self, y, X, iter=2000, intercept=False, ab=(1, 1), link="logit", classification=False):
        self.y = np.asarray(y).reshape(-1, 1)
        self.X = np.asarray(X)
        self.iter = iter
        self.intercept = intercept
        self.ab = ab
        self.link = link  # Specify the link function
        self.classify = classification
        
        if self.intercept:
            self.X = np.column_stack((np.ones(self.X.shape[0]), self.X))

        self.n, self.p = self.X.shape
        
        # Initialize storage
        self.beta_samples = np.zeros((self.iter, self.p))
        self.lambda_samples = np.ones((self.iter, self.p))
        self.tau_samples = np.ones(self.iter)
        self.sigma_samples = np.ones(self.iter) if self.ab else np.ones(self.iter) * 1.0

    def inverse_link(self, eta):
        """Applies the inverse link function (g⁻¹(η))."""
        if self.link == "identity":
            return eta
        elif self.link == "logit":
            return 1 / (1 + np.exp(-eta))  # Inverse of logit (sigmoid)
        elif self.link == "log":
            return np.exp(eta)  # Inverse of log-link
        elif self.link == "squared":
            return np.sqrt(eta)  # Inverse of squared-link
        else:
            raise ValueError(f"Unknown link function: {self.link}")


    def sample_beta(self, mu_n, Lambda_n_inv, sigma):
        return np.random.multivariate_normal(mean=mu_n.flatten(), cov=(sigma ** 2) * Lambda_n_inv)

    def sample_sigma(self, a_n, b_0, yty, mu_n, Lambda_n):
        b_n = b_0 + 0.5 * (yty - mu_n.T @ Lambda_n @ mu_n)
        return np.sqrt(1 / np.random.gamma(a_n, 1 / b_n))

    def sample_lambda(self, beta, sigma, tau):
        mu2_j = (beta / (sigma * tau)) ** 2
        gamma_l = np.random.exponential(scale=2 / np.maximum(mu2_j, 1e-6))  # Prevent zero
        return 1 / np.sqrt(np.maximum(gamma_l, 1e-6))  # Avoid division by zero

    def sample_tau(self, lambda_, beta, sigma):
        shape_tau = 0.5 * (len(lambda_) + 1)
        mu2_tau = np.sum((beta / (sigma * lambda_)) ** 2)
        gamma_t = np.random.gamma(shape=shape_tau, scale=2 / np.maximum(mu2_tau, 1e-6))  # Prevent zero
        return 1 / np.sqrt(np.maximum(gamma_t, 1e-6))  # Avoid division by zero

    def fit(self):
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y
        yty = self.y.T @ self.y
        a_0, b_0 = self.ab
        a_n = a_0 + self.n / 2 if self.ab else None
        
        for it in range(1, self.iter):
            Lambda0 = np.diag(1 / (self.lambda_samples[it - 1, :] ** 2 * self.tau_samples[it - 1] ** 2 + 1e-4))
            Lambda_n = XtX + Lambda0
            Lambda_n_inv = solve(Lambda_n, np.eye(self.p), assume_a='pos')
            mu_n = Lambda_n_inv @ Xty
            
            # Sample beta
            self.beta_samples[it, :] = self.sample_beta(mu_n, Lambda_n_inv, self.sigma_samples[it - 1])
            
            # Compute linear predictor and apply inverse link function
            eta = self.X @ self.beta_samples[it, :]
            
            Ey = self.inverse_link(eta)
            
            if self.classify:
                y_pred = np.random.binomial(1, Ey).astype(np.float32)
            else:
                y_pred = Ey

            if self.ab:
                self.sigma_samples[it] = self.sample_sigma(a_n, b_0, yty, mu_n, Lambda_n)

            # Sample lambda and tau
            self.lambda_samples[it, :] = self.sample_lambda(self.beta_samples[it, :], self.sigma_samples[it], self.tau_samples[it - 1])
            self.tau_samples[it] = self.sample_tau(self.lambda_samples[it, :], self.beta_samples[it, :], self.sigma_samples[it])

        return {
            "eta": self.X @ self.beta_samples[-1, :],
            "beta": self.beta_samples[self.iter//4:, :],
            "lambda": self.lambda_samples[self.iter//4:, :],
            "tau": self.tau_samples[self.iter//4:],
            "sigma": self.sigma_samples[self.iter//4:] if self.ab else None,
            "y_pred": self.inverse_link(self.X @ self.beta_samples[-1, :]),
            "link": self.link
        }
