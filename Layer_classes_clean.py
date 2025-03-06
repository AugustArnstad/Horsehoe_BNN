import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.distributions import InverseGamma
import autograd.numpy.random as npr


def reparametrize(mu, logvar, sampling=True):
    if sampling:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
    else:
        return mu
    
def diag_gaussian_entropy(log_std, D):
    return 0.5 * D * (1.0 + torch.log(torch.tensor(2 * np.pi, device=log_std.device))) + torch.sum(log_std)
  
def inv_gamma_entropy(a, b):
    # Ensure a and b are PyTorch tensors on the correct device
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)  # Convert to tensor if it's a float
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32)  # Convert to tensor if it's a float

    return torch.sum(a + torch.log(b) + torch.lgamma(a) - (1 + a) * torch.digamma(a))

def log_normal_entropy(log_std, mu, D):
    return torch.sum(log_std + mu + 0.5) + (D / 2) * torch.log(torch.tensor(2 * np.pi, device=log_std.device))


# --- Layer Implementation ---
# Note that Ghosh considers all weights INCIDENT to a node j in layer l from a node i in layer l-1 to get the shrinkage tau_j
# while Ullrich considers all weights EMANATING from a node i in layer l-1 to a node j in layer l to get the shrinkage tau_i
# This affects mainly the dimension of the local shrinkage tensor z_mu and z_logvar

class MeanFieldHorseshoeLayer_Ghosh(nn.Module):
    """
    A linear layer with a Horseshoe prior (noncentered parameterization).

    Variational posteriors:
      - Base weight:  q(w) = N(w; w_mu, exp(0.5 * w_logvar))
      - Local scale:   q(λ) = LN(λ; lambda_mu, exp(0.5 * lambda_logvar))
      - Global scale:  q(τ) = LN(τ; tau_mu, exp(0.5 * tau_logvar))

    The effective weight is computed as:
         w_eff = (τ * λ) ⊙ w
    """
    def __init__(self, in_features, out_features, prior_precision=1., tau_0=np.sqrt(1), k_a_z=0.5, k_a_s=0.5, k_b_z=1., clip_var=None, prune=False):
        super(MeanFieldHorseshoeLayer_Ghosh, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.w_mu_q = Parameter(torch.Tensor(out_features, in_features))
        self.w_logvar = Parameter(torch.Tensor(out_features, in_features))

        self.z_a_prior = k_a_z
        self.z_b_prior = k_b_z  
        self.s_a_prior = k_a_s
        self.s_b_prior = tau_0

        self.bias_mu_q = Parameter(torch.Tensor(out_features))
        self.bias_logvar = Parameter(torch.Tensor(out_features))
        
        self.clip_var = clip_var
        
        self.threshold = None
        
        self.reset_parameters()

    
    def reset_parameters(self):
        stdv = math.sqrt(2.) / math.sqrt(self.in_features)
        self.w_mu_q.data.normal_(0, stdv)
        self.w_logvar.data.normal_(-5.0, 1)
        
        self.initialize_scale_from_prior()
        
        self.bias_mu_q.data.zero_()
        self.bias_logvar.data.fill_(-8.0)
        
    def initialize_scale_from_prior(self):
        self.alpha_z = torch.tensor(self.z_a_prior, dtype=torch.float32).expand(self.out_features)
        self.lambda_z = torch.distributions.InverseGamma(self.alpha_z, 1.0 / self.z_b_prior**2).sample()
        
        self.alpha_s = torch.tensor(self.s_a_prior, dtype=torch.float32)
        self.lambda_s = torch.distributions.InverseGamma(self.alpha_s, 1.0 / self.s_b_prior**2).sample()
        
        sample_z = torch.abs(self.z_b_prior * torch.tensor(
            npr.randn(self.out_features) / npr.randn(self.out_features), dtype=torch.float32
        ))
        self.z_mu = nn.Parameter(torch.log(torch.sqrt(sample_z)), requires_grad=True)
            
        sample_s = torch.abs(self.z_b_prior * torch.tensor(
            npr.randn(1) / npr.randn(1), dtype=torch.float32
        ))
        self.s_mu = nn.Parameter(torch.log(torch.sqrt(sample_s)), requires_grad=True)
        
        self.z_logvar = nn.Parameter(torch.tensor((npr.randn(self.out_features) - 10.), dtype=torch.float32), requires_grad=True)
        self.s_logvar = nn.Parameter(torch.tensor((npr.randn(1) - 10.), dtype=torch.float32), requires_grad=True)

    
    def forward(self, x):
        scale_mu = 0.5 * (self.z_mu + self.s_mu.expand_as(self.z_mu))
        
        scale_logvar = torch.log(0.25 * (self.z_logvar.exp() + self.s_logvar.exp().expand_as(self.z_logvar)))
        
        scale = torch.exp(reparametrize(scale_mu, scale_logvar, sampling=self.training))

        mu_w = x @ self.w_mu_q.t() + self.bias_mu_q
        v_w = x.pow(2) @ self.w_logvar.exp().t() + self.bias_logvar.exp()
        out = (v_w.sqrt() / x.shape[1]**0.5) * torch.randn_like(mu_w) + mu_w
        
        output = out * scale
        
        self.threshold = scale_logvar.exp() - scale_mu
        return output

    
    def get_shrinkage_rates(self):
        
        scale_mu = 0.5 * (self.z_mu + self.s_mu.expand_as(self.z_mu))
        
        scale_logvar = torch.log(0.25 * (self.z_logvar.exp() + self.s_logvar.exp().expand_as(self.z_logvar)))
        
        scale = torch.exp(reparametrize(scale_mu, scale_logvar, sampling=self.training))
        
        return 1/(1+scale**2)
    

    def get_threshold(self):
        if self.threshold is None: 
            with torch.no_grad():
                scale_mu = 0.5 * (self.z_mu + self.s_mu.expand_as(self.z_mu))
                scale_logvar = torch.log(0.25 * (self.z_logvar.exp() + self.s_logvar.exp().expand_as(self.z_logvar)))
                self.threshold = scale_logvar.exp() - scale_mu  # Compute threshold manually
        return self.threshold

    
    def compute_posterior_params(self):
        
        scale_mu = 0.5 * (self.z_mu + self.s_mu.expand_as(self.z_mu))
        
        scale_logvar = torch.log(0.25 * (self.z_logvar.exp() + self.s_logvar.exp().expand_as(self.z_logvar)))
        
        
        self.post_weight_var = (torch.exp(scale_logvar.exp()) - 1)*torch.exp(2*scale_mu + scale_logvar.exp())*(self.w_logvar.exp()+self.w_mu_q)+self.w_logvar.exp()*torch.exp(2*scale_mu+scale_logvar.exp())
        self.post_weight_mu = self.w_mu_q * (torch.exp(scale_mu + 0.5 * scale_logvar.exp()))
        return self.post_weight_mu, self.post_weight_var
    
    def clip_variances(self):
        if self.clip_var:
            self.w_logvar.data.clamp_(max=math.log(self.clip_var))
            self.bias_logvar.data.clamp_(max=math.log(self.clip_var))
    
    def entropy(self):
        entropy_w = diag_gaussian_entropy(0.5 * self.w_logvar.detach(), self.w_mu_q.numel()) 
        entropy_bias = diag_gaussian_entropy(0.5 * self.bias_logvar.detach(), self.bias_mu_q.numel())
        
        entropy_z = log_normal_entropy(0.5 * self.z_logvar.detach(), self.z_mu.detach(), self.z_mu.numel())
        entropy_s = log_normal_entropy(0.5 * self.s_logvar.detach(), self.s_mu.detach(), self.s_mu.numel())
        
        entropy_lambda_z = inv_gamma_entropy(self.alpha_z.detach(), 1.0 / self.z_b_prior**2)
        entropy_lambda_s = inv_gamma_entropy(self.alpha_s.detach(), 1.0 / self.s_b_prior**2)
        
        H = entropy_w + entropy_bias + entropy_z + entropy_s + entropy_lambda_z + entropy_lambda_s
        
        return H

    
    def EPw_Gaussian(self, prior_precision=1.0):
        """
        Computes E_q[ln p(w)] for a Gaussian prior p(w) = N(0, 1/prior_precision)
        """
        device = self.w_mu_q.device  
        D = self.w_mu_q.numel()  
        zs = torch.tensor(1.0 / prior_precision, dtype=torch.float32, device=device)  # Ensure zs is a tensor

        a = -0.5 * D * torch.log(torch.tensor(2 * np.pi, device=device)) \
            - 0.5 * D * torch.log(zs**2) \
            - 0.5 * (torch.dot(self.w_mu_q.view(-1), self.w_mu_q.view(-1)) + torch.sum(self.w_logvar.exp() ** 2)) / (zs ** 2)

        return a

    
    def EP_Gamma(self, Egamma, Elog_gamma):
        """ Computes E_q[ln p(γ)] for a Gamma prior """
        device = self.w_mu_q.device

        return self.noise_a * torch.log(torch.tensor(self.noise_b, dtype=torch.float32, device=device)) \
            - torch.lgamma(torch.tensor(self.noise_a, dtype=torch.float32, device=device)) \
            + (-self.noise_a - 1) * Elog_gamma - self.noise_b * Egamma

    
    def EPtaulambda(self, tau_mu, tau_sigma, tau_a_prior=0.5, lambda_a_prior=0.5, lambda_b_prior=1):
        """ Computes E[ln p(\tau | \lambda)] + E[ln p(\lambda)] """

        device = self.w_mu_q.device

        # Ensure inputs are tensors
        lambda_b_prior_sq_inv = torch.tensor(1 / (lambda_b_prior ** 2), dtype=torch.float32, device=device)
        tau_a_prior = torch.tensor(tau_a_prior, dtype=torch.float32, device=device)
        lambda_a_prior = torch.tensor(lambda_a_prior, dtype=torch.float32, device=device)

        etau_given_lambda = (
            - torch.lgamma(tau_a_prior)  
            - tau_a_prior * (torch.log(lambda_b_prior_sq_inv) - torch.digamma(lambda_a_prior)) 
            + (-tau_a_prior - 1.) * tau_mu
            - torch.exp(-tau_mu + 0.5 * tau_sigma ** 2) * (lambda_a_prior / lambda_b_prior_sq_inv)
        )

        elambda = (
            - torch.lgamma(lambda_a_prior)  
            + lambda_a_prior * torch.log(lambda_b_prior_sq_inv)  
            + (-lambda_a_prior - 1.) * (torch.log(lambda_b_prior_sq_inv) - torch.digamma(lambda_a_prior))
            - lambda_a_prior
        )

        return torch.sum(etau_given_lambda) + torch.sum(elambda)

    def kl_divergence(self):
        H_q = self.entropy()
        
        E_weights = self.EPw_Gaussian()

        E_z = self.EPtaulambda(self.z_mu, self.z_logvar.exp(), self.z_a_prior, self.alpha_z, self.z_b_prior)
        E_s = self.EPtaulambda(self.s_mu, self.s_logvar.exp(), self.s_a_prior, self.alpha_s, self.s_b_prior)

        
        KL = - E_weights - E_z - E_s - H_q
        
        return KL

        
    def mc_kl_divergence(self, num_samples=10):
        """ Computes the KL divergence using Monte Carlo sampling instead of analytical expectations. """
        device = self.w_mu_q.device
        H_q = self.entropy()  # Entropy term

        kl_samples = []

        for _ in range(num_samples):
            w_sample = self.w_mu_q + torch.randn_like(self.w_mu_q) * torch.exp(0.5 * self.w_logvar)

            # Expand z_sample across input dimension (columns)
            z_sample = torch.exp(self.z_mu + torch.randn_like(self.z_mu) * torch.exp(0.5 * self.z_logvar))  
            z_sample = z_sample.view(-1, 1).expand(self.out_features, self.in_features)  

            # Expand s_sample across the entire weight matrix
            s_sample = torch.exp(self.s_mu + torch.randn_like(self.s_mu) * torch.exp(0.5 * self.s_logvar))  
            s_sample = s_sample.expand_as(w_sample)

            # Compute log prior p(w | z, s) with correctly shaped tensors
            log_p_w = -0.5 * torch.log(torch.tensor(2 * np.pi, device=device)) - 0.5 * torch.log((z_sample * s_sample) ** 2) - 0.5 * (w_sample ** 2) / (z_sample * s_sample) ** 2
            log_p_w = log_p_w.sum()

            # Compute log prior p(z)
            log_p_z = -0.5 * torch.log(torch.tensor(self.z_b_prior ** 2, device=device)) - (3 / 2) * torch.log(z_sample) - torch.tensor((1 / self.z_b_prior ** 2), device=device) * (1 / (2 * z_sample))
            log_p_z = log_p_z.sum()

            # Compute log prior p(s)
            log_p_s = -0.5 * torch.log(torch.tensor(float(self.s_b_prior ** 2), dtype=torch.float32, device=device)) - (3 / 2) * torch.log(s_sample) - torch.tensor((1 / self.s_b_prior ** 2), dtype=torch.float32, device=device) * (1 / (2 * s_sample))
            log_p_s = log_p_s.sum()

            # Compute total log-prior
            log_p_theta = log_p_w + log_p_z + log_p_s
            kl_samples.append(log_p_theta)

        # Compute expectation via Monte Carlo
        E_q_log_p = torch.stack(kl_samples).mean()

        # Compute final KL divergence
        KL = -E_q_log_p - H_q
        
        return KL

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features})"



class MeanFieldHorseshoeLayer_Ullrich(nn.Module):
    """
    A linear layer with a Horseshoe prior (noncentered parameterization).

    Variational posteriors:
      - Base weight:  q(w) = N(w; w_mu, exp(0.5 * w_logvar))
      - Local scale:   q(λ) = LN(λ; lambda_mu, exp(0.5 * lambda_logvar))
      - Global scale:  q(τ) = LN(τ; tau_mu, exp(0.5 * tau_logvar))

    The effective weight is computed as:
         w_eff = (τ * λ) ⊙ w
    """
    def __init__(self, in_features, out_features, prior_precision=1., tau_0=np.sqrt(1), k_a_z=0.5, k_a_s=0.5, k_b_z=1., clip_var=None, prune=False):
        super(MeanFieldHorseshoeLayer_Ullrich, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.w_mu_q = Parameter(torch.Tensor(out_features, in_features))
        self.w_logvar = Parameter(torch.Tensor(out_features, in_features))

        self.z_a_prior = k_a_z
        self.z_b_prior = k_b_z  
        self.s_a_prior = k_a_s
        self.s_b_prior = tau_0
        
        self.prune = prune
        
        self.bias_mu_q = Parameter(torch.Tensor(out_features))
        self.bias_logvar = Parameter(torch.Tensor(out_features))
        
        self.clip_var = clip_var
        
        self.threshold = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = math.sqrt(2.) / math.sqrt(self.in_features)
        self.w_mu_q.data.normal_(0, stdv)
        self.w_logvar.data.normal_(-5.0, 1)
        
        self.initialize_scale_from_prior()
        
        self.bias_mu_q.data.zero_()
        self.bias_logvar.data.fill_(-8.0)
        
    def initialize_scale_from_prior(self):
        self.alpha_z = torch.tensor(self.z_a_prior, dtype=torch.float32).expand(self.in_features)
        self.lambda_z = torch.distributions.InverseGamma(self.alpha_z, 1.0 / self.z_b_prior**2).sample()
        
        self.alpha_s = torch.tensor(self.s_a_prior, dtype=torch.float32)
        self.lambda_s = torch.distributions.InverseGamma(self.alpha_s, 1.0 / self.s_b_prior**2).sample()
        
        sample_z = torch.abs(self.z_b_prior * torch.tensor(
            npr.randn(self.in_features) / npr.randn(self.in_features), dtype=torch.float32
        ))
        self.z_mu = nn.Parameter(torch.log(torch.sqrt(sample_z)), requires_grad=True)
            
        sample_s = torch.abs(self.z_b_prior * torch.tensor(
            npr.randn(1) / npr.randn(1), dtype=torch.float32
        ))
        self.s_mu = nn.Parameter(torch.log(torch.sqrt(sample_s)), requires_grad=True)
        
        self.z_logvar = nn.Parameter(torch.tensor((npr.randn(self.in_features) - 10.), dtype=torch.float32), requires_grad=True)
        self.s_logvar = nn.Parameter(torch.tensor((npr.randn(1) - 10.), dtype=torch.float32), requires_grad=True)

    
    def forward(self, x):
        scale_mu = 0.5 * (self.z_mu + self.s_mu.expand_as(self.z_mu))
        
        scale_logvar = torch.log(0.25 * (self.z_logvar.exp() + self.s_logvar.exp().expand_as(self.z_logvar)))
        
        scale = torch.exp(reparametrize(scale_mu, scale_logvar, sampling=self.training))
            
        H_hat = x*scale

        w_mu = F.linear(H_hat, self.w_mu_q, self.bias_mu_q)
        w_var = F.linear(H_hat.pow(2), self.w_logvar.exp(), self.bias_logvar.exp())
        
        self.threshold = scale_logvar.exp() - scale_mu
        
        return reparametrize(w_mu, w_var.log(), sampling=self.training)

    
    def get_shrinkage_rates(self):
        
        scale_mu = 0.5 * (self.z_mu + self.s_mu.expand_as(self.z_mu))
        
        scale_logvar = torch.log(0.25 * (self.z_logvar.exp() + self.s_logvar.exp().expand_as(self.z_logvar)))
        
        scale = torch.exp(reparametrize(scale_mu, scale_logvar, sampling=self.training))
        
        return 1/(1+scale**2)

    def get_threshold(self):
        return self.threshold
    
    def compute_posterior_params(self):
        
        scale_mu = 0.5 * (self.z_mu + self.s_mu.expand_as(self.z_mu))
        
        scale_logvar = torch.log(0.25 * (self.z_logvar.exp() + self.s_logvar.exp().expand_as(self.z_logvar)))
        
        
        self.post_weight_var = (torch.exp(scale_logvar.exp()) - 1)*torch.exp(2*scale_mu + scale_logvar.exp())*(self.w_logvar.exp()+self.w_mu_q)+self.w_logvar.exp()*torch.exp(2*scale_mu+scale_logvar.exp())
        self.post_weight_mu = self.w_mu_q * (torch.exp(scale_mu + 0.5 * scale_logvar.exp()))
        return self.post_weight_mu, self.post_weight_var
    
    def clip_variances(self):
        if self.clip_var:
            self.w_logvar.data.clamp_(max=math.log(self.clip_var))
            self.bias_logvar.data.clamp_(max=math.log(self.clip_var))
    
    def entropy(self):
        entropy_w = diag_gaussian_entropy(0.5 * self.w_logvar.detach(), self.w_mu_q.numel()) 
        entropy_bias = diag_gaussian_entropy(0.5 * self.bias_logvar.detach(), self.bias_mu_q.numel())
        
        entropy_z = log_normal_entropy(0.5 * self.z_logvar.detach(), self.z_mu.detach(), self.z_mu.numel())
        entropy_s = log_normal_entropy(0.5 * self.s_logvar.detach(), self.s_mu.detach(), self.s_mu.numel())
        
        entropy_lambda_z = inv_gamma_entropy(self.alpha_z.detach(), 1.0 / self.z_b_prior**2)
        entropy_lambda_s = inv_gamma_entropy(self.alpha_s.detach(), 1.0 / self.s_b_prior**2)
        
        H = entropy_w + entropy_bias + entropy_z + entropy_s + entropy_lambda_z + entropy_lambda_s
        
        return H
    
    def EPw_Gaussian(self, prior_precision=1.0):
        """
        Computes E_q[ln p(w)] for a Gaussian prior p(w) = N(0, 1/prior_precision)
        """
        device = self.w_mu_q.device
        D = self.w_mu_q.numel()  # Number of elements in the weight tensor
        zs = torch.tensor(1.0 / prior_precision, dtype=torch.float32, device=device)  # Prior standard deviation

        a = -0.5 * D * torch.log(torch.tensor(2 * torch.pi, device=self.w_mu_q.device)) \
            - 0.5 * D * torch.log(torch.tensor(zs**2, device=self.w_mu_q.device)) \
            - 0.5 * (torch.dot(self.w_mu_q.view(-1), self.w_mu_q.view(-1)) + torch.sum(self.w_logvar.exp() ** 2)) / (zs ** 2)
        
        return a
    
    def EPtaulambda(self, tau_mu, tau_sigma, tau_a_prior=0.5, lambda_a_prior=0.5, lambda_b_prior=1):
        """ Computes E[ln p(\tau | \lambda)] + E[ln p(\lambda)] """

        device = self.w_mu_q.device

        lambda_b_prior_sq_inv = torch.tensor(1 / (lambda_b_prior ** 2), dtype=torch.float32, device=device)
        tau_a_prior = torch.tensor(tau_a_prior, dtype=torch.float32, device=device)
        lambda_a_prior = torch.tensor(lambda_a_prior, dtype=torch.float32, device=device)

        etau_given_lambda = (
            - torch.lgamma(tau_a_prior)  
            - tau_a_prior * (torch.log(lambda_b_prior_sq_inv) - torch.digamma(lambda_a_prior)) 
            + (-tau_a_prior - 1.) * tau_mu
            - torch.exp(-tau_mu + 0.5 * tau_sigma ** 2) * (lambda_a_prior / lambda_b_prior_sq_inv)
        )

        elambda = (
            - torch.lgamma(lambda_a_prior)  
            + lambda_a_prior * torch.log(lambda_b_prior_sq_inv)  
            + (-lambda_a_prior - 1.) * (torch.log(lambda_b_prior_sq_inv) - torch.digamma(lambda_a_prior))
            - lambda_a_prior
        )

        return torch.sum(etau_given_lambda) + torch.sum(elambda)
        
        
    def kl_divergence(self):
        H_q = self.entropy()
        
        E_weights = self.EPw_Gaussian()

        E_z = self.EPtaulambda(self.z_mu, self.z_logvar.exp(), self.z_a_prior, self.alpha_z, self.z_b_prior)
        E_s = self.EPtaulambda(self.s_mu, self.s_logvar.exp(), self.s_a_prior, self.alpha_s, self.s_b_prior)
        
        KL = - E_weights - E_z - E_s - H_q
        
        return KL
    
    
    def mc_kl_divergence(self, num_samples=10):
        """
        Computes the KL divergence using Monte Carlo sampling instead of analytical expectations.
        
        Parameters:
            num_samples (int): Number of Monte Carlo samples.
        
        Returns:
            KL estimate using MC samples.
        """
        device = self.w_mu_q.device
        H_q = self.entropy()  # Entropy term

        kl_samples = []

        for _ in range(num_samples):

            w_sample = self.w_mu_q + torch.randn_like(self.w_mu_q) * torch.exp(0.5 * self.w_logvar)

            
            z_sample = torch.exp(self.z_mu + torch.randn_like(self.z_mu) * torch.exp(0.5 * self.z_logvar))  
            z_sample = z_sample.view(1, -1).expand(self.out_features, self.in_features)  # (1, d_{l-1}) → (d_l, d_{l-1})

            
            # Expand s_sample across the entire weight matrix
            s_sample = torch.exp(self.s_mu + torch.randn_like(self.s_mu) * torch.exp(0.5 * self.s_logvar))  # Scalar
            s_sample = s_sample.expand_as(w_sample)  # Now shape: (d_{l-1}, d_l)

            # Compute log prior p(w | z, s) with correctly shaped tensors
            log_p_w = -0.5 * torch.log(2 * torch.pi * (z_sample * s_sample) ** 2) - 0.5 * (w_sample ** 2) / (z_sample * s_sample) ** 2
            log_p_w = log_p_w.sum()

            # Compute log prior p(z)
            log_p_z = -0.5 * torch.log(torch.tensor(self.z_b_prior ** 2, device=device)) - (3 / 2) * torch.log(z_sample) - torch.tensor((1 / self.z_b_prior ** 2)) * (1 / (2 * z_sample))
            log_p_z = log_p_z.sum()

            # Compute log prior p(s)
            log_p_s = -0.5 * torch.log(torch.tensor(float(self.s_b_prior ** 2), dtype=torch.float32, device=device)) - (3 / 2) * torch.log(s_sample) - torch.tensor((1 / self.s_b_prior ** 2), dtype=torch.float32, device=device) * (1 / (2 * s_sample))
            log_p_s = log_p_s.sum()

            # Compute total log-prior
            log_p_theta = log_p_w + log_p_z + log_p_s
            kl_samples.append(log_p_theta)

        # Compute expectation via Monte Carlo
        E_q_log_p = torch.stack(kl_samples).mean()

        # Compute final KL divergence
        KL = -E_q_log_p - H_q
        
        return KL
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features})"
    


class MeanFieldHorseshoeConv2D(nn.Module):
    """
    Convolutional layer with a Horseshoe prior following Algorithm 4.

    Shrinkage is applied per **output feature map**, meaning all weights
    contributing to the same output channel share the same shrinkage factor.

    Uses local reparameterization for variance reduction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, clip_var=None,
                 tau_0=np.sqrt(1), k_a_z=0.5, k_a_s=0.5, k_b_z=1.):
        super(MeanFieldHorseshoeConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.clip_var = clip_var

        # Variational Parameters
        self.w_mu_q = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.w_logvar = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        
        self.z_a_prior = k_a_z
        self.z_b_prior = k_b_z  
        self.s_a_prior = k_a_s
        self.s_b_prior = tau_0
        
        self.bias_mu_q = nn.Parameter(torch.Tensor(out_channels))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_channels))
        
        # **Shrinkage scale per output feature map**
        self.z_mu = nn.Parameter(torch.ones(out_channels, 1, 1, 1))  
        self.z_logvar = nn.Parameter(torch.full((out_channels, 1, 1, 1), -9.0))
        
        self.s_mu = nn.Parameter(torch.ones(1, 1, 1, 1))  
        self.s_logvar = nn.Parameter(torch.full((1, 1, 1, 1), -9.0))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.w_mu_q.data.normal_(0, stdv)
        self.bias_mu_q.data.zero_()
        self.w_logvar.data.fill_(-9.0)
        self.bias_logvar.data.fill_(-9.0)
        self.initialize_scale_from_prior()
        # self.z_mu.data.normal_(1, 1e-2)
        # self.z_logvar.data.normal_(-3, 1e-2)
        # self.s_mu.data.normal_(1, 1e-2)
        # self.s_logvar.data.normal_(-3, 1e-2)
        

    def initialize_scale_from_prior(self):
        """Initialize z_mu and s_mu using the prior distribution."""
        self.alpha_z = torch.tensor(self.z_a_prior, dtype=torch.float32).expand(self.out_channels)
        self.lambda_z = torch.distributions.InverseGamma(self.alpha_z, 1.0 / self.z_b_prior**2).sample()
        
        self.alpha_s = torch.tensor(self.s_a_prior, dtype=torch.float32)
        self.lambda_s = torch.distributions.InverseGamma(self.alpha_s, 1.0 / self.s_b_prior**2).sample()
        
        # Sample z using an Inverse Gamma prior, then reshape to match (out_channels, 1, 1, 1)
        sample_z = torch.abs(self.z_b_prior * torch.randn(self.out_channels) / torch.randn(self.out_channels))
        sample_z = torch.log(torch.sqrt(sample_z)).view(self.out_channels, 1, 1, 1)  # Ensure correct shape
        
        # Sample s using an Inverse Gamma prior, then reshape to match (1, 1, 1, 1)
        sample_s = torch.abs(self.z_b_prior * torch.randn(1) / torch.randn(1))
        sample_s = torch.log(torch.sqrt(sample_s)).view(1, 1, 1, 1)  # Ensure correct shape
        
        # Assign parameters with correct shapes
        self.z_mu = nn.Parameter(sample_z, requires_grad=True)
        self.s_mu = nn.Parameter(sample_s, requires_grad=True)
        
        # Initialize log-variances with correct shapes
        self.z_logvar = nn.Parameter(torch.randn(self.out_channels).sub(10.).view(self.out_channels, 1, 1, 1), requires_grad=True)
        self.s_logvar = nn.Parameter(torch.randn(1).sub(10.).view(1, 1, 1, 1), requires_grad=True)


    def clip_variances(self):
        """Clips variance parameters to prevent numerical instability."""
        if self.clip_var:
            self.w_logvar.data.clamp_(max=math.log(self.clip_var))
            self.bias_logvar.data.clamp_(max=math.log(self.clip_var))

    def compute_posterior_params(self):
        """Compute the posterior mean and variance using the hierarchical shrinkage structure."""
        scale_mu = 0.5 * (self.z_mu + self.s_mu)
        scale_logvar = torch.log(0.25 * (self.z_logvar.exp() + self.s_logvar.exp()))

        self.post_weight_var = (torch.exp(scale_logvar) - 1) * torch.exp(2 * scale_mu + scale_logvar) * \
                               (self.w_logvar.exp() + self.w_mu_q.pow(2)) + \
                               self.w_logvar.exp() * torch.exp(2 * scale_mu + scale_logvar)

        self.post_weight_mu = self.w_mu_q * (torch.exp(scale_mu + 0.5 * scale_logvar))
        return self.post_weight_mu, self.post_weight_var

    def forward(self, x):
        batch_size = x.shape[0]
        
        mu_activation = F.conv2d(x, self.w_mu_q, self.bias_mu_q, self.stride, self.padding, self.dilation, self.groups)
        var_activation = F.conv2d(x.pow(2), self.w_logvar.exp(), self.bias_logvar.exp(), self.stride, self.padding, self.dilation, self.groups)


        scale_mu = 0.5 * (self.z_mu + self.s_mu.expand_as(self.z_mu))  # Shape: [N_f, 1, 1, 1]
        scale_mu = scale_mu.permute(1, 2, 3, 0).expand(batch_size, 1, 1, self.out_channels)  # Shape: [batch_size, 1, 1, N_f]

        scale_logvar = torch.log(0.25 * (self.z_logvar.exp() + self.s_logvar.exp().expand_as(self.z_logvar)))  # [N_f, 1, 1, 1]
        scale_logvar = scale_logvar.permute(1, 2, 3, 0).expand(batch_size, 1, 1, self.out_channels)  # Shape: [batch_size, 1, 1, N_f]

        log_z = reparametrize(scale_mu, scale_logvar, sampling=self.training)  # [K, 1, 1, N_f]

        z = torch.exp(log_z)  # Z has shape (out_channels, 1, 1, 1)
        
        eps = torch.randn_like(mu_activation)
        z = z.view(batch_size, self.out_channels, 1, 1)  # Shape [64, 32, 1, 1]
        output = mu_activation * z + torch.sqrt(var_activation) * (z ** 2) * eps

        return output

    def kl_divergence(self):
        """Compute the KL divergence for the variational parameters."""
        # KL(q(z)||p(z)) - Approximation from Molchanov et al.
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self.z_logvar - torch.log(self.z_mu.pow(2) + 1e-8)
        kl_z = -torch.sum(k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * F.softplus(-log_alpha) - k1)

        # KL(q(w|z,s) || p(w)) assuming a Gaussian prior
        kl_w = -0.5 * (self.w_logvar - self.w_logvar.exp() - self.w_mu_q.pow(2) + 1).sum()

        # KL for bias terms
        kl_b = -0.5 * (self.bias_logvar - self.bias_logvar.exp() - self.bias_mu_q.pow(2) + 1).sum()

        return kl_z + kl_w + kl_b
    
    
    def __repr__(self):
        p = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            p += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            p += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            p += ', output_padding={output_padding}'
        if self.groups != 1:
            p += ', groups={groups}'
        if self.bias is None:
            p += ', bias=False'
        p += ')'
        return p.format(name=self.__class__.__name__, **self.__dict__)
    
    
    