import torch
import torch.nn as nn
import numpy as np


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN)
    
    Trains a neural network to solve differential equations by
    incorporating physical constraints directly into the loss function.
    """
    
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1):
        """
        Initialize PINN architecture.
        
        Args:
            input_dim: Dimension of input (e.g., 2 for (x, t))
            hidden_dim: Number of neurons in hidden layers
            output_dim: Dimension of output (usually 1 for scalar PDEs)
        """
        super(PINN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass with tanh activation (common for PINNs).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


class PDELoss:
    """
    Custom loss function combining data loss and PDE residual loss.
    """
    
    def __init__(self, model, lambda_pde=1.0):
        """
        Args:
            model: The PINN model
            lambda_pde: Weight for PDE residual loss vs data loss
        """
        self.model = model
        self.lambda_pde = lambda_pde
        
    def data_loss(self, x_data, u_data):
        """
        Compute loss on known data points.
        """
        u_pred = self.model(x_data)
        return torch.mean((u_pred - u_data) ** 2)
    
    def pde_residual_loss(self, x_pde):
        """
        Compute residual of the PDE (example: simple diffusion equation).
        For a custom PDE, modify this function.
        """
        x_pde.requires_grad = True
        u = self.model(x_pde)
        
        # Compute gradients for PDE residual
        # Example: du/dt - d2u/dx2 = 0 (diffusion equation)
        u_grad = torch.autograd.grad(
            outputs=u,
            inputs=x_pde,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        du_dt = u_grad[:, 1:2]  # Gradient w.r.t. time
        du_dx = u_grad[:, 0:1]  # Gradient w.r.t. space
        
        # Second derivative d2u/dx2
        d2u_dx2 = torch.autograd.grad(
            outputs=du_dx,
            inputs=x_pde,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True
        )[0][:, 0:1]
        
        # PDE residual: du/dt - d2u/dx2 = 0
        pde_residual = du_dt - d2u_dx2
        
        return torch.mean(pde_residual ** 2)
    
    def total_loss(self, x_data, u_data, x_pde):
        """
        Combined loss: data loss + lambda * pde loss
        """
        loss_data = self.data_loss(x_data, u_data)
        loss_pde = self.pde_residual_loss(x_pde)
        return loss_data + self.lambda_pde * loss_pde
