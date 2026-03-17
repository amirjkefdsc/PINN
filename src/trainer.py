import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class PINNTrainer:
    """
    Trainer class for Physics-Informed Neural Networks.
    """
    
    def __init__(self, model, loss_fn, device='cpu', learning_rate=0.001):
        """
        Initialize trainer.
        
        Args:
            model: PINN model instance
            loss_fn: Loss function (PDELoss instance)
            device: 'cpu' or 'cuda'
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
    def train(self, x_data, u_data, x_pde, epochs=1000, verbose=True):
        """
        Train the PINN model.
        
        Args:
            x_data: Training data points (shape: [N, input_dim])
            u_data: Solutions at training points (shape: [N, 1])
            x_pde: Points where PDE is enforced (shape: [M, input_dim])
            epochs: Number of training epochs
            verbose: Print loss every N epochs
        """
        x_data = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        u_data = torch.tensor(u_data, dtype=torch.float32).to(self.device)
        x_pde = torch.tensor(x_pde, dtype=torch.float32).to(self.device)
        
        loss_history = []
        
        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        
        for epoch in iterator:
            self.optimizer.zero_grad()
            
            # Compute combined loss
            loss = self.loss_fn.total_loss(x_data, u_data, x_pde)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            loss_history.append(loss.item())
            
            if verbose and (epoch + 1) % 100 == 0:
                iterator.set_description(f"Loss: {loss.item():.6f}")
        
        return loss_history
    
    def predict(self, x):
        """
        Make predictions on new data.
        
        Args:
            x: Input points (shape: [N, input_dim])
            
        Returns:
            Predictions (shape: [N, 1])
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            predictions = self.model(x_tensor)
            
        return predictions.cpu().numpy()
    
    def save(self, filepath):
        """Save model weights."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")
