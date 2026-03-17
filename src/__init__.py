"""PINN: Physics-Informed Neural Networks"""

from .pinn_model import PINN, PDELoss
from .trainer import PINNTrainer

__version__ = "0.1.0"
__all__ = ["PINN", "PDELoss", "PINNTrainer"]
