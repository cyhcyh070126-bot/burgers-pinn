"""Standard and global artificial-viscosity PINNs for a Burgers shock."""
from .core import TrainingConfig, VanillaBurgersPINN, scalar_residual

__all__ = ["TrainingConfig", "VanillaBurgersPINN", "scalar_residual"]
