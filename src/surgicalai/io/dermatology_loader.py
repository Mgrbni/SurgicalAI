"""Dermatology image loader and preprocessor."""

from pathlib import Path
import numpy as np
from PIL import Image

def load_derm_image(path: Path) -> np.ndarray:
    """Load and preprocess dermatology image.
    
    Args:
        path: Path to image file
        
    Returns:
        RGB image array normalized to [0,1]
    """
    raise NotImplementedError
