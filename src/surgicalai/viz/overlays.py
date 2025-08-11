def save_overlay(image, gradcam, out_path):
    # Dummy overlay save
    from PIL import Image
    import numpy as np
    overlay = Image.fromarray((np.zeros_like(gradcam) * 255).astype(np.uint8))
    overlay.save(out_path)
