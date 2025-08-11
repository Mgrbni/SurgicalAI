import os

try:
    import torch
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_WEIGHTS_PATH = os.environ.get('MODEL_WEIGHTS_PATH', None)


def load_model():
    if not TORCH_AVAILABLE:
        print("[Vision] torch not available, using dummy model.")
        return None
    try:
        model = models.resnet50(pretrained=True)
        model.eval()
        return model
    except Exception as e:
        print(f"[Vision] Failed to load ResNet-50: {e}. Using dummy model.")
        return None


def predict(model, image_path):
    if not TORCH_AVAILABLE or model is None:
        print("[Vision] Dummy predict: returning deterministic results.")
        return {"melanoma": 0.5, "nevus": 0.3, "bcc": 0.1, "scc": 0.05, "benign_other": 0.05}, None
    # Real inference stub
    # You would add image loading and preprocessing here
    # For now, return dummy
    return {"melanoma": 0.5, "nevus": 0.3, "bcc": 0.1, "scc": 0.05, "benign_other": 0.05}, None
