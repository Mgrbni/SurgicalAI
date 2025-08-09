from surgicalai.vision.classifier import LesionClassifier
from surgicalai.vision.gradcam import GradCAM
from PIL import Image
import numpy as np


def test_gradcam():
    clf = LesionClassifier()
    gc = GradCAM(clf.model)
    img = Image.new('RGB',(224,224),(0,0,0))
    cam = gc(img, 0)
    assert cam.shape == (224,224)
