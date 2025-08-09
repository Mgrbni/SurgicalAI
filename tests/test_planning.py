import numpy as np
from surgicalai.planning.flap_designer import design_flap


def test_design_flap():
    vertices = np.random.rand(10,3)
    heat = np.random.rand(10)
    flap = design_flap(vertices, heat)
    assert 'pivot' in flap
