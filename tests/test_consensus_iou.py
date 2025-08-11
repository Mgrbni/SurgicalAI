from surgicalai_demo.multimodel import locate_lesion
import numpy as np


def test_consensus_iou_threshold():
    img = np.zeros((400,400,3), dtype='uint8')
    res = locate_lesion(img)
    iou = res['consensus']['meta']['iou']
    assert 0 <= iou <= 1
