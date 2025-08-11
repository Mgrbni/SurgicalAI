"""Smoke tests for multi-class inference calibration (demo).

These tests ensure:
- Softmax over all configured classes returns probs summing ~1
- Melanoma probability not systematically dominating for random benign-like images
- Top-k ordering consistent with probabilities
- Grad-CAM heatmaps generated for top-3 classes

NOTE: Uses placeholder backbone producing deterministic pseudo-random logits.
"""
from pathlib import Path
from server.model_infer import run_inference, CLASS2IDX, IDX2CLASS
from server.settings import SETTINGS
from PIL import Image

def _make_tmp_image(tmp_path: Path, name: str, color: tuple[int,int,int]):
    p = tmp_path / name
    Image.new('RGB', (64,64), color).save(p)
    return p

def test_probability_sum_and_classes(tmp_path):
    img = _make_tmp_image(tmp_path, 'sample1.png', (120,80,60))
    res = run_inference(img, tmp_path)
    assert set(res.probs.keys()) == set(SETTINGS.class_names)
    assert abs(sum(res.probs.values()) - 1.0) < 1e-6

def test_topk_consistency(tmp_path):
    img = _make_tmp_image(tmp_path, 'sample2.png', (10,200,10))
    res = run_inference(img, tmp_path)
    # topk sorted descending
    probs = res.probs
    for i in range(len(res.topk)-1):
        assert res.topk[i][1] >= res.topk[i+1][1]
    # each topk element matches dict
    for lbl,p in res.topk:
        assert abs(p - probs[lbl]) < 1e-8

def test_gradcam_generation(tmp_path):
    img = _make_tmp_image(tmp_path, 'sample3.png', (30,30,220))
    res = run_inference(img, tmp_path)
    for lbl,_ in res.topk:
        assert (tmp_path / f'heatmap_{lbl}.png').exists()

def test_no_universal_melanoma_domination(tmp_path):
    # Generate few varied images; ensure melanoma not always top1
    melanoma_top_count = 0
    N = 5
    for i in range(N):
        img = _make_tmp_image(tmp_path, f'var_{i}.png', (i*40 % 255, (255-40*i)%255, (30*i)%255))
        res = run_inference(img, tmp_path)
        if res.topk[0][0] == 'melanoma':
            melanoma_top_count += 1
    # Expect at least one non-melanoma top-1
    assert melanoma_top_count < N
