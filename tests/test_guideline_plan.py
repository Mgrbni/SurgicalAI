from surgicalai_demo.guidelines import plan


def test_guideline_plan_nevus():
    r = plan('nevus_compound','upper_lip',6.0, borders_clear=True)
    assert r.margins_mm == '1–2 mm if excised for cosmesis'
    assert any('primary' in o for o in r.reconstruction_options)


def test_guideline_plan_melanoma_defer():
    r = plan('melanoma','upper_lip',8.0, borders_clear=False, breslow_mm=0.8)
    assert r.defer_flap is True
    assert any('Defer' in o for o in r.reconstruction_options)
