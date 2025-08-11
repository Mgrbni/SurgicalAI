from surgicalai_demo.risk_factors import adjust_probs


def test_bayes_adjust_basic():
    priors = {"melanoma":0.2,"bcc":0.2,"scc":0.1,"nevus":0.3,"seborrheic_keratosis":0.15,"benign_other":0.05}
    res = adjust_probs(priors, {"uv_exposure":"high","family_history_melanoma":True})
    assert abs(sum(res.posteriors.values()) - 1) < 1e-6
    assert res.posteriors['melanoma'] > priors['melanoma']

