"""Tests for surgical guidelines engine."""

import pytest
from pathlib import Path
import yaml

from surgicalai.cognition.guidelines import (
    load_rules,
    validate_sources,
    query_rules,
    FacialSubunitGuidelines
)
from surgicalai.cognition.contraindications import GuidelineGate

@pytest.fixture
def test_rules():
    """Load the rules for testing."""
    rules_path = Path(__file__).parent.parent / "src" / "surgicalai" / "rules" / "face_rules.yaml"
    with open(rules_path) as f:
        return yaml.safe_load(f)

def test_rules_load(test_rules):
    """Test that rules load correctly."""
    assert test_rules["meta"]["version"] == "0.1"
    assert "diagnoses" in test_rules
    assert "melanoma" in test_rules["diagnoses"]

def test_source_validation(test_rules):
    """Test source validation logic."""
    assert validate_sources(test_rules)

def test_melanoma_guidelines(test_rules):
    """Test melanoma guideline queries."""
    guidelines = query_rules("melanoma", "nasal_dorsum", test_rules)
    assert isinstance(guidelines, FacialSubunitGuidelines)
    assert guidelines.margins.source_ids

def test_h_zone_gate():
    """Test H-zone contraindication gate."""
    gate = GuidelineGate()
    decision = gate.check("melanoma", "nasal_tip", melanoma_prob=0.1)
    assert not decision.allow
    assert "H-zone" in decision.reason
    assert decision.citations

def test_melanoma_prob_gate():
    """Test melanoma probability gate."""
    gate = GuidelineGate()
    decision = gate.check("melanoma", "cheek_lateral", melanoma_prob=0.35)
    assert not decision.allow
    assert "probability" in decision.reason.lower()
    assert decision.citations

def test_bcc_margin_control():
    """Test BCC margin control requirements."""
    gate = GuidelineGate()
    decision = gate.check(
        "bcc",
        "nasal_ala",
        high_risk_features=["margin_control"]
    )
    assert decision.allow
