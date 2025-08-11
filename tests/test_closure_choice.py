from surgicalai_demo.langer_lines import analyze_orientation_and_closure
import yaml, json
from pathlib import Path

def test_closure_choice_primary():
    rules = yaml.safe_load(open('data/oncology_rules.yaml','r',encoding='utf-8'))
    res = analyze_orientation_and_closure('upper_lip', (50.0,50.0), 6.0, rules)
    assert 'primary' in res.closure_recommendation.lower()
    assert 0 <= res.success_score <= 100
