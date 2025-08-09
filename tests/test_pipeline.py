from surgicalai.pipeline.integration import Pipeline
from surgicalai.config import CONFIG

def test_pipeline_run():
    pipe = Pipeline()
    res = pipe.run(CONFIG.data / 'samples/face_mock.obj', run_id='test')
    assert 'probs' in res and 'heat' in res
