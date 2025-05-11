import pytest
from llm_training.evaluation import evaluate_model

class DummyModel:
    pass

def dummy_simple_evaluate(**kwargs):
    return {"results": {kwargs['tasks'][0]: {"accuracy": 1.0}}}

@pytest.fixture
def patch_eval(monkeypatch):
    monkeypatch.setattr("llm_training.evaluation.LlamaCausalLMTensor.from_pretrained", lambda *a, **k: DummyModel())
    monkeypatch.setattr("llm_training.evaluation.HFLM", lambda *a, **k: DummyModel())
    monkeypatch.setattr("llm_training.evaluation.simple_evaluate", dummy_simple_evaluate)
    monkeypatch.setattr("llm_training.evaluation.TaskManager", lambda: object())

def test_evaluate_model(patch_eval, tmp_path):
    config = {
        "model_name": "dummy",
        "device": "cpu",
        "datasets": ["testset"],
        "num_fewshot": 0,
        "batch_size": 1,
        "output_dir": str(tmp_path),
        "is_tensorized": False,
    }
    results = evaluate_model(config)
    assert "testset" in results 