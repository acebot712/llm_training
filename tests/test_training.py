import pytest
from llm_training.training import train_model

class DummyTrainer:
    def train(self): pass
    def save_model(self, output_dir): pass

class DummyModel:
    config = type('obj', (object,), {'use_cache': False})()

class DummyTokenizer:
    eos_token = "<EOS>"
    pad_token = "<PAD>"
    padding_side = "right"

@pytest.fixture
def patch_hf(monkeypatch):
    monkeypatch.setattr("llm_training.training.AutoModelForCausalLM.from_pretrained", lambda *a, **k: DummyModel())
    monkeypatch.setattr("llm_training.training.AutoTokenizer.from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr("llm_training.training.load_from_disk", lambda *a, **k: {"train": [1,2], "test": [3]})
    monkeypatch.setattr("llm_training.training.SFTConfig", lambda **k: object())
    monkeypatch.setattr("llm_training.training.DataCollatorForLanguageModeling", lambda *a, **k: object())
    monkeypatch.setattr("llm_training.training.SFTTrainer", lambda **k: DummyTrainer())
    monkeypatch.setattr("llm_training.training.wandb", type("WandB", (), {"login": staticmethod(lambda key: None), "init": staticmethod(lambda project: None)}))

def test_train_model(patch_hf):
    config = {
        "model_name": "dummy",
        "dataset_path": "dummy",
        "output_dir": "dummy",
        "run_name": "dummy",
        "wandb_api_key": "test_key"
    }
    train_model(config)
