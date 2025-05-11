import pytest
from llm_training.data import prepare_data

class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        self.eos_token = "<EOS>"
    def apply_chat_template(self, conversations, tokenize=False):
        return "<chat>"

@pytest.fixture
def patch_hf(monkeypatch):
    monkeypatch.setattr("llm_training.data.AutoTokenizer.from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr("llm_training.data.load_dataset", lambda *a, **k: type("DS", (), {"train_test_split": lambda self, test_size: {"train": [1,2], "test": [3]}})())
    monkeypatch.setattr("llm_training.data.DatasetDict", dict)

def test_prepare_data(patch_hf):
    dataset = prepare_data("dummy", "dummy")
    assert "train" in dataset and "test" in dataset 