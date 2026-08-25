"""
tests/conftest.py

shared pytest fixtures for testing app.py inference functions.

Design goals:
- No real Hugging Face models are downloaded
- No real network calls are made
- extract_classification()/extract_classification_pairs() are tested 
  with fully mocked models, and tokenizers, so the tests run fast and 
  don't depend on external services.

  If adding new tests to this suite, the fixtures mock_tokenizer, 
  mock_model_factory, patch_sent_tokenize are available automatically 
  in any test file in this directory - pytest discovers conftest.py 
  fixtures by name, so don't need to import them.
"""





import os

os.environ["SKIP_MODEL_LOAD"] = '1'  # Set this environment variable to skip model loading during tests

import nltk
nltk.data.find = lambda *args, **kwargs: None


import app 

import pytest
import torch 
from unittest.mock import  MagicMock

def make_logits(num_labels: int, values = None) -> torch.Tensor:

    """
    Build a fake logits tensor shaped (1, num_labels), as if returned 
    by huggingFace sequence-classification model's forward pass.


    pass `values` when a test need to control which label argmax()
    will pick. Omit it when a test just needs a valid-shaped tensor 
    and doesn;t care about the actual numbers.
    """

    if values is not None:
        assert len(values) == num_labels, "values length must match num_labels"
        return torch.tensor([values], dtype=torch.float32)
    
    return torch.tensor([[float(i) for i in range(num_labels)]], dtype=torch.float32)


@pytest.fixture
def mock_tokenizer():

    """
    A face Hugging face tokenizer.

    app.py inference functions only need the tokenizer's return value to 
    behave like a dict that can be unpacked into the model call via **inputs
    as they never inspect the actual token ID's the plain dict of placeholder 
    tensors is enough here.
    """
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1,2,3]]),
        "attention_mask": torch.tensor([[1,1,1]])
    }
    return tokenizer


@pytest.fixture
def mock_model_factory():
    """
    A factory for fake hugging face models.

    returns a functions (rather than single ready-made mock) so each test can 
    independently control num_labels and the exact logits returned, this 
    matters for tests that assert a specific predicted label.

    Usage:
        model = mock_model_factory(num_labels = 4, logits_values=[0.1, 0.9, 0.2, 0.05])

    """
    def _make(num_labels: int, logits_values=None) -> MagicMock:
        model = MagicMock()
        model.config.num_labels = num_labels

        logits = make_logits(num_labels, logits_values)
        model.return_value.logits = logits
        return model
    return _make


@pytest.fixture
def patch_sent_tokenizer(monkeypatch):

    """
    This fixture replaces app.py sent_tokenize with a fake that returns a caller-supplied 
    list of sentences, instead of running real NLTK sentance spllitting.

    Patch target note: we patch "app.sent_tokenize", not "nltk.tokenize.sent_tokenize". 
    app.py did `from nltk.tokenize import sent_tokenize`, which binds the name into app's 
    own module namespace — patching the original nltk module would have no effect on what
    app.py actually calls.

    Usage in a test:
        patch_sent_tokenize(["First sentence.", "Second sentence."])
    """

    def _patch(sentences: list):
        monkeypatch.setattr("app.sent_tokenize", lambda text: sentences)
    return _patch

    

