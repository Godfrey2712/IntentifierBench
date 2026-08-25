
"""
tests/test_inference.py
Unit test for app.py's core inference functions.

Models and tokenizers are mocked, no real hugging face models are loaded and no network calls are made.
Refer conftest.py for how SKIP_MODEL_LOAD and the nltk.data.find path make this possible.
"""




import importlib
import numpy as np 
import nltk
from unittest.mock import MagicMock

import app 
from app import extract_classification, extract_classification_pairs



# ===========================================================
# extract_classification
# ===========================================================

def test_extract_classification_normal_text(mock_tokenizer, mock_model_factory, patch_sent_tokenizer):

    """
    With multiple sentences, the functions should classify each 
    one independently and average their logits into a single document embedding.
    """

    sentences = ["Sentence one.", "Sentence two", "Sentence three."]
    patch_sent_tokenizer(sentences)

    model = mock_model_factory(num_labels = 4, logits_values=[0.1, 0.9, 0.2, 0.05])

    predicted_labels, document_embedding = extract_classification(
        "irrelevant raw text", model, mock_tokenizer
    )

    assert len(predicted_labels) == len(sentences)

    assert document_embedding.shape ==(4,)

    assert model.call_count == len(sentences)
    assert mock_tokenizer.call_count == len(sentences)



def test_extract_classification_empty_text(mock_tokenizer, mock_model_factory, patch_sent_tokenizer):

    """
    With no sentences, the function should short-circuit and return a zero vector
    by never touching the model or tokenizer.
    """

    patch_sent_tokenizer([])

    model = mock_model_factory(num_labels = 4)

    predicted_labels , document_embedding = extract_classification(
        "irrelevant raw text", model, mock_tokenizer
    )

    assert predicted_labels == []
    assert np.array_equal(document_embedding, np.zeros(4))


    model.asser_not_called()
    mock_tokenizer.assert_not_called()

# ===========================================================
# extract_classification_pairs
# ===========================================================

def test_extract_classification_pairs_one_sentence(mock_tokenizer, mock_model_factory, patch_sent_tokenizer):

    """
    A single sentence can't form a pair , so the fucntion should 
    short-circuit the same way the empty-text case does.
    """

    patch_sent_tokenizer(["Only one sentence."])

    model = mock_model_factory(num_labels =3)

    predicted_labels, document_embedding = extract_classification_pairs(
        "irrelevant raw text", model, mock_tokenizer
    )


    assert predicted_labels == []
    assert np.array_equal(document_embedding, np.zeros(3))
    
    
    model.asser_not_called()
    mock_tokenizer.assert_not_called()


def test_extract_classification_pairs_normal_input(mock_tokenizer, mock_model_factory, patch_sent_tokenizer):


    """
    With more than three sentences, the function should classify consecutive 
    sentence pairs, not all-apirs or any other scheme.
    """

    sentences =["S1.", "S2.", "S3."]
    patch_sent_tokenizer(sentences)

    model = mock_model_factory(num_labels=3, logits_values=[0.2,0.7,0.1])

    predicted_labels, document_embedding = extract_classification_pairs(
        "irrelevant raw text", model, mock_tokenizer
    )

    assert len(predicted_labels) == len(sentences) -1
    assert model.call_count == len(sentences) -1


    mock_tokenizer.asser_any_call(
        sentences[0], sentences[1], return_tensors="pt", padding=True, truncation=True
    )

    mock_tokenizer.asser_any_call(
        sentences[1], sentences[2], return_tensors="pt", padding=True, truncation=True
    )

    assert document_embedding.shape ==(3,)

# ===========================================================
# extract_classification_pairs
# ===========================================================

def test_app_reimport_does_not_trigger_downloads(monkeypatch):

    """
    This function confirms re-importing app.py never reaches nltk.download
    with patch applied to nltk.data.find in conftest.py and HF model-loading is 
    skipped by SKIP_MODEL_LOAD.

    Note: importlib.reload mutates the real app module in place 
    for the rest of the test session. 

    """

    download_spy = MagicMock()
    monkeypatch.setattr(nltk, "download", download_spy)

    importlib.reload(app)

    download_spy.assert_not_called()
    assert app.illoc_model is None
    assert app.pair_model is None

