# Authorship Identification App

Author: GitHub Copilot  
Model: Raptor mini (Preview)

## Overview

`app.py` runs a Flask web app to identify the author of an unknown document by comparing:
- Model 1: single-sentence illocutionary-force classification
- Model 2: consecutive-sentence pair classification
- Both: concatenated embeddings for joint comparison

The app uses pre-trained Hugging Face models:
- `Godfrey2712/amf_illoc_force_intent_recognition` (`illoc_model`)
- `Godfrey2712/arg_mining_us2016_locutions` (`pair_model`)

Similarity is via cosine similarity from `sklearn.metrics.pairwise.cosine_similarity`.

## Repository Layout

- `app.py` — Flask backend and core logic
- `templates/index.html` — web UI
- `static/scripts.js` — browser logic + chart rendering
- `static/style.css` — CSS
- `documents/` — known + unknown example documents
- `pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23/` — PAN19 dataset folder

## Core Functions

- [`extract_classification`](app.py)
  - single-sentence classification
  - returns `(predicted_labels, document_embedding)`
- [`extract_classification_pairs`](app.py)
  - consecutive sentence pair classification
  - returns `(predicted_labels, document_embedding)`
- `/identify` route in [`app.py`](app.py)
  - computes unknown_doc profiles and author similarity signatures
  - returns JSON with `author`, `author_profiles`, `unknown_profile_model1`, `unknown_profile_model2`, etc.

## UI behavior

- `index.html` has model-selection radios and known/unknown documents.
- `static/scripts.js`:
  - `identifyAuthor(doc)` POST `/identify`
  - response-driven display `author-result`
  - `displayProfile(...)` draws Chart.js bar charts for model profiles
- Labels come from:
  - `illoc_labels = illoc_model.config.id2label`
  - `illoc_pairs_labels = pair_model.config.id2label`

## Prerequisites

- Python 3.10+ recommended
- `pip install -r requirements.txt`
- `nltk` required data:
  - `punkt` or `punkt_tab` (per code uses `nltk.download('punkt_tab')`)
- GPU preferable but not required

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r [requirements.txt](http://_vscodecontentref_/1)