# IntentifierBench

**Author:** Godfrey Inyama

---

## Overview

IntentifierBench is an open-source benchmark and web-based system for authorship identification using discourse-aware representations. Unlike traditional stylometric approaches that rely on lexical or syntactic features, this framework models communicative behaviour through:

- Illocutionary force classification (sentence-level intent)
- Argumentative relation modelling (inter-sentence discourse structure)

Authorship attribution is formulated as a similarity matching problem between document-level embeddings derived from these discourse signals.

---

## Key Features

- Discourse-driven authorship identification
- Neuro-symbolic representation (neural models + linguistic structure)
- Interactive visualisation of author profiles
- Multiple model configurations:
    - Model 1: Illocutionary force classification
    - Model 2: Sentence-pair argumentative classification
    - Both: Combined embeddings
- Flask-based web interface

---

## Repository Layout

```
.
├── app.py
├── preprocessing.py
├── templates/
│   └── index.html
├── static/
│   ├── scripts.js
│   └── style.css
├── documents_pan19/
├── pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23/
├── results/
└── README.md
```

---

## Methodology

1. **Sentence-Level Encoding**: Each document is segmented into sentences and classified into illocutionary forces.
2. **Discourse-Level Encoding**: Adjacent sentence pairs are analysed for argumentative relations.
3. **Document Embeddings**: Embeddings are aggregated using mean pooling and optionally concatenated.
4. **Authorship Attribution**: Cosine similarity is used to match unknown documents to candidate authors.

---

## Core Functions

- `extract_classification`
- `extract_classification_pairs`
- `/identify` route (Flask endpoint)

---

## UI Behaviour

Users can:
- Select a model configuration
- Choose an unknown document
- View predicted author and profile visualisations

---

## Prerequisites

- Python 3.10+
- pip
- Optional GPU

---

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/Godfrey2712/IntentifierBench.git
cd IntentifierBench
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install NLTK Tokenizer

```bash
python -c "import nltk; nltk.download('punkt')"
```

### 5. Prepare Dataset

**Option A:** Use existing `documents_pan19/`

**Option B:** Place PAN19 dataset in `pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23/`

```bash
python preprocessing.py
```

### 6. Run Application

```bash
python app.py
```

### 7. Open Browser

```
http://127.0.0.1:5000/
```

### 8. Perform Authorship Identification

Select model, choose unknown document, view results.

---

## Notes

- First run may take time due to model loading
- Ensure dataset paths are correct
- GPU improves performance

---

## References

### PAN2019 Dataset

```bibtex
@dataset{kestemont_2019_3530313,
    author       = {Kestemont, Mike and Stamatatos, Efstathios and Manjavacas, Enrique and Daelemans, Walter and Potthast, Martin and Stein, Benno},
    title        = {PAN19 Authorship Analysis: Cross-Domain Authorship Attribution},
    month        = nov,
    year         = 2019,
    publisher    = {Zenodo},
    doi          = {10.5281/zenodo.3530313},
    url          = {https://doi.org/10.5281/zenodo.3530313},
}
```

### Pre-trained Models

```bibtex
@misc{godfrey_inyama_2026,
    author       = {Inyama, Godfrey},
    title        = {arg_mining_us2016_locutions},
    year         = {2026},
    howpublished = {https://huggingface.co/Godfrey2712/arg_mining_us2016_locutions},
    doi          = {10.57967/hf/8217},
}

@misc{godfrey_inyama_2026_intent,
    author       = {Inyama, Godfrey},
    title        = {amf_illoc_force_intent_recognition},
    year         = {2026},
    howpublished = {https://huggingface.co/Godfrey2712/amf_illoc_force_intent_recognition},
    doi          = {10.57967/hf/8221},
}
```

### Thesis

```bibtex
@phdthesis{inyama2024exploiting,
    title={Exploiting Illocutionary Forces in Dialogue Structures for Enhancing Authorship Identification},
    author={Inyama, Godfrey},
    school={University of Dundee},
    year={2025}
}
```

---

## Summary

IntentifierBench provides a discourse-aware, neuro-symbolic framework for authorship identification based on communicative behaviour rather than surface-level text features.