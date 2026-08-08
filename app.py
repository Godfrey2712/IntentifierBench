from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import re
import nltk
import numpy as np
import time

# Ensure tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

app = Flask(__name__)

# Allow tests to skip heavy model loading
SKIP_MODEL_LOAD = os.environ.get('SKIP_MODEL_LOAD') == '1'

# -------------------------------
# ⏱ Utility Timer
# -------------------------------
def log_time(stage, start_time):
    elapsed = time.time() - start_time
    print(f"[TIME] {stage}: {elapsed:.4f} seconds")


# -------------------------------
# Load Models (timed)
# -------------------------------
illoc_model = None
illoc_tokenizer = None
pair_model = None
pair_tokenizer = None
illoc_labels = {}
illoc_pairs_labels = {}

if not SKIP_MODEL_LOAD:
    start = time.time()

    illoc_model = AutoModelForSequenceClassification.from_pretrained(
        "Godfrey2712/amf_illoc_force_intent_recognition"
    )
    illoc_tokenizer = AutoTokenizer.from_pretrained(
        "Godfrey2712/amf_illoc_force_intent_recognition"
    )

    pair_model = AutoModelForSequenceClassification.from_pretrained(
        "Godfrey2712/arg_mining_us2016_locutions"
    )
    pair_tokenizer = AutoTokenizer.from_pretrained(
        "Godfrey2712/arg_mining_us2016_locutions"
    )

    log_time("Model Loading", start)

    # Labels
    illoc_labels = illoc_model.config.id2label
    illoc_pairs_labels = pair_model.config.id2label
else:
    # Provide safe defaults so importing app in tests doesn't break early
    illoc_model = None
    illoc_tokenizer = None
    pair_model = None
    pair_tokenizer = None
    illoc_labels = {}
    illoc_pairs_labels = {}

DOCUMENTS_PATH = "./documents_pan19/documents00001/"


# -------------------------------
# Model 1 (Sentence Classification)
# -------------------------------
def extract_classification(text, model, tokenizer):
    start = time.time()

    sentences = sent_tokenize(text)
    if not sentences:
        return [], np.zeros(model.config.num_labels)

    sentence_embeddings = []
    predicted_labels = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = torch.argmax(logits, dim=1).item()
        predicted_labels.append(predicted_label)
        sentence_embeddings.append(logits.squeeze().cpu().numpy())

    document_embedding = np.mean(sentence_embeddings, axis=0)

    log_time("Model1 Processing", start)
    return predicted_labels, document_embedding


# -------------------------------
# Model 2 (Sentence Pair Classification)
# -------------------------------
def extract_classification_pairs(text, model, tokenizer):
    start = time.time()

    sentences = sent_tokenize(text)

    if len(sentences) < 2:
        return [], np.zeros(model.config.num_labels)

    sentence_pair_embeddings = []
    predicted_labels = []

    for i in range(len(sentences) - 1):
        inputs = tokenizer(
            sentences[i],
            sentences[i + 1],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = torch.argmax(logits, dim=1).item()
        predicted_labels.append(predicted_label)
        sentence_pair_embeddings.append(logits.squeeze().cpu().numpy())

    document_embedding = np.mean(sentence_pair_embeddings, axis=0)

    log_time("Model2 Processing", start)
    return predicted_labels, document_embedding


# -------------------------------
# Filename → Author
# -------------------------------
def extract_author_from_filename(filename):
    if filename.startswith("document_") and filename.lower().endswith(".txt"):
        return filename[len("document_"):-len(".txt")]

    if re.match(r"candidate\d+\.txt$", filename, re.IGNORECASE):
        return os.path.splitext(filename)[0]

    if re.match(r"unknown\d+\.txt$", filename, re.IGNORECASE):
        return os.path.splitext(filename)[0]

    return os.path.splitext(filename)[0]


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    files = os.listdir(DOCUMENTS_PATH)
    known_docs = [f for f in files if "unknown" not in f]
    unknown_docs = [f for f in files if "unknown" in f]

    return render_template(
        'index.html',
        known_docs=known_docs,
        unknown_docs=unknown_docs
    )


# -------------------------------
# Main Logic (TIMED)
# -------------------------------
@app.route('/identify', methods=['POST'])
def identify_author():
    total_start = time.time()

    unknown_doc_name = request.form['unknown_doc']
    selected_model = request.form['selected_model']

    # ---------------------------
    # Validate filename against whitelist
    # ---------------------------
    try:
        available_files = os.listdir(DOCUMENTS_PATH)
    except Exception as e:
        return jsonify({'error': f'Documents directory not found: {str(e)}'}), 500

    if unknown_doc_name not in available_files:
        return jsonify({'error': 'Invalid filename: file not found in documents directory.'}), 400

    # Safe join after validation
    unknown_doc_path = os.path.join(DOCUMENTS_PATH, unknown_doc_name)

    # ---------------------------
    # Read Unknown Document
    # ---------------------------
    start = time.time()
    with open(unknown_doc_path, 'r') as file:
        unknown_text = file.read()
    log_time("Read Unknown Document", start)

    # ---------------------------
    # Extract Unknown Embeddings
    # ---------------------------
    unknown_labels_1, unknown_embedding_1 = None, None
    unknown_labels_2, unknown_embedding_2 = None, None

    if selected_model in ['model1', 'both']:
        unknown_labels_1, unknown_embedding_1 = extract_classification(
            unknown_text, illoc_model, illoc_tokenizer
        )

    if selected_model in ['model2', 'both']:
        unknown_labels_2, unknown_embedding_2 = extract_classification_pairs(
            unknown_text, pair_model, pair_tokenizer
        )

    # ---------------------------
    # Process Known Documents
    # ---------------------------
    author_similarities = {}
    author_profiles = {}

    known_docs = [f for f in os.listdir(DOCUMENTS_PATH) if "unknown" not in f]

    for doc in known_docs:
        doc_start = time.time()

        author_name = extract_author_from_filename(doc)

        with open(os.path.join(DOCUMENTS_PATH, doc), 'r') as file:
            known_text = file.read()

        known_labels_1, known_embedding_1 = None, None
        known_labels_2, known_embedding_2 = None, None

        if selected_model in ['model1', 'both']:
            known_labels_1, known_embedding_1 = extract_classification(
                known_text, illoc_model, illoc_tokenizer
            )

        if selected_model in ['model2', 'both']:
            known_labels_2, known_embedding_2 = extract_classification_pairs(
                known_text, pair_model, pair_tokenizer
            )

        # -----------------------
        # Similarity Calculation
        # -----------------------
        similarity = None

        if selected_model == 'model1' and unknown_embedding_1 is not None:
            similarity = cosine_similarity(
                [unknown_embedding_1], [known_embedding_1]
            )[0][0]

        elif selected_model == 'model2' and unknown_embedding_2 is not None:
            similarity = cosine_similarity(
                [unknown_embedding_2], [known_embedding_2]
            )[0][0]

        elif selected_model == 'both':
            if all(v is not None for v in [
                unknown_embedding_1, known_embedding_1,
                unknown_embedding_2, known_embedding_2
            ]):
                unknown_embedding = np.concatenate(
                    [unknown_embedding_1, unknown_embedding_2]
                )
                known_embedding = np.concatenate(
                    [known_embedding_1, known_embedding_2]
                )

                similarity = cosine_similarity(
                    [unknown_embedding], [known_embedding]
                )[0][0]

        if similarity is not None:
            author_similarities[author_name] = similarity

        # Store profiles safely
        author_profiles[author_name] = {
            'model1_labels': known_labels_1,
            'model2_labels': known_labels_2
        }

        log_time(f"Processed {author_name}", doc_start)

    print(f"\nAuthor Similarities: {author_similarities}\n")

    if not author_similarities:
        return jsonify({
            'error': 'No similarities calculated.'
        }), 400

    most_similar_author = max(
        author_similarities,
        key=author_similarities.get
    )

    log_time("TOTAL REQUEST TIME", total_start)

    return jsonify({
        'author': most_similar_author,
        'author_profiles': author_profiles,
        'unknown_profile_model1': unknown_labels_1,
        'unknown_profile_model2': unknown_labels_2,
        'labels': illoc_labels if selected_model in ['model1', 'both'] else None,
        'labels_pairs': illoc_pairs_labels if selected_model in ['model2', 'both'] else None
    })


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
