from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import nltk
import numpy as np

#nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

# Load the models and tokenizers
illoc_model = AutoModelForSequenceClassification.from_pretrained("Godfrey2712/amf_illoc_force_intent_recognition")
illoc_tokenizer = AutoTokenizer.from_pretrained("Godfrey2712/amf_illoc_force_intent_recognition")

pair_model = AutoModelForSequenceClassification.from_pretrained("Godfrey2712/arg_mining_us2016_locutions")
pair_tokenizer = AutoTokenizer.from_pretrained("Godfrey2712/arg_mining_us2016_locutions")

# Extract id2label from the model config
illoc_labels = illoc_model.config.id2label
illoc_pairs_labels = pair_model.config.id2label

# Directory where documents are stored
DOCUMENTS_PATH = "./documents_pan19/documents00001/"

def extract_classification(text, model, tokenizer):
    sentences = sent_tokenize(text)
    sentence_embeddings = []
    predicted_labels = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        predicted_label = torch.argmax(logits, dim=1).item()
        predicted_labels.append(predicted_label)
        sentence_embeddings.append(logits.squeeze().numpy())
    
    document_embedding = np.mean(sentence_embeddings, axis=0)
    return predicted_labels, document_embedding

def extract_classification_pairs(text, model, tokenizer):
    sentences = sent_tokenize(text)
    sentence_pair_embeddings = []
    predicted_labels = []

    for i in range(len(sentences) - 1):
        sentence_1 = sentences[i]
        sentence_2 = sentences[i + 1]
        inputs = tokenizer(sentence_1, sentence_2, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = torch.argmax(logits, dim=1).item()
        predicted_labels.append(predicted_label)
        sentence_pair_embeddings.append(logits.squeeze().numpy())

    document_embedding = np.mean(sentence_pair_embeddings, axis=0)
    return predicted_labels, document_embedding

def extract_author_from_filename(filename):
    if filename.startswith("document_") and filename.lower().endswith(".txt"):
        return filename[len("document_"):-len(".txt")]
    m = re.match(r"candidate\d+\.txt$", filename, re.IGNORECASE)
    if m:
        return os.path.splitext(filename)[0]  # candidate00001 etc
    m2 = re.match(r"unknown\d+\.txt$", filename, re.IGNORECASE)
    if m2:
        return os.path.splitext(filename)[0]  # unknown00001 etc
    # fallback for generic names
    return os.path.splitext(filename)[0]

@app.route('/')
def index():
    known_docs = [f for f in os.listdir(DOCUMENTS_PATH) if "unknown" not in f]
    unknown_docs = [f for f in os.listdir(DOCUMENTS_PATH) if "unknown" in f]
    return render_template('index.html', known_docs=known_docs, unknown_docs=unknown_docs)

@app.route('/identify', methods=['POST'])
def identify_author():
    unknown_doc_name = request.form['unknown_doc']
    selected_model = request.form['selected_model']
    unknown_doc_path = os.path.join(DOCUMENTS_PATH, unknown_doc_name)

    # Read the unknown document
    with open(unknown_doc_path, 'r') as file:
        unknown_text = file.read()

    author_similarities = {}
    author_profiles = {}
    known_docs = [f for f in os.listdir(DOCUMENTS_PATH) if "unknown" not in f]

    # Initialize variables for unknown labels and embeddings
    unknown_labels_1, unknown_embedding_1 = None, None
    unknown_labels_2, unknown_embedding_2 = None, None

    # Extract embeddings based on selected models
    if selected_model in ['model1', 'both']:
        unknown_labels_1, unknown_embedding_1 = extract_classification(unknown_text, illoc_model, illoc_tokenizer)

    if selected_model in ['model2', 'both']:
        unknown_labels_2, unknown_embedding_2 = extract_classification_pairs(unknown_text, pair_model, pair_tokenizer)

    for doc in known_docs:
        with open(os.path.join(DOCUMENTS_PATH, doc), 'r') as file:
            known_text = file.read()

        # Initialize known labels and embeddings
        known_labels_1, known_embedding_1 = None, None
        known_labels_2, known_embedding_2 = None, None

        if selected_model in ['model1', 'both']:
            known_labels_1, known_embedding_1 = extract_classification(known_text, illoc_model, illoc_tokenizer)

        if selected_model in ['model2', 'both']:
            known_labels_2, known_embedding_2 = extract_classification_pairs(known_text, pair_model, pair_tokenizer)

        # Calculate similarity based on the selected model
        if selected_model == 'model1':
            if unknown_embedding_1 is not None and known_embedding_1 is not None:
                similarity = cosine_similarity([unknown_embedding_1], [known_embedding_1])[0][0]
                author_name = extract_author_from_filename(doc)
                author_similarities[author_name] = similarity

        elif selected_model == 'model2':
            if unknown_embedding_2 is not None and known_embedding_2 is not None:
                similarity = cosine_similarity([unknown_embedding_2], [known_embedding_2])[0][0]
                author_name = extract_author_from_filename(doc)
                author_similarities[author_name] = similarity

        elif selected_model == 'both':
            # Ensure both embeddings are present
            if unknown_embedding_1 is not None and known_embedding_1 is not None and \
               unknown_embedding_2 is not None and known_embedding_2 is not None:
                
                # Concatenate the embeddings
                unknown_embedding = np.concatenate([unknown_embedding_1, unknown_embedding_2])
                known_embedding = np.concatenate([known_embedding_1, known_embedding_2])
                
                # Calculate cosine similarity
                similarity = cosine_similarity([unknown_embedding], [known_embedding])[0][0]
                author_name = extract_author_from_filename(doc)
                author_similarities[author_name] = similarity

        # Store profiles for the author
        author_profiles[author_name] = {
            'model1_labels': known_labels_1,
            'model2_labels': known_labels_2
        }

    print(f"Author Similarities: {author_similarities}")

    # Check if author_similarities is empty before using max
    if not author_similarities:
        return jsonify({
            'error': 'No similarities calculated. Please check the documents and model outputs.'
        }), 400  # Return an error response

    # Identify the most similar author
    most_similar_author = max(author_similarities, key=author_similarities.get)

    return jsonify({
        'author': most_similar_author,
        'author_profiles': author_profiles,
        'unknown_profile_model1': unknown_labels_1,
        'unknown_profile_model2': unknown_labels_2,
        'labels': illoc_labels if selected_model in ['model1', 'both'] else None,
        'labels_pairs': illoc_pairs_labels if selected_model in ['model2', 'both'] else None  # Add this for Model 2 labels
    })


if __name__ == "__main__":
    app.run(debug=True)
