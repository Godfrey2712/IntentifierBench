from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import numpy as np

app = Flask(__name__)

# Load the models and tokenizers
illoc_model = AutoModelForSequenceClassification.from_pretrained("Godfrey2712/amf_illoc_force_intent_recognition")
illoc_tokenizer = AutoTokenizer.from_pretrained("Godfrey2712/amf_illoc_force_intent_recognition")

pair_model = AutoModelForSequenceClassification.from_pretrained("Godfrey2712/arg_mining_us2016_locutions")
pair_tokenizer = AutoTokenizer.from_pretrained("Godfrey2712/arg_mining_us2016_locutions")

# Extract id2label from the model config
illoc_labels = illoc_model.config.id2label

# Directory where documents are stored
DOCUMENTS_PATH = "./documents/"

def extract_classification(text, model, tokenizer):
    # Split the text into sentences
    sentences = sent_tokenize(text)
    
    sentence_embeddings = []
    predicted_labels = []

    for sentence in sentences:
        # Tokenize the sentence
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Get the predicted label for the sentence
        predicted_label = torch.argmax(logits, dim=1).item()
        predicted_labels.append(predicted_label)

        # Store the embedding (logits) for the sentence
        sentence_embeddings.append(logits.squeeze().numpy())
    
    # Aggregate sentence embeddings by averaging
    document_embedding = np.mean(sentence_embeddings, axis=0)
    
    return predicted_labels, document_embedding

# Route for home page to display documents
@app.route('/')
def index():
    known_docs = [f for f in os.listdir(DOCUMENTS_PATH) if "unknown" not in f]
    unknown_docs = [f for f in os.listdir(DOCUMENTS_PATH) if "unknown" in f]
    return render_template('index.html', known_docs=known_docs, unknown_docs=unknown_docs)

# Route to identify the author of unknown documents
@app.route('/identify', methods=['POST'])
def identify_author():
    unknown_doc_name = request.form['unknown_doc']
    unknown_doc_path = os.path.join(DOCUMENTS_PATH, unknown_doc_name)

    # Read the unknown document
    with open(unknown_doc_path, 'r') as file:
        unknown_text = file.read()

    # Extract classification (for each sentence) and the aggregated embeddings for the unknown document
    unknown_labels, unknown_embedding = extract_classification(unknown_text, illoc_model, illoc_tokenizer)

    # Compare embeddings with known author documents
    author_similarities = {}
    author_profiles = {}
    known_docs = [f for f in os.listdir(DOCUMENTS_PATH) if "unknown" not in f]

    for doc in known_docs:
        with open(os.path.join(DOCUMENTS_PATH, doc), 'r') as file:
            known_text = file.read()

        # Extract classification (for each sentence) and the aggregated embeddings for each known document
        known_labels, known_embedding = extract_classification(known_text, illoc_model, illoc_tokenizer)
        similarity = cosine_similarity([unknown_embedding], [known_embedding])[0][0]

        author_name = doc.split('_')[1].split('.')[0]  # Extract author name from filename
        author_similarities[author_name] = similarity
        author_profiles[author_name] = known_labels  # Save the sentence-level illocutionary force classifications

    # Identify the most similar author
    most_similar_author = max(author_similarities, key=author_similarities.get)

    # Return similarity, author profiles, unknown document classification, and model labels
    return jsonify({
        'author': most_similar_author,
        'author_profiles': author_profiles,
        'unknown_profile': unknown_labels,
        'labels': illoc_labels
    })
if __name__ == "__main__":
    app.run(debug=True)
