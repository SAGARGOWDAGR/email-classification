import sys
import subprocess
import streamlit as st
import re
import joblib
import time
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Workaround for missing joblib (already in use for Hugging Face Spaces)
try:
    import joblib
except ImportError:
    st.warning("Joblib not found. Attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib

# Download stopwords
nltk.download('stopwords', quiet=True)

# Define regex rules
regex_rules = {
    "full_name": r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\b",
    "email": r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
    "phone_number": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}(?:x\d+)?\b",
    "dob": r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b",
    "aadhar_num": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "credit_debit_no": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "cvv_no": r"\b(?<!\d)\d{3}(?!\d)\b",
    "expiry_no": r"\b(0[1-9]|1[0-2])\/?([0-9]{2})\b"
}

# Masking function with entity tracking
def apply_regex_masking_with_entities(text):
    if not isinstance(text, str):
        return "", []
    masked = text
    entities = []
    for entity_type, pattern in regex_rules.items():
        matches = [(m.start(), m.end(), m.group(), entity_type) for m in re.finditer(pattern, text)]
        for start, end, value, etype in matches:
            entities.append({"position": [start, end], "classification": etype, "entity": value})
            masked = masked[:start] + f"[{etype}]" + masked[end:]
    return masked, entities

# Preprocessing with stemming
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

@st.cache_data
def preprocess_text(text):
    words = text.lower().split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load pre-trained model, vectorizer, and selector
@st.cache_resource
def load_model_and_vectorizer():
    print("Loading model, vectorizer, and selector...")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'svm_email_classifier_fast.pkl')
        vectorizer_path = os.path.join(base_dir, 'tfidf_vectorizer_fast.pkl')
        selector_path = os.path.join(base_dir, 'feature_selector_fast.pkl')
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        selector = joblib.load(selector_path)
        return model, vectorizer, selector
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

svm_model, vectorizer, selector = load_model_and_vectorizer()

def classify_email(email_text):
    if svm_model is None or vectorizer is None or selector is None:
        return {"error": "Model files could not be loaded. Please check the files in the repository."}

    start_time = time.time()
    mask_start = time.time()
    masked_email, entities = apply_regex_masking_with_entities(email_text)
    mask_time = time.time() - mask_start
    preprocess_start = time.time()
    processed_email = preprocess_text(masked_email)
    preprocess_time = time.time() - preprocess_start
    vectorize_start = time.time()
    email_tfidf = vectorizer.transform([processed_email])
    email_selected = selector.transform(email_tfidf)
    vectorize_time = time.time() - vectorize_start
    predict_start = time.time()
    category = svm_model.predict(email_selected)[0]
    predict_time = time.time() - predict_start
    total_time = time.time() - start_time

    response = {
        "input_email_body": email_text,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category,
        "timing": {
            "masking": f"{mask_time:.2f} seconds",
            "preprocessing": f"{preprocess_time:.2f} seconds",
            "vectorization": f"{vectorize_time:.2f} seconds",
            "prediction": f"{predict_time:.2f} seconds",
            "total": f"{total_time:.2f} seconds"
        }
    }
    return response

# Streamlit UI and API-like functionality
