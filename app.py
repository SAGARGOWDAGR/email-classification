import streamlit as st
import re
import joblib
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

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
    model = joblib.load('svm_email_classifier_fast.pkl')
    vectorizer = joblib.load('tfidf_vectorizer_fast.pkl')
    selector = joblib.load('feature_selector_fast.pkl')
    return model, vectorizer, selector

svm_model, vectorizer, selector = load_model_and_vectorizer()

def classify_email(email_text):
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
    st.write(f"Masking Time: {mask_time:.2f} seconds")
    st.write(f"Preprocessing Time: {preprocess_time:.2f} seconds")
    st.write(f"Vectorization Time: {vectorize_time:.2f} seconds")
    st.write(f"Prediction Time: {predict_time:.2f} seconds")
    st.write(f"Total Time: {total_time:.2f} seconds")
    response = {
        "input_email_body": email_text,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
    return response

# Streamlit UI
st.title("Email Classification with Optimized SVM")
email_input = st.text_area("Enter Email Body", height=200)
if st.button("Classify Email"):
    if email_input:
        with st.spinner("Classifying..."):
            result = classify_email(email_input)
            st.json(result)
    else:
        st.error("Please enter an email body.")
st.markdown("""
This application uses an optimized SVM model with TF-IDF features to classify emails into categories (Billing Issues, Technical Support, Account Management). 
PII (e.g., names, credit card numbers) is masked before processing. The model has been tuned for better accuracy and speed.
""")
