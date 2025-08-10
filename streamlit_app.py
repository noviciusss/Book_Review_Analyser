import streamlit as st
import pickle
import numpy as np
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data if not already present
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

# Download NLTK data
download_nltk_data()

# --- Load Models and Initialize Components ---
@st.cache_resource
def load_models_and_components():
    """Load the trained models and initialize text processing components."""
    try:
        # Load the Word2Vec model
        with open('w2v_model.pkl', 'rb') as f:
            w2v_model = pickle.load(f)
        
        # Load the Logistic Regression classifier
        with open('lr_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        
        # Initialize text processing components
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        return w2v_model, classifier, stop_words, lemmatizer
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load models and components
w2v_model, classifier, stop_words, lemmatizer = load_models_and_components()

# --- Text Processing Functions ---
def clean_text(raw_text, stop_words):
    """
    Cleans raw text by performing the following steps:
    1. Removes HTML tags
    2. Removes URLs
    3. Removes special characters and numbers
    4. Converts to lowercase
    5. Removes stopwords
    6. Removes extra whitespace
    
    Args:
        raw_text (str): The original text string
        stop_words (set): Set of stopwords to remove
    
    Returns:
        str: The cleaned text
    """
    if not raw_text or pd.isna(raw_text):
        return ""
    
    # Remove HTML tags
    text = BeautifulSoup(str(raw_text), "html.parser").get_text()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters and numbers, keep only letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase and tokenize
    words = text.lower().split()
    
    # Remove stopwords
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 1]
    
    # Join words back and remove extra whitespace
    cleaned_text = " ".join(meaningful_words).strip()
    
    return cleaned_text

def lemmatize_words(text, lemmatizer):
    """
    Lemmatizes words in the text.
    
    Args:
        text (str): Input text
        lemmatizer: WordNet lemmatizer instance
    
    Returns:
        str: Lemmatized text
    """
    if not text:
        return ""
    
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def get_average_vector(token_list, model, vector_size=100):
    """
    Calculates the average vector for a list of tokens using Word2Vec model.
    
    Args:
        token_list (list): List of tokens/words
        model: Trained Word2Vec model
        vector_size (int): Size of word vectors
    
    Returns:
        numpy.ndarray: Average vector representation
    """
    avg_vector = np.zeros((vector_size,), dtype="float32")
    num_words_in_model = 0
    
    for word in token_list:
        if word in model.wv:
            avg_vector = np.add(avg_vector, model.wv[word])
            num_words_in_model += 1
    
    if num_words_in_model > 0:
        avg_vector = np.divide(avg_vector, num_words_in_model)
    
    return avg_vector

def preprocess_text(text, stop_words, lemmatizer):
    """
    Complete text preprocessing pipeline.
    
    Args:
        text (str): Raw input text
        stop_words (set): Set of stopwords
        lemmatizer: WordNet lemmatizer instance
    
    Returns:
        str: Fully preprocessed text
    """
    cleaned = clean_text(text, stop_words)
    lemmatized = lemmatize_words(cleaned, lemmatizer)
    return lemmatized

def predict_sentiment(text, w2v_model, classifier, stop_words, lemmatizer):
    """
    Predicts sentiment of the given text.
    
    Args:
        text (str): Input text
        w2v_model: Trained Word2Vec model
        classifier: Trained classifier
        stop_words (set): Set of stopwords
        lemmatizer: WordNet lemmatizer instance
    
    Returns:
        tuple: (prediction, confidence, processed_text)
    """
    # Preprocess the text
    processed_text = preprocess_text(text, stop_words, lemmatizer)
    
    if not processed_text:
        return None, None, ""
    
    # Tokenize and vectorize
    tokens = processed_text.split()
    if not tokens:
        return None, None, processed_text
    
    # Get average vector
    vector = get_average_vector(tokens, w2v_model, 100)
    
    # Check if vector has meaningful content
    if np.all(vector == 0):
        return None, None, processed_text
    
    # Reshape for prediction
    model_input = np.array([vector])
    
    # Make prediction
    prediction = classifier.predict(model_input)[0]
    
    # Get prediction probabilities for confidence
    try:
        probabilities = classifier.predict_proba(model_input)[0]
        confidence = max(probabilities)
    except:
        confidence = None
    
    return prediction, confidence, processed_text

# --- Streamlit App UI ---
st.set_page_config(
    page_title="Kindle Review Sentiment Analyzer",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with app information
st.sidebar.title("About")
st.sidebar.info(
    """
    This app analyzes the sentiment of Kindle book reviews using:
    - **Word2Vec** for text vectorization
    - **Logistic Regression** for classification
    - Trained on **10,000+** Amazon Kindle reviews
    
    **Preprocessing steps:**
    1. HTML tag removal
    2. URL removal
    3. Special character cleaning
    4. Stopword removal
    5. Lemmatization
    6. Word2Vec vectorization
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance:**")
st.sidebar.markdown("- Training Accuracy: ~85%")
st.sidebar.markdown("- Features: 100D Word2Vec vectors")

# Main app
st.title("📖 Kindle Book Review Sentiment Analyzer")
st.markdown("---")

st.write("""
Enter a book review from the Kindle store below, and this app will predict its sentiment using a machine learning model 
trained on thousands of Amazon Kindle reviews.
""")

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "Enter your review text here:",
        height=150,
        placeholder="Type your book review here... For example: 'This book was amazing! I loved the characters and the plot was very engaging.'"
    )

with col2:
    st.markdown("### Sample Review")
    if st.button("Use Sample Review"):
        sample_review = "This book was absolutely fantastic! The characters were well-developed and the plot kept me engaged throughout. I highly recommend it to anyone who enjoys this genre."
        st.session_state.sample_review = sample_review

# Use sample review if button was clicked
if 'sample_review' in st.session_state:
    user_input = st.session_state.sample_review
    del st.session_state.sample_review

# Analysis section
if st.button("🔍 Analyze Sentiment", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            # Make prediction
            prediction, confidence, processed_text = predict_sentiment(
                user_input, w2v_model, classifier, stop_words, lemmatizer
            )
            
            if prediction is not None:
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("### 👍 Positive Sentiment")
                        st.balloons()
                    else:
                        st.error("### 👎 Negative Sentiment")
                
                with col2:
                    if confidence:
                        st.metric("Confidence", f"{confidence:.2%}")
                
                # Show processed text
                with st.expander("🔧 View Preprocessed Text"):
                    st.write("**Original Text:**")
                    st.write(user_input)
                    st.write("**Processed Text:**")
                    st.write(processed_text if processed_text else "No meaningful words found after preprocessing")
                
            else:
                st.warning("⚠️ Could not analyze the text. Please try with a different review that contains more meaningful words.")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# Additional features
st.markdown("---")
st.subheader("💡 Tips for Better Results")

tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.markdown("""
    **✅ Good practices:**
    - Use complete sentences
    - Include descriptive words
    - Mention specific aspects of the book
    - Express clear opinions
    """)

with tips_col2:
    st.markdown("""
    **❌ Avoid:**
    - Very short texts (< 10 words)
    - Only special characters or numbers
    - Text in languages other than English
    - Pure HTML or code
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Made with ❤️ using Streamlit | Data from Amazon Kindle Store Reviews
    </div>
    """,
    unsafe_allow_html=True
)
