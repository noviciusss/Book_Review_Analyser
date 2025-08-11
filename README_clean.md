# Kindle Review Sentiment Analyzer

This Streamlit app analyzes the sentiment of Kindle book reviews using machine learning.

## Features

- **Word2Vec** text vectorization
- **Logistic Regression** classification
- Trained on 10,000+ Amazon Kindle reviews
- Real-time sentiment prediction
- Text preprocessing pipeline
- Confidence scores
- Beautiful UI with Streamlit

## Installation

1. Make sure you have Python installed (3.7 or higher) or use the included conda environment

2. Install required packages:
```bash
# If using system Python:
pip install -r requirements.txt

# If using the included conda environment:
.conda\python.exe -m pip install -r requirements.txt
```

3. Run the app:
```bash
# If using system Python:
streamlit run streamlit_app.py

# If using the included conda environment:
.conda\python.exe -m streamlit run streamlit_app.py

# Or simply double-click run_app.bat on Windows
```

## Usage

1. Open the app in your browser (usually http://localhost:8501)
2. Enter a book review in the text area
3. Click "Analyze Sentiment" to get the prediction
4. View the results with confidence score and preprocessed text

## Model Details

- **Algorithm**: Logistic Regression with Word2Vec features
- **Vector Size**: 100 dimensions
- **Training Data**: Amazon Kindle Store reviews
- **Accuracy**: ~85% on test data

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- nltk
- beautifulsoup4
- gensim

The app will work with missing dependencies but with reduced functionality:
- Without NLTK: Basic stopword removal and no lemmatization
- Without BeautifulSoup4: No HTML tag removal

## Files

- `streamlit_app.py`: Main Streamlit application
- `lr_classifier.pkl`: Trained Logistic Regression model
- `w2v_model.pkl`: Trained Word2Vec model
- `Setiment_Analysis_kindle.ipynb`: Jupyter notebook with model training
- `requirements.txt`: Required Python packages
- `run_app.bat`: Windows batch file to easily run the app
- `setup.bat`: Windows setup script to install dependencies

## Example Reviews

**Positive Example:**
"This book was absolutely fantastic! The characters were well-developed and the plot kept me engaged throughout. I highly recommend it to anyone who enjoys this genre."

**Negative Example:**
"I was really disappointed with this book. The story was confusing and the characters were not believable. I couldn't finish it."

## Screenshots

![App Interface](https://via.placeholder.com/800x400?text=Streamlit+App+Interface)

## How It Works

1. **Text Preprocessing**: The app cleans the input text by removing HTML tags, URLs, special characters, and stopwords
2. **Lemmatization**: Words are reduced to their root form using NLTK's WordNetLemmatizer
3. **Vectorization**: The processed text is converted to numerical vectors using the trained Word2Vec model
4. **Classification**: The Logistic Regression model predicts sentiment based on the vector representation
5. **Results**: The app displays the predicted sentiment with confidence score

## Model Training

The model was trained on a dataset of Amazon Kindle book reviews with the following steps:
- Data cleaning and preprocessing
- Text tokenization and lemmatization
- Word2Vec model training for word embeddings
- Logistic Regression classifier training
- Model evaluation and optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Made with ❤️ using Streamlit and scikit-learn**
