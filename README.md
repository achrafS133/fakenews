# Fake News Detection Project

A comprehensive machine learning project for detecting fake news articles using natural language processing and various classification algorithms.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Methodology](#methodology)
- [Results](#results)
- [Models Evaluated](#models-evaluated)

## 🎯 Overview

This project implements a fake news detection system that analyzes news articles to classify them as either real or fake. The system uses advanced NLP techniques including text preprocessing, feature extraction, sentiment analysis, and multiple machine learning algorithms to achieve high accuracy in classification.

## 📊 Dataset

The project uses the **WELFake_Dataset.csv** dataset, which contains:
- **Title**: The headline of the news article
- **Text**: The body content of the article
- **Label**: Binary classification (1 = Real news, 0 = Fake news)

**Dataset Statistics:**
- Total articles: 48,390 (after preprocessing)
- Real articles: 52.40%
- Fake articles: 47.60%

## ✨ Features

### Data Preprocessing
- **Text Cleaning**: Removal of URLs, punctuation, special characters, and non-English content
- **Stopword Removal**: Elimination of common English stopwords
- **Lemmatization**: Converting words to their root forms
- **Tokenization**: Breaking text into individual tokens
- **Null Value Handling**: Removal of missing or blank entries

### Feature Extraction
- **Sentiment Analysis**: Polarity and subjectivity scores for titles and text
- **Basic Features**: Word count, character count, average word length, numeric count
- **TF-IDF**: Term Frequency-Inverse Document Frequency analysis
- **N-grams**: Bigram and trigram analysis

### Exploratory Data Analysis
- Distribution analysis of real vs fake news
- Word clouds for visual representation
- N-gram frequency analysis
- Topic modeling using LDA (Latent Dirichlet Allocation)
- Correlation analysis between features

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup Steps

1. **Clone or download the repository**
   ```bash
   cd fakenews
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install pandas scikit-learn seaborn matplotlib preprocessor num2words langdetect nltk textblob unidecode wordcloud gensim spacy scipy
   ```

5. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

6. **Download spaCy English model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 💻 Usage

1. **Open the Jupyter Notebook**
   ```bash
   jupyter notebook "fake news usa project.ipynb"
   ```

2. **Ensure the dataset is in the same directory**
   - Make sure `WELFake_Dataset.csv` is in the project root directory

3. **Run the notebook cells sequentially**
   - The notebook is organized into sections:
     - Data Preprocessing
     - Exploratory Data Analysis
     - Model Training and Evaluation

## 📁 Project Structure

```
fakenews/
│
├── fake news usa project.ipynb    # Main Jupyter notebook
├── WELFake_Dataset.csv            # Dataset file
├── README.md                      # This file
└── venv/                          # Virtual environment (not included in repo)
```

## 📦 Dependencies

### Core Libraries
- **pandas** (2.2.3): Data manipulation and analysis
- **numpy** (1.26.4): Numerical computing
- **scikit-learn** (1.6.1): Machine learning algorithms
- **scipy** (1.13.1): Scientific computing

### NLP Libraries
- **nltk** (3.9.1): Natural language processing toolkit
- **textblob** (0.19.0): Text processing and sentiment analysis
- **spacy** (3.8.6): Advanced NLP library
- **gensim** (4.3.3): Topic modeling and word embeddings
- **preprocessor** (1.1.3): Text preprocessing utilities
- **langdetect** (1.0.9): Language detection
- **unidecode** (1.4.0): Unicode normalization
- **num2words** (0.5.14): Number to word conversion

### Visualization
- **matplotlib** (3.10.3): Plotting library
- **seaborn** (0.13.2): Statistical data visualization
- **wordcloud** (1.9.4): Word cloud generation

## 🔬 Methodology

### 1. Data Preprocessing Pipeline
   - Text normalization and cleaning
   - Stopword removal
   - Lemmatization
   - Tokenization
   - Feature engineering

### 2. Feature Engineering
   - Sentiment analysis (polarity and subjectivity)
   - Statistical features (word count, character count, etc.)
   - TF-IDF vectorization
   - N-gram extraction

### 3. Model Training
   - Train-test split (80-20)
   - Multiple algorithms tested with both Count Vectorizer and TF-IDF Vectorizer
   - Hyperparameter tuning
   - Model evaluation using accuracy and F1-score

## 📈 Results

### Model Performance Comparison

The following models were evaluated on the test set:

| Model | Vectorizer | Accuracy | F1-Score |
|-------|-----------|----------|----------|
| **SVM** | TF-IDF | **0.94** | **0.94** |
| SVM | Count | 0.93 | 0.93 |
| Logistic Regression | Count | 0.92 | 0.92 |
| Logistic Regression | TF-IDF | 0.92 | 0.92 |
| Random Forest | Count | 0.92 | 0.92 |
| Random Forest | TF-IDF | 0.92 | 0.92 |
| Decision Tree | Count | 0.91 | 0.91 |
| Decision Tree | TF-IDF | 0.90 | 0.90 |
| Gradient Boosting | Count | 0.91 | 0.91 |
| Gradient Boosting | TF-IDF | 0.91 | 0.91 |
| Multinomial Naive Bayes | Count | 0.85 | 0.85 |
| Multinomial Naive Bayes | TF-IDF | 0.84 | 0.84 |

**Best Model**: Support Vector Machine (SVM) with TF-IDF Vectorizer
- **Accuracy**: 94%
- **F1-Score**: 94%
- **Precision**: 95% (Fake), 93% (Real)
- **Recall**: 92% (Fake), 96% (Real)

## 🤖 Models Evaluated

1. **Decision Tree Classifier**
2. **Logistic Regression**
3. **Random Forest Classifier**
4. **Gradient Boosting Classifier**
5. **Support Vector Machine (SVM)**
6. **Multinomial Naive Bayes**

Each model was tested with both:
- **Count Vectorizer**: Bag of words representation
- **TF-IDF Vectorizer**: Term frequency-inverse document frequency representation

## 📝 Notes

- The dataset was cleaned to remove null values and non-English content
- N-gram analysis showed that n-gram range of (1,4) achieved the best F1-score (0.9247) in preliminary testing
- The project includes comprehensive exploratory data analysis with visualizations
- Topic modeling was performed to identify common themes in real vs fake news

## 🔮 Future Improvements

- Implement deep learning models (LSTM, BERT, etc.)
- Add more sophisticated feature engineering
- Implement ensemble methods
- Create a web interface for real-time prediction
- Expand dataset with more recent articles
- Add cross-validation for more robust evaluation

## 📄 License

This project is for educational purposes.

## 👤 Author

Fake News Detection Project

---

**Note**: Make sure to have the `WELFake_Dataset.csv` file in the project directory before running the notebook.

