import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set style for plots
plt.style.use('ggplot')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Tokenization
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
    
    def train(self, X_train, y_train):
        # Vectorize the text data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        # Train the model
        self.model.fit(X_train_vec, y_train)
        return self
    
    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
    
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        return accuracy_score(y_true, y_pred)

def load_and_prepare_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Display basic info
    print("Dataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def main():
    # For demonstration, we'll use a sample dataset
    # In a real scenario, you would load your own dataset
    print("Loading sample dataset...")
    # Example: This should be replaced with your actual dataset
    # df = pd.read_csv('your_dataset.csv')
    
    # For demonstration, creating a sample dataset
    data = {
        'text': [
            'I love this product! It works great!',
            'This is terrible. I hate it!',
            'Not bad, but could be better.',
            'Amazing experience, highly recommended!',
            'Waste of money, very disappointed.'
        ],
        'sentiment': [1, 0, 0, 1, 0]  # 1 for positive, 0 for negative
    }
    df = pd.DataFrame(data)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Clean the text data
    print("\nPreprocessing text data...")
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Initialize and train the model
    print("\nTraining the model...")
    analyzer = SentimentAnalyzer()
    analyzer.train(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    accuracy = analyzer.evaluate(X_test, y_test)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    
    # Example prediction
    test_texts = [
        "I really enjoyed using this product!",
        "This was a complete waste of time.",
        "It's okay, nothing special."
    ]
    
    print("\nSample Predictions:")
    for text in test_texts:
        cleaned_text = preprocessor.clean_text(text)
        prediction = analyzer.predict([cleaned_text])
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        print(f"\nText: {text}
Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
