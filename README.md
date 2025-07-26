# Sentiment Analysis with NLP

This project demonstrates a complete pipeline for performing sentiment analysis on text data using Natural Language Processing (NLP) techniques. The implementation includes text preprocessing, feature extraction, model training, and evaluation.

## Project Structure

```
sentiment-analysis-nlp/
├── sentiment_analysis.py     # Main Python script with the implementation
├── requirements.txt          # Required Python packages
└── README.md                # This file
```

## Features

- Text preprocessing (tokenization, stopword removal, lemmatization)
- TF-IDF vectorization for feature extraction
- Logistic Regression model for sentiment classification
- Model evaluation with classification report and confusion matrix
- Example predictions on custom text

## Prerequisites

- Python 3.7+
- Required Python packages (install using `pip install -r requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-nlp.git
   cd sentiment-analysis-nlp
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

## Usage

1. Prepare your dataset in a CSV file with at least two columns:
   - `text`: The text data
   - `sentiment`: The sentiment label (0 for negative, 1 for positive)

2. Update the `main()` function in `sentiment_analysis.py` to load your dataset:
   ```python
   # Replace the sample data with your dataset
   df = pd.read_csv('your_dataset.csv')
   ```

3. The script will:
   - Preprocess the text data
   - Split it into training and testing sets
   - Train the model
   - Evaluate the model's performance
   - Make predictions on sample text

## Model Performance

The model's performance will be displayed in the console, including:
- Classification report (precision, recall, f1-score)
- Confusion matrix visualization
- Overall accuracy

## Example Output

```
Loading sample dataset...

Preprocessing text data...

Training the model...

Evaluating the model...

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

Model Accuracy: 1.00

Sample Predictions:

Text: I really enjoyed using this product!
Sentiment: Positive

Text: This was a complete waste of time.
Sentiment: Negative

Text: It's okay, nothing special.
Sentiment: Negative
```

## Customization

You can customize the model by:
1. Adjusting the preprocessing steps in the `TextPreprocessor` class
2. Changing the vectorizer parameters in the `SentimentAnalyzer` class
3. Trying different classification models from scikit-learn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
