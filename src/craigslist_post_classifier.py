import json
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# Load data
def load_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        try:
            # Attempt to convert the first line to an integer. If successful, skip it.
            int(lines[0].strip())
            data = [json.loads(line) for line in lines[1:]]
        except ValueError:
            # If the first line is not an integer, parse the entire file
            data = [json.loads(line) for line in lines]
    return data


# Preprocess and split data
def preprocess_data(data):
    texts = [entry['heading'] for entry in data]  # Use 'heading' for training
    labels = [entry['category'] for entry in data]  # Categories as labels
    return texts, labels


# Train a Naive Bayes classifier
def train_classifier(texts, labels):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(texts, labels)
    return model


# Predict categories
def predict(model, new_data):
    texts = [entry['heading'] for entry in new_data]  # Extract texts from new data
    predictions = model.predict(texts)
    return predictions


def load_data_from_stdin():
    input_data = []
    for line in sys.stdin:
        # Assuming each line in stdin is a valid JSON object
        json_object = json.loads(line.strip())
        input_data.append(json_object)
    return input_data


# Main execution logic
def main(training_file, input_data):
    # Load and prepare training data
    train_data = load_data(training_file)
    train_texts, train_labels = preprocess_data(train_data)

    # Train the model
    model = train_classifier(train_texts, train_labels)

    # Assuming now we have a different way to read input, such as manually for testing:
    # new_data = [{'heading': 'Test post heading'}]  # Example new data
    # You should replace this line with actual data loading in production or competition
    new_data = load_data_from_stdin()  # Uncomment this line for actual stdin input
    predictions = predict(model, new_data)

    # Output predictions
    for prediction in predictions:
        print(prediction)
