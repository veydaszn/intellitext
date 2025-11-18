from preprocess import load_and_preprocess
from train_model import train_classifier
from predict import predict_text


print("=== IntelliText: AI Text Classification Toolkit ===")


# Load and preprocess data
X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess("../data/sample_dataset.csv")


# Train model
model, acc = train_classifier(X_train, y_train, X_test, y_test)
print(f"\nModel trained successfully! Accuracy: {acc * 100:.2f}%")


# Predict
while True:
user_input = input("\nEnter text to classify (or 'exit'): ")
if user_input.lower() == "exit":
break
prediction = predict_text(model, vectorizer, user_input)
print("Predicted Label:", prediction)
