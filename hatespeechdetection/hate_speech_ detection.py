from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load hate speech dataset, assuming CSV file contains headers and is comma-separated
data = pd.read_csv(r'C:\Users\madhu\OneDrive\Desktop\hatespeechdetection\labeled_data.csv')

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['class'], test_size=0.2, random_state=42)

# Creating a pipeline with TF-IDF vectorizer and LinearSVC classifier
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

# Training the model
model.fit(X_train, y_train)

# Predictions on test set to measure accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get text from form
    text = request.form['text']
    print("Received text:", text)
    
    # Perform prediction
    prediction = model.predict([text])[0]
    print("Prediction:", prediction)
    
    # Provide appropriate message based on prediction
    output = "Hate Speech" if prediction == 1 else "Non-Hate Speech"
    message = "This is hate speech. Please be respectful." if prediction == 1 else "This is not hate speech. Keep up the positive expression!"
    print("Output:", output)
    print("Message:", message)
    
    return render_template('result.html', text=text, prediction=output, message=message, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
