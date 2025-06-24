import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the datasets
train_data = pd.read_csv('training.csv')
valid_data = pd.read_csv('validation.csv')

# Combine train and validation data
full_train_data = pd.concat([train_data, valid_data])

# Check for NaN values and handle them (drop rows with NaN)
full_train_data.dropna(subset=['label'], inplace=True)
X_train = full_train_data['text']
y_train = full_train_data['label']

# Create vectorizer with tweaked settings
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')  # Tweak the vectorizer settings

X_train_vec = vectorizer.fit_transform(X_train)

# Train a model (LogisticRegression in this case)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save the model and vectorizer
with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and Vectorizer Saved Successfully!")

# Evaluate on test data (optional step, can be skipped if not needed)
test_data = pd.read_csv('test.csv')
X_test = test_data['text']
y_test = test_data['label']

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

print("🔍 Test Data Report:")
print(classification_report(y_test, y_pred))
