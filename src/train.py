import pandas as pd
import pickle
from sklearn.model_selction import train_test_split
sklearn.feature_extraction.text import TfidfVectorizer
sklearn.linear_model import LogisticRegression
sklearn.metrics import accuracy_score

preprocessing import clean_text
# now we load the data
data = pd.read_csv("../data/arabic_sentiment.csv")
#clean the data
data["text"] = data['text'].apply(clean_text)
#labels & features
x = data['text']
y = data['label']
#convert text to numbers
vectorizer = TfidVectorizer(ngram_range=(1, 2),  max_features=5000)
X_vec = vectorizer.fit_transform(X)
#split data
X_train, X_test, y_train, y_test = train_test_split( X_vec, y, test_size=0.2, random_state=42)
# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
with open("../model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("../vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")
