import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
df = pd.read_csv("data/clean_news.csv")

# Drop rows with missing values
df = df.dropna(subset=["clean_text", "label"])

# Features and Labels
X = df["clean_text"]
y = df["label"]


# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vectors = vectorizer.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
import joblib
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved in 'model/' folder.")
import pickle

# Save the model
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

