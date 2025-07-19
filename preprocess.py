import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only once)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load the combined dataset
df = pd.read_csv("data/news.csv")

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Save cleaned dataset
df.to_csv("data/clean_news.csv", index=False)

print("✅ Text preprocessing complete! Saved to 'data/clean_news.csv'")
print(df[["text", "clean_text", "label"]].head(2))
