# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re

# Download the stopwords dataset and set up the English stopwords list
nltk.download("stopwords") 
stop_words_eng = set(stopwords.words("english"))

# %% 1. Data Loading
# Load the SMS Spam Collection dataset
# Note: 'latin-1' encoding is typically required for this specific dataset
df = pd.read_csv("sms_spam.csv", encoding="latin-1")

# Extract the text messages (v2) and their corresponding labels (v1)
documents = df["v2"]
labels = df["v1"]

# %% 2. Text Preprocessing
def clean_text(text):

    text = text.lower() # Convert to lowercase
    text = re.sub(r"\d+", "", text) # Remove digits
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation marks
    
    # Filter out words shorter than 3 characters and remove stopwords
    text = " ".join([word for word in text.split() if len(word) > 2 and word not in stop_words_eng])
    
    return text

# Apply the cleaning function to all documents in the dataset
cleaned_doc = [clean_text(row) for row in documents]

# %% 3. TF-IDF Vectorization
# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the cleaned documents and transform them into a TF-IDF matrix
X = vectorizer.fit_transform(cleaned_doc)

# Get the vocabulary (list of unique words)
features_names = vectorizer.get_feature_names_out()

# Calculate the average TF-IDF score for each word across all documents
tfidf_score = X.mean(axis=0).A1

# %% 4. Results & Visualization
# Create a DataFrame to easily view the words and their corresponding TF-IDF scores
df_tfidf = pd.DataFrame({"word": features_names, "tfidf_score": tfidf_score})

# Sort the DataFrame by TF-IDF score in descending order to see the most important words
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)

# Print the top 10 most significant words in the dataset
print("Top 10 words by average TF-IDF score:")
print(df_tfidf_sorted.head(10))


