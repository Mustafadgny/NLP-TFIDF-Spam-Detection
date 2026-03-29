# NLP TF-IDF Vectorization on SMS Spam Dataset 📩

This repository demonstrates how to process text data and extract meaningful features using **TF-IDF (Term Frequency-Inverse Document Frequency)**. The script is applied to the popular SMS Spam Collection dataset to identify the most significant words used in spam and ham (normal) messages.

## 🚀 Pipeline Features
1. **Data Loading:** Handling the SMS Spam dataset (using `latin-1` encoding).
2. **Text Preprocessing:** - Lowercasing
   - Removing digits and punctuation (`re`)
   - Filtering short words and English stopwords (`nltk`)
3. **TF-IDF Extraction:** - Converting the cleaned text list into a sparse TF-IDF matrix using `sklearn`.
   - Calculating the average importance score of each word across the entire corpus.
4. **Analysis:** Creating a Pandas DataFrame to sort and display the top 10 most impactful words in the dataset.

## 🛠️ Libraries Used
- `pandas`
- `scikit-learn` (`TfidfVectorizer`)
- `nltk` (Stopwords)
- `re` (Regex)

## 💻 How to Run
1. Ensure the required libraries are installed (`pip install pandas scikit-learn nltk`).
2. Download the `sms_spam.csv` dataset and place it in the same directory as the script.
3. Run the script. It will output the top 10 words with the highest average TF-IDF scores.
