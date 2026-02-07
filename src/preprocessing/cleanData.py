import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Keep negation words
stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')
stop_words.discard('never')


def clean_review(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

def clean_dataset(input_path, output_path):
    print("Loading dataset...")
    df = pd.read_csv(input_path)

    print("Cleaning reviews...")
    df["clean_review"] = df["review"].apply(clean_review)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    print("Saving cleaned dataset...")
    df.to_csv(output_path, index=False)

    print("Cleaning done")

if __name__ == "__main__":
    clean_dataset(
        "data/IMDB Dataset.csv",
        "data/cleaned_imdb.csv"
    )