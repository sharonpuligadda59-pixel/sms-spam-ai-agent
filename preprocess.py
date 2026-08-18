import re
import string

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "for", "with", "at",
    "by", "from", "up", "down", "this", "that", "these", "those", "it",
    "its", "i", "you", "he", "she", "we", "they", "them", "his", "her",
    "your", "my", "our", "their", "as", "so", "not", "no", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "should",
    "just", "than", "then", "there", "here", "am", "im",
}

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)
