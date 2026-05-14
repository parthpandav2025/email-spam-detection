import re
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

stopwords = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]' , ' ' , text)
    words = text.split()
    words = [i for i in words if i not in stopwords]
    
    return " ".join(words)
