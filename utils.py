import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Keep negations: "not good" must stay different from "good"
STOP = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
LEM = WordNetLemmatizer()

def clean(text: str) -> str:
    t = text.lower()                                  # normalization
    t = re.sub(r"http\S+|www\.\S+", " ", t)           # URLs
    t = re.sub(r"@\w+", " ", t)                       # user mentions
    t = re.sub(r"#", "", t)                           # keep hashtag word, drop '#'
    t = re.sub(r"[^a-z\s]", " ", t)                   # punctuation, digits, emojis
    tokens = [LEM.lemmatize(w, pos="v") for w in t.split()]   # running -> run
    tokens = [LEM.lemmatize(w) for w in tokens]               # flights -> flight
    return " ".join(w for w in tokens if w not in STOP and len(w) > 1)