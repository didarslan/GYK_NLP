import re
import nltk

nltk.download('stopwords')

stopwords = set(nltk.corpus.stopwords.words('english'))

def preprocess(text):
    text = clean_text(text)
    text = remove_mention(text)
    text = remove_stopwords(text)
    return text.strip()

def clean_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def remove_mention(text):
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    return text

def remove_stopwords(text):
    text = " ".join([word for word in text.split() if word not in stopwords])  # Remove stopwords
    return text


# Data Augmentation

# Random Shuffle Data Augmentation

# I feel very sad -> sad label
# feel I very sad -> sad label
# sad I feel very -> sad label


def augment_text(text):
    # Example augmentation: synonym replacement
    words = text.split()
    random.shuffle(words)  # Shuffle words for simplicity
    return ' '.join(words)