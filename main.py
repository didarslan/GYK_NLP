# main.py

# NLP ile neler yapıyoruz? => 
# Metin Sınıflandırma -> E posta spam mı değil mi?
# Duygu Analizi -> (mutlu mu ? üzgün mü?)
# Özetleme -> 
# Metin Üretimi -> 
# Chatbot 
# Named Entity Recognition


# Bölüm 1 Müfredat Konuları

# Kütüphaneler -> numpy/pandas 
# NLTK => Temel nlp işlemleri yapan.
# scikit-learn


import nltk 
nltk.download('punkt_tab') # punkt_tab => Tokenizer

text = "Natural Language Processing is a branch of artificial intelligence."

# Tokenization
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
print(tokens)
#

# Stop-Word Removal
# is,the,on,at,in
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english')) #Dosyadaki kelimeleri oku.
filtered_tokens = [word for word in tokens if word not in stop_words]
print(filtered_tokens)
#

# Lemmatization -> Kök haline getirme
# running -> run
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
# v =>verb -> fiil
# n =>noun -> isim
# a => adjective -> sıfat
# r => adverb (zarf)
print(lemmatizer.lemmatize('running', pos='n'))


# Pos tagging => Part of Speech Tagging
nltk.download('averaged_perceptron_tagger_eng')
from nltk import pos_tag

pos_tags = pos_tag(filtered_tokens)
print(pos_tags)
#


# NER => Named Entity Recognition
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

from nltk import ne_chunk
tree = ne_chunk(pos_tags)
print(tree)
#


# You have chosen
# YoU hAvE ChOSen

# Metin temizleme ve ön işleme 
# Lowercasing
text = "Natural Language Processing is, a branch of artificial intelligence. %100"

text = text.lower()
print(text)
#

# Remove Punctuation
import re
text = re.sub(r'[^\w\s]', '', text) #Regex => Regular Expression
print(text)
#

#
text = re.sub(r'\d+', '', text)
print(text)
#


# Vectorize Etmek

# Bag Of Words
corpus = [
    "Natural Language Processing is a branch of artificial intelligence.",
    "I love studying NLP.",
    "Language is a tool for communication.",
    "Language models can understand texts."
]
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
#

# Tf-Idf -> Term Frequency - Inverse Document Frequency

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer2 = TfidfVectorizer()
X2 = vectorizer2.fit_transform(corpus)

print(vectorizer2.get_feature_names_out())
print(X2.toarray())


# Fonkisyon 

# pipeline => 
# 1-Tokenization - lowercasing 
# 2- Stopwords Temizliği
# 3- Lemmatization
# 4- TF-IDF Vektörleştirme
# 5- Feature isimlerini ve arrayi ekrana yazdır.

# generate a corpus of 10 about AI in english
corpus = [
    "Artificial Intelligence is the future.",
    "AI is changing the world.",
    "AI is a branch of computer science.",
]

# Word Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

# Her kelimeye sayısal bir vektör atamak için kullanılır. 
# Bu vektörler, kelimelerin anlamını ve bağlamını yakalamaya çalışır.
# Aynı bağlamdaki kelimeler, uzayda birbirine yakın vektörlere sahip olur.

# Araba [0.1, 0.2, 0.3........, 0.9] 100 veya 300 boyutlu bir vektör.

# Ek özellik = Vektör cebiri yapılabilir

# vec("king") - vec("man") + vec("women") = vec("queen")

# Nerede kullanılır?
# 1- Chatbotlar
# 2- Semantic search (Anlamlı arama)

corpus = [
    "NLP çok eğlenceli bir alandır.",
    "Doğal dil işleme çok önemli",
    "Eğlenceli projeler yapıyoruz",
]

import gensim 
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Her cümleyi tokenize et kelime listesine çevir
tokenized_senrtences = [word_tokenize(sentence.lower()) for sentence in corpus]
print(tokenized_senrtences)
# Word2Vec Modeli
model = Word2Vec(sentences=tokenized_senrtences, vector_size=100, window=5, min_count=1, workers=4)
print(model.wv['nlp'])  # 'eğlenceli' kelimesinin vektör temsili
