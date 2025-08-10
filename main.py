# NLP nedir?

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
#nltk.download('punkt_tab') # punkt_tab => Tokenizer

text = "Natural Language Processing is a branch of artificial intelligence."

# Tokenization
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
print(tokens)
#

# Stop-Word Removal
# is,the,on,at,in
from nltk.corpus import stopwords
#nltk.download('stopwords')

stop_words = set(stopwords.words('english')) #Dosyadaki kelimeleri oku.
filtered_tokens = [word for word in tokens if word not in stop_words]
print(filtered_tokens)
#

# Lemmatization -> Kök haline getirme
# running -> run
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
# v =>verb -> fiil
# n =>noun -> isim
# a => adjective -> sıfat
# r => adverb (zarf)
print(lemmatizer.lemmatize('running', pos='n'))


# Pos tagging => Part of Speech Tagging
#nltk.download('averaged_perceptron_tagger_eng')
from nltk import pos_tag

pos_tags = pos_tag(filtered_tokens)
print(pos_tags)
#


# NER => Named Entity Recognition
#nltk.download('maxent_ne_chunker_tab')
#nltk.download('words')

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
#

# N-gram Modeller , Word Embeddings

# N-gram Modeller

# Bir metindeki kelimelerin (ya da karakterlerin) ardışık gruplar halinde oluşturulmasıdır.

# NLP çok eğlenceli alan
# Unigram-Bigram-Trigram 


# Unigram (1n) = ["NLP", "çok", "eğlenceli", "alan"]
# Bigram (2n) = ["NLP çok", "çok eğlenceli", "eğlenceli alan"]
# Trigram (3n) = ["NLP çok eğlenceli", "çok eğlenceli alan"]

# Otomatik tamamlama, spam tespiti, yazım önerisi.
# Nerede kullanılır? => Dilin anlamını anlamaz. Sadece istatistiksel olarak kullanılır.

# Apple is a fruit.
# Apple is a company.

corpus = [
    "NLP çok eğlenceli alan",
    "Doğal dil işleme çok önemli",
    "Eğlenceli projeler yapıyoruz"
]

from sklearn.feature_extraction.text import TfidfVectorizer
# Unigram ve Bigram birlikte kullanılır.
vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True)

X = vectorizer.fit_transform(corpus)

print(f"Feature Names: {vectorizer.get_feature_names_out()}")
print(f"X: {X.toarray()}")

# Word Embedding
# Her kelimeye sayısal bir vektör ata. Bu vektörler sayesinde:
# Kelimeler arasındaki anlamsal yakınlık öğreniliyor.
# Aynı bağlam geçen kelimeler, uzayda da birbirine yakın olur.

# Araba -> [0.21, -0.43, 0.92, ........, 0.01] 100 veya 300+ boyutlu.

# Güzel ek özellik => Vektör cebiri bile yapılabilir.
# vec("king") - vec("man") + vec("woman") = vec("queen")

# Nerede kullanılır? 

# Derin öğrenme.
# Chatbot, anlamsal arama

corpus = [
    "NLP çok eğlenceli alan",
    "Doğal dil işleme çok önemli",
    "Eğlenceli projeler yapıyoruz"
]
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

#her cümleyi tokenize et. kelime listesi oluştur.
# kelimeleri parçala, liste haline getir.
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in corpus]
print("******")
print(tokenized_sentences)

model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=2)

print("*******")
print(model.wv['nlp'])
print("*******")
print(model.wv.most_similar('nlp'))


# Sentence Embedding


corpus = [
    "NLP çok eğlenceli alan",
    "NLP çok önemli",
    "Eğlenceli projeler yapıyoruz"
]
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in corpus]

model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=2)

# Ortalama Vektör Alma
import numpy as np

def sentence_vector(sentence):
    words = word_tokenize(sentence.lower())
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    return np.zeros(100)

vec1 = sentence_vector(corpus[0])
vec2 = sentence_vector(corpus[1])

print(vec1)
print(vec2)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# cosinesimilarty(a,b) = a.b / |a| * |b| => -1,1 arasında değer döner.
#
# Average Word Embedding

print(cosine_similarity(vec1, vec2))

# Sentence-BERT (SBERT)
# Sentence-BERT, cümleleri vektörlere dönüştürmek için kullanılan bir modeldir.
