import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import re
def clean_text(text):
    text = text.lower()  # mettre en minuscules
    text = re.sub(r'\d+', '', text)  # supprimer les chiffres
    text = ' '.join([word for word in text.split() if len(word) >= 3]) # on garde les mots de 3 lettres et plus
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

df = pd.read_csv("cannabis_recreatif_lemmatise.csv", encoding="utf-8", sep=';')

df2 = df["Pensez vous que le dispositif actuel permet de lutter efficacement contre les trafics"].copy().dropna()
df2= df2.apply(clean_text)
df2.to_csv("question2.csv", index=False, sep=";", encoding="utf-8")
print(df2.head)
df2 = pd.read_csv("question2.csv", encoding="utf-8", sep=';')
corpus = df2["Pensez vous que le dispositif actuel permet de lutter efficacement contre les trafics"].astype(str).tolist()
# Nettoyage basique : minuscules, suppression caractères non alphabétiques
#textes = corpus.str.lower().str.replace('[^a-zàâçéèêëîïôûùüÿñæœ ]', ' ', regex=True)
print(corpus[:5])
#Vectorisation avec CountVectorizer
count_vectorizer = CountVectorizer(lowercase=True)
X_count = count_vectorizer.fit_transform(corpus)
matrice_vector = pd.DataFrame.sparse.from_spmatrix(X_count, columns=count_vectorizer.get_feature_names_out())
print("=== Matrice CountVectorizer ===")
print(matrice_vector.head())