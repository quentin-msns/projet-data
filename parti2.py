import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt

df = pd.read_csv("cannabis_recreatif_lemmatise.csv", encoding="utf-8", sep=';')

# 1️⃣ Fusionne toutes les colonnes textuelles en un seul texte par ligne
df['texte_complet'] = df.apply(
    lambda x: ' '.join(x.dropna().astype(str)), axis=1
)
df['texte_complet'] = df['texte_complet'].apply(lambda x: x.encode('utf-8').decode('utf-8', errors='ignore') if isinstance(x, str) else x)
corpus = df['texte_complet'].tolist()
# 2️⃣ Récupère le corpus (liste de textes)
import re
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # mettre en minuscules
    text = re.sub(r'\d+', '', text)  # supprimer les chiffres
    text = re.sub(r'[^\w\s]', '', text)  # supprimer la ponctuation
    # supprimer les mots de moins de 3 lettres
    text = ' '.join([word for word in text.split() if len(word) >= 3])
    text = re.sub(r'\s+', ' ', text).strip()  # nettoyer les espaces
    return text

corpus = [clean_text(doc) for doc in corpus]



# 3️⃣ Vectorisation avec CountVectorizer
count_vectorizer = CountVectorizer(lowercase=True)
X_count = count_vectorizer.fit_transform(corpus)
df_sparse = pd.DataFrame.sparse.from_spmatrix(X_count, columns=count_vectorizer.get_feature_names_out())

print("=== Matrice CountVectorizer ===")
print(df_sparse.head())

# 4️⃣ Vectorisation avec TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer( lowercase=True,min_df=5,max_df=0.80)
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
df_tfidf = pd.DataFrame.sparse.from_spmatrix(
    X_tfidf, 
    columns=tfidf_vectorizer.get_feature_names_out()
)
print("\n=== Matrice TF-IDF ===")
print(df_tfidf.head())
# 5️⃣ (Optionnel) Top 15 mots les plus importants globalement
word_importance = df_tfidf.sum(axis=0).sort_values(ascending=False).head(15)

plt.figure(figsize=(10,5))
word_importance.plot(kind='bar')
plt.title("Top 15 mots les plus importants (TF-IDF global)")
plt.xlabel("Mot")
plt.ylabel("Score TF-IDF total")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
