import sqlite3
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import eigsh
from pathlib import Path

# -------------------------------------------------------------
# 1) Connexion à SQLite
# -------------------------------------------------------------
base_dir = Path(__file__).resolve().parent
db_path = base_dir/ "data" /"cannabis.db"
conn = sqlite3.connect(db_path)

print("✔️ Connexion OK")

# -------------------------------------------------------------
# 2) Récupérer les textes lemmatisés
# -------------------------------------------------------------
df = pd.read_sql_query("""
    SELECT id, full_text
    FROM questions_lemmatisees_top1000
    ORDER BY id
""", conn)
df['id'] = range(1, len(df) + 1)
df.to_sql('questions_lemmatisees_top1000', conn, if_exists='replace', index=False)

texts = df["full_text"].fillna("").tolist()

print(f"✔️ {len(texts)} textes chargés")

# -------------------------------------------------------------
# 3) TF-IDF
# -------------------------------------------------------------
vectorizer = TfidfVectorizer(min_df=2)
tfidf = vectorizer.fit_transform(texts)

print("✔️ TF-IDF calculé :", tfidf.shape)

# -------------------------------------------------------------
# 4) Matrice de similarité cosinus
# -------------------------------------------------------------
# produit matriciel (plus rapide)
similarity = (tfidf @ tfidf.T).toarray()

print("✔️ Matrice de similarité créée :", similarity.shape)

# -------------------------------------------------------------
# 5) Sauvegarde dans SQLite
#  → format long : (i, j, value)
# -------------------------------------------------------------
similarity_df = pd.DataFrame([
    (i, j, similarity[i, j])
    for i in range(similarity.shape[0])
    for j in range(similarity.shape[1])
], columns=["i", "j", "value"])

similarity_df.to_sql("matrice_q2", conn, if_exists="replace", index=False)

print("✔️ Matrice enregistrée dans TABLE 'matrice_q2'")

# -------------------------------------------------------------
# 6) ACP = valeurs propres (2 premières)
# -------------------------------------------------------------
# convertir en matrice creuse pour eigsh
M = csr_matrix(similarity)

vals, vecs = eigsh(M, k=2, which="LM")

coords = vecs[:, :2] * np.sqrt(vals[:2])

print("✔️ PCA calculée")

# -------------------------------------------------------------
# 7) Extraire les mots fréquents
# -------------------------------------------------------------
def top_words(text, n=5):
    words = text.split()
    if len(words) == 0:
        return ""
    freq = pd.Series(words).value_counts().head(n).index
    return " ".join(freq)

df["top_words"] = df["full_text"].apply(top_words)

# -------------------------------------------------------------
# 8) Sauvegarde PCA dans SQL
# -------------------------------------------------------------
df_pca = pd.DataFrame({
    "id": df["id"],
    "x": coords[:, 0],
    "y": coords[:, 1],
    "top_words": df["top_words"]
})

df_pca.to_sql("pca_q2", conn, if_exists="replace", index=False)

print("🎉 PCA enregistrée dans 'pca_q2' !")

conn.close()
