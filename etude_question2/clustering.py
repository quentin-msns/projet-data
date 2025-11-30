import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent

# Connexion à la base de données
db_path = base_dir / "question2.db"
engine = create_engine(f'sqlite:///{db_path}')

# Charger les top textes depuis la base de données
df_top = pd.read_sql("SELECT * FROM top_texts", engine)
col_name = df_top.columns[0]
corpus = df_top[col_name].astype(str).tolist()

# Recompute TF-IDF (même paramètres que dans tf_idf.py)
tfidf_vectorizer = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.80)
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# Clustering K-Means (supposons k=5 clusters, ajustable)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_tfidf)

# Charger les composantes PCA depuis analyse_pca.py (vecteurs propres)
from scipy import sparse
df_matrix = pd.read_sql("SELECT * FROM similarity_matrix", engine)
M = sparse.csr_matrix((df_matrix['value'], (df_matrix['row'], df_matrix['col'])), shape=(500, 500))
from scipy.sparse.linalg import eigsh
vals, vecs = eigsh(M, k=2, which='LM')
x = vecs[:, 0]
y = vecs[:, 1]

# Visualisation avec couleurs par cluster
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, y, c=clusters, cmap='viridis', s=5, alpha=0.8)
plt.title("Clustering K-Means sur les documents (projection PCA)")
plt.colorbar(scatter, label='Cluster')
plt.show()

# Sauvegarde des clusters dans la base de données
df_clusters = pd.DataFrame({'cluster': clusters})
df_clusters.to_sql('clusters', engine, if_exists='replace', index=False)
print("Clusters sauvegardés dans la base de données.")
