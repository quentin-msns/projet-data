import pandas as pd
import numpy as np
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

from scipy import sparse
df_matrix = pd.read_sql("SELECT * FROM similarity_matrix", engine)
M = sparse.csr_matrix((df_matrix['value'], (df_matrix['row'], df_matrix['col'])), shape=(750, 750))
from scipy.sparse.linalg import eigsh
vals, vecs = eigsh(M, k=2, which='LM')
x = vecs[:, 0]
y = vecs[:, 1]


X_pca = np.column_stack((x, y))

def kmeans_custom(X, k, random_state=42):
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    centroids_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[centroids_indices].copy()
    
    max_iter = 100
    tol = 1e-4
    for _ in range(max_iter):

        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
  
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        centroids = new_centroids
    
    return clusters, centroids

# Clustering K-Means on PCA data (supposons k=5 clusters, ajustable)
k = 3
clusters, centroids = kmeans_custom(X_pca, k, random_state=42)

# Analyser les caractéristiques des clusters
df_top['cluster'] = clusters

# Fonction pour parser l'âge
def parse_age(age_str):
    if age_str is None or age_str == '':
        return None
    import re
    match = re.search(r'\d+', str(age_str))
    return int(match.group()) if match else None

df_top['age_parsed'] = df_top['age'].apply(parse_age)

# Visualisation avec couleurs par cluster
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, y, c=clusters, cmap='viridis', s=5, alpha=0.8)
plt.title("Clustering K-Means sur les documents (projection PCA)")
plt.colorbar(scatter, label='Cluster')

# Ajouter les moyennes d'âge sur les centroids
for cluster_id in range(k):
    cluster_data = df_top[df_top['cluster'] == cluster_id]
    avg_age = cluster_data['age_parsed'].mean()
    centroid_x, centroid_y = centroids[cluster_id]
    plt.text(centroid_x, centroid_y, f'{avg_age:.1f}', fontsize=12, ha='center', va='center', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.show()

# Sauvegarde des clusters dans la base de données
df_clusters = pd.DataFrame({'cluster': clusters})
df_clusters.to_sql('clusters', engine, if_exists='replace', index=False)
print("Clusters sauvegardés dans la base de données.")

# Statistiques par cluster
color_names = ['yellow', 'blue', 'green']
for cluster_id in range(k):
    cluster_data = df_top[df_top['cluster'] == cluster_id]
    avg_age = cluster_data['age_parsed'].mean()
    color = color_names[cluster_id]
    print(f"Cluster {cluster_id}: Moyenne âge = {avg_age:.2f}, Couleur = {color}")
