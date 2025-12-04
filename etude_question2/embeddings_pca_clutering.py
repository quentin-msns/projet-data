import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent

# Connexion à la base de données
db_path = base_dir / "question2.db"
engine = create_engine(f'sqlite:///{db_path}')

df = pd.read_sql_table("embeddings_q2", engine)

# Convertir JSON → vecteurs
X = np.array([json.loads(v) for v in df["embedding"]])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0)
df["cluster"] = kmeans.fit_predict(X)
df.to_sql("clusters_q2", engine, if_exists="replace", index=False)

# Afficher les résultats
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cluster sizes:")
print(df["cluster"].value_counts())

# Visualisation
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('PCA Clustering Results')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(base_dir / "clustering_plot.png")
plt.show()
