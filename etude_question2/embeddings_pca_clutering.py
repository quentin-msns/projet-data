import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
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

# DataFrame pour la visualisation
df_coords = pd.DataFrame({
    "x": X_pca[:, 0],
    "y": X_pca[:, 1],
    "cluster": df["cluster"],
    "sexe": df["sexe"],
    "age": df["age"],
    "profession": df["profession"]
})

# Visualisation interactive
fig = px.scatter(
    df_coords,
    x="x", y="y",
    color="cluster",
    hover_data=["sexe", "age", "profession"],
    title="PCA Clustering Results",
)
fig.update_traces(marker=dict(size=8, opacity=0.7))
# Personnaliser le tooltip
fig.update_traces(hovertemplate='Sexe: %{customdata[0]}<br>Age: %{customdata[1]}<br>Profession: %{customdata[2]}<extra></extra>')
fig.show()
