import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from pathlib import Path
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from collections import Counter

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

def parse_age(age_str):
    if pd.isna(age_str):
        return np.nan
    # Extract numbers from "18 à 29 ans" -> [18, 29]
    import re
    nums = re.findall(r'\d+', str(age_str))
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2
    elif len(nums) == 1:
        return int(nums[0])
    else:
        return np.nan

def get_top_words(text, n=5):
    if not isinstance(text, str):
        return ""
    words = text.split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(n)
    return ", ".join([word for word, count in top_words])

df["age_numeric"] = df["age"].apply(parse_age)

# Statistiques par cluster
print("\nStatistiques par cluster:")
for cluster in df["cluster"].unique():
    cluster_data = df[df["cluster"] == cluster]
    avg_age = cluster_data["age_numeric"].mean()
    print(f"\nCluster {cluster}:")
    print(f"  Moyenne d'âge: {avg_age:.2f}")
    print("  Effectifs par profession:")
    print(cluster_data["profession"].value_counts().to_string())
    # Top mots par cluster
    all_words = " ".join(cluster_data["reponse_lem"].dropna())
    top_words_cluster = get_top_words(all_words, 5)
    print(f"  Top mots: {top_words_cluster}")

df["top_words"] = df["reponse_lem"].apply(lambda x: get_top_words(x, 5))

# DataFrame pour la visualisation
df_coords = pd.DataFrame({
    "x": X_pca[:, 0],
    "y": X_pca[:, 1],
    "cluster": df["cluster"],
    "sexe": df["sexe"],
    "age": df["age"],
    "profession": df["profession"],
    "top_words": df["top_words"]
})

# Visualisation interactive
fig = px.scatter(
    df_coords,
    x="x", y="y",
    color="cluster",
    hover_data=["sexe", "age", "profession", "top_words"],
    title="PCA Clustering Results",
)
fig.update_traces(marker=dict(size=8, opacity=0.7))
# Personnaliser le tooltip
fig.update_traces(hovertemplate='Sexe: %{customdata[0]}<br>Age: %{customdata[1]}<br>Profession: %{customdata[2]}<br>Top mots: %{customdata[3]}<extra></extra>')
fig.show()
