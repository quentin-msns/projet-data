from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import umap
import networkx as nx
import plotly.graph_objects as go

# =============== PARAMÈTRES ===============
k = 5         # nombre de voisins dans le graphe
n_top_words = 5
n = 750      # nombre de lignes utilisées

# =============== CHARGEMENT MATRICE ===============
base_dir = Path(__file__).resolve().parent
db_path = base_dir / "question2.db"
engine = create_engine(f'sqlite:///{db_path}')

df_matrix = pd.read_sql("SELECT * FROM similarity_matrix", engine)

M = sparse.csr_matrix(
    (df_matrix['value'], (df_matrix['row'], df_matrix['col'])),
    shape=(n, n)
)

# matrice de distances
D = 1 - M.toarray()


# =============== UMAP ===============
um = umap.UMAP(metric="precomputed", n_neighbors=15, min_dist=0.1, random_state=42)
coords = um.fit_transform(D)


# =============== Chargement données textuelles ===============
df_text = pd.read_sql("SELECT * FROM lemmatized_texts", engine)

# on garde les 750 plus longs si nécessaire
col_text = df_text.columns[0]
df_text["word_count"] = df_text[col_text].astype(str).str.split().apply(len)
df_text = df_text.sort_values("word_count", ascending=False).head(n).reset_index(drop=True)

# =============== Extraction des top N mots les plus fréquents ===============
def top_words(text, n=5):
    if not isinstance(text, str):
        return ""
    words = text.split()
    freq = pd.Series(words).value_counts().head(n).index
    return " ".join(freq)

df_text["top_words"] = df_text[col_text].apply(lambda t: top_words(t, n_top_words))


# =============== Construction du graphe k-NN ===============
nbrs = NearestNeighbors(n_neighbors=k, metric="precomputed")
nbrs.fit(D)

neighbors = nbrs.kneighbors(D, return_distance=False)

G = nx.Graph()
for i in range(n):
    for j in neighbors[i]:
        if i != j:
            G.add_edge(i, j)


# =============== Préparation des arêtes pour Plotly ===============
edge_x = []
edge_y = []

for u, v in G.edges():
    edge_x += [coords[u, 0], coords[v, 0], None]
    edge_y += [coords[u, 1], coords[v, 1], None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    line=dict(width=0.5, color='gray'),
    hoverinfo='none'
)


# =============== Préparation des noeuds ===============
hover_text = [
    f"<b>Top mots :</b> {df_text.loc[i, 'top_words']}<br>"
    for i in range(n)
]

node_trace = go.Scatter(
    x=coords[:, 0],
    y=coords[:, 1],
    mode='markers',
    hovertemplate="%{text}<extra></extra>",
    text=hover_text
)


# =============== PLOT FINAL ===============
fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    title="Graphe k-NN avec UMAP + Top 5 mots les plus fréquents",
    width=1200,
    height=800,
    showlegend=False
)

fig.show()
