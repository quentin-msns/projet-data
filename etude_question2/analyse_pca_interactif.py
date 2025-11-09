import pandas as pd
import numpy as np
import plotly.express as px
from scipy import sparse
from scipy.sparse.linalg import eigsh

# Charger la matrice
M = sparse.load_npz("data/resultats/matrice_similarite_q2_500.npz")

# Calcul des 2 premiers vecteurs propres
vals, vecs = eigsh(M, k=2, which='LM')

# Coords projetées
coords = vecs[:, :2] * np.sqrt(vals[:2])

# Charger le DataFrame des textes (ou les 500 plus longs que tu avais)
df = pd.read_csv("data/donnees/question2.csv", sep=";", encoding="utf-8")

# Option : récupérer les 5 mots les plus fréquents pour chaque ligne
def top_words(text, n=5):
    if not isinstance(text, str):
        return ""
    words = text.split()
    freq = pd.Series(words).value_counts().head(n).index
    return " ".join(freq)

df["top_words"] = df.iloc[:, 0].apply(top_words)

# DataFrame des coordonnées
df_coords = pd.DataFrame({
    "x": coords[:, 0],
    "y": coords[:, 1],
    "texte": df.iloc[:, 0],
    "top_words": df["top_words"]
})

# Graphique interactif
fig = px.scatter(
    df_coords,
    x="x", y="y",
    hover_data={"texte": True, "top_words": True},
    title="Cartographie des documents (2 vecteurs propres)",
)
fig.update_traces(marker=dict(size=8, opacity=0.7))
fig.show()
