import pandas as pd
import numpy as np
import plotly.express as px
from scipy import sparse
from scipy.sparse.linalg import eigsh
from pathlib import Path
from sqlalchemy import create_engine
base_dir = Path(__file__).resolve().parent
n=500
# Connexion à la base de données
db_path = base_dir / "cannabis.db"
engine = create_engine(f'sqlite:///{db_path}')

# Charger la matrice de similarité depuis la base de données
df_matrix = pd.read_sql("SELECT * FROM similarity_matrix", engine)
M = sparse.csr_matrix((df_matrix['value'], (df_matrix['row'], df_matrix['col'])), shape=(n, n))

# Calcul des 2 premiers vecteurs propres
vals, vecs = eigsh(M, k=2, which='LM')

# Normalisation du signe des vecteurs propres pour un résultat identique partout
for i in range(vecs.shape[1]):
    if vecs[0, i] < 0:       # si la première valeur du vecteur propre est négative
        vecs[:, i] *= -1     # on change le signe (flip)

# Coords projetées
coords = vecs[:, :2] * np.sqrt(vals[:2])

# Charger le DataFrame des textes depuis la base de données (top 500 avec démographiques)
df_lemmatized = pd.read_sql("SELECT * FROM lemmatized_texts", engine)
col_name = df_lemmatized.columns[0]
df_lemmatized['word_count'] = df_lemmatized[col_name].astype(str).str.split().apply(len)
df_sorted = df_lemmatized.sort_values(by='word_count', ascending=False)
df_top500 = df_sorted.head(n).copy()
df_top500.drop(columns=['word_count'], inplace=True)

# Option : récupérer les 5 mots les plus fréquents pour chaque ligne
def top_words(text, n=5):
    if not isinstance(text, str):
        return ""
    words = text.split()
    freq = pd.Series(words).value_counts().head(n).index
    return " ".join(freq)

df_top500["top_words"] = df_top500[col_name].apply(top_words)

# DataFrame des coordonnées
df_coords = pd.DataFrame({
    "x": coords[:, 0],
    "y": coords[:, 1],
    "texte": df_top500[col_name],
    "top_words": df_top500["top_words"],
    "sexe": df_top500["sexe"],
    "age": df_top500["age"],
    "profession": df_top500["profession"]
})

# Graphique interactif
fig = px.scatter(
    df_coords,
    x="x", y="y",
    color="sexe",
    hover_data=["top_words", "sexe", "age", "profession"],
    title="Cartographie des documents",
)
fig.update_traces(marker=dict(size=8, opacity=0.7))
# Personnaliser le tooltip avec des retours à la ligne
fig.update_traces(hovertemplate='Top words: %{customdata[0]}<br>Sexe: %{customdata[1]}<br>Age: %{customdata[2]}<br>Profession: %{customdata[3]}<extra></extra>')
fig.show()