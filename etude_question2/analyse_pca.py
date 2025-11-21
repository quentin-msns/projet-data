from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd
base_dir = Path(__file__).resolve().parent

# Connexion à la base de données
db_path = base_dir / "question2.db"
engine = create_engine(f'sqlite:///{db_path}')

# Charger la matrice de similarité depuis la base de données
df_matrix = pd.read_sql("SELECT * FROM similarity_matrix", engine)
M = sparse.csr_matrix((df_matrix['value'], (df_matrix['row'], df_matrix['col'])), shape=(500, 500))
print(M.shape)
print(f"{M.nnz} valeurs non nulles")


# Calcul des 2 plus grands vecteurs propres (en valeur absolue)
vals, vecs = eigsh(M, k=2, which='LM')

print("Valeurs propres :", vals)
print("Vecteurs propres shape :", vecs.shape)


# Projection 2D
x = vecs[:, 0]
y = vecs[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=5, alpha=0.8)
plt.title("Cartographie basée sur la matrice de similarité des 500 plus longs documents")

plt.show()

