from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from pathlib import Path
base_dir = Path(__file__).resolve().parent
file_path = base_dir.parent / "data" / "resultats" / "matrice_similarite_q2_500.npz"
M = sparse.load_npz(file_path) # on charge la matrice de similarité des 500 plus longues lignes
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

