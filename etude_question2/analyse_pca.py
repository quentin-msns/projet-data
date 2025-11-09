from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
M = sparse.load_npz("matrice_similarite_q2_500.npz") # on charge la matrice de similarité des 500 plus longues lignes
print(M.shape)
print(f"{M.nnz} valeurs non nulles")


# Calcul des 2 plus grands vecteurs propres (en valeur absolue)
vals, vecs = eigsh(M, k=2, which='LM')

print("Valeurs propres :", vals)
print("Vecteurs propres shape :", vecs.shape)


# Projection 2D
x = vecs[:, 0]
y = vecs[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=2, alpha=0.8)
plt.title("Cartographie spectrale basée sur la matrice de similarité")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.show()

