import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pathlib import Path
from scipy import sparse
base_dir = Path(__file__).resolve().parent

# Chemin vers le fichier (en remontant d’un dossier)
file_path = base_dir.parent / "data" / "donnees" / "question2_lemmatise2.csv"

# Lecture du CSV
df = pd.read_csv(file_path, encoding="utf-8", sep=';')
col_name = df.columns[0]
taille  = 500
#crée une nouvelle colonne avec le nombre de mots par ligne
df['word_count'] = df[col_name].astype(str).str.split().apply(len)
df_sorted = df.sort_values(by='word_count', ascending=False)#trie par nombre de mots décroissant
df_top10000 = df_sorted.head(taille).copy()#garde seulement les 100 lignes les plus longues
df_top10000.drop(columns=['word_count'], inplace=True)# Supprime la colonne word_count 

file_path = base_dir.parent / "data" / "donnees" / f"question2_lemmatise2_{taille}lignes.csv"
df_top10000.to_csv(file_path, sep=";", encoding="utf-8", index=False)
print("fichier sauvegardé :", file_path)
print(df_top10000)
corpus = df_top10000["Pensez vous que le dispositif actuel permet de lutter efficacement contre les trafics"].astype(str).tolist()
print(corpus[:5])
#Vectorisation avec CountVectorizer
count_vectorizer = CountVectorizer(lowercase=True)
X_count = count_vectorizer.fit_transform(corpus)
matrice_vector = pd.DataFrame.sparse.from_spmatrix(X_count, columns=count_vectorizer.get_feature_names_out())
print("=== Matrice CountVectorizer ===")
print(matrice_vector.head())

#vectorisation avec TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer( lowercase=True,min_df=5,max_df=0.80)
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
df_tfidf = pd.DataFrame.sparse.from_spmatrix(
    X_tfidf, 
    columns=tfidf_vectorizer.get_feature_names_out()
)
print("\n=== Matrice TF-IDF ===")
print(df_tfidf.head())

# similarité entre document
def similarite(d1,d2):
    produit_scalaire = d1.dot(d2)
    norm_1 = np.sqrt((d1 ** 2).sum())
    norm_2 = np.sqrt((d2 ** 2).sum())
    if norm_1 == 0 or norm_2 == 0:  # éviter la division par zéro
        return 0.0
    return produit_scalaire/(norm_1*norm_2)

matrice_similarite =[]

# Nombre de lignes
nb_lignes = df_tfidf.shape[0]
print("Nombre de lignes :", nb_lignes)



#paramètres
n = nb_lignes  # taille de la matrice
dtype = np.float32  # pour gagner de la place

#initialisation : matrice creuse au format COO (très pratique pour construire)
rows = []
cols = []
values = []

# on remplit le triangle inférieur (sans diagonale)
for i in range(n):
    if i%10 ==0:
        print(i/n *100,"%") 
    for j in range(i+1):  # triangle inférieur
        
        if i == j: #1 sur la diagonale
            rows.append(i)
            cols.append(j)
            values.append(1)
        else:
            d1 = df_tfidf.iloc[i]
            d2 = df_tfidf.iloc[j]
            sim = similarite(d1, d2)
            sim = round(sim, 2)  # arrondi à 2 décimales
            
            if sim != 0:
                rows.append(i)
                cols.append(j)
                values.append(sim)

#conversion en matrice creuse CSR
sparse_matrix = sparse.csr_matrix((values, (rows, cols)), shape=(n, n), dtype=dtype)

print(f"Nombre de valeurs non nulles : {sparse_matrix.nnz}")
print(f"Taille approximative : {sparse_matrix.data.nbytes / 1e6:.2f} Mo")

#sauvegarde compressée
file_path = base_dir.parent / "data" / "resultats" / f"matrice_similarite_q2_{taille}.npz"
sparse.save_npz(f"matrice_similarite_q2_{taille}.npz", sparse_matrix)
print("✅ Matrice sparse sauvegardée avec succès.")
