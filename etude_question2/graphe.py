from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import networkx as nx

base_dir = Path(__file__).resolve().parent
n = 750

# Charger la matrice sparse
db_path = base_dir / "question2.db"
engine = create_engine(f'sqlite:///{db_path}')

df_matrix = pd.read_sql("SELECT * FROM similarity_matrix", engine)

M = sparse.csr_matrix(
    (df_matrix['value'], (df_matrix['row'], df_matrix['col'])),
    shape=(n, n)
)

# Matrice de similarité dense
M_dense = M.toarray()

# Distances = 1 - similarité
D = 1 - M_dense


# k-NN sur distances pré-calculées
k = 5
nbrs = KNeighborsClassifier(n_neighbors=k, metric="precomputed")

nbrs.fit(D)




