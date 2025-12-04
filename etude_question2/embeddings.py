import pandas as pd
import json
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from pathlib import Path
from sqlalchemy import create_engine

base_dir = Path(__file__).resolve().parent

# Connexion à la base de données
db_path = base_dir / "question2.db"
engine = create_engine(f'sqlite:///{db_path}')


# Charger la table lemmatisee
df = pd.read_sql_query("SELECT * FROM lemmatized_texts", engine)


# Charger le modèle
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# (meilleur que all-MiniLM-L6-v2 pour le français)

# Encoder
emb = model.encode(df["reponse_lem"].tolist()).tolist()

# Convertir en JSON pour SQL
df["embedding"] = [json.dumps(v) for v in emb]

# Sauver une nouvelle table
df.to_sql("embeddings_q2", engine, if_exists="replace", index=False)

print("✔️ Table SQL 'embeddings_q2' créée (avec embeddings SentenceTransformer).")
