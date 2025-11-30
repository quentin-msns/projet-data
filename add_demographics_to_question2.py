import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# Chemin vers la base de données
db_path = Path("etude_question2/question2.db")
engine = create_engine(f'sqlite:///{db_path}')

# Charger la table existante 'lemmatized_texts'
df_lemmatized = pd.read_sql("SELECT * FROM lemmatized_texts", engine)

# Chemin vers le fichier CSV original
csv_path = Path("data/donnees/cannabis_recreatif.csv")

# Lecture du CSV original
df_original = pd.read_csv(csv_path, encoding="latin1", sep=';')

# Sélection des colonnes démographiques
demographics = df_original[[
    'Vous êtes ? \xa0',  # Sexe
    'Vous avez ?',  # Âge
    'Quelle est votre catégorie socio-professionnelle ?'  # Profession
]].copy()

# Renommer les colonnes
demographics.columns = ['sexe', 'age', 'profession']

# Assumer que les lignes correspondent par index (même ordre)
df_lemmatized['sexe'] = demographics['sexe']
df_lemmatized['age'] = demographics['age']
df_lemmatized['profession'] = demographics['profession']

# Sauvegarde de la table mise à jour
df_lemmatized.to_sql('lemmatized_texts', engine, if_exists='replace', index=False)

print("Colonnes démographiques ajoutées à la table 'lemmatized_texts'.")
