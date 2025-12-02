import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# Nombre de réponses à extraire
N = 750

# Chemin vers le fichier CSV
csv_path = Path(__file__).resolve().parent.parent / "data" / "donnees" / "cannabis_recreatif.csv"

# Lecture du CSV
df = pd.read_csv(csv_path, encoding="latin1", sep=';')

# Sélection des colonnes pertinentes
df_selected = df[[
    'Vous pouvez préciser votre réponse..1',  # Réponse à la question 2
    'Vous êtes ? \xa0',  # Sexe
    'Vous avez ?',  # Âge
    'Quelle est votre catégorie socio-professionnelle ?'  # Profession
]].copy()

# Renommer les colonnes pour plus de clarté
df_selected.columns = ['response', 'sexe', 'age', 'profession']

# Calculer le nombre de mots dans la réponse
df_selected['word_count'] = df_selected['response'].astype(str).str.split().apply(len)

# Trier par nombre de mots décroissant et prendre les N premiers
df_top500 = df_selected.sort_values(by='word_count', ascending=False).head(N).copy()

# Supprimer la colonne word_count si non nécessaire
df_top500.drop(columns=['word_count'], inplace=True)
print(df_top500.head())
# Chemin vers la base de données
db_path = Path("question2.db")
engine = create_engine(f'sqlite:///{db_path}')

# Sauvegarde dans la table 'texte_brut'
df_top500.to_sql('texte_brut', engine, if_exists='replace', index=False)

print(f"Les {N} réponses les plus longues ont été sauvegardées dans la table 'texte_brut' de question2.db.")
