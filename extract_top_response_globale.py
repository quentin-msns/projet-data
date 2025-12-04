import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# Nombre de réponses à extraire
N = 500

# Chemin vers le fichier CSV
csv_path = Path(__file__).resolve().parent / "data" / "donnees" / "cannabis_recreatif.csv"

# Lecture du CSV
df = pd.read_csv(csv_path, encoding="latin1", sep=';')

# Sélection des colonnes pertinentes
df_selected = df[[
    "Vous pouvez préciser votre réponse.",
    "Vous pouvez préciser votre réponse..1",
    "Quel(s) autre(s) avantage(s) verriez-vous à l\x92assouplissement de la politique actuelle ?",
    "Selon vous y aurait-il une ou plusieurs autres priorités budgétaires ?",
    "Pour quelle(s) raison(s) ?",
    "Y a-t-il une ou des raisons supplémentaires pour lesquelles vous vous opposez à sa dépénalisation ou sa légalisation ?",
    'Vous êtes ? \xa0',  # Sexe
    'Vous avez ?',  # Âge
    'Quelle est votre catégorie socio-professionnelle ?'  # Profession
]].copy()

# Fusionner les 6 premières colonnes en une seule colonne 'response'
df_selected['response'] = df_selected.iloc[:, 0:6].fillna('').astype(str).agg(' '.join, axis=1)

# Sélectionner les colonnes finales
df_selected = df_selected[['response', df_selected.columns[6], df_selected.columns[7], df_selected.columns[8]]]

# Renommer les colonnes pour plus de clarté
df_selected.columns = ['response', 'sexe', 'age', 'profession']

# Calculer le nombre de mots dans la réponse
df_selected['word_count'] = df_selected['response'].astype(str).str.split().apply(len)

# Trier par nombre de mots décroissant et prendre les N premiers
df_top500 = df_selected.sort_values(by='word_count', ascending=False).head(N).copy()

# Supprimer la colonne word_count si non nécessaire
df_top500.drop(columns=['word_count'], inplace=True)
print(df_top500.head())


# Sauvegarde dans la table 'texte_brut'
df_top500.to_csv('top_responses.csv', index=False, encoding='utf-8', sep=';')

print(f"Les {N} réponses les plus longues ont été sauvegardées dans le csv.")
