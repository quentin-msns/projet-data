import pandas as pd
from pathlib import Path
from unidecode import unidecode
import spacy
from sqlalchemy import create_engine

# Charger Spacy FR
nlp = spacy.load("fr_core_news_sm")

def lemmatize_text(text: str) -> str:
    """Renvoie le texte lemmatisé en français"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    doc = nlp(text)
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(lemmas)


# --- Chargement du CSV brut ---
base_dir = Path(__file__).resolve().parent
file_path = base_dir.parent / "etude_question2" /"top_responses.csv"

df = pd.read_csv(file_path, encoding="utf-8", sep=';')

# Assurer que les colonnes existent
expected_cols = ['response', 'sexe', 'age', 'profession']
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

# --- Nettoyage des accents uniquement sur la colonne texte ---
df["reponse_clean"] = df["response"].astype(str).map(unidecode)

# --- Lemmatisation ---
df["reponse_lem"] = df["reponse_clean"].map(lemmatize_text)

# --- Préparation du tableau final ---
df_sql = df[["response", "reponse_clean", "reponse_lem", "sexe", "age", "profession"]]

# --- Sauvegarde dans une nouvelle table SQL ---
db_path = base_dir / "question2.db"
engine = create_engine(f"sqlite:///{db_path}")

df_sql.to_sql("lemmatized_texts", engine, if_exists="replace", index=False)

print("Table 'lemmatized_texts' sauvegardée correctement dans question2.db")
