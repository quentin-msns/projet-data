import pandas as pd
from pathlib import Path
from unidecode import unidecode
import spacy
from sqlalchemy import create_engine
from collections import Counter

# Charger Spacy FR
nlp = spacy.load("fr_core_news_sm")

def clean_common_words(text: str) -> str:
    """Enlève les mots usuels et les retours à la ligne de la colonne response"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    # Enlever les retours à la ligne
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Liste des mots usuels à enlever
    common_words = ['un', 'le', 'la', 'les', 'de', 'du', 'des', 'et', 'à', 'a', 'en', 'sur', 'dans', 'par', 'pour', 'avec', 'sans', 'sous', 'entre', 'contre', 'chez', 'vers', 'pendant', 'depuis', 'jusque', 'malgré', 'quoique', 'bien', 'que', 'qui', 'quoi', 'dont', 'où', 'lequel', 'laquelle', 'lesquels', 'lesquelles', 'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'notre', 'votre', 'leur', 'ma', 'ta', 'sa', 'mes', 'tes', 'ses', 'nos', 'vos', 'leurs', 'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'nous', 'vous', 'le', 'la', 'les', 'lui', 'leur', 'y', 'en', 'moi', 'toi', 'soi', 'nous', 'vous', 'eux', 'elles', 'même', 'mêmes', 'tel', 'telle', 'tels', 'telles', 'tout', 'toute', 'tous', 'toutes', 'autre', 'autres', 'même', 'mêmes', 'quel', 'quelle', 'quels', 'quelles', 'tel', 'telle', 'tels', 'telles', 'combien', 'comment', 'où', 'quand', 'pourquoi', 'comment', 'si', 'oui', 'non', 'peut-être', 'peut', 'être', 'avoir', 'être', 'faire', 'aller', 'venir', 'voir', 'savoir', 'pouvoir', 'devoir', 'vouloir', 'falloir', 'mettre', 'prendre', 'donner', 'dire', 'venir', 'partir', 'arriver', 'passer', 'rester', 'tomber', 'tenir', 'porter', 'entrer', 'sortir', 'parler', 'écouter', 'regarder', 'chercher', 'trouver', 'perdre', 'gagner', 'jouer', 'chanter', 'danser', 'lire', 'écrire', 'dessiner', 'peindre', 'courir', 'marcher', 'nager', 'voler', 'manger', 'boire', 'dormir', 'réveiller', 'habiter', 'travailler', 'étudier', 'apprendre', 'enseigner', 'connaître', 'comprendre', 'penser', 'croire', 'aimer', 'détester', 'préférer', 'vouloir', 'pouvoir', 'devoir', 'falloir', 'mettre', 'prendre', 'donner', 'dire', 'venir', 'partir', 'arriver', 'passer', 'rester', 'tomber', 'tenir', 'porter', 'entrer', 'sortir', 'parler', 'écouter', 'regarder', 'chercher', 'trouver', 'perdre', 'gagner', 'jouer', 'chanter', 'danser', 'lire', 'écrire', 'dessiner', 'peindre', 'courir', 'marcher', 'nager', 'voler', 'manger', 'boire', 'dormir', 'réveiller', 'habiter', 'travailler', 'étudier', 'apprendre', 'enseigner', 'connaître', 'comprendre', 'penser', 'croire', 'aimer', 'détester', 'préférer']
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in common_words]
    return ' '.join(filtered_words)

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



file_path = Path(__file__).resolve().parent / "top_responses.csv"

df = pd.read_csv(file_path, encoding="utf-8", sep=';')

# Assurer que les colonnes existent
expected_cols = ['response', 'sexe', 'age', 'profession']
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

# --- Nettoyage des mots usuels et retours à la ligne ---
df["response_clean"] = df["response"].astype(str).map(clean_common_words)

# --- Nettoyage des accents uniquement sur la colonne texte ---
df["reponse_clean"] = df["response_clean"].astype(str).map(unidecode)

# --- Comptage des mots et suppression du top n% ---
def remove_top_n_percent_words(text: str, n: float) -> str:
    """Enlève les mots du top n% les plus fréquents"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    all_words = []
    for t in df["reponse_clean"]:
        if isinstance(t, str):
            all_words.extend(t.split())
    word_counts = Counter(all_words)
    total_unique_words = len(word_counts)
    top_n_percent_count = int(total_unique_words * (n / 100))
    top_n_percent_words = set([word for word, _ in word_counts.most_common(top_n_percent_count)])
    words = text.split()
    filtered_words = [word for word in words if word not in top_n_percent_words]
    return ' '.join(filtered_words)

# Utiliser n=10 pour 10%
df["reponse_clean"] = df["reponse_clean"].map(lambda x: remove_top_n_percent_words(x, 10))

# --- Lemmatisation ---
df["reponse_lem"] = df["reponse_clean"].map(lemmatize_text)

# --- Préparation du tableau final ---
df_sql = df[["reponse_lem", "sexe", "age", "profession"]]

# --- Sauvegarde dans une nouvelle table SQL ---
db_path = "cannabis.db"
engine = create_engine(f"sqlite:///{db_path}")

df_sql.to_sql("lemmatized_texts", engine, if_exists="replace", index=False)

print("Table 'lemmatized_texts' sauvegardée correctement dans cannabis.db")
