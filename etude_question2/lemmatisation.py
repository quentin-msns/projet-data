import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from unidecode import unidecode
import spacy
from scipy import sparse
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

from pathlib import Path
base_dir = Path(__file__).resolve().parent

# Chemin vers le fichier (en remontant d’un dossier)
file_path = base_dir.parent / "data" / "donnees" / "question2.csv"

# Lecture du CSV
df = pd.read_csv(file_path, encoding="utf-8", sep=';')

# Appliquer la suppression d'accents sur tout le DataFrame

df = df.map(lambda x: unidecode(str(x)) if isinstance(x, str) else x)
df = df.map(lemmatize_text) 

#on sauve le dataframe lemmatizé
chemin_fichier = base_dir.parent / "data" / "donnees" / "question2_lemmatise2.csv"
df.to_csv(chemin_fichier, sep=";", encoding="utf-8", index=False)