import pandas as pd
import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("cannabis_recreatif.csv", encoding="latin1", sep=';')

# print(df.head)
import spacy

# chargement du modele spaCy pour la langue francaise
nlp = spacy.load("fr_core_news_sm")

# fonction pour lemmatiser le texte
def lemmatize_text(text):
    if pd.isna(text):
        return text
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

df["texte_lemmatise"] = df["Vous pouvez préciser votre réponse."].apply(lemmatize_text)
df.to_csv("cannabis_recreatif_lemmatise.csv", index=False, sep=';', encoding='utf-8')
# exemple sur une chaine simple (gardé pour référence)
doc = nlp("Les étudiants apprennent le traitement automatique du langage naturel avec spaCy.")
lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
#print(lemmatized_text)
