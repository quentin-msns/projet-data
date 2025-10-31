import pandas as pd
import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("cannabis_recreatif.csv", encoding="latin1", sep=';')

print(df.head)
import spacy

# chargement du modele spaCy pour la langue francaise
nlp = spacy.load("fr_core_news_sm")

doc = nlp(input_text)

# extraction des mots lemmatises
lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])