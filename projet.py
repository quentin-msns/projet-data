import pandas as pd
import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("cannabis_recreatif.csv", encoding="latin1", sep=';')
df1 = df.head(10).copy()
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

df1["texte_lemmatise"] = df1["Vous pouvez préciser votre réponse."].apply(lemmatize_text)
df1.to_csv("cannabis_recreatif_lemmatise_10.csv", index=False, sep=';', encoding='utf-8')
# exemple sur une chaine simple (gardé pour référence)
#print(df.columns)
print(df1.columns.tolist())

print(df1[["Vous pouvez préciser votre réponse.", 'Vous pouvez préciser votre réponse..1','Quel(s) autre(s) avantage(s) verriez-vous à lassouplissement de la politique actuelle ?', 'Selon vous y aurait-il une ou plusieurs autres priorités budgétaires ?','Pour quelle(s) raison(s) ?','Y a-t-il une ou des raisons supplémentaires pour lesquelles vous vous opposez à sa dépénalisation ou sa légalisation ?','Vous pouvez déposer ici une contribution écrite. ']])