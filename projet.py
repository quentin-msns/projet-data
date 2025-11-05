import pandas as pd
import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("cannabis_recreatif.csv", encoding="latin1", sep=';')
df1 = df.head(10).copy()
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
#print(df1.columns.tolist())

df_questions_ouvertes = df[["Vous pouvez préciser votre réponse.", 'Vous pouvez préciser votre réponse..1','Quel(s) autre(s) avantage(s) verriez-vous à l\x92assouplissement de la politique actuelle ?',
                'Selon vous y aurait-il une ou plusieurs autres priorités budgétaires ?','Pour quelle(s) raison(s) ?',
                'Y a-t-il une ou des raisons supplémentaires pour lesquelles vous vous opposez à sa dépénalisation ou sa légalisation ?']]


df_questions_ouvertes.columns=['Pensez vous que le dispositif actuel de répression de la consommation de cannabis permet d’en limiter l’ampleur',
                               'Pensez vous que le dispositif actuel permet de lutter efficacement contre les trafics',
                               'Quels autres avantages verriez vous à l’assouplissement de la politique actuelle',
                               'Selon vous y aurait-il une ou plusieurs autres priorités budgétaires',
                               'Pour quelles raisons En cas de légalisation ou de dépénalisation, seriez vous favorable à la possibilité pour les particuliers de cultiver à des fins personnelles un nombre de pieds de cannabis fixé par la loi',
                               'Y a-t-il une ou des raisons supplémentaires pour lesquelles vous vous opposez à sa dépénalisation ou sa légalisation']
print(df_questions_ouvertes)


df_questions_ouvertes_lemmatise=df_questions_ouvertes.copy()

for col in df_questions_ouvertes_lemmatise.columns:
    df_questions_ouvertes_lemmatise[col] = df_questions_ouvertes_lemmatise[col].map(lemmatize_text)
    
df_questions_ouvertes_lemmatise.to_csv("cannabis_recreatif_lemmatise.csv", index=False, sep=';', encoding='utf-8')

