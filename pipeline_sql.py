import pandas as pd
import spacy
import re
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---------------------------------------------------------------
# 1) Charger spaCy
# ---------------------------------------------------------------
nlp = spacy.load("fr_core_news_md")

# ---------------------------------------------------------------
# 2) Charger le CSV brut
# ---------------------------------------------------------------
df = pd.read_csv("data/donnees/cannabis_recreatif.csv", encoding="latin1", sep=';')

df_questions = df[[
    "Vous pouvez préciser votre réponse.",
    "Vous pouvez préciser votre réponse..1",
    "Quel(s) autre(s) avantage(s) verriez-vous à l\x92assouplissement de la politique actuelle ?",
    "Selon vous y aurait-il une ou plusieurs autres priorités budgétaires ?",
    "Pour quelle(s) raison(s) ?",
    "Y a-t-il une ou des raisons supplémentaires pour lesquelles vous vous opposez à sa dépénalisation ou sa légalisation ?"
]]

df_questions.columns = ["q1", "q2", "q3", "q4", "q5", "q6"]

# ---------------------------------------------------------------
# 3) Nettoyage léger
# ---------------------------------------------------------------
def light_clean(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df_clean = df_questions.copy()
for col in df_clean.columns:
    df_clean[col] = df_clean[col].map(light_clean)

# ---------------------------------------------------------------
# 4) Compter les mots
# ---------------------------------------------------------------
def count_words(text):
    if not isinstance(text, str) or text == "":
        return 0
    return len(text.split())

df_clean["word_count"] = df_clean.apply(
    lambda row: sum(count_words(row[col]) for col in df_questions.columns),
    axis=1
)

# ---------------------------------------------------------------
# 5) Sélection des 1000 lignes les plus longues
# ---------------------------------------------------------------
df_top1000 = df_clean.sort_values(by="word_count", ascending=False).head(1000)

# ---------------------------------------------------------------
# 6) Lemmatisation
# ---------------------------------------------------------------
def lemmatize_text(text):
    if not isinstance(text, str) or text == "":
        return ""
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if token.is_alpha])

df_lemm = df_top1000.copy()
for col in ["q1","q2","q3","q4","q5","q6"]:
    print(f"Lemmatisation de {col}")
    df_lemm[col] = df_lemm[col].map(lemmatize_text)

# ---------------------------------------------------------------
# 7) SQL : enregistrement des textes lemmatisés
# ---------------------------------------------------------------
engine = create_engine("sqlite:///data/cannabis.db")
df_lemm.to_sql("questions_lemmatisees_top1000", engine, if_exists="replace", index=False)

print("\n✔️ Table SQL 'questions_lemmatisees_top1000' créée !")

# ---------------------------------------------------------------
# 8) TF-IDF sur les 1000 lignes lemmatisées (tout concaténé)
# ---------------------------------------------------------------
df_lemm["full_text"] = df_lemm[["q1","q2","q3","q4","q5","q6"]].agg(" ".join, axis=1)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
)

tfidf_matrix = vectorizer.fit_transform(df_lemm["full_text"])
feature_names = vectorizer.get_feature_names_out()

# ---------------------------------------------------------------
# 9) Construction d’un tableau TF-IDF exploitable
# ---------------------------------------------------------------
tfidf_matrix = vectorizer.fit_transform(df_lemm["full_text"])
feature_names = vectorizer.get_feature_names_out()

# Conversions correctes en 1D
tfidf_moyen = np.array(tfidf_matrix.mean(axis=0)).flatten()
tfidf_max = np.array(tfidf_matrix.max(axis=0).toarray()).flatten()  # max nécessite toarray()
documents_apparition = np.array((tfidf_matrix > 0).sum(axis=0)).flatten()

tfidf_scores = pd.DataFrame({
    "mot": feature_names,
    "tfidf_moyen": tfidf_moyen,
    "tfidf_max": tfidf_max,
    "documents_apparition": documents_apparition,
})

# tri (plus important d'abord)
tfidf_scores = tfidf_scores.sort_values(by="tfidf_moyen", ascending=False)

# ---------------------------------------------------------------
# 10) Enregistrement en SQL
# ---------------------------------------------------------------
tfidf_scores.to_sql("tfidf_top1000", engine, if_exists="replace", index=False)

print("✔️ Table SQL 'tfidf_top1000' créée !")
print("🔥 Pipeline complet terminé.")
