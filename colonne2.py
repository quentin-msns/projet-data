import pandas as pd
import spacy
import re

# Chargement du modèle spaCy français (meilleur que le "sm")
# Si besoin, installe-le avec :
#   python -m spacy download fr_core_news_md
nlp = spacy.load("fr_core_news_md")

# === 1️⃣ Lecture du CSV brut ===
df = pd.read_csv("cannabis_recreatif.csv", encoding="latin1", sep=';')

# === 2️⃣ Sélection des questions ouvertes ===
df_questions_ouvertes = df[[
    "Vous pouvez préciser votre réponse.",
    "Vous pouvez préciser votre réponse..1",
    "Quel(s) autre(s) avantage(s) verriez-vous à l\x92assouplissement de la politique actuelle ?",
    "Selon vous y aurait-il une ou plusieurs autres priorités budgétaires ?",
    "Pour quelle(s) raison(s) ?",
    "Y a-t-il une ou des raisons supplémentaires pour lesquelles vous vous opposez à sa dépénalisation ou sa légalisation ?"
]]

# === 3️⃣ Renommage (on garde tes noms originaux exacts) ===
df_questions_ouvertes.columns = [
    "Pensez vous que le dispositif actuel de répression de la consommation de cannabis permet d’en limiter l’ampleur",
    "Pensez vous que le dispositif actuel permet de lutter efficacement contre les trafics",
    "Quels autres avantages verriez vous à l’assouplissement de la politique actuelle",
    "Selon vous y aurait-il une ou plusieurs autres priorités budgétaires",
    "Pour quelles raisons En cas de légalisation ou de dépénalisation, seriez vous favorable à la possibilité pour les particuliers de cultiver à des fins personnelles un nombre de pieds de cannabis fixé par la loi",
    "Y a-t-il une ou des raisons supplémentaires pour lesquelles vous vous opposez à sa dépénalisation ou sa légalisation"
]

# === 4️⃣ Fonctions de nettoyage et lemmatisation ===

def clean_text(text: str) -> str:
    if pd.isna(text):
        return text
    text = str(text).encode("latin1", errors="ignore").decode("latin1", errors="ignore")
    text = text.lower()  # mettre en minuscules
    text = re.sub(r'\d+', '', text)  # supprimer les chiffres
    text = ' '.join([word for word in text.split() if len(word) >= 3]) # on garde les mots de 3 lettres et plus
    text = re.sub(r'\s+', ' ', text).strip()  
    # Supprime tout caractère non alphabétique
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

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

# === 5️⃣ Application du nettoyage ===
df_questions_ouvertes_lemmatise = df_questions_ouvertes.copy()
df_questions_ouvertes_lemmatise = df_questions_ouvertes_lemmatise.map(clean_text)

# === 6️⃣ Application de la lemmatisation seulement à la deuxième colonne ===
col = "Pensez vous que le dispositif actuel permet de lutter efficacement contre les trafics"
print(f"→ Lemmatisation de la colonne : {col}")
df2 = df_questions_ouvertes_lemmatise[col].map(lemmatize_text)

# Supprime les lignes totalement vides
df2 = df2.dropna(how="all")
# === 6️⃣ Sauvegarde du résultat ===
df2.to_csv("question2.csv", index=False, sep=";", encoding="utf-8")
print("\n✅ Fichier 'question2.csv' créé avec succès !")
