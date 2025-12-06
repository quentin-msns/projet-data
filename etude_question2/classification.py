from pathlib import Path
from sqlalchemy import create_engine, inspect
import pandas as pd

# Définir le chemin vers la base de données
base_dir = Path(__file__).resolve().parent
db_path = base_dir / "question2.db"
engine = create_engine(f'sqlite:///{db_path}')

# Charger les réponses depuis la table top_texts
df = pd.read_sql("SELECT * FROM top_texts", engine)
print("Columns in top_texts:", df.columns.tolist())
# Assuming the first column is the text column
text_col = df.columns[0]
age_col = 'age' if 'age' in df.columns else df.columns[1]  # Assuming age is the second column
df = df[[text_col, age_col]].rename(columns={text_col: 'reponse', age_col: 'age'})

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df["reponse"])
def parse_age(age_str):
    # Extract the first number from the string, e.g., '30 à 39 ans' -> 30
    import re
    match = re.search(r'\d+', age_str)
    return int(match.group()) if match else 0

def age_class(age):
    if age < 18: return 0  # Moins de 18 ans
    if age < 30: return 1  # 18 à 29 ans
    if age < 40: return 2  # 30 à 39 ans
    if age < 50: return 3  # 40 à 49 ans
    if age < 65: return 4  # 50 à 64 ans
    if age < 75: return 5  # 65 à 74 ans
    return 6  # 75 ans et plus

df["age"] = df["age"].apply(parse_age)
df["age_class"] = df.age.apply(age_class)
y = df["age_class"]
from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X, y)
from sklearn.metrics import classification_report

pred = model.predict(X)
print(classification_report(y, pred))
