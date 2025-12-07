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
# Remove rows where age is None or empty
df = df[df['age'].notna() & (df['age'] != '')]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df["reponse"])
def parse_age(age_str):
    if age_str is None:
        return 0
    # Extract the first number from the string, e.g., '30 à 39 ans' -> 30
    import re
    match = re.search(r'\d+', age_str)
    return int(match.group()) if match else 0

def age_class(age):
    if age < 30: return 1  # 18 à 29 ans
    if age < 50: return 2  # 30 à 49 ans
    if age >=50: return 3  # 50ans et plus

df["age"] = df["age"].apply(parse_age)
df["age_class"] = df.age.apply(age_class)
y = df["age_class"]

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=5000, class_weight="balanced")
model.fit(X_train, y_train)


pred = model.predict(X_test)
print(classification_report(y_test, pred, zero_division=0))
print(df["age_class"].value_counts())
