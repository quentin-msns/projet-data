from pathlib import Path
from sqlalchemy import create_engine, inspect
import pandas as pd

# Définir le chemin vers la base de données
base_dir = Path(__file__).resolve().parent
db_path = base_dir / "question2.db"
engine = create_engine(f'sqlite:///{db_path}')

# Charger les réponses depuis la table classification_texts
df = pd.read_sql("SELECT * FROM classification_texts", engine)
print("Columns in top_texts:", df.columns.tolist())

text_col = df.columns[0]
age_col = 'age' if 'age' in df.columns else df.columns[1] 
df = df[[text_col, age_col]].rename(columns={text_col: 'reponse', age_col: 'age'})

df = df[df['age'].notna() & (df['age'] != '')]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,3),min_df=0.01, max_df=0.9,lowercase=True)

X = vectorizer.fit_transform(df["reponse"])
def parse_age(age_str):
    if age_str is None:
        return 0
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

from sklearn.metrics import classification_report, precision_score, accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

precisions = {}
accuracies = {}
from sklearn.linear_model import LogisticRegression
print("Training Logistic Regression model...")
model = LogisticRegression(
    C=2.0,
    penalty='l2',
    solver='liblinear',
    class_weight='balanced',
    max_iter=200
)
model.fit(X_train, y_train)


pred = model.predict(X_test)
print("Regression Logistique - Classification Report:")
print(classification_report(y_test, pred, zero_division=0))
precisions['Logistic Regression'] = precision_score(y_test, pred, average='macro', zero_division=0)
accuracies['Logistic Regression'] = accuracy_score(y_test, pred)
print(df["age_class"].value_counts())

"""
from sklearn.ensemble import RandomForestClassifier
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=30,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    n_jobs=-1
)
model.fit(X_train, y_train)


pred = model.predict(X_test)
print("Random Forest - Classification Report:")
print(classification_report(y_test, pred, zero_division=0))
precisions['Random Forest'] = precision_score(y_test, pred, average='macro', zero_division=0)
"""
print("gradient boosting classifier")
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    learning_rate=0.05,
    n_estimators=300,
    max_depth=3,
    subsample=0.8
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Gradient Boosting - Classification Report:")
print(classification_report(y_test, pred, zero_division=0))
precisions['Gradient Boosting'] = precision_score(y_test, pred, average='macro', zero_division=0)
accuracies['Gradient Boosting'] = accuracy_score(y_test, pred)

print("\n SVm Linear")
from sklearn.svm import LinearSVC

model = LinearSVC(
    C=0.5,
    class_weight='balanced'
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("SVM Linear - Classification Report:")
print(classification_report(y_test, pred, zero_division=0))
accuracies['SVM Linear'] = accuracy_score(y_test, pred)


print("\n Bayes Naif ")
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(
    alpha=0.5
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Naive Bayes - Classification Report:")
print(classification_report(y_test, pred, zero_division=0))
precisions['Naive Bayes'] = precision_score(y_test, pred, average='macro', zero_division=0)
accuracies['Naive Bayes'] = accuracy_score(y_test, pred)

print("\nAccuracy for each model:")
for model_name, accuracy in accuracies.items():
    print(f"{model_name}: {accuracy:.4f}")
