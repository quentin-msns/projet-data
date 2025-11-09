# Projet Analyse Questionnaire Cannabis Récréatif
**Auteur : Hugo Drion et Quentin Mieussens**
Ce projet contient plusieurs scripts et données pour l'analyse d'un questionnaire sur le cannabis récréatif. L'objectif est de nettoyer, lemmatiser, calculer le TF-IDF et visualiser les résultats.

---

## Arborescence des fichiers

├── data/
│ ├── donnees/ # Contient les fichiers CSV bruts et traités
│ ├── packages/ # Librairies ou modules personnalisés si nécessaires
│ └── resultats/ # Graphiques, matrices ou fichiers de sortie générés
├── etude_question2/
│ ├── lemmatisation.py # Script pour lemmatiser la deuxième colonne du questionnaire
│ ├── tf_idf.py # Calcul du TF-IDF pour la question 2
│ └── cartographie.py # Visualisation des documents projetés sur les 2 vecteurs propres
├── colonne.py # Crée un CSV avec la colonne lemmatisée, en supprimant les mots de moins de 3 lettres et les chiffres
├── lemmatisation_globale.py # Sauvegarde un CSV avec les 7 colonnes des questions ouvertes lemmatisées
└── tf_idf_global.py # Import du CSV lemmatisé, calcul du TF-IDF global et génération d’un graphique avec les 15 mots les plus fréquents

---

## Description des scripts

### `colonne2.py`
- Nettoie et lemmatise une colonne spécifique du questionnaire.
- Supprime les mots de moins de 3 lettres et les chiffres.
- Sauvegarde un CSV avec la colonne traitée.

### `lemmatisation_globale.py`
- Lemmatisation de toutes les 7 colonnes de questions ouvertes.
- Sauvegarde un CSV global contenant toutes les colonnes lemmatisées.

### `tf_idf_global.py`
- Charge le CSV lemmatisé global.
- Calcule le TF-IDF global.
- Génère un graphique montrant les 15 mots les plus fréquents dans l’ensemble des réponses.

### `etude_question2/lemmatisation.py`
- Lemmatisation spécifique à la question 2 du questionnaire.

### `etude_question2/tf_idf.py`
- Calcul du TF-IDF pour la question 2.
- Analyse des termes les plus significatifs.

### `etude_question2/cartographie.py`
- Projection des documents sur les deux premiers vecteurs propres.
- Génération d’une cartographie des réponses pour visualisation.

---

## Usage

1. Placer les fichiers CSV bruts dans `data/donnees`.
2. Exécuter les scripts dans l’ordre souhaité :
   - Pour lemmatiser : `colonne.py` ou `lemmatisation_globale.py`
   - Pour analyser les TF-IDF : `tf_idf_global.py` ou `etude_question2/tf_idf.py`
   - Pour visualiser : `etude_question2/cartographie.py`
3. Les résultats (CSV, graphiques) seront sauvegardés dans `data/resultats`.

---

## Prérequis

- Python 3.8 ou supérieur
- Librairies Python :
  - `pandas`
  - `numpy`
  - `spacy` (`fr_core_news_md` pour le français)
  - `matplotlib` ou `plotly` pour les graphiques
  - `scipy` pour les matrices sparse
- Installer le modèle français spaCy si nécessaire :
```bash
python -m spacy download fr_core_news_md
```
## Notes

Les scripts appliquent un nettoyage préalable des textes : suppression des chiffres, mots courts et caractères spéciaux.

Les résultats TF-IDF et cartographie sont générés uniquement à partir des données lemmatisées.