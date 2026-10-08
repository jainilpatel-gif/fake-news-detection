# 📰 Fake News Detection with NLP & Machine Learning

Classify news articles as **REAL** or **FAKE** using TF-IDF features and classical ML models.
Built on the ISOT *Fake and Real News* dataset (`True.csv`, `Fake.csv`).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/fake-news-detection/blob/main/notebooks/Fake_News_Detection.ipynb)

## 1. Problem statement
Misinformation spreads faster than fact-checkers can respond. This project builds a text classifier that
flags an article as likely fake or real from its title and body alone.

## 2. Dataset
| File | Rows | Source | Label |
|------|------|--------|-------|
| `True.csv` | 21,417 | Reuters news articles | 1 (Real) |
| `Fake.csv` | 23,481 | Websites flagged as unreliable | 0 (Fake) |

Columns: `title`, `text`, `subject`, `date`. After removing duplicates and empty texts: **38,658 articles**.

**Use cases:** social-media moderation, browser plug-ins, newsroom pre-screening, teaching NLP / text classification.

### ⚠️ Data-leakage findings (handled in the notebook)
- ~99.8% of real articles contain "(Reuters)"; only ~1.3% of fake ones do → tag is stripped.
- `subject` values never overlap between classes → column is **not** used as a feature.
- 6,240 duplicate/empty rows removed; 630 fake articles had empty text.
- Fake-news dates are partly corrupted (URLs/text in the date field), so the timeline plot uses valid dates only.

## 3. Tools & libraries
Python 3 · pandas · NumPy · scikit-learn (TF-IDF, Logistic Regression, Linear SVM, Naive Bayes, Random Forest) ·
matplotlib · seaborn · wordcloud · joblib · Google Colab · Git/GitHub

## 4. Visualisations
| Plot | What it shows |
|------|---------------|
| `class_balance.png` | Fairly balanced classes (Fake ≈ 45%, Real ≈ 55% after cleaning) |
| `subject_by_class.png` | Subject labels are disjoint between classes (leakage) |
| `text_length.png` | Length distributions overlap; length alone isn't a reliable signal |
| `timeline.png` | Volume of articles per month |
| `top_words.png` | Most frequent words per class |
| `model_comparison.png` | Accuracy of four models |
| `confusion_matrix.png` | Errors of the best model |
| `feature_importance.png` | Words that push a prediction toward Fake or Real |

## 5. Results (20% hold-out test set, 7,732 articles)
| Model | Accuracy |
|-------|----------|
| **Linear SVM** | **99.04%** |
| Logistic Regression | 98.78% |
| Random Forest | 97.74% |
| Naive Bayes | 95.89% |

Best model (Linear SVM): precision/recall ≈ 0.99 for both classes.

**Interpretation:** fake articles lean on web/social vocabulary ("video", "featured image", "getty images", "breaking", "watch"),
while real ones use wire-style reporting language (weekday names, "said statement", "spokesman", "reporters", "minister").
The model detects **writing style and source habits**, not factual truth.

## 6. Limitations
- Data is 2015–2018 and mostly US politics; expect lower accuracy on other topics, years or sources.
- Real = one outlet (Reuters), Fake = flagged sites → the model may partly learn "wire-agency style".
- Validate on another dataset (LIAR, FakeNewsNet) before drawing real-world conclusions.

## 7. Run it
**Google Colab:** click the badge above (after pushing to GitHub), upload `True.csv` + `Fake.csv` to `/content/`, then *Runtime → Run all*.

**Locally**
```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
# place True.csv and Fake.csv in data/ and update the paths in the notebook
jupyter notebook notebooks/Fake_News_Detection.ipynb
```

## 8. Repo structure
```
fake-news-detection/
├── notebooks/Fake_News_Detection.ipynb
├── images/                # generated plots
├── data/README.md         # where to get the CSVs (CSVs not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Future work
Add a Streamlit demo · try BERT/DistilBERT · cross-dataset evaluation · explainability with LIME/SHAP.

## License
MIT. Dataset © its original authors (Ahmed, Traore, Saad — ISOT, University of Victoria); cite them if you publish.
