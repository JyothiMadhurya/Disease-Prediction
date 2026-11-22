# Disease Prediction & Treatment Recommendation

A machine-learning system that predicts likely diseases from a free-text list of symptoms and recommends a matching treatment.

## Results

**88.3% accuracy** on the test set (`Accuracy: 0.8833`), with a per-disease classification report and a chart that groups diseases by F1-score (perfect, moderate, poor).

## How it works

1. **Text preprocessing:** symptoms are lower-cased, split into individual terms and turned into features with **TF-IDF**.
2. **Model:** a **stacking ensemble** with Random Forest and XGBoost as base models and Logistic Regression as the meta-model, trained with stratified K-fold cross-validation.
3. **Treatment lookup:** each predicted disease is mapped to its treatment from a separate treatment dataset.
4. **Two ways to use it:**
   - **Interactive:** type comma-separated symptoms and get a list of possible diseases with treatments.
   - **Batch:** predict diseases and treatments for every row in a file and save them to `disease_diagnosis_treatment.csv`.

## Data

The notebook expects three files that are not included in this repository:

| File | Columns |
|---|---|
| `disease_train.csv.xls` | `symptoms`, `disease` |
| `disease_test.csv.xls` | `symptoms`, `disease` |
| `treatment_dataset.csv.xls` | `disease`, `treatment` |

## Run it

```bash
pip install pandas numpy scikit-learn xgboost matplotlib tabulate
jupyter notebook "ML Project.ipynb"
```

Update the file paths in the notebook to point to your copies of the datasets.

## Tech

Python · scikit-learn · XGBoost · pandas · NumPy · Matplotlib

See `readme.txt` for the original detailed project notes.
