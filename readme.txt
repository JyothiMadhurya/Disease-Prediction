# Disease Prediction and Treatment Recommendation System

## Overview
This project implements a machine-learning pipeline for predicting diseases based on symptoms and recommending corresponding treatments. The model leverages advanced natural language processing (NLP) techniques and machine learning algorithms, including ensemble methods like stacking classifiers. Visualizations and detailed performance analysis are also provided.

## Key Features
Symptom-based Disease Prediction: Predicts the most probable diseases from a given list of symptoms using TF-IDF vectorization and classification models.
Treatment Recommendation: Maps predicted diseases to their respective treatments using a provided treatment dataset.
User Interaction: Supports both automated file-based predictions and interactive symptom diagnosis through user input.
Performance Visualization: Provides a pie chart illustrating the distribution of diseases based on model performance categories (e.g., perfect, moderate, or poor classification).

## Data
### Input Datasets
1. **Training Dataset (`disease_train.csv.xls`)**:
   - Contains `symptoms` and `disease` columns to train the machine learning models.
2. **Treatment Dataset (`treatment_dataset.csv.xls`)**:
   - Contains `disease` and corresponding `treatment` columns.
3. **Test Dataset (`disease_test.csv.xls`)**:
   - Contains `symptoms` and `disease` columns for model evaluation.

### Output Dataset
- The final predictions and treatments are saved in `disease_diagnosis_treatment.csv`.

## Implementation
### Preprocessing
1. **TF-IDF Vectorization**: Text data of symptoms is converted into numerical features.
2. **Symptom Parsing**: Splits symptoms into lists for individual analysis and standardizes text to lowercase.

### Model Architecture
- **Ensemble Stacking Classifier**:
  - **Base Models**: Random Forest, XGBoost
  - **Meta Model**: Logistic Regression
  - Cross-validation is performed using stratified K-Folds.

### Performance Metrics
- **Accuracy**: Overall correctness of predictions.
- **Classification Report**: Detailed breakdown of precision, recall, and F1-score for each disease.

### Visualization
A pie chart demonstrates the classification performance across diseases. Categories include:
- Perfect (F1-score = 1.00)
- Moderate (0.50 ≤ F1-score < 1.00)
- Poor (F1-score < 0.50)

### User Interaction
1. **Interactive Diagnosis**: Users input symptoms as a comma-separated list. The system predicts potential diseases and their treatments.
2. **Batch Processing**: Predicts diseases and treatments for multiple samples stored in a file (e.g., `prediction_test.csv.xls`).

## Requirements
- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - tabulate
  - warnings

## How to Use
1. Ensure all required datasets (`disease_train.csv.xls`, `treatment_dataset.csv.xls`, and `disease_test.csv.xls`) are in the specified paths.
2. Install required libraries:
   
   pip install pandas numpy scikit-learn xgboost matplotlib tabulate
   
3. Run the Jupyter Notebook (`specific name`) using:
   
   jupyter notebook specific.ipynb
 
4. Follow the prompts or adjust the file paths in the script to test predictions.

## Output
- Prediction and treatment mapping are saved in `disease_diagnosis_treatment.csv`.
- Performance visualization is saved as `disease_classification_performance.png`.

## Customization
- Update file paths to point to your local datasets.

