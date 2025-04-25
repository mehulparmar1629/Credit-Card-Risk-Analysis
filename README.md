# Credit Risk Analysis Using Data Mining Techniques

## Overview

This project explores **Credit Risk Analysis** using advanced data mining and machine learning techniques. The goal is to predict borrower defaults and provide insights into credit risk management for financial institutions. The analysis is based on the **German Credit Dataset** and other sources, employing robust models and feature selection methods to ensure reliable predictions.

---

## Features

- Preprocessing using label encoding, normalization, and SMOTE for class balancing.
- Exploratory Data Analysis (EDA) for data insights.
- Feature selection using Information Gain and Recursive Feature Elimination.
- Ensemble learning with Random Forest, Gradient Boosting, and XGBoost.
- Final model: A **stacked classifier** integrating multiple base models.

---

## Technologies

- **Languages**: Python
- **Libraries**: 
  - Data Processing: `pandas`, `numpy`, `scikit-learn`, `imblearn`
  - Visualization: `matplotlib`, `seaborn`, `shap`
  - Modeling: `xgboost`, `lightgbm`, `tensorflow`

---

## Dataset

- Source: [Kaggle](https://www.kaggle.com) and [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- **Files**:
  - `application_record.csv`: Static customer data.
  - `credit_record.csv`: Dynamic behavior data.

**Target variable**: Default risk (binary: default = 1, non-default = 0).

---

## Steps to Reproduce

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repo_name.git
   cd repo_name
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook Credit_Risk_Analysis.ipynb
   ```

---

## Results

- **Top models**: Random Forest, Gradient Boosting, XGBoost.
- **Best performance**: Stacked classifier with:
  - Accuracy: 85%
  - AUC: 0.83
  - F1 Score: 0.85

---

## Challenges & Future Scope

### Challenges:
- Class imbalance even after applying SMOTE.
- Interpretability of ensemble models.

### Future Scope:
- Incorporate SHAP values for explainability.
- Explore deep tabular transformers for structured data.
- Real-time deployment via Flask API.

---

## Contributions

Feel free to contribute! Please fork the repository, create a branch, and submit a pull request.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
