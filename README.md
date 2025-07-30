# Fraud Detection in E-commerce and Financial Transactions

## Overview

This project aims to build and evaluate machine learning models for detecting fraudulent transactions in both e-commerce and financial (credit card) datasets. The workflow covers comprehensive exploratory data analysis (EDA), advanced feature engineering, robust model development, and interpretability, ensuring actionable insights for fraud prevention.

## Usage Flow

### 1. Clone and Set Up the Project

Start by cloning the repository and setting up your environment:

```bash
git clone https://github.com/ayanasamuel8/fraud-detection.git
cd fraud-detection
```

#### Environment Setup

- **Using Pip (Recommended for local development):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
- **Using Docker (For reproducible environments):**
    ```bash
    docker build -t fraud-detection-app .
    docker run -it -p 8888:8888 -v "%cd%:/app" fraud-detection-app
    ```

### 2. Data Preparation

All raw data is stored in `data/00_raw/`. Preprocessing and feature engineering are performed in the notebook:

- `notebooks/01_eda_and_preprocessing.ipynb`

Key steps include:
- Cleaning and transforming time-based and categorical features
- Engineering predictive features (e.g., Bayesian country risk, device-user anomalies)
- Scaling and encoding data for modeling

Processed datasets are saved to `data/01_processed/`.

### 3. Model Development

Modeling is organized by dataset:

- **E-commerce:**  
    `notebooks/02_model_building_ecommerce.ipynb`
- **Credit Card:**  
    `notebooks/03_model_building_creaditcard.ipynb`

Each notebook covers:
- Data loading and preprocessing
- Training multiple classifiers (Logistic Regression, XGBoost)
- Hyperparameter tuning
- Evaluation using metrics for imbalanced data (Precision, Recall, F1, AUPRC)
- Feature importance analysis

### 4. Model Explainability

Interpretability is essential for trust and actionable insights:

- `notebooks/04_model_interpretation.ipynb` uses SHAP for:
    - **Global explanations:** SHAP summary plots highlight influential features
    - **Local explanations:** SHAP force plots show feature contributions for individual predictions

**Visual Placeholders:**
- ![SHAP Summary Plot Placeholder](reports/analysis_images/shap_summary_placeholder.png)
- ![SHAP Force Plot Placeholder](reports/analysis_images/shap_force_placeholder.png)

### 5. Continuous Integration & Quality Assurance

Automated CI/CD ensures reliability:

- **Testing:** All code in `src/` and `tests/` is covered by `pytest`
- **Linting:** Enforced via `flake8` for PEP8 compliance
- **GitHub Actions:** `.github/workflows/ci.yml` runs tests, linting, and Docker builds on every push/pull request

### 6. Project Structure

```
├── data/
│   ├── 00_raw/              # Original data
│   └── 01_processed/        # Cleaned, feature-engineered data
├── models/                  # Saved model artifacts
├── notebooks/               # Jupyter notebooks for analysis and modeling
├── reports/                 # Visualizations and analysis outputs
├── src/                     # Source code for data processing and modeling
├── tests/                   # Unit tests
├── Dockerfile               # Containerization setup
└── requirements.txt         # Dependencies
```

## Next Steps

- Continue model development and evaluation
- Integrate model deployment (API) and monitoring
- Expand test coverage and validation for new features and models

---

**To get started:**  
1. Set up your environment  
2. Run preprocessing and modeling notebooks  
3. Explore model explanations  
4. Contribute and extend with new features or models

For questions or contributions, please open an issue or pull request on GitHub.
