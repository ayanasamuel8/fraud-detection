# Fraud Detection in E-commerce and Financial Transactions

## Overview
This project develops and evaluates machine learning models to detect fraudulent transactions across two distinct datasets: an e-commerce dataset and a credit card transaction dataset. The primary goal is to perform in-depth exploratory data analysis (EDA), extensive feature engineering, and build robust models to identify and prevent fraud effectively.

## Current Progress

The project is currently in the **Data Preprocessing and Feature Engineering** phase. All work for this phase is complete and documented in the `notebooks/01_eda_and_preprocessing.ipynb` notebook.

### Key Accomplishments:

1.  **E-commerce Dataset (`Fraud_Data.csv`):**
    *   **Data Cleaning:** Corrected data types for time-based columns and IP addresses.
    *   **Advanced Feature Engineering:** Created several powerful predictive features, including:
        *   `time_to_purchase`: The time elapsed between a user's signup and their first purchase.
        *   `country_risk_rate`: A **Bayesian-adjusted fraud rate** for each country, providing a statistically robust risk score that avoids issues with low transaction counts.
        *   `device_user_count`: A critical anomaly detection feature that counts the number of unique user accounts associated with a single device ID.
        *   `purchase_hour` & `purchase_day_of_week`: Time-based features to capture behavioral patterns.
    *   **Final Processing:** The cleaned data with newly engineered features has been one-hot encoded and saved to `data/01_processed/ecommerce_fraud_processed.csv`.

2.  **Credit Card Dataset (`creditcard.csv`):**
    *   **Data Scaling:** The `Time` and `Amount` columns have been standardized using `StandardScaler` to prepare them for modeling.
    *   **Final Processing:** The scaled dataset has been saved to `data/01_processed/creditcard_processed.csv`.

## Project Structure

```
├── data/
│   ├── 00_raw/              # Original, immutable data files
│   └── 01_processed/        # Cleaned and feature-engineered data
├── models/                  # Saved model artifacts (e.g., .pkl files)
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb  # (Completed) EDA and feature engineering
│   ├── 02_model_building_ecommerce.ipynb # (Upcoming) Modeling for e-commerce data
│   └── 03_model_building_creaditcard.ipynb # (Upcoming) Modeling for credit card data
├── reports/                 # Generated reports or visualizations
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # Scripts for data processing logic
│   ├── modeling.py          # Scripts for model training and evaluation
│   └── utils.py             # Utility functions (e.g., load/save data)
├── tests/                   # Unit tests for source code
├── Dockerfile               # Defines the environment for containerization
└── requirements.txt         # Project dependencies
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ayanasamuel8/fraud-detection.git
    cd fraud-detection
    ```

2.  **Set up the environment:**
    *   **Option A: Using Pip**
        Create a virtual environment and install the required packages:
        ```bash
        python -m venv .venv
        source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
        pip install -r requirements.txt
        ```
    *   **Option B: Using Docker**
        Build and run the Docker container:
        ```bash
        docker build -t fraud-detection-app .
        docker run -it -p 8888:8888 -v "%cd%:/app" fraud-detection-app
        ```

3.  **Run the Notebooks:**
    Launch Jupyter Lab or Jupyter Notebook and navigate to the `notebooks/` directory to explore the analysis and run the modeling notebooks.
    ```bash
    jupyter lab
    ```

## Model Building Progress

Model development is underway in the following notebooks:

- `notebooks/02_model_building_ecommerce.ipynb`: Implements and compares multiple classification algorithms (Logistic Regression, XGBoost) for the e-commerce dataset. Includes hyperparameter tuning and evaluation using metrics suitable for imbalanced data.
- `notebooks/03_model_building_creaditcard.ipynb`: Applies similar modeling and evaluation techniques to the credit card dataset.

Each notebook documents:
- Data loading and preprocessing
- Model training and validation
- Performance metrics (Precision, Recall, F1, AUPRC)
- Feature importance analysis

## Continuous Integration (CI)

Automated CI is set up to ensure code quality and reproducibility:

- **Testing:** All code in the `src/` and `tests/` directories is automatically tested using `pytest`.
- **Linting:** Code style is checked with `flake8` to enforce PEP8 standards.
- **GitHub Actions:** The repository includes a workflow file (`.github/workflows/ci.yml`) that runs tests and linting on every push and pull request, ensuring that only high-quality code is merged.
- **Docker Build:** The CI pipeline also verifies that the Docker image builds successfully.

## Next Steps

*   Continue model development and evaluation in the modeling notebooks.
*   Integrate model deployment (API) and monitoring solutions.
*   Expand test coverage and add more robust validation for new features and models.