
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    roc_auc_score
)
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.
    
    Parameters:
    data (pd.DataFrame): The DataFrame to save.
    file_path (str): The path where the CSV file will be saved.
    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def plot_confusion_matrix(y_true, y_pred, title):
    """
    Generates and displays a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Trains a given model, evaluates it, and prints a detailed report.

    Args:
        model: The machine learning model instance to train.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        model_name (str): A name for the model for printing headers.

    Returns:
        The trained model object.
    """
    print(f"--- Training and Evaluating: {model_name} ---")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions and get probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Evaluation ---
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    print(f"Area Under the Precision-Recall Curve (AUC-PR): {auc_pr:.4f}\n")

    # Display the confusion matrix
    plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix for {model_name}")

    return model