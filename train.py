"""
Model Training & Evaluation Script for Siemens PdM Digital Twin
"""
import os
import sys

# Ensure package directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_loader import load_data
from src.pdm_engine import train_pdm_models

def main():
    print("==================================================")
    print("🚀 Siemens Asset Digital Twin & PdM - Model Training")
    print("==================================================")
    df = load_data()
    metrics = train_pdm_models(df)
    print("\nTraining Run Completed Successfully!")
    print(f"• RUL Regressor RMSE: {metrics['rul_rmse']:.2f} hrs")
    print(f"• Fault Type Classifier Accuracy: {metrics['fault_type_accuracy']*100:.2f}%")
    print(f"• Component Classifier Accuracy: {metrics['comp_accuracy']*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
