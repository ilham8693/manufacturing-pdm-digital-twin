"""
Predictive Maintenance (PdM) & Digital Twin Physics & ML Engine
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RUL_MODEL_PATH = os.path.join(MODELS_DIR, "rul_model.pkl")
FAULT_TYPE_MODEL_PATH = os.path.join(MODELS_DIR, "fault_type_model.pkl")
FAULT_COMP_MODEL_PATH = os.path.join(MODELS_DIR, "fault_comp_model.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")

# Standardized Feature Set
FEATURE_COLS = [
    'Vibration_X', 'Vibration_Y', 'Vibration_Z', 'RMS_Vibration', 'Peak_Vibration',
    'Bearing_Temperature', 'Motor_Temperature', 'Gearbox_Temperature', 'Oil_Temperature',
    'Oil_Viscosity', 'Oil_Particle_Count', 'Coolant_Temperature', 'Coolant_Flow_Rate',
    'Hydraulic_Pressure', 'Pneumatic_Pressure', 'Air_Flow_Rate',
    'Voltage_Phase_A', 'Voltage_Phase_B', 'Voltage_Phase_C',
    'Current_Phase_A', 'Current_Phase_B', 'Current_Phase_C',
    'Power_Factor', 'Power_Consumption', 'Energy_Efficiency_Index',
    'Shaft_Speed_RPM', 'Load_Torque', 'Tool_Wear_Level', 'Acoustic_Emission_Level'
]

def classify_iso_vibration(rms_vibration: float) -> Tuple[str, str]:
    """
    Evaluates vibration velocity severity based on ISO 10816-3 Standard (Medium/Large Machines):
      - Zone A (< 1.8 mm/s): Good condition (Newly commissioned).
      - Zone B (1.8 - 4.5 mm/s): Satisfactory (Unrestricted long-term operation).
      - Zone C (4.5 - 7.1 mm/s): Unsatisfactory (Remedial maintenance required).
      - Zone D (> 7.1 mm/s): Unacceptable (Immediate shutdown danger).
    """
    if rms_vibration < 1.8:
        return "Zone A: Good", "#10b981"
    elif rms_vibration < 4.5:
        return "Zone B: Satisfactory", "#3b82f6"
    elif rms_vibration < 7.1:
        return "Zone C: Unsatisfactory", "#f59e0b"
    else:
        return "Zone D: Unacceptable / Critical", "#ef4444"

def calculate_health_index(row: Any) -> float:
    """
    Computes a composite Asset Health Index (0% - 100%) incorporating 4 physical domains:
      - Rotodynamic & Vibration Health (35%)
      - Thermal Dissipation & Temperature Gradient (25%)
      - Tribology, Viscosity & Debris Contamination (25%)
      - 3-Phase Electrical Quality & Power Factor (15%)
    """
    # 1. Vibration Health
    rms_vib = float(row.get('RMS_Vibration', 1.0))
    vib_penalty = min(100.0, (rms_vib / 7.1) * 100.0)
    vib_health = max(0.0, 100.0 - vib_penalty)
    
    # 2. Thermal Health
    b_temp = float(row.get('Bearing_Temperature', 60.0))
    m_temp = float(row.get('Motor_Temperature', 100.0))
    temp_penalty = min(100.0, max(0.0, (b_temp - 50.0) * 1.5 + (m_temp - 90.0) * 0.8))
    thermal_health = max(0.0, 100.0 - temp_penalty)
    
    # 3. Oil / Tribology Health
    particles = float(row.get('Oil_Particle_Count', 500.0))
    oil_penalty = min(100.0, (particles / 1000.0) * 100.0)
    oil_health = max(0.0, 100.0 - oil_penalty)
    
    # 4. Electrical Quality Health
    pf = float(row.get('Power_Factor', 0.85))
    elec_health = min(100.0, max(0.0, pf * 100.0))
    
    composite = (vib_health * 0.35) + (thermal_health * 0.25) + (oil_health * 0.25) + (elec_health * 0.15)
    return round(float(np.clip(composite, 0.0, 100.0)), 1)

def train_pdm_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains and serializes:
      1. Gradient Boosting Regressor for Remaining Useful Life (RUL) estimation.
      2. Random Forest Classifier for Failure Mode identification (Mechanical/Electrical/Software).
      3. Random Forest Classifier for Damaged Component identification (Bearing/Motor/Gearbox).
    """
    print(f"Preparing dataset with {len(df):,} samples...")
    X = df[FEATURE_COLS].copy().fillna(0)
    
    # 1. RUL Model
    y_rul = df['RUL']
    X_train, X_test, y_train, y_test = train_test_split(X, y_rul, test_size=0.2, random_state=42)
    
    print("Training RUL Gradient Boosting Regressor...")
    rul_model = HistGradientBoostingRegressor(max_iter=100, max_depth=8, learning_rate=0.1, random_state=42)
    rul_model.fit(X_train, y_train)
    rul_preds = rul_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, rul_preds)))
    print(f"✅ RUL Model RMSE: {rmse:.2f} hours")
    joblib.dump(rul_model, RUL_MODEL_PATH)
    
    # 2. Fault Type Model
    df_fault_type = df.dropna(subset=['Failure_Type'])
    ft_acc = 0.0
    if len(df_fault_type) > 0:
        print(f"Training Fault Type Classifier on {len(df_fault_type):,} labeled samples...")
        X_ft = df_fault_type[FEATURE_COLS].fillna(0)
        y_ft = df_fault_type['Failure_Type']
        X_tr, X_ts, y_tr, y_ts = train_test_split(X_ft, y_ft, test_size=0.2, random_state=42)
        ft_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        ft_model.fit(X_tr, y_tr)
        ft_acc = float(accuracy_score(y_ts, ft_model.predict(X_ts)))
        joblib.dump(ft_model, FAULT_TYPE_MODEL_PATH)
        print(f"✅ Fault Type Classifier Accuracy: {ft_acc*100:.2f}%")
        
    # 3. Fault Component Model
    df_comp = df.dropna(subset=['Failure_Component_Class'])
    comp_acc = 0.0
    if len(df_comp) > 0:
        print(f"Training Component Classifier on {len(df_comp):,} labeled samples...")
        X_comp = df_comp[FEATURE_COLS].fillna(0)
        y_comp = df_comp['Failure_Component_Class']
        X_ctr, X_cts, y_ctr, y_cts = train_test_split(X_comp, y_comp, test_size=0.2, random_state=42)
        comp_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        comp_model.fit(X_ctr, y_ctr)
        comp_acc = float(accuracy_score(y_cts, comp_model.predict(X_cts)))
        joblib.dump(comp_model, FAULT_COMP_MODEL_PATH)
        print(f"✅ Component Classifier Accuracy: {comp_acc*100:.2f}%")
        
    joblib.dump(FEATURE_COLS, FEATURE_NAMES_PATH)
    print("All models successfully saved to models/ directory.")
    return {"rul_rmse": rmse, "fault_type_accuracy": ft_acc, "comp_accuracy": comp_acc}

def load_models(auto_retrain_on_error: bool = True) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Loads serialized machine learning models from disk.
    If an environment mismatch occurs (e.g. NumPy 1.x vs 2.x BitGenerator pickle difference),
    it automatically retrains the models on the fly for seamless execution.
    """
    try:
        if not (os.path.exists(RUL_MODEL_PATH) and os.path.exists(FAULT_TYPE_MODEL_PATH) and os.path.exists(FAULT_COMP_MODEL_PATH)):
            raise FileNotFoundError("Model artifacts not found.")
            
        rul_model = joblib.load(RUL_MODEL_PATH)
        ft_model = joblib.load(FAULT_TYPE_MODEL_PATH)
        comp_model = joblib.load(FAULT_COMP_MODEL_PATH)
        return rul_model, ft_model, comp_model
    except Exception as e:
        if auto_retrain_on_error:
            print(f"⚙️ [pdm_engine] Auto-training models for current environment due to: {e}")
            from src.data_loader import load_data
            df = load_data()
            train_pdm_models(df)
            rul_model = joblib.load(RUL_MODEL_PATH)
            ft_model = joblib.load(FAULT_TYPE_MODEL_PATH)
            comp_model = joblib.load(FAULT_COMP_MODEL_PATH)
            return rul_model, ft_model, comp_model
        else:
            return None, None, None

