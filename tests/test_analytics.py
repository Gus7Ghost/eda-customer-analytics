# ==============================================================================
# Pruebas Unitarias de Analítica de Datos y Tests Estadísticos
# Autor: Gustavo Bryan López Cosme | UNI - FIIS
# ==============================================================================

import pytest
import os
import sys
import pandas as pd

# Asegurar importación de módulo padre
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eda_engine import load_data, run_descriptive_analysis, compute_cohort_retention

def test_dataset_integrity():
    df = load_data()
    assert df is not None
    assert len(df) == 2500
    assert not df.isnull().values.any(), "El dataset no debe contener valores nulos"
    
    expected_cols = [
        "customer_id", "region", "contract_type", "payment_method",
        "tenure_months", "cohort", "monthly_charges", "total_charges",
        "support_tickets", "usage_score", "churn"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Columna {col} no encontrada"

def test_churn_bounds():
    df = load_data()
    unique_churn = set(df["churn"].unique())
    assert unique_churn.issubset({0, 1}), "La variable target churn debe ser binaria (0 o 1)"
    
    churn_rate = df["churn"].mean()
    assert 0.05 < churn_rate < 0.40, "La tasa de churn debe ser realista (entre 5% y 40%)"

def test_cohort_calculation():
    df = load_data()
    cohort_summary = compute_cohort_retention(df)
    assert not cohort_summary.empty
    assert "retencion_pct" in cohort_summary.columns
    assert (cohort_summary["retencion_pct"] >= 0).all() and (cohort_summary["retencion_pct"] <= 100).all()
