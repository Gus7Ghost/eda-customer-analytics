# ==============================================================================
# Generador de Dataset Analítico de Clientes y Transacciones
# Autor: Gustavo Bryan López Cosme | UNI - FIIS
# ==============================================================================

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_customer_dataset(n_samples=2500, seed=42):
    np.random.seed(seed)
    
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_samples + 1)]
    
    # 1. Variables Demográficas y de Cuenta
    regions = np.random.choice(["Lima", "Arequipa", "La Libertad", "Cusco", "Piura"], size=n_samples, p=[0.50, 0.15, 0.15, 0.10, 0.10])
    contract_types = np.random.choice(["Mes a Mes", "Anual", "Bianual"], size=n_samples, p=[0.55, 0.30, 0.15])
    payment_methods = np.random.choice(["Débito Automático", "Tarjeta de Crédito", "Transferencia Bancaria", "Efectivo/Ventanilla"], size=n_samples, p=[0.35, 0.30, 0.20, 0.15])
    
    # 2. Antigüedad (Tenure en meses)
    tenure_months = np.random.exponential(scale=18, size=n_samples).astype(int) + 1
    tenure_months = np.clip(tenure_months, 1, 72)
    
    # 3. Métricas de Consumo y Soporte
    monthly_charges = np.random.normal(loc=145.0, scale=45.0, size=n_samples)
    monthly_charges = np.clip(monthly_charges, 35.0, 320.0).round(2)
    
    # Total charges proporcional a tenure con ligera variabilidad
    total_charges = (monthly_charges * tenure_months * np.random.uniform(0.92, 1.05, size=n_samples)).round(2)
    
    # Tickets de soporte técnico (Poisson)
    support_tickets = np.random.poisson(lam=1.8, size=n_samples)
    
    # Score de uso de plataforma (0.0 a 10.0)
    usage_score = np.random.beta(a=5, b=2, size=n_samples) * 10
    usage_score = np.clip(usage_score, 0.5, 10.0).round(1)
    
    # 4. Probabilidad de Fuga (Churn) basada en factores reales
    # Factores de riesgo: Contrato mes a mes (+), tickets altos (+), bajo uso (+), tenure bajo (+)
    churn_prob = (
        0.15 
        + 0.28 * (contract_types == "Mes a Mes")
        - 0.12 * (contract_types == "Bianual")
        + 0.06 * np.clip(support_tickets - 2, 0, 10)
        - 0.03 * (tenure_months / 12.0)
        - 0.04 * (usage_score / 2.0)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)
    
    # 5. Cohorte de Registro (YYYY-MM)
    # Asignamos cohorte retrospectiva según tenure
    from datetime import datetime, timedelta
    ref_date = datetime(2026, 10, 1)
    cohort_dates = [ref_date - timedelta(days=int(t * 30.4)) for t in tenure_months]
    cohorts = [d.strftime("%Y-%m") for d in cohort_dates]
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "region": regions,
        "contract_type": contract_types,
        "payment_method": payment_methods,
        "tenure_months": tenure_months,
        "cohort": cohorts,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "usage_score": usage_score,
        "churn": churn
    })
    
    output_path = os.path.join(DATA_DIR, "customer_churn_analytics.csv")
    df.to_csv(output_path, index=False)
    print(f"[Dataset] Generado exitosamente: {output_path} ({len(df)} registros)")
    return df

if __name__ == "__main__":
    generate_customer_dataset()
