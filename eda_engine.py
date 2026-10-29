# ==============================================================================
# Motor Analítico: Análisis Exploratorio de Datos (EDA) y Pruebas Estadísticas
# Autor: Gustavo Bryan López Cosme | UNI - FIIS
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_churn_analytics.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Configuración visual profesional
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

def load_data():
    if not os.path.exists(DATA_PATH):
        from generate_dataset import generate_customer_dataset
        return generate_customer_dataset()
    return pd.read_csv(DATA_PATH)

def run_descriptive_analysis(df):
    print("\n" + "=" * 80)
    print("  1. RESUMEN ESTADÍSTICO Y DISTRIBUCIONES (UNIVARIADO)")
    print("=" * 80)
    
    num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "usage_score"]
    stats_df = df[num_cols].describe().T
    stats_df["skewness"] = df[num_cols].skew()
    stats_df["kurtosis"] = df[num_cols].kurtosis()
    print(stats_df[["mean", "std", "min", "50%", "max", "skewness", "kurtosis"]].round(2))
    
    churn_rate = df["churn"].mean() * 100
    print(f"\n[KPI Base] Tasa Global de Fuga (Churn Rate): {churn_rate:.2f}%")
    print(f"[Registros Totales]: {len(df):,} clientes | Columnas: {len(df.columns)}")
    return stats_df

def run_hypothesis_tests(df):
    print("\n" + "=" * 80)
    print("  2. CONTRASTE DE HIPÓTESIS ESTADÍSTICAS (BIVARIADO)")
    print("=" * 80)
    
    churners = df[df["churn"] == 1]
    retained = df[df["churn"] == 0]
    
    # Hipótesis 1: Tickets de soporte entre churners vs retenidos (Mann-Whitney U)
    u_stat, u_p_val = stats.mannwhitneyu(churners["support_tickets"], retained["support_tickets"], alternative="two-sided")
    print(f"H1: Diferencia en Tickets de Soporte:")
    print(f"    - Media Churn: {churners['support_tickets'].mean():.2f} | Media Retenidos: {retained['support_tickets'].mean():.2f}")
    print(f"    - Test Mann-Whitney U: Estadístico={u_stat:.1f}, p-valor={u_p_val:.4e} -> {'Rechaza H0 (Significativo p < 0.001)' if u_p_val < 0.05 else 'No Rechaza H0'}")

    # Hipótesis 2: Antigüedad (Tenure) entre churners vs retenidos (T-test)
    t_stat, t_p_val = stats.ttest_ind(churners["tenure_months"], retained["tenure_months"], equal_var=False)
    print(f"\nH2: Diferencia en Antigüedad (Tenure en Meses):")
    print(f"    - Media Churn: {churners['tenure_months'].mean():.2f} meses | Media Retenidos: {retained['tenure_months'].mean():.2f} meses")
    print(f"    - Welch's T-test: t={t_stat:.2f}, p-valor={t_p_val:.4e} -> {'Rechaza H0 (Significativo p < 0.001)' if t_p_val < 0.05 else 'No Rechaza H0'}")

    # Hipótesis 3: Tipo de Contrato vs Churn (Chi-Cuadrado de Independencia)
    contingency = pd.crosstab(df["contract_type"], df["churn"])
    chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)
    print(f"\nH3: Independencia entre Tipo de Contrato y Churn:")
    print(contingency)
    print(f"    - Chi-Cuadrado: chi2={chi2:.2f}, gl={dof}, p-valor={chi_p:.4e} -> {'Asociación Altamente Significativa' if chi_p < 0.05 else 'Independiente'}")

def generate_visualizations(df):
    print("\n" + "=" * 80)
    print("  3. GENERANDO GRÁFICOS Y FIGURAS ANALÍTICAS (REPORTS/FIGURES/)")
    print("=" * 80)
    
    # Figura 1: Distribución de Cargos Mensuales por Churn
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x="monthly_charges", hue="churn", kde=True, ax=axes[0], palette=["#2563eb", "#dc2626"], alpha=0.5)
    axes[0].set_title("Distribución de Cargos Mensuales (S/.) por Estado de Churn", fontweight="bold")
    axes[0].set_xlabel("Cargo Mensual (S/.)")
    axes[0].set_ylabel("Frecuencia")
    
    sns.boxplot(data=df, x="contract_type", y="monthly_charges", hue="churn", ax=axes[1], palette=["#2563eb", "#dc2626"])
    axes[1].set_title("Cargos Mensuales por Tipo de Contrato", fontweight="bold")
    axes[1].set_xlabel("Tipo de Contrato")
    axes[1].set_ylabel("Cargo Mensual (S/.)")
    
    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, "01_distribution_charges.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print(f"[Graficado] {p1}")

    # Figura 2: Matriz de Correlación de Spearman (Robustez no lineal)
    num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "usage_score", "churn"]
    corr = df[num_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Matriz de Correlación de Spearman (Métricas vs Churn)", fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, "02_correlation_matrix.png")
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    print(f"[Graficado] {p2}")

    # Figura 3: Tasa de Fuga por Tipo de Contrato y Región
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    contract_churn = df.groupby("contract_type")["churn"].mean().reset_index()
    contract_churn["churn"] *= 100
    sns.barplot(data=contract_churn, x="contract_type", y="churn", ax=axes[0], palette="Reds_r")
    axes[0].set_title("Tasa de Churn (%) por Tipo de Contrato", fontweight="bold")
    axes[0].set_ylabel("Tasa de Churn (%)")
    axes[0].set_xlabel("Tipo de Contrato")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 7), textcoords="offset points", fontweight="bold")

    region_churn = df.groupby("region")["churn"].mean().reset_index()
    region_churn["churn"] *= 100
    sns.barplot(data=region_churn, x="region", y="churn", ax=axes[1], palette="Blues_r")
    axes[1].set_title("Tasa de Churn (%) por Región Geográfica", fontweight="bold")
    axes[1].set_ylabel("Tasa de Churn (%)")
    axes[1].set_xlabel("Región")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 7), textcoords="offset points", fontweight="bold")

    plt.tight_layout()
    p3 = os.path.join(FIGURES_DIR, "03_churn_by_contract_and_region.png")
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    print(f"[Graficado] {p3}")

def compute_cohort_retention(df):
    print("\n" + "=" * 80)
    print("  4. ANÁLISIS DE RETENCIÓN DE COHORTES (COHORT RETENTION)")
    print("=" * 80)
    
    # Filtrar últimas 8 cohortes
    recent_cohorts = sorted(df["cohort"].unique())[-8:]
    cohort_df = df[df["cohort"].isin(recent_cohorts)].copy()
    
    cohort_summary = cohort_df.groupby("cohort").agg(
        total_clientes=("customer_id", "count"),
        churners=("churn", "sum"),
        retencion_pct=("churn", lambda x: round((1 - x.mean()) * 100, 1))
    ).reset_index()
    
    print(cohort_summary.to_string(index=False))
    return cohort_summary

if __name__ == "__main__":
    df = load_data()
    run_descriptive_analysis(df)
    run_hypothesis_tests(df)
    generate_visualizations(df)
    compute_cohort_retention(df)
    print("\n[OK] Análisis Exploratorio (EDA) completado con éxito.")
