# scripts/pricing_models.py

"""
Modelagem de Pricing de Seguros
--------------------------------
Este script implementa duas abordagens para cálculo de prêmio esperado:

1. Prêmio Puro: Frequência × Severidade usando GLMs clássicos
2. Claim Cost direto: previsão do valor do sinistro usando Gradient Boosting

Saídas:
- Sumário dos GLMs
- Predições de prêmio puro
- RMSE do modelo de claim cost
"""

# === 1. Importações ===
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# === 2. Carregar dados limpos com caminho robusto ===
base_path = os.path.dirname(os.path.abspath(__file__))  # pasta do script
df_path = os.path.join(base_path, "../outputs/clean_dataset.csv")  # ajusta relativo ao script

if not os.path.exists(df_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {df_path}")

df = pd.read_csv(df_path)
print(f"CSV carregado de: {df_path}")

# Seleção de variáveis
features = ["age", "credit_score", "coverage_amount", "deductible", 
            "marital_status", "occupation", "risk_profile"]

# Criar dummies para categóricas
df_model = pd.get_dummies(df[features + ["has_claim", "claim_severity", "claim_amount"]], drop_first=True)

# ====================================
# 3. Abordagem 1: Prêmio Puro (GLM)
# ====================================

print("\n=== ABORDAGEM 1: PRÊMIO PURO ===")

# --- Frequência ---
X_freq = df_model.drop(columns=["has_claim", "claim_severity", "claim_amount"])
y_freq = df_model["has_claim"]
X_freq_sm = sm.add_constant(X_freq)
glm_freq = sm.GLM(y_freq, X_freq_sm, family=sm.families.Poisson())
freq_result = glm_freq.fit()
print("\n--- Frequência (GLM Poisson) ---")
print(freq_result.summary())

df_model["pred_freq"] = freq_result.predict(X_freq_sm)

# --- Severidade ---
df_sev = df_model[df_model["has_claim"]==1]
X_sev = df_sev.drop(columns=["has_claim", "claim_severity", "claim_amount"])
y_sev = df_sev["claim_severity"]
X_sev_sm = sm.add_constant(X_sev)
glm_sev = sm.GLM(y_sev, X_sev_sm, family=sm.families.Gamma(link=sm.families.links.log()))
sev_result = glm_sev.fit()
print("\n--- Severidade (GLM Gama) ---")
print(sev_result.summary())

df_sev["pred_sev"] = sev_result.predict(X_sev_sm)
df_model["pred_sev"] = df_model["has_claim"].map(df_sev.set_index(df_sev.index)["pred_sev"]).fillna(0)

# --- Prêmio Puro ---
df_model["pure_premium"] = df_model["pred_freq"] * df_model["pred_sev"]
print("\nExemplo de Prêmio Puro (5 primeiros clientes):")
print(df_model[["pred_freq", "pred_sev", "pure_premium"]].head())

# ====================================
# 4. Abordagem 2: Claim Cost direto (LightGBM)
# ====================================

print("\n=== ABORDAGEM 2: CLAIM COST DIRETO ===")

X_full = df_model.drop(columns=["has_claim", "claim_severity", "claim_amount", "pred_freq", "pred_sev", "pure_premium"])
y_full = df_model["claim_amount"]

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1
}

gbm = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=[lgb_test], early_stopping_rounds=20, verbose_eval=50)

y_pred_gbm = gbm.predict(X_test, num_iteration=gbm.best_iteration)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_gbm))
print(f"\nRMSE do modelo de Claim Cost (LightGBM): {rmse:.2f}")

# Mostrar primeiras previsões
print("\nExemplo de predição Claim Cost (5 primeiros clientes do teste):")
print(pd.DataFrame({"y_true": y_test.values[:5], "y_pred": y_pred_gbm[:5]}))
