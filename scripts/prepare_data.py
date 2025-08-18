# scripts/prepare_data.py

import pandas as pd
import os

# Caminhos dos arquivos
DATA_PATH = os.path.join("data", "data_synthetic.csv")
OUTPUT_PATH = os.path.join("outputs", "clean_dataset.csv")

def prepare_data():
    # === 1. Ler dados ===
    df = pd.read_csv(DATA_PATH)

    # === 2. Selecionar variáveis relevantes ===
    cols = [
        "Customer ID", "Age", "Gender", "Marital Status", "Occupation",
        "Income Level", "Education Level", "Geographic Information",
        "Location", "Coverage Amount", "Premium Amount", "Deductible",
        "Policy Type", "Claim History", "Risk Profile", "Previous Claims History",
        "Credit Score"
    ]
    df = df[cols]

    # === 3. Renomear colunas para padronizar ===
    df = df.rename(columns={
        "Customer ID": "customer_id",
        "Age": "age",
        "Gender": "gender",
        "Marital Status": "marital_status",
        "Occupation": "occupation",
        "Income Level": "income_level",
        "Education Level": "education_level",
        "Geographic Information": "geo_info",
        "Location": "location",
        "Coverage Amount": "coverage_amount",
        "Premium Amount": "premium_amount",
        "Deductible": "deductible",
        "Policy Type": "policy_type",
        "Claim History": "claim_history",
        "Risk Profile": "risk_profile",
        "Previous Claims History": "prev_claims",
        "Credit Score": "credit_score"
    })

    # === 4. Criar variáveis derivadas ===
    # Variável binária de ocorrência de sinistro (se houve histórico de claim > 0)
    df["has_claim"] = df["claim_history"].apply(lambda x: 1 if x > 0 else 0)

    # Frequência de sinistro aproximada (usando histórico anterior)
    df["claim_frequency"] = df["prev_claims"].fillna(0)

    # Severidade aproximada (valor médio do claim se houve)
    df["claim_severity"] = df["claim_history"] / df["claim_frequency"].replace(0, 1)

    # Loss ratio (sinistro/prêmio)
    df["loss_ratio"] = df["claim_history"] / df["premium_amount"].replace(0, 1)

    # === 5. Tratamento de valores faltantes ===
    df = df.fillna({
        "occupation": "Unknown",
        "education_level": "Unknown",
        "geo_info": "Unknown",
        "location": "Unknown",
        "credit_score": df["credit_score"].median()
    })

    # === 6. Salvar dataset limpo ===
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Dataset limpo salvo em {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_data()
