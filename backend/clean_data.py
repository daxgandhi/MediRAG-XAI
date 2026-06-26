import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "datasets"

def clean_dataset():
    train_path = DATA_DIR / "Training.csv"
    test_path  = DATA_DIR / "Testing.csv"

    if not train_path.exists():
        print("Training.csv not found!")
        return

    print("Cleaning dataset...")
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # 1. Drop Unnamed columns
    df_train = df_train.loc[:, ~df_train.columns.str.contains("^Unnamed")]
    df_test  = df_test.loc[:, ~df_test.columns.str.contains("^Unnamed")]

    # 2. Clean disease names (prognosis)
    disease_map = {
        "Diabetes ": "Diabetes",
        "Hypertension ": "Hypertension",
        "Peptic ulcer diseae": "Peptic Ulcer Disease",
        "Osteoarthristis": "Osteoarthritis",
        "(vertigo) Paroymsal  Positional Vertigo": "Paroxysmal Positional Vertigo",
        "Dimorphic hemmorhoids(piles)": "Hemorrhoids (Piles)",
        "hepatitis A": "Hepatitis A",
        "Bronchial Asthma": "Bronchial Asthma",
        "Cervical spondylosis": "Cervical Spondylosis",
        "Alcoholic hepatitis": "Alcoholic Hepatitis",
        "Heart attack": "Heart Attack",
        "Varicose veins": "Varicose Veins",
        "Urinary tract infection": "Urinary Tract Infection",
        "Fungal infection": "Fungal Infection",
        "Drug Reaction": "Drug Reaction",
        "Chicken pox": "Chicken Pox",
        "Common Cold": "Common Cold",
        "Chronic cholestasis": "Chronic Cholestasis",
    }

    def clean_disease(d):
        d = str(d).strip()
        if d in disease_map:
            return disease_map[d]
        return d.title() if d.islower() else d

    df_train["prognosis"] = df_train["prognosis"].apply(clean_disease)
    df_test["prognosis"]  = df_test["prognosis"].apply(clean_disease)

    # Save cleaned
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("✅ Dataset cleaned successfully! Overwrote Training.csv and Testing.csv")
    print("New classes sample:")
    print(df_train["prognosis"].unique()[:10])

if __name__ == "__main__":
    clean_dataset()
