# ===============================
# ✅ LOAD + TEST MODEL
# ===============================
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ===============================
# ✅ Load stored model bundle
# ===============================
with open("xgb_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
encoder = saved["encoder"]
categorical_cols = saved["categorical_cols"]
numeric_cols = saved["numeric_cols"]

# ===============================
# ✅ Load dataset
# ===============================
df = pd.read_csv("preprocessed_supply_chain_resilience_dataset.csv")

target = "Supply_Risk_Flag"
X = df.drop(columns=[target])
y = df[target]

# ===============================
# ✅ Encode categorical
# ===============================
X_cat = encoder.transform(X[categorical_cols])

X_cat_df = pd.DataFrame(
    X_cat,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X.index
)

# Combine
X_final = pd.concat([X[numeric_cols], X_cat_df], axis=1)

# ===============================
# ✅ Predict
# ===============================
preds = model.predict(X_final)

print("\nAccuracy:", accuracy_score(y, preds))
print("\nClassification Report:\n", classification_report(y, preds))
