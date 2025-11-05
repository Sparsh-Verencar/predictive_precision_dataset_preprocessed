# ====================================
# ✅ Import Libraries
# ====================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt

# ====================================
# ✅ Load Data
# ====================================
df = pd.read_csv("preprocessed_supply_chain_resilience_dataset.csv")

target = "Supply_Risk_Flag"
X = df.drop(columns=[target])
y = df[target]

# ====================================
# ✅ Separate Columns
# ====================================
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols     = X.select_dtypes(exclude=["object"]).columns

# ====================================
# ✅ Train-Test Split
# (Before encoding to avoid leakage)
# ====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ====================================
# ✅ Fit OHE ONLY on Training (prevents leakage!)
# ====================================
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(X_train[categorical_cols])

# Transform
X_train_cat = encoder.transform(X_train[categorical_cols])
X_test_cat  = encoder.transform(X_test[categorical_cols])

# Build DataFrames
X_train_cat_df = pd.DataFrame(
    X_train_cat,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_train.index
)

X_test_cat_df = pd.DataFrame(
    X_test_cat,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_test.index
)

# Merge with numeric
X_train_final = pd.concat([X_train[numeric_cols], X_train_cat_df], axis=1)
X_test_final  = pd.concat([X_test[numeric_cols],  X_test_cat_df],  axis=1)

# ====================================
# ✅ XGBoost Model
# ====================================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    device="cuda"     # ✅ GPU accelerate training
)

# ====================================
# ✅ Train (with progress)
# ====================================
model.fit(
    X_train_final,
    y_train,
    eval_set=[(X_test_final, y_test)],
    verbose=True
)

# ====================================
# ✅ Predict safely (CPU)
# To avoid GPU–CPU mismatch warning
# ====================================
model.set_params(device="cpu")
preds = model.predict(X_test_final)

# ====================================
# ✅ Results
# ====================================
print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

# ====================================
# ✅ Save Model + Encoder
# ====================================
save_dict = {
    "model": model,
    "encoder": encoder,
    "categorical_cols": list(categorical_cols),
    "numeric_cols": list(numeric_cols)
}

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(save_dict, f)

print("\n✅ Model, encoder, and column info saved successfully!")

# ====================================
# ✅ Feature Importance (save image)
# ====================================
fig, ax = plt.subplots(figsize=(12, 8))
xgb.plot_importance(model, max_num_features=20, ax=ax)
plt.tight_layout()
plt.savefig("feature_importance.png")
print("✅ Feature importance saved as feature_importance.png")
