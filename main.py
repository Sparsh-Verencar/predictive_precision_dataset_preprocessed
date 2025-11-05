# ===============================
#  ✅  XGBoost Ensemble Training
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ===============================
# ✅ LOAD DATA
# ===============================
# Replace with your CSV path
df = pd.read_csv("preprocessed_supply_chain_resilience_dataset.csv")

# Ensure your DataFrame is available as df
# target column
target = "Supply_Risk_Flag"

X = df.drop(columns=[target])
y = df[target]

# ===============================
# ✅ COLUMN TYPES
# ===============================
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

# ===============================
# ✅ PREPROCESSING
# ===============================
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

# ===============================
# ✅ TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# ✅ XGBOOST MODEL
# ===============================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    device="cuda" 
)

# ===============================
# ✅ PIPELINE
# ===============================
clf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', model)
])

# ===============================
# ✅ TRAIN
# ===============================
clf.fit(
    X_train, y_train,
    model__eval_set=[(X_test, y_test)],
    model__verbose=True
)


# ===============================
# ✅ PREDICT
# ===============================
preds = clf.predict(X_test)

# ===============================
# ✅ METRICS
# ===============================
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

# ===============================
# ✅ FEATURE IMPORTANCE (Optional)
# ===============================
import matplotlib.pyplot as plt
import xgboost as xgbmodule

xgb = clf.named_steps['model']
xgbmodule.plot_importance(xgb)
plt.show()
