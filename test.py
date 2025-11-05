import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, roc_auc_score
)
import pickle

# =====================================================
# LOAD PRE-PROCESSED DATA
# =====================================================
df = pd.read_csv("processed_dataset.csv")

target = "Supply_Risk_Flag"

# -------------------------
# Model-1: WITH Delay_Days
# -------------------------
X1 = df.drop(columns=[target])
y = df[target]

X1_train, X1_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=42
)

model1 = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model1.fit(X1_train, y_train)
pred1 = model1.predict(X1_test)

print("\n=========== MODEL 1 (With Delay_Days) ===========")
print("Accuracy:", accuracy_score(y_test, pred1))
print("F1 Score:", f1_score(y_test, pred1))
print(classification_report(y_test, pred1))

# Save
pickle.dump(model1, open("model_with_delay.pkl", "wb"))


# -------------------------
# Model-2: WITHOUT Delay_Days
# -------------------------
X2 = df.drop(columns=[target, "Delay_Days"])

X2_train, X2_test, y_train2, y_test2 = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

model2 = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model2.fit(X2_train, y_train2)
pred2 = model2.predict(X2_test)

print("\n=========== MODEL 2 (Without Delay_Days) ===========")
print("Accuracy:", accuracy_score(y_test2, pred2))
print("F1 Score:", f1_score(y_test2, pred2))
print(classification_report(y_test2, pred2))

# Save
pickle.dump(model2, open("model_without_delay.pkl", "wb"))


# =====================================================
# COMPARISON METRICS
# =====================================================
acc1 = accuracy_score(y_test, pred1)
acc2 = accuracy_score(y_test2, pred2)

f11 = f1_score(y_test, pred1)
f12 = f1_score(y_test2, pred2)

print("\nModel-1 Accuracy (With Delay_Days):", acc1)
print("Model-2 Accuracy (Without Delay_Days):", acc2)

print("\nModel-1 F1 (With Delay_Days):", f11)
print("Model-2 F1 (Without Delay_Days):", f12)


# =====================================================
# 1) Accuracy Comparison Bar Graph
# =====================================================
plt.figure(figsize=(6,4))
plt.bar(["With Delay", "Without Delay"], [acc1, acc2])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("accuracy_comparison.png", dpi=300)


# =====================================================
# 2) F1 Score Comparison
# =====================================================
plt.figure(figsize=(6,4))
plt.bar(["With Delay", "Without Delay"], [f11, f12])
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.savefig("f1_comparison.png", dpi=300)


# =====================================================
# 3) ROC Curve
# =====================================================
proba1 = model1.predict_proba(X1_test)[:,1]
proba2 = model2.predict_proba(X2_test)[:,1]

fpr1, tpr1, _ = roc_curve(y_test, proba1)
fpr2, tpr2, _ = roc_curve(y_test2, proba2)

auc1 = roc_auc_score(y_test, proba1)
auc2 = roc_auc_score(y_test2, proba2)

plt.figure(figsize=(6,5))
plt.plot(fpr1, tpr1, label=f"With Delay (AUC={auc1:.3f})")
plt.plot(fpr2, tpr2, label=f"Without Delay (AUC={auc2:.3f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("roc_comparison.png", dpi=300)


# =====================================================
# 4) Confusion Matrices
# =====================================================
cm1 = confusion_matrix(y_test, pred1)
cm2 = confusion_matrix(y_test2, pred2)

fig, axs = plt.subplots(1,2, figsize=(10,4))

sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title("With Delay")

sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=axs[1])
axs[1].set_title("Without Delay")

plt.savefig("confusion_matrices.png", dpi=300)
