# =============================================================================
# Iris Flower Classification
# Author: Fahim Ahmed | BUET Urban & Regional Planning
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings("ignore")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current folder:", os.getcwd())
# ── 1. Load Dataset ───────────────────────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 55)
print("       IRIS FLOWER CLASSIFICATION — OVERVIEW")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print(f"Classes       : {list(iris.target_names)}")
print(f"\nClass distribution:\n{df['species'].value_counts()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic statistics:\n{df.describe().round(2)}")

# ── 2. Exploratory Data Analysis (EDA) ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Iris Dataset — Feature Distributions by Species", fontsize=14, fontweight="bold")

features = iris.feature_names
colors = {"setosa": "#4C72B0", "versicolor": "#DD8452", "virginica": "#55A868"}

for ax, feature in zip(axes.flatten(), features):
    for species, color in colors.items():
        subset = df[df["species"] == species][feature]
        ax.hist(subset, alpha=0.6, label=species, color=color, bins=15, edgecolor="white")
    ax.set_title(feature.replace(" (cm)", "").title(), fontsize=11)
    ax.set_xlabel("cm")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("iris_eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] iris_eda_distributions.png")

# Pairplot
sns.set_theme(style="ticks")
pair = sns.pairplot(df, hue="species", palette=colors, plot_kws={"alpha": 0.7}, height=2.2)
pair.fig.suptitle("Iris Pairplot — Feature Relationships", y=1.02, fontsize=13, fontweight="bold")
pair.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] iris_pairplot.png")

# Correlation heatmap (numeric only)
fig, ax = plt.subplots(figsize=(7, 5))
corr = df.drop(columns="species").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("iris_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] iris_correlation.png")

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
X = df.drop(columns="species")
y = iris.target  # numeric labels (0, 1, 2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain size : {X_train_sc.shape[0]} samples")
print(f"Test size  : {X_test_sc.shape[0]} samples")

# ── 4. Model Training & Comparison ───────────────────────────────────────────
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, random_state=42),
}

print("\n" + "=" * 55)
print("          MODEL COMPARISON (5-Fold Cross-Validation)")
print("=" * 55)

results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="accuracy")
    model.fit(X_train_sc, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_sc))
    results[name] = {"cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "test_acc": test_acc}
    print(f"\n{name}")
    print(f"  CV Accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

# ── 5. Best Model — Detailed Evaluation ──────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["test_acc"])
best_model = models[best_name]
y_pred = best_model.predict(X_test_sc)

print("\n" + "=" * 55)
print(f"  BEST MODEL: {best_name}")
print("=" * 55)
print(f"\nTest Accuracy: {results[best_name]['test_acc']:.4f}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] iris_confusion_matrix.png")

# ── 6. Model Comparison Bar Chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
names  = list(results.keys())
test_accs = [results[n]["test_acc"] for n in names]
cv_means  = [results[n]["cv_mean"]  for n in names]

x = np.arange(len(names))
width = 0.35
bars1 = ax.bar(x - width/2, cv_means,  width, label="CV Mean Accuracy", color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + width/2, test_accs, width, label="Test Accuracy",    color="#DD8452", alpha=0.85)

ax.set_ylim(0.85, 1.02)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison — Iris Classification", fontsize=13, fontweight="bold")
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("iris_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] iris_model_comparison.png")

# ── 7. Feature Importance (Random Forest) ────────────────────────────────────
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({"Feature": iris.feature_names, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(feat_df["Feature"], feat_df["Importance"], color="#55A868", edgecolor="white")
ax.set_xlabel("Importance Score")
ax.set_title("Random Forest — Feature Importances", fontsize=12, fontweight="bold")
for i, v in enumerate(feat_df["Importance"]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("iris_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] iris_feature_importance.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("                     SUMMARY")
print("=" * 55)
for name, res in results.items():
    print(f"  {name:<28} Test Acc: {res['test_acc']:.4f}")
print(f"\n  Best model → {best_name}")
print(f"  Test Accuracy: {results[best_name]['test_acc'] * 100:.2f}%")
print("\n[Done] All plots saved. Ready for GitHub upload.")
