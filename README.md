# Breast-Cancer Cytology Classifier (MLP, scikit-learn)

> **TL;DR** — Leakage-safe pipeline (`StandardScaler → MLPClassifier(lbfgs)`) that predicts **malignant (1)** vs **benign (0)** from 5 cytology features.  
> On held-out data: **Accuracy ≈ 0.89**, **Malignant recall ≈ 0.86**.  
> Stratified 5-fold CV AUC ≈ **0.97–0.98**. Dataset is mildly imbalanced (**~63:37**).

---

## 1) Project goal
Build a small, reproducible tabular ML pipeline to classify **breast lesions** as **benign (0)** or **malignant (1)** using five summary features from fine-needle aspiration (FNA) cytology.

This is an **educational** project — **not** a medical device and **not** for clinical use.

---

## 2) Data

- File: `Breast_cancer_data.csv` (569 rows × 6 columns)  
- **Features (5):** `mean_radius`, `mean_texture`, `mean_perimeter`, `mean_area`, `mean_smoothness`  
- **Target:** `diagnosis` (binary) — **1 = malignant**, **0 = benign**  
- Class balance: **1 → 357 (62.7%)**, **0 → 212 (37.3%)** → ~**63:37**

> These features are *derived from microscope images* (cytology). The model learns from tabular features — it does **not** read medical images directly.

---

## 3) Method

- **Pipeline:** `StandardScaler → MLPClassifier` (scikit-learn)  
- **Model:** `MLPClassifier(solver='lbfgs', random_state=42, max_iter=[1000], hidden_layer_sizes=[(100,)] , alpha=[1e-4])`  
- **Validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`  
- **Metrics:** Accuracy, ROC-AUC, Precision/Recall/F1 per class; Confusion Matrix  
- **Why lbfgs?** Small dataset; quasi-Newton often converges fast and stably on shallow networks.

---

## 4) Results (update with your exact numbers)

**Held-out test set**
- **Accuracy:** ~**0.89**  
- **Malignant (class 1) recall:** ~**0.857**  
- **Precision/Recall/F1 by class:** see classification report in the notebook  
- **Confusion matrix:** included in figures

**Cross-validation (train data, stratified 5-fold)**
- **ROC-AUC:** ~**0.97–0.98** (MLP)  
- Baseline comparison: Perceptron (reported in notebook)

---

## 5) How to run (reproducible)

### A) Environment
Create a virtual env (Python 3.10+ recommended) and install:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
