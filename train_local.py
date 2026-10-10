
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# 1. Load the original dataset
df = pd.read_csv("credit_risk_dataset.csv")
df = df.drop_duplicates().copy()

numeric_features = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

categorical_features = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

required = numeric_features + categorical_features + ["loan_status"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

# 2. Prepare the features and target
df["loan_status"] = pd.to_numeric(
    df["loan_status"], errors="coerce"
)
df = df.dropna(subset=["loan_status"])

df["loan_status"] = df["loan_status"].astype(int)

X = df[numeric_features + categorical_features].copy()
y = df["loan_status"]

for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# 3. Split the data
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 4. Prepare numerical and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 5. Train XGBoost locally
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if negative_count == 0 or positive_count == 0:
    raise ValueError("The dataset must contain both target classes.")

classifier = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=negative_count / positive_count,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier),
])

print("Training model...")
model.fit(X_train, y_train)

# 6. Select a classification threshold using validation data
probabilities = model.predict_proba(X_valid)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_valid, probabilities
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-9)
)

best_index = int(np.argmax(f1_values))
best_threshold = float(thresholds[best_index])

predictions = (probabilities >= best_threshold).astype(int)

print("Validation ROC-AUC:", round(
    roc_auc_score(y_valid, probabilities), 4
))
print("Validation F1:", round(
    f1_score(y_valid, predictions), 4
))
print("Selected threshold:", round(best_threshold, 4))

# 7. Save NEW files, leaving the downloaded files untouched
joblib.dump(model, "credit_risk_model_local.pkl")
joblib.dump(best_threshold, "best_threshold_local.pkl")

# 8. Confirm these new files can be loaded on this computer
loaded_model = joblib.load("credit_risk_model_local.pkl")
loaded_threshold = joblib.load("best_threshold_local.pkl")

check = loaded_model.predict_proba(X_valid.iloc[:5])[:, 1]

print("Reload test passed.")
print("Threshold loaded:", loaded_threshold)
print("Sample probabilities:", check)
print("New local model saved successfully.")
