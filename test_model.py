import joblib
import sklearn

print("Scikit-learn version:", sklearn.__version__)

model = joblib.load("credit_risk_model.pkl")
threshold = joblib.load("best_threshold.pkl")

print("Model loaded successfully!")
print("Threshold:", threshold)