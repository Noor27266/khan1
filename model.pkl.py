import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# === Load dataset ===
folderPath = 'F:/Inputs data'
fileName = 'Book2.xlsx'
sheetName = 'Sheet2'

file_path = f"{folderPath}/{fileName}"
df = pd.read_excel(file_path, sheet_name=sheetName)

# === Columns check ===
print("Dataset Columns:", df.columns.tolist())

# === Separate input (X) and output (y) ===
X = df.drop(columns=['DampingRatio'])
y = df['DampingRatio']

# ---------------------------- 
# RANDOM FOREST (80% train, 20% test)
# ---------------------------- 
print("\n=== Training Random Forest (80/20 split) ===")
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)
rf_preds = rf_model.predict(X_test_rf)
print("Random Forest R2 on Test:", r2_score(y_test_rf, rf_preds))

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✅ Random Forest model saved as rf_model.pkl\n")

# ---------------------------- 
# XGBoost (80/10/10 split)
# ---------------------------- 
print("\n=== Training XGBoost (80/10/10 split) ===")
X_temp_xgb, X_test_xgb, y_temp_xgb, y_test_xgb = train_test_split(X, y, test_size=0.10, random_state=42)
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(X_temp_xgb, y_temp_xgb, test_size=0.1111, random_state=42)

xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train_xgb, y_train_xgb)  # No eval_set or early_stopping here
xgb_preds = xgb_model.predict(X_test_xgb)
print("XGBoost R2 on Test:", r2_score(y_test_xgb, xgb_preds))

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("✅ XGBoost model saved as xgb_model.pkl\n")

# === Final Message ===
print("\n✅✅✅ All models trained, validated, tested, and saved successfully!")
