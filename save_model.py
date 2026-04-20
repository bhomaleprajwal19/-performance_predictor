

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Load & clean data (same as your notebook) ──────────────────────────────
df = pd.read_csv('student_performance_dirty.csv')

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

le = LabelEncoder()
df['Extracurricular Activities'] = le.fit_transform(df['Extracurricular Activities'])

X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities',
        'Sleep Hours', 'Sample Question Papers Practiced']]
y = df['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 1. Linear Regression ───────────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# ── 2. Random Forest (Tuned) ───────────────────────────────────────────────
rf = RandomForestRegressor(random_state=42)
param_rf = {'n_estimators': [50, 100], 'max_depth': [10, 15]}
grid_rf = GridSearchCV(rf, param_rf, cv=3, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)

# ── 3. XGBoost (Tuned) ────────────────────────────────────────────────────
xgb = XGBRegressor(random_state=42)
param_xgb = {'n_estimators': [50, 100], 'max_depth': [6, 10], 'learning_rate': [0.1, 0.2]}
grid_xgb = GridSearchCV(xgb, param_xgb, cv=3, scoring='r2', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_scaled)

# ── Print comparison ───────────────────────────────────────────────────────
models = {
    'Linear Regression':     (y_pred_lr,  lr),
    'Random Forest (Tuned)': (y_pred_rf,  best_rf),
    'XGBoost (Tuned)':       (y_pred_xgb, best_xgb),
}

print(f"\n{'Model':<25} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
print("-" * 52)
for name, (pred, _) in models.items():
    r2   = r2_score(y_test, pred)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:<25} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f}")

# ── Save all models + scaler ───────────────────────────────────────────────
joblib.dump(lr,       'model_lr.pkl')
joblib.dump(best_rf,  'model_rf.pkl')
joblib.dump(best_xgb, 'model_xgb.pkl')
joblib.dump(scaler,   'scaler.pkl')

print("\n✅ Saved: model_lr.pkl, model_rf.pkl, model_xgb.pkl, scaler.pkl") 