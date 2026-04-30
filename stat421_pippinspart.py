# STAT421 Group 1 — Member Pippin
# XGBoost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor


df = pd.read_csv("insurance.csv")
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

df = pd.get_dummies(df, columns=['region'], drop_first=True)

df = df.astype(float)

x = df.drop(columns=["charges"])
y = df['charges']

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=74)

#XGBoost regressor
xg_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
xg_model.fit(x_train, y_train)

y_pred_xg = xg_model.predict(x_val)


#Results
mse = mean_squared_error(y_val, y_pred_xg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred_xg)
r2 = r2_score(y_val, y_pred_xg)

print(f"R2 Score: {r2:.4f}")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")


plt.figure(figsize=(6, 6))
plt.scatter(y_val, y_pred_xg, alpha=0.5, color='purple')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("XGBoost Predicted Values")
plt.title(f"Model Performance: XGBoost")
plt.show()