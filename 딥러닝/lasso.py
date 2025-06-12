import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("C:/Users/bjh20/source/repos/딥러닝/딥러닝/merged_data_2025.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

df = df.loc["2005-01-01":"2025-04-30"]

target_col = "Total CPI"
X = df.drop(columns=[target_col])
y = df[target_col]

X = X.fillna(method="ffill").fillna(method="bfill")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(X_scaled, y)

coefs = pd.Series(lasso.coef_, index=X.columns)
non_zero_coefs = coefs[coefs != 0]
importance = np.abs(non_zero_coefs)
importance_pct = (importance / importance.sum()) * 100

result_df = pd.DataFrame({
    "feature": importance_pct.index,
    "coef": non_zero_coefs.values,
    "importance": importance_pct.values
}).sort_values(by="importance", ascending=False).reset_index(drop=True)
result_df["rank"] = result_df.index + 1
result_df = result_df[["rank", "feature", "coef", "importance"]]

output_dir = "C:/Users/bjh20/source/repos/딥러닝/딥러닝"
os.makedirs(output_dir, exist_ok=True)
result_df.to_csv(os.path.join(output_dir, "lasso_importance_2025.csv"), index=False)

print("저장 완료", os.path.join(output_dir, "lasso_importance_2025.csv"))
